// How fast is one batch-1 quantized matvec, in isolation, on this model's real shapes?
//
// The decode token spends 24.0 ms of its 30.0 ms of device time in MUL_MAT, 341 nodes per
// token, and the measured per-tensor rates are 29-63 GB/s on a card that streams at 128-155.
// docs/STRATA-PORT.md section 7 has where those numbers come from; this exists to measure the
// op itself, the way fa_bench measures attention, because an end-to-end A/B cannot resolve a
// few percent on this box.
//
// Shapes and dtypes are the model's own, read from the GGUF: the LM head, which is a single
// 6.63 ms matvec, and the recurrent layers' projections, which are 341 of the nodes. Nothing
// here is a guess about a kernel's performance; every row is a real op with the real dtype.
//
// The row to change is the vec kernel's resident-blocks-per-SM launch bound, which is 1 in
// llama.cpp. That is a compiler hint, not a grid size: calc_nwarps and calc_rows_per_block
// are compile-time functions of the type and column count alone, so the block count and the
// order of the reduction over K do not depend on it. Which is what makes a sweep here
// bit-exact where the same sweep on the attention kernel was not.
//
// Not part of the serving path; a measurement tool.

#include "amp/model/gguf.h"
#include "amp/bytes.h"
#include "amp/format.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Shape {
    const char * label;
    ggml_type    type;
    int64_t      k;   // reduction length, the contiguous dim of the weight
    int64_t      n;   // output rows
};

struct Row {
    double   ms;
    double   gbps;
    double   base_ms;
    double   base_gbps;
    uint64_t checksum;
};

// Best-effort achievable DRAM bandwidth, same kernel shape as fa_bench so the two tools'
// percentages are comparable. The card idles at 315 MHz and cannot be clock-locked, so this
// is re-measured per row and the row reports its own baseline.
double measure_stream(ggml_backend_t backend, uint64_t bytes, double & ms_out) {
    ggml_init_params ip = {};
    ip.mem_size   = 256 * 1024;
    ip.no_alloc   = true;
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, bytes / sizeof(float));
    ggml_tensor * one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_cgraph * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ggml_mul(ctx, a, one));

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        ggml_free(ctx);
        return 0.0;
    }
    const float one_host = 1.0f;
    ggml_backend_tensor_set(one, &one_host, 0, sizeof(float));

    for (int i = 0; i < 3; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    double best = 0.0;
    ms_out = 0.0;
    for (int i = 0; i < 5; i++) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_graph_compute(backend, gf);
        const double ms = (double) (ggml_time_us() - t0) / 1e3;
        const double gbps = 2.0 * (double) bytes / (ms * 1e-3) / 1e9;
        if (gbps > best) {
            best = gbps;
            ms_out = ms;
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return best;
}

Row measure_matvec(ggml_backend_t backend, const Shape & s, int passes) {
    ggml_init_params ip = {};
    // Enough for the graph, the scratch the op needs, and the context the dequant produces.
    ip.mem_size = 512 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * w   = ggml_new_tensor_2d(ctx, s.type, s.k, s.n);
    ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k, 1);
    ggml_tensor * dst = ggml_mul_mat(ctx, w, x);
    ggml_cgraph * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        fprintf(stderr, "matvec-bench: allocation failed for %s\n", s.label);
        ggml_free(ctx);
        return {};
    }

    std::vector<float> xh((size_t) s.k);
    for (int64_t i = 0; i < s.k; i++) {
        xh[(size_t) i] = 0.5f + 0.001f * (float) (i % 251);
    }
    ggml_backend_tensor_set(x, xh.data(), 0, xh.size() * sizeof(float));
    // A weight that is not all zero, so nothing can be folded away.
    {
        std::vector<uint8_t> junk(4096, 0x5a);
        ggml_backend_tensor_set(w, junk.data(), 0, junk.size());
    }

    for (int i = 0; i < 3; i++) {
        ggml_backend_graph_compute(backend, gf);
    }

    double best = 1e30;
    uint64_t sum = 0;
    for (int i = 0; i < passes; i++) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_graph_compute(backend, gf);
        const double ms = (double) (ggml_time_us() - t0) / 1e3;
        if (ms < best) {
            best = ms;
        }
        if (i == 0) {
            std::vector<float> out((size_t) s.n);
            ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));
            for (float v : out) {
                sum += (uint64_t) (int64_t) v;
            }
        }
    }

    const double bytes = (double) ggml_nbytes(w);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    Row r;
    r.ms       = best;
    r.gbps     = bytes / (best * 1e-3) / 1e9;
    r.checksum = sum;
    return r;
}

} // namespace

int main(int argc, char ** argv) {
    const char * model  = nullptr;
    int          passes = 7;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (a == "--passes" && i + 1 < argc) {
            passes = atoi(argv[++i]);
        } else if (a == "--help" || a == "-h") {
            printf(
                "usage: amp-matvec-bench --model <gguf> [--passes N]\n"
                "\n"
                "Times one batch-1 quantized matvec for the shapes this model actually uses,\n"
                "at q8_0/q8_0 KV and whatever dtype each projection was quantized to, and\n"
                "reports the achieved rate against a measured streaming baseline.\n");
            return 0;
        }
    }

    if (model == nullptr) {
        fprintf(stderr, "amp-matvec-bench: --model is required\n");
        return 2;
    }

    auto gguf = amp::GGUFFile::open(model);
    if (!gguf) {
        fprintf(stderr, "amp-matvec-bench: cannot read %s: %s\n", model, gguf.message().c_str());
        return 1;
    }

    // The dtypes are not assumed: they are the file's. output.weight is the LM head, a single
    // matvec that costs 6.63 ms per token, and it is not the dtype a reader would guess.
    const char * want[] = { "output.weight", "blk.0.ssm_out.weight", "blk.0.ffn_gate_shexp.weight" };
    std::vector<Shape> shapes;
    for (const char * name : want) {
        const auto * t = gguf->find(name);
        if (t == nullptr) {
            continue;
        }
        const int64_t k = t->ne[0];
        const int64_t n = t->ne[1];
        if (k % 256 != 0) {
            // A quantized row must be a whole number of 256-element groups or the op silently
            // takes a padded path and the number is a phantom, the same trap fa_bench asserts
            // against for n_kv.
            fprintf(stderr, "amp-matvec-bench: %s has K=%lld, not a multiple of 256; skipping\n",
                    name, (long long) k);
            continue;
        }
        shapes.push_back({ name, (ggml_type) t->type, k, n });
    }

    if (shapes.empty()) {
        fprintf(stderr, "amp-matvec-bench: none of the reference tensors are present\n");
        return 1;
    }

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "amp-matvec-bench: no GPU backend\n");
        return 1;
    }

    printf("amp-matvec-bench\n");
    printf("  model   %s\n", model);
    printf("  card    %s\n\n", ggml_backend_name(backend));
    printf("  %-30s %-9s %9s %8s %10s %10s %8s\n",
           "tensor", "type", "K", "N", "ms", "GB/s", "%stream");

    for (const Shape & s : shapes) {
        double   base_ms = 0.0;
        const double base = measure_stream(backend, 512ull << 20, base_ms);
        const Row r = measure_matvec(backend, s, passes);
        if (r.ms <= 0.0) {
            continue;
        }
        printf("  %-30s %-9s %9lld %8lld %10.4f %10.1f %7.0f%%\n",
               s.label, ggml_type_name(s.type), (long long) s.k, (long long) s.n,
               r.ms, r.gbps, base > 0.0 ? 100.0 * r.gbps / base : 0.0);
        printf("       %zu MiB of weights, %.2f MiB/s/row-block-equivalent, checksum %llu\n",
               (size_t) (ggml_row_size(s.type, s.k) * (size_t) s.n) / (1024 * 1024),
               r.gbps, (unsigned long long) r.checksum);
    }

    printf("\n  batch 1, one query column. A rate far below %%stream is not a bandwidth\n"
           "  problem, it is the kernel not issuing enough memory in flight.\n");

    ggml_backend_free(backend);
    return 0;
}
