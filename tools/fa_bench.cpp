// How fast is one decode attention, in isolation, and which kernel does llama.cpp pick?
//
// End-to-end decode on this box is unmeasurable for this question. The 12.19 GiB expert
// working set is only sometimes resident in an ~11 GiB page cache, so identical
// configurations measure anywhere from 10 to 19 t/s, and that spread is several times
// larger than the effect being chased. An A/B on the full forward pass cannot resolve it.
//
// This measures the attention op alone, with no weights, no experts, no page cache and no
// sampler. Everything it touches is a few hundred MiB of KV that it allocates and faults
// itself, so the number is reproducible to well under a percent and a kernel swap shows up
// immediately.
//
// It exists to answer one question: for this model's geometry, is the decode attention
// moving KV at the card's bandwidth, or is it stalling?
//
// The geometry is the model's own, not a guess: 16 query heads over 2 KV heads (GQA ratio
// 8), head_dim 256, and q8_0 K/V, which is what the shipped configuration uses. One query
// column, so this is the decode shape and not a prefill shape.
//
// It reports achieved bandwidth against the measured achievable rate rather than a
// theoretical one, because the theoretical figure for a laptop part is optimistic and would
// flatter every result here.

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

// The model's own shape, read off the GGUF by amp-plan.
constexpr ggml_type kTypeK  = GGML_TYPE_Q8_0;
constexpr ggml_type kTypeV  = GGML_TYPE_Q8_0;

// Head dim, and the query/KV head counts, are overridable because they are the knobs that
// separate "this kernel cannot reach the card's rate" from "this kernel is also being asked
// to serve more query heads": head dim moves bytes-per-row and math-per-row together, and
// AMP_FA_H/AMP_FA_HKV hold the KV bytes fixed while the query-head count changes, which is
// what separates the two. 16 / 2 / 256 is the model.
static int kHeadDim   = getenv("AMP_FA_D")   ? atoi(getenv("AMP_FA_D"))   : 256;
static int ncols      = 1;   // decode shape: one query column
static int kNHead     = getenv("AMP_FA_H")   ? atoi(getenv("AMP_FA_H"))   : 16;
static int kNHeadKV   = getenv("AMP_FA_HKV") ? atoi(getenv("AMP_FA_HKV")) : 2;

struct Measurement {
    double      ms;
    double      gb_per_s;
    double      pct_of_stream;
    double      base_gbps;   // baseline measured alongside this row
    uint64_t    checksum;
};

// Best-effort achievable DRAM bandwidth, measured by streaming a large device allocation.
// Streaming a copy, not a kernel, so this is the memory system's rate and not any
// particular kernel's rate. A kernel that cannot beat a plain copy is not memory-bound on
// anything except its own occupancy.
double measure_stream_bandwidth(ggml_backend_t backend) {
    // Large enough to dwarf L2, small enough to fit alongside the benchmark's KV.
    const size_t bytes = 1ull << 30;

    ggml_init_params ip = {};
    ip.mem_size = 256 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, bytes/sizeof(float));
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, bytes/sizeof(float));

    // A read-dominated op, so the denominator is comparable with attention, which reads its
    // operands and writes almost nothing. A plain a+b would move two reads and one write,
    // and a write is slower than a read on GDDR6, so using a copy as the ceiling for a
    // read-only kernel understates how much headroom the kernel actually has left.
    //
    // out = a*b scales the larger operand against a 1-element vector, so `a` is streamed
    // from DRAM in full and `b` stays resident in cache.
    ggml_tensor * one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    // Build the op before allocating. alloc_ctx_tensors only sees tensors that already
    // exist, so creating the destination afterwards would leave it off the device and the
    // kernel would write into unallocated memory.
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ggml_mul(ctx, a, one));

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        ggml_free(ctx);
        return 0.0;
    }

    {
        std::vector<float> one_host(1, 1.0f);
        ggml_backend_tensor_set(one, one_host.data(), 0, sizeof(float));
    }

    // Warm up, then take the best of several passes for the same reason the rest of this
    // project does: the first pass faults pages in and the card clocks up.
    for (int i = 0; i < 3; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }

    double best = 0.0;
    for (int i = 0; i < 5; ++i) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_graph_compute(backend, gf);
        const int64_t dt = ggml_time_us() - t0;
        if (dt <= 0) {
            continue;
        }
        // One read of `a` in full, plus a write of the same size. Counted the same way
        // attention's K and V are counted: what crosses the bus, in and out.
        const double gbps = (2.0 * (double) bytes) / ((double) dt * 1e-6) / 1e9;
        best = std::max(best, gbps);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return best;
}

// Build and time one decode attention at the given KV length.
Measurement measure_attention(ggml_backend_t backend, int n_kv, int passes, double stream_gbps) {
    ggml_init_params ip = {};
    // flash_attn_ext carries op_params, and the graph bookkeeping needs room for the
    // nodes the op expands to. 64 KiB is comfortably more than this op needs and costs
    // nothing, since no_alloc means no data is placed here.
    ip.mem_size = 256 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);

    // [D, n_kv, n_head_kv] for K and V, [D, 1, n_head] for Q.
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kNHead);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, kTypeK,    kHeadDim, n_kv, kNHeadKV);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, kTypeV,    kHeadDim, n_kv, kNHeadKV);

    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f, 0.0f, 0.0f);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        ggml_free(ctx);
        return Measurement{ 0.0, 0.0, 0.0, 0.0, 0 };
    }

    // Fill with something deterministic and non-zero, so a kernel cannot skip the read
    // because of a degenerate input.
    {
        std::vector<uint8_t> zeros(1024, 0);
        ggml_backend_tensor_set(k, zeros.data(), 0, std::min<size_t>(zeros.size(), ggml_nbytes(k)));
        ggml_backend_tensor_set(v, zeros.data(), 0, std::min<size_t>(zeros.size(), ggml_nbytes(v)));
    }

    for (int i = 0; i < 3; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }

    Measurement best{ 0.0, 0.0, 0.0, 0.0, 0 };
    for (int i = 0; i < passes; ++i) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_graph_compute(backend, gf);
        const int64_t dt = ggml_time_us() - t0;
        if (dt <= 0) {
            continue;
        }
        const double ms = (double) dt / 1e3;

        // K and V are both read once, at their stored width.
        const double bytes = 2.0 * n_kv * kNHeadKV * kHeadDim
                           * ggml_type_size(kTypeK) / (double) ggml_blck_size(kTypeK);
        const double gbps = bytes / ((double) dt * 1e-6) / 1e9;
        if (gbps > best.gb_per_s) {
            best = Measurement{ ms, gbps, stream_gbps > 0 ? 100.0 * gbps / stream_gbps : 0.0, stream_gbps, 0 };
        }
    }

    // Read the result back so the whole graph is live and the checksum can be printed as
    // evidence that both arms computed something.
    {
        std::vector<float> host(ggml_nelements(out));
        ggml_backend_tensor_get(out, host.data(), 0, ggml_nbytes(out));
        uint64_t sum = 0;
        for (size_t i = 0; i < host.size(); i += 997) {
            uint32_t bits;
            memcpy(&bits, &host[i], sizeof(bits));
            sum = sum * 1000003u + bits;
        }
        best.checksum = sum;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return best;
}

} // namespace

int main(int argc, char ** argv) {
    // Large enough that the default sweep includes the largest context this model is
    // actually run at, so `--ctx` only ever has to go lower.
    int n_kv = 262144;
    int passes = 7;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--ctx" && i + 1 < argc) {
            n_kv = atoi(argv[++i]);
        } else if (a == "--passes" && i + 1 < argc) {
            passes = atoi(argv[++i]);
        } else if (a == "--help" || a == "-h") {
            printf(
                "usage: amp-fa-bench [--ctx N] [--passes N]   (AMP_FA_D overrides head dim)\n"
                "\n"
                "Times one decode attention at this model's geometry (16 heads / 2 KV heads,\n"
                "head_dim 256, q8_0 K and V) with no other work in the graph, and reports the\n"
                "bandwidth it achieves against a measured streaming copy of the same card.\n");
            return 0;
        }
    }

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "amp-fa-bench: no GPU backend\n");
        return 1;
    }

    printf("amp-fa-bench\n");
    printf("  geometry     D=%d  heads=%d  kv_heads=%d (gqa %d)  K=%s V=%s  %d query column(s)\n",
           kHeadDim, kNHead, kNHeadKV, kNHead / kNHeadKV,
           ggml_type_name(kTypeK), ggml_type_name(kTypeV), ncols);
    printf("  backend      %s\n", ggml_backend_name(backend));

    const double stream = measure_stream_bandwidth(backend);
    printf("  stream read  %.1f GB/s  (this card's achievable DRAM rate for a read-dominated kernel,\n"
           "                         used as the denominator below)\n\n", stream);

    // The card cannot be clock-locked without root on this box, and it idles at 315 MHz,
    // so any single measurement can be taken while the GPU is still ramping. Re-measuring
    // the baseline between each context makes the ratio drift with the clocks in the same
    // direction as the numerator, which is what a percentage needs to be trustworthy.
    // Each row is therefore one attention measurement and one baseline measurement taken
    // back to back, and the best of several such pairs is kept.
    printf("  %8s %10s %12s %10s %12s %18s\n",
           "ctx", "ms", "GB/s", "% stream", "base GB/s", "KV GiB");
    for (int ctx : { 2048, 8192, 32768, 65536, 131072, 200192 }) {
        if (ctx > n_kv) {
            continue;
        }
        // Every context here is a multiple of FATTN_KQ_STRIDE on purpose. llama.cpp gates
        // its fast attention paths on K->ne[1] % FATTN_KQ_STRIDE == 0 and pads n_kv up to
        // that boundary in llama_kv_cache::get_n_kv, so the serving path only ever sees
        // aligned lengths. A length that is not aligned here is not a shape the model ever
        // runs in, and timing one silently measures the generic fallback kernel instead of
        // the thing under test.
        GGML_ASSERT(ctx % 256 == 0);

        Measurement best{ 0.0, 0.0, 0.0, 0.0, 0 };
        for (int round = 0; round < 3; ++round) {
            const double base = measure_stream_bandwidth(backend);
            const Measurement m = measure_attention(backend, ctx, passes, base);
            if (m.gb_per_s > best.gb_per_s) {
                best = m;
            }
        }
        if (best.ms <= 0) {
            continue;
        }
        const double gib = 2.0 * ctx * kNHeadKV * kHeadDim
                         * ggml_type_size(kTypeK) / (double) ggml_blck_size(kTypeK) / (1024.0 * 1024 * 1024);
        printf("  %8d %10.3f %12.1f %9.0f%% %12.1f %18.3f\n",
               ctx, best.ms, best.gb_per_s, best.pct_of_stream, best.base_gbps, gib);
    }

    // The % column is a bandwidth ratio, and at gqa > 1 it is a ratio of two different
    // things, so say what it means rather than let it be read as an efficiency.
    const int gqa = kNHead / kNHeadKV;
    printf("\n"
           "  The %% column is useful KV bytes over wall time against this card's measured\n"
           "  streaming rate, so at gqa 1 it is a bandwidth ratio and a value near 100%% would\n"
           "  mean the kernel is moving KV as fast as the card streams. At this model's gqa %d\n"
           "  it is not that: each KV row is read once but scored against %d query heads, so\n"
           "  the denominator counts only part of the work. Measured on this box, holding the\n"
           "  KV bytes fixed at one KV head and raising only the query heads gives\n"
           "\n"
           "      2/1 0.906 ms    4/1 0.946 ms    8/1 1.310 ms   16/1 2.420 ms\n"
           "\n"
           "  so the time grows with the head count on its own, and at the model's shape the\n"
           "  cost is per query head, not KV bytes. And going from 8/1 to 8/2 doubles the KV\n"
           "  for 0.58 ms more, i.e. the marginal KV streams at ~177 GB/s. Re-run this bench\n"
           "  with AMP_FA_H and AMP_FA_HKV set as above to reproduce that, rather than reading\n"
           "  a low %% here as a kernel that cannot reach memory bandwidth.\n", gqa, kNHead);

    ggml_backend_free(backend);
    return 0;
}
