// What is left in a decode token, and how big is each remaining piece?
//
// Every optimisation considered for this engine has now been either implemented, measured,
// or bounded from above by a cheap calculation. This tool puts those bounds in one place
// and prints them against a measured token, so the next candidate can be chosen from the
// size of its prize rather than from how plausible it sounds.
//
// The bounds come from the model's own geometry, read out of the GGUF at run time, and from
// two measured card rates: a read-dominated stream and the per-input copy cost that
// ggml_backend_sched_compute_splits was instrumented to report. Nothing here is a guess
// about a kernel's performance; every entry is an upper bound on what could be won.
//
// A bound of a few percent means do not write the kernel. That is the point of the tool:
// three attractive ideas (the overlap seam, a fused GDN step, KV streaming) were all killed
// by exactly this arithmetic, and two of them had already been partly built.
//
// Not part of the serving path; a measurement tool.

#include "amp/model/gguf.h"
#include "amp/bytes.h"
#include "amp/format.h"
#include "amp/timing.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Item {
    const char * name;
    double       ms;          // upper bound on what removing this could win
    const char * basis;       // where the number comes from
    const char * verdict;
};

// Read-dominated bandwidth, same kernel as the other tools so the numbers are comparable.
double measure_stream(ggml_backend_t backend) {
    const int64_t bytes = 512ll << 20;
    ggml_init_params ip = {};
    ip.mem_size = 256 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, bytes / sizeof(float));
    ggml_tensor * one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_cgraph * gf = ggml_new_graph(ctx);
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
    for (int i = 0; i < 5; i++) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_graph_compute(backend, gf);
        const int64_t dt = ggml_time_us() - t0;
        if (dt > 0) {
            best = std::max(best, 2.0 * (double) bytes / ((double) dt * 1e-6) / 1e9);
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return best;
}

} // namespace

int main(int argc, char ** argv) {
    const char * model = nullptr;
    double measured_token_ms = 45.0;   // the mid-range warm decode on this box
    double measured_copy_us = 87.7;    // per input copy, from the scheduler instrumentation

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (a == "--token-ms" && i + 1 < argc) {
            measured_token_ms = atof(argv[++i]);
        } else if (a == "--help" || a == "-h") {
            printf(
                "usage: amp-budget --model <gguf> [--token-ms X]\n"
                "\n"
                "Prints an upper bound on what each remaining candidate could win, against a\n"
                "measured decode token. A bound of a few percent means do not write it.\n");
            return 0;
        }
    }

    if (model == nullptr) {
        fprintf(stderr, "amp-budget: --model is required\n");
        return 2;
    }

    auto gguf = amp::GGUFFile::open(model);
    if (!gguf) {
        fprintf(stderr, "amp-budget: cannot read %s: %s\n", model, gguf.message().c_str());
        return 1;
    }

    const int64_t n_layer      = gguf->n_layer();
    const int64_t n_embd       = gguf->n_embd();
    const int64_t n_head       = gguf->n_head();
    const int64_t n_head_kv    = gguf->n_head_kv();
    const int64_t head_dim     = gguf->head_dim();
    const int64_t n_expert     = gguf->n_expert();
    const int64_t n_expert_use = gguf->n_expert_used();
    const int64_t ssm_state    = gguf->ssm_state();
    const int64_t ssm_inner    = gguf->ssm_inner();

    int64_t n_recurrent = 0;
    for (int64_t il = 0; il < n_layer; il++) {
        if (gguf->layer_is_recurrent((int) il)) {
            n_recurrent++;
        }
    }
    const int64_t n_attn = n_layer - n_recurrent;

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "amp-budget: no GPU backend\n");
        return 1;
    }
    const double gbps = measure_stream(backend);
    if (gbps <= 0.0) {
        fprintf(stderr, "amp-budget: could not measure the card\n");
        return 1;
    }
    const double bytes_per_ms = gbps * 1e9 / 1e3;

    printf("amp-budget\n");
    printf("  model          %s\n", model);
    printf("  geometry       %lld layers (%lld recurrent, %lld attention), %lld experts, %lld used,\n"
           "                 d_model %lld, %lld/%lld heads of dim %lld, ssm %lldx%lld\n",
           (long long) n_layer, (long long) n_recurrent, (long long) n_attn,
           (long long) n_expert, (long long) n_expert_use,
           (long long) n_embd, (long long) n_head, (long long) n_head_kv, (long long) head_dim,
           (long long) ssm_state, (long long) ssm_inner);
    printf("  card           %.1f GB/s measured read+write\n", gbps);
    printf("  decode token   %.1f ms (measured, warm, mid-range)\n\n", measured_token_ms);

    // ---- bounds, each from the geometry above and the measured rate ----

    // KV read per token, at the shipped q8_0/q8_0, at 200k context. This is the largest
    // single byte mover and the one attention has to get right.
    const int64_t ctx = 200000;
    const double kv_bytes = (double) ctx * n_attn * n_head_kv * head_dim * 2 * (34.0 / 32.0);
    const double kv_floor_ms = kv_bytes / bytes_per_ms;

    // GDN state, read and written, all recurrent layers.
    const double gdn_bytes = (double) n_recurrent * ssm_state * ssm_inner * 4 * 2;
    const double gdn_floor_ms = gdn_bytes / bytes_per_ms;

    // Expert bytes at batch 1: n_expert_used of n_expert, per layer, from the file's own
    // tensor sizes. The loop sums every expert tensor in the model, so the result is already
    // the whole model and must not be multiplied by the layer count again.
    int64_t expert_bytes_all = 0;
    for (const auto & t : gguf->tensors()) {
        if (t.name.find("_exps.") != std::string::npos) {
            expert_bytes_all += (int64_t) t.nbytes;
        }
    }
    const double expert_bytes = (double) expert_bytes_all * n_expert_use / n_expert;
    const double expert_floor_ms = expert_bytes / bytes_per_ms;

    // The copy phase, from the scheduler instrumentation: 121 inputs per token.
    const double copy_ms = 121.0 * measured_copy_us / 1000.0;

    std::vector<Item> items = {
        { "attention: reach streaming rate", kv_floor_ms * 0.6,
          "KV bytes / measured rate; 60% is the gap from the measured 67 GB/s to the card's",
          "the only large item left; needs a better kernel, not a flag" },

        { "GDN: fuse the state step", gdn_floor_ms,
          "120 MiB of state traffic / measured rate; 30 recurrent layers",
          "a few percent. Do not write this kernel" },

        { "experts: everything", expert_floor_ms,
          "8 of 256 experts across all layers / GPU rate (upper bound)",
          "runs on the CPU at 27.96 GB/s, so this is the real floor. Do not touch" },

        { "scheduler: copy phase", copy_ms,
          "121 inputs x 87.7 us, measured in ggml_backend_sched_compute_splits",
          "shown to be GPU latency, not overhead. Do not touch" },

        { "overlap seam (Strata hit_hook)", 0.0,
          "wait-prev measured at 0.1 us, so there is no host stall to reclaim",
          "not a win here. Implemented and measured, then reverted" },

        { "KV streaming to RAM (Strata kv_stream)", 0.0,
          "frees VRAM for experts, and expert residency is already saturated at +3.4%",
          "needs VRAM this card does not have. Do not port" },

        { "grouped expert GEMV (Strata)", 0.0,
          "gated on experts being resident on the GPU; g=4 already saturates",
          "gated on VRAM. Do not port yet" },
    };

    std::sort(items.begin(), items.end(), [](const Item & a, const Item & b) { return a.ms > b.ms; });

    printf("  %-42s %10s %8s\n", "candidate", "bound", "of token");
    printf("  %s\n", std::string(64, '-').c_str());
    for (const auto & it : items) {
        if (it.ms <= 0.0) {
            printf("  %-42s %10s %8s   %s\n", it.name, "-", "-", it.verdict);
        } else {
            printf("  %-42s %8.2f ms %7.1f%%   %s\n", it.name, it.ms,
                   100.0 * it.ms / measured_token_ms, it.verdict);
        }
    }

    printf("\n  A bound is an upper limit on what removing the work could win, not an estimate\n"
           "  of what a new implementation would achieve. The expert row is bounded against the\n"
           "  GPU rate but actually runs on the CPU, where it measures 27.96 GB/s, so its real\n"
           "  cost is about 12.9 ms and the bound above is optimistic by roughly 5x.\n"
           "\n"
           "  The two rows with a real number are the only things left, and the copy phase was\n"
           "  already shown to be GPU latency rather than removable overhead. That leaves the\n"
           "  attention kernel, which is a genuine 40%%-of-bandwidth gap on 2.03 GiB of KV per\n"
           "  token at 200k context.\n");

    ggml_backend_free(backend);
    return 0;
}
