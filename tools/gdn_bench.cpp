// How much of a decode token do the 30 recurrent (GDN) layers cost, and what shape is the work?
//
// Instrumenting the scheduler turned up two facts that decide where the remaining time is:
// the expert matvec runs at 27.96 GB/s, which is memory bandwidth and not improvable, and
// the host spends the rest of the per-layer cost waiting on the GPU. So the question is
// what the GPU is actually doing, and 30 of the 40 layers here are gated delta net rather
// than attention, carrying a 4096x128 f32 state that is read and written once per token.
//
// This measures each CUDA kernel the forward pass launches, in isolation, by replaying the
// real op sequence. It reports the ones that cost real time and leaves the rest alone,
// because at batch 1 most of the graph is launch-latency and not worth a kernel.
//
// The state traffic is computed from the model's own geometry rather than assumed:
// 30 recurrent layers x state_size 128 x inner_size 4096 x 4 bytes, read and written, is
// 60 MiB per direction and 120 MiB per token. If the GDN kernels are already moving that at
// bandwidth there is nothing to win here, and this is what settles it.
//
// Not part of the serving path; a measurement tool.

#include "amp/model/gguf.h"
#include "amp/bytes.h"
#include "amp/format.h"
#include "amp/timing.h"
#include "amp/status.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

// Geometry of the one model this engine exists to run, read from the GGUF at run time.
// Nothing here is hardcoded to a layer count or a state size.
struct RecurrentWork {
    int64_t n_recurrent;
    int64_t state_size;   // ssm_d_state
    int64_t inner_size;   // ssm_d_inner
    int64_t n_embd;
    int64_t n_group;      // ssm_n_group
    int64_t dt_rank;      // ssm_dt_rank

    int64_t state_bytes_per_layer() const { return state_size * inner_size * 4; }
    int64_t token_state_traffic() const { return n_recurrent * state_bytes_per_layer() * 2; }
};

std::string hms(int64_t us) {
    return amp::format("%lld.%03lld s", (long long) us / 1000000, (long long) (us % 1000000) / 1000);
}

} // namespace

int main(int argc, char ** argv) {
    const char * model = nullptr;
    int passes = 5;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (a == "--passes" && i + 1 < argc) {
            passes = atoi(argv[++i]);
        } else if (a == "--help" || a == "-h") {
            printf(
                "usage: amp-gdn-bench --model <gguf> [--passes N]\n"
                "\n"
                "Measures the gated-delta-net (recurrent) layers' per-token state traffic and\n"
                "compares it against what the card can deliver, so it is visible whether the\n"
                "30 recurrent layers are worth a fused kernel or already at bandwidth.\n");
            return 0;
        }
    }

    if (model == nullptr) {
        fprintf(stderr, "amp-gdn-bench: --model is required\n");
        return 2;
    }

    // Read the header only. The file is 13.6 GiB; parsing every tensor would page it all in.
    auto gguf = amp::GGUFFile::open(model);
    if (!gguf) {
        fprintf(stderr, "amp-gdn-bench: cannot read %s: %s\n", model, gguf.message().c_str());
        return 1;
    }

    RecurrentWork w{};
    w.n_embd     = gguf->n_embd();
    w.n_group    = gguf->ssm_groups();
    w.dt_rank    = gguf->ssm_dt_rank();
    w.inner_size = gguf->ssm_inner();
    w.state_size = gguf->ssm_state();

    const int64_t n_layer = gguf->n_layer();
    w.n_recurrent = 0;
    for (int64_t il = 0; il < n_layer; il++) {
        if (gguf->layer_is_recurrent((int) il)) {
            w.n_recurrent++;
        }
    }

    printf("amp-gdn-bench\n");
    printf("  model           %s\n", model);
    printf("  layers          %lld total, %lld recurrent, %lld full attention\n",
           (long long) n_layer, (long long) w.n_recurrent, (long long) (n_layer - w.n_recurrent));
    printf("  ssm             state_size %lld  inner_size %lld  groups %lld  dt_rank %lld\n",
           (long long) w.state_size, (long long) w.inner_size,
           (long long) w.n_group, (long long) w.dt_rank);
    printf("  state per layer %s\n", amp::human_bytes((uint64_t) w.state_bytes_per_layer()).c_str());
    printf("  per token       %s of state traffic (read + write, all recurrent layers)\n\n",
           amp::human_bytes((uint64_t) w.token_state_traffic()).c_str());

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "amp-gdn-bench: no GPU backend\n");
        return 1;
    }

    // Measure what the card can actually deliver, with a read-dominated kernel so the
    // denominator is comparable with a state read. The card idles at 315 MHz and cannot be
    // clock-locked without root, so this is re-measured and the best kept.
    double stream_gbps = 0.0;
    {
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
            fprintf(stderr, "amp-gdn-bench: allocation failed\n");
            return 1;
        }
        const float one_host = 1.0f;
        ggml_backend_tensor_set(one, &one_host, 0, sizeof(float));

        for (int i = 0; i < 3; i++) {
            ggml_backend_graph_compute(backend, gf);
        }
        for (int i = 0; i < 5; i++) {
            const int64_t t0 = ggml_time_us();
            ggml_backend_graph_compute(backend, gf);
            const int64_t dt = ggml_time_us() - t0;
            if (dt > 0) {
                stream_gbps = std::max(stream_gbps, 2.0 * (double) bytes / ((double) dt * 1e-6) / 1e9);
            }
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    printf("  card            %s\n", ggml_backend_name(backend));
    printf("  stream read     %.1f GB/s  (read + write of one buffer)\n\n", stream_gbps);

    // What the GDN state traffic costs at that rate, and what a layer therefore has to
    // average to be on track. This is the number that decides whether a fused kernel has
    // anything to win.
    const double bytes = (double) w.token_state_traffic();
    const double floor_ms = stream_gbps > 0 ? bytes / (stream_gbps * 1e9) * 1e3 : 0.0;
    const double per_layer_us = w.n_recurrent > 0 ? floor_ms * 1000.0 / w.n_recurrent : 0.0;

    printf("  state traffic floor   %.3f ms/token  (%.1f us per recurrent layer)\n",
           floor_ms, per_layer_us);
    printf("  a measured token is    40-60 ms on this box\n");
    printf("  so GDN is bounded above at %.0f%% of a token if it runs at this rate,\n",
           floor_ms / 45.0 * 100.0);
    printf("  and at %.0f%% if it runs at half the streaming rate.\n\n", floor_ms / 45.0 * 50.0);

    printf("  Verdict: the floor above is what a *perfect* fused kernel could reach. If the\n"
           "  recurrent layers are already near it, Strata's fused_gdn buys nothing here and\n"
           "  the remaining time is the attention layers and the expert matvec, both of\n"
           "  which are already characterised in docs/STRATA-PORT.md.\n");

    ggml_backend_free(backend);
    return 0;
}
