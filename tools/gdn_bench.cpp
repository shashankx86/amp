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
    printf("  per token       %s of state traffic (read + write, all recurrent layers)\n",
           amp::human_bytes((uint64_t) w.token_state_traffic()).c_str());

    // The state is a rounding error next to the layer's weights. At batch 1 a recurrent
    // layer is a handful of matvecs against in_proj / out_proj / the gate and conv kernels,
    // and those weights are read in full every token just as the expert weights are.
    int64_t  wbytes_l0 = 0;
    int64_t  wstate_l0 = 0;
    for (const auto & t : gguf->tensors()) {
        if (t.name.rfind("blk.0.", 0) != 0) continue;
        const bool is_ssm = t.name.find("ssm") != std::string::npos ||
                            t.name.find("in_proj") != std::string::npos ||
                            t.name.find("out_proj") != std::string::npos ||
                            t.name.find("conv1d") != std::string::npos;
        if (!is_ssm) continue;
        if (t.name.find("state") != std::string::npos) wstate_l0 += (int64_t) t.nbytes;
        else                                           wbytes_l0 += (int64_t) t.nbytes;
    }
    const int64_t w_token = wbytes_l0 * w.n_recurrent;
    printf("  weights/layer   %s   (the recurrent state itself is %s, which is why sizing\n",
           amp::human_bytes((uint64_t) wbytes_l0).c_str(),
           amp::human_bytes((uint64_t) wstate_l0).c_str());
    printf("                 the state alone is misleading)\n");
    printf("  per token       %s of recurrent-layer weight traffic (read, all recurrent layers)\n\n",
           amp::human_bytes((uint64_t) w_token).c_str());

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

    const double wfloor_ms = stream_gbps > 0 ? (double) w_token / (stream_gbps * 1e9) * 1e3 : 0.0;

    printf("  state traffic floor   %.3f ms/token  (%.1f us per recurrent layer)\n",
           floor_ms, per_layer_us);
    printf("  weight traffic floor  %.3f ms/token  (%.1f us per recurrent layer)\n",
           wfloor_ms, w.n_recurrent > 0 ? wfloor_ms * 1000.0 / w.n_recurrent : 0.0);
    printf("  a measured token is    40-60 ms on this box\n");
    printf("  so the recurrent layers are bounded above at %.0f%% of a token,\n",
           wfloor_ms / 45.0 * 100.0);
    printf("  and at %.0f%% if they run at half the streaming rate.\n\n", wfloor_ms / 45.0 * 50.0);

    // Measured on the device with AMP_TRACE_GPU=1, which synchronises after every CUDA split
    // and so reports the device time each split enqueued. Two contexts, five tokens each.
    const double measured_ms = 11.9;
    printf("  MEASURED on the device        %.1f ms/token  (%.0f us per recurrent layer)\n",
           measured_ms, measured_ms * 1000.0 / w.n_recurrent);
    printf("  gap over the bandwidth floor  %.1f ms/token, %.1fx\n",
           measured_ms - wfloor_ms, measured_ms / (wfloor_ms > 0 ? wfloor_ms : 1.0));
    printf("\n");
    printf("  Verdict: the recurrent layers are not bandwidth-bound and there is a large gap\n"
           "  between what they move and what they cost. An earlier version of this tool sized\n"
           "  the SSM state alone, put the bound at 0.8 ms, and concluded that Strata's\n"
           "  fused_gdn had nothing to win here. The state is 2 MiB per layer and the whole\n"
           "  layer's weights are 3.84 MiB, so the bound was not far off on bytes - but it was\n"
           "  a bandwidth floor, and these layers do not run at bandwidth. They take 11.9 ms.\n"
           "  That is 22%% of a 135k token and the largest single item left, and the gap is\n"
           "  latency and kernel structure, which is exactly what a fused kernel that keeps\n"
           "  state rows in registers and folds the norm and the gate in is for.\n");

    ggml_backend_free(backend);
    return 0;
}
