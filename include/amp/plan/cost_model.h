// CostModel: amp's analytic model of where time goes on this machine.
//
// Every constant here is a measurement from ../NOTES.md, not a guess. The model exists so
// the planner can rank thousands of (ubatch, device-split) candidates in microseconds
// instead of benchmarking each one for minutes.
#pragma once

#include <cstdint>
#include <string>

namespace amp {

// How bytes that miss the page cache reach the compute units. This distinction is the
// core of amp's advantage over llama.cpp and must not be blurred in the model:
//
//   kPageFault - llama.cpp today: 4 KiB synchronous faults on the compute thread, 555 MB/s,
//                essentially no overlap with compute.
//   kPrefetch  - amp: 440 KiB sequential reads issued ahead of use, ~2.0 GB/s, overlapped
//                with compute. The same bytes, three times the bandwidth.
enum class IoMode { kPageFault, kPrefetch };

struct CostModelConstants {
    // Prefill, cache-resident (i.e. pure compute). Measured: 11,568-token prompt in 47.6 s.
    double prefill_ceiling_tps      = 240.0;
    // Ubatch at which GEMM efficiency saturates (measured sweet spot).
    int64_t ubatch_saturation      = 1024;
    double ubatch_efficiency_floor = 0.35;
    // NVMe under *demand paging with readahead* (measured with mincore, this is the path amp
    // and llama.cpp both use): 1737-1804 MiB/s cold, 100% of pages stay resident afterwards.
    // Do NOT use the 555 MB/s O_DIRECT 4K-QD32 fio number here: that is random access with no
    // readahead, which is not what expert streaming looks like.
    double bandwidth_fault_bps     = 1.85e9;
    // Fully resident: pure RAM copy speed of the same range (measured 3516-3620 MiB/s).
    double bandwidth_resident_bps  = 3.5e9;
    // Explicit 440 KiB async reads issued ahead of use (amp's IReadScheduler): same bytes as
    // the fault path but decoupled from the compute thread, so this is a ceiling, not a rate.
    double bandwidth_seq_bps       = 2.0e9;
    // Fraction of I/O time that hides behind compute. Demand faults are synchronous (0);
    // explicit prefetch overlaps (measured-shape estimate, calibrate in M5).
    double io_overlap              = 0.75;
    // Decode: measured 11-14 t/s with 38 of 40 layers' experts on the CPU.
    double decode_ms_per_cpu_layer = 1.79;
    // Decode fault latency when misses are *not* hidden. Measured: 2.8 t/s (357 ms/token) with
    // the CPU expert set exceeding the page cache, against 30 t/s (33 ms/token) when it fits.
    // That is ~28 us per 4 KiB fault: once the working set stops fitting, decode is
    // fault-latency-bound, not bandwidth-bound.
    double decode_fault_latency_us = 28.0;
    // The working set is scanned cyclically, so once it exceeds the cache, LRU reuse collapses
    // instead of degrading linearly. This cliff is what makes "how many expert layers go on the
    // GPU" a decode decision and not merely a VRAM one.
    double cache_overflow_tolerance = 0.03;  // up to 3% over is still fine
    // Compute buffer cost per ubatch token (measured: 1513 MiB at ubatch 2048 with 2-6 expert
    // layers on the GPU, 1992 MiB with 8 - the MoE gather/scatter intermediates grow with the
    // number of GPU-resident expert layers).
    double vram_bytes_per_ubatch_token = 0.72 * 1024 * 1024;
    // Extra compute-buffer bytes per GPU-resident expert layer (~80 MiB/layer at ubatch 2048).
    double vram_bytes_per_gpu_expert_layer = 80.0 * 1024 * 1024;
    // Never plan right up to the edge: the allocator fragments, and a failed context init costs
    // far more than a slightly smaller ubatch.
    double vram_safety = 0.92;
    // What a CUDA context costs before any of amp's own buffers exist: the driver's own allocation
    // plus the context's bookkeeping. Measured here as the gap between the planner's estimate and the
    // free VRAM actually reported after llama_init_from_model(): 119 MiB free against a 5.56 GiB
    // estimate at 200k context, i.e. ~400-500 MiB unaccounted for. Ignoring it makes the planner pick
    // plans that the runtime then has to halve twice to fit.
    double vram_context_bytes = 450.0 * 1024 * 1024;
    // Ceiling on the page cache we plan against. The theoretical figure (RAM minus the OS reserve) is
    // ~12.9 GiB here, but the largest working set ever measured resident on this box is 11.15 GiB, and
    // the decode cliff is real: 10.25 GiB of CPU experts gave 25.6-29.9 t/s, 11.54 GiB gave 2.8-4.8.
    // Planning against the theoretical number therefore disables the cliff term in the cost model -
    // every candidate looks fully resident, including the ones that thrash.
    double cache_ceiling_bytes = 11.2 * 1024 * 1024 * 1024;
    // Score weights for the planner.
    double weight_prefill = 0.65;
    double weight_decode  = 0.35;
};

class CostModel {
public:
    CostModel() = default;
    explicit CostModel(const CostModelConstants & c) : c_(c) {}

    // Defaults measured on this box; env overrides: AMP_CEILING_TPS, AMP_BW_RANDOM_MBS, ...
    static CostModel for_this_machine();
    static CostModel from_environment();

    const CostModelConstants & constants() const { return c_; }

    // GEMM efficiency relative to the saturation point.
    double ubatch_efficiency(int64_t ubatch) const;

    // Prefill time for n tokens, split into ubatches of `ubatch`, where
    // `stream_bytes_per_ubatch` must be fetched from NVMe each ubatch.
    double prefill_ms(int64_t n_tokens, int64_t ubatch, uint64_t stream_bytes_per_ubatch,
                      IoMode mode = IoMode::kPageFault) const;
    // tokens/second (prefill_ms returns milliseconds)
    double prefill_tps(int64_t n_tokens, int64_t ubatch, uint64_t stream_bytes_per_ubatch,
                       IoMode mode = IoMode::kPageFault) const;

    // Decode: one token at a time. n_cpu_layers do expert work on the CPU,
    // `bytes_per_token` may still have to come from NVMe.
    double decode_ms(int64_t n_cpu_layers, uint64_t bytes_per_token) const;
    double decode_tps(int64_t n_cpu_layers, uint64_t bytes_per_token) const;

    // Bytes the CPU must pull per generated token: every active expert of every CPU layer.
    uint64_t decode_bytes_per_token(int64_t n_layer, int64_t n_expert_used,
                                    uint64_t bytes_per_expert_layer) const;

    std::string describe() const;

private:
    CostModelConstants c_{};
};

} // namespace amp
