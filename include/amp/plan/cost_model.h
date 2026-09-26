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
    // NVMe, mmap random 4 KiB faults (fio, O_DIRECT, sustained).
    double bandwidth_random_bps    = 555e6;
    // NVMe, sequential / large-chunk reads (fio 128 KiB-1 MiB QD16-32).
    double bandwidth_seq_bps       = 2000e6;
    // Fraction of I/O time that hides behind compute when prefetching well.
    double io_overlap              = 0.75;
    // Decode: measured 11-14 t/s with 38 of 40 layers' experts on the CPU.
    double decode_ms_per_cpu_layer = 1.79;
    // Compute buffer cost per ubatch token (measured: 1513 MiB at ub 2048).
    double vram_bytes_per_ubatch_token = 0.72 * 1024 * 1024;
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
