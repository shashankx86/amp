#include "amp/plan/cost_model.h"

#include "amp/bytes.h"
#include "amp/format.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace amp {

namespace {
double env_double(const char * name, double def) {
    const char * v = getenv(name);
    return v ? atof(v) : def;
}
} // namespace

CostModel CostModel::for_this_machine() {
    return CostModel(CostModelConstants{});
}

CostModel CostModel::from_environment() {
    CostModelConstants c;
    c.prefill_ceiling_tps        = env_double("AMP_CEILING_TPS", c.prefill_ceiling_tps);
    c.bandwidth_fault_bps        = env_double("AMP_BW_FAULT_MBS", c.bandwidth_fault_bps / 1e6) * 1e6;
    c.bandwidth_resident_bps     = env_double("AMP_BW_RESIDENT_MBS", c.bandwidth_resident_bps / 1e6) * 1e6;
    c.bandwidth_seq_bps          = env_double("AMP_BW_SEQ_MBS", c.bandwidth_seq_bps / 1e6) * 1e6;
    c.io_overlap                 = env_double("AMP_IO_OVERLAP", c.io_overlap);
    c.vram_bytes_per_ubatch_token = env_double("AMP_VRAM_PER_UBATCH_TOKEN_MB",
                                               c.vram_bytes_per_ubatch_token / (1024.0 * 1024.0)) *
                                   1024.0 * 1024.0;
    c.vram_context_bytes = env_double("AMP_VRAM_CONTEXT_MB",
                                     c.vram_context_bytes / (1024.0 * 1024.0)) *
                           1024.0 * 1024.0;
    c.cache_ceiling_bytes = env_double("AMP_CACHE_CEILING_GB",
                                      c.cache_ceiling_bytes / (1024.0 * 1024.0 * 1024.0)) *
                           1024.0 * 1024.0 * 1024.0;
    c.decode_ms_per_cpu_layer    = env_double("AMP_DECODE_MS_PER_LAYER", c.decode_ms_per_cpu_layer);
    return CostModel(c);
}

double CostModel::ubatch_efficiency(int64_t ubatch) const {
    if (ubatch <= 0) {
        return c_.ubatch_efficiency_floor;
    }
    // Sub-linear: small batches are latency- and cache-inefficient, large ones saturate.
    const double ratio = (double) std::min<int64_t>(ubatch, c_.ubatch_saturation) /
                         (double) c_.ubatch_saturation;
    const double eff  = std::pow(std::max(ratio, 1e-3), 0.6);
    return std::min(1.0, std::max(c_.ubatch_efficiency_floor, eff));
}

double CostModel::prefill_ms(int64_t n_tokens, int64_t ubatch, uint64_t stream_bytes_per_ubatch,
                             IoMode mode) const {
    if (n_tokens <= 0) {
        return 0.0;
    }
    const int64_t ub = std::max<int64_t>(1, ubatch);
    const int64_t n_ubatches = (n_tokens + ub - 1) / ub;

    // compute
    const double tps_compute = c_.prefill_ceiling_tps * ubatch_efficiency(ub);
    const double t_compute   = (double) n_tokens / tps_compute;

    // io: only the bytes we cannot keep resident
    double t_io = 0.0;
    if (stream_bytes_per_ubatch > 0) {
        const bool   prefetch = (mode == IoMode::kPrefetch);
        const double bw       = prefetch ? c_.bandwidth_seq_bps : c_.bandwidth_fault_bps;
        const double hidden   = prefetch ? c_.io_overlap : 0.0;
        const double secs     = (double) (stream_bytes_per_ubatch * (uint64_t) n_ubatches) / bw;
        t_io                  = secs * (1.0 - hidden);
    }

    return (t_compute + t_io) * 1e3;  // milliseconds
}

double CostModel::prefill_tps(int64_t n_tokens, int64_t ubatch, uint64_t stream_bytes_per_ubatch,
                              IoMode mode) const {
    const double ms = prefill_ms(n_tokens, ubatch, stream_bytes_per_ubatch, mode);
    return ms > 0 ? (double) n_tokens * 1e3 / ms : 0.0;
}

double CostModel::decode_ms(int64_t n_cpu_layers, uint64_t bytes_per_token) const {
    const double t_compute = c_.decode_ms_per_cpu_layer * (double) std::max<int64_t>(0, n_cpu_layers);
    if (bytes_per_token == 0) {
        return t_compute;
    }
    // bytes_per_token is the *missing* byte count, already scaled by the reuse factor the caller
    // derived. Cost it as page faults rather than bandwidth: a 4 KiB synchronous fault costs
    // ~28 us of latency, and that is what actually dominates decode once the cache is exceeded.
    const double t_faults = ((double) bytes_per_token / 4096.0) * c_.decode_fault_latency_us / 1e3;
    const double t_io     = t_faults * (1.0 - 0.5 * c_.io_overlap);  // half hides behind compute
    return std::max(t_compute, t_io) + 0.25 * std::min(t_compute, t_io);
}

double CostModel::decode_tps(int64_t n_cpu_layers, uint64_t bytes_per_token) const {
    const double ms = decode_ms(n_cpu_layers, bytes_per_token);
    return ms > 0 ? 1e3 / ms : 0.0;
}

uint64_t CostModel::decode_bytes_per_token(int64_t n_layer, int64_t n_expert_used,
                                           uint64_t bytes_per_expert_layer) const {
    // Every token routes to n_expert_used experts per layer; each expert costs
    // bytes_per_expert_layer / n_expert. Without per-expert info we approximate with the
    // per-layer figure divided by the expert count, which is what the caller passes in.
    (void) n_layer;
    if (n_expert_used <= 0) {
        return 0;
    }
    return bytes_per_expert_layer * (uint64_t) n_expert_used;
}

std::string CostModel::describe() const {
    return format(
        "ceiling=%.0f t/s  bw_fault=%.0f MB/s  bw_resident=%.0f MB/s  bw_prefetch=%.0f MB/s  "
        "overlap=%.2f  vram/ubatch-token=%.2f MiB  decode=%.2f ms/cpu-layer",
        c_.prefill_ceiling_tps, c_.bandwidth_fault_bps / 1e6, c_.bandwidth_resident_bps / 1e6,
        c_.bandwidth_seq_bps / 1e6, c_.io_overlap,
        c_.vram_bytes_per_ubatch_token / (1024.0 * 1024.0), c_.decode_ms_per_cpu_layer);
}

} // namespace amp
