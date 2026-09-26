// MemoryPlanner: given the model, the box, and the goal, decide
//   - how many MoE expert layers live on the GPU,
//   - how big the ubatch is,
//   - which byte ranges must stay page-cache resident and which may be streamed.
//
// The key non-obvious result (measured, see ../NOTES.md 5.3): VRAM spent on a larger
// ubatch beats VRAM spent on GPU-resident experts, because NVMe traffic per token scales
// as 1/ubatch while GPU residency only helps decode. The planner searches that trade-off
// explicitly instead of hard-coding it.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "amp/io/stream_reader.h"
#include "amp/model/geometry.h"
#include "amp/plan/cost_model.h"
#include "amp/status.h"

namespace amp {

struct DeviceBudget {
    uint64_t vram_total    = 0;   // bytes
    uint64_t vram_reserved = 0;   // desktop / other processes
    uint64_t ram_total     = 0;
    uint64_t os_reserve    = 2ull * 1024 * 1024 * 1024;  // RAM we refuse to plan to use
    uint64_t cache_potential = 0;   // RAM minus OS reserve: what we could hold if idle
    uint64_t cache_realistic = 0;   // what is actually achievable right now
    uint64_t cache_free_now = 0;  // page cache headroom detected right now
    uint64_t cache_budget  = 0;   // what planning assumes (potential, not momentary)
    uint32_t n_cpu_threads = 8;

    uint64_t vram_usable() const {
        return vram_total > vram_reserved ? vram_total - vram_reserved : vram_total;
    }
    // planning should use the budget, not the momentary reading: a browser holding 3 GiB
    // must not convince the planner that the model can never be resident.
    uint64_t cache_for_planning() const { return cache_budget > 0 ? cache_budget : cache_free_now; }
};

// Reads /proc/meminfo and the NVIDIA driver for the real numbers.
// `cost_constants` supplies the page-cache ceiling the plan is held to.
DeviceBudget detect_device_budget(const CostModelConstants & cost_constants = CostModelConstants{});

struct PlannerOptions {
    int64_t   n_ctx          = 200000;
    CacheType cache_k        = CacheType::kQ8_0;
    CacheType cache_v        = CacheType::kQ4_0;
    int64_t   ubatch_min     = 256;
    int64_t   ubatch_max     = 2048;
    int32_t   max_expert_layers_gpu = 8;
    bool      prefer_decode  = false;  // tie-break towards generation
    bool      allow_streaming = true;   // stream the non-resident tail instead of thrashing
    // fraction of the CPU expert set we insist on keeping resident (0..1]
    double    resident_fraction = 1.0;
};

enum class RangeKind { kExpert, kOther };

struct RangePlan {
    ReadRange  range;
    int        layer    = -1;
    RangeKind  kind     = RangeKind::kOther;
    ReadPolicy policy   = ReadPolicy::kCacheWarm;
    std::string note;
};

struct CandidatePlan {
    int32_t   n_expert_layers_gpu = 0;
    int64_t   ubatch              = 0;
    int64_t   vram_bytes          = 0;
    int64_t   compute_bytes       = 0;
    int64_t   kv_bytes            = 0;
    int64_t   expert_bytes_cpu    = 0;
    int64_t   expert_bytes_stream = 0;
    double    predicted_prefill_tps = 0.0;              // with amp's prefetching
    double    predicted_prefill_tps_faults = 0.0;      // llama.cpp-equivalent (page faults)
    double    predicted_decode_tps  = 0.0;   // at the planned cache budget
    double    predicted_decode_tps_now = 0.0; // at the cache free *right now*
    int64_t   cpu_expert_bytes_planned = 0;
    int64_t   cache_needed        = 0;   // page cache this candidate requires for fast decode
    double    score               = 0.0;
    bool      fits                = false;
    std::string reject_reason;
};

struct ExecutionPlan {
    int32_t   n_expert_layers_gpu = 0;  // layers [n_layer-g, n_layer) on GPU
    int64_t   ubatch              = 0;
    CacheType cache_k             = CacheType::kQ8_0;
    CacheType cache_v             = CacheType::kQ4_0;

    int64_t   kv_bytes            = 0;
    int64_t   compute_bytes       = 0;
    int64_t   vram_total          = 0;
    int64_t   vram_budget         = 0;
    int64_t   expert_bytes_gpu    = 0;
    int64_t   expert_bytes_cpu    = 0;
    int64_t   resident_bytes      = 0;   // expert bytes we expect to keep cached
    int64_t   stream_bytes        = 0;   // expert bytes that may hit NVMe per ubatch
    int64_t   decode_bytes_stream = 0;   // per generated token

    double    predicted_prefill_tps = 0.0;
    double    predicted_decode_tps  = 0.0;      // at the planned cache budget
    double    predicted_decode_tps_now = 0.0;   // with the cache free right now
    int64_t   cache_needed          = 0;       // page cache this plan wants for fast decode

    std::vector<RangePlan> resident;   // warm these at start-up / keep them hot
    std::vector<RangePlan> stream;     // fetch these per ubatch (or per token for decode)
    std::vector<std::string> notes;
    std::vector<CandidatePlan> top_candidates;

    bool uses_streaming() const { return !stream.empty(); }
    std::string to_string() const;
};

class MemoryPlanner {
public:
    static Result<ExecutionPlan> plan(const ModelGeometry & geo, const DeviceBudget & budget,
                                      const PlannerOptions & opts, const CostModel & cost);

    // Exposed for tools/tests: rank all candidates without picking one.
    static std::vector<CandidatePlan> rank(const ModelGeometry & geo, const DeviceBudget & budget,
                                           const PlannerOptions & opts, const CostModel & cost,
                                           size_t keep_top = 8);
};

} // namespace amp
