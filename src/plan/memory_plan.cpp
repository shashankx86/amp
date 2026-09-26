#include "amp/plan/memory_plan.h"

#include "amp/bytes.h"
#include "amp/format.h"
#include "amp/log.h"
#include "amp/timing.h"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace amp {

namespace {

// VRAM used by everything except MoE experts: embeddings, output head, attention/SSM
// weights and the KV cache. The compute buffer is added per candidate.
int64_t vram_fixed_bytes(const ModelGeometry & geo, const PlannerOptions & opts) {
    const int64_t non_expert_weights = geo.total_other_bytes() - geo.router_bytes();
    const int64_t kv                 = geo.kv_bytes(opts.n_ctx, opts.cache_k, opts.cache_v);
    return non_expert_weights + kv;
}

uint64_t read_sysfs_uint(const char * path) {
    std::ifstream f(path);
    if (!f) {
        return 0;
    }
    unsigned long long v = 0;
    f >> v;
    return (uint64_t) v;
}

uint64_t nvidia_smi_mem(const char * field) {
    const std::string cmd = std::string("nvidia-smi --query-gpu=") + field +
                            " --format=csv,noheader,nounits 2>/dev/null | head -1";
    FILE * p = popen(cmd.c_str(), "r");
    if (!p) {
        return 0;
    }
    char buf[64] = { 0 };
    if (!fgets(buf, sizeof(buf), p)) {
        pclose(p);
        return 0;
    }
    pclose(p);
    return strtoull(buf, nullptr, 10);
}

int64_t cpu_expert_bytes(const ModelGeometry & geo, int32_t gpu_expert_layers) {
    int64_t total = 0;
    for (int64_t il = 0; il < geo.n_layer() - gpu_expert_layers; il++) {
        total += geo.layers()[(size_t) il].expert_bytes;
    }
    return total;
}

int64_t gpu_expert_bytes(const ModelGeometry & geo, int32_t gpu_expert_layers) {
    int64_t total = 0;
    for (int64_t il = geo.n_layer() - gpu_expert_layers; il < geo.n_layer(); il++) {
        total += geo.layers()[(size_t) il].expert_bytes;
    }
    return total;
}

// Chooses which expert ranges stay page-cache resident. Right now this is purely
// capacity-driven: with the whole CPU expert set exceeding the cache, *which* ranges we
// pin does not change prefill throughput (misses per ubatch = total - resident either way).
// It matters for decode, where reuse is local to the active experts, so once amp has
// router statistics this becomes frequency-ordered. That swap is deliberately isolated here.
std::vector<RangePlan> select_resident_ranges(const ModelGeometry & geo, int32_t gpu_expert_layers,
                                              int64_t budget_bytes) {
    std::vector<RangePlan> out;
    int64_t                used = 0;
    for (int64_t il = 0; il < geo.n_layer() - gpu_expert_layers; il++) {
        for (const auto & r : geo.expert_ranges((int) il)) {
            if (used + (int64_t) r.length > budget_bytes) {
                continue;  // try smaller ranges from later layers
            }
            RangePlan rp;
            rp.range  = r;
            rp.layer  = (int) il;
            rp.kind   = RangeKind::kExpert;
            rp.policy = ReadPolicy::kCacheWarm;
            out.push_back(rp);
            used += (int64_t) r.length;
        }
    }
    return out;
}

} // namespace

DeviceBudget detect_device_budget(const CostModelConstants & cost_constants) {
    DeviceBudget b;
    const MemInfo mi = read_meminfo();
    b.ram_total     = mi.mem_total_bytes;
    b.cache_free_now = mi.cache_headroom();
    // What we are willing to plan for: everything except the OS reserve, but never more than
    // RAM minus what non-cache processes currently hold plus what is already cached.
    const uint64_t potential = mi.mem_total_bytes > b.os_reserve ? mi.mem_total_bytes - b.os_reserve : 0;
    const uint64_t realistic = mi.cached_bytes + (mi.mem_total_bytes - mi.mem_available_bytes > mi.cached_bytes
                                                      ? mi.mem_total_bytes - mi.mem_available_bytes - mi.cached_bytes
                                                      : 0);
    // Plan for the *achievable* cache (RAM minus OS reserve), not for whatever happens to be free
    // this second: the operating mode we are planning for has other RAM users closed. The
    // momentary reading is still reported, as the "tg now" column, so the cost of not closing
    // them is visible rather than silently baked into the ranking.
    // Capped at the largest working set ever measured resident, so the cost model's overflow (decode
    // cliff) term can actually fire. See CostModelConstants::cache_ceiling_bytes.
    const uint64_t ceiling = (uint64_t) cost_constants.cache_ceiling_bytes;
    b.cache_budget    = potential < ceiling ? potential : ceiling;
    b.cache_potential = potential;
    b.cache_realistic = realistic;

    uint64_t total_mib = nvidia_smi_mem("memory.total");
    if (total_mib == 0) {
        const uint64_t sysfs = read_sysfs_uint("/sys/class/drm/card0/device/mem_info_vram_total");
        total_mib = sysfs / (1024 * 1024);
    }
    b.vram_total    = total_mib * kMiB;
    b.vram_reserved = nvidia_smi_mem("memory.used") * kMiB;
    if (b.vram_reserved == 0) {
        b.vram_reserved = 128ull * kMiB;
    }
    const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    b.n_cpu_threads = (uint32_t) std::max(1L, ncpu / 2);
    return b;
}

std::vector<CandidatePlan> MemoryPlanner::rank(const ModelGeometry & geo, const DeviceBudget & budget,
                                               const PlannerOptions & opts, const CostModel & cost,
                                               size_t keep_top) {
    std::vector<CandidatePlan> cands;

    const int64_t            n_layer  = geo.n_layer();
    const uint64_t           vram_cap = budget.vram_usable();
    const CostModelConstants cc       = cost.constants();
    // The CUDA context's own cost is included: it exists before any buffer of ours is allocated.
    // Measured as the gap between this estimate and the free VRAM the driver reports after context
    // init - 119 MiB free against a 5.56 GiB estimate at 200k context.
    const int64_t fixed = vram_fixed_bytes(geo, opts) + (int64_t) cc.vram_context_bytes;

    // Decode prediction as a function of the page cache we assume. Evaluated twice: once for
    // the budget we plan against, once for what is free right now, because the difference between
    // those two numbers is exactly the "close the browser" advice.
    auto predict_decode = [&](int64_t cpu_experts, int64_t cpu_layers, int64_t cache_avail) -> double {
        const double overflow = cpu_experts > 0
            ? std::max(0.0, (double) (cpu_experts - cache_avail) / (double) cpu_experts)
            : 0.0;
        const double tolerance = cc.cache_overflow_tolerance;
        double       resident_frac;
        if (overflow <= 0.0) {
            resident_frac = 1.0;
        } else if (overflow <= tolerance) {
            resident_frac = 1.0 - (overflow / tolerance) * 0.25;
        } else {
            const double t = (overflow - tolerance) / (1.0 - tolerance);
            resident_frac = 0.75 * (1.0 - std::min(1.0, t));
        }
        const uint64_t per_token_total = (uint64_t) ((double) cpu_experts *
                                                     (double) geo.n_expert_used() /
                                                     (double) std::max<int64_t>(1, geo.n_expert()));
        const uint64_t per_token_miss  = (uint64_t) ((double) per_token_total * (1.0 - resident_frac));
        return cost.decode_tps(cpu_layers, per_token_miss);
    };


    std::vector<int64_t> ubatches;
    for (int64_t ub = opts.ubatch_min; ub <= opts.ubatch_max; ub *= 2) {
        ubatches.push_back(ub);
    }
    for (int64_t ub : { 384, 640, 768, 896, 1280, 1536, 1792 }) {
        if (ub >= opts.ubatch_min && ub <= opts.ubatch_max) {
            ubatches.push_back(ub);
        }
    }
    std::sort(ubatches.begin(), ubatches.end());
    ubatches.erase(std::unique(ubatches.begin(), ubatches.end()), ubatches.end());

    const int32_t g_max = std::min<int32_t>(opts.max_expert_layers_gpu, (int32_t) n_layer);

    for (int32_t g = 0; g <= g_max; g++) {
        const int64_t gpu_experts = gpu_expert_bytes(geo, g);
        const int64_t cpu_experts = cpu_expert_bytes(geo, g);

        for (int64_t ub : ubatches) {
            CandidatePlan c;
            c.n_expert_layers_gpu = g;
            c.ubatch              = ub;
            c.expert_bytes_cpu    = cpu_experts;
            c.kv_bytes            = geo.kv_bytes(opts.n_ctx, opts.cache_k, opts.cache_v);
            c.compute_bytes = (int64_t) (cc.vram_bytes_per_ubatch_token * (double) ub) +
                                (int64_t) (cc.vram_bytes_per_gpu_expert_layer * (double) g);
            // The safety factor covers the *estimate* only. Weights and the KV cache are exact byte
            // counts read out of the GGUF, so applying a margin to the total lets a plan that is
            // physically too large look affordable - which is exactly what happened at 200k: a
            // 6.05 GiB plan was accepted against 5.89 GiB usable, and the runtime then had to halve
            // the ubatch three times to fit. Better to reject it here and pick the largest ubatch
            // that genuinely fits.
            c.vram_bytes = fixed + gpu_experts +
                           (int64_t) ((double) c.compute_bytes * cc.vram_safety);

            if ((uint64_t) c.vram_bytes > vram_cap) {
                c.fits          = false;
                c.reject_reason = format("vram %s > usable %s",
                                         human_bytes((uint64_t) c.vram_bytes).c_str(),
                                         human_bytes(vram_cap).c_str());
                cands.push_back(c);
                continue;
            }
            c.fits = true;

            // Residency: the CPU expert set versus what the page cache can hold.
            const int64_t target_resident =
                (int64_t) ((double) cpu_experts * clampd(opts.resident_fraction, 0.0, 1.0));
            const int64_t resident = std::min<int64_t>(target_resident, (int64_t) budget.cache_for_planning());
            c.expert_bytes_stream  = (!opts.allow_streaming) ? 0
                                     : (cpu_experts > resident ? (cpu_experts - resident) : 0);

            c.predicted_prefill_tps = cost.prefill_tps(2048, ub, (uint64_t) c.expert_bytes_stream, IoMode::kPrefetch);
            c.predicted_prefill_tps_faults =
                cost.prefill_tps(2048, ub, (uint64_t) c.expert_bytes_stream, IoMode::kPageFault);

            const int64_t cpu_layers = n_layer - g;
            c.predicted_decode_tps     = predict_decode(cpu_experts, cpu_layers, resident);
            c.predicted_decode_tps_now = predict_decode(cpu_experts, cpu_layers,
                                                        (int64_t) budget.cache_free_now);
            c.cpu_expert_bytes_planned = cpu_experts;
            c.cache_needed             = cpu_experts;

            c.score = cc.weight_prefill * std::log(std::max(0.01, c.predicted_prefill_tps)) +
                      cc.weight_decode * std::log(std::max(0.01, c.predicted_decode_tps));
            cands.push_back(c);
        }
    }

    std::stable_sort(cands.begin(), cands.end(),
                     [](const CandidatePlan & a, const CandidatePlan & b) {
                         if (a.fits != b.fits) {
                             return a.fits;
                         }
                         if (a.fits) {
                             return a.score > b.score;
                         }
                         return a.vram_bytes < b.vram_bytes;
                     });

    if (cands.size() > keep_top) {
        cands.resize(keep_top);
    }
    return cands;
}

Result<ExecutionPlan> MemoryPlanner::plan(const ModelGeometry & geo, const DeviceBudget & budget,
                                          const PlannerOptions & opts, const CostModel & cost) {
    const std::vector<CandidatePlan> cands = rank(geo, budget, opts, cost, 8);
    if (cands.empty()) {
        return Status::Error("planner produced no candidates");
    }

    ExecutionPlan p;
    p.top_candidates = cands;

    const CandidatePlan * best = nullptr;
    for (const auto & c : cands) {
        if (c.fits) {
            best = &c;
            break;
        }
    }
    if (!best) {
        std::string why;
        for (const auto & c : cands) {
            if (!c.reject_reason.empty()) {
                why = c.reject_reason;
                break;
            }
        }
        return Status::Errorf("no configuration fits: %s", why.c_str());
    }

    p.n_expert_layers_gpu = best->n_expert_layers_gpu;
    p.ubatch              = best->ubatch;
    p.cache_k             = opts.cache_k;
    p.cache_v             = opts.cache_v;
    p.kv_bytes            = best->kv_bytes;
    p.compute_bytes       = best->compute_bytes;
    p.vram_total          = best->vram_bytes;
    p.vram_budget         = (int64_t) budget.vram_usable();
    p.expert_bytes_gpu    = gpu_expert_bytes(geo, p.n_expert_layers_gpu);
    p.expert_bytes_cpu    = best->expert_bytes_cpu;
    p.resident_bytes      = p.expert_bytes_cpu - best->expert_bytes_stream;
    p.stream_bytes        = best->expert_bytes_stream;
    p.predicted_prefill_tps = best->predicted_prefill_tps;
    p.predicted_decode_tps  = best->predicted_decode_tps;

    p.cache_needed = p.expert_bytes_cpu;
    p.predicted_decode_tps_now = best->predicted_decode_tps_now;
    const double resident_frac =
        p.expert_bytes_cpu > 0 ? (double) p.resident_bytes / (double) p.expert_bytes_cpu : 1.0;
    p.decode_bytes_stream = (uint64_t) ((double) p.expert_bytes_cpu *
                                        (double) geo.n_expert_used() /
                                        (double) std::max<int64_t>(1, geo.n_expert()) *
                                        (1.0 - resident_frac));

    // resident / stream range lists for the CPU layers
    p.resident = select_resident_ranges(geo, p.n_expert_layers_gpu, p.resident_bytes);
    {
        std::vector<RangePlan> all = select_resident_ranges(geo, p.n_expert_layers_gpu,
                                                             p.expert_bytes_cpu);
        for (const auto & r : all) {
            bool is_resident = false;
            for (const auto & rr : p.resident) {
                if (rr.range == r.range) {
                    is_resident = true;
                    break;
                }
            }
            if (!is_resident) {
                RangePlan sp = r;
                sp.policy   = ReadPolicy::kStream;
                sp.note     = "expert bytes beyond cache capacity";
                p.stream.push_back(sp);
            }
        }
    }

    p.notes.push_back(format("expert layers on GPU: %d (highest %d layers), CPU expert bytes %s",
                             p.n_expert_layers_gpu, p.n_expert_layers_gpu,
                             human_bytes((uint64_t) p.expert_bytes_cpu).c_str()));
    p.notes.push_back(format("resident %s, stream %s per ubatch",
                             human_bytes((uint64_t) p.resident_bytes).c_str(),
                             human_bytes((uint64_t) p.stream_bytes).c_str()));
    if (p.predicted_decode_tps_now + 0.5 * p.predicted_decode_tps < p.predicted_decode_tps) {
        p.notes.push_back(format(
            "DECODE IS CACHE-BOUND RIGHT NOW: this plan needs %s of page cache and only %s is "
            "free; close the browser (or anything else holding RAM) to go from %.1f to %.1f t/s",
            human_bytes((uint64_t) p.cache_needed).c_str(), human_bytes(budget.cache_free_now).c_str(),
            p.predicted_decode_tps_now, p.predicted_decode_tps));
    }
    p.notes.push_back(format("ubatch %lld, compute buffer %s, total VRAM %s of %s free",
                             (long long) p.ubatch, human_bytes((uint64_t) p.compute_bytes).c_str(),
                             human_bytes((uint64_t) p.vram_total).c_str(),
                             human_bytes((uint64_t) p.vram_budget).c_str()));
    if (p.stream_bytes > 0) {
        p.notes.push_back("working set exceeds page cache: the tail will be streamed with async reads");
    }
    if (p.n_expert_layers_gpu == 0) {
        p.notes.push_back("no expert layers on GPU: maximum VRAM headroom, slowest decode");
    }

    AMP_INFO("amp: plan: ", p.to_string());
    return p;
}

std::string ExecutionPlan::to_string() const {
    std::string s = format(
        "g=%d ubatch=%lld vram=%s/%s pp=%.0f t/s tg=%.1f t/s resident=%s stream=%s",
        n_expert_layers_gpu, (long long) ubatch, human_bytes((uint64_t) vram_total).c_str(),
        human_bytes((uint64_t) vram_budget).c_str(), predicted_prefill_tps, predicted_decode_tps,
        human_bytes((uint64_t) resident_bytes).c_str(), human_bytes((uint64_t) stream_bytes).c_str());
    for (const auto & n : notes) {
        s += "\n    - " + n;
    }
    return s;
}

} // namespace amp
