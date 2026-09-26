#include "amp/plan/preflight.h"

#include "amp/bytes.h"
#include "amp/log.h"
#include "amp/plan/memory_plan.h"
#include "amp/timing.h"

#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cstdlib>
#include <list>
#include <string>
#include <vector>

namespace amp {

namespace {

// ---------------------------------------------------------------------------
// argv scan: the only reliable "did the user set this flag" test.
//
// llama.cpp's parser keeps no record of which args were supplied (seen_args at
// common/arg.cpp:814 is local to parse_cli_args and discarded on return), and the
// fields we care about have no surviving unset sentinel: postprocess_cpu_params
// resolves n_threads < 0 during the parse itself (common/common.cpp:291-298), and
// n_batch/n_ubatch/n_ctx default to real values (common/common.h:450-452). So we
// scan argv. llama.cpp's parser matches whole tokens and does not accept
// "--flag=value" (common/arg.cpp:819-825), so a successfully-parsed argv only
// contains the two-token form; the "=" form is handled defensively anyway.
// ---------------------------------------------------------------------------

bool argv_has(const std::vector<std::string> & argv, const std::string & flag) {
    for (const auto & a : argv) {
        if (a == flag || a.rfind(flag + "=", 0) == 0) {
            return true;
        }
    }
    return false;
}

const char * argv_value(const std::vector<std::string> & argv, const std::string & flag) {
    for (size_t i = 0; i < argv.size(); i++) {
        const std::string & a = argv[i];
        if (a == flag && i + 1 < argv.size()) {
            return argv[i + 1].c_str();
        }
        if (a.rfind(flag + "=", 0) == 0) {
            return a.c_str() + flag.size() + 1;
        }
    }
    return nullptr;
}

struct UserFlags {
    bool fit_on = false;   // explicit "--fit on": the fitter is in charge, full no-op
    bool fit_off = false;  // explicit "--fit off"
    bool c = false, b = false, ub = false;
    bool ctk = false, ctv = false, fa = false;
    bool t = false, tb = false;
    bool ctxcp = false, cms = false, cram = false, lzm = false;
};

UserFlags scan_user_flags(const std::vector<std::string> & argv) {
    UserFlags f;
    f.c     = argv_has(argv, "-c") || argv_has(argv, "--ctx-size");
    f.b     = argv_has(argv, "-b") || argv_has(argv, "--batch-size");
    f.ub    = argv_has(argv, "-ub") || argv_has(argv, "--ubatch-size");
    f.ctk   = argv_has(argv, "-ctk") || argv_has(argv, "--cache-type-k");
    f.ctv   = argv_has(argv, "-ctv") || argv_has(argv, "--cache-type-v");
    f.fa    = argv_has(argv, "-fa") || argv_has(argv, "--flash-attn");
    f.t     = argv_has(argv, "-t") || argv_has(argv, "--threads");
    f.tb    = argv_has(argv, "-tb") || argv_has(argv, "--threads-batch");
    f.ctxcp = argv_has(argv, "-ctxcp") || argv_has(argv, "--ctx-checkpoints") || argv_has(argv, "--swa-checkpoints");
    f.cms   = argv_has(argv, "-cms") || argv_has(argv, "--checkpoint-min-step");
    f.cram  = argv_has(argv, "-cram") || argv_has(argv, "--cache-ram");
    f.lzm   = argv_has(argv, "-lzm") || argv_has(argv, "--lazy-mode");
    if (argv_has(argv, "-fit") || argv_has(argv, "--fit")) {
        const char * v = argv_value(argv, "-fit");
        if (!v) {
            v = argv_value(argv, "--fit");
        }
        // llama.cpp's own truthiness (common/arg.cpp:1331-1337); the -fit handler only
        // accepts on/off (common/arg.cpp:2864-2870), so one of the two always matches.
        f.fit_on  = v && common_arg_utils::is_truthy(v);
        f.fit_off = v && common_arg_utils::is_falsey(v);
    }
    return f;
}

ggml_type to_ggml(CacheType t) {
    switch (t) {
        case CacheType::kF32:    return GGML_TYPE_F32;
        case CacheType::kF16:    return GGML_TYPE_F16;
        case CacheType::kBF16:   return GGML_TYPE_BF16;
        case CacheType::kQ8_0:   return GGML_TYPE_Q8_0;
        case CacheType::kQ5_1:   return GGML_TYPE_Q5_1;
        case CacheType::kQ5_0:   return GGML_TYPE_Q5_0;
        case CacheType::kQ4_1:   return GGML_TYPE_Q4_1;
        case CacheType::kQ4_0:   return GGML_TYPE_Q4_0;
        case CacheType::kQ4_K:   return GGML_TYPE_Q4_K;
        case CacheType::kIQ4_NL: return GGML_TYPE_IQ4_NL;
        case CacheType::kQ3_K:   return GGML_TYPE_Q3_K;
        case CacheType::kQ2_K:   return GGML_TYPE_Q2_K;
        case CacheType::kQ6_K:   return GGML_TYPE_Q6_K;
    }
    return GGML_TYPE_F16;
}

// Fills params.tensor_buft_overrides with "pin the experts of layers [0, n_cpu) to the
// CPU" - the public-API equivalent of llama.cpp's -ncmoe.
//
// The vector was already padded to llama_max_tensor_buft_overrides() (4096,
// src/llama.cpp:90-92) with {nullptr, nullptr} entries by common_params_parse_ex
// (common/arg.cpp:940-944), which runs inside common_params_parse BEFORE this
// preflight. We write into the padded slots in place and keep the tail null.
//
// Two landmines this function exists to avoid:
//   - common_model_params_to_llama asserts back().pattern == nullptr
//     (common/common.cpp:1706) and GGML_ASSERT is always active (ggml/include/ggml.h:288),
//     so a non-null back() aborts the process before the model loads.
//   - the tensor loader walks the array until pattern == nullptr
//     (src/llama-model-loader.cpp:1237) and builds a std::regex from each pattern, so a
//     missing sentinel reads past the end: segfault or silent garbage.
// We never push_back past the sentinel; n_cpu <= 40 layers * 3 tensors = 120 entries,
// far below the 4096 slots.
void write_expert_cpu_overrides(common_params & params, int64_t n_cpu) {
    auto & ov = params.tensor_buft_overrides;
    // Defensive: the padding is supposed to be there. Restore it rather than overflow.
    const size_t ntbo = llama_max_tensor_buft_overrides();
    while (ov.size() < ntbo) {
        ov.push_back({nullptr, nullptr});
    }
    // The pattern strings must outlive the load. llama.cpp's own helper keeps them in a
    // static list for the same reason (common/common.h:1147-1154); a std::list never
    // moves existing nodes, so the c_str() pointers stay valid.
    static std::list<std::string> patterns;
    const ggml_backend_buffer_type_t cpu = ggml_backend_cpu_buffer_type();
    static const char * suffixes[] = { "ffn_gate_exps.weight", "ffn_up_exps.weight",
                                       "ffn_down_exps.weight" };
    size_t i = 0;
    for (int64_t il = 0; il < n_cpu; il++) {
        for (const char * suffix : suffixes) {
            patterns.push_back("blk." + std::to_string(il) + "." + suffix);
            if (i < ov.size()) {
                ov[i].pattern = patterns.back().c_str();
                ov[i].buft    = cpu;
            }
            i++;
        }
    }
    // Re-null the tail (already null from the padding; belt and braces).
    if (i < ov.size()) {
        ov[i].pattern = nullptr;
        ov[i].buft    = nullptr;
    }
}

bool user_set_layout(const common_params & params) {
    // n_gpu_layers: -1 is "auto" and also the unset default (common/common.h:473);
    // anything else is an explicit layer count. -ngl auto / -ngl -1 both leave -1.
    if (params.n_gpu_layers != -1) {
        return true;
    }
    // tensor_buft_overrides: after parse every entry is null unless the user passed
    // -ncmoe/-cmoe/-ncffn/-ot (arg.cpp:2752-2782), which push to the front.
    return !params.tensor_buft_overrides.empty() &&
           params.tensor_buft_overrides[0].pattern != nullptr;
}

} // namespace

Result<std::vector<std::string>> apply_preflight(const PreflightOptions & opts,
                                                 common_params & params,
                                                 const std::vector<std::string> & argv) {
    std::vector<std::string> notes;
    auto note = [&](const std::string & s) {
        notes.push_back(s);
        AMP_INFO("amp-preflight: ", s);
    };
    const UserFlags f = scan_user_flags(argv);

    // -- Rule 0: the user explicitly asked for llama.cpp's fitter. It is in charge of
    //    device memory and would throw on any layout we set (common/fit.cpp:463-465,
    //    484-486), with the failure ignored by the caller (common/common.cpp:1320).
    //    Full no-op, per the "never override an explicit choice" rule.
    if (f.fit_on) {
        note("user passed --fit on: leaving device-memory fitting to llama.cpp (no amp plan)");
        return notes;
    }

    // -- Open the model and plan. Without a local GGUF there is nothing to plan; the
    //    server downloads HF repos later (tools/server/server.cpp:395), after us.
    if (params.model.path.empty()) {
        note("no local --model: nothing to plan (router or download mode)");
        return notes;
    }
    auto gguf_res = GGUFFile::open(params.model.path);
    if (!gguf_res.ok()) {
        note("cannot open model '" + params.model.path + "': " + gguf_res.message() +
             " - skipping plan");
        return notes;
    }
    GGUFFile & gguf = *gguf_res;
    auto geo_res = ModelGeometry::build(gguf);
    if (!geo_res.ok()) {
        note("cannot build geometry: " + geo_res.message() + " - skipping plan");
        return notes;
    }
    const ModelGeometry & geo = *geo_res;

    const CostModel     cost   = CostModel::from_environment();
    const DeviceBudget  budget = detect_device_budget(cost.constants());

    // An explicit -ub pins the ubatch the planner may choose, so the VRAM estimate
    // matches what the context will actually be created with.
    PlannerOptions po;
    po.n_ctx   = opts.n_ctx;
    po.cache_k = opts.cache_k;
    po.cache_v = opts.cache_v;
    if (f.ub) {
        const char * v = argv_value(argv, "-ub");
        if (!v) {
            v = argv_value(argv, "--ubatch-size");
        }
        po.ubatch_min = po.ubatch_max = std::max<int64_t>(32, std::atoll(v));
    }
    auto plan_res = MemoryPlanner::plan(geo, budget, po, cost);
    if (!plan_res.ok()) {
        note("planner failed: " + plan_res.message() + " - continuing with llama.cpp defaults");
        return notes;
    }
    const ExecutionPlan & plan = *plan_res;

    // -- Rule 1: the layout. The user's explicit layout wins; otherwise the plan's
    //    n_expert_layers_gpu becomes tensor_buft_overrides (-ncmoe's public equivalent,
    //    amp/runtime/buft_overrides.cpp). n_gpu_layers stays -1 so llama.cpp assigns
    //    every non-overridden tensor to the GPU - the configuration BENCH.md was
    //    measured with (the old server did the same, src/server/service.cpp:333-345).
    if (user_set_layout(params)) {
        note("user set a device layout (-ngl/-ncmoe/-ot): leaving n_gpu_layers=" +
             std::to_string(params.n_gpu_layers) + " and tensor_buft_overrides untouched");
    } else {
        const int64_t n_cpu = std::max<int64_t>(0, geo.n_layer() - plan.n_expert_layers_gpu);
        write_expert_cpu_overrides(params, n_cpu);
        note("layout: " + std::to_string(plan.n_expert_layers_gpu) + " expert layers on GPU, " +
             std::to_string(n_cpu) + " pinned to CPU (" +
             human_bytes((uint64_t) plan.expert_bytes_cpu) + " CPU expert set)");
    }

    // -- Rule 2: fit_params. Default is true (common/common.h:476). The fitter throws
    //    on any layout it did not create (common/fit.cpp:463-465, 484-486) and the
    //    caller ignores the failure (common/common.cpp:1320), so leaving it on would
    //    only burn a full no_alloc model load at startup (common/fit.cpp:264). amp's
    //    planner replaces it. An explicit "-fit off" is already false; we only change
    //    the default.
    if (params.fit_params) {
        params.fit_params = false;
        note("fit_params: off (amp's planner replaces llama.cpp's fitter)");
    } else {
        note("fit_params: already off");
    }

    // -- Rule 3: context size. The planner's KV/VRAM numbers are at opts.n_ctx (200k,
    //    the measured fit). Only set it when the user did not pass -c.
    if (f.c) {
        note("n_ctx: leaving user's -c " + std::to_string(params.n_ctx));
    } else {
        params.n_ctx = (int32_t) opts.n_ctx;
        note("n_ctx: " + std::to_string(params.n_ctx) +
             " (planned; native 262144 does not fit the VRAM budget)");
    }

    // -- Rule 4: ubatch. The planner picked the largest ubatch whose VRAM estimate
    //    fits (src/plan/memory_plan.cpp:206-216, including the measured 450 MiB context
    //    overhead, include/amp/plan/cost_model.h:61-66, and the 0.92 safety factor).
    //    This replaces the old init-and-retry loop (src/server/service.cpp:354-403),
    //    which is impossible before a context exists. Tradeoff: if the estimate is
    //    wrong, llama_init_from_model fails loudly (common/common.cpp:1398-1402)
    //    instead of backing off - see docs/PREFLIGHT.md "what could still go wrong".
    if (!f.ub) {
        params.n_ubatch = (int32_t) plan.ubatch;
        note("n_ubatch: " + std::to_string(plan.ubatch) +
             " (largest ubatch that fits the measured VRAM budget)");
    } else {
        note("n_ubatch: leaving user's -ub " + std::to_string(params.n_ubatch));
    }
    // llama.cpp clamps n_ubatch to n_batch (src/llama-context.cpp:248). The default
    // n_batch (2048) already covers the planner's range; only raise it for consistency
    // when the user pinned a larger -ub without -b.
    if (!f.b && params.n_batch < params.n_ubatch) {
        params.n_batch = params.n_ubatch;
        note("n_batch: raised to " + std::to_string(params.n_batch) + " to cover n_ubatch");
    }

    // -- Rule 5: KV cache dtypes. The field default is F16 (common/common.h:587-588),
    //    so an unset field is indistinguishable from an explicit "-ctk f16" - and f16
    //    is not what BENCH.md measured. The argv scan is the only reliable test.
    if (!f.ctk) {
        params.cache_type_k = to_ggml(opts.cache_k);
        note("cache_type_k: " + std::string(ggml_type_name(params.cache_type_k)) + " (measured baseline)");
    } else {
        note("cache_type_k: leaving user's -ctk " +
             std::string(ggml_type_name(params.cache_type_k)));
    }
    if (!f.ctv) {
        params.cache_type_v = to_ggml(opts.cache_v);
        note("cache_type_v: " + std::string(ggml_type_name(params.cache_type_v)) + " (measured baseline)");
    } else {
        note("cache_type_v: leaving user's -ctv " +
             std::string(ggml_type_name(params.cache_type_v)));
    }

    // -- Rule 6: flash attention. The old server forced it on
    //    (src/server/service.cpp:367); AUTO (the default, common/common.h:499) resolves
    //    to it on CUDA but say so explicitly.
    if (!f.fa) {
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        note("flash_attn: on (measured with it)");
    } else {
        note("flash_attn: leaving user's -fa");
    }

    // -- Rule 7: threads. postprocess_cpu_params resolves the -1 default during the
    //    parse (common/common.cpp:291-298), so the field cannot tell us whether -t was
    //    passed; the argv scan is the only test. n_threads_batch follows n_threads
    //    (common/common.cpp:1729-1730).
    if (!f.t) {
        const int64_t nt = opts.n_threads > 0 ? opts.n_threads : (int64_t) budget.n_cpu_threads;
        params.cpuparams.n_threads = (int32_t) nt;
        note("n_threads: " + std::to_string(nt) + " (ncpu/2 on this box)");
    } else {
        note("n_threads: leaving user's -t " +
             std::to_string(params.cpuparams.n_threads));
    }

    // -- Rule 8: safety guards. These are not layout choices; they only ever REDUCE
    //    llama.cpp server defaults that are dangerous on this machine, and they never
    //    override a flag the user set.
    //    n_ctx_checkpoints: default 32 (common/common.h:629). Each checkpoint stores
    //    the full memory state at its position - ~1.6 GiB at 200k ctx (KV plus the
    //    62.81 MiB recurrent state) - so the default ring can reach ~50 GiB.
    if (!f.ctxcp && params.n_ctx_checkpoints > opts.n_ctx_checkpoints) {
        note("n_ctx_checkpoints: " + std::to_string(opts.n_ctx_checkpoints) +
             " (clamped from " + std::to_string(params.n_ctx_checkpoints) +
             "; each is ~1.6 GiB at 200k ctx)");
        params.n_ctx_checkpoints = opts.n_ctx_checkpoints;
    }
    //    cache_ram_mib: default 8192 (common/common.h:632). The prompt cache is
    //    anonymous RAM (tools/server/server-task.cpp:1711-1758) that evicts the
    //    model's page cache - the decode cliff, AGENT.md.
    if (!f.cram && params.cache_ram_mib > opts.cache_ram_mib) {
        note("cache_ram_mib: " + std::to_string(opts.cache_ram_mib) +
             " (clamped from " + std::to_string(params.cache_ram_mib) +
             "; the prompt cache evicts the model's page cache)");
        params.cache_ram_mib = opts.cache_ram_mib;
    }

    // -- Rule 9: the page-cache warm. llama.cpp's model load performs a full sequential
    //    fault-in of the GGUF: init_mappings(true) (src/llama-model.cpp:1748) mmaps
    //    with MAP_POPULATE (src/llama-mmap.cpp:480) plus posix_fadvise(SEQUENTIAL)
    //    (:475) and MADV_WILLNEED over the whole file (:500-504); lazy AUTO marks
    //    nothing lazy because no tensor exceeds 4 GiB (src/llama-model-loader.cpp:1094).
    //    So by default we do NOT warm: the load re-reads everything anyway, and its
    //    sequential populate evicts the file front (where the hot set lives), which
    //    would leave a deeper cold spot than not warming at all. opts.warm opts into
    //    the old amp-warm behavior: lazy_mode ON so the load does not populate, then a
    //    sequential warm of the ranges the plan calls resident.
    if (opts.warm) {
        if (!f.lzm) {
            params.lazy_mode = LLAMA_LAZY_MODE_ON;
            note("lazy_mode: on (warm requested; stops the load's MAP_POPULATE from evicting the warmed pages)");
        } else if (params.lazy_mode == LLAMA_LAZY_MODE_ON) {
            note("lazy_mode: already on (user's -lzm)");
        } else {
            note("lazy_mode: user's -lzm " + std::to_string((int) params.lazy_mode) +
                 " left as-is; the warm may not survive the load's populate");
        }
        if (!plan.resident.empty()) {
            std::vector<ReadRange> ranges;
            ranges.reserve(plan.resident.size());
            uint64_t total = 0;
            for (const auto & rp : plan.resident) {
                ranges.push_back(rp.range);
                total += rp.range.length;
            }
            AMP_INFO("amp-preflight: warming ", human_bytes(total), " in ", ranges.size(),
                     " ranges (plan.resident) ...");
            WarmProgress prog;
            prog.on_progress = [](const WarmProgress & wp) {
                if (!wp.finished) {
                    AMP_INFO("  ", human_bytes(wp.bytes_done), " / ", human_bytes(wp.bytes_total),
                             "  ", human_rate(wp.bytes_per_sec));
                }
            };
            const Stopwatch wsw;
            const Status wst = warm_ranges_blocking(gguf.file().fd(), ranges, opts.warm_chunk, &prog);
            const double secs = wsw.elapsed_s();
            if (!wst.ok()) {
                // A failed warm is a performance problem, not a correctness one.
                AMP_WARN("amp-preflight: warm failed (", wst.message(),
                         ") - continuing with demand paging");
            } else {
                AMP_INFO("amp-preflight: warmed ", human_bytes(total), " in ", secs,
                         " s  (", human_rate(secs > 0 ? (double) total / secs : 0.0), ")");
            }
            note("warm: " + human_bytes(total) + " in " + std::to_string(secs) + " s");
        } else {
            note("warm: plan.resident is empty - nothing to warm");
        }
    } else {
        note("warm: off (the load's MAP_POPULATE + fadvise(SEQUENTIAL) already reads the whole file)");
    }

    note("plan: ubatch=" + std::to_string(plan.ubatch) +
         " gpu_expert_layers=" + std::to_string(plan.n_expert_layers_gpu) +
         " kv=" + human_bytes((uint64_t) plan.kv_bytes) +
         " predicted " + std::to_string((int) plan.predicted_prefill_tps) + " t/s prefill / " +
         std::to_string((int) plan.predicted_decode_tps) + " t/s decode");
    for (const auto & n : plan.notes) {
        note("plan: " + n);
    }
    return notes;
}

} // namespace amp
