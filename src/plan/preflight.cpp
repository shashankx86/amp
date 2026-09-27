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
    bool np = false;   // explicit --parallel/-np: the user wants N concurrent slots
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
    f.np    = argv_has(argv, "-np") || argv_has(argv, "--parallel");
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

// The planner budgets VRAM from CacheType, but the authoritative resolved value after
// common_params_parse is a ggml_type. This is the reverse of to_ggml, and it must exist:
// without it the planner sizes the KV cache with the default dtypes while the context is
// actually created with whatever the user asked for, and the estimate is silently wrong.
CacheType from_ggml(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:   return CacheType::kF32;
        case GGML_TYPE_F16:   return CacheType::kF16;
        case GGML_TYPE_BF16:  return CacheType::kBF16;
        case GGML_TYPE_Q8_0:  return CacheType::kQ8_0;
        case GGML_TYPE_Q5_1:  return CacheType::kQ5_1;
        case GGML_TYPE_Q5_0:  return CacheType::kQ5_0;
        case GGML_TYPE_Q4_1:  return CacheType::kQ4_1;
        case GGML_TYPE_Q4_0:  return CacheType::kQ4_0;
        case GGML_TYPE_Q4_K:  return CacheType::kQ4_K;
        case GGML_TYPE_IQ4_NL:return CacheType::kIQ4_NL;
        case GGML_TYPE_Q3_K:  return CacheType::kQ3_K;
        case GGML_TYPE_Q2_K:  return CacheType::kQ2_K;
        case GGML_TYPE_Q6_K:  return CacheType::kQ6_K;
        default:              return CacheType::kF16;   // fp types have no fixed width
    }
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

    // -- Safety clamps, applied BEFORE the plan on purpose.
    //
    //    These do not depend on the plan, and the plan can fail: MemoryPlanner::plan returns
    //    "no configuration fits" when VRAM is contended, and that path returns early. When the
    //    clamps sat after the plan, a planner failure meant serving with llama.cpp's defaults of
    //    32 context checkpoints and 8 GiB of anonymous prompt cache - precisely the OOM-prone
    //    configuration, on the one occasion the box is already under pressure. Found by
    //    tests/test_preflight.cpp, which is the only reason it was caught at all.
    //
    //    They stay *after* the --fit check, so an explicit --fit on remains a total no-op.

    //    n_parallel is the same kind of trap, and a much bigger one. The server example sets
    //    params.n_parallel = -1 ("auto", common/arg.cpp:1400), which server.cpp:156-159 expands
    //    to FOUR concurrent slots with kv_unified. Four concurrent generations each want the same
    //    shared ~10.9 GiB CPU expert working set, and on this box that thrashes rather than
    //    shares: measured 0.64 t/s with two slots active, against 28.4 t/s with one.
    //
    //    This is not hypothetical and not a corner case. An agentic client has two requests open
    //    by design - OpenCode asks for a conversation title while the main answer streams - so
    //    the default is a guaranteed slowdown for the exact workload this server exists for.
    //    The deleted hand-rolled server serialised generations for this reason.
    //
    //    Overridable with --parallel N, because someone batching independent prompts may want
    //    the throughput and accept the memory cost.
    //
    //    It is 2, not 1. Forcing a single slot was right about the thrash and wrong about the
    //    consequence, and the second mistake cost real time in the field. An agentic client that
    //    tests the API it is being served by - which is what a model asked to verify a config
    //    does - sends a short unrelated request mid-conversation. With one slot that request is
    //    served *by the slot holding the conversation*, and the conversation's cached prefix does
    //    not survive it. Measured at a 35,001-token context with a self-test request after every
    //    second turn: **2 full re-prefills, worst turn 110.6 s, 239.3 s over 6 turns.** With two
    //    slots the self-test goes to the other slot: **0 re-prefills, worst turn 6.3 s, 27.1 s.**
    //    An 8.8x difference on the number a user actually waits through.
    //
    //    Two slots costs nothing in decode: 27.42 t/s at one slot against 27.25 at two and 27.67
    //    at four, all inside the run-to-run spread. The 0.64 t/s collapse needs several slots
    //    generating *simultaneously* against the shared expert set, which an agentic turn never
    //    does - it finishes generating, then runs tools. So keep the thrash protection, lose the
    //    single-slot trap.
    if (!f.np && params.n_parallel < 2) {
        note("n_parallel: 2 (llama-server defaults to 4 concurrent slots, which thrash the "
             "shared CPU expert set on this box; 2 keeps a self-test or side request from "
             "evicting the conversation's cached prefix, which costs an 8.8x worse worst turn. "
             "Use --parallel N to override)");
        params.n_parallel = 2;
    }

    //    fit_params belongs with them. Default is true (common/common.h:476), and the fitter
    //    would fight whatever layout we set (fit.cpp:463-486, with the failure ignored at
    //    common.cpp:1320). Disabling it even when OUR plan fails is deliberate: the fitter also
    //    silently reduces n_ctx to whatever fits (fit.cpp:393-457), and a quietly shrunk context
    //    is a worse outcome than a clean, predictable default. The user can always ask for --fit.
    if (!f.fit_on && params.fit_params) {
        params.fit_params = false;
        note("fit_params: off (amp's planner replaces llama.cpp's fitter)");
    }
    // -- Safety clamps. These are not layout choices; they only ever REDUCE llama.cpp server
    //    defaults that are dangerous on this machine, and they never override a flag the user set.
    //    n_ctx_checkpoints: default 32 (common/common.h:629). Each checkpoint stores the full
    //    memory state at its position - ~1.6 GiB at 200k ctx (KV plus the 62.81 MiB recurrent
    //    state) - so the default ring can reach ~50 GiB.
    if (!f.ctxcp && params.n_ctx_checkpoints > opts.n_ctx_checkpoints) {
        note("n_ctx_checkpoints: " + std::to_string(opts.n_ctx_checkpoints) +
             " (clamped from " + std::to_string(params.n_ctx_checkpoints) +
             "; each is ~1.6 GiB at 200k ctx)");
        params.n_ctx_checkpoints = opts.n_ctx_checkpoints;
    }
    //    cache_ram_mib: default 8192 (common/common.h:632). The prompt cache is anonymous RAM
    //    (tools/server/server-task.cpp:1711-1758) that evicts the model's page cache - the decode
    //    cliff, AGENT.md.
    if (!f.cram && params.cache_ram_mib > opts.cache_ram_mib) {
        note("cache_ram_mib: " + std::to_string(opts.cache_ram_mib) +
             " (clamped from " + std::to_string(params.cache_ram_mib) +
             "; the prompt cache evicts the model's page cache)");
        params.cache_ram_mib = opts.cache_ram_mib;
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
    // ModelGeometry::build yields a Result<unique_ptr<ModelGeometry>>, so the geometry itself is
    // the second dereference (same shape as the old service.cpp's `**geo_res`).
    const ModelGeometry & geo = **geo_res;

    const CostModel     cost   = CostModel::from_environment();
    const DeviceBudget  budget = detect_device_budget(cost.constants());

    // An explicit -ub pins the ubatch the planner may choose, so the VRAM estimate
    // matches what the context will actually be created with.
    PlannerOptions po;
    // Plan against the context the server will really create, not opts.n_ctx. params.n_ctx is
    // the resolved truth at this point: common_params_parse has already folded in -c/--ctx-size
    // and LLAMA_ARG_CTX_SIZE. It is 0 when neither was given, which means "use the model's
    // native context" rather than "a context of zero", hence the fallback.
    //
    // This is not cosmetic. Planning for 200k while the user asked for 32k reserves 1.55 GiB of
    // KV that will never be allocated, which costs GPU expert layers and shrinks the ubatch:
    // the planner ends up optimising against a budget that does not exist.
    const int64_t plan_ctx = params.n_ctx > 0 ? (int64_t) params.n_ctx : opts.n_ctx;
    if (plan_ctx != opts.n_ctx) {
        note("plan against the requested context: " + std::to_string(plan_ctx) +
             " tokens (not the " + std::to_string(opts.n_ctx) + " default)");
    }
    po.n_ctx   = plan_ctx;
    // The KV dtypes must be the ones that will ACTUALLY be in effect after every rule has run,
    // not the ones currently sitting in params. Rule 5 (below) overwrites params.cache_type_{k,v}
    // with q8_0/q4_0 unless the user pinned them, and llama.cpp's field default is F16
    // (common/common.h:587-588) - so reading params before Rule 5 plans against f16.
    //
    // That is not a rounding error. f16 KV is 2.46x the bytes of q8_0/q4_0, so at 200k context
    // the plan reserved 3.81 GiB of VRAM for a cache that is really 1.55 GiB. Measured cost at
    // -c 200000: 4 GPU expert layers and a 2.7x smaller ubatch (g=4/ubatch 1024 -> g=0/ubatch 384).
    //
    // So: honour the user's -ctk/-ctv when present, otherwise plan with the default we are about
    // to apply. This is the same user-intent test Rule 5 uses, deliberately, so the plan and the
    // applied configuration cannot disagree.
    po.cache_k = f.ctk ? from_ggml(params.cache_type_k) : opts.cache_k;
    po.cache_v = f.ctv ? from_ggml(params.cache_type_v) : opts.cache_v;
    if (f.ctk || f.ctv) {
        note("plan against the requested KV dtypes: " + std::string(ggml_type_name(params.cache_type_k)) +
             "/" + std::string(ggml_type_name(params.cache_type_v)));
    }
    if (f.ub) {
        const char * v = argv_value(argv, "-ub");
        if (!v) {
            v = argv_value(argv, "--ubatch-size");
        }
        po.ubatch_min = po.ubatch_max = std::max<int64_t>(32, std::atoll(v));
    }
    //
    // A plan failure is NOT a reason to skip the rules that do not depend on the plan. Only
    // Rules 1 and 4 consume it; context, KV dtypes, flash-attn and threads are all decided
    // without it, and they are what make the server behave predictably on a box too small for
    // the plan. So the plan is optional and only the two consumers are guarded.
    auto               plan_res = MemoryPlanner::plan(geo, budget, po, cost);
    const bool         have_plan = plan_res.ok();
    // Only meaningful when have_plan; both consumers below are guarded on it.
    static const ExecutionPlan kNoPlan{};
    const ExecutionPlan &      plan = have_plan ? *plan_res : kNoPlan;
    if (!have_plan) {
        note("planner failed: " + plan_res.message() +
             " - keeping llama.cpp's device layout, applying the rest of the safe configuration");
    }

    // -- Rule 1: the layout. The user's explicit layout wins; otherwise the plan's
    //    n_expert_layers_gpu becomes tensor_buft_overrides (-ncmoe's public equivalent,
    //    amp/runtime/buft_overrides.cpp). n_gpu_layers stays -1 so llama.cpp assigns
    //    every non-overridden tensor to the GPU - the configuration BENCH.md was
    //    measured with (the old server did the same, src/server/service.cpp:333-345).
    if (!have_plan) {
        note("no plan: leaving n_gpu_layers=" + std::to_string(params.n_gpu_layers) +
             " and tensor_buft_overrides untouched");
    } else if (user_set_layout(params)) {
        note("user set a device layout (-ngl/-ncmoe/-ot): leaving n_gpu_layers=" +
             std::to_string(params.n_gpu_layers) + " and tensor_buft_overrides untouched");
    } else {
        const int64_t n_cpu = std::max<int64_t>(0, geo.n_layer() - plan.n_expert_layers_gpu);
        write_expert_cpu_overrides(params, n_cpu);
        note("layout: " + std::to_string(plan.n_expert_layers_gpu) + " expert layers on GPU, " +
             std::to_string(n_cpu) + " pinned to CPU (" +
             human_bytes((uint64_t) plan.expert_bytes_cpu) + " CPU expert set)");
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
    if (f.ub) {
        note("n_ubatch: leaving user's -ub " + std::to_string(params.n_ubatch));
    } else if (have_plan) {
        params.n_ubatch = (int32_t) plan.ubatch;
        note("n_ubatch: " + std::to_string(plan.ubatch) +
             " (largest ubatch that fits the measured VRAM budget)");
    } else {
        // ExecutionPlan::ubatch defaults to 0, so without this guard a planner failure would
        // set n_ubatch = 0 - worse than leaving llama.cpp's default in place.
        note("n_ubatch: leaving llama.cpp's default (no plan)");
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
