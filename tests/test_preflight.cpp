// Tests for the preflight's central promise: it must NEVER override a flag the user passed,
// and it must be a complete no-op when the user has asked for a specific layout or for
// llama.cpp's own fitter.
//
// This is a hard constraint from the project's goals, and it is the kind of promise that rots
// silently: the preflight shipped with two planning bugs that cost 4 GPU expert layers each and
// never once failed a test, because nothing asserted what it *chose*.
//
// These need a real GGUF (AMP_TEST_MODEL) because the preflight opens the model to plan against
// it. That is a header parse plus the planner - no weight loading, so it is fast.
#include "test_harness.h"

#include "amp/plan/preflight.h"

#include "arg.h"    // common_params
#include "ggml.h"

#include <string>
#include <vector>

using namespace amp;

namespace {

// A common_params as common_params_parse would leave it: n_gpu_layers -1, the override array
// padded to llama_max_tensor_buft_overrides() null entries, and F16 KV (llama.cpp's default).
common_params parsed_like_llama_cpp() {
    common_params p;
    p.tensor_buft_overrides.clear();
    const size_t ntbo = llama_max_tensor_buft_overrides();
    p.tensor_buft_overrides.reserve(ntbo);
    for (size_t i = 0; i < ntbo; i++) {
        p.tensor_buft_overrides.push_back({ nullptr, nullptr });
    }
    return p;
}

const char * model_path() { return getenv("AMP_TEST_MODEL"); }

bool has_note(const std::vector<std::string> & notes, const char * needle) {
    for (const auto & n : notes) {
        if (n.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::string notes_joined(const std::vector<std::string> & notes) {
    std::string out;
    for (const auto & n : notes) {
        out += n;
        out += " | ";
    }
    return out;
}

} // namespace

AMP_TEST(preflight_no_model_is_a_noop) {
    // Router / download mode: there is nothing to plan, and the preflight must not object.
    common_params   p = parsed_like_llama_cpp();
    p.model.path.clear();
    PreflightOptions o;
    const auto       r = apply_preflight(o, p, std::vector<std::string>{ "amp-server" });
    AMP_CHECK_MSG(r.ok(), "preflight should not fail without a model: " + r.message());
    AMP_CHECK_EQ(p.n_ctx, 0);
    AMP_CHECK_EQ(p.n_gpu_layers, -1);
}

AMP_TEST(preflight_fit_flag_is_a_total_noop) {
    // "--fit on" hands device memory to llama.cpp's fitter, which throws on our layout
    // (fit.cpp:463-486) with the failure ignored (common.cpp:1320). The preflight must not
    // touch a single field.
    const char * m = model_path();
    if (!m) {
        return;
    }
    common_params   p = parsed_like_llama_cpp();
    p.model.path    = m;
    p.n_gpu_layers  = -1;
    PreflightOptions o;
    const auto       r = apply_preflight(o, p, std::vector<std::string>{ "amp-server", "-m", m, "-fit", "on" });
    AMP_CHECK_MSG(r.ok(), "should not fail: " + r.message());
    AMP_CHECK_MSG(has_note(*r, "--fit"), "should say it deferred to the fitter, got: " + notes_joined(*r));
    AMP_CHECK_EQ(p.n_gpu_layers, -1);
    // fit_params already defaults to true (common/common.h:476), so "untouched" means still true.
    // The preflight's normal job is to turn it OFF; under an explicit -fit on it must not.
    AMP_CHECK_MSG(p.fit_params, "must not disable a user's --fit");
    AMP_CHECK_EQ(p.n_ctx, 0);
}

AMP_TEST(preflight_respects_explicit_gpu_layers) {
    // -ngl is the clearest "the user chose the layout" signal. It must win outright.
    const char * m = model_path();
    if (!m) {
        return;
    }
    common_params   p = parsed_like_llama_cpp();
    p.model.path    = m;
    p.n_gpu_layers  = 20;   // as if the user passed -ngl 20
    PreflightOptions o;
    const auto       r = apply_preflight(o, p, std::vector<std::string>{ "amp-server", "-m", m, "-ngl", "20" });
    AMP_CHECK_MSG(r.ok(), "should not fail: " + r.message());
    AMP_CHECK_EQ(p.n_gpu_layers, 20);
    AMP_CHECK_MSG(p.tensor_buft_overrides[0].pattern == nullptr,
                  "must not write expert overrides when the user set a layout");
}

AMP_TEST(preflight_respects_explicit_context_and_ubatch) {
    // -c was ignored by the planner, which cost 4 GPU expert layers at -c 32768.
    const char * m = model_path();
    if (!m) {
        return;
    }
    common_params   p = parsed_like_llama_cpp();
    p.model.path    = m;
    p.n_ctx         = 8192;
    p.n_ubatch      = 256;
    PreflightOptions o;
    const auto r = apply_preflight(
        o, p, std::vector<std::string>{ "amp-server", "-m", m, "-c", "8192", "-ub", "256" });
    AMP_CHECK_MSG(r.ok(), "should not fail: " + r.message());
    AMP_CHECK_EQ(p.n_ctx, 8192);
    AMP_CHECK_EQ(p.n_ubatch, 256);
}

AMP_TEST(preflight_terminates_the_override_array) {
    // The one genuinely dangerous failure mode: common.cpp:1706 asserts the last entry is null
    // and the tensor loader walks to pattern == nullptr, so a missing sentinel aborts or
    // segfaults. Assert the entries are contiguous and the array ends with the sentinel.
    const char * m = model_path();
    if (!m) {
        return;
    }
    common_params   p = parsed_like_llama_cpp();
    p.model.path    = m;
    PreflightOptions o;
    o.n_ctx = 4096;   // small, so this is not a cache-bound no-op
    const auto r = apply_preflight(o, p, std::vector<std::string>{ "amp-server", "-m", m });
    AMP_CHECK_MSG(r.ok(), "should not fail: " + r.message());

    const auto & ov   = p.tensor_buft_overrides;
    AMP_CHECK_MSG(!ov.empty(), "override array must not be empty");
    AMP_CHECK_MSG(ov.back().pattern == nullptr, "the array must end with a null sentinel");

    // The real invariant: the non-null entries form a CONTIGUOUS PREFIX and everything from
    // the first null onwards is null. Asserting "the first non-null is followed by a null"
    // instead - which is what this test did - is only true when exactly one override is
    // written, so it passed for the wrong reason: whenever the planner failed (VRAM held by a
    // running server) it wrote nothing and the loop never ran. It only failed once VRAM was
    // free and the planner actually chose a layout, writing 3 x n_gpu_layers entries.
    size_t n_real = 0;
    while (n_real < ov.size() && ov[n_real].pattern != nullptr) {
        n_real++;
    }
    for (size_t i = n_real; i < ov.size(); i++) {
        if (ov[i].pattern != nullptr) {
            AMP_CHECK_MSG(false,
                          "override entries must be a contiguous prefix: entry " +
                              std::to_string(i) + " is non-null after " +
                              std::to_string(n_real) + " nulls");
            break;
        }
    }
    // Expert overrides are written in groups of 3 (gate/up/down) per layer, so the count is
    // either 0 (no plan, or a user-set layout) or a positive multiple of 3.
    AMP_CHECK_MSG(n_real % 3 == 0,
                  "expert overrides come in groups of 3, got " + std::to_string(n_real));
    AMP_CHECK_MSG(n_real < ov.size(), "the prefix must leave room for the null sentinel");
}

AMP_TEST(preflight_clamps_the_dangerous_defaults) {
    // llama.cpp's own defaults would exhaust RAM on this box: 32 context checkpoints, each a
    // full serialized sequence state (~1.6 GiB at 200k), and 8 GiB of anonymous prompt cache
    // that evicts the model's page cache. See AGENT.md.
    const char * m = model_path();
    if (!m) {
        return;
    }
    common_params   p = parsed_like_llama_cpp();
    p.model.path    = m;
    p.n_ctx_checkpoints = 32;    // llama.cpp's default
    p.cache_ram_mib     = 8192;  // llama.cpp's default
    p.fit_params        = true;  // llama.cpp's default
    PreflightOptions    o;
    o.n_ctx = 4096;
    const auto r = apply_preflight(o, p, std::vector<std::string>{ "amp-server", "-m", m });
    AMP_CHECK_MSG(r.ok(), "should not fail: " + r.message());
    AMP_CHECK_EQ(p.n_ctx_checkpoints, 2);
    // 0, not the old 512. 512 is still enough for two or three 18k-token prompt entries to fill
    // it and start evicting the model's page cache, which measured 2.6x on sustained prefill.
    AMP_CHECK_EQ(p.cache_ram_mib, 0);
    {
        // ...unless the user asks for a cache, in which case they get one.
        // common_params_parse would already have applied the VALUE; the preflight only has to
        // notice that the user supplied the flag at all. So set both, as the real pipeline does.
        common_params   q = parsed_like_llama_cpp();
        q.model.path    = m;
        q.cache_ram_mib = 2048;   // as "-cram 2048" would have left it
        PreflightOptions qo;
        qo.n_ctx = 4096;
        const auto r2 = apply_preflight(qo, q,
                                        std::vector<std::string>{ "amp-server", "-m", m, "-cram", "2048" });
        AMP_CHECK_MSG(r2.ok(), "should not fail: " + r2.message());
        AMP_CHECK_EQ(q.cache_ram_mib, 2048);
    }
    AMP_CHECK_MSG(!p.fit_params, "fit_params must be off so it cannot overwrite our layout");
}

AMP_TEST(preflight_plan_failure_still_leaves_a_usable_config) {
    // When VRAM is contended the planner legitimately fails. That path must still leave the
    // server in a safe, working configuration - the clamps applied, dtypes correct, and a real
    // ubatch. It used to return early and skip all of it, which meant serving with 32 context
    // checkpoints and 8 GiB of anonymous prompt cache precisely when the box was under pressure.
    //
    // n_ubatch is the subtle one: ExecutionPlan::ubatch defaults to 0, so a plan failure that
    // still assigned it would set n_ubatch = 0.
    const char * m = model_path();
    if (!m) {
        return;
    }
    common_params   p = parsed_like_llama_cpp();
    p.model.path    = m;
    p.n_ctx         = 0;
    p.n_ubatch      = 2048;   // llama.cpp's default, as common_params_parse would leave it
    PreflightOptions o;
    // A context and cache size that cannot fit even an idle 6 GB card, to force the failure
    // path deterministically regardless of what else is running.
    o.n_ctx          = 4000000;
    o.n_ctx_checkpoints = 2;
    const auto r = apply_preflight(o, p, std::vector<std::string>{ "amp-server", "-m", m });
    AMP_CHECK_MSG(r.ok(), "should not fail: " + r.message());
    AMP_CHECK_MSG(has_note(*r, "planner failed"), "expected the planner to fail, got: " + notes_joined(*r));
    AMP_CHECK_EQ(p.n_ctx_checkpoints, 2);
    AMP_CHECK_MSG(p.cache_type_k == GGML_TYPE_Q8_0,
                  std::string("K cache must still be q8_0, got ") + ggml_type_name(p.cache_type_k));
    AMP_CHECK_MSG(p.cache_type_v == GGML_TYPE_Q8_0,
                  std::string("V cache must still be q8_0, got ") + ggml_type_name(p.cache_type_v));
    AMP_CHECK_MSG(p.n_ubatch == 2048,
                  std::string("n_ubatch must be left untouched without a plan, got ") +
                      std::to_string(p.n_ubatch) + " (0 would mean plan.ubatch was assigned)");
    AMP_CHECK_MSG(p.n_ctx > 0, "n_ctx must be set even without a plan, got " + std::to_string(p.n_ctx));
}

AMP_TEST(preflight_forces_one_slot_unless_asked) {
    // The server example sets n_parallel = -1 ("auto", arg.cpp:1400), which server.cpp:156-159
    // expands to FOUR concurrent slots. On this box that is a 44x decode collapse, because every
    // concurrent generation wants the same shared ~10.9 GiB CPU expert set.
    //
    // One, and the cost is deliberate. n_ctx is the total across slots, so two slots would halve
    // the context a single conversation gets: -c 200000 becomes 100096. That is worse than the
    // stall it avoids for anyone who wants 200k in one session. --parallel 2 still works.
    const char * m = model_path();
    if (!m) {
        return;
    }
    {
        common_params   p = parsed_like_llama_cpp();
        p.model.path    = m;
        p.n_parallel    = -1;   // what common_params_parse leaves for the server example
        PreflightOptions o;
        o.n_ctx = 4096;
        const auto r = apply_preflight(o, p, std::vector<std::string>{ "amp-server", "-m", m });
        AMP_CHECK_MSG(r.ok(), "should not fail: " + r.message());
        AMP_CHECK_EQ(p.n_parallel, 1);
    }
    {
        // ...unless the user asked for a specific count, in either direction.
        common_params   p = parsed_like_llama_cpp();
        p.model.path    = m;
        // As common_params_parse would leave it: the flag's VALUE is already applied, and the
        // preflight only needs to detect that the user supplied the flag at all.
        p.n_parallel    = 3;
        PreflightOptions o;
        o.n_ctx = 4096;
        const auto r = apply_preflight(o, p,
                                       std::vector<std::string>{ "amp-server", "-m", m, "--parallel", "3" });
        AMP_CHECK_MSG(r.ok(), "should not fail: " + r.message());
        AMP_CHECK_EQ(p.n_parallel, 3);
    }
    {
        common_params   p = parsed_like_llama_cpp();
        p.model.path    = m;
        p.n_parallel    = 1;
        PreflightOptions o;
        o.n_ctx = 4096;
        const auto r = apply_preflight(o, p,
                                       std::vector<std::string>{ "amp-server", "-m", m, "--parallel", "1" });
        AMP_CHECK_MSG(r.ok(), "should not fail: " + r.message());
        AMP_CHECK_EQ(p.n_parallel, 1);
    }
}

AMP_TEST(preflight_applies_the_measured_kv_dtypes) {
    // The regression test for the worst of the two planning bugs. llama.cpp's field default is
    // F16; the preflight previously planned at F16 and then applied q8_0/q4_0, so the plan
    // reserved 2.46x the VRAM the cache actually needs. The rule that survives is the one that
    // says plan against what will actually be applied, not what the field happened to hold.
    //
    // V is q8_0 rather than the older q4_0. llama.cpp's own default is f16/f16, so q4_0 was
    // already a reduction, and published measurements put q4_0 V about 7x further from f16 than
    // q8_0 V is. The extra VRAM comes out of one GPU expert layer, which is the right trade:
    // decode stays far above its floor and that layer was worth little to it.
    const char * m = model_path();
    if (!m) {
        return;
    }
    common_params   p = parsed_like_llama_cpp();
    p.model.path    = m;
    p.cache_type_k  = GGML_TYPE_F16;
    p.cache_type_v  = GGML_TYPE_F16;
    PreflightOptions o;
    o.n_ctx = 4096;
    const auto r = apply_preflight(o, p, std::vector<std::string>{ "amp-server", "-m", m });
    AMP_CHECK_MSG(r.ok(), "should not fail: " + r.message());
    AMP_CHECK_MSG(p.cache_type_k == GGML_TYPE_Q8_0,
                  std::string("K cache must be q8_0, got ") + ggml_type_name(p.cache_type_k));
    AMP_CHECK_MSG(p.cache_type_v == GGML_TYPE_Q8_0,
                  std::string("V cache must be q8_0, got ") + ggml_type_name(p.cache_type_v));
    {
        // ...unless the user names a dtype, in which case it is theirs.
        common_params   q = parsed_like_llama_cpp();
        q.model.path    = m;
        q.cache_type_v  = GGML_TYPE_F16;   // as "-ctv f16" would have left it
        PreflightOptions qo;
        qo.n_ctx = 4096;
        const auto r2 = apply_preflight(qo, q,
                                        std::vector<std::string>{ "amp-server", "-m", m, "-ctv", "f16" });
        AMP_CHECK_MSG(r2.ok(), "should not fail: " + r2.message());
        AMP_CHECK_MSG(q.cache_type_v == GGML_TYPE_F16,
                      std::string("an explicit -ctv must survive, got ") + ggml_type_name(q.cache_type_v));
    }
}

AMP_TEST(model_config_variant_sets_kv_dtypes) {
    // A preset must be indistinguishable from typing its flags: same fields, and the same
    // "the user asked" bits so the planner does not undo it.
    const char * m = model_path();
    if (!m) {
        return;
    }
    PreflightOptions o;
    o.n_ctx = 4096;

    {
        common_params   p = parsed_like_llama_cpp();
        p.model.path    = m;
        p.cache_type_k  = GGML_TYPE_F16;
        p.cache_type_v  = GGML_TYPE_F16;
        const auto r = apply_preflight(o, p, std::vector<std::string>{
            "amp-server", "-m", m, "--model-config", "occamy-1.0-apex-i-miniplus:v1" });
        AMP_CHECK_MSG(r.ok(), "should not fail: " + r.message());
        AMP_CHECK_MSG(p.cache_type_v == GGML_TYPE_Q4_0,
                      std::string("v1 must set q4_0 V, got ") + ggml_type_name(p.cache_type_v));
    }
    {
        common_params   p = parsed_like_llama_cpp();
        p.model.path    = m;
        p.cache_type_k  = GGML_TYPE_F16;
        p.cache_type_v  = GGML_TYPE_F16;
        const auto r = apply_preflight(o, p, std::vector<std::string>{
            "amp-server", "-m", m, "--model-config", "occamy-1.0-apex-i-miniplus:v3" });
        AMP_CHECK_MSG(r.ok(), "should not fail: " + r.message());
        AMP_CHECK_MSG(p.cache_type_v == GGML_TYPE_F16,
                      std::string("v3 must set f16 V, got ") + ggml_type_name(p.cache_type_v));
        // f16 KV does not fit at 200k on this box, so the preset has to bring its own context.
        AMP_CHECK_MSG(p.n_ctx == 131072,
                      "v3 must lower n_ctx to fit f16, got " + std::to_string(p.n_ctx));
    }
}

AMP_TEST(model_config_yields_to_an_explicit_flag) {
    // The whole point of a preset: it fills gaps, it does not fight the command line. This was
    // briefly inverted, and the symptom was silent - the preset's values won and nothing said so.
    const char * m = model_path();
    if (!m) {
        return;
    }
    PreflightOptions o;
    o.n_ctx = 4096;

    common_params   p = parsed_like_llama_cpp();
    p.model.path    = m;
    p.n_ctx         = 200000;                       // as "-c 200000" would have left it
    p.cache_type_v  = GGML_TYPE_Q8_0;               // as "-ctv q8_0" would have left it
    const auto r = apply_preflight(o, p, std::vector<std::string>{
        "amp-server", "-m", m, "--model-config", "occamy-1.0-apex-i-miniplus:v3", "-c", "200000", "-ctv", "q8_0" });
    AMP_CHECK_MSG(r.ok(), "should not fail: " + r.message());
    AMP_CHECK_MSG(p.cache_type_v == GGML_TYPE_Q8_0,
                  std::string("an explicit -ctv must beat the preset, got ") + ggml_type_name(p.cache_type_v));
    AMP_CHECK_MSG(p.n_ctx == 200000,
                  "an explicit -c must beat the preset, got " + std::to_string(p.n_ctx));
}

AMP_TEST(model_config_sampler_profiles) {
    // The four published sampler profiles, checked against the values in the table rather than
    // against themselves. :s1 is the one that catches the most: it is the only profile using
    // presence_penalty and enable_thinking, and those are the two keys that are not plain floats.
    //
    // Note what the expected values say about a key the profile omits. :s1 does not name
    // repetition_penalty, and the expected value is amp's default 1.05, not llama.cpp's 1.00. A
    // profile is a partial description of a sampler, and the rest of the sampler is amp's, not the
    // upstream default's - otherwise :s1 would silently be a different sampler depending on what
    // llama.cpp happened to ship, and the four profiles would not be comparable to each other.
    const char * m = model_path();
    if (!m) {
        return;
    }
    PreflightOptions o;
    o.n_ctx = 4096;

    struct Row { const char * spec; float temp; float top_p; int top_k; float min_p;
                 float rep; float presence; int thinking; };
    const Row rows[] = {
        { "occamy-1.0-apex-i-miniplus:s1", 1.00f, 0.95f, 20, 0.05f, 1.05f, 1.5f, 1 },
        { "occamy-1.0-apex-i-miniplus:s2", 0.20f, 0.95f, 20, 0.05f, 1.05f, 0.0f, -1 },
        { "occamy-1.0-apex-i-miniplus:s3", 0.35f, 0.95f, 20, 0.00f, 1.00f, 0.0f, -1 },
        { "occamy-1.0-apex-i-miniplus:s4", 0.60f, 0.95f, 20, 0.05f, 1.08f, 0.0f, -1 },
    };
    for (const Row & row : rows) {
        common_params p = parsed_like_llama_cpp();
        p.model.path   = m;
        const auto r = apply_preflight(o, p, std::vector<std::string>{
            "amp-server", "-m", m, "--model-config", row.spec });
        AMP_CHECK_MSG(r.ok(), std::string(row.spec) + " must apply: " + r.message());
        const std::string at = std::string(" in ") + row.spec;
        AMP_CHECK_MSG(p.sampling.temp == row.temp,
                      "temperature" + at + ", got " + std::to_string(p.sampling.temp));
        AMP_CHECK_MSG(p.sampling.top_p == row.top_p,
                      "top_p" + at + ", got " + std::to_string(p.sampling.top_p));
        AMP_CHECK_MSG(p.sampling.top_k == row.top_k,
                      "top_k" + at + ", got " + std::to_string(p.sampling.top_k));
        AMP_CHECK_MSG(p.sampling.min_p == row.min_p,
                      "min_p" + at + ", got " + std::to_string(p.sampling.min_p));
        AMP_CHECK_MSG(p.sampling.penalty_repeat == row.rep,
                      "repetition_penalty" + at + ", got " + std::to_string(p.sampling.penalty_repeat));
        AMP_CHECK_MSG(p.sampling.penalty_present == row.presence,
                      "presence_penalty" + at + ", got " + std::to_string(p.sampling.penalty_present));
        AMP_CHECK_MSG(p.enable_reasoning == row.thinking,
                      "enable_thinking" + at + ", got " + std::to_string(p.enable_reasoning));
    }
}

AMP_TEST(model_config_sampler_flags_win_per_key) {
    // Per key, not per group: one --temp must not stop the preset from supplying the rest, and one
    // --presence-penalty must not stop it from supplying enable_thinking.
    const char * m = model_path();
    if (!m) {
        return;
    }
    PreflightOptions o;
    o.n_ctx = 4096;

    common_params p = parsed_like_llama_cpp();
    p.model.path   = m;
    p.sampling.temp = 0.35f;                       // as "--temp 0.35" would have left it
    const auto r = apply_preflight(o, p, std::vector<std::string>{
        "amp-server", "-m", m, "--model-config", "occamy-1.0-apex-i-miniplus:s1", "--temp", "0.35" });
    AMP_CHECK_MSG(r.ok(), "should not fail: " + r.message());
    AMP_CHECK_MSG(p.sampling.temp == 0.35f,
                  "the flag must win, got " + std::to_string(p.sampling.temp));
    AMP_CHECK_MSG(p.sampling.penalty_present == 1.5f,
                  "the preset must still supply presence_penalty, got " +
                      std::to_string(p.sampling.penalty_present));
    AMP_CHECK_MSG(p.enable_reasoning == 1,
                  "the preset must still enable thinking, got " + std::to_string(p.enable_reasoning));
}

AMP_TEST(model_config_rejects_a_bad_sampler_value) {
    // A typo in a config file is the failure this whole mechanism exists to rule out, so an
    // out-of-range number is a hard error just like an unknown key. temperature is the case worth
    // having: llama.cpp's own flag accepts 9.5, so nothing upstream would catch it.
    const char * m = model_path();
    if (!m) {
        return;
    }
    PreflightOptions o;
    o.n_ctx = 4096;
    setenv("AMP_MODEL_CONFIGS", "/nonexistent-model-configs.conf", 1);

    struct Case { const char * body; };
    const Case cases[] = {
        { "[t:b]\ntemperature = 9.5\n" },
        { "[t:b]\ntop_p = 3.0\n" },
        { "[t:b]\nmin_p = -1\n" },
        { "[t:b]\nrepetition_penalty = 0\n" },
        { "[t:b]\nenable_thinking = maybe\n" },
        { "[t:b]\ntemperaturre = 0.3\n" },
    };
    for (const Case & c : cases) {
        const std::string path = "/tmp/amp-bad-sampler.conf";
        FILE * fp = fopen(path.c_str(), "w");
        AMP_CHECK_MSG(fp != nullptr, "could not write " + path);
        if (!fp) {
            return;
        }
        fputs(c.body, fp);
        fclose(fp);
        setenv("AMP_MODEL_CONFIGS", path.c_str(), 1);

        common_params p = parsed_like_llama_cpp();
        p.model.path   = m;
        const auto r = apply_preflight(o, p, std::vector<std::string>{
            "amp-server", "-m", m, "--model-config", "t:b" });
        AMP_CHECK_MSG(!r.ok(), std::string("must be rejected: ") + c.body);
    }
    unsetenv("AMP_MODEL_CONFIGS");
}

AMP_TEST(model_config_rejects_a_bad_spec_instead_of_guessing) {
    // A quality dial that quietly does nothing is the worst failure mode it can have, so an
    // unknown variant is a hard error. Bare numbers are accepted as well as the vN spelling.
    const char * m = model_path();
    if (!m) {
        return;
    }
    PreflightOptions o;
    o.n_ctx = 4096;

    {
        common_params p = parsed_like_llama_cpp();
        p.model.path  = m;
        const auto r = apply_preflight(o, p, std::vector<std::string>{
            "amp-server", "-m", m, "--model-config", "occamy-1.0-apex-i-miniplus:v2" });
        AMP_CHECK_MSG(r.ok(), "the bare vN form must work: " + r.message());
    }
    {
        common_params p = parsed_like_llama_cpp();
        p.model.path  = m;
        const auto r = apply_preflight(o, p, std::vector<std::string>{
            "amp-server", "-m", m, "--model-config", "occamy-1.0-apex-i-miniplus:2" });
        AMP_CHECK_MSG(r.ok(), "a bare number must work too: " + r.message());
    }
    {
        common_params p = parsed_like_llama_cpp();
        p.model.path  = m;
        const auto r = apply_preflight(o, p, std::vector<std::string>{
            "amp-server", "-m", m, "--model-config", "occamy-1.0-apex-i-miniplus:v99" });
        AMP_CHECK_MSG(!r.ok(), "an unknown variant must fail rather than fall back to a default");
    }
    {
        common_params p = parsed_like_llama_cpp();
        p.model.path  = m;
        const auto r = apply_preflight(o, p, std::vector<std::string>{
            "amp-server", "-m", m, "--model-config", "no-such-model:v1" });
        AMP_CHECK_MSG(!r.ok(), "an unknown preset must fail rather than fall back to a default");
    }
}
