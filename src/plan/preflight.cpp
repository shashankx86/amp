#include "amp/plan/preflight.h"

#include "amp/bytes.h"
#include "amp/log.h"
#include "amp/plan/memory_plan.h"
#include "amp/timing.h"

#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <unistd.h>
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

// ---------------------------------------------------------------------------
// Sampler keys: the config-file spelling, the flags that set them, and where the value lands.
//
// A table rather than a UserFlags bit per key, because the planner treats the samplers as one
// axis (a whole profile, chosen by name) while the command line treats them as independent flags.
// A table is also the only way to keep "an explicit flag beats the preset" and "these are the
// defaults" from drifting apart: one row per key carries its flag spelling, its field, its bit and
// its default, so adding a key cannot forget any of the four.
//
// Four things in the rows are not optional:
//
//   alt_flag  Every flag here is a real llama-server flag, and the alternates are the other
//             spellings its own parser accepts. A flag amp lists but llama.cpp lacks would be a
//             setting that never takes effect, which is the exact failure this mechanism exists
//             to prevent.
//
//   bit       llama.cpp gates most samplers behind a user_sampling_config bit, and a sampler with
//             its bit clear is overwritten by the GGUF's own sampling metadata
//             (common/common.cpp:1217, 1235-1242). Setting the field without the bit would let
//             the model file quietly undo the preset, so the bit is set with the value. This is
//             the same pair of statements llama.cpp's own flag handlers make (common/arg.cpp:
//             2015-2017). A bit of 0 means llama.cpp has no bit for that sampler - true of
//             presence_penalty, whose own flag handler (common/arg.cpp:2101-2108) sets no bit
//             either, so amp matching that is matching upstream rather than losing a guard.
//
//   dflt      The value the planner uses when neither a flag nor a preset said anything. These
//             are the deterministic-coding profile and are the reason this table exists rather
//             than a dozen lines in the rule that sets them: a profile is a set of values chosen
//             together, and a rule that names them one at a time is how a profile ends up half
//             applied. An empty dflt means "amp has no opinion", which is different from a
//             default of 0 and is why the column is a string.
//
//   kind      Only presence_penalty and enable_thinking are not plain floats. enable_thinking is
//             not a sampler at all: it is a chat-template variable, and llama.cpp exposes it as
//             -rea/--reasoning, which sets two fields at once (common/arg.cpp:3688-3700).
enum SamplerKind { kSampFloat, kSampTopK, kSampThinking };

struct SamplerKey {
    const char * key;         // key as written in a model config
    const char * long_flag;   // canonical flag
    const char * alt_flag;    // llama.cpp's other accepted spelling, or nullptr
    SamplerKind  kind;
    float common_params_sampling::* dst;   // kSampFloat only
    uint64_t     bit;                        // 0 when llama.cpp defines no bit
    const char * dflt;                       // amp's default, or nullptr for no opinion
};

static const SamplerKey kSamplerKeys[] = {
    { "temperature",        "--temp",           "--temperature", kSampFloat,
      &common_params_sampling::temp, common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TEMP,
      "0.20" },
    { "top_p",              "--top-p",          nullptr,          kSampFloat,
      &common_params_sampling::top_p, common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_P,
      "0.95" },
    { "top_k",              "--top-k",          nullptr,          kSampTopK,
      nullptr,             common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K,
      "20" },
    { "min_p",              "--min-p",          nullptr,          kSampFloat,
      &common_params_sampling::min_p, common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIN_P,
      "0.05" },
    { "repetition_penalty", "--repeat-penalty", nullptr,          kSampFloat,
      &common_params_sampling::penalty_repeat, common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT,
      "1.05" },
    { "presence_penalty",   "--presence-penalty", nullptr,        kSampFloat,
      &common_params_sampling::penalty_present, 0,
      nullptr },
    { "enable_thinking",    "--reasoning",      "-rea",           kSampThinking,
      nullptr, 0,
      nullptr },
};

static const SamplerKey * find_sampler_key(const std::string & k) {
    for (const SamplerKey & s : kSamplerKeys) {
        if (k == s.key) {
            return &s;
        }
    }
    return nullptr;
}

// Did the command line already fix this sampler? Checked per key rather than once for the whole
// group, so that passing only --temp still lets a preset supply top_k, min_p and the rest.
static bool sampler_flag_given(const std::vector<std::string> & argv, const std::string & key) {
    const SamplerKey * s = find_sampler_key(key);
    if (!s) {
        return false;
    }
    return argv_has(argv, s->long_flag) || (s->alt_flag && argv_has(argv, s->alt_flag));
}

// Parses one value, or returns false. The accepted ranges are llama.cpp's own, not new ones:
// top_p and min_p are probabilities whose documented disabled values are 1.0 and 0.0
// (common/arg.cpp:2030, 2038), repeat-penalty must be finite and above 0 or llama.cpp refuses to
// start (common/arg.cpp:2091-2094), and presence-penalty need only be finite (2105-2106).
// top_k is an int32, 0 meaning disabled.
//
// One range is deliberately stricter than upstream. llama.cpp clamps temperature at 0 and imposes
// no ceiling, so `--temp 9.5` starts and samples from a near-uniform distribution. That is fine for
// a flag someone typed deliberately and not fine for a line in a config file read months later by
// someone who meant 0.95, which is the same failure an unknown key already is. So the ceiling is
// amp's, and 2.0 is far above any temperature that still means something.
static bool parse_sampler_float(const SamplerKey & sk, const std::string & v, float & out) {
    char *   end = nullptr;
    const double d = strtod(v.c_str(), &end);
    if (end == v.c_str() || (end && *end != '\0') || !std::isfinite(d)) {
        return false;
    }
    if (sk.key == std::string("repetition_penalty") && d <= 0.0) {
        return false;
    }
    if ((sk.key == std::string("top_p") || sk.key == std::string("min_p")) &&
        (d < 0.0 || d > 1.0)) {
        return false;
    }
    if (sk.key == std::string("temperature") && (d < 0.0 || d > 2.0)) {
        return false;
    }
    out = (float) d;
    return true;
}

// enable_thinking is on/off/auto, and the spellings are llama.cpp's own is_truthy/is_falsey
// (common/arg.cpp:3691-3696), so "auto" is accepted and means: leave the template to decide.
static bool parse_thinking(const std::string & v, int & out) {
    if (common_arg_utils::is_truthy(v)) {
        out = 1;
    } else if (common_arg_utils::is_falsey(v)) {
        out = 0;
    } else if (common_arg_utils::is_autoy(v)) {
        out = -1;
    } else {
        return false;
    }
    return true;
}

// Applies one key's value, plus its bit when llama.cpp has one. Shared by the preset path and the
// default path so the two cannot set the same field differently. The value must already have been
// validated by parse_sampler_float or parse_thinking.
static void apply_sampler(const SamplerKey & sk, const std::string & v, common_params & params) {
    switch (sk.kind) {
        case kSampThinking: {
            int on = -1;
            parse_thinking(v, on);
            params.enable_reasoning = on;
            // Both fields, because llama.cpp's -rea handler sets both (common/arg.cpp:3691-3696)
            // and setting only one leaves the template and the response parser disagreeing about
            // whether a reply should have been a thinking reply at all.
            if (on >= 0) {
                params.default_template_kwargs["enable_thinking"] = on ? "true" : "false";
            }
            return;
        }
        case kSampTopK: {
            float parsed = 0.0f;
            parse_sampler_float(sk, v, parsed);
            params.sampling.top_k = (int32_t) parsed;
            break;
        }
        case kSampFloat:
        default: {
            float parsed = 0.0f;
            parse_sampler_float(sk, v, parsed);
            params.sampling.*(sk.dst) = parsed;
            break;
        }
    }
    if (sk.bit) {
        params.sampling.user_sampling_config |= sk.bit;
    }
}

// ---------------------------------------------------------------------------
// Model configs: named presets with a quality dial.
//
// A preset is applied as if the user had typed its flags - the same fields are set and the same
// UserFlags bits are raised - so the planner will not override them and there is only one code
// path. An explicit flag still wins, because scan_user_flags() sees the flag and the preset only
// fills in what the flag did not set.
//
// The file format is "key = value" under a "[name:vN]" header, parsed here rather than with a
// library, because the preflight is the one piece of amp that has to stay small and dependency
// free. An unknown key is a hard error: a typo in a quality dial that silently does nothing is
// worse than a refusal to start.
static std::string trim(const std::string & in) {
    size_t a = in.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) {
        return "";
    }
    size_t b = in.find_last_not_of(" \t\r\n");
    return in.substr(a, b - a + 1);
}

static std::string model_configs_path() {
    if (const char * env = getenv("AMP_MODEL_CONFIGS")) {
        return env;
    }

    // Resolve next to the repository root via /proc/self/exe, so the file is found no matter what
    // the working directory is. This was a real bug: a relative path works when the server is
    // started from the repo root and silently finds nothing from build/, which is exactly where
    // ctest runs from. A quality dial that cannot find its own file is worse than none.
    char        self[4096];
    const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n > 0) {
        self[n] = '\0';
        std::string dir(self);
        size_t slash = dir.find_last_of('/');
        if (slash != std::string::npos) {
            // <root>/build/bin/amp-server -> <root>/configs/model-configs.conf
            dir = dir.substr(0, slash);
            slash = dir.find_last_of('/');
            if (slash != std::string::npos) {
                dir = dir.substr(0, slash);
                slash = dir.find_last_of('/');
                if (slash != std::string::npos) {
                    return dir.substr(0, slash) + "/configs/model-configs.conf";
                }
            }
        }
    }

    return "configs/model-configs.conf";
}

// Reads one "[name:vN]" block. `section` is the BARE name, e.g. "occamy:v2" - the brackets are
// the file's syntax, not part of the name, and mixing the two is an easy way to never match.
static bool read_variant(const std::string & path, const std::string & section,
                        std::vector<std::pair<std::string, std::string>> & out,
                        std::string & err) {
    FILE * fp = fopen(path.c_str(), "r");
    if (!fp) {
        err = "cannot open " + path;
        return false;
    }
    char   raw[1024];
    bool   in = false, found = false;
    while (fgets(raw, sizeof(raw), fp)) {
        std::string line = trim(raw);
        // Strip comments, but only outside a value, so a '#' can appear in one.
        const size_t hash = line.find('#');
        if (hash != std::string::npos) {
            line = trim(line.substr(0, hash));
        }
        if (line.empty()) {
            continue;
        }
        if (line.front() == '[') {
            if (in) {
                break;                      // left the section without finding anything more
            }
            std::string header = line.substr(1, line.size() - 2);
            if (header == section) {
                in = true;
            }
            continue;
        }
        if (!in) {
            continue;
        }
        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
            err = path + ": malformed line in [" + section + "]: " + line;
            fclose(fp);
            return false;
        }
        out.emplace_back(trim(line.substr(0, eq)), trim(line.substr(eq + 1)));
        found = true;
    }
    fclose(fp);
    if (!found) {
        err = path + ": no such variant [" + section + "]";
    }
    return found;
}

// Splits a --model-config value into the preset name and the "[name:suffix]" header to look for.
// Returns false when no variant was named, which the caller reports as a listing.
//
// The suffix is whatever the config file's headers use, so a second axis can be added without
// touching the parser: ":v2" is the KV quality dial, ":s2" a sampler profile. A bare number still
// means the v axis, because "name:2" is the obvious thing to type and there is no reason to make
// it an error.
static bool resolve_variant(const std::string & arg, std::string & name, std::string & want) {
    name = arg;
    want.clear();
    const size_t colon = arg.find(':');
    if (colon == std::string::npos) {
        return false;
    }
    name = arg.substr(0, colon);
    const std::string suffix = arg.substr(colon + 1);
    const bool bare_number =
        !suffix.empty() && suffix.find_first_not_of("0123456789") == std::string::npos;
    want = name + (bare_number ? ":v" : ":") + suffix;
    return true;
}

static bool list_variants(const std::string & path, const std::string & name, std::string & out) {
    FILE * fp = fopen(path.c_str(), "r");
    if (!fp) {
        return false;
    }
    // Every "[name:...]" header, whatever the suffix letter. Listing one axis and not the other
    // would leave the user unable to discover half the dial.
    const std::string prefix = name + ":";
    char raw[1024];
    out  = "variants of \"" + name + "\" in " + path + ":\n";
    bool any = false;
    while (fgets(raw, sizeof(raw), fp)) {
        std::string line = trim(raw);
        const size_t hash = line.find('#');
        if (hash != std::string::npos) {
            line = trim(line.substr(0, hash));
        }
        if (line.size() < prefix.size() + 2 || line.front() != '[' || line.back() != ']') {
            continue;
        }
        const std::string header = line.substr(1, line.size() - 2);
        if (header.compare(0, prefix.size(), prefix) == 0) {
            out += "  --model-config " + header + "\n";
            any = true;
        }
    }
    fclose(fp);
    return any;
}

struct UserFlags {
    bool fit_on = false;   // explicit "--fit on": the fitter is in charge, full no-op
    bool fit_off = false;  // explicit "--fit off"
    bool c = false, b = false, ub = false;
    bool ctk = false, ctv = false, fa = false;
    bool t = false, tb = false;
    bool ctxcp = false, cms = false, cram = false, lzm = false;
    bool np = false;   // explicit --parallel/-np: the user wants N concurrent slots
    bool mc = false;   // explicit --model-config: a preset supplies some settings
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
    f.mc    = argv_has(argv, "-mc") || argv_has(argv, "--model-config");
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

// Splits an extra_args value into argv tokens: whitespace-separated, with double quotes grouping a
// token that contains spaces. The quoting is the only concession to path arguments, and it is
// there because "--chat-template-file /some dir/template.jinja" is a real flag and silently
// splitting it into two arguments would be a confusing way to fail.
static std::vector<std::string> split_args(const std::string & v) {
    std::vector<std::string> out;
    std::string              cur;
    bool                     in_quote = false, have = false;
    for (const char c : v) {
        if (c == '"') {
            in_quote = !in_quote;
            have     = true;
        } else if (!in_quote && (c == ' ' || c == '\t')) {
            if (have) {
                out.push_back(cur);
            }
            cur.clear();
            have = false;
        } else {
            cur += c;
            have = true;
        }
    }
    if (have) {
        out.push_back(cur);
    }
    return out;
}

std::vector<std::string> model_config_extra_args(const std::string & spec, std::string & err) {
    err.clear();
    std::string name, want;
    if (spec.empty() || !resolve_variant(spec, name, want)) {
        return {};
    }
    std::vector<std::pair<std::string, std::string>> kv;
    if (!read_variant(model_configs_path(), want, kv, err)) {
        err.clear();   // apply_preflight() raises the authoritative error for a bad spec
        return {};
    }
    for (const auto & p : kv) {
        if (p.first == "extra_args") {
            return split_args(p.second);
        }
    }
    return {};
}

Result<std::vector<std::string>> apply_preflight(const PreflightOptions & opts,
                                                 common_params & params,
                                                 const std::vector<std::string> & argv) {
    std::vector<std::string> notes;
    // A model config may set n_ubatch, but PlannerOptions does not exist until the plan is set
    // up further down, so the value is carried here and applied there alongside the -ub path.
    int64_t mc_ubatch = 0;
    auto note = [&](const std::string & s) {
        notes.push_back(s);
        AMP_INFO("amp-preflight: ", s);
    };
    UserFlags f = scan_user_flags(argv);

    // -- Model config preset. Applied first, before every rule, because it works by raising the
    //    same UserFlags bits a typed flag would raise: everything downstream then treats these as
    //    user choices and leaves them alone, and an explicit flag still wins because
    //    scan_user_flags() already saw it. One code path, no special cases in the rules.
    if (f.mc) {
        const char * spec = argv_value(argv, "-mc");
        if (!spec) {
            spec = argv_value(argv, "--model-config");
        }
        if (!spec) {
            note("--model-config needs a value, as <name>:v<N> or <name> to list");
            return Status::Error("bad --model-config");
        }

        const std::string path  = model_configs_path();
        const std::string arg  = spec;
        std::string       name;
        std::string       want;
        if (!resolve_variant(arg, name, want)) {
            std::string listing;
            if (!list_variants(path, name, listing)) {
                note("no variants of \"" + name + "\" in " + path);
                return Status::Error("unknown model config");
            }
            note(listing);
            return Status::Error("no variant selected");
        }

        std::vector<std::pair<std::string, std::string>> kv;
        std::string err;
        if (!read_variant(path, want, kv, err)) {
            note("model-config: " + err);
            return Status::Error("model config not found");
        }

        std::string applied, skipped, label;
        for (const auto & p : kv) {
            const std::string & k = p.first;
            const std::string & v = p.second;

            if (k == "label") { label = v; continue; }

            // extra_args was already spliced into argv by amp-server before common_params_parse
            // ran, so llama.cpp has parsed it and the preflight has seen it in argv like any other
            // flag. Re-applying it here would double every argument in it.
            if (k == "extra_args") {
                if (!skipped.empty()) {
                    skipped += ", ";
                }
                skipped += k + " (already on the command line)";
                continue;
            }

            // If the command line already set this one, the preset must not touch it: an
            // explicit flag outranks a preset, and silently overwriting it would be the exact
            // opposite of what --model-config promises.
            const bool already = (k == "cache_type_k"   && f.ctk)   ||
                                 (k == "cache_type_v"   && f.ctv)   ||
                                 (k == "n_ctx"          && f.c)     ||
                                 (k == "cache_ram_mib"  && f.cram)  ||
                                 (k == "n_parallel"     && f.np)    ||
                                 (k == "n_ubatch"       && f.ub)    ||
                                 (k == "n_threads"      && f.t)     ||
                                 (find_sampler_key(k) && sampler_flag_given(argv, k));
            if (already) {
                // Skip before the dispatch chain, not by compounding its conditions: a compound
                // condition would fall through to the unknown-key branch and reject a perfectly
                // good key just because the user also passed the flag.
                if (!skipped.empty()) {
                    skipped += ", ";
                }
                skipped += k;
                continue;
            }

            if (k == "cache_type_k") {
                params.cache_type_k = to_ggml(cache_type_from_string(v));
                if (params.cache_type_k == GGML_TYPE_COUNT) {
                    note("model-config: unknown cache_type_k \"" + v + "\"");
                    return Status::Error("bad model config");
                }
                f.ctk = true;
            } else if (k == "cache_type_v") {
                params.cache_type_v = to_ggml(cache_type_from_string(v));
                if (params.cache_type_v == GGML_TYPE_COUNT) {
                    note("model-config: unknown cache_type_v \"" + v + "\"");
                    return Status::Error("bad model config");
                }
                f.ctv = true;
            } else if (k == "n_ctx") {
                params.n_ctx = (int32_t) atoi(v.c_str());
                f.c = true;
            } else if (k == "cache_ram_mib") {
                params.cache_ram_mib = (int32_t) atoi(v.c_str());
                f.cram = true;
            } else if (k == "n_parallel") {
                params.n_parallel = (int32_t) atoi(v.c_str());
                f.np = true;
            } else if (k == "n_ubatch") {
                // `po` does not exist yet this early; carried out and applied with the -ub path.
                mc_ubatch = std::max<int64_t>(32, atoll(v.c_str()));
                f.ub = true;
            } else if (k == "n_threads") {
                params.cpuparams.n_threads      = (int32_t) atoi(v.c_str());
                params.cpuparams_batch.n_threads = (int32_t) atoi(v.c_str());
                f.t = true;
            } else if (const SamplerKey * sk = find_sampler_key(k)) {
                // The samplers are llama-server request defaults, so they apply to every request
                // that does not carry its own value. A client that sends temperature still wins,
                // which is why the docs say to set them here rather than in the client.
                const bool ok = sk->kind == kSampThinking
                                    ? [&] { int t; return parse_thinking(v, t); }()
                                    : [&] { float f; return parse_sampler_float(*sk, v, f); }();
                if (!ok) {
                    note("model-config: bad value \"" + v + "\" for " + k + " in " + want +
                         (sk->kind == kSampThinking
                              ? " (expected on, off or auto)"
                              : " (expected a number in that sampler's own range)"));
                    return Status::Error("bad model config");
                }
                apply_sampler(*sk, v, params);
            } else {
                // Never silently ignore a key. A config file that quietly does nothing is the
                // worst failure mode a quality dial can have.
                note("model-config: unknown key \"" + k + "\" in " + want);
                return Status::Error("bad model config");
            }

            if (!applied.empty()) {
                applied += ", ";
            }
            applied += k + "=" + v;
        }

        std::string line = "model-config " + name + (label.empty() ? "" : " (" + label + ")") +
                           ": " + (applied.empty() ? "nothing new" : applied);
        if (!skipped.empty()) {
            line += "; kept from the command line: " + skipped;
        }
        note(line);
    }

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

    //    n_parallel is 1, deliberately. llama.cpp's own default is -1 ("auto",
    //    common/arg.cpp:1400), which server.cpp:156-159 expands to FOUR concurrent slots.
    //    Four concurrent generations each want the same shared ~10.9 GiB CPU expert working set,
    //    and on this box that thrashes rather than shares: measured 0.64 t/s with two slots
    //    active, against 28.4 t/s with one. So one slot it is.
    //
    //    It was briefly 2, because a mid-conversation side request - a model testing the API it
    //    is being served by - is served by the slot holding the conversation and destroys its
    //    cached prefix. At 35k context that cost a 110.6 s worst turn against 6.3 s with two
    //    slots. But two slots also halve the context each conversation gets, because llama.cpp's
    //    n_ctx is the total across slots: -c 200000 with two slots is 100096 per conversation,
    //    not 200192.
    //
    //    One slot is the right default here because a 200k single conversation is worth more than
    //    surviving an occasional side request, and the side request is rare while the context is
    //    not. The trade is real and measured in both directions; if you are willing to give up
    //    half the context, --parallel 2 removes the stall.
    if (!f.np && params.n_parallel != 1) {
        note("n_parallel: 1 (llama-server defaults to 4 concurrent slots, which thrash the "
             "shared CPU expert set on this box; 1 also keeps the full -c as one conversation's "
             "context, since n_ctx is the total across slots. --parallel 2 avoids a stall when "
             "the agent tests its own API, at half the context per conversation)");
        params.n_parallel = 1;
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
    //    (tools/server/server-task.cpp) holding KV snapshots keyed by prompt, and on a 14.3 GiB
    //    box carrying a 13.66 GiB model it competes directly with the page cache that both prefill
    //    and decode depend on.
    //
    //    It used to be clamped to 512 here, on the reasoning that a smaller cache would protect
    //    the page cache. Measured, 512 is still far too much to do that: each 18k-token prompt
    //    caches about 200 MiB, so two or three fill it and the churn starts knocking out model
    //    pages. Sustained prefill over 6 distinct 18k prompts, one server process:
    //
    //        -cram 0     median 273.3 t/s   (133 -> 333 -> 231 -> 284 -> 272 -> 273; warms, holds)
    //        -cram 512   median 103.8 t/s   (180 ->  93 ->  82 ->  98 -> 110 -> 110; churns)
    //
    //    **2.6x on prompt processing**, and it costs nothing in agentic terms: with a 35k context
    //    and a self-test request every second turn, -cram 0 gives an 8.7 s worst turn against
    //    10.0 s at 512, still 6/6 cache hits and 0 full re-prefills. Prefix reuse comes from the
    //    slot's own KV, not from this RAM cache - which is what the measurement shows, and what
    //    the previous note assumed rather than checked.
    //
    //    So disable it outright on this box. 0 is llama.cpp's own "off" value for -cram.
    if (!f.cram && params.cache_ram_mib != 0) {
        note("cache_ram_mib: 0 (the prompt cache is anonymous RAM competing with the page cache a "
             "13.66 GiB model needs; 512 is enough to evict it and costs 2.6x on sustained prefill. "
             "Prefix reuse comes from the slot KV, not this cache. Use -cram N to override)");
        params.cache_ram_mib = 0;
    }

    // -- Sampler defaults: the deterministic-coding profile, applied only where neither the command
    //    line nor a model config said anything. The test is the user_sampling_config bit rather
    //    than a separate "did a preset set this" flag, because the bit is already the record of
    //    that fact and duplicating it would be a second thing to keep in step.
    //
    //    Why these values rather than llama.cpp's: the defaults are a general chat sampler, tuned
    //    for prose. This model's job here is tool-calling code generation, where a sampled token
    //    is a malformed edit at best and a hallucinated file write at worst. The bit set is the
    //    whole reason this is a default and not a recommendation: with the bit clear the GGUF's
    //    own sampling metadata overrides it, so a "default" that can be silently replaced by the
    //    model file is not a default.
    //
    //    A client that sends its own temperature still wins per request, so anything that needs
    //    this pinned must not send the field. The model config file is the place to pin it.
    bool touched = false;
    for (const SamplerKey & sk : kSamplerKeys) {
        // No dflt means amp has no opinion on that key, which is not the same as a default of 0:
        // presence_penalty and enable_thinking stay at llama.cpp's own values unless a preset or a
        // flag names them.
        if (!sk.dflt) {
            continue;
        }
        if (sk.bit && (params.sampling.user_sampling_config & sk.bit)) {
            continue;
        }
        // Also skip on the flag, not only on the bit. In production the bit is enough, because
        // llama.cpp's own parser sets it for any sampler passed on the command line
        // (common/arg.cpp:2015-2017). Checking the flag as well means the default cannot overwrite
        // a value that reached params by some other route, which is the failure this loop would
        // otherwise cause silently and only under a caller's own construction of params.
        if (sampler_flag_given(argv, sk.key)) {
            continue;
        }
        apply_sampler(sk, sk.dflt, params);
        touched = true;
    }
    if (touched) {
        const auto num = [](float v) { return std::to_string((double) v); };
        note(std::string("samplers: ") + num(params.sampling.temp) + " / top_p " +
             num(params.sampling.top_p) + " / top_k " + std::to_string(params.sampling.top_k) +
             " / min_p " + num(params.sampling.min_p) + " / repeat " +
             num(params.sampling.penalty_repeat) +
             " (deterministic-coding defaults for tool-calling work; a flag, or a model config "
             "variant such as :s2, overrides any of them, and a request that carries its own "
             "value overrides them all)");
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
    if (f.ub && mc_ubatch > 0) {
        po.ubatch_min = po.ubatch_max = mc_ubatch;
    } else if (f.ub) {
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
