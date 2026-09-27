#include "amp/plan/model_config.h"
#include "amp/log.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

namespace amp {
namespace {

using json = nlohmann::json;

// Keys beginning with "_" are comments in the file and are ignored everywhere, so the file can
// document itself without the loader growing a comment syntax.
bool is_comment_key(const std::string & k) {
    return !k.empty() && k[0] == '_';
}

std::vector<std::string> variant_keys(const json & variants) {
    std::vector<std::string> out;
    for (const auto & el : variants.items()) {
        if (!is_comment_key(el.key())) {
            out.push_back(el.key());
        }
    }
    return out;
}

// Accept both "2" and "v2" for the same variant, because the file uses v-prefixed keys and
// people will type the bare number.
std::string resolve_variant(const json & variants, const std::string & want) {
    if (variants.contains(want) && !is_comment_key(want)) {
        return want;
    }
    if (!want.empty() && (want[0] == 'v' || want[0] == 'V')) {
        const std::string bare = want.substr(1);
        if (variants.contains(bare) && !is_comment_key(bare)) {
            return bare;
        }
    } else {
        const std::string pref = "v" + want;
        if (variants.contains(pref) && !is_comment_key(pref)) {
            return pref;
        }
    }
    return "";
}

json read_file(const std::string & path, std::string & err) {
    std::ifstream in(path);
    if (!in) {
        err = "cannot open " + path;
        return json();
    }
    try {
        json j;
        in >> j;
        return j;
    } catch (const std::exception & e) {
        err = path + ": " + e.what();
        return json();
    }
}

std::string join(const std::vector<std::string> & v, const char * sep) {
    std::string out;
    for (size_t i = 0; i < v.size(); i++) {
        if (i) {
            out += sep;
        }
        out += v[i];
    }
    return out;
}

} // namespace

std::string default_model_config_path() {
    if (const char * p = std::getenv("AMP_MODEL_CONFIG")) {
        if (*p) {
            return p;
        }
    }
    // Relative to the working directory, which is how every other path in this project is
    // handled: the server is always started from the repository root.
    return "config/model-configs.json";
}

bool parse_model_config_spec(const std::string & spec, std::string & name, std::string & variant,
                             std::string & err) {
    name.clear();
    variant.clear();

    const size_t colon = spec.find(':');
    if (colon == std::string::npos) {
        if (spec.empty()) {
            err = "empty --model-config value; expected NAME:VARIANT, e.g. occamy:v2";
            return false;
        }
        name = spec;
        return true;
    }

    name    = spec.substr(0, colon);
    variant = spec.substr(colon + 1);
    if (name.empty() && variant.empty()) {
        err = "--model-config value is just ':'; expected NAME:VARIANT, e.g. occamy:v2";
        return false;
    }
    return true;
}

Result<ModelConfig> load_model_config(const std::string & spec) {
    std::string name;
    std::string variant;
    std::string err;

    if (!parse_model_config_spec(spec, name, variant, err)) {
        return Err<ModelConfig>(err);
    }

    const std::string path = default_model_config_path();
    const json root = read_file(path, err);
    if (!err.empty()) {
        return Err<ModelConfig>("model config: " + err +
                                " (set AMP_MODEL_CONFIG to point somewhere else)");
    }
    if (!root.contains("configs")) {
        return Err<ModelConfig>("model config: " + path + " has no \"configs\" object");
    }
    const json & configs = root["configs"];

    if (name.empty()) {
        if (configs.contains("default") && !is_comment_key("default")) {
            name = configs["default"].get<std::string>();
        } else if (configs.size() == 1) {
            name = configs.begin().key();
        } else {
            return Err<ModelConfig>("model config: " + path + " has no default config; use "
                                    "--model-config NAME:VARIANT. Available: " +
                                    list_model_configs());
        }
    }

    if (!configs.contains(name)) {
        return Err<ModelConfig>("model config: no config named \"" + name + "\" in " + path +
                                ". Available: " + list_model_configs());
    }
    const json & cfg = configs[name];
    if (!cfg.contains("variants")) {
        return Err<ModelConfig>("model config: \"" + name + "\" has no \"variants\" object");
    }
    const json & variants = cfg["variants"];

    std::string want = variant;
    if (want.empty()) {
        if (variants.contains("default") && !is_comment_key("default")) {
            want = variants["default"].get<std::string>();
        } else {
            return Err<ModelConfig>("model config: \"" + name + "\" needs a variant; use "
                                    "--model-config " + name + ":VARIANT. Available: " +
                                    name + ":" + join(variant_keys(variants), ", "));
        }
    }

    const std::string key = resolve_variant(variants, want);
    if (key.empty()) {
        return Err<ModelConfig>("model config: \"" + name + "\" has no variant \"" + want +
                                "\". Available: " + name + ":" +
                                join(variant_keys(variants), ", "));
    }

    const json & v = variants[key];

    ModelConfig out;
    out.name    = name;
    out.variant = key;

    try {
        if (v.contains("label") && !is_comment_key("label")) {
            out.label = v.value("label", std::string());
        }
        if (v.contains("quality") && !is_comment_key("quality")) {
            out.quality = v.value("quality", 0);
        }
        if (v.contains("n_ctx") && !is_comment_key("n_ctx")) {
            out.n_ctx         = v.at("n_ctx").get<int64_t>();
            out.has_n_ctx     = true;
        }
        if (v.contains("cache_type_k") && !is_comment_key("cache_type_k")) {
            out.cache_type_k     = v.at("cache_type_k").get<std::string>();
            out.has_cache_type_k = true;
        }
        if (v.contains("cache_type_v") && !is_comment_key("cache_type_v")) {
            out.cache_type_v     = v.at("cache_type_v").get<std::string>();
            out.has_cache_type_v = true;
        }
        if (v.contains("cache_ram_mib") && !is_comment_key("cache_ram_mib")) {
            out.cache_ram_mib     = v.at("cache_ram_mib").get<int32_t>();
            out.has_cache_ram_mib = true;
        }
        if (v.contains("n_ctx_checkpoints") && !is_comment_key("n_ctx_checkpoints")) {
            out.n_ctx_checkpoints     = v.at("n_ctx_checkpoints").get<int32_t>();
            out.has_n_ctx_checkpoints = true;
        }
        if (v.contains("n_ubatch") && !is_comment_key("n_ubatch")) {
            out.n_ubatch     = v.at("n_ubatch").get<int64_t>();
            out.has_n_ubatch = true;
        }
        if (v.contains("n_threads") && !is_comment_key("n_threads")) {
            out.n_threads     = v.at("n_threads").get<int64_t>();
            out.has_n_threads = true;
        }
    } catch (const std::exception & e) {
        return Err<ModelConfig>("model config: " + name + ":" + key + ": " + e.what());
    }

    return Ok(out);
}

std::string list_model_configs() {
    const std::string path = default_model_config_path();
    std::string err;
    const json root = read_file(path, err);
    if (!err.empty() || !root.contains("configs")) {
        return "(none: " + err + ")";
    }
    std::vector<std::string> out;
    for (const auto & cfg : root["configs"].items()) {
        if (is_comment_key(cfg.key()) || !cfg.value().contains("variants")) {
            continue;
        }
        for (const std::string & v : variant_keys(cfg.value()["variants"])) {
            out.push_back(cfg.key() + ":" + v);
        }
    }
    return join(out, ", ");
}

} // namespace amp
