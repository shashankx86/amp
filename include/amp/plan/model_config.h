#pragma once

// Named model configurations: a set of DEFAULTS selected by name and variant.
//
//     amp-server --model M --model-config occamy:v2
//
// Why a file rather than code: the presets are the values docs/BENCH.md measured, and
// measurements get revised. Putting them in a data file means retuning one is an edit rather
// than a rebuild, and it keeps the numbers next to the numbers they came from.
//
// The one rule that matters: a config supplies DEFAULTS and never overrides. Any flag the user
// passed on the command line still wins, which is the same contract the preflight already has
// for llama.cpp flags, and it is why `--model-config occamy:v2 -ctv f16` behaves sensibly
// instead of silently ignoring the -ctv.

#include "amp/result.h"

#include <cstdint>
#include <string>

namespace amp {

// One resolved variant. Every field is optional: absent means "leave the built-in default
// alone", which is what lets a config file carry only the keys it cares about.
struct ModelConfig {
    std::string name;     // the config group, e.g. "occamy"
    std::string variant;  // the variant key, e.g. "v2"
    std::string label;    // human-readable, for the startup log
    int         quality = 0;

    bool        has_n_ctx             = false;
    int64_t     n_ctx                 = 0;
    bool        has_cache_type_k      = false;
    std::string cache_type_k;
    bool        has_cache_type_v      = false;
    std::string cache_type_v;
    bool        has_cache_ram_mib     = false;
    int32_t     cache_ram_mib         = 0;
    bool        has_n_ctx_checkpoints = false;
    int32_t     n_ctx_checkpoints     = 0;
    bool        has_n_ubatch          = false;
    int64_t     n_ubatch              = 0;
    bool        has_n_threads         = false;
    int64_t     n_threads             = 0;
};

// Where the config file lives. AMP_MODEL_CONFIG overrides the path, which is how you point at
// your own presets without touching the tree.
std::string default_model_config_path();

// Parse "name:variant". A bare "name" selects the config's "default" variant if it has one.
// A bare ":variant" or an empty name selects the file's default config.
// Returns false and fills `err` on a malformed spec.
bool parse_model_config_spec(const std::string & spec, std::string & name, std::string & variant,
                             std::string & err);

// Load and resolve one variant. Fails if the file is missing or unreadable, if the config or
// variant is not in it, or if a value cannot be parsed. Never falls back silently: a config
// the user asked for by name and did not get is a mistake worth reporting.
Result<ModelConfig> load_model_config(const std::string & spec);

// List the available "name:variant" pairs, for `--model-config help` and for the error message
// when a user asks for one that does not exist. An empty string means the file was unreadable.
std::string list_model_configs();

} // namespace amp
