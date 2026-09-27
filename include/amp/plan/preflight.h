// Preflight: amp's entire contribution to the llama-server-based amp-server.
//
// amp-server's main() is:
//
//     common_params params;
//     common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER);
//     apply_preflight(opts, params, argv);        // <-- this module
//     llama_server(params, nullptr, nullptr);
//
// The preflight turns amp's MemoryPlanner into common_params fields BEFORE the model
// loads. It never overrides an argument the user explicitly passed, it is a no-op when
// the user asked for a specific device layout or for llama.cpp's own fitter, and it
// never loads the model, creates a context, or touches a backend (llama_server() does
// all of that).
//
// The per-field rules, the "how do we know the user set it" tests, and the llama.cpp
// citations live in docs/PREFLIGHT.md. The short version:
//
//   - llama.cpp's parser keeps NO record of which args were supplied (the seen_args
//     set at common/arg.cpp:814 is local to parse_cli_args and is discarded), and most
//     common_params fields have no unset sentinel that survives parsing. So the
//     preflight scans argv itself; that is the only reliable test for most fields.
//   - The one exception that IS reliably detectable: n_gpu_layers (-1 = auto/unset,
//     common/common.h:473) and tensor_buft_overrides (empty = unset, common/common.h:527).
//   - tensor_buft_overrides is walked by the tensor loader until pattern == nullptr
//     (src/llama-model-loader.cpp:1237) and common_model_params_to_llama asserts the
//     last entry is null (common/common.cpp:1706, GGML_ASSERT is always active). The
//     buffer is padded to llama_max_tensor_buft_overrides() = 4096 by
//     common_params_parse_ex (common/arg.cpp:940-944) BEFORE this preflight runs, so
//     the preflight writes into the padded slots in place and never push_backs.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "amp/model/geometry.h" // CacheType
#include "amp/status.h"
#include "arg.h"                // common_params (llama.cpp)

namespace amp {

struct PreflightOptions {
    // Page-cache warm before the model load. Default off: llama.cpp's model load
    // performs a full sequential fault-in of the GGUF (MAP_POPULATE,
    // src/llama-mmap.cpp:480), so a pre-load warm is redundant unless lazy_mode is
    // forced on. See docs/PREFLIGHT.md section 6.
    bool warm = false;
    uint64_t warm_chunk = 2ull * 1024 * 1024; // per pread() in the warm

    // Context size to plan for, and to set when the user did not pass -c. 200000 is
    // the measured optimum: the KV cache (8320 B/token) plus the compute buffers fit
    // the 6 GB card there, the model's native 262144 does not.
    int64_t n_ctx = 200000;

    // KV cache dtypes. Must stay q8_0/q4_0: those are the dtypes the BENCH.md
    // numbers were measured with (zero-quality-loss constraint, AGENT.md).
    CacheType cache_k = CacheType::kQ8_0;
    CacheType cache_v = CacheType::kQ8_0;

    // CPU threads when the user did not pass -t. 8 = ncpu/2 on this 8C/16T box.
    int64_t n_threads = 8;

    // Safety guards for llama.cpp server defaults that are dangerous on this machine.
    // n_ctx_checkpoints: llama.cpp defaults to 32 (common/common.h:629); each checkpoint
    // at 200k ctx stores the full memory state (~1.6 GiB), so the default ring can
    // reach ~50 GiB. cache_ram_mib: llama.cpp defaults to 8192 (common/common.h:632);
    // the prompt cache is anonymous RAM that evicts the model's page cache.
    int32_t n_ctx_checkpoints = 2;
    int32_t cache_ram_mib = 512;

    bool verbose = false; // log every decision, not just the plan
};

// Mutates params in place. Returns notes for logging (empty on a full no-op).
// Never throws; failures are reported as a failed Result or as a note + skip.
Result<std::vector<std::string>> apply_preflight(const PreflightOptions & opts,
                                                 common_params & params,
                                                 const std::vector<std::string> & argv);

} // namespace amp
