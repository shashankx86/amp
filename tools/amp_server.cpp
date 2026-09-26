// amp-server: llama.cpp's OpenAI-compatible server, with amp's memory plan in front of it.
//
// This file is deliberately tiny. Everything the server does - the 40+ routes, the slot
// lifecycle, prompt caching, sampling, the chat templates, tool calls, reasoning formats,
// embeddings, Anthropic and OpenAI Responses APIs - is llama.cpp's, linked in statically from
// the same pinned commit we already vendor. We do not reimplement any of it.
//
// amp's entire contribution is the ~200 lines of preflight that turn the measured memory plan
// into `common_params` fields *before* the model loads. The plan is the product: the expert
// working set is 12.19 GiB against a ~9.6 GiB page cache on a 6 GB card, so where the experts
// live is worth 7x on decode. That decision is expressible through public API
// (`tensor_buft_overrides`, the same mechanism as llama.cpp's `-ncmoe`), so it belongs here
// rather than in a hand-written inference loop.
//
//   ./amp-server --model /path/to/model.gguf --port 8081
//
// Every flag llama-server accepts is accepted here, because argv goes straight to
// `common_params_parse`. The preflight never overrides one the user passed explicitly.
#include <cstdlib>
#include <string>
#include <vector>

#include "amp/log.h"
#include "amp/plan/preflight.h"

#include "arg.h"     // common_params_parse, common_params
#include "common.h"  // common_init, LLAMA_EXAMPLE_SERVER

// Declared in tools/server/server.cpp:43. External linkage, but not in any header, so we
// declare it here. argv == nullptr is the "invoked as a library" signal (server.cpp:120,
// `is_run_by_cli = (argv == nullptr)`), which is what keeps router mode out of the way.
// Call it directly rather than llama_server(argc, argv) so the preflight can run in between.
int llama_server(common_params & params, int argc, char ** argv);

namespace {

bool env_flag(const char * name, bool fallback) {
    const char * v = std::getenv(name);
    if (v == nullptr || *v == '\0') {
        return fallback;
    }
    return !(std::string(v) == "0" || std::string(v) == "false" || std::string(v) == "off");
}

} // namespace

int main(int argc, char ** argv) {
    // Must precede anything that logs, and llama_server() calls common_log_flush() on the way
    // out, so the two halves share one logging setup.
    common_init();

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    // Preflight knobs are environment variables rather than new flags on purpose: argv is
    // llama.cpp's, and inventing flags here would mean intercepting and stripping them before
    // common_params_parse sees them. That is a parser we would have to keep in sync with
    // upstream for no benefit. See docs/PREFLIGHT.md.
    amp::PreflightOptions opts;
    opts.warm     = env_flag("AMP_WARM", false);
    opts.verbose  = env_flag("AMP_VERBOSE", false);
    if (const char * ctx = std::getenv("AMP_CTX")) {
        opts.n_ctx = std::strtoll(ctx, nullptr, 10);
    }

    std::vector<std::string> args;
    args.reserve((size_t) argc);
    for (int i = 0; i < argc; i++) {
        args.emplace_back(argv[i]);
    }

    // Mutates params in place. No backend init, no model load, no context: llama_server() owns
    // all of that, and doing any of it here would either double-initialise the backend or read
    // the 12 GB of weights twice.
    const amp::Result<std::vector<std::string>> notes = amp::apply_preflight(opts, params, args);
    if (!notes.ok()) {
        // A failed plan is a performance problem, not a correctness one. Say so and let the
        // server start on its own defaults rather than refusing to serve.
        AMP_WARN("amp: preflight failed (", notes.message(), ") - starting with llama.cpp defaults");
    } else {
        for (const std::string & n : *notes) {
            AMP_INFO("amp: ", n);
        }
    }

    return llama_server(params, 0, nullptr);
}
