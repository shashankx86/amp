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
#include <algorithm>
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

    // --model-config is amp's own flag, and common_params_parse rejects anything it does not
    // know, so it is removed from the vector handed to llama.cpp and kept in the one handed to
    // the preflight. This is a single well-defined filter rather than a second argument parser,
    // which is what "intercept and strip" would otherwise have meant; everything else in argv
    // still goes straight through, and llama.cpp stays the only owner of its own flags.
    // Other preflight knobs remain environment variables for the same reason. See
    // docs/PREFLIGHT.md.
    std::vector<std::string> args;
    args.reserve((size_t) argc);
    for (int i = 0; i < argc; i++) {
        args.emplace_back(argv[i]);
    }

    const bool has_mc =
        std::find(args.begin(), args.end(), "--model-config") != args.end() ||
        std::find(args.begin(), args.end(), "-mc") != args.end();

    // The value, however it was spelled: "--model-config x", "--model-config=x", same for -mc.
    std::string mc_value;
    for (size_t i = 0; i < args.size(); i++) {
        const std::string & a = args[i];
        if ((a == "--model-config" || a == "-mc") && i + 1 < args.size()) {
            mc_value = args[i + 1];
            break;
        }
        if (a.rfind("--model-config=", 0) == 0) { mc_value = a.substr(15); break; }
        if (a.rfind("-mc=", 0) == 0)            { mc_value = a.substr(4);  break; }
    }
    const std::string * mc_spec = mc_value.empty() ? nullptr : &mc_value;

    // A model config's `extra_args` become real argv entries, here and before common_params_parse.
    // Putting them any later would mean the preflight had to learn a second argument parser and
    // that every flag llama.cpp has would need a matching case here; this way llama.cpp parses
    // them exactly as it parses the user's own, and the preflight then sees them in argv and so
    // treats them as user-supplied. That last part is the point of doing it this early: a flag
    // that arrived via a config is a flag the user asked for, and must not be planner-overridden.
    //
    // Inserted straight after argv[0], and before the user's own arguments, so that a flag present
    // in both the config and the command line is won by the command line: llama.cpp assigns each
    // occurrence in turn, so the last one parsed is the one in effect (common/arg.cpp:790-812
    // applies each handler as it walks argv). This is the same precedence every other key in the
    // config file already has, and putting these last would silently invert it for every flag.
    if (mc_spec) {
        std::string err;
        const std::vector<std::string> extra = amp::model_config_extra_args(*mc_spec, err);
        if (!err.empty()) {
            fprintf(stderr, "amp: model-config %s: %s\n", mc_spec->c_str(), err.c_str());
        }
        if (!extra.empty()) {
            args.insert(args.begin() + 1, extra.begin(), extra.end());
        }
    }

    std::vector<std::string> llama_argv;
    llama_argv.reserve((size_t) argc);
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i] == "--model-config" || args[i] == "-mc") {
            i++;   // skip the value too
            continue;
        }
        if (args[i].rfind("--model-config=", 0) == 0 || args[i].rfind("-mc=", 0) == 0) {
            continue;
        }
        llama_argv.push_back(args[i]);
    }

    // common_params_parse takes char**, so the filtered vector needs a stable char* per entry.
    std::vector<char *> llama_argp;
    llama_argp.reserve(llama_argv.size());
    for (std::string & a : llama_argv) {
        llama_argp.push_back(a.data());
    }

    common_params params;
    if (!common_params_parse((int) llama_argp.size(), llama_argp.data(), params,
                             LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }
    amp::PreflightOptions opts;
    opts.warm     = env_flag("AMP_WARM", false);
    opts.verbose  = env_flag("AMP_VERBOSE", false);
    if (const char * ctx = std::getenv("AMP_CTX")) {
        opts.n_ctx = std::strtoll(ctx, nullptr, 10);
    }

    // "--model-config <name>" with no ":vN" is a query, not a configuration. Serving after it
    // would mean the listing is printed and then a 13.66 GiB model is loaded behind it, which is
    // not what someone asking what variants exist wants. Run the preflight to produce the
    // listing, print it, and stop.
    const bool mc_query = has_mc && mc_spec &&
                          mc_spec->find(':') == std::string::npos;

    // Mutates params in place. No backend init, no model load, no context: llama_server() owns
    // all of that, and doing any of it here would either double-initialise the backend or read
    // the 12 GB of weights twice.
    const amp::Result<std::vector<std::string>> notes = amp::apply_preflight(opts, params, args);

    if (mc_query) {
        for (const std::string & n : *notes) {
            AMP_INFO("amp: ", n);
        }
        return 0;
    }

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
