// TEMPORARY build spike: proves that llama.cpp's own tools/server can be linked into an amp binary
// and driven through its exported entry point. Delete once tools/amp_server.cpp exists.
#include "arg.h"
#include "common.h"
#include "log.h"

// Declared in tools/server/server.cpp:43, external linkage, not in a header. argv == nullptr is the
// "invoked as a library" signal (server.cpp:120 `is_run_by_cli = (argv == nullptr)`), which is what
// keeps router mode out of the way.
int llama_server(common_params & params, int argc, char ** argv);

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }
    return llama_server(params, 0, nullptr);
}
