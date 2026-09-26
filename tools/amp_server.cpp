// amp-server: the OpenAI-compatible server.
//
//   ./build/bin/amp-server --model PATH [--port 8081] [--ctx 200000] ...
//
// Same flags as llama-server where they mean the same thing, so existing clients (OpenCode,
// OpenWebUI, curl, anything speaking /v1/chat/completions) work unchanged.
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "amp/bytes.h"
#include "amp/format.h"
#include "amp/log.h"
#include "amp/server/http_server.h"
#include "amp/server/openai_api.h"
#include "amp/server/service.h"
#include "amp/timing.h"

using namespace amp;

namespace {
InferenceService *     g_service = nullptr;
http::Server *        g_server  = nullptr;

// A signal handler must do almost nothing, so: flip the stop flag (atomic) and shut the listening
// socket down so the poll() in serve() returns. Without the shutdown, the process ignored SIGTERM
// entirely - the old handler only interrupted generation and serve() never looked at the flag.
void on_signal(int) {
    if (g_service) {
        g_service->interrupt();
    }
    if (g_server) {
        g_server->stop();
    }
}
} // namespace

static void usage() {
    printf(R"(amp-server - OpenAI-compatible server for one model

usage: amp-server --model PATH [options]

  --model PATH, -m PATH   GGUF model (required, or set AMP_MODEL)
  --host HOST             bind address (default 127.0.0.1)
  --port PORT             bind port (default 8081)
  --ctx N                 context length (default 200000)
  --parallel N            concurrent sequences (default 1; >1 splits VRAM for KV)
  --ctk TYPE              K cache type (default q8_0)
  --ctv TYPE              V cache type (default q4_0)
  --gpu-layers N          expert layers on the GPU (default: from the planner)
  --ubatch N              ubatch (default: from the planner, reduced to fit VRAM)
  --threads N             CPU threads (default 8)
  --temp F                default temperature (default 0.6)
  --top-p F               default top-p (default 0.95)
  --top-k N               default top-k (default 20)
  --seed N                default seed
  --n-predict N           default max_tokens when a request omits it (default 256)
  --api-key KEY           require Authorization: Bearer KEY
  --model-id NAME         model name reported by /v1/models (default amp)
  --no-prefetch           disable the page-cache prefetcher
  --no-warm               do not warm the page cache at start-up (faster start, slower first token)
  --warm-chunk MB         warm read size in MiB (default 16)
  --api-key-file PATH     read the API key from a file
  -v, --verbose           debug logging
  -h, --help              this help
)");
}

int main(int argc, char ** argv) {
    ServerConfig cfg;
    std::string  api_key_file;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char * w) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "amp-server: %s needs a value\n", w);
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--model" || a == "-m") cfg.model_path = next("--model");
        else if (a == "--host") cfg.host = next("--host");
        else if (a == "--port") cfg.port = (uint16_t) atoi(next("--port").c_str());
        else if (a == "--ctx" || a == "-c") cfg.n_ctx = atoll(next("--ctx").c_str());
        else if (a == "--parallel" || a == "-np") cfg.n_seq = atoi(next("--parallel").c_str());
        else if (a == "--ctk") cfg.cache_k = cache_type_from_string(next("--ctk"));
        else if (a == "--ctv") cfg.cache_v = cache_type_from_string(next("--ctv"));
        else if (a == "--gpu-layers" || a == "-ngl") cfg.n_gpu_expert_layers = atoi(next("--gpu-layers").c_str());
        else if (a == "--ubatch") cfg.n_ubatch = atoll(next("--ubatch").c_str());
        else if (a == "--threads" || a == "-t") cfg.n_threads = atoi(next("--threads").c_str());
        else if (a == "--temp") cfg.temperature = (float) atof(next("--temp").c_str());
        else if (a == "--top-p") cfg.top_p = (float) atof(next("--top-p").c_str());
        else if (a == "--top-k") cfg.top_k = atoi(next("--top-k").c_str());
        else if (a == "--seed") cfg.seed = (uint32_t) strtoul(next("--seed").c_str(), nullptr, 10);
        else if (a == "--n-predict" || a == "-n") cfg.n_predict = atoi(next("--n-predict").c_str());
        else if (a == "--api-key") cfg.api_key = next("--api-key");
        else if (a == "--api-key-file") api_key_file = next("--api-key-file");
        else if (a == "--model-id") cfg.model_id = next("--model-id");
        else if (a == "--no-prefetch") cfg.prefetch = false;
        else if (a == "--no-warm") cfg.warm = false;
        else if (a == "--warm-chunk") cfg.warm_chunk = (uint64_t) atoll(next("--warm-chunk").c_str()) << 20;
        else if (a == "-v" || a == "--verbose") { cfg.verbose = true; set_log_level(LogLevel::kDebug); }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "amp-server: unknown argument '%s'\n", a.c_str()); usage(); return 2; }
    }

    if (cfg.model_path.empty()) {
        const char * env = getenv("AMP_MODEL");
        cfg.model_path = env ? env : "";
    }
    if (cfg.model_path.empty()) {
        fprintf(stderr, "amp-server: --model is required (or set AMP_MODEL)\n");
        return 2;
    }
    if (!api_key_file.empty()) {
        FILE * f = fopen(api_key_file.c_str(), "r");
        if (!f) {
            fprintf(stderr, "amp-server: cannot open %s\n", api_key_file.c_str());
            return 1;
        }
        char buf[256] = { 0 };
        if (fgets(buf, sizeof(buf), f)) {
            cfg.api_key = buf;
            while (!cfg.api_key.empty() && (cfg.api_key.back() == '\n' || cfg.api_key.back() == '\r')) {
                cfg.api_key.pop_back();
            }
        }
        fclose(f);
    }

    auto svc_res = InferenceService::create(cfg);
    if (!svc_res.ok()) {
        fprintf(stderr, "amp-server: %s\n", svc_res.message().c_str());
        return 1;
    }
    InferenceService & svc = **svc_res;
    g_service = &svc;

    auto srv_res = http::Server::listen(cfg.host, cfg.port);
    if (!srv_res.ok()) {
        fprintf(stderr, "amp-server: %s\n", srv_res.message().c_str());
        return 1;
    }
    const std::unique_ptr<http::Server> & srv = *srv_res;
    g_server = srv.get();
    OpenAIApi::register_routes(*srv, svc, cfg);

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    const MemInfo mi = read_meminfo();
    printf("amp-server: listening on http://%s:%u\n", cfg.host.c_str(), srv->port());
    printf("  %s\n", svc.status_line().c_str());
    printf("  page cache: %s cached, %s headroom  (close the browser for full decode speed)\n",
           human_bytes(mi.cached_bytes).c_str(), human_bytes(mi.cache_headroom()).c_str());
    fflush(stdout);

    (void) srv->serve();
    AMP_INFO("amp: stopped");
    return 0;
}
