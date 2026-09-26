// amp-infer: amp's forward path, benchmarked against the llama.cpp numbers in docs/BENCH.md.
//
//   ./build/amp-infer --model PATH --prompt "..." --n-predict 128
//   ./build/amp-infer --model PATH --prompt-file big.txt --no-prefetch   # A/B baseline
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "amp/bytes.h"
#include "amp/format.h"
#include "amp/log.h"
#include "amp/model/geometry.h"
#include "amp/plan/cost_model.h"
#include "amp/plan/memory_plan.h"
#include "amp/runtime/model_runtime.h"
#include "amp/timing.h"

using namespace amp;

namespace {

void usage() {
    printf(R"(amp-infer - amp forward path benchmark

usage: amp-infer --model PATH [options]

options:
  --model PATH        GGUF model (required, or set AMP_MODEL)
  --prompt TEXT       prompt text (default: a short built-in prompt)
  --prompt-file PATH  read the prompt from a file (use a long one for real prefill numbers)
  --n-predict N       tokens to generate (default 64)
  --ctx N             context size (default 200000)
  --ctk TYPE          K cache type (default q8_0)
  --ctv TYPE          V cache type (default q4_0)
  --gpu-layers N      expert layers on the GPU (default: from the planner)
  --ubatch N          ubatch size (default: from the planner)
  --threads N         CPU threads (default 8)
  --no-prefetch       disable the page-cache prefetcher (baseline A/B)
  --forward-warm      warm expert ranges forwards instead of in reverse
  --drop-cache        evict the model from the page cache first
  --repeat N          repeat the whole prefill+decode N times (default 1)
  --temp F            sampler temperature (default 0.6)
  --top-p F           sampler top-p (default 0.95)
  --top-k N           sampler top-k (default 20)
  --json              machine-readable output
  -v, --verbose       debug logging
)");
}

std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return std::string();
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

} // namespace

int main(int argc, char ** argv) {
    RuntimeConfig cfg;
    std::string   prompt_file;
    std::string   prompt;
    int32_t       n_predict     = 64;
    int32_t       repeat        = 1;
    int32_t       gpu_layers    = -1;
    int64_t       ubatch        = -1;
    bool          drop_cache    = false;
    bool          json          = false;
    std::string   dump_output;
    std::string   dump_logprobs;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char * w) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "amp-infer: %s needs a value\n", w); exit(2); }
            return argv[++i];
        };
        if (a == "--model" || a == "-m") cfg.model_path = next("--model");
        else if (a == "--prompt") prompt = next("--prompt");
        else if (a == "--prompt-file") prompt_file = next("--prompt-file");
        else if (a == "--n-predict" || a == "-n") n_predict = atoi(next("--n-predict").c_str());
        else if (a == "--ctx" || a == "-c") cfg.n_ctx = atoll(next("--ctx").c_str());
        else if (a == "--ctk") cfg.cache_k = cache_type_from_string(next("--ctk"));
        else if (a == "--ctv") cfg.cache_v = cache_type_from_string(next("--ctv"));
        else if (a == "--gpu-layers") gpu_layers = atoi(next("--gpu-layers").c_str());
        else if (a == "--ubatch") ubatch = atoll(next("--ubatch").c_str());
        else if (a == "--threads") cfg.n_threads = atoi(next("--threads").c_str());
        else if (a == "--no-prefetch") cfg.prefetch = false;
        else if (a == "--forward-warm") cfg.reverse_warm = false;
        else if (a == "--drop-cache") drop_cache = true;
        else if (a == "--repeat") repeat = atoi(next("--repeat").c_str());
        else if (a == "--temp") cfg.temperature = (float) atof(next("--temp").c_str());
        else if (a == "--top-p") cfg.top_p = (float) atof(next("--top-p").c_str());
        else if (a == "--top-k") cfg.top_k = atoi(next("--top-k").c_str());
        else if (a == "--dump-output") dump_output = next("--dump-output");
        else if (a == "--dump-logprobs") dump_logprobs = next("--dump-logprobs");
        else if (a == "--json") json = true;
        else if (a == "-v" || a == "--verbose") { cfg.verbose = true; set_log_level(LogLevel::kDebug); }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "amp-infer: unknown argument '%s'\n", a.c_str()); usage(); return 2; }
    }

    if (cfg.model_path.empty()) {
        const char * env = getenv("AMP_MODEL");
        cfg.model_path = env ? env : "";
    }
    if (cfg.model_path.empty()) {
        fprintf(stderr, "amp-infer: --model is required (or set AMP_MODEL)\n");
        return 2;
    }
    if (!prompt_file.empty()) {
        prompt = read_file(prompt_file);
    }
    if (prompt.empty()) {
        prompt =
            "You are a helpful assistant. Explain, in detail, how a modern CPU's out-of-order "
            "execution works: branch prediction, the reorder buffer, register renaming, and why "
            "they improve instruction-level parallelism. Use concrete numbers where you can.";
    }

    // ---- model + plan ----
    auto gguf_res = GGUFFile::open(cfg.model_path);
    if (!gguf_res.ok()) { fprintf(stderr, "amp-infer: %s\n", gguf_res.message().c_str()); return 1; }
    const GGUFFile & gguf = *gguf_res;
    auto geo_res = ModelGeometry::build(gguf);
    if (!geo_res.ok()) { fprintf(stderr, "amp-infer: %s\n", geo_res.message().c_str()); return 1; }
    const ModelGeometry & geo = **geo_res;

    PlannerOptions opts;
    opts.n_ctx  = cfg.n_ctx;
    opts.cache_k = cfg.cache_k;
    opts.cache_v = cfg.cache_v;
    DeviceBudget budget = detect_device_budget();
    const CostModel cost = CostModel::from_environment();
    auto plan_res = MemoryPlanner::plan(geo, budget, opts, cost);
    if (!plan_res.ok()) { fprintf(stderr, "amp-infer: %s\n", plan_res.message().c_str()); return 1; }
    ExecutionPlan plan = *plan_res;
    if (gpu_layers >= 0) { plan.n_expert_layers_gpu = gpu_layers; }
    if (ubatch > 0) { plan.ubatch = ubatch; }
    cfg.n_ubatch = plan.ubatch;
    cfg.n_batch  = (int32_t) std::max<int64_t>(cfg.n_ubatch, (int64_t) cfg.n_batch);
    cfg.n_threads_batch = cfg.n_threads;

    if (!json) {
        printf("amp-infer: %s\n", cfg.model_path.c_str());
        printf("  plan: g=%d ubatch=%lld vram=%s predicted pp=%.0f t/s tg=%.1f t/s\n",
               plan.n_expert_layers_gpu, (long long) plan.ubatch,
               human_bytes((uint64_t) plan.vram_total).c_str(), plan.predicted_prefill_tps,
               plan.predicted_decode_tps);
        printf("  prefetch=%s order=%s\n", cfg.prefetch ? "on" : "off",
               cfg.reverse_warm ? "reverse" : "forward");
    }

    // ---- optional cold start ----
    if (drop_cache) {
        // POSIX_FADV_DONTNEED on the expert ranges only; the header stays cached.
        for (int64_t il = 0; il < geo.n_layer(); il++) {
            for (const auto & r : geo.expert_ranges((int) il)) {
                (void) gguf.file().drop_cache(r.offset, r.length);
            }
        }
        if (!json) { printf("  dropped expert pages from the page cache\n"); }
    }

    // ---- runtime ----
    auto rt_res = ModelRuntime::create(cfg, plan, geo, gguf.file());
    if (!rt_res.ok()) { fprintf(stderr, "amp-infer: %s\n", rt_res.message().c_str()); return 1; }
    const std::unique_ptr<ModelRuntime> & rt = *rt_res;

    auto tok_res = rt->tokenize(prompt, /*add_special=*/ true);
    if (!tok_res.ok()) { fprintf(stderr, "amp-infer: %s\n", tok_res.message().c_str()); return 1; }
    const std::vector<llama_token> & toks = *tok_res;

    double pp_sum = 0.0, tg_sum = 0.0;
    int64_t pp_toks = 0, tg_toks = 0;
    for (int r = 0; r < repeat; r++) {
        const Status st = rt->prefill(toks);
        if (!st.ok()) { fprintf(stderr, "amp-infer: %s\n", st.message().c_str()); return 1; }
        std::string out;
        auto gen = rt->generate(n_predict, &out);
        if (!gen.ok()) { fprintf(stderr, "amp-infer: %s\n", gen.message().c_str()); return 1; }
        pp_sum  += rt->stats().prefill_ms;
        tg_sum  += rt->stats().decode_ms;
        pp_toks  = rt->stats().prefill_tokens;
        tg_toks  = rt->stats().decode_tokens;
        if (r + 1 < repeat) {
            // Reset the KV for the next repetition.
            llama_memory_clear(llama_get_memory((llama_context *) nullptr), true);
        }
    }

    if (!dump_output.empty()) {
        std::ofstream f(dump_output, std::ios::binary);
        f << rt->last_text();
    }
    if (!dump_logprobs.empty()) {
        std::ofstream f(dump_logprobs);
        for (const auto & pos : rt->logprobs()) {
            for (size_t i = 0; i < pos.size(); i++) {
                f << (i ? "\t" : "") << pos[i].token << "\t" << std::setprecision(9) << pos[i].logprob;
            }
            f << "\n";
        }
    }

    const RuntimeStats & st = rt->stats();
    const MemInfo        mi = read_meminfo();

    if (json) {
        printf("{\n");
        printf("  \"prompt_tokens\": %lld,\n", (long long) toks.size());
        printf("  \"generated_tokens\": %lld,\n", (long long) tg_toks);
        printf("  \"prefill_ms\": %.1f,\n", pp_sum);
        printf("  \"prefill_tps\": %.2f,\n", pp_sum > 0 ? pp_toks * 1e3 / pp_sum : 0.0);
        printf("  \"decode_tps\": %.2f,\n", tg_sum > 0 ? tg_toks * 1e3 / tg_sum : 0.0);
        printf("  \"ubatch\": %lld,\n", (long long) cfg.n_ubatch);
        printf("  \"gpu_expert_layers\": %d,\n", plan.n_expert_layers_gpu);
        printf("  \"prefetch\": %s,\n", cfg.prefetch ? "true" : "false");
        printf("  \"reverse_warm\": %s,\n", cfg.reverse_warm ? "true" : "false");
        printf("  \"warm_bytes_issued\": %llu,\n", (unsigned long long) st.warm_bytes_issued);
        printf("  \"predicted_prefill_tps\": %.2f,\n", plan.predicted_prefill_tps);
        printf("  \"cached_bytes\": %llu\n", (unsigned long long) mi.cached_bytes);
        printf("}\n");
        return 0;
    }

    printf("\n== result (%lld prompt tokens, %lld generated) ==\n", (long long) toks.size(),
           (long long) tg_toks);
    printf("  ubatch    %lld (after VRAM verification)\n", (long long) cfg.n_ubatch);
    printf("  prefill   %8.2f s   %7.1f t/s   (predicted %.0f t/s)\n", pp_sum / 1e3,
           pp_sum > 0 ? pp_toks * 1e3 / pp_sum : 0.0, plan.predicted_prefill_tps);
    printf("  decode    %8.2f s   %7.1f t/s   (predicted %.1f t/s)\n", tg_sum / 1e3,
           tg_sum > 0 ? tg_toks * 1e3 / tg_sum : 0.0, plan.predicted_decode_tps);
    printf("  warm      %s issued in %s calls\n", human_bytes(st.warm_bytes_issued).c_str(),
           human_count(st.warm_calls).c_str());
    printf("  cache     %s cached, %s free\n", human_bytes(mi.cached_bytes).c_str(),
           human_bytes(mi.mem_available_bytes).c_str());
    return 0;
}
