// amp-plan: inspect the model, detect the box, and show the ranked device/ubatch plans.
//
//   ./build/amp-plan --model /path/to/model.gguf
//   ./build/amp-plan --model ... --ctx 65536 --json
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "amp/bytes.h"
#include "amp/format.h"
#include "amp/log.h"
#include "amp/model/geometry.h"
#include "amp/plan/cost_model.h"
#include "amp/plan/memory_plan.h"
#include "amp/timing.h"

using namespace amp;

namespace {

void usage() {
    printf(R"(amp-plan - memory planner for a single-model inference engine

usage: amp-plan --model PATH [options]

options:
  --model PATH        GGUF model (required)
  --ctx N             context length to plan for (default 200000)
  --ctk TYPE          K cache type (default q8_0)
  --ctv TYPE          V cache type (default q8_0)
  --ubatch-max N      largest ubatch to consider (default 2048)
  --max-gpu-layers N  cap on expert layers placed on the GPU (default 8)
  --cache-target B    page cache budget, e.g. 11.5GiB (default: auto-detected)
  --vram-total B      override total VRAM (default: auto-detected)
  --vram-reserved B   override VRAM already in use (default: auto-detected)
  --no-streaming      assume everything is kept resident (no streaming tail)
  --prefer-decode     bias the plan towards generation: at 200k this moves a layer of experts
                      from the CPU to the GPU, 3.3% faster decode, at 181 -> 119 t/s prefill
  --resident-frac F   fraction of the CPU expert set to keep resident (0..1, default 1.0)
  --top N             number of candidates to print (default 8)
  --json              machine-readable output
  -v, --verbose       debug logging
)");
}

const char * env_or(const char * name, const char * def) {
    const char * v = getenv(name);
    return v ? v : def;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model;
    PlannerOptions opts;
    DeviceBudget   budget;
    bool           budget_overridden = false;
    bool           json             = false;
    // Rows of the candidate table to print. Was parsed from --top and then never read, so the
    // documented flag silently did nothing.
    size_t         top              = 8;

    const char * cache_k_str = nullptr;
    const char * cache_v_str = nullptr;
    uint64_t     cache_budget = 0, vram_total = 0, vram_reserved = 0;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "amp-plan: %s needs a value\n", what);
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--model" || a == "-m") {
            model = next("--model");
        } else if (a == "--ctx" || a == "-c") {
            opts.n_ctx = atoll(next("--ctx").c_str());
        } else if (a == "--ctk") {
            cache_k_str = strdup(next("--ctk").c_str());
        } else if (a == "--ctv") {
            cache_v_str = strdup(next("--ctv").c_str());
        } else if (a == "--ubatch-max") {
            opts.ubatch_max = atoll(next("--ubatch-max").c_str());
        } else if (a == "--max-gpu-layers") {
            opts.max_expert_layers_gpu = atoi(next("--max-gpu-layers").c_str());
        } else if (a == "--cache-target") {
            parse_bytes(next("--cache-target"), &cache_budget);
            budget_overridden = true;
        } else if (a == "--vram-total") {
            parse_bytes(next("--vram-total"), &vram_total);
            budget_overridden = true;
        } else if (a == "--vram-reserved") {
            parse_bytes(next("--vram-reserved"), &vram_reserved);
            budget_overridden = true;
        } else if (a == "--no-streaming") {
            opts.allow_streaming = false;
        } else if (a == "--prefer-decode") {
            opts.prefer_decode = true;
        } else if (a == "--resident-frac") {
            opts.resident_fraction = atof(next("--resident-frac").c_str());
        } else if (a == "--top") {
            top = (size_t) atoi(next("--top").c_str());
        } else if (a == "--json") {
            json = true;
        } else if (a == "-v" || a == "--verbose") {
            set_log_level(LogLevel::kDebug);
        } else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            fprintf(stderr, "amp-plan: unknown argument '%s'\n", a.c_str());
            usage();
            return 2;
        }
    }

    if (model.empty()) {
        model = env_or("AMP_MODEL", "");
    }
    if (model.empty()) {
        fprintf(stderr, "amp-plan: --model is required (or set AMP_MODEL)\n");
        return 2;
    }
    opts.cache_k = cache_type_from_string(cache_k_str ? cache_k_str : "q8_0");
    opts.cache_v = cache_type_from_string(cache_v_str ? cache_v_str : "q8_0");

    // ---- model ----
    auto gguf_res = GGUFFile::open(model);
    if (!gguf_res.ok()) {
        fprintf(stderr, "amp-plan: %s\n", gguf_res.message().c_str());
        return 1;
    }
    const GGUFFile & gguf = *gguf_res;
    auto geo_res = ModelGeometry::build(gguf);
    if (!geo_res.ok()) {
        fprintf(stderr, "amp-plan: %s\n", geo_res.message().c_str());
        return 1;
    }
    const ModelGeometry & geo = **geo_res;

    // ---- box ----
    budget = detect_device_budget(CostModel::from_environment().constants());
    if (vram_total) {
        budget.vram_total = vram_total;
    }
    if (vram_reserved) {
        budget.vram_reserved = vram_reserved;
    }
    if (cache_budget) {
        budget.cache_budget = cache_budget;
        budget_overridden    = true;
    }

    const CostModel cost = CostModel::from_environment();
    const auto      plan_res = MemoryPlanner::plan(geo, budget, opts, cost);
    if (!plan_res.ok()) {
        fprintf(stderr, "amp-plan: %s\n", plan_res.message().c_str());
        return 1;
    }
    const ExecutionPlan & plan = *plan_res;

    if (json) {
        printf("{\n");
        printf("  \"model\": \"%s\",\n", model.c_str());
        printf("  \"architecture\": \"%s\",\n", geo.architecture().c_str());
        printf("  \"file_bytes\": %llu,\n", (unsigned long long) gguf.file_size());
        printf("  \"expert_bytes_total\": %lld,\n", (long long) geo.total_expert_bytes());
        printf("  \"other_bytes_total\": %lld,\n", (long long) geo.total_other_bytes());
        printf("  \"layer_expert_bytes_min\": %lld,\n", (long long) geo.min_layer_expert_bytes());
        printf("  \"layer_expert_bytes_max\": %lld,\n", (long long) geo.max_layer_expert_bytes());
        printf("  \"kv_bytes_per_token\": %lld,\n", (long long) geo.kv_bytes_per_token(opts.cache_k, opts.cache_v));
        printf("  \"vram_total\": %llu,\n", (unsigned long long) budget.vram_total);
        printf("  \"vram_usable\": %llu,\n", (unsigned long long) budget.vram_usable());
        printf("  \"cache_budget\": %llu,\n", (unsigned long long) budget.cache_budget);
        printf("  \"chosen\": {\n");
        printf("    \"n_expert_layers_gpu\": %d,\n", plan.n_expert_layers_gpu);
        printf("    \"ubatch\": %lld,\n", (long long) plan.ubatch);
        printf("    \"vram_bytes\": %lld,\n", (long long) plan.vram_total);
        printf("    \"compute_bytes\": %lld,\n", (long long) plan.compute_bytes);
        printf("    \"kv_bytes\": %lld,\n", (long long) plan.kv_bytes);
        printf("    \"resident_bytes\": %lld,\n", (long long) plan.resident_bytes);
        printf("    \"stream_bytes\": %lld,\n", (long long) plan.stream_bytes);
        printf("    \"predicted_prefill_tps\": %.2f,\n", plan.predicted_prefill_tps);
        printf("    \"predicted_decode_tps\": %.2f\n", plan.predicted_decode_tps);
        printf("  },\n");
        printf("  \"candidates\": [\n");
        const size_t n_json = std::min(top, plan.top_candidates.size());
        for (size_t i = 0; i < n_json; i++) {
            const auto & c = plan.top_candidates[i];
            printf("    {\"g\": %d, \"ubatch\": %lld, \"vram\": %lld, \"fits\": %s, "
                   "\"stream\": %lld, \"pp_tps\": %.1f, \"tg_tps\": %.1f, \"score\": %.3f}%s\n",
                   c.n_expert_layers_gpu, (long long) c.ubatch, (long long) c.vram_bytes,
                   c.fits ? "true" : "false", (long long) c.expert_bytes_stream,
                   c.predicted_prefill_tps, c.predicted_decode_tps, c.score,
                   i + 1 < n_json ? "," : "");
        }
        printf("  ]\n}\n");
        return 0;
    }

    // ---- human report ----
    printf("== model ==\n");
    printf("  file              %s (%s)\n", model.c_str(),
           human_bytes(gguf.file_size()).c_str());
    printf("  architecture      %s\n", geo.architecture().c_str());
    printf("  layers            %lld  (full attention on %zu, recurrent %zu)\n",
           (long long) geo.n_layer(), geo.full_attention_layers().size(),
           geo.n_layer() - (int64_t) geo.full_attention_layers().size());
    printf("  experts           %lld used of %lld, expert_ff=%lld shared_ff=%lld\n",
           (long long) geo.n_expert_used(), (long long) geo.n_expert(),
           (long long) geo.expert_ff(), (long long) geo.shared_ff());
    printf("  expert bytes      %s total, %s..%s per layer\n",
           human_bytes((uint64_t) geo.total_expert_bytes()).c_str(),
           human_bytes((uint64_t) geo.min_layer_expert_bytes()).c_str(),
           human_bytes((uint64_t) geo.max_layer_expert_bytes()).c_str());
    printf("  bytes per expert  %s (per layer)\n",
           human_bytes((uint64_t) geo.bytes_per_expert(0)).c_str());
    printf("  other weights     %s (embeddings %s, routers %s, shared experts %s)\n",
           human_bytes((uint64_t) geo.total_other_bytes()).c_str(),
           human_bytes((uint64_t) geo.embedding_bytes()).c_str(),
           human_bytes((uint64_t) geo.router_bytes()).c_str(),
           human_bytes((uint64_t) geo.shared_expert_bytes()).c_str());
    printf("  kv cache          %s/token at ctx=%lld -> %s total (k=%s v=%s)\n",
           human_bytes((uint64_t) geo.kv_bytes_per_token(opts.cache_k, opts.cache_v)).c_str(),
           (long long) opts.n_ctx,
           human_bytes((uint64_t) geo.kv_bytes(opts.n_ctx, opts.cache_k, opts.cache_v)).c_str(),
           to_string(opts.cache_k), to_string(opts.cache_v));
    printf("  ssm state         %s per sequence\n", human_bytes((uint64_t) geo.ssm_state_bytes()).c_str());

    printf("\n== box ==\n");
    printf("  ram               %s\n", human_bytes(budget.ram_total).c_str());
    printf("  page cache        plan against %s (potential), %s achievable now, %s free right now%s\n",
           human_bytes(budget.cache_budget).c_str(), human_bytes(budget.cache_realistic).c_str(),
           human_bytes(budget.cache_free_now).c_str(), budget_overridden ? " [override]" : "");
    printf("  vram              %s total, %s in use, %s usable\n",
           human_bytes(budget.vram_total).c_str(), human_bytes(budget.vram_reserved).c_str(),
           human_bytes(budget.vram_usable()).c_str());
    printf("  cpu threads       %u\n", budget.n_cpu_threads);
    printf("  cost model        %s\n", cost.describe().c_str());

    printf("\n== chosen plan ==\n");
    printf("  expert layers gpu %d of %lld  (layers %lld..%lld)\n", plan.n_expert_layers_gpu,
           (long long) geo.n_layer(), (long long) (geo.n_layer() - plan.n_expert_layers_gpu),
           (long long) (geo.n_layer() - 1));
    printf("  ubatch            %lld\n", (long long) plan.ubatch);
    printf("  vram planned      %s  (fixed %s + gpu experts %s + compute %s)\n",
           human_bytes((uint64_t) plan.vram_total).c_str(),
           human_bytes((uint64_t) (plan.vram_total - plan.expert_bytes_gpu - plan.compute_bytes)).c_str(),
           human_bytes((uint64_t) plan.expert_bytes_gpu).c_str(),
           human_bytes((uint64_t) plan.compute_bytes).c_str());
    printf("  resident (cache)  %s across %zu ranges\n",
           human_bytes((uint64_t) plan.resident_bytes).c_str(), plan.resident.size());
    printf("  streamed tail     %s across %zu ranges (%.1f MiB/token worst case)\n",
           human_bytes((uint64_t) plan.stream_bytes).c_str(), plan.stream.size(),
           plan.decode_bytes_stream / (1024.0 * 1024.0));
    printf("  predicted         prefill %.0f t/s, decode %.1f t/s (%.1f t/s with the cache free right now)\n",
           plan.predicted_prefill_tps, plan.predicted_decode_tps, plan.predicted_decode_tps_now);
    for (const auto & n : plan.notes) {
        printf("    - %s\n", n.c_str());
    }

    printf("\n== candidate ranking ==\n");
    printf("  %4s %8s %10s %6s %10s %9s %8s %8s\n", "g", "ubatch", "vram", "fits", "stream/ub",
           "pp t/s", "tg t/s", "tg now");
    const size_t n_show = std::min(top, plan.top_candidates.size());
    for (size_t i = 0; i < n_show; i++) {
        const auto & c = plan.top_candidates[i];
        printf("  %4d %8lld %10s %6s %10s %9.1f %8.1f %8.1f%s\n", c.n_expert_layers_gpu,
               (long long) c.ubatch, human_bytes((uint64_t) c.vram_bytes).c_str(),
               c.fits ? "yes" : "NO", human_bytes((uint64_t) c.expert_bytes_stream).c_str(),
               c.predicted_prefill_tps, c.predicted_decode_tps, c.predicted_decode_tps_now,
               c.reject_reason.empty() ? "" : ("  (" + c.reject_reason + ")").c_str());
    }
    return 0;
}
