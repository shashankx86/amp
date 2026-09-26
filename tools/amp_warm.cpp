// amp-warm: make the model's hot bytes actually resident, and measure whether it worked.
//
// This is the single cheapest performance lever in the whole project: the same bytes read
// as random 4 KiB faults cost 555 MB/s on this NVMe, and as sequential reads ~2.0 GB/s.
// Turning a cold working set into a resident one is worth more than any kernel change.
//
//   ./build/amp-warm --model PATH --what plan|experts|all
//   ./build/amp-warm --model PATH --what plan --verify
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "amp/bytes.h"
#include "amp/format.h"
#include "amp/io/warmer.h"
#include "amp/log.h"
#include "amp/model/geometry.h"
#include "amp/plan/cost_model.h"
#include "amp/plan/memory_plan.h"
#include "amp/timing.h"

using namespace amp;

namespace {

void usage() {
    printf(R"(amp-warm - page-cache warmer and residency verifier

usage: amp-warm --model PATH [options]

options:
  --model PATH      GGUF model (required)
  --what WHAT       plan (default: use the planner's resident set), experts, all, none
  --ctx N           context for planning (default 200000)
  --chunk B         read chunk size (default 2MiB)
  --gpu-layers N    override: expert layers on the GPU
  --drop-cache      evict the range first (POSIX_FADV_DONTNEED) so the warm is measurable
  --verify          re-read after warming and report achieved bandwidth
  --json            machine-readable output
  -v, --verbose     debug logging
)");
}

} // namespace

int main(int argc, char ** argv) {
    std::string model;
    std::string what = "plan";
    int64_t     ctx = 200000;
    uint64_t    chunk = 2ull * 1024 * 1024;
    int32_t     gpu_layers_override = -1;
    bool        drop_cache = false, verify = false, json = false;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char * w) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "amp-warm: %s needs a value\n", w); exit(2); }
            return argv[++i];
        };
        if (a == "--model" || a == "-m") model = next("--model");
        else if (a == "--what") what = next("--what");
        else if (a == "--ctx" || a == "-c") ctx = atoll(next("--ctx").c_str());
        else if (a == "--chunk") parse_bytes(next("--chunk"), &chunk);
        else if (a == "--gpu-layers") gpu_layers_override = atoi(next("--gpu-layers").c_str());
        else if (a == "--drop-cache") drop_cache = true;
        else if (a == "--verify") verify = true;
        else if (a == "--json") json = true;
        else if (a == "-v" || a == "--verbose") set_log_level(LogLevel::kDebug);
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "amp-warm: unknown argument '%s'\n", a.c_str()); usage(); return 2; }
    }

    if (model.empty()) {
        const char * env = getenv("AMP_MODEL");
        model = env ? env : "";
    }
    if (model.empty()) {
        fprintf(stderr, "amp-warm: --model is required (or set AMP_MODEL)\n");
        return 2;
    }

    auto gguf_res = GGUFFile::open(model);
    if (!gguf_res.ok()) {
        fprintf(stderr, "amp-warm: %s\n", gguf_res.message().c_str());
        return 1;
    }
    const GGUFFile & gguf = *gguf_res;
    auto geo_res = ModelGeometry::build(gguf);
    if (!geo_res.ok()) {
        fprintf(stderr, "amp-warm: %s\n", geo_res.message().c_str());
        return 1;
    }
    const ModelGeometry & geo = **geo_res;

    // ---- decide the ranges to warm ----
    std::vector<ReadRange> ranges;
    std::string            mode = what;

    if (what == "plan") {
        PlannerOptions opts;
        opts.n_ctx = ctx;
        const CostModel    cost   = CostModel::from_environment();
        const DeviceBudget budget = detect_device_budget(cost.constants());
        auto plan_res = MemoryPlanner::plan(geo, budget, opts, cost);
        if (!plan_res.ok()) {
            fprintf(stderr, "amp-warm: planning failed: %s\n", plan_res.message().c_str());
            return 1;
        }
        const ExecutionPlan & plan = *plan_res;
        if (gpu_layers_override >= 0) {
            mode = format("experts(cpu-only,%d gpu layers)", gpu_layers_override);
            for (int64_t il = 0; il < geo.n_layer() - gpu_layers_override; il++) {
                for (const auto & r : geo.expert_ranges((int) il)) {
                    ranges.push_back(r);
                }
            }
        } else {
            for (const auto & rp : plan.resident) {
                ranges.push_back(rp.range);
            }
            mode = format("plan (g=%d, ubatch=%lld)", plan.n_expert_layers_gpu,
                          (long long) plan.ubatch);
        }
    } else if (what == "experts") {
        mode = "experts (all layers)";
        for (int64_t il = 0; il < geo.n_layer(); il++) {
            for (const auto & r : geo.expert_ranges((int) il)) {
                ranges.push_back(r);
            }
        }
    } else if (what == "all") {
        mode = "whole file";
        ranges.push_back(ReadRange{ 0, gguf.file_size() });
    } else if (what == "none") {
        mode = "none";
    } else {
        fprintf(stderr, "amp-warm: --what must be plan|experts|all|none\n");
        return 2;
    }

    uint64_t total = 0;
    for (const auto & r : ranges) {
        total += r.length;
    }

    const MemInfo before = read_meminfo();
    if (!json) {
        printf("amp-warm: model   %s\n", model.c_str());
        printf("amp-warm: target  %s, %s in %zu ranges\n", mode.c_str(),
               human_bytes(total).c_str(), ranges.size());
        printf("amp-warm: cache   before %s (cached %s, available %s)\n",
               human_bytes(before.cache_headroom()).c_str(),
               human_bytes(before.cached_bytes).c_str(),
               human_bytes(before.mem_available_bytes).c_str());
    }

    if (drop_cache && !ranges.empty()) {
        if (!json) {
            printf("amp-warm: dropping %s from the page cache first ...\n", human_bytes(total).c_str());
        }
        for (const auto & r : ranges) {
            (void) gguf.file().drop_cache(r.offset, r.length);
        }
    }

    // ---- warm ----
    double seconds = 0.0;
    double bps     = 0.0;
    if (total > 0) {
        const Stopwatch sw;
        uint64_t        last_report = 0;
        WarmProgress    prog;
        prog.on_progress = [&](const WarmProgress & p) {
            if (json) {
                return;
            }
            if (p.bytes_done - last_report >= (1ull << 30)) {
                last_report = p.bytes_done;
                printf("  %s / %s  %s  (%.1f s)\n", human_bytes(p.bytes_done).c_str(),
                       human_bytes(p.bytes_total).c_str(),
                       human_rate(p.bytes_per_sec).c_str(), p.seconds);
                fflush(stdout);
            }
        };
        const Status st = warm_ranges_blocking(gguf.file().fd(), ranges, chunk, &prog);
        if (!st.ok()) {
            fprintf(stderr, "amp-warm: %s\n", st.message().c_str());
            return 1;
        }
        seconds = sw.elapsed_s();
        bps     = seconds > 0 ? (double) total / seconds : 0.0;
    }

    const MemInfo after = read_meminfo();

    double verify_bps = 0.0;
    if (verify && total > 0) {
        const Stopwatch sw;
        std::vector<uint8_t> buf((size_t) chunk);
        uint64_t             done = 0;
        for (const auto & r : ranges) {
            for (uint64_t off = 0; off < r.length; off += chunk) {
                const size_t len = (size_t) std::min<uint64_t>(chunk, r.length - off);
                if (!gguf.file().pread_exact(r.offset + off, buf.data(), len).ok()) {
                    break;
                }
                done += len;
            }
        }
        verify_bps = sw.elapsed_s() > 0 ? (double) done / sw.elapsed_s() : 0.0;
    }

    if (json) {
        printf("{\n");
        printf("  \"mode\": \"%s\",\n", mode.c_str());
        printf("  \"ranges\": %zu,\n", ranges.size());
        printf("  \"bytes\": %llu,\n", (unsigned long long) total);
        printf("  \"seconds\": %.3f,\n", seconds);
        printf("  \"bytes_per_second\": %.0f,\n", bps);
        printf("  \"cached_before\": %llu,\n", (unsigned long long) before.cached_bytes);
        printf("  \"cached_after\": %llu,\n", (unsigned long long) after.cached_bytes);
        printf("  \"verify_bytes_per_second\": %.0f\n", verify_bps);
        printf("}\n");
        return 0;
    }

    printf("amp-warm: warmed   %s in %.2f s = %s\n", human_bytes(total).c_str(), seconds,
           human_rate(bps).c_str());
    printf("amp-warm: cache   after  %s (cached %s, available %s)\n",
           human_bytes(after.cache_headroom()).c_str(), human_bytes(after.cached_bytes).c_str(),
           human_bytes(after.mem_available_bytes).c_str());
    if (verify) {
        printf("amp-warm: verify  re-read at %s (this is the rate the model will see)\n",
               human_rate(verify_bps).c_str());
    }
    const double needed = (double) geo.total_expert_bytes();
    if (after.cache_headroom() > 0 && (double) after.cache_headroom() < needed) {
        printf("amp-warn: page cache holds %s but the CPU expert set needs %s -- the tail will "
               "come from NVMe (see plan --no-streaming to see the cost)\n",
               human_bytes(after.cache_headroom()).c_str(), human_bytes((uint64_t) needed).c_str());
    }
    return 0;
}
