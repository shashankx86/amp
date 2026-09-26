// Tests for the cost model and the memory planner. The point of these is to lock in the
// non-obvious result from ../NOTES.md: on this box, VRAM spent on a bigger ubatch beats
// VRAM spent on GPU-resident experts, so the planner must prefer large ubatches.
#include "test_harness.h"

#include "amp/bytes.h"
#include "amp/format.h"
#include "amp/plan/cost_model.h"
#include "amp/plan/memory_plan.h"

using namespace amp;

namespace {

// The measured box, spelled out so the tests do not depend on detection.
DeviceBudget this_box() {
    DeviceBudget b;
    b.vram_total    = 6141ull * kMiB;   // RTX 4050 Laptop
    b.vram_reserved = 104ull * kMiB;    // kwin/wayland
    b.ram_total     = 14ull * kGiB;
    b.cache_budget  = (uint64_t) (11.5 * (double) kGiB);
    b.n_cpu_threads = 8;
    return b;
}

} // namespace

AMP_TEST(cost_model_ubatch_efficiency_is_monotonic) {
    const CostModel c = CostModel::for_this_machine();
    double prev = -1;
    for (int64_t ub : { 64, 128, 256, 512, 1024, 2048, 4096 }) {
        const double e = c.ubatch_efficiency(ub);
        AMP_CHECK_MSG(e >= prev, format("efficiency dropped at ubatch %lld", (long long) ub));
        AMP_CHECK(e > 0.0 && e <= 1.0);
        prev = e;
    }
    AMP_CHECK_NEAR(c.ubatch_efficiency(1024), 1.0, 1e-9);
    AMP_CHECK(c.ubatch_efficiency(2048) >= 0.999);
}

AMP_TEST(cost_model_io_bounds_prefill) {
    const CostModel c = CostModel::for_this_machine();
    // no streaming -> the measured cache-resident ceiling
    const double tps_clean = c.prefill_tps(2048, 2048, 0);
    AMP_CHECK_NEAR(tps_clean, 240.0, 1.0);

    // streaming 10 GiB per ubatch costs throughput, and the cost depends on HOW we read it
    const double tps_fault = c.prefill_tps(2048, 2048, 10ull * kGiB, IoMode::kPageFault);
    const double tps_pre   = c.prefill_tps(2048, 2048, 10ull * kGiB, IoMode::kPrefetch);
    AMP_CHECK_MSG(tps_fault < tps_clean, format("page faults %.0f vs clean %.0f", tps_fault, tps_clean));
    // the same bytes via 440 KiB prefetches at 2 GB/s, overlapped with compute, are much cheaper
    AMP_CHECK_MSG(tps_pre > tps_fault * 1.15,
                  format("prefetch %.0f should beat page faults %.0f", tps_pre, tps_fault));
    AMP_CHECK_MSG(tps_pre < tps_clean, "prefetching cannot exceed the compute ceiling");

    // ...and a bigger ubatch must recover more (this is the whole -ub 512 -> 2048 story)
    const double tps_512 = c.prefill_tps(2048, 512, 10ull * kGiB, IoMode::kPageFault);
    const double tps_2048 = c.prefill_tps(2048, 2048, 10ull * kGiB, IoMode::kPageFault);
    AMP_CHECK_MSG(tps_512 < tps_2048, format("ub512 %.1f should be worse than ub2048 %.1f",
                                              tps_512, tps_2048));
}

AMP_TEST(cost_model_decode) {
    const CostModel c = CostModel::for_this_machine();
    const double t_resident = c.decode_tps(38, 0);
    const double t_streaming = c.decode_tps(38, 8ull * kMiB);
    AMP_CHECK_MSG(t_resident > t_streaming, "streaming should slow decode");
    // matches the measured 11-14 t/s ballpark for a fully resident CPU path
    AMP_CHECK_MSG(t_resident > 10.0 && t_resident < 30.0, format("%.1f t/s", t_resident));
    // fewer CPU layers -> faster decode
    AMP_CHECK(c.decode_tps(20, 0) > t_resident);
}

AMP_TEST(planner_prefers_large_ubatch_over_gpu_experts) {
    const char * model = getenv("AMP_TEST_MODEL");
    model = model ? model
                  : "/home/e0u/localhost/models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf";
    auto gguf_res = GGUFFile::open(model);
    if (!gguf_res.ok()) {
        printf("(skipped: model not found) ");
        return;
    }
    auto geo_res = ModelGeometry::build(*gguf_res);
    const ModelGeometry & geo = **geo_res;

    PlannerOptions opts;  // defaults: ctx 200k, q8_0/q4_0, ubatch <= 2048
    const CostModel cost = CostModel::for_this_machine();
    const DeviceBudget budget = this_box();

    auto plan_res = MemoryPlanner::plan(geo, budget, opts, cost);
    AMP_CHECK(plan_res.ok());
    const ExecutionPlan & plan = *plan_res;

    // the chosen plan must actually fit
    AMP_CHECK_MSG((uint64_t) plan.vram_total <= budget.vram_usable(),
                  format("vram %s > usable %s", human_bytes((uint64_t) plan.vram_total).c_str(),
                         human_bytes(budget.vram_usable()).c_str()));

    // and it should land on the large-ubatch end, which is the measured optimum
    AMP_CHECK_MSG(plan.ubatch >= 1024, format("ubatch %lld is smaller than expected",
                                              (long long) plan.ubatch));

    // with the working set slightly over the cache, the plan must stream the tail
    AMP_CHECK_MSG(plan.stream_bytes > 0 || plan.resident_bytes >= plan.expert_bytes_cpu,
                  "planner neither resident nor streaming");
}

AMP_TEST(planner_handles_impossible_budgets) {
    const char * model = getenv("AMP_TEST_MODEL");
    model = model ? model
                  : "/home/e0u/localhost/models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf";
    auto gguf_res = GGUFFile::open(model);
    if (!gguf_res.ok()) {
        return;
    }
    auto geo_res = ModelGeometry::build(*gguf_res);
    const ModelGeometry & geo = **geo_res;

    DeviceBudget tiny = this_box();
    tiny.vram_total    = 512ull * kMiB;  // smaller than weights + KV
    PlannerOptions opts;
    const CostModel cost = CostModel::for_this_machine();
    auto res = MemoryPlanner::plan(geo, tiny, opts, cost);
    AMP_CHECK_MSG(!res.ok(), "planner should refuse an impossible VRAM budget");
}

AMP_TEST(planner_candidate_ranking_is_sorted) {
    const char * model = getenv("AMP_TEST_MODEL");
    model = model ? model
                  : "/home/e0u/localhost/models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf";
    auto gguf_res = GGUFFile::open(model);
    if (!gguf_res.ok()) {
        return;
    }
    auto geo_res = ModelGeometry::build(*gguf_res);
    const ModelGeometry & geo = **geo_res;

    PlannerOptions opts;
    const auto cands = MemoryPlanner::rank(geo, this_box(), opts, CostModel::for_this_machine(), 32);
    AMP_CHECK(!cands.empty());
    for (size_t i = 1; i < cands.size(); i++) {
        if (cands[i - 1].fits && cands[i].fits) {
            AMP_CHECK_MSG(cands[i - 1].score >= cands[i].score, "candidates not sorted by score");
        }
    }
    // the first candidate must fit
    if (!cands.empty()) {
        AMP_CHECK_MSG(cands[0].fits, "best candidate does not fit");
    }
}
