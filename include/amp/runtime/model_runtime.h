// ModelRuntime: amp's forward path.
//
// Deliberately built on llama.cpp's public C API rather than a hand-rolled graph. That is a
// strategic choice, not laziness: the arithmetic is then *literally* the same code llama.cpp runs,
// which is the only way "zero quality loss" is guaranteed rather than argued. What amp replaces
// is everything around it - how bytes reach that code, in what order, and when.
//
//   amp_runtime   owns llama.h: model, context, tokenizer, sampler
//   amp_io        owns the page cache and prefetch streams
//   amp_plan      supplies the device split and ubatch size
#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "amp/io/stream_reader.h"
#include "amp/model/geometry.h"
#include "amp/plan/memory_plan.h"
#include "amp/status.h"
#include "amp/runtime/buft_overrides.h"
#include "ggml-backend.h"
#include "llama.h"

namespace amp {

struct RuntimeConfig {
    std::string model_path;
    int64_t     n_ctx            = 200000;
    CacheType   cache_k          = CacheType::kQ8_0;
    CacheType   cache_v          = CacheType::kQ4_0;
    int32_t     n_expert_layers_gpu = 2;
    int64_t     n_ubatch         = 2048;
    int32_t     n_batch          = 2048;
    int32_t     n_threads        = 8;
    int32_t     n_threads_batch  = 8;
    bool        flash_attn       = true;
    bool        prefetch         = true;
    bool        reverse_warm     = true;   // see Prefetcher for why this is not a typo
    float       temperature      = 0.6f;
    float       top_p            = 0.95f;
    int32_t     top_k            = 20;
    uint32_t    seed             = LLAMA_DEFAULT_SEED;
    bool        verbose          = false;
};

struct RuntimeStats {
    int64_t prefill_tokens     = 0;
    double  prefill_ms         = 0.0;
    int64_t decode_tokens      = 0;
    double  decode_ms          = 0.0;
    uint64_t warm_bytes_issued = 0;
    uint64_t warm_calls        = 0;
    int64_t  n_decode_calls    = 0;

    double prefill_tps() const { return prefill_ms > 0 ? prefill_tokens * 1e3 / prefill_ms : 0.0; }
    double decode_tps() const { return decode_ms > 0 ? decode_tokens * 1e3 / decode_ms : 0.0; }
};

// Keeps the *next* ubatch's expert bytes in the page cache while the current one computes.
//
// Two details make this work, and both come from measurements rather than intuition:
//
//  1. The working set (12.19 GiB) is larger than the page cache (~9.6 GiB usable), so the access
//     pattern is a cyclic scan and LRU gives ~0% reuse. Warming in *reverse* layer order fixes
//     that: the compute thread walks layers 0..39, so if the tail of the warm is layers 0..N, the
//     layers it needs first are the ones that stayed resident. Warming forwards leaves the warm
//     layers cached, which are exactly the ones the compute reaches last, and it misses them again.
//  2. Warming must overlap compute. The compute thread's own demand faults are synchronous, so a
//     cold fault both waits for the NVMe and drains queue depth. If the bytes are already resident
//     the fault is served from RAM at 3.5 GiB/s instead of 1.8 GiB/s from the drive.
class Prefetcher {
public:
    Prefetcher(const ModelGeometry & geo, const ExecutionPlan & plan, const MappedFile & file,
               const RuntimeConfig & cfg, std::unique_ptr<IReadScheduler> sched);

    // Queue the expert bytes for the upcoming *prefill* ubatch. Non-blocking.
    void prime_for_ubatch();

    // Decode touches only the 8 active experts per layer (~51 MiB), not the whole 12 GiB set, so
    // prefetching during decode would evict the resident set for no benefit. Measured: doing so
    // dropped decode from 21.6 t/s to 0.4-1.0 t/s.
    void prime_for_decode() { (void) cursor_; }

    // How many ranges were skipped because mincore() said they were already resident.
    uint64_t ranges_skipped_resident() const { return skipped_resident_; }
    uint64_t resident_checks() const { return resident_checks_; }
    // Called after each decode call; keeps the queue from running away.
    void settle(uint32_t timeout_ms = 0);
    // Bring the first-issued (i.e. most-at-risk) layers back into the cache.
    void rebalance();

    uint64_t total_bytes() const;
    uint64_t bytes_issued() const { return bytes_issued_; }
    uint64_t calls() const { return calls_; }
    // By value, not by reference: IReadScheduler::warm_stats() returns a WarmStats by value
    // (stream_reader.h:61), so returning a const& to it would dangle.
    WarmStats        warm_stats() const { return sched_->warm_stats(); }

private:
    void submit(const ReadRange & r);

    const ModelGeometry &          geo_;
    const ExecutionPlan &          plan_;
    const RuntimeConfig &          cfg_;
    std::unique_ptr<IReadScheduler> sched_;
    std::vector<ReadRange>         cpu_expert_ranges_;  // in warm order
    size_t                         cursor_ = 0;
    uint64_t                       bytes_issued_ = 0;
    uint64_t                       calls_ = 0;
    uint64_t                       skipped_resident_ = 0;
    uint64_t                       resident_checks_ = 0;
    const MappedFile *             file_ = nullptr;
};

class ModelRuntime {
public:
    ~ModelRuntime();

    static Result<std::unique_ptr<ModelRuntime>> create(const RuntimeConfig & cfg,
                                                        const ExecutionPlan & plan,
                                                        const ModelGeometry & geo,
                                                        const MappedFile & file);

    // Raw prompt text -> tokens. add_special=true for a chat-formatted prompt.
    Result<std::vector<llama_token>> tokenize(const std::string & text, bool add_special = true) const;

    // Processes the whole prompt in ubatches, priming the page cache ahead of each one.
    Status prefill(const std::vector<llama_token> & tokens);

    // Generates up to max_new tokens. Returns the generated tokens.
    Result<std::vector<llama_token>> generate(int32_t max_new, std::string * text_out = nullptr);

    // Full detokenized text of the last generate() call (for quality diffing).
    const std::string & last_text() const { return last_text_; }

    // Per-position top-k (token, logprob) for the last generate() call. This is how amp
    // demonstrates quality parity: identical logits, not merely similar text.
    struct TopK {
        int32_t token;
        float   logprob;
    };
    const std::vector<std::vector<TopK>> & logprobs() const { return logprobs_; }
    int32_t top_k_track() const { return top_k_track_; }

    int64_t n_past() const { return n_past_; }
    const RuntimeStats & stats() const { return stats_; }
    const Prefetcher &  prefetcher() const { return *prefetch_; }
    int32_t             n_vocab() const;

private:
    ModelRuntime() = default;

    mutable RuntimeConfig                cfg_;
    const ExecutionPlan *                plan_ = nullptr;
    const ModelGeometry *                geo_  = nullptr;
    // per-tensor buffer-type overrides, so expert tensors can live on the GPU independently of the
    // layer count (the -ncmoe equivalent)
    ExpertCpuOverrides overrides_;
    llama_model *                        model_ = nullptr;
    llama_context *                      ctx_   = nullptr;
    const llama_vocab *                  vocab_ = nullptr;
    llama_sampler *                      smpl_  = nullptr;
    std::unique_ptr<Prefetcher>          prefetch_;
    int64_t                              n_past_ = 0;
    RuntimeStats                         stats_;
    std::vector<llama_token>             decode_batch_;  // reusable batch storage
    std::string                          last_text_;
    std::vector<std::vector<TopK>>       logprobs_;
    int32_t                              top_k_track_ = 5;
};

} // namespace amp
