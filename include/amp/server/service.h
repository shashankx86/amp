// InferenceService: the engine behind the HTTP API.
//
// Owns the model, the sequences, the prefix cache and the chat templates, and runs one request at a
// time. That is not a simplification: this box has 6 GB of VRAM and a 14 GiB working set, so a
// second concurrent generation would double the KV and evict the expert weights we depend on.
// Streaming is a callback, not a second code path, so the streamed and non-streamed responses cannot
// drift apart.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "amp/model/geometry.h"
#include "amp/plan/memory_plan.h"
#include "amp/io/warmer.h"
#include "amp/runtime/prefix_cache.h"
#include "amp/server/task_queue.h"
#include "amp/status.h"
#include "llama.h"

namespace amp {

struct ServerConfig {
    std::string model_path;
    int64_t     n_ctx               = 200000;
    CacheType   cache_k             = CacheType::kQ8_0;
    CacheType   cache_v             = CacheType::kQ4_0;
    int32_t     n_seq               = 1;
    int32_t     n_gpu_expert_layers = -1;   // -1 = planner's choice
    int64_t     n_ubatch            = -1;   // -1 = planner's choice
    int32_t     n_threads           = 8;
    bool        prefetch            = true;
    std::string host                = "127.0.0.1";
    bool        warm                = true;   // warm the plan's resident ranges at start-up
    uint64_t    warm_chunk          = 16ull << 20;
    uint16_t    port                = 8081;
    std::string api_key;                    // empty = no auth
    std::string model_id           = "amp";
    std::string log_prompts_dir;            // empty = off
    bool        verbose             = false;
    float       temperature = 0.6f;
    float       top_p       = 0.95f;
    int32_t     top_k       = 20;
    uint32_t    seed        = 0xFFFFFFFFu;
    int32_t     n_predict   = 256;         // default when the request omits max_tokens
};

// One message of a chat request, in amp's own types so the API layer never includes llama-common.
struct ChatMessage {
    std::string role;
    std::string content;
    // Reasoning round-tripped by the client. We read all three aliases on the way in: OpenCode
    // sends "reasoning_text", other clients send "reasoning_content" or "reasoning". Accepting only
    // the canonical name is what makes agentic clients re-evaluate the whole tail of the
    // conversation every turn - see ../NOTES.md 7.2.
    std::string reasoning_content;
    bool        has_reasoning = false;
    std::string tool_call_id;
    std::string tool_name;
};

struct GenerateParams {
    int32_t                 max_tokens = 0;
    float                   temperature = -1.0f;
    float                   top_p       = -1.0f;
    int32_t                 top_k       = -1;
    float                   min_p       = -1.0f;
    std::vector<std::string> stop;
    bool                    ignore_eos = false;
    uint32_t                seed        = 0xFFFFFFFFu;
    // The model's own thinking tags, so the reasoning/content split follows the template instead of
    // assuming <think>. Empty falls back to <think>.
    std::string             think_start = "<think>";
    std::string             think_end   = "</think>";
    // Split the answer into reasoning/content at all? /v1/completions wants the raw text, because
    // that endpoint never applied a chat template in the first place.
    bool                    split_thinking = true;
};

struct StreamChunk {
    std::string text;        // content delta
    std::string reasoning;   // reasoning delta
    llama_token token = 0;
    bool        first = false;
    bool        done  = false;
};

enum class FinishReason { kEos, kStop, kLength, kInterrupted };

struct GenerationResult {
    std::string              content;    // after the thinking split
    std::string              reasoning;  // the <think>...</think> part
    std::string              raw;        // everything generated, tags included
    std::vector<llama_token> generated;
    int32_t                  n_prompt_tokens    = 0;
    int32_t                  n_prompt_cached    = 0;
    int32_t                  n_prompt_computed  = 0;
    double                   prefill_ms = 0.0;
    double                   decode_ms  = 0.0;
    FinishReason             finish = FinishReason::kLength;
    // False when the memory could not rewind and the prompt had to be evaluated from scratch. The
    // answer is still correct; the client is told so it can explain the pause.
    bool                     cache_rewound_exactly = true;

    double prefill_tps() const {
        return prefill_ms > 0 ? (double) n_prompt_computed * 1e3 / prefill_ms : 0.0;
    }
    double decode_tps() const {
        return decode_ms > 0 ? (double) generated.size() * 1e3 / decode_ms : 0.0;
    }
    const char * finish_str() const {
        switch (finish) {
            case FinishReason::kEos:         return "stop";
            case FinishReason::kStop:        return "stop";
            case FinishReason::kInterrupted: return "stop";
            case FinishReason::kLength:      return "length";
        }
        return "length";
    }
};

// Rendered chat prompt plus the tags needed to split the answer back into reasoning and content.
struct RenderedChat {
    std::string prompt;
    bool        supported_thinking = false;
    std::string thinking_start      = "<think>";
    std::string thinking_end        = "</think>";
};

struct ChatTemplates;   // holds llama-common's common_chat_templates, defined in the .cpp

class InferenceService {
public:
    ~InferenceService();

    static Result<std::unique_ptr<InferenceService>> create(const ServerConfig & cfg);

    // Renders a chat with the model's own jinja template, with preserve_reasoning enabled.
    Result<RenderedChat> render_chat(const std::vector<ChatMessage> & messages,
                                     const std::string & tools_json = "");

    // Prompt tokens -> completion, with an optional streaming callback.
    //
    // Serialized: the work runs on the queue's single worker, because a llama_context is not
    // reentrant and an agentic client will happily have two requests in flight at once. See
    // server/task_queue.h for why that is not hypothetical.
    Result<GenerationResult> generate(const std::vector<llama_token> & prompt_tokens,
                                      const GenerateParams & params,
                                      const std::function<void(const StreamChunk &)> & on_chunk);

    // Blocks until every request submitted so far has finished. Used by tests.
    void drain() { queue_.drain(); }
    std::string queue_status() const { return queue_.describe(); }
    TaskQueue::Stats queue_stats() const { return queue_.stats(); }

    // Tokenize plain text (adds BOS if the model's tokenizer does).
    Result<std::vector<llama_token>> tokenize(const std::string & text, bool add_special = true) const;

    void interrupt();

    const ServerConfig &   config() const { return cfg_; }
    const ExecutionPlan &  plan() const { return plan_; }
    const ModelGeometry &  geometry() const { return geo_; }
    const PrefixCacheStats & cache_stats() const;
    std::string status_line() const;
    uint64_t    total_generated() const { return total_generated_; }
    double      last_prefill_tps() const { return last_prefill_tps_; }
    double      last_decode_tps() const { return last_decode_tps_; }
    const WarmStats & warm_stats() const { return warm_stats_; }
    double   checkpoint_save_ms() const { return cache_->checkpoint_save_ms(); }
    double   checkpoint_load_ms() const { return cache_->checkpoint_load_ms(); }
    uint64_t checkpoint_bytes() const { return cache_->checkpoint_bytes(); }

private:
    // The un-serialized body of generate(). Only the queue's worker may call it.
    Result<GenerationResult> generate_locked(const std::vector<llama_token> & prompt_tokens,
                                             const GenerateParams & params,
                                             const std::function<void(const StreamChunk &)> & on_chunk);

    InferenceService();
    InferenceService(const InferenceService &) = delete;
    InferenceService & operator=(const InferenceService &) = delete;

    ServerConfig                   cfg_;
    ExecutionPlan                  plan_;
    ModelGeometry                  geo_;
    llama_model *                  model_ = nullptr;
    llama_context *                ctx_   = nullptr;
    const llama_vocab *            vocab_ = nullptr;
    std::unique_ptr<PrefixCache>   cache_;
    TaskQueue                      queue_;
    std::shared_ptr<ChatTemplates> tmpls_;   // complete type lives in the .cpp
    std::atomic<bool>              interrupt_{false};
    uint64_t                       total_generated_ = 0;
    double                         last_prefill_tps_ = 0.0;
    double                         last_decode_tps_  = 0.0;
    WarmStats                      warm_stats_;
};

} // namespace amp
