#include "amp/server/service.h"

#include "amp/runtime/buft_overrides.h"

#include "amp/bytes.h"
#include "amp/format.h"
#include "amp/log.h"
#include "amp/timing.h"
#include "amp/util/json.h"

#include "chat.h"              // llama-common: jinja rendering + thinking tags
#include "common.h"            // llama-common: common_tokenize
#include "reasoning-budget.h"  // llama-common: the thinking-budget sampler

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <condition_variable>
#include <mutex>

namespace amp {

namespace {

ggml_type to_ggml(CacheType t) {
    switch (t) {
        case CacheType::kF32:    return GGML_TYPE_F32;
        case CacheType::kF16:    return GGML_TYPE_F16;
        case CacheType::kBF16:   return GGML_TYPE_BF16;
        case CacheType::kQ8_0:   return GGML_TYPE_Q8_0;
        case CacheType::kQ5_1:   return GGML_TYPE_Q5_1;
        case CacheType::kQ5_0:   return GGML_TYPE_Q5_0;
        case CacheType::kQ4_1:   return GGML_TYPE_Q4_1;
        case CacheType::kQ4_0:   return GGML_TYPE_Q4_0;
        case CacheType::kQ4_K:   return GGML_TYPE_Q4_K;
        case CacheType::kIQ4_NL: return GGML_TYPE_IQ4_NL;
        case CacheType::kQ3_K:   return GGML_TYPE_Q3_K;
        case CacheType::kQ2_K:   return GGML_TYPE_Q2_K;
        case CacheType::kQ6_K:   return GGML_TYPE_Q6_K;
    }
    return GGML_TYPE_F16;
}

uint64_t cuda_free_bytes() {
    for (int i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char *      name = ggml_backend_dev_name(dev);
        if (!name || !strstr(name, "CUDA")) {
            continue;
        }
        ggml_backend_dev_props props;
        memset(&props, 0, sizeof(props));
        ggml_backend_dev_get_props(dev, &props);
        if (props.memory_free > 0) {
            return (uint64_t) props.memory_free;
        }
    }
    return 0;
}

// The reasoning block is tracked by the sampler's state machine rather than by searching the output
// for a tag. llama.cpp does it this way and it is the only robust option: the model emits the closing
// tag with or without surrounding whitespace depending on how it was sampled (a budget-forced close
// produced "...Analyze</think>" with no newline at all), so a string search silently fails and the
// whole answer lands in reasoning_content. The state machine cannot be fooled that way.
llama_sampler * make_sampler(const GenerateParams & p, const ServerConfig & cfg,
                             const llama_vocab * vocab, const std::vector<llama_token> & prefill,
                             llama_sampler ** rbudget_out) {
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf                  = true;
    llama_sampler * chain       = llama_sampler_chain_init(sp);

    const float    temp = p.temperature >= 0.0f ? p.temperature : cfg.temperature;
    const int32_t  tk   = p.top_k >= 0 ? p.top_k : cfg.top_k;
    const float    tp   = p.top_p >= 0.0f ? p.top_p : cfg.top_p;
    const float    mp   = p.min_p;
    const uint32_t seed = p.seed != 0xFFFFFFFFu ? p.seed : cfg.seed;

    // Reasoning budget, llama.cpp's mechanism. A thinking model that spends the whole token budget
    // inside <think>...</think> returns *no content at all*, which for a short request (a
    // conversation title, a one-line summary) means the caller gets nothing usable. The sampler
    // counts the tokens spent in the reasoning block and, when the budget runs out, forces the end
    // tag - so generation continues into actual content.
    //
    // The trick llama.cpp uses, and the reason this works at all: the sampler starts in IDLE and is
    // fed the prefill tokens, so a template whose generation prompt already ends inside <think> is
    // detected by replaying the prompt through it. No special-casing of the template.
    int32_t budget = p.reasoning_budget;
    if (budget == -2) {
        budget = cfg.reasoning_budget;
    }
    if (budget == -2) {
        // Not specified anywhere. Derived: a short request that produces only reasoning is useless to
        // its caller, so give thinking half the budget. Long requests are left alone - a real answer
        // should think as long as it needs.
        budget = (p.max_tokens > 0 && p.max_tokens <= 256) ? std::max(4, p.max_tokens / 2) : -1;
    }
    llama_sampler * rbudget = nullptr;
    // Always build the tracker (INT_MAX budget when unlimited) so the reasoning/content split is
    // state-driven even for long requests. Only *forcing* is conditional on the budget.
    const int32_t budget_for_sampler = (budget < 0) ? INT32_MAX : budget;
    if (!p.think_start.empty() && !p.think_end.empty() && vocab) {
        const std::vector<llama_token> start = common_tokenize(vocab, p.think_start, false, true);
        std::vector<llama_token>       forced = common_tokenize(vocab, p.think_end, false, true);
        if (!start.empty() && !forced.empty()) {
            std::vector<std::vector<llama_token>> ends;
            ends.push_back(forced);
            for (const auto & t : p.think_end_alternatives) {
                std::vector<llama_token> alt = common_tokenize(vocab, t, false, true);
                if (!alt.empty()) {
                    ends.push_back(std::move(alt));
                }
            }
            if (!p.reasoning_budget_message.empty()) {
                const std::vector<llama_token> msg = common_tokenize(vocab, p.reasoning_budget_message, false, true);
                forced.insert(forced.begin(), msg.begin(), msg.end());
            }
            rbudget = common_reasoning_budget_init(vocab, { start }, ends, forced, budget_for_sampler);
            for (const llama_token t : prefill) {
                llama_sampler_accept(rbudget, t);
            }
        }
    }

    if (rbudget_out) {
        *rbudget_out = rbudget;
    }
    if (temp <= 0.0f) {
        // Greedy still needs the budget sampler in front of it: forcing a token is a logits
        // operation, and it is what ends the reasoning block.
        if (rbudget) {
            llama_sampler_chain_add(chain, rbudget);
        }
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        return chain;
    }
    if (tk > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(tk));
    }
    if (mp >= 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(mp, 1));
    }
    if (tp > 0.0f && tp < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(tp, 1));
    }
    if (rbudget) {
        llama_sampler_chain_add(chain, rbudget);
    }
    llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    return chain;
}

// Splits generated text into reasoning and content using the template's thinking tags. Streaming
// needs this to be incremental, so the state machine is kept per generation.
// Splits generated text into reasoning and content.
//
// Two things this has to get right, both learned the hard way on this model:
//
//  - generation *starts inside* a reasoning block. The template's generation prompt ends with the
//    opening think tag (`<|im_start|>assistant\n<think>\n`), so the model never emits one and a
//    split triggered by the opening tag never fires. That is how a conversation title came back as
//    "The user said hi. This is a short, conversational greeting. According to the rules, I should..."
//    - a chain of thought delivered as the answer.
//  - the closing tag arrives with *variable* surrounding whitespace: naturally as "\n</think>",
//    but when the reasoning budget forces it, as "</think>" with no newline at all. Matching only the
//    exact tag string silently fails, and then the answer lands in reasoning_content too.
class ThinkingSplitter {
public:
    ThinkingSplitter(std::string start, std::string end, bool starts_inside = false)
        : start_(std::move(start)), end_(std::move(end)), in_reasoning_(starts_inside) {
        start_variants_ = tag_variants(start_);
        end_variants_   = tag_variants(end_);
    }

    void feed(const std::string & piece, std::string & reasoning_out, std::string & content_out) {
        buf_ += piece;
        while (!buf_.empty()) {
            if (in_reasoning_) {
                const Match m = find_tag(buf_, end_variants_);
                if (m.at == std::string::npos) {
                    const size_t keep = partial_tag_len(buf_, end_variants_);
                    if (buf_.size() > keep) {
                        reasoning_out.append(buf_, 0, buf_.size() - keep);
                        buf_.erase(0, buf_.size() - keep);
                    }
                    return;
                }
                // The tag itself is part of the reasoning block, so it does not leak into content.
                reasoning_out.append(buf_, 0, m.at);
                buf_.erase(0, m.at + m.len);
                in_reasoning_ = false;
                continue;
            }
            const Match m = find_tag(buf_, start_variants_);
            if (m.at == std::string::npos) {
                const size_t keep = partial_tag_len(buf_, start_variants_);
                if (buf_.size() > keep) {
                    content_out.append(buf_, 0, buf_.size() - keep);
                    buf_.erase(0, buf_.size() - keep);
                }
                return;
            }
            content_out.append(buf_, 0, m.at);
            buf_.erase(0, m.at + m.len);
            in_reasoning_ = true;
        }
    }

    void flush(std::string & reasoning_out, std::string & content_out) {
        if (buf_.empty()) {
            return;
        }
        (in_reasoning_ ? reasoning_out : content_out).append(buf_);
        buf_.clear();
    }

    bool in_reasoning() const { return in_reasoning_; }

private:
    struct Match {
        size_t at  = std::string::npos;
        size_t len = 0;
    };

    // The tag as written, plus the same tag with surrounding whitespace trimmed, so "\n</think>",
    // "</think>" and "\n</think>\n" all match.
    static std::vector<std::string> tag_variants(const std::string & tag) {
        std::vector<std::string> out;
        if (tag.empty()) {
            return out;
        }
        out.push_back(tag);
        size_t b = tag.find_first_not_of(" \t\r\n");
        size_t e = tag.find_last_not_of(" \t\r\n");
        if (b == std::string::npos) {
            return out;
        }
        const std::string trimmed = tag.substr(b, e - b + 1);
        if (trimmed != tag) {
            out.push_back(trimmed);
        }
        return out;
    }

    static Match find_tag(const std::string & buf, const std::vector<std::string> & tags) {
        Match best;
        for (const auto & t : tags) {
            const size_t at = buf.find(t);
            if (at != std::string::npos && (best.at == std::string::npos || at < best.at)) {
                best = Match{ at, t.size() };
            }
        }
        return best;
    }

    // How much of the tail could still turn into a tag: never emit a partial one.
    static size_t partial_tag_len(const std::string & buf, const std::vector<std::string> & tags) {
        size_t keep = 0;
        for (const auto & t : tags) {
            const size_t max_keep = std::min(buf.size(), t.size() - 1);
            for (size_t k = max_keep; k > keep; k--) {
                if (buf.compare(buf.size() - k, k, t, 0, k) == 0) {
                    keep = k;
                    break;
                }
            }
        }
        return keep;
    }

    std::string              start_, end_, buf_;
    std::vector<std::string> start_variants_, end_variants_;
    bool                     in_reasoning_ = false;
};

} // namespace

// llama-common's template handle, kept out of the public header.
struct ChatTemplates {
    common_chat_templates_ptr tmpls;
};

InferenceService::InferenceService() = default;

InferenceService::~InferenceService() {
    if (ctx_) {
        llama_free(ctx_);
    }
    if (model_) {
        llama_model_free(model_);
    }
    llama_backend_free();
}

Result<std::unique_ptr<InferenceService>> InferenceService::create(const ServerConfig & cfg) {
    llama_backend_init();

    auto gguf_res = GGUFFile::open(cfg.model_path);
    if (!gguf_res.ok()) {
        return gguf_res.status().context("opening model");
    }
    auto geo_res = ModelGeometry::build(*gguf_res);
    if (!geo_res.ok()) {
        return geo_res.status().context("model geometry");
    }

    PlannerOptions opts;
    opts.n_ctx   = cfg.n_ctx;
    opts.cache_k = cfg.cache_k;
    opts.cache_v = cfg.cache_v;
    const CostModel     cost   = CostModel::from_environment();
    const DeviceBudget  budget = detect_device_budget(cost.constants());
    auto plan_res = MemoryPlanner::plan(**geo_res, budget, opts, cost);
    if (!plan_res.ok()) {
        return plan_res.status().context("planning");
    }
    ExecutionPlan plan = *plan_res;
    if (cfg.n_gpu_expert_layers >= 0) {
        plan.n_expert_layers_gpu = cfg.n_gpu_expert_layers;
    }
    if (cfg.n_ubatch > 0) {
        plan.ubatch = cfg.n_ubatch;
    }

    auto svc = std::unique_ptr<InferenceService>(new InferenceService());
    svc->cfg_  = cfg;
    svc->plan_ = plan;
    svc->geo_  = **geo_res;
    const ModelGeometry & geo = svc->geo_;

    llama_model_params mparams = llama_model_default_params();
    mparams.load_mode    = LLAMA_LOAD_MODE_MMAP;   // anonymous load froze this machine once
    mparams.lazy_mode    = LLAMA_LAZY_MODE_OFF;
    mparams.n_gpu_layers = -1;                    // everything not overridden goes to the GPU
    mparams.split_mode   = LLAMA_SPLIT_MODE_NONE;

    // The plan becomes explicit buffer-type overrides: the public way to say "keep the experts of
    // these layers on the CPU", which is what -ncmoe does inside llama.cpp.
    ExpertCpuOverrides overrides;
    overrides.build(geo, plan.n_expert_layers_gpu);
    if (!overrides.empty()) {
        mparams.tensor_buft_overrides = overrides.data();
    }

    svc->model_ = llama_load_model_from_file(cfg.model_path.c_str(), mparams);
    if (!svc->model_) {
        return Status::Errorf("llama_load_model_from_file('%s') failed", cfg.model_path.c_str());
    }
    svc->vocab_ = llama_model_get_vocab(svc->model_);

    // Context init can succeed and then fail inside the CUDA graph's scratch pool, so verify real
    // free VRAM afterwards and back the ubatch off until there is margin.
    int64_t     ubatch     = plan.ubatch;
    const int64_t floor_ub  = 128;
    const uint64_t want_vram = 320ull * kMiB;
    for (int attempt = 0; attempt < 8; attempt++) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = (uint32_t) cfg.n_ctx;
        cp.n_batch         = (uint32_t) std::max<int64_t>(ubatch, 2048);
        cp.n_ubatch        = (uint32_t) ubatch;
        cp.n_seq_max       = (uint32_t) std::max(1, cfg.n_seq);
        cp.kv_unified      = false;
        cp.n_threads       = cfg.n_threads;
        cp.n_threads_batch = cfg.n_threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k          = to_ggml(cfg.cache_k);
        cp.type_v          = to_ggml(cfg.cache_v);
        cp.no_perf         = true;

        svc->ctx_ = llama_init_from_model(svc->model_, cp);
        if (svc->ctx_) {
            const uint64_t free_vram = cuda_free_bytes();
            if (free_vram == 0 || free_vram >= want_vram) {
                plan.ubatch = ubatch;
                svc->plan_  = plan;
                break;
            }
            if (ubatch <= floor_ub) {
                // Out of room to back off any further. Accept it - refusing to start would be worse -
                // but say plainly that the context is over budget, because that is the state in which
                // a later allocation can fail inside the CUDA allocator rather than here.
                plan.ubatch = ubatch;
                svc->plan_  = plan;
                AMP_WARN("amp: only ", human_bytes(free_vram),
                         " VRAM free and the ubatch is already at the floor (", ubatch,
                         "): the context is over budget. Lower --ctx or --gpu-layers.");
                break;
            }
            AMP_WARN("amp: only ", human_bytes(free_vram), " VRAM free after init, want ",
                     human_bytes(want_vram), " -> ubatch ", ubatch, " -> ",
                     std::max<int64_t>(floor_ub, ubatch / 2));
            llama_free(svc->ctx_);
            svc->ctx_ = nullptr;
            ubatch = std::max<int64_t>(floor_ub, ubatch / 2);
            continue;
        }
        if (ubatch <= floor_ub) {
            return Status::Errorf("context init failed even at ubatch %lld", (long long) ubatch);
        }
        ubatch = std::max<int64_t>(floor_ub, ubatch / 2);
    }
    if (!svc->ctx_) {
        return Status::Error("context init failed");
    }

    svc->tmpls_ = std::make_shared<ChatTemplates>();
    try {
        svc->tmpls_->tmpls = common_chat_templates_init(svc->model_, "");
    } catch (const std::exception & e) {
        return Status::Errorf("chat template init failed: %s", e.what());
    }

    svc->cache_ = std::make_unique<PrefixCache>(svc->ctx_, std::max(1, cfg.n_seq));

    // ---- warm the page cache ----
    // Demand paging with readahead does run at 1.8 GiB/s, but the first token still pays fault
    // latency, and decode (8 active experts per layer) is latency-bound rather than
    // bandwidth-bound. Warming the ranges the plan calls resident - in the plan's order, which ends
    // on the low layers that compute reaches first - converts that into a steady state.
    if (cfg.warm && !plan.resident.empty()) {
        std::vector<ReadRange> ranges;
        ranges.reserve(plan.resident.size());
        for (const auto & rp : plan.resident) {
            ranges.push_back(rp.range);
        }
        uint64_t total = 0;
        for (const auto & r : ranges) {
            total += r.length;
        }
        const MemInfo before = read_meminfo();
        AMP_INFO("amp: warming ", human_bytes(total), " in ", ranges.size(), " ranges (",
                 human_bytes(before.cache_headroom()), " headroom) ...");
        WarmProgress prog;
        prog.on_progress = [](const WarmProgress & wp) {
            if (wp.finished) {
                return;
            }
            AMP_INFO("  ", human_bytes(wp.bytes_done), " / ", human_bytes(wp.bytes_total), "  ",
                     human_rate(wp.bytes_per_sec), "  (", wp.seconds, " s)");
        };
        const Stopwatch wsw;
        const Status    wst = warm_ranges_blocking(gguf_res->file().fd(), ranges, cfg.warm_chunk, &prog);
        const double    secs = wsw.elapsed_s();
        if (!wst.ok()) {
            // A failed warm is a performance problem, not a correctness one: carry on and let the
            // kernel fault the pages in on demand.
            AMP_WARN("amp: warm failed (", wst.message(), ") - continuing with demand paging");
        } else {
            svc->warm_stats_.bytes_issued = total;
            svc->warm_stats_.bytes_queued  = total;
            svc->warm_stats_.ranges_queued = ranges.size();
            svc->warm_stats_.syscalls      = (total + cfg.warm_chunk - 1) / cfg.warm_chunk;
            AMP_INFO("amp: warmed ", human_bytes(total), " in ", secs, " s  (",
                     human_rate(secs > 0 ? (double) total / secs : 0.0), ")");
        }
    }

    AMP_INFO("amp: server ready: ctx=", cfg.n_ctx, " ubatch=", plan.ubatch, " seq=",
             std::max(1, cfg.n_seq), " gpu_expert_layers=", plan.n_expert_layers_gpu, " kv=",
             to_string(cfg.cache_k), "/", to_string(cfg.cache_v), " threads=", cfg.n_threads);
    for (const auto & n : plan.notes) {
        AMP_INFO("amp: plan: ", n);
    }
    return svc;
}

Result<RenderedChat> InferenceService::render_chat(const std::vector<ChatMessage> & messages,
                                                   const std::string & tools_json,
                                                   const std::string & template_kwargs) {
    RenderedChat out;
    if (!tmpls_ || !tmpls_->tmpls) {
        return Status::Error("chat templates not initialised");
    }

    common_chat_templates_inputs in;
    in.add_generation_prompt = true;
    in.use_jinja             = true;
    in.enable_thinking       = true;
    // The whole point: keep the assistant's own reasoning in the rendered prompt so the next turn's
    // prefix matches. Without this, a thinking model re-evaluates the tail of every turn.
    in.chat_template_kwargs["preserve_reasoning"] = "true";
    // Caller-supplied kwargs win, so enable_thinking=false actually takes effect.
    if (!template_kwargs.empty()) {
        if (auto parsed = json::Value::parse(template_kwargs); parsed.ok() && parsed->is_object()) {
            for (const auto & kv : parsed->as_object()) {
                const json::Value & v = kv.second;
                if (v.is_string()) {
                    in.chat_template_kwargs[kv.first] = v.as_string();
                } else if (v.is_bool()) {
                    in.chat_template_kwargs[kv.first] = v.as_bool() ? "true" : "false";
                }
            }
        }
    }

    for (const auto & m : messages) {
        common_chat_msg cm;
        cm.role    = m.role;
        cm.content = m.content;
        if (m.has_reasoning) {
            cm.reasoning_content = m.reasoning_content;
        }
        if (!m.tool_call_id.empty()) {
            cm.tool_call_id = m.tool_call_id;
        }
        if (!m.tool_name.empty()) {
            cm.tool_name = m.tool_name;
        }
        in.messages.push_back(std::move(cm));
    }
    if (!tools_json.empty()) {
        // Tools are optional for this model, but if a client sends them we hand them to the template
        // rather than dropping them silently. Parsed with llama-common's JSON so the types match.
        in.tools = common_chat_tools_parse_oaicompat(common_json::parse(tools_json));
    }

    const common_chat_params p = common_chat_templates_apply(tmpls_->tmpls.get(), in);
    out.prompt              = p.prompt;
    out.supported_thinking = p.supports_thinking;
    if (p.supports_thinking) {
        if (!p.thinking_start_tag.empty()) {
            out.thinking_start = p.thinking_start_tag;
        }
        if (!p.thinking_end_tags.empty()) {
            out.thinking_end = p.thinking_end_tags.front();
            for (size_t i = 1; i < p.thinking_end_tags.size(); i++) {
                if (!p.thinking_end_tags[i].empty()) {
                    out.thinking_end_alternatives.push_back(p.thinking_end_tags[i]);
                }
            }
        }
        // Does the prompt leave the model *inside* a reasoning block? Check the rendered text rather
        // than trusting a flag: what matters is the bytes the model will continue from.
        const std::string trimmed = [&] {
            size_t e = out.prompt.size();
            while (e > 0 && (out.prompt[e - 1] == ' ' || out.prompt[e - 1] == '\n' ||
                             out.prompt[e - 1] == '\r' || out.prompt[e - 1] == '\t')) {
                e--;
            }
            return out.prompt.substr(0, e);
        }();
        out.thinking_open_at_start =
            !out.thinking_start.empty() && trimmed.size() >= out.thinking_start.size() &&
            trimmed.compare(trimmed.size() - out.thinking_start.size(), out.thinking_start.size(),
                            out.thinking_start) == 0;
    }
    return out;
}

Result<std::vector<llama_token>> InferenceService::tokenize(const std::string & text,
                                                            bool add_special) const {
    if (!vocab_) {
        return Status::Error("no vocab");
    }
    int32_t n = llama_tokenize(vocab_, text.data(), (int32_t) text.size(), nullptr, 0, add_special,
                               /*parse_special=*/ true);
    if (n < 0) {
        n = -n;
    }
    std::vector<llama_token> out((size_t) n);
    const int32_t got = llama_tokenize(vocab_, text.data(), (int32_t) text.size(), out.data(), n,
                                       add_special, /*parse_special=*/ true);
    if (got < 0) {
        return Status::Errorf("llama_tokenize failed (%d)", got);
    }
    out.resize((size_t) got);
    return out;
}

void InferenceService::interrupt() { interrupt_ = true; }

const PrefixCacheStats & InferenceService::cache_stats() const { return cache_->stats(); }

std::string InferenceService::status_line() const {
    return format("model=%s ctx=%lld ubatch=%lld gpu_expert_layers=%d threads=%d | %s",
                  cfg_.model_id.c_str(), (long long) cfg_.n_ctx, (long long) plan_.ubatch,
                  plan_.n_expert_layers_gpu, cfg_.n_threads, cache_->describe().c_str());
}

Result<GenerationResult> InferenceService::generate(
    const std::vector<llama_token> & prompt_tokens, const GenerateParams & params,
    const std::function<void(const StreamChunk &)> & on_chunk) {
    // One generation at a time, on the queue's worker. The caller blocks until it is done, so the
    // streaming callback runs on the worker thread - which is fine, because that connection belongs
    // to exactly this request.
    std::mutex              done_mu;
    std::condition_variable done_cv;
    bool                    finished = false;
    Result<GenerationResult> out(Status::Error("generation was not run"));

    queue_.post([&] {
        out = generate_locked(prompt_tokens, params, on_chunk);
        {
            std::lock_guard<std::mutex> lock(done_mu);
            finished = true;
        }
        done_cv.notify_all();
    });

    {
        std::unique_lock<std::mutex> lock(done_mu);
        done_cv.wait(lock, [&] { return finished; });
    }
    return out;
}

Result<GenerationResult> InferenceService::generate_locked(
    const std::vector<llama_token> & prompt_tokens, const GenerateParams & params,
    const std::function<void(const StreamChunk &)> & on_chunk) {
    if (prompt_tokens.empty()) {
        return Status::Error("empty prompt");
    }
    if (prompt_tokens.size() + 8 > (size_t) cfg_.n_ctx) {
        return Status::Errorf("prompt of %zu tokens exceeds the %lld-token context",
                              prompt_tokens.size(), (long long) cfg_.n_ctx);
    }
    interrupt_ = false;

    GenerationResult res;
    res.n_prompt_tokens = (int32_t) prompt_tokens.size();

    // Pick the sequence with the longest common token prefix, drop what diverged, decode the rest.
    // The append case (the new turn extends the previous one, which is what an agentic client does)
    // needs no rewind at all: the cache is already exactly the prefix, so we only decode the tail.
    const int     seq     = cache_->select(prompt_tokens);
    const size_t  lcp     = cache_->common_prefix(seq, prompt_tokens);
    const int64_t n_past0 = cache_->n_past(seq);
    bool          rewound_exactly = true;
    if (lcp < (size_t) n_past0) {
        PrefixCache::Rewind rw;
        const Status        st = cache_->rewind(seq, (int64_t) lcp, &rw);
        if (!st.ok()) {
            return st;
        }
        rewound_exactly = rw.exact;
    }
    int64_t n_past = cache_->n_past(seq);
    // Invariant, asserted rather than assumed: everything the KV holds must be a prefix of this
    // prompt. If the bookkeeping ever disagrees, drop the sequence and start over - a slow answer is
    // recoverable, an answer to a different question is not.
    if (n_past < 0 || (size_t) n_past > lcp || (size_t) n_past > prompt_tokens.size()) {
        AMP_WARN("amp: cache position ", n_past, " is not a prefix of the prompt (lcp ", (int64_t) lcp,
                 ") - dropping the sequence");
        (void) cache_->invalidate(seq);
        n_past = 0;
    }
    res.n_prompt_cached   = (int32_t) n_past;
    res.n_prompt_computed = (int32_t) (prompt_tokens.size() - (size_t) n_past);
    res.cache_rewound_exactly = rewound_exactly;

    // ---- prefill ----
    // The extended batch API does *not* produce logits unless a token opts in, unlike the legacy
    // llama_batch API which always output the last token.
    const Stopwatch sw;
    int32_t          idx_sample = -1;

    // llama_process() asserts that a batch is no larger than n_ubatch (llama-context.cpp), and
    // llama_batch_ext is sized to n_batch, so the caller has to split. A prompt longer than the
    // ubatch used to fail outright with "prefill batch full at token 2048" - which is every real
    // conversation, since agent clients resend the whole history every turn.
    const int64_t ubatch_max = (int64_t) llama_n_ubatch(ctx_);
    auto run_batch = [&](int64_t from, int64_t to, bool mark_output) -> Status {
        for (int64_t start = from; start < to; start += ubatch_max) {
            const int64_t end = std::min(start + ubatch_max, to);
            const bool    last = (end >= to);
            llama_batch_ext * batch = llama_batch_ext_init(ctx_);
            int32_t           idx   = -1;
            for (int64_t i = start; i < end; i++) {
                idx = llama_batch_ext_add_token(batch, (llama_seq_id) seq, prompt_tokens[(size_t) i]);
                if (idx < 0) {
                    llama_batch_ext_free(batch);
                    return Status::Errorf("prefill batch full at token %lld (ubatch %lld)",
                                          (long long) i, (long long) ubatch_max);
                }
                const llama_pos pos = (llama_pos) i;
                llama_batch_ext_set_pos(batch, idx, &pos);
            }
            // Only the very last token of the whole prefill needs to produce logits, and that is what
            // llama_sampler_sample() reads below.
            if (mark_output && last) {
                llama_batch_ext_set_output_logits(batch, idx, true);
                idx_sample = idx;
            }
            const int32_t rc = llama_process(ctx_, LLAMA_PROCESS_TYPE_DECODE, batch);
            llama_batch_ext_free(batch);
            if (rc != 0) {
                return Status::Errorf("prefill failed (rc=%d)", rc);
            }
        }
        return Status::OK();
    };

    // Prefill stops one token short, and the checkpoint is taken there. Reason: the next turn
    // re-renders this prompt plus more text, and BPE merges across the boundary - the final token of
    // the cached prompt is often a *split* of a longer token in the next rendering, so the common
    // prefix comes up exactly one token short. A checkpoint one token back is still fully reusable;
    // recomputing that one token costs a single batch, while being one token too long costs a full
    // re-evaluation (the recurrent layers cannot rewind).
    const int64_t n_tokens   = (int64_t) prompt_tokens.size();
    const int64_t ckpt_at    = n_tokens > 0 ? n_tokens - 1 : 0;
    if (n_past < ckpt_at) {
        const Status st = run_batch(n_past, ckpt_at, /*mark_output=*/ n_past >= n_tokens);
        if (!st.ok()) {
            return st;
        }
        n_past = ckpt_at;
    }
    {
        // Snapshot the prompt boundary before generating.
        const Status cst = cache_->checkpoint(seq, std::vector<llama_token>(
                                                  prompt_tokens.begin(),
                                                  prompt_tokens.begin() + (size_t) ckpt_at));
        if (!cst.ok()) {
            AMP_WARN("amp: prompt-boundary checkpoint failed (", cst.message(),
                     ") - the next turn will re-evaluate the prompt");
        }
    }
    if (n_past < n_tokens) {
        const Status st = run_batch(n_past, n_tokens, /*mark_output=*/ true);
        if (!st.ok()) {
            return st;
        }
        n_past = n_tokens;
    }
    res.prefill_ms   = sw.elapsed_s() * 1e3;
    last_prefill_tps_ = res.prefill_tps();


    // ---- decode ----
    llama_sampler * rbudget = nullptr;
    llama_sampler * smpl     = make_sampler(params, cfg_, vocab_, prompt_tokens, &rbudget);
    std::vector<char> piece(512);
    // Fallback for the paths with no tracker (a raw completion has no template and therefore no
    // thinking tags): a string search, which is all that is possible without the template.
    ThinkingSplitter splitter(params.split_thinking ? params.think_start : std::string(),
                              params.split_thinking ? params.think_end : std::string(),
                              params.split_thinking && params.thinking_open_at_start);
    const Stopwatch  dsw;
    bool             hit_stop = false;
    bool             hit_eos  = false;
    res.finish       = FinishReason::kLength;

    if (on_chunk) {
        StreamChunk c;
        c.first = true;
        on_chunk(c);
    }

    for (int32_t i = 0; i < params.max_tokens; i++) {
        if (interrupt_) {
            break;
        }
        const llama_token id = llama_sampler_sample(smpl, ctx_, idx_sample);
        // Read the state *before* accepting: accepting the closing tag is what moves the tracker to
        // DONE, and the tag itself belongs to the reasoning side. Sampling-time state is the state the
        // token was generated in, which is the only correct question to ask.
        llama_sampler_accept(smpl, id);
        const bool eog = llama_vocab_is_eog(vocab_, id);

        if (!eog) {
            res.generated.push_back(id);
            const int32_t n = llama_token_to_piece(vocab_, id, piece.data(), (int32_t) piece.size(),
                                                   0, /*special=*/ false);
            if (n > 0) {
                const std::string text(piece.data(), (size_t) n);
                res.raw += text;
                // Which side of the reasoning block is this token on? The tracker was fed the prefill,
                // so a prompt that ends inside <think> is already in the reasoning state before the
                // first generated token, and DONE means the block has closed.
                std::string text_out, reasoning_out;
                splitter.feed(text, reasoning_out, text_out);
                res.content   += text_out;
                res.reasoning += reasoning_out;
                if (on_chunk && (!text_out.empty() || !reasoning_out.empty())) {
                    StreamChunk c;
                    c.text      = text_out;
                    c.reasoning = reasoning_out;
                    c.token     = id;
                    on_chunk(c);
                }
                for (const auto & s2 : params.stop) {
                    if (!s2.empty() && res.content.find(s2) != std::string::npos) {
                        hit_stop = true;
                    }
                }
            }
        }
        if (eog && !params.ignore_eos) {
            hit_eos = true;
            break;
        }
        if (hit_stop || i + 1 >= params.max_tokens) {
            break;
        }
        // Feed the sampled token to get the next logits: one token, at batch index 0.
        llama_batch_ext * batch = llama_batch_ext_init(ctx_);
        const int32_t    idx    = llama_batch_ext_add_token(batch, (llama_seq_id) seq, id);
        const llama_pos  pos    = (llama_pos) n_past;
        llama_batch_ext_set_pos(batch, idx, &pos);
        llama_batch_ext_set_output_logits(batch, idx, true);
        const int32_t rc = llama_process(ctx_, LLAMA_PROCESS_TYPE_DECODE, batch);
        llama_batch_ext_free(batch);
        if (rc != 0) {
            llama_sampler_free(smpl);
            // Leave the cache consistent with the KV, whatever the client does next.
            (void) cache_->restore(seq, std::vector<llama_token>(
                                               prompt_tokens.begin(),
                                               prompt_tokens.begin() + (size_t) ckpt_at));
            return Status::Errorf("decode failed (rc=%d)", rc);
        }
        idx_sample = idx;
        n_past++;
    }
    llama_sampler_free(smpl);
    {
        std::string r, c;
        splitter.flush(r, c);
        res.reasoning += r;
        res.content   += c;
    }

    res.decode_ms = dsw.elapsed_s() * 1e3;
    if (interrupt_) {
        res.finish = FinishReason::kInterrupted;
    } else if (hit_eos || hit_stop) {
        res.finish = hit_eos ? FinishReason::kEos : FinishReason::kStop;
    }
    last_decode_tps_  = res.decode_tps();
    total_generated_ += res.generated.size();

    // Put the KV back at the prompt boundary. The generated tokens are deliberately *not* kept:
    // the next turn re-renders the assistant message from text, and the sampled ids are almost never
    // byte-identical to that re-rendering, so caching them would guarantee a rewind we cannot perform.
    // The prompt is what the next turn reliably reproduces.
    if (cache_->has_checkpoint(seq)) {
        const Status rst = cache_->restore(seq, std::vector<llama_token>(
                                                  prompt_tokens.begin(),
                                                  prompt_tokens.begin() + (size_t) ckpt_at));
        if (!rst.ok()) {
            AMP_WARN("amp: could not restore the prompt checkpoint (", rst.message(),
                     ") - the next turn re-evaluates from scratch");
            (void) cache_->invalidate(seq);   // also clears the KV, so the two never disagree
        }
    } else {
        (void) cache_->invalidate(seq);
    }

    if (on_chunk) {
        StreamChunk c;
        c.done = true;
        on_chunk(c);
    }
    return res;
}

} // namespace amp
