#include "amp/runtime/model_runtime.h"

#include "amp/runtime/buft_overrides.h"

#include "amp/bytes.h"
#include "amp/format.h"
#include "amp/log.h"
#include "amp/timing.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

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

// Free VRAM on the CUDA device, via ggml's device properties (no shelling out).
uint64_t cuda_free_bytes() {
    const int n = ggml_backend_dev_count();
    for (int i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
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

} // namespace

// ---------------------------------------------------------------- Prefetcher

Prefetcher::Prefetcher(const ModelGeometry & geo, const ExecutionPlan & plan, const MappedFile & file,
                       const RuntimeConfig & cfg, std::unique_ptr<IReadScheduler> sched)
    : geo_(geo), plan_(plan), cfg_(cfg), sched_(std::move(sched)), file_(&file) {
    // Only the CPU-resident expert layers need warming; GPU-resident ones never touch the page cache.
    const int64_t n_cpu = geo_.n_layer() - plan_.n_expert_layers_gpu;
    for (int64_t il = 0; il < n_cpu; il++) {
        for (const auto & r : geo_.expert_ranges((int) il)) {
            cpu_expert_ranges_.push_back(r);
        }
    }
    if (cfg_.reverse_warm) {
        // The compute thread walks layers 0 -> n_layer-1, so warm the *last* layer first: the warm
        // then ends on the low layers, which are the ones the compute reaches first and which are
        // therefore the ones that survive in the cache. See the class comment.
        std::reverse(cpu_expert_ranges_.begin(), cpu_expert_ranges_.end());
    }
    AMP_INFO("amp: prefetcher: ", cpu_expert_ranges_.size(), " ranges, ",
             human_bytes(total_bytes()).c_str(), cfg_.reverse_warm ? " (reverse order)" : " (forward order)");
}

uint64_t Prefetcher::total_bytes() const {
    uint64_t n = 0;
    for (const auto & r : cpu_expert_ranges_) {
        n += r.length;
    }
    return n;
}

void Prefetcher::submit(const ReadRange & r) {
    if (r.empty() || !cfg_.prefetch) {
        return;
    }
    // Do not re-read what we already hold. Readahead on resident pages is a no-op in the kernel
    // but still costs a syscall per 4 MiB chunk; a single mincore() per range tells us for less.
    if (file_) {
        resident_checks_++;
        const double res = file_->resident_fraction(r.offset, r.length).value_or(0.0);
        if (res >= 0.9) {
            skipped_resident_++;
            return;
        }
    }
    ReadRequest req;
    req.range  = r;
    req.policy = ReadPolicy::kCacheWarm;  // these bytes are reused every ubatch
    if (sched_->submit(req).ok()) {
        bytes_issued_ += r.length;
        calls_++;
    }
    // Queue-full is expected under pressure and is not an error: the next ubatch re-submits.
}

void Prefetcher::prime_for_ubatch() {
    if (!cfg_.prefetch || cpu_expert_ranges_.empty()) {
        return;
    }
    // Walk a bounded number of ranges per call so that issuing never becomes the bottleneck
    // itself, and so the worker pool has time to actually pull the bytes in.
    const size_t budget_ranges = std::min<size_t>(cpu_expert_ranges_.size(), 64);
    for (size_t i = 0; i < budget_ranges; i++) {
        submit(cpu_expert_ranges_[cursor_]);
        cursor_ = (cursor_ + 1) % cpu_expert_ranges_.size();
    }
}

void Prefetcher::settle(uint32_t timeout_ms) {
    if (cfg_.prefetch && timeout_ms > 0) {
        (void) sched_->wait_idle(timeout_ms);
    }
}

void Prefetcher::rebalance() {
    if (!cfg_.prefetch) {
        return;
    }
    // Re-prime from the cursor: whatever fell out of the cache is by definition somewhere
    // behind the cursor, so continuing the walk is the cheapest way to pull it back.
    prime_for_ubatch();
}

// ---------------------------------------------------------------- ModelRuntime

ModelRuntime::~ModelRuntime() {
    if (smpl_) {
        llama_sampler_free(smpl_);
    }
    if (ctx_) {
        llama_free(ctx_);
    }
    if (model_) {
        llama_model_free(model_);
    }
}

Result<std::unique_ptr<ModelRuntime>> ModelRuntime::create(const RuntimeConfig & cfg,
                                                           const ExecutionPlan & plan,
                                                           const ModelGeometry & geo,
                                                           const MappedFile & file) {
    llama_backend_init();

    auto rt = std::unique_ptr<ModelRuntime>(new ModelRuntime());
    rt->cfg_  = cfg;
    rt->plan_ = &plan;
    rt->geo_  = &geo;

    llama_model_params mparams = llama_model_default_params();
    // Weights are always mmapped: an anonymous load of 14.65 GB froze this machine once, and
    // file-backed pages are the whole premise (evictable, re-readable, warmable on demand).
    mparams.load_mode     = LLAMA_LOAD_MODE_MMAP;
    mparams.lazy_mode     = LLAMA_LAZY_MODE_OFF;
    mparams.n_gpu_layers  = -1;   // everything that is not overridden goes to the GPU
    mparams.split_mode    = LLAMA_SPLIT_MODE_NONE;

    // The plan decides how many expert layers live on the GPU; the shared helper builds the
    // null-terminated override array (see buft_overrides.h for why the sentinel matters).
    rt->overrides_.build(geo, plan.n_expert_layers_gpu);
    if (!rt->overrides_.empty()) {
        mparams.tensor_buft_overrides = rt->overrides_.data();
        AMP_INFO("amp: pinning ", rt->overrides_.size(), " expert tensors to the CPU (layers 0..",
                 geo.n_layer() - plan.n_expert_layers_gpu - 1, ")");
    }

    rt->model_ = llama_model_load_from_file(cfg.model_path.c_str(), mparams);
    if (!rt->model_) {
        return Status::Errorf("llama_model_load_from_file('%s') failed", cfg.model_path.c_str());
    }
    rt->vocab_ = llama_model_get_vocab(rt->model_);

    // The plan is a prediction; the allocator has the last word. Try the planned ubatch and back
    // off geometrically until the context fits, because a failed init costs far more than a
    // slightly smaller batch. Each attempt is logged so the plan can be corrected.
    int64_t ubatch = cfg.n_ubatch;
    const int64_t ubatch_floor = 128;
    for (int attempt = 0; attempt < 8; attempt++) {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx           = (uint32_t) cfg.n_ctx;
        cparams.n_batch         = (uint32_t) std::max<int64_t>(ubatch, cfg.n_batch);
        cparams.n_ubatch        = (uint32_t) ubatch;
        cparams.n_seq_max       = 1;
        cparams.kv_unified      = false;   // matches the measured llama.cpp server config
        cparams.n_threads       = cfg.n_threads;
        cparams.n_threads_batch = cfg.n_threads_batch;
        cparams.flash_attn_type = cfg.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                                 : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cparams.type_k          = to_ggml(cfg.cache_k);
        cparams.type_v          = to_ggml(cfg.cache_v);
        cparams.no_perf         = true;

        rt->ctx_ = llama_init_from_model(rt->model_, cparams);
        if (rt->ctx_) {
            // Init can succeed and compute still OOM inside the CUDA graph's scratch pool
            // (observed: ggml_cuda_pool_vmm::alloc failure at ubatch 2048). So verify the
            // *actual* free VRAM and back off if the margin is thin.
            const uint64_t free_vram = cuda_free_bytes();
            const uint64_t want = 384ull * kMiB;
            if (free_vram >= want || ubatch <= ubatch_floor) {
                if (attempt > 0 || free_vram < want) {
                    AMP_WARN("amp: ubatch ", ubatch, " with ", human_bytes(free_vram),
                             " VRAM free (margin target ", human_bytes(want).c_str(), ")");
                }
                rt->cfg_.n_ubatch = ubatch;
                break;
            }
            AMP_WARN("amp: only ", human_bytes(free_vram), " VRAM free after init, want ",
                     human_bytes(want).c_str(), " -> reducing ubatch ", ubatch, " -> ",
                     std::max<int64_t>(ubatch_floor, ubatch / 2));
            llama_free(rt->ctx_);
            rt->ctx_ = nullptr;
            ubatch = std::max<int64_t>(ubatch_floor, ubatch / 2);
            continue;
        }
        if (ubatch <= ubatch_floor) {
            return Status::Errorf(
                "llama_init_from_model failed even at ubatch %lld; VRAM is too small for "
                "ctx=%lld with %d GPU expert layers (try fewer GPU expert layers)",
                (long long) ubatch, (long long) cfg.n_ctx, plan.n_expert_layers_gpu);
        }
        ubatch = std::max<int64_t>(ubatch_floor, ubatch / 2);
    }
    if (!rt->ctx_) {
        return Status::Error("llama_init_from_model failed for unknown reasons");
    }

    // Assigned unconditionally so that --logprobs-n 0 actually turns the tracking off; with
    // the old guard a 0 fell through and left the runtime at its own default.
    rt->top_k_track_ = cfg.top_k_track;
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf                  = true;
    rt->smpl_                   = llama_sampler_chain_init(sp);   // a chain is a llama_sampler
    if (cfg.top_k > 0) {
        llama_sampler_chain_add(rt->smpl_, llama_sampler_init_top_k(cfg.top_k));
    }
    if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) {
        llama_sampler_chain_add(rt->smpl_, llama_sampler_init_top_p(cfg.top_p, 1));
    }
    if (cfg.temperature > 0.0f) {
        llama_sampler_chain_add(rt->smpl_, llama_sampler_init_temp(cfg.temperature));
        llama_sampler_chain_add(rt->smpl_, llama_sampler_init_dist(cfg.seed));
    } else {
        llama_sampler_chain_add(rt->smpl_, llama_sampler_init_greedy());
    }

    ReadSchedulerConfig sched_cfg;
    sched_cfg.warmer.chunk_bytes = 4ull * 1024 * 1024;
    sched_cfg.warmer.threads     = 2;
    sched_cfg.warmer.queue_depth = 1024;
    sched_cfg.enable_streaming    = false;   // streaming arrives with the expert-major runtime (M5)
    rt->prefetch_ = std::make_unique<Prefetcher>(geo, plan, file, cfg,
                                                make_read_scheduler(sched_cfg, file.fd()));

    AMP_INFO("amp: runtime ready: ctx=", cfg.n_ctx, " ubatch=", cfg.n_ubatch,
             " gpu_expert_layers=", plan.n_expert_layers_gpu, " threads=", cfg.n_threads);
    return rt;
}

int32_t ModelRuntime::n_vocab() const {
    return vocab_ ? llama_vocab_n_tokens(vocab_) : 0;
}

Result<std::vector<llama_token>> ModelRuntime::tokenize(const std::string & text, bool add_special) const {
    if (!vocab_) {
        return Status::Error("no vocab");
    }
    // First call with n_tokens=0 returns the required size.
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

Status ModelRuntime::reset() {
    // The real context, not a null one. See the header for why that distinction mattered.
    llama_memory_clear(llama_get_memory(ctx_), true);
    n_past_ = 0;
    last_text_.clear();
    logprobs_.clear();
    return Status::OK();
}

Status ModelRuntime::prefill(const std::vector<llama_token> & tokens) {
    if (tokens.empty()) {
        return Status::Error("nothing to prefill");
    }
    // Prefill always starts from an empty cache, so a second call cannot silently append to the
    // previous one's KV and produce a sequence that is neither the prompt nor the prompt twice.
    const Status rst = reset();
    if (!rst.ok()) {
        return rst;
    }
    const int32_t ub = (int32_t) std::max<int64_t>(1, cfg_.n_ubatch);

    // Warm the cache before the first ubatch: the whole point is that the very first pass over
    // the experts is not a synchronous-fault storm.
    if (prefetch_) {
        prefetch_->prime_for_ubatch();
    }

    const Stopwatch sw;
    size_t         done = 0;
    while (done < tokens.size()) {
        const size_t n = std::min<size_t>((size_t) ub, tokens.size() - done);
        llama_batch batch = llama_batch_get_one((llama_token *) (tokens.data() + done), (int32_t) n);
        const int32_t rc = llama_decode(ctx_, batch);
        if (rc != 0) {
            return Status::Errorf("llama_decode failed during prefill (rc=%d)", rc);
        }
        done += n;
        n_past_ += (int64_t) n;
        stats_.n_decode_calls++;
        // Prime the next ubatch while this one is already in flight. Issuing is cheap; the
        // worker pool does the reading.
        if (prefetch_ && done < tokens.size()) {
            prefetch_->prime_for_ubatch();
        }
    }
    stats_.prefill_tokens = (int64_t) done;
    stats_.prefill_ms     = sw.elapsed_s() * 1e3;
    if (prefetch_) {
        stats_.warm_bytes_issued = prefetch_->bytes_issued();
        stats_.warm_calls        = prefetch_->calls();
    }
    return Status::OK();
}

// Scores a FIXED token sequence. Same as generate() except the next token is read from `tokens`
// rather than sampled, so two engines with different KV dtypes evaluate the identical prefix at
// every position and their per-position distributions are directly comparable.
//
// The distribution is captured BEFORE the token at that position is consumed, exactly as in
// generate(), so position i's top-k is the distribution over the token that appears at position i.
// That is the convention the logprob dumps already use, so existing captures stay comparable.
Result<std::vector<llama_token>> ModelRuntime::score(const std::vector<llama_token> & tokens) {
    if (!smpl_) {
        return Status::Error("no sampler");
    }
    decode_batch_.assign(1, 0);
    logprobs_.clear();

    const Stopwatch sw;
    std::vector<llama_token> scored;
    scored.reserve(tokens.size());

    for (size_t i = 0; i < tokens.size(); i++) {
        if (prefetch_) {
            prefetch_->prime_for_decode();
        }

        if (top_k_track_ > 0) {
            const float * logits = llama_get_logits_ith(ctx_, -1);
            if (logits) {
                const int n = llama_vocab_n_tokens(vocab_);
                std::vector<std::pair<float, int32_t>> all;
                all.reserve((size_t) n);
                for (int t = 0; t < n; t++) {
                    all.emplace_back(logits[t], t);
                }
                const int k = std::min(top_k_track_, n);
                std::partial_sort(all.begin(), all.begin() + k, all.end(),
                                  [](const auto & a, const auto & b) { return a.first > b.first; });
                const float maxl = all.front().first;
                double      sum  = 0.0;
                for (const auto & kv : all) {
                    sum += std::exp((double) (kv.first - maxl));
                }
                const double logZ = (double) maxl + std::log(sum);
                std::vector<TopK> tk;
                for (int j = 0; j < k; j++) {
                    tk.push_back({ all[(size_t) j].second,
                                   (float) ((double) all[(size_t) j].first - logZ) });
                }
                logprobs_.push_back(std::move(tk));
            }
        }

        const llama_token id = tokens[i];
        // The sampler still sees the forced token, so penalties and any stateful sampler stay in
        // step with what the arm is being asked to score. It is not sampling this.
        llama_sampler_accept(smpl_, id);
        if (llama_vocab_is_eog(vocab_, id)) {
            scored.push_back(id);
            break;
        }
        scored.push_back(id);

        if (i + 1 < tokens.size()) {
            decode_batch_[0] = id;
            llama_batch batch = llama_batch_get_one(decode_batch_.data(), 1);
            const int32_t rc = llama_decode(ctx_, batch);
            if (rc != 0) {
                return Status::Errorf("llama_decode failed while scoring (rc=%d)", rc);
            }
            n_past_++;
            stats_.n_decode_calls++;
        }
    }
    stats_.decode_tokens = (int64_t) scored.size();
    stats_.decode_ms    = sw.elapsed_s() * 1e3;
    return scored;
}

Result<std::vector<llama_token>> ModelRuntime::generate(int32_t max_new, std::string * text_out) {
    std::vector<llama_token> out;
    if (!smpl_) {
        return Status::Error("no sampler");
    }
    decode_batch_.assign(1, 0);
    logprobs_.clear();

    const Stopwatch sw;
    // Phase timers for AMP_TRACE_DECODE. Every decode token is prefetch + decode + sample, and
    // the sample and the prefetch are host work that has nothing to do with the model, so
    // without this the only way to attribute a token is to infer it from a sweep.
    const bool  trace    = getenv("AMP_TRACE_DECODE") != nullptr;
    double      t_pref = 0.0, t_dec = 0.0, t_smpl = 0.0, t_logp = 0.0, t_acc_ms = 0.0, t_gap_ms = 0.0, t_call_ms = 0.0;
    int64_t     t0 = 0;
    // The KV cache already holds the whole prompt, so the logits for the *next* token are ready.
    // Sample first, then feed the sampled token: decoding before sampling would append a spurious
    // token (this bug shifted every generation by one and was caught by the logit parity check).
    for (int32_t i = 0; i < max_new; i++) {
        t0 = trace ? ggml_time_us() : 0;
        if (prefetch_) {
            // Deliberately not prefetching the full expert set during decode: decode only needs the
            // active experts, and pulling 12 GiB per token evicts the resident set. See
            // Prefetcher::prime_for_decode().
            prefetch_->prime_for_decode();
        }
        if (trace) { t_pref += (double) (ggml_time_us() - t0); t0 = ggml_time_us(); }

        // Capture the distribution we are about to sample from, so quality can be compared
        // numerically against another engine rather than by eyeballing text.
        if (top_k_track_ > 0) {
            const float * logits = llama_get_logits_ith(ctx_, -1);
            if (logits) {
                const int n = llama_vocab_n_tokens(vocab_);
                std::vector<std::pair<float, int32_t>> all;
                all.reserve((size_t) n);
                for (int t = 0; t < n; t++) {
                    all.emplace_back(logits[t], t);
                }
                const int k = std::min(top_k_track_, n);
                std::partial_sort(all.begin(), all.begin() + k, all.end(),
                                  [](const auto & a, const auto & b) { return a.first > b.first; });
                const float maxl = all.front().first;
                double      sum  = 0.0;
                for (const auto & kv : all) {
                    sum += std::exp((double) (kv.first - maxl));
                }
                const double logZ = (double) maxl + std::log(sum);
                std::vector<TopK> tk;
                for (int j = 0; j < k; j++) {
                    tk.push_back({ all[(size_t) j].second,
                                   (float) ((double) all[(size_t) j].first - logZ) });
                }
                logprobs_.push_back(std::move(tk));
            }
        }

        const int64_t t_a = trace ? ggml_time_us() : 0;
        const llama_token id = llama_sampler_sample(smpl_, ctx_, -1);
        const int64_t t_b = trace ? ggml_time_us() : 0;
        llama_sampler_accept(smpl_, id);
        if (trace) {
            const int64_t t_now = ggml_time_us();
            t_gap_ms += (double) (t_a - t0);
            t_call_ms += (double) (t_b - t_a);
            t_acc_ms += (double) (t_now - t_b);
            t_smpl += (double) (t_now - t0);
            t0 = t_now;
        }
        if (llama_vocab_is_eog(vocab_, id)) {
            break;
        }
        out.push_back(id);

        if (i + 1 < max_new) {
            decode_batch_[0] = id;
            llama_batch batch = llama_batch_get_one(decode_batch_.data(), 1);
            const int32_t rc = llama_decode(ctx_, batch);
            if (trace) { t_dec += (double) (ggml_time_us() - t0); }
            if (rc != 0) {
                return Status::Errorf("llama_decode failed during decode (rc=%d)", rc);
            }
            n_past_++;
            stats_.n_decode_calls++;
        }

        if (cfg_.verbose && (out.size() % 32 == 0)) {
            const double el = sw.elapsed_s();
            AMP_INFO("amp: decode ", out.size(), " tokens, ", format("%.2f", (double) out.size() / el),
                     " t/s cumulative");
        }
    }
    stats_.decode_tokens = (int64_t) out.size();
    stats_.decode_ms    = sw.elapsed_s() * 1e3;

    if (trace && out.size() > 0) {
        const double n   = (double) out.size();
        const double tot = t_pref + t_dec + t_smpl + t_logp;
        AMP_INFO("amp: decode phases per token (", out.size(), " tokens, ",
                 format("%.2f", stats_.decode_ms / n), " ms/token)",
                 "  prefetch ",      format("%.3f", t_pref / 1e3 / n), " ms",
                 "  llama_decode ",  format("%.3f", t_dec  / 1e3 / n), " ms",
                 "  sampler ",       format("%.3f", t_smpl  / 1e3 / n), " ms",
                 "    gap ", format("%.3f", t_gap_ms / 1e3 / n),
                 "    call ", format("%.3f", t_call_ms / 1e3 / n),
                 "    accept ", format("%.3f", t_acc_ms / 1e3 / n), " ms",
                 "  logits_fetch ", format("%.3f", t_logp  / 1e3 / n), " ms",
                 "  unaccounted ",   format("%.3f", (stats_.decode_ms * 1e3 - tot) / 1e3 / n), " ms");
    }

    // Detokenize the whole completion. Pieces can be partial UTF-8, so accumulate bytes and
    // strip a trailing incomplete sequence.
    std::string text;
    text.reserve(out.size() * 4);
    std::vector<char> buf(256);
    for (const llama_token t : out) {
        const int32_t n = llama_token_to_piece(vocab_, t, buf.data(), (int32_t) buf.size(), 0, false);
        if (n > 0) {
            text.append(buf.data(), (size_t) n);
        }
    }
    last_text_ = text;
    if (text_out) {
        *text_out = text;
    }
    return out;
}

} // namespace amp
