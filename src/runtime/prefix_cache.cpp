#include "amp/runtime/prefix_cache.h"

#include "amp/bytes.h"
#include "amp/format.h"
#include "amp/timing.h"

#include <algorithm>

namespace amp {

PrefixCache::PrefixCache(llama_context * ctx, int n_seq) : ctx_(ctx) {
    seqs_.resize((size_t) std::max(1, n_seq));
}

size_t PrefixCache::common_prefix(int seq, const std::vector<llama_token> & tokens) const {
    std::lock_guard<std::mutex> lock(mu_);
    const Seq &                 s = seqs_[(size_t) seq];
    const size_t                n = std::min(s.tokens.size(), tokens.size());
    size_t                      i = 0;
    while (i < n && s.tokens[i] == tokens[i]) {
        i++;
    }
    return i;
}

int PrefixCache::select(const std::vector<llama_token> & tokens) {
    std::lock_guard<std::mutex> lock(mu_);
    int      best      = 0;
    size_t   best_len  = 0;
    uint64_t best_time = 0;
    for (size_t i = 0; i < seqs_.size(); i++) {
        const Seq & s = seqs_[i];
        const size_t n = std::min(s.tokens.size(), tokens.size());
        size_t       l = 0;
        while (l < n && s.tokens[l] == tokens[l]) {
            l++;
        }
        if (l > best_len || (l == best_len && s.last_used > best_time)) {
            best      = (int) i;
            best_len  = l;
            best_time = s.last_used;
        }
    }
    seqs_[(size_t) best].last_used = ++clock_;

    stats_.requests++;
    const size_t live = (size_t) std::min<int64_t>(seqs_[(size_t) best].n_past, (int64_t) best_len);
    stats_.tokens_reused += live;
    stats_.tokens_computed += tokens.size() - live;
    if (live == tokens.size() && !tokens.empty()) {
        stats_.full_hits++;
    } else if (live > 0) {
        stats_.partial_hits++;
    }
    return best;
}

int64_t PrefixCache::n_past(int seq) const {
    std::lock_guard<std::mutex> lock(mu_);
    return seqs_[(size_t) seq].n_past;
}

// Restores `s.ckpt` into sequence `sid`. Caller holds mu_. Returns false and invalidates the
// checkpoint on any failure, so a broken restore can never be mistaken for a valid cache.
static bool restore_locked(PrefixCache::Seq & s, llama_context * ctx, llama_seq_id sid,
                           const std::vector<llama_token> & prompt_tokens) {
    if (!s.ckpt_valid) {
        return false;
    }
    // set_data overwrites the sequence wholesale, but clear first so a failure cannot leave a
    // half-restored sequence behind (which would then be read as a valid cache).
    (void) llama_memory_seq_rm(llama_get_memory(ctx), sid, 0, -1);
    const size_t got = llama_state_seq_set_data(ctx, s.ckpt.data(), s.ckpt.size(), sid);
    if (got != s.ckpt.size()) {
        s.ckpt_valid = false;
        s.tokens.clear();
        s.n_past = 0;
        return false;
    }
    // The token list is *set*, not resized: the previous contents were prompt+answer, and a blind
    // resize would leave the generated tail in place as if it were part of the cached prompt.
    s.tokens = prompt_tokens;
    s.n_past = std::min<int64_t>(s.ckpt_tokens, (int64_t) s.tokens.size());
    return true;
}

Status PrefixCache::checkpoint(int seq, const std::vector<llama_token> & tokens) {
    std::lock_guard<std::mutex> lock(mu_);
    Seq &                       s = seqs_[(size_t) seq];
    const llama_seq_id sid       = (llama_seq_id) seq;

    const size_t size = llama_state_seq_get_size(ctx_, sid);
    if (size == 0) {
        s.ckpt_valid = false;
        return Status::Error("llama_state_seq_get_size returned 0");
    }
    s.ckpt.resize(size);
    const Stopwatch sw;
    const size_t    got = llama_state_seq_get_data(ctx_, s.ckpt.data(), size, sid);
    ckpt_save_ms_ += sw.elapsed_s() * 1e3;
    if (got != size) {
        s.ckpt_valid = false;
        return Status::Errorf("llama_state_seq_get_data wrote %zu of %zu bytes", got, size);
    }
    s.ckpt_tokens = (int64_t) tokens.size();
    s.ckpt_valid  = true;
    return Status::OK();
}

Status PrefixCache::restore(int seq, const std::vector<llama_token> & prompt_tokens) {
    std::lock_guard<std::mutex> lock(mu_);
    Seq &                       s = seqs_[(size_t) seq];
    if (!s.ckpt_valid) {
        return Status::Error("no checkpoint for this sequence");
    }
    const llama_seq_id sid    = (llama_seq_id) seq;
    const Stopwatch    sw;
    const bool         ok     = restore_locked(s, ctx_, sid, prompt_tokens);
    ckpt_load_ms_ += sw.elapsed_s() * 1e3;
    if (!ok) {
        return Status::Error("checkpoint restore failed (see the log)");
    }
    return Status::OK();
}

bool PrefixCache::has_checkpoint(int seq) const {
    std::lock_guard<std::mutex> lock(mu_);
    return seqs_[(size_t) seq].ckpt_valid;
}

uint64_t PrefixCache::checkpoint_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t                     n = 0;
    for (const auto & s : seqs_) {
        n += s.ckpt.size();
    }
    return n;
}

// The last *occupied* position, or -1 when the sequence is empty. Note this is a position index, not
// a count: a cache holding 25 tokens reports 24. Comparing it against a token count is an off-by-one
// that silently "verifies" a rewind which never happened.
int64_t PrefixCache::memory_pos_max(int seq) const {
    return (int64_t) llama_memory_seq_pos_max(llama_get_memory(ctx_), (llama_seq_id) seq) - 1;
}

Status PrefixCache::rewind(int seq, int64_t from, Rewind * out) {
    std::lock_guard<std::mutex> lock(mu_);
    Seq &                       s = seqs_[(size_t) seq];
    out->usable_from            = 0;
    out->exact                  = false;
    if (from >= s.n_past) {
        // Nothing diverged, or the request is already behind the cache.
        out->usable_from = from;
        out->exact       = true;
        return Status::OK();
    }

    llama_memory_i * mem = llama_get_memory(ctx_);
    (void) llama_memory_seq_rm(mem, (llama_seq_id) seq, (llama_pos) from, (llama_pos) s.n_past);

    // Trust the position, not the return value.
    if (memory_pos_max(seq) == from - 1) {
        s.tokens.resize((size_t) std::max<int64_t>(0, from));
        s.n_past      = from;
        out->usable_from = from;
        out->exact       = true;
        stats_.rewinds_exact++;
        return Status::OK();
    }

    // The memory would not rewind (recurrent layers, no snapshot ring). If the request still covers
    // the prompt boundary we checkpointed, restoring that state puts the KV back at a usable
    // position: a memcpy instead of a re-evaluation, which is the whole point of the checkpoint.
    // Only usable if the new prompt reproduces the checkpoint: the checkpoint must sit at or before
    // the common prefix. If it sits *after* the divergence point, the restored state contains tokens
    // this request does not have, and reusing it would answer a different question.
    if (s.ckpt_valid && s.ckpt_tokens <= from) {
        const Stopwatch sw;
        // The prompt tokens are already s.tokens up to `from`; the checkpoint may sit further along,
        // so restore against the cache's own (truncated) list.
        std::vector<llama_token> known(s.tokens.begin(), s.tokens.begin() + (size_t) from);
        if (restore_locked(s, ctx_, (llama_seq_id) seq, known)) {
            ckpt_load_ms_ += sw.elapsed_s() * 1e3;
            stats_.rewinds_restored++;
            out->usable_from = s.n_past;
            out->exact       = true;
            return Status::OK();
        }
        // Restore failed: fall through to the full clear rather than trust a broken state.
        s.ckpt_valid = false;
    }

    // Otherwise drop the whole sequence and prefill again: a full re-evaluation, but never a wrong
    // answer and never a position-continuity abort.
    (void) llama_memory_seq_rm(mem, (llama_seq_id) seq, 0, -1);
    s.tokens.clear();
    s.n_past         = 0;
    out->usable_from = 0;
    out->exact       = false;
    stats_.rewinds_cleared++;
    stats_.evictions++;
    return Status::OK();
}

void PrefixCache::commit(int seq, const std::vector<llama_token> & tokens) {
    std::lock_guard<std::mutex> lock(mu_);
    Seq &                       s = seqs_[(size_t) seq];
    s.tokens = tokens;
    s.n_past = (int64_t) tokens.size();
}

void PrefixCache::invalidate(int seq) {
    std::lock_guard<std::mutex> lock(mu_);
    Seq &                       s = seqs_[(size_t) seq];
    stats_.evictions++;
    // Clear the KV too: the cache bookkeeping and the memory module must never disagree.
    (void) llama_memory_seq_rm(llama_get_memory(ctx_), (llama_seq_id) seq, 0, -1);
    s.tokens.clear();
    s.n_past      = 0;
    s.ckpt_valid  = false;
    s.ckpt_tokens = 0;
}

void PrefixCache::clear() {
    std::lock_guard<std::mutex> lock(mu_);
    for (size_t i = 0; i < seqs_.size(); i++) {
        (void) llama_memory_seq_rm(llama_get_memory(ctx_), (llama_seq_id) i, 0, -1);
        seqs_[i].tokens.clear();
        seqs_[i].n_past      = 0;
        seqs_[i].ckpt_valid  = false;
        seqs_[i].ckpt_tokens = 0;
    }
    stats_ = PrefixCacheStats{};
}

const PrefixCacheStats & PrefixCache::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
}

std::string PrefixCache::describe() const {
    std::lock_guard<std::mutex> lock(mu_);
    const PrefixCacheStats &     st = stats_;
    std::string                  s = format("requests=%s full=%s partial=%s reused=%s computed=%s",
                                           human_count(st.requests).c_str(),
                                           human_count(st.full_hits).c_str(),
                                           human_count(st.partial_hits).c_str(),
                                           human_count(st.tokens_reused).c_str(),
                                           human_count(st.tokens_computed).c_str());
    if (st.rewinds_exact || st.rewinds_restored || st.rewinds_cleared) {
        s += format(" rewinds=%s exact/%s restored/%s cleared",
                    human_count(st.rewinds_exact).c_str(),
                    human_count(st.rewinds_restored).c_str(),
                    human_count(st.rewinds_cleared).c_str());
    }
    for (size_t i = 0; i < seqs_.size(); i++) {
        s += format(" | seq%zu: %s cached", i, human_bytes(seqs_[i].tokens.size() * 4).c_str());
    }
    return s;
}

} // namespace amp
