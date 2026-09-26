// Token-level prefix cache.
//
// The single most valuable thing a server does for an agentic client. Clients like OpenCode resend
// the entire conversation every turn, so a request usually shares a long prefix with the previous
// one. Matching at the *token* level and keeping that prefix in the KV cache turns "re-read the
// whole 30k-token history" into "decode the few hundred new tokens".
//
// This is also where the bug that costs llama.cpp users 1-8 minutes per turn lives: if the client
// does not round-trip the assistant's reasoning, the rendered prompt diverges at the first token of
// the last assistant turn and everything after it - response text, tool calls, tool results - is
// re-evaluated. The cache here is deliberately dumb about *why* a prompt changed; it just finds the
// longest common token prefix, which is the only correct definition.
#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "amp/status.h"
#include "llama.h"

namespace amp {

struct PrefixCacheStats {
    uint64_t requests          = 0;
    uint64_t full_hits         = 0;   // the entire prompt was already in the KV cache
    uint64_t partial_hits      = 0;
    uint64_t tokens_reused     = 0;
    uint64_t tokens_computed   = 0;
    uint64_t evictions         = 0;   // times a sequence had to be dropped
    uint64_t rewinds_exact     = 0;   // the divergent tail was removed in place
    uint64_t rewinds_restored  = 0;   // rewound by restoring the prompt-boundary checkpoint
    uint64_t rewinds_cleared   = 0;   // the memory could not rewind, so we re-prefilled from 0
    double   hit_rate() const {
        const uint64_t total = tokens_reused + tokens_computed;
        return total ? (double) tokens_reused / (double) total : 0.0;
    }
};

class PrefixCache {
public:
    PrefixCache(llama_context * ctx, int n_seq);

    // Longest common token prefix between the cached sequence and `tokens`.
    size_t common_prefix(int seq, const std::vector<llama_token> & tokens) const;

    // Pick the sequence with the longest common prefix. Ties go to the most recently used.
    int select(const std::vector<llama_token> & tokens);

    // Number of tokens whose KV is live for `seq` right now.
    int64_t n_past(int seq) const;

    // What a rewind actually achieved. This model cannot always rewind: 30 of its 40 layers are
    // recurrent (linear attention), and a recurrent state is a running summary, so a mid-sequence
    // erase is only possible through a bounded snapshot ring (n_rs_seq, which costs 62.8 MiB of VRAM
    // per snapshot on this model and is off by default). llama_memory_seq_rm() reports failure in
    // that case, but it has also been observed to return true without moving the position, so the
    // position is verified rather than trusted.
    struct Rewind {
        int64_t usable_from = 0;   // first position the caller may write at
        bool    exact       = false;  // true: [from, n_past) was removed; false: sequence was cleared
    };

    // Drop cached KV for [from, n_past) of `seq`, so the next decode can rewrite it. Falls back to
    // clearing the whole sequence (and reporting usable_from = 0) when the memory cannot rewind.
    Status rewind(int seq, int64_t from, Rewind * out);

    // The position the memory module currently holds for `seq`, i.e. how far it can be rewound.
    int64_t memory_pos_max(int seq) const;

    // ---- prompt-boundary checkpoints ------------------------------------------------
    //
    // 30 of this model's 40 layers are recurrent, so a mid-sequence erase is impossible: a recurrent
    // state is a running summary, not a per-token record. That makes the naive "cache prompt+answer
    // and rewind on the next turn" design useless here - the next turn's prompt almost never matches
    // the sampled tokens exactly, and one token short is already too short.
    //
    // So amp checkpoints the sequence at the *prompt boundary* and restores it after generation.
    // The KV then always sits exactly at the end of the prompt, and the next turn - which an agentic
    // client builds by appending to the same history - simply extends it. Cost is one save and one
    // restore per turn, proportional to the conversation's KV, and it turns a 30k-token re-evaluation
    // into a memcpy.
    Status checkpoint(int seq, const std::vector<llama_token> & tokens);
    // `prompt_tokens` must be the tokens the checkpoint was taken over: the restore sets the cache's
    // token list from them, which is what the next request's common prefix is measured against.
    Status restore(int seq, const std::vector<llama_token> & prompt_tokens);
    bool   has_checkpoint(int seq) const;
    double checkpoint_save_ms() const { return ckpt_save_ms_; }
    double checkpoint_load_ms() const { return ckpt_load_ms_; }
    uint64_t checkpoint_bytes() const;

    // Record that `tokens` (and everything generated after them) is now live in the KV cache.
    void commit(int seq, const std::vector<llama_token> & tokens);

    void invalidate(int seq);
    void clear();

    const PrefixCacheStats & stats() const;
    std::string describe() const;

private:
public:
    struct Seq {
        std::vector<llama_token> tokens;   // what the KV currently holds
        int64_t                   n_past  = 0;
        uint64_t                  last_used = 0;
        std::vector<uint8_t>      ckpt;          // sequence state at the prompt boundary
        int64_t                   ckpt_tokens = 0;
        bool                      ckpt_valid  = false;
    };

    llama_context *        ctx_;
    mutable std::mutex     mu_;
    std::vector<Seq>       seqs_;
    uint64_t               clock_ = 0;
    PrefixCacheStats       stats_;
    double                 ckpt_save_ms_ = 0.0;
    double                 ckpt_load_ms_ = 0.0;
};

} // namespace amp
