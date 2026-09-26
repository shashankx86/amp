#include "amp/model/geometry.h"

#include "amp/bytes.h"
#include "amp/format.h"
#include "amp/log.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace amp {

CacheType cache_type_from_string(const std::string & s) {
    if (s == "f32")  return CacheType::kF32;
    if (s == "f16")  return CacheType::kF16;
    if (s == "bf16") return CacheType::kBF16;
    if (s == "q8_0") return CacheType::kQ8_0;
    if (s == "q5_1") return CacheType::kQ5_1;
    if (s == "q5_0") return CacheType::kQ5_0;
    if (s == "q4_1") return CacheType::kQ4_1;
    if (s == "q4_0") return CacheType::kQ4_0;
    if (s == "q4_K") return CacheType::kQ4_K;
    if (s == "iq4_nl") return CacheType::kIQ4_NL;
    if (s == "q3_K") return CacheType::kQ3_K;
    if (s == "q2_K") return CacheType::kQ2_K;
    if (s == "q6_K") return CacheType::kQ6_K;
    return CacheType::kF16;
}

const char * to_string(CacheType t) {
    switch (t) {
        case CacheType::kF32:    return "f32";
        case CacheType::kF16:    return "f16";
        case CacheType::kBF16:   return "bf16";
        case CacheType::kQ8_0:   return "q8_0";
        case CacheType::kQ5_1:   return "q5_1";
        case CacheType::kQ5_0:   return "q5_0";
        case CacheType::kQ4_1:   return "q4_1";
        case CacheType::kQ4_0:   return "q4_0";
        case CacheType::kQ4_K:   return "q4_K";
        case CacheType::kIQ4_NL: return "iq4_nl";
        case CacheType::kQ3_K:   return "q3_K";
        case CacheType::kQ2_K:   return "q2_K";
        case CacheType::kQ6_K:   return "q6_K";
    }
    return "?";
}

namespace {
int64_t blck(CacheType t) {
    switch (t) {
        case CacheType::kF32: case CacheType::kF16: case CacheType::kBF16: return 1;
        case CacheType::kQ8_0: return 32;
        case CacheType::kQ5_1: case CacheType::kQ5_0: case CacheType::kQ4_1:
        case CacheType::kQ4_0: case CacheType::kIQ4_NL: return 32;
        default: return 256;
    }
}
int64_t bytes(CacheType t) {
    switch (t) {
        case CacheType::kF32:  return 4;
        case CacheType::kF16:  return 2;
        case CacheType::kBF16: return 2;
        case CacheType::kQ8_0: return 34;
        case CacheType::kQ5_1: return 24;
        case CacheType::kQ5_0: return 22;
        case CacheType::kQ4_1: return 20;
        case CacheType::kQ4_0: return 18;
        case CacheType::kQ4_K: return 144;
        case CacheType::kIQ4_NL: return 18;
        case CacheType::kQ3_K: return 110;
        case CacheType::kQ2_K: return 84;
        case CacheType::kQ6_K: return 210;
    }
    return 2;
}
} // namespace

int64_t cache_type_bytes(CacheType t) { return bytes(t); }
int64_t cache_type_blck(CacheType t) { return blck(t); }

Result<std::unique_ptr<ModelGeometry>> ModelGeometry::build(const GGUFFile & gguf) {
    auto g = std::make_unique<ModelGeometry>();
    g->gguf_        = &gguf;
    g->arch_        = gguf.architecture();
    g->n_layer_     = gguf.n_layer();
    g->n_embd_      = gguf.n_embd();
    g->n_expert_    = gguf.n_expert();
    g->n_expert_used_ = gguf.n_expert_used();
    g->expert_ff_   = gguf.expert_ff();
    g->shared_ff_   = gguf.shared_ff();
    g->n_head_      = gguf.n_head();
    g->n_head_kv_   = gguf.n_head_kv();
    g->head_dim_    = gguf.head_dim();
    g->n_ctx_train_ = gguf.n_ctx_train();
    g->rope_dim_    = gguf.rope_dim();
    g->rope_base_   = gguf.rope_base();
    g->rms_eps_     = gguf.rms_eps();
    g->rope_sections_ = gguf.rope_sections();
    g->ssm_inner_   = gguf.ssm_inner();
    g->ssm_state_   = gguf.ssm_state();
    g->ssm_groups_  = gguf.ssm_groups();
    g->ssm_dt_rank_ = gguf.ssm_dt_rank();
    g->ssm_conv_kernel_ = gguf.ssm_conv_kernel();
    g->data_offset_ = gguf.data_offset();

    if (g->n_layer_ <= 0 || g->n_layer_ > 512) {
        return Status::Errorf("unsupported block_count %lld", (long long) g->n_layer_);
    }
    if (g->n_expert_ <= 0) {
        return Status::Errorf("unsupported expert_count %lld", (long long) g->n_expert_);
    }

    g->layers_.resize((size_t) g->n_layer_);
    g->expert_ranges_.resize((size_t) g->n_layer_);
    for (int64_t il = 0; il < g->n_layer_; il++) {
        LayerInfo & li = g->layers_[(size_t) il];
        li.index     = (int) il;
        li.recurrent = gguf.layer_is_recurrent((int) il);
        if (!li.recurrent) {
            g->full_attn_layers_.push_back((int) il);
        }

        const std::string pfx = "blk." + std::to_string(il) + ".";
        for (const auto & t : gguf.tensors()) {
            if (t.name.compare(0, pfx.size(), pfx) != 0) {
                continue;
            }
            const bool is_expert = t.name.find("_exps.") != std::string::npos;
            const bool is_shared = t.name.find("_shexp.") != std::string::npos;
            const bool is_router = t.name.find("ffn_gate_inp") != std::string::npos;

            if (is_router) {
                g->router_bytes_ += (int64_t) t.nbytes;
            } else if (is_shared) {
                g->shared_expert_bytes_ += (int64_t) t.nbytes;
            }
            if (is_expert) {
                li.expert_bytes += (int64_t) t.nbytes;
                g->expert_ranges_[il].push_back(
                    ReadRange{ gguf.tensor_file_offset(t), t.nbytes });
            } else {
                li.other_bytes += (int64_t) t.nbytes;
            }
            if (li.expert_offset == 0 || (uint64_t) li.expert_bytes == t.nbytes) {
                // remember the lowest offset of each group
                const uint64_t off = gguf.tensor_file_offset(t);
                if (li.expert_offset == 0 || off < li.expert_offset) {
                    li.expert_offset = is_expert ? off : li.expert_offset;
                }
            }
            if (li.other_offset == 0 || gguf.tensor_file_offset(t) < li.other_offset) {
                if (!is_expert) {
                    li.other_offset = gguf.tensor_file_offset(t);
                }
            }
        }
        // expert ranges in file order (gate, up, down as stored)
        std::sort(g->expert_ranges_[il].begin(), g->expert_ranges_[il].end(),
                  [](const ReadRange & a, const ReadRange & b) { return a.offset < b.offset; });

        g->total_expert_bytes_ += li.expert_bytes;
        g->total_other_bytes_  += li.other_bytes;
        g->max_layer_expert_bytes_ = std::max(g->max_layer_expert_bytes_, li.expert_bytes);
        if (g->min_layer_expert_bytes_ == 0 || li.expert_bytes < g->min_layer_expert_bytes_) {
            g->min_layer_expert_bytes_ = li.expert_bytes;
        }
    }

    for (const auto & t : gguf.tensors()) {
        if (t.name == "token_embd.weight" || t.name == "output.weight") {
            g->embedding_bytes_ += (int64_t) t.nbytes;
        }
    }
    // Global tensors (embeddings, output head) are fixed VRAM cost as well; keep them in the
    // "other" total so the planner reserves for them, but report them separately too.
    g->total_other_bytes_ += g->embedding_bytes_;

    if (g->full_attn_layers_.empty()) {
        return Status::Error("model has no full-attention layers");
    }
    AMP_DEBUG("amp: geometry: ", g->summary());
    return g;
}

int64_t ModelGeometry::bytes_per_expert(int il) const {
    if (il < 0 || il >= (int) layers_.size()) {
        return 0;
    }
    return layers_[(size_t) il].expert_bytes / std::max<int64_t>(1, n_expert_);
}

std::vector<ReadRange> ModelGeometry::expert_ranges(int il) const {
    if (il < 0 || il >= (int) expert_ranges_.size()) {
        return {};
    }
    return expert_ranges_[il];
}

int64_t ModelGeometry::kv_bytes_per_token(CacheType k, CacheType v) const {
    // Per attention layer and per token: n_head_kv * head_dim elements of K, and the same of V.
    // Bytes-per-element is fractional for quantised types, so this must be computed in double.
    const double elems = (double) (n_head_kv_ * head_dim_);
    const double bk    = (blck(k) > 0) ? (double) bytes(k) / (double) blck(k) : 0.0;
    const double bv    = (blck(v) > 0) ? (double) bytes(v) / (double) blck(v) : 0.0;
    const double per_layer = elems * (bk + bv);
    return (int64_t) llround(per_layer * (double) full_attn_layers_.size());
}

int64_t ModelGeometry::kv_bytes(int64_t n_tokens, CacheType k, CacheType v) const {
    return n_tokens * kv_bytes_per_token(k, v);
}

int64_t ModelGeometry::ssm_state_bytes() const {
    // state_size x inner_size per recurrent layer, fp32
    int64_t total = 0;
    for (const auto & li : layers_) {
        if (li.recurrent) {
            total += ssm_state_ * ssm_inner_ * 4;
        }
    }
    return total;
}

std::string ModelGeometry::summary() const {
    return format(
        "arch=%s layers=%lld embd=%lld experts=%lld/%lld expert_ff=%lld shared_ff=%lld "
        "heads=%lld/%lld head_dim=%lld ctx_train=%lld attn_layers=%zu",
        arch_.c_str(), (long long) n_layer_, (long long) n_embd_, (long long) n_expert_used_,
        (long long) n_expert_, (long long) expert_ff_, (long long) shared_ff_, (long long) n_head_,
        (long long) n_head_kv_, (long long) head_dim_, (long long) n_ctx_train_,
        full_attn_layers_.size());
}

} // namespace amp
