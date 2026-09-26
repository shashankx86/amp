// ModelGeometry: everything the runtime and the planner need to know about the model,
// derived once from the GGUF header. Pure data + queries, no I/O.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "amp/io/warmer.h"
#include "amp/model/gguf.h"
#include "amp/status.h"

namespace amp {

// Quantised cache type for K and V, mirroring llama.cpp's -ctk/-ctv.
enum class CacheType {
    kF32, kF16, kBF16, kQ8_0, kQ5_1, kQ5_0, kQ4_1, kQ4_0, kQ4_K, kIQ4_NL, kQ3_K, kQ2_K, kQ6_K,
};

CacheType    cache_type_from_string(const std::string & s);
const char * to_string(CacheType t);
int64_t      cache_type_bytes(CacheType t);  // 0 means "variable/fp"
int64_t      cache_type_blck(CacheType t);

struct LayerInfo {
    int     index        = 0;
    bool    recurrent    = false;   // SSM/GDN layer vs full attention
    int64_t expert_bytes = 0;       // ffn_{gate,up,down}_exps
    int64_t other_bytes  = 0;       // attention / ssm / shared expert / router
    uint64_t expert_offset = 0;     // absolute file offset of the first expert tensor
    uint64_t other_offset  = 0;
};

class ModelGeometry {
public:
    static Result<std::unique_ptr<ModelGeometry>> build(const GGUFFile & gguf);

    // ---- identity ----
    const std::string & architecture() const { return arch_; }
    int64_t n_layer() const { return n_layer_; }
    int64_t n_embd() const { return n_embd_; }
    int64_t n_expert() const { return n_expert_; }
    int64_t n_expert_used() const { return n_expert_used_; }
    int64_t expert_ff() const { return expert_ff_; }
    int64_t shared_ff() const { return shared_ff_; }
    int64_t n_head() const { return n_head_; }
    int64_t n_head_kv() const { return n_head_kv_; }
    int64_t head_dim() const { return head_dim_; }
    int64_t n_ctx_train() const { return n_ctx_train_; }
    int64_t rope_dim() const { return rope_dim_; }
    double  rope_base() const { return rope_base_; }
    float   rms_eps() const { return rms_eps_; }
    const std::vector<int64_t> & rope_sections() const { return rope_sections_; }
    int64_t ssm_inner() const { return ssm_inner_; }
    int64_t ssm_state() const { return ssm_state_; }
    int64_t ssm_groups() const { return ssm_groups_; }
    int64_t ssm_dt_rank() const { return ssm_dt_rank_; }
    int64_t ssm_conv_kernel() const { return ssm_conv_kernel_; }

    const std::vector<LayerInfo> & layers() const { return layers_; }
    const std::vector<int> &      full_attention_layers() const { return full_attn_layers_; }

    // ---- sizes ----
    int64_t total_expert_bytes() const { return total_expert_bytes_; }
    int64_t total_other_bytes() const { return total_other_bytes_; }
    int64_t total_tensor_bytes() const { return total_expert_bytes_ + total_other_bytes_; }
    int64_t embedding_bytes() const { return embedding_bytes_; }
    int64_t router_bytes() const { return router_bytes_; }
    int64_t shared_expert_bytes() const { return shared_expert_bytes_; }
    int64_t max_layer_expert_bytes() const { return max_layer_expert_bytes_; }
    int64_t min_layer_expert_bytes() const { return min_layer_expert_bytes_; }

    // bytes of a single expert in a single layer (gate+up+down)
    int64_t bytes_per_expert(int il) const;
    // the three expert tensor ranges of one layer, in file order
    std::vector<ReadRange> expert_ranges(int il) const;

    // KV cache bytes for n_past tokens with the given cache types.
    int64_t kv_bytes(int64_t n_tokens, CacheType k, CacheType v) const;
    int64_t kv_bytes_per_token(CacheType k, CacheType v) const;
    // SSM recurrent state bytes (per sequence, independent of context length)
    int64_t ssm_state_bytes() const;

    // Active parameter count per token (for FLOP estimates): 8 experts + shared expert.
    int64_t active_experts_per_token() const { return n_expert_used_ + 1; }

    std::string summary() const;

private:
    std::string              arch_;
    int64_t                  n_layer_ = 0, n_embd_ = 0, n_expert_ = 0, n_expert_used_ = 0;
    int64_t                  expert_ff_ = 0, shared_ff_ = 0, n_head_ = 0, n_head_kv_ = 0;
    int64_t                  head_dim_ = 0, n_ctx_train_ = 0, rope_dim_ = 0;
    double                   rope_base_ = 10000.0;
    float                    rms_eps_ = 1e-6f;
    std::vector<int64_t>     rope_sections_;
    int64_t                  ssm_inner_ = 0, ssm_state_ = 0, ssm_groups_ = 0;
    int64_t                  ssm_dt_rank_ = 0, ssm_conv_kernel_ = 0;
    std::vector<LayerInfo>   layers_;
    std::vector<int>         full_attn_layers_;
    std::vector<std::vector<ReadRange>> expert_ranges_;
    int64_t                  total_expert_bytes_ = 0, total_other_bytes_ = 0;
    int64_t                  embedding_bytes_ = 0, router_bytes_ = 0, shared_expert_bytes_ = 0;
    int64_t                  max_layer_expert_bytes_ = 0, min_layer_expert_bytes_ = 0;
    uint64_t                 data_offset_ = 0;
    const GGUFFile *         gguf_ = nullptr;
};

} // namespace amp
