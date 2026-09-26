// GGUF v3 header reader.
//
// Reads metadata + tensor infos only: never pulls tensor payloads into RAM. Byte sizes
// come from ggml (ggml_type_size/ggml_blck_size) so amp's view of the file is exactly
// llama.cpp's view.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "amp/io/mapped_file.h"
#include "amp/status.h"
#include "ggml.h"

namespace amp {

using KVArrayValue = std::variant<int64_t, double, std::string>;

struct KVValue {
    enum class Type { BOOL, INT, UINT, FLOAT, STRING, ARRAY, UNKNOWN };

    Type                     type = Type::UNKNOWN;
    bool                     b    = false;
    int64_t                  i    = 0;
    uint64_t                 u    = 0;
    double                   f    = 0.0;
    std::string              s;
    std::vector<int64_t>     ai;
    std::vector<double>      af;
    std::vector<std::string> as;

    int64_t     as_int(int64_t def = 0) const;
    double      as_double(double def = 0.0) const;
    std::string as_string(const std::string & def = std::string()) const;
    bool        as_bool(bool def = false) const;
    size_t      array_size() const;
};

struct TensorInfo {
    std::string name;
    ggml_type   type   = GGML_TYPE_F32;
    int64_t     ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
    int         n_dims = 0;
    uint64_t    offset = 0;  // relative to the tensor data section
    uint64_t    nbytes = 0;  // ggml_type_size * n_elements / ggml_blck_size

    int64_t     n_elements() const;
    std::string shape() const;
    std::string type_name() const;
};

class GGUFFile {
public:
    static Result<GGUFFile> open(const std::string & path);

    const MappedFile &           file() const { return map_; }
    const std::string &          path() const { return path_; }
    const std::vector<TensorInfo> & tensors() const { return tensors_; }
    const std::map<std::string, KVValue> & all_kv() const { return kv_; }
    const std::string &          architecture() const { return arch_; }
    uint64_t                     alignment() const { return alignment_; }
    uint64_t                     data_offset() const { return data_offset_; }
    uint64_t                     file_size() const { return map_.size(); }

    const TensorInfo * find(const std::string & name) const;
    const KVValue *    kv(const std::string & suffix) const;  // arch-prefixed, e.g. "block_count"

    uint64_t tensor_file_offset(const TensorInfo & t) const { return data_offset_ + t.offset; }

    // ---- convenience accessors (all arch-prefixed) ----
    int64_t n_layer() const;
    int64_t n_embd() const;
    int64_t n_expert() const;
    int64_t n_expert_used() const;
    int64_t expert_ff() const;
    int64_t shared_ff() const;
    int64_t n_head() const;
    int64_t n_head_kv() const;
    int64_t head_dim() const;
    int64_t full_attn_interval() const;
    int64_t n_ctx_train() const;
    int64_t ssm_inner() const;
    int64_t ssm_state() const;
    int64_t ssm_groups() const;
    int64_t ssm_dt_rank() const;
    int64_t ssm_conv_kernel() const;
    int64_t rope_dim() const;
    double  rope_base() const;
    float   rms_eps() const;
    std::vector<int64_t> rope_sections() const;

    std::vector<int64_t> recurrent_layers() const;
    // true when layer il is a recurrent (SSM) layer
    bool layer_is_recurrent(int il) const;
    // layers with full attention, ascending
    std::vector<int> full_attention_layers() const;

    std::string chat_template() const;

private:
    Status parse_header();

    MappedFile                        map_;
    std::string                       path_;
    std::string                       arch_ = "llama";
    std::vector<TensorInfo>           tensors_;
    std::map<std::string, KVValue>    kv_;
    uint64_t                          alignment_   = 32;
    uint64_t                          data_offset_ = 0;
};

} // namespace amp
