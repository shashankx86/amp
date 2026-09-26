#include "amp/model/gguf.h"

#include "amp/bytes.h"
#include "amp/format.h"
#include "amp/log.h"

#include <cstring>
#include <set>

namespace amp {

namespace {
constexpr uint32_t kGGUFMagic = 0x46554747;  // "GGUF" little-endian

// The tokenizer blob is ~15 MiB of strings we never need for planning; skip it so
// opening a 14.65 GB model stays instant.
const std::set<std::string> kSkippedKeys = {
    "tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type",
    "tokenizer.ggml.scores", "tokenizer.ggml.token_to_id",
};
} // namespace

// ---------------------------------------------------------------- KVValue

int64_t KVValue::as_int(int64_t def) const {
    switch (type) {
        case Type::INT:   return i;
        case Type::UINT:  return (int64_t) u;
        case Type::FLOAT: return (int64_t) f;
        case Type::BOOL:  return b ? 1 : 0;
        default:          return def;
    }
}

double KVValue::as_double(double def) const {
    switch (type) {
        case Type::FLOAT: return f;
        case Type::INT:   return (double) i;
        case Type::UINT:  return (double) u;
        case Type::BOOL:  return b ? 1.0 : 0.0;
        default:          return def;
    }
}

std::string KVValue::as_string(const std::string & def) const {
    return type == Type::STRING ? s : def;
}

bool KVValue::as_bool(bool def) const {
    switch (type) {
        case Type::BOOL:  return b;
        case Type::INT:   return i != 0;
        case Type::UINT:  return u != 0;
        case Type::FLOAT: return f != 0.0;
        default:          return def;
    }
}

size_t KVValue::array_size() const {
    if (type == Type::ARRAY) {
        return !ai.empty() ? ai.size() : (!af.empty() ? af.size() : as.size());
    }
    return 0;
}

// ---------------------------------------------------------------- TensorInfo

int64_t TensorInfo::n_elements() const {
    int64_t n = 1;
    for (int i = 0; i < n_dims; i++) {
        n *= ne[i];
    }
    return n;
}

std::string TensorInfo::shape() const {
    std::string s = "[";
    for (int i = 0; i < n_dims; i++) {
        if (i) {
            s += ",";
        }
        s += std::to_string(ne[i]);
    }
    return s + "]";
}

std::string TensorInfo::type_name() const {
    return ggml_type_name(type);
}

// ---------------------------------------------------------------- GGUFFile

Result<GGUFFile> GGUFFile::open(const std::string & path) {
    auto map_res = MappedFile::open_read(path);
    if (!map_res.ok()) {
        return map_res.status();
    }
    GGUFFile f;
    f.map_ = map_res.take();
    f.path_ = path;
    const Status st = f.parse_header();
    if (!st.ok()) {
        return st;
    }
    return f;
}

Status GGUFFile::parse_header() {
    const uint8_t * p   = map_.data();
    const size_t   end = map_.size();
    size_t         off = 0;

    auto need = [&](size_t n) { return off + n <= end; };
    auto rd_u32 = [&]() -> uint32_t {
        uint32_t v;
        memcpy(&v, p + off, 4);
        off += 4;
        return v;
    };
    auto rd_u64 = [&]() -> uint64_t {
        uint64_t v;
        memcpy(&v, p + off, 8);
        off += 8;
        return v;
    };
    auto rd_str = [&]() -> std::string {
        const uint64_t n = rd_u64();
        if (!need(n)) {
            return std::string();
        }
        std::string s((const char *) (p + off), (size_t) n);
        off += (size_t) n;
        return s;
    };

    if (!need(8)) {
        return Status::Error("file too small to be GGUF");
    }
    if (rd_u32() != kGGUFMagic) {
        return Status::Errorf("'%s' is not a GGUF file (bad magic)", path_.c_str());
    }
    const uint32_t version = rd_u32();
    if (version < 2 || version > 3) {
        return Status::Errorf("unsupported GGUF version %u", version);
    }
    const uint64_t n_tensors = rd_u64();
    const uint64_t n_kv      = rd_u64();
    if (!need(n_tensors * 24 + n_kv * 24)) {
        return Status::Error("implausible GGUF header (truncated?)");
    }

    // ---- key/value metadata ----
    for (uint64_t i = 0; i < n_kv; i++) {
        const std::string key = rd_str();
        const uint32_t    type = rd_u32();

        KVValue v;
        v.type = KVValue::Type::UNKNOWN;
        std::vector<KVValue> array_elems;
        const bool          keep = kSkippedKeys.find(key) == kSkippedKeys.end();

        switch (type) {
            case 0: { if (!need(1)) return Status::Error("truncated bool");  uint8_t  x; memcpy(&x, p+off,1); off+=1; v.type=KVValue::Type::BOOL; v.b = x!=0; if(keep) v.i=v.b; break; }
            case 1: { if (!need(1)) return Status::Error("truncated u8");   uint8_t  x; memcpy(&x, p+off,1); off+=1; v.type=KVValue::Type::UINT; v.u=x; v.i=x; break; }
            case 2: { if (!need(2)) return Status::Error("truncated i16");  int16_t  x; memcpy(&x, p+off,2); off+=2; v.type=KVValue::Type::INT; v.i=x; break; }
            case 3: { if (!need(2)) return Status::Error("truncated u16");  uint16_t x; memcpy(&x, p+off,2); off+=2; v.type=KVValue::Type::UINT; v.u=x; v.i=x; break; }
            case 4: { if (!need(4)) return Status::Error("truncated i32");  int32_t  x; memcpy(&x, p+off,4); off+=4; v.type=KVValue::Type::INT; v.i=x; break; }
            case 5: { if (!need(4)) return Status::Error("truncated u32");  uint32_t x; memcpy(&x, p+off,4); off+=4; v.type=KVValue::Type::UINT; v.u=x; v.i=x; break; }
            case 6: { if (!need(4)) return Status::Error("truncated f32");  float    x; memcpy(&x, p+off,4); off+=4; v.type=KVValue::Type::FLOAT; v.f=x; break; }
            case 7: { if (!need(1)) return Status::Error("truncated bool1"); uint8_t  x; memcpy(&x, p+off,1); off+=1; v.type=KVValue::Type::BOOL; v.b=x!=0; break; }
            case 8: {
                const std::string s = rd_str();
                v.type = KVValue::Type::STRING;
                v.s    = s;
                break;
            }
            case 9: {
                const uint32_t et = rd_u32();
                const uint64_t n  = rd_u64();
                v.type = KVValue::Type::ARRAY;
                for (uint64_t j = 0; j < n; j++) {
                    KVValue ev;
                    switch (et) {
                        case 1: { uint8_t  x; memcpy(&x,p+off,1); off+=1; ev.type=KVValue::Type::UINT; ev.u=x; break; }
                        case 2: { int16_t  x; memcpy(&x,p+off,2); off+=2; ev.type=KVValue::Type::INT;  ev.i=x; break; }
                        case 3: { uint16_t x; memcpy(&x,p+off,2); off+=2; ev.type=KVValue::Type::UINT; ev.u=x; break; }
                        case 4: { int32_t  x; memcpy(&x,p+off,4); off+=4; ev.type=KVValue::Type::INT;  ev.i=x; break; }
                        case 5: { uint32_t x; memcpy(&x,p+off,4); off+=4; ev.type=KVValue::Type::UINT; ev.u=x; break; }
                        case 6: { float    x; memcpy(&x,p+off,4); off+=4; ev.type=KVValue::Type::FLOAT; ev.f=x; break; }
                        case 7: { uint8_t  x; memcpy(&x,p+off,1); off+=1; ev.type=KVValue::Type::BOOL;  ev.b=x!=0; ev.i=ev.b?1:0; break; }
                        case 8: {
                            const uint64_t sl = rd_u64();
                            if (!need(sl)) {
                                return Status::Error("truncated string array");
                            }
                            ev.type = KVValue::Type::STRING;
                            ev.s    = std::string((const char *) (p + off), (size_t) sl);
                            off += (size_t) sl;
                            break;
                        }
                        default:
                            return Status::Errorf("unsupported array element type %u in key '%s'", et,
                                                  key.c_str());
                    }
                    if (ev.type == KVValue::Type::STRING) {
                        if (keep) {
                            v.as.push_back(std::move(ev.s));
                        }
                    } else if (ev.type == KVValue::Type::UINT) {
                        if (keep) {
                            v.ai.push_back((int64_t) ev.u);
                        }
                    } else {
                        if (keep) {
                            v.ai.push_back(ev.type == KVValue::Type::FLOAT ? (int64_t) ev.f : ev.i);
                        }
                    }
                }
                break;
            }
            case 10: { if (!need(8)) return Status::Error("truncated i64"); int64_t  x; memcpy(&x,p+off,8); off+=8; v.type=KVValue::Type::INT;  v.i=x; break; }
            case 11: { if (!need(8)) return Status::Error("truncated u64"); uint64_t x; memcpy(&x,p+off,8); off+=8; v.type=KVValue::Type::UINT; v.u=x; v.i=(int64_t)x; break; }
            case 12: { if (!need(8)) return Status::Error("truncated f64"); double   x; memcpy(&x,p+off,8); off+=8; v.type=KVValue::Type::FLOAT; v.f=x; break; }
            default:
                return Status::Errorf("unsupported KV type %u for key '%s'", type, key.c_str());
        }
        kv_[key] = std::move(v);
    }

    if (const auto it = kv_.find("general.architecture"); it != kv_.end()) {
        arch_ = it->second.as_string("llama");
    }
    alignment_ = 32;
    if (const auto it = kv_.find("general.alignment"); it != kv_.end()) {
        alignment_ = (uint64_t) it->second.as_int(32);
    }

    // ---- tensor infos ----
    tensors_.reserve((size_t) n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        TensorInfo t;
        t.name           = rd_str();
        const uint32_t nd = rd_u32();
        if (nd == 0 || nd > GGML_MAX_DIMS) {
            return Status::Errorf("tensor '%s' has %u dims", t.name.c_str(), nd);
        }
        t.n_dims = (int) nd;
        for (uint32_t d = 0; d < nd; d++) {
            t.ne[d] = (int64_t) rd_u64();
        }
        t.type   = (ggml_type) rd_u32();
        t.offset = rd_u64();

        const int64_t nel = t.n_elements();
        const int64_t bl  = ggml_blck_size(t.type);
        const int64_t tsz = ggml_type_size(t.type);
        if (bl <= 0 || tsz <= 0) {
            return Status::Errorf("tensor '%s' has invalid type %d", t.name.c_str(), (int) t.type);
        }
        t.nbytes = (uint64_t) ((nel / bl) * tsz);
        tensors_.push_back(std::move(t));
    }

    data_offset_ = align_up((uint64_t) off, alignment_);
    AMP_DEBUG("amp: parsed ", path_, ": arch=", arch_, " tensors=", tensors_.size(),
              " kv=", kv_.size(), " data_offset=", human_bytes(data_offset_));
    return Status::OK();
}

const TensorInfo * GGUFFile::find(const std::string & name) const {
    for (const auto & t : tensors_) {
        if (t.name == name) {
            return &t;
        }
    }
    return nullptr;
}

const KVValue * GGUFFile::kv(const std::string & suffix) const {
    const std::string key = arch_ + "." + suffix;
    const auto        it  = kv_.find(key);
    return it == kv_.end() ? nullptr : &it->second;
}

int64_t GGUFFile::n_layer() const { const auto * v = kv("block_count"); return v ? v->as_int() : 0; }
int64_t GGUFFile::n_embd() const { const auto * v = kv("embedding_length"); return v ? v->as_int() : 0; }
int64_t GGUFFile::n_expert() const { const auto * v = kv("expert_count"); return v ? v->as_int() : 0; }
int64_t GGUFFile::n_expert_used() const { const auto * v = kv("expert_used_count"); return v ? v->as_int() : 0; }
int64_t GGUFFile::expert_ff() const { const auto * v = kv("expert_feed_forward_length"); return v ? v->as_int() : 0; }
int64_t GGUFFile::shared_ff() const { const auto * v = kv("expert_shared_feed_forward_length"); return v ? v->as_int() : 0; }
int64_t GGUFFile::n_head() const { const auto * v = kv("attention.head_count"); return v ? v->as_int() : 0; }
int64_t GGUFFile::n_head_kv() const { const auto * v = kv("attention.head_count_kv"); return v ? v->as_int(1) : 1; }
int64_t GGUFFile::head_dim() const {
    if (const auto * v = kv("attention.key_length")) {
        return v->as_int();
    }
    const int64_t h = n_head();
    return h ? n_embd() / h : 0;
}
int64_t GGUFFile::full_attn_interval() const {
    const auto * v = kv("full_attention_interval");  // key is <arch>.full_attention_interval
    return v ? v->as_int(1) : 1;
}
int64_t GGUFFile::n_ctx_train() const { const auto * v = kv("context_length"); return v ? v->as_int() : 0; }
int64_t GGUFFile::ssm_inner() const { const auto * v = kv("ssm.inner_size"); return v ? v->as_int() : 0; }
int64_t GGUFFile::ssm_state() const { const auto * v = kv("ssm.state_size"); return v ? v->as_int() : 0; }
int64_t GGUFFile::ssm_groups() const { const auto * v = kv("ssm.group_count"); return v ? v->as_int() : 0; }
int64_t GGUFFile::ssm_dt_rank() const { const auto * v = kv("ssm.time_step_rank"); return v ? v->as_int() : 0; }
int64_t GGUFFile::ssm_conv_kernel() const { const auto * v = kv("ssm.conv_kernel"); return v ? v->as_int() : 0; }
int64_t GGUFFile::rope_dim() const { const auto * v = kv("rope.dimension_count"); return v ? v->as_int() : 0; }
double  GGUFFile::rope_base() const { const auto * v = kv("rope.freq_base"); return v ? v->as_double(10000.0) : 10000.0; }

float GGUFFile::rms_eps() const {
    const auto * v = kv("attention.layer_norm_rms_epsilon");
    return v ? (float) v->as_double(1e-6) : 1e-6f;
}

std::vector<int64_t> GGUFFile::rope_sections() const {
    const auto * v = kv("rope.dimension_sections");
    return v ? v->ai : std::vector<int64_t>();
}

std::vector<int64_t> GGUFFile::recurrent_layers() const {
    const auto * v = kv("attention.recurrent_layers");
    return v ? v->ai : std::vector<int64_t>();
}

bool GGUFFile::layer_is_recurrent(int il) const {
    const std::vector<int64_t> r = recurrent_layers();
    if ((size_t) il < r.size()) {
        return r[(size_t) il] != 0;
    }
    // Fall back to the interval rule: attention every full_attention_interval layers.
    const int64_t interval = full_attn_interval();
    return interval <= 0 ? false : ((il % interval) != (interval - 1));
}

std::vector<int> GGUFFile::full_attention_layers() const {
    std::vector<int> out;
    const int64_t    n = n_layer();
    for (int64_t i = 0; i < n; i++) {
        if (!layer_is_recurrent((int) i)) {
            out.push_back((int) i);
        }
    }
    return out;
}

std::string GGUFFile::chat_template() const {
    const auto it = kv_.find("tokenizer.chat_template");
    return it == kv_.end() ? std::string() : it->second.as_string();
}

} // namespace amp
