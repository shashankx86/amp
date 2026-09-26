// Tests for the GGUF reader and the derived geometry.
// These assert the *measured* facts about the real model, so a regression in the parser
// (or a model swap) shows up immediately.
#include "test_harness.h"

#include "amp/bytes.h"
#include "amp/format.h"
#include "amp/model/geometry.h"
#include "amp/model/gguf.h"

#include <cstdlib>

using namespace amp;

namespace {
const char * model_path() {
    const char * env = getenv("AMP_TEST_MODEL");
    if (env) {
        return env;
    }
    return "/home/e0u/localhost/models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf";
}
} // namespace

AMP_TEST(gguf_rejects_garbage) {
    const Status st = MappedFile::open_read("/nonexistent/definitely-not-here.gguf").status();
    AMP_CHECK_MSG(!st.ok(), st.message());
}

AMP_TEST(gguf_parses_the_model_header) {
    auto res = GGUFFile::open(model_path());
    if (!res.ok()) {
        printf("(skipped: %s) ", res.message().c_str());
        return;
    }
    const GGUFFile & g = *res;

    AMP_CHECK_EQ(g.architecture(), std::string("qwen35moe"));
    AMP_CHECK_EQ(g.n_layer(), 40);
    AMP_CHECK_EQ(g.n_embd(), 2048);
    AMP_CHECK_EQ(g.n_expert(), 256);
    AMP_CHECK_EQ(g.n_expert_used(), 8);
    AMP_CHECK_EQ(g.expert_ff(), 512);
    AMP_CHECK_EQ(g.shared_ff(), 512);
    AMP_CHECK_EQ(g.n_head(), 16);
    AMP_CHECK_EQ(g.n_head_kv(), 2);
    AMP_CHECK_EQ(g.head_dim(), 256);
    AMP_CHECK_EQ(g.n_ctx_train(), 262144);
    AMP_CHECK_EQ(g.full_attn_interval(), 4);
    AMP_CHECK_EQ(g.tensors().size(), 733u);
    AMP_CHECK_EQ(g.rope_dim(), 64);
    AMP_CHECK_NEAR(g.rope_base(), 1e7, 1.0);
    AMP_CHECK_EQ(g.ssm_inner(), 4096);
    AMP_CHECK_EQ(g.ssm_state(), 128);
    AMP_CHECK_EQ(g.ssm_conv_kernel(), 4);

    // hybrid: full attention every 4th layer -> 10 attention layers, 30 recurrent
    AMP_CHECK_EQ(g.full_attention_layers().size(), 10u);
    AMP_CHECK_EQ(g.full_attention_layers()[0], 3);
    AMP_CHECK_EQ(g.full_attention_layers()[9], 39);

    // the chat template is required for the server
    AMP_CHECK_MSG(g.chat_template().size() > 1000, "chat template missing");
}

AMP_TEST(gguf_tensor_sizes_match_ggml) {
    auto res = GGUFFile::open(model_path());
    if (!res.ok()) {
        return;
    }
    const GGUFFile & g = *res;

    const TensorInfo * tok = g.find("token_embd.weight");
    AMP_CHECK(tok != nullptr);
    if (tok) {
        AMP_CHECK_EQ(tok->n_elements(), 2048ll * 248320ll);
        AMP_CHECK_EQ(ggml_type_name(tok->type), std::string(ggml_type_name(GGML_TYPE_Q3_K)));
        // Q3_K = 110 bytes / 256 elements
        AMP_CHECK_EQ(tok->nbytes, (uint64_t) ((tok->n_elements() / 256) * 110));
    }
    const TensorInfo * out = g.find("output.weight");
    AMP_CHECK(out != nullptr);
    if (out) {
        AMP_CHECK_EQ(ggml_type_name(out->type), std::string(ggml_type_name(GGML_TYPE_Q6_K)));
    }
    // no MTP / nextn tensors at all: this model has no multi-token-prediction head
    for (const auto & t : g.tensors()) {
        AMP_CHECK_MSG(t.name.find("nextn") == std::string::npos, "unexpected nextn tensor: " + t.name);
        AMP_CHECK_MSG(t.name.find("mtp") == std::string::npos, "unexpected mtp tensor: " + t.name);
    }
}

AMP_TEST(geometry_byte_accounting) {
    auto gguf_res = GGUFFile::open(model_path());
    if (!gguf_res.ok()) {
        return;
    }
    auto geo_res = ModelGeometry::build(*gguf_res);
    AMP_CHECK(geo_res.ok());
    const ModelGeometry & geo = **geo_res;

    // measured: 12.188 GiB of experts, 1.454 GiB of everything else
    AMP_CHECK_NEAR((double) geo.total_expert_bytes(), 12.188 * (double) kGiB, 0.05 * (double) kGiB);
    AMP_CHECK_NEAR((double) geo.total_other_bytes(), 1.454 * (double) kGiB, 0.05 * (double) kGiB);

    // per-layer expert bytes: 330 MiB (Q3_K, layers 0-9 and 30-39), 294 MiB (IQ3_XXS, 10-29)
    AMP_CHECK_NEAR((double) geo.layers()[0].expert_bytes, 330.0 * (double) kMiB, 1.0 * (double) kMiB);
    AMP_CHECK_NEAR((double) geo.layers()[15].expert_bytes, 294.0 * (double) kMiB, 1.0 * (double) kMiB);
    AMP_CHECK_NEAR((double) geo.max_layer_expert_bytes(), 330.0 * (double) kMiB, 1.0 * (double) kMiB);
    AMP_CHECK_NEAR((double) geo.min_layer_expert_bytes(), 294.0 * (double) kMiB, 1.0 * (double) kMiB);

    // every CPU-resident expert layer costs 294-330 MiB: this is why VRAM buys so little
    AMP_CHECK_EQ(geo.layers().size(), 40u);
    for (const auto & li : geo.layers()) {
        AMP_CHECK(li.expert_bytes > 0);
        AMP_CHECK_EQ(geo.expert_ranges(li.index).size(), 3u);  // gate, up, down
    }

    // bytes per expert per layer: ~1.29 MiB (3 x ~440 KiB)
    const int64_t per_expert = geo.bytes_per_expert(0);
    AMP_CHECK(per_expert > 1200 * 1024 && per_expert < 1400 * 1024);

    // routers are F32 and must never be quantised away
    AMP_CHECK_NEAR((double) geo.router_bytes(), 80.0 * (double) kMiB, 2.0 * (double) kMiB);
    AMP_CHECK_NEAR((double) geo.shared_expert_bytes(), 82.8 * (double) kMiB, 2.0 * (double) kMiB);
}

AMP_TEST(geometry_kv_cache_math) {
    auto gguf_res = GGUFFile::open(model_path());
    if (!gguf_res.ok()) {
        return;
    }
    auto geo_res = ModelGeometry::build(*gguf_res);
    const ModelGeometry & geo = **geo_res;

    // per attention layer and token: 2 kv heads x 256 dim = 512 elements of K and 512 of V
    // f32: 10 layers x 1024 elements x 4 B = 40960 B/token
    AMP_CHECK_EQ(geo.kv_bytes_per_token(CacheType::kF32, CacheType::kF32), 40960ll);

    // q8_0 K (34 B / 32 elem) + q4_0 V (18 B / 32 elem) -> fractional bytes/element
    const int64_t per_tok = geo.kv_bytes_per_token(CacheType::kQ8_0, CacheType::kQ4_0);
    AMP_CHECK_EQ(per_tok, 8320ll);  // 10 x 512 x (1.0625 + 0.5625)

    // 1.55 GiB at 200k. Cross-checked against the measured VRAM base: 3473 MiB total
    // = 1489 MiB other weights + ~1534 MiB KV + ~450 MiB compute buffer at ubatch 512.
    const int64_t kv200k = geo.kv_bytes(200000, CacheType::kQ8_0, CacheType::kQ4_0);
    AMP_CHECK_MSG(kv200k > 1.50 * (int64_t) kGiB && kv200k < 1.60 * (int64_t) kGiB,
                  format("kv@200k = %s", human_bytes((uint64_t) kv200k).c_str()));

    // This identity is what the VRAM planner depends on. Measured on this box:
    // 3473 MiB base = 1489 MiB weights + 1587 MiB KV@200k + ~450 MiB compute buffer @ub512.
    const int64_t weights_kv = (int64_t) geo.total_other_bytes() + kv200k;
    AMP_CHECK_MSG(weights_kv > 2950 * (int64_t) kMiB && weights_kv < 3100 * (int64_t) kMiB,
                  format("weights+kv = %s", human_bytes((uint64_t) weights_kv).c_str()));
    const int64_t with_compute = weights_kv + 450 * (int64_t) kMiB;
    AMP_CHECK_MSG(with_compute > 3400 * (int64_t) kMiB && with_compute < 3550 * (int64_t) kMiB,
                  format("weights+kv+compute = %s vs measured 3473 MiB",
                         human_bytes((uint64_t) with_compute).c_str()));
}

AMP_TEST(geometry_expert_ranges_are_file_ordered) {
    auto gguf_res = GGUFFile::open(model_path());
    if (!gguf_res.ok()) {
        return;
    }
    auto geo_res = ModelGeometry::build(*gguf_res);
    const ModelGeometry & geo = **geo_res;

    const uint64_t file_limit = gguf_res->file_size();
    uint64_t        sum       = 0;
    for (int il = 0; il < geo.n_layer(); il++) {
        const auto ranges = geo.expert_ranges(il);
        AMP_CHECK_EQ(ranges.size(), 3u);
        for (size_t i = 1; i < ranges.size(); i++) {
            AMP_CHECK(ranges[i - 1].offset < ranges[i].offset);
        }
        for (const auto & r : ranges) {
            AMP_CHECK(r.length > 0);
            AMP_CHECK_MSG(r.end() <= file_limit, "expert range outside the file");
            sum += r.length;
        }
    }
    AMP_CHECK_EQ(sum, (uint64_t) geo.total_expert_bytes());
}
