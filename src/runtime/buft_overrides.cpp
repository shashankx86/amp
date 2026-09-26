#include "amp/runtime/buft_overrides.h"

#include "amp/log.h"

#include "ggml-backend.h"

#include <algorithm>

namespace amp {

void ExpertCpuOverrides::build(const ModelGeometry & geo, int32_t gpu_expert_layers) {
    const int64_t n_cpu = std::max<int64_t>(0, geo.n_layer() - gpu_expert_layers);
    if (n_cpu <= 0) {
        return;
    }
    const size_t count = (size_t) n_cpu * 3;
    patterns_.reserve(count);
    bufts_.reserve(count);
    entries_.reserve(count + 1);

    const ggml_backend_buffer_type_t cpu = ggml_backend_cpu_buffer_type();
    static const char * suffixes[] = { "ffn_gate_exps.weight", "ffn_up_exps.weight",
                                       "ffn_down_exps.weight" };
    for (int64_t il = 0; il < n_cpu; il++) {
        for (const char * suffix : suffixes) {
            patterns_.push_back("blk." + std::to_string(il) + "." + suffix);
            bufts_.push_back(cpu);
            llama_model_tensor_buft_override ov{};
            ov.pattern = patterns_.back().c_str();
            ov.buft    = cpu;
            entries_.push_back(ov);
        }
    }
    // Sentinel: llama.cpp iterates the array until pattern == nullptr. Without this it walks off the
    // end and the process dies inside the tensor loader - which is exactly what happened before this
    // helper existed.
    entries_.push_back(llama_model_tensor_buft_override{ nullptr, nullptr });

    AMP_DEBUG("amp: pinned ", size(), " expert tensors to the CPU (layers 0..", n_cpu - 1, ")");
}

} // namespace amp
