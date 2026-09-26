// Buffer-type overrides: "keep the MoE experts of these layers on the CPU".
//
// This is the public-API equivalent of llama.cpp's -ncmoe / -ot. It lives in one place on purpose:
// getting it wrong is a segfault, because llama.cpp walks the override array until it finds an entry
// with a null pattern (src/llama-model-loader.cpp). Both the server and the CLI need it, so neither
// gets to hand-roll the array.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "amp/model/geometry.h"
#include "llama.h"

namespace amp {

class ExpertCpuOverrides {
public:
    // Pins ffn_{gate,up,down}_exps.weight of layers [0, n_layer - gpu_expert_layers) to the CPU.
    // Everything else is left for n_gpu_layers to place.
    void build(const ModelGeometry & geo, int32_t gpu_expert_layers);

    // nullptr when there is nothing to override, so it is always safe to assign.
    const llama_model_tensor_buft_override * data() const {
        return entries_.empty() ? nullptr : entries_.data();
    }
    size_t size() const { return entries_.empty() ? 0 : entries_.size() - 1; }  // minus the sentinel
    bool   empty() const { return size() == 0; }

private:
    // The patterns must outlive the llama_load_model_from_file call, and the strings must be
    // reserved up front so no reallocation can invalidate the c_str() pointers we hand out.
    std::vector<std::string>                     patterns_;
    std::vector<ggml_backend_buffer_type_t>      bufts_;
    std::vector<llama_model_tensor_buft_override> entries_;   // null-terminated
};

} // namespace amp
