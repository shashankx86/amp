// Small text helpers shared by the API layer. Kept free of llama.cpp types so they can be tested
// without a model.
#pragma once

#include <string>

namespace amp {

// Normalises a client-supplied reasoning field so it round-trips through a chat template.
//
// The template owns the thinking tags: it writes `<|im_start|>assistant\n<think>\n{reasoning}\n
// </think>` around the reasoning field. So the value that must be stored is the *inner* text. But
// clients differ on what they send back: a server that reports reasoning_content without the tags
// produces a client that echoes the inner text (correct), while a client that echoes a raw
// `<think>...</think>` block from elsewhere would get the tags applied twice - and then the
// re-rendered prompt no longer matches the cached one, so every turn silently loses its prefix.
//
// Being liberal here costs nothing and removes that whole failure mode. Only the exact leading and
// trailing tags are removed, and only when they wrap the whole string, so prose that merely mentions
// a tag is left alone.
std::string normalize_reasoning(const std::string & reasoning,
                                const std::string & start_tag = "<think>",
                                const std::string & end_tag   = "</think>");

} // namespace amp
