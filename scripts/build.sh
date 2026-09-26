#!/usr/bin/env bash
# Convenience wrapper: fetch pinned deps, configure, build, run tests.
#
# The first invocation compiles ggml's CUDA kernels from the vendored llama.cpp, so expect it to take
# a while (order 15 minutes). After that this is seconds.
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=${AMP_BUILD_TYPE:-Release}

# LLAMA_BUILD_TOOLS has to be ON to get tools/{ui,server,mtmd} defined (llama.cpp's root CMakeLists
# gates the whole tools/ tree on it, not on LLAMA_BUILD_SERVER). That also *defines* every other
# llama.cpp tool target -- llama-bench, llama-cli, llama-perplexity, llama-quantize and friends --
# so we must name our targets rather than build `all`, or a plain build compiles all of them.
TARGETS=(amp-server amp-infer amp-plan amp-warm amp_tests)

./scripts/fetch_deps.sh
cmake -B build -DCMAKE_BUILD_TYPE="$CFG"
cmake --build build -j"$(nproc)" --target "${TARGETS[@]}" "$@"
ctest --test-dir build --output-on-failure
