#!/usr/bin/env bash
# Convenience wrapper: fetch pinned deps, configure, build, run tests.
#
# The first invocation compiles ggml's CUDA kernels from the vendored llama.cpp, so expect it to take
# a while (order 15 minutes). After that this is seconds.
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=${AMP_BUILD_TYPE:-Release}

./scripts/fetch_deps.sh
cmake -B build -DCMAKE_BUILD_TYPE="$CFG"
cmake --build build -j"$(nproc)" "$@"
ctest --test-dir build --output-on-failure
