#!/usr/bin/env bash
# Convenience wrapper: configure, build, run tests.
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=${AMP_BUILD_TYPE:-Release}
if [ ! -f build/CMakeCache.txt ]; then
  cmake -B build -DCMAKE_BUILD_TYPE="$CFG" -DAMP_LLAMA_ROOT=../llama.cpp
fi
cmake --build build -j"$(nproc)" "$@"
ctest --test-dir build --output-on-failure
