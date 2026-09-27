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
TARGETS=(amp-server amp-infer amp-plan amp-warm amp-kernel-bound amp_tests)

./scripts/fetch_deps.sh
cmake -B build -DCMAKE_BUILD_TYPE="$CFG"
cmake --build build -j"$(nproc)" --target "${TARGETS[@]}" "$@"

# The preflight tests need a real GGUF to plan against, and they SKIP SILENTLY without one
# (model_path() is getenv("AMP_TEST_MODEL"), and every test does `if (!m) return;`). That meant
# ctest reported "100% tests passed" while 7 of the tests did nothing at all, which is how a stale
# cache_ram_mib assertion survived a behaviour change. Point them at the model if there is one, and
# fail loudly if there is not, so a green ctest can be trusted again.
MODEL="${AMP_TEST_MODEL:-/home/e0u/localhost/models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf}"
if [ -f "$MODEL" ]; then
    AMP_TEST_MODEL="$MODEL" ctest --test-dir build --output-on-failure
else
    echo "build.sh: no model at $MODEL, so the preflight tests would skip silently." >&2
    echo "          Set AMP_TEST_MODEL to a GGUF and re-run, or the suite proves nothing." >&2
    exit 1
fi
