# amp

A purpose-built inference engine for one model: **Occamy-1.0 APEX-I-MiniPlus-V2.1-Abliterated**
(`qwen35moe`, 40 blocks, hybrid SSM+attention, 256 experts / 8 used, 262144 native context).

Target machine: RTX 4050 Laptop 6 GB + Ryzen 7 7735HS (8C/16T, AVX2) + 14 GiB RAM + NVMe.

**Design goal: beat llama.cpp here with zero quality loss.** See `AGENT.md` for the rules and
`../NOTES.md` for the measurements that motivate every decision.

## Why this model needs its own engine

It is a 12.19 GiB MoE expert working set (256 experts/layer, 8 active) that has to be cyclically
re-read for every ubatch, competing against a ~11.5 GiB page cache on a 6 GB GPU. The arithmetic is
cheap (the model runs at ~240 t/s prefill when the weights are resident); the *memory system* is the
product. Everything amp does is about residency, I/O-compute overlap, and scheduling — never about
changing the numbers.

## Status

| Milestone | State |
|---|---|
| M0 scaffold (modules, build, tests) | done |
| M1 GGUF reader + geometry | done |
| M2 page-cache warmer | done — measured 1.92 GiB/s cold, whole 12.19 GiB set in 6.3 s |
| M2b memory planner + cost model | done — predicts 225 t/s prefill / 12.4 t/s decode at 200k |
| M3 forward path + prefetcher | done — parity harness; see BENCH.md for measured rates |
| M3b quality parity | done — bit-deterministic, no algorithmic difference |
| M4 GPU offload of dense + N expert layers | done — expressed through `tensor_buft_overrides` |
| M5 page-cache warm at start-up | done — 9.6 GiB in ~4 s, 2-3 GiB/s |
| M6 server (OpenAI API, prefix cache, checkpoints) | done — `amp-server`, 35 t/s decode at 200k |
| M6b concurrency + streaming correctness | done — one-worker task queue, always-finished streams |
| M3c own graph (observable router) | next — needed to prefetch the 8 active experts per layer |
| M7 decode optimization | planned |

## Thinking models

This is a reasoning model: the chat template's generation prompt ends *inside* a `<think>` block, so
the model reasons before answering and never emits the opening tag. amp therefore

- reads the client's reasoning from `reasoning_content`, `reasoning_text` **and** `reasoning` — the
  alias llama.cpp ignores, which makes OpenCode re-evaluate whole tool-call turns;
- separates reasoning from content with tags matched whitespace-tolerantly, in responses and streams;
- offers llama.cpp's reasoning budget (`--reasoning-budget`, `reasoning_budget_tokens`), off by
  default because this quant tends to loop after a forced close.

What the model's template supports, and what amp does not implement yet, is listed in `RUNNING.md`.

## Tools

```bash
# what the model is, what the box can do, and which configuration is predicted fastest
./build/bin/amp-plan --model ../models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf

# head-to-head against llama-server on the same prompt, plus a logit-level quality diff
./scripts/head2head.sh /tmp/opencode/amp_bench_prompt.txt 128
./scripts/parity.py

# make the hot bytes actually resident, and measure whether it worked
./build/bin/amp-warm --model ../models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf \
                 --what experts --drop-cache --verify
```

Measurements and open questions: `docs/BENCH.md`. Design: `docs/ARCHITECTURE.md`.
Working rules and hard constraints: `AGENT.md`.

## Build and run

```bash
./scripts/fetch_deps.sh          # vendored llama.cpp at the pinned commit
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

M=/home/e0u/localhost/models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf
./build/bin/amp-plan  --model $M                    # what the planner decided, and why
./build/bin/amp-warm  --model $M --what plan        # pull the expert set into the page cache
./build/bin/amp-infer --model $M --prompt "hi" -n 128
./scripts/head2head.sh                          # amp vs llama-server on the same prompt
./scripts/parity.py                             # quality parity, three ways
```

**amp is not a server yet** — for chat today keep using the `llama-server` command in `../RUN.md`.
Full instructions: [`RUNNING.md`](RUNNING.md).

Requires the local llama.cpp build (`../llama.cpp/build/bin/libllama*.so`); its location is baked
into the binaries with RPATH, so no `LD_LIBRARY_PATH` is needed. amp links llama.cpp so the
quantized math is literally the same code llama.cpp runs.
