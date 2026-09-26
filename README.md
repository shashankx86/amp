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
| M3 forward path (CPU) | next |
| M4 GPU offload | planned |
| M5 async expert prefetch | planned |
| M6 server | planned |
| M7 decode optimization | planned |

## Tools

```bash
# what the model is, what the box can do, and which configuration is predicted fastest
./build/amp-plan --model ../models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf

# make the hot bytes actually resident, and measure whether it worked
./build/amp-warm --model ../models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf \
                 --what experts --drop-cache --verify
```

Measurements and open questions: `docs/BENCH.md`. Design: `docs/ARCHITECTURE.md`.
Working rules and hard constraints: `AGENT.md`.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DAMP_LLAMA_ROOT=../llama.cpp
cmake --build build -j$(nproc)
```

Requires the local llama.cpp build (`../llama.cpp/build/bin/libggml*.so`) — amp links ggml so the
quantized math is literally the same code llama.cpp runs.
