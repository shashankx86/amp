# AGENT.md — working rules for `amp`

Read this first, every session. It is written to survive compaction.

## What amp is

A purpose-built inference engine for **exactly one model**:
`/home/e0u/localhost/models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf`
(arch `qwen35moe`, 40 blocks, hybrid SSM+attention, 256 experts / 8 used, native ctx 262144).

**Goal: beat llama.cpp's speed on this machine with ZERO quality loss.**
Zero quality loss is a hard constraint, not a preference. Concretely that means:
bit-identical quantized math (same dequant + same dot products as ggml), same RMSNorm/softmax/RoPE
order of operations, same KV cache dtypes (`-ctk q8_0 -ctv q4_0`), same sampling, mmap'd weights
(never copy weights into anonymous RAM), no quantization, no pruning, no approximate top-k.

## Non-negotiable environment facts (measured, see ../NOTES.md)

- VRAM: 6141 MiB total, ~104 MiB desktop. **The whole MoE expert set is 12.188 GiB — it can never fit in VRAM.**
- RAM: 14.3 GiB usable; page cache (~11.5 GiB observed) is the real budget, and the CPU expert working
  set is 11.54 GiB at `-ncmoe 38`. We are operating right at the cache cliff.
- NVMe (Kingston QLC, LUKS+btrfs): 555 MB/s @4K-QD32, ~2.0 GB/s @128K/1M. The Kioxia is 6-13x slower — unusable.
- CPU: 8C/16T Zen 3, AVX2 + FMA + F16C, **no AVX-512, no VNNI**. 8 physical cores.
- Compute ceiling measured: ~240 t/s prefill, 11-14 t/s decode (llama.cpp, cache-resident).
- Per-layer MoE expert bytes: 330.0 MiB (Q3_K, layers 0-9 and 30-39) / 294.0 MiB (IQ3_XXS, layers 10-29).
- Per expert per layer: 3 tensors x ~440 KiB (gate/up `[2048,512]`, down `[512,2048]`), ~1.29 MiB total.

## Established by measurement (see docs/BENCH.md — do not re-derive)

- Demand paging **with readahead** runs at **1.8 GiB/s** and leaves **100 % of pages resident**
  (mincore-verified). Fully resident reads run at **3.5 GiB/s**. The 555 MB/s O_DIRECT fio number is
  random access without readahead and is NOT the right constant — the cost model was wrong until
  this was measured.
- `amp-warm` pulls the entire 12.19 GiB expert set in at **1.92 GiB/s (6.3 s)**.
- The practical page-cache ceiling on this box is **~9.6 GiB** while other processes hold ~5 GiB, so
  the 12.19 GiB working set does **not** fit. Plan for a resident hot set plus a streamed tail.
- llama.cpp's 12 t/s collapse was a *latency* problem (synchronous faults, LRU thrash), not a
  bandwidth problem.
- KV cache is **8320 B/token** at k=q8_0/v=q4_0 → 1.55 GiB at 200k, 0.51 GiB at 65k.
- The planner independently reproduces the measured llama.cpp optimum (large ubatch beats GPU
  expert residency) and predicts **240 t/s prefill / 16.4 t/s decode** at 200k with prefetching.
- **Head-to-head vs llama-server (best config, same prompt, same cache state): 3.1x prefill
  (104.4 vs 34.1 t/s) and 5.4x decode (25.6 vs 4.76 t/s).** The win is the plan (6 expert layers on
  the GPU -> 10.25 GiB CPU set that fits the cache), amplified by the prefetcher.
- **Decode has a 10x cliff, not a slope**: 10.25 GiB of CPU experts -> 25.6-29.9 t/s, 11.54 GiB ->
  2.8-4.8 t/s. Cyclic scan + LRU means exceeding the cache by 1.3 GiB collapses reuse. Modelled as a
  cliff, and `amp-plan` prints "tg now" next to "tg t/s".
- **Quality: amp is bit-deterministic (max logprob delta 0.000000 run to run).** Moving experts
  between CPU and GPU shifts tail logprobs by up to ~0.5 and did not change the output; llama.cpp has
  the same property between its own -ngl settings. No algorithmic difference exists to find: same
  ggml kernels, same weights, same KV dtypes, same sampler.

## Architecture decisions already made

1. **Reuse ggml kernels from the local llama.cpp build** (`../llama.cpp/build/bin/libggml*.so`). This makes
   "zero quality loss" structurally guaranteed: the math is literally the same code. amp replaces the
   *runtime*, not the arithmetic.
2. **Memory planning is the product.** The bottleneck is not FLOPs, it is that a 12.19 GiB expert working
   set is cyclically scanned through a ~11.5 GiB page cache. Wins come from residency control and
   I/O-compute overlap, not from new math.
3. **Expert-major scheduling**: process the ubatch expert-by-expert (like `ggml_mul_mat_id`) so each expert's
   ~1.29 MiB is read exactly once per ubatch, and prefetch the next expert/layer asynchronously.
4. **Warm the page cache deliberately** with large sequential reads at startup. Random 4K faults cost
   555 MB/s; sequential reads of the same bytes cost ~2.0 GB/s.

## Working rules

- **Commit as you go.** No commit batching at the end. Small, working, committable increments.
  Conventional-ish prefixes: `feat:`, `fix:`, `perf:`, `docs:`, `bench:`.
- **Measure before and after every perf claim.** `bench/` numbers go in `docs/BENCH.md` with the config
  used. Never state a speedup without a measurement in the same commit.
- **Never run heavy tests without saying so first.** The user's machine froze once from a
  14.75 GB anonymous allocation, and `--load-mode none` / `--direct-io` are permanently banned.
- Do not launch a second `llama-server` while the user's is running (check `pgrep -x llama-server` and
  `nvidia-smi`); they contend for the same 6 GB of VRAM.
- `pkill -f "port XXXX"` matches its own command line and kills the shell. Use `pkill -x <name>`.
- `/tmp` is **tmpfs**. Never put I/O benchmark files there — they land in RAM and give false results.
- Keep the model file's extents contiguous (+16% read bandwidth). If it is ever re-copied, use
  `cp --reflink=never --sparse=never`, `sync`, then swap the name.

## Parity protocol traps (do not re-learn these)

- `n_probs` together with `top_k`/`top_p` makes llama-server renormalise the reported logprobs.
  Differences of ~2.4 logprob appear that are pure protocol error. `temperature 0` alone is greedy
  and leaves the distribution intact.
- `--jinja` on a raw `/completion` request applies the chat template and changes the prompt, so a
  parity comparison against `amp-infer --prompt` becomes meaningless.
- Greedy argmax flips are only meaningful next to the top-1/top-2 gap. Report the gap.

## Reference material

- `../NOTES.md` — every measurement, the performance model, VRAM arithmetic, the proven root causes.
- `../RUN.md` — the best llama.cpp command and the operational checklist.
- `../llama.cpp/` — source at git `1ab7e5a`. `../ik_llama.cpp/` — alternate kernels, slower on this model
  (117.6 vs 160.9 t/s prefill), needs `GGML_CUDA_NO_PINNED=1`.

## Milestones

- [x] M0 scaffold + this file
- [x] M1 GGUF header reader + geometry (validated against measured byte counts)
- [x] M2 page-cache warmer (measured: 1.92 GiB/s, 12.19 GiB in 6.3 s)
- [x] M2b cost model + memory planner (predicts 225 t/s / 12.4 t/s at 200k)
- [ ] M3 forward path: tokenize -> embed -> 40 blocks -> output, CPU-only, bit-comparable to llama.cpp
- [ ] M4 GPU offload of attention/SSM + N expert layers, with the VRAM planner
- [ ] M5 expert-major async prefetch overlapping I/O with compute
- [ ] M6 server: OpenAI-compatible API, slot prompt cache, context checkpoints, reasoning alias fix
- [ ] M7 decode optimization (the other half of the product: 11-14 t/s today)
