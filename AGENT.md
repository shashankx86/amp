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

1. **Reuse ggml kernels by vendoring llama.cpp** (`third_party/llama.cpp`, pinned in
   `third_party/DEPS.lock`, fetched by `scripts/fetch_deps.sh`, built from source and linked
   statically into `build/bin`). This makes "zero quality loss" structurally guaranteed: the math is
   literally the same code. amp replaces the *runtime*, not the arithmetic. Static linking also means
   no RPATH into someone else's build tree and no `LD_LIBRARY_PATH`.
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
- [x] M3 forward path + prefetcher + VRAM verification (`amp-infer`)
- [x] M4 GPU offload expressed through the public `tensor_buft_overrides` API (= `-ncmoe`)
- [x] M5 page-cache warming at start-up; decode-time prefetch of active experts **removed** on purpose
- [x] M6 server: OpenAI-compatible API, SSE, token-level prefix cache, prompt-boundary checkpoints,
      reasoning alias fix — `amp-server`, driven by `scripts/smoke_server.py`
- [ ] M3c amp's own ggml graph, so the router is observable and the 8 active experts per layer can be
      prefetched during decode (the only remaining decode win; impossible through `llama.h`)
- [ ] M7 close the gap between amp's decode and the measured ceiling (25.6 t/s was with a plan chosen
      for 6 GPU expert layers; the server currently plans 8 at short ctx)
- [ ] Fold `InferenceService`'s forward path and `ModelRuntime`'s into one implementation. They are
      two copies of the same loop today: the CLI has the prefetcher, the server has the cache. This is
      known duplication, deliberately left until the server is proven, and it is the first thing to fix.

## Server rules learned the hard way

- **A `llama_context` is not reentrant.** Two generations on one context corrupt the KV and the
  sampler, and the process dies inside `ggml_abort` — every in-flight stream returns zero bytes. This
  is not hypothetical: OpenCode issues a side request (conversation title) while the main stream is
  running, on a second connection, and my HTTP layer happily served both on one context. Fixed with
  `amp::TaskQueue` (one worker, FIFO), and `InferenceService::generate()` is now serialized *inside*
  the service so no handler path can forget. Reproduce with section 7 of
  `scripts/smoke_server.py`; before the fix all four streams were empty and the server was dead.
- **A stream must always end with `finish_reason`, then `[DONE]`.** Otherwise the Vercel AI SDK
  reports "OpenAI Chat stream ended without finish_reason" and retries forever, which presents as a
  mysterious hang rather than an error. On failure amp now sends the error object *and* a proper
  finishing chunk — strictly more robust than llama.cpp, which sends the error and stops.
- **A failed SSE write means the client is gone.** Stop generating (`interrupt()`) instead of
  computing tokens nobody will read. llama.cpp cancels the task on disconnect for the same reason.
- Handler exceptions are caught and turned into a 500 with OpenAI's error shape, mirroring
  `server-http.cpp`. An escaping exception through the worker thread would drop the connection.
- Error bodies use OpenAI's `type` strings (`invalid_request_error`, `server_error`, ...), because
  clients branch on them.
- Copy response shapes from `tools/server/server-task.cpp`, not from memory: `system_fingerprint` on
  every chunk, and `stream_options.include_usage` producing a trailing chunk with **empty** `choices`
  and only `usage` (the spec requires it; the AI SDK is fine either way but be exact).

## Facts about this model that are easy to get wrong

- **30 of the 40 layers are recurrent** (linear attention / SSM), only 10 use the KV cache. That is
  why the KV is 8320 B/token and why the recurrent state is a flat 62.81 MiB (40 layers, f32, 1 cell).
- **A recurrent state cannot be partially erased.** `llama_memory_seq_rm` only rewinds one through a
  bounded snapshot ring (`n_rs_seq`, 62.81 MiB of VRAM *per snapshot*, 0 by default) and for M-RoPE
  models the position check then aborts the decode outright. Any design that assumes "truncate the KV
  to the common prefix" is broken on this model — see the prompt-boundary checkpoint in
  `include/amp/runtime/prefix_cache.h`.
- **`llama_memory_seq_pos_max` returns a position index, not a count.** 25 cached tokens report 24.
  Verifying a rewind against a token count silently "succeeds" on a rewind that never happened; this
  cost a real position-continuity abort before it was found.
- **BPE merges across the prompt boundary.** The next turn's rendering splits the cached prompt's last
  token differently, so the common prefix is routinely *one* token short of the prompt end. The
  checkpoint therefore sits one token before the end and recomputes that token.
- **The chat template owns the thinking tags** and renders `<think>
` as the generation prompt, so the
  model reasons by default. The value stored for an assistant turn is the *inner* text; see
  `include/amp/util/text.h` for why accepting the tagged form matters.
- **The extended batch API does not produce logits unless asked.** `llama_batch_ext_set_output_logits`
  per token, then `llama_sampler_sample(smpl, ctx, idx)` with that idx. Passing `-1` with no output
  token aborts inside `ggml_abort`.
- **`tensor_buft_overrides` is walked until `pattern == nullptr`.** A missing sentinel reads past the
  array end: a segfault in the tensor loader, or silent garbage in the CLI. One helper builds it now.
