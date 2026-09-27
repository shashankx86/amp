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

- **Decode splits roughly one third / two thirds between the expert matvec and everything else.**
  The expert share is **~30-36 %** by three independent routes (349 MiB/token at the measured
  29.21 GB/s = 12.5 ms of a 34.57 ms token; the `-ncmoe` A/B solved two ways, both 29 %).
  **This supersedes an earlier claim of 3.7 %, which was an arithmetic bug** — the share of expert
  bytes *removed* was computed as `1 - 1.29/12.19 = 0.894` instead of `0.106`, an 8.4x error. Do
  not restore the old number. If you write a script that turns one A/B pair into a decomposition,
  check it against a second method before believing it.
- **The expert matvec is memory-bandwidth-bound, and already at bandwidth.** `ggml mul_mat` over
  an `iq3_xxs` expert matrix at batch 1 runs at **29.21 GB/s** against **28.69 GB/s** for a plain
  8-thread read of the same bytes — 101.8 % of plain, so the dequant work is entirely hidden behind
  the memory traffic. Consequence: **prefetching cannot help the MoE**, not because the hardware
  prefetcher is clever but because there is no exposed latency left; and a cheaper dequant kernel
  would not help either. The expert path improves only by moving fewer bytes, i.e. quantization,
  which the project rules out. Measure with `amp-kernel-bound`; pin ggml's thread count with
  `ggml_backend_cpu_set_n_threads` or the comparison is meaningless.
- **VRAM *is* a binding constraint, reversing the earlier advice.** g=0 -> g=4 measures +3.4 % for
  10.6 % of expert bytes moved off the CPU, so g=8 is worth roughly +7 % and all-experts-on-GPU
  roughly **1.46x**. 6 GB cannot hold g=8. Unreachable on this laptop, but it is the only large
  multiplier left, and on a bigger card that is where the win is.
- **Decode I/O is a non-issue and page faults are not on the critical path.** Over 384 tokens:
  `read_bytes` 88.61 MiB total = **236 KiB/token**, `majflt` 56/token, and `stime` **0.1 %** of busy
  time. Note that DRAM stalls bill as *user* time, so a small `stime` rules out I/O and kernel work
  but does **not** by itself show decode is arithmetic-bound.
- **Decode threads are already optimal.** `-t 8` (one per physical core) measures 32.97 t/s against
  27.95 at 4, 29.96 at 12 and 21.19 at 16. The planner's `ncpu / 2` default is right; SMT siblings
  contend. Do not "improve" this.
- **N-gram speculative decoding does nothing here.** Lossless and therefore the one remaining lever
  that avoids the arithmetic: +3 % on a counting prompt (inside the 22 % session variance) and
  **+0.2 %** on a 300-token templated list, which is where drafting should pay best. The model is too
  entropic at the token level for n-grams to find matches.
- **Context length barely costs decode.** 8,192 -> 35.61, 65,536 -> 35.27, 200,000 -> 34.68 t/s.
  The 200k target costs **2.6 %** even though the planner gives up GPU expert layers (8 -> 4) as the
  KV grows.
- **Steady decode varies ~22 % between sessions** on this box (28.39 / 30.39 / 31.44 / 33.94 /
  34.68 t/s at 200k), driven by page-cache warmth. A single-session A/B is only trustworthy if both
  engines are measured in that same session. This is how the project came to hold both "35.25 t/s"
  and "28.39 t/s" without either being a lie.

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
- **Head-to-head vs llama-server, MEASURED AGAIN 2026-09-26 and CORRECTED.** The old claim was
  3.1x prefill and 5.4x decode (25.6 vs 4.76 t/s). Re-measured with 5 identical requests per
  engine, alternating order, rates from each server's own `timings`:

  | engine | request 1 | steady decode |
  |---|---|---|
  | amp-server | 26.01 t/s | **28.39 t/s** |
  | llama-server (documented best) | **1.96 t/s** | **30.05 t/s** |

  **At steady state they are the same, within noise.** The old 5.4x compared amp's warm state
  against llama-server's cold one. 28.39 t/s is inside the 25.6-29.9 t/s range already recorded
  below; 35.25 t/s was the outlier. Do not quote a speedup over llama-server — it does not
  reproduce. See docs/BENCH.md, "the 7.4x decode claim does not reproduce".
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
      reasoning alias fix — `amp-server`, driven by `scripts/parity_test.py`
- [x] M3c — **implemented, measured, TIE.** It did not need to own the ggml graph after all: inside
      `ggml_compute_forward_mul_mat_id` the router output is already host-resident, so the active
      experts are observable with no sync. Two variants measured (same-node, and predictive
      cross-layer, which is the form the milestone promised), nine paired comparisons spanning
      -1.9 % to +1.7 %, mean **+0.2 %**. Quality is provably unchanged: KL exactly 0.000000e+00 both
      directions, JS 0, max |delta logprob| 0, 512/512 top-1, gated at `max-kl 0.0`.
      **Why it cannot win:** the expert matvec already runs at 101.8 % of a plain read of its own
      bytes, so there is no exposed latency for a prefetcher to hide. The code path is verified
      live — depth 4096 KiB costs 1.9 %, which is the loop's instruction-issue cost.
      Kept as `third_party/patches/m3c-expert-prefetch.patch`, off by default, because
      `third_party/llama.cpp` is gitignored and an edit made in that tree is not versioned.
- [x] M7 **spent.** Every quality-neutral decode lever has been measured: threads already optimal
      (`-t 8` beats 4/12/16), expert placement 3.4 %, context length 2.6 %, n-gram speculation
      0.2 %, expert prefetch 0.2 %. There is no gap left to close on the CPU side.
- [ ] Profile the **30 recurrent layers** (of 40; the other 10 are full attention). This is now the
      only substantial unmeasured region: the MoE's third of decode is measured, understood and
      closed, so the other two thirds is what remains. A sequential SSM recurrence parallelises
      badly and is the obvious suspect. No per-op profile exists, because `perf` is absent and there
      is no sudo — `/proc` counters say what decode waits on but not which op. Needs a sampling
      profiler, a debug ggml with `GGML_SCHED_DEBUG`, or per-op timers in the CPU backend.
- [x] Fold the two forward paths into one — resolved by deletion rather than refactoring. The server
      (`InferenceService`) is gone; `amp-server` is llama.cpp's server. `ModelRuntime` survives as
      `amp-infer`, the measurement harness, because it has the prefetcher and can be A/B'd against
      the server without a network round trip. The duplication was a symptom of maintaining our own
      server, and it went away with it.
- [x] Server: replace the hand-rolled one with llama.cpp's. 40+ routes, 100% of the flag surface,
      `docs/PARITY.md` as the contract, `scripts/parity_test.py` as the test (15 sections pass,
      8 features honestly skipped). Measured bit-identical to upstream at matched placement.

## What this model's reasoning surface actually is

From the `tokenizer.chat_template` in the GGUF (7764 chars), and `tools/server/server-task.cpp`:

- **Generation starts inside a think block.** `{%- if enable_thinking is defined and enable_thinking is
  false %}` → `<think>\n\n</think>\n\n` (pre-closed, answer directly); else `<think>\n`.
- **`preserve_thinking`** decides whether an assistant turn's reasoning is re-rendered; the template
  *also* preserves anything after `ns.last_query_index` automatically, so the current turn needs no flag.
  llama.cpp's public name for this is `preserve_reasoning`, mapped onto the real variable names by
  `jinja::caps_apply_preserve_reasoning` (which also derives `clear_thinking`,
  `truncate_history_thinking`, `drop_thinking`). Setting the raw name as a kwarg skips that mapping.
- **`reasoning_content`** is accepted on assistant messages; if absent, the template splits it out of
  `content` itself on `</think>`. Both conventions round-trip.
- **Not supported by the template:** `reasoning_effort` / `reasoning_strength`.
- **Tool calls** are rendered by the template; amp does not yet emit structured `tool_calls`.
- **API-level variants** (llama.cpp): `reasoning_format` ∈ `none` | `auto` | `deepseek` |
  `deepseek-legacy`; `reasoning_budget_tokens` / `thinking_budget_tokens` / `--reasoning-budget`
  (`-1` unrestricted, `0` immediate, `N`); `reasoning_budget_message`; `reasoning_control`.
- **SOLVED (2026-09-26) - the bug was ours, not upstream's.** The old hand-rolled server could not
  make `enable_thinking: false` work. llama.cpp's own server does, with no patch. The real gate is
  `server-context.cpp:1463-1464`:

      template_supports_thinking = params_base.use_jinja
                                && common_chat_templates_support_enable_thinking(...);
      enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;

  and the per-request value is applied at `server-common.cpp:1339-1346`. Verified with
  `/apply-template`: default renders `<|im_start|>assistant\n<think>\n`; `enable_thinking:false`
  renders `<|im_start|>assistant\n<think>\n\n</think>\n\n`. End to end, same question, same
  answer `4`: 104 completion tokens with thinking on, **2** with it off.
  The earlier hypothesis was wrong on both counts: the autoparser's reasoning-mode detection comes
  from template analysis in `chat-diff-analyzer.cpp`, not from `enable_thinking`, and the
  specialized-template path was never involved. Lesson: when a reimplementation disagrees with
  upstream, suspect the reimplementation first, and prove it with `/apply-template` before
  theorising.

## Thinking-model rules learned the hard way (this template, this quant)

- **The generation prompt ends *inside* a reasoning block.** `/apply-template` shows the rendered
  prompt ending `<|im_start|>assistant\n<think>\n`, so the model reasons without ever emitting an
  opening tag. Any split triggered by the opening tag never fires and the entire chain of thought is
  delivered as `content` — that is what a conversation title came out as. The splitter must be told
  the block is already open, which `render_chat` derives from the rendered text (does it end with the
  template's thinking start tag?).
- **The closing tag arrives with variable whitespace.** Naturally `\n</think>`; forced by the
  reasoning budget, `</think>` with no newline. Match the tag *and* the whitespace-trimmed tag.
- **`llama_process()` asserts a batch is no larger than `n_ubatch`** (llama-context.cpp), and
  `llama_batch_ext` is sized to `n_batch`. The caller must split the prefill. Not splitting it failed
  every prompt over the ubatch with "prefill batch full at token 2048" — i.e. every real conversation,
  since an agent client resends the whole history each turn.
- **To get a non-reasoning answer, use the template's switch, not a forced close.**
  `chat_template_kwargs: {"enable_thinking": false}` makes the template pre-close the think block.
  The reasoning budget (`reasoning_budget_tokens`) also works and is effective — measured 100
  completion tokens down to 12 — but forcing a close mid-thought alters generation, and at a very
  tight budget (8 tokens) 1 sample in 3 returned a visibly doubled answer (`4\n</think>\n\n4`).
  4 of 4 samples at 16-32 were clean. The severe looping in older notes was on the deleted server's
  sampling path and did not reproduce. It stays opt-in; see docs/BENCH.md.
- **The reasoning budget is llama.cpp's `common_reasoning_budget_init`** and it arms itself by
  *replaying the prefill tokens* through the sampler — no template special-casing. Feed it with
  `llama_sampler_accept` before the first generated token.
- Use `common_tokenize()` for tags, not a hand-rolled "call with a null buffer to count" version: the
  counting call returns 0 for a bare special token, which silently disables the whole budget path.
- A short request from a thinking model can legitimately produce **no content at all**. That is the
  spec-correct answer, not a bug: the budget is the client's lever, or `enable_thinking: false`.

## Server rules learned the hard way

- **A `llama_context` is not reentrant.** Two generations on one context corrupt the KV and the
  sampler, and the process dies inside `ggml_abort` — every in-flight stream returns zero bytes. This
  is not hypothetical: OpenCode issues a side request (conversation title) while the main stream is
  running, on a second connection. Our HTTP layer happily served both on one context; four concurrent
  streams returned zero bytes each and the process died.

  Fixed in the hand-rolled server with `amp::TaskQueue` (one worker, FIFO), which no longer exists.
  What handles it now is llama.cpp's slot model plus our `n_parallel = 1` default — see the next
  bullet, which is the more important half of the lesson. Regression test: section 14 of
  `scripts/parity_test.py`.
- **Concurrent slots are not free, and on this model they are actively destructive.**
  `llama-server` defaults `n_parallel = -1` ("auto", `common/arg.cpp:1400`), which
  `tools/server/server.cpp:156-159` expands to **four** concurrent slots with `kv_unified`. Every
  concurrent generation wants the same shared ~10.9 GiB CPU expert set, and on a box with ~10.9 GiB
  of usable page cache that is not sharing, it is thrashing: measured **0.64 t/s** with two slots
  active against 28.4 t/s with one — a 44x collapse, found only by running the OpenCode harness
  (it stalled ~50 min on one prompt). The preflight now sets `n_parallel = 1` unless `--parallel N`
  is passed. General rule: **on a model whose decode is bound by a working set larger than cache,
  "parallel" slots divide a fixed resource and the default must be 1.**
  **The cost, which is a real capability loss:** `n > 1` (several choices in one response) is bounded
  by the slot count, so with one slot it returns a typed 400. Stock llama-server can serve it. We
  traded that for the 44x. `parity_test.py` section 19 asserts the typed 400 rather than pretending
  the feature works, and RUNNING.md states the trade so it is the user's call.
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

## llama.cpp defaults that are dangerous on this box (clamped by the preflight)

These are llama.cpp's own defaults, not amp's, and each is a way to OOM or thrash a 6 GB card with
14 GiB of RAM. `amp::PreflightOptions` clamps them and says so in the startup log:

| default | value | why it is dangerous here | clamped to |
|---|---|---|---|
| `n_ctx_checkpoints` | 32 | each checkpoint is a **full** serialized sequence state (`common_prompt_checkpoint::data_tgt`, filled by `llama_state_seq_get_data_ext`) - 8320 B/token of KV plus the 62.81 MiB recurrent state, so ~1.6 GiB each at 200k, and the ring reaches tens of GiB | 2 |
| `cache_ram_mib` | 8192 | an *anonymous* RAM prompt cache that evicts the model's page cache - the exact mechanism behind the 10x decode cliff | 512 |
| `fit_params` | on | llama.cpp's fitter throws on our layout (`fit.cpp:463-486`), the failure is ignored (`common.cpp:1320`), and a *successful* fit would overwrite the buft overrides and re-fill the GPU | off |

## Facts about this model that are easy to get wrong

- **30 of the 40 layers are recurrent** (linear attention / SSM), only 10 use the KV cache. That is
  why the KV is 8320 B/token and why the recurrent state is a flat 62.81 MiB (40 layers, f32, 1 cell).
- **A recurrent state cannot be partially erased.** `llama_memory_seq_rm` only rewinds one through a
  bounded snapshot ring (`n_rs_seq`, 62.81 MiB of VRAM *per snapshot*, 0 by default) and for M-RoPE
  models the position check then aborts the decode outright. Any design that assumes "truncate the KV
  to the common prefix" is broken on this model. Our prompt-boundary checkpoint was a workaround and
  is deleted; llama.cpp solves it properly by asking the model what it supports via
  `common_context_seq_rm_type` (`common/common.h:989-992`) and falling back to a full re-process
  (`tools/server/server-context.cpp:3379`).
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
