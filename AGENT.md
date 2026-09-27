# AGENT.md, working rules for `amp`

Read this first, every session. It is written to survive compaction.

## What amp is

A purpose-built inference engine for **exactly one model**:
`/home/e0u/localhost/models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf`
(arch `qwen35moe`, 40 blocks, hybrid SSM+attention, 256 experts / 8 used, native ctx 262144).

**Goal: beat llama.cpp's speed on this machine with ZERO quality loss.**
Zero quality loss is a hard constraint, not a preference. Concretely that means:
bit-identical quantized math (same dequant + same dot products as ggml), same RMSNorm/softmax/RoPE
order of operations, same KV cache dtypes (`-ctk q8_0 -ctv q8_0`), same sampling, mmap'd weights
(never copy weights into anonymous RAM), no quantization, no pruning, no approximate top-k.

## Non-negotiable environment facts (measured, see ../NOTES.md)

- VRAM: 6141 MiB total, ~104 MiB desktop. **The whole MoE expert set is 12.188 GiB and can never fit in VRAM.**
- RAM: 14.3 GiB usable; page cache (~11.5 GiB observed) is the real budget, and the CPU expert working
  set is 11.54 GiB at `-ncmoe 38`. We are operating right at the cache cliff.
- NVMe (Kingston QLC, LUKS+btrfs): 555 MB/s @4K-QD32, ~2.0 GB/s @128K/1M. The Kioxia is 6 to 13x slower, which makes it unusable.
- CPU: 8C/16T Zen 3, AVX2 + FMA + F16C, **no AVX-512, no VNNI**. 8 physical cores.
- Compute ceiling measured: ~240 t/s prefill, 11-14 t/s decode (llama.cpp, cache-resident).
- Per-layer MoE expert bytes: 330.0 MiB (Q3_K, layers 0-9 and 30-39) / 294.0 MiB (IQ3_XXS, layers 10-29).
- Per expert per layer: 3 tensors x ~440 KiB (gate/up `[2048,512]`, down `[512,2048]`), ~1.29 MiB total.

## Established by measurement (see docs/BENCH.md, do not re-derive)

Measure prefill with a sequence of distinct prompts or the number means nothing. The same
configuration, restarted three times, measured 132.5, 178.4 and 311.3 t/s on one identical
18,265-token prompt. Decode over those same runs varied only between 26.4 and 27.7. The reason
is the working set. Prefill at batch ~1024 routes most of the 256 experts per layer through the
CPU matmul, so it reads nearly all 13.66 GiB of the weights and thrashes a page cache that
holds about 11 GiB of it. Decode at batch 1 touches 8 experts and stays resident. Use
`scripts/bench_prefill.py`, which builds a fresh prompt per measurement so nothing is served
from the prompt cache, and quote the median with the spread beside it. An entire `n_ubatch`
sweep in this project was run on single requests and every conclusion had to be thrown out.

`cache_ram_mib` is 0. The old clamp to 512 reasoned that a smaller prompt cache would protect the
model's page cache. The direction is right — 0 is not slower, and at 200k over ten distinct 18k
prompts it measures 559.8 t/s against 509.5 t/s at 512, so **about 1.1x**. It was previously
recorded here as 3x, which is wrong: the 85.0 t/s that supported it does not reproduce at any
prompt count, and the "collapses and never recovers" sequence it was read from was one unlucky
session. Re-measured, both arms climb out of the cold request and hold. Do not quote 3x. Prefix
reuse comes from the slot's own KV rather than this cache, and turning it off cost nothing
agentically: an 8.7 s worst turn against 10.0 s at 512.

One protocol trap, because it will hide an eviction effect even when the effect is real:
`bench_prefill.py --tokens N` builds prompts that land *larger* than N (8192 requested, ~11.2k
delivered). At 11.2k a 512 MiB cache needs 5.7 prompts to fill, so a 5-prompt run cannot show it.
Run past the point where the cache fills, and check that it did.

**Quality parity needs TEACHER FORCING, and getting this wrong produces a large, confident,
completely meaningless number.** A per-position KL comparison where each engine samples its own
tokens puts the two arms on different prefixes at the first argmax difference, and every later
position compares two different questions. Measured at 131k that reported **444 of 512 positions
"disagreeing"** at median KL 1.7e-05 while the dtype under test had almost no effect. The fix is
`amp-infer --score-file` against a fixture from `--emit-score`: every arm then evaluates the
identical prefix and every position is comparable. Also note the logprob window — `top_k_track_`
was 5, and a token dropping out of a 5-wide window reads as a disagreement; it is 32 now and
`--logprobs-n` sets it. Run the determinism control (two runs, same config) and the placement
control (same dtype, g=3 vs g=6) before believing any dtype number: placement alone is a
**1.2e-04** median KL floor, and both q8_0 configs sit 26-29x above it at 131k.

Decode splits 40% expert bandwidth and 60% everything else, measured 2026-09-28 by subtraction
(349 MiB of experts at a measured 30.2 GB/s = 11.29 ms of a 28.25 ms token; `ggml mul_mat` at
batch 1 runs at 30.2 GB/s on GDN layers and 29.5 on attention, against plain reads of 29.1 and
29.0, so no expert kernel is running badly). The 60% is not an expert kernel. Three
independent routes put the expert share at 30 to 36 %: 349 MiB per token at the measured
29.21 GB/s is 12.5 ms of a 34.57 ms token, and the `-ncmoe` A/B solved two ways gives 29 % both
times.

That 3.7 % figure this section used to carry was an arithmetic bug, and the bug is worth
remembering. The share of expert bytes *removed* was computed as `1 - 1.29/12.19 = 0.894` instead
of `0.106`, an 8.4x error that inverted the conclusion. Do not restore the old number. When you
turn one A/B pair into a decomposition of a whole system, check it against a second method
before believing it.

The expert matvec is memory-bandwidth-bound and already sitting at bandwidth. `ggml mul_mat`
over an `iq3_xxs` expert matrix at batch 1 runs at 29.21 GB/s, against 28.69 GB/s for a plain
8-thread read of the very same bytes. That is 101.8 % of plain, which means the dequant work is
entirely hidden behind the memory traffic.

So prefetching cannot help the MoE, and not because the hardware prefetcher is clever. There is
no exposed latency left to hide. A cheaper dequant kernel would not help either, for the same
reason. The expert path improves only by moving fewer bytes, which means quantization, which
this project rules out. Measure it with `amp-kernel-bound`, and pin ggml's thread count with
`ggml_backend_cpu_set_n_threads` or the comparison means nothing.

VRAM is a binding constraint, which reverses the earlier advice in this file. g=0 to g=4
measures +3.4 % for 10.6 % of expert bytes moved off the CPU. Then placement saturates: g=4 to
g=7 measures 28.89 against 26.44 t/s, slightly worse, and g=6 OOMs at 200k. The ceiling is
therefore the measured +3.4 %, not a compounding 1.46x. That 1.46x came from the same kind of
linear extrapolation that produced the 3.7 % error, and it was wrong for the same reason.

The odd part is that the GPU sits 70 % idle through all of this, at 29 % utilisation and 18.7 W
of a 140 W budget. Moving work onto it still does not help decode. At batch 1 a GPU GEMV has
almost no parallelism to exploit and just competes with the 30 recurrent layers already resident
there. What the spare capacity does help is prefill, where batches are big enough to fill it:
g=4 to g=7 is +27 % prefill. The real trade is context length against prefill rate, not VRAM
against decode speed. `-c 64000` gives +27 % prefill for -8.5 % decode.

Decode I/O is a non-issue and page faults are nowhere near the critical path. Over 384 tokens,
`read_bytes` totals 88.61 MiB, or 236 KiB per token, `majflt` is 56 per token, and `stime` is
0.1 % of busy time. One caveat matters here: DRAM stalls bill as *user* time, so a small
`stime` rules out I/O and kernel work without showing that decode is arithmetic-bound. That
distinction is what made the 3.7 % error possible in the first place.

Decode threads are already optimal. `-t 8` (one per physical core) measures 32.97 t/s, against
27.95 at 4, 29.96 at 12 and 21.19 at 16. The planner's `ncpu / 2` default is right. SMT siblings
contend, so do not "improve" this.

N-gram speculative decoding does nothing here. It is lossless and was therefore the one
remaining lever that avoids the arithmetic, but it buys +3 % on a counting prompt, which is
inside the 22 % session variance, and +0.2 % on a 300-token templated list, which is exactly
where drafting should pay best. The model is too entropic at the token level for n-grams to find
matches.

Context length barely costs decode: 8,192 gives 35.61 t/s, 65,536 gives 35.27, and 200,000 gives
34.68. The 200k target costs 2.6 % even though the planner gives up GPU expert layers as the KV
grows, 8 down to 4.

Steady decode varies about 22 % between sessions on this box, driven by page-cache warmth:
28.39, 30.39, 31.44, 33.94 and 34.68 t/s at 200k. An A/B is only trustworthy if both arms are
measured in the same session. This is also how the project came to hold both "35.25 t/s" and
"28.39 t/s" without either being a lie.

Demand paging with readahead runs at 1.8 GiB/s and leaves 100 % of pages resident, verified
with mincore. Fully resident reads run at 3.5 GiB/s. The 555 MB/s O_DIRECT fio number is random
access without readahead and is not the right constant. The cost model was wrong until this got
measured.

`amp-warm` pulls the entire 12.19 GiB expert set in at 1.92 GiB/s, which takes 6.3 s.

The practical page-cache ceiling on this box is about 9.6 GiB while other processes hold around
5 GiB, so the 12.19 GiB working set does not fit. Plan for a resident hot set plus a streamed
tail.

llama.cpp's 12 t/s collapse was a latency problem, from synchronous faults and LRU thrash, not a
bandwidth problem.

KV cache at the shipped q8_0/q8_0 is about 10.9 KB per token, measured at 2.03 GiB at 200k and
0.64 GiB at 131k. That is higher than the 8320 B/token the old q8_0/q4_0 pair gave, which was
1.55 GiB at 200k. The planner pays for the difference by dropping one GPU expert layer, which
costs decode almost nothing and buys 8.8x closer to f16.

The planner independently reproduces the measured llama.cpp optimum, where a large ubatch beats
GPU expert residency, and predicts 240 t/s prefill and 16.4 t/s decode at 200k with prefetching.
- **Head-to-head vs llama-server, MEASURED AGAIN 2026-09-26 and CORRECTED.** The old claim was
  3.1x prefill and 5.4x decode (25.6 vs 4.76 t/s). Re-measured with 5 identical requests per
  engine, alternating order, rates from each server's own `timings`:

  | engine | request 1 | steady decode |
  |---|---|---|
  | amp-server | 26.01 t/s | **28.39 t/s** |
  | llama-server (documented best) | **1.96 t/s** | **30.05 t/s** |

  **At steady state they are the same, within noise.** The old 5.4x compared amp's warm state
  against llama-server's cold one. 28.39 t/s is inside the 25.6-29.9 t/s range already recorded
  below; 35.25 t/s was the outlier. Do not quote a speedup over llama-server, because it does not
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
- `/tmp` is **tmpfs**. Never put I/O benchmark files there. They land in RAM and give false results.
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

- `../NOTES.md`: every measurement, the performance model, VRAM arithmetic, the proven root causes.
- `../RUN.md`: the best llama.cpp command and the operational checklist.
- `../llama.cpp/`: source at git `1ab7e5a`. `../ik_llama.cpp/` has alternate kernels, slower on this model
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
      reasoning alias fix. `amp-server`, driven by `scripts/parity_test.py`
- [x] M3c. **Implemented, measured, and a tie.** It did not need to own the ggml graph after all. Inside
      `ggml_compute_forward_mul_mat_id` the router output is already host-resident, so the active
      experts are observable with no sync. Two variants measured (same-node, and predictive
      cross-layer, which is the form the milestone promised), nine paired comparisons spanning
      -1.9 % to +1.7 %, mean **+0.2 %**. Quality is provably unchanged: KL exactly 0.000000e+00 both
      directions, JS 0, max |delta logprob| 0, 512/512 top-1, gated at `max-kl 0.0`.
      **Why it cannot win:** the expert matvec already runs at 101.8 % of a plain read of its own
      bytes, so there is no exposed latency for a prefetcher to hide. The code path is verified
      live. Depth 4096 KiB costs 1.9 %, which is about what the loop's instruction issue alone should cost.
      Kept as `third_party/patches/m3c-expert-prefetch.patch`, off by default, because
      `third_party/llama.cpp` is gitignored and an edit made in that tree is not versioned.
- [x] M7 **spent.** Every quality-neutral decode lever has been measured: threads already optimal
      (`-t 8` beats 4/12/16), expert placement 3.4 %, context length 2.6 %, n-gram speculation
      0.2 %, expert prefetch 0.2 %. There is no gap left to close on the CPU side.
- [x] Profile decode per-op. **Done, and it refuted the hypothesis.** `perf` works here. Extract it
      from the local pacman cache (`/var/cache/pacman/pkg/perf-*.pkg.tar.zst`) into a user dir, no
      sudo; `perf_event_open` was verified to work at `perf_event_paranoid=2` for our own process.
      The Release binary is not stripped and has all 126 `ggml_compute_forward_*` symbols, so no
      debug rebuild is needed. Result over 352k samples of pure decode: **100 % of CPU-side work is
      `ggml_vec_dot_iq3_xxs_q8_K` (21.2 %) + `ggml_vec_dot_q3_K_q8_K` (16.6 %)**, and `ssm_conv`,
      `ssm_scan`, `flash_attn_ext`, `argsort` and `get_rows` are **absent entirely**. They are not
      slow, they are not on the CPU: only expert weights are pinned to CPU, so all 30 recurrent
      layers and all 10 attention layers run on the GPU. **Do not go looking for SSM wins on the CPU
      side; there is no CPU work there.** The MoE expert matvec is the whole CPU story and it is
      already at memory bandwidth.
- [x] **Never set `OMP_WAIT_POLICY=passive` on this box.** ggml is OpenMP-built, so decode spends
      ~1000 barriers per token and the 8 threads spin at each: **53.6 % of CPU cycles**. That is load
      balancing rather than waste. Spinning measures 29.56 t/s against 22.53 t/s for sleeping, a 31 %
      penalty. That 53.6 % is the price of 8-thread efficiency, not a bug to fix.
- [x] **Tool-call latency: root-caused, reproduced and fixed.** It was `n_parallel: 1`. A model asked
      to verify a config sends a short self-test request to the API it is being served by; with one
      slot that request is served by the slot holding the conversation and the cached prefix does
      not survive it. At a 35,001-token context that costs **2 full re-prefills and a 110.6 s worst
      turn**; at 2 slots, **0 re-prefills and 6.3 s**. Decode is unaffected (27.42 / 27.25 / 27.67
      t/s at 1 / 2 / 4 slots) because the 44x collapse needs *concurrent* generation, which an
      agentic turn never does. **Default is now 2.** Reproduce with
      `scripts/bench_agentic.py --interleave N --base-tokens 35000`; it only bites above ~30k tokens
      of context, so a small-context benchmark will show 0 re-prefills and hide the bug entirely.
      `get_perf.sh` installs the profiler unprivileged and `build/` is gitignored, so neither
      leaves the tree dirty.
- [ ] Three latency hypotheses were falsified on the way and are recorded so they are not retested:
      `cache_ram_mib` 512 / 2048 / 4096 all give 0 re-prefills and ~2.2 s median turns; squeezing
      5 GiB of RAM does the same (`MAP_POPULATE` refills the page cache at load, so a fresh server
      is warm even on a tight box); and OpenMP barrier spinning is not a cost but a benefit.
- [ ] **Unexplained: long prompts are truncated to about half.** A 104,884-char prompt (36,530
      tokens by `/tokenize`) was processed as 18,269 tokens. Not the tokenizer, which tokenises 4x
      the text to 106,239 as one string. If a real prompt is silently halved that is worse than any
      speed problem here. Uninvestigated.
- [x] Fold the two forward paths into one, resolved by deletion rather than refactoring. The server
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
  false %}` -> `<think>\n\n</think>\n\n` (pre-closed, answer directly); else `<think>\n`.
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
  delivered as `content`, which is what a conversation title came out as. The splitter must be told
  the block is already open, which `render_chat` derives from the rendered text (does it end with the
  template's thinking start tag?).
- **The closing tag arrives with variable whitespace.** Naturally `\n</think>`; forced by the
  reasoning budget, `</think>` with no newline. Match the tag *and* the whitespace-trimmed tag.
- **`llama_process()` asserts a batch is no larger than `n_ubatch`** (llama-context.cpp), and
  `llama_batch_ext` is sized to `n_batch`. The caller must split the prefill. Not splitting it failed
  every prompt over the ubatch with "prefill batch full at token 2048", which is every real conversation,
  since an agent client resends the whole history each turn.
- **To get a non-reasoning answer, use the template's switch, not a forced close.**
  `chat_template_kwargs: {"enable_thinking": false}` makes the template pre-close the think block.
  The reasoning budget (`reasoning_budget_tokens`) also works, and is effective: measured 100
  completion tokens down to 12. But forcing a close mid-thought alters generation, and at a very
  tight budget (8 tokens) 1 sample in 3 returned a visibly doubled answer (`4\n</think>\n\n4`).
  4 of 4 samples at 16-32 were clean. The severe looping in older notes was on the deleted server's
  sampling path and did not reproduce. It stays opt-in; see docs/BENCH.md.
- **The reasoning budget is llama.cpp's `common_reasoning_budget_init`** and it arms itself by
  *replaying the prefill tokens* through the sampler, with no template special-casing. Feed it with
  `llama_sampler_accept` before the first generated token.
- Use `common_tokenize()` for tags, not a hand-rolled "call with a null buffer to count" version: the
  counting call returns 0 for a bare special token, which silently disables the whole budget path.
- A short request from a thinking model can legitimately produce **no content at all**. That is the
  spec-correct answer, not a bug: the budget is the client's lever, or `enable_thinking: false`.

## Server rules learned the hard way

- **A `llama_context` is not reentrant.** Two generations on one context corrupt the KV and the
  sampler, and the process dies inside `ggml_abort`, so every in-flight stream returns zero bytes. This
  is not hypothetical: OpenCode issues a side request (conversation title) while the main stream is
  running, on a second connection. Our HTTP layer happily served both on one context; four concurrent
  streams returned zero bytes each and the process died.

  Fixed in the hand-rolled server with `amp::TaskQueue` (one worker, FIFO), which no longer exists.
  What handles it now is llama.cpp's slot model plus our `n_parallel = 1` default. See the next
  bullet, which is the more important half of the lesson. Regression test: section 14 of
  `scripts/parity_test.py`.
- **Concurrent slots are not free, and on this model they are actively destructive.**
  `llama-server` defaults `n_parallel = -1` ("auto", `common/arg.cpp:1400`), which
  `tools/server/server.cpp:156-159` expands to **four** concurrent slots with `kv_unified`. Every
  concurrent generation wants the same shared ~10.9 GiB CPU expert set, and on a box with ~10.9 GiB
  of usable page cache that is not sharing, it is thrashing: measured **0.64 t/s** with two slots
  active against 28.4 t/s with one. A 44x collapse, found only by running the OpenCode harness
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
  finishing chunk, which is strictly more robust than llama.cpp, since it sends the error and stops.
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
