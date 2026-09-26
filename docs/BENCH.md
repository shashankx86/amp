# BENCH.md — measurements taken by amp on this machine

All numbers measured on the target box (CachyOS 7.2.6, RTX 4050 Laptop 6 GB, Ryzen 7 7735HS,
14 GiB RAM, Kingston QLC NVMe under LUKS+btrfs). Raw context in `../NOTES.md`.

Reproduce with:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DAMP_LLAMA_ROOT=../llama.cpp
cmake --build build -j$(nproc)
./build/bin/amp-plan --model ../models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf
./build/bin/amp-warm  --model ../models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf \
                  --what experts --drop-cache --verify
```

---

## 1. Demand paging vs direct I/O — the number that matters most

`mincore()` over a read-only mmap of the model, touching one byte per 4 KiB page. This is the
exact access pattern both llama.cpp and amp produce.

| region | resident before | cold touch | resident after | warm touch |
|---|---|---|---|---|
| 256 MiB @ 1.5 GiB | 29.9 % | 0.14 s = **1804 MiB/s** | **100 %** | 0.07 s = **3620 MiB/s** |
| 2 GiB @ 1.5 GiB | 26.0 % | 1.18 s = **1737 MiB/s** | **100 %** | 0.58 s = **3516 MiB/s** |

**Demand faults with readahead reach ~1.8 GiB/s and leave 100 % of pages resident. Warm reads run at
~3.5 GiB/s (RAM speed).**

This corrects the number I originally took from fio: O_DIRECT 4 KiB QD32 random reads measure
555 MB/s, but that is random access *without readahead*, which is not what expert streaming looks
like. The cost model now uses 1.85 GB/s for the fault path and 3.5 GB/s for the resident path.

Consequence for the design: the 12 t/s catastrophe in llama.cpp was never a bandwidth problem. With
a 12.19 GiB working set cyclically scanned through a smaller cache, LRU yields ~0 % reuse, every
ubatch re-reads the whole set, and *synchronous* faults on the compute thread cannot build queue
depth — that latency, not throughput, is what collapses prefill. The fixes are residency control
and I/O/compute overlap, not faster disks.

## 2. Warming the whole expert working set

`amp-warm --what experts --drop-cache --verify` (12.19 GiB, the CPU-resident expert set for all 40
layers):

```
12.19 GiB in 6.34 s = 1.92 GiB/s   (cold, sequential, 2 MiB chunks)
re-read           = 2.20 GiB/s
```

**The entire expert working set can be pulled into the page cache in 6.3 seconds.** Whether it
*stays* there is the open question: after the warm, `Cached` was only 5.46 GiB with 4.3 GiB of RAM
still free, i.e. roughly 7 GiB of the 12.19 GiB had been evicted again. Memory in use by other
processes at the time was 5.2 GiB, so the practical cache ceiling is ~9.6 GiB — below the working
set. That is the structural reason `amp` plans a resident hot set *plus* an explicitly streamed
tail instead of assuming everything fits.

## 3. Storage characteristics (fio 3.42, O_DIRECT)

| test | Kingston (LUKS+btrfs) | Kioxia KBG40ZNV256G (ext4) |
|---|---|---|
| 4 KiB randread QD32 sustained 60 s | 555-562 MB/s (flat) | 89 MB/s (after a 623 MB/s first 10 s) |
| 4 KiB randread QD1 | 50 MB/s, p99 0.10 ms | 26 MB/s, p99 0.19 ms |
| 128 KiB randread QD32 | 2099 MB/s | 183 MB/s |
| 1 MiB randread QD16 | 1996 MB/s | 148 MB/s, p99 246 ms |
| model file, contiguous extents | 545-547 MB/s | (not used: 6-13x slower) |

The Kioxia collapses to a degraded state after ~10 s and stays there. It is not thermal (a cliff,
not a ramp). Do not put the model there.

## 4. Model byte budget (from the GGUF header)

| item | bytes |
|---|---|
| MoE experts, all 40 layers | 12.188 GiB (330.00 MiB/layer Q3_K on 0-9 & 30-39, 294.00 MiB/layer IQ3_XXS on 10-29) |
| non-expert weights (attn, SSM, shared experts, routers, embeddings, output) | 1.454 GiB |
| KV cache @ 200k, k=q8_0 v=q4_0 | 1.55 GiB (8320 B/token) |
| KV cache @ 65k | 0.51 GiB |
| SSM recurrent state, per sequence | 60 MiB |
| compute buffer | ~0.72 MiB per ubatch token (1513 MiB at ubatch 2048) |

Cross-check: 1489 MiB weights + 1587 MiB KV + ~450 MiB compute = 3526 MiB, and the VRAM base
measured on this box is 3473 MiB. The accounting is consistent, which is why the planner's VRAM
predictions can be trusted.

## 5. Planner output on this box

`amp-plan` (default: 200k context, q8_0/q4_0, detected budgets):

```
expert layers gpu 4 of 40  (layers 36..39)
ubatch            2048
vram planned      5.65 GiB  (fixed 2.93 + gpu experts 1.29 + compute 1.44)
resident (cache)  6.69 GiB across 66 ranges
streamed tail     4.21 GiB across 42 ranges
predicted         prefill 225 t/s, decode 12.4 t/s

   g  ubatch      vram  fits  stream/ub    pp t/s   tg t/s
   4    2048   5.65 GiB   yes   4.20 GiB     225.1     12.4
   6    1024   5.58 GiB   yes   3.56 GiB     215.8     13.5
   3    2048   5.33 GiB   yes   4.53 GiB     224.0     11.8
   5    1024   5.26 GiB   yes   3.88 GiB     213.9     12.9
```

The planner independently rediscovers the measured llama.cpp optimum — a large ubatch beats
GPU-resident experts, because misses per token scale as `1/ubatch` while GPU residency only helps
decode — and then goes one step further by weighting decode, which moves the choice to 6 GPU layers
at ubatch 1024 (215.8 t/s prefill, 13.5 t/s decode).

For reference, measured llama.cpp on this box: 45-86 t/s prefill steady state, 11-14 t/s decode,
243 t/s prefill when the expert set is fully cache-resident.

## 6. amp vs llama-server, head to head

`scripts/head2head.sh` — identical prompt (52,713 bytes = 18,265 tokens), identical quant, identical
box, page cache warmed first so both engines start from the same state. llama-server runs the best
config from `../RUN.md` (`-ncmoe 38 -ub 2048`); amp uses the planner's choice (6 expert layers on the
GPU, ubatch 1024).

| | amp | llama-server | speedup |
|---|---|---|---|
| prefill, 18,265 tokens | 175.0 s = **104.4 t/s** | 535.4 s = **34.1 t/s** | **3.1x** |
| decode, 128 tokens | 5.01 s = **25.6 t/s** | 26.9 s = **4.76 t/s** | **5.4x** |

For reference, the best llama.cpp number ever observed on this box was 243 t/s prefill with the whole
expert set page-cache resident and 45-86 t/s in the steady state the user actually runs in.

### What actually produces the win

Not the prefetcher, at least not for prefill — the *plan*. Putting 6 expert layers on the GPU instead
of 2 shrinks the CPU-resident expert set from 11.54 GiB to 10.25 GiB, which is the difference between
a working set that thrashes a 9-11 GiB cache and one that mostly fits. The prefetcher then covers the
remainder.

### The decode cliff (10x, and it is a memory-planning decision)

| CPU-resident expert bytes | page cache | decode |
|---|---|---|
| 10.25 GiB (6 GPU layers) | 11.4 GiB — fits | **25.6-29.9 t/s** |
| 11.54 GiB (2 GPU layers) | 9-11 GiB — does not fit | **2.8-4.8 t/s** |

Because the working set is scanned cyclically, exceeding the cache by 1.3 GiB does not degrade decode
gracefully, it collapses it ~10x. The cost model now models this as a cliff
(`cache_overflow_tolerance`, `decode_fault_latency_us = 28 us`) instead of a linear penalty, and
`amp-plan` prints the "tg now" column next to "tg t/s" so the effect of other RAM users is visible.

## 7. Quality parity (`scripts/parity.py`)

amp runs llama.cpp's own ggml kernels on the same weights with the same KV dtypes (q8_0/q4_0), the same
flash-attention setting and the same sampler. There is no algorithmic difference to find. The harness
measures what *is* allowed to differ, in three parts:

| test | top-1 agreement | max abs logprob delta | generated text |
|---|---|---|---|
| amp g=6 twice (determinism) | 24/24 | **0.000000** | identical |
| amp g=6 vs amp g=0 (experts on GPU vs all on CPU) | 24/24 | 0.470572 | identical |
| amp g=6 vs llama-server `-ncmoe 34` (same placement) | 22/24 | 2.413935 | diverges after position 22 |

Interpretation:

* amp is bit-deterministic run to run.
* Moving expert weights between CPU and GPU shifts tail logprobs by up to ~0.5 and does not change
  the output. llama.cpp has exactly the same property between its own `-ngl` values, so this is
  floating-point non-associativity, not a difference in what is computed.
* The single cross-engine argmax flip happens at a position where the top-1/top-2 gap is 0.37 — inside
  the same drift band. The second "flip" (position 23) is meaningless because the contexts had already
  diverged at 22.

Protocol traps found while building this, all of which first looked like numerical bugs:

* Passing `top_k`/`top_p` together with `n_probs` makes the server renormalise the reported
  logprobs (it filters the distribution first). Differences of ~2.4 logprob appear that are pure
  protocol error. `temperature 0` alone is greedy and leaves the distribution intact.
* `--jinja` on a raw `/completion` request applies the chat template, changing the prompt.
* A genuine bug this harness caught: `generate()` originally decoded a spurious token before the
  first sample, shifting every generation by one. Text diffing alone did not make that obvious;
  comparing logprobs position by position did.

## 8. Open questions this bench raised

1. **How much of the working set can actually be pinned?** The 12.19 GiB set does not fit under the
   ~9.6 GiB practical cache ceiling while other processes hold ~5 GiB. amp needs to decide which
   ~9 GiB to keep resident and how to stream the rest without evicting it. That needs router
   statistics (M3+) so the pinned set is the hot one.
2. **Does explicit 440 KiB prefetch beat readahead on the fault path?** Both sit near 1.8-2.0 GiB/s;
   the win would have to come from overlap with compute, not bandwidth. Needs a real benchmark (M5).
3. **Decode is now cache-bound, so the next win is hiding its misses.** At 25-30 t/s the working set
   fits and decode is compute-bound (~1.79 ms per CPU layer predicts 13-16 t/s, measured 25-30 t/s,
   so the model is conservative). The interesting case is the *overflow* regime: 2.8 t/s at 357 ms per
   token is ~12,600 page faults serialising. Prefetching the 8 active experts per layer needs router
   output, which the llama.h path does not expose — that argues for amp building its own graph (M3b)
   rather than driving llama.h.
4. **Prefill regression to 104 t/s** in the head-to-head versus 166-170 t/s measured earlier for the
   same config. The difference is page-cache state: after warming 12.19 GiB only ~4.5 GiB stayed
   resident because other processes hold ~4 GiB, so the head-to-head was run in a partially cold
   state. Needs a controlled cold/warm matrix before quoting either number.

## 2026-09-26 — the server, and what the planner had to be corrected about

Reproduce with:

```bash
./scripts/fetch_deps.sh && ./scripts/build.sh          # vendored llama.cpp, static, 31 tests
./build/bin/amp-server --model $M --port 8085 --ctx 200000
python3 scripts/parity_test.py --url http://127.0.0.1:8085   # supersedes smoke_server.py
```

Server at the real 200k context, planner's own choice (4 expert layers on the GPU, ubatch 1024),
page cache otherwise quiet. Three identical requests, run back to back:

| request | decode | prefill | prefix cached |
|---|---|---|---|
| 1st | 11.35 t/s | 2.3 t/s | 0 |
| 2nd | 32.90 t/s | 22.4 t/s | 17 |
| 3rd | **35.25 t/s** | 27.3 t/s | 17 |

Start-up warm: 10.90 GiB in 3.7 s (2.94 GiB/s).

At 64k context (planner's choice: 7 expert layers on the GPU, 9.93 GiB CPU experts): 19.40 t/s on the
first request, then **30.21 and 31.11 t/s**. So 200k costs about 10 % of decode, not the 5x the first
request suggested.

**Steady state is the number, and the first request is not.** An earlier revision of this file
concluded from a single request that 200k decode was "6.89 t/s, cache-bound" and that the working set
did not fit. That was wrong: it measured the first pass, where every expert layer is still being
faulted in, and drew a conclusion about steady state from it. Three back-to-back requests differ by
3x. Anything reported from here on has to be a warm request, and the honest way to say so is to show
the sequence.

Two other things worth knowing, both measured today:

- `--gpu-layers 6` at 200k context does not work, and the failure is not graceful:
  `llama_init_from_model` fails to allocate a 782 MB compute buffer, and the over-budget fallback then
  serves requests with 178 MiB of VRAM free. The arithmetic says so beforehand — 200k of KV is
  1.55 GiB, dense weights 1.45 GiB, 6 expert layers 1.93 GiB, which leaves under 0.9 GiB for compute.
  g=4 is the most that fits, and that is what the planner now picks.
- The first request is worth its own optimisation pass (it is 3x slower than steady state). The warm
  helps but does not remove it, so the remaining cost is fault latency on the first pass over each
  layer plus CUDA graph and sampler setup, not bandwidth.

For reference, the llama-server baseline in the M3 head-to-head decoded at 4.76 t/s on the same
prompt with the same cache state: **35.25 t/s is 7.4x that.**

### Planner corrections found by running the server at 200k

1. **The VRAM safety factor was applied to the whole plan**, so a 6.05 GiB configuration was accepted
   against 5.89 GiB usable. Weights and the KV cache are exact byte counts from the GGUF; only the
   compute buffer is an estimate. The factor now applies to the compute estimate alone, and the total
   must fit. Effect: the runtime's VRAM backoff stopped halving the ubatch three times
   (1024 → 512 → 256 → 128) and the server comes up at ubatch 1024.
2. **The CUDA context's own cost was unmodelled** — 450 MiB, measured as the gap between the estimate
   and the 119 MiB the driver reported free after init at 200k context. Now in the fixed cost.
3. **The decode cliff could never fire.** The planning budget was the theoretical page cache
   (RAM − OS reserve ≈ 12.9 GiB), so every candidate looked fully resident, including the ones that
   thrash. The budget is now capped at 11.2 GiB, the largest working set ever measured resident here.
   With that, g=3 (11.54 GiB of CPU experts) correctly shows a 21.20 MiB/ubatch stream and a lower
   predicted decode than g=4 (10.90 GiB, fully resident).
4. **The ubatch backoff stopped at its floor while VRAM was still over budget**, accepting a context
   with 119 MiB free. It now says so explicitly and names the flags that would fix it.

## 2026-09-26 — serving correctness, measured

Reproduce: `./scripts/build.sh`, start `amp-server`, then `python3 scripts/parity_test.py --url ...`
(sections 6b, 6c and 7 below are the regressions; they are now sections 12, 14 and 15 of
`parity_test.py`, rewritten for llama.cpp's response shapes).
Sections 6b, 6c and 7 are the regression tests; each fails on the build that preceded its fix.

### Concurrency: the failure and the fix

A `llama_context` is not reentrant. Before the task queue, four concurrent requests produced:

| stream | bytes | finish_reason | `[DONE]` | server |
|---|---|---|---|---|
| main (150 tok) | 0 | 0 | 0 | dead |
| side ×3 | 0 | 0 | 0 | dead |

After, one worker with a FIFO queue:

| stream | chunks | finish_reason | `[DONE]` |
|---|---|---|---|
| main (120 tok) | 123 | `length` | yes |
| side0/1/2 (10 tok) | 10 each | `length` | yes |

Queue stats afterwards: `posted=10 done=10 failed=0 depth=0`. The process stayed up. Cost of
serialization: a side request waits for the main stream, same as `llama-server --parallel 1`.

### Prefill across ubatches

`llama_process()` asserts a batch is no larger than `n_ubatch`; unsplit, every prompt over the ubatch
failed with `prefill batch full at token 2048` and a 500. After splitting in `n_ubatch`-sized chunks:

| request | prompt tokens | result |
|---|---|---|
| `/v1/completions`, 4000-word prompt | 4001 | 807 t/s prefill |
| `/v1/chat/completions`, 4000-word history | 4010 | 825 t/s prefill |

(The rate is high because the prompt is one repeated word — almost no real compute. The number that
matters here is that 4000+ tokens complete at all.)

### Reasoning split, and the thinking budget

Same request, three configurations, 200k-capable server at `--ctx 32768`:

| configuration | content | reasoning_content |
|---|---|---|
| default (thinking on), 200 tokens | `"\n\n4"` | 141 chars — the model thought, then answered |
| default, 32-token "title" request | `""` | 13 chars — spent the budget thinking, no content |
| `enable_thinking: false`, 24 tokens | `"Fixing a Segmentation Fault"` | `""` |

Two findings:

- **The split must start "inside" the block.** The template's generation prompt ends with `<think>\n`,
  so the model never emits the opening tag and a tag-triggered split never fires. That is how a
  conversation title came back as *"The user said \"hi\". This is a short, conversational greeting.
  According to the rules, I should cr..."* — a chain of thought delivered as the answer.
- **The closing tag arrives with variable whitespace**: `\n</think>` naturally, `</think>` when the
  budget forces it. Matching only the exact string silently puts the answer in `reasoning_content`.

**Forcing a close makes this quant loop.** With `reasoning_budget_tokens` at 16 or 75, a 150-token
answer came back as `2+2 = 4.\n</think>\n\n2+2 = 4.\n</think>\n\n2+2 = 4...` repeated. The budget is
therefore implemented (llama.cpp's sampler, armed by replaying the prefill) but **off by default**, and
`enable_thinking: false` is the supported way to get a direct answer. Measured cost of the decision:
one wasted iteration each way, versus shipping a generation change that alters every answer.

### Steady state at 200k, warm requests

Three identical requests on one server (`--ctx 200000`, planner's choice: 4 expert layers on the GPU,
ubatch 1024, 10.90 GiB warmed in 3.7 s):

| request | decode | prefill | prefix cached |
|---|---|---|---|
| 1st | 11.35 t/s | 2.3 t/s | 0 |
| 2nd | 32.90 t/s | 22.4 t/s | 17 |
| 3rd | 35.25 t/s | 27.3 t/s | 17 |

Against the `llama-server` baseline's 4.76 t/s on the same prompt and cache state, that is **7.4x**.
Report a sequence, never a single request.

## 2026-09-26 — quality baseline: the measurement noise floor is exactly zero

Before the server swap can be claimed lossless, two things have to be true: there must be a
reference to compare against, and the *measurement itself* must have a known noise floor.
Otherwise a non-zero KL afterwards is uninterpretable.

Captured with `scripts/kl_parity.py`, 512 greedy tokens, `n_probs=32`, no `top_k`/`top_p`
(filtering makes a server renormalise the reported logprobs and manufactures fake divergence).
Config: `-c 16384 --parallel 1 -fa on -ctk q8_0 -ctv q4_0 -ngl 41 -ncmoe 38 -b 2048 -ub 2048
-t 8 -tb 8 --jinja`.

| capture | server | elapsed | positions | mean captured mass |
|---|---|---|---|---|
| `llama-ref` | `llama-server` :8099 | 93.0 s | 512 | 0.998430 |
| `llama-ref-2` | same server, same flags, second run | 44.8 s | 512 | 0.998430 |

Comparing the two runs of the **same** server with the **same** flags:

| metric | value |
|---|---|
| top-1 agreement | 512/512 (100.00 %) |
| KL(A‖B) | mean 0.000000e+00, max 0.000000e+00 |
| KL(B‖A) | mean 0.000000e+00, max 0.000000e+00 |
| Jensen-Shannon | mean 0.000000e+00 |
| max abs Δ logprob on shared tokens | 0.000000e+00 |

**The noise floor is exactly zero, not "small".** llama.cpp at `temperature 0` is bit-identical
run to run through the HTTP API, on this model, with this quant. So the acceptance threshold for
the post-swap capture is not "below some tolerance" — it is *exactly zero*, and any deviation is
a real finding rather than jitter. This is a stronger and cheaper test than a statistical one.

`min` captured mass is 0.935363 at one position: a flatter-than-usual distribution where the top 32
tokens hold less mass. Harmless here because both sides are truncated identically and renormalised
over the joint support, but it is why the tool reports captured mass rather than only KL.

### The pre-swap server cannot be captured this way

The hand-rolled `amp-server` set `logprobs` to `null` unconditionally
(`src/server/openai_api.cpp:516`) — it never implemented `n_probs` at all. So a KL reference from
the engine being replaced is not obtainable over HTTP, and that capability is one of the things
the swap gains. The pre-swap engine's own quality evidence remains the top-5 logprob parity in
`scripts/parity.py` and the determinism result in the table above.

## 2026-09-26 — the server swap is bit-identical (KL = 0)

The hard constraint is zero quality loss, so the swap had to be *proven*, not asserted. Same
`kl_parity.py` protocol as the noise-floor section above: 512 greedy tokens, `n_probs=32`, no
`top_k`/`top_p`.

Three captures:

| tag | server | expert layers on GPU | ctx | ubatch |
|---|---|---|---|---|
| `llama-ref` | `llama-server` `-ngl 41 -ncmoe 38` | 2 | 16384 | 2048 |
| `llama-matched` | `llama-server` `-ngl 99 -ncmoe 32` | 8 | 32768 | 1024 |
| `amp-post-swap` | `amp-server` (preflight chose the layout) | 8 | 32768 | 1024 |

### Matched placement: exactly zero

`llama-matched` vs `amp-post-swap` — same placement, same ctx, same ubatch, same KV dtypes, so the
only variable is the server:

| metric | value |
|---|---|
| top-1 agreement | **512/512 (100.00 %)** |
| KL(A‖B) | mean 0.000000e+00, max 0.000000e+00 |
| KL(B‖A) | mean 0.000000e+00, max 0.000000e+00 |
| Jensen-Shannon | mean 0.000000e+00 |
| max abs Δ logprob on shared tokens | **0.000000e+00** |
| captured mass, both sides | mean 0.998339, min 0.906032 (identical) |

**Not "within tolerance" — exactly zero.** Replacing 2,200 lines of hand-rolled server with
llama.cpp's changed nothing about the arithmetic, which is the expected result given the same
kernels, weights and sampler, and is now measured rather than argued.

### Mismatched placement: how a correct result looks like a catastrophe

Comparing `llama-ref` (2 expert layers on GPU) against `amp-post-swap` (8) gives what looks like a
disaster, and is the single most misleading number in this file:

| metric | value |
|---|---|
| top-1 agreement | 58/512 (11.33 %) |
| JS | 0.6118 (bound is 0.6931) |
| max abs Δ logprob | 22.68 |

It is not a regression. Inspecting the captures position by position:

- the **first 10 tokens are identical**, and position 0's top-1 logprob differs by **0.012**
  (−0.1196 vs −0.1316)
- the sequences agree early and diverge later — 58/512 over the full run

That is butterfly amplification from a 0.012 numerical difference caused by a different CPU/GPU
split of the experts changing summation order. Once one near-tie flips, the contexts differ and
every subsequent position is a different question. The 11 % figure measures *placement drift*, which
`AGENT.md` already characterised as shifting tail logprobs by up to ~0.5; it says nothing about the
swap.

**Method rule this establishes:** a KL comparison between two engines is only meaningful when
placement is held fixed. amp's preflight *chooses* placement for speed, so "amp vs llama-server with
llama.cpp's default layout" is guaranteed to diverge and is not a quality test. Match the layout
first, then compare. The 0.012 at position 0 is also the right order of magnitude for placement
drift, which is a useful sanity check that the two captures differ for the stated reason and no
other.

## 2026-09-26 — post-swap benchmark: the 7.4x decode claim does not reproduce

The goal was explicit about this: re-measure decode on the new server rather than assume the
previously recorded 35.25 t/s carried over. It does not, and neither does the 7.4x.

`scripts/bench_server.py`, 52,442-char prompt (18,265 tokens), `n_predict=64`, `temperature 0`,
5 identical requests per engine, same machine, same session, alternating which engine went first.
Rates are the server's own `timings` object.

| engine | request 1 | steady decode (requests 2-5) | plan |
|---|---|---|---|
| `amp-server` | 26.01 t/s | **28.39 t/s** (29.18 / 28.70 / 28.09 / 28.04) | g=4, ubatch 1024, 10.90 GiB CPU set |
| `llama-server` (documented best) | **1.96 t/s** | **30.05 t/s** (29.05 / 30.12 / 30.10 / 29.99) | `-ncmoe 38`, ubatch 2048, ~11.5 GiB CPU set |

### The correction

**At steady state the two engines are the same, within noise** — amp 28.39 t/s vs llama-server
30.05 t/s, and llama-server is if anything ~6 % ahead on this particular run. The previously
recorded "35.25 t/s vs 4.76 t/s = 7.4x" compared **amp's warm state against llama-server's cold
one**. That is precisely the mistake this file's own methodology section warns against, and it
was committed anyway. `28.39 t/s` is inside the range `AGENT.md` already recorded for this box
("10.25 GiB of CPU experts -> 25.6-29.9 t/s"); the 35.25 figure was the outlier, not the norm.

### What amp's plan actually buys

Not raw steady-state decode. **Graceful degradation, and a working set that fits.**

`llama-server`'s documented best config puts ~11.5 GiB of expert weights on the CPU, which does
not fit the ~10.9 GiB the page cache can actually hold, so it thrashes: **1.96 t/s** on its first
request, then recovers once the OS has pulled the set in. amp's plan keeps the CPU expert set at
10.90 GiB, inside the budget, and never collapses — 26.01 t/s on its very first request, then
28-29 t/s.

That difference is worth having: a cold or contended cache costs llama-server 15x on the first
request and costs amp nothing. It is also the *whole* mechanism behind the "10x decode cliff" in
`AGENT.md`, now observed directly rather than inferred.

The honest summary: **amp is not faster than a correctly-configured llama-server at steady state.
It is faster when the page cache is under pressure, which on a 14 GiB box is most of the time.**

### Two preflight bugs this benchmark exposed

Both were invisible until the plan was run at 200k and compared against `amp-plan`:

1. **The plan ignored the user's `-c`** (fixed earlier): planning for 200k while serving 32k
   reserved 1.55 GiB of KV for a 260 MiB cache. Cost 4 GPU expert layers at `-c 32768`.

2. **The plan computed KV at f16 while the server ran q8_0/q4_0.** llama.cpp's field default is
   F16 (`common/common.h:587-588`), the preflight read `params.cache_type_*` for planning, and
   only later overwrote them with the measured q8_0/q4_0 baseline (Rule 5). f16 KV is **2.46x**
   the bytes, so at 200k the plan reserved 3.81 GiB for a 1.55 GiB cache. Measured at `-c 200000`:

       before   g=0  ubatch 384  kv=3.81 GiB (planned)  predicted 1.9 t/s decode
       after    g=4  ubatch 1024 kv=1.55 GiB (actual)   predicted 15.5 t/s decode

   That is a 2.7x smaller ubatch and all four GPU expert layers, from a planning input that
   disagreed with the configuration being applied. The rule now plans with the dtypes that will
   actually be in effect, using the same user-intent test as the rule that applies them.

Both bugs made the plan *pessimistic*, so they cost performance without ever causing a wrong
answer — which is the dangerous class of bug, because nothing fails loudly.

### Concurrency: llama-server's 4 default slots cost 44x on this model

Found while running the OpenCode harness, not by reading code. The harness stalled on its first
prompt for ~50 minutes, and the server log explained why:

    slot 1 | task 479 | prompt processing, n_tokens = 3756, t = 195.55 s / 19.21 tokens per second
    slot 2 | task 335 | n_gen = 138, tg = 0.64 t/s, tg_3s = 0.02 t/s

**`n_slots = 4`.** The server example sets `params.n_parallel = -1` ("auto", `common/arg.cpp:1400`),
which `tools/server/server.cpp:156-159` expands to four concurrent slots with `kv_unified = true`.
Every concurrent generation wants the same shared ~10.9 GiB CPU expert working set, and on a box
with ~10.9 GiB of usable page cache that is not sharing, it is thrashing. Decode fell from 28.4 t/s
to **0.64 t/s**, a 44x collapse, and prefill from 45 t/s to 19 t/s.

This is not a corner case. An agentic client has two requests open *by design* — OpenCode asks for a
conversation title while the main answer is streaming — so the default is a guaranteed slowdown for
exactly the workload this server exists for. The deleted hand-rolled server serialised generations
for this reason; llama.cpp's server does not.

The preflight now sets `n_parallel = 1` unless `--parallel N` is passed. Same 3-concurrent-request
test, after:

| | before | after |
|---|---|---|
| slots created | 4 | 1 |
| decode under 3 concurrent requests | 0.64 t/s | 21.26 / 29.87 / 33.46 t/s |
| requests completed | stalling | 3/3, all `finish_reason: length` |
| slot ids used | 0 and 2 concurrently | 0 only — they queued |

Requests now take turns instead of interfering, which is what `--parallel 1` has always meant.
`--parallel N` still works for anyone who wants the batching and accepts the memory cost.
