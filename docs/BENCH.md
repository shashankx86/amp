# BENCH.md, measurements taken by amp on this machine

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

## 1. Demand paging vs direct I/O: the number that matters most

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
depth. Latency, not throughput, is what collapses prefill. The fixes are residency control
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
processes at the time was 5.2 GiB, so the practical cache ceiling is about 9.6 GiB, below the working
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
| KV cache @ 200k, k=q8_0 v=q4_0 (historical) | 1.55 GiB (8320 B/token) |
| KV cache @ 200k, k=q8_0 v=q8_0 (current)  | 2.03 GiB - measured |
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

The planner independently rediscovers the measured llama.cpp optimum, that a large ubatch beats
GPU-resident experts, because misses per token scale as `1/ubatch` while GPU residency only helps
decode, and then goes one step further by weighting decode, which moves the choice to 6 GPU layers
at ubatch 1024 (215.8 t/s prefill, 13.5 t/s decode).

For reference, measured llama.cpp on this box: 45-86 t/s prefill steady state, 11-14 t/s decode,
243 t/s prefill when the expert set is fully cache-resident.

## 6. amp vs llama-server, head to head

`scripts/head2head.sh`: identical prompt (52,713 bytes = 18,265 tokens), identical quant, identical
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

Not the prefetcher, at least not for prefill, but the *plan*. Putting 6 expert layers on the GPU instead
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
* The single cross-engine argmax flip happens at a position where the top-1/top-2 gap is 0.37, inside
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
   output, which the llama.h path does not expose. That argues for amp building its own graph (M3b)
   rather than driving llama.h.
4. **Prefill regression to 104 t/s** in the head-to-head versus 166-170 t/s measured earlier for the
   same config. The difference is page-cache state: after warming 12.19 GiB only ~4.5 GiB stayed
   resident because other processes hold ~4 GiB, so the head-to-head was run in a partially cold
   state. Needs a controlled cold/warm matrix before quoting either number.

## 2026-09-26: the server, and what the planner had to be corrected about

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
  serves requests with 178 MiB of VRAM free. The arithmetic says so beforehand, because 200k of KV is
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
2. **The CUDA context's own cost was unmodelled**, at 450 MiB measured as the gap between the estimate
   and the 119 MiB the driver reported free after init at 200k context. Now in the fixed cost.
3. **The decode cliff could never fire.** The planning budget was the theoretical page cache
   (RAM − OS reserve ≈ 12.9 GiB), so every candidate looked fully resident, including the ones that
   thrash. The budget is now capped at 11.2 GiB, the largest working set ever measured resident here.
   With that, g=3 (11.54 GiB of CPU experts) correctly shows a 21.20 MiB/ubatch stream and a lower
   predicted decode than g=4 (10.90 GiB, fully resident).
4. **The ubatch backoff stopped at its floor while VRAM was still over budget**, accepting a context
   with 119 MiB free. It now says so explicitly and names the flags that would fix it.

## 2026-09-26: serving correctness, measured

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

(The rate is high because the prompt is one repeated word, so almost no real compute. The number that
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
  According to the rules, I should cr..."*, a chain of thought delivered as the answer.
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

## 2026-09-26: quality baseline: the measurement noise floor is exactly zero

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
the post-swap capture is not "below some tolerance" but *exactly zero*, and any deviation is
a real finding rather than jitter. This is a stronger and cheaper test than a statistical one.

`min` captured mass is 0.935363 at one position: a flatter-than-usual distribution where the top 32
tokens hold less mass. Harmless here because both sides are truncated identically and renormalised
over the joint support, but it is why the tool reports captured mass rather than only KL.

### The pre-swap server cannot be captured this way

The hand-rolled `amp-server` set `logprobs` to `null` unconditionally
(`src/server/openai_api.cpp:516`), because it never implemented `n_probs` at all. So a KL reference from
the engine being replaced is not obtainable over HTTP, and that capability is one of the things
the swap gains. The pre-swap engine's own quality evidence remains the top-5 logprob parity in
`scripts/parity.py` and the determinism result in the table above.

## 2026-09-26: the server swap is bit-identical (KL = 0)

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

`llama-matched` vs `amp-post-swap`: same placement, same ctx, same ubatch, same KV dtypes, so the
only variable is the server:

| metric | value |
|---|---|
| top-1 agreement | **512/512 (100.00 %)** |
| KL(A‖B) | mean 0.000000e+00, max 0.000000e+00 |
| KL(B‖A) | mean 0.000000e+00, max 0.000000e+00 |
| Jensen-Shannon | mean 0.000000e+00 |
| max abs Δ logprob on shared tokens | **0.000000e+00** |
| captured mass, both sides | mean 0.998339, min 0.906032 (identical) |

**Not "within tolerance", exactly zero.** Replacing 2,200 lines of hand-rolled server with
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
- the sequences agree early and diverge later, 58/512 over the full run

That is butterfly amplification from a 0.012 numerical difference caused by a different CPU/GPU
split of the experts changing summation order. Once one near-tie flips, the contexts differ and
every subsequent position is a different question. The 11 % figure measures *placement drift*, which
`AGENT.md` already characterised as shifting tail logprobs by up to ~0.5; it says nothing about the
swap.

The method rule this establishes is that a KL comparison between two engines is only meaningful when
placement is held fixed. amp's preflight *chooses* placement for speed, so "amp vs llama-server with
llama.cpp's default layout" is guaranteed to diverge and is not a quality test. Match the layout
first, then compare. The 0.012 at position 0 is also the right order of magnitude for placement
drift, which is a useful sanity check that the two captures differ for the stated reason and no
other.

## 2026-09-26: post-swap benchmark: the 7.4x decode claim does not reproduce

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

**At steady state the two engines are the same, within noise.** Amp 28.39 t/s vs llama-server
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
10.90 GiB, inside the budget, and never collapses: 26.01 t/s on its very first request, then
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
answer, which is the dangerous class of bug because nothing fails loudly.

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

This is not a corner case. An agentic client has two requests open *by design*: OpenCode asks for a
conversation title while the main answer streams. So the default is a guaranteed slowdown for
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

## 2026-09-27: acceptance: the real OpenCode client, 3 tool-using prompts

`scripts/harness/run.sh` drives the installed `opencode` CLI (`opencode run --auto --format json`)
against the running server, in a pristine copy of a small fixture project, three times.

| prompt | wall | exit | tool calls | result |
|---|---|---|---|---|
| `01-read-and-report` | 499.1 s | 0 | `read` | ok |
| `02-fix-failing-test` | 163.4 s | 0 | `edit`, `glob`, `read`, `shell` | ok |
| `03-add-feature-and-test` | 232.9 s | 0 | `edit`, `read`, `shell` | ok |

`exit 0` only means the client process succeeded, so the transcripts were read to confirm the model
actually did the work:

- **02** ran the suite three times (before, after, verify) and made exactly one edit, removing the
  `/ 100` from `total_value_cents`, then reported all four tests passing. That is the correct
  diagnosis and the minimal fix; it did not change the test to make it pass, which the prompt
  explicitly warned against.
- **03** made three edits (the new function reusing the existing `low_stock` helper, its import,
  and its test), ran the suite, and reported the real output.

Two things worth recording from the timings:

- **Prompt 2 ran 3x faster than prompt 1** (163 s vs 499 s). Prompt 1 pays the cold prompt: a
  several-thousand-token prefill of OpenCode's system prompt plus every tool definition, at
  12-15 t/s while the expert set is still settling. Once warm, the same model decodes at 22-24 t/s.
- The same run against `llama-server`, before the `n_parallel = 1` fix, **stalled ~50 minutes on
  prompt 1 alone**. That stall is what surfaced the four-slot thrash documented above. Reading the
  server log is what turned "the harness is slow" into "the harness is 44x slower than it should be".

This is the acceptance test the project's goals ask for: a real agentic client holding multi-turn
conversations with tool calls, which is how the server is actually used.

### Reasoning budget: measured, and the old "it loops" note needs qualifying

The reasoning budget is llama.cpp's sampler (`common_reasoning_budget_init`), reached through
`reasoning_budget_tokens`. It counts tokens spent inside the reasoning block and forces the end tag
when the budget runs out, so a short request cannot spend its whole allowance thinking.

Same question ("What is 2+2? Answer with just the number."), thinking on, `max_tokens` 100-120:

| budget | completion tokens | answer | samples |
|---|---|---|---|
| unrestricted (default) | 100 | `4` | 1, clean |
| 32 | 36 | `4` | 3, all clean |
| 16 | 20 | `4` | 1, clean |
| 8 | 12 | `4` | 2 clean, **1 returned `4\n</think>\n\n4`** |

So the mechanism works and is effective, 100 tokens down to 12, but a **very** tight budget is
genuinely risky: at 8 tokens, one sample in three came back with a visibly doubled answer. The
severe looping described in earlier notes (`2+2 = 4. </think> 2+2 = 4. </think> ...`) was measured on
the deleted hand-rolled server's sampling path and did **not** reproduce here; 4 of 4 samples at
16-32 were clean.

**Decision: it stays opt-in, off by default.** Not because it is broken, but because forcing a close
mid-thought alters generation for every request that sets it, and at tight budgets it can visibly
degrade the answer. `enable_thinking: false` remains the supported way to get a direct answer. It is
free of that risk because the template pre-closes the block instead of the sampler forcing it.

## 2026-09-27: where decode is *not* bound: expert placement is worth 3.4%

`README.md` and `AGENT.md` have carried M3c, amp's own ggml graph so the router is observable
and the 8 active experts per layer can be prefetched during decode, as "the only remaining decode
win" for months. Its thesis is that decode is bound by **CPU memory latency** on the expert reads.
That is testable without writing a graph: if it is true, moving expert layers from the CPU to the
GPU should help substantially, because VRAM is both faster and lower-latency than RAM.

`llama-server`, 200k context, identical flags apart from `-ncmoe`, page cache warmed with
`amp-warm --what plan` before each run, 3 requests each, steady state = median of requests 2-3:

| `-ncmoe` | expert layers on GPU | CPU expert set | steady decode |
|---|---|---|---|
| 40 | 0 | 12.19 GiB | **30.39 t/s** |
| 36 | 4 (amp's plan) | 10.90 GiB | **31.44 t/s** |
| 32 | 8 | 9.61 GiB | **OOM** — exceeds 6 GB VRAM |
| 28 | 12 | 8.32 GiB | **OOM** |

**Moving four of forty layers' experts from RAM to VRAM buys 3.4 %.** It also shrinks the CPU
working set by 1.3 GiB, so this is not placement-neutral in either direction.

**What this does and does not say.** It is a bound, not a proof: if decode were dominated by the CPU
expert-memory path, g=4 versus g=0 would have shown a large gap, and it shows 3 %. So the CPU memory
path is not the dominant cost, which substantially lowers the expected value of M3c. Prefetching
hides the latency of a path that is not where the time goes. It does not prove prefetch is worthless
(latency inside the CPU path could still matter), and it says nothing about a better *kernel
schedule*, which M3c would also allow. But building a graph on the strength of a hypothesis this
size would be betting on an unmeasured guess, and the honest next step is a per-layer decode profile
before any of it.

It also fixes the VRAM ceiling with a measurement rather than an estimate: **g=4 is the maximum at
200k context**, because g=8 does not fit. That matches the planner's choice and `AGENT.md`'s earlier note.

### The page cache cannot hold the CPU expert set

`amp-warm --what plan` after warming: `cache after 7.27 GiB (cached 10.99 GiB, available 11.20 GiB)`
against a 12.19 GiB CPU expert working set. **Roughly 4 GiB of it always streams from NVMe** on this
box. That is the structural reason decode sits at ~31 t/s rather than higher, and it is not
fixable by scheduling: it is 14 GiB of RAM against a 12.19 GiB cyclically-scanned working set.

### Decode across context lengths: 200k costs 2.6 %, and two measurements agree

The goal asked for decode at multiple context lengths, not just 200k. This matters here because
context size changes the *plan*: the KV cache grows into the same 6 GB of VRAM the expert layers
want, so the planner gives layers away as context grows. 4 requests each, steady state = median of
requests 2-4, same machine, same session:

| context | GPU expert layers | KV | steady decode |
|---|---|---|---|
| 8,192 | 8 | 65 MiB | **35.61 t/s** |
| 65,536 | 7 | 520 MiB | **35.27 t/s** |
| 200,000 | 4 | 1.55 GiB | **34.68 t/s** |

**Going from 8k to the full 200k target costs 2.6 %.** The plan really does give up expert layers
(8 -> 4) as the KV grows, and it barely matters.

That is the same answer as the `-ncmoe` sweep above, arrived at independently: there, moving expert
layers CPU->GPU was worth 3.4 %; here, the planner *chose* 4 layers instead of 8 and lost 2.6 %. Two
different experiments, same magnitude. That is the strongest evidence in this file that decode on
this box is not bound by where the expert weights live - and it is the measurement that most
undercuts M3c, whose entire premise is that the CPU expert path is the bottleneck.

### Decode on this box varies ~20 % between sessions

Steady-state decode at 200k, measured five times across this session:

    28.39   30.39   31.44   33.94   34.68   t/s          spread 22 %

The page cache is warmer later in a session, and other processes take RAM from it. This is the
methodological point the whole file keeps circling: **a single-session A/B is only trustworthy if
both engines are measured in that same session**, which is why the 28.39-vs-30.05 comparison was run
alternating on one machine rather than quoted from two different days. It also explains how the
project came to believe both "35.25 t/s" and "28.39 t/s" at different times without either being a
lie: they are different cache states. Neither is a speedup over llama-server.

## 2026-09-27: every quality-neutral decode lever, exhausted

Follow-up to the placement result above. The remaining levers were parallelism, prefetching, and
speculation. All three were measured; prefetching is recorded below in full.

### M3c: implemented, measured, and a tie

The section this replaces got the arithmetic wrong and reached the opposite conclusion. The
error is worth recording, because it is the second time in this project that a confident number
came out of a script nobody sanity-checked against a second method.

The original claim was that the `-ncmoe` sweep implies expert weight fetching is **3.7 %** of
decode. It is not. The script computed the share of expert bytes *removed* as
`1 - 1.29/12.19 = 0.894` when it is `1.29/12.19 = 0.106`, an 8.4x error, and everything
downstream of it was wrong:

| | original (wrong) | corrected |
|---|---|---|
| share of expert bytes removed | 0.894 | 0.106 |
| expert weight fetching | 3.7 % of decode | **~30-36 % of decode** |
| all experts on GPU | 1.04x | 1.46x, later **refuted**, see below |

So M3c was not obviously worthless, and it was built and measured rather than argued away.

**Implementation.** The milestone assumed an observable router required owning the whole ggml
graph, which is why it was called impossible through `llama.h`. It is not necessary. Inside
`ggml_compute_forward_mul_mat_id` the router output is *already host-resident*: `ids->data` is
dereferenced directly to build `matrix_row_counts`, because CPU-resident expert layers run the
router on the CPU. The active experts are therefore observable with no sync and no new plumbing.
The patch prefetches the head of each selected expert's weight matrix, and each thread prefetches
the expert it is about to compute, mirroring the existing `cur_a` chunking so no barrier is
needed. It lives in `third_party/patches/m3c-expert-prefetch.patch` and is applied by
`fetch_deps.sh`, because `third_party/llama.cpp` is gitignored and an edit made directly in that
tree is not version controlled.

Two variants were measured, because prefetching in the same node that consumes the data gives
almost no time lead:

- **same-node**: prefetch the experts this token just selected. Memory-level parallelism across
  the 8 selected experts, but microseconds of lead.
- **predictive** (negative depth): while layer L computes, prefetch what the *previous* token
  selected for layer L+1. Expert routing is stable across adjacent tokens and one layer of
  compute is ~0.9 ms here, four orders of magnitude more lead than a DRAM latency. This is the
  form the milestone actually promised.

**Measurement.** Same server process for both arms. `AMP_M3C_PREFETCH_FILE` is re-read at most
twice a second, so depth can be switched without a restart; a per-arm restart would measure
session noise rather than the prefetch. Arms interleaved and order-reversed per round, first
request after each switch discarded. 200k, warm cache, 2 requests per visit.

| variant | depth | result |
|---|---|---|
| same-node | 16 KiB | +0.9 % |
| same-node | 64 KiB | +0.7 % |
| same-node | 4096 KiB | -1.9 % |
| predictive | -16 KiB | -0.5 % |
| predictive | -128 KiB | +0.9 %, then **-0.2 %** over 4 further rounds |

Nine paired comparisons in total span **-1.9 % to +1.7 %**, mean about **+0.2 %**. The ordering
is not even consistent between rounds. **Tie.**

The 4096 KiB arm is what makes the tie interpretable rather than merely unexplained. That depth
issues ~13,400 prefetches per expert and cost 1.9 %, which is about what the loop's instruction
issue alone should cost. So the code path is definitely live: the hint fires, and firing it at
already-resident lines is nearly free. There is nothing left for it to fetch.

### Why: the expert matvec already runs at memory bandwidth

A tie is consistent with two opposite explanations, so the tie alone does not say which. Either
the bytes are already available (nothing to fetch), or dequant is the wall (earlier bytes cannot
help). `tools/kernel_bound.cpp` separates them by timing a plain 8-thread read of an expert
weight matrix against the real `ggml` `mul_mat` over the same bytes at batch 1, sweeping the whole
98 MiB tensor because one expert alone fits in L2 and would flatter the kernel.

Layer 20, `iq3_xxs`, 8 threads, with `ggml`'s thread count pinned explicitly:

| | rate |
|---|---|
| plain read of the same bytes | 28.69 GB/s |
| `ggml mul_mat`, batch 1 | **29.21 GB/s** (101.8 % of plain) |

**The expert matvec moves its weights as fast as a plain read of them.** The dequant work is
entirely hidden behind the memory traffic. Thread scaling is near-linear and saturates at 8:

| threads | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| `mul_mat` | 4.80 | 9.17 | 17.75 | **29.21** | 29.27 GB/s |
| plain read | 27.64 | 31.62 | 31.07 | 28.49 | 27.34 GB/s |

The 16-thread flatness independently reproduces the SMT cliff in the `-t` sweep above.

This is a stronger statement than "the hardware prefetcher already covers it". The kernel is not
leaving bandwidth unused, it is leaving *nothing* unused. There is no exposed latency to hide,
because it already runs at the speed the memory system delivers the bytes. **M3c cannot win, and
the reason is a property of the kernel rather than of the prefetcher.**

### How much of decode is the expert matvec, three ways

| method | share of decode |
|---|---|
| 349 MiB/token at the measured 29.21 GB/s = 12.5 ms of 34.57 ms | **36 %** |
| `-ncmoe` A/B, solved as `t = a + b` | 29 % |
| `-ncmoe` A/B, marginal effect per byte moved | 29 % |

Three independent routes agree inside their noise: **expert reads are about a third of decode**,
and that third is memory-bound with nothing left to prefetch. The other two thirds is not the MoE.

### Quality: provably unchanged

A prefetch is a hint and cannot alter a value, so this is structural rather than merely measured.
Confirmed anyway, prefetch off versus predictive 128 KiB, 512 positions, `n_probs=32`, with the
gate set to require *exactly* zero:

```
KL(A || B)  0.000000e+00 over 512/512   max 0.000000e+00   inf at 0/512
KL(B || A)  0.000000e+00 over 512/512   max 0.000000e+00   inf at 0/512
JS(A, B)    0.000000e+00
|delta logprob| on shared tokens: mean 0.000000e+00  max 0.000000e+00
top-1 agreement 512/512 (100.00 %)
captured mass, both sides: mean 0.998502  min 0.934985
```

Zero quality loss, at zero speed gain. Captures committed as `quality/m3c-off.json` and
`quality/m3c-on.json`.

### Threads: already optimal

`-t` varies generation threads, `-tb` held at 8. 200k, warm cache, 4 requests each, steady median:

| `-t` | steady decode |
|---|---|
| 4 | 27.95 t/s |
| **8 (default)** | **32.97 t/s** |
| 12 | 29.96 t/s |
| 16 | 21.19 t/s |

Eight is one thread per physical core on this 8C/16T part, and it is already the peak. Twelve and
sixteen are SMT siblings contending for the same execution units and are *worse*. The planner's
`ncpu / 2 = 8` default is right, so there is nothing to win here.

### N-gram speculative decoding: no win on this model

`--spec-type ngram-simple` is lossless - drafts are verified against the target distribution - so
it was the one remaining lever that does not touch the arithmetic. It does nothing here:

| prompt | baseline | ngram-simple 8/4 |
|---|---|---|
| "Count from one to forty", 64 tokens | 32.32 t/s | 33.36 t/s (+3 %, inside the 22 % session variance) |
| templated status-code list, 300 tokens | 33.51 t/s | 33.57 t/s (+0.2 %) |

The 300-token templated case was the fair test: n-gram prediction should pay best on output that
repeats a structure, and it still moved nothing. This model is too entropic at the token level for
drafting to find matches.

### What this leaves

Measured and spent: concurrency (**44x**, the only large win), the KV-dtype planning fix (2.3x on
plan quality), expert placement (**3.4 %**), context length (**2.6 %**), threads (**already
optimal**), n-gram speculation (**0.2 %**), expert prefetch (**0.2 %, a tie**).

**This reverses the earlier claim that VRAM is not the binding constraint.** That rested on the
3.7 % figure, and it was wrong. The corrected picture is that expert reads are about a third of
decode and are memory-bandwidth-bound on the CPU, so moving experts to the GPU removes both the
memory traffic and the CPU arithmetic for those layers. Measured: g=0 -> g=4 is **+3.4 %** for
10.6 % of expert bytes moved. The linear extrapolation that followed from this - g=8 worth ~+7 %,
every expert on the GPU 1.46x - **was wrong, and is retracted below.**

6 GB of VRAM cannot hold g=8, so on this box that headroom is unreachable. It is still real
headroom, and it is the only large multiplier left. This laptop's 4050 cannot be upgraded, so
this is a fact about the hardware rather than an avenue, but anyone reproducing this on a bigger
card should know that VRAM is where the win is, not threads and not prefetching.

A cheaper dequant kernel would **not** help the MoE: the dequant is already hidden behind memory
traffic, so the expert path can only be improved by moving fewer bytes, which means quantization,
which the project rules out.

### The per-op profile: the SSM layers were never the problem

This is the measurement that was missing, and it refutes the hypothesis the previous section
recorded. `perf` is installed (extracted from the local pacman cache, no sudo, no system install);
`perf_event_open` was verified to work at `perf_event_paranoid=2` for our own process first. The
binary is not stripped and carries 126 `ggml_compute_forward_*` symbols, so no debug rebuild was
needed.

Decode, 352,440 samples at 999 Hz, prefill pinned to 4 tokens so this is pure decode at 29.7 t/s:

| symbol | share of CPU cycles |
|---|---|
| `libgomp` (OpenMP barrier spin) | **53.6 %** |
| `ggml_vec_dot_iq3_xxs_q8_K` | 21.2 % |
| `ggml_vec_dot_q3_K_q8_K` | 16.6 % |
| `ggml_graph_compute_thread` | 1.2 % |
| `__vdso_clock_gettime` | 1.3 % |
| `libcuda` | 1.3 % |

**Essentially 100 % of the CPU-side decode work is the two quantized dot-product kernels.**
`ssm_conv`, `ssm_scan`, `flash_attn_ext`, `argsort` (the router) and `get_rows` do not appear at
all, at a 0.001 % reporting threshold. They are not slow; they are **not on the CPU**. Only expert
weights are pinned to CPU (`-ot ffn_.*exps.weight=CPU`), so all 30 recurrent layers, all 10
attention layers and the routers run on the GPU under `-ngl 99`.

So the previous note in this file, which called the recurrent layers "the obvious suspect for the
96 %", was wrong twice over: the MoE is not 96 % of decode, and the recurrent layers are not where
the CPU time is. The CPU's entire job during decode is the 36 layers x 3 matrices of expert matvec,
which the kernel-bound measurement above already showed to be running at memory bandwidth.

**The 53.6 % barrier spin is load balancing, not waste.** ggml is built with OpenMP, so a decode
graph of ~1000+ nodes costs ~1000+ barriers per token, and the 8 threads spin at each one. Making
them sleep instead is worse:

| `OMP_WAIT_POLICY` | steady decode |
|---|---|
| active (default, spin) | **29.56 t/s** |
| passive (sleep) | 22.53 t/s |

A 31 % penalty for sleeping. Never set `OMP_WAIT_POLICY=passive` on this box, and treat that 53.6 %
as the price of 8-thread efficiency on a kernel whose threads finish at different times.

### Agentic loop: the cache is not the problem

Throughput is the wrong metric for tool use. A tool-using client sends a large context once and then
sends that context plus a small delta every turn, so a turn's cost is dominated by *how much got
re-prefilled*, and the pain is the worst turn rather than the median.
`scripts/bench_agentic.py` measures that: 1 initial context plus N turns, each appending a ~1 KB
tool result, reporting the max turn and flagging any turn that re-prefilled the whole context.

18,265-token base context, 8 turns, 48 tokens generated per turn:

| `-cram` | median turn | max turn | full re-prefills |
|---|---|---|---|
| 512 (amp's clamp) | 2.3 s | 4.7 s | **0** |
| 2048 | 2.0 s | 4.5 s | **0** |
| 4096 | 2.2 s | 4.6 s | **0** |

**`cache_ram_mib` is not the cause of the tool-call stalls.** The hypothesis was that amp's clamp
from 8192 to 512 was thrashing the prompt cache and forcing full re-prefills. It is not: at every
setting the prefix is reused and no turn re-prefills. The clamp stays, because its stated reason
(that the prompt cache evicts the model's page cache) is a real trade against decode, but it is not
the explanation for the 234-second stall in the field.

Squeezing 5 GiB of RAM to imitate a memory-starved session also did not reproduce it: prefill stayed
at 383 t/s and the loop at 2.9 s median with 0 re-prefills. `MAP_POPULATE` at load refills the page
cache, so a fresh server is warm even when the box is tight.

### The tool-call stall, reproduced and fixed: it was the single slot

The stall reported from the field was 234 seconds of "prompt processing" on one turn. The
hypotheses above were all wrong. What actually happened is that **the model was testing the API it
was being served by**: a simple prompt, then a request to configure the client, then self-test
requests, then continued work. A self-test is a short unrelated prompt sent to the same server
mid-conversation.

amp forced `n_parallel: 1` to stop four concurrent slots thrashing the shared CPU expert set
(measured 0.64 t/s against 28.4). The thrash protection was right. Forcing a **single** slot was
not, and the consequence was the bug: with one slot, the self-test request is served *by the slot
holding the conversation*, and the conversation's cached prefix does not survive it. The server log
shows it exactly - a 34,774-token context, an 11-token probe, a release at `n_tokens = 216`, then
a 35,603-token re-prefill at 151 t/s.

Reproduced with `scripts/bench_agentic.py --interleave`, 35,001-token context, a self-test request
after every second turn, 6 turns:

| `--parallel` | full re-prefills | median turn | **worst turn** | total, 6 turns |
|---|---|---|---|---|
| 1 | **2** | 5.5 s | **110.6 s** | 239.3 s |
| 2 | **0** | 4.6 s | **6.3 s** | **27.1 s** |

**8.8x on the number a user actually waits through.** Two slots costs nothing in decode:

| `--parallel` | 1 | 2 | 4 |
|---|---|---|---|
| steady decode | 27.42 | 27.25 | 27.67 t/s |

All inside the run-to-run spread. The 0.64 t/s collapse needs several slots generating
**simultaneously** against the shared expert set, and an agentic turn never does that - it finishes
generating, then runs tools. So the fix keeps the thrash protection and drops the single-slot trap:
`n_parallel` now defaults to 2. `--parallel 1` still works for anyone who wants it.

Note this only reproduces at scale. At an 18,265-token context the same probe caused 0 re-prefills,
because the prefix still fits comfortably; at 35,001 tokens it cost 110 s. The bug was invisible at
the size the rest of this file benchmarks.

### An unexplained truncation, left open

While building the 35k fixture, a 104,884-character prompt (36,530 tokens, verified via
`/tokenize`) was reported by the server as processing 18,269 prompt tokens. Repeated content is not
a plausible explanation, since 4x the text tokenises to 106,239 tokens as one string. Something
truncates a long prompt to about half. Not investigated, and recorded here rather than forgotten:
if a real prompt is silently halved, that is worse than any speed question.

**This supersedes the 1.46x figure above, which was wrong in the same way the 3.7 % one was.** It was
a linear extrapolation from two A/B points, and placement does not stay linear. Measured directly,
200k context, decode and prefill on the same 18,265-token prompt:

| `-c` | GPU expert layers | prefill | decode |
|---|---|---|---|
| 200000 | 4 | 198.8 t/s | **28.89 t/s** |
| 64000 | 7 | **252.9 t/s (+27 %)** | 26.44 t/s |
| 16000 | 8 | (context too small to measure) | - |

**Moving expert layers to the GPU helps prefill a great deal and does nothing for decode - it makes
decode slightly worse.** At batch 1 the GPU GEMV cannot exploit its parallelism and simply competes
with the 30 recurrent layers already resident there; at batch 1024 it is far more efficient per byte
than the CPU. The GPU is not the constraint for decode and there is spare capacity for prefill.

So the placement ceiling is the measured g=0 -> g=4 gain of **+3.4 %**, not a compounding 1.46x.
Forcing more at 200k is not even possible: `-ncmoe 34` (g=6) and `-ncmoe 32` (g=8) both OOM at load,
so 6 GB caps this at g=4 to g=5. The remaining honest trade is context length against prefill rate:
`-c 64000` buys **+27 % prefill** for **-8.5 % decode** and 64k of context instead of 200k. Whether
that is worth it depends on how often you re-prefill, and for a cached agentic loop the answer is
usually no, because steady-state turns are decode-bound.

## 2026-09-27: prefill-first retune: q8_0/q8_0 KV and no RAM prompt cache

The objective: maximise prompt processing, keep decode above 15 t/s (aim 20+), switch the KV cache
to q8_0/q8_0, land it all at 200k context, and lose no quality.

### The measurement problem came first, and it invalidated my own sweep

Prefill on this box has a huge run-to-run spread. The same configuration, restarted three times,
measured **132.5, 178.4 and 311.3 t/s** on an identical 18,265-token prompt. Decode over the same
runs varied only 26.4 to 27.7. An entire `n_ubatch` sweep was run on single requests before this was
noticed, and every conclusion drawn from it was discarded.

The cause is the working set, and it changes how prefill has to be measured. Prefill runs at batch
~1024, which routes most or all of the 256 experts per layer through the CPU matmul, so it reads
essentially the whole 13.66 GiB of weights and thrashes a page cache holding about 11 GiB of it.
Decode at batch 1 touches 8 experts per layer - small enough to stay resident - which is why decode is
stable to 5 % and prefill spans 2.35x. `scripts/bench_prefill.py` now generates a distinct prompt per
measurement so nothing is served from the prompt cache, and reports the median with the spread beside
it.

### The win: the RAM prompt cache was evicting the model

`cache_ram_mib` is llama.cpp's prompt cache - anonymous RAM holding KV snapshots keyed by prompt.
The preflight clamped it 8192 -> 512 on the reasoning that a smaller cache would protect the model's
page cache. Measured, **512 is still far too much to do that.** Each 18k-token prompt caches about
200 MiB, so two or three fill it and the churn starts knocking out model pages.

Sustained prefill over 6 distinct 18k prompts, one server process, alternating to cancel drift, KV
held at q8_0/q8_0 so this is the lever in isolation:

| `-cram` | prefill median | prefill min |
|---|---|---|
| **0** | **244.8 t/s** | 192.2 t/s |
| 512 | 85.0 t/s | 77.0 t/s |
| 512 | 77.9 t/s | 54.3 t/s |

> **SUPERSEDED IN MAGNITUDE 2026-09-28 — the direction holds, the 3x does not. Re-measured
> at the same 200k context with ten distinct 18k prompts: `-cram 0` median 559.8 t/s against
> `-cram 512` median 509.5 t/s, so ~1.10x, not ~3x.** The original 85.0 t/s for `-cram 512` does
> not reproduce and never did under this protocol; the collapse shape below was one session.
> What survives is that `-cram 0` is not slower, which is why the default stands. Keep the
> 3x figure out of any summary; quote ~1.1x or quote the direction.

The shape that was read as the mechanism, re-run ten prompts deep:

| `-cram` | sequence of prefill t/s, prompts 1..10 | median |
|---|---|---:|
| 0 | 267, 519, 558, 562, 559, 564, 560, 562, 564, 560 | **559.8** |
| 512 | 288, 483, 517, 522, 496, 524, 528, 507, 495, 512 | **509.5** |

Both climb out of the cold first request and then hold. Neither collapses. The original
"180 -> 93 -> 82 -> 98 -> 110 -> 110" was the signature of a session that happened to be unlucky,
read as a mechanism. A 512 MiB cache holding 143 MiB entries fills after 3.6 prompts, so ten
prompts is where the effect *should* appear if it exists — and it does not.

**A caution about the original protocol:** `bench_prefill.py --tokens N` builds prompts that land
larger than N (8192 asked for, ~11.2k delivered). At 11.2k a 512 MiB cache needs 5.7 prompts to
fill, so a 5-prompt run cannot show the effect even if it is real. Any future test of this lever
has to run enough prompts to overflow the cache, which at 18k tokens means at least 4.

It costs nothing in agentic terms, which is what the previous note assumed rather than checked. At
35k context with a self-test request every second turn: `-cram 0` gives an **8.7 s worst turn**
against 10.0 s at 512, still 6/6 cache hits and 0 full re-prefills. Prefix reuse comes from the
slot's own KV, not from this RAM cache.

Still unexplained: the old configuration (`-ctv q4_0 -cram 512`) measured 205.1 and 229.7 t/s in a
back-to-back head-to-head, which is far above the 77.9-85.0 t/s that `-cram 512` gives with
q8_0/q8_0. The only difference is the V dtype, which should not move prefill by 2.5x, and that arm
also got g=4 where the q8_0 arm got g=3 - the wrong direction to explain it. Recorded as a loose
end rather than explained away. The `cram` lever itself reproduces cleanly at ~3x; this anomaly is
about the old arm being faster than expected, not about the new one being slow.

### n_ubatch and friends were already optimal

Measured on the same prompt, and the early single-request numbers for this were wrong:

| setting | prefill |
|---|---|
| **n_ubatch 1024 (planner's choice)** | **320.1 t/s** (first read; see the caveat above) |
| n_ubatch 768 | 141.0 t/s |
| n_ubatch 512 | 180.8 t/s |
| n_ubatch 256 | 133.6 t/s |
| n_ubatch 1536 (g falls to 2) | 248.9 t/s |
| n_ubatch 2048 (g falls to 1) | 242.7 t/s |
| `-tb 12` | 288.6 t/s |
| `-tb 16` | 221.0 t/s |
| `-tb 8` explicit | 173.0 t/s |

> **SUPERSEDED 2026-09-28 — the "above 1024 loses" half of this is now measured to be
> wrong, and the correction is in the 2026-09-28 section below.** The 2048 point is not a
> wash: it is 38% *faster* than 1024 under the same harness, and the displacement argument does
> not survive. The "below 1024 falls off a cliff" half still holds. The failure was measuring a
> `n_ubatch` sweep with single requests, which is the exact mistake that had already
> invalidated an earlier sweep in this project — and `n_ubatch 1024` is still the planner's
> choice, for a reason given in the 2026-09-28 section.

The compute buffer is not tradeable: below 1024 prefill falls off a cliff, and above it the GPU
expert layers the buffer displaces cost more than the batch size gains. More batch threads hurt.
These are single-request numbers and the ordering is what carries the conclusion, not the values.

### Quality: q8_0/q8_0 is 8.8x closer to f16 than what we shipped

Reference is **f16/f16**, llama.cpp's own default and strictly more precise than either arm. All
three arms at 32,768 context with placement forced identical at g=7 (`-ncmoe 33`), so the KV dtype is
the only variable. The committed captures are `quality/kv-f16.json`, `quality/kv-q8q8.json`,
`quality/kv-q8q4.json` and the control `quality/ctl-f16-{A,B}.json`.

**The engine is bit-deterministic.** Two independent runs of the identical f16/f16 configuration give
`KL = 0.000000e+00` both directions, `JS = 0`, `max|delta logprob| = 0`, 512/512 top-1, zero
flips. So every difference below is the KV dtype and nothing else.

The 512-position metric is saturated once the two arms start generating different text, so the
signal is in the positions before the first token flip:

| arm vs f16/f16 | first token flip | med KL | max KL | med &#124;delta logprob&#124; |
|---|---|---|---|---|
| f16 vs f16 (control) | none in 512 | **0.000e+00** | 0.000e+00 | 0.0000 |
| **q8_0/q8_0 (new)** | 25 | **2.065e-07** | 3.002e-04 | 0.7695 |
| q8_0/q4_0 (old) | 48 | **1.809e-06** | 1.809e-06 | 0.8546 |

**q8_0/q8_0 is 8.8x closer to f16 than the configuration we shipped**, which matches the published
ratio direction (q4_0 V about 7x further from f16 than q8_0 V).

The honest part: **neither q8_0 configuration is lossless against f16 on this model.** Both diverge
measurably - median |delta logprob| around 0.8 nats, median KL 2e-7 to 2e-6 - where published
measurements on other architectures report q8_0 KV as near-lossless. This is a `qwen35moe` hybrid
with only 2 KV heads at key/value length 256, and there is upstream work noting that uniform q8_0 KV
breaks specific architectures. **f16/f16 does not fit at 200k on this box at all** - it fails with
"failed to allocate compute pp buffers" even with 0 GPU expert layers, and first fits at 131,072. So
q8_0/q8_0 is not a choice between lossless and fast; it is the closest available to the reference.

### What the change costs and buys

At 200k context, no flags beyond the defaults:

| | old | new |
|---|---|---|
| KV dtypes | q8_0 / q4_0 | **q8_0 / q8_0** |
| `-cram` | 512 | **0** |
| GPU expert layers | 4 | 3 |
| KV size | 1.55 GiB | 2.03 GiB |
| prefill median | 205.1 t/s | **236.1 t/s** |
| decode median | 27.73 t/s | **28.25 t/s** |
| distance from f16 (med KL) | 1.809e-06 | **2.065e-07** |

Decode gains nothing and loses nothing: 28.25 t/s is far above the 15 t/s floor and the 20 t/s aim.
Prefill improves, and quality improves 8.8x. The price is one GPU expert layer, which the placement
work already showed is worth little to decode.

Caveat on the prefill column: a reversed-order rerun gave new 224.4 against old 229.7, i.e. the
opposite sign. So **the old-versus-new prefill difference is inside the noise** and only the
`-cram` lever (measured in isolation, alternating, ~3x) should be treated as established. The 205 ->
236 figure is what one ordering produced and is not a claim this document makes.

### A false green, fixed

`AMP_TEST_MODEL` is unset and every preflight test begins `if (!m) return;`, so **ctest had been
reporting "100% tests passed" while 7 of the tests did nothing at all.** That is how a stale
`cache_ram_mib == 512` assertion survived a behaviour change unnoticed. `build.sh` now points ctest
at the model and exits non-zero if there is not one, and the new expectations are asserted:
`cache_ram_mib == 0`, `cache_type_v == GGML_TYPE_Q8_0`, and both surviving an explicit
`-cram N` and `-ctv f16`.

## 2026-09-28: mining Strata, and a harness built but not yet run

### What Strata is, and why the ideas transfer

`/home/e0u/localhost/Strata` is a from-scratch engine for Qwen3.8-Flash-Next: 53k lines, 48
layers (36 Gated DeltaNet + 12 QSA), 512 experts/layer, top-10, its own CUDA kernels, an MTP
draft layer, a VRAM-resident expert cache, and a single-request Python server.

That model is **the same architecture class as Occamy**, which is what makes it worth mining
rather than merely reading:

| | Strata (Qwen3.8-Flash-Next) | amp (Occamy) |
|---|---|---|
| layers | 36 GDN + 12 QSA | **30 GDN + 10 QSA** (`full_attention_interval=4`) |
| experts / used | 512 / 10 | 256 / 8 |
| expert bytes per token | 663.6 MB | **349 MiB** |
| measured bandwidth | ~40 GB/s | **29.21 GB/s** (101.8% of a plain read) |
| CPU expert time per token | 16.2 ms | **12.5 ms** |
| token time | ~53 ms | **34.57 ms** |
| **expert share of a token** | 31% | **36%** |

The last row is the whole argument. amp had already found, from the other direction, that the
expert matvec runs at 101.8% of a plain read of the same bytes, i.e. bandwidth-bound with
nothing left to prefetch. Strata measured the same shape and then did something about it. Its
pool hides only 1.055 ms of 19.035 because the residual chain is strictly serial.

### The expert cache: what it actually returned, and why we are not building it

Strata's largest single idea is a VRAM-resident expert cache. It dominates the codebase - a
`STRP` profile format, a `(layer, expert)` residency table, five grouped CUDA kernels, five
wiring sites, an adaptive tier on its own thread.

Measured, from their own source comments: pool drain **19.076 -> 10.312 ms/token**, and end to
end **-2.7 ms on a ~48.7 ms token, i.e. -5.5%**, not the 36% the byte counts suggest. The hit
rate is `h = 0.6447` (ten-fold leave-one-out, `[0.6110, 0.6750]`; their in-sample 0.6573 and
single-prompt 0.4720 are explicitly labelled not-the-metric), the CPU's remaining half has to
hide under GPU work with its own 26.32 ms/token floor, and their first buffer ordering dropped
the drain from 18.2 to 10.2 ms while **the token did not move at all**.

Not worth building here, for three independently sufficient reasons:

1. **The ceiling is 5.5%.** That is the return on the most sophisticated idea in the project.
2. **VRAM does not allow it.** Their 4,096 slots x 1,382,400 B is 4 GiB of expert cache on a
   12 GB card. amp has 5.89 GiB usable and the current plan already spends 5.69 GiB of it
   (measured: "ubatch 1024, compute buffer 977.28 MiB, total VRAM 5.69 GiB of 5.89 GiB free"),
   leaving ~200 MiB - about 170 expert slots.
3. **llama.cpp's offload is tensor-granular.** `-ncmoe` and `-ot` are per-layer. Per-*expert*
   residency needs a new ggml op, a backend-scheduler change, and a kernel.

Recorded so the next person does not re-derive it: **this is the idea to revisit first if amp
ever stops being a llama.cpp derivative.**

### What is ruled out permanently: MTP speculation

Strata's MTP draft layer gives 1.6-1.8x (2.4-3.2 tokens per pass, 79-86% measured acceptance,
verified bit-identical by construction). A byte scan of all 13.66 GB of the Occamy GGUF found
**no `nextn`, `draft`, `mtp` or `eagle` tensor name and no such KV key**. There is no draft
head on this model. Not a tuning problem, not a flag.

### What IS open: n-gram prompt-lookup speculation

Strata's `SuffixDrafter` is llama.cpp's `ngram-map-k`, almost exactly: 4 candidate m-grams per
key n-gram (`COMMON_NGRAM_MAX_VALUES 4` against their `WAYS = 4`), per-candidate
`n_accepted` tracking for acceptance statistics, and a match-length gate (`min_hits`) that is
their "only where its measured acceptance says it pays". Defaults are 12-gram keys, 48-token
drafts.

Strata measured **6-11% on code edits, prose unchanged**, and recorded the negative result too:
forcing the lookup drafter whenever it proposed more than the MTP **lost 2-8% on ordinary
text**. So the question is not "is it faster" but two questions - does it help where it should,
and does it hurt where it should not.

**Status: measured. Both configurations are a loss. See below.**

```
python3 scripts/gen_spec_prompts.py                  # fixtures, derived from live source
setsid nohup scripts/sweep_spec.sh 3 5 > /tmp/opencode/sweep.log 2>&1 < /dev/null &
python3 scripts/summarise_spec.py /tmp/opencode      # paired differences + a noise verdict
```

Three things in the harness that are not obvious, each from a way this measurement could
otherwise have lied:

- **Readiness is a real completion, not `/health`.** `/health` answers 200 about two seconds in,
  while the 13.66 GiB model is still being read off disk. The first version used it and would
  have benchmarked an empty server while producing numbers that looked fine.
- **The arm label must be corroborated by the server's own log.** The per-request
  `speculative.type` override is compiled out (`tools/server/server-schema.cpp:197` is `#if
  0`), so each arm costs a process start and a full model load. A driver that died between
  restarting the server and writing its state file would mislabel every number, silently.
  `bench_spec.py --require` refuses to record numbers it cannot corroborate.
- **Summarise pairs on the sweep's pair index and reports the spread, not a ratio of pooled
  medians.** With a 22% session-to-session drift, a ratio of pooled medians charges the drift
  to whichever arm ran in the faster half of the afternoon. When the per-pair spread is wider
  than 10 pp the summary prints that the effect is below this box's noise instead of quoting a
  mean. Unit-tested against synthetic signal, synthetic noise, and a deliberately unmatched run.

The code workload is generated from `src/plan/preflight.cpp` by brace-matching a live function
rather than embedding a copy - the first version quoted a function that had already been
rewritten, so it was asking the model to reproduce code that no longer existed.

## 2026-09-28: n-gram speculation, measured, and it is a loss

Two arms, 3 paired sessions each, alternating base/speculative to cancel the 22% session
drift, 5 requests per session with the first dropped, 128 generated tokens, `-c 200000`,
decode only (`cache_prompt: true`, so the slot serves the prefix and requests 2+ prefill
4 tokens). Every number below is the engine's own `timings.predicted_per_second`.

| workload | arm | drafted | accepted | median t/s | paired vs base |
|---|---|---:|---:|---:|---:|
| code | base | - | - | 35.38 | - |
| code | `ngram-map-k` (default, 12-gram key) | **0** | - | 35.09 | -0.5% (tie, 4.7 pp spread) |
| code | `ngram-map-k` (3-gram key, m=16) | 936 | 26.9% | **26.12** | **-26.5%** (1.4 pp spread) |
| prose | `ngram-map-k` (default) | 48 | **0.0%** | 34.67 | -0.9% (3.4 pp spread) |

**Neither configuration helps. The 3-gram one costs a quarter of decode throughput.**

### The first result was not a null result, it was a broken experiment

`ngram-map-k` at its defaults uses `size_n = 12`: a **12-token exact** key. On a 128-token
answer the engine drafted **zero** tokens on the code workload, and 48 tokens with **zero
accepted** on prose. The `-0.5%` above is two arms of nothing, not a tie between two
configurations.

Strata's SuffixDrafter keys on a **trigram** (`WAYS = 4` candidates, `min_match = 3`,
drafts up to 5). That is a completely different matcher from llama.cpp's default, and
importing Strata's 6-11% while running llama.cpp's defaults would have measured the wrong
thing while looking like a refutation. Retested with `--spec-ngram-map-k-size-n 3
--spec-ngram-map-k-size-m 16`, the drafter fires, and the result is the -26.5% above.

**The harness now refuses to report a speculative arm that drafted nothing**
(`bench_spec.py` exits non-zero with "the drafter never fired"), and `summarise_spec.py`
prints the draft count and acceptance beside every comparison and classifies the effect
against the per-pair spread rather than just checking the spread is small. An earlier
version printed "separable" for a -0.5% effect with a 4.7 pp spread, which is a claim about
the noise and not about the effect. Unit-tested against tie, never-fired, real-effect and
single-pair cases.

### Why it loses, from the measurement with no cost model in it

From the engine's own accounting: 78 drafted, 21 accepted, **mean accepted run 2.75**, so
7.6 verify passes produced 21 tokens and each pass proposed ~10.2 tokens. A pass therefore
costs 38.3 ms/token x 2.75 = **105 ms**, against **28 ms** for a one-token pass.

**A ~10-token verify window on this model costs 3.7x a single-token pass. Break-even needs
an accepted run of 3.7 consecutive tokens; the measured run was 2.75 - 1.35x too short.**

This is Strata's exact break-even condition failing. Their MTP drafter accepts 0.89 / 0.86 /
0.85 at steps 1-3, i.e. runs of 3+, so it clears 3.7. An n-gram drafter on a code answer does
not: the model paraphrases the function rather than reproducing it verbatim, so matches are
short and scattered. Strata's 6-11% came from a *learned* MTP head, not from the lookup.

Their cost model is also the reason this is close rather than catastrophic: "the dense
weights are read ONCE for T tokens", so a window amortises the dense read and only the
union of missed experts multiplies (1.75x at T=2, 2.40x at 3, 3.05x at 4). The measurement
confirms the amortisation is real - a 10-token window costs 3.7x, not 10x - which is why the
loss is 26% and not 90%. The 3.7x is the expert term: 16 tokens' worth of routing is close
to the whole working set, not a small multiple of one token's 8.

### Verdict

**Rejected. Do not enable speculative decoding on this model.** No n-gram configuration is
available that both fires and breaks even: the one that fires needs accepted runs of 3.7 and
gets 2.75, and the configuration that would need shorter runs is the 12-gram default that
never fires at all. Strata's own negative result predicted exactly this - "forcing the
lookup drafter whenever it proposes more than the MTP lost 2-8% on ordinary text" - and here
there is no MTP to compare against, so the lookup is always the loser.

The one Strata speculation idea that would work on this model is the MTP head, and there is
no MTP head: a byte scan of all 13.66 GB of the Occamy GGUF found no `nextn`, `draft`, `mtp`
or `eagle` tensor. Speculative decoding on Occamy is not a tuning problem.

### The last n-gram shape, and why the idea is now exhausted

Halving the verify window from ~10 tokens to ~5.5 was the only remaining shape worth
testing, and it also loses:

| window | drafted/pass | accepted run A | pass cost | break-even A | short by | decode |
|---|---:|---:|---:|---:|---:|---:|
| `m=16` (T~10.2) | 10.2 | 2.75 | 3.72x t1 | 3.72 | 1.35x | **-26.2%** |
| `m=5` (T~5.5) | 5.5 | 2.62 | 3.29x t1 | 3.29 | 1.26x | **-20.3%** |

The shape of that table is the finding. Halving the window cut the per-pass cost by only
12% (3.72x -> 3.29x) while the accepted run got slightly *worse* (2.75 -> 2.62), so the
deficit barely moved. **The accepted run length is set by the model and the workload, not
by the window size** - it is how often Occamy reproduces a span of its context verbatim
rather than paraphrasing it, and no flag changes that.

Break-even needs A ~ 3.3-3.7 consecutive accepted tokens. Strata's MTP head delivers
0.89 / 0.86 / 0.85 at steps 1-3, i.e. runs of 3+, and clears it. A lookup drafter gets 2.6
on this workload and cannot clear it at any window size. **n-gram speculation on Occamy is
closed.** No further tuning of `-lcs`, `-lcd` or `--spec-ngram-*` is worth spending on.

## 2026-09-28: the decode budget, measured rather than inferred

Strata's one strong lead that had not been tried here: **the recurrent mixers are probably
the dominant non-expert cost.** They measured the mixers at 55.8% of their GPU floor -
10.88 ms across 36 GDN layers and 4.56 ms across 12 QSA, corrected total 14.039 ms/token.
amp has 30 GDN + 10 QSA, and had never profiled the non-expert part of a token.

That number is from another model, so it is a hypothesis here. Measuring it needed three
things, two of which this box does not have.

### What the box can and cannot measure

`perf` is present at `/usr/bin/perf` but `perf_event_paranoid = 2`, so hardware counters are
not available to an unprivileged process. `perf record -e cpu-cycles` **silently degrades to
a software `cpu-clock` event** and still writes a data file, so a naive profile looks like it
worked. What comes out is 56% `clock_gettime` in libc and 37%
`pthread_cond_clockwait` - the CUDA driver's event handler thread spinning while it waits for
the GPU, which is the thing being measured and not a consumer of decode time. Sampling the
process with `-p` is worse than useless here. `-t <tid>` on the busy compute thread gives the
same two symbols. **There is no usable per-symbol CPU profile on this box without
`perf_event_paranoid` lowered, and there is no sudo.**

So per-symbol attribution is off the table. The question is still answerable, by subtraction
and by an A/B that does not need a profiler.

### The isolated kernel rate, on both layer kinds

`tools/kernel-bound` (it takes the model as `argv[1]`, not `--model`):

| layer | kind | `ggml mul_mat` batch 1 | plain read |
|---|---|---:|---:|
| 1 | GDN | 30.19 GB/s | 29.10 GB/s |
| 2 | GDN | 30.20 | |
| 3 | GDN | 30.20 | |
| 5 | GDN | 30.29 | |
| 6 | GDN | 30.21 | |
| 7 | GDN | 30.21 | |
| 9 | GDN | 30.10 | |
| 20 | attention | 29.52 | 28.99 |

Seven GDN layers and one attention layer all run at **~30.2 GB/s**, i.e. at the plain-read
rate, on both layer kinds. There is no layer whose matvec is unusually slow, and no
layer-kind difference in the expert path. Whatever the non-expert 60% of a token is, it is
not an expert kernel running badly.

### The budget, by subtraction

| | ms/token | share |
|---|---:|---:|
| measured token (35.4 t/s) | 28.25 | 100% |
| expert bytes: 349 MiB at 30.2 GB/s | 11.29 | **40%** |
| everything else | 16.96 | **60%** |

Strata's prior, scaled from 36 to amp's 30 GDN layers, predicts **~9.07 ms/token** of
recurrent-mixer cost against amp's 16.96 ms of non-expert time. Same order of magnitude, so
the prior holds as a prior - and the point of writing this down is that the number is now
amp's to measure rather than imported. The honest statement is: **40% of decode is expert
bandwidth, 60% is something else, and "something else" is not an expert kernel.**

`profile_decode.py` independently confirms the bandwidth side: 786% CPU across threads,
99.8% of it user time, **0.2% system** (so page-fault handling is not on the critical path),
0.14 MiB read from disk across the whole run, 0.1 major faults per token, 12.17 GiB of the
12.19 GiB expert working set resident in page cache.

### What this rules out, and what it does not

Ruled out by measurement: any expert-side latency trick. The matvec is at the memory rate,
the working set is resident, the disk is idle, and the kernel time is 0.2% of busy. There is
no exposed latency to hide and no bytes to stop re-reading.

Not ruled out, and now the only remaining place with 60% of a token in it: the 30 GDN
recurrent layers and the 10 attention layers, and the per-layer launch and sync overhead
around them. Strata's caveat is the relevant one - their GDN expense is ~13 latency-bound
kernel launches per block under **WDDM on Windows** (0.3-0.4 ms per graph launch, 96
launches per token), while on Linux with graph capture a launch is 0.805 us. amp is on
Linux, so the specific disease does not transfer, but the measurement that would settle it -
how many kernel launches per token, and what each costs - needs an instrument this box does
not have. Recorded as the next question with the reason it is still open, not as a finding.

## 2026-09-28: Strata's 2048-token prompt chunk beats amp's 1024, by 38%

Strata processes prompts in **2,048-token chunks** with the experts streamed to the GPU over
PCIe. amp's measured optimum was 1,024, and its `n_ubatch` sweep claimed 2,048 was 24% *worse*
(243 vs 320 t/s). That was measured with single requests, which this project has already
recorded as the mistake that invalidated an earlier sweep. Re-measured with
`scripts/bench_prefill.py` — five distinct ~8,192-token prompts, one server process, first
request reported separately by the script — the sign flips:

| `n_ubatch` | GPU expert layers | total VRAM | prefill median | spread |
|---:|---:|---|---:|---:|
| 1024 (planner's choice) | 7 | 5.57 / 5.89 GiB | 604.4 t/s | 3.42x |
| **2048 (Strata's chunk)** | 6 | 5.84 / 5.89 GiB | **836.5 t/s** | 1.28x |

**+38% prefill**, for one fewer GPU expert layer. Run again with `-ncmoe 33` to hold the
placement fixed and separate the two effects:

| arm | prefill median | spread |
|---|---:|---:|
| `-ub 2048 -ncmoe 33` (placement held) | 836.8 t/s | 3.16x |
| `-ub 2048` (placement free, g=6) | 836.5 t/s | 1.31x |

Identical. **The displacement is not the mechanism** — losing a GPU expert layer costs nothing
measurable here, because during prefill a 2048-token batch activates so many experts that the
extra resident layer is a rounding error. The win is the batch size itself, and it is the same
+38% whether or not the placement is pinned.

### Why the earlier sweep had it backwards

Three things differ from the 2026-09-27 run, and none of them is the hardware:

1. **`cache_ram_mib` is now 0.** The old sweep ran with the 512 MiB RAM prompt cache on, which
   is a measured ~3x on sustained prefill, and that penalty is *per prompt*. A 2,048-token
   ubatch against a 512 MiB cache that holds ~2.5 such chunks interacts badly; with no cache
   there is nothing to thrash.
2. **KV is q8_0/q8_0**, not q8_0/q4_0. Smaller, so a bigger compute buffer fits.
3. **Single request per point.** The old numbers were 1536 -> 248.9 and 2048 -> 242.7 t/s, from
   a sweep whose spread this project later measured at 2.35x. The current `n_ubatch 1024` arm
   still reports a 3.42x spread; its *median* is 604 t/s, nearly double the 320 t/s the old
   sweep recorded for the same configuration. The old sweep was reading cold-cache requests.

### What was given up, and what this does not claim

2048 needs a 1.91 GiB compute buffer against 1 GiB at 1024, and total VRAM rises to 5.84 of
5.89 GiB — **50 MiB free**, which is below Strata's 256 MiB "requests may stall" threshold and
below the 119 MiB at which amp's own preflight already warns. At 200,000 context the buffer
scales with the batch and this does not fit at all.

**So 2048 is right at short context and wrong at 200k, and the planner's 1024 is not a
mistake — it is the only value that fits the target configuration.** The honest conclusion is
that `n_ubatch` should be a function of context, not a constant, and that the constant was
chosen for the case that matters:

| context | best `n_ubatch` | why |
|---|---:|---|
| 32,768 | **2048** | +38% prefill, 5.84/5.89 GiB fits |
| 200,000 | 1024 | 2048's buffer does not fit; 1024 is the largest that does |

Not yet implemented: making the planner pick 2048 when the budget allows. It is a two-line
change to the same code that already sizes the buffer, and the measurement says it is worth
+38% of prefill for every user at a context where it fits. Recorded as the concrete next step
rather than a claim.

### Shipped: the planner now finds this by itself

`ubatch_saturation` 1024 -> 2048 in `CostModelConstants`, with the measurement in the comment
and an `AMP_UBATCH_SATURATION` override. No flags, `-c 32768`:

| | prefill median | decode median | GPU expert layers | compute buffer | total VRAM |
|---|---:|---:|---:|---:|---|
| **shipped (planner picks 2048)** | **828.1 t/s** | 36.98 t/s | 6 | 1.91 GiB | 5.84 / 5.89 GiB |
| `-ub 1024` for comparison | 601.2 t/s | 36.88 t/s | 7 | 1.27 GiB | 5.57 / 5.89 GiB |

**+37.7% prefill, decode unchanged** (36.98 vs 36.88 t/s). Decode is unchanged because
`n_ubatch` is a prefill batching parameter and the decode path reads the expert matvec at
batch 1, already measured at 30.2 GB/s against a 28.99 GB/s plain read.

At `-c 200000` nothing changes: the planner still lands on 1024, because 2048's buffer leaves
50 MiB. Verified the server loads and reports `n_ubatch: 1024`, 5.69 / 5.89 GiB, 234 MiB free.
The planner is now context-sensitive on this knob, and the VRAM check is what makes it so.

The test that pinned the wrong constant, `cost_model_ubatch_efficiency_is_monotonic`, asserted
`ubatch_efficiency(1024) == 1.0` — so it was *enforcing* the error. It now asserts saturation at
2048, that 1024 is strictly below it, and that the 2048/1024 ratio brackets the measured 1.38x,
so if the measurement stops being represented in the model the test fails rather than the
constant quietly staying wrong.

## 2026-09-28: the `cram` claim was overstated at 3x; it is about 1.1x

Not a Strata item — this is a claim this project made in its own earlier work, re-measured while
looking for the next thing and found to be wrong in magnitude. It matters because the same
number appears in `AGENT.md` as an established invariant, and a 3x figure in a durable rule is
the kind of thing that stops anyone from re-checking the next one.

Re-measured at the target 200k context, ten distinct ~18k prompts, one server process per arm:

| `-cram` | prefill t/s, prompts 1..10 | median |
|---|---|---:|
| **0** | 267, 519, 558, 562, 559, 564, 560, 562, 564, 560 | **559.8** |
| 512 | 288, 483, 517, 522, 496, 524, 528, 507, 495, 512 | **509.5** |

**~1.10x, not ~3x.** The earlier 85.0 t/s for `-cram 512` does not reproduce under this
protocol at any prompt count tried. Both arms climb out of the cold first request and then hold;
neither collapses. The "180 -> 93 -> 82 -> 98 -> 110 -> 110" sequence that was read as the
eviction mechanism was one unlucky session, and a shape read off a single session is not a
mechanism — the same mistake this project has already made twice and recorded.

**The direction survives, so the default is unchanged**: `-cram 0` is not slower and is slightly
faster, and at 4096 MiB (a plausible user value) the difference disappears entirely into the
noise. The claim is now written as ~1.1x everywhere, with the 3x marked superseded in place
rather than deleted, because "this was measured and was wrong" is more useful to the next
reader than a quietly corrected number.

**A protocol trap worth recording.** `bench_prefill.py --tokens N` builds prompts that land
*larger* than N — 8192 requested, ~11.2k delivered. At 11.2k a 512 MiB cache needs 5.7 prompts
to fill, so a 5-prompt run cannot expose an eviction effect even if it is real. This project's
own earlier test used 18k prompts, which is why the effect, if it exists, needs at least 4
prompts to appear. Any future test of a cache-vs-page-cache lever has to run past the point
where the cache fills, and check that it did.

## 2026-09-28: the v1 preset measured at long context, which required fixing the quality instrument first

Strata's finding to check: their 4-bit KV with a 256-point Hadamard rotation **costs real
precision, and the cost grows with context** - document perplexity +8% at 1K and +12% at 8K.
amp's `v1` preset is `q8_0` K / `q4_0` V and had only ever been measured at 32,768 context. So
the claim was untested here at exactly the place it would matter.

### The instrument was wrong first, in a way that would have produced a false result

`kl_parity.py capture` generates greedily and records the distribution at each position. Two
arms therefore sample their own tokens, so the first argmax difference puts them on **different
prefixes**, and every later position compares two different questions. Measured at 131,072 with
placement pinned: **444 of 512 positions "disagreed"**, median KL 1.7e-05, and the per-block
breakdown showed the distribution wandering with no relationship to the dtype. That number
measures the divergence cascade, not the KV dtype.

The fix is teacher forcing — score a *fixed* token sequence so every position is comparable.
Three changes made that possible, and each was a latent trap:

- **`ModelRuntime::score()`** (`src/runtime/model_runtime.cpp`) generates the same way but reads
  the next token from a supplied sequence instead of sampling it. The sampler is still fed each
  token, so penalties and any stateful sampler stay in step.
- **`--emit-score`** on `amp-infer` writes a reference run's tokens as a fixture, and
  **`--score-file`** makes every arm score those exact tokens.
- **`--logprobs-n`**, because `top_k_track_` was a hard-coded **5**. A token falling out of a
  5-wide window in one arm and not the other reads as a disagreement that is really the window's
  edge. With 5 candidates, teacher-forced `q8_0/q8_0` reported 92.2% top-1 disagreement while
  the underlying first-pair argmax agreed at positions 0, 1 and 2. It is now 32 and configurable.

### Controls first, because they decide whether the rest means anything

| control | result |
|---|---|
| f16 vs f16, same config, two runs | **byte-identical** |
| `q8_0/q8_0` vs itself, two runs | **byte-identical** |
| `q8_0/q8_0` at g=3 vs g=6 | median KL **1.217e-04**, 8/512 top-1 flips |

The engine is bit-deterministic on the scored path, and **placement alone moves the median KL
by 1.2e-04** — the same quantized matmul accumulated on CPU and GPU sums in a different order.
That is the floor any dtype number has to be read against.

### The result

512 teacher-forced positions, `-c 131072`, g=3 for every arm, 32-wide window, reference f16/f16.
**Every arm's f16 reference is the same fixture**, so the only variable is the KV dtype.

| arm | top-1 flip | median KL(f16‖arm) | mean | median KL(arm‖f16) | median Δ top-1 logprob | vs placement floor |
|---|---:|---:|---:|---:|---:|---:|
| `q8_0/q4_0` (v1) | 60/512 (11.7%) | **3.551e-03** | 1.152e-02 | 3.658e-03 | 0.0640 | 29.2x |
| `q8_0/q8_0` (v2) | 54/512 (10.5%) | **3.179e-03** | 9.520e-03 | 3.565e-03 | 0.0537 | 26.1x |

### What this does and does not say

**The dtype effect is real and an order of magnitude above the placement floor** — 26-29x it,
not 1.2x, so this is not float reordering in disguise.

**But v1 and v2 are within 12% of each other at 131k** (3.551e-03 against 3.179e-03), where at
32k v2 was 8.8x closer to f16. **The gap between q8_0 and q4_0 in V largely closes at long
context.** That is the opposite of Strata's shape — their q4_0 penalty *grew* with context
(+8% at 1K, +12% at 8K) — and it is consistent with what this model is: only 2 KV heads at
key/value length 256, so the V cache is small and the long-context cost lands in attention
rounding rather than in V's own error.

**So the honest reading of Strata's warning here is: the v1 preset is not measurably worse than
v2 at long context, and therefore is not the quality risk it was assumed to be.** v2 stays the
default because it is no worse and its 32k advantage is real, but `v1` is no longer a suspect
at 200k. What *is* real at 131k is that **both** q8_0 configs are 26-29x further from f16 than
placement noise, which is a larger gap than the 32k measurement suggested and is worth knowing
before claiming q8_0 is near-lossless on this model.

Per-block, the divergence does not compound with position (v1: 1.0e-02 early, 3.6e-04 late;
v2: 1.1e-02 early, 5.7e-04 late), so this is a stationary per-token rounding cost, not an
error that accumulates down a long context.

## 2026-09-28: two bugs in the measurement instrument, and the logits-scan idea does not transfer

Chasing Strata's "993 KB D2H plus a 248,320-float host NaN scan every token, 16% of the token"
needed `amp-infer --repeat` to average out run-to-run noise, and `--repeat` was broken twice over.

### Bug 1: a NULL context

`--repeat` reset its KV with `llama_memory_clear(llama_get_memory((llama_context *) nullptr), true)`
— a **null context**, which is undefined behaviour. `ModelRuntime::reset()` now uses the real
context, and `prefill()` calls it so every pass starts from an empty cache rather than appending
to the previous one's.

### Bug 2: the reported rate was wrong by exactly the repeat count

`pp_toks`/`tg_toks` were **assigned** from the last pass while `pp_sum`/`tg_sum` **accumulated**
across passes, so `--repeat 3` reported 512 tokens over three passes' worth of time. Every
repeated measurement read 1/3 of the true rate. That is why `--repeat 3` looked like a 3x
slowdown (9.5 t/s) while a single pass in the same session read 31.7 t/s — and it is a strong
reason this instrument was never used to average noise, because it reported a number that was
wrong by construction.

Fixed to `+=`. Now consistent: `--repeat 1` 19.27 t/s over 512 tokens, `--repeat 3` 29.71 t/s
over 1536 (the single-pass figure is lower because one 512-token pass still pays graph build).
**1536 tokens per measurement is the trustworthy number this instrument produces**, which is
what makes the next section possible.

### Strata's logits-scan finding does NOT transfer, and now it is measured rather than assumed

Their per-token host phase is 993 KB of D2H plus a 248,320-float NaN scan, 16% of a 53 ms token.
**amp's vocabulary is 3,115,143 tokens** — 12.5x theirs — so the same shape would be 11.9 MiB of
f32 logits per token, a 3.1M-element `partial_sort`, and a 3.1M-term `exp()` loop, every token.

Measured with the fixed instrument, 1536 tokens per arm:

| arm | decode |
|---|---:|
| `--logprobs-n 0` (no top-k capture at all) | 28.04 t/s |
| `--logprobs-n 32` (the 3.1M partial_sort + exp) | 28.87 t/s |

**Indistinguishable.** The whole vocab scan is inside the noise of a ~28 ms token, so it is
under ~3 ms and therefore not a lever. The reason Strata's version cost 16% and this does not is
scale in the wrong direction for them: 248k floats is a small fraction of a 53 ms token's
attention and expert work, while 3.1M pairs of `partial_sort` work is a different thing — but
measured, it still does not matter, because decode here is 40% expert bandwidth and the
remaining time is GPU-side layer work, not host-side scanning.

**Consequence: the `top_k_track_` 5 -> 32 widening stays**, because it is free and it fixed the
quality instrument. It is recorded here as measured-free rather than assumed-free.

## 2026-09-28: a g-sweep for the 60%, and an arithmetic error it caught

The open question was what the 60% of a token that is not expert bandwidth *is*. Strata's prior
says the recurrent GDN mixers; amp had never measured it, and `perf` cannot on this box
(`perf_event_paranoid = 2`, and `perf record -e cpu-cycles` silently degrades to a software
event so the output looks valid). So: measure the marginal cost of a CPU expert layer by sweeping
`g` and fitting. `amp-infer --repeat 3`, 1536 tokens per point, 2 runs per point, `-c 8192`.

| g | CPU expert set | ms/token | t/s |
|---:|---:|---:|---:|
| 0 | 12.19 GiB | 50.56 | 19.78 |
| 2 | 11.58 GiB | 37.74 | 26.50 |
| 4 | 10.97 GiB | 36.11 | 27.69 |
| 6 | 10.36 GiB | 31.36 | 31.89 |
| 8 | 9.75 GiB | 30.34 | 32.96 |
| 10 | 9.14 GiB | **28.06** | 35.63 |
| 12 | 8.53 GiB | 29.04 | 34.43 |

### g=0 and g=2 are not on the same curve, and fitting all seven is invalid

g=0 puts 12.19 GiB of experts on the CPU against a measured ~11.2 GiB usable page cache, and
this model has a documented decode cliff: 10.25 GiB gives 25.6-29.9 t/s, 11.54 GiB gives
2.8-4.8 t/s. So **g=0..2 is caching-bound and g>=4 is bandwidth-bound** — one line through all
seven points describes a cliff with a ramp. Fitting all seven gives `ms = -19.70 + 1.601*cpu`,
i.e. a **negative** intercept and experts at 175% of the token, which is physically impossible.
R² was 0.78, a perfectly respectable-looking fit on data that cannot be right. That is the
clearest argument in this project for refusing to read an intercept off a regression.

### In the planner's own regime (g>=6) the fit is clean

`ms/token = 15.41 + 0.461 * (CPU expert layers)`, **R² = 0.954**, residuals +0.27 +0.18 -1.18
+0.73 ms. At g=6 (34 CPU layers) that is 15.67 ms of expert-attributable time in a 31.36 ms token.

### An arithmetic error, caught by the sweep disagreeing with itself

My first pass divided **total** expert bytes by 40 layers, giving "312 MiB per layer" and 10.3 ms
per layer — 432 ms for a token measured at 28-50 ms, which is nonsense. **A token routes 8 of 256
experts per layer, so it reads 1/32 of each layer's bytes.** Corrected: 9.75 MiB per layer,
0.339 ms of bandwidth per CPU expert layer, 13.54 ms for a fully-CPU token against the
documented 349 MiB/token.

The correction matters because it is what makes the two estimates agree:

| estimate | expert | non-expert |
|---|---:|---:|
| (a) pure-bandwidth subtraction, 349 MiB at 30.2 GB/s | 37% | 63% |
| (b) g-sweep marginal slope, 0.461 ms x 34 layers | 50% | 50% |

**Two independent routes bracket the 40/60 split the earlier measurement reported, so that
conclusion stands.** What is new is (b): the marginal cost of a CPU expert layer is **0.461 ms
against 0.339 ms of pure bandwidth, so 1.36x**. That 0.12 ms per layer — about 4 ms per token —
is not memory. It is the per-expert work around the memory: gather, dequant, the group-scatter
back into the residual, and the launch per expert.

This is the first *measured* number on the other side of the 60%, and it is the same shape as
Strata's `moe_grouped_s2` motivation ("this cannot be a per-layer grouped kernel and the three-way
split is not an optimisation", h_layer = 0.0456 ruling out the cheap shapes). Their answer was to
group experts into one kernel and get 10.2 ms off a 19.0 ms pool drain. amp's per-expert overhead
is smaller in absolute terms but is the same phenomenon, and it is a kernel question rather than
a bandwidth question — the only kind left on this box.

### The finding with immediate operational value: g=10 is the turning point

**g=12 is slower than g=10** (29.04 vs 28.06 ms). Adding GPU expert layers stops paying at about
10 of 40 and then reverses, because the compute buffer grows ~80 MiB per GPU expert layer and the
GPU begins competing with the CPU for the same memory path. So `max_expert_layers_gpu` above ~10
is not a free safety margin — it is a region where the planner can choose a slower plan. The
planner's VRAM-driven choice lands at g=3..6 at 200k, well inside the paying region.

`--gpu-layers 14` is **refused** by the planner (fails model load) but prints the refusal to
stderr only, so a script that captures stdout sees an empty file and reads it as a hang. Worth
knowing: `amp-infer --gpu-layers 14` never starts, it does not run slowly.

## 2026-09-28: the 1.36x is not threads, and its cheap fix does not exist on this kernel

Two follow-ups on the g-sweep's marginal cost of **0.461 ms per CPU expert layer against 0.339 ms
of pure bandwidth**. If that 0.12 ms/layer (~4 ms/token, 13% of decode) is recoverable, it is the
last lever left, and it is a kernel question rather than a bandwidth one. Strata's answer to
exactly this shape is `moe_grouped_s2`, motivated by their `h_layer = 0.0456` ruling out the
cheap kernel shapes.

### Not thread count

g=6, `-c 8192`, `--repeat 3`, 1536 tokens per point, 2 runs each. nproc = 16.

| threads | ms/token | vs 8 |
|---:|---:|---:|
| 4 | 37.42 | +7.8% |
| 6 | 35.32 | +1.7% |
| **8 (planner default, ncpu/2)** | **34.72** | - |
| 12 | 35.94 | +3.5% |
| 16 | 43.75 | **+26.0%** |

**8 is optimal and 16 is 26% worse.** If the overhead were latency in the gather/scatter, extra
threads would recover some of it; they lose instead, which is the signature of a
bandwidth-and-barrier-bound path rather than an under-parallelised one. The planner's `ncpu/2`
default is right and there is nothing here.

### The mechanism, and why its cheap fix does not exist here

`ggml_compute_forward_mul_mat_id` (ggml-cpu.c:1550) parallelises **over output rows within one
expert** (`nchunk1`/`dr1`) and iterates experts **serially** in the outer loop. At batch 1 that is
8 sequential expert matvecs per layer, 40 layers, so ~320 serial chunks per token. And the
isolated matvec measures 29-30 GB/s on one *contiguous* expert tensor, while real decode does 320
*scattered* ~10 MiB reads out of a 12.19 GiB set. Scattered reads cost TLB walks, and the 1.36x
is consistent with that.

The cheap fix for TLB pressure is huge pages, and the expert set is 3.2 million 4 KiB pages:

```
/sys/kernel/mm/transparent_hugepage/enabled   ->  [always] madvise never
/proc/meminfo: FileHugePages:    0 kB          <- the GGUF mapping gets none
```

Tested directly rather than assumed (`/tmp/opencode/thp/thp.c`, 3 GiB file, mmap + touch, with
and without `madvise(MADV_HUGEPAGE)`): `madvise` **returns success** and `FileHugePages` stays
**0 kB**, and the plain-mmap control is also 0. **THP on file mappings is unavailable on this
kernel**, so the TLB fix cannot be had by advising the existing mapping.

The only route to huge pages here is Strata's: copy the experts into an **anonymous** arena and
advise that (`ArenaExpertSource`, 33.97 GB anonymous + `lock_resident`, measured 1.79x better
than a warm mmap). AnonHugePages does work on this box. **That is exactly the
`--load-mode none` path this project has already recorded as freezing the box** — a 14.75 GB
anonymous load, once. So the one remaining version of this idea is the one version already known
to be unsafe here.

**Conclusion: the ~4 ms/token of non-bandwidth cost in the expert path is real, measured, and
not reachable on this box by threads, by huge pages, or by any memory layout short of the
anonymous load that has already crashed it.** Writing a grouped expert kernel against
`ggml_compute_forward_mul_mat_id` would be the remaining move, and it is a genuine upstream-scale
change — a new `ggml` op plus a kernel — for a ceiling of ~13% of decode, on a path whose
measured penalty is 1.36x bandwidth rather than the 4-5x that would make it worth the build.
Recorded as the one open lever with its ceiling stated, not as work to do.
