# Where the work is, 2026-09-30

`docs/STRATA-PORT.md` has the measurements and the reasoning. This is the state of the
tree, what shipped, and what is left.

## State

The vendored `third_party/llama.cpp` carries **two uncommitted changes**:

| file | change | why |
|---|---|---|
| `ggml-cuda/fattn-vec.cuh` | `__launch_bounds__(nthreads, 1)` -> `, 4` | 1.22x on the attention kernel, bit-exact at 200k, worth 1% of a token |
| `ggml-cuda/ggml-cuda.cu` | buffer get/set copy on the compute stream | correctness: they copied on `cudaStreamPerThread` and synchronised that, which is not ordered against the stream the data was produced on. Measured 0.8% the wrong way over 3 interleaved pairs, so it is not a speed change |

They are uncommitted because `third_party/llama.cpp/AGENTS.md` says not to commit without
explicit approval and that was not given for this tree. There is also a pre-existing
`ggml-cpu.c` patch that predates this work.

Everything in `amp/` is committed. `amp_tests` passes. Shipped configuration: `q8_0`/`q8_0`
KV, `g=3` expert layers, `ubatch 1280`, 8 threads, 18.0 t/s decode and 407 t/s prefill at
134k context on this box.

The two measurement instruments from this session are saved as
`docs/amp-split-instrumentation.patch` (per-split host phases, and per-op device time with
`AMP_NO_CUDA_GRAPH=1`) and removed from the vendored tree. Re-apply to reproduce sections 6
and 7 of `STRATA-PORT.md`.

## Bugs found and fixed

0. **`--seed` did not exist.** `RuntimeConfig` had the field from the start and nothing set
   it, so it stayed at `LLAMA_DEFAULT_SEED`, which `llama_sampler_init_dist` resolves from
   the system clock. With the shipped default sampler every run produced different text. Two
   runs with `--seed 1234` are now byte-identical.
0b. **Every token was accepted twice.** `llama_sampler_sample` accepts before returning and
   `generate()` accepted again. `score()` legitimately needs its own. Harmless today (every
   accept in this chain is a no-op), verified bit-identical with a fixed seed.
1. **`--score-file` did not read back what `--emit-score` wrote.** The fixture is token ids;
   the reader tokenised it as text, so the newlines became tokens. Two runs of the same
   binary disagreed on 127 of 128 positions. The whole quality gate depended on this.
2. **The top-k logprob tracking ran on the hot path by default**, sorting the whole
   248,320-entry vocab every token for a report nobody asked for. Now off by default, and
   `--logprobs-n 0` actually disables it (the old guard let a 0 fall through).
3. **The engine ran `q8_0`/`q4_0` KV, not the `q8_0`/`q8_0` it documents.** Every other
   declaration in the tree said `kQ8_0`; this one said `kQ4_0`. Cost of the fix, as
   `AGENT.md` predicted: KV grows 1.55 -> 2.03 GiB at 200k and ubatch drops 2048 -> 1280.
4. **`prefer_decode` was a documented option that nothing read** and had no flag. It now
   ranks on decode alone, with a test. It does not help - see below.
5. **Four `amp-infer --help` lines** that did not match the code, two of which the quality
   gate depends on reading correctly.

## Three of my own conclusions this session were wrong, and the corrections

- **The GDN bound was wrong by 15x.** `tools/gdn_bench` sized the SSM *state* (2 MiB/layer),
  put the recurrent layers at 0.8 ms, and `fused_gdn` was retired on that number. They take
  **11.9 ms**. The whole layer's weights are 3.84 MiB and the layers run at 9.3 GB/s against
  a card that streams at 128.
- **Then the recurrence turned out not to be the cost either.** Per-op device time shows
  `SSM_CONV` and `SSM_SCAN` do not appear in the list at all, under 0.01 ms per token. The
  11.9 ms is 341 plain matvecs.
- **The 9-26 ms "input copy" cost was never copies.** With the device drain measured
  separately, the host's copy phase is 0.51-0.56 ms and the same time appears on the device.
  That figure has been carried in this document for two sessions.

## Where a token goes, measured

At 135k context, 55.5 ms:

| | ms | share | measured by |
|---|---|---|---|
| attention | 18.9 | 34% | device drain, and an independent context sweep at 18.3 |
| recurrent layers | 11.9 | 21% | device drain |
| expert matvec | 16.3 | 29% | host launch phase, and `perf` |
| copies at layer boundaries | 0.6 | 1% | host copy phase |
| sampler, unexplained | ~4.4 | 8% | see below |
| outside the split loop | ~2.6 | 5% | not decomposed |

The device is busy 30.0 of 55.5 ms, so it is idle 46% of the token. The CPU's 16.3 ms of
expert matvec is exactly that idle window. They cannot be overlapped - layer N+1's input is
layer N's post-MoE output - which is also why the `hit_hook` seam has nothing to reclaim.

## Where this engine actually stands

Per decode token at 200k, from the per-split, per-op and per-tensor measurements:

| | bytes/token | device | rate |
|---|---|---|---|
| attention KV, q8_0, 10 layers | 2.176 GB | GPU | 80 GB/s |
| the LM head, q6_K | 398 MB | GPU | 62.9 GB/s in situ |
| expert weights, 8 of 256 per layer | 413 MB | CPU | 27.96 GB/s |
| recurrent projections | 103 MB | GPU | 29 GB/s |
| GPU total | 2.68 GB in 30.0 ms | | 89 GB/s |
| CPU total | 0.41 GB in 16.3 ms | | 25.3 GB/s |

**114 GB/s of a ~155 GB/s ceiling, 74% of it, with both processors on one memory path.**

The MUL_MAT item that was the largest live candidate is **closed, and it is not a kernel
problem.** `tools/matvec_bench.cpp` measures the op in isolation for the model's real
shapes and dtypes:

| tensor | ms | GB/s |
|---|---|---|
| `output.weight` (the LM head) | 2.24 | 178 |
| `blk.0.ssm_out.weight` | 0.028 | 123 |
| `blk.0.ffn_gate_shexp.weight` | 0.007 | 143 |

At or above the card's rate. And the vec matvec's `__launch_bounds__(..., 1)`, the same lever
that was worth 1.22x on the attention kernel and is provably bit-exact here, was swept at
1/2/4/8/16 and changed nothing - because a matvec launches one block per output row, so the
grid is 512 to 248320 blocks and the GPU is already saturated. Occupancy is the lever only
when the grid is small.

What is left of the 2.24 ms isolated against 6.63 ms in situ is contention, and it measures
directly: the same benchmark with 8 CPU threads streaming RAM at the rate the expert matvec
uses drops the head 10% and `ssm_out` 39%.

Prefill was the other half of the engine and had never been examined. It is 332 s at 200k,
eleven times a 512-token decode. Its expert work is 2.38 TFLOP per 1280-token ubatch, 37 of
40 layers on the CPU, in 3.17 s - **753 GFLOPS, 74% of this CPU's AVX2 peak.** The part is a
Ryzen 7 7735HS with `avx avx2 f16c fma` and no `avx512` and no `amx`, so there is no wider
ISA, and thread count does not help (8 threads 423/417 t/s, 12 threads 408, 16 threads
425) because it is FP-throughput-bound and SMT shares the FP units. DRAM is not the
constraint: prefill reads 14.6 GB per ubatch, 4.6 GB/s, a sixth of what the same path
delivers during decode.

**So both regimes sit at about three quarters of a hardware limit, for different reasons:**
decode on a shared memory path at 74%, prefill on CPU FP throughput at 74%.

## The g=4 question, settled

Three clean interleaved pairs said g=4 was 2.2% better on decode at an identical ubatch, and
I recorded it as available. It is not, because I had only measured decode:

| g | prefill, 34k tokens | decode, 1024 tokens |
|---|---|---|
| 3 (the default) | 79.1 s, 81.1 s | 57.27 s |
| 4 | 222.3 s, 220.6 s | 56.01 s |

**One extra expert layer on the GPU makes prefill 2.8x slower for 2.2% of decode.** At batch 1
a layer reads 8 of its 256 experts, so a GPU-resident layer reads 294 MiB and is nearly
free; at batch 1280 it reads all 256, and the GPU does 2 TFLOP of small per-expert GEMMs in a
layer whose neighbours are waiting on the CPU. g=6 fits at no ubatch the engine runs.

The planner's prefill-heavy weighting is the measurement, not conservatism. `--prefer-decode`
is kept because the documented option was dead, but it trades 2.2% of decode for 2.8x of
prefill and its help now says so in measured numbers rather than the cost model's, which was
wrong by 2.8x on this axis.

## What is left

**The sampler's 4.4 ms is closed, and it was never the sampler.** `llama-context.cpp:2087`
has the end-of-decode `synchronize()` commented out, which is upstream and correct: the host
returns while the graph runs, and the wait lands on whichever call blocks first, which was
`llama_sampler_sample`. `AMP_TRACE_DECODE` now times an explicit `llama_synchronize` right
after decode, and the number moves exactly: the sampler falls from 4.789 ms to 0.362 ms and
a 4.411 ms post-decode drain appears. That drain is a constant 3.4-4.7 ms independent of
context, prefetcher and CUDA graphs, it grows ~0.35 ms per GPU-resident expert layer, and it
**cannot be filled** - the next token needs the id the drain is producing.

**fused GDN is closed, for a measured reason.** The recurrence's own weights are 115 MiB/token
(`ssm_out` is 90% of it) and it has no `z` or `x_proj` tensor, so the earlier 775 MB guess was
wrong by 7x. The recurrence moves 120 MiB/token of state, which is 0.81 ms at the streaming
rate, and the elementwise ops it is decomposed into are <= 0.8 ms of real device time once the
instrument's own ~3.3 us per node is subtracted out of the ~3.3 ms they appear to cost. A fused
kernel would swap 0.8 ms of glue for a 0.81 ms single pass. It saves nothing.

**The one live item is the decode attention kernel, and it is at a structural local optimum.**
It reports 48% of the card's streaming rate. The cost is per query head - holding the KV bytes
fixed at one KV head and raising only the query heads gives 0.906 / 0.946 / 1.310 / 2.420 ms for
2 / 4 / 8 / 16 heads - and the marginal KV streams at 177 GB/s, so the redundancy is already
free. Every knob is at its best value: launch bound 4 of 2/3/4/6/8/12, split-K 5 of
5/10/16/24/40/80, `nthreads_KQ` 32 of 32/16/8/4, `REG:128 STACK:0`, row pitch 260 of
260/272/288, head dim 256 of 64/128/256.

The reason is a crossed constraint, not a missing trick. The tile loop runs a K pass then a V
pass with a barrier between, the four co-resident blocks march in lockstep so the memory system
sees all-K then all-V, and the two cannot be in flight together: a K tile per warp is 64
registers per lane against a kernel already at its 128-register cap, which would halve
occupancy, and a K tile per block is 33,280 B, so double-buffering it is 66 KB of an SM's 100 KB,
which would leave one block. And the good coalescing that produces 82.5 GB/s is the same thing
that puts 32 lanes on one 256-byte row, which is what forces the 5-shuffle reduction. Buying
the overlap costs 2-4x the occupancy to get 2x the overlap.

## Traps

- **`prompt200k.txt` and `p200k.txt` are different prompts.** The gate recipe uses
  `prompt200k.txt`; `p200k.txt` is unrelated text. Comparing a run against a stored reference
  with the wrong one reports 9 top-1 disagreements and a median KL of 8.4e-3 that look exactly
  like a numerical regression, and pure stock reproduces it too. Check the prompt file before
  believing a gate failure.
- The launch-bound second argument and the split-K fan-out are coupled. On this box the
  wave-efficiency search lands on 5 because 16 heads x 5 splits == 20 SMs x 4 blocks, which is
  a coincidence of the geometry rather than a property of the search; a sweep of 2/3/4/6/8/12
  puts `parallel_blocks` at 5/7/5/7/8/8, and the order the softmax is summed in follows it.
- `#pragma unroll` does not macro-expand, so a `#pragma unroll SOME_MACRO` fails to compile.
  An unbuildable variant measured as a stale binary, and two sweeps in this session were run
  against stale binaries before that was noticed. Always confirm the build relinked.
- `cmake --build build -D...` re-configures and persists the flag into the cache; a later plain
  `cmake -S . -B build` does not clear it. Check `CMAKE_CUDA_FLAGS` in `build/CMakeCache.txt`
  after reconfiguring.
- `/tmp` is tmpfs, so `p200k.txt` at 1 MiB and `prompt200k.txt` at 910 KiB are both
  page-cache resident and neither reads as a large file.
- **Do not `git checkout` a vendored file to drop a measurement instrument from it.** Doing
  that on `ggml-cuda.cu` also silently reverted the shipped compute-stream fix, and
  `grep -c ctx->stream` still reported 24 hits because it matches the unrelated
  `ctx->stream()` accessor, so the loss looked like it had not happened. Extract the
  instrument's hunks into a patch first and apply the saved shipped patch back, then confirm
  the diffstat is byte-for-byte what it was before instrumenting.
- A per-node CUDA event pair measures the **inter-kernel gap**, not the kernel, whenever the
  CUDA graph is disabled - which is the only way to get per-node device times at all here.
  With ~700 nodes per decode token the bubble dominates; `CONT` puts it at ~3.3 us per node.
  Use `AMP_OPS_MATCH` to price one op class at a time and treat unfiltered per-op tables as
  upper bounds.
