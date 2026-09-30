# Findings from the Strata port, 2026-09-29

Starting point: the note in `AGENT.md` that M7 is spent and "there is no gap left to close
on the CPU side". That is correct, and it is also the reason the project stalled. The
optimization target was never the CPU. These are the measurements that moved it.

## 1. Decode is not compute-bound on either device

`scripts/decompose_decode.sh` samples one decode phase and reports both duty cycles.

| | measured |
|---|---|
| GPU kernels resident | 73-79% median |
| GPU DRAM bandwidth | ~30% of peak |
| CPU | 1.2-1.4 of 16 cores (7-8%) |

The CPU is nearly idle and the GPU is not saturating its memory system. Neither device is
the bottleneck on its own, which is the signature of a pipeline that is stopping and
starting rather than one that is working hard. Section 4 identifies what it is waiting on.

Every conclusion in `AGENT.md` about the expert matvec being "the whole CPU story" is true
and irrelevant: the CPU has 14 of its 16 cores free, so making the expert matvec faster
cannot make decode faster. That is why M7 read as "spent" while 10x sat on the table.

## 2. The GPU is not close to its bandwidth ceiling, but it is not 3.8x off either

Per decode token at 200k context, from the real GGUF geometry
(10 attention layers, 2 KV heads, head_dim 256, q8_0 = 34 B / 32 elems):

| term | bytes/token |
|---|---|
| KV read | 2.03 GiB |
| GDN state r+w, 30 recurrent layers | 120 MiB |
| **total GPU traffic** | **2.14 GiB** |

`amp-fa-bench` measures one decode attention in isolation, with no weights and no page
cache, against a read-dominated baseline measured on the same card in the same run:

| ctx | ms | GB/s | % of achievable |
|---|---|---|---|
| 32,768 | 0.540 | 66.0 | 39% |
| 65,536 | 1.065 | 67.0 | 39% |
| 131,072 | 2.118 | 67.3 | 40% |
| 200,192 | 3.228 | 67.5 | 40% |

Reproducible to well under a percent once the baseline is re-measured alongside each row,
which is necessary because the card idles at 315 MHz and cannot be clock-locked without
root on this box.

**A correction to an earlier draft of this file.** It claimed the GPU was "3.8x off its own
bandwidth ceiling" and that ~90 ms of a 100 ms token was spent moving KV. That was wrong,
and it came from dividing by a bandwidth figure that was never measured. The isolated
benchmark puts one 200k-context attention at 3.2 ms, so 10 of them is ~32 ms, and the
remaining token time is elsewhere. The 26%-of-peak figure from the duty-cycle sampler is
real but is a sampling artifact of `nvidia-smi` on a card that is mostly idle; it is not a
measure of what the attention kernel achieves, and the two must not be combined.

So the correct statement is narrower: the attention kernel runs at about 40% of what this
card can deliver, and it is flat in context length, which means the shortfall is per-position
work rather than per-byte work. That is a real kernel-efficiency gap, worth roughly 2 ms per
attention layer at 200k, but it is not the 10x the first draft implied.

## 3. The kernel llama.cpp picks, and why the obvious "fix" is wrong

`ggml_cuda_get_best_fattn_kernel` (`ggml/src/ggml-cuda/fattn.cu:541`) is called with
`cc = 8.9` (Ada), `Q->ne[0] = 256`, `Q->ne[1] = 1` at decode, and **quantized** K and V.
That takes the branch at `fattn.cu:650`, which returns `BEST_FATTN_KERNEL_VEC`. Verified
directly, not inferred, by tracing the chosen kernel at each context: `VEC` at every one.

`fattn-vec`'s inner loop (`fattn-vec.cuh:254-290`) does, per KV position, a `vec_dot_KQ`
over D=256, a `warp_reduce_sum`, and then a block-wide reduction through shared memory for
the running softmax max and sum. That is the per-position cost that keeps the kernel at 40%
of achievable bandwidth instead of near it, and it is why the achieved rate is flat in
context length: the work per position does not shrink as the context grows.

**The MMA kernel is not the answer, and it is worth being explicit about why**, because it
is the obvious thing to try and it is a trap. `MMA_F16` requires `need_f16_K` and
`need_f16_V` (`fattn.cu:731-738`), which means the whole q8_0 KV cache is dequantized to f16
on *every* decode step before the math runs. At 200k context across 10 layers that is:

| path | bytes moved per token |
|---|---|
| vec: read q8_0 directly | 2.03 GiB |
| mma: read q8_0, write f16, read f16 | 9.67 GiB |

5.06x the traffic, to do a matrix-multiply faster. The `vec` kernel exists precisely because
for small batches and quantized KV it is the right choice, and at this context length forcing
MMA makes things worse.

A first attempt at this fix did appear to show MMA 1.5x faster. That was an artifact: the
forced-MMA path crashed with an illegal memory access after the first context (it needs an
f16 scratch buffer the benchmark did not allocate), so the "MMA" column was one short-context
number compared against a full sweep. The fix was reverted. Timing a path that aborts early
is not a measurement.

## 4. Where a decode token actually goes, measured inside the scheduler

`perf` says 71% of host cycles are in the CUDA driver, mostly `cuMemcpyHtoDAsync` and
`cudaStreamSynchronize`. That is real but it does not say *why*. The scheduler was
instrumented to time three phases of every split - waiting for the previous split, copying
split inputs, and launching - and the result overturns the obvious reading.

Per split, on a warm 33k-context decode, 38 CPU splits and 38 CUDA splits per token:

| phase | CPU split | CUDA split |
|---|---|---|
| wait for previous split | **0.0001 ms** | 0.0000 ms |
| copy split inputs | **0.244 ms** | 0.027 ms |
| launch / compute | 0.387 ms | 0.014 ms |
| **total, per split** | **0.625 ms** | 0.041 ms |

**The synchronisation hypothesis is wrong.** `wait-prev` is 0.1 us, not 634 us. There is no
host-side barrier stalling, so there is nothing for an overlap schedule to reclaim. The
earlier "96% of the token is barrier latency" figure came from solving two equations whose
prefill and decode arms do not move the same bytes - prefill touches far more distinct
experts, so `w` and `B` were never separable that way. It is withdrawn.

What the 0.387 ms of CPU "launch" is: it is `ggml_graph_compute` on the CPU backend, which
is synchronous, so the expert matvec runs inline. 8 experts x 1.29 MiB = 10.32 MiB in
0.387 ms is **27.96 GB/s**, which is the 28-30 GB/s the existing notes already measured for
this path. `AGENT.md` was right: the expert matvec is at memory bandwidth, and a better
kernel or a deeper prefetcher cannot help it.

That still leaves a real and *stable* cost, and it is the one thing in this table that is
addressable:

**0.244 ms per layer of input copying, 9.5 ms per token, about 18% of a decode token.** The
bytes involved are trivial - 8 KiB down, 32 B down, 64 KiB up, 73,760 B per layer, 2.8 MiB
per token. At 0.244 ms for three copies that is 79 us per copy, where an 8 KiB
`cudaMemcpyAsync` should cost 5-10 us. The cost is the **number of driver round trips**, not
the bytes: 3 copies x 40 layers = 120 per token, each one a separate call into the driver,
and each `GGML_TENSOR_FLAG_INPUT` copy preceded by a stream synchronise
(`ggml/src/ggml-backend.cpp:1677-1684`).

Crucially this number is reproducible to within 1% across runs, unlike the launch time,
which swings 8 to 16 ms per layer with page-cache state. That makes it the one cost here
that can be improved and then *verified*.

### What the 121 copies actually are, and why two fixes for them did not work

The 121 inputs per token, from the scheduler's own copy loop:

| tensor | bytes | direction | per copy |
|---|---|---|---|
| `model.input_embed` | 8192 | CPU -> CUDA | 229 us (once) |
| `attn_post_norm-N (reshaped)` | 8192 | CUDA -> CPU | 34-162 us |
| `ffn_moe_topk-N` | 32 | CUDA -> CPU | 7-11 us |
| `ffn_moe_down-N` | 65536 | CPU -> CUDA | 64-77 us |

Two hypotheses were tested against this and both were wrong, which is worth recording
because the measurements look like a fix until you check where the time is inside them.

**Fix 1, wrong: the copies are on the wrong stream.** `ggml_backend_tensor_get/set` go
through the *buffer* interface, which has no handle on the backend's compute stream, so they
used `cudaStreamPerThread` and then synchronised *that* stream
(`ggml-cuda.cu`, `buffer_get_tensor` / `buffer_set_tensor`). Those two streams are not
ordered against each other, so the synchronise did not mean "the data is ready". Giving the
buffer context the backend's stream and copying on it is more correct and did cut the
individual read-backs from 44-162 us to 13-23 us. **The token did not get faster**: total
per-input time was 87.7 us before and 87.7 us after. The per-copy improvement was real and
was entirely hidden by waiting for the GPU further down.

**Fix 2, wrong: the time is a wait, not a copy.** Instrumenting the body of the copy loop
separated the wait issued at the top of the branch from the copy itself, and the wait is
**0.0 us**. Events are in use (`ggml_backend_event_wait`, not a stream sync), so nothing
stalls there. The 87 us is inside `ggml_backend_tensor_copy`, and for the D2H cases that is
`cudaStreamSynchronize` on the compute stream, which returns when the GPU has actually
produced the tensor. Those 8 KiB of `attn_post_norm` exist at the *end* of a layer's GPU
work, so reading them means waiting for that layer's kernels, and the wait is the GPU doing
its job, not overhead.

So the 0.244 ms per layer is mostly **the GPU's own latency being attributed to the copy
phase**, because the copy is the first place the host has to wait for it. It is not
reclaimable by making the copies cheaper, and it is not a barrier that an overlap schedule
can hide, because the CPU cannot start its expert matvec before the router has run.

**What remains true and useful:** the expert matvec is at 27.96 GB/s and is not the
problem; the copies are not the problem; and the per-layer cost is dominated by GPU work the
host is waiting on. Any remaining win has to come from making the GPU's per-layer work
faster or from overlapping it with the CPU's, and the two Strata ideas that would do that -
grouped expert GEMV, and the fused GDN step - are both kernel work, not scheduling.

## 5. Attention: a 22% speedup that is bit-exact at the target context

This is the one candidate with a large prize. It was built, measured, rejected, and then
reinstated, because the first rejection tested the wrong context. The whole sequence is
worth recording, since the error was mine and it was invisible without a control.

**The change.** `flash_attn_ext_vec` is declared

```cuda
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 1)   // fattn-vec.cuh:20
```

with 128 threads per block. The trailing `1` is a *minimum resident blocks per SM* hint,
which leaves 4 warps per SM for a kernel whose inner loop is a chain of dependent global
loads. Raising it to 4 is one token.

**The speed.** `tools/fa_bench`, one decode attention, this model's geometry, best of 9
passes, three repeats, spread under 0.5%:

| min blocks/SM | 32k | 131k | 200k |
|---|---|---|---|
| 1 (stock) | 0.541 ms | 2.112 ms | 3.229 ms |
| 4 | 0.439 ms | 1.727 ms | 2.638 ms |
| 8 | 0.481 ms | 1.878 ms | 2.856 ms |

1.22x, with an interior optimum, so it is not just "more is better". Ten of the forty
layers are attention layers, and at 200k the attention is 2.03 GiB of KV per token moving
at 67 GB/s, which is about 30 ms of a 45 ms token. So this is worth roughly **12% of a
decode token**, the largest single item found in this whole exercise.

### The first rejection, and why it was wrong

The change appeared to break the quality gate badly: teacher-forced, 32 logprobs over 128
positions, 12 of 128 top-1 disagreements, median KL 2.8e-3, max delta 2.74 nats. Against a
placement floor of 1.2e-4 that is 23x over, and the obvious reading was that the kernel's
partition of K had moved.

The reasoning behind that reading was sound: when `gridDim.y == 1` a block normalises its
own output (`fattn-vec.cuh:498`) and the summation order is fixed, while when
`gridDim.y > 1` it writes an unnormalised partial plus `(KQ_max, KQ_sum)` to `dst_meta`
(`:501`, `:512`) for a log-sum-exp reduce. `parallel_blocks` seeds that fan-out from
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` (`fattn-common.cuh:1131`), so a launch-bound
change can move the partition, and the partition is the arithmetic.

**The error was the context, not the reasoning.** The test prompt was about 1k tokens. The
fan-out at that size is small enough that the occupancy seed binds and the partition really
does move. The engine is used at 200k, where the seed does not bind. Instrumenting
`launch_fattn` in the real serving path:

| prompt | ntiles_KV | fan-out, min_blocks=1 | fan-out, min_blocks=4 |
|---|---|---|---|
| para (1k tok) | 3 | **2** | **3** |
| p8k | 9 | 5 | 5 |
| p32k | 33 | 5 | 5 |
| p64k | 66 | 5 | 5 |
| p128k | 131 | 5 | 5 |

From about 1k tokens of KV upward the wave-efficiency search settles the fan-out at 5 on
this card and the launch bound stops mattering. It was rejected on the one row where it
does matter, which is the row where the attention costs 0.038 ms.

### The isolation test that settles it

If the fan-out is the only channel, then holding it equal must make the launch bound
arithmetically invisible. Pinning `parallel_blocks` to 5 in both builds and replaying one
fixture:

```
positions            128
top-1 disagreements  0 / 128
median KL            0.000000e+00 nats
BIT-IDENTICAL
```

So the launch bound on its own changes nothing. The speed and the quality difference are
both carried entirely by the fan-out, and at the context this engine runs at, the fan-out
does not move.

**The earlier claim that speed and bit-exactness are the same knob in this kernel is
withdrawn.** They are the same knob only where the fan-out is small enough to be seed-bound.
That is a real limitation and it is stated in the shipped comment, because the two numbers
are read from the same source and must not be tuned independently.

### The gate, at the shipped dtype and the shipped context

Run at 200k context with the engine's real configuration, `q8_0` K and `q8_0` V, 128
teacher-forced positions, 32 logprobs each, all three arms on one fixture:

| comparison | result |
|---|---|
| stock, sampled vs stock, replayed (control) | bit-identical |
| **stock vs `min_blocks = 4`** | **bit-identical** |

Speed, `tools/fa_bench`, one decode attention, q8_0/q8_0, best of 9, three repeats on the
shipped build: **3.23 ms -> 2.64 ms** at 200k. The kernel is confirmed to be the one that
runs: instrumenting the dispatch prints, during decode at 200k,

```
[FATTN] VEC nq=1 nkv=2 D=256 n_kv=200192 K=q8_0 V=q8_0
```

which is exactly the geometry the benchmark models, at the full context and the shipped
dtypes.

### What it is worth end to end: 1%, not 10%

End to end at 200k, interleaved A/B so cache and temperature drift hit both arms equally,
2048 generated tokens so the decode window is 113 s rather than a few seconds:

| | decode | |
|---|---|---|
| stock | 113.44 s / 113.51 s | 18.05, 18.04 t/s |
| min_blocks = 4 | 112.49 s / 112.11 s | 18.21, 18.27 t/s |

**1.0% faster, repeatable, and the method is precise: the within-arm spread is 0.06%.**

The arithmetic says it should be about ten times that. `fa_bench` puts one attention at
2.64 ms, ten attention layers makes 26.4 ms, and a token is 55.5 ms, so attention would be
48% of the token and a 22% cut on it should be worth ~10%. It is worth 1.0%.

So the isolated benchmark overstates the in-situ cost of this kernel by roughly an order of
magnitude. The most likely reason is the memory the kernel actually walks: `fa_bench` gives
K and V two separate contiguous 218 MB buffers, while in situ the KV cache is one 2.03 GiB
buffer holding all ten layers' K and V together, so the per-layer access pattern, the TLB
behaviour and the L2 residency are all different. `fa_bench` is still the right tool for
comparing *kernels*, and the wrong tool for predicting *tokens*.

The honest summary of this item: a real, free, bit-exact 1.22x on the attention kernel that
is worth 1% of a decode token, and a demonstration that on this box an attention kernel
measured in isolation says almost nothing about what the token costs. Every bandwidth claim
in this document that was derived from a single isolated kernel should be read with that in
mind, including the 30 ms of attention per token that the byte counts suggest.

### The bug this found in the gate itself

Running the 200k comparison produced 127 of 128 top-1 disagreements, far worse than the
small-context result, with 3273 of 4096 top-32 tokens appearing in only one arm. A
reassociation cannot do that. The control settled it: **the same stock binary, sampled and
dumped in one run and replayed in another, disagreed the same way.**

The cause was in `amp-infer`, not in the kernel. `--emit-score` writes the fixture as one
token id per line, and `--score-file` read it back by *tokenising it as text*, so the
newlines became tokens and byte-level tokens did not survive the round trip. The two arms
scored different continuations, which is precisely the failure teacher forcing exists to
prevent. `--score-file` now reads a bare list of ids as ids and only falls back to
tokenising when the file is not one, so a hand-written text fixture still works.

That bug was live while an earlier kernel result was being judged. The small-context
measurements were unaffected, because both of their arms replayed through the same
misreading and therefore scored the same sequence. But the gate was one procedure change
away from being meaningless, and nothing in it would have said so.

### A second bug this found: the shipped KV dtype was not the one documented

The gate had to be re-run, because the measurements leading up to it had all been taken at a
different KV dtype than the engine claims to use. `RuntimeConfig::cache_v` defaulted to
`kQ4_0` (`include/amp/runtime/model_runtime.h:32`) while every other declaration in the tree
- `PlannerOptions`, `PreflightOptions` - said `kQ8_0`, `tests/test_preflight.cpp:238`
asserted `Q8_0`, and `AGENT.md` documented `q8_0/q8_0` as the shipped default with the
reasoning for it. `--model-config NAME:VARIANT` is opt-in, so **without it the engine ran
`q8_0`/`q4_0`** while every plan, test and document described `q8_0`/`q8_0`.

`tools/amp_plan.cpp` had the same default as a hardcoded string. Both are now `q8_0`. The
cost is the one `AGENT.md` predicted: the KV cache grows from 1.55 GiB to 2.03 GiB at 200k
and the planner drops the ubatch from 2048 to 1280. That is the price of the requirement, and
it was being paid in quality while the documentation claimed otherwise.

### What still does not qualify

The remaining 40%-of-bandwidth gap is not reachable by moving threads. Split-K is worth
2.0x on this kernel, measured by forcing `gridDim.y = 1` (6.46 ms against 3.23 ms), and
split-K is what makes the result order-sensitive. So the rest of the gap needs a kernel
that adds parallelism *within* a chunk instead of splitting the chunk, keeping each K
position in the accumulator that assigns it today. Strata's `qsa_decode_attn` is that
shape - one block per 64-cell chunk per KV head, all `n_head / n_head_kv` query heads that
share a KV head served from one read of the chunk - and it has to clear this same gate
before it ships.
## 6. Where a decode token actually goes, at the context it is used at

The bounds above all came from isolated measurements, and section 5 showed that one of them
was wrong by an order of magnitude. So this measures the token directly. Three instruments,
none of which touches the vendored tree except the two that are described as reverted.

### `AMP_TRACE_DECODE`: the phases of amp's own decode loop

Four phases per token, timed in `ModelRuntime::generate`:

| phase | 3.4k ctx | 33k ctx | 135k ctx |
|---|---|---|---|
| prefetch | 0.000 | 0.000 | 0.000 |
| `llama_decode` | 32.48 | 34.17 | 49.58 |
| sampler | 3.89 | 4.20 | 5.90 |
| logprob tracking | 0.000 | 0.000 | 0.000 |
| **total** | **36.37** | **38.36** | **55.48** |

The prefetcher is free at the call site, and the logprob tracking is now off. The sampler is
10.6% of a token at 135k, which is the one number here that did not exist before.

### The context sweep: the KV slope is the attention, exactly

No instrumentation, just decode time against prompt length, 1024 generated tokens each:

| prompt | tokens | ms/token |
|---|---|---|
| p32k | 8365 | 39.18 |
| p64k | 16661 | 37.15 |
| p128k | 33359 | 40.23 |
| prompt200k | 134572 | 56.33 |

The two end points carry the fit, the middle two carry the noise:

```
slope 0.1359 us per KV position per token      (10 layers, q8_0 K+V, 2 KV heads, D=256)
fixed 38.04 ms per token, independent of context
```

0.1359 us per position is 10,880 B / 0.1359 us = **80 GB/s in situ**, against the 155 GB/s
the card streams and the 67 GB/s `fa_bench` reports for the kernel in isolation. So the
in-situ attention is running *faster* than the isolated benchmark of the same kernel, which
is the other half of why the 1.22x kernel win was worth only 1% of a token: the kernel is
not the constraint in situ, the memory layout around it is.

At 200k that is 27.2 ms of attention in a 65 ms token, 42%.

### The fixed 38 ms

`perf` over a 2048-token decode at 8k context, grouped by symbol:

| share of host cycles | where |
|---|---|
| 47.2% | `libgomp` - the OpenMP runtime: barriers and spin |
| 22.7% | `ggml_vec_dot_iq3_xxs_q8_K` - expert matvec, layers 10-29 |
| 20.0% | `ggml_vec_dot_q3_K_q8_K` - expert matvec, layers 0-9 and 30-39 |
| 2.5% | graph dispatch and `clock_gettime` |

The expert matvec at 42.7% of cycles matches the 14.6 ms it is independently known to take
(413 MiB per token at 27.96 GB/s), so the OpenMP time is the other 47% and it is *spin*:
scaling is memory-bound, and a thread sweep confirms it. Four configurations, 2048 tokens,
two runs each:

| threads | decode | |
|---|---|---|
| 2 | 128.4 / 135.3 s | 1.84x slower than 8 |
| 4 | 86.3 / 83.8 s | 1.19x slower |
| 8 | 70.3 / 73.1 s | the optimum, and the default |
| 16 | 109.9 / 112.7 s | 1.55x slower; 8 is 8 physical cores, 16 is SMT |

So the thread count is already right and the expert path is at bandwidth. Neither is a
target.

### The sampler's 5.9 ms, and the 4.4 ms of it that is not accounted for

The sampler's own arithmetic is negligible, measured two ways:

| | cost |
|---|---|
| build the 248,320-entry candidate array | 0.05 ms in situ, 0.11 ms isolated |
| `std::partial_sort`, k=20 | 0.03 ms in situ, 0.06 ms isolated |
| `llama_sampler_accept` | 0.000 ms |

And it is not the top-k logprob tracking, which is off by default now and cost 0.16 ms when
it was on, and it is not `output_reorder()`, measured at 0.000 ms with zero swaps. All four
`llama_get_sampled_*_ith` entry points call `ctx->synchronize()`, so the phase is where the
host first waits for the device, and it does grow with context: 3.89 ms at 3.4k, 4.20 ms at
33k, 5.90 ms at 135k.

**But 4.4 ms of it is unexplained and is recorded as an open lead.** A bracket around the
`llama_sampler_sample` call from the caller says 5.90 ms. A bracket inside that same
function, from its first statement to its last, says 1.19 ms. Both are wall clock and they
cannot both be right.

Two candidate explanations, one tested and refuted:

- *The OpenMP spin pool is thrashing memory underneath the sampler.* Tested with
  `OMP_WAIT_POLICY=passive` and `GOMP_SPINCOUNT=0`. The sampler phase does not move
  (5.90 -> 5.74 ms), so this is not it. The experiment was worth running for another
  reason: the spin is worth **12%** end to end. 56.6/57.7 s spinning against 63.6/64.2 s
  passive, and it is 6 ms of that gap inside `llama_decode`, which is exactly the
  `libgomp` 47% the profile shows. The default is right and is now known to be.
- *The sampler is blocked, not computing.* This one the profile supports and the timing
  cannot settle. `perf` over the same decode shows 47.2% `libgomp`, 42.7% the two matvecs
  and 2.5% everything else - and no symbol for the sampler at all. If the sampler burned
  4.4 ms of CPU per 35 ms token it would be ~13% of the cycles and would be plainly
  visible. It is not there, so the time is most likely spent blocked in
  `ctx->synchronize()`, which `perf`'s cycle event does not sample.

That is the reading, and it is not proven. Settling it needs an instrument that samples
the host while it is inside that call, which this session did not build.

What is safe to say either way: 10.6% of a decode token sits in a function whose entire
purpose is to pick a token from 248,320 logits, and almost none of it is the selection.

### What this changes about the plan

The map is now, for a 135k token of 55.5 ms:

| | ms | |
|---|---|---|
| attention, from the sweep slope | 18.3 | 33%, at 80 GB/s in situ |
| expert matvec, from perf | 14.6 | 26%, at 27.96 GB/s, already at bandwidth |
| sampler's unexplained portion | ~4.4 | 8% |
| the rest of `llama_decode` | ~18 | 32%, GDN, projections, norms, copies, launches, graph |

The two byte-movers are both at or near their hardware limits. The only item that is not
accounted for and is not obviously a hardware limit is the sampler's 4.4 ms, which is the
next thing to chase, and the "rest of llama_decode" at 18 ms, which nobody has decomposed.

## 7. The 18 ms that nobody had decomposed, and a bound of mine that was wrong by 15x

The previous section left "the rest of `llama_decode`, ~18 ms, not decomposed" as the
open question. Two things settled it, both of which overturned something this document
previously asserted.

### The device time, measured by draining each split

`AMP_TRACE_GPU=1` synchronises the CUDA backend after every CUDA split and reports the
device time that split enqueued. It serialises the pipeline more than the scheduler already
does, so the token gets slower, and the device time it reports is unaffected. Per token:

| split | backend | n | 3.4k ctx | 135k ctx |
|---|---|---|---|---|
| attention | CUDA | 9 | 2.03 ms | **18.88 ms** |
| recurrent | CUDA | 29 | 9.77 ms | **11.89 ms** |
| experts | CPU | 37 | 15.72 ms | 16.32 ms |
| input copies at the layer boundaries | CPU | 37 | 0.51 ms | 0.56 ms |

The attention number lands on the context sweep's independent estimate of 18.3 ms within 3%,
from a completely different instrument, which is the cross-check that makes the rest
trustworthy.

**The input copies are 0.5 ms, not 9.** The same run without the extra synchronise reports
the CPU input-copy phase as 9.08 ms at 3.4k and 26.54 ms at 135k. The difference is entirely
where the wait lands: with the drain measured, the host's copy phase falls to 0.51 ms and the
same time appears in the device column. So the 9-26 ms "copy cost" that this document has
carried for two sessions is the host waiting for the device, and the copies themselves are
three round trips of 73 KiB per layer boundary.

### The compute-stream copy fix, measured properly, is not a speed change

`ggml_backend_tensor_get/set` reach the device through the *buffer* interface, which has no
handle on the stream the work was enqueued on, so they copied on `cudaStreamPerThread` and
synchronised that stream - which is not ordered against the compute stream, so the
synchronise did not mean "the data is ready". Giving the buffer context the compute stream
and copying on it removes that hazard. It is kept as a correctness change, not a speed one,
because three interleaved pairs at 135k say it is not a speed one:

| | decode, 1024 tokens | mean | spread |
|---|---|---|---|
| `cudaStreamPerThread` (stock) | 56.18, 57.07, 57.03 | 56.76 s | 0.89 |
| compute stream (the fix) | 57.23, 57.45, 56.98 | 57.22 s | 0.47 |

0.8% the wrong way, and less than the spread. This is the same result the change gave in
section 4 and in the first pass at it: cutting individual read-backs from 44-162 us to
13-23 us moves nothing, because the host was waiting for the device either way. It was worth
1.29 ms on the input-copy *phase* in one instrumented run and 0 ms end to end, which is what
the phase measurement was really saying.

The A/B also produced a trap worth recording: a straggler from an aborted earlier script was
holding 5360 MiB of the 6141, so the next six runs each failed to plan with `no
configuration fits: vram 4.01 GiB > usable 668.00 MiB`, and one of them raced a rebuild. The
script now checks free VRAM before every run. On a 6 GB card a leaked inference process is
not a nuisance, it invalidates the next six measurements.

### Which matmuls, and the LM head

Per-tensor device time, same 200-token run, the layer index and numeric suffixes collapsed so
the stems are per-tensor-kind:

| tensor | ms/token | calls/token |
|---|---|---|
| `node_#` (unnamed) | 9.49 | 130 |
| **`result_output`** | **6.63** | **1** |
| `linear_attn_out-#` | 3.53 | 30 |
| `z-#` | 3.02 | 30 |
| `Qcur_full-#` | 2.74 | 10 |
| `ffn_shexp-#` | 1.05 | 40 |
| `ffn_moe_logits-#` | 0.93 | 40 |
| `shared_expert_gate-#` | 0.51 | 40 |
| `ffn_moe_down-#` (MUL_MAT_ID) | 0.67 | 3 |

`result_output` is the LM head and it is **one matvec costing 6.63 ms, 12% of a token.** It is
`Q6_K` at 248320 x 2048, so 397.9 MiB read per token, which is 62.9 GB/s. The card streams at
128-155. So the head is at about half the achievable rate and there is roughly 3.4 ms in it.

Worth recording the wrong turn this took: assuming the head was f16 - which is what a
`-logprobs-n` reading of "248320 logits" suggests - gives 970 MiB and 153 GB/s, i.e. exactly at
the limit and nothing to win. The model is fully quantized, `file type = IQ3_XXS - 3.0625 bpw`,
with a Q6_K output head. The whole conclusion flips on the dtype, and the dtype is in the file.

The GDN projections are the other pair: `linear_attn_out` 3.53 ms and `z` 3.02 ms for 30
layers, 103 MiB of Q3_K between them, which is 29 GB/s. Against the card that is a 4.4x gap,
and it is the same gap as the 3.84 MiB-per-layer reading in the section above.

**So the largest remaining item is not the recurrence, not the attention, and not the experts.
It is that batch-1 quantized matmuls on this card run at 29-63 GB/s where the card does
128-155.** MUL_MAT is 24.0 ms of a 30.0 ms device budget; at streaming rate it would be about
10 ms, and 14 ms is 25% of a decode token.

Whether that is reachable at zero quality loss is the open question, and the structure of the
answer matters. A matvec's only order-sensitive operation is the reduction over K. Changing
which block computes which output row does not change it and is free; splitting K across
blocks does change it and is not. So the prize is a scheduling and occupancy change, not a
rewriting of the reduction - the same distinction that killed the attention occupancy change
in section 5, and it needs the same discipline to tell them apart.

### A bound of mine that was wrong by 15x

`tools/gdn_bench.cpp` sized the recurrent layers by their SSM **state**: 2 MiB per layer,
120 MiB per token, a floor of 0.8 ms against 155 GB/s. It concluded that Strata's `fused_gdn`
had nothing to win, and the idea was retired on that number.

The device measurement says the recurrent layers take **11.9 ms**, which is 22% of a 135k
token. The state is not the layer. The whole layer's weights are 3.84 MiB, so 115 MiB per
token across 30 layers, a bandwidth floor of 0.79 ms - and the layers take 15x that.

**They are not bandwidth-bound, and that is the finding.** 3.84 MiB per layer at 11.9 ms is
9.3 GB/s on a card that streams at 128. The working set is small enough to sit in L2, so this
is latency and kernel structure: a handful of dependent kernels per layer (in_proj, conv1d,
the gated delta recurrence, out_proj, the norm and gate) each paying launch and dependent-load
latency at batch 1, with the recurrence itself inherently serial.

So `fused_gdn` is the largest item left, not a rounding error. What it would have to do is
what Strata's does: hold 32 state rows per thread in registers so the state never
round-trips, and fold the norm and the sigmoid gate into the same kernel so those are not
separate dependent launches either. The prize is bounded by the 11.1 ms of gap between the
0.79 ms floor and the 11.9 ms measured, and a fused kernel that reached the floor would be
worth about 20% of a decode token.

That is a real piece of work - a new CUDA kernel for a recurrence, validated against the same
teacher-forced gate - and it is the one thing on the list whose size is now known.

### The token, complete

At 135k, of 55.5 ms, everything attributed:

| | ms | share | at its limit? |
|---|---|---|---|
| attention, device | 18.9 | 34% | no, 80 GB/s in situ |
| recurrent layers, device | 11.9 | 21% | **no, 15x its bandwidth floor** |
| expert matvec, host | 16.3 | 29% | yes, 27.96 GB/s |
| copies at layer boundaries | 0.6 | 1% | yes |
| CUDA split launches, host | 0.6 | 1% | yes |
| sampler's unexplained portion | ~4.4 | 8% | unknown |
| outside the split loop, in `llama_decode` | ~2.6 | 5% | not decomposed |

Two thirds of the token is now measured rather than inferred, and the one item with a large
gap to its hardware limit is the recurrent layers - which is the item this document had
already retired.

### Which kernel, and the finding that the recurrence is not the problem

`AMP_TRACE_OPS=1` times every node with a CUDA event pair and accumulates device
milliseconds per op; `AMP_NO_CUDA_GRAPH=1` is required alongside it, because decode runs as a
captured graph that replays as a single unit and cannot be attributed per op. The graph is
what makes 0.41 ms per recurrent layer *kernel execution* rather than launch overhead, and
with it bypassed the attribution is per op. 200 decode tokens, 3-token prompt, so prefill is
negligible:

| op | ms/token | nodes/token |
|---|---|---|
| **MUL_MAT** | **24.04** | 341 |
| GET_ROWS | 1.58 | 61 |
| ADD | 0.82 | 60 |
| CONCAT | 0.60 | 30 |
| FLASH_ATTN_EXT | 0.57 | 10 |
| MUL_MAT_ID | 0.55 | 3 |
| UNARY | 0.51 | 70 |
| CPY | 0.45 | 30 |
| MUL | 0.40 | 40 |
| SET_ROWS | 0.21 | 20 |
| ROPE | 0.19 | 20 |
| **total device** | **30.0** | |

Total device time of 30.0 ms per token against the 18.9 + 11.9 = 30.8 ms the split drain
measured, from a third instrument.

**`SSM_CONV` and `SSM_SCAN` do not appear in the list at all.** The gated delta recurrence -
the thing `fused_gdn` exists to accelerate - costs under 0.01 ms per token. The 11.9 ms of
recurrent splits is 341 plain matvecs: `in_proj`, `out_proj`, the shared expert, the router,
the qkv and output projections. The recurrence was never the cost, which is a third thing
this document had wrong about the recurrent layers, after the state-only bound and the
"not bandwidth-bound" reading.

**MUL_MAT is 24.0 ms of a 30.0 ms device budget, at roughly a quarter of the card's
streaming rate.** 3.84 MiB of weights per recurrent layer in 0.41 ms is 9.3 GB/s where the
card streams at 128. At batch 1 each of those is a small matvec, and 341 of them are chained
serially in one stream, so each pays its own launch and dependent-load latency with nothing
to overlap against.

### The consequence: the GPU is idle 46% of the token

Device time is 30.0 ms and the token is 55.5 ms, so the device is busy 54% of the time. The
CPU's 16.3 ms of expert matvec is exactly the GPU's idle window. They cannot be overlapped
within a token - layer N+1's input is layer N's post-MoE output, so the chain is serial, and
that is the same reason the `hit_hook` seam has nothing to reclaim - but expert work *moved*
onto the GPU would land in idle time and be close to free. All 37 CPU expert layers would fit
in 25.5 ms of GPU idle at 0.41 ms each; VRAM is what stops it, at roughly 2.4 GiB free
against 294 MiB per expert layer, so the planner's `g=3` is a VRAM answer and not a
throughput answer.

That makes the expert-layer count a real tuning question rather than a fixed constant, and
`g` is already a flag. The next thing to establish is where the measured optimum sits now
that the KV cache is `q8_0`/`q8_0` rather than `q8_0`/`q4_0`, since the larger cache is what
crowds the expert layers out of VRAM in the first place.

## 8. Attention efficiency, in isolation

`amp-fa-bench` measures one decode attention alone: ~67 GB/s, about 40% of this card's
achievable read bandwidth, flat from 32k to 200k context. The flatness says the shortfall
is per-position work, not per-byte work, and the per-position cost is the block-wide
reduction in `fattn-vec`'s inner loop. Closing that gap is worth roughly 1.5 ms per
attention layer, ~15 ms per token. Real, and second in line behind the barriers.

The 256-alignment cliff is worth recording because it cost real time here. Passing
`n_kv = 200000` to a raw `flash_attn_ext` measures 15.7 ms; passing 199936 measures 3.3 ms,
for essentially the same bytes. The difference is `K->ne[1] % FATTN_KQ_STRIDE` (256), which
gates `can_use_vector_kernel` at `fattn.cu:635`. **This is a benchmark artifact, not a
shipping bug**: `llama_kv_cache::get_n_kv` (`src/llama-kv-cache.cpp:1250-1264`) already
pads `n_kv` up to a multiple of 256 so the graph stays constant across batches, so the
serving path only ever sees aligned lengths. The tool now asserts alignment so this cannot
be rediscovered as a phantom 5x.

## Correctness constraint that shapes the fix

Any change here must be bit-exact or provably equivalent, because the project's whole
quality argument is "same ggml kernels, same order of operations". That rules out
approximate attention, FP8/INT8 KV, and a different reduction order in the softmax. It does
not rule out a different *thread mapping*, nor a different *schedule* over the same
operations, which is what is actually broken here. The gate stays `scripts/kl_parity.py`
with teacher forcing, per the note in `AGENT.md` about per-position KL without it being
meaningless.

## What the user asked for, and what is answerable from Strata

Strata is a from-scratch CUDA engine for a 125 B model on a 12 GB card; amp is llama.cpp
plus a memory plan on a 6 GB card. Most of Strata's kernel tree does not transfer, and
pretending otherwise would waste the session. Assessed against this box:

| Strata idea | Transfers? | Why |
|---|---|---|
| `hit_hook` two-phase CPU/GPU seam | **yes, and it is the whole prize** | the 96%-of-decode barrier cost is exactly what it removes |
| Grouped expert GEMV, constant launches per layer | yes, later | only pays off once experts actually live on the GPU; g=4 already saturates here |
| Per-layer expert-cache quotas, adaptive residency | partly | 6 GB cannot hold enough of a 12.19 GiB set to matter; VRAM is the binding constraint |
| `kv_stream` KV-in-RAM paging | no | its point is freeing VRAM for experts, and expert residency is already saturated at +3.4% |
| `fused_gdn` state-in-registers | yes, second | 30 recurrent layers x 2 MiB state, currently a separate round trip per layer |
| MTP / n-gram speculation | no | no MTP in this model, and n-gram was already measured at +0.2% |
| Warp-per-row mapping, hi/lo FP16 MMA, cp.async | yes, for attention | see section 5; the 40% of bandwidth |
| `__constant__` codebook is slow, use shared memory | no | applies to Strata's own codebook formats, not to GGUF's |
| Hugepages for weights | no, and now disproven | `tools/page_walk.cpp`: this kernel gives zero huge pages and no gain |

## Order of work, by measured value

`tools/budget.cpp` prints this table from the model's own geometry and a measured card rate,
so it can be re-run rather than believed. Against a 45 ms warm decode token:

| candidate | bound | of token | verdict |
|---|---|---|---|
| attention: reach streaming rate | 8.5 ms | 19% | partly taken: 1.22x on the kernel, 1% on the token |
| scheduler copy phase | 10.6 ms | 24% | measured to be GPU latency, not overhead |
| experts, everything | 12.9 ms | 29% | already at 27.96 GB/s on the CPU |
| GDN: fuse the state step | 0.8 ms | 2% | a few percent. Do not write this kernel |
| overlap seam (`hit_hook`) | - | - | implemented, measured, reverted |
| KV streaming (`kv_stream`) | - | - | needs VRAM this card does not have |
| grouped expert GEMV | - | - | gated on VRAM; g=4 already saturates |

**Rewritten by sections 6 and 7, which measured the token instead of bounding it.** The table
above is the old set of bounds, kept so the corrections are visible. Two of them were wrong:

- *Attention, 19%, "the only real kernel work left".* The kernel work was taken (1.22x,
  bit-exact) and the kernel is worth 1% of a token. In situ the attention runs at 80 GB/s.
- *GDN, 0.8 ms, 2%, "do not write this kernel".* Wrong by 15x. The bound sized the SSM state
  rather than the layer, and the layer does not run at bandwidth: 11.9 ms against a 0.79 ms
  floor. It is now the largest item on the list.

The token at 135k, everything attributed: recurrent layers 11.9 ms on the device, attention
18.9 ms on the device, expert matvec 16.3 ms on the host, layer-boundary copies 0.6 ms,
sampler ~4.4 ms unexplained, 2.6 ms outside the split loop. Of those, the expert matvec is at
memory bandwidth and the attention is at 80 GB/s. The recurrent layers are 15x off their own
bandwidth floor, and that is the one with a large gap left.

**Shipped this session:** raising the vec kernel's resident-blocks-per-SM from 1 to 4
(`fattn-vec.cuh:29`). One decode attention at 200k goes from 3.23 ms to 2.64 ms, 1.22x, and
it is bit-identical at the shipped dtype and context. End to end it is worth **1%**, not the
10% the isolated number predicts, because the attention is not the share of the token its
byte count suggests. See section 5.

That is the most important methodological result in this document: every bound on this list
that came from a single isolated kernel is suspect by the same order of magnitude, and the
largest one - attention, 19% of a token - is the one that turned out to be wrong. The two
entries below it were checked end to end and held.

The expert row is bounded against the GPU rate but runs on the CPU, where it measures
27.96 GB/s; its real cost is 12.9 ms and it is already at memory bandwidth, so it is a
floor rather than an opportunity.

**One item survives, and it is a kernel: the attention gap.** ~67 GB/s against ~170 GB/s
achievable, flat from 32k to 200k context, so the shortfall is per-position work. The cost
is the block-wide reduction `fattn-vec` performs once per KV position
(`ggml/src/ggml-cuda/fattn-vec.cuh:254-290`). Strata's answer to a slow decode attention is
`qsa_decode_attn`: one block per (64-cell chunk, KV head) so all the query heads sharing a
KV head are served from a single read of the chunk, per-lane `float4` loads, and split-K
partials merged with a log-sum-exp pass.

**It was built, measured, and rejected.** Raising the kernel's occupancy to 4 blocks per SM
is a 1.22x win on the attention, and it changes the arithmetic: 12 of 128 teacher-forced
top-1 tokens disagree and the largest logprob moves by 2.74 nats. The cause is not the
occupancy itself but the split-K it selects: `parallel_blocks` decides how K is partitioned
across blocks, and the merge of the resulting partials is order-sensitive. Forcing
`gridDim.y = 1` disables the split, is bit-exact by construction, and costs 2.0x. Speed and
bit-exactness are the same knob in this kernel. See section 5 for the full separation.

So the attention gap is real and worth 22% of the attention, and it is **not reachable
without giving up bit-exactness**. The only version that would qualify is a kernel that
adds parallelism *within* a chunk rather than splitting the chunk, which is what Strata's
`qsa_decode_attn` does and what would have to be written and gated here.

**Retired:** the CPU/GPU overlap seam. It was the largest item two drafts ago and it is not a
win on this box, because there is no host-side stall for it to reclaim: `wait-prev` is
0.1 us and the copy-phase wait is 0.0 us. The per-layer cost is GPU work the host is waiting
on, so a schedule that starts the CPU pool earlier cannot start it earlier, because the
router has not run.

**Retired:** the fused GDN step. 30 of the 40 layers are recurrent, which looked like the
obvious place for Strata's register-resident state. `tools/gdn_bench.cpp` bounds it: 120 MiB
of state traffic per token against 155 GB/s is a floor of 0.812 ms, 27 us per recurrent
layer, under 2% of a token even at half the streaming rate. There is no prize.

## The measurement trap on this box, restated because it cost real time

Identical configuration, same session, minutes apart:

| run | decode |
|---|---|
| p8k, rep 1 | 11.12 t/s |
| p8k, rep 2 | 24.28 t/s |
| p8k, rep 3 | 21.78 t/s |
| g=0 (all experts CPU) | 5.54 t/s |
| g=0, same command again | 22.14 t/s |

A 4x spread on a fixed config. Any claim of the form "X is N% faster" measured this way is
meaningless, and one such claim in an earlier draft of this file was exactly that. The two
measurements that survived scrutiny are the ones that do not depend on page-cache state: a
duty-cycle ratio from a single process, and the prefill/decode contrast in section 4, which
compares two phases *of the same run*.
