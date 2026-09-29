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

## 4. The actual bottleneck: 80 GPU-CPU barriers per token

This is the finding that matters, and it was not visible until the process was sampled
rather than timed end to end.

`perf record` over a 200k-context prefill plus 200 decode tokens, grouped by shared object:

| share of all CPU cycles | where |
|---|---|
| **71.4%** | `libcuda.so` (the driver) |
| 11.1% | `vdso` (`clock_gettime`) |
| 8.1% | `libgomp` (OpenMP barriers) |
| **5.9%** | `amp-infer`, i.e. actual model arithmetic |

Broken down by symbol, the driver time is two calls:

| share of all CPU cycles | call |
|---|---|
| **43.7%** | `ggml_backend_cuda_set_tensor_async` -> `cuMemcpyHtoDAsync_v2` |
| **39.3%** | `cudaStreamSynchronize` |

So the model spends most of its time not computing, and the two things it is doing are
host-to-device copies and waiting for the device.

**Why.** The MoE experts are pinned to the CPU (`-ncmoe`), so the router's chosen expert
ids have to travel GPU -> CPU before the CPU can compute anything, and the expert output
has to travel back. Tracing the scheduler's copies (`ggml_backend_tensor_copy`, called from
`ggml_backend_sched_compute_splits`) shows the per-layer pattern, repeated for all 40 layers
on every token:

```
D2H  attn_post_norm-N (reshaped)   8192 B
D2H  ffn_moe_topk-N                   32 B     <- the router's choice
H2D  ffn_moe_down-N               65536 B
```

That is 2.8 MiB per token in total, which is nothing. The cost is not bandwidth. Each
`GGML_TENSOR_FLAG_INPUT` copy forces a full `cudaStreamSynchronize` first
(`ggml/src/ggml-backend.cpp:1677-1684`), so every layer boundary is a hard
**GPU-drain -> CPU-compute -> re-upload** barrier. 40 layers means 80 such barriers per
token, and each one stops the pipeline dead until both sides finish.

Arithmetic that fits: 52.7 ms per token over 40 layers is 1318 us per layer, which is the
right order for "drain the device, compute 8 experts from RAM, copy 64 KiB back, refill the
pipe". The GPU is 79% resident but its DRAM is only ~30% utilised, and the CPU is using 1.36
of 16 cores. Neither device is doing useful work most of the time, because they are taking
turns.

### How big the barrier cost is, from two measurements that do not depend on the profiler

The barrier count is per-layer and per-token, and it does not depend on batch size. So
prefill, which pays 80 barriers once per 2048 tokens, and decode, which pays 80 barriers for
one token, do identical arithmetic work per token. Two equations, two unknowns:

| measured | value |
|---|---|
| prefill | 1.97 ms/token at batch 2048, so 4034 ms per ubatch |
| decode | 52.7 ms/token at batch 1 |

Writing `w` for the per-token bandwidth-bound work and `B` for the per-layer barrier cost:

```
2048w + 80B = 4034
   w + 80B =   52.7
```

which gives **w = 1.95 ms/token** and **B = 0.634 ms per layer**, so `80B = 50.8 ms`.

**96% of a decode token is barrier latency.** The model's arithmetic is ~2 ms. This is also
why prefill and decode differ by 26.8x on identical bytes, which is the clearest single
statement of the problem: nothing about the work changed, only how often the pipeline was
stopped.

**This is exactly the problem Strata's `hit_hook` seam exists to solve.** Strata splits each
layer's graph into a pre-graph ending at the router and a post-graph beginning at the
combine, rings a doorbell, and runs the CPU expert pool for layer N *while the GPU is
already working on layer N+1's mixer*. Its own measurements record the same effect: a single
change moved the drain from 18.2 ms to 10.2 ms while the token did not move at all, because
the GPU's half had been moved *behind* the CPU instead of *beside* it, and a second buffer
plus one `add_inplace` bought the overlap back (`include/strata/core/hit_hook.hpp:23-27`).

That is the port with the largest expected value, and it is a scheduling change rather than
an arithmetic one, so it preserves bit-exactness. Two things make it non-trivial here:

1. llama.cpp's scheduler is built around "compute this split, then synchronise", and the CPU
   expert split genuinely cannot start until the router result exists. The overlap therefore
   has to be a producer/consumer seam, not a relaxed barrier.
2. Strata overlaps the GPU computing *some* experts of a layer against the CPU computing the
   *others*. amp pins whole layers to the CPU, so there is no within-layer split to overlap;
   the seam has to run the CPU pool for layer N against GPU work for layer N+1, which is the
   cross-layer variant Strata also implements.

Strata's own record is the warning that matters here: the naive version of this change
looked like a large win and was worth nothing, because it moved the GPU's work *behind* the
CPU instead of *beside* it. Any attempt at this must be measured on the token, not on the
drain.

## 5. Attention efficiency, second

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

1. **Overlap the CPU expert pool with GPU work** (96% of decode). Largest single item.
2. **Attention kernel efficiency** (40% of achievable bandwidth, ~15 ms/token). Kernel work.
3. **Fuse the GDN step** so the 30 recurrent layers stop costing a round trip each.
4. Everything else in Strata's tree is either already in llama.cpp or needs more VRAM than
   this card has.

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
