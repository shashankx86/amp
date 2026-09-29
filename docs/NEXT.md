# Where the work is, 2026-09-29

Read `docs/STRATA-PORT.md` first: it has the measurements. This file is the state of the
tree and the next concrete step.

## Done and committed

- `4b1395e` — `tools/page_walk.cpp`, `tools/fa_bench.cpp`, `scripts/decompose_decode.sh`,
  wired into `CMakeLists.txt`.
- `c9e5b79` — `docs/STRATA-PORT.md`, the bottleneck analysis and the transfer assessment.

`third_party/llama.cpp` is **stock**. Two changes were tried there and reverted: a
`fattn-vec` -> `MMA` kernel-selection change, and temporary tracing. Both are described in
STRATA-PORT.md so they are not re-attempted. Only the pre-existing `ggml-cpu.c` vendored
patch remains modified.

## The finding

96% of a decode token is GPU-CPU barrier latency, not arithmetic: ~50.8 ms of a 52.7 ms
token, 0.634 ms per layer, 80 barriers per token. The CPU has 14 of 16 cores idle and the
GPU's DRAM is ~30% utilised, because they take turns rather than overlapping.

## Next step: the overlap, in Strata's shape

Strata's `hit_hook` (`include/strata/core/hit_hook.hpp`) is the thing to port. Its shape:

```
pre[N]   graph up to and including the router   -> rings a doorbell
         CPU pool for layer N                   -> runs WHILE the GPU does pre[N+1]
post[N]  combine and the rest of layer N
```

Constraints to respect, all of them learned by Strata at cost:

1. **The split decision must be made before the CPU pool starts**, on the ids for *this*
   layer. Strata's first version decided inside the pool callback, read the previous layer's
   ids, and produced a plausible-looking token with KL 9.69e-02 -> 1.03e+00 and top-1
   0.867 -> 0.333 (`hit_hook.hpp:30-37`). If amp does this, the failure is silent.
2. **Ordering is the whole point.** Calling the GPU half once, after the pool, cut the drain
   18.2 -> 10.2 ms and moved the token not at all, because the GPU's work had been moved
   *behind* the CPU rather than *beside* it. A second buffer plus one `add` bought it back
   (`hit_hook.hpp:23-27`).
3. **Measure on the token, never on the drain.** Point 2 is a 1.8x win on the drain and
   0% on the token.

Where the seam goes in amp: `ggml_backend_sched_compute_splits`
(`ggml/src/ggml-backend.cpp:1646-1830`) currently does compute-split / sync / copy-inputs
in a strictly serial loop. The `GGML_TENSOR_FLAG_INPUT` branch at lines 1677-1684 is the
`cudaStreamSynchronize` that costs 39.3% of all cycles.

Expected shape of the change, smallest first:

- **Do nothing clever, just do it in one process**: let the CPU expert split for layer N run
  while the CUDA backend is still executing layer N's GPU queue, and use the existing
  `ggml_backend_event_t` to order the result rather than a host-side wait. This alone should
  recover most of the 50.8 ms, because the work already exists and is only serialized.
- Only if that is not enough, split the layer graph at the router, as Strata does.

## The gate

`scripts/kl_parity.py`, teacher-forced, per the note in `AGENT.md` about per-position KL
being meaningless without it. Placement already shifts logprobs by ~1.2e-04 (the documented
floor), so the comparison must hold placement fixed and compare decode output token-for-token
at greedy, reporting the top-1/top-2 gap. Any KL above the placement floor is a bug, not
noise.

## Also worth doing, lower value

- Attention runs at ~67 GB/s of ~170 GB/s achievable, flat in context length
  (`tools/fa-bench`). The per-position cost is the block-wide reduction in `fattn-vec`'s
  inner loop. Worth ~15 ms/token. Kernel work, not a flag.
- The 30 recurrent GDN layers each cost a separate state round trip; Strata's `fused_gdn`
  keeps 32 state rows per thread in registers and folds the norm and gate into the same
  kernel. Worth roughly 120 MiB/token of avoidable traffic.

## Traps, all of which cost time this session

- End-to-end decode varies **5.5 to 22 t/s on a fixed config** in the same session. Nothing
  measured that way is usable.
- `nvidia-smi` `utilization.memory` on this card idles at 11% and reads ~30% under load. It
  is not a bandwidth measurement. Do not divide anything by it.
- The GPU cannot be clock-locked (no root) and idles at 315 MHz, so any single timing is a
  guess until the card is warm. `tools/fa_bench` re-measures its baseline per row for this.
- `/tmp` is tmpfs. `bench_prefill.py` exists because prefill measured on a single request
  means nothing.
- `pkill -f` matches its own command line. Use `pkill -x`.
