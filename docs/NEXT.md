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

## The next piece of work

**Batch-1 quantized matmuls run at 29-63 GB/s on a card that streams at 128-155.** MUL_MAT
is 24.0 ms of the 30.0 ms device budget, 341 nodes per token, and at streaming rate it would
be about 10 ms. 14 ms is 25% of a decode token.

The two largest single nodes:

| node | ms/token | dtype | bytes/token | GB/s |
|---|---|---|---|---|
| `result_output` (the LM head) | 6.63 | Q6_K 248320x2048 | 397.9 MiB | 62.9 |
| `linear_attn_out` + `z` (30 layers) | 6.55 | Q3_K | 103 MiB | 29 |

The order-sensitive part of a matvec is only the reduction over K. Moving rows between
blocks does not change it and is free; splitting K across blocks does change it and is not.
So the prize is a scheduling and occupancy change, and it has to be separated the same way
the attention occupancy change was - by pinning the order-dependent part and measuring the
rest.

Second: the sampler's unexplained 4.4 ms. Its own work is 0.08 ms; the OpenMP explanation is
tested and refuted; `perf` shows no sampler symbol at all, so it is most likely blocked in
`ctx->synchronize()`, which `perf`'s cycle event does not sample. Settling it needs `perf -g`
with `-fno-omit-frame-pointer`, since without frame pointers the call graph was unusable.

Third: the 2.2% available from `g=4` at ubatch 1280, measured over three clean interleaved
pairs (57.27 s -> 56.01 s). **It is a placement change, so it must not become the default
under a hard no-quality-loss constraint** - `--gpu-layers 4` already exposes it. The planner
rejects that combination on a VRAM model that over-predicts; the runtime runs it fine.

## Strata

`git pull` is clean at `v0.1.24` (`3ce2523`). The three newest commits do not transfer:
`731899f` is QSA cell selection, an architecture this model does not have; `3c974b9` is
timing instrumentation. `fused_gdn` is Strata's best idea for this model and it is now
reinstated, but for a different reason than Strata's: not the recurrence, which is free, but
the projections around it.

## The gate

```bash
M=../models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf
./build/bin/amp-infer --model $M --prompt-file /tmp/opencode/prompt200k.txt --ctx 200000 \
    --n-predict 128 --temp 0 --logprobs-n 32 --seed 1234 --emit-score /tmp/fix.txt
# ... change something ...
./build/bin/amp-infer --model $M --prompt-file /tmp/opencode/prompt200k.txt --ctx 200000 \
    --n-predict 128 --temp 0 --logprobs-n 32 --seed 1234 --score-file /tmp/fix.txt \
    --dump-logprobs /tmp/after.tsv
python3 scripts/compare_logprobs.py /tmp/before.tsv /tmp/after.tsv
```

**Always run a stock-vs-stock control through the same procedure first.** That is what caught
bug 1, and it is the only reason this session's wrong conclusions were caught before they
were written down. Every arm must replay; none may sample-and-dump.

Last run, 200k, `q8_0`/`q8_0`, 128 positions, 32 logprobs each: control bit-identical,
shipped against stock **bit-identical**.

## Traps

- End-to-end decode varies 5.5-22 t/s across a session. With a 2048-token window it drops to
  0.06% within-arm spread; with 1024 it is 0.5%. Short windows cannot see a 1-2% effect.
- **A leaked inference process holds 5.4 GiB of the 6 and invalidates the next six runs**,
  each of which fails to plan with `no configuration fits: vram 4.01 GiB > usable 668.00 MiB`.
  Check free VRAM before every measurement.
- `nvidia-smi memory.used` reads a flat value across runs that differ by 294 MiB per expert
  layer, so it cannot resolve a marginal cost. Use llama.cpp's own buffer accounting.
- The GPU cannot be clock-locked (no root) and idles at 315 MHz; measure a baseline alongside.
- `n_kv` must be a multiple of 256 or the raw `flash_attn_ext` silently times the generic
  fallback.
- Decode runs as a captured CUDA graph, so per-node device times are impossible without
  bypassing the graph, and launch overhead is not a candidate for anything in the decode path.
- `/tmp` is tmpfs. `pkill -f` matches its own command line, and a `pkill` in the same command
  as a patch script kills the patch.
