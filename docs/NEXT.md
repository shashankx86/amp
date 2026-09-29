# Where the work is, 2026-09-30 (end of session)

`docs/STRATA-PORT.md` has the measurements and the reasoning. This is the state of the
tree and what is left.

## State

One change is live in the vendored tree and is **not committed** -
`third_party/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh`, the vec attention kernel's
`__launch_bounds__(nthreads, 1)` -> `, 4`. It is 12 lines including the comment, it is
bit-exact at 200k with the shipped dtypes, and it is worth 1% of a decode token. It is left
uncommitted because `third_party/llama.cpp/AGENTS.md` says not to commit without explicit
approval, and that instruction was not given for this tree.

Everything else in `amp/` is committed. `amp_tests` passes.

## Bugs found and fixed

0. **`--seed` did not exist.** `RuntimeConfig` has had a `seed` field since the file was
   created and nothing set it, so it stayed at `LLAMA_DEFAULT_SEED`, which
   `llama_sampler_init_dist` resolves from the system clock. With the shipped default config
   - temp 0.6, top_p 0.95, top_k 20 - every run of the same build on the same input
   produced different text. Nothing was reproducible and nothing about the sampling path
   could be compared; the quality gate only works because it pins `--temp 0`, which is not
   what ships. Two runs with `--seed 1234` are now byte-identical.
0b. **Every token was accepted twice.** `llama_sampler_sample` calls
   `llama_sampler_accept` before returning and `generate()` called it again. `score()` needs
   its own, because it never calls `llama_sampler_sample`, so the two are now asymmetric on
   purpose. Harmless today - every accept in this chain is a no-op - and verified
   bit-identical under the shipped sampling config with a fixed seed.
1. **`--score-file` did not read back what `--emit-score` wrote.** The fixture is a list of
   token ids; the reader tokenised it as text, so newlines became tokens and byte-level
   tokens did not survive. Two runs of the *same binary* disagreed on 127 of 128 positions.
   The whole quality gate depended on this and nothing in it would have said so.
2. **The top-k logprob tracking ran on the hot path by default.** It builds and
   partial-sorts the whole 248,320-entry vocab every token to produce a report nobody
   asked for. Measured at 0.16 ms, so 0.3% of a token, but it is pure overhead and
   `--logprobs-n` turns it on when comparing engines. `--logprobs-n 0` now actually
   disables it: the old guard was `if (cfg.top_k_track > 0) assign`, so a 0 fell through and
   left the runtime at its own default.
3. **The engine ran `q8_0`/`q4_0` KV, not the `q8_0`/`q8_0` it documents.**
   `RuntimeConfig::cache_v` was the only declaration in the tree that said `kQ4_0`;
   `PlannerOptions`, `PreflightOptions`, a test and `AGENT.md` all said `kQ8_0`. The
   requirement is q8_0/q8_0, and it was being violated silently. Cost of the fix, as
   `AGENT.md` predicted: KV grows 1.55 -> 2.03 GiB at 200k and ubatch drops 2048 -> 1280.
3. **Three `amp-infer --help` lines that did not match the code**, including two
   (`--logprobs-n` default, `--dump-logprobs` format) that the quality gate depends on
   reading correctly.

## The result, and the lesson attached to it

`__launch_bounds__(128, 1)` gives 4 warps per SM. Raising it to 4 is 1.22x on the attention
kernel and **bit-exact at 200k** - verified teacher-forced, 128 positions, 32 logprobs each,
with a stock-vs-stock control that also passed.

It was rejected the first time, correctly reasoned and wrongly tested: the gate ran on a
1k-token prompt, which is the one context where the occupancy seed binds the split-K fan-out.
From about 1k tokens of KV upward the fan-out is 5 either way. Isolation test: pin the
fan-out in both builds and the launch bound alone is bit-identical.

**It is worth 1% of a decode token, not the 10% the isolated benchmark predicts.** Two
interleaved rounds, 2048 tokens, 113 s decode window: 113.44/113.51 s stock, 112.49/112.11 s
patched. Within-arm spread 0.06%, so the method is precise.

`tools/fa_bench` is the wrong tool for predicting tokens. It gives K and V as two separate
contiguous 218 MB buffers; in situ the KV cache is one 2.03 GiB buffer holding all ten
layers together. It remains the right tool for comparing kernels against each other.

## What this invalidates in the earlier analysis

Every bound derived from a single isolated kernel is suspect by about the same factor. The
largest one on the list - attention at 19% of a token - was wrong. The two below it were
checked end to end and held: the expert matvec at 27.96 GB/s (12.9 ms, confirmed in the
scheduler) and the GDN state floor at 0.8 ms.

Also note: at 200k, **f16 KV is faster than q8_0 KV** (17.9 vs 16.1 t/s) despite 2.46x the
bytes, so the q8_0 path is dequantisation-bound rather than bandwidth-bound. That is a
finding about the constraint, not an argument against it.

## Where a token goes, measured 2026-09-30

This was the open question at the end of the last session, and it is now answered.
`docs/STRATA-PORT.md` section 6 has the numbers. For a 135k token of 55.5 ms:

| | ms | share | at hardware limit? |
|---|---|---|---|
| attention | 18.3 | 33% | 80 GB/s in situ, yes |
| expert matvec | 14.6 | 26% | 27.96 GB/s, yes |
| sampler, unexplained portion | ~4.4 | 8% | unknown, this is the lead |
| the rest of `llama_decode` | ~18 | 32% | not decomposed |

Tools: `AMP_TRACE_DECODE=1` times the four phases of amp's own decode loop. A context sweep
(no instrumentation, 1024 generated tokens per point) gives the KV slope and therefore the
attention share exactly. `perf record` over a 2048-token decode gives the cycle split.

Both byte-movers are at their hardware limits, so neither is a target. 8 threads is
optimal and scaling is memory-bound, confirmed by a four-point thread sweep. The OpenMP
spin is worth 12% end to end (`OMP_WAIT_POLICY=passive` costs 56.6 s -> 63.6 s), so the
default is right.

## The next piece of work

**The sampler's unexplained 4.4 ms.** The sampler's own arithmetic is 0.08 ms: building the
248,320-entry candidate array is 0.05 ms and the k=20 `partial_sort` is 0.03 ms, measured
both in situ and in isolation. It is not the logprob tracking (now off by default, 0.16 ms
when on) and not `output_reorder` (0.000 ms, zero swaps). All four
`llama_get_sampled_*_ith` entry points call `ctx->synchronize()`, so the phase is where the
host first waits for the device, and it does grow with context: 3.89 ms at 3.4k, 4.20 at
33k, 5.90 at 135k.

A bracket around the `llama_sampler_sample` call from the caller reads 5.90 ms. A bracket
inside the same function, first statement to last, reads 1.19 ms. Both are wall clock and
they disagree, and that is the whole lead.

Two candidate explanations, one already tested:

- *OpenMP spin thrashing memory under the sampler.* **Refuted.** `OMP_WAIT_POLICY=passive`
  and `GOMP_SPINCOUNT=0` leave the phase at 5.74 ms.
- *The sampler is blocked, not computing.* Supported by the profile and not settled by the
  timing. `perf` shows 47.2% `libgomp`, 42.7% the two matvecs, 2.5% everything else, and no
  sampler symbol at all. A function burning 4.4 ms of CPU per 35 ms token would be ~13% of
  the cycles and impossible to miss. So the time is most likely blocked in
  `ctx->synchronize()`, which `perf`'s cycle event does not sample.

Settling it needs an instrument that samples the host while it is inside that call. The
vendored-tree timers that were used to bisect it (`llama-sampler.cpp`,
`llama-context.cpp`) have been reverted; `perf` without frame pointers gave no usable call
graph, so the next attempt should either build the binary with `-fno-omit-frame-pointer` and
run `perf record -g`, or add a ring-buffer sampler that records the instruction pointer on a
timer thread.

**Second:** the 18 ms of `llama_decode` that is neither attention nor the expert matvec.
Nobody has decomposed it. The scheduler split instrumentation from earlier in this series -
three phases per split in `ggml_backend_sched_compute_splits` - is the direct instrument and
is worth reinstating, extended to separate the ten attention layers from the thirty
recurrent ones.

## Strata

`git pull` is clean at `v0.1.24` (`3ce2523`). The three newest commits were never read
before this session and none of them transfers:

- `731899f` QSA selection on tensor cores, +14.6% prompt at 128K - this model has full
  attention, there is no cell selection to speed up. Its commit message does carry one
  relevant data point: Strata ships a 3xTF32 change that it labels "not bitwise (another
  summation order)" and calls "the size of any FP32-level change".
- `3c974b9` GDN phase split in prefill timing - instrumentation, not an optimisation, and it
  confirms the same projections / conv+gates / recurrence / output decomposition.
- `3ce2523` version bump.

## The gate

```bash
M=../models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf
./build/bin/amp-infer --model $M --prompt-file /tmp/opencode/p200k.txt --ctx 200000 \
    --n-predict 128 --temp 0 --logprobs-n 32 --emit-score /tmp/fix.txt
# ... change something ...
./build/bin/amp-infer --model $M --prompt-file /tmp/opencode/p200k.txt --ctx 200000 \
    --n-predict 128 --temp 0 --logprobs-n 32 --score-file /tmp/fix.txt \
    --dump-logprobs /tmp/after.tsv
python3 scripts/compare_logprobs.py /tmp/before.tsv /tmp/after.tsv
```

**Always run a stock-vs-stock control through the same procedure first.** That is what
caught bug 1, and it is the only reason this session's two wrong conclusions were caught
before they were written down. Every arm must replay; none may sample-and-dump, because
that is the asymmetry the fixture bug hid behind.

Zero top-1 disagreements, or it does not ship.

## Traps, each of which cost time

- End-to-end decode varies **5.5 to 22 t/s** across a session. With a long enough decode
  window (2048 tokens, 113 s) it drops to 0.06% within-arm spread and becomes usable. Short
  runs cannot see a 1% effect.
- `nvidia-smi` `utilization.memory` is not a bandwidth measurement on this card.
- The GPU cannot be clock-locked (no root) and idles at 315 MHz; measure a baseline alongside.
- `n_kv` must be a multiple of 256 or the raw `flash_attn_ext` silently times the generic
  fallback. The serving path pads it; `amp-fa-bench` asserts it.
- Timing a code path that aborts early is not a measurement.
- `/tmp` is tmpfs. `pkill -f` matches its own command line.
- Check that a kernel is actually on the live path before optimising it. `qwen35moe.cpp`
  never calls `ggml_flash_attn_ext`; it goes through `build_attn_mha`, and a trace placed at
  the wrong `return` in the dispatch suggested for a while that no FA kernel ran at all.
