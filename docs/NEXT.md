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

## Three bugs found and fixed

1. **`--score-file` did not read back what `--emit-score` wrote.** The fixture is a list of
   token ids; the reader tokenised it as text, so newlines became tokens and byte-level
   tokens did not survive. Two runs of the *same binary* disagreed on 127 of 128 positions.
   The whole quality gate depended on this and nothing in it would have said so.
2. **The engine ran `q8_0`/`q4_0` KV, not the `q8_0`/`q8_0` it documents.**
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

## The next piece of work

Not the attention kernel. The remaining ideas are bounded, and `tools/budget.cpp` prints the
bounds from the model's own geometry:

| candidate | bound | verdict |
|---|---|---|
| attention kernel | 19% | 1% actually realised; the rest needs a kernel that adds parallelism within a chunk |
| scheduler copy phase | 10.6 ms | measured to be GPU latency, not overhead |
| experts | 12.9 ms | already at 27.96 GB/s on the CPU |
| GDN fused state | 0.8 ms | under 2%, do not write it |
| overlap seam, KV streaming, grouped expert GEMV | - | no host stall to reclaim, or need VRAM this card lacks |

The one substantive thing left is understanding **where a 55 ms decode token actually goes
at 200k**, because the isolated measurements have now twice been shown not to predict it.
`scripts/decompose_decode.sh` samples the duty cycle; the scheduler instrumentation from
earlier in this series (three phases per split) is the more direct instrument and is worth
reinstating, extended to separate the attention layers from the recurrent ones.

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
