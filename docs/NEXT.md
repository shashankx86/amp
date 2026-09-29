# Where the work is, 2026-09-29 (end of session)

`docs/STRATA-PORT.md` has the measurements and the reasoning. This is the state of the
tree and the one thing left to do.

## State

`third_party/llama.cpp` is **stock**, apart from a pre-existing `ggml-cpu.c` patch that
predates this work. Four changes were built and measured in that tree and all four were
reverted; they are described in STRATA-PORT.md so none is re-attempted:

| change | result |
|---|---|
| `fattn-vec` -> MMA kernel selection | appeared to be 1.5x; the forced path crashes on the first context, so the number was a short-context timing compared against a full sweep. MMA also dequantizes the whole q8_0 KV to f16 every step, 5.06x the traffic. |
| buffer `get_tensor`/`set_tensor` on the compute stream | correct and cut individual read-backs from 44-162 us to 13-23 us; changed the token by 0.0%, because the wait is for the GPU finishing the layer. |
| `__launch_bounds__(128, 4)` on `fattn-vec` | a real 1.22x on the attention, and it changes the arithmetic: 12/128 teacher-forced top-1 disagreements, median KL 2.8e-3 against a placement floor of 1.2e-4. |
| `gridDim.y` pinned to 1 | bit-exact, and 2.0x slower. Proves split-K is both the speed and the quality loss. |

New tools, all committed and building: `amp-page-walk`, `amp-fa-bench`, `amp-gdn-bench`,
`amp-budget`, `scripts/decompose_decode.sh`, `scripts/compare_logprobs.py`.

`amp_tests` passes. The one bug found in amp's own code is fixed: `--emit-score`,
`--score-file`, `--dump-output`, `--dump-logprobs` and `--logprobs-n` all worked and all
were missing from `amp-infer --help`.

## What is left, and why it is a single item

`amp-budget` bounds every candidate. Against a 45 ms warm decode token:

| candidate | bound | verdict |
|---|---|---|
| scheduler copy phase | 10.6 ms | measured to be GPU latency, not overhead |
| attention kernel | 8.5 ms | **the only live item** |
| experts | 12.9 ms | already at 27.96 GB/s on the CPU |
| GDN fused state | 0.8 ms | under 2%, not worth writing |
| overlap seam, KV streaming, grouped expert GEMV | - | need VRAM this card does not have, or have no host stall to reclaim |

The attention kernel runs at 67 GB/s of ~170 GB/s achievable, flat from 32k to 200k
context, so the shortfall is per-position work. It is **not** reachable by changing the
thread mapping: split-K is worth 2.0x and is the same thing that makes the result differ
from stock, and the project requires bit-exactness.

## The next piece of work

Write a decode attention that adds parallelism *within* a chunk rather than splitting the
chunk, so the speed does not come from repartitioning K. Strata's `qsa_decode_attn` is the
shape to adapt: one block per 64-cell chunk per KV head, all `n_head / n_head_kv` = 8 query
heads that share a KV head served from a single read of the chunk, per-lane `float4` loads,
and a split-K merge that is either disabled or kept in the same order as stock.

Files that matter:

- `ggml/src/ggml-cuda/fattn-vec.cuh` - the kernel to beat, and the exact summation order
  to reproduce. `vec_dot_KQ` and the `KQ_max`/`KQ_sum` update at `:268-303` are the order.
- `ggml/src/ggml-cuda/fattn-common.cuh:1131-1204` - where `parallel_blocks` is chosen.
- `ggml/src/ggml-cuda/fattn.cu:541` - the dispatch that picks VEC, and the reason it
  picks VEC for this geometry (quantized K/V, Ada, batch 1).
- Strata's `include/strata/kernels/qsa_decode_attn.hpp` and
  `src/kernels/cuda/qsa_decode_attn.cu` - the design, and the split-K merge to port.

## The gate

```bash
M=../models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf
./build/bin/amp-infer --model $M --prompt-file /tmp/opencode/para.txt --ctx 200000 \
    --n-predict 128 --temp 0 --logprobs-n 32 --emit-score /tmp/fix.txt
# ... change something ...
./build/bin/amp-infer --model $M --prompt-file /tmp/opencode/para.txt --ctx 200000 \
    --n-predict 128 --temp 0 --logprobs-n 32 --score-file /tmp/fix.txt \
    --dump-logprobs /tmp/after.tsv
python3 scripts/compare_logprobs.py /tmp/before.tsv /tmp/after.tsv
```

Zero top-1 disagreements, or it does not ship. Generating text with both builds and
diffing it is not a substitute: after the first argmax flip the two arms are on different
prefixes, which on this model reported 444 of 512 positions disagreeing at median KL
1.7e-05 for a change that did almost nothing.

## Traps, each of which cost time this session

- End-to-end decode varies **5.5 to 22 t/s on a fixed config**. Nothing measured that way
  is usable. `amp-fa-bench` and the duty-cycle tools exist because of this.
- `nvidia-smi` `utilization.memory` is not a bandwidth measurement on this card.
- The GPU cannot be clock-locked (no root) and idles at 315 MHz; a single timing is a guess
  until the card is warm, so measure the baseline alongside the thing.
- `n_kv` must be a multiple of 256 or the raw `flash_attn_ext` silently times the generic
  fallback and looks like a 5x win. The serving path pads it; the benchmark asserts it.
- Timing a code path that aborts early is not a measurement. That is how the MMA result
  first looked like a 1.5x win.
- `/tmp` is tmpfs. `pkill -f` matches its own command line.
