# Running amp

**Status check first: `amp` is not a server yet.** It has a working forward path, a memory
planner, a page-cache warmer and a quality-parity harness, but no HTTP API. For actual chat use
today, keep using `llama-server` with the command in `../RUN.md`. `amp` today is the toolchain that
beats it, plus the plan that makes it fast.

M6 (the OpenAI-compatible server) is on the milestone list and is the next real milestone.

## Build

```bash
cd /home/e0u/localhost/amp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DAMP_LLAMA_ROOT=../llama.cpp
cmake --build build -j$(nproc)
```

Requires the local llama.cpp build (`../llama.cpp/build/bin/libllama*.so`). The location is baked
into the binaries via RPATH, so the tools run with no `LD_LIBRARY_PATH`.

Check it:

```bash
./build/amp_tests          # 19 cases, asserts the measured facts about the model
```

## Everyday commands

Model path used throughout below:

```bash
M=/home/e0u/localhost/models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf
```

### 1. See what the planner decided (start here)

```bash
./build/amp-plan --model $M
```

Prints the model's byte budget, the box's limits, the chosen configuration, and a ranked candidate
table. Two things to look for:

- `predicted prefill / decode` — what the cost model expects
- `DECODE IS CACHE-BOUND RIGHT NOW` — appears when other programs are holding RAM that the expert
  weights need. Close the browser and re-run; that is usually a 10x decode difference.

Useful flags: `--ctx N`, `--json`, `--gpu-layers N`, `--ubatch N`, `--top N`.

### 2. Run inference

```bash
./build/amp-infer --model $M --prompt "explain MoE routing" --n-predict 128
```

Realistic prompt (52 KB, ~18k tokens) — the numbers in `docs/BENCH.md` come from this:

```bash
./build/amp-infer --model $M --prompt-file /tmp/opencode/amp_bench_prompt.txt --n-predict 128 --ctx 200000
```

Flags that matter: `--gpu-layers` / `--ubatch` (override the plan), `--no-prefetch` (A/B baseline),
`--forward-warm` (warm in layer order instead of reverse), `--drop-cache` (cold start),
`--temp 0` (greedy), `--dump-output PATH` / `--dump-logprobs PATH` (for parity work).

### 3. Warm the page cache

```bash
./build/amp-warm --model $M --what plan --verify
```

Pulls the planner's resident set in with large sequential reads and reports the achieved bandwidth.
`--what experts` warms all 40 expert layers, `--what all` the whole file, `--drop-cache` evicts first
so the measurement is honest.

This is the highest-leverage thing to do before a long session: 12.19 GiB comes in at ~1.9 GiB/s
(6.3 s), and the CPU-expert working set then stays resident instead of thrashing.

### 4. Compare against llama-server

```bash
./scripts/head2head.sh /tmp/opencode/amp_bench_prompt.txt 128
```

Warms the cache, runs both engines on the identical prompt, prints prefill and decode for each.
Requires port 8099 to be free.

### 5. Verify quality parity

```bash
./scripts/parity.py
```

Runs three comparisons (amp vs amp for determinism, amp g=6 vs g=0 for placement drift, amp vs
llama-server) and reports top-1 token agreement, max logprob delta, and the top-1/top-2 gap at any
position where the argmax flips. Expect: 0.000000 determinism, identical text for the placement
comparison.

## Reading the results

| symptom | meaning |
|---|---|
| prefill ~35-65 t/s | CPU expert set does not fit the page cache — check `amp-plan`, close the browser, warm first |
| prefill >200 t/s | working set is resident; you are at the compute ceiling (~240 t/s) |
| decode <5 t/s | cache overflow cliff: too many expert layers on the CPU for the available cache |
| decode 25-30 t/s | resident, compute-bound, normal |
| `amp-warn: ubatch reduced` | the planner's VRAM estimate was optimistic; the runtime backed off to fit |

## Environment overrides (all optional)

Every constant in the cost model can be overridden, mostly for calibration experiments:

```bash
AMP_LOG_LEVEL=debug ./build/amp-infer ...     # trace|debug|info|warn|error
AMP_CEILING_TPS=200 ./build/amp-plan ...      # prefill ceiling, tokens/s
AMP_BW_FAULT_MBS=1850 AMP_BW_RESIDENT_MBS=3500 AMP_BW_SEQ_MBS=2000 ./build/amp-plan ...
AMP_IO_OVERLAP=0.6 ./build/amp-plan ...       # how much I/O hides behind compute
AMP_VRAM_PER_UBATCH_TOKEN_MB=0.72 ./build/amp-plan ...
AMP_DECODE_MS_PER_LAYER=1.79 ./build/amp-plan ...
AMP_TEST_MODEL=/path/to/other.gguf ./build/amp_tests
```

## Safety rules that are not negotiable

- Never `--load-mode none` and never `--direct-io`. A 14.65 GB anonymous load froze this machine
  once. `amp` only ever mmaps.
- Close the browser before a long inference session; it holds several GiB that the expert weights
  need, and the difference is 10x on decode.
- Do not start `amp-infer` while a `llama-server` is running: they will fight over the same 6 GB of
  VRAM and 14 GiB of page cache, and both will be slow and possibly unstable.
- The Kioxia SSD (`/run/media/e0u/D1`) is 6-13x slower than the Kingston for this workload. Keep the
  model where it is.
