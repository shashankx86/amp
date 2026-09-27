# Running amp

**`amp-server` serves chat.** Point your OpenAI-compatible client at it and it replaces
`llama-server`. Everything is vendored and static: one clone, one build tree, no `LD_LIBRARY_PATH`,
no RPATH into another checkout.

## Build

```bash
cd /home/e0u/localhost/amp
./scripts/fetch_deps.sh                              # clones llama.cpp at the pinned commit (~13 s)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)                       # first build compiles the CUDA kernels: ~15 min
```

`third_party/DEPS.lock` holds the pin. The fetch is idempotent — re-running it only re-checks the
commit. The first build is slow because ggml's CUDA kernels are compiled from scratch; after that,
incremental builds are seconds.

Check it:

```bash
./build/bin/amp_tests        # 31 cases: measured facts about the model, the planner, JSON, text
```

## Serving chat

```bash
M=/home/e0u/localhost/models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf
./build/bin/amp-server --model $M --port 8081 -c 200000
```

**`amp-server` is llama.cpp's server.** Every flag `llama-server` accepts is accepted here, because
argv goes straight to `common_params_parse`. The endpoint list is therefore llama.cpp's, not ours:
`/v1/chat/completions` and `/v1/completions` (with SSE), `/v1/messages` (Anthropic),
`/v1/responses` (OpenAI Responses), `/v1/embeddings`, `/v1/rerank`, `/infill`, `/tokenize`,
`/detokenize`, `/apply-template`, the `*/input_tokens` counting routes, `/props`, `/health`,
`/v1/models`, `/metrics`, `/slots`, `/slots/:id`, `/lora-adapters`. `docs/PARITY.md` is the full
contract, including which features cannot work on this box and why.

Note the flag names are llama.cpp's, not amp's old ones: **`-c` / `--ctx-size`, not `--ctx`.**

amp's own flags are environment variables, so they cannot collide with upstream's:

| variable | default | effect |
|---|---|---|
| `AMP_CTX` | 200000 | context to plan for when you do not pass `-c` |
| `AMP_WARM` | off | page-cache warm before the model load (see below) |
| `AMP_VERBOSE` | off | log every preflight decision, not just the plan |

### What amp adds, and what it deliberately does not

The preflight turns the memory plan into `common_params` before the model loads. It **never
overrides a flag you passed**, and it is a complete no-op if you pass `--fit` or your own device
layout (`-ngl`, `-ncmoe`, `-ot`). At start-up it logs every decision it made, for example:

    amp: plan: g=4 ubatch=1024 kv=1.55 GiB predicted 240 t/s prefill / 15 t/s decode
    amp: layout: 4 expert layers on GPU, 36 pinned to CPU (10.90 GiB CPU expert set)
    amp: n_ctx_checkpoints: 2 (clamped from 32; each is ~1.6 GiB at 200k ctx)
    amp: cache_ram_mib: 512 (clamped from 8192; the prompt cache evicts the model's page cache)
    amp: warm: off (the load's MAP_POPULATE + fadvise(SEQUENTIAL) already reads the whole file)

Those three clamps matter: llama.cpp's own defaults for `n_ctx_checkpoints` and `cache_ram_mib` are
32 and 8192, and on this box they are a way to exhaust RAM. `docs/AGENT.md` has the reasoning.

**It is not faster than a well-configured `llama-server` at steady state.** Measured with 5
identical requests per engine: amp 28.39 t/s decode, llama-server 30.05 t/s. What the plan buys is
that the CPU expert set fits the page cache (10.90 GiB) where llama-server's documented config does
not (~11.5 GiB), so amp does not collapse — llama-server's first request runs at 1.96 t/s, amp's at
26 t/s. See `docs/BENCH.md`.

### Performance: what to tune, and what to leave alone

**Leave `-t` alone.** `-t 8` (one thread per physical core) is the measured peak. Raising it makes
things worse:

| `-t` | 4 | **8 (default)** | 12 | 16 |
|---|---|---|---|---|
| steady decode | 27.95 | **32.97** | 29.96 | 21.19 t/s |

**Do not reach for these**, all measured to be worth nothing on this model:

- **N-gram speculative decoding** (`--spec-type ngram-simple`) — lossless, but +0.2 % on templated
  output, which is where it should pay best. The model is too entropic for drafts to match.
- **Expert prefetch** (M3c, `AMP_M3C_PREFETCH_KIB`) — built, measured, and a tie at +0.2 % over
  nine paired comparisons. The expert matvec already runs at 101.8 % of a plain read of its own
  bytes, so there is no exposed latency for a prefetcher to hide. Off by default; see
  `docs/BENCH.md` before enabling it for any reason.
- **A smaller context to "free" VRAM.** The 200k target costs 2.6 % of decode (8,192 -> 35.61,
  65,536 -> 35.27, 200,000 -> 34.68 t/s). Use the context you need; the trade is not worth making.

**The planner disables llama.cpp's RAM prompt cache, and that is deliberate.** `-cram 0`, not a
reduced value: on a 14.3 GiB box carrying a 13.66 GiB model, any prompt cache competes with the page
cache that both prefill and decode run on, and 512 MiB is still enough to lose ~3x on prompt
processing. Pass `-cram N` if you want one back - the first request after a mid-conversation side
request may then cost a full re-prefill.

**VRAM is a real constraint, but placement saturates, so do not expect a win from it.** g=0 -> g=4
measures +3.4 % for 10.6 % of expert bytes moved off the CPU, but g=4 -> g=7 then measures 28.89 ->
26.44 t/s, i.e. slightly *worse*, and forcing g=6 at 200k OOMs at load. The GPU is 70 % idle during
decode and moving work onto it still does not help: at batch 1 a GEMV cannot exploit parallelism and
competes with the recurrent layers already resident. An earlier note here claimed 1.46x for
all-experts-on-GPU; that was a linear extrapolation and it is wrong.

**The one real trade is context length against prefill rate.** Cutting `-c` to 64000 gives the planner
room for 7 GPU expert layers instead of 4, which measures **+27 % prefill** (198.8 -> 252.9 t/s) for
**-8.5 % decode** (28.89 -> 26.44 t/s). Worth it if you re-prefill often; usually not worth it in a
cached agentic loop, where steady-state turns are decode-bound.

The measurement still missing is a per-op profile. **30 of the 40 layers are recurrent** and 10 are
full attention; together with everything outside the expert matvec that is the other two thirds of
decode, and it has never been profiled. `perf` is not installed and there is no sudo, so this needs
a sampling profiler, a debug ggml with `GGML_SCHED_DEBUG`, or per-op timers in the CPU backend.
`amp-kernel-bound` answers the memory-versus-arithmetic question for a single tensor if you want to
check the reasoning above yourself.

Beware the noise: steady decode varies **~22 % between sessions** with page-cache warmth
(28.39-34.68 t/s at 200k across one day). Compare two configurations inside the same session, or
the difference is meaningless.

### Thinking models

This is a reasoning model, and its template renders the generation prompt already **inside** a
`<think>` block, so it reasons before answering and never emits the opening tag.

| capability | how | status |
|---|---|---|
| thinking on/off | `chat_template_kwargs: {"enable_thinking": false}` | **works** — verified with `/apply-template` |
| thinking on/off (server-wide) | `--reasoning off` | works |
| reasoning field names from a client | `reasoning_content`, `reasoning_text`, `reasoning` | all three, via llama.cpp |
| reasoning split in responses | automatic | `reasoning_content` separate from `content` |
| `reasoning_format` | `none` / `auto` / `deepseek` / `deepseek-legacy` | works, upstream's implementation |
| structured `tool_calls` | `tools` + `tool_choice` | works, with a real id and parsed `arguments` |
| thinking budget | `reasoning_budget_tokens` | works (100 -> 12 completion tokens); **opt-in**, see below |
| `reasoning_effort` | request field | accepted and **ignored**: this template has no such capability |

`enable_thinking: false` is the supported way to get a direct answer. It is a large win on short
requests: the same question costs **104 completion tokens thinking and 2 not thinking**, same answer.
The reasoning budget also works and is effective (100 -> 12 tokens), but it forces a `</think>`
mid-thought, and at a very tight budget (8 tokens) 1 sample in 3 returned a doubled answer. It stays
opt-in; `docs/BENCH.md` has the table.

Always inspect `POST /apply-template` when a turn misbehaves — it shows the exact prompt the server
will evaluate, for free.

### Concurrency

llama.cpp's server owns slot scheduling, so a `llama_context` is never reentered. Our old
hand-rolled server got this wrong and died under four concurrent streams; section 14 of
`scripts/parity_test.py` is the regression test for it.

**amp runs one slot by default, and that has a cost you should know about.** Stock
`llama-server` resolves `--parallel` to 4 (`arg.cpp:1400` -> `server.cpp:156-159`). Four
concurrent generations each want the same shared ~10.9 GiB CPU expert working set, and on a
box with ~10.9 GiB of usable page cache that is thrashing, not sharing: measured **0.64 t/s**
with two slots active against 28.4 t/s with one. So the preflight sets `n_parallel = 1` and
requests queue instead of interfering.

What that costs:

| | `n_parallel = 1` (default) | `--parallel 4` |
|---|---|---|
| concurrent requests | queue, each at full speed | thrash to ~0.6 t/s |
| `n > 1`, several choices in one response | **400**, bounded by slot count | works |

`n > 1` needs more than one slot, so it is unavailable by default and returns a typed
`invalid_request_error`. Stock llama-server can serve it. This is a deliberate trade, not an
oversight — but it is a capability stock llama-server has and amp does not by default, so it
should be your call rather than mine. Pass `--parallel N` if you need it.

Re-measured with one slot, three simultaneous requests: 3/3 completed, all with
`finish_reason`, decode 21.26 / 29.87 / 33.46 t/s, and only slot 0 ever used.

Point OpenCode at it by setting `baseURL` to `http://127.0.0.1:8081/v1` in
`~/.config/opencode/opencode.json`.

### Tests

```bash
# every llama.cpp route and response shape this box can actually exercise
python3 scripts/parity_test.py --url http://127.0.0.1:8081

# the real client: 3 prompts that use tools, in a pristine workspace each
./scripts/harness/run.sh --url http://127.0.0.1:8081

# quality: KL divergence and determinism against a reference capture
python3 scripts/kl_parity.py capture --url http://127.0.0.1:8081 --tag amp --out quality/amp.json
python3 scripts/kl_parity.py compare --a quality/llama-matched.json --b quality/amp.json

# speed: prefill and decode as a sequence, never a single request
python3 scripts/bench_server.py --url http://127.0.0.1:8081 --tag amp --n 5
```

A KL comparison is only meaningful with **placement held fixed** — amp's plan deliberately chooses a
different device layout from llama.cpp's default, which on its own produces ~11 % top-1 agreement
from a 0.012 logprob difference. Match the layout, then compare; see `docs/BENCH.md`.

## Everyday commands

Model path used throughout below:

```bash
M=/home/e0u/localhost/models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf
```

### 1. See what the planner decided (start here)

```bash
./build/bin/amp-plan --model $M
```

Prints the model's byte budget, the box's limits, the chosen configuration, and a ranked candidate
table. Two things to look for:

- `predicted prefill / decode` — what the cost model expects
- `DECODE IS CACHE-BOUND RIGHT NOW` — appears when other programs are holding RAM that the expert
  weights need. Close the browser and re-run; that is usually a 10x decode difference.

Useful flags: `--ctx N`, `--json`, `--gpu-layers N`, `--ubatch N`, `--top N`.

### 2. Run inference

```bash
./build/bin/amp-infer --model $M --prompt "explain MoE routing" --n-predict 128
```

Realistic prompt (52 KB, ~18k tokens) — the numbers in `docs/BENCH.md` come from this:

```bash
./build/bin/amp-infer --model $M --prompt-file /tmp/opencode/amp_bench_prompt.txt --n-predict 128 --ctx 200000
```

Flags that matter: `--gpu-layers` / `--ubatch` (override the plan), `--no-prefetch` (A/B baseline),
`--forward-warm` (warm in layer order instead of reverse), `--drop-cache` (cold start),
`--temp 0` (greedy), `--dump-output PATH` / `--dump-logprobs PATH` (for parity work).

### 3. Warm the page cache

```bash
./build/bin/amp-warm --model $M --what plan --verify
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
AMP_LOG_LEVEL=debug ./build/bin/amp-infer ...   # trace|debug|info|warn|error
AMP_CEILING_TPS=200 ./build/bin/amp-plan ...    # prefill ceiling, tokens/s
AMP_BW_FAULT_MBS=1850 AMP_BW_RESIDENT_MBS=3500 AMP_BW_SEQ_MBS=2000 ./build/bin/amp-plan ...
AMP_IO_OVERLAP=0.6 ./build/bin/amp-plan ...     # how much I/O hides behind compute
AMP_VRAM_PER_UBATCH_TOKEN_MB=0.72 ./build/bin/amp-plan ...
AMP_DECODE_MS_PER_LAYER=1.79 ./build/bin/amp-plan ...
AMP_TEST_MODEL=/path/to/other.gguf ./build/bin/amp_tests
```

## Safety rules that are not negotiable

- Never `--load-mode none` and never `--direct-io`. A 14.65 GB anonymous load froze this machine
  once. `amp` only ever mmaps.
- Close the browser before a long inference session; it holds several GiB that the expert weights
  need, and the difference is 10x on decode.
- Do not start `amp-server` (or `amp-infer`) while a `llama-server` is running: they will fight over
  the same 6 GB of VRAM and 14 GiB of page cache, and both will be slow and possibly unstable. The
  planner will also refuse to fit the model, which is the intended behaviour — free the VRAM first.
- `pkill -x amp-server` sends SIGTERM and the server exits cleanly. If a run left one behind, check
  `pgrep -a amp-server` — a stale instance holds 5.4 GB of VRAM and every other instance will then
  fail to plan.
- The Kioxia SSD (`/run/media/e0u/D1`) is 6-13x slower than the Kingston for this workload. Keep the
  model where it is.
