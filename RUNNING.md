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
./build/bin/amp-server --model $M --port 8081 --ctx 200000
```

It plans its own memory layout, warms the page cache (about 5 s for 9.6 GiB), then serves:

| endpoint | purpose |
|---|---|
| `POST /v1/chat/completions` | chat, SSE streaming, `reasoning_content` split out of `<think>` blocks |
| `POST /v1/completions` | raw text completion, think block included |
| `POST /apply-template` | the prompt the server will evaluate — use this to debug a cache miss |
| `POST /tokenize` | token ids for a prompt |
| `GET /props` | plan, prefix-cache stats, page-cache and VRAM state |
| `GET /health`, `GET /v1/models` | liveness, model list |

Aliases without the `/v1` prefix are registered too. `--api-key KEY` (or `--api-key-file`) requires
`Authorization: Bearer`. Useful flags: `--parallel N` (extra sequences, costs KV), `--gpu-layers N` /
`--ubatch N` (override the plan), `--no-warm` (start fast, first request slow), `--n-predict N`
(default `max_tokens`), `--reasoning-budget N` (see below).

### Thinking models: what this one supports

It is a reasoning model, and its template is unusual in a way that breaks naive servers. The rendered
generation prompt ends **inside** a think block — `<|im_start|>assistant\n<think>\n` — so the model
reasons before answering and *never emits the opening tag*. `GET`/`POST /apply-template` shows you
exactly what will be evaluated; use it whenever a turn misbehaves.

| capability | how | status in amp |
|---|---|---|
| thinking on/off | `chat_template_kwargs: {"enable_thinking": false}` | wired, **but not taking effect** — see below |
| preserve reasoning in history | `preserve_thinking` (llama.cpp calls it `preserve_reasoning` and maps it) | current turn is preserved automatically by the template; older turns need the flag |
| reasoning field names accepted from a client | `reasoning_content`, `reasoning_text`, `reasoning` | all three |
| thinking budget | `--reasoning-budget N`, or `reasoning_budget_tokens` per request: `-1` unrestricted (default), `0` close immediately, `N` tokens | works, off by default |
| `reasoning_format` (`none`/`auto`/`deepseek`/`deepseek-legacy`) | request field | **not implemented**; amp always behaves like `deepseek` |
| `reasoning_effort` | request field | template does not implement it |
| tool calls | the template renders `tool_calls` | rendered into the prompt; **no structured `tool_calls` in responses yet** |

`--reasoning-budget` is llama.cpp's mechanism (`common_reasoning_budget_init`), armed by replaying the
prefill tokens through the sampler. It is off by default on purpose: forcing a close mid-thought makes
this quant loop (`2+2 = 4. </think> 2+2 = 4. </think> ...`), so it would trade a parsing bug for a
generation bug. `--reasoning-budget -1` restores unrestricted thinking.

**Known issue:** `enable_thinking: false` does not currently change the rendered prompt — it stays
`<|im_start|>assistant\n<think>\n`. It is routed through the field llama.cpp reads
(`common_chat_templates_inputs::enable_thinking`, because `chat-auto-parser-helpers.cpp:322` overwrites
the kwarg), so the remaining cause is the autoparser's reasoning-mode detection pinning the open tag.
Verified with `/apply-template`, not assumed.

### Concurrency

One generation at a time, by design: a `llama_context` is not reentrant, and OpenCode issues a side
request (conversation title) while the main stream runs. Requests queue instead of racing — see
section 7 of `smoke_server.py`. On the pre-task-queue build, four concurrent streams returned zero bytes
each and the process died.

Point OpenCode at it by setting `baseURL` to `http://127.0.0.1:8081/v1` in
`~/.config/opencode/opencode.json`.

Check a running server end to end — health, raw completion, chat, the agentic prefix-cache pattern,
divergence, streaming, errors, **a prompt longer than the ubatch**, **the reasoning split**, and
**four concurrent streams**:

```bash
./scripts/smoke_server.py --url http://127.0.0.1:8081
```

Seven sections; the last three are regression tests for bugs that reached real use.

Every answer carries `amp_timings`, including `prompt_cached` and `cache_rewound_exactly`, so you can
tell a cache hit from a re-evaluation instead of guessing from a pause. `GET /props` shows the
lifetime totals.

### The one behaviour worth knowing

This model has 30 recurrent layers out of 40, so its KV cannot be rewound. amp therefore checkpoints
the sequence at the *prompt boundary* and restores it after each answer, which is what lets the next
turn extend the cache. A turn whose history diverges from the cached one is still answered correctly
— it just re-evaluates the prompt, and says so in `cache_rewound_exactly: false`.

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
