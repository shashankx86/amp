# amp

A purpose-built inference engine for one model: **Occamy-1.0 APEX-I-MiniPlus-V2.1-Abliterated**
(`qwen35moe`, 40 blocks, hybrid SSM+attention, 256 experts / 8 used, 262144 native context).

Target machine: RTX 4050 Laptop 6 GB + Ryzen 7 7735HS (8C/16T, AVX2) + 14 GiB RAM + NVMe.

**`amp-server` is llama.cpp's server, with amp's memory plan in front of it.** The 40+ routes, the
sampler, the chat templates, tool calls, reasoning formats, slot management and every CLI flag are
upstream's, vendored at a pinned commit and linked statically. amp's own contribution is a ~90-line
preflight that turns a measured memory plan into `common_params` before the model loads.

```
argv ──► common_params_parse ──► apply_preflight ──► llama_server(...)
         (every llama.cpp flag)   (amp's plan)        (llama.cpp's server)
```

## Why this model needs a plan

The expert working set is 12.19 GiB (256 experts/layer, 8 active) and has to be cyclically re-read
for every ubatch, competing against a ~10.9 GiB page cache on a 6 GB card. The arithmetic is cheap —
~240 t/s prefill when the weights are resident — so the *memory system* is the product. Where the
experts live is worth 44x on decode under concurrency and 15x on a cold cache, and that is
expressible through public API (`tensor_buft_overrides`, the same mechanism as `-ncmoe`), so it
belongs in a preflight rather than a hand-written inference loop.

**Zero quality loss is a hard constraint**, and it is structural rather than aspirational: same
ggml kernels, same weights, same KV dtypes, same sampler, statically linked from the same commit.
Measured, not asserted — with placement held fixed, llama.cpp's server and amp's produce
**bit-identical** output distributions (KL = 0.000000e+00 both directions, JS = 0, max
|Δ logprob| = 0 over 512 greedy tokens).

## Status

| Milestone | State |
|---|---|
| M0 scaffold (modules, build, tests) | done |
| M1 GGUF reader + geometry | done |
| M2 page-cache warmer | done — 1.92 GiB/s cold, 12.19 GiB in 6.3 s |
| M2b memory planner + cost model | done |
| M3 forward path + prefetcher (`amp-infer`) | done — the measurement harness |
| M3b quality parity | done — bit-deterministic; KL = 0 vs upstream at matched placement |
| M4 GPU offload via `tensor_buft_overrides` | done — the public equivalent of `-ncmoe` |
| M5 page-cache warm at start-up | done — opt-in via `AMP_WARM=1` |
| **M6 server: llama.cpp's, with amp's plan** | **done** — 40+ routes, tool calls, reasoning formats |
| M6b preflight correctness | done — 7 tests assert it never overrides an explicit flag |
| M6c parity test suite | done — 15 sections pass, 8 features honestly skipped |
| M3c own graph (observable router) | **refuted** — expert fetching is 3.7 % of decode, so prefetch has a ~4 % ceiling |
| M7 decode headroom | **spent** — every quality-neutral lever measured and exhausted |
| M7b profile the 30 recurrent layers | the only untested avenue; a sequential SSM recurrence is the suspect for the 96 % |

## On speed, honestly

At steady state amp is **not** faster than a correctly configured `llama-server`: measured over 5
identical requests each, amp sustains 28.39 t/s decode and llama-server 30.05 t/s. An earlier
version of this file claimed 7.4x; that was amp's warm state measured against llama-server's cold
one, and it did not reproduce.

What the plan does buy is that it does not collapse. `llama-server`'s documented config needs
~11.5 GiB of CPU experts — more than this box keeps resident — so its first request runs at
**1.96 t/s** before recovering. amp keeps the CPU set at 10.90 GiB, inside the budget, and its first
request runs at 26 t/s. It also forces a single slot by default, because `llama-server`'s four
concurrent slots thrash the same shared working set (0.64 t/s measured). Full numbers, including
the mistakes, in `docs/BENCH.md`.

**Decode is compute-bound, and that reframes the hardware question.** Expert weight fetching is
**3.7 %** of decode; the other 96.3 % is arithmetic. So the 6 GB of VRAM is *not* the binding
constraint — g=8 would only be worth ~3 % — and more or faster GPU would change little for this
model. Measured and spent: threads are already optimal (`-t 8` beats 4/12/16), placement is worth
3.4 %, the 200k context costs 2.6 %, and n-gram speculation is worth 0.2 %. The one untested
avenue is the 30 recurrent layers out of 40, whose sequential recurrence is the obvious suspect
for the 96 % — the opposite of the MoE framing the old M3c milestone assumed.

## Thinking models

This is a reasoning model whose chat template renders the generation prompt already *inside* a
`<think>` block, so it reasons before answering and never emits the opening tag.

- `chat_template_kwargs: {"enable_thinking": false}` **works** and is the supported way to get a
  direct answer: the same question costs 104 completion tokens thinking and 2 not thinking.
- `reasoning_content` is split out of `content` automatically, in responses and streams.
- Structured `tool_calls` are emitted with a real id and parsed `arguments`. A thinking model needs
  ~66 tokens to reach a tool call; with thinking disabled, 27.
- `reasoning_format` (`none`/`auto`/`deepseek`/`deepseek-legacy`) is honoured.
- `reasoning_effort` is accepted and **ignored** — this template has no such capability.
- The reasoning budget is upstream's sampler, left off by default: this quant loops after a forced
  `</think>`.

## Build and run

```bash
./scripts/fetch_deps.sh && ./scripts/build.sh      # vendored llama.cpp, pinned; first build ~15 min

M=/home/e0u/localhost/models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf
./build/bin/amp-server --model $M --port 8081 -c 200000
```

Point an OpenAI-compatible client at `http://127.0.0.1:8081/v1`. Every `llama-server` flag works,
so the context flag is `-c` / `--ctx-size`. `docs/PARITY.md` is the full API contract.

## Tools and tests

```bash
./build/bin/amp-plan  --model $M             # what the planner decided, and why
./build/bin/amp-warm  --model $M --what plan # pull the expert set into the page cache
./build/bin/amp-infer --model $M --prompt hi # the forward path, with the prefetcher

python3 scripts/parity_test.py --url http://127.0.0.1:8081   # 15 sections over the API surface
./scripts/harness/run.sh --url http://127.0.0.1:8081          # the real client, 3 tool-using prompts
python3 scripts/bench_server.py --url http://127.0.0.1:8081 --n 5   # prefill/decode as a sequence
python3 scripts/kl_parity.py compare --a quality/llama-matched.json --b quality/amp-post-swap.json
```

Measurements and open questions: `docs/BENCH.md`. Design: `docs/ARCHITECTURE.md`. Operating guide:
[`RUNNING.md`](RUNNING.md). Working rules and hard constraints: `AGENT.md`. Background:
`../NOTES.md`, `../RUN.md`.
