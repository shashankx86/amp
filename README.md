# amp

llama.cpp's server with a memory plan in front of it. One model, one machine.

**Occamy-1.0 APEX-I-MiniPlus-V2.1-Abliterated** (`qwen35moe`, 40 blocks: 30 recurrent and 10
attention, 256 experts of which 8 are active). RTX 4050 Laptop 6 GB, Ryzen 7 7735HS, 14 GiB RAM.

The 40+ routes, sampler, chat templates, tool calls, reasoning formats, slot management and every
CLI flag are upstream's, vendored at a pinned commit and linked statically. amp adds a preflight
of about 200 lines that turns a measured memory plan into `common_params` before the model loads.
No amp HTTP layer.

```
argv ──► common_params_parse ──► apply_preflight ──► llama_server(...)
         every llama.cpp flag    amp's plan         llama.cpp's server
```

## Quick start

```bash
./scripts/fetch_deps.sh && ./scripts/build.sh     # first build ~15 min, it compiles CUDA

M=/home/e0u/localhost/models/Occamy-1.0-APEX-I-MiniPlus-V2.1-Abliterated.gguf
./build/bin/amp-server --model $M --port 8081 -c 200000
```

Point an OpenAI-compatible client at `http://127.0.0.1:8081/v1`. The context flag is `-c`, same
as upstream.

`amp-server` blocks and serves, so wrap it in `timeout` when checking it rather than using it. It
holds 4.5 GB of VRAM while looking idle.

## What the plan actually buys

llama.cpp on this box needs about 11.5 GiB of CPU expert weights, and the page cache holds about
10.9. Out of the box that gap makes llama.cpp thrash. amp sizes the working set to fit.

| | llama.cpp as shipped | amp | |
|---|---|---|---|
| prefill, 18k prompt | 45-86 t/s | **244.8 t/s** | 2.8-5.4x |
| decode, first request after a start | 1.96 t/s | **26.01 t/s** | 13x |
| decode, warm and steady | 30.05 t/s | 28.39 t/s | tie |
| decode, 4 slots generating at once | 0.64 t/s | 21-33 t/s | up to 44x |

The third row is the one place amp is not ahead, and it is a tie inside run-to-run noise: this box
varies about 20% between sessions.

The fourth row is why the preflight overrides `n_parallel`. llama.cpp's four default slots each
walk the same 11.5 GiB expert set, and when they generate simultaneously they collapse to 0.64
t/s. Two slots instead of four measured 21-33 t/s under the same three-concurrent-request test,
and cost nothing in steady decode.

**TBD.** Three things are unmeasured at the shipped configuration and should not be read into the
table above:

| | status |
|---|---|
| per-slot context at the shipped `n_parallel` | TBD. The 200k single-session figure was measured at `n_parallel 1`, which the shipped config no longer uses. |
| the four sampler profiles, `:s1` through `:s4` | TBD. They are transcribed from the model's published table and verified to reach the server, but none has been measured head to head. `:s2` is the current default. |
| which profile is best for agentic tool use | TBD. This is the most useful missing measurement in the project. |

The 7.4x an earlier version of this file claimed was amp warm against llama.cpp cold. It did not
reproduce, and the table above replaces it.

## Configuration presets

`--model-config <name>:<variant>` reads `configs/model-configs.conf`. A preset is exactly
equivalent to typing its flags, and a command line flag beats it. Unknown keys and out-of-range
values are hard errors, never silent no-ops.

```bash
./build/bin/amp-server --model $M -mc occamy-1.0-apex-i-miniplus   # lists variants, exits
```

Two independent axes. `v` sets KV cache dtypes, `s` sets samplers, and a preset may set both.

**KV quality.** Measured as KL divergence against f16/f16 at matched placement.

| | KV | median KL | |
|---|---|---|---|
| `:v1` | q8_0 / q4_0 | 1.809e-06 | smallest KV, furthest from f16 |
| `:v2` | q8_0 / q8_0 | 2.065e-07 | the default, 8.8x closer to f16 than v1 |
| `:v3` | f16 / f16 | 0 | `n_ctx` drops to 131072, f16 does not fit at 200k here |

**Samplers.** Transcribed from the model's published table.

| | workload | settings |
|---|---|---|
| `:s1` | co-work, agentic, tool use | `temperature` 1.00, `top_p` 0.95, `top_k` 20, `presence_penalty` 1.5, `enable_thinking` true |
| `:s2` | deterministic coding | `temperature` 0.20, `top_p` 0.95, `top_k` 20, `min_p` 0.05, `repetition_penalty` 1.05 |
| `:s3` | independent SWE-bench baseline | `temperature` 0.35, `top_p` 0.95, `top_k` 20, `min_p` 0, `repetition_penalty` 1.0 |
| `:s4` | general multilingual | `temperature` 0.60, `top_p` 0.95, `top_k` 20, `repetition_penalty` 1.08 |

`:s2` and `:s4` give temperature as a range upstream. The preset takes the low end; the high end
is `--model-config ...:s2 --temp 0.35`.

`extra_args` passes any llama-server flag through, including ones this project does not know
about. It lands in argv before llama.cpp parses, so upstream stays the only argument parser.

```ini
[my-preset]
extra_args = --port 8099 --alias my-model --no-webui
```

Samplers live here rather than in the client on purpose. The OpenAI-compatible provider does pass
unrecognised `options` into the request body, but OpenCode never forwards them: a request carrying
`temperature` arrives at the server with the server's own default. Verified, not assumed.

## Zero quality loss in the output

Same ggml kernels, same weights, same KV dtypes, same sampler, statically linked from the same
commit. With placement held fixed, amp and llama.cpp produce bit-identical output distributions:
KL 0.000000e+00 both directions, JS 0, max absolute logprob difference 0, over 512 greedy tokens.
The engine is deterministic enough that a repeated run also gives exactly zero, so this is a real
zero rather than a small number.

Placement has to be held fixed for that comparison to mean anything. Mismatched placement produces
about 11% top-1 disagreement from a 0.012 logprob difference, which looks catastrophic and is only
a different layout. `scripts/kl_parity.py` is the gate.

## Thinking and reasoning

The template renders the generation prompt already inside a `<think>` block, so the model reasons
before answering and never emits the opening tag.

- `chat_template_kwargs: {"enable_thinking": false}` gives a direct answer. The same question costs
  104 completion tokens thinking and 2 not thinking.
- `reasoning_content` is split out of `content` in responses and streams.
- Tool calls come back structured, with a real id and parsed `arguments`. Reaching one costs about
  66 tokens with thinking on, 27 with it off.
- `reasoning_format` (`none`/`auto`/`deepseek`/`deepseek-legacy`) works. `reasoning_effort` is
  accepted and ignored; this template has no such capability.

## Tools

```bash
./build/bin/amp-plan  --model $M             # what the planner decided, and why
./build/bin/amp-warm  --model $M --what plan # pull the expert set into the page cache
./build/bin/amp-infer --model $M --prompt hi # the forward path, with the prefetcher
```

## Measuring it

```bash
python3 scripts/parity_test.py --url http://127.0.0.1:8081          # API surface, 15 sections
./scripts/harness/run.sh --url http://127.0.0.1:8081                # real client, 3 tool prompts
python3 scripts/bench_server.py --url http://127.0.0.1:8081 --n 5   # speed
python3 scripts/kl_parity.py compare --a quality/a.json --b quality/b.json
```

One trap worth stating, because it invalidated a whole sweep here. Prefill has to be measured as a
sequence of requests. The same config restarted three times measured 132.5, 178.4 and 311.3 t/s on
one identical prompt, because prefill at batch 1024 thrashes the page cache. Decode at batch 1 does
not, and varied only 26.4 to 27.7.

`scripts/bench_prefill.py` exists for that reason.

## Where to read more

| | |
|---|---|
| [`AGENT.md`](AGENT.md) | working rules, hard constraints, and the facts not to re-derive |
| [`RUNNING.md`](RUNNING.md) | operating guide |
| [`docs/BENCH.md`](docs/BENCH.md) | every measurement with its date, including the retracted ones |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | module layout and why each piece exists |
| [`docs/PREFLIGHT.md`](docs/PREFLIGHT.md) | the preflight field by field, with llama.cpp citations |
| [`docs/PARITY.md`](docs/PARITY.md) | the API contract, request and response shapes |
