# amp

llama.cpp's server with a memory plan in front of it, tuned for one model on one machine:
**Occamy-1.0 APEX-I-MiniPlus-V2.1-Abliterated** (`qwen35moe`, 40 blocks, 30 recurrent and 10
attention, 256 experts of which 8 are active, 262144 native context).

Machine: RTX 4050 Laptop 6 GB, Ryzen 7 7735HS (8C/16T, AVX2), 14 GiB RAM, NVMe.

```
argv ──► common_params_parse ──► apply_preflight ──► llama_server(...)
         every llama.cpp flag    amp's plan         llama.cpp's server
```

The 40+ routes, the sampler, chat templates, tool calls, reasoning formats, slot management and
every CLI flag are upstream's, vendored at a pinned commit and linked statically. amp contributes
a preflight of about 200 lines that turns a measured memory plan into `common_params` before the
model loads. There is no amp HTTP layer.

## Build and run

```bash
./scripts/fetch_deps.sh && ./scripts/build.sh      # first build ~15 min, it compiles CUDA

M=/home/e0u/localhost/models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf
./build/bin/amp-server --model $M --port 8081 -c 200000
```

Point an OpenAI-compatible client at `http://127.0.0.1:8081/v1`. The context flag is `-c`, same as
upstream. `docs/PARITY.md` is the full API contract.

`amp-server` blocks and serves, so wrap it in `timeout` when you are checking it rather than using
it. It holds 4.5 GB of VRAM while looking idle.

## Configuration presets

`--model-config <name>:<variant>` applies a named preset from `configs/model-configs.conf`. A
preset is exactly equivalent to typing its flags, and an explicit flag on the command line beats
it. An unknown key or an out-of-range value is a hard error, never a silent no-op.

Pass a bare name to list what exists. It prints and exits without loading the model.

```bash
./build/bin/amp-server --model $M -mc occamy-1.0-apex-i-miniplus
```

Two independent axes. The `v` variants set KV cache dtypes, the `s` variants set samplers, and a
preset may set keys from both.

| variant | KV dtypes | median KL vs f16 | note |
|---|---|---|---|
| `:v1` | q8_0 / q4_0 | 1.809e-06 | smallest KV, furthest from f16 |
| `:v2` | q8_0 / q8_0 | 2.065e-07 | the default, 8.8x closer to f16 than v1 |
| `:v3` | f16 / f16 | 0 | `n_ctx` drops to 131072, f16 does not fit at 200k here |

| variant | workload | temp | other |
|---|---|---|---|
| `:s1` | co-work, agentic, tool use | 1.00 | `presence_penalty` 1.5, thinking on |
| `:s2` | deterministic coding | 0.20 | `min_p` 0.05, `repetition_penalty` 1.05 |
| `:s3` | independent SWE-bench baseline | 0.35 | `min_p` 0, `repetition_penalty` 1.0 |
| `:s4` | general multilingual | 0.60 | `repetition_penalty` 1.08 |

All four are transcribed from the model's published sampler table, not invented. Rows `:s2` and
`:s4` give temperature as a range; the profile takes the low end and the high end is one flag away,
`--model-config ...:s2 --temp 0.35`.

`extra_args` carries any llama-server flag, including ones this project does not know about:

```ini
[my-preset]
extra_args = --port 8099 --alias my-model --no-webui
```

It is spliced into argv before llama.cpp parses it, so upstream stays the only argument parser.
Command line wins on a conflict.

Samplers are configured here rather than in the client on purpose. The OpenAI-compatible provider
passes unrecognised `options` straight into the request body, but OpenCode never actually forwards
them: a request carrying `temperature` in the body arrives at llama-server with llama-server's own
default. Verified, not assumed.

## Zero quality loss

A hard constraint, and a structural one. Same ggml kernels, same weights, same KV dtypes, same
sampler, statically linked from the same commit.

With placement held fixed, llama.cpp's server and amp produce bit-identical output distributions.
KL is 0.000000e+00 in both directions, JS is 0, max absolute logprob difference is 0, over 512
greedy tokens. The engine is deterministic enough that a repeated run gives KL of exactly zero, so
this is a real zero rather than a small number.

Placement has to be held fixed for the comparison to mean anything. Mismatched placement produces
about 11% top-1 disagreement from a 0.012 logprob difference, which looks like a catastrophe and is
just a different layout. `scripts/kl_parity.py` is the gate.

## Speed, honestly

At steady state amp is not faster than a correctly configured `llama-server`. Over 5 identical
requests each, amp sustains 28.39 t/s decode and llama-server 30.05 t/s. An earlier version of
this file claimed 7.4x; that was amp warm against llama-server cold, and it did not reproduce.

What the plan buys is that it does not collapse. llama-server's documented configuration needs
about 11.5 GiB of CPU experts, more than this box keeps resident, so its first request runs at
1.96 t/s before recovering. amp holds the CPU set at 10.90 GiB, inside budget, and its first
request runs at 26 t/s. It also forces a single slot, because llama-server's four concurrent slots
thrash the same shared working set, measured at 0.64 t/s.

`docs/BENCH.md` has every number with its date and caveats, including the ones that were retracted.

## Where decode goes

The expert matvec is 30 to 36% of decode, measured three ways. It is also already at memory
bandwidth: it runs at 29.21 GB/s against 28.69 GB/s for a plain read of the same bytes, so the
dequant is fully hidden behind memory traffic.

That closes the MoE from both sides. Prefetching cannot help, because there is no exposed latency
left to hide. M3c was built anyway and measured at a mean +0.2% over nine paired comparisons, a
tie, with quality provably unchanged. A cheaper dequant kernel would not help either. The only
thing that moves the expert path is fewer bytes, which means quantization, which this project
rules out.

Expert placement saturates. g=0 to g=4 measures +3.4% for 10.6% of expert bytes moved off the CPU;
g=4 to g=7 measures 28.89 to 26.44 t/s, slightly worse; g=6 runs out of VRAM at 200k. An earlier
1.46x "all experts on GPU" extrapolation was wrong and is retracted. The GPU sits 70% idle during
decode and moving work onto it still does not help, because at batch 1 a GEMV cannot exploit
parallelism and competes with the recurrent layers already resident.

CPU-side levers, all measured and spent: threads are already optimal at `-t 8`, placement is worth
3.4%, the 200k context costs 2.6%, n-gram speculation 0.2%, expert prefetch 0.2%.

The unmeasured part is the other two thirds of decode: 30 recurrent layers plus 10 attention
layers, never profiled per-op. A per-op profile of the CPU side found that 100% of it is
`ggml_vec_dot_iq3_xxs_q8_K` and `ggml_vec_dot_q3_K_q8_K`, and that 53.6% is OpenMP barrier spin,
which is load balancing worth 31%. That closed the MoE question. The GPU side has never been
profiled per-op.

## Thinking models

The chat template renders the generation prompt already inside a `<think>` block, so the model
reasons before answering and never emits the opening tag.

`chat_template_kwargs: {"enable_thinking": false}` is the supported way to get a direct answer. The
same question costs 104 completion tokens thinking and 2 not thinking. `reasoning_content` is
split out of `content` automatically, in responses and in streams. Structured `tool_calls` come
back with a real id and parsed `arguments`; a thinking model needs about 66 tokens to reach a tool
call against 27 with thinking off. `reasoning_format` (`none`/`auto`/`deepseek`/`deepseek-legacy`)
is honoured. `reasoning_effort` is accepted and ignored, because this template has no such
capability. The reasoning budget is upstream's sampler, off by default.

## Tools and tests

```bash
./build/bin/amp-plan  --model $M             # what the planner decided, and why
./build/bin/amp-warm  --model $M --what plan # pull the expert set into the page cache
./build/bin/amp-infer --model $M --prompt hi # the forward path, with the prefetcher

python3 scripts/parity_test.py --url http://127.0.0.1:8081   # 15 sections over the API surface
./scripts/harness/run.sh --url http://127.0.0.1:8081          # the real client, 3 tool-using prompts
python3 scripts/bench_server.py --url http://127.0.0.1:8081 --n 5   # prefill/decode as a sequence
python3 scripts/kl_parity.py compare --a quality/a.json --b quality/b.json
```

Prefill needs a sequence of requests, not one. The same config restarted three times measured
132.5, 178.4 and 311.3 t/s on one identical prompt, because prefill at batch 1024 thrashes the page
cache. Decode at batch 1 does not, and varied only 26.4 to 27.7. An entire `n_ubatch` sweep was
invalidated by this and had to be thrown away.

`scripts/bench_prefill.py` and `scripts/bench_agentic.py` exist because of that.

## Where things are

| file | what is in it |
|---|---|
| `AGENT.md` | working rules and hard constraints. Read this first in a new session. |
| [`RUNNING.md`](RUNNING.md) | operating guide |
| `docs/BENCH.md` | every measurement, with dates, including retracted ones |
| `docs/ARCHITECTURE.md` | module layout and why each piece exists |
| `docs/PREFLIGHT.md` | the preflight, field by field, with the llama.cpp citations |
| `docs/PARITY.md` | the API contract, request and response shapes |
