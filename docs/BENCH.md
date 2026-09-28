# BENCH.md, measurements from this machine

RTX 4050 Laptop 6 GB, Ryzen 7 7735HS, 14 GiB RAM, Kingston QLC NVMe under LUKS+btrfs. Model:
Occamy-1.0 APEX-I-MiniPlus-V2.1-Abliterated.

This is a table of results, not a log of what was tried. The old version of this file was a dated
diary of 1200 lines and finding a number meant reading through the sessions that produced it. What
survived the rewrite is the measurement, the method, and the caveats that decide whether the number
means anything.

## Rules for anything measured here

**A speed claim needs a measurement in this file in the same commit.** Not asserted, not
extrapolated from a related figure.

**Prefill must be measured as a sequence of requests, never one.** The same config restarted three
times measured 132.5, 178.4 and 311.3 t/s on one identical prompt, because prefill at batch 1024
thrashes the page cache. Decode at batch 1 does not, and varied only 26.4 to 27.7. An entire
`n_ubatch` sweep was invalidated by this and discarded. `scripts/bench_prefill.py` exists for this
reason and is not optional.

**Always run the determinism control first.** Two identical-config runs give KL of exactly
0.000000e+00, JS 0, max absolute logprob difference 0, 512/512 top-1 agreement. Without that
control, a large KL result is indistinguishable from an instrument fault, and one earlier f16
comparison that looked catastrophic turned out to be the dtype.

**Hold placement fixed when comparing KV dtypes.** Mismatched placement gives about 11% top-1
disagreement from a 0.012 logprob difference. That looks like a quality regression and is only a
different layout.

## Baseline

Head to head, same prompt (18,265 tokens), same quant, same session, alternating which engine went
first. llama-server runs its best documented config (`-ncmoe 38 -ub 2048`); amp runs the planner's
choice.

| | amp | llama-server | |
|---|---|---|---|
| prefill, 18k prompt | **244.8 t/s** | 45-86 t/s | 2.8-5.4x |
| decode, first request after start | **26.01 t/s** | 1.96 t/s | 13x |
| decode, warm and steady | 28.39 t/s | 30.05 t/s | tie |
| decode, 4 slots generating at once | 21-33 t/s | 0.64 t/s | up to 44x |

Prefill is the median over six distinct 18k prompts at `-cram 0` with q8_0/q8_0 KV. The llama.cpp
range is what it does steady-state; its best ever observed here was 243 t/s with the expert set
fully cache-resident.

**Decode is a tie and that is the honest result.** The 30.05 vs 28.39 gap is inside this box's
run-to-run spread of about 20%, and decode has been recorded anywhere from 26.4 to 30.05 across
sessions. An earlier claim of 7.4x compared amp's warm state against llama-server's cold one. It
did not reproduce and it should not have been published.

**Where amp does win is not collapsing.** llama-server's documented config puts about 11.5 GiB of
expert weights on the CPU, against a page cache that holds 9.6-10.9 GiB depending on what else is
running. So its first request runs at 1.96 t/s
until the OS pulls the set in, and four slots generating at once collapse it to 0.64 t/s because
all four walk the same set. amp sizes the working set to fit and never does either.

## Memory system

| measurement | result |
|---|---|
| demand fault, cold, 4 KiB touched with readahead | 1737-1804 MiB/s, leaves 100% of pages resident |
| demand fault, warm | 3516-3620 MiB/s |
| `amp-warm` full expert set (12.19 GiB) | 6.34 s = 1.92 GiB/s |
| practical page cache ceiling | 9.6 GiB with ~5 GiB held by other processes, 10.9 GiB when it is mostly idle. Both are under the 12.19 GiB working set. |
| 4 KiB randread QD32 sustained (O_DIRECT) | 555-562 MB/s, flat |
| 128 KiB randread QD32 | 2099 MB/s |
| model file, contiguous | 545-547 MB/s |

The fio O_DIRECT 4 KiB number (555 MB/s) is not the relevant one. Expert streaming is sequential
with readahead, which measures 1.8 GiB/s, six times faster. The cost model uses 1.85 GB/s for the
fault path and 3.5 GB/s for the resident path.

The working set does not fit, and that is the structural fact the whole design rests on. With
12.19 GiB cyclically scanned through a smaller cache, LRU yields near-zero reuse and every ubatch
re-reads the entire set. What collapses is latency, not throughput: synchronous faults on the
compute thread cannot build queue depth. The fix is residency control plus I/O and compute
overlap, never a faster disk.

## Model byte budget

| item | bytes |
|---|---|
| MoE experts, 40 layers | 12.188 GiB (330 MiB/layer Q3_K on layers 0-9 and 30-39, 294 MiB/layer IQ3_XXS on 10-29) |
| other weights | 1489 MiB |
| KV cache at 65k context, q8_0/q8_0 | 0.51 GiB |
| SSM recurrent state, per sequence | 60 MiB |
| compute buffer | about 0.72 MiB per ubatch token |

Cross-check: 1489 weights + 1587 KV + 450 compute = 3526 MiB against a measured VRAM base of
3473 MiB. The accounting closes, which is why the planner's VRAM predictions can be trusted.

## Where decode time goes

| | share of a token |
|---|---|
| expert matvec, measured three ways | 30-36% |
| OpenMP barrier spin inside that | 53.6% of CPU samples |
| the other two thirds (30 recurrent + 10 attention layers) | not broken down per-op |

The expert matvec runs at 29.21 GB/s against 28.69 GB/s for a plain read of the same bytes. The
dequant is fully hidden behind memory traffic, so the kernel is at the memory rate and there is no
exposed latency for a prefetcher to fill. A per-op profile of 352,440 samples found 100% of CPU work
in two kernels, `ggml_vec_dot_iq3_xxs_q8_K` (21.2%) and `ggml_vec_dot_q3_K_q8_K` (16.6%).

That closes the MoE from both sides. Prefetching cannot help, because there is no latency left to
hide. A cheaper dequant kernel cannot help either. The only lever is moving fewer bytes, which means
quantization, which this project rules out.

Expert prefetch (M3c) was built anyway and measured at a mean +0.2% over nine paired comparisons: a
tie, with quality provably identical. It stays off by default. Patch in
`third_party/patches/m3c-expert-prefetch.patch`.

## Placement

| move | result |
|---|---|
| g=0 to g=4 | +3.4% decode for 10.6% of expert bytes moved to GPU |
| g=4 to g=7 | 28.89 to 26.44 t/s, worse |
| g=6 at 200k context | out of VRAM |
| g=10 | the turning point on this box |

Placement saturates. An earlier 1.46x "all experts on GPU" extrapolation came from fitting two
points on a curve that is not linear across the low end, and it was wrong. The GPU sits about 70%
idle during decode and moving work onto it still does not help, because at batch 1 a GEMV cannot
exploit parallelism and competes with the recurrent layers already resident.

## Levers measured and spent

| lever | result |
|---|---|
| threads | `-t 8` beats 4, 12 and 16 |
| expert prefetch | +0.2%, a tie |
| n-gram speculative decoding | loss. The accepted run length is a property of the model, not the window. |
| placement | 3.4%, then saturates |
| 200k context | costs 2.6% |
| `-cram 512` to `-cram 0` | about 1.1x prefill, not the 3x first claimed |
| prompt chunk 1024 to 2048 | +37.7% prefill, decode unchanged |

`OMP_WAIT_POLICY=passive` costs 31% (29.56 to 22.53 t/s) and must never be set.

## Quality

| KV | median KL vs f16/f16 |
|---|---|
| q8_0 / q4_0 | 1.809e-06 |
| q8_0 / q8_0 | 2.065e-07 |
| f16 / f16 | 0 |

q8_0/q8_0 is 8.8x closer to f16 than q4_0 V, and it is the shipped default. The KL metric saturates
once two arms generate different text, so only positions before the first token flip carry signal.
That is why the instrument scores a fixed token sequence rather than letting each arm generate.

## Things that are open

Nothing in this table has a number. That is deliberate: these are the questions where a number would
be the wrong output.

| | status |
|---|---|
| the 60% of decode outside the expert matvec | **TBD.** Never profiled per-op. Needs a GPU-side instrument this box does not have. |
| which sampler profile is best for agentic tool use | **TBD.** `:s1` through `:s4` all reach the server correctly, none has been measured against the others. `:s2` is the default. |
| a 104,884-character prompt processed as 18,269 tokens | **TBD, and a possible correctness bug.** Not the tokenizer; 4x the text tokenizes to 106,239 as a single string. |
| per-slot context at the shipped `n_parallel` | **TBD.** The 200k single-session figure was measured at `n_parallel 1`, which the shipped config no longer uses. |
| the planner's optimistic prefill prediction | **TBD.** It scores the `kPrefetch` I/O mode, which amp does not implement. `predicted_prefill_tps_faults` is the honest figure, and the offset is constant across candidates so ranking is unaffected. |

The truncation is the only item on this list that is not a performance question.

## Reproducing

```bash
./scripts/fetch_deps.sh && ./scripts/build.sh
M=/home/e0u/localhost/models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf

./build/bin/amp-plan --model $M                    # what the planner decided, and why
./build/bin/amp-warm --model $M --what experts     # pull the expert set into the page cache

# speed, always as a sequence
python3 scripts/bench_server.py  --url http://127.0.0.1:8081 --n 5
python3 scripts/bench_prefill.py --url http://127.0.0.1:8081
python3 scripts/bench_agentic.py --url http://127.0.0.1:8081

# quality, with the determinism control first
python3 scripts/kl_parity.py compare --a quality/a.json --b quality/b.json
```

## What was removed from this file

The dated session entries, roughly 1200 lines of them. They recorded the reasoning behind each
figure, which was worth writing down once and is not worth carrying forever. What is worth keeping
is here: the numbers, the method that produced them, and the two or three results that are wrong in
a way someone is likely to re-derive. The git history has the rest.
