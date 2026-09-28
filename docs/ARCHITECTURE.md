# ARCHITECTURE.md

How amp is put together, and why each piece exists. Module boundaries are enforced by the build
(one CMake target per module); interfaces exist wherever a second implementation is plausible.

```
  tools/       amp-plan  amp-warm  amp-server  amp-infer  kernel-bound
      |
  plan/        cost model, memory/device planner, preflight   [pure, testable]
      |
  model/       GGUF header reader + derived geometry          [pure, testable]
      |
  io/          mmap, page-cache warmer, streaming reads       [the memory system]
      |
  runtime/     the forward path used by amp-infer

  src/         format, bytes, log, timing                    [shared utilities]
```

Dependencies point downward only. `model/` and `plan/` are pure functions of the model file, so they
are fully testable without touching hardware; `io/` is the only module that does I/O.

`runtime/` and the preflight in `plan/` are the two pieces added after the first four settled, and
both are described at the end.

## Modules

### shared utilities (src/*.cpp)
`Status`/`Result<T>`, the only error channel across module boundaries. No exceptions cross an
interface. Also a tiny `format()`, byte parsing/printing, a leveled logger, `Stopwatch`, and
`read_meminfo()`. The meminfo reader exists because page-cache residency is the single most
important signal in this project; it is a first-class utility rather than something buried in a tool.

### io/
The memory system. Three pieces behind interfaces:

- `MappedFile`: read-only mmap. Never `MAP_POPULATE`, never an anonymous copy. A 14.65 GB
  anonymous load froze this machine once; file-backed pages are evictable and re-readable, which is
  what makes a working set larger than RAM survivable at all.
- `IPageCacheWarmer`: asynchronous page-cache warming. `readahead()` in 2 MiB chunks from a small
  worker pool, so the block layer accumulates queue depth instead of the compute thread stalling on
  each 4 KiB fault. Plus `warm_range_blocking()` for start-up, where blocking is the point.
- `IStreamReader`: `pread()` pool for bytes that should *not* enter the page cache. This is the
  escape hatch from LRU thrash: stream the cold tail into a small reusable buffer and keep the hot
  set pinned. io_uring slots in behind this interface later.
- `IReadScheduler`: composite that dispatches per `ReadPolicy` (`kCacheWarm` vs `kStream`). The
  runtime states an intent; the backends stay swappable.

### model/
`GGUFFile` parses the header only: 41 KV pairs and 733 tensor infos, a few hundred KB, with
sizes taken from ggml itself so amp's view of the file is byte-for-byte llama.cpp's view. That is
what makes "zero quality loss" structural instead of aspirational: the arithmetic is the same code.

`ModelGeometry` derives the facts the rest of the engine needs: per-layer expert bytes, per-expert
bytes, the attention/recurrent split, KV bytes per token, SSM state size.

### plan/
`CostModel` holds the measured constants of this box and turns a configuration into predicted
tokens/second. `MemoryPlanner` searches (expert layers on GPU) x (ubatch), scores with
`0.65 * log(prefill) + 0.35 * log(decode)`, and emits an `ExecutionPlan`: device assignment,
resident ranges to warm, streamed ranges, VRAM totals, and the top-N candidate ranking.

The cost model scores each candidate twice, once per I/O mode, and `predicted_prefill_tps_faults`
is the one that matches what actually ships:

| mode | bandwidth | overlap | corresponds to |
|---|---|---|---|
| `kPageFault` | 1.85 GB/s | none (synchronous) | what the server really does |
| `kPrefetch` | 2.0 GB/s | 0.75 | an overlap amp does not implement |

**The planner's headline `predicted prefill` figure uses `kPrefetch` and is therefore optimistic.**
`predicted_prefill_tps_faults` is the honest one. This is left as-is rather than corrected because
the fitter's job is to *rank* candidates, and the offset is the same for all of them, so the
ranking is unaffected. TBD: report the faults number in `amp-plan` output alongside the other, or
switch the headline to it.

## Design rules

1. **Never change the numbers.** Same ggml kernels, same KV dtypes, same mmap'd weights, F32
   routers, no quantization anywhere. Every speedup must come from memory behaviour, scheduling or
   kernel selection that is provably equivalent.
2. **Memory planning is the product.** This model's arithmetic is cheap (~240 t/s ceiling). What
   costs time is a 12.19 GiB expert working set competing with a ~9.6 GiB practical page cache.
3. **Plan against potential, report reality.** The planner budgets against RAM minus OS reserve,
   capped by what is realistically achievable; it reports both that and the momentary free cache, so
   an open browser cannot make the model look permanently un-cacheable.
4. **Measure before claiming.** `docs/BENCH.md` carries the numbers, and a perf claim without a
   measurement in the same commit is not acceptable.
5. **Bounded everything.** Queues, in-flight bytes, and thread pools all have explicit limits;
   unbounded prefetch on a 6 GB box is how you get an OOM instead of a speedup.

### runtime/

The forward path for `amp-infer` (the benchmark tool), on llama.cpp's kernels.

- `model_runtime.cpp`: tokenize, embed, 40 blocks (30 recurrent + 10 attention), sample. Uses the
  legacy `llama_batch` API; the prefetcher walks expert ranges ahead of the compute thread.
- `buft_overrides.{h,cpp}`: builds the null-terminated `tensor_buft_overrides` array that pins
  `ffn_*_exps.weight` to the CPU for the layers the plan left there (the `-ncmoe` equivalent).
  One helper, because a missing sentinel is a segfault and two call sites had it.

### plan/preflight, the whole of amp's server contribution

`src/plan/preflight.{h,cpp}`, plus `tools/amp_server.cpp` which is ~90 lines. There is no amp HTTP
layer, no amp OpenAI translation, no amp slot management, and no amp prompt cache. All of it is
llama.cpp's `tools/server`, linked in statically, driven through its exported entry point:

    common_params params;
    common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER);  // every upstream flag
    apply_preflight(opts, params, argv);                            // amp's plan -> common_params
    llama_server(params, 0, nullptr);                               // upstream's server

What the preflight actually does, and why each part exists:

1. **Never overrides an explicit flag.** llama.cpp discards its record of which args were supplied
   (`seen_args` at `common/arg.cpp:814` is local to the parser), so the preflight scans `argv`
   itself. The only reliable field-level tests are `n_gpu_layers` (-1 = unset) and
   `tensor_buft_overrides` (null first entry = unset).
2. **Device layout.** The plan's expert-layer count becomes `tensor_buft_overrides`, written *in
   place* into the buffer `common_params_parse` already padded to 4096 entries. Never `push_back`
   past the sentinel: `common.cpp:1706` asserts on it and the tensor loader walks to it.
3. **Context, ubatch, KV dtypes, threads, flash-attn.** Set only when the user did not pass them.
4. **Three clamps on llama.cpp defaults that are dangerous on this box**: `n_ctx_checkpoints` 32 to
   2 (each checkpoint is a *full* serialized sequence state, ~1.6 GiB at 200k), `cache_ram_mib` 8192
   to 512 (anonymous RAM that evicts the model's page cache), and `fit_params` off.
5. **Plans with the configuration it will actually apply.** This sounds obvious and was wrong twice:
   planning with `opts.n_ctx` instead of the user's `-c` cost 4 GPU expert layers, and planning with
   llama.cpp's default f16 KV while then setting q8_0/q4_0 cost another 4 layers and a 2.7x smaller
   ubatch. Both are recorded in docs/BENCH.md.

It never initialises a backend, loads the model, or creates a context. `llama_server()` owns all
of that, and doing any of it twice would read 12 GB of weights twice.

## One design decision worth recording

The hand-rolled server was deleted, about 3,800 lines including its prefix cache. The prefix cache
was real work and it is still the interesting part of the decision: 30 of 40 layers here are
recurrent, so the KV cannot be rewound, and the cache snapshotted the sequence at the prompt
boundary to work around that. llama.cpp solves it better, by asking the model what it supports
(`common_context_seq_rm_type`, `common.h:989-992`) and falling back to a full re-process with a log
line pointing at the upstream PR (`server-context.cpp:3379`).

Maintaining a version of a solved problem was the mistake, not writing it. Git has the deleted
code if anyone needs to know what it did.

## Known duplication

`ModelRuntime` (the `amp-infer` CLI) and llama.cpp's server context are two forward loops. This one
is deliberate and worth keeping: the CLI is the measurement harness, it has the prefetcher, and it
can be A/B'd against the server without a network round trip. It is not on the serving path.
