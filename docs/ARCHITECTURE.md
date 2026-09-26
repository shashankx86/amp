# ARCHITECTURE.md

How amp is put together, and why each piece exists. Module boundaries are enforced by the build
(one CMake target per module); interfaces exist wherever a second implementation is plausible.

```
  tools/            amp-plan, amp-warm, (later) amp-server, amp-bench
      |
  amp_plan          cost model + memory/device planner        [pure, testable]
      |
  amp_model         GGUF header reader + derived geometry     [pure, testable]
      |
  amp_io            mmap, page-cache warmer, streaming reads  [the memory system]
      |
  amp_util          format, bytes, log, timing, meminfo
```

Dependencies point downward only. `amp_model` and `amp_plan` are pure functions of the model file,
so they are fully testable without touching hardware; `amp_io` is the only module that does I/O.

## Modules

### amp_util
`Status`/`Result<T>` (the only error channel across module boundaries — no exceptions cross an
interface), a tiny `format()`, byte parsing/printing, a leveled logger, `Stopwatch`, and
`read_meminfo()`. The meminfo reader exists because page-cache residency is the single most
important signal in this project; it is a first-class utility rather than something buried in a tool.

### amp_io
The memory system. Three pieces behind interfaces:

- `MappedFile` — read-only mmap. Never `MAP_POPULATE`, never an anonymous copy. A 14.65 GB
  anonymous load froze this machine once; file-backed pages are evictable and re-readable, which is
  what makes a working set larger than RAM survivable at all.
- `IPageCacheWarmer` — asynchronous page-cache warming. `readahead()` in 2 MiB chunks from a small
  worker pool, so the block layer accumulates queue depth instead of the compute thread stalling on
  each 4 KiB fault. Plus `warm_range_blocking()` for start-up, where blocking is the point.
- `IStreamReader` — `pread()` pool for bytes that should *not* enter the page cache. This is the
  escape hatch from LRU thrash: stream the cold tail into a small reusable buffer and keep the hot
  set pinned. io_uring slots in behind this interface later.
- `IReadScheduler` — composite that dispatches per `ReadPolicy` (`kCacheWarm` vs `kStream`). The
  runtime states an intent; the backends stay swappable.

### amp_model
`GGUFFile` parses the header only — 41 KV pairs and 733 tensor infos, a few hundred KB, with
sizes taken from ggml itself so amp's view of the file is byte-for-byte llama.cpp's view. That is
what makes "zero quality loss" structural instead of aspirational: the arithmetic is the same code.

`ModelGeometry` derives the facts the rest of the engine needs: per-layer expert bytes, per-expert
bytes, the attention/recurrent split, KV bytes per token, SSM state size.

### amp_plan
`CostModel` holds the measured constants of this box and turns a configuration into predicted
tokens/second. `MemoryPlanner` searches (expert layers on GPU) x (ubatch), scores with
`0.65 * log(prefill) + 0.35 * log(decode)`, and emits an `ExecutionPlan`: device assignment,
resident ranges to warm, streamed ranges, VRAM totals, and the top-N candidate ranking.

The `IoMode` distinction is deliberate and load-bearing:

| mode | bandwidth | overlap | who |
|---|---|---|---|
| `kPageFault` | 1.85 GB/s | none (synchronous) | llama.cpp today |
| `kPrefetch` | 2.0 GB/s | 0.75 | amp |

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
   measurement in the same commit is not acceptable. Two "measured" constants were already wrong
   when first written (KV byte math, fault bandwidth) and the tests caught both.
5. **Bounded everything.** Queues, in-flight bytes, and thread pools all have explicit limits;
   unbounded prefetch on a 6 GB box is how you get an OOM instead of a speedup.

### amp_runtime

The forward path for `amp-infer` (the benchmark tool), on llama.cpp's kernels.

- `model_runtime.cpp` — tokenize, embed, 40 blocks (30 recurrent + 10 attention), sample. Uses the
  legacy `llama_batch` API; the prefetcher walks expert ranges ahead of the compute thread.
- `buft_overrides.{h,cpp}` — builds the null-terminated `tensor_buft_overrides` array that pins
  `ffn_*_exps.weight` to the CPU for the layers the plan left there (the `-ncmoe` equivalent).
  One helper, because a missing sentinel is a segfault and two call sites had it.

### amp_preflight — the whole of amp's server contribution

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
2. **Device layout** — the plan's expert-layer count becomes `tensor_buft_overrides`, written *in
   place* into the buffer `common_params_parse` already padded to 4096 entries. Never `push_back`
   past the sentinel: `common.cpp:1706` asserts on it and the tensor loader walks to it.
3. **Context, ubatch, KV dtypes, threads, flash-attn** — set only when the user did not pass them.
4. **Three clamps on llama.cpp defaults that are dangerous on this box**: `n_ctx_checkpoints` 32 to
   2 (each checkpoint is a *full* serialized sequence state, ~1.6 GiB at 200k), `cache_ram_mib` 8192
   to 512 (anonymous RAM that evicts the model's page cache), and `fit_params` off.
5. **Plans with the configuration it will actually apply.** This sounds obvious and was wrong twice:
   planning with `opts.n_ctx` instead of the user's `-c` cost 4 GPU expert layers, and planning with
   llama.cpp's default f16 KV while then setting q8_0/q4_0 cost another 4 layers and a 2.7x smaller
   ubatch. Both are recorded in docs/BENCH.md.

It never initialises a backend, loads the model, or creates a context — `llama_server()` owns all
of that, and doing any of it twice would read 12 GB of weights twice.

## What was deleted, and why

The hand-rolled server (`src/server/`, ~2,200 lines) plus the code that only existed to serve it
(`prefix_cache`, `util/json`, `util/text`, ~1,600 more) is gone. It implemented 7 endpoints;
llama.cpp's server implements 40+ and is maintained against the commit we pin.

The prefix cache is the deletion worth arguing for, because it was real work: 30 of 40 layers here
are recurrent, so the KV cannot be rewound, and it snapshotted the sequence at the prompt boundary
to work around that. llama.cpp solves it better — it asks the model what it supports via
`common_context_seq_rm_type` (`common.h:989-992`), keeps context checkpoints with `pos_min`/`pos_max`
ranges, and falls back to a full re-process with a log line pointing at the upstream PR that added
it for hybrid/recurrent memory (`server-context.cpp:3379`). Maintaining our own version of a solved
problem was the mistake, not writing it.

## Known duplication

`ModelRuntime` (the `amp-infer` CLI) and llama.cpp's server context are two forward loops. This one
is deliberate and worth keeping: the CLI is the measurement harness, it has the prefetcher, and it
can be A/B'd against the server without a network round trip. It is not on the serving path.
