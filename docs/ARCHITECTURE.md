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

The forward path (`amp-infer`) and the two pieces of state that make a server viable.

- `model_runtime.cpp` — tokenize, embed, 40 blocks (30 recurrent + 10 attention), sample. Uses the
  legacy `llama_batch` API; the prefetcher walks expert ranges ahead of the compute thread.
- `buft_overrides.{h,cpp}` — builds the null-terminated `tensor_buft_overrides` array that pins
  `ffn_*_exps.weight` to the CPU for the layers the plan left there (the `-ncmoe` equivalent).
  One helper, because a missing sentinel is a segfault and two call sites had it.
- `prefix_cache.{h,cpp}` — token-level longest-common-prefix matching across sequences, with
  **prompt-boundary checkpoints**: the sequence state is snapshotted at the end of the prompt and
  restored after generation, because 30 of this model's 40 layers are recurrent and cannot be rewound.

### amp_server

The HTTP layer, the engine, and the OpenAI translation.

- `http_server.cpp` — HTTP/1.1 with chunked framing, a small thread pool, SSE. Written from scratch;
  ~400 lines is cheaper than a dependency in a project that vendors exactly one C++ library.
- `task_queue.cpp` — **one worker, FIFO**. A `llama_context` is not reentrant, and agentic clients put
  two requests in flight at once (OpenCode asks for a conversation title while the main stream runs).
  Serialization lives inside `InferenceService::generate()` so no handler path can bypass it.
- `service.cpp` — owns the model, the context, the sequences, the prefix cache and the chat templates.
  Runs one generation at a time, warms the page cache at start-up, verifies VRAM after context init and
  backs the ubatch off until it fits.
- `openai_api.cpp` — request/response translation. Shapes are copied field for field from
  `tools/server/server-task.cpp`: the chunk envelope including `system_fingerprint`, the
  `stream_options.include_usage` trailing chunk with empty `choices`, and OpenAI's error `type`
  strings. A stream always ends with a `finish_reason` and then `[DONE]`, including on failure.

### amp_util additions

- `json.{h,cpp}` — minimal JSON, with every converting constructor explicit (see AGENT.md for the bug
  that forced it).
- `text.{h,cpp}` — `normalize_reasoning`, so a client that echoes a tagged `<think>...</think>` block
  cannot make the re-rendered prompt diverge on every turn.

## Known duplication

`ModelRuntime` (CLI) and `InferenceService` (server) are two implementations of the same forward loop:
the CLI has the prefetcher, the server has the cache and the task queue. Deliberate — the server was
built against the reference implementation rather than by refactoring a benchmark harness — and it is
the first thing to fix now that both are proven.
