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

## Planned modules (not yet written)

- `amp_backend` — compute backend abstraction over ggml (CPU, CUDA), with the VRAM planner feeding
  buffer placement decisions.
- `amp_runtime` — tokenization, the 40-block forward (attention + SSM + MoE), KV/SSM state,
  expert-major scheduling, sampler.
- `amp_server` — OpenAI-compatible HTTP API, slot prompt cache (token-level prefix matching),
  context checkpoints, and the `reasoning_content` / `reasoning_text` alias fix that makes OpenCode
  clients stop re-evaluating whole tool-call turns.
