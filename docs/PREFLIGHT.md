# PREFLIGHT.md, amp's contribution to the llama-server-based `amp-server`

`amp-server`'s `main()` is now:

```cpp
common_params params;
common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER);   // full llama.cpp flag parity
auto notes = apply_preflight(opts, params, argv);                 // amp's entire contribution
for (auto & n : notes) log_line(...);
llama_server(params, nullptr, nullptr);                           // blocks until shutdown
```

`apply_preflight` (`src/plan/preflight.cpp`, `include/amp/plan/preflight.h`) turns amp's
`MemoryPlanner` into `common_params` fields **before** the model loads. It never loads the
model, never creates a context, never calls `llama_backend_init`/`llama_numa_init`
(`llama_server()` does all of that, `tools/server/server.cpp:111-112`), and never overrides
an argument the user explicitly passed.

All citations are to the vendored llama.cpp at `third_party/llama.cpp` (git `1ab7e5a`,
`third_party/DEPS.lock`).

---

## 1. The ordered sequence: `main()` to the first HTTP request

| # | Step | Where | Notes |
|---|------|-------|-------|
| 1 | `common_params params;` | amp `main` | all defaults from `common/common.h:448-759` |
| 2 | `common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)` | `common/arg.cpp:1277` | **inside** this call: config file (`arg.cpp:769`), env vars (`arg.cpp:781-804`), CLI (`arg.cpp:878`), `postprocess_cpu_params` resolves `n_threads<0` (`arg.cpp:880-884`, `common/common.cpp:288-310`), and `tensor_buft_overrides` is **padded to 4096** (`arg.cpp:940-944`) |
| 3 | `apply_preflight(opts, params, argv)` | `src/plan/preflight.cpp` | opens the GGUF header (`GGUFFile::open`), builds geometry, detects the budget (`detect_device_budget`, nvidia-smi), runs `MemoryPlanner::plan`, then mutates `params` (section 3 table). No backend init, no model load, no context. |
| 4 | `llama_server(params, nullptr, nullptr)` | `tools/server/server.cpp:119` | `is_run_by_cli = true` (`server.cpp:120`) |
| 5 | `llama_backend_init(); llama_numa_init(params.numa);` | `server.cpp:111-112` | `numa` defaults to `GGML_NUMA_STRATEGY_DISABLED` (`common/common.h:494`) -> no-op |
| 6 | `common_params_print_info(params, true)` | `server.cpp:144`, `common/common.cpp:397-419` | prints verbosity; device enumeration only at TRACE (`common.cpp:409`) -> no CUDA context created |
| 7 | `n_parallel < 0` -> `n_parallel = 4; kv_unified = true;` | `server.cpp:156-161` | server default (`-np` is `-1` for `LLAMA_EXAMPLE_SERVER`, `arg.cpp:1399-1400`) |
| 8 | `ctx_http.init(params)`, route registration, `mcp_mgr.start` | `server.cpp:194-374` | no model work |
| 9 | `ctx_http.start()` | `server.cpp:462` | **HTTP listener is up before the model loads**, so `/health` works during load |
| 10 | `ctx_server.load_model(params)` | `server.cpp:475`, `server-context.cpp:1006` | copies params -> `params_base` (`server-context.cpp:1013`), then `common_init_from_params` (`server-context.cpp:1099`) |
| 11 | `common_init_result` ctor | `common/common.cpp:1290-1409` | `common_model_params_to_llama` (`:1292`), `common_context_params_to_llama` (`:1293`), **fit runs here if `fit_params`** (`:1295-1327`), `llama_model_load_from_file` (`:1329`), `llama_init_from_model` (`:1398`) |
| 12 | model load: `init_mappings(true, ...)` | `src/llama-model.cpp:1748` | `mmap` + **`MAP_POPULATE`** (`src/llama-mmap.cpp:480`) + `posix_fadvise(SEQUENTIAL)` (`:475`) + `MADV_WILLNEED` whole file (`:500-504`) — **the whole GGUF is faulted into the page cache here** (section 6) |
| 13 | context init | `src/llama-context.cpp:84+` | `n_batch=min(n_ctx,·)` (`:246`), `n_ubatch=min(n_batch,·)` (`:248`), `n_ctx` padded to 256 (`:289`), `n_ctx_seq` split or unified (`:291-305`) |
| 14 | `routes.update_meta`, `ctx_http.is_ready = true` | `server.cpp:482-483` | first request can be served |
| 15 | `ctx_server.start_loop()` | `server.cpp:542` | blocks |

**Where the warm goes (section 6):** the warm must be **after the plan** (it warms
`plan.resident`) and its effect must survive the load. Because step 12 fault-in would evict a
pre-load warm, the warm either (a) is skipped by default, or (b) opts in with
`lazy_mode = ON` so step 12 does not populate. Both are implemented inside `apply_preflight`
(step 3), because step 3 is the only amp-controlled code that runs before step 12.

---

## 2. Does the startup already read the whole file? (warm redundancy)

**Yes.** With the default `load_mode = AUTO` and `lazy_mode = AUTO`:

- `use_mmap = true` (`src/llama-model-loader.cpp:559`), and `lazy` stays `AUTO` because no
  tensor exceeds the 4 GiB `auto_min_size` (`src/llama-model-loader.cpp:1094) ->
  `lazy_ranges` is empty.
- Empty `lazy_ranges` ⇒ `MAP_POPULATE` (`src/llama-mmap.cpp:480`) **and**
  `posix_madvise(WILLNEED)` over the whole file (`:500-504`) **and**
  `posix_fadvise(SEQUENTIAL)` (`:475`).

So `llama_model_load_from_file` performs a full sequential fault-in of the 12.19 GiB GGUF
at readahead speed. There is **no** `mlock` (load_mode AUTO ≠ MMAP_MLOCK,
`src/llama-model.cpp:1748`), **no** warmup inference (`llama_model_warmup` does not exist in
this tree; `cparams.warmup` is forced `false` at `src/llama-context.cpp:123`; the only
`warmup` consumers are graph-building hints in `src/llama-graph.cpp:1472`), and the fit,
if it ran, would load the model a **second** time with `no_alloc` (`common/fit.cpp:56-60`,
reached from `common/common.cpp:1320`). Disabling the fit (Rule 2) removes that second read.

**Consequence for the warm:** a warm *before* the load cannot leave the hot set more
resident than the load's own tail-heavy eviction allows, because the sequential populate evicts the
file front (where the hot set lives), so a pre-load warm front-loads I/O but leaves a
*deeper* cold spot than not warming. Hence the default is **no warm**; `opts.warm` opts into
the old amp-warm behavior with `lazy_mode = ON` to protect it.

---

## 3. Field table: who sets it, how we detect user intent, default when unset

**The core problem:** llama.cpp's parser keeps **no** record of which args were supplied.
`seen_args` (`common/arg.cpp:814`) is local to `parse_cli_args` and discarded on return.
Most fields have no surviving unset sentinel: `postprocess_cpu_params` resolves
`n_threads < 0` *during* the parse (`common/common.cpp:291-298`), and `n_batch`/`n_ubatch`/
`n_ctx` default to real values (`common/common.h:450-452`). So for most fields the preflight
**scans `argv` itself**, which is the only reliable test. The two exceptions
(`n_gpu_layers`, `tensor_buft_overrides`) have reliable field tests.

| `common_params` field | Set by (arg) | "User set it" test | Default when unset |
|---|---|---|---|
| `tensor_buft_overrides` | `-ncmoe`/`-cmoe`/`-ncffn`/`-ot` (`arg.cpp:2752-2782`) | **field test**: any entry with non-`nullptr` pattern (user entries are pushed to the front before the padding) | plan's `n_expert_layers_gpu` -> pin first `n_layer - g` layers' experts (the `-ncmoe` equivalent) |
| `n_gpu_layers` | `-ngl` (`arg.cpp:2784-2801`) | **field test**: `!= -1` (−1 is both "auto" and the default, `common.h:473`) | left at −1 (all non-overridden tensors to GPU — the measured config) |
| `fit_params` | `-fit` (`arg.cpp:2864-2870`) | **argv**: `-fit` present; value via `is_truthy`/`is_falsey` (`arg.cpp:1331-1337`) | set **false** (our planner replaces the fitter) |
| `n_ctx` | `-c` (`arg.cpp:1635-1645`) | **argv** | `opts.n_ctx` (200000) |
| `n_ubatch` | `-ub` (`arg.cpp:1672-1678`) | **argv** | `plan.ubatch` (largest that fits VRAM) |
| `n_batch` | `-b` (`arg.cpp:1665-1671`) | **argv** | default 2048; raised only to cover a larger `n_ubatch` |
| `cache_type_k` | `-ctk` (`arg.cpp:2433-2445`) | **argv** (field default F16 is indistinguishable from explicit) | `opts.cache_k` (q8_0) |
| `cache_type_v` | `-ctv` (`arg.cpp:2446-2458`) | **argv** | `opts.cache_v` (q4_0) |

**And the same intent test decides what the planner is *given*, not just what is set.** See
"the plan must use the configuration that will be applied" below - this is not a detail, it
was worth 4 GPU expert layers and a 2.7x smaller ubatch at `-c 200000`.
| `flash_attn_type` | `-fa` (`arg.cpp:1751-1765`) | **argv** (default AUTO, `common.h:499`) | `LLAMA_FLASH_ATTN_TYPE_ENABLED` |
| `cpuparams.n_threads` | `-t` (`arg.cpp:1513-1522`) | **argv** (field resolved during parse) | `opts.n_threads` / `budget.n_cpu_threads` (8) |
| `cpuparams_batch.n_threads` | `-tb` (`arg.cpp:1523-1532`) | not set | follows `n_threads` (`common.cpp:1729-1730`) |
| `n_ctx_checkpoints` | `-ctxcp` (`arg.cpp:1694-1701`) | **argv** | clamped to `opts.n_ctx_checkpoints` (2) |
| `cache_ram_mib` | `-cram` (`arg.cpp:1712-1719`) | **argv** | clamped to `opts.cache_ram_mib` (512) |
| `lazy_mode` | `-lzm` (`arg.cpp:2706-2714`) | **argv** | left AUTO, **or** forced ON when `opts.warm` |
| `kv_unified` | `-kvu` (`arg.cpp:1720-1727`) | not set | left alone, see below |
| `n_parallel` (-> `n_seq_max`) | `-np` (`arg.cpp:2540-2549`) | not set | left alone; the server forces 4 + `kv_unified` (`server.cpp:156-161`) |
| `checkpoint_min_step` | `-cms` (`arg.cpp:1702-1711`) | not set | left at default 8192 |

**`-ngl-experts` does not exist in this llama.cpp version.** The only expert-placement
flags are `-ncmoe` / `-cmoe` / `-ncffn` (`arg.cpp:2755-2782`) and the generic
`-ot`/`--override-tensor` (`arg.cpp:2749-2754`); there is no `--n-gpu-layers-experts`
anywhere in `arg.cpp`. Expert placement is therefore expressed the only way this version
allows: `n_gpu_layers` (which layers are offloaded) plus `tensor_buft_overrides` (which
tensors inside the non-offloaded layers stay on the CPU), exactly what the old runtime did
with `n_gpu_layers = -1` + overrides (`src/server/service.cpp:333-345`).

**`kv_unified` / `n_seq_max` are not set, deliberately.** The server maps `n_seq_max ←
n_parallel` (`common/common.cpp:1722`) and forces `kv_unified = true` whenever
`n_parallel < 0` (`server.cpp:156-161`), which is the server default (`arg.cpp:1399-1400`).
Setting `params.kv_unified = false` in the preflight would be silently overridden. The KV
pool is `n_ctx_seq = n_ctx` when unified and `n_ctx / n_seq_max` per slot otherwise
(`src/llama-context.cpp:291-305`). For this model both give the same 200k pool, so leaving
the server default (unified, 4 slots) is correct *and* gives amp real concurrency for the
first time. `-ctk`/`-ctv` are the reachable mappings for `type_k`/`type_v`
(`common/common.cpp:1751-1752`). `n_ctx_checkpoints`/`checkpoint_min_step` are **not** in
`llama_context_params`. The server reads them straight from `params_base`
(`server-context.cpp:1365-1367`).

**No-op rules.**
- `-fit on` (explicit) ⇒ **full no-op**: the fitter is in charge of device memory and would
  throw on any layout we set (`common/fit.cpp:463-465`, `484-486`).
- Any layout intent (`-ngl` with a value, or any of `-ncmoe`/`-cmoe`/`-ncffn`/`-ot`) ⇒ the
  preflight leaves `n_gpu_layers` and `tensor_buft_overrides` untouched, but still applies the
  non-layout defaults and the Rule-8 safety guards (a manual `-ngl` says nothing about
  checkpoint count or prompt-cache RAM).

---

## 4. The `tensor_buft_overrides` padding and segfault question, definitive answer

**Claim:** the preflight can safely write overrides into `params.tensor_buft_overrides`
because llama.cpp has *already padded the buffer* by the time the preflight runs.

**Call order (all inside step 2):**

1. `common_params_parse` (`common/arg.cpp:1277`)
2. -> `common_params_parse_ex` (`arg.cpp:762`)
3. -> `parse_cli_args()`. User `-ot`/`-ncmoe`/... handlers `push_back` real entries
   (`arg.cpp:2752-2782`)
4. -> **padding**: `common_params_parse_ex` pads the vector to
   `llama_max_tensor_buft_overrides()` with `{nullptr, nullptr}` (`arg.cpp:940-944`)
5. -> back in `common_params_parse`, returning to amp's `main`
6. -> `apply_preflight`. The vector has exactly 4096 entries (`src/llama.cpp:90-92`),
   unused ones null.

So **the padding is already done**; the preflight must **not** push past it. It writes our
overrides into slots `[0, k)` and keeps slot `k` and beyond null (`write_expert_cpu_overrides`,
`src/plan/preflight.cpp:108-141`), with a defensive re-pad if the invariant is somehow broken.

**Why the sentinel is non-negotiable (two landmines):**

- `common_model_params_to_llama` asserts `tensor_buft_overrides.back().pattern == nullptr`
  (`common/common.cpp:1706`). `GGML_ASSERT` is **always active**, mapping to `GGML_ABORT`
  (`ggml/include/ggml.h:288`), not disabled by `NDEBUG`. A non-null `back()` aborts the
  process at `common.cpp:1706`, before the model loads. The arg.cpp:940-944 padding is what
  makes this assert pass even when the user passed no overrides.
- The tensor loader walks `for (overrides = ...; overrides->pattern != nullptr; ++overrides)`
  and builds a `std::regex` from each pattern (`src/llama-model-loader.cpp:1235-1259`). A
  missing sentinel reads past the array end: segfault, or silent garbage (a regex built from
  a wild pointer). This is the failure mode AGENT.md:214-215 records.

**Pattern lifetime:** the override patterns are `const char *` into strings that must
outlive the load. The preflight keeps them in a function-local `static std::list<std::string>`
, the same technique llama.cpp's own `llm_add_n_cpu_ffn_overrides` uses
(`common/common.h:1147-1154`); `std::list` never moves existing nodes, so the `c_str()`
pointers stay valid for the process lifetime.

---

## 5. llama.cpp's fitter (`--fit`) vs amp's planner

**What it does:** `common_fit_params` (`common/fit.cpp:879`) loads the model with `no_alloc`
(`fit.cpp:56-60`), measures per-device memory with a real context (`fit.cpp:66`, `:264`),
and if the projection does not fit, reduces `n_ctx` (`fit.cpp:393-457`) and then iteratively
assigns layers/devices back-to-front (`fit.cpp:488-876`), writing `n_gpu_layers`,
`tensor_split` and `tensor_buft_overrides` as it goes. Its target is "leave `fit_params_target`
(default 1 GiB, `common.h:481`) free per device".

**When it runs:** inside `common_init_result`'s constructor (`common/common.cpp:1295-1327`),
i.e. **after** our preflight (step 3) and **before** `llama_model_load_from_file`
(`common.cpp:1329`). It receives `params.tensor_buft_overrides.data()` as a writable
4096-entry buffer (`common.cpp:1322`), the second consumer that depends on the arg.cpp
padding.

**Would it fight amp's planner? Yes, three ways:**

1. **It throws on our layout.** `if (mparams->n_gpu_layers != default_mparams.n_gpu_layers)`
   -> `"n_gpu_layers already set by user, abort"` (`fit.cpp:463-465`); same for a non-null
   first override (`fit.cpp:484-486`). The exception is caught (`fit.cpp:894-900`) and the
   status is **ignored** by the caller (`common.cpp:1320`), so the plan survives, but...
2. **It wastes a full no_alloc model load** before throwing (`fit.cpp:264`, plus `:271-278`
   when `n_ctx == 0`), and prints a scary `failed to fit params` warning.
3. **A successful fit would overwrite our overrides** (`fit.cpp:568-585`, `:620-621`) and
   re-fill the GPU to leave 1 GiB free, the opposite of the planner's 6-8-expert-layer
   choice. And it would reduce `n_ctx` if the user left it at 0 (`fit.cpp:393-457`).

**Resolution:** the preflight sets `fit_params = false` (Rule 2) unless the user explicitly
passed `-fit on` (full no-op, section 3). This is not overriding an explicit choice,
`fit_params` defaults to `true` (`common.h:476`) and the user did not ask for it. Setting
`n_ctx` explicitly (Rule 3) additionally protects the context from reduction
(`fit.cpp:454-456`).

---

## 6. The VRAM-adaptive ubatch, recommendation

**Old behaviour** (`src/server/service.cpp:354-403`): create the context, measure free VRAM
via `ggml_backend_dev_get_props`, and if under budget free the context and retry with
`ubatch/2` (8 attempts, floor 128). This **cannot survive the move**: the preflight runs
before any context exists, and the rules forbid creating one.

**Recommended: pick the ubatch analytically, from the existing planner.** The planner
already ranks candidates by `vram_bytes` against `vram_usable()`
(`src/plan/memory_plan.cpp:142-216`), where `vram_bytes` includes the measured 450 MiB CUDA
context overhead (`include/amp/plan/cost_model.h:61-66`) and a 0.92 safety factor on the
compute buffers (`cost_model.h:58-60`). The preflight measures the budget with
`detect_device_budget()` (nvidia-smi, `src/plan/memory_plan.cpp:121-130`, no backend init
needed) and lets the planner choose. This is the simplest approach that keeps the measured
behaviour: the process still never dies from a too-large ubatch, because the planner
*rejects* candidates that do not fit (`memory_plan.cpp:209-216`) and picks the largest that
does.

**Tradeoff (be explicit about it):** the old loop measured *actual* free VRAM after init;
the analytic path estimates it. If the estimate is wrong by more than the safety margin,
`llama_init_from_model` returns `NULL` and llama-server exits with a clear
`failed to create context` (`common/common.cpp:1398-1402`) instead of backing off. The
mitigations: the constants were calibrated on this exact machine (`cost_model.h:58-66`
records the measurement), the 0.92 factor is conservative, and the failure mode is a clean
error message, not a freeze. The first validation run should diff the planner's
`vram_bytes` against what the server logs at context creation (`src/llama-context.cpp:307-319`)
and the memory breakdown printed at shutdown (`server.cpp:550-553`,
`common/fit.cpp:906-1086`), and adjust `vram_context_bytes` / `vram_safety` if they disagree.
The alternative of leaving the fit on to choose the ubatch is worse. The fit throws on our
layout (section 5) and its 1 GiB-free target fights the planner.

---

## 7. Landmines that would silently undo amp's plan (or double the memory)

1. **The model load's full-file fault-in** (`src/llama-mmap.cpp:480`, `:500-504`), which reads
   the whole GGUF front-to-back and evicts the file front. This is why the default warm is
   off (section 2). *Not* a problem for the layout itself: the hot set (10.25 GiB) fits the
   ~11.5 GiB cache, so after the load only the hot set's first ~0.7 GiB is cold and the
   first decode pass faults it in with readahead (no `MADV_RANDOM` is applied on the
   default path), after which it stays resident.
2. **`n_ctx_checkpoints` = 32 default** (`common.h:629`). Each checkpoint stores the full
   memory state at its position: ~1.6 GiB at 200k ctx (8320 B/token KV **plus** the 62.81
   MiB recurrent state). The ring (`server-context.cpp:2331-2339`) can reach ~50 GiB. On a
   14.3 GiB machine this OOMs. Clamped to 2 unless the user set `-ctxcp`.
3. **`cache_ram_mib` = 8192 default** (`common.h:632`). The prompt cache is anonymous RAM
   (`server-task.cpp:1711-1758`) bounded at 8 GiB; every cached prompt state evicts model
   pages -> the decode cliff. Clamped to 512 unless the user set `-cram`.
4. **The fit** (section 5), disabled by Rule 2.
5. **A second context**, which only the fit creates (`common/fit.cpp:66`). With the fit
   off there is exactly one `llama_init_from_model` (`common.cpp:1398`). No mmproj, no
   draft model.
6. **`common_memory_breakdown_print`**, which runs once after the server loop exits
   (`server.cpp:550-553`), read-only. Noisy at shutdown, harmless.
7. **`common_numa`**: `llama_numa_init(params.numa)` (`server.cpp:112`) with
   `numa = GGML_NUMA_STRATEGY_DISABLED` by default (`common.h:494`) -> no-op. If a user
   passes `--numa`, the big buffers are page-cache-backed (mmap) and unaffected; only the
   small anonymous ones would be interleaved.
8. **`llama_mmap` unmaps the gaps** between tensor ranges after load
   (`src/llama-model-loader.cpp:1779-1785`), which is only alignment gaps and no tensor bytes.

---

## 8. What could still go wrong

- **The analytic ubatch is wrong** (section 6), giving a clean startup failure rather than a freeze.
  Calibrate `vram_context_bytes` / `vram_safety` against the post-init free VRAM on first run.
- **The kernel's LRU behaviour under `MAP_POPULATE` differs from the model in section 2.**
  The claim "a pre-load warm deepens the cold spot" follows from front-to-back eviction;
  the kernel's readahead windows and reclaim heuristics are not guaranteed to match.
  **Resolve empirically:** `mincore()` the hot set's pages after the load, with and without
  a pre-load warm, and confirm which leaves more of the hot set resident.
- **`opts.warm` + `lazy_mode = ON` suppresses readahead for the GPU tail upload.** With lazy
  ON, the whole file is marked `MADV_RANDOM` (`src/llama-mmap.cpp:505-507`), so the load's
  upload of the tail experts plus dense weights (~2.8 GiB) runs at ~555 MB/s, not ~2 GB/s,
  a one-time ~5 s. Acceptable for the deterministic residency it buys; documented here.
- **A user passing `-fit on` gets the full no-op** (section 3), including no checkpoint /
  prompt-cache clamps. That is their explicit choice; the RAM risks in section 7 remain.
- **`-c 0` (max context) or a user `-c` above 200000**: the KV cache no longer fits the VRAM
  budget and `llama_init_from_model` fails. The error is clear; the fix is a smaller `-c`.
- **Future llama.cpp changes**: the padding (arg.cpp:940-944), the fit's throw checks, and
  the `MAP_POPULATE` load are all version-pinned (`third_party/DEPS.lock`); if the pin moves,
  re-verify sections 2, 4 and 5.
- **The preflight is ~430 lines** (vs the ~300 guideline). The design is 9 small rules plus
  a warm; the length is the `file:line` citations and per-field notes this task requires, not
  complexity. The one genuinely chunky helper (`to_ggml`, 16 cases) is a mechanical enum
  mapping copied from `src/server/service.cpp:26-43`.

---

## 9. Verification status

**Verified by reading the source (this document):** every `file:line` claim above, the parse
order and padding (`arg.cpp`), the conversion asserts (`common.cpp:1706`,
`common.cpp:1292-1327`), the fit's throw/catch/ignore behavior (`fit.cpp`, `common.cpp:1320`),
the loader's sentinel walk (`llama-model-loader.cpp:1237`), the `MAP_POPULATE` + fadvise +
WILLNEED load path (`llama-model.cpp:1748`, `llama-mmap.cpp:475-507`), the server's
kv_unified/n_parallel forcing (`server.cpp:156-161`), the context param clamps
(`llama-context.cpp:246-248`), and the absence of a warmup inference and of `llama_model_warmup`.

**Assumed / not yet measured:** that the planner's `vram_bytes` matches reality within the
0.92 margin on the first run (section 6 tradeoff); the exact LRU behaviour under `MAP_POPULATE`
(section 8); and that `opts.warm`'s one-time ~5 s tail upload is acceptable. All three are
empirical and should be measured when the module is first compiled and run.

## The plan must use the configuration that will be applied

Two bugs, both found by running the plan at 200k and comparing it against `amp-plan -c 200000`,
which disagreed. Both were *pessimistic*, so they cost performance and never produced a wrong
answer - the dangerous class, because nothing fails loudly.

### 1. The plan ignored the user's `-c`

`PlannerOptions::n_ctx` was set from `opts.n_ctx` (200000) rather than from what the server would
actually create. Asking for `-c 32768` still reserved 200k worth of KV: 1.55 GiB of VRAM budget for
a cache that would be 260 MiB.

Fix: plan against `params.n_ctx`, which `common_params_parse` has already resolved from `-c` /
`--ctx-size` / `LLAMA_ARG_CTX_SIZE`. It is 0 when neither was given, which means "use the model's
native context", not "a context of zero" - hence the fallback to `opts.n_ctx`.

Measured at `-c 32768`:

| | expert layers on GPU | CPU expert set | KV budgeted |
|---|---|---|---|
| before | 4 | 10.90 GiB | 1.55 GiB (should be 260 MiB) |
| after | **8** | **9.61 GiB** | 260 MiB |

### 2. The plan computed KV at f16 while the server ran q8_0/q4_0

llama.cpp's field default is F16 (`common/common.h:587-588`). The preflight read
`params.cache_type_*` for planning and only *afterwards* overwrote them with the measured
q8_0/q4_0 baseline (Rule 5). f16 KV is **2.46x** the bytes, so at 200k the plan reserved 3.81 GiB
for a cache that is really 1.55 GiB - which crushed the VRAM budget for everything else.

Measured at `-c 200000`:

| | expert layers on GPU | ubatch | KV | predicted decode |
|---|---|---|---|---|
| before | 0 | 384 | 3.81 GiB (planned) | 1.9 t/s |
| after | **4** | **1024** | 1.55 GiB (actual) | 15.5 t/s |

Fix: `po.cache_k = f.ctk ? from_ggml(params.cache_type_k) : opts.cache_k`, and the same for v. The
`f.ctk` / `f.ctv` flags are the *same* user-intent test Rule 5 uses to decide whether to override,
so the plan and the applied configuration cannot drift apart.

**The general rule: a planner must be given the configuration it is planning for.** Reading a
"resolved" field before your own rules have run is the same bug wearing a different hat.

## Model configs: `--model-config <name>:v<N>`

Named presets with a quality dial, in `configs/model-configs.conf`. Three variants for
`occamy-1.0-apex-i-miniplus`, ordered by KV quality:

| variant | KV | median KL vs f16/f16 | notes |
|---|---|---|---|
| `:v1` | q8_0 / q4_0 | 1.809e-06 | the old default, smallest KV |
| `:v2` | q8_0 / q8_0 | 2.065e-07 | the default now, 8.8x closer to f16 |
| `:v3` | f16 / f16 | 0 | f16 KV; `n_ctx` cut to 131072 because f16 does not fit at 200k |

Three design decisions, each of which cost something to get right:

**A preset is exactly equivalent to typing its flags.** It sets the same `common_params` fields and
raises the same `UserFlags` bits a typed flag raises, so every downstream rule treats it as a user
choice and leaves it alone. There is no second code path. The alternative - a separate "preset
applies after the rules" step - would have meant two places where a setting can be decided, and
therefore two places to be wrong.

**An explicit flag outranks the preset.** The first version had this inverted and the symptom was
silent: `--model-config ...:v3 -c 200000 -ctv q8_0` applied the *preset's* 131072 and f16, because
the preset wrote `params` unconditionally after `scan_user_flags` had already recorded the flags.
The guard has to skip a key *before* the dispatch chain rather than by compounding each condition,
because a compound condition falls through to the unknown-key branch and rejects a perfectly good
key. The log now reports which settings the preset set and which were kept from the command line.

**The file is found relative to the executable, not the working directory.** A relative path works
when the server is started from the repo root and silently finds nothing from `build/`, which is
where ctest runs. That is how the config was missing during development. Resolved via
`/proc/self/exe`; `AMP_MODEL_CONFIGS` overrides.

Two smaller things, both of which bit:

- `--model-config` is amp's flag and `common_params_parse` rejects anything it does not know, so it
  is filtered out of the vector handed to llama.cpp and kept for the preflight. This reverses the
  earlier decision to use environment variables only, which is what a comment in
  `tools/amp_server.cpp` used to justify.
- `--model-config <name>` with no variant is a *query*, and must not fall through to serving. It
  prints the variants and exits 0; the first version printed the listing and then loaded 13.66 GiB
  behind it.

An unknown key or variant is a hard error, never a silent no-op: a quality dial that quietly does
nothing is the worst failure mode it can have.
