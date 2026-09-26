# PARITY.md — llama.cpp server functional parity contract

This document defines what "100% functional parity with llama.cpp's server" means for
`amp-server`. It is written so that `scripts/parity_test.py` (or any automated smoke
test) can be generated directly from it. Every non-obvious claim cites `file:line` in
the vendored llama.cpp checkout at `third_party/llama.cpp` (pinned commit `1ab7e5a`).

**Scope.** The new `amp-server` binary links llama.cpp's `tools/server` and calls the
exported `int llama_server(common_params & params, int argc, char * argv)`
(`tools/server/server.cpp:43`, defined at `server.cpp:119`). The CLI entry point
`llama_server(int argc, char ** argv)` (`server.cpp:88`) parses argv via
`common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)` (`server.cpp:107`).
Therefore **the entire llama.cpp server flag surface is accepted**, not just the flags
the old amp server used.

**Model under test.** `/home/e0u/localhost/models/Occamy-1.0-APEX-I-MiniPlus-V2.1-Abliterated.gguf`
(arch `qwen35moe`, text-only, no mmproj, no LoRA adapters, no draft model).

---

## 1. Route table

All routes are registered in `tools/server/server.cpp:244-373`. Unless noted, a
route requires the API key (if `--api-key` is set) and is gated by the
`middleware_server_state` 503-while-loading check (`server-http.cpp:307-328`).

"Public" = no API key required even when `--api-key` is set. The public set is exactly
`/health`, `/v1/health`, plus all embedded Web UI asset paths (`server-http.cpp:251-258`).

| Method | Path(s) | Handler | Request body | Response | Public | Notes |
|---|---|---|---|---|---|---|
| GET | `/health`, `/v1/health` | `get_health` (`server-context.cpp:4654`) | — | `{"status":"ok"}` | **yes** | Also served while sleeping/loading. |
| GET | `/metrics` | `get_metrics` (`server-context.cpp:4667`) | — | Prometheus text | no | Requires `--metrics`; else 501 `not_supported_error`. |
| GET | `/props` | `get_props` (`server-context.cpp:4804`) | — | props JSON (§3.10) | no | Served from cache while sleeping. |
| POST | `/props` | `post_props` (`server-context.cpp:4816`) | ignored | `{"success":true}` | no | Requires `--props`; else 501. **No-op** — the handler body is `// update any props here` (`server-context.cpp:4822`). |
| GET | `/models`, `/v1/models` | `get_models` (`server-context.cpp:5072`) | — | models JSON (§3.11) | no | Served from cache while sleeping. |
| POST | `/completion` (legacy), `/completions` (legacy), `/v1/completions` | `post_completions` / `post_completions_oai` (`server-context.cpp:4906-4928`) | completion body (§2.2) | completion JSON (§3.4) | no | `/completion`+`/completions` return llama-native format; `/v1/completions` returns OAI `text_completion`. |
| POST | `/chat/completions` (legacy), `/v1/chat/completions` | `post_chat_completions` (`server-context.cpp:4930`) | chat body (§2.1) | chat JSON (§3.1–3.3) | no | Both paths are identical. |
| POST | `/v1/chat/completions/control` | `post_control` (`server-context.cpp:4950`) | `{"id":"<cmpl_id>","action":"reasoning_end"}` | `{"success":bool,"message"?:str}` | no | Only `action="reasoning_end"` is accepted; anything else → 400 `unknown control action`. Requires the original request to have set `reasoning_control:true`; else `{"success":false,"message":"reasoning control not enabled for this completion"}`. |
| POST | `/v1/responses`, `/responses` | `post_responses_oai` (`server-context.cpp:4987`) | Responses API body, converted to chat (`server-chat.cpp:6`) | Responses JSON (§3.5) | no | `previous_response_id` is rejected (`server-chat.cpp:10-12`). |
| POST | `/v1/audio/transcriptions`, `/audio/transcriptions` | `post_transcriptions_oai` (`server-context.cpp:5009`) | multipart or JSON with `file` | `{"type":"transcript.text.done","text":...,"usage":{...}}` | no | **Requires mtmd + audio**; else 501 `The current model does not support audio input.` |
| POST | `/v1/messages` | `post_anthropic_messages` (`server-context.cpp:5037`) | Anthropic Messages body, converted (`server-chat.cpp:348`) | Anthropic JSON (§3.6) | no | |
| POST | `/infill` | `post_infill` (`server-context.cpp:4828`) | `{"input_prefix":str,"input_suffix":str,"prompt"?:str,"input_extra"?:[{text,filename?}]}` | llama-native completion JSON | no | Requires FIM tokens in vocab; else 501 `Infill is not supported by this model: ...`. |
| POST | `/embedding` (legacy), `/embeddings` (legacy), `/v1/embeddings` | `post_embeddings` / `post_embeddings_oai` (`server-context.cpp:5139-5145`) | `{"input":str|array,"encoding_format"?:"float"|"base64"}` | embeddings JSON (§3.7) | no | Requires `--embeddings`; else 501. `/v1/embeddings` with pooling `none` → 400 `Pooling type 'none' is not OAI compatible`. |
| POST | `/rerank` (legacy), `/reranking` (legacy), `/v1/rerank`, `/v1/reranking` | `post_rerank` (`server-context.cpp:5147`) | `{"query":str,"documents"|"texts":[str],"top_n"?:int,"return_text"?:bool}` | rerank JSON (§3.8) | no | Requires `--reranking`; else 501. `texts` key → TEI format; `documents` → Jina format. |
| POST | `/tokenize` | `post_tokenize` (`server-context.cpp:5084`) | `{"content":str|array,"add_special"?:bool,"parse_special"?:bool,"with_pieces"?:bool}` | `{"tokens":[...]}` or `{"tokens":[{"id":n,"piece":str\|[bytes]}]}` | no | `with_pieces:true` returns per-token pieces. |
| POST | `/detokenize` | `post_detokenize` (`server-context.cpp:5125`) | `{"tokens":[int]}` | `{"content":str}` | no | |
| POST | `/apply-template` | `post_apply_template` (`server-context.cpp:5060`) | chat body (same as `/v1/chat/completions`) | `{"prompt":str}` | no | Renders the template without running inference. |
| POST | `/chat/completions/input_tokens` (legacy), `/v1/chat/completions/input_tokens` | `post_chat_completions_tok` → `handle_count_tokens` (`server-context.cpp:5488`) | chat body | `{"input_tokens":N,"object":"response.input_tokens"}` | no | |
| POST | `/responses/input_tokens` (legacy), `/v1/responses/input_tokens` | `post_responses_tok_oai` → `handle_count_tokens` | Responses body | `{"input_tokens":N,"object":"response.input_tokens"}` | no | |
| POST | `/v1/messages/count_tokens` | `post_anthropic_count_tokens` → `handle_count_tokens` | Anthropic Messages body | `{"input_tokens":N}` | no | No `object` key (Anthropic format). |
| GET | `/lora-adapters` | `get_lora_adapters` (`server-context.cpp:5227`) | — | `[{"id":n,"path":str,"scale":f,"task_name":str,"prompt_prefix":str,"alora_invocation_string"?:str,"alora_invocation_tokens"?:[int]}]` | no | |
| POST | `/lora-adapters` | `post_lora_adapters` (`server-context.cpp:5255`) | `[{"id":n,"scale":f},...]` | `{"success":true}` | no | Body must be an array; else 400 `Request body must be an array`. |
| GET | `/slots` | `get_slots` (`server-context.cpp:4729`) | query: `fail_on_no_slot` | `[{slot...}]` (§3.12) | no | Requires `--slots`; else 501. `?fail_on_no_slot=1` with 0 idle → 503 `no slot available`. |
| POST | `/slots/:id_slot` | `post_slots` (`server-context.cpp:4771`) | query: `action=save\|restore\|erase`; body for save/restore: `{"filename":str}` | action-specific JSON (§3.13) | no | Requires `--slot-save-path`; else 501. Invalid slot id → 400 `Invalid slot ID`. Invalid action → 400 `Invalid action`. |
| GET | `/v1/stream` | `server_stream_make_get_handler` (`server-stream.cpp:454`) | query: `conv_id=str`,`from`=int | `text/event-stream` replay | no | Resumable streaming. 400 if `conv_id` missing or `from` lost; 404 if session gone. |
| POST | `/v1/streams/lookup` | `server_stream_make_lookup_handler` (`server-stream.cpp:503`) | `{"conversation_ids":[str]}` | `[{"conversation_id":str,"is_done":bool,"total_bytes":n,"started_at":n,"completed_at":n}]` | no | Matches exact id and `<id>::<model>` prefixes. |
| DELETE | `/v1/stream` | `server_stream_make_delete_handler` (`server-stream.cpp:561`) | query: `conv_id=str` | 204 empty | no | Idempotent. |
| GET/POST | `/cors-proxy` | `proxy_handler_get/post` or 403 (`server.cpp:338-344`) | — | proxy response or 403 `feature_disabled` | no | Only registered when `--ui-mcp-proxy` is set; otherwise returns 403 `{"error":{"message":"this feature is disabled","type":"feature_disabled"}}`. |
| GET/POST | `/tools` | `tools.handle_get/post` (`server-tools.cpp:2061,2077`) | POST: `{"tool":str,"params":obj,"stream"?:bool}` | tool-specific JSON | no | Only registered when `--tools` or MCP servers are configured; otherwise 403 `feature_disabled`. |
| POST | `/models` | `post_router_models` (`server-models.cpp:2152`) | `{"model":hf_repo_id}` | `{"success":true}` | no | **Router mode only** (`--models-dir`). Downloads a model from HF. |
| POST | `/models/load` | `post_router_models_load` (`server-models.cpp:2022`) | `{"model":name}` | `{"success":true}` | no | Router mode only. |
| POST | `/models/unload` | `post_router_models_unload` (`server-models.cpp:2116`) | `{"model":name}` | `{"success":true}` | no | Router mode only. |
| GET | `/models/sse` | `get_router_models_sse` (`server-models.cpp:2134`) | — | `text/event-stream` | no | Router mode only. |
| DELETE | `/models` | `del_router_models` (`server-models.cpp:2200`) | query: `model=name` | `{"success":true}` | no | Router mode only. |

**GCP/Vertex compat.** When `AIP_MODE=PREDICTION` is set, `register_gcp_compat()`
(`server-http.cpp:797`) adds `POST <AIP_PREDICT_ROUTE>` (default `/predict`) and
`GET <AIP_HEALTH_ROUTE>`. The predict route dispatches on a `@requestFormat` field
(`server-http.cpp:823-922`). This is env-var activated, not a normal flag.

**Unknown routes** return 404 with `{"error":{"message":"File Not Found","type":"not_found_error","code":404}}`
(`server-http.cpp:199-213`).

---

## 2. Request field matrix

### 2.1 `/v1/chat/completions` (and `/chat/completions`, `/v1/responses`, `/v1/messages`, `/apply-template`, `*/input_tokens`)

Parsed by `oaicompat_chat_params_parse` (`server-common.cpp:1151-1427`) then
`eval_llama_cmpl_schema` (`server-schema.cpp:519-571`).

**Chat-specific fields** (consumed by `oaicompat_chat_params_parse`, not passed to the sampler):

| Field | Type | Default | Lands in | Notes |
|---|---|---|---|---|
| `messages` | array | **required** | `inputs.messages` (`server-common.cpp:1291`) | Each must have `role`; non-assistant must have `content`; assistant must have `content` or `tool_calls` (`server-common.cpp:1214-1226`). |
| `tools` | array | `[]` | `inputs.tools` (`server-common.cpp:1292`) | Requires `--jinja`; else runtime error `tools param requires --jinja flag` (`server-common.cpp:1164-1166`). |
| `tool_choice` | string\|object | `"auto"` | `inputs.tool_choice` (`server-common.cpp:1293`) | Requires `--jinja` if != `"auto"` (`server-common.cpp:1167-1169`). |
| `parallel_tool_calls` | bool | `caps["supports_parallel_tool_calls"]` | `inputs.parallel_tool_calls` (`server-common.cpp:1297`) | |
| `add_generation_prompt` | bool | `true` | `inputs.add_generation_prompt` (`server-common.cpp:1298`) | |
| `continue_final_message` | bool | `false` | `inputs.continue_final_message` (`server-common.cpp:1299-1301`) | Mutually exclusive with `add_generation_prompt` (`server-common.cpp:1310-1312`). |
| `reasoning_format` | string | CLI `--reasoning-format` (default `deepseek`) | `inputs.reasoning_format` (`server-common.cpp:1320-1322`) | One of `none`,`auto`,`deepseek`,`deepseek-legacy`; else runtime error (`chat.cpp:889-903`). |
| `chat_template_kwargs` | object | CLI `--chat-template-kwargs` | `inputs.chat_template_kwargs` (`server-common.cpp:1332-1336`) | Merged over CLI defaults. `enable_thinking` must be unquoted `true`/`false`; quoted string → 400 (`server-common.cpp:1339-1346`). |
| `reasoning_effort` | string | — | `inputs.enable_thinking` / `chat_template_kwargs` (`server-common.cpp:1349-1357`) | `"none"` disables thinking; other values stored as `reasoning_effort` kwarg. |
| `reasoning_budget_tokens` / `thinking_budget_tokens` | int | CLI `--reasoning-budget` (default `-1`) | `llama_params["reasoning_budget_tokens"]` (`server-common.cpp:1390-1394`) | Only active when template has thinking end tags (`server-common.cpp:1396`). |
| `reasoning_budget_message` | string | CLI `--reasoning-budget-message` | `llama_params["reasoning_budget_message"]` (`server-common.cpp:1400`) | |
| `reasoning_control` | bool | `false` | `llama_params["reasoning_control"]` (`server-common.cpp:1401`) | Enables runtime reasoning-end via `/v1/chat/completions/control`. |
| `response_format` | object | — | `json_schema` (`server-common.cpp:1186-1199`) | `type:"json_object"` → `response_format.schema`; `type:"json_schema"` → `response_format.json_schema.schema`; `type:"text"` or absent → no constraint; other → 400. |
| `json_schema` | object\|string | — | `inputs.json_schema` (`server-common.cpp:1294`) | Empty object `{}` → `{"type":"object"}` (`server-common.cpp:1202-1204`). Mutually exclusive with `grammar` (`server-common.cpp:1181-1183`). |
| `grammar` | string | `""` | `inputs.grammar` (`server-common.cpp:1295`) | |
| `logprobs` | bool | `false` | `llama_params["n_probs"]` (`server-common.cpp:1407-1411`) | Value = `top_logprobs` (default 20). Rejected with tools+stream (`server-common.cpp:1408-1410`). |
| `top_logprobs` | int\|null | — | — | Requires `logprobs:true`; else 400 (`server-common.cpp:1412-1414`). |
| `stop` | string\|array | `[]` | `llama_params["stop"]` (`server-common.cpp:1173-1177`) | String is wrapped in a 1-element array. |
| `stream` | bool | `false` | `task_params.stream` | |
| `stream_options.include_usage` | bool | `false` | `task_params.include_usage` (`server-schema.cpp:26-29`) | |

**Chat message extensions** (parsed by `common_chat_msgs_parse_oaicompat`, `chat.cpp:373-472`):

| Field | Type | Lands in | Notes |
|---|---|---|---|
| `role` | string | `msg.role` | Required (`chat.cpp:387-389`). |
| `content` | string\|array\|null | `msg.content` / `msg.content_parts` | String or array of `{type:"text"\|"media_marker",text}` (`chat.cpp:394-417`). |
| `reasoning_content` | string | `msg.reasoning_content` | (`chat.cpp:453-455`). |
| `tool_calls` | array | `msg.tool_calls` | Each `{type:"function",function:{name,arguments},id?}` (`chat.cpp:418-447`). |
| `tool_call_id` | string | `msg.tool_call_id` | (`chat.cpp:459-461`). |
| `name` | string | `msg.tool_name` | (`chat.cpp:456-458`). |

**Note:** `reasoning_text` and `reasoning` are **not** accepted as input message fields
by `common_chat_msgs_parse_oaicompat` — only `reasoning_content`. (The response side
emits `reasoning_content`; see §3.1.)

**Passthrough fields.** Any field not listed above is copied verbatim into
`llama_params` (`server-common.cpp:1419-1424`) and then evaluated by
`eval_llama_cmpl_schema`. This is how llama.cpp-specific fields (`mirostat`, `dry_*`,
`xtc_*`, `samplers`, `lora`, `logit_bias`, `preserved_tokens`, `grammar_triggers`,
`chat_format`, `generation_prompt`, `parse_tool_calls`, `chat_parser`, `echo`,
`ignore_eos`, `min_keep`, `backend_sampling`, `post_sampling_probs`, `dry_sequence_breakers`,
`reasoning_budget_start_tag`, `reasoning_budget_end_tags`/`reasoning_budget_end_tag`,
`id_slot`, `message_delimiters`, `response_fields`, `t_max_predict_ms`, `n_indent`,
`n_keep`, `n_discard`, `n_cmpl`/`n`, `n_cache_reuse`, `sse_ping_interval`,
`return_tokens`, `return_progress`, `verbose`, `timings_per_token`) reach the sampler.

### 2.2 `/v1/completions` (and `/completion`, `/completions`)

Parsed by `oaicompat_completion_params_parse` (`server-common.cpp:1036-1072`) then
`eval_llama_cmpl_schema`.

| Field | Type | Default | Lands in | Notes |
|---|---|---|---|---|
| `prompt` | string\|array\|mixed | **required** | `llama_params["prompt"]` | Missing → runtime error `"prompt" is required` (`server-common.cpp:1039-1041`). |
| `stop` | string\|array | `[]` | `llama_params["stop"]` | String wrapped in array (`server-common.cpp:1044-1048`). |
| `echo` | bool | `false` | — | `true` → runtime error `Only no echo is supported` (`server-common.cpp:1051-1053`). |
| `best_of` | — | — | — | **Rejected**: `Unsupported param: best_of` (`server-common.cpp:1056-1061`). |
| `suffix` | — | — | — | **Rejected**: `Unsupported param: suffix` (`server-common.cpp:1056-1061`). |
| `n_predict` | int | CLI `--predict` | `task_params.n_predict` | Overrides `max_tokens` if both present (`server-common.cpp:1065-1068`). |

All other fields pass through to `eval_llama_cmpl_schema` (same as §2.1 passthrough).

### 2.3 Full sampler field list (`eval_llama_cmpl_schema`, `server-schema.cpp:11-517`)

| Field | Type | Default | Struct field | Limits / notes |
|---|---|---|---|---|
| `verbose` | bool | `verbosity > 9` | `task_params.verbose` | |
| `timings_per_token` | bool | `false` | `task_params.timings_per_token` | |
| `stream` | bool | `false` | `task_params.stream` | |
| `stream_options.include_usage` | bool | `false` | `task_params.include_usage` | |
| `cache_prompt` | bool | CLI `--cache-prompt` | `task_params.cache_prompt` | |
| `return_tokens` | bool | `false` | `task_params.return_tokens` | |
| `return_progress` | bool | `false` | `task_params.return_progress` | |
| `sse_ping_interval` | int | CLI `--sse-ping-interval` (30) | `task_params.sse_ping_interval` | Hard limit `[-1, INT32_MAX]`. |
| `n_predict` / `max_completion_tokens` / `max_tokens` | int | CLI `--predict` (-1) | `task_params.n_predict` | Hard limit `[-1, INT32_MAX]`. |
| `n_indent` | int | `0` | `task_params.n_indent` | Hard limit `[0, INT32_MAX]`. |
| `n_keep` | int | CLI `--keep` (0) | `task_params.n_keep` | Hard limit `[-1, INT32_MAX]`. |
| `n_discard` | int | `0` | `task_params.n_discard` | Hard limit `[0, INT32_MAX]`. |
| `n_cmpl` / `n` | int | `1` | `task_params.n_cmpl` | Hard limit `[1, n_parallel]`. |
| `n_cache_reuse` | int | CLI `--cache-reuse` (0) | `task_params.n_cache_reuse` | Hard limit `[0, INT32_MAX]`. |
| `t_max_predict_ms` | int | `-1` | `task_params.t_max_predict_ms` | Hard limit `[-1, INT64_MAX]`. |
| `t_max_prompt_ms` | — | — | — | **Not implemented** (TODO at `server-schema.cpp:71-72`). |
| `response_fields` | array | `[]` | `task_params.response_fields` | Slash paths unnest (`server-schema.cpp:78-82`). |
| `top_k` | int | CLI `--top-k` | `sampling.top_k` | Soft limit `[0, INT32_MAX]`. |
| `top_p` | float | CLI `--top-p` | `sampling.top_p` | Soft limit `[0,1]`. |
| `min_p` | float | CLI `--min-p` | `sampling.min_p` | Soft limit `[0,1]`. |
| `top_n_sigma` | float | CLI | `sampling.top_n_sigma` | |
| `xtc_probability` | float | CLI | `sampling.xtc_probability` | Soft limit `[0,1]`. |
| `xtc_threshold` | float | CLI | `sampling.xtc_threshold` | Soft limit `[0,1]`. |
| `typical_p` | float | CLI `--typical-p` | `sampling.typ_p` | |
| `temperature` | float | CLI `--temp` | `sampling.temp` | Soft limit `[0,inf]`. |
| `dynatemp_range` | float | CLI | `sampling.dynatemp_range` | |
| `dynatemp_exponent` | float | CLI | `sampling.dynatemp_exponent` | |
| `repeat_last_n` | int | CLI `--repeat-last-n` | `sampling.penalty_last_n` | Hard limit `[0, INT32_MAX]`. |
| `repeat_penalty` | float | CLI `--repeat-penalty` | `sampling.penalty_repeat` | |
| `frequency_penalty` | float | CLI | `sampling.penalty_freq` | |
| `presence_penalty` | float | CLI | `sampling.penalty_present` | |
| `dry_multiplier` | float | CLI | `sampling.dry_multiplier` | |
| `dry_base` | float | CLI | `sampling.dry_base` | Values `<1.0` replaced with default (`server-schema.cpp:144-147`). |
| `dry_allowed_length` | int | CLI | `sampling.dry_allowed_length` | Hard limit `[0, INT32_MAX]`. |
| `dry_penalty_last_n` | int | CLI | `sampling.dry_penalty_last_n` | Hard limit `[0, INT32_MAX]`. |
| `mirostat` | int | CLI `--mirostat` | `sampling.mirostat` | Soft limit `[0,2]`. |
| `mirostat_tau` | float | CLI | `sampling.mirostat_tau` | |
| `mirostat_eta` | float | CLI | `sampling.mirostat_eta` | |
| `adaptive_target` | float | CLI | `sampling.adaptive_target` | Soft limit `[-FLT_MAX, 1.0]`. |
| `adaptive_decay` | float | CLI | `sampling.adaptive_decay` | Hard limit `[0, 0.99]`. |
| `seed` | int | CLI `--seed` (-1) | `sampling.seed` | |
| `n_probs` / `logprobs` | int | `0` | `sampling.n_probs` | `logprobs` is an alias used when `n_probs` absent (`server-schema.cpp:179-181`). |
| `min_keep` | int | `0` | `sampling.min_keep` | Hard limit `[0, INT32_MAX]`. |
| `backend_sampling` | bool | `false` | `sampling.backend_sampling` | |
| `post_sampling_probs` | bool | `false` | `task_params.post_sampling_probs` | |
| `lora` | array | `[]` | `task_params.lora` | `[{"id":n,"scale":f}]`; unlisted adapters default to scale 0 (`server-schema.cpp:229-237`). |
| `dry_sequence_breakers` | array | `[]` | `sampling.dry_sequence_breakers` | Non-empty array of strings (`server-schema.cpp:242-249`). |
| `json_schema` / `grammar` | object/string | — | `sampling.grammar` | `json_schema` takes precedence (`server-schema.cpp:252-285`). |
| `grammar_lazy` | bool | `false` | `sampling.grammar_lazy` | |
| `chat_format` | int | — | `chat_parser_params.format` | (`server-schema.cpp:295-300`). |
| `reasoning_format` | string | CLI `--reasoning-format` | `chat_parser_params.reasoning_format` | (`server-schema.cpp:302-308`). |
| `generation_prompt` | string | — | `chat_parser_params.generation_prompt` + `sampling.generation_prompt` | (`server-schema.cpp:310-316`). |
| `parse_tool_calls` | bool | `false` | `chat_parser_params.parse_tool_calls` | |
| `chat_parser` | string | — | `chat_parser_params.parser` | (`server-schema.cpp:321-325`). |
| `continue_final_message` | bool | `false` | `chat_parser_params.is_continuation` | (`server-schema.cpp:327-332`). |
| `echo` | bool | `false` | `chat_parser_params.echo` | |
| `preserved_tokens` | array | `[]` | `sampling.preserved_tokens` | (`server-schema.cpp:341-351`). |
| `grammar_triggers` | array | `[]` | `sampling.grammar_triggers` | (`server-schema.cpp:353-382`). |
| `reasoning_control` | bool | `false` | `sampling.reasoning_control` | |
| `reasoning_budget_tokens` | int | `-1` | `sampling.reasoning_budget_tokens` | Hard limit `[-1, INT32_MAX]`. |
| `reasoning_budget_start_tag` | string | — | `sampling.reasoning_budget_start` | (`server-schema.cpp:391-396`). |
| `reasoning_budget_end_tags` / `reasoning_budget_end_tag` | array/string | — | `sampling.reasoning_budget_end` | (`server-schema.cpp:398-417`). |
| `reasoning_budget_message` | string | — | `sampling.reasoning_budget_forced` | (`server-schema.cpp:419-432`). |
| `logit_bias` | array\|object | `[]` | `sampling.logit_bias` | Array of `[token,bias]` or object; `false` bias = ban (`server-schema.cpp:434-473`). |
| `ignore_eos` | bool | `false` | `sampling.ignore_eos` | (`server-schema.cpp:475-485`). |
| `stop` | string\|array | CLI `--reverse-prompt` | `task_params.antiprompt` | Falls back to CLI default if request provides none (`server-schema.cpp:487-503`). |
| `samplers` | array\|string | CLI `--samplers` | `sampling.samplers` | (`server-schema.cpp:505-514`). |

**Disabled:** All `speculative.*` fields are `#if 0`'d out (`server-schema.cpp:197-227`).
They are accepted by the schema builder but never registered.

---

## 3. Response shape contract

All shapes below are the exact JSON keys emitted by the cited `to_json_*` functions.
`system_fingerprint` is `llama_build_info()` on every response and every stream chunk.

### 3.1 Non-streaming chat response (`to_json_oaicompat_chat`, `server-task.cpp:414-460`)

```json
{
  "choices": [{"finish_reason": "stop|length|tool_calls", "index": 0, "message": {...}}],
  "created": 1234567890,
  "model": "<model_name>",
  "system_fingerprint": "<build_info>",
  "object": "chat.completion",
  "usage": {"completion_tokens": N, "prompt_tokens": N, "total_tokens": N, "prompt_tokens_details": {"cached_tokens": N}},
  "id": "chatcmpl-<random>"
}
```

`message` is `common_chat_msg::to_json_oaicompat` (`chat.cpp:187-265`):
`{"role":"assistant","content":str,"reasoning_content"?:str,"name"?:str,"tool_call_id"?:str,"tool_calls"?:[{"type":"function","function":{"name":str,"arguments":obj},"id"?:str}]}`.
`content` is `""` when empty (`chat.cpp:230-232`); `reasoning_content` omitted when empty (`chat.cpp:233-235`).

`finish_reason` (`server-task.cpp:415-425`):
- `"stop"` — stopped by EOS or stop word, no tool calls.
- `"tool_calls"` — stopped by EOS or stop word, with tool calls.
- `"length"` — hit `n_predict` limit or context limit.

Optional: `logprobs` (if `n_probs>0`), `timings` (if stats set), `__verbose` (if `verbose`).

### 3.2 Streaming chat response (`to_json_oaicompat_chat_stream`, `server-task.cpp:462-526`)

Returns a JSON **array** of chunk objects (sent as multiple SSE `data:` lines,
`server-common.cpp:1620-1637`). Each chunk:

```json
{"choices":[{"finish_reason": null, "index": 0, "delta": {...}}],
 "created": N, "id": "chatcmpl-...", "model": "...", "system_fingerprint": "...", "object": "chat.completion.chunk"}
```

- First chunk (if `n_decoded==1` or progress): `delta` = `{"role":"assistant","content":null}` (`server-task.cpp:1134-1139`).
- Content chunks: `delta` = `{"content":"...", "reasoning_content"?: "...", "tool_calls"?: [...]}` (`server-chat.cpp:621-649`).
- Final chunk: `delta` = `{}`, `finish_reason` set (`server-task.cpp:487-500`).
- If `include_usage`: trailing chunk with `"choices": []` and `usage` (`server-task.cpp:502-514`).
- Stream ends with `data: [DONE]\n\n` (`server-context.cpp:4455`).

### 3.3 `include_usage` trailing chunk

`{"choices":[],"created":N,"id":"chatcmpl-...","model":"...","system_fingerprint":"...","object":"chat.completion.chunk","usage":{...}}`
(`server-task.cpp:505-513`). Empty `choices` array, usage only.

### 3.4 Non-streaming completion (`to_json_oaicompat`, `server-task.cpp:374-412`)

```json
{"choices":[{"text":str,"index":0,"logprobs":null|{...},"finish_reason":"stop|length"}],
 "created":N,"model":"...","system_fingerprint":"...","object":"text_completion",
 "usage":{"completion_tokens":N,"prompt_tokens":N,"total_tokens":N,"prompt_tokens_details":{"cached_tokens":N}},
 "id":"chatcmpl-..."}
```

`finish_reason`: `"stop"` if EOS/stop-word, else `"length"` (`server-task.cpp:382-385`).
Note: completions never emit `"tool_calls"` as a finish_reason.

### 3.5 Streaming completion (`to_json_oaicompat` partial, `server-task.cpp:1073-1109`)

Same shape as §3.4 but `finish_reason` is `null` on every chunk, no `usage`, ends with `[DONE]`.

### 3.6 `/v1/responses` non-streaming (`to_json_oaicompat_resp`, `server-task.cpp:528-597`)

```json
{"completed_at":N,"created_at":N,"id":"resp_...","model":"...","object":"response","output":[...],"status":"completed","usage":{"input_tokens":N,"output_tokens":N,"total_tokens":N,"input_tokens_details":{"cached_tokens":N}}}
```

`output` items: `reasoning` (id `rs_...`), `message` (id `msg_...`), `function_call` (id `fc_...`, `call_id` `call_...`).
Streaming variant emits `response.created`, `response.in_progress`, `response.output_item.added/done`, `response.reasoning_text.delta`, `response.output_text.delta`, `response.content_part.added/done`, `response.function_call_arguments.delta`, `response.completed` (`server-task.cpp:599-714`).

### 3.7 `/v1/messages` (Anthropic) non-streaming (`to_json_anthropic`, `server-task.cpp:731-795`)

```json
{"id":"chatcmpl-...","type":"message","role":"assistant","content":[{"type":"thinking","thinking":str,"signature":""},{"type":"text","text":str},{"type":"tool_use","id":str,"name":str,"input":obj}],"model":"...","stop_reason":"end_turn|tool_use|max_tokens","stop_sequence":str|null,"usage":{"cache_read_input_tokens":N,"input_tokens":N,"output_tokens":N}}
```

`stop_reason` (`server-task.cpp:732-735`): `"end_turn"` (EOS/stop, no tools), `"tool_use"` (EOS/stop, with tools), `"max_tokens"` (limit).
Streaming variant emits `message_start`, `content_block_start/delta/stop`, `message_delta`, `message_stop` (`server-task.cpp:797-983`).

### 3.8 `/v1/embeddings` (`format_embeddings_response_oaicompat`, `server-common.cpp:1429-1473`)

```json
{"model":"...","object":"list","usage":{"prompt_tokens":N,"total_tokens":N},"data":[{"embedding":[...],"index":0,"object":"embedding"}]}
```

With `encoding_format:"base64"`: `{"embedding":"<base64>","index":0,"object":"embedding","encoding_format":"base64"}`.

### 3.9 `/v1/rerank` (`format_response_rerank`, `server-common.cpp:1475-1519`)

Jina format (default):
```json
{"model":"...","object":"list","usage":{"prompt_tokens":N,"total_tokens":N},"results":[{"index":n,"relevance_score":f}]}
```

TEI format (when `texts` key present): bare array `[{"index":n,"score":f,"text"?:str}]` (`server-common.cpp:1506`).

### 3.10 `/props` (`get_res_props`, `server-context.cpp:4599-4644`)

```json
{"default_generation_settings":{"params":{...},"n_ctx":N},"total_slots":N,"model_alias":"...","model_ftype":"...","model_path":"...","modalities":{"vision":false,"video":false,"audio":false},"media_marker":"<__media_...>","endpoint_slots":bool,"endpoint_props":bool,"endpoint_metrics":bool,"ui":...,"ui_settings":{...},"chat_template":"...","chat_template_caps":{...},"bos_token":"...","eos_token":"...","build_info":"...","is_sleeping":false,"cors_proxy_enabled":bool}
```

If `--jinja` and a tool-use template exists, also `chat_template_tool_use` (`server-context.cpp:4637-4641`).

### 3.11 `/v1/models` (`get_res_models`, `server-context.cpp:4566-4597`)

```json
{"models":[{"name":"...","model":"...","modified_at":"","size":"","digest":"","type":"model","description":"","tags":[""],"capabilities":["completion"],"parameters":"","details":{"parent_model":"","format":"gguf","family":"","families":[""],"parameter_size":"","quantization_level":""}}],"object":"list","data":[{"id":"...","aliases":[],"tags":[],"object":"model","created":N,"owned_by":"llamacpp","meta":{"vocab_type":...,"n_vocab":N,"n_ctx":N,"n_ctx_train":N,"n_embd":N,"n_params":N,"size":N,"ftype":"..."}}]}
```

`capabilities` is `["completion","multimodal"]` if mtmd, else `["completion"]` (`server-context.cpp:4580`).

### 3.12 `/slots` (`server_slot::to_json`, `server-context.cpp:686-720`)

```json
{"id":0,"n_ctx":N,"speculative":bool,"is_processing":bool,"id_task":N,"n_prompt_tokens":N,"n_prompt_tokens_processed":N,"n_prompt_tokens_cache":N,"params":{...},"next_token":[{"has_next_token":bool,"has_new_line":bool,"n_remain":N,"n_decoded":N}],"prompt":"...","generated":"..."}
```

`prompt`/`generated` omitted when `only_metrics` (`server-context.cpp:713-716`).

### 3.13 `/slots/:id_slot` action responses

- save: `{"id_slot":n,"filename":"...","n_saved":N,"n_written":N,"timings":{"save_ms":f}}` (`server-task.cpp:1622-1633`)
- restore: `{"id_slot":n,"filename":"...","n_restored":N,"n_read":N,"timings":{"restore_ms":f}}` (`server-task.cpp:1635-1643`)
- erase: `{"id_slot":n,"n_erased":N}` (`server-task.cpp:1649-1654`)

### 3.14 `/metrics` (`server_task_result_metrics::to_metrics`, `server-task.cpp:1523-1617`)

Prometheus text format, content-type `text/plain; version=0.0.4`. Counters:
`llamacpp:prompt_tokens_total`, `prompt_tokens_cached_total`, `prompt_seconds_total`,
`tokens_predicted_total`, `tokens_predicted_seconds_total`, `n_decode_total`,
`n_tokens_max`, `spec_decode_num_draft_tokens_total`,
`spec_decode_num_accepted_tokens_total`, `spec_decode_num_drafts_total`.
Gauges: `prompt_tokens_seconds`, `predicted_tokens_seconds`, `requests_processing`,
`requests_deferred`, `n_busy_slots_per_decode`. Optional per-position counter
`spec_decode_num_accepted_tokens_per_pos_total{position="N"}`.

---

## 4. Error contract

### 4.1 Error body shape

All errors are `{"error":{"code":int,"message":str,"type":str}}`
(`format_error_response`, `server-common.cpp:36-78`).

### 4.2 `error_type` → HTTP status mapping (`server-common.cpp:39-72`)

| `error_type` | HTTP | `type` string | OpenAI-spec? |
|---|---|---|---|
| `ERROR_TYPE_INVALID_REQUEST` | 400 | `invalid_request_error` | yes |
| `ERROR_TYPE_AUTHENTICATION` | 401 | `authentication_error` | no (OpenAI uses `invalid_request_error` / `authentication_error` is Anthropic) |
| `ERROR_TYPE_NOT_FOUND` | 404 | `not_found_error` | no (OpenAI uses `invalid_request_error`) |
| `ERROR_TYPE_SERVER` | 500 | `server_error` | no (OpenAI uses `server_error` — yes) |
| `ERROR_TYPE_PERMISSION` | 403 | `permission_error` | no |
| `ERROR_TYPE_NOT_SUPPORTED` | 501 | `not_supported_error` | no |
| `ERROR_TYPE_UNAVAILABLE` | 503 | `unavailable_error` | no |
| `ERROR_TYPE_EXCEED_CONTEXT_SIZE` | 400 | `exceed_context_size_error` | no |

**Non-OpenAI status codes emitted intentionally:** 401, 403, 404, 501, 503, and the
400-with-`exceed_context_size_error` body. These are llama.cpp's own taxonomy, not
OpenAI's. A smoke test must accept them as correct.

### 4.3 Exception → status mapping (`ex_wrapper`, `server.cpp:54-86`)

- `std::invalid_argument` → 400 `invalid_request_error`.
- Other `std::exception` → 500 `server_error`.
- Unknown → 500 `server_error`.

### 4.4 Middleware errors (bypass `ex_wrapper`)

- 401 `authentication_error` — invalid/missing API key (`server-http.cpp:290-304`).
- 503 `unavailable_error` — server not ready (model loading) (`server-http.cpp:307-328`).
- 404 `not_found_error` — unknown route (`server-http.cpp:199-213`).
- 403 `feature_disabled` — `/tools` or `/cors-proxy` when not enabled (`server.cpp:315-325`).

### 4.5 Context-size errors (`server-context.cpp:3196-3215`)

- Prompt > `n_ctx` (no split): 400 `exceed_context_size_error`, body includes
  `n_prompt_tokens` and `n_ctx` (`server-task.h:481-483`, `server-task.cpp:1501-1508`).
- Prompt >= `n_ctx` (split allowed): 400 `exceed_context_size_error`.
- Prompt > `n_ubatch` (no split): 500 `server_error` (`server-context.cpp:3184-3194`).

---

## 5. CLI flag surface

The new `amp-server` passes argv to `common_params_parse(..., LLAMA_EXAMPLE_SERVER)`
(`server.cpp:107`). The flags below are **all** flags accepted for the server example,
extracted from `common_params_parser_init` (`arg.cpp:1391+`). Flags tagged
`LLAMA_EXAMPLE_COMMON` are inherited by all examples except `LLAMA_EXAMPLE_DOWNLOAD`
(`arg.cpp:1432-1438`).

### 5.1 Model / context

| Flag | Env | Default | Notes |
|---|---|---|---|
| `-m`, `--model` | `LLAMA_ARG_MODEL` | — | Model path. |
| `-mu`, `--model-url` | `LLAMA_ARG_MODEL_URL` | — | Download from URL. |
| `-hf`, `-hfr`, `--hf-repo` | `LLAMA_ARG_HF_REPO` | — | HF repo id. |
| `-hff`, `--hf-file` | `LLAMA_ARG_HF_FILE` | — | HF file. |
| `-hft`, `--hf-token` | `HF_TOKEN` | — | HF token. |
| `-dr`, `--docker-repo` | `LLAMA_ARG_DOCKER_REPO` | — | Docker repo. |
| `-c`, `--ctx-size` | `LLAMA_ARG_CTX_SIZE` | 0 (from model) | 0 = model default. |
| `--kv-unified-per-slot` | `LLAMA_ARG_KV_UNIFIED_PER_SLOT` | unset | Per-slot context limit. |
| `-n`, `--predict`, `--n-predict` | `LLAMA_ARG_N_PREDICT` | -1 | -1 = infinity. |
| `-b`, `--batch-size` | `LLAMA_ARG_BATCH` | 512 | Logical batch. |
| `-ub`, `--ubatch-size` | `LLAMA_ARG_UBATCH` | 512 | Physical batch. |
| `--keep` | — | 0 | Tokens to keep on shift. |
| `--swa-full` | `LLAMA_ARG_SWA_FULL` | false | Full SWA cache. |
| `-ctxcp`, `--ctx-checkpoints`, `--swa-checkpoints` | `LLAMA_ARG_CTX_CHECKPOINTS` | 0 | Context checkpoints per slot. |
| `-cms`, `--checkpoint-min-step` | `LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT` | 0 | Min spacing between checkpoints. |
| `-cram`, `--cache-ram` | `LLAMA_ARG_CACHE_RAM` | -1 | Prompt cache MiB limit; -1 = no limit, 0 = disable. |
| `-kvu`, `--kv-unified` / `-no-kvu`, `--no-kv-unified` | `LLAMA_ARG_KV_UNIFIED` | auto | Unified KV buffer. |
| `--cache-idle-slots` / `--no-cache-idle-slots` | `LLAMA_ARG_CACHE_IDLE_SLOTS` | enabled | Save idle slots to cache. |
| `--context-shift` / `--no-context-shift` | `LLAMA_ARG_CONTEXT_SHIFT` | enabled | Context shift on overflow. |
| `-fa`, `--flash-attn` | `LLAMA_ARG_FLASH_ATTN` | auto | Flash attention. |
| `-dt`, `--defrag-thold` | `LLAMA_ARG_DEFRAG_THOLD` | 0.1 | KV defrag threshold. |
| `--spm-infill` | — | disabled | SPM infill pattern. |
| `--chunk-size` | — | — | Prompt chunk size. |
| `--chunk-separator` | — | — | Chunk separator. |

### 5.2 Batching / parallelism

| Flag | Env | Default | Notes |
|---|---|---|---|
| `-np`, `--parallel` | `LLAMA_ARG_N_PARALLEL` | -1 (auto) | Number of slots. |
| `-cb`, `--cont-batching` / `-nocb`, `--no-cont-batching` | `LLAMA_ARG_CONT_BATCHING` | enabled | Continuous batching. |
| `-pps` | — | false | Shared prompt across sequences. |
| `-tgs` | — | false | Separated generation across sequences. |

### 5.3 GPU offload

| Flag | Env | Default | Notes |
|---|---|---|---|
| `-ngl`, `--gpu-layers`, `--n-gpu-layers` | `LLAMA_ARG_N_GPU_LAYERS` | 999 | Layers on GPU. |
| `-sm`, `--split-mode` | `LLAMA_ARG_SPLIT_MODE` | none | GPU split mode. |
| `-ts`, `--tensor-split` | `LLAMA_ARG_TENSOR_SPLIT` | — | Tensor split ratios. |
| `-mg`, `--main-gpu` | `LLAMA_ARG_MAIN_GPU` | 0 | Main GPU index. |
| `-kvo`, `--kv-offload` / `-nkvo`, `--no-kv-offload` | `LLAMA_ARG_KV_OFFLOAD` | enabled | KV cache offload. |
| `--repack` / `-nr`, `--no-repack` | `LLAMA_ARG_REPACK` | disabled | Weight repacking. |
| `--no-host` | `LLAMA_ARG_NO_HOST` | false | Disable host offload. |
| `--op-offload` / `--no-op-offload` | — | false | Offload host ops. |
| `-dev`, `--device` | `LLAMA_ARG_DEVICE` | — | Device selection. |
| `--list-devices` | — | — | List devices and exit. |
| `-ot`, `--override-tensor` | `LLAMA_ARG_OVERRIDE_TENSOR` | — | Tensor override. |
| `-cmoe`, `--cpu-moe` | `LLAMA_ARG_CPU_MOE` | false | CPU MoE. |
| `-ncmoe`, `--n-cpu-moe` | `LLAMA_ARG_N_CPU_MOE` | 0 | CPU MoE layers. |
| `-ncffn`, `--n-cpu-ffn` | `LLAMA_ARG_N_CPU_FFN` | 0 | CPU FFN layers. |
| `--rpc` | `LLAMA_ARG_RPC` | — | RPC backend. |

### 5.4 Sampling

| Flag | Env | Default |
|---|---|---|
| `-s`, `--seed` | — | -1 |
| `--samplers` | — | default chain |
| `--sampler-seq`, `--sampling-seq` | — | — |
| `--ignore-eos` | — | false |
| `--temp`, `--temperature` | `LLAMA_ARG_TEMPERATURE` | 0.8 |
| `--top-k` | `LLAMA_ARG_TOP_K` | 40 |
| `--top-p` | `LLAMA_ARG_TOP_P` | 0.95 |
| `--min-p` | `LLAMA_ARG_MIN_P` | 0.05 |
| `--top-nsigma`, `--top-n-sigma` | — | -1 |
| `--xtc-probability` | — | 0 |
| `--xtc-threshold` | — | 0.1 |
| `--typical`, `--typical-p` | — | 1.0 |
| `--repeat-last-n` | — | 64 |
| `--repeat-penalty` | `LLAMA_ARG_REPEAT_PENALTY` | 1.0 |
| `--presence-penalty` | `LLAMA_ARG_PRESENCE_PENALTY` | 0.0 |
| `--frequency-penalty` | `LLAMA_ARG_FREQUENCY_PENALTY` | 0.0 |
| `--dry-multiplier` | — | 0.0 |
| `--dry-base` | — | 1.75 |
| `--dry-allowed-length` | — | 2 |
| `--dry-penalty-last-n` | — | -1 |
| `--dry-sequence-breaker` | — | — |
| `--adaptive-target` | — | -1 |
| `--adaptive-decay` | — | 0.99 |
| `--dynatemp-range` | — | 0.0 |
| `--dynatemp-exp` | — | 1.0 |
| `--mirostat` | — | 0 |
| `--mirostat-lr` | — | 0.1 |
| `--mirostat-ent` | — | 5.0 |
| `-l`, `--logit-bias` | — | — |
| `--grammar` | — | — |
| `--grammar-file` | — | — |
| `-j`, `--json-schema` | — | — |
| `-jf`, `--json-schema-file` | — | — |
| `-bs`, `--backend-sampling` | `LLAMA_ARG_BACKEND_SAMPLING` | false |
| `--pooling` | `LLAMA_ARG_POOLING` | none |

### 5.5 Cache / prompt-cache

| Flag | Env | Default |
|---|---|---|
| `--cache-prompt` / `--no-cache-prompt` | `LLAMA_ARG_CACHE_PROMPT` | enabled |
| `--cache-reuse` | `LLAMA_ARG_CACHE_REUSE` | 0 |
| `-ctk`, `--cache-type-k` | `LLAMA_ARG_CACHE_TYPE_K` | f16 |
| `-ctv`, `--cache-type-v` | `LLAMA_ARG_CACHE_TYPE_V` | f16 |
| `-cram`, `--cache-ram` | `LLAMA_ARG_CACHE_RAM` | -1 |
| `-kvu`, `--kv-unified` | `LLAMA_ARG_KV_UNIFIED` | auto |
| `--cache-idle-slots` | `LLAMA_ARG_CACHE_IDLE_SLOTS` | enabled |
| `-ctxcp`, `--ctx-checkpoints` | `LLAMA_ARG_CTX_CHECKPOINTS` | 0 |
| `-cms`, `--checkpoint-min-step` | `LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT` | 0 |
| `--context-shift` | `LLAMA_ARG_CONTEXT_SHIFT` | enabled |

### 5.6 Reasoning

| Flag | Env | Default | Notes |
|---|---|---|---|
| `--jinja` / `--no-jinja` | `LLAMA_ARG_JINJA` | enabled | Jinja chat templates. |
| `--reasoning-format` | `LLAMA_ARG_THINK` | `deepseek` | `none`/`auto`/`deepseek`/`deepseek-legacy`. |
| `-rea`, `--reasoning` | `LLAMA_ARG_REASONING` | auto | `on`/`off`/`auto`. |
| `--reasoning-effort` | `LLAMA_ARG_REASONING_EFFORT` | default | `default`/`minimal`/`low`/`medium`/`high`/`xhigh`/`max`. |
| `--reasoning-budget` | `LLAMA_ARG_THINK_BUDGET` | -1 | -1 unrestricted, 0 immediate, N budget. |
| `--reasoning-budget-message` | `LLAMA_ARG_THINK_BUDGET_MESSAGE` | — | Message before forced end tag. |
| `--reasoning-preserve` / `--no-reasoning-preserve` | `LLAMA_ARG_REASONING_PRESERVE` | auto | Preserve reasoning in history. |
| `--chat-template` | `LLAMA_ARG_CHAT_TEMPLATE` | — | Override chat template. |
| `--chat-template-file` | `LLAMA_ARG_CHAT_TEMPLATE_FILE` | — | Template file. |
| `--chat-template-kwargs` | `LLAMA_ARG_CHAT_TEMPLATE_KWARGS` | — | JSON kwargs. |
| `--skip-chat-parsing` / `--no-skip-chat-parsing` | `LLAMA_ARG_SKIP_CHAT_PARSING` | disabled | Force pure content. |
| `--prefill-assistant` / `--no-prefill-assistant` | `LLAMA_ARG_PREFILL_ASSISTANT` | enabled | Prefill assistant response. |

### 5.7 Speculative decoding

All `--spec-draft-*` and `--spec-*` flags (`arg.cpp:3750+`). Requires a draft model or
ngram config. **Not usable with this model** (no draft model available).

### 5.8 API key / host / port / CORS

| Flag | Env | Default |
|---|---|---|
| `--api-key` | `LLAMA_API_KEY` | — |
| `--api-key-file` | `LLAMA_ARG_API_KEY_FILE` | — |
| `--host` | `LLAMA_ARG_HOST` | 127.0.0.1 |
| `--port` | `LLAMA_ARG_PORT` | 8080 |
| `--path` | `LLAMA_ARG_STATIC_PATH` | — |
| `--reuse-port` | `LLAMA_ARG_REUSE_PORT` | enabled |
| `--cors-origins` | `LLAMA_ARG_CORS_ORIGINS` | — |
| `--cors-methods` | — | — |
| `--cors-headers` | — | — |
| `--cors-credentials` / `--no-cors-credentials` | `LLAMA_ARG_CORS_CREDENTIALS` | — |
| `--api-prefix` | `LLAMA_ARG_API_PREFIX` | — |
| `--ssl-key-file` | `LLAMA_ARG_SSL_KEY_FILE` | — |
| `--ssl-cert-file` | `LLAMA_ARG_SSL_CERT_FILE` | — |
| `-to`, `--timeout` | `LLAMA_ARG_TIMEOUT` | — |
| `--threads-http` | `LLAMA_ARG_THREADS_HTTP` | auto |
| `--sse-ping-interval` | `LLAMA_ARG_SSE_PING_INTERVAL` | 30 |

### 5.9 Logging

| Flag | Env |
|---|---|
| `--log-disable` | — |
| `--log-file` | `LLAMA_ARG_LOG_FILE` |
| `--log-jsonl` / `--no-log-jsonl` | `LLAMA_ARG_LOG_JSONL` |
| `--log-prompts-dir` | — |
| `--log-colors` | `LLAMA_ARG_LOG_COLORS` |
| `-v`, `--verbose`, `--log-verbose` | — |
| `-lv`, `--verbosity`, `--log-verbosity` | `LLAMA_ARG_LOG_VERBOSITY` |
| `--log-prefix` / `--no-log-prefix` | `LLAMA_ARG_LOG_PREFIX` |
| `--log-timestamps` / `--no-log-timestamps` | `LLAMA_ARG_LOG_TIMESTAMPS` |

### 5.10 Other server flags

| Flag | Env | Notes |
|---|---|---|
| `--embedding` / `--embeddings` | `LLAMA_ARG_EMBEDDINGS` | Enable embeddings. |
| `--rerank` / `--reranking` | `LLAMA_ARG_RERANKING` | Enable reranking. |
| `--metrics` | `LLAMA_ARG_ENDPOINT_METRICS` | Enable /metrics. |
| `--props` | `LLAMA_ARG_ENDPOINT_PROPS` | Enable POST /props. |
| `--slots` / `--no-slots` | `LLAMA_ARG_ENDPOINT_SLOTS` | Enable /slots. |
| `--slot-save-path` | — | Slot save/restore path. |
| `--media-path` | — | Media file base path. |
| `--models-dir` | `LLAMA_ARG_MODELS_DIR` | Router mode model dir. |
| `--models-preset` | `LLAMA_ARG_MODELS_PRESET` | Router preset. |
| `--models-max` | `LLAMA_ARG_MODELS_MAX` | Max router models. |
| `--models-autoload` / `--no-models-autoload` | `LLAMA_ARG_MODELS_AUTOLOAD` | Auto-load on startup. |
| `--tools` | `LLAMA_ARG_TOOLS` | Enable server tools. |
| `--tools-runtime` | `LLAMA_ARG_TOOLS_RUNTIME` | Tools runtime isolate. |
| `--mcp-servers-config` | `LLAMA_ARG_MCP_SERVERS_CONFIG` | MCP config file. |
| `--mcp-servers-json` | `LLAMA_ARG_MCP_SERVERS_JSON` | MCP JSON config. |
| `--ui-mcp-proxy` / `--no-ui-mcp-proxy` | `LLAMA_ARG_UI_MCP_PROXY` | CORS proxy for MCP. |
| `--ui` / `--webui` / `--no-ui` / `--no-webui` | `LLAMA_ARG_UI` | Web UI. |
| `--ui-config` / `--webui-config` | `LLAMA_ARG_UI_CONFIG` | UI config JSON. |
| `--ui-config-file` / `--webui-config-file` | `LLAMA_ARG_UI_CONFIG_FILE` | UI config file. |
| `-ag`, `--agent` / `-no-ag`, `--no-agent` | `LLAMA_ARG_AGENT` | Agent mode. |
| `-sps`, `--slot-prompt-similarity` | — | Slot prompt similarity. |
| `--sleep-idle-seconds` | — | Sleep after N idle seconds. |
| `--lora` | — | LoRA adapter path. |
| `--lora-scaled` | — | Scaled LoRA. |
| `--lora-init-without-apply` | — | Load LoRA without applying. |
| `--numa` | `LLAMA_ARG_NUMA` | NUMA optimization. |
| `-lm`, `--load-mode` | `LLAMA_ARG_LOAD_MODE` | Weight load mode. |
| `-lzm`, `--lazy-mode` | `LLAMA_ARG_LAZY_MODE` | Lazy loading. |
| `--offline` | `LLAMA_ARG_OFFLINE` | Offline mode. |
| `--check-tensors` | — | Validate tensor data. |
| `--override-kv` | — | Override model metadata. |
| `--control-vector` | — | Control vector. |
| `--control-vector-scaled` | — | Scaled control vector. |
| `--control-vector-layer-range` | — | Control vector layer range. |
| `-a`, `--alias` | `LLAMA_ARG_ALIAS` | Model alias. |
| `--tags` | `LLAMA_ARG_TAGS` | Model tags. |
| `--embd-normalize` | — | Embedding normalization. |
| `--warmup` / `--no-warmup` | — | Warmup run. |
| `--perf` / `--no-perf` | `LLAMA_ARG_PERF` | Performance timings. |

### 5.11 Load-bearing for THIS model on THIS machine

**Critical (must be set correctly):**
- `-m` — model path.
- `-ngl` — GPU layers. The 12.19 GiB expert set cannot fit in 6 GB VRAM; most layers must stay on CPU.
- `-c` — context size. Native 262144; KV is 8320 B/token → 1.55 GiB at 200k, 0.51 GiB at 65k.
- `-ncmoe` / `-cmoe` — CPU MoE control (amp's `-ncmoe` maps to this).
- `-np` — parallel slots. Default -1 (auto).
- `-b` / `-ub` — batch sizes. Prefill must be split to ubatch.
- `-ctk` / `-ctv` — KV cache dtypes. amp uses `q8_0`/`q4_0`.
- `--jinja` — must be enabled (default) for the chat template.
- `--reasoning-format` — default `deepseek`; controls reasoning extraction.
- `-t` / `-tb` — CPU threads. 8 cores.

**Dangerous here:**
- `-ngl 999` (default) — attempts to offload all layers; will OOM on 6 GB VRAM with this model.
- `-c 262144` (native) — 1.55 GiB KV at 200k; may be too large with 14 GiB RAM.
- `--cache-ram -1` (default) — no prompt cache limit; could consume all RAM.
- `--models-autoload` — router mode only; not applicable.
- `--spec-*` — no draft model available.
- `--lora` — no adapter file available.
- `--mmproj` — no mmproj file; multimodal routes will 501.
- `--mcp-servers-config` — no MCP server available.
- `--sleep-idle-seconds` — sleep mode changes `/props`, `/metrics`, `/v1/models` to serve cached responses.

---

## 6. Reasoning surface

### 6.1 `reasoning_format` values (`common_reasoning_format`, `common.h:419-427`)

| Value | Enum | Behavior |
|---|---|---|
| `none` | `COMMON_REASONING_FORMAT_NONE` | Thoughts left unparsed in `message.content`. |
| `auto` | `COMMON_REASONING_FORMAT_AUTO` | Same as `deepseek`: `message.reasoning_content`. |
| `deepseek` | `COMMON_REASONING_FORMAT_DEEPSEEK` | Extract thinking tag contents into `message.reasoning_content`, including in streaming deltas. |
| `deepseek-legacy` | `COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY` | Extract into `reasoning_content`, but keep `<think>` tags in `content` in stream mode. |

Default: `COMMON_REASONING_FORMAT_DEEPSEEK` (`common.h:649`). The `--reasoning-format`
help text says "(default: auto)" (`arg.cpp:3680`) but the actual default is `deepseek`;
`auto` and `deepseek` behave identically (`common.h:421`).

Set per-request via the `reasoning_format` field (`server-common.cpp:1320-1322`) or
via `--reasoning-format` CLI (`arg.cpp:3674-3684`). Unknown value → runtime error
`Unknown reasoning format: <value>` (`chat.cpp:902`).

### 6.2 Thinking budget mechanism

The reasoning budget is a **sampler** (`common_reasoning_budget_init`,
`reasoning-budget.cpp:261-269`), not a template feature. State machine
(`reasoning-budget.h:10-16`):

```
IDLE -> COUNTING -> WAITING_UTF8 -> FORCING -> DONE
```

- **IDLE**: passthrough, watching for start sequence.
- **COUNTING**: counting down tokens, watching for natural end sequence.
- **WAITING_UTF8**: budget exhausted, allowing UTF-8 completion.
- **FORCING**: forces `forced_tokens` token-by-token (all other logits → -inf).
- **DONE**: passthrough forever.

Armed when (`sampling.cpp:311-323`):
- `reasoning_budget_start` and `reasoning_budget_end` are non-empty, **and**
- `grammar_lazy` is true, **or** `reasoning_budget_tokens >= 0`, **or** `reasoning_control` is true.

The budget is fed by replaying prefill tokens through `llama_sampler_accept`
(`sampling.cpp:319-322`), matching AGENT.md's "arms itself by replaying the prefill
tokens".

Request fields (`server-common.cpp:1388-1403`):
- `reasoning_budget_tokens` / `thinking_budget_tokens` — budget in tokens. `-1` = disabled (default), `0` = immediate end, `N` = budget.
- `reasoning_budget_message` — prepended to the forced end tag.
- `reasoning_control` — if true, the budget sampler is created on demand so reasoning can be ended at runtime via `/v1/chat/completions/control` with `action:"reasoning_end"` (`server-context.cpp:2479-2489`).

CLI: `--reasoning-budget N` (`arg.cpp:3715-3722`), `--reasoning-budget-message`
(`arg.cpp:3723-3729`).

### 6.3 How `enable_thinking` reaches the template

1. CLI `--reasoning on|off|auto` sets `params.enable_reasoning` (1/0/-1) and
   `default_template_kwargs["enable_thinking"]` (`arg.cpp:3685-3702`).
2. Server computes `enable_thinking = (enable_reasoning != 0) && template_supports_thinking`
   (`server-context.cpp:1463-1464`). `template_supports_thinking` requires `--jinja`
   and `common_chat_templates_support_enable_thinking()`.
3. `chat_params.enable_thinking` is set (`server-context.cpp:1485`).
4. Per-request, `inputs.enable_thinking = opt.enable_thinking` (`server-common.cpp:1323`),
   overridable by `chat_template_kwargs.enable_thinking` (`server-common.cpp:1339-1346`)
   or `reasoning_effort:"none"` (`server-common.cpp:1349-1357`).
5. In `common_chat_templates_apply_jinja`, `params.enable_thinking = inputs.enable_thinking`
   (`chat.cpp:1245`), passed to the template as `enable_thinking` (`chat.cpp:920`).
6. The autoparser also receives it: `tmpl_params.enable_thinking = params.enable_thinking`
   and `tmpl_params.extra_context["enable_thinking"] = params.enable_thinking`
   (`chat-auto-parser-helpers.cpp:317,322`).

**Note (from AGENT.md):** the autoparser's reasoning-mode detection
(`autoparser::reasoning_mode`, `chat-auto-parser.h:85-89`) is derived from template
analysis (R1-R3 comparisons in `chat-diff-analyzer.cpp`), not directly from
`enable_thinking`. The `enable_thinking` flag controls whether the template renders
a thinking block; the autoparser independently detects whether the template has
reasoning markers.

### 6.4 `preserve_reasoning` → `preserve_thinking` capability mapping

`jinja::caps_apply_preserve_reasoning` (`caps.cpp:22-27`):

```cpp
ctx.set_val("preserve_thinking",         enabled);
ctx.set_val("clear_thinking",            !enabled);
ctx.set_val("truncate_history_thinking", !enabled);
ctx.set_val("drop_thinking",             !enabled);
```

Applied in `common_chat_template_direct_apply_impl` (`chat.cpp:940-943`) when the
`preserve_reasoning` kwarg is a boolean.

Default: `preserve_reasoning` is set to `"true"` in `default_template_kwargs` if not
explicitly specified (`arg.cpp:958-961`). Overridable via `--reasoning-preserve` /
`--no-reasoning-preserve` (`arg.cpp:3731-3742`) or `chat_template_kwargs.preserve_reasoning`.

The template must have the `supports_preserve_reasoning` cap (`caps.cpp:95,537`);
otherwise the kwarg is ignored with a warning (`server-context.cpp:1499-1509`).

### 6.5 `reasoning_effort`

- CLI `--reasoning-effort` sets `default_template_kwargs["reasoning_effort"]`
  (`arg.cpp:3703-3714`). `"default"` removes it.
- Request-level `reasoning_effort` field (`server-common.cpp:1349-1357`): `"none"`
  disables thinking; other values stored as `reasoning_effort` kwarg.
- Applied via `jinja::caps_apply_reasoning_effort` (`caps.cpp:29-33`), which sets both
  `reasoning_effort` and `reasoning_strength`.
- Template must have `supports_reasoning_effort` cap (`caps.cpp:96,564`).

**Note (from AGENT.md):** this template does **not** support `reasoning_effort` /
`reasoning_strength`. The kwarg is accepted but ignored.

---

## 7. Testability notes

### 7.1 Testable with this GGUF, no network, black-box HTTP

| Feature | Testable | How |
|---|---|---|
| `/health`, `/v1/health` | **yes** | GET, assert `{"status":"ok"}`. |
| `/v1/models`, `/models` | **yes** | GET, assert shape (§3.11). |
| `/props` | **yes** | GET, assert keys (§3.10). |
| `/v1/completions` | **yes** | POST with `prompt`, assert shape (§3.4). |
| `/v1/chat/completions` | **yes** | POST with `messages`, assert shape (§3.1). |
| Streaming (SSE) | **yes** | `stream:true`, assert chunks + `[DONE]`. |
| `include_usage` | **yes** | `stream_options.include_usage:true`, assert trailing chunk. |
| Reasoning split | **yes** | Default `deepseek` format; assert `reasoning_content` present. |
| `enable_thinking:false` | **yes** | `chat_template_kwargs:{"enable_thinking":false}`. |
| Reasoning budget | **yes** | `reasoning_budget_tokens:N`; assert forced end tag. |
| `/v1/chat/completions/control` | **yes** | `reasoning_control:true` + `action:"reasoning_end"`. |
| `/v1/responses` | **yes** | POST Responses body, assert shape (§3.6). |
| `/v1/messages` | **yes** | POST Anthropic body, assert shape (§3.7). |
| `/v1/messages/count_tokens` | **yes** | POST, assert `{"input_tokens":N}`. |
| `/v1/chat/completions/input_tokens` | **yes** | POST, assert `{"input_tokens":N,"object":"response.input_tokens"}`. |
| `/tokenize`, `/detokenize` | **yes** | POST, assert shapes. |
| `/apply-template` | **yes** | POST chat body, assert `{"prompt":str}`. |
| `/infill` | **maybe** | Requires FIM tokens in vocab; test will 501 if absent. |
| `/v1/embeddings` | **maybe** | Requires `--embeddings`; model must support pooling. |
| `/v1/rerank` | **maybe** | Requires `--reranking`; model must have RANK pooling. |
| `/metrics` | **yes** | Requires `--metrics`; assert Prometheus text. |
| `/slots` | **yes** | Requires `--slots`; assert array. |
| `/slots/:id_slot` | **yes** | Requires `--slot-save-path`; save/restore/erase. |
| `/lora-adapters` GET | **yes** | Returns `[]` (no adapters loaded). |
| `/lora-adapters` POST | **yes** | Returns `{"success":true}` (no-op with no adapters). |
| `/v1/stream` resumable | **yes** | Send `X-Conversation-Id` header on a stream, then GET `/v1/stream?conv_id=...`. |
| `/v1/streams/lookup` | **yes** | POST `{"conversation_ids":[...]}`. |
| DELETE `/v1/stream` | **yes** | Returns 204. |
| Error contract | **yes** | Assert status codes and `error.type` strings (§4). |
| API key | **yes** | `--api-key`, assert 401 without, 200 with. |
| CORS | **yes** | OPTIONS preflight, assert headers. |
| `logprobs` / `n_probs` | **yes** | `logprobs:true, top_logprobs:5`. |
| `tools` / `tool_choice` | **yes** | Template supports tool calls; assert `tool_calls` in response. |
| `response_format` | **yes** | `type:"json_object"` with schema. |
| `grammar` / `json_schema` | **yes** | Constrained generation. |
| `stop` | **yes** | Stop strings. |
| `seed` | **yes** | Deterministic output. |
| `n` / `n_cmpl` | **yes** | Multiple completions. |
| `temperature`, `top_k`, `top_p`, `min_p` | **yes** | Sampling params. |
| `mirostat` | **yes** | Mirostat sampling. |
| `dry_*` | **yes** | DRY sampling. |
| `xtc_*` | **yes** | XTC sampling. |
| `logit_bias` | **yes** | Token bias. |
| `samplers` | **yes** | Custom sampler order. |
| `cache_prompt` | **yes** | Assert `prompt_tokens_details.cached_tokens > 0` on 2nd request. |
| `chat_template_kwargs` | **yes** | `enable_thinking`, `preserve_reasoning`. |
| `reasoning_format` | **yes** | `none`/`deepseek`/`deepseek-legacy`. |
| `reasoning_effort` | **no** | Template does not support it; accepted but ignored. |
| `continue_final_message` | **yes** | Assistant continuation. |
| `add_generation_prompt` | **yes** | Toggle generation prompt. |
| `parallel_tool_calls` | **yes** | Tool call rendering. |

### 7.2 NOT testable here (and why)

| Feature | Why not testable |
|---|---|
| Multimodal (`image_url`, `input_audio`, `input_video`) | No mmproj file; model is text-only. Routes return 501 `image input is not supported - hint: if this is unexpected, you may need to provide the mmproj` (`server-common.cpp:1239-1241`). |
| `/v1/audio/transcriptions` | Requires mtmd + audio; returns 501 `The current model does not support audio input.` (`server-context.cpp:5012-5014`). |
| LoRA adapters (actual application) | No adapter file available. `/lora-adapters` GET returns `[]`; POST returns `{"success":true}` but no adapter is loaded. |
| Speculative decoding | No draft model. All `--spec-*` flags are unusable. |
| MCP servers | No MCP server available. `/tools` returns 403 `feature_disabled`. |
| Router mode (`/models`, `/models/load`, etc.) | Requires `--models-dir` and HF repo access. |
| `/cors-proxy` | Requires `--ui-mcp-proxy`; otherwise 403. |
| `--sleep-idle-seconds` | Sleep mode changes `/props`, `/metrics`, `/v1/models` to serve cached responses; testable but changes semantics. |
| `--ui` (Web UI) | Testable (GET `/` returns HTML) but not a functional parity concern. |
| `--tools` (server tools) | Requires `--tools` flag and tool definitions; otherwise 403. |
| GCP/Vertex compat | Requires `AIP_MODE=PREDICTION` env var. |

### 7.3 Features that are accepted but no-ops or ignored

| Feature | Behavior |
|---|---|
| `POST /props` | Returns `{"success":true}` but does nothing (`server-context.cpp:4822` comment: `// update any props here`). |
| `reasoning_effort` | Accepted, stored as kwarg, but template does not support it → ignored. |
| `t_max_prompt_ms` | Not implemented (`server-schema.cpp:71-72` TODO). |
| `echo` (completions) | Rejected: `Only no echo is supported` (`server-common.cpp:1051-1053`). |
| `best_of`, `suffix` (completions) | Rejected: `Unsupported param: ...` (`server-common.cpp:1056-1061`). |
| `speculative.*` request fields | `#if 0`'d out in schema (`server-schema.cpp:197-227`). |
| `previous_response_id` (Responses) | Rejected (`server-chat.cpp:10-12`). |
| `input_file` (Responses) | Rejected (`server-chat.cpp:93`). |

---

## 8. Key differences from the old amp server

1. **Route count:** 7 endpoints → 40+ (including legacy aliases).
2. **Error taxonomy:** OpenAI `type` strings → llama.cpp's own (`not_found_error`,
   `not_supported_error`, `unavailable_error`, `exceed_context_size_error`,
   `permission_error`, `authentication_error`). Status codes 401/403/404/501/503 are
   emitted intentionally.
3. **Reasoning:** `reasoning_content` is now split out by default (`deepseek` format);
   the old amp server required manual splitting.
4. **Streaming:** Resumable streaming via `X-Conversation-Id` + `/v1/stream`.
5. **Token counting:** `/v1/chat/completions/input_tokens`, `/v1/responses/input_tokens`,
   `/v1/messages/count_tokens`.
6. **Slot management:** `/slots`, `/slots/:id_slot` with save/restore/erase.
7. **Metrics:** Prometheus `/metrics`.
8. **Models:** `/v1/models` with full metadata.
9. **Props:** `/props` with template, caps, endpoints.
10. **Control:** `/v1/chat/completions/control` for runtime reasoning-end.
11. **Anthropic:** `/v1/messages` with full Anthropic Messages API.
12. **Responses:** `/v1/responses` with OpenAI Responses API.
13. **Embeddings/Rerank:** `/v1/embeddings`, `/v1/rerank` (with `--embeddings`/`--reranking`).
14. **Infill:** `/infill` (if FIM tokens present).
15. **Tools:** `/tools` (if `--tools` or MCP).
16. **CORS proxy:** `/cors-proxy` (if `--ui-mcp-proxy`).
17. **CLI surface:** ~140 server-relevant flags vs. the old amp server's ~10.

---

## Verification status (added 2026-09-27)

This document is a description of llama.cpp's server. This section records what is actually
*verified*, so the two cannot drift apart silently.

| | count |
|---|---|
| route registrations in `tools/server/server.cpp` (`ctx_http.get/post/del/put`) | 54 |
| distinct routes reachable (with `/v1` and legacy aliases) | 62 in section 7.1's testable list |
| CLI flags exposed by `amp-server --help` | 399 |
| `scripts/parity_test.py` sections | 17 |
| sections passing against a real server | 16 |
| section that reports what cannot be tested here | 1 (8 features, each with a reason) |

`amp-server` is llama.cpp's server with an ~90-line preflight, so route and flag coverage is
inherited rather than reimplemented. The parity claim rests on three things, each verified:

1. **The link is real.** `amp-server` is built from `tools/amp_server.cpp` (90 lines) calling the
   exported `llama_server()` (`tools/server/server.cpp:43`), linking `llama-server-impl`. Build
   commit `d354cd9`; the link itself was proven earlier in `5975bd7`.
2. **The behaviour is llama.cpp's.** `parity_test.py` asserts response key sets, `finish_reason`
   values, the `include_usage` trailing chunk, error `type` strings and the non-OpenAI status codes
   against a running server. It found two real defects in itself (a 120 s default timeout that a cold
   first generation exceeds, and a 32-token tool-call budget that a *thinking* model cannot meet).
3. **The arithmetic is unchanged.** With placement held fixed, `amp-server` and `llama-server`
   produce bit-identical output distributions: KL = 0.000000e+00 in both directions, JS = 0,
   max abs delta logprob = 0, 512/512 top-1 agreement. See `docs/BENCH.md`.

**What this document does not claim.** It describes upstream behaviour, including things upstream
itself does not implement: `POST /props` is a no-op, `t_max_prompt_ms` and `speculative.*` are
accepted and ignored, and `reasoning_effort` has no effect on a template without that capability.
Section 7.3 lists them. A test must not assert them.
