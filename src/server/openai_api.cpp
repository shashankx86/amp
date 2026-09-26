#include "amp/server/openai_api.h"

#include "amp/bytes.h"
#include "amp/format.h"
#include "amp/log.h"
#include "amp/timing.h"
#include "amp/util/json.h"
#include "amp/util/text.h"

#include <atomic>
#include <chrono>
#include <ctime>
#include <mutex>

namespace amp {

using json::Value;

namespace {

std::atomic<uint64_t> g_req_counter{1};

int64_t now_s() {
    return (int64_t) std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// llama.cpp stamps every chunk with system_fingerprint (llama_build_info()). Clients ignore it; it
// is here so responses are shaped identically for anything that diffs them.
const char * fingerprint() { return "amp-" AMP_VERSION; }

std::string make_id(const char * prefix) {
    return format("%s-%llx-%llu", prefix, (unsigned long long) now_s(),
                  (unsigned long long) g_req_counter.fetch_add(1));
}

// Error body, shaped like llama.cpp's format_error_response(). Clients branch on `type`, so these are
// OpenAI's type strings rather than something amp-specific.
const char * error_type_for(int code) {
    switch (code) {
        case 400: return "invalid_request_error";
        case 401: return "authentication_error";
        case 403: return "permission_error";
        case 404: return "not_found_error";
        default:  return "server_error";
    }
}

Value error_body(int code, const std::string & msg) {
    Value err = Value::object();
    err.set("message", msg);
    err.set("type", error_type_for(code));
    err.set("code", (int64_t) code);
    Value body = Value::object();
    body.set("error", std::move(err));
    return body;
}

Status json_error(http::ResponseWriter & w, int code, const std::string & msg) {
    (void) w.send_headers(code, "application/json", false);
    return w.write(error_body(code, msg).dump());
}

bool authorized(const http::Request & req, const ServerConfig & cfg) {
    if (cfg.api_key.empty()) {
        return true;
    }
    const std::string a = req.header("authorization");
    return a == "Bearer " + cfg.api_key || a == cfg.api_key;
}

std::vector<std::string> parse_stop(const Value * stop) {
    std::vector<std::string> out;
    if (!stop) {
        return out;
    }
    if (stop->is_string()) {
        out.push_back(stop->as_string());
    } else if (stop->is_array()) {
        for (const auto & v : stop->as_array()) {
            if (v.is_string()) {
                out.push_back(v.as_string());
            }
        }
    }
    return out;
}

// The envelope every chunk carries: id, created, model, system_fingerprint, object. Same fields in
// the same shape as llama.cpp's chat.completion.chunk.
Value stream_envelope(const std::string & id, int64_t created, const std::string & model,
                      const char * object) {
    Value env = Value::object();
    env.set("id", id);
    env.set("object", object);
    env.set("created", created);
    env.set("model", model);
    env.set("system_fingerprint", fingerprint());
    return env;
}

Value chunk_with(const Value & env, const Value & delta, Value finish_reason, int index) {
    Value choice = Value::object();
    choice.set("index", (int64_t) index);
    choice.set("delta", delta);
    choice.set("finish_reason", std::move(finish_reason));
    Value choices = Value::array();
    choices.push(std::move(choice));
    Value chunk = env;
    chunk.set("choices", std::move(choices));
    return chunk;
}

// stream_options.include_usage, as OpenAI and llama.cpp both spell it.
bool wants_usage(const Value & body) {
    const Value * so = body.get("stream_options");
    if (so && so->is_object()) {
        const Value * iu = so->get("include_usage");
        if (iu && iu->as_bool(false)) {
            return true;
        }
    }
    return false;
}

Value usage_object(const GenerationResult & r, bool include_prompt) {
    Value u = Value::object();
    if (include_prompt) {
        u.set("prompt_tokens", (int64_t) r.n_prompt_tokens);
        // Reported honestly: tokens whose KV was already live when the request arrived. A client that
        // reads this can tell a cache hit from a full re-evaluation.
        Value d = Value::object();
        d.set("cached_tokens", (int64_t) r.n_prompt_cached);
        u.set("prompt_tokens_details", std::move(d));
    }
    u.set("completion_tokens", (int64_t) r.generated.size());
    u.set("total_tokens", (int64_t) (r.n_prompt_tokens + r.generated.size()));
    return u;
}

Value timings_object(const GenerationResult & r) {
    Value t = Value::object();
    t.set("prompt_n", (int64_t) r.n_prompt_computed);
    t.set("prompt_ms", r.prefill_ms);
    t.set("prompt_per_second", r.prefill_tps());
    t.set("prompt_cached", (int64_t) r.n_prompt_cached);
    t.set("predicted_n", (int64_t) r.generated.size());
    t.set("predicted_ms", r.decode_ms);
    t.set("predicted_per_second", r.decode_tps());
    t.set("cache_hit_rate", r.n_prompt_tokens > 0
                                ? (double) r.n_prompt_cached / (double) r.n_prompt_tokens
                                : 0.0);
    t.set("cache_rewound_exactly", r.cache_rewound_exactly);
    return t;
}

// Sampling parameters, shared by both completion endpoints. -1 / negative means "use the server
// default", so a request that omits a field inherits the CLI defaults exactly.
GenerateParams parse_sampling(const Value & body, const ServerConfig & cfg) {
    GenerateParams gp;
    gp.max_tokens = cfg.n_predict;
    if (const Value * v = body.get("max_tokens"); v) {
        gp.max_tokens = (int32_t) v->as_int(cfg.n_predict);
    } else if (const Value * v = body.get("max_completion_tokens"); v) {
        gp.max_tokens = (int32_t) v->as_int(cfg.n_predict);
    } else if (const Value * v = body.get("n_predict"); v) {
        gp.max_tokens = (int32_t) v->as_int(cfg.n_predict);
    }
    if (const Value * v = body.get("temperature"); v) {
        gp.temperature = (float) v->as_double(cfg.temperature);
    }
    if (const Value * v = body.get("top_p"); v) {
        gp.top_p = (float) v->as_double(cfg.top_p);
    }
    if (const Value * v = body.get("top_k"); v) {
        gp.top_k = (int32_t) v->as_int(cfg.top_k);
    }
    if (const Value * v = body.get("min_p"); v) {
        gp.min_p = (float) v->as_double(-1.0f);
    }
    if (const Value * v = body.get("seed"); v) {
        gp.seed = (uint32_t) v->as_int(0);
    }
    if (const Value * v = body.get("ignore_eos"); v) {
        gp.ignore_eos = v->as_bool(false);
    }
    gp.stop = parse_stop(body.get("stop"));
    return gp;
}

// Build a chat message from the request, accepting every reasoning field name clients use.
ChatMessage parse_message(const Value & m) {
    ChatMessage out;
    out.role    = m.get("role") ? m.get("role")->as_string("user") : "user";
    const Value *content = m.get("content");
    if (content) {
        if (content->is_string()) {
            out.content = content->as_string();
        } else if (content->is_array()) {
            // Multimodal-style content arrays: keep the text parts, ignore the rest.
            for (const auto & part : content->as_array()) {
                const Value *txt = part.get("text");
                if (txt && txt->is_string()) {
                    out.content += txt->as_string();
                }
            }
        }
    }
    // Reasoning round-trip aliases. OpenCode sends "reasoning_text"; the canonical field is
    // "reasoning_content"; some clients send "reasoning". Missing this is what makes agentic clients
    // re-evaluate the whole tail of the conversation every turn (see ../NOTES.md 7.2).
    for (const char * key : { "reasoning_content", "reasoning_text", "reasoning" }) {
        if (const Value * r = m.get(key); r && r->is_string() && !r->as_string().empty()) {
            // The template owns the thinking tags, so the stored value is the inner text. Accepting
            // the tagged form as well is what keeps a client's echo from diverging the prompt on
            // every turn - see util/text.h.
            out.reasoning_content = normalize_reasoning(r->as_string());
            out.has_reasoning     = !out.reasoning_content.empty();
            break;
        }
    }
    if (const Value * tc = m.get("tool_call_id"); tc && tc->is_string()) {
        out.tool_call_id = tc->as_string();
    }
    if (const Value * nm = m.get("name"); nm && nm->is_string()) {
        out.tool_name = nm->as_string();
    }
    // OpenAI tool-call responses put the call in `tool_calls`; render their arguments as content so
    // the template sees something sensible.
    if (const Value * tcs = m.get("tool_calls"); tcs && tcs->is_array()) {
        for (const auto & tc : tcs->as_array()) {
            const Value * fn = tc.get("function");
            if (!fn) {
                continue;
            }
            const std::string nm  = fn->get("name") ? fn->get("name")->as_string() : "";
            const std::string arg = fn->get("arguments") ? fn->get("arguments")->as_string() : "";
            out.content += format("\n%s(%s)", nm.c_str(), arg.c_str());
        }
    }
    return out;
}

} // namespace

void OpenAIApi::register_routes(http::Server & srv, InferenceService & svc, const ServerConfig & cfg) {
    srv.route("GET", "/health", [s = &svc](const http::Request & r, http::ResponseWriter & w) {
        return health(r, w, *s);
    });
    srv.route("GET", "/v1/models", [&](const http::Request & r, http::ResponseWriter & w) {
        return models(r, w, svc, cfg);
    });
    srv.route("GET", "/props", [&](const http::Request & r, http::ResponseWriter & w) {
        return props(r, w, svc);
    });
    srv.route("POST", "/tokenize", [&](const http::Request & r, http::ResponseWriter & w) {
        return tokenize(r, w, svc);
    });
    srv.route("POST", "/apply-template", [&](const http::Request & r, http::ResponseWriter & w) {
        return apply_template(r, w, svc);
    });

    // A throw inside a handler must not take down the worker thread or leave the client with a
    // half-written response. llama.cpp wraps its handlers the same way (server-http.cpp) and turns
    // any escaping exception into a 500 with the OpenAI error shape.
    auto guard = [](Status st, http::ResponseWriter & w) {
        if (st.ok()) {
            return st;
        }
        if (w.headers_sent()) {
            return st;   // mid-stream: the stream path already reported it
        }
        return json_error(w, 500, st.message());
    };
    auto comp = [&](const http::Request & r, http::ResponseWriter & w) {
        try {
            return guard(completions(r, w, svc, cfg), w);
        } catch (const std::exception & e) {
            return guard(Status::Errorf("%s", e.what()), w);
        } catch (...) {
            return guard(Status::Error("unknown exception"), w);
        }
    };
    srv.route("POST", "/v1/completions", comp);
    srv.route("POST", "/completion", comp);

    auto chat = [&](const http::Request & r, http::ResponseWriter & w) {
        try {
            return guard(chat_completions(r, w, svc, cfg), w);
        } catch (const std::exception & e) {
            return guard(Status::Errorf("%s", e.what()), w);
        } catch (...) {
            return guard(Status::Error("unknown exception"), w);
        }
    };
    srv.route("POST", "/v1/chat/completions", chat);
    srv.route("POST", "/chat/completions", chat);
}

Status OpenAIApi::health(const http::Request &, http::ResponseWriter & w, InferenceService & svc) {
    Value v = Value::object();
    v.set("status", "ok");
    v.set("slots_idle", (int64_t) 1);
    v.set("model", svc.config().model_id);
    v.set("detail", svc.status_line());
    (void) w.send_headers(200, "application/json", false);
    return w.write(v.dump());
}

Status OpenAIApi::models(const http::Request &, http::ResponseWriter & w,
                                InferenceService & svc, const ServerConfig & cfg) {
    Value m  = Value::object();
    m.set("id", cfg.model_id);
    m.set("object", "model");
    m.set("created", now_s());
    m.set("owned_by", "amp");
    Value perms = Value::object();
    perms.set("context_length", (int64_t) cfg.n_ctx);
    perms.set("max_tokens", (int64_t) cfg.n_predict);
    m.set("amp", std::move(perms));

    Value list = Value::array();
    list.push(std::move(m));
    Value root = Value::object();
    root.set("object", "list");
    root.set("data", std::move(list));
    (void) w.send_headers(200, "application/json", false);
    return w.write(root.dump());
}

Status OpenAIApi::props(const http::Request &, http::ResponseWriter & w, InferenceService & svc) {
    const MemInfo mi = read_meminfo();
    const auto &  st = svc.cache_stats();

    Value plan = Value::object();
    plan.set("expert_layers_gpu", (int64_t) svc.plan().n_expert_layers_gpu);
    plan.set("ubatch", svc.plan().ubatch);
    plan.set("vram_bytes", svc.plan().vram_total);
    plan.set("vram_usable", svc.plan().vram_budget);
    plan.set("kv_bytes", svc.plan().kv_bytes);
    plan.set("compute_bytes", svc.plan().compute_bytes);
    plan.set("resident_bytes", svc.plan().resident_bytes);
    plan.set("stream_bytes", svc.plan().stream_bytes);
    plan.set("predicted_prefill_tps", svc.plan().predicted_prefill_tps);
    plan.set("predicted_decode_tps", svc.plan().predicted_decode_tps);

    Value cache = Value::object();
    cache.set("requests", (int64_t) st.requests);
    cache.set("full_hits", (int64_t) st.full_hits);
    cache.set("partial_hits", (int64_t) st.partial_hits);
    cache.set("tokens_reused", (int64_t) st.tokens_reused);
    cache.set("tokens_computed", (int64_t) st.tokens_computed);
    cache.set("hit_rate", st.hit_rate());
    cache.set("rewinds_exact", (int64_t) st.rewinds_exact);
    cache.set("rewinds_restored", (int64_t) st.rewinds_restored);
    cache.set("rewinds_cleared", (int64_t) st.rewinds_cleared);
    cache.set("checkpoint_bytes", (int64_t) svc.checkpoint_bytes());
    cache.set("checkpoint_save_ms_total", svc.checkpoint_save_ms());
    cache.set("checkpoint_load_ms_total", svc.checkpoint_load_ms());

    Value sys = Value::object();
    sys.set("ram_total", mi.mem_total_bytes);
    sys.set("cached_bytes", mi.cached_bytes);
    sys.set("cache_headroom", mi.cache_headroom());
    sys.set("mem_available", mi.mem_available_bytes);

    Value root = Value::object();
    root.set("model", svc.config().model_id);
    root.set("status", svc.status_line());
    root.set("total_generated_tokens", (int64_t) svc.total_generated());
    root.set("plan", std::move(plan));
    root.set("prefix_cache", std::move(cache));

    const TaskQueue::Stats q = svc.queue_stats();
    Value                   queue = Value::object();
    queue.set("posted", (int64_t) q.posted);
    queue.set("done", (int64_t) q.done);
    queue.set("failed", (int64_t) q.failed);
    queue.set("in_flight", (int64_t) q.in_flight);
    queue.set("depth", (int64_t) q.depth);
    root.set("queue", std::move(queue));
    root.set("system", std::move(sys));
    (void) w.send_headers(200, "application/json", false);
    return w.write(root.dump(2));
}

Status OpenAIApi::tokenize(const http::Request & req, http::ResponseWriter & w,
                                 InferenceService & svc) {
    auto body = Value::parse(req.body);
    if (!body.ok()) {
        return json_error(w, 400, body.message());
    }
    const std::string text = body->get("content") ? body->get("content")->as_string() : req.body;
    const bool        add_special = body->get("add_special") ? body->get("add_special")->as_bool(true) : true;
    auto              toks = svc.tokenize(text, add_special);
    if (!toks.ok()) {
        return json_error(w, 400, toks.message());
    }
    Value arr = Value::array();
    for (const auto t : *toks) {
        arr.push((int64_t) t);
    }
    Value root = Value::object();
    root.set("tokens", std::move(arr));
    (void) w.send_headers(200, "application/json", false);
    return w.write(root.dump());
}

// The rendered prompt, as text and as token count. This is the endpoint to reach for when a turn
// does not hit the prefix cache: it shows exactly what the server will evaluate, so a divergence in
// the client's history is visible instead of inferred from a slow response.
Status OpenAIApi::apply_template(const http::Request & req, http::ResponseWriter & w,
                                 InferenceService & svc) {
    auto body = Value::parse(req.body);
    if (!body.ok()) {
        return json_error(w, 400, body.message());
    }
    const Value *msgs = body->get("messages");
    if (!msgs || !msgs->is_array() || msgs->as_array().empty()) {
        return json_error(w, 400, "missing or empty 'messages'");
    }
    std::vector<ChatMessage> messages;
    for (const auto & m : msgs->as_array()) {
        messages.push_back(parse_message(m));
    }
    std::string tools_json;
    if (const Value * t = body->get("tools"); t && t->is_array() && t->size() > 0) {
        tools_json = t->dump();
    }
    auto rendered = svc.render_chat(messages, tools_json);
    if (!rendered.ok()) {
        return json_error(w, 500, rendered.message());
    }
    auto toks = svc.tokenize(rendered->prompt, /*add_special=*/ true);
    const int64_t n_tokens = toks.ok() ? (int64_t) toks->size() : -1;
    if (toks.ok()) {
        // Round-tripping the rendered text back through the tokenizer is the whole point: the cache
        // compares token ids, so this is what has to be stable for a turn to be reused.
        auto again = svc.tokenize(rendered->prompt, /*add_special=*/ true);
        if (!again.ok() || *again != *toks) {
            return json_error(w, 500, "tokenizer is not deterministic for this prompt");
        }
    }
    Value root = Value::object();
    root.set("prompt", rendered->prompt);
    root.set("n_tokens", n_tokens);
    root.set("supports_thinking", rendered->supported_thinking);
    root.set("thinking_start", rendered->thinking_start);
    root.set("thinking_end", rendered->thinking_end);
    (void) w.send_headers(200, "application/json", false);
    return w.write(root.dump());
}

Status OpenAIApi::completions(const http::Request & req, http::ResponseWriter & w,
                                     InferenceService & svc, const ServerConfig & cfg) {
    if (!authorized(req, cfg)) {
        return json_error(w, 401, "invalid api key");
    }
    auto body = Value::parse(req.body);
    if (!body.ok()) {
        return json_error(w, 400, body.message());
    }
    const Value *prompt = body->get("prompt");
    if (!prompt) {
        return json_error(w, 400, "missing 'prompt'");
    }
    std::string text;
    if (prompt->is_string()) {
        text = prompt->as_string();
    } else if (prompt->is_array()) {
        for (const auto & v : prompt->as_array()) {
            if (v.is_string()) {
                text += v.as_string();
            }
        }
    } else {
        return json_error(w, 400, "'prompt' must be a string or an array of strings");
    }

    GenerateParams gp = parse_sampling(*body, cfg);
    // No chat template was applied, so there is nothing to split: the caller wants what the model
    // actually emitted, <think> blocks included.
    gp.split_thinking = false;

    const bool stream = (body->get("stream") ? body->get("stream")->as_bool(false) : false) ||
                        req.wants_stream();

    auto toks = svc.tokenize(text, /*add_special=*/ true);
    if (!toks.ok()) {
        return json_error(w, 400, toks.message());
    }

    const std::string id      = make_id("cmpl");
    const int64_t     created = now_s();

    if (!stream) {
        auto res = svc.generate(*toks, gp, nullptr);
        if (!res.ok()) {
            return json_error(w, 500, res.message());
        }
        Value choice = Value::object();
        choice.set("index", (int64_t) 0);
        choice.set("text", res->raw);
        choice.set("logprobs", Value());
        choice.set("finish_reason", res->finish_str());
        Value root = Value::object();
        root.set("id", id);
        root.set("object", "text_completion");
        root.set("created", created);
        root.set("model", cfg.model_id);
        Value choices = Value::array();
        choices.push(std::move(choice));
        root.set("choices", std::move(choices));
        root.set("usage", usage_object(*res, true));
        root.set("amp_timings", timings_object(*res));
        (void) w.send_headers(200, "application/json", false);
        return w.write(root.dump());
    }

    (void) w.send_headers(200, "text/event-stream", /*chunked=*/ true);

    // Envelope, built once, matching llama.cpp's chunk shape field for field.
    const Value env = stream_envelope(id, created, cfg.model_id, "text_completion");
    const bool  include_usage = wants_usage(*body);

    // A write that fails means the client is gone. Stop generating rather than burn tokens into a
    // dead socket - llama.cpp cancels the task when the connection drops, and so does this.
    bool client_gone = false;
    auto emit        = [&](const Value & chunk) -> Status {
        if (client_gone) {
            return Status::OK();
        }
        const Status st = w.write_sse(chunk.dump());
        if (!st.ok()) {
            client_gone = true;
        }
        return st;
    };

    auto sse_result = svc.generate(*toks, gp, [&](const StreamChunk & c) {
        if (client_gone) {
            return;
        }
        if (c.first) {
            Value delta = Value::object();
            delta.set("role", "assistant");
            (void) emit(chunk_with(env, delta, Value(), 0));
        } else if (!c.done && !c.text.empty()) {
            Value delta = Value::object();
            delta.set("content", c.text);
            (void) emit(chunk_with(env, delta, Value(), 0));
        }
    });

    // Whatever happened, the stream ends the way the OpenAI spec says it must: a chunk carrying
    // finish_reason, then the [DONE] sentinel. A client that never sees finish_reason reports
    // "stream ended without finish_reason" and retries - which is worse than any error we could
    // have sent it, so on failure we send the error *and* still finish the stream properly.
    if (!sse_result.ok()) {
        if (!client_gone) {
            Value err = error_body(500, sse_result.message());
            (void) w.write_sse(err.dump());
        }
        (void) emit(chunk_with(env, Value::object(), Value("stop"), 0));
    } else {
        (void) emit(chunk_with(env, Value::object(), Value(sse_result->finish_str()), 0));
    }
    if (!client_gone && include_usage && sse_result.ok()) {
        Value usage = usage_object(*sse_result, /*include_prompt=*/ true);
        Value u     = env;
        u.set("choices", Value::array());   // OpenAI: empty choices on the usage-only chunk
        u.set("usage", std::move(usage));
        u.set("amp_timings", timings_object(*sse_result));
        (void) w.write_sse(u.dump());
    }
    if (!client_gone) {
        (void) w.write_sse_done();
    }
    if (client_gone) {
        svc.interrupt();
    }
    return sse_result.ok() ? Status::OK()
                           : Status::Error("stream terminated early: " + sse_result.message());
}

Status OpenAIApi::chat_completions(const http::Request & req, http::ResponseWriter & w,
                                          InferenceService & svc, const ServerConfig & cfg) {
    if (!authorized(req, cfg)) {
        return json_error(w, 401, "invalid api key");
    }
    auto body = Value::parse(req.body);
    if (!body.ok()) {
        return json_error(w, 400, body.message());
    }
    const Value *msgs = body->get("messages");
    if (!msgs || !msgs->is_array() || msgs->as_array().empty()) {
        return json_error(w, 400, "missing or empty 'messages'");
    }

    std::vector<ChatMessage> messages;
    messages.reserve(msgs->as_array().size());
    for (const auto & m : msgs->as_array()) {
        messages.push_back(parse_message(m));
    }

    GenerateParams gp = parse_sampling(*body, cfg);
    gp.split_thinking = true;

    std::string tools_json;
    if (const Value * t = body->get("tools"); t && t->is_array() && t->size() > 0) {
        tools_json = t->dump();
    }

    auto rendered = svc.render_chat(messages, tools_json);
    if (!rendered.ok()) {
        return json_error(w, 500, rendered.message());
    }
    auto toks = svc.tokenize(rendered->prompt, /*add_special=*/ true);
    if (!toks.ok()) {
        return json_error(w, 400, toks.message());
    }

    const bool     stream = (body->get("stream") ? body->get("stream")->as_bool(false) : false) ||
                        req.wants_stream();
    const std::string id      = make_id("chatcmpl");
    const int64_t     created = now_s();

    if (!stream) {
        auto res = svc.generate(*toks, gp, nullptr);
        if (!res.ok()) {
            return json_error(w, 500, res.message());
        }
        Value msg = Value::object();
        msg.set("role", "assistant");
        msg.set("content", res->content);
        if (!res->reasoning.empty()) {
            msg.set("reasoning_content", res->reasoning);
        }
        Value choice = Value::object();
        choice.set("index", (int64_t) 0);
        choice.set("message", std::move(msg));
        choice.set("finish_reason", res->finish_str());
        Value root = Value::object();
        root.set("id", id);
        root.set("object", "chat.completion");
        root.set("created", created);
        root.set("model", cfg.model_id);
        Value choices = Value::array();
        choices.push(std::move(choice));
        root.set("choices", std::move(choices));
        root.set("usage", usage_object(*res, true));
        root.set("amp_timings", timings_object(*res));
        (void) w.send_headers(200, "application/json", false);
        return w.write(root.dump());
    }

    (void) w.send_headers(200, "text/event-stream", /*chunked=*/ true);

    const Value env           = stream_envelope(id, created, cfg.model_id, "chat.completion.chunk");
    const bool  include_usage = wants_usage(*body);
    bool        client_gone   = false;

    // A failed write means the client hung up. Stop generating rather than compute tokens nobody
    // will read - llama.cpp cancels the task when the connection drops.
    auto emit = [&](const Value & chunk) -> Status {
        if (client_gone) {
            return Status::OK();
        }
        const Status st = w.write_sse(chunk.dump());
        if (!st.ok()) {
            client_gone = true;
        }
        return st;
    };

    auto sse_result = svc.generate(*toks, gp, [&](const StreamChunk & c) {
        if (client_gone) {
            return;
        }
        if (c.first) {
            Value delta = Value::object();
            delta.set("role", "assistant");
            (void) emit(chunk_with(env, delta, Value(), 0));
            return;
        }
        if (c.done || (c.text.empty() && c.reasoning.empty())) {
            return;
        }
        Value delta = Value::object();
        if (!c.text.empty()) {
            delta.set("content", c.text);
        }
        if (!c.reasoning.empty()) {
            delta.set("reasoning_content", c.reasoning);
        }
        (void) emit(chunk_with(env, delta, Value(), 0));
    });

    // The stream must always end with a finish_reason and then the [DONE] sentinel. A client that
    // never sees finish_reason reports "stream ended without finish_reason" and retries, so on
    // failure we send the error *and* still finish the stream the way the spec requires.
    if (!sse_result.ok()) {
        if (!client_gone) {
            (void) w.write_sse(error_body(500, sse_result.message()).dump());
        }
        (void) emit(chunk_with(env, Value::object(), Value("stop"), 0));
    } else {
        (void) emit(chunk_with(env, Value::object(), Value(sse_result->finish_str()), 0));
    }
    if (!client_gone && include_usage && sse_result.ok()) {
        Value u = env;
        u.set("choices", Value::array());   // OpenAI: the usage-only chunk carries empty choices
        u.set("usage", usage_object(*sse_result, /*include_prompt=*/ true));
        u.set("amp_timings", timings_object(*sse_result));
        (void) w.write_sse(u.dump());
    }
    if (!client_gone) {
        (void) w.write_sse_done();
    }
    if (client_gone) {
        svc.interrupt();
    }
    return sse_result.ok() ? Status::OK()
                           : Status::Error("stream terminated early: " + sse_result.message());
}
} // namespace amp
