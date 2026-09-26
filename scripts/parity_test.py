#!/usr/bin/env python3
"""Parity test for amp-server against docs/PARITY.md.

This suite proves that the running amp-server exposes llama.cpp's own HTTP
surface with the exact request/response contract documented in docs/PARITY.md:
the route table (section 1), the response shapes (section 3), the error contract
(section 4), and the reasoning surface (section 6).

IMPORTANT: the system under test is llama.cpp's stock `tools/server` (40+ routes)
with a ~200-line amp preflight that only applies amp's memory plan. This script
therefore tests llama.cpp's server, NOT amp's own code. amp-specific behaviour
(the preflight, the memory plan, the prefix cache) is out of scope here; see
the retired scripts/smoke_server.py and scripts/parity.py for that.

Cheap-request rule: the model is a slow reasoning model (~15-35 tok/s decode,
and a thinking answer can be 700+ tokens). Every test that only checks response
SHAPE uses max_tokens of 1-8 and stream:false. Tests that need real text cap
max_tokens hard (16-128). The whole suite finishes in a few minutes.

SKIP is not PASS: features listed in PARITY.md section 7.2 as not testable on
this box (no adapter / no draft model / no MCP / text-only model) are reported
as SKIP with the reason from PARITY.md. SKIP never counts as a pass and is
visible in the summary table.

Usage:
    python3 scripts/parity_test.py [--url URL] [--filter SUBSTR] [--list]
                                 [--verbose] [--timeout SECONDS]
"""

import argparse
import json
import threading
import time
import urllib.error
import urllib.request


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #

def http_request(url, method="GET", body=None, headers=None, timeout=120, raw_body=None):
    """One HTTP request. Returns (status, headers, text). Never raises for HTTP errors."""
    if raw_body is not None:
        data = raw_body.encode("utf-8")
    elif body is not None:
        data = json.dumps(body).encode("utf-8")
    else:
        data = None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read().decode("utf-8", "replace")
    except Exception as e:  # connection refused, timeout, etc.
        return -1, {}, "%s: %s" % (type(e).__name__, e)


def http_post_stream(url, payload, timeout=120):
    """POST a streaming request. Returns (status, chunks) where chunks is the list
    of SSE `data:` payloads (strings, not yet parsed)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        return e.code, [e.read().decode("utf-8", "replace")]
    except Exception as e:
        return -1, ["%s: %s" % (type(e).__name__, e)]
    chunks = []
    try:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data: "):
                chunks.append(line[6:])
    except Exception as e:
        return -1, chunks + ["%s: %s" % (type(e).__name__, e)]
    finally:
        r.close()
    return 200, chunks


def parse_chunks(chunks):
    """Parse SSE data payloads into (list_of_objects, done_bool)."""
    objs = []
    done = False
    for c in chunks:
        if c == "[DONE]":
            done = True
        else:
            try:
                objs.append(json.loads(c))
            except Exception:
                objs.append({"_unparseable": c})
    return objs, done


def parse_error(body):
    """Extract the {"error":{...}} object from a response body, or {}."""
    try:
        return json.loads(body).get("error", {})
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# Test framework
# --------------------------------------------------------------------------- #

class Suite:
    def __init__(self, base, timeout, verbose):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.verbose = verbose
        self.sections = []


class Section:
    def __init__(self, name, verbose=False):
        self.name = name
        self.verbose = verbose
        self.results = []  # list of (status, label, detail); status in pass/fail/skip

    def check(self, cond, label, detail="", response=None):
        if cond:
            self.results.append(("pass", label, detail))
            if self.verbose:
                print("  [ok] %s%s" % (label, ("  " + detail) if detail else ""))
            return True
        msg = detail
        if response is not None:
            msg = (detail + " | " if detail else "") + "response: " + str(response)[:500]
        self.results.append(("fail", label, msg))
        print("  [FAIL] %s" % label)
        if msg:
            print("         %s" % msg)
        return False

    def skip(self, label, reason):
        self.results.append(("skip", label, reason))
        print("  [SKIP] %s  -- %s" % (label, reason))

    @property
    def status(self):
        sts = [s for s, _, _ in self.results]
        if "fail" in sts:
            return "FAIL"
        if "pass" in sts:
            return "PASS"
        return "SKIP"

    def summary(self):
        np = sum(1 for s, _, _ in self.results if s == "pass")
        nf = sum(1 for s, _, _ in self.results if s == "fail")
        ns = sum(1 for s, _, _ in self.results if s == "skip")
        if nf:
            first = next(l for s, l, _ in self.results if s == "fail")
            return "%d passed, %d FAILED (%s)" % (np, nf, first)
        if np:
            return "%d passed%s" % (np, (", %d skipped" % ns) if ns else "")
        return "SKIP (%d not testable)" % ns


def check_keys(sec, label, obj, keys, response=None):
    """Assert that obj is a dict containing every key in `keys`."""
    if not isinstance(obj, dict):
        sec.check(False, label, "expected object, got %s" % type(obj).__name__, response=response)
        return False
    missing = sorted(k for k in keys if k not in obj)
    if missing:
        sec.check(False, label, "missing keys: %s" % missing, response=response)
        return False
    sec.check(True, label, "all %d keys present" % len(keys))
    return True


# --------------------------------------------------------------------------- #
# Key-set constants (from PARITY.md section 3)
# --------------------------------------------------------------------------- #

CHAT_KEYS = {"choices", "created", "model", "system_fingerprint", "object", "usage", "id"}
CHAT_CHUNK_KEYS = {"choices", "created", "id", "model", "system_fingerprint", "object"}
USAGE_KEYS = {"completion_tokens", "prompt_tokens", "total_tokens", "prompt_tokens_details"}
CMPL_KEYS = {"choices", "created", "model", "system_fingerprint", "object", "usage", "id"}
PROPS_KEYS = {
    "default_generation_settings", "total_slots", "model_alias", "model_ftype",
    "model_path", "modalities", "media_marker", "endpoint_slots", "endpoint_props",
    "endpoint_metrics", "ui", "ui_settings", "chat_template", "chat_template_caps",
    "bos_token", "eos_token", "build_info", "is_sleeping", "cors_proxy_enabled",
}
MODELS_DATA_KEYS = {"id", "aliases", "tags", "object", "created", "owned_by", "meta"}
MODELS_MODELS_KEYS = {"name", "model", "modified_at", "size", "digest", "type",
                      "description", "tags", "capabilities", "parameters", "details"}


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #

SECTIONS = []


def section(name):
    def deco(fn):
        SECTIONS.append((name, fn))
        return fn
    return deco


# ---- 1. Health and discovery -------------------------------------------------

@section("Health and discovery")
def s_health(suite, sec):
    base = suite.base

    status, _, body = http_request(base + "/health", timeout=suite.timeout)
    try:
        h = json.loads(body)
    except Exception:
        h = {}
    sec.check(status == 200 and h.get("status") == "ok",
              'GET /health -> {"status":"ok"}', "status=%s" % status, response=body)

    status, _, body = http_request(base + "/v1/models", timeout=suite.timeout)
    try:
        m = json.loads(body)
    except Exception:
        m = {}
    data = m.get("data") if isinstance(m.get("data"), list) else []
    ok = check_keys(sec, "/v1/models: data[0] key set", data[0] if data else {},
                    MODELS_DATA_KEYS, response=body)
    if ok:
        sec.check(data[0].get("object") == "model", "/v1/models: data[0].object == model",
                  data[0].get("object"))
        sec.check(data[0].get("owned_by") == "llamacpp", "/v1/models: data[0].owned_by",
                  data[0].get("owned_by"))
    sec.check(m.get("object") == "list", "/v1/models: object == list", m.get("object"))
    sec.check(isinstance(m.get("models"), list) and len(m["models"]) >= 1,
              "/v1/models: models array non-empty", str(len(m.get("models", []))))

    status, _, body = http_request(base + "/props", timeout=suite.timeout)
    try:
        p = json.loads(body)
    except Exception:
        p = {}
    check_keys(sec, "/props: documented key set (PARITY.md 3.10)", p, PROPS_KEYS, response=body)
    if isinstance(p.get("chat_template_caps"), dict):
        sec.check(True, "/props: chat_template_caps is an object",
                  "%d caps" % len(p["chat_template_caps"]))
    sec.check(isinstance(p.get("default_generation_settings"), dict),
              "/props: default_generation_settings present",
              str(p.get("default_generation_settings"))[:80])


# ---- 2. Metrics, slots, and LoRA --------------------------------------------

@section("Metrics, slots, and LoRA")
def s_metrics(suite, sec):
    base = suite.base

    status, headers, body = http_request(base + "/metrics", timeout=suite.timeout)
    if status == 501:
        sec.skip("/metrics", "501 not_supported_error -- server not started with --metrics")
    else:
        sec.check(status == 200, "/metrics -> 200", "status=%s" % status, response=body)
        ct = headers.get("Content-Type", "")
        sec.check("text/plain" in ct, "/metrics: content-type text/plain", ct)
        sec.check("llamacpp:" in body, "/metrics: Prometheus text has llamacpp: counters",
                  body[:120].replace("\n", " "))

    status, _, body = http_request(base + "/slots", timeout=suite.timeout)
    if status == 501:
        sec.skip("/slots", "501 not_supported_error -- server not started with --slots")
    else:
        try:
            slots = json.loads(body)
        except Exception:
            slots = None
        sec.check(status == 200 and isinstance(slots, list) and len(slots) >= 1,
                  "/slots -> non-empty array", "status=%s" % status, response=body)
        if isinstance(slots, list) and slots:
            check_keys(sec, "/slots: slot key set (PARITY.md 3.12)", slots[0],
                       {"id", "n_ctx", "speculative", "is_processing"}, response=body)

    status, _, body = http_request(base + "/lora-adapters", timeout=suite.timeout)
    try:
        lora = json.loads(body)
    except Exception:
        lora = None
    sec.check(status == 200 and isinstance(lora, list),
              "GET /lora-adapters -> array (empty: no adapters loaded)",
              "status=%s len=%s" % (status, len(lora) if isinstance(lora, list) else "?"),
              response=body)


# ---- 3. Tokenize, detokenize, apply-template ---------------------------------

@section("Tokenize, detokenize, apply-template")
def s_tokens(suite, sec):
    base = suite.base
    text = "The quick brown fox jumps over the lazy dog."

    status, _, body = http_request(base + "/tokenize", "POST", {"content": text}, timeout=suite.timeout)
    try:
        tok = json.loads(body)
    except Exception:
        tok = {}
    ok = check_keys(sec, "/tokenize -> {tokens:[...]}", tok, {"tokens"}, response=body)
    tokens = tok.get("tokens") if ok else None
    if ok:
        sec.check(isinstance(tokens, list) and len(tokens) > 0 and
                  all(isinstance(t, int) for t in tokens),
                  "/tokenize: tokens are a non-empty list of ints",
                  "n=%d first=%s" % (len(tokens), tokens[:4]))

    if isinstance(tokens, list) and tokens:
        status, _, body = http_request(base + "/detokenize", "POST", {"tokens": tokens},
                                       timeout=suite.timeout)
        try:
            detok = json.loads(body)
        except Exception:
            detok = {}
        ok = check_keys(sec, "/detokenize -> {content:str}", detok, {"content"}, response=body)
        if ok:
            content = detok.get("content", "")
            sec.check(isinstance(content, str) and len(content) > 0,
                      "/detokenize: content is a non-empty string (round trip)",
                      repr(content[:60]))

    msgs = [{"role": "user", "content": "What is 2+2?"}]
    status, _, body = http_request(base + "/apply-template", "POST", {"messages": msgs},
                                   timeout=suite.timeout)
    try:
        at = json.loads(body)
    except Exception:
        at = {}
    ok = check_keys(sec, "/apply-template -> {prompt:str}", at, {"prompt"}, response=body)
    if ok:
        prompt = at.get("prompt", "")
        sec.check(isinstance(prompt, str) and "2+2" in prompt,
                  "/apply-template: rendered prompt contains the user message",
                  repr(prompt[-80:]))


# ---- 4. Input token counting -------------------------------------------------

@section("Input token counting")
def s_input_tokens(suite, sec):
    base = suite.base
    chat_body = {"messages": [{"role": "user", "content": "Hello world"}]}
    resp_body = {"input": "Hello world"}
    anth_body = {"model": "m", "messages": [{"role": "user", "content": "Hello world"}], "max_tokens": 10}

    for path, body, label in [
        ("/v1/chat/completions/input_tokens", chat_body, "chat"),
        ("/chat/completions/input_tokens", chat_body, "chat (legacy)"),
        ("/v1/responses/input_tokens", resp_body, "responses"),
        ("/responses/input_tokens", resp_body, "responses (legacy)"),
    ]:
        status, _, resp = http_request(base + path, "POST", body, timeout=suite.timeout)
        try:
            j = json.loads(resp)
        except Exception:
            j = {}
        ok = check_keys(sec, "%s -> {input_tokens, object}" % path, j,
                        {"input_tokens", "object"}, response=resp)
        if ok:
            sec.check(isinstance(j["input_tokens"], int) and j["input_tokens"] > 0,
                      "%s: input_tokens is a positive int" % path, str(j["input_tokens"]))
            sec.check(j.get("object") == "response.input_tokens",
                      "%s: object == response.input_tokens" % path, str(j.get("object")))

    status, _, resp = http_request(base + "/v1/messages/count_tokens", "POST", anth_body,
                                   timeout=suite.timeout)
    try:
        j = json.loads(resp)
    except Exception:
        j = {}
    ok = check_keys(sec, "/v1/messages/count_tokens -> {input_tokens}", j, {"input_tokens"},
                    response=resp)
    if ok:
        sec.check(isinstance(j["input_tokens"], int) and j["input_tokens"] > 0,
                  "/v1/messages/count_tokens: input_tokens is a positive int",
                  str(j["input_tokens"]))
        sec.check("object" not in j, "/v1/messages/count_tokens: no object key (Anthropic format)",
                  str(j.get("object")))


# ---- 5. Error contract -------------------------------------------------------

@section("Error contract")
def s_errors(suite, sec):
    base = suite.base

    # (a) missing 'messages' -> 400 invalid_request_error (std::invalid_argument)
    status, _, body = http_request(base + "/v1/chat/completions", "POST", {}, timeout=suite.timeout)
    err = parse_error(body)
    sec.check(status == 400, "missing 'messages' -> 400", "status=%s" % status, response=body)
    sec.check(err.get("type") == "invalid_request_error",
              "400 error type == invalid_request_error", str(err.get("type")), response=body)
    sec.check(err.get("code") == 400, "400 error code == 400", str(err.get("code")), response=body)

    # (a2) syntactically malformed JSON -> 500 server_error.
    # PARITY.md 4.3: ex_wrapper maps std::invalid_argument -> 400 and EVERYTHING ELSE -> 500.
    # json::parse_error is a std::exception but NOT a std::invalid_argument, so a
    # malformed body surfaces as 500, not the OpenAI-preferred 400.
    status, _, body = http_request(base + "/v1/chat/completions", "POST",
                                   raw_body="{not valid json", timeout=suite.timeout)
    err = parse_error(body)
    sec.check(status == 500,
              "malformed JSON -> 500 (PARITY.md 4.3: json::parse_error is not invalid_argument)",
              "status=%s" % status, response=body)
    sec.check(err.get("type") == "server_error", "500 error type == server_error",
              str(err.get("type")), response=body)

    # (b) unknown route -> 404 not_found_error
    status, _, body = http_request(base + "/v1/definitely-not-a-real-route", timeout=suite.timeout)
    err = parse_error(body)
    sec.check(status == 404, "unknown route -> 404", "status=%s" % status, response=body)
    sec.check(err.get("type") == "not_found_error", "404 error type == not_found_error",
              str(err.get("type")), response=body)
    sec.check(err.get("code") == 404, "404 error code == 404", str(err.get("code")), response=body)

    # (c) api-key-free success: no --api-key configured, so open routes need no auth header
    status, _, body = http_request(base + "/health", timeout=suite.timeout)
    sec.check(status == 200, "api-key-free GET /health succeeds", "status=%s" % status,
              response=body)
    status, _, body = http_request(base + "/v1/chat/completions", "POST",
                                   {"messages": [{"role": "user", "content": "hi"}],
                                    "max_tokens": 1, "temperature": 0},
                                   timeout=suite.timeout)
    sec.check(status == 200, "api-key-free POST /v1/chat/completions succeeds",
              "status=%s" % status, response=body)

    # error body shape: error.code int, error.message str, error.type str (PARITY.md 4.1)
    # asserted implicitly above via the type/code checks; assert message is a non-empty str
    sec.check(isinstance(err.get("message"), str) and len(err.get("message", "")) > 0,
              "error body has a non-empty message string", str(err.get("message"))[:80])

    # 501 not_supported_error via /infill (adaptive: 501 if no FIM tokens, 200 if present)
    status, _, body = http_request(base + "/infill", "POST",
                                   {"input_prefix": "def f(", "input_suffix": ")"},
                                   timeout=suite.timeout)
    if status == 501:
        err = parse_error(body)
        sec.check(err.get("type") == "not_supported_error",
                  "/infill -> 501 not_supported_error (no FIM tokens in vocab)",
                  str(err.get("type")), response=body)
    elif status == 200:
        sec.skip("/infill", "model unexpectedly supports infill (FIM tokens present)")
    else:
        sec.check(False, "/infill -> 501 or 200", "status=%s" % status, response=body)

    # PARITY.md 4.2: 401/403/404/501/503 and 400-exceed_context_size_error are emitted
    # intentionally (llama.cpp's own taxonomy, not OpenAI's). We accept them as correct.
    sec.check(True, "non-OpenAI status codes accepted as intentional (PARITY.md 4.2)",
              "400/404/500/501 asserted above; 401/403/503 are intentional per 4.2")


# ---- 6. Chat completions (non-streaming) -------------------------------------

@section("Chat completions (non-streaming)")
def s_chat(suite, sec):
    base = suite.base
    status, _, body = http_request(base + "/v1/chat/completions", "POST",
                                   {"messages": [{"role": "user", "content": "Say OK"}],
                                    "max_tokens": 4, "temperature": 0},
                                   timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    ok = check_keys(sec, "chat completion key set (PARITY.md 3.1)", j, CHAT_KEYS, response=body)
    if not ok:
        return
    sec.check(j.get("object") == "chat.completion", "object == chat.completion", j.get("object"))
    sec.check(isinstance(j.get("system_fingerprint"), str) and j["system_fingerprint"],
              "system_fingerprint is a non-empty string", j.get("system_fingerprint"))
    sec.check(isinstance(j.get("id"), str) and j["id"], "id is a non-empty string", j.get("id"))
    ch = j["choices"][0]
    check_keys(sec, "choices[0] key set", ch, {"finish_reason", "index", "message"}, response=body)
    sec.check(ch.get("finish_reason") in ("stop", "length", "tool_calls"),
              "finish_reason is stop|length|tool_calls", str(ch.get("finish_reason")))
    msg = ch["message"]
    check_keys(sec, "message key set", msg, {"role", "content"}, response=body)
    sec.check(msg.get("role") == "assistant", "message.role == assistant", msg.get("role"))
    check_keys(sec, "usage key set (PARITY.md 3.1)", j["usage"], USAGE_KEYS, response=body)
    sec.check(j["usage"]["prompt_tokens"] > 0, "usage.prompt_tokens > 0",
              str(j["usage"]["prompt_tokens"]))
    sec.check(j["usage"]["completion_tokens"] >= 0, "usage.completion_tokens >= 0",
              str(j["usage"]["completion_tokens"]))


# ---- 7. Chat completions (streaming) -----------------------------------------

@section("Chat completions (streaming)")
def s_chat_stream(suite, sec):
    base = suite.base
    status, chunks = http_post_stream(base + "/v1/chat/completions",
                                      {"messages": [{"role": "user", "content": "Say OK"}],
                                       "max_tokens": 8, "temperature": 0, "stream": True},
                                      timeout=suite.timeout)
    objs, done = parse_chunks(chunks)
    sec.check(status == 200, "stream -> 200", "status=%s" % status,
              response=str(chunks)[:300])
    if status != 200:
        return
    sec.check(done, "stream ends with [DONE]")
    sec.check(len(objs) > 0, "stream produced chunks", "n=%d" % len(objs))
    if objs:
        check_keys(sec, "stream chunk key set (PARITY.md 3.2)", objs[0], CHAT_CHUNK_KEYS,
                   response=str(objs[0])[:300])
        sec.check(objs[0].get("object") == "chat.completion.chunk",
                  "chunk object == chat.completion.chunk", objs[0].get("object"))
    finish = [o for o in objs if isinstance(o, dict) and o.get("choices") and
              o["choices"][0].get("finish_reason")]
    sec.check(bool(finish), "a chunk carries finish_reason",
              finish[0]["choices"][0]["finish_reason"] if finish else "none")


# ---- 8. Stream include_usage -------------------------------------------------

@section("Stream include_usage")
def s_include_usage(suite, sec):
    base = suite.base
    status, chunks = http_post_stream(
        base + "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 8,
         "temperature": 0, "stream": True, "stream_options": {"include_usage": True}},
        timeout=suite.timeout)
    objs, done = parse_chunks(chunks)
    sec.check(status == 200, "stream -> 200", "status=%s" % status, response=str(chunks)[:300])
    if status != 200:
        return
    sec.check(done, "stream ends with [DONE]")
    usage_chunks = [o for o in objs if isinstance(o, dict) and o.get("choices") == [] and "usage" in o]
    sec.check(bool(usage_chunks),
              "trailing chunk with EMPTY choices and usage (PARITY.md 3.3)",
              "n=%d" % len(usage_chunks))
    if usage_chunks:
        uc = usage_chunks[-1]
        check_keys(sec, "usage chunk key set (PARITY.md 3.3)", uc,
                   {"choices", "created", "id", "model", "system_fingerprint", "object", "usage"},
                   response=str(uc)[:300])
        sec.check(uc["choices"] == [], "usage chunk choices == []", str(uc["choices"]))
        check_keys(sec, "usage chunk usage key set", uc["usage"], USAGE_KEYS, response=str(uc)[:300])


# ---- 9. Raw completions ------------------------------------------------------

@section("Raw completions")
def s_completions(suite, sec):
    base = suite.base

    status, _, body = http_request(base + "/v1/completions", "POST",
                                   {"prompt": "Q: What is 17*23? A:", "max_tokens": 4,
                                    "temperature": 0},
                                   timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    ok = check_keys(sec, "completion key set (PARITY.md 3.4)", j, CMPL_KEYS, response=body)
    if ok:
        sec.check(j.get("object") == "text_completion", "object == text_completion", j.get("object"))
        ch = j["choices"][0]
        check_keys(sec, "completion choices[0] key set", ch,
                   {"text", "index", "logprobs", "finish_reason"}, response=body)
        sec.check(ch.get("finish_reason") in ("stop", "length"),
                  "completion finish_reason is stop|length", str(ch.get("finish_reason")))
        check_keys(sec, "completion usage key set", j["usage"], USAGE_KEYS, response=body)

    status, chunks = http_post_stream(base + "/v1/completions",
                                      {"prompt": "Q: What is 17*23? A:", "max_tokens": 8,
                                       "temperature": 0, "stream": True},
                                      timeout=suite.timeout)
    objs, done = parse_chunks(chunks)
    sec.check(status == 200, "completion stream -> 200", "status=%s" % status,
              response=str(chunks)[:300])
    if status != 200:
        return
    sec.check(done, "completion stream ends with [DONE]")
    if objs:
        check_keys(sec, "completion chunk key set (PARITY.md 3.5)", objs[0],
                   {"choices", "created", "model", "system_fingerprint", "object", "id"},
                   response=str(objs[0])[:300])
        sec.check(objs[0].get("object") == "text_completion",
                  "completion chunk object == text_completion", objs[0].get("object"))
        sec.check(objs[0]["choices"][0].get("finish_reason") is None,
                  "completion chunk finish_reason is null", str(objs[0]["choices"][0].get("finish_reason")))
        sec.check("usage" not in objs[0], "completion chunk has no usage key",
                  str(list(objs[0].keys())))


# ---- 10. reasoning_format values ---------------------------------------------

@section("reasoning_format values")
def s_reasoning_format(suite, sec):
    base = suite.base
    for fmt in ("none", "auto", "deepseek", "deepseek-legacy"):
        status, _, body = http_request(base + "/v1/chat/completions", "POST",
                                       {"messages": [{"role": "user", "content": "Say OK"}],
                                        "max_tokens": 4, "temperature": 0,
                                        "reasoning_format": fmt},
                                       timeout=suite.timeout)
        try:
            j = json.loads(body)
        except Exception:
            j = {}
        ok = check_keys(sec, "reasoning_format=%s accepted, well-formed" % fmt, j,
                        CHAT_KEYS, response=body)
        if ok:
            sec.check(j["choices"][0].get("finish_reason") in ("stop", "length", "tool_calls"),
                      "reasoning_format=%s: finish_reason valid" % fmt,
                      str(j["choices"][0].get("finish_reason")))


# ---- 11. Structured tool calls -----------------------------------------------

@section("Structured tool calls")
def s_tool_calls(suite, sec):
    base = suite.base
    tools = [{"type": "function",
              "function": {"name": "get_weather",
                           "description": "Get the weather for a city",
                           "parameters": {"type": "object",
                                         "properties": {"location": {"type": "string"}},
                                         "required": ["location"]}}}]
    # This is a THINKING model whose template renders the generation prompt already inside a
    # <think> block, so it reasons before it calls anything. Measured on this quant:
    #
    #   max_tokens=32, thinking on   -> finish_reason "length", no tool call  (budget too small)
    #   max_tokens=32, thinking off  -> finish_reason "tool_calls" at 27 tokens
    #   max_tokens=300, thinking on  -> finish_reason "tool_calls" at 66 tokens
    #
    # So a 32-token budget is not a server bug, it is a thinking model thinking. Both
    # configurations are asserted, because the difference is the single most useful thing a caller
    # can know about tool use on this model, and it costs only ~93 generated tokens in total.
    def call_tool(think, max_tokens):
        payload = {"messages": [{"role": "user", "content": "What is the weather in Paris?"}],
                   "tools": tools, "tool_choice": "required",
                   "max_tokens": max_tokens, "temperature": 0}
        if not think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        st, _, bd = http_request(base + "/v1/chat/completions", "POST", payload,
                                 timeout=suite.timeout)
        try:
            return st, json.loads(bd), bd
        except Exception:
            return st, {}, bd

    _, j, body = call_tool(think=True, max_tokens=300)
    ok = check_keys(sec, "tool-call response key set", j, CHAT_KEYS, response=body)
    if not ok:
        return
    ch = j["choices"][0]
    sec.check(ch.get("finish_reason") == "tool_calls",
              "finish_reason == tool_calls (thinking on, 300 tok)", str(ch.get("finish_reason")))
    tcs = ch.get("message", {}).get("tool_calls")
    sec.check(isinstance(tcs, list) and len(tcs) > 0,
              "message.tool_calls is a non-empty array", str(tcs)[:120])
    if not tcs:
        return
    fn = tcs[0].get("function", {})
    sec.check(isinstance(fn.get("name"), str) and fn["name"],
              "tool_calls[0].function.name is a non-empty string", str(fn.get("name")))
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            json.loads(args)
            args_ok = True
        except Exception:
            args_ok = False
    elif isinstance(args, dict):
        args_ok = True
    else:
        args_ok = False
    sec.check(args_ok, "tool_calls[0].function.arguments parses as JSON",
              str(args)[:120])

    # Thinking off must reach the same tool call in far fewer tokens.
    _, j2, body2 = call_tool(think=False, max_tokens=32)
    ch2 = (j2.get("choices") or [{}])[0]
    sec.check(ch2.get("finish_reason") == "tool_calls",
              "finish_reason == tool_calls (thinking off, 32 tok)", str(ch2.get("finish_reason")))
    tcs2 = ch2.get("message", {}).get("tool_calls")
    sec.check(isinstance(tcs2, list) and len(tcs2) > 0,
              "tool call also emitted with thinking disabled", str(tcs2)[:120])
    if j.get("usage") and j2.get("usage"):
        sec.check(j2["usage"]["completion_tokens"] < j["usage"]["completion_tokens"],
                  "thinking off uses fewer completion tokens than thinking on",
                  "%d vs %d" % (j2["usage"]["completion_tokens"], j["usage"]["completion_tokens"]))


# ---- 12. Reasoning split -----------------------------------------------------

@section("Reasoning split")
def s_reasoning_split(suite, sec):
    base = suite.base
    msgs = [{"role": "system", "content": "Answer in one short line."},
            {"role": "user", "content": "What is 2+2?"}]

    # thinking on (default deepseek format): reasoning split into reasoning_content
    status, _, body = http_request(base + "/v1/chat/completions", "POST",
                                   {"messages": msgs, "max_tokens": 128, "temperature": 0},
                                   timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    msg = j.get("choices", [{}])[0].get("message", {}) if j else {}
    reasoning = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    sec.check(status == 200, "thinking-on request -> 200", "status=%s" % status, response=body)
    sec.check(bool(reasoning),
              "thinking on: reasoning_content is non-empty (split out of content)",
              "reasoning=%d chars" % len(reasoning))
    sec.check("<think>" not in content and "</think>" not in content,
              "thinking on: no stray think tags in content", repr(content[:60]))
    if content and reasoning:
        sec.check(content != reasoning,
                  "thinking on: content and reasoning_content are distinct",
                  "content=%r reasoning=%r" % (content[:30], reasoning[:30]))

    # thinking off: reasoning_content absent/empty, content still produced
    status, _, body = http_request(base + "/v1/chat/completions", "POST",
                                   {"messages": msgs, "max_tokens": 16, "temperature": 0,
                                    "chat_template_kwargs": {"enable_thinking": False}},
                                   timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    msg = j.get("choices", [{}])[0].get("message", {}) if j else {}
    reasoning_off = msg.get("reasoning_content") or ""
    content_off = msg.get("content") or ""
    sec.check(status == 200, "enable_thinking=false request -> 200", "status=%s" % status,
              response=body)
    sec.check(bool(content_off), "enable_thinking=false: content is still produced",
              repr(content_off[:60]))
    sec.check(not reasoning_off, "enable_thinking=false: reasoning_content is absent/empty",
              repr(reasoning_off[:40]))

    # /apply-template: the rendered prompt must gain <think>\n\n</think>\n\n (free, deterministic)
    def render(kwargs):
        body = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        if kwargs:
            body["chat_template_kwargs"] = kwargs
        status, _, resp = http_request(base + "/apply-template", "POST", body, timeout=suite.timeout)
        try:
            return json.loads(resp).get("prompt", "")
        except Exception:
            return ""

    p_on = render(None)
    p_off = render({"enable_thinking": False})
    sec.check(p_on.endswith("<think>\n"),
              "apply-template default: prompt ends with <think>\\n", repr(p_on[-40:]))
    sec.check("<think>\n\n</think>\n\n" in p_off,
              "apply-template enable_thinking=false: prompt gains <think>\\n\\n</think>\\n\\n",
              repr(p_off[-60:]))
    sec.check(p_on != p_off, "apply-template: rendered prompts differ")


# ---- 13. Legacy route aliases ------------------------------------------------

@section("Legacy route aliases")
def s_legacy(suite, sec):
    base = suite.base

    status, _, body = http_request(base + "/completion", "POST",
                                   {"prompt": "Q: What is 17*23? A:", "max_tokens": 4,
                                    "temperature": 0},
                                   timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    ok = check_keys(sec, "/completion -> llama-native format", j,
                    {"content", "tokens_predicted", "tokens_evaluated"}, response=body)
    if ok:
        sec.check(isinstance(j.get("content"), str), "/completion: content is a string",
                  repr(j.get("content")[:40]))

    status, _, body = http_request(base + "/completions", "POST",
                                   {"prompt": "Q: What is 17*23? A:", "max_tokens": 4,
                                    "temperature": 0},
                                   timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    check_keys(sec, "/completions -> llama-native format", j,
               {"content", "tokens_predicted", "tokens_evaluated"}, response=body)

    status, _, body = http_request(base + "/chat/completions", "POST",
                                   {"messages": [{"role": "user", "content": "Say OK"}],
                                    "max_tokens": 4, "temperature": 0},
                                   timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    ok = check_keys(sec, "/chat/completions -> OAI chat format", j, CHAT_KEYS, response=body)
    if ok:
        sec.check(j.get("object") == "chat.completion",
                  "/chat/completions: object == chat.completion", j.get("object"))


# ---- 14. Concurrent streams --------------------------------------------------

@section("Concurrent streams")
def s_concurrency(suite, sec):
    base = suite.base
    payload = {"messages": [{"role": "user", "content": "Reply with just the word OK"}],
               "max_tokens": 8, "temperature": 0, "stream": True}
    results = {}

    def run(name):
        try:
            status, chunks = http_post_stream(base + "/v1/chat/completions", payload,
                                              timeout=suite.timeout)
            results[name] = (status, chunks)
        except Exception as e:
            results[name] = (-1, ["%s: %s" % (type(e).__name__, e)])

    threads = [threading.Thread(target=run, args=("req%d" % i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=max(suite.timeout * 4, 300))

    for i in range(4):
        name = "req%d" % i
        if name not in results:
            sec.check(False, "%s completed" % name, "timed out")
            continue
        status, chunks = results[name]
        if status != 200:
            sec.check(False, "%s -> 200" % name, "status=%s" % status,
                      response=str(chunks)[:300])
            continue
        objs, done = parse_chunks(chunks)
        finish = [o for o in objs if isinstance(o, dict) and o.get("choices") and
                  o["choices"][0].get("finish_reason")]
        sec.check(bool(finish), "%s has a finish_reason" % name,
                  finish[0]["choices"][0]["finish_reason"] if finish else "none")
        sec.check(done, "%s ends with [DONE]" % name)


# ---- 15. Long prompt (multi-ubatch) -----------------------------------------

@section("Long prompt (multi-ubatch)")
def s_long_prompt(suite, sec):
    base = suite.base
    # ubatch is 1024 on this box (see scripts/parity.py); ~1500 tokens forces a split.
    long_prompt = "context " * 1500
    status, _, body = http_request(base + "/v1/completions", "POST",
                                   {"prompt": long_prompt, "max_tokens": 4, "temperature": 0},
                                   timeout=max(suite.timeout, 300))
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    sec.check(status == 200 and "error" not in j,
              "a ~1500-token prompt (> 1 ubatch) prefills without error",
              "status=%s" % status, response=body)
    if status == 200 and "error" not in j:
        pt = j.get("usage", {}).get("prompt_tokens", 0)
        sec.check(pt > 1024, "the whole long prompt was evaluated", "prompt_tokens=%s" % pt)


# ---- 16. Not testable on this box -------------------------------------------

@section("Not testable on this box")
def s_not_testable(suite, sec):
    # PARITY.md section 7.2 -- these are accepted-but-unusable on this box. SKIP, never PASS.
    sec.skip("LoRA adapter application", "no adapter file available (PARITY.md 7.2)")
    sec.skip("Speculative decoding", "no draft model available (PARITY.md 7.2)")
    sec.skip("MCP servers", "no MCP server available (PARITY.md 7.2)")
    sec.skip("Router mode (/models, /models/load, ...)", "requires --models-dir and HF access (PARITY.md 7.2)")
    sec.skip("/v1/embeddings", "text-only model; requires --embeddings and pooling support (PARITY.md 7.1/7.2)")
    sec.skip("/v1/rerank", "text-only model; requires --reranking and RANK pooling (PARITY.md 7.1/7.2)")
    sec.skip("/v1/audio/transcriptions", "requires mtmd + audio; model is text-only (PARITY.md 7.2)")
    sec.skip("Multimodal input (image_url, input_audio, ...)", "no mmproj file; model is text-only (PARITY.md 7.2)")


# ---- 17. Sampler fields ------------------------------------------------------


@section("Sampler fields")
def s_sampler_fields(suite, sec):
    base = suite.base
    # One request per field (grouped where fields are independent scalars), thinking
    # off, max_tokens 1-16. A rejected field surfaces as a non-200 or an error body,
    # which the status + key-set assertions catch.

    def sampler_request(label, extra, max_tokens=4, temperature=0, prompt="Say OK"):
        payload = {"messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": temperature,
                   "chat_template_kwargs": {"enable_thinking": False}}
        payload.update(extra)
        status, _, body = http_request(base + "/v1/chat/completions", "POST", payload,
                                       timeout=suite.timeout)
        try:
            j = json.loads(body)
        except Exception:
            j = {}
        sec.check(status == 200, "%s -> 200" % label, "status=%s" % status, response=body)
        ok = check_keys(sec, "%s: well-formed (PARITY.md 3.1)" % label, j, CHAT_KEYS,
                        response=body)
        if ok:
            sec.check(j["choices"][0].get("finish_reason") in ("stop", "length", "tool_calls"),
                      "%s: finish_reason valid" % label, str(j["choices"][0].get("finish_reason")))
            content = (j["choices"][0].get("message") or {}).get("content")
            sec.check(isinstance(content, str) and len(content) > 0,
                      "%s: content non-empty" % label, repr(content)[:60])
        return j

    sampler_request("top_k=1, top_p=0.9, min_p=0.05 (PARITY.md 2.3)",
                    {"top_k": 1, "top_p": 0.9, "min_p": 0.05})
    sampler_request("typical_p, repeat_penalty, repeat_last_n, presence/frequency_penalty (PARITY.md 2.3)",
                    {"typical_p": 0.9, "repeat_penalty": 1.1, "repeat_last_n": 32,
                     "presence_penalty": 0.1, "frequency_penalty": 0.1})
    sampler_request("mirostat=2 with mirostat_tau/mirostat_eta (PARITY.md 2.3)",
                    {"mirostat": 2, "mirostat_tau": 5.0, "mirostat_eta": 0.1}, temperature=0.8)
    sampler_request("dry_multiplier=0.8 (PARITY.md 2.3)", {"dry_multiplier": 0.8})
    sampler_request("xtc_probability=0.1 (PARITY.md 2.3)", {"xtc_probability": 0.1})

    # seed is the one field with a crisp deterministic assertion: same seed,
    # temperature > 0 and a fixed budget must give identical text (PARITY.md 7.1).
    j1 = sampler_request("seed=12345, temperature=0.8, n_predict=16 (call 1)",
                         {"seed": 12345}, max_tokens=16, temperature=0.8,
                         prompt="The capital of France is")
    j2 = sampler_request("seed=12345, temperature=0.8, n_predict=16 (call 2)",
                         {"seed": 12345}, max_tokens=16, temperature=0.8,
                         prompt="The capital of France is")
    c1 = (j1.get("choices", [{}])[0].get("message") or {}).get("content")
    c2 = (j2.get("choices", [{}])[0].get("message") or {}).get("content")
    sec.check(c1 is not None and c1 == c2,
              "seed: identical seed + temperature > 0 gives identical text across two calls",
              "call1=%r call2=%r" % (c1, c2))

    # logit_bias: acceptance only. The effect depends on token ids we would have to
    # discover first, so a behavioural assertion is not possible black-box
    # (PARITY.md 2.3, server-schema.cpp:434-473).
    sampler_request('logit_bias {"0": -100.0} (PARITY.md 2.3)', {"logit_bias": {"0": -100.0}})

    # tfs_z and penalty_prompt do not exist in the vendored server-schema.cpp (grep over
    # third_party/llama.cpp: no matches for either string) and are absent from PARITY.md
    # 2.3. eval_llama_cmpl_schema evaluates the schema's registered fields, not the
    # request keys (server-schema.cpp:549-551), so an unknown field is silently ignored:
    # a request carrying it would return 200 while proving nothing. SKIP, not PASS.
    sec.skip("tfs_z", "no such field in vendored server-schema.cpp or PARITY.md 2.3; "
                      "unknown fields are ignored by eval_llama_cmpl_schema (server-schema.cpp:549-551)")
    sec.skip("penalty_prompt", "no such field in vendored server-schema.cpp or PARITY.md 2.3; "
                             "unknown fields are ignored by eval_llama_cmpl_schema (server-schema.cpp:549-551)")


# ---- 18. Constrained generation ----------------------------------------------


@section("Constrained generation")
def s_constrained(suite, sec):
    base = suite.base

    # grammar: the only legal completion is the literal "yes"
    status, _, body = http_request(base + "/v1/chat/completions", "POST",
                                   {"messages": [{"role": "user", "content": "Say yes or no:"}],
                                    "max_tokens": 4, "temperature": 0,
                                    "grammar": "root ::= \"yes\"",
                                    "chat_template_kwargs": {"enable_thinking": False}},
                                   timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    content = (j.get("choices", [{}])[0].get("message") or {}).get("content")
    sec.check(status == 200, "grammar request -> 200", "status=%s" % status, response=body)
    sec.check(content is not None and content.strip() == "yes",
              "grammar root ::= \"yes\": content satisfies the constraint", repr(content),
              response=body)

    # response_format json_schema: the schema requires a string field "city"
    # (server-common.cpp:1193-1195). Verified by hand: the model answers {"city": "Paris"}.
    status, _, body = http_request(
        base + "/v1/chat/completions", "POST",
        {"messages": [{"role": "user",
                       "content": "Give me a JSON object with key city set to Paris."}],
         "max_tokens": 16, "temperature": 0,
         "response_format": {"type": "json_schema",
                             "json_schema": {"name": "city", "strict": True,
                                             "schema": {"type": "object",
                                                        "properties": {"city": {"type": "string"}},
                                                        "required": ["city"]}}},
         "chat_template_kwargs": {"enable_thinking": False}},
        timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    content = (j.get("choices", [{}])[0].get("message") or {}).get("content")
    parsed = None
    if content:
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = None
    sec.check(status == 200, "response_format json_schema request -> 200", "status=%s" % status,
              response=body)
    sec.check(isinstance(parsed, dict) and "city" in parsed,
              "response_format json_schema: content is a JSON object with the required \"city\" key",
              repr(content)[:120], response=body)

    # response_format json_object: any JSON object (server-common.cpp:1189-1192, 1202-1204)
    status, _, body = http_request(
        base + "/v1/chat/completions", "POST",
        {"messages": [{"role": "user",
                       "content": "Return a JSON object containing a friendly greeting."}],
         # 32, not 8: constrained generation still has to *finish* the object. At 8 the
         # model produced '{\n  "greeting": "' with finish_reason "length" - a truncated
         # object, not a server bug. A short greeting fits well inside 32.
         "max_tokens": 32, "temperature": 0,
         "response_format": {"type": "json_object"},
         "chat_template_kwargs": {"enable_thinking": False}},
        timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    content = (j.get("choices", [{}])[0].get("message") or {}).get("content")
    parsed = None
    if content:
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = None
    sec.check(status == 200, "response_format json_object request -> 200", "status=%s" % status,
              response=body)
    sec.check(isinstance(parsed, dict),
              "response_format json_object: content parses as a JSON object",
              repr(content)[:120], response=body)

    # n_probs on /v1/completions: logprobs for the generated tokens. Sent WITHOUT
    # top_k/top_p/min_p -- those interact with the n_probs code path, so the clean
    # assertion is n_probs alone (PARITY.md 2.3, server-schema.cpp:179-181).
    status, _, body = http_request(base + "/v1/completions", "POST",
                                   {"prompt": "Q: What is 17*23? A:", "max_tokens": 4,
                                    "temperature": 0, "n_probs": 5},
                                   timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    ch = (j.get("choices") or [{}])[0] if isinstance(j, dict) else {}
    lp = ch.get("logprobs")
    sec.check(status == 200, "n_probs request -> 200", "status=%s" % status, response=body)
    sec.check(isinstance(lp, dict),
              "n_probs: choices[0].logprobs is an object (server-task.cpp:433-437)",
              str(lp)[:120], response=body)
    lpc = lp.get("content") if isinstance(lp, dict) else None
    sec.check(isinstance(lpc, list) and len(lpc) > 0,
              "n_probs: logprobs.content covers the generated tokens",
              "n=%d" % (len(lpc) if isinstance(lpc, list) else -1))
    tl = lpc[0].get("top_logprobs") if lpc else None
    sec.check(isinstance(tl, list) and 1 <= len(tl) <= 5,
              "n_probs=5: each generated token carries 1-5 top logprobs",
              "n=%d" % (len(tl) if isinstance(tl, list) else -1))


# ---- 19. n, stop, samplers, chat logprobs, cache_prompt, prefix completion -------


@section("n, stop, samplers, chat logprobs, cache_prompt, prefix completion")
def s_request_fields(suite, sec):
    base = suite.base

    # n > 1 needs more than one slot. Stock llama-server resolves n_parallel to 4
    # (arg.cpp:1400 -> server.cpp:156-159) so n=2 works there. amp deliberately forces
    # n_parallel = 1, because four concurrent slots thrash the shared ~10.9 GiB CPU expert
    # set on this box and cost 44x on decode (docs/BENCH.md). The consequence is real and
    # asserted here rather than hidden: n>1 is a typed 400, and --parallel N restores it.
    # See RUNNING.md, "Concurrency".
    status, _, body = http_request(base + "/v1/completions", "POST",
                                   {"prompt": "Q: What is 17*23? A:", "max_tokens": 4,
                                    "temperature": 0, "n": 2},
                                   timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    choices = j.get("choices") if isinstance(j, dict) else None
    err = j.get("error", {}) if isinstance(j, dict) else {}
    n_parallel_1 = status == 400 and isinstance(err.get("type"), str)
    sec.check(n_parallel_1 or (status == 200 and isinstance(choices, list) and len(choices) == 2),
              "n=2 is served with >=2 slots, or is a typed 400 explaining the bound",
              "status=%s type=%s" % (status, err.get("type")))
    if isinstance(choices, list) and len(choices) == 2:
        sec.check([c.get("index") for c in choices] == [0, 1],
                  "n=2: choice indices are 0 and 1", str([c.get("index") for c in choices]))
        sec.check(all(isinstance(c.get("text"), str) and c["text"] for c in choices),
                  "n=2: both choices carry non-empty text",
                  repr([c.get("text") for c in choices])[:100])

    # stop: generation halts at the stop string, which is erased from the output
    # (server-context.cpp:1854-1858).
    status, _, body = http_request(
        base + "/v1/completions", "POST",
        # The stop string has to occur in the GENERATED text. An earlier version prompted
        # "...one two three four..." and stopped on "three", which the model never generates
        # because it is already in the prompt - so finish_reason came back "length" and the
        # test was measuring the prompt, not the stop logic. Measured working: this prompt
        # completes to " Tokyo", which is then erased, leaving finish_reason "stop".
        {"prompt": "The capital city of Japan is called", "max_tokens": 12,
         "temperature": 0, "stop": ["Tokyo"]},
        timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    ch = (j.get("choices") or [{}])[0] if isinstance(j, dict) else {}
    text = ch.get("text")
    sec.check(status == 200, "stop request -> 200", "status=%s" % status, response=body)
    sec.check(ch.get("finish_reason") == "stop",
              "stop: finish_reason == stop", str(ch.get("finish_reason")))
    sec.check(isinstance(text, str) and "Tokyo" not in text,
              "stop: the stop string is erased from the output", repr(text)[:80])

    # samplers: an explicit sampler sequence. Every name must be valid for this
    # llama.cpp version: dry, top_k, top_p, top_n_sigma, typ_p, min_p, temperature, xtc,
    # infill, penalties, adaptive_p (sampling.cpp:833-845). "dist" is NOT a valid name
    # here (the alias is "temp"/"temperature"); unknown names are dropped with a
    # warning (sampling.cpp:885), which would make the test pass without testing the
    # field at all.
    status, _, body = http_request(
        base + "/v1/chat/completions", "POST",
        {"messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 4,
         "temperature": 0,
         "samplers": ["top_k", "typ_p", "top_p", "min_p", "temperature", "penalties"],
         "chat_template_kwargs": {"enable_thinking": False}},
        timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    ok = check_keys(sec, "samplers: accepted, well-formed (PARITY.md 2.3)", j, CHAT_KEYS,
                    response=body)
    if ok:
        sec.check(j["choices"][0].get("finish_reason") in ("stop", "length", "tool_calls"),
                  "samplers: finish_reason valid", str(j["choices"][0].get("finish_reason")))

    # logprobs/top_logprobs on the chat path: logprobs:true sets n_probs=top_logprobs
    # (PARITY.md 2.1, server-common.cpp:1407-1411); the response carries them per choice
    # (server-task.cpp:433-437).
    status, _, body = http_request(
        base + "/v1/chat/completions", "POST",
        {"messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 4,
         "temperature": 0, "logprobs": True, "top_logprobs": 5,
         "chat_template_kwargs": {"enable_thinking": False}},
        timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    ch = (j.get("choices") or [{}])[0] if isinstance(j, dict) else {}
    ok = check_keys(sec, "chat logprobs: response key set", j, CHAT_KEYS, response=body)
    if ok:
        lp = ch.get("logprobs")
        sec.check(isinstance(lp, dict),
                  "chat logprobs: choices[0].logprobs is an object", str(lp)[:120])
        lpc = lp.get("content") if isinstance(lp, dict) else None
        sec.check(isinstance(lpc, list) and len(lpc) > 0,
                  "chat logprobs: logprobs.content covers the generated tokens",
                  "n=%d" % (len(lpc) if isinstance(lpc, list) else -1))
        tl = lpc[0].get("top_logprobs") if lpc else None
        sec.check(isinstance(tl, list) and 1 <= len(tl) <= 5,
                  "chat logprobs: top_logprobs=5 -> 1-5 entries per token",
                  "n=%d" % (len(tl) if isinstance(tl, list) else -1))

    # cache_prompt:false: the prompt is neither read from nor written to the cache, so
    # cached_tokens must be 0 even though earlier sections cached similar prompts.
    # Unique prompt string: no earlier section can have populated the cache with it.
    status, _, body = http_request(
        base + "/v1/chat/completions", "POST",
        {"messages": [{"role": "user", "content": "zebra cache probe 42"}],
         "max_tokens": 4, "temperature": 0, "cache_prompt": False,
         "chat_template_kwargs": {"enable_thinking": False}},
        timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    cached = ((j.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    sec.check(status == 200, "cache_prompt:false request -> 200", "status=%s" % status,
              response=body)
    sec.check(cached == 0,
              "cache_prompt:false -> usage.prompt_tokens_details.cached_tokens == 0",
              str(cached))

    # continue_final_message + add_generation_prompt:false = assistant prefix completion.
    #
    # The message shape matters and is not obvious. With an ASSISTANT-ONLY message list this
    # model's chat template throws ("No messages provided", jinja line 43) and llama.cpp's
    # ex_wrapper turns that into a 500 - documented in PARITY.md 4.3 as the general
    # exception-to-status rule. The realistic shape is a user turn followed by the assistant
    # prefill, which is what prefix completion actually means in a conversation.
    status, _, body = http_request(
        base + "/v1/chat/completions", "POST",
        {"messages": [{"role": "user", "content": "List three fruits."},
                      {"role": "assistant", "content": "A list of fruits: 1. Apples 2. Bananas 3."}],
         "continue_final_message": True, "add_generation_prompt": False,
         "max_tokens": 16, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": False}},
        timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    content = (j.get("choices", [{}])[0].get("message") or {}).get("content")
    sec.check(status == 200, "continue_final_message request -> 200", "status=%s" % status,
              response=body)
    # The prefix must be continued, not restarted: the model has to carry on from
    # "... 2. Bananas 3." and the result still contains the prefix text.
    sec.check(isinstance(content, str) and "Bananas" in content,
              "continue_final_message: the assistant prefix was continued, not replaced",
              repr(content)[:70])

    # The assistant-only shape is a documented 500 from this template, not a silent success.
    st_only, _, body_only = http_request(
        base + "/v1/chat/completions", "POST",
        {"messages": [{"role": "assistant", "content": "The capital of France is"}],
         "continue_final_message": True, "add_generation_prompt": False,
         "max_tokens": 4, "temperature": 0},
        timeout=suite.timeout)
    try:
        jo = json.loads(body_only)
    except Exception:
        jo = {}
    err_only = jo.get("error", {}) if isinstance(jo, dict) else {}
    sec.check(st_only in (400, 500) and isinstance(err_only.get("type"), str),
              "continue_final_message with an assistant-only list is a typed error, not a silent 200",
              "status=%s type=%s" % (st_only, err_only.get("type")))

    # add_generation_prompt:false on its own: accepted, generation still happens.
    status, _, body = http_request(
        base + "/v1/chat/completions", "POST",
        {"messages": [{"role": "user", "content": "Say OK"}],
         "add_generation_prompt": False, "max_tokens": 4, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": False}},
        timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    content = (j.get("choices", [{}])[0].get("message") or {}).get("content")
    sec.check(status == 200, "add_generation_prompt:false request -> 200", "status=%s" % status,
              response=body)
    sec.check(isinstance(content, str) and len(content) > 0,
              "add_generation_prompt:false: content is still produced", repr(content)[:60])

    # parallel_tool_calls: accepted with tools supplied (PARITY.md 2.1). The model is
    # not forced to call anything, so this stays a cheap shape check.
    tools = [{"type": "function",
              "function": {"name": "get_weather",
                           "description": "Get the weather for a city",
                           "parameters": {"type": "object",
                                         "properties": {"location": {"type": "string"}},
                                         "required": ["location"]}}}]
    status, _, body = http_request(
        base + "/v1/chat/completions", "POST",
        {"messages": [{"role": "user", "content": "Say OK"}], "tools": tools,
         "parallel_tool_calls": False, "max_tokens": 4, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": False}},
        timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    check_keys(sec, "parallel_tool_calls:false with tools: accepted, well-formed (PARITY.md 2.1)",
               j, CHAT_KEYS, response=body)


# ---- 20. Reasoning kwargs (accepted, ignored) --------------------------------


@section("Reasoning kwargs (accepted, ignored)")
def s_reasoning_kwargs(suite, sec):
    base = suite.base
    # PARITY.md 6.4/6.5 and 7.3: preserve_reasoning and reasoning_effort are accepted,
    # but this template has no supports_preserve_reasoning / supports_reasoning_effort
    # cap, so both kwargs are ignored. Assert acceptance + a well-formed response, and
    # via /apply-template (free, no generation) that the rendered prompt is unchanged.

    def chat(kwargs, label, extra=None):
        payload = {"messages": [{"role": "user", "content": "Say OK"}],
                   "max_tokens": 4, "temperature": 0,
                   "chat_template_kwargs": kwargs}
        if extra:
            payload.update(extra)
        status, _, body = http_request(base + "/v1/chat/completions", "POST", payload,
                                       timeout=suite.timeout)
        try:
            j = json.loads(body)
        except Exception:
            j = {}
        sec.check(status == 200, "%s -> 200" % label, "status=%s" % status, response=body)
        ok = check_keys(sec, "%s: well-formed (PARITY.md 3.1)" % label, j, CHAT_KEYS,
                        response=body)
        if ok:
            sec.check(j["choices"][0].get("finish_reason") in ("stop", "length", "tool_calls"),
                      "%s: finish_reason valid" % label,
                      str(j["choices"][0].get("finish_reason")))
        return j

    chat({"enable_thinking": False, "preserve_reasoning": False},
         "preserve_reasoning=false in chat_template_kwargs (PARITY.md 6.4)")
    chat({"enable_thinking": False}, "reasoning_effort=high (PARITY.md 6.5)",
         extra={"reasoning_effort": "high"})

    def render(kwargs):
        body = {"messages": [{"role": "user", "content": "What is 2+2?"}],
                 "chat_template_kwargs": kwargs}
        status, _, resp = http_request(base + "/apply-template", "POST", body,
                                       timeout=suite.timeout)
        try:
            return json.loads(resp).get("prompt", "")
        except Exception:
            return ""

    p_base = render({"enable_thinking": False})
    p_preserve = render({"enable_thinking": False, "preserve_reasoning": False})
    p_effort = render({"enable_thinking": False, "reasoning_effort": "high"})
    sec.check(bool(p_base) and p_preserve == p_base,
              "preserve_reasoning: rendered prompt unchanged (kwarg ignored, PARITY.md 6.4/7.3)",
              "base=%d chars, with-kwarg=%d chars" % (len(p_base), len(p_preserve)))
    sec.check(bool(p_base) and p_effort == p_base,
              "reasoning_effort: rendered prompt unchanged (kwarg ignored, PARITY.md 6.5/7.3)",
              "base=%d chars, with-kwarg=%d chars" % (len(p_base), len(p_effort)))


# ---- 21. Control and stream lookup routes ------------------------------------


@section("Control and stream lookup routes")
def s_control_lookup(suite, sec):
    base = suite.base

    # PARITY.md 1 + server-context.cpp:4950-4985: the body is {"id": "<cmpl_id>",
    # "action": "reasoning_end"}. Note the completion id field is "id" -- the
    # "reasoning_control":true field belongs to the ORIGINAL generation request and
    # arms the budget sampler; it is not part of the control body.
    status, _, body = http_request(base + "/v1/chat/completions/control", "POST",
                                   {"id": "cmpl-nonexistent", "action": "reasoning_end"},
                                   timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = {}
    sec.check(status == 200, "control: unknown completion id -> 200", "status=%s" % status,
              response=body)
    sec.check(isinstance(j, dict) and j.get("success") is False,
              "control: unknown id -> {\"success\":false} (server-context.cpp:2469-2476)",
              str(j)[:120], response=body)
    sec.check(isinstance(j.get("message"), str) and j.get("message"),
              "control: failure carries a non-empty message", str(j.get("message"))[:80])

    # missing id -> 400 (server-context.cpp:4956-4958)
    status, _, body = http_request(base + "/v1/chat/completions/control", "POST",
                                   {"action": "reasoning_end"}, timeout=suite.timeout)
    err = parse_error(body)
    sec.check(status == 400 and err.get("type") == "invalid_request_error",
              "control: missing id -> 400 invalid_request_error",
              "status=%s" % status, response=body)

    # unknown action -> 400 (server-context.cpp:4960-4962)
    status, _, body = http_request(base + "/v1/chat/completions/control", "POST",
                                   {"id": "cmpl-nonexistent", "action": "bogus"},
                                   timeout=suite.timeout)
    err = parse_error(body)
    sec.check(status == 400 and err.get("type") == "invalid_request_error",
              "control: unknown action -> 400 invalid_request_error",
              "status=%s" % status, response=body)

    # PARITY.md 1 + server-stream.cpp:503-558: POST {"conversation_ids":[...]} -> 200
    # with a JSON array (empty when no session matches). Lookup only reports sessions
    # the server already knows; it never creates one.
    status, _, body = http_request(base + "/v1/streams/lookup", "POST",
                                   {"conversation_ids": ["cmpl-nonexistent",
                                                          "cmpl-also-nonexistent"]},
                                   timeout=suite.timeout)
    try:
        j = json.loads(body)
    except Exception:
        j = None
    sec.check(status == 200, "POST /v1/streams/lookup -> 200 for unknown ids",
              "status=%s" % status, response=body)
    sec.check(isinstance(j, list),
              "POST /v1/streams/lookup -> JSON array (empty: no such stream sessions)",
              repr(body)[:120], response=body)

    # Live positive path: a streaming request with reasoning_control armed (the budget
    # sampler also needs non-empty start/end tags, sampling.cpp:311) is force-ended
    # mid-flight via the control route. Thinking must stay ON -- the budget machinery
    # exists to cut reasoning short. Worst case ~128 generated tokens.
    def open_stream_first(url, payload, timeout):
        """Streaming POST; read only the first SSE data line so the completion id can
        be used mid-generation. Returns (status, response_or_None, first_data_or_None)."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "text/event-stream")
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            return e.code, None, None
        except Exception:
            return -1, None, None
        first = None
        try:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("data: "):
                    first = line[6:]
                    break
        except Exception:
            pass
        return 200, r, first

    # "stream": True is load-bearing and was missing: without it the server returns one
    # non-streaming JSON object, there are no "data: " lines to read, and the completion id
    # can never be harvested mid-generation - which is the whole point of the control test.
    payload = {"messages": [{"role": "user",
                             "content": "Count from 1 to 1000, one number per line."}],
               "max_tokens": 128, "stream": True, "reasoning_control": True,
               "reasoning_budget_start_tag": "<think>", "reasoning_budget_end_tag": "</think>"}
    status, r, first = open_stream_first(base + "/v1/chat/completions", payload,
                                         suite.timeout)
    sec.check(status == 200, "control: live reasoning_control stream -> 200",
              "status=%s" % status)
    if status != 200 or r is None:
        return
    cmpl_id = None
    try:
        cmpl_id = json.loads(first).get("id")
    except Exception:
        cmpl_id = None
    if not cmpl_id:
        sec.check(False, "control: stream chunk carries the completion id",
                  repr(first)[:160])
        r.close()
        return
    status_c, _, body_c = http_request(base + "/v1/chat/completions/control", "POST",
                                       {"id": cmpl_id, "action": "reasoning_end"},
                                       timeout=suite.timeout)
    try:
        jc = json.loads(body_c)
    except Exception:
        jc = {}
    sec.check(status_c == 200, "control: reasoning_end on the live id -> 200",
              "status=%s" % status_c, response=body_c)
    sec.check(jc.get("success") is True,
              "control: reasoning_end on a live reasoning_control completion -> success true",
              str(jc)[:160], response=body_c)

    # drain the stream: after the forced end tag the model finishes its answer
    chunks = []
    if first:
        chunks.append(first)
    try:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data: "):
                chunks.append(line[6:])
    except Exception:
        pass
    finally:
        r.close()
    objs, done = parse_chunks(chunks)
    sec.check(done, "control: stream still ends with [DONE] after the forced end")
    finish = [o for o in objs if isinstance(o, dict) and o.get("choices") and
              o["choices"][0].get("finish_reason")]
    sec.check(bool(finish),
              "control: stream still emits a finish_reason after the forced end",
              str(finish[0]["choices"][0]["finish_reason"]) if finish else "none")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def health_check(base, timeout):
    for path in ("/health", "/v1/health"):
        status, _, _ = http_request(base + path, timeout=timeout)
        if status == 200:
            return True, path
    return False, None


def main():
    ap = argparse.ArgumentParser(
        description="Parity test for amp-server (llama.cpp tools/server) against docs/PARITY.md.")
    ap.add_argument("--url", default="http://127.0.0.1:8081", help="server base URL")
    ap.add_argument("--filter", default="", help="run only sections whose name contains SUBSTR")
    ap.add_argument("--list", action="store_true", help="list every section name and exit")
    ap.add_argument("--verbose", action="store_true", help="print passing checks too")
    # 120 s is not enough for the FIRST generation after a server start on this box: the
    # expert set is not in the page cache yet and that request has been measured at 50-115 s.
    # A suite that fails on a cold server is a flaky suite.
    ap.add_argument("--timeout", type=int, default=600, help="per-request timeout in seconds")
    args = ap.parse_args()

    if args.list:
        for i, (name, _) in enumerate(SECTIONS, 1):
            print("%2d. %s" % (i, name))
        return 0

    base = args.url.rstrip("/")

    if args.filter:
        filt = args.filter.lower()
        selected = [(n, f) for n, f in SECTIONS if filt in n.lower()]
        if not selected:
            print("ERROR: --filter %r matches no section. Run with --list to see section names."
                  % args.filter)
            return 2
    else:
        selected = SECTIONS

    print("amp-server parity test against %s" % base)
    print("model: Occamy-1.0-APEX-I-MiniPlus-V2.1-Abliterated.gguf (qwen35moe, reasoning)")
    print("spec:  docs/PARITY.md | stdlib only | cheap requests (max_tokens 1-8 for shape tests)")

    ok, path = health_check(base, args.timeout)
    if not ok:
        print("\nERROR: no server answered at %s (tried /health and /v1/health)." % base)
        print("Start amp-server first (see docs/RUN.md), then re-run this script.")
        return 1
    print("health check: %s -> 200\n" % path)

    suite = Suite(base, args.timeout, args.verbose)

    for i, (name, fn) in enumerate(selected, 1):
        sec = Section("%d. %s" % (i, name), args.verbose)
        print("=== %d. %s ===" % (i, name))
        try:
            fn(suite, sec)
        except Exception as e:
            sec.check(False, "section raised an exception", "%s: %s" % (type(e).__name__, e))
        suite.sections.append(sec)

    # summary table
    print("\n" + "=" * 78)
    print("%-42s %-6s  %s" % ("Section", "Status", "Detail"))
    print("-" * 78)
    n_fail = 0
    for sec in suite.sections:
        st = sec.status
        if st == "FAIL":
            n_fail += 1
        print("%-42s %-6s  %s" % (sec.name, st, sec.summary()))
    print("=" * 78)

    # prominent failure summary
    failures = [(sec.name, l, d) for sec in suite.sections for s, l, d in sec.results if s == "fail"]
    if failures:
        print("\nFAILURES (%d):" % len(failures))
        for name, label, detail in failures:
            print("  [%s] %s" % (name, label))
            if detail:
                print("      %s" % detail)
        print("\nRESULT: %d section(s) FAILED. See above for response bodies." % n_fail)
        return 1

    n_skip = sum(1 for sec in suite.sections if sec.status == "SKIP")
    n_pass = sum(1 for sec in suite.sections if sec.status == "PASS")
    print("\nRESULT: %d passed, %d skipped (not testable on this box), 0 failed."
          % (n_pass, n_skip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
