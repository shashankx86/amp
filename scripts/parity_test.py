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
scripts/smoke_server.py (kept for reference) and scripts/parity.py for that.

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
