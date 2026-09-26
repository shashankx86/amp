#!/usr/bin/env python3
"""End-to-end check of amp-server: raw completions, chat + reasoning, SSE streaming, prefix cache.

Not a unit test - it drives a running server the way OpenCode does, including the reasoning
round-trip that llama.cpp silently drops.

usage: scripts/smoke_server.py [--url http://127.0.0.1:8081] [--max-tokens 96]
"""
import argparse
import json
import time
import urllib.request

FAIL = []


def check(cond, label, detail=""):
    mark = "ok  " if cond else "FAIL"
    if not cond:
        FAIL.append(label)
    print(f"  [{mark}] {label}" + (f"  {detail}" if detail else ""))


def post(url, payload, stream=False, timeout=900):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    if stream:
        req.add_header("Accept", "text/event-stream")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if not stream:
            return json.loads(r.read())
        chunks = []
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                chunks.append(None)
                continue
            chunks.append(json.loads(body))
        return chunks


def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--max-tokens", type=int, default=96)
    args = ap.parse_args()
    base = args.url.rstrip("/")

    print(f"amp-server smoke test against {base}\n")

    # ---- 1. liveness + diagnostics -------------------------------------------------
    print("1. health and models")
    h = get(f"{base}/health")
    check(h.get("status") == "ok", "/health status ok", h.get("detail", "")[:60])
    m = get(f"{base}/v1/models")
    check(m.get("object") == "list" and len(m["data"]) == 1, "/v1/models shape")
    check(m["data"][0].get("id"), "model id present", m["data"][0].get("id", ""))

    # ---- 2. raw completion ---------------------------------------------------------
    print("\n2. /v1/completions returns the raw generation (think block included)")
    t0 = time.time()
    c = post(f"{base}/v1/completions",
             {"prompt": "Q: What is 17*23? A:", "n_predict": args.max_tokens, "temperature": 0})
    dt = time.time() - t0
    text = c["choices"][0]["text"]
    check(len(c["choices"][0]["text"]) > 8 or c["choices"][0]["finish_reason"] == "length",
          "completion produced text", repr(text[:60]))
    check(c["choices"][0]["finish_reason"] in ("stop", "length"), "finish_reason valid",
          c["choices"][0]["finish_reason"])
    check(c["usage"]["completion_tokens"] > 0, "usage.completion_tokens",
          str(c["usage"]["completion_tokens"]))
    check("prompt_tokens" in c["usage"], "usage.prompt_tokens")
    t = c.get("amp_timings", {})
    check(t.get("predicted_per_second", 0) > 0, "timings present",
          f"prefill {t.get('prompt_per_second', 0):.1f} t/s, decode {t.get('predicted_per_second', 0):.2f} t/s")
    print(f"       {dt:.1f}s wall for {c['usage']['completion_tokens']} tokens")

    # ---- 3. chat + reasoning split -------------------------------------------------
    print("\n3. /v1/chat/completions: template rendering and the reasoning split")
    msgs = [{"role": "user", "content": "What is 17*23? Answer with just the number."}]
    ch = post(f"{base}/v1/chat/completions",
              {"messages": msgs, "max_tokens": args.max_tokens, "temperature": 0})
    msg = ch["choices"][0]["message"]
    check(msg["role"] == "assistant", "role is assistant")
    check("content" in msg, "content field present")
    check("reasoning_content" in msg or len(msg.get("content", "")) > 0,
          "reasoning or content present",
          f"reasoning={len(msg.get('reasoning_content',''))} content={len(msg.get('content',''))}")

    # ---- 4. prefix cache: the agentic pattern ---------------------------------------
    # This is what OpenCode does every turn: resend the whole conversation, including the assistant
    # turn verbatim. The cached sequence is prompt+completion, so the common prefix should cover
    # everything and only the new user message needs evaluating.
    print("\n4a. prefix cache: agentic append (history resent verbatim)")
    before = get(f"{base}/props")["prefix_cache"]
    t_a = msg.get("content", "") or "391"
    follow = msgs + [{"role": "assistant", "content": t_a,
                      "reasoning_content": msg.get("reasoning_content", "")},
                     {"role": "user", "content": "Now multiply that by 2. Answer with just the number."}]
    c2 = post(f"{base}/v1/chat/completions",
              {"messages": follow, "max_tokens": args.max_tokens, "temperature": 0})
    t2 = c2["amp_timings"]
    total = t2["prompt_n"] + t2["prompt_cached"]
    check(t2["prompt_cached"] > 0, "append reused the cached prefix",
          f"cached {t2['prompt_cached']}/{total} tokens ({100 * t2['cache_hit_rate']:.0f}%)")
    check(t2["cache_rewound_exactly"] is True, "no rewind needed for an append")
    check(c2["usage"]["prompt_tokens_details"]["cached_tokens"] == t2["prompt_cached"],
          "usage reports the cached count honestly",
          str(c2["usage"]["prompt_tokens_details"]["cached_tokens"]))
    check(bool(c2["choices"][0]["message"].get("content")), "answer is non-empty",
          repr(c2["choices"][0]["message"].get("content", "")[:60]))
    # Not asserted: the exact wording. This is a base-model quant, and greedy decoding of "multiply
    # that by 2" can legitimately be conversational. Output *quality* is gated by scripts/parity.py
    # against llama-server, which is a much stronger check than a string match here.

    # ---- 4b. divergence must be handled, not crash ----------------------------------
    print("\n4b. divergent prompt: correct answer, no position abort")
    other = [{"role": "user", "content": "What is the capital of France? One word."}]
    c3 = post(f"{base}/v1/chat/completions",
              {"messages": other, "max_tokens": args.max_tokens, "temperature": 0})
    check(bool(c3["choices"][0]["message"].get("content")), "divergent request answered",
          repr(c3["choices"][0]["message"].get("content", "")[:40]))
    t3 = c3["amp_timings"]
    check(t3["prompt_n"] == c3["usage"]["prompt_tokens"] or t3["prompt_n"] > 0,
          "divergent prompt was evaluated", f"computed {t3['prompt_n']} tokens")
    print(f"       cache_rewound_exactly={t3['cache_rewound_exactly']} "
          f"(false means the recurrent memory cannot rewind, so a full re-prefill is the only "
          f"correct option)")

    after = get(f"{base}/props")["prefix_cache"]
    check(after["requests"] > before["requests"], "props cache stats advanced",
          f"{before['requests']} -> {after['requests']} requests")
    check(after["rewinds_cleared"] + after.get("rewinds_exact", 0) >= 1,
          "the divergence was recorded as a rewind",
          f"exact={after.get('rewinds_exact', 0)} cleared={after['rewinds_cleared']}")

    # ---- 5. streaming --------------------------------------------------------------
    print("\n5. SSE streaming")
    chunks = post(f"{base}/v1/chat/completions",
                  {"messages": [{"role": "user", "content": "Count: 1 2 3"}],
                   "max_tokens": 24, "temperature": 0, "stream": True}, stream=True)
    check(chunks and chunks[-1] is None, "stream terminated with [DONE]",
          f"{len(chunks)} events")
    check(chunks[0]["choices"][0]["delta"].get("role") == "assistant", "first chunk has the role")
    finish = [c for c in chunks if c and c["choices"][0].get("finish_reason")]
    check(bool(finish), "a chunk carries finish_reason",
          finish[0]["choices"][0]["finish_reason"] if finish else "")
    text = "".join(c["choices"][0]["delta"].get("content", "")
                   for c in chunks if c and c["choices"][0].get("delta"))
    check(len(text) > 0, "streamed deltas reassemble into text", repr(text[:50]))
    if finish and "usage" in finish[0]:
        check(finish[0]["usage"]["completion_tokens"] > 0, "final chunk carries usage")

    # ---- 6. error handling ---------------------------------------------------------
    print("\n6. errors are clean, not crashes")
    try:
        post(f"{base}/v1/chat/completions", {"messages": []})
        check(False, "empty messages rejected")
    except urllib.error.HTTPError as e:
        check(e.code == 400, "empty messages -> 400", str(e.code))
    try:
        post(f"{base}/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}],
                                             "max_tokens": 10 ** 9})
        check(True, "absurd max_tokens does not take the server down")
    except Exception as e:  # noqa: BLE001
        check(False, "absurd max_tokens handled", str(e)[:80])
    check(get(f"{base}/health").get("status") == "ok", "server still healthy afterwards")

    # ---- 6b. a prompt longer than one ubatch ---------------------------------------
    # llama_process() asserts a batch is no larger than n_ubatch, so the caller must split the
    # prefill. This used to fail outright with "prefill batch full at token 2048", which is every
    # real conversation, because an agent client resends the whole history every turn.
    print("\n6b. a prompt longer than the ubatch (the agent resends everything every turn)")
    long_prompt = "context " * 4000          # ~4000 tokens, well past ubatch 1024
    lc = post(f"{base}/v1/completions", {"prompt": long_prompt, "n_predict": 8, "temperature": 0})
    check("error" not in lc, "a 4000-token prompt prefills", str(lc.get("error", {}).get("message", ""))[:80])
    if "error" not in lc:
        check(lc["amp_timings"]["prompt_n"] > 2048, "all of it was evaluated",
              f"{lc['amp_timings']['prompt_n']} tokens")
    lm = post(f"{base}/v1/chat/completions",
              {"messages": [{"role": "user", "content": long_prompt}], "max_tokens": 8, "temperature": 0})
    check("error" not in lm, "a 4000-token chat history prefills",
          str(lm.get("error", {}).get("message", ""))[:80])

    # ---- 6c. the reasoning split --------------------------------------------------
    # This template ends its generation prompt with the opening think tag, so the model reasons
    # without ever emitting one. Splitting on the tag therefore has to start "inside" the block, or
    # the whole chain of thought is delivered as the answer - which is what a conversation title
    # looked like: "The user said hi. This is a short, conversational greeting. According to the
    # rules, I should...".
    print("\n6c. reasoning is separated from content")
    r1 = post(f"{base}/v1/chat/completions",
              {"messages": [{"role": "system", "content": "Answer in one short line."},
                            {"role": "user", "content": "What is 2+2?"}],
               "max_tokens": 200, "temperature": 0})
    content = r1["choices"][0]["message"].get("content") or ""
    reasoning = r1["choices"][0]["message"].get("reasoning_content") or ""
    check("</think>" not in content, "no stray think tag in content", repr(content[:60]))
    check(not content.lower().startswith("here's a thinking process"),
          "content is not a chain of thought", repr(content[:60]))
    check(bool(reasoning) or bool(content), "reasoning or content is populated",
          f"reasoning={len(reasoning)} content={len(content)}")
    if content:
        check("4" in content, "the answer actually reached content", repr(content[:60]))
    # The clean way to ask a thinking model for a direct answer: the template's own switch.
    r2 = post(f"{base}/v1/chat/completions",
              {"messages": [{"role": "system", "content": "Generate a short title. Under 8 words."},
                            {"role": "user", "content": "how do I fix a segfault"}],
               "max_tokens": 24, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    t2 = (r2["choices"][0]["message"].get("content") or "").strip()
    check(bool(t2) and len(t2) < 120, "enable_thinking=false yields a direct answer", repr(t2[:80]))

    # ---- 7. concurrency: the regression test that matters -------------------------
    # A llama_context is not reentrant. Agentic clients put two requests in flight at once all the
    # time (OpenCode asks for a conversation title while the main stream is running), and before the
    # task queue this produced zero-byte streams and took the process down with them.
    print("\n7. concurrent streams (the OpenCode pattern: a main stream plus a side request)")
    import threading

    results = {}

    def run(name, body, n_predict):
        try:
            results[name] = ("ok", post(f"{base}/v1/chat/completions", body, stream=True))
        except Exception as e:  # noqa: BLE001
            results[name] = ("error", str(e)[:120])

    long_body = {"messages": [{"role": "user", "content": "Write two sentences about caches."}],
                 "max_tokens": 120, "temperature": 0.7, "stream": True,
                 "stream_options": {"include_usage": True}}
    short_body = {"messages": [{"role": "user", "content": "Reply with just the word OK"}],
                  "max_tokens": 8, "temperature": 0, "stream": True}
    threads = [threading.Thread(target=run, args=("main", long_body, 120))]
    threads += [threading.Thread(target=run, args=(f"side{i}", short_body, 8)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=900)

    for name in ["main", "side0", "side1", "side2"]:
        if name not in results:
            check(False, f"{name} stream completed", "timed out")
            continue
        kind, payload = results[name]
        if kind != "ok":
            check(False, f"{name} stream completed", payload)
            continue
        frags = [c for c in payload if c is not None]
        finish = [c for c in frags if c["choices"] and c["choices"][0].get("finish_reason")]
        check(bool(frags) and bool(finish),
              f"{name} stream has content and a finish_reason",
              f"{len(frags)} chunks, finish_reason="
              f"{finish[0]['choices'][0]['finish_reason'] if finish else 'MISSING'}")
    check(payload[-1] is None, "streams end with [DONE]")
    usage_chunks = [c for c in results["main"][1] if c and not c.get("choices")]
    check(bool(usage_chunks), "include_usage produces the usage-only chunk",
          str(usage_chunks[0].get("usage")) if usage_chunks else "missing")
    check(get(f"{base}/health").get("status") == "ok", "server survived the concurrent load")
    q = get(f"{base}/props").get("queue", {})
    check(q.get("failed", 0) == 0, "no task threw", str(q))

    print("\n" + ("FAILED: " + ", ".join(FAIL) if FAIL else "all checks passed"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
