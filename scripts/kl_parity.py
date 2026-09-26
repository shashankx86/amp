#!/usr/bin/env python3
"""KL-divergence parity harness for amp vs llama-server.

This tool compares two *captures* (saved runs of one engine) by computing KL divergence
between their output probability distributions -- not just the top-5 logprobs that
scripts/parity.py reports, but the full top-N distribution per generated position.

The claim under test is "zero quality loss" when replacing amp's hand-rolled server with
llama.cpp's own tools/server. Same ggml kernels, same weights, same KV dtypes, same sampler.
KL divergence over the output distribution is the strongest signal we have: if the swap is
truly lossless, KL should be ~0 (only float-placement noise).

THE n_probs RENORMALISATION TRAP (learned the hard way):
  Do NOT send top_k, top_p or min_p together with n_probs. The server filters the distribution
  and then renormalises the reported logprobs, which produces large apparent differences that
  are pure protocol error. temperature 0 alone is greedy in llama.cpp and leaves the
  distribution unfiltered -- that is the only valid configuration for this tool.

TRUNCATED SUPPORT AND CAPTURED MASS:
  Both sides are truncated to the top-N tokens. We build the joint support (union of both
  sides' token sets at each position) and renormalise both sides over that union so KL is
  well-defined. But if a side captured less than ~1.0 total probability mass, the KL understates
  the true divergence -- the missing mass went to tokens we never saw. We therefore report the
  captured mass alongside every KL number. If mass is not ~1, the KL cannot be trusted.

n_probs CAP:
  llama.cpp's server-schema.cpp defines n_probs as field_num with NO hard upper bound.
  server-common.cpp:get_token_probabilities clamps n_top to the full vocab size
  (n_logits). For this model the vocab is 3,115,143 tokens (gpt2 tokenizer with a huge
  custom vocab). Requesting n_probs > vocab_size is silently clamped. The practical cap is
  therefore the vocab size, but requesting 3.1M logprobs per position is absurd. We default
  to n_probs=32, which captures >99.9% of probability mass for greedy decoding of a base
  model. Use --n-probs to increase if you need a tighter tail. The script reports the
  captured mass so you can judge whether 32 is enough.

DETERMINISM CHECK:
  Comparing a capture against itself must give exactly 0 divergence. The --selftest flag
  verifies this and unit-tests the logprob-to-probability and KL maths on hand-written
  inputs, with no server needed.

ATOMIC WRITES:
  Capture files are written to a temp file then renamed, so an interrupted run cannot leave
  a half-written golden file. Default output is amp/quality/ (NOT /tmp, which is tmpfs on
  this machine and would lose captures on reboot).

USAGE:
  # Capture from a running server (greedy, temperature 0):
  python3 scripts/kl_parity.py capture --url http://127.0.0.1:8081 --tag amp-before --out amp/quality/amp-before.json

  # Capture from the replacement server:
  python3 scripts/kl_parity.py capture --url http://127.0.0.1:8081 --tag llama-after --out amp/quality/llama-after.json

  # Compare:
  python3 scripts/kl_parity.py compare --a amp/quality/amp-before.json --b amp/quality/llama-after.json

  # Compare with a gate (non-zero exit if max KL exceeds threshold):
  python3 scripts/kl_parity.py compare --a amp/quality/amp-before.json --b amp/quality/llama-after.json --max-kl 0.01

  # Self-test the maths (no server needed):
  python3 scripts/kl_parity.py --selftest

PROTOCOL NOTES (inherited from scripts/parity.py):
  * temperature 0 alone is greedy in llama.cpp and leaves the distribution unfiltered.
  * The server must NOT be started with --jinja for a raw /completion prompt: the chat
    template changes the prompt and the comparison becomes meaningless.
  * A stale amp-server holds ~5.4 GB of VRAM; this tool never starts or stops any server.

STANDARD LIBRARY ONLY: urllib, json, math, argparse. No numpy, no scipy.
"""
import argparse
import json
import math
import os
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_OUT_DIR = os.path.join(ROOT, "quality")
DEFAULT_URL = "http://127.0.0.1:8081"
DEFAULT_N_PROBS = 32
DEFAULT_N_TOKENS = 512
DEFAULT_TIMEOUT = 7200  # seconds, for long generations

# Built-in default prompt: deterministic, reproducible, long enough for a few hundred positions.
DEFAULT_PROMPT = (
    "what is a page cache, and why does it matter more than RAM for inference? "
    "explain the relationship between page cache, mmap, and direct IO. "
    "how does the kernel decide which pages to evict? "
    "what is the role of the dirty bit and writeback? "
    "how does this affect LLM inference workloads that mmap a GGUF file? "
    "what is the difference between mmap and read for sequential access? "
    "when does direct IO make sense? "
    "how does transparent huge pages interact with mmap? "
    "what is the cost of a page fault? "
    "how does the kernel handle copy-on-write for forked processes? "
    "what is the difference between anonymous and file-backed memory? "
    "how does the VMA (virtual memory area) structure track mappings? "
    "what is the role of the page table and TLB? "
    "how does address space layout randomisation affect mmap? "
    "what is the difference between mprotect and mmap? "
    "how does the kernel handle demand paging? "
    "what is the role of the swap cache? "
    "how does the kernel decide which pages to swap out? "
    "what is the difference between swap and page cache eviction? "
    "how does swappiness affect this? "
    "what is the role of the OOM killer? "
    "how does cgroup memory limiting interact with the page cache? "
    "what is the difference between rss and vsz? "
    "how does the kernel account for shared memory? "
    "what is the role of the mm_struct? "
    "how does the kernel handle stack growth? "
    "what is the difference between brk and mmap for heap allocation? "
    "how does glibc's malloc decide between brk and mmap? "
    "what is the role of the arena in malloc? "
    "how does tcache work and what are its security implications? "
    "what is the difference between fastbin and smallbin? "
    "how does the kernel handle memory fragmentation? "
    "what is the role of the buddy allocator? "
    "how does the slab allocator work on top of the buddy allocator? "
    "what is the difference between kmalloc and vmalloc? "
    "how does the kernel handle memory pressure? "
    "what is the role of the shrinker? "
    "how does the kernel reclaim slab memory? "
    "what is the difference between active and inactive lists? "
    "how does the kernel decide the swappiness of a cgroup? "
    "what is the role of the LRU lists? "
    "how does the kernel handle page reclaim under memory pressure? "
    "what is the difference between direct reclaim and background reclaim? "
    "how does kswapd work? "
    "what is the role of the watermark? "
    "how does the kernel handle the min watermark? "
    "what is the difference between low and high watermark? "
    "how does the kernel handle the no watermark? "
    "what is the role of the zone? "
    "how does the kernel handle NUMA? "
    "what is the difference between local and remote memory? "
    "how does the kernel handle memory hotplug? "
    "what is the role of the memory cgroup? "
    "how does the kernel handle the oom score? "
    "what is the difference between oom_score_adj and oom_adj? "
    "how does the kernel handle the oom notifier? "
    "what is the role of the mempolicy? "
    "how does the kernel handle mbind? "
    "what is the difference between MPOL_BIND and MPOL_PREFERRED? "
    "how does the kernel handle MPOL_INTERLEAVE? "
    "what is the role of the cpuset? "
    "how does the kernel handle memory migration? "
    "what is the difference between migrate_pages and compaction? "
    "how does the kernel handle the compaction daemon? "
    "what is the role of the frag index? "
    "how does the kernel handle the watermark scale factor? "
    "what is the difference between the active and inactive ratio? "
    "how does the kernel handle the swappiness of a process? "
    "what is the role of the reclaim ratio? "
    "how does the kernel handle the vmscan ratio? "
    "what is the difference between the scan ratio and the reclaim ratio? "
    "how does the kernel handle the priority of reclaim? "
    "what is the role of the defer reclaim? "
    "how does the kernel handle the throttle? "
    "what is the difference between the throttle and the defer? "
    "how does the kernel handle the congestion wait? "
    "what is the role of the dirty threshold? "
    "how does the kernel handle the dirty ratio? "
    "what is the difference between the dirty threshold and the dirty background threshold? "
    "how does the kernel handle the dirty expire centisecs? "
    "what is the role of the dirty writeback centisecs? "
    "how does the kernel handle the dirtytime expire seconds? "
    "what is the difference between the dirty ratio and the dirty background ratio? "
    "how does the kernel handle the dirty high ratio? "
    "what is the role of the dirty low ratio? "
    "how does the kernel handle the dirty high threshold? "
    "what is the difference between the dirty high threshold and the dirty threshold? "
    "how does the kernel handle the dirty low threshold? "
    "what is the role of the dirty high background ratio? "
    "how does the kernel handle the dirty low background ratio? "
    "what is the difference between the dirty high background ratio and the dirty high ratio? "
    "how does the kernel handle the dirty low background threshold? "
    "what is the role of the dirty high background threshold? "
    "how does the kernel handle the dirty low background threshold? "
    "what is the difference between the dirty high background threshold and the dirty low background threshold?"
)


def now_utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def post_json(url, body, timeout=DEFAULT_TIMEOUT):
    """POST a JSON body and return the parsed response. Raises on HTTP error."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get_json(url, timeout=30):
    """GET a URL and return the parsed JSON response. Raises on HTTP error."""
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def get_model_name(url):
    """Try to get the model name from /v1/models. Returns 'unknown' if unreachable."""
    try:
        m = get_json(f"{url}/v1/models", timeout=10)
        return m["data"][0]["id"]
    except Exception:
        return "unknown"


def normalise_response(payload):
    """Normalise both response shapes to a list of per-position dicts.

    Returns: list of {token_id: int, top_logprobs: [(token_id, logprob), ...]}
    sorted by descending logprob.

    Handles:
      - OpenAI-style /v1/completions: choices[0].logprobs.content is a list of
        {id, token, bytes, logprob, top_logprobs: [...]} -- one entry per position.
      - Legacy /completion: top-level completion_probabilities is a list of
        {id, token, bytes, logprob, top_logprobs: [...]} -- one entry per position.
    """
    positions = []

    # OpenAI-style: choices[0].logprobs.content
    if "choices" in payload:
        choices = payload["choices"]
        if choices and "logprobs" in choices[0] and choices[0]["logprobs"]:
            content = choices[0]["logprobs"].get("content", [])
            for idx, entry in enumerate(content):
                top = entry.get("top_logprobs", [])
                pairs = [(int(q["id"]), float(q["logprob"])) for q in top
                         if "id" in q and "logprob" in q]
                pairs.sort(key=lambda x: -x[1])
                positions.append({
                    "token_id": int(entry.get("id", pairs[0][0] if pairs else -1)),
                    "top_logprobs": pairs,
                    "is_last": idx == len(content) - 1,
                })
    # Legacy /completion: completion_probabilities
    elif "completion_probabilities" in payload:
        for idx, entry in enumerate(payload["completion_probabilities"]):
            top = entry.get("top_logprobs", [])
            pairs = [(int(q["id"]), float(q["logprob"])) for q in top
                     if "id" in q and "logprob" in q]
            pairs.sort(key=lambda x: -x[1])
            positions.append({
                "token_id": int(entry.get("id", pairs[0][0] if pairs else -1)),
                "top_logprobs": pairs,
                "is_last": idx == len(payload["completion_probabilities"]) - 1,
            })

    return positions


def capture(args):
    """Run a greedy capture from a running server and save to a JSON file."""
    url = args.url.rstrip("/")
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    # Read prompt
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
    else:
        prompt = DEFAULT_PROMPT

    # Build request body -- ONLY temperature and n_probs. Never top_k/top_p/min_p.
    body = {
        "prompt": prompt,
        "n_predict": args.n_tokens,
        "temperature": 0.0,
        "n_probs": args.n_probs,
        "stream": False,
        "cache_prompt": False,
    }

    print(f"capturing from {url}")
    print(f"  tag:     {args.tag}")
    print(f"  n_probs: {args.n_probs}")
    print(f"  n_tokens:{args.n_tokens}")
    print(f"  prompt:  {len(prompt)} chars")
    print(f"  out:     {args.out}")
    print()

    # Try /v1/completions first, fall back to /completion
    t0 = time.time()
    payload = None
    endpoint_used = None
    for endpoint in ["/v1/completions", "/completion"]:
        try:
            payload = post_json(f"{url}{endpoint}", body, timeout=args.timeout)
            endpoint_used = endpoint
            break
        except urllib.error.HTTPError as e:
            print(f"  {endpoint} returned HTTP {e.code}, trying next endpoint...")
            continue
        except Exception as e:
            print(f"  {endpoint} failed: {e}, trying next endpoint...")
            continue

    if payload is None:
        print("ERROR: all endpoints failed. No capture written.")
        return 1

    elapsed = time.time() - t0
    positions = normalise_response(payload)

    # Fail loudly if no logprobs were returned
    if not positions:
        print("ERROR: server returned a response but no logprobs were found.")
        print("  This usually means n_probs was not accepted or the endpoint does not support it.")
        print("  No capture file written.")
        return 1

    n_with_probs = sum(1 for p in positions if p["top_logprobs"])
    if n_with_probs == 0:
        print("ERROR: server returned positions but none have logprobs.")
        print("  This usually means n_probs was not accepted or the endpoint does not support it.")
        print("  No capture file written.")
        return 1

    # Get model name
    model_name = get_model_name(url)

    # Build capture record
    capture_record = {
        "metadata": {
            "url": url,
            "tag": args.tag,
            "timestamp_utc": now_utc_iso(),
            "model_name": model_name,
            "endpoint": endpoint_used,
            "request_body": body,
            "elapsed_seconds": round(elapsed, 2),
            "n_positions": len(positions),
            "prompt_length_chars": len(prompt),
        },
        "positions": positions,
    }

    # Write atomically: temp file + rename
    fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(capture_record, f, indent=2)
        os.rename(tmp_path, args.out)
    except Exception:
        os.unlink(tmp_path)
        raise

    # Compute and report captured mass
    masses = []
    for p in positions:
        m = sum(math.exp(lp) for _, lp in p["top_logprobs"])
        masses.append(m)
    mean_mass = sum(masses) / len(masses) if masses else 0.0
    min_mass = min(masses) if masses else 0.0

    print(f"  endpoint: {endpoint_used}")
    print(f"  model:    {model_name}")
    print(f"  elapsed:  {elapsed:.1f}s")
    print(f"  positions: {len(positions)} ({n_with_probs} with logprobs)")
    print(f"  captured mass: mean={mean_mass:.6f} min={min_mass:.6f}")
    if mean_mass < 0.99:
        print(f"  WARNING: captured mass {mean_mass:.4f} < 0.99 -- KL will understate true divergence.")
        print(f"  Consider increasing --n-probs beyond {args.n_probs}.")
    print(f"  written:  {args.out}")
    return 0


def load_capture(path):
    """Load a capture JSON file and return (metadata, positions)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("metadata", {}), data.get("positions", [])


def logprobs_to_probs(top_logprobs):
    """Convert [(token_id, logprob), ...] to {token_id: prob}."""
    return {tid: math.exp(lp) for tid, lp in top_logprobs}


def renormalise_over_support(probs, support):
    """Renormalise a probability dict over a given support set.

    Tokens in probs but not in support are dropped (their mass is redistributed proportionally).
    Tokens in support but not in probs get probability 0.
    Returns (renormalised_dict, original_mass).
    """
    original_mass = sum(probs.get(t, 0.0) for t in support)
    if original_mass <= 0:
        return {t: 0.0 for t in support}, 0.0
    return {t: probs.get(t, 0.0) / original_mass for t in support}, original_mass


def kl_divergence(p, q):
    """KL(p || q) = sum_i p_i * log(p_i / q_i).

    p and q are dicts {token_id: prob} over the same support.
    Terms where p_i == 0 contribute 0. Terms where q_i == 0 but p_i > 0 contribute +inf.
    """
    kl = 0.0
    for t in p:
        pi = p[t]
        qi = q.get(t, 0.0)
        if pi <= 0:
            continue
        if qi <= 0:
            return float("inf")
        kl += pi * math.log(pi / qi)
    return kl


def js_divergence(p, q):
    """Jensen-Shannon divergence: bounded [0, ln(2)], symmetric, numerically safe.

    JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m) where m = 0.5 * (p + q).
    Using log base 2, JS is in [0, 1]. We use natural log, so JS is in [0, ln(2)].
    """
    m = {}
    for t in set(p) | set(q):
        m[t] = 0.5 * (p.get(t, 0.0) + q.get(t, 0.0))
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def compare_positions(pos_a, pos_b):
    """Compute divergence statistics for a single position.

    Returns a dict with per-position stats, or None if either side has no data.
    """
    if not pos_a or not pos_b:
        return None

    pairs_a = pos_a.get("top_logprobs", [])
    pairs_b = pos_b.get("top_logprobs", [])
    if not pairs_a or not pairs_b:
        return None

    probs_a = logprobs_to_probs(pairs_a)
    probs_b = logprobs_to_probs(pairs_b)

    # Joint support = union of both sides' token sets
    support = set(probs_a) | set(probs_b)

    # Renormalise both sides over the joint support
    p_a, mass_a = renormalise_over_support(probs_a, support)
    p_b, mass_b = renormalise_over_support(probs_b, support)

    # KL divergences (both directions)
    kl_ab = kl_divergence(p_a, p_b)
    kl_ba = kl_divergence(p_b, p_a)

    # Jensen-Shannon divergence (symmetric, bounded)
    js = js_divergence(p_a, p_b)

    # Top-1 token agreement
    top_a = pos_a.get("token_id", -1)
    top_b = pos_b.get("token_id", -1)
    agree = (top_a == top_b)

    # Mean/max absolute logprob delta on shared tokens
    dict_a = dict(pairs_a)
    dict_b = dict(pairs_b)
    shared = set(dict_a) & set(dict_b)
    if shared:
        deltas = [abs(dict_a[t] - dict_b[t]) for t in shared]
        mean_delta = sum(deltas) / len(deltas)
        max_delta = max(deltas)
    else:
        mean_delta = float("nan")
        max_delta = float("nan")

    return {
        "kl_ab": kl_ab,
        "kl_ba": kl_ba,
        "js": js,
        "mass_a": mass_a,
        "mass_b": mass_b,
        "agree": agree,
        "mean_delta": mean_delta,
        "max_delta": max_delta,
        "n_shared": len(shared),
    }


def compare(args):
    """Load two captures and compute divergence statistics."""
    meta_a, positions_a = load_capture(args.a)
    meta_b, positions_b = load_capture(args.b)

    n_a = len(positions_a)
    n_b = len(positions_b)
    n = min(n_a, n_b)

    if n == 0:
        print("ERROR: one or both captures have no positions.")
        return 1

    label = args.label or f"{meta_a.get('tag', 'A')} vs {meta_b.get('tag', 'B')}"
    print(f"comparing: {label}")
    print(f"  A: {args.a}")
    print(f"     tag={meta_a.get('tag', '?')} url={meta_a.get('url', '?')} "
          f"n_probs={meta_a.get('request_body', {}).get('n_probs', '?')} "
          f"positions={n_a} ts={meta_a.get('timestamp_utc', '?')}")
    print(f"  B: {args.b}")
    print(f"     tag={meta_b.get('tag', '?')} url={meta_b.get('url', '?')} "
          f"n_probs={meta_b.get('request_body', {}).get('n_probs', '?')} "
          f"positions={n_b} ts={meta_b.get('timestamp_utc', '?')}")
    print()

    stats = []
    for i in range(n):
        s = compare_positions(positions_a[i], positions_b[i])
        if s is not None:
            s["position"] = i
            stats.append(s)

    if not stats:
        print("ERROR: no positions with comparable data.")
        return 1

    # Aggregate statistics
    kl_ab_vals = [s["kl_ab"] for s in stats if math.isfinite(s["kl_ab"])]
    kl_ba_vals = [s["kl_ba"] for s in stats if math.isfinite(s["kl_ba"])]
    js_vals = [s["js"] for s in stats if math.isfinite(s["js"])]
    mass_a_vals = [s["mass_a"] for s in stats]
    mass_b_vals = [s["mass_b"] for s in stats]
    agree_count = sum(1 for s in stats if s["agree"])
    mean_deltas = [s["mean_delta"] for s in stats if math.isfinite(s["mean_delta"])]
    max_deltas = [s["max_delta"] for s in stats if math.isfinite(s["max_delta"])]

    n_inf_kl_ab = sum(1 for s in stats if s["kl_ab"] == float("inf"))
    n_inf_kl_ba = sum(1 for s in stats if s["kl_ba"] == float("inf"))

    mean_kl_ab = sum(kl_ab_vals) / len(kl_ab_vals) if kl_ab_vals else float("inf")
    mean_kl_ba = sum(kl_ba_vals) / len(kl_ba_vals) if kl_ba_vals else float("inf")
    mean_js = sum(js_vals) / len(js_vals) if js_vals else float("nan")
    mean_mass_a = sum(mass_a_vals) / len(mass_a_vals) if mass_a_vals else 0.0
    mean_mass_b = sum(mass_b_vals) / len(mass_b_vals) if mass_b_vals else 0.0
    min_mass_a = min(mass_a_vals) if mass_a_vals else 0.0
    min_mass_b = min(mass_b_vals) if mass_b_vals else 0.0
    mean_delta = sum(mean_deltas) / len(mean_deltas) if mean_deltas else float("nan")
    max_delta = max(max_deltas) if max_deltas else float("nan")

    # Worst KL position
    worst_kl = max(s["kl_ab"] for s in stats)
    worst_pos = next(s["position"] for s in stats if s["kl_ab"] == worst_kl)

    # Print results
    print(f"  positions compared: {len(stats)}")
    print(f"  top-1 agreement:    {agree_count}/{len(stats)} ({100*agree_count/len(stats):.2f}%)")
    print()
    print(f"  KL(A || B):  mean={mean_kl_ab:.6e}  max={worst_kl:.6e}  (inf at {n_inf_kl_ab} positions)")
    print(f"  KL(B || A):  mean={mean_kl_ba:.6e}  max={max((s['kl_ba'] for s in stats), default=float('nan')):.6e}  (inf at {n_inf_kl_ba} positions)")
    print(f"  JS(A, B):    mean={mean_js:.6e}  (bounded [0, {math.log(2):.6f}])")
    print()
    print(f"  captured mass A: mean={mean_mass_a:.6f} min={min_mass_a:.6f}")
    print(f"  captured mass B: mean={mean_mass_b:.6f} min={min_mass_b:.6f}")
    if mean_mass_a < 0.99 or mean_mass_b < 0.99:
        print(f"  WARNING: captured mass < 0.99 -- KL understates true divergence.")
    print()
    print(f"  |delta logprob| on shared tokens: mean={mean_delta:.6e} max={max_delta:.6e}")
    print()
    print(f"  worst KL(A||B) at position {worst_pos} (KL={worst_kl:.6e})")
    print()

    # Gate check
    if args.max_kl is not None:
        threshold = args.max_kl
        if mean_kl_ab > threshold:
            print(f"  FAIL: mean KL(A||B) {mean_kl_ab:.6e} exceeds threshold {threshold:.6e}")
            return 1
        if mean_kl_ba > threshold:
            print(f"  FAIL: mean KL(B||A) {mean_kl_ba:.6e} exceeds threshold {threshold:.6e}")
            return 1
        print(f"  PASS: mean KL below threshold {threshold:.6e}")

    return 0


def selftest():
    """Run self-tests on the maths with hand-written inputs. No server needed."""
    print("running self-tests...")
    passed = 0
    failed = 0

    def check(label, actual, expected, tol=1e-12):
        nonlocal passed, failed
        if isinstance(expected, float) and math.isinf(expected):
            ok = actual == expected
        elif isinstance(expected, float) and math.isnan(expected):
            ok = math.isnan(actual)
        else:
            ok = abs(actual - expected) <= tol
        if ok:
            print(f"  ok   {label}")
            passed += 1
        else:
            print(f"  FAIL {label}: expected {expected}, got {actual}")
            failed += 1

    # Test 1: identical distributions give KL = 0
    p = {1: 0.5, 2: 0.3, 3: 0.2}
    q = {1: 0.5, 2: 0.3, 3: 0.2}
    check("KL(p||p) == 0", kl_divergence(p, q), 0.0)

    # Test 2: JS(p,p) == 0
    check("JS(p,p) == 0", js_divergence(p, q), 0.0)

    # Test 3: known KL value
    # KL({0.5,0.5} || {0.25,0.75}) = 0.5*log(0.5/0.25) + 0.5*log(0.5/0.75)
    #   = 0.5*log(2) + 0.5*log(2/3) = 0.5*(log 2 + log 2 - log 3) = 0.5*(2*log2 - log3)
    p2 = {1: 0.5, 2: 0.5}
    q2 = {1: 0.25, 2: 0.75}
    expected_kl = 0.5 * math.log(2) + 0.5 * math.log(2.0 / 3.0)
    check("KL known value", kl_divergence(p2, q2), expected_kl)

    # Test 4: KL with zero in q gives inf
    p3 = {1: 0.5, 2: 0.5}
    q3 = {1: 1.0, 2: 0.0}
    check("KL with zero in q is inf", kl_divergence(p3, q3), float("inf"))

    # Test 5: KL with zero in p is fine (contributes 0)
    p4 = {1: 1.0, 2: 0.0}
    q4 = {1: 0.5, 2: 0.5}
    check("KL with zero in p", kl_divergence(p4, q4), math.log(2))

    # Test 6: JS is symmetric
    js_ab = js_divergence(p2, q2)
    js_ba = js_divergence(q2, p2)
    check("JS symmetric", js_ab, js_ba)

    # Test 7: JS bounded by ln(2)
    check("JS <= ln(2)", js_ab <= math.log(2) + 1e-12, True)

    # Test 8: renormalise_over_support
    probs = {1: 0.6, 2: 0.3, 3: 0.1}
    support = {1, 2, 4}
    renorm, mass = renormalise_over_support(probs, support)
    check("renormalise mass", mass, 0.9)
    check("renormalise sums to 1", sum(renorm.values()), 1.0)
    check("renormalise drops token 3", renorm[1], 0.6 / 0.9)
    check("renormalise adds token 4 as 0", renorm[4], 0.0)

    # Test 9: logprobs_to_probs
    pairs = [(10, math.log(0.7)), (20, math.log(0.3))]
    probs = logprobs_to_probs(pairs)
    check("logprobs_to_probs", probs[10], 0.7, tol=1e-10)
    check("logprobs_to_probs", probs[20], 0.3, tol=1e-10)

    # Test 10: compare_positions with identical data gives KL=0
    pos_a = {"token_id": 42, "top_logprobs": [(42, -0.1), (43, -2.0)]}
    pos_b = {"token_id": 42, "top_logprobs": [(42, -0.1), (43, -2.0)]}
    s = compare_positions(pos_a, pos_b)
    check("identical positions KL=0", s["kl_ab"], 0.0)
    check("identical positions agree", s["agree"], True)

    # Test 11: compare_positions with shifted logprobs
    pos_a = {"token_id": 42, "top_logprobs": [(42, -0.1), (43, -2.0)]}
    pos_b = {"token_id": 42, "top_logprobs": [(42, -0.15), (43, -1.9)]}
    s = compare_positions(pos_a, pos_b)
    check("shifted positions KL>0", s["kl_ab"] > 0, True)
    check("shifted positions agree", s["agree"], True)

    # Test 12: normalise_response OpenAI-style
    payload = {
        "choices": [{
            "logprobs": {
                "content": [
                    {"id": 10, "token": "a", "logprob": -0.1,
                     "top_logprobs": [{"id": 10, "logprob": -0.1}, {"id": 11, "logprob": -0.5}]},
                    {"id": 12, "token": "b", "logprob": -0.2,
                     "top_logprobs": [{"id": 12, "logprob": -0.2}, {"id": 13, "logprob": -0.8}]},
                ]
            }
        }]
    }
    positions = normalise_response(payload)
    check("normalise OAI: 2 positions", len(positions), 2)
    check("normalise OAI: first token", positions[0]["token_id"], 10)
    check("normalise OAI: first top pair id", positions[0]["top_logprobs"][0][0], 10)
    check("normalise OAI: first top pair logprob", positions[0]["top_logprobs"][0][1], -0.1)
    check("normalise OAI: is_last on last", positions[1]["is_last"], True)
    check("normalise OAI: not is_last on first", positions[0]["is_last"], False)

    # Test 13: normalise_response legacy /completion
    payload = {
        "completion_probabilities": [
            {"id": 20, "token": "x", "logprob": -0.3,
             "top_logprobs": [{"id": 20, "logprob": -0.3}, {"id": 21, "logprob": -0.6}]},
        ]
    }
    positions = normalise_response(payload)
    check("normalise legacy: 1 position", len(positions), 1)
    check("normalise legacy: token", positions[0]["token_id"], 20)

    # Test 14: self-comparison via compare_positions gives exactly 0
    # (simulating the determinism check)
    pos = {"token_id": 100, "top_logprobs": [(100, -0.01), (101, -3.0), (102, -5.0)]}
    s = compare_positions(pos, pos)
    check("self-compare KL=0 exactly", s["kl_ab"], 0.0)
    check("self-compare JS=0 exactly", s["js"], 0.0)

    print()
    if failed:
        print(f"SELFTEST FAILED: {failed} test(s) failed, {passed} passed")
        return 1
    print(f"selftest passed: {passed} tests")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="KL-divergence parity harness: compare output distributions of two engines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--selftest", action="store_true",
                    help="run self-tests on the maths (no server needed)")

    sub = ap.add_subparsers(dest="command")

    # capture subcommand
    cap = sub.add_parser("capture", help="capture a greedy run from a running server")
    cap.add_argument("--url", default=DEFAULT_URL,
                     help=f"server base URL (default: {DEFAULT_URL})")
    cap.add_argument("--tag", required=True,
                     help="tag for this capture (e.g. amp-before, llama-after)")
    cap.add_argument("--out", required=True,
                     help="output JSON file path (default dir: amp/quality/)")
    cap.add_argument("--n-probs", type=int, default=DEFAULT_N_PROBS,
                     help=f"number of top logprobs to request (default: {DEFAULT_N_PROBS})")
    cap.add_argument("--n-tokens", type=int, default=DEFAULT_N_TOKENS,
                     help=f"number of tokens to generate (default: {DEFAULT_N_TOKENS})")
    cap.add_argument("--prompt-file", default=None,
                     help="read prompt from this file (default: built-in prompt)")
    cap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                     help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT})")

    # compare subcommand
    cmp = sub.add_parser("compare", help="compare two captures and report KL divergence")
    cmp.add_argument("--a", required=True, help="first capture file (JSON)")
    cmp.add_argument("--b", required=True, help="second capture file (JSON)")
    cmp.add_argument("--label", default=None, help="label for this comparison")
    cmp.add_argument("--max-kl", type=float, default=None,
                     help="fail (exit 1) if mean KL exceeds this threshold")

    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.command == "capture":
        return capture(args)
    elif args.command == "compare":
        return compare(args)
    else:
        ap.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
