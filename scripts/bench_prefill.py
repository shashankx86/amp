#!/usr/bin/env python3
"""Measure prefill throughput as a sequence, because a single prefill number on this box is noise.

    python3 scripts/bench_prefill.py --tokens 18265 --n 5
    python3 scripts/bench_prefill.py --tokens 35000 --n 5 --tag q8q8

Why this exists
---------------

Prefill throughput on this machine has an enormous run-to-run spread. The same server
configuration, restarted three times, measured **132.5, 178.4 and 311.3 tokens/s** on an
identical 18,265-token prompt, while decode over the same runs varied only 26.4 to 27.7.

The cause is the working set, and it is worth stating because it changes how prefill must be
measured. Prefill runs at batch ~1024, which routes most or all of the 256 experts per layer
through the CPU matmul, so it reads essentially the whole 13.66 GiB of weights and thrashes a
page cache that holds about 11 GiB of it. Decode at batch 1 touches only 8 experts per layer, a
small enough set to stay resident, which is why decode is stable and prefill is not. Any
prefill figure taken from one request is therefore a sample of a distribution that spans more
than 2x, and the project's own rule already says to report a sequence and never a single
request. This applies to prefill harder than to anything else measured here.

So: every prompt is distinct, generated with per-prompt unique content, so that no measurement
can be served from the server's prompt cache and each one pays a real prefill. The median is
the number to quote, and the min and max are reported beside it because that range is the
honest uncertainty on any prefill claim this project makes.

This script starts nothing and stops nothing.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

WORDS = (
    "planner override resident streaming expert layer context batch ubatch decode prefill "
    "kernel bandwidth latency quantisation checkpoint slot router recurrence attention cache "
    "resident threshold measured benchmark variance throughput placement arithmetic"
).split()


def post(url, body, timeout):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize(url, text, timeout):
    req = urllib.request.Request(f"{url}/tokenize", data=json.dumps({"content": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return len(json.loads(r.read())["tokens"])


def build(url, seed_text, target_tokens, unique, timeout, blake=0):
    """Grow a prompt to ~target_tokens, with content unique to this prompt.

    Uniqueness matters twice over. It stops the server's prompt cache from serving the
    measurement, and it stops any de-duplication in the tokenizer from quietly returning a
    shorter stream than the one being timed.
    """
    # A per-prompt marker at both ends, derived from the index, guarantees the token stream
    # differs even if the filler repeats.
    text = f"### record {blake} {unique} begin\n" + seed_text
    n = tokenize(url, text, timeout)
    i = 0
    while n < target_tokens:
        block = "\n".join(
            f"item {blake}.{i}.{j}: {WORDS[(i * 7 + j * 3 + blake) % len(WORDS)]} "
            f"{WORDS[(i * 11 + j * 5 + blake * 2) % len(WORDS)]} {j}"
            for j in range(200))
        text += f"\n\n## part {blake}.{i}\n" + block
        n = tokenize(url, text, timeout)
        i += 1
        if i > 200:
            break
    return text + f"\n### record {blake} end\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--tag", default="prefill")
    ap.add_argument("--n", type=int, default=5, help="distinct prompts to prefill (>=5 to see the spread)")
    ap.add_argument("--tokens", type=int, default=18265, help="approximate prompt size")
    ap.add_argument("--seed-file", default="/tmp/opencode/amp_bench_prompt.txt")
    ap.add_argument("--gen", type=int, default=8, help="tokens to generate after (small on purpose)")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    with open(args.seed_file, encoding="utf-8") as f:
        seed = f.read()

    print(f"tag     : {args.tag}")
    print(f"plan    : {args.n} distinct ~{args.tokens}-token prompts, one server process, "
          f"{args.gen} tokens generated after each")
    print()

    rates, rows = [], []
    for i in range(args.n):
        # Different filler per prompt, so nothing can be served from the prompt cache.
        prompt = build(args.url, seed, args.tokens, f"variant{i}", args.timeout, blake=i * 101)
        body = {"prompt": prompt, "n_predict": args.gen, "temperature": 0.0, "stream": False}
        t0 = time.time()
        try:
            r = post(f"{args.url}/v1/completions", body, args.timeout)
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"  prompt {i+1} failed: {e}", file=sys.stderr)
            return 1
        wall = time.time() - t0
        t = r.get("timings") or {}
        pn = t.get("prompt_n") or 0
        pp = t.get("prompt_per_second") or 0.0
        if pp:
            rates.append(pp)
        rows.append({"i": i, "prompt_n": pn, "prefill_tps": pp, "wall_s": wall,
                     "decode_tps": t.get("predicted_per_second")})
        print(f"  prompt {i+1}/{args.n}: {pn:>6} tok prefill at {pp:7.1f} t/s | wall {wall:6.1f}s")

    if not rates:
        print("no prefill rates captured", file=sys.stderr)
        return 1

    print()
    print("summary")
    print(f"  prefill median  {statistics.median(rates):7.1f} t/s   <- quote this one")
    print(f"  prefill min     {min(rates):7.1f} t/s")
    print(f"  prefill max     {max(rates):7.1f} t/s")
    print(f"  spread          {max(rates) / min(rates):7.2f}x   <- the honest uncertainty")
    # Only report decode if it was generated in enough tokens to mean anything. A handful of
    # tokens rounds to 0.0 and printing that would be a misleading line in a report.
    dec = [x["decode_tps"] for x in rows
           if isinstance(x.get("decode_tps"), (int, float)) and x["decode_tps"] > 0]
    if dec and args.gen >= 64:
        print(f"  decode median   {statistics.median(dec):7.1f} t/s   (much more stable)")
    elif dec:
        print(f"  (decode not reported: only --gen {args.gen} tokens, too few to be meaningful. "
              f"Use scripts/bench_server.py for decode.)")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"tag": args.tag, "target_tokens": args.tokens, "rows": rows,
                       "median": statistics.median(rates), "min": min(rates), "max": max(rates)}, f, indent=2)
        print(f"\n  written: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
