#!/usr/bin/env python3
"""Benchmark a running OpenAI-compatible server: prefill and decode, as a sequence.

    python3 scripts/bench_server.py --url http://127.0.0.1:8081 --tag amp
    python3 scripts/bench_server.py --url http://127.0.0.1:8099 --tag llama --n 3

Design notes, all of them learned the hard way:

* **Report a sequence, never a single request.** The first request pays the cold page-cache
  fault for the expert set and is routinely 3x slower than steady state. Quoting request 1 is
  quoting a cold start. Every run here sends N identical requests and prints all of them.

* **Use the server's own `timings` object**, not wall-clock arithmetic. It reports prompt_n,
  prompt_per_second, predicted_n and predicted_per_second as measured inside the engine, so the
  number is not distorted by HTTP, JSON, or this script.

* **`temperature: 0`** and no top_k/top_p. The decode rate must not depend on which tokens come
  out, and a filtered distribution makes the server renormalise reported logprobs.

* **The same prompt for every engine being compared.** A different prompt is a different
  measurement. Pass --prompt-file explicitly for any comparison.

* This script starts nothing and stops nothing. Two servers on this box fight over 6 GB of VRAM
  and produce numbers that are worse than useless - check `pgrep` and `nvidia-smi` yourself.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request


def post(url, body, timeout):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--tag", default="server", help="label for the report")
    ap.add_argument("--n", type=int, default=3, help="identical requests to send (>=2 to see steady state)")
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--prompt-file", default="/tmp/opencode/amp_bench_prompt.txt")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.n < 2:
        print("warning: fewer than 2 requests cannot distinguish cold start from steady state",
              file=sys.stderr)

    try:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt = f.read()
    except OSError as e:
        print(f"cannot read prompt file {args.prompt_file}: {e}", file=sys.stderr)
        return 2

    print(f"tag    : {args.tag}")
    print(f"url    : {args.url}")
    print(f"prompt : {len(prompt)} chars from {args.prompt_file}")
    print(f"plan   : {args.n} identical requests, n_predict={args.n_predict}, temperature=0")
    print()

    rows = []
    for i in range(args.n):
        body = {"prompt": prompt, "n_predict": args.n_predict, "temperature": 0.0, "stream": False}
        t0 = time.time()
        try:
            r = post(f"{args.url}/v1/completions", body, args.timeout)
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"  request {i+1} failed: {e}", file=sys.stderr)
            return 1
        wall = time.time() - t0
        t = r.get("timings") or {}
        row = {
            "request": i + 1,
            "wall_s": round(wall, 2),
            "prompt_tokens": t.get("prompt_n"),
            "prefill_tps": t.get("prompt_per_second"),
            "generated": t.get("predicted_n"),
            "decode_tps": t.get("predicted_per_second"),
        }
        rows.append(row)
        print(f"  request {i+1}/{args.n}: prefill {row['prefill_tps']} t/s over "
              f"{row['prompt_tokens']} tok | decode {row['decode_tps']} t/s over "
              f"{row['generated']} tok | wall {row['wall_s']}s")

    def steady(key):
        # Drop the first request: it is the cold page-cache fault, not the engine's speed.
        vals = [r[key] for r in rows[1:] if isinstance(r.get(key), (int, float))]
        return vals

    pp, dd = steady("prefill_tps"), steady("decode_tps")
    print()
    if pp:
        print(f"  steady prefill (requests 2..{args.n}): median {statistics.median(pp):.2f} t/s")
    if dd:
        print(f"  steady decode  (requests 2..{args.n}): median {statistics.median(dd):.2f} t/s")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"tag": args.tag, "url": args.url, "prompt_file": args.prompt_file,
                       "n_predict": args.n_predict, "rows": rows}, f, indent=2)
        print(f"  written: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
