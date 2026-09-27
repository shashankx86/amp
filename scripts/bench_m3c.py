#!/usr/bin/env python3
"""A/B the M3c expert prefetch inside one server session.

    python3 scripts/bench_m3c.py --depths 0 16 64 --rounds 4

Why this shape
--------------

The single hardest-won fact about benchmarking on this box is that steady
decode varies by about 22 % between sessions with page-cache warmth. That makes
any A/B which starts a server per configuration meaningless: the difference
between the configurations is smaller than the difference between the sessions.

So both arms run inside **one** server process. `AMP_M3C_PREFETCH_FILE` points
at a file holding the prefetch depth in KiB, which the CPU backend re-reads at
most twice a second. Writing 0 turns the hint off; writing N turns it on with an
N KiB depth. Nothing else about the process changes, and the prefetch is a hint
that cannot alter results, so switching mid-session is safe by construction.

Arms are interleaved rather than run in blocks, and the order is reversed on
alternate rounds. Block ordering would let any monotonic drift in cache warmth
be charged to the prefetch.

The first request after each switch is discarded: it pays for the file re-read
landing and for any cache effects of the previous arm.

This script starts nothing and stops nothing. Check `pgrep` and `nvidia-smi`
yourself before trusting a number.
"""

import argparse
import json
import os
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


def set_depth(path, kib):
    # Write via a temp file and rename, so the backend never reads a half-written
    # integer.
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(f"{kib}\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 16],
                    help="prefetch depths in KiB to compare (0 = prefetch off)")
    ap.add_argument("--rounds", type=int, default=3, help="interleaved rounds per depth")
    ap.add_argument("--reqs", type=int, default=2, help="measured requests per arm visit")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--control-file", default="/tmp/opencode/m3c_depth")
    ap.add_argument("--settle", type=float, default=1.5,
                    help="seconds to wait after a switch, so the backend re-reads it")
    ap.add_argument("--prompt-file", default="/tmp/opencode/amp_bench_prompt.txt")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if len(args.depths) < 2:
        print("need at least two depths to compare", file=sys.stderr)
        return 2

    with open(args.prompt_file, encoding="utf-8") as f:
        prompt = f.read()

    set_depth(args.control_file, 0)
    body = {"prompt": prompt, "n_predict": args.n_predict, "temperature": 0.0, "stream": False}

    print(f"url     : {args.url}")
    print(f"control : {args.control_file} (depth in KiB, 0 = off)")
    print(f"depths  : {args.depths}")
    print(f"plan    : {args.rounds} rounds x {args.reqs} requests, "
          f"n_predict={args.n_predict}, arms interleaved and order-reversed")
    print()

    def visit(depth):
        set_depth(args.control_file, depth)
        time.sleep(args.settle)
        rates = []
        for i in range(args.reqs + 1):          # one discarded, see module docstring
            try:
                r = post(f"{args.url}/v1/completions", body, args.timeout)
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                print(f"  depth {depth} request {i+1} failed: {e}", file=sys.stderr)
                return None
            t = r.get("timings") or {}
            if i == 0:
                print(f"  depth {depth:>4} KiB  settle request: "
                      f"{t.get('predicted_per_second'):.2f} t/s (discarded)")
                continue
            rates.append(t.get("predicted_per_second"))
        med = statistics.median(rates)
        print(f"  depth {depth:>4} KiB  {len(rates)} requests: "
              f"{', '.join(f'{x:.2f}' for x in rates)} -> median {med:.2f} t/s")
        return med

    print("warming (pays prefill and settles the page cache)...")
    set_depth(args.control_file, 0)
    t0 = time.time()
    post(f"{args.url}/v1/completions", body, args.timeout)
    print(f"  warm done in {time.time() - t0:.1f}s")
    print()

    results = {d: [] for d in args.depths}
    for rnd in range(args.rounds):
        order = args.depths if rnd % 2 == 0 else list(reversed(args.depths))
        print(f"round {rnd + 1}/{args.rounds} (order {order}):")
        for d in order:
            med = visit(d)
            if med is None:
                return 1
            results[d].append(med)
        print()

    print("summary")
    for d in args.depths:
        vals = results[d]
        print(f"  depth {d:>4} KiB : median of rounds {statistics.median(vals):6.2f} t/s   "
              f"rounds [{', '.join(f'{v:.2f}' for v in vals)}]")

    base = args.depths[0]
    if base in results and results[base]:
        bmed = statistics.median(results[base])
        print()
        for d in args.depths:
            if d == base:
                continue
            dmed = statistics.median(results[d])
            print(f"  depth {d:>4} vs {base:>4}: {100 * (dmed - bmed) / bmed:+.1f} %")
        if len(args.depths) > 2:
            print()
            print("  (a difference smaller than a few percent is inside the run-to-run"
                  " spread of this box; see docs/BENCH.md)")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"depths": args.depths, "rounds": args.rounds,
                       "reqs": args.reqs, "n_predict": args.n_predict,
                       "results": {str(d): v for d, v in results.items()}}, f, indent=2)
        print(f"\n  written: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
