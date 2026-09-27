#!/usr/bin/env python3
"""Measure what an agentic loop actually feels like: per-turn latency, not tokens/second.

    python3 scripts/bench_agentic.py --turns 8
    python3 scripts/bench_agentic.py --turns 8 --tag cram-2048 -cram 2048

Why a single request cannot measure this
----------------------------------------

Decode throughput is the wrong number for agentic use. A tool-using client does not
send one long prompt; it sends a large context once and then, on every subsequent
turn, sends that same context plus a small delta (a tool result, a file read). The
server is expected to reuse the cached prefix and prefill only the delta.

So the cost of a turn is dominated by *how much got re-prefilled*, and the pain is
the worst turn, not the median. A 35k-token re-prefill at 150 tokens/s is 234
seconds of the user watching a spinner; that is invisible in any tokens/second
figure and is the entire difference between usable and not.

This script therefore reports, per turn, how many prompt tokens were actually
processed against how many were new, and calls out any turn that re-prefilled the
whole context. Those are the stalls.

Design notes
------------

* **temperature 0** and no top_k/top_p, matching bench_server.py.
* **The delta is realistic**: a tool result of a few hundred tokens, which is what a
  `read` or `glob` returns. Large deltas would flatter the cache.
* **The base context comes from the standard bench prompt** so the size matches the
  rest of the project's measurements rather than being tuned to suit the result.
* Reports the max turn prominently, because that is the number a user feels.
* This script starts nothing and stops nothing. Two servers on this box fight over
  6 GB of VRAM and make every number meaningless.
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

# A plausible tool result: a short report plus a couple of file excerpts. This is
# the size of delta a real `read`/`glob`/`shell` turn contributes.
DELTA = """
Tool: read
Result: src/plan/preflight.cpp
  1: // amp's whole contribution to the server: turn MemoryPlanner into common_params
  2: // before the model is opened. Nothing here changes llama.cpp's arithmetic.
  3: #include "preflight.h"
  4:
  5: #include "memory_plan.h"
  6: #include "gguf_info.h"
  7:
  8: #include <cstdio>
  9: #include <string>
 10:
 11: namespace amp {
 12:
 13: // Writes overrides in place into the buffer common_params_parse already padded to
 14: // 4096 bytes. push_back past the sentinel aborts in common.cpp, so never grow argv.
 15: static void push_flag(std::vector<std::string> & argv, const std::string & flag) {
 16:     argv.push_back(flag);
 17: }
 18:
 19: } // namespace amp
"""


def post(url, body, timeout):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--tag", default="agentic")
    ap.add_argument("--turns", type=int, default=8, help="turns after the initial context")
    ap.add_argument("--gen", type=int, default=48, help="tokens generated per turn")
    ap.add_argument("--base-chars", type=int, default=0,
                    help="0 = use the whole standard bench prompt as the base context")
    ap.add_argument("--prompt-file", default="/tmp/opencode/amp_bench_prompt.txt")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--interleave", type=int, default=0,
                    help="after every N turns, send one short unrelated request, the way an "
                         "agent tests the API it is being served by")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    with open(args.prompt_file, encoding="utf-8") as f:
        prompt = f.read()
    if args.base_chars:
        prompt = prompt[:args.base_chars]

    print(f"tag     : {args.tag}")
    print(f"url     : {args.url}")
    print(f"plan    : 1 initial context + {args.turns} turns, each adding a ~{len(DELTA)}-char "
          f"tool result, {args.gen} tokens generated per turn")
    if args.interleave:
        print(f"          plus one short unrelated request every {args.interleave} turns, the way "
              f"an agent tests the API it is being served by")
    print()

    context = prompt
    rows = []
    stalls = []

    for turn in range(args.turns + 1):
        body = {"prompt": context, "n_predict": args.gen, "temperature": 0.0, "stream": False}
        t0 = time.time()
        try:
            r = post(f"{args.url}/v1/completions", body, args.timeout)
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"  turn {turn} failed: {e}", file=sys.stderr)
            return 1
        wall = time.time() - t0
        t = r.get("timings") or {}

        pn = t.get("prompt_n") or 0
        pp = t.get("prompt_per_second") or 0.0
        dn = t.get("predicted_n") or 0
        dp = t.get("predicted_per_second") or 0.0

        prefill_ms = 1000.0 * pn / pp if pp else 0.0
        decode_ms = 1000.0 * dn / dp if dp else 0.0

        kind = "initial"
        if turn > 0:
            # A turn that reprocessed roughly the whole context is a cache miss.
            kind = "FULL RE-PREFILL" if pn > len(prompt) * 0.25 else "delta"
            if kind == "FULL RE-PREFILL":
                stalls.append((turn, pn, wall))

        rows.append({"turn": turn, "prompt_n": pn, "prefill_ms": prefill_ms,
                     "decode_ms": decode_ms, "wall_s": wall, "kind": kind})

        label = f"turn {turn:>2}" if turn else "  init"
        print(f"  {label}: prompt {pn:>6} tok in {prefill_ms / 1000:6.1f}s "
              f"({pp:6.1f} t/s) | decode {dn:>3} tok {dp:6.1f} t/s | wall {wall:6.1f}s  {kind}")

        # The model replies, and the client's next turn appends the tool result.
        context = context + "\nassistant: ok\nuser:\n" + DELTA

        # An agent that tests its own API injects an unrelated short request into the same
        # server. On a single-slot server that request is served by the slot holding the agent's
        # context, and the context does not survive. This is the stall.
        if args.interleave and turn and turn % args.interleave == 0:
            probe = {"prompt": "Reply with the single word: pong",
                     "n_predict": 8, "temperature": 0.0, "stream": False}
            t1 = time.time()
            try:
                post(f"{args.url}/v1/completions", probe, args.timeout)
                pw = time.time() - t1
                rows.append({"turn": f"{turn}+probe", "prompt_n": 0, "prefill_ms": 0.0,
                             "decode_ms": pw * 1000.0, "wall_s": pw, "kind": "probe"})
                print(f"  probe  : unrelated 8-token request, wall {pw:.1f}s "
                      f"(sent by the agent to its own host)")
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                print(f"  probe failed: {e}", file=sys.stderr)
                return 1

    turn_rows = [r for r in rows[1:] if isinstance(r["turn"], int)]
    probes = [r for r in rows if r["kind"] == "probe"]
    walls = [r["wall_s"] for r in turn_rows]
    prefills = [r["prefill_ms"] for r in turn_rows]
    deltas = [r for r in turn_rows if r["kind"] == "delta"]
    delta_walls = [r["wall_s"] for r in deltas]

    print()
    print("summary")
    print(f"  initial context prefill : {rows[0]['prefill_ms'] / 1000:.1f}s "
          f"({rows[0]['prompt_n']} tokens)")
    if walls:
        print(f"  per-turn wall, median   : {statistics.median(walls):6.1f}s")
        print(f"  per-turn wall, MAX      : {max(walls):6.1f}s   <- what the user waits through")
        print(f"  per-turn wall, total    : {sum(walls):6.1f}s over {len(walls)} turns")
    if prefills:
        print(f"  per-turn prefill, median: {statistics.median(prefills) / 1000:6.2f}s")
    if delta_walls:
        print(f"  cache-hit turns only    : median {statistics.median(delta_walls):.1f}s "
              f"over {len(delta_walls)}/{len(turn_rows)} turns")
    if probes:
        print(f"  self-test requests       : {len(probes)}, "
              f"{sum(p['wall_s'] for p in probes):.1f}s total")
    print(f"  FULL RE-PREFILL stalls  : {len(stalls)}"
          + (f"  -> turns {[t for t, _, _ in stalls]}, "
             f"{max(w for _, _, w in stalls):.0f}s worst" if stalls else "  (none)"))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"tag": args.tag, "turns": args.turns, "gen": args.gen,
                       "base_chars": len(prompt), "rows": rows,
                       "stalls": stalls}, f, indent=2)
        print(f"\n  written: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
