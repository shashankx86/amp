#!/usr/bin/env python3
"""Measure speculative decoding against nothing, alternating arms, as a sequence.

    python3 scripts/bench_spec.py --arms base,ngram-map-k --pairs 3 --n 6

WHY THIS IS A SEPARATE SCRIPT AND NOT A CALL TO bench_server.py

The thing being measured needs a different workload and a different control than throughput:

* **The workload is the point.** Strata measured prompt-lookup speculation at 6-11% on code
  edits and *unchanged* on ordinary prose, and it also measured the negative case - forcing the
  lookup drafter whenever it proposed more than the MTP lost 2-8% on ordinary text. So "is
  n-gram speculation faster" is not a single question, it is two: does it help where it should,
  and does it hurt where it should not. A benchmark that only runs one of them will happily
  report a win that is a loss in the other half of the workload.

* **Both arms need alternating sessions.** llama.cpp's per-request `speculative.type` override
  is compiled out (`tools/server/server-schema.cpp:197` is `#if 0`), so the arm cannot be
  switched inside one process. Steady decode on this box varies about 22% between sessions and
  drifted 30.5 -> 27.2 t/s inside a single 15-minute session, so one session per arm is a
  comparison of two noise draws. This alternates base/spec/base/spec/... and compares medians
  over pairs, which cancels a monotonic drift almost exactly and leaves the variance visible.

* **The first request of every session is dropped.** It pays the cold page-cache fault for the
  expert set and is routinely several times slower than steady state. It is not a measurement of
  anything except the disk.

* **It reports the engine's own `timings.predicted_per_second`**, not wall-clock, so HTTP and
  JSON are not in the number, and it records `predicted_n` so a run where the model simply
  stopped early cannot masquerade as a fast one.

* **Nothing is started or stopped here.** Two servers on this box fight over 6 GB of VRAM. The
  caller owns the process lifetime; see scripts/sweep_spec.sh.
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


def wait_ready(url, deadline_s):
    """Block until the server can actually serve, or give up. Returns seconds waited, or None.

    NOT /health. `/health` answers 200 as soon as the router is listening, which on this box is
    about two seconds - while the 13.66 GiB model is still being read off disk. A first version
    of this script used it and would happily have benchmarked an unloaded server, producing
    numbers that looked fine and meant nothing. The only honest readiness signal is a request
    that comes back with a real token in it.
    """
    probe = {"model": "amp", "messages": [{"role": "user", "content": "hi"}],
             "max_tokens": 1, "temperature": 0, "stream": False}
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        # A connect-refused is instant, but a port that accepts and then never answers would
        # otherwise sit in urlopen for the full 30 s probe timeout, so the deadline here is
        # deliberately short and a failure just means "not ready yet".
        try:
            r = post(url + "/v1/chat/completions", probe, 5)
            if r.get("choices"):
                return time.time() - t0
        except Exception:
            pass
        time.sleep(3)
    return None


def one_request(url, prompt, n_predict, timeout):
    body = {
        "model": "amp",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_predict,
        "temperature": 0,
        "stream": False,
        "cache_prompt": True,
    }
    t0 = time.time()
    r = post(url + "/v1/chat/completions", body, timeout)
    wall = time.time() - t0
    t = r.get("timings", {})
    return {
        "prompt_n": t.get("prompt_n"),
        "prompt_tps": t.get("prompt_per_second"),
        "n": t.get("predicted_n"),
        "tps": t.get("predicted_per_second"),
        # The engine's own draft accounting (tools/server/server-common.cpp puts draft_n and
        # draft_n_accepted in `timings`). Without these, "speculation did not help" and
        # "speculation never ran" are indistinguishable - and a sweep with a mis-set match
        # length produces exactly the second while looking like the first.
        "draft_n": t.get("draft_n", 0) or 0,
        "draft_accepted": t.get("draft_n_accepted", 0) or 0,
        "wall": wall,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--arms", default="base,spec",
                    help="comma-separated arm names; each must be a label the driver script knows")
    ap.add_argument("--pairs", type=int, default=3, help="how many times to visit each arm")
    ap.add_argument("--n", type=int, default=5, help="requests per session (first is dropped)")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--state", default="/tmp/opencode/spec_state.json",
                    help="written by the driver so this script knows which arm it is serving")
    ap.add_argument("--require", default="", help="comma-separated substrings that must ALL appear "
                    "in the server log before the first request; guards against measuring a "
                    "server that started with different flags than the driver intended")
    ap.add_argument("--forbid", default="", help="comma-separated substrings, NONE of which may "
                    "appear in the server log. This is the only sound way to confirm the BASELINE "
                    "arm, which has no line of its own: the base arm is defined by the absence of "
                    "speculation, so a positive requirement proves nothing about it.")
    ap.add_argument("--prompt-file", default="/tmp/opencode/spec_code.txt")
    ap.add_argument("--workload", default="code", help="label only, recorded in the output")
    ap.add_argument("--log", default="/tmp/opencode/spec_server.log",
                    help="server log to corroborate the arm against")
    ap.add_argument("--pair", type=int, default=0,
                    help="the sweep's pair number, recorded in the output file name so the "
                         "summariser pairs base and spec from the SAME visit")
    args = ap.parse_args()

    try:
        with open(args.prompt_file) as f:
            prompt = f.read()
    except OSError as e:
        print(f"FATAL: cannot read --prompt-file {args.prompt_file}: {e}", file=sys.stderr)
        return 2

    # Which arm is the server currently configured for? The driver sets this before each
    # restart, because a client cannot discover it and guessing would silently mislabel the run.
    try:
        with open(args.state) as f:
            arm = json.load(f)["arm"]
    except Exception:
        print("FATAL: no state file, cannot tell which arm is loaded. "
              "Run this through scripts/sweep_spec.sh, or write the state file yourself.",
              file=sys.stderr)
        return 2

    # The arm label comes from a file the driver writes, so a driver that died between restarting
    # the server and writing the state file would silently mislabel every number. Requiring the
    # server's own log to corroborate the arm turns that from a silent lie into a loud failure.
    # Read the log BEFORE the first request: once requests start flowing, the file is full of
    # timing lines and a substring search is no longer evidence of anything.
    if args.require or args.forbid:
        try:
            with open(args.log) as f:
                head = f.read()
        except OSError as e:
            print(f"FATAL: cannot read --log {args.log}: {e}", file=sys.stderr)
            return 2
        missing = [r for r in args.require.split(",") if r and r not in head]
        if missing:
            print(f"FATAL: server log does not contain {missing!r}, so this is not the arm the "
                  f"state file claims. Refusing to record numbers for the wrong configuration.",
                  file=sys.stderr)
            return 2
        present = [r for r in args.forbid.split(",") if r and r in head]
        if present:
            print(f"FATAL: server log contains {present!r}, which the claimed arm forbids. "
                  f"This server is NOT the baseline. Refusing to record its numbers as baseline.",
                  file=sys.stderr)
            return 2
        print(f"# corroborated in log: require={args.require or '(none)'} "
              f"forbid={args.forbid or '(none)'}", flush=True)

    waited = wait_ready(args.url, args.timeout)
    if waited is None:
        print(f"FATAL: {args.url} never served a request", file=sys.stderr)
        return 2

    print(f"# arm={arm} workload={args.workload} ready_s={waited:.0f} "
          f"prompt_chars={len(prompt)} n_predict={args.n_predict} n={args.n}", flush=True)

    rows = []
    for i in range(args.n):
        r = one_request(args.url, prompt, args.n_predict, args.timeout)
        rows.append(r)
        print(f"  req {i}: tps={r['tps']} n={r['n']} prompt_n={r['prompt_n']} "
              f"prompt_tps={r['prompt_tps']} wall={r['wall']:.1f}s "
              f"draft={r['draft_accepted']}/{r['draft_n']}", flush=True)
        if i == 0:
            print("  (cold: dropped from the median)", flush=True)

    # Everything but the first request. Also drop any request that produced a different number
    # of tokens than asked for, because a short completion has a different per-token cost and
    # would quietly flatter the arm that produced it.
    steady = [r for r in rows[1:] if r["n"] == args.n_predict and r["tps"]]
    dropped = len(rows) - 1 - len(steady)
    if not steady:
        print(f"FATAL: no steady-state request produced {args.n_predict} tokens", file=sys.stderr)
        return 1

    out = {
        "arm": arm,
        "workload": args.workload,
        "pair": args.pair,
        "steady_tps": [r["tps"] for r in steady],
        "median_tps": statistics.median(r["tps"] for r in steady),
        "mean_tps": statistics.fmean(r["tps"] for r in steady),
        "min_tps": min(r["tps"] for r in steady),
        "max_tps": max(r["tps"] for r in steady),
        "prompt_tps": [r["prompt_tps"] for r in steady if r["prompt_tps"]],
        "dropped_short_or_cold": dropped + 1,
    }
    print(f"RESULT {json.dumps(out)}", flush=True)
    drafted = sum(r["draft_n"] for r in steady)
    accepted = sum(r["draft_accepted"] for r in steady)
    out["drafted_total"] = drafted
    out["draft_accepted_total"] = accepted
    out["draft_acceptance"] = (accepted / drafted) if drafted else None

    # A speculative arm that never drafted a token has not been measured, it has merely been
    # run. This is not hypothetical: a first sweep of this harness reported "-0.5%, a tie"
    # when the truth was that the drafter's match length was 12 tokens against a workload
    # whose best match was shorter, so it proposed nothing at all. The engine logged
    # "0 accepted / 48 generated" in one workload and no draft line whatsoever in the other.
    # A tie is a result; a drafter that never fired is a broken experiment.
    if arm != "base" and drafted == 0:
        print("FATAL: this arm is supposed to speculate but the engine drafted ZERO tokens in "
              f"{len(steady)} steady-state requests. The result is meaningless and will not be "
              "written as a measurement. Check the drafter's match length against the workload "
              "(--spec-ngram-*-size-n is a strict exact-match length, not a hint).", file=sys.stderr)
        return 1

    suffix = f"_{args.pair}" if args.pair else ""
    with open(f"/tmp/opencode/spec_{args.workload}_{arm}{suffix}.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
