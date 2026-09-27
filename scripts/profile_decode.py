#!/usr/bin/env python3
"""Profile where decode time actually goes: compute, page faults, or I/O.

    python3 scripts/profile_decode.py --tag warm
    python3 scripts/evict_model_cache.py $M && python3 scripts/profile_decode.py --tag cold

The question this exists to answer
----------------------------------

Every performance claim in this project is supposed to be measured. The one that
was *inferred* rather than measured is how much of decode is expert-weight
fetching, and that number is what decides whether prefetching (M3c) can possibly
help. It was previously derived from two A/B points by solving `t = a + b`, which
produced 3.7%; correcting the algebra gave 32%. Two inferences disagreeing is a
reason to measure directly.

`perf` is not installed here and there is no sudo, so this uses `/proc` instead.
For the question at hand that is not a downgrade, it is a better instrument:

* `/proc/PID/io:read_bytes` counts bytes that actually came from the block
  device. If it is ~0, the model is not waiting on the disk, full stop.
* `majflt` counts page faults that required I/O; `minflt` counts those served
  from the page cache. This separates "reading from disk" from "walking page
  tables", which is exactly the distinction a prefetcher is supposed to exploit.
* `stime` versus `utime` separates kernel page-fault handling from arithmetic.
* `voluntary_ctxt_switches` counts times a thread blocked. I/O wait shows up
  here, spin-wait does not.

A sampling profiler would have given per-op attribution inside the ggml graph.
That is a different question (it is where the *compute* goes, which the recurrent
layers own) and it is not the question blocking M3c.

Design notes
------------

* **The counters are the measurement, the server is only the workload driver.**
  Decode rate is still read from the server's own `timings` object so that
  per-token normalisation uses the engine's own token count.
* **Report a sequence, never a single request,** for the reason
  `bench_server.py` gives: request 1 is the cold page-cache fault.
* **A warm run and a cold run answer different questions.** Warm says what
  steady state costs. Cold says what a prefetcher could possibly recover. A
  prefetch that only helps cold is still worth having, but it must be described
  as a cold-start win and not as a decode win.
* **This script starts nothing and stops nothing.** Two servers on this box
  fight over 6 GB of VRAM and make every number meaningless.
"""

import argparse
import glob
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

CLOCK_TICKS = os.sysconf("SC_CLK_TCK")


def post(url, body, timeout):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def find_pid(binary):
    """Find the server process without shelling out to pgrep."""
    for d in glob.glob("/proc/[0-9]*"):
        try:
            with open(f"{d}/comm", encoding="utf-8") as f:
                if f.read().strip() != binary:
                    continue
            with open(f"{d}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if binary in cmd:
            return int(os.path.basename(d)), cmd.strip()
    return None, None


def read_counters(pid):
    """Snapshot everything we need in one pass over /proc."""
    c = {}

    # schedstat gives on-CPU nanoseconds, which is far finer than the 10 ms
    # clock-tick granularity of utime/stime. Summed over threads.
    on_cpu_ns = 0
    for d in glob.glob(f"/proc/{pid}/task/*"):
        try:
            with open(f"{d}/schedstat", encoding="utf-8") as f:
                on_cpu_ns += int(f.read().split()[0])
        except (OSError, IndexError, ValueError):
            pass
    c["on_cpu_ns"] = on_cpu_ns

    # utime/stime summed over threads: nine threads at 10 ms each give roughly
    # nine times the resolution of one, which is what makes the split usable.
    utime = stime = 0
    for d in glob.glob(f"/proc/{pid}/task/*"):
        try:
            with open(f"{d}/stat", encoding="utf-8") as f:
                fields = f.read().rsplit(") ", 1)[1].split()
        except (OSError, IndexError):
            continue
        # After comm, field 11 is state, so utime is index 11 and stime index 12.
        utime += int(fields[11])
        stime += int(fields[12])
    c["utime_ticks"] = utime
    c["stime_ticks"] = stime

    with open(f"/proc/{pid}/io", encoding="utf-8") as f:
        for line in f:
            k, _, v = line.partition(":")
            if k in ("rchar", "read_bytes", "wchar", "write_bytes"):
                c[k] = int(v)

    with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
        fields = f.read().rsplit(") ", 1)[1].split()
    c["minflt"] = int(fields[7])   # field 10
    c["majflt"] = int(fields[9])   # field 12

    with open(f"/proc/{pid}/status", encoding="utf-8") as f:
        for line in f:
            k, _, v = line.partition(":")
            if k in ("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches"):
                c[k] = int(v)
    return c


def diff(a, b):
    """b - a, for keys present in both."""
    return {k: b[k] - a[k] for k in a if k in b}


def meminfo_cached_gib():
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Cached:"):
                return int(line.split()[1]) / 2**20
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--binary", default="amp-server", help="process name to profile")
    ap.add_argument("--tag", default="decode")
    ap.add_argument("--n", type=int, default=3, help="measured requests (>=2 to see steady state)")
    ap.add_argument("--warmup", type=int, default=1, help="unmeasured requests first")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--prompt-file", default="/tmp/opencode/amp_bench_prompt.txt")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    pid, cmd = find_pid(args.binary)
    if pid is None:
        print(f"no {args.binary} process found; start the server first", file=sys.stderr)
        return 1
    if "llama-server" in cmd or "amp-server" not in cmd:
        pass  # both are fine; we profile whichever was asked for

    with open(args.prompt_file, encoding="utf-8") as f:
        prompt = f.read()

    print(f"tag    : {args.tag}")
    print(f"pid    : {pid}")
    print(f"cmd    : {cmd[:110]}")
    print(f"plan   : {args.warmup} warmup + {args.n} measured requests, "
          f"n_predict={args.n_predict}, temperature=0")
    print(f"cached : {meminfo_cached_gib():.2f} GiB page cache before the run")
    print()

    body = {"prompt": prompt, "n_predict": args.n_predict, "temperature": 0.0, "stream": False}

    for i in range(args.warmup):
        t0 = time.time()
        try:
            r = post(f"{args.url}/v1/completions", body, args.timeout)
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"  warmup {i+1} failed: {e}", file=sys.stderr)
            return 1
        t = r.get("timings") or {}
        print(f"  warmup {i+1}/{args.warmup}: decode {t.get('predicted_per_second')} t/s "
              f"over {t.get('predicted_n')} tok | wall {time.time() - t0:.1f}s")

    before = read_counters(pid)
    rows = []
    for i in range(args.n):
        t0 = time.time()
        try:
            r = post(f"{args.url}/v1/completions", body, args.timeout)
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"  request {i+1} failed: {e}", file=sys.stderr)
            return 1
        wall = time.time() - t0
        t = r.get("timings") or {}
        rows.append({
            "request": i + 1,
            "wall_s": wall,
            "generated": t.get("predicted_n"),
            "decode_tps": t.get("predicted_per_second"),
            "prefill_tps": t.get("prompt_per_second"),
        })
        print(f"  request {i+1}/{args.n}: decode {t.get('predicted_per_second')} t/s "
              f"over {t.get('predicted_n')} tok | wall {wall:.1f}s")
    after = read_counters(pid)

    d = diff(before, after)
    tokens = sum(x["generated"] or 0 for x in rows)
    wall = sum(x["wall_s"] for x in rows)
    if tokens == 0 or wall == 0:
        print("no tokens generated; cannot normalise", file=sys.stderr)
        return 1

    ut = d["utime_ticks"] / CLOCK_TICKS
    st = d["stime_ticks"] / CLOCK_TICKS
    busy = ut + st
    on_cpu = d["on_cpu_ns"] / 1e9

    print()
    print(f"measured: {tokens} tokens in {wall:.1f}s wall "
          f"({tokens / wall:.2f} t/s overall, {len(rows)} requests)")
    print(f"cached  : {meminfo_cached_gib():.2f} GiB page cache after the run")
    print()
    print("per token:")
    print(f"  wall                    {1000 * wall / tokens:8.2f} ms")
    print(f"  on-CPU (schedstat)      {1000 * on_cpu / tokens:8.2f} ms "
          f"({100 * on_cpu / wall:.1f}% of wall)")
    print(f"  user (utime)            {1000 * ut / tokens:8.2f} ms "
          f"({100 * ut / busy:.1f}% of busy)")
    print(f"  system (stime)          {1000 * st / tokens:8.2f} ms "
          f"({100 * st / busy:.1f}% of busy)")
    print()
    print("memory behaviour:")
    print(f"  minor faults            {d['minflt'] / tokens:8.1f} /token "
          f"({d['minflt']} total)")
    print(f"  major faults            {d['majflt'] / tokens:8.1f} /token "
          f"({d['majflt']} total)")
    print(f"  disk read_bytes         {d['read_bytes'] / 2**20:8.2f} MiB "
          f"({d['read_bytes'] / 2**30:.3f} GiB total)")
    print(f"  disk read per token     {d['read_bytes'] / tokens / 1024:8.1f} KiB")
    print(f"  voluntary switches      {d['voluntary_ctxt_switches'] / tokens:8.2f} /token")
    print(f"  involuntary switches    {d['nonvoluntary_ctxt_switches'] / tokens:8.2f} /token")
    print()

    dd = [x["decode_tps"] for x in rows if isinstance(x.get("decode_tps"), (int, float))]
    verdict = []
    if d["read_bytes"] == 0:
        verdict.append("read_bytes is 0: no block-device I/O at all, so nothing is "
                       "waiting on the disk")
    if d["majflt"] == 0:
        verdict.append("majflt is 0: no fault needed I/O, the expert set is "
                       "page-cache resident")
    if busy and st / busy < 0.05:
        verdict.append(f"system time is {100 * st / busy:.1f}% of busy: page-fault "
                       "handling is not on the critical path")
    if dd:
        verdict.append(f"steady decode median {statistics.median(dd):.2f} t/s")
    for v in verdict:
        print(f"  - {v}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"tag": args.tag, "pid": pid, "cmd": cmd, "rows": rows,
                       "tokens": tokens, "wall_s": wall, "delta": d,
                       "decode_tps_median": statistics.median(dd) if dd else None,
                       "verdict": verdict}, f, indent=2)
        print(f"\n  written: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
