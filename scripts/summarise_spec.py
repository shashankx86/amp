#!/usr/bin/env python3
"""Summarise a speculative-decoding sweep as paired differences.

    python3 scripts/summarise_spec.py /tmp/opencode

Reads the spec_<workload>_<arm>.json files that scripts/bench_spec.py writes, and reports the
one number that matters: the paired per-session difference, not a difference of pooled medians.

Why paired, and why not "median(spec) / median(base)":

* Sessions drift. Decode moves 22% between sessions on this box and 30.5 -> 27.2 t/s inside a
  single 15-minute session. A ratio of pooled medians silently charges the drift to whichever
  arm happened to run in the faster half of the afternoon. Pairing base/session N against
  spec/session N, and taking the differences, cancels any drift that is smooth in time - which
  is what drift is - and leaves the variance that is not.

* The variance is the finding when the effect is small. Strata measured 6-11% for prompt-lookup
  speculation on code edits. If the per-pair differences here scatter wider than that, then
  whatever the mean says is not measurable on this box and the honest report is "below the
  noise", not a rounded-up figure. `spread` in the output is the check for that.
"""
import json
import pathlib
import statistics
import sys


def pair_index(path):
    """The sweep's pair number, from a filename like spec_code_map-k_2.json. None if absent."""
    stem = path.stem                      # spec_code_map-k_2
    tail = stem.rsplit("_", 1)[-1]        # "2"
    return int(tail) if tail.isdigit() else None


def main():
    work = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode")

    rows = {}
    # A later run of the same arm overwrites this filename, so pairing "by position" would pair
    # session 3 of a 5-pair sweep against session 3 of a 3-pair one and quietly compare
    # different sessions. Keying by the loop's own pair index - carried in the filename by
    # bench_spec.py's caller and recorded here when present - makes a mismatched pairing visible
    # instead of statistically meaningless.
    for f in sorted(work.glob("spec_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(d, dict) or "median_tps" not in d:
            continue
        d["_file"] = f.name
        d.setdefault("pair", pair_index(f))
        rows.setdefault(d.get("workload", "?"), {}).setdefault(d["arm"], []).append(d)

    if not rows:
        print("no sweep results found in", work)
        return 1

    for workload, arms in sorted(rows.items()):
        print(f"\n=== workload: {workload} ===")
        base = arms.get("base", [])
        specs = {a: v for a, v in arms.items() if a != "base"}
        if not base:
            print("  no base arm, nothing to compare against")
            continue
        print(f"  base: {len(base)} session(s), medians "
              f"{[round(d['median_tps'], 2) for d in base]} t/s")
        for arm, ds in sorted(specs.items()):
            print(f"  {arm}: {len(ds)} session(s), medians "
                  f"{[round(d['median_tps'], 2) for d in ds]} t/s")

        # Pair on the sweep's pair index, so a base and a spec run from the same visit are
        # compared and nothing else is.
        for arm, ds in sorted(specs.items()):
            bmap = {d.get("pair"): d for d in base}
            pairs, unmatched = [], 0
            for s in ds:
                b = bmap.get(s.get("pair"))
                if b is None or b.get("workload") != s.get("workload"):
                    unmatched += 1
                    continue
                pairs.append((b, s))
            if unmatched:
                print(f"\n  {arm}: {unmatched} session(s) had no matching base visit and were "
                      f"NOT counted. A ratio of pooled medians would have included them.")
            if not pairs:
                print(f"\n  {arm} vs base: no matched pairs")
                continue
            diffs = [100.0 * (s["median_tps"] - b["median_tps"]) / b["median_tps"]
                     for b, s in pairs]
            med = statistics.median(diffs)
            spread = (max(diffs) - min(diffs)) if len(diffs) > 1 else float("nan")
            print(f"\n  {arm} vs base, paired:")
            print(f"    per-pair: {[round(d, 1) for d in diffs]} %")
            print(f"    median  : {med:+.1f} %   spread: {spread:.1f} pp"
                  f"   n={len(diffs)} session pairs")

            # Draft accounting, and the verdict. A first version of this printed "separable"
            # whenever the spread was under 10 pp, which is a claim about the noise being
            # small and NOT about the effect being real: a measured -0.5% with a 4.7 pp spread
            # is a tie, and saying "separable" invited reading it as a result. So the test is
            # the effect against the spread.
            acc = [d.get("draft_acceptance") for d in ds
                   if d.get("draft_acceptance") is not None]
            drafted = sum(d.get("drafted_total", 0) or 0 for d in ds)
            if drafted:
                rate = sum(d.get("draft_accepted_total", 0) or 0
                           for d in ds) / drafted
                print(f"    drafted {drafted} tokens, {rate*100:.1f}% accepted")
            else:
                print("    drafted 0 tokens - THE DRAFTER NEVER FIRED. Not a measurement.")

            if len(diffs) > 1 and abs(med) < spread:
                print(f"    VERDICT: TIE. The effect ({med:+.1f} %) is smaller than the "
                      f"per-pair spread ({spread:.1f} pp), so this measures nothing either way.")
            elif len(diffs) > 1:
                print(f"    VERDICT: {med:+.1f} %, and it is larger than the per-pair spread "
                      f"({spread:.1f} pp).")
            else:
                print("    VERDICT: one session pair is a hypothesis, not a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
