#!/usr/bin/env python3
"""Compare two teacher-forced logprob dumps produced by amp-infer --dump-logprobs.

This is the gate for any change to the CUDA path. It exists because the obvious way to
check a change is wrong: generating text with both builds and diffing it tells you only
whether the *sampled* tokens differ, and a single argmax flip early on sends the two runs
down different prefixes, so every later position compares two different questions. Measured
on this model, that method reported 444 of 512 positions "disagreeing" at a median KL of
1.7e-05 while the thing under test had almost no effect at all.

The comparison is therefore built on teacher forcing. One build emits a fixture
(`--emit-score`, a token id per line); every arm then replays exactly that sequence with
`--score-file` and dumps per-position logprobs. Both arms then evaluate byte-identical
prefixes, so every position is comparable and a difference is a real difference.

Both files must have been produced against the same fixture and with the same
`--logprobs-n`. The script checks the shape and refuses to compare files that disagree
about it, because a top-32 dump compared against a top-5 dump silently compares different
token sets.

Usage:
    amp-infer ... --emit-score fix.txt
    amp-infer ... --score-file fix.txt --logprobs-n 32 --dump-logprobs a.tsv
    amp-infer ... --score-file fix.txt --logprobs-n 32 --dump-logprobs b.tsv
    compare_logprobs.py a.tsv b.tsv [--max-kl 1e-9] [--placement-floor 1.2e-4]
"""

import argparse
import math
import sys


def load(path):
    """Read a --dump-logprobs file: one position per line, tab-separated token/logprob pairs."""
    positions = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            cells = line.split("\t")
            if len(cells) % 2 != 0:
                raise SystemExit(f"{path}:{lineno}: odd number of fields, expected token/logprob pairs")
            try:
                pairs = [(int(cells[i]), float(cells[i + 1])) for i in range(0, len(cells), 2)]
            except ValueError as e:
                raise SystemExit(f"{path}:{lineno}: {e}")
            if not pairs:
                raise SystemExit(f"{path}:{lineno}: empty position")
            positions.append(pairs)
    return positions


def kl(p, q):
    """KL(p || q) in nats over the shared support, plus the mass each side has outside it.

    The two supports differ whenever a token leaves one arm's top-k, which is expected and
    is reported rather than silently dropped, because a token dropping out of the window
    is not the same event as a token moving.
    """
    dp = dict(p)
    dq = dict(q)
    shared = dp.keys() & dq.keys()
    out = 0.0
    for t in shared:
        pp = dp[t]
        out += math.exp(pp) * (pp - dq[t])
    mass_p = sum(math.exp(v) for t, v in dp.items() if t not in dq)
    mass_q = sum(math.exp(v) for t, v in dq.items() if t not in dp)
    return out, mass_p, mass_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--max-kl", type=float, default=1e-9,
                    help="median KL ceiling; default rejects anything not bit-equivalent")
    ap.add_argument("--placement-floor", type=float, default=1.2e-4,
                    help="median KL that a placement change alone produces, reported for context")
    args = ap.parse_args()

    A = load(args.a)
    B = load(args.b)

    if len(A) != len(B):
        raise SystemExit(f"different position counts: {len(A)} vs {len(B)}. Same fixture?")

    n_head = [len(p) for p in A]
    n_head_b = [len(p) for p in B]
    if n_head != n_head_b:
        raise SystemExit("different --logprobs-n between the two files; the token sets are not comparable")

    kls, top1_mismatch, compared, only_a, only_b = [], 0, 0, 0, 0
    max_delta = 0.0

    for pa, pb in zip(A, B):
        if pa[0][0] != pb[0][0]:
            top1_mismatch += 1
        da, db = dict(pa), dict(pb)
        only_a += len(da.keys() - db.keys())
        only_b += len(db.keys() - da.keys())
        for t, v in da.items():
            if t in db:
                max_delta = max(max_delta, abs(v - db[t]))
        k, _, _ = kl(pa, pb)
        kls.append(k)
        compared += len(da)

    kls_sorted = sorted(kls)
    median = kls_sorted[len(kls_sorted) // 2] if kls_sorted else 0.0
    worst = max(kls) if kls else 0.0
    identical = all(k == 0.0 for k in kls)

    print(f"positions            {len(A)}")
    print(f"logprobs compared    {compared}  ({n_head[0]} per position)")
    print(f"tokens only in A     {only_a}")
    print(f"tokens only in B     {only_b}")
    print(f"top-1 disagreements  {top1_mismatch} / {len(A)}")
    print(f"median KL            {median:.6e} nats")
    print(f"worst  KL            {worst:.6e} nats")
    print(f"max |delta logprob|  {max_delta:.6e}")
    print()

    if identical:
        print("BIT-IDENTICAL: every compared logprob is exactly equal.")
        return 0

    print(f"reference: a placement change alone (g=3 vs g=6) measures a median KL of")
    print(f"          {args.placement_floor:.2e} on this model, so anything at or below that")
    print(f"          floor is movement from where the arithmetic ran, not a defect in it.")

    if top1_mismatch:
        print(f"FAIL: {top1_mismatch} top-1 disagreements. Zero quality loss requires 0.")
        return 1
    if median > args.max_kl:
        print(f"FAIL: median KL {median:.3e} exceeds --max-kl {args.max_kl:.3e}.")
        return 1

    print(f"PASS: no top-1 disagreements and median KL {median:.3e} within --max-kl.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
