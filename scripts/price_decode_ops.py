#!/usr/bin/env python3
"""Price each op class in a decode token, with the per-node event pair applied only to the
ops being asked about.

The instrument times a node by recording an event either side of ggml_cuda_compute_forward and
synchronising on the second, which drains the pipeline. With ~700 nodes per decode token,
doing that unconditionally measures the drain rather than the kernels, so AMP_OPS_MATCH
restricts the timing to one op class at a time and the drain is paid once per node of that
class.

Two runs differing only in --n-predict difference the decode out: prefill, warmup and model
load are identical between them and cancel.
"""
import re
import subprocess
import sys

MODEL = "../models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf"
PROMPT = "/tmp/opencode/fix200k2.txt"
N_DEC = 500
TOT = re.compile(r"total device time ([\d.]+) ms over (\d+) nodes \((\d+) timed\)")

CLASSES = [
    "MUL_MAT", "MUL_MAT_ID", "FLASH_ATTN_EXT", "GET_ROWS", "ADD", "MUL", "UNARY",
    "CONCAT", "CPY", "CONT", "ROPE", "SET_ROWS", "GLU", "SCALE",
]


def run(match, n_predict):
    env = {"AMP_TRACE_OPS": "1", "AMP_NO_CUDA_GRAPH": "1", "PATH": "/usr/bin:/bin"}
    if match:
        env["AMP_OPS_MATCH"] = match
    cmd = ["./build/bin/amp-infer", "--model", MODEL, "--prompt-file", PROMPT,
           "--ctx", "8192", "--n-predict", str(n_predict), "--temp", "0", "--seed", "1234"]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True).stderr
    m = TOT.search(out)
    if not m:
        raise SystemExit("no OPS line for match=%r n=%d\n%s" % (match, n_predict, out[-800:]))
    return float(m.group(1)), int(m.group(2)), int(m.group(3))


def main():
    classes = sys.argv[1:] or CLASSES
    print("ctx 8192, %d decoded tokens, per-node event pair on the named class only\n" % N_DEC)
    print("%-16s %10s %10s %12s" % ("class", "ms/token", "nodes/tok", "us/node"))
    tot_ms = tot_nodes = 0.0
    for c in classes:
        a_ms, a_n, a_t = run(c, 1)
        b_ms, b_n, b_t = run(c, N_DEC + 1)
        ms = (b_ms - a_ms) / N_DEC
        n = (b_t - a_t) / N_DEC
        per = (ms * 1000.0 / n) if n > 0 else 0.0
        tot_ms += ms
        tot_nodes += n
        print("%-16s %10.4f %10.1f %12.2f" % (c, ms, n, per))
    print("%-16s %10.4f %10.1f" % ("SUM", tot_ms, tot_nodes))


if __name__ == "__main__":
    main()
