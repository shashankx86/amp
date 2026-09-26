#!/usr/bin/env python3
"""amp <-> llama.cpp quality parity harness.

The claim under test is "zero quality loss". amp runs llama.cpp's own ggml kernels on the same
weights with the same KV dtypes and the same sampler, so there is no algorithmic difference at
all. What can still differ is floating-point *placement*: the same quantized matmul accumulated on
the CPU and on the GPU sums in a different order. llama.cpp has exactly the same property between
its own -ngl / -ncmoe settings, so this harness measures three things separately:

  1. determinism        - same config twice: must be bit-identical
  2. placement drift    - amp g=6 vs amp g=0: argmax must agree, logprobs drift by float noise
  3. cross-engine       - amp vs llama-server: argmax agreement, and the top1-top2 gap wherever
                          the argmax flips (a flip is only meaningful if the gap is tiny)

Protocol notes, learned the hard way:
  * Do NOT send top_k/top_p with n_probs. Filtering the distribution makes the server renormalise the
    reported logprobs, which looks like a huge numerical difference that is not one.
  * temperature 0 alone is greedy in llama.cpp and leaves the distribution unfiltered.
  * The server must NOT be started with --jinja for a raw /completion prompt: the chat template
    changes the prompt and the comparison becomes meaningless.

usage: scripts/parity.py [prompt] [n_tokens]
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = "/tmp/opencode/parity"
MODEL = os.path.join(ROOT, "..", "models",
                     "Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf")
PORT = 8093
PROMPT = sys.argv[1] if len(sys.argv) > 1 else \
    "what is a page cache, and why does it matter more than RAM for inference?"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 24

os.makedirs(OUT, exist_ok=True)
os.environ["LD_LIBRARY_PATH"] = os.path.join(ROOT, "..", "llama.cpp", "build", "bin")


def load_logprobs(path):
    out = []
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if not parts or parts == [""]:
            continue
        out.append([(int(parts[i]), float(parts[i + 1])) for i in range(0, len(parts) - 1, 2)])
    return out


def run_amp(tag, g=6, ubatch=1024):
    lp, txt = f"{OUT}/amp_{tag}.tsv", f"{OUT}/amp_{tag}.txt"
    subprocess.run([os.path.join(ROOT, "build/amp-infer"), "--model", MODEL, "--prompt", PROMPT,
                    "--n-predict", str(N), "--ctx", "16384", "--temp", "0",
                    "--ubatch", str(ubatch), "--gpu-layers", str(g),
                    "--dump-logprobs", lp, "--dump-output", txt],
                   cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return load_logprobs(lp), open(txt, encoding="utf-8").read()


def run_llama(gpu_expert_layers=6):
    log = f"{OUT}/llama.log"
    srv = subprocess.Popen(
        [os.path.join(ROOT, "../llama.cpp/build/bin/llama-server"), "-m", MODEL, "-c", "16384",
         "--parallel", "1", "-fa", "on", "-ctk", "q8_0", "-ctv", "q4_0", "-ngl", "41",
         "-ncmoe", str(40 - gpu_expert_layers), "-b", "1024", "-ub", "1024", "-t", "8", "-tb", "8",
         "--host", "127.0.0.1", "--port", str(PORT)],
        stdout=open(log, "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            if "listening on" in open(log).read():
                break
            time.sleep(1)
        time.sleep(2)
        body = json.dumps({"prompt": PROMPT, "n_predict": N, "temperature": 0.0, "n_probs": 5,
                           "stream": False, "cache_prompt": False}).encode()
        r = json.loads(urllib.request.urlopen(
            urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                   headers={"Content-Type": "application/json"}, method="POST"),
            timeout=7200).read())
    finally:
        srv.terminate()
        srv.wait()
    # normalise the server payload to [[(id, logprob), ...]] ordered by descending logprob
    norm = []
    for e in r["completion_probabilities"]:
        top = sorted(e.get("top_logprobs", []), key=lambda q: -q["logprob"])
        norm.append([(q["id"], q["logprob"]) for q in top])
    return norm, r.get("content", "")


def compare(a_lp, b_lp, label):
    n = min(len(a_lp), len(b_lp))
    same, maxd, flips = 0, 0.0, []
    for i in range(n):
        # a_lp rows are amp's [[ (token, logprob), ... ]]; b_lp rows may come straight from the
        # server's payload (dicts keyed by id), so normalise both sides to (id -> logprob) plus a
        # separately tracked argmax.
        a, a_arg = dict(a_lp[i]), a_lp[i][0][0]
        b = dict(b_lp[i])
        b_arg = b_lp[i][0][0] if b_lp[i] else -1
        if not a or not b:
            continue
        same += (a_arg == b_arg)
        for t in set(a) & set(b):
            maxd = max(maxd, abs(a[t] - b[t]))
        if a_arg != b_arg:
            vals = sorted(b.values(), reverse=True)
            flips.append((i, (vals[0] - vals[1]) if len(vals) > 1 else float("nan")))
    print(f"  {label}")
    print(f"    positions {n}, top-1 agreement {same}/{n}, max |delta logprob| {maxd:.6f}")
    for i, gap in flips:
        print(f"    argmax flipped at position {i}; top1-top2 gap there = {gap:.4f}")
    return same, n, maxd, flips


def main():
    print(f"prompt: {PROMPT[:80]}{'...' if len(PROMPT) > 80 else ''}   n_tokens={N}\n")

    a1, t1 = run_amp("det1")
    a2, t2 = run_amp("det2")
    print("1) determinism (amp g=6 twice)")
    compare(a1, a2, "amp vs amp")
    print(f"    generated text identical: {t1 == t2}\n")

    a0, t0 = run_amp("g0", g=0)
    print("2) placement drift (amp g=6 vs amp g=0: same kernels, different device for experts)")
    compare(a1, a0, "amp g=6 vs g=0")
    print(f"    generated text identical: {t1 == t0}\n")

    lp, tl = run_llama()
    print("3) cross-engine (amp g=6 vs llama-server -ncmoe 34, identical placement)")
    compare(a1, lp, "amp vs llama-server")
    print(f"    generated text identical: {t1 == tl}")
    open(f"{OUT}/amp.txt", "w", encoding="utf-8").write(t1)
    open(f"{OUT}/llama.txt", "w", encoding="utf-8").write(tl)


if __name__ == "__main__":
    main()
