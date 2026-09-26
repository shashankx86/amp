#!/usr/bin/env bash
# Head-to-head: amp-infer vs llama-server on the identical prompt, same quant, same box.
#   scripts/head2head.sh [prompt_file] [n_predict]
set -uo pipefail
cd "$(dirname "$0")/.."

PROMPT=${1:-/tmp/opencode/amp_bench_prompt.txt}
NPRED=${2:-64}
MODEL=../models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf
export LD_LIBRARY_PATH=../llama.cpp/build/bin
PORT=8099

echo "### prompt: $PROMPT ($(wc -c < "$PROMPT") bytes), n_predict=$NPRED"
echo

# Both engines must face the same page-cache reality, so warm it first and say so.
echo "--- warming the page cache (both engines get the same starting point)"
./build/amp-warm --model "$MODEL" --what experts 2>/dev/null | grep -E "warmed|cache"
echo

# ---------------------------------------------------------------- amp
echo "--- amp (planner-chosen split, prefetch on)"
./build/amp-infer --model "$MODEL" --prompt-file "$PROMPT" --n-predict "$NPRED" --ctx 200000 \
  2>/dev/null | grep -E "plan:|prefill |decode "
echo

# ---------------------------------------------------------------- llama.cpp
echo "--- llama-server (best config from ../RUN.md: ncmoe 38, ub 2048)"
../llama.cpp/build/bin/llama-server -m "$MODEL" \
  -c 200000 --parallel 1 -fa on -ctk q8_0 -ctv q4_0 \
  -ngl 41 -ncmoe 38 -b 2048 -ub 2048 -t 8 -tb 8 \
  --jinja --checkpoint-min-step 2048 --ctx-checkpoints 32 \
  --host 127.0.0.1 --port $PORT > /tmp/opencode/llama_h2h.log 2>&1 &
LLAMA_PID=$!
trap 'kill $LLAMA_PID 2>/dev/null' EXIT

for i in $(seq 1 180); do
  grep -q "server is listening\|listening on" /tmp/opencode/llama_h2h.log 2>/dev/null && break
  sleep 1
done
sleep 2

python3 - "$PROMPT" "$NPRED" "$PORT" <<'PY'
import json, sys, time, urllib.request
prompt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
n_pred = int(sys.argv[2]); port = sys.argv[3]
body = json.dumps({"prompt": prompt, "n_predict": n_pred, "temperature": 0.0, "stream": False,
                   "cache_prompt": True}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
t0 = time.time()
r = json.loads(urllib.request.urlopen(req, timeout=7200).read())
wall = time.time() - t0
timings = r.get("timings", {})
pp_ms = timings.get("prompt_ms", 0.0); pp_n = timings.get("prompt_n", 0)
tg_ms = timings.get("predicted_ms", 0.0); tg_n = timings.get("predicted_n", 0)
print("  llama-server prefill: " + (f"{pp_n} tokens in {pp_ms/1000:.2f}s = {pp_n*1000/pp_ms:.1f} t/s" if pp_ms else "n/a"))
print("  llama-server decode : " + (f"{tg_n} tokens in {tg_ms/1000:.2f}s = {tg_n*1000/tg_ms:.2f} t/s" if tg_ms else "n/a"))
print(f"  (request wall time {wall:.1f}s)")
PY

echo
echo "--- llama-server log timings"
grep -E "prompt eval time|eval time" /tmp/opencode/llama_h2h.log | tail -4
kill $LLAMA_PID 2>/dev/null
wait $LLAMA_PID 2>/dev/null
