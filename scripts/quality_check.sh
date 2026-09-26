#!/usr/bin/env bash
# Quality parity check: amp vs llama-server, greedy decoding, identical prompt.
# The claim under test is "zero quality loss" - the arithmetic is llama.cpp's own, so the
# generated text must match token for token. Any difference here is a bug, not a tuning gap.
set -uo pipefail
cd "$(dirname "$0")/.."

PROMPT=${1:-/tmp/opencode/small_prompt.txt}
NPRED=${3:-48}
MODEL=../models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf
OUT=/tmp/opencode/quality
PORT=8098
# Only the llama-server *baseline* needs this; amp's own binaries are statically linked against the
# vendored llama.cpp and have no shared-library dependencies.
export LD_LIBRARY_PATH=../llama.cpp/build/bin
mkdir -p "$OUT"

printf "what is a page cache, and why does it matter more than RAM for inference? " > "$OUT/prompt.txt"
[ -s "$PROMPT" ] && cp "$PROMPT" "$OUT/prompt_full.txt"

echo "### greedy, n_predict=$NPRED"
echo

echo "--- amp"
./build/bin/amp-infer --model "$MODEL" --prompt "$(cat "$OUT/prompt.txt")" --n-predict "$NPRED" \
  --ctx 32768 --temp 0 --top-k 0 --top-p 1.0 --ubatch 1024 --gpu-layers 6 \
  --dump-output "$OUT/amp.txt" 2>/dev/null | grep -E "prefill |decode "
echo

echo "--- llama-server (same greedy settings)"
../llama.cpp/build/bin/llama-server -m "$MODEL" -c 32768 --parallel 1 -fa on \
  -ctk q8_0 -ctv q4_0 -ngl 41 -ncmoe 34 -b 1024 -ub 1024 -t 8 -tb 8 \
  --host 127.0.0.1 --port $PORT > "$OUT/llama.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null' EXIT
for i in $(seq 1 180); do
  grep -q "listening on" "$OUT/llama.log" 2>/dev/null && break
  sleep 1
done
sleep 2

python3 - "$OUT/prompt.txt" "$NPRED" "$PORT" "$OUT/llama.txt" <<'PY'
import json, sys, urllib.request
prompt = open(sys.argv[1], encoding="utf-8").read()
body = json.dumps({"prompt": prompt, "n_predict": int(sys.argv[2]), "temperature": 0.0,
                   "top_k": 1, "top_p": 1.0, "stream": False, "cache_prompt": False}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{sys.argv[3]}/completion", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
open(sys.argv[4], "w", encoding="utf-8").write(r.get("content", ""))
print("  wrote", sys.argv[4])
PY

kill $PID 2>/dev/null; wait $PID 2>/dev/null
echo
echo "### diff"
if diff -q "$OUT/amp.txt" "$OUT/llama.txt" >/dev/null 2>&1; then
  echo "  IDENTICAL ($(wc -c < "$OUT/amp.txt") bytes) - amp's output matches llama.cpp byte for byte"
else
  echo "  DIFFERENT:"
  diff <(fold -w 100 "$OUT/amp.txt") <(fold -w 100 "$OUT/llama.txt") | head -30
  echo
  echo "  amp   ($(wc -c < "$OUT/amp.txt") bytes): $(head -c 220 "$OUT/amp.txt")"
  echo "  llama ($(wc -c < "$OUT/llama.txt") bytes): $(head -c 220 "$OUT/llama.txt")"
fi
