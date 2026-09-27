#!/usr/bin/env bash
# Sweep the VRAM levers that decide what fits at a given context, and record what each costs.
#
#   ./scripts/sweep_plan.sh "<label>" [-c N] [extra server flags...]
#
# Why this exists
# ---------------
#
# Moving the KV cache from q8_0/q4_0 to q8_0/q8_0 needs about 0.5 GiB more VRAM at 200k, and there
# is only ~0.3 GiB of slack. Something has to give, and the candidates pull in opposite directions:
#
#   n_ubatch down  -> frees the compute buffer (1.03 GiB at 1024), but smaller prefill batches
#   n_gpu_expert_layers down -> frees ~0.31 GiB per layer, but the placement measurement says GPU
#                                expert layers are worth +27 % prefill and -8.5 % decode
#
# So the trade cannot be reasoned about from first principles; it has to be measured on both axes
# at once. This script reports both, plus the VRAM actually used, so the choice is made on numbers.
#
# Prefill is measured on a *fresh* prompt each time, because a prompt-cache hit reports only a few
# tokens and measures nothing. Decode is the steady median over several requests.
#
# It starts and stops servers itself, and refuses to run if one is already up, because two servers
# on this box fight over 6 GB of VRAM and make every number meaningless.
set -uo pipefail
cd "$(dirname "$0")/.."

LABEL="$1"; shift
M="${AMP_MODEL:-/home/e0u/localhost/models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf}"
URL="http://127.0.0.1:8081"
LOG="/tmp/opencode/sweep_$(echo "$LABEL" | tr -c 'a-zA-Z0-9' '_').log"

if pgrep -x amp-server >/dev/null 2>&1; then
    echo "REFUSING: an amp-server is already running (holds ~5.4 GB of VRAM). pkill -x amp-server" >&2
    exit 1
fi

cleanup() { pkill -x amp-server >/dev/null 2>&1; sleep 3; }
trap cleanup EXIT

nohup ./build/bin/amp-server --model "$M" --port 8081 "$@" > "$LOG" 2>&1 &

ok=0
for _ in $(seq 1 180); do
    if curl -s -m 2 "$URL/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
    if ! pgrep -x amp-server >/dev/null 2>&1; then break; fi
    sleep 1
done

if [ "$ok" != "1" ]; then
    printf '%-34s FAILED TO START  %s\n' "$LABEL" \
        "$(grep -ioE 'out of memory|error[^\n]{0,60}' "$LOG" | head -1 | tr -d '\n')"
    exit 1
fi

PLAN="$(grep -m1 'plan: ubatch=' "$LOG" | sed 's/.*plan: //')"
G="$(echo "$PLAN" | sed -n 's/.*gpu_expert_layers=\([0-9]*\).*/\1/p')"
UB="$(echo "$PLAN" | sed -n 's/.*ubatch=\([0-9]*\).*/\1/p')"
KV="$(echo "$PLAN" | sed -n 's/.*kv=\([0-9.]*\) GiB.*/\1/p')"
SLOT="$(grep -m1 -oE 'n_ctx_slot = [0-9]+' "$LOG" | grep -oE '[0-9]+$')"
VRAM="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -dc '0-9')"

# Fresh prompt, so prefill is a real prefill. Then several requests for a decode median.
OUT="$(python3 scripts/bench_server.py --url "$URL" --tag sweep --n 3 --n-predict 256 2>&1)"
PP="$(echo "$OUT" | grep 'request 1' | sed 's/.*prefill \([0-9.]*\) t\/s.*/\1/')"
PPN="$(echo "$OUT" | grep 'request 1' | sed 's/.*prefill [0-9.]* t\/s over \([0-9]*\) tok.*/\1/')"
DD="$(echo "$OUT" | grep 'steady decode' | sed 's/.*median \([0-9.]*\) t\/s.*/\1/')"

printf '%-34s g=%-3s ubatch=%-5s kv=%-6s slot=%-7s vram=%-6s  prefill %-8s (%s tok)  decode %-7s\n' \
    "$LABEL" "${G:-?}" "${UB:-?}" "${KV:-?}" "${SLOT:-?}" "${VRAM}" "${PP:-FAIL}" "${PPN:-?}" "${DD:-FAIL}"
