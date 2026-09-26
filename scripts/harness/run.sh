#!/usr/bin/env bash
#
# Drive the real OpenCode client against a running inference server.
#
# This is the acceptance test. Everything else in scripts/ checks that a specific
# mechanism works; this checks that a real agentic client can hold a multi-turn
# conversation with tool calls against the server, which is how the server is
# actually used.
#
# Each prompt runs in a pristine copy of the fixture workspace, so the results do
# not depend on what a previous run changed. Prompts 2 and 3 edit files.
#
# Usage:
#   ./scripts/harness/run.sh                       # all prompts, server on :8081
#   ./scripts/harness/run.sh --url http://127.0.0.1:8099   # a different server
#   ./scripts/harness/run.sh --model ref/occamy    # a provider defined in the generated config
#   ./scripts/harness/run.sh --only 2              # one prompt by number
#
# Notes:
#   * --auto is mandatory. Without it OpenCode stops on the first tool call waiting
#     for a permission answer that a script cannot give, and the run hangs.
#   * Results go under scripts/harness/runs/<timestamp>/. Never /tmp: it is tmpfs on
#     this machine, so anything written there lives in RAM.
#   * This script does NOT start a server. It health-checks one and refuses to run
#     if it is not up, because two servers on this box fight over 6 GB of VRAM.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
AMP_ROOT="$(cd "$HERE/../.." && pwd)"

URL="http://127.0.0.1:8081"
MODEL="amp/occamy"
ONLY=""
KEEP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --url)   URL="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --only)  ONLY="$2"; shift 2 ;;
        --keep)  KEEP=1; shift ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

command -v opencode >/dev/null || { echo "opencode not on PATH" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
# A stale server holds ~5.4 GB of VRAM and makes every measurement meaningless.
if pgrep -x amp-server >/dev/null 2>&1 && pgrep -x llama-server >/dev/null 2>&1; then
    echo "REFUSING: both amp-server and llama-server are running. They contend for the same" >&2
    echo "6 GB of VRAM. Stop one first (pkill -x <name>)." >&2
    exit 1
fi

if ! curl -fsS --max-time 5 "$URL/health" >/dev/null 2>&1; then
    echo "REFUSING: no server answering at $URL/health" >&2
    echo "Start one, e.g.:" >&2
    echo "  $AMP_ROOT/build/bin/amp-server --model <gguf> --port 8081" >&2
    exit 1
fi

HEALTH="$(curl -fsS --max-time 5 "$URL/health")"
echo "server: $URL"
echo "health: $HEALTH"

# ---------------------------------------------------------------- run dir
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$HERE/runs/$TS"
mkdir -p "$OUT"

# OpenCode discovers config by walking up from the working directory, so putting
# opencode.json next to (not inside) the workspace keeps the fixture clean.
# Provider shape copied from the user's working ~/.config/opencode/opencode.json
# (package + settings.baseURL). Do not "modernise" this to npm/options without
# checking the providers guide - the working shape is the one known to load.
cat > "$OUT/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "providers": {
    "amp": {
      "name": "amp-server (under test)",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": { "baseURL": "$URL/v1" },
      "models": {
        "occamy": {
          "name": "Occamy 1.0 APEX-I MiniPlus V2.1",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"] },
          "limit": { "context": 200000, "output": 8192 },
          "compatibility": { "reasoningField": "reasoning_content" }
        }
      }
    },
    "ref": {
      "name": "reference server (llama.cpp baseline)",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": { "baseURL": "http://127.0.0.1:8099/v1" },
      "models": {
        "occamy": {
          "name": "Occamy 1.0 APEX-I MiniPlus V2.1 (reference)",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"] },
          "limit": { "context": 200000, "output": 8192 },
          "compatibility": { "reasoningField": "reasoning_content" }
        }
      }
    }
  }
}
JSON

cp -r "$HERE/workspace" "$OUT/ws"
WS="$OUT/ws"

# ---------------------------------------------------------------- run prompts
FAILED=0
SUMMARY="$OUT/summary.md"
{
    echo "# Harness run $TS"
    echo
    echo "server: $URL"
    echo "model:  $MODEL"
    echo
    echo "| prompt | wall s | exit | tool calls | verdict |"
    echo "|---|---|---|---|---|"
} > "$SUMMARY"

for p in "$HERE"/prompts/*.md; do
    name="$(basename "$p" .md)"
    # Files are zero-padded (01-, 02-, 03-) so they sort in order, but a user types --only 1.
    # 10# strips the padding so both work. Comparing the strings silently skipped every prompt.
    num=$((10#${name%%-*}))
    if [ -n "$ONLY" ] && [ "$num" -ne "$ONLY" ]; then
        continue
    fi

    echo
    echo "=== $name ==="

    # Pristine workspace per prompt: prompts 2 and 3 edit files, so without this
    # a second run of the suite would start from an already-fixed tree and the
    # results would not be comparable.
    rm -rf "$WS"
    cp -r "$HERE/workspace" "$WS"

    start=$(date +%s.%N)
    set +e
    ( cd "$WS" && opencode run --auto --model "$MODEL" --format json "$(cat "$p")" ) \
        > "$OUT/$name.json" 2> "$OUT/$name.err"
    rc=$?
    set -e
    end=$(date +%s.%N)
    wall=$(awk "BEGIN{printf \"%.1f\", $end-$start}")

    # Tool calls appear in the JSON stream; count distinct tool names rather than
    # counting events so a retried call is not double-counted.
    tools=$(grep -o '"tool"[[:space:]]*:[[:space:]]*"[a-z_]*"' "$OUT/$name.json" 2>/dev/null \
            | sed 's/.*"\([a-z_]*\)"$/\1/' | sort -u | tr '\n' ',' | sed 's/,$//')
    ntools=$(printf '%s' "$tools" | tr -cd ',' | wc -c)

    if [ "$rc" -eq 0 ]; then verdict="ok"; else verdict="EXIT $rc"; FAILED=1; fi
    if [ -z "$tools" ]; then verdict="$verdict (NO TOOL CALLS)"; fi

    printf '| %s | %s | %s | %s | %s |\n' "$name" "$wall" "$rc" "${tools:-none}" "$verdict" \
        >> "$SUMMARY"

    echo "  wall ${wall}s  exit $rc  tools: ${tools:-NONE}"
    if [ "$rc" -ne 0 ]; then
        echo "  --- stderr ---"
        tail -20 "$OUT/$name.err" | sed 's/^/  /'
    fi
done

# The fixture must be left exactly as we found it, or the next run is not comparable.
rm -rf "$WS"

echo
echo "=== summary ==="
cat "$SUMMARY"
echo
echo "artifacts: $OUT"

[ "$KEEP" -eq 1 ] || true
[ "$FAILED" -eq 0 ] || { echo "one or more prompts failed - see $SUMMARY" >&2; exit 1; }
