#!/usr/bin/env bash
# Alternating A/B of speculative decoding against nothing, across two workloads.
#
#   scripts/sweep_spec.sh [pairs] [requests-per-session]
#
# WHY A SEPARATE DRIVER RATHER THAN A LOOP IN THE BENCHMARK
#
# llama.cpp's per-request `speculative.type` override is compiled out
# (tools/server/server-schema.cpp:197 is `#if 0`), so the arm cannot change inside a running
# server. That makes each arm cost a full process start and a 13.66 GiB model load, and it is
# the reason this alternates instead of running all of one arm and then all of the other:
# steady decode on this box varies about 22% between sessions and drifted 30.5 -> 27.2 t/s
# inside one 15-minute session, so a blocked design measures that drift as a result.
#
# The arms are visited base, spec, base, spec, ... and compared as paired differences.
#
# SAFETY, and the reason this is a script and not something to improvise
#
# * Never two servers. This box has 6 GB of VRAM and one of them dies or thrashes. The loop
#   waits for the previous server to be gone before starting the next, and refuses to continue
#   if one is already running when it begins.
# * Never `--load-mode none` or `--direct-io`. An anonymous 14.75 GiB load froze this box once.
# * Run it detached. The kernel OOM killer is a live hazard when a 13.66 GiB model is resident
#   alongside everything else; an OOM kill should take down this sweep, not the session driving
#   it. So: setsid, output to a log, and poll the log rather than sit in the foreground.
#
#   setsid nohup scripts/sweep_spec.sh 3 5 > /tmp/opencode/sweep.log 2>&1 < /dev/null &
#
#   tail -f /tmp/opencode/sweep.log
set -uo pipefail

PAIRS="${1:-3}"
REQS="${2:-5}"
PORT="${AMP_SPEC_PORT:-8081}"
MODEL="${AMP_TEST_MODEL:-/home/e0u/localhost/models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf}"
CTX="${AMP_SPEC_CTX:-200000}"
WORK="${AMP_SPEC_WORK:-/tmp/opencode}"
BIN="${AMP_BIN:-./build/bin/amp-server}"
LOG="$WORK/sweep.log"

# Arm definitions: label -> extra argv, and the substring that must appear in the server's own
# log before bench_spec.py will believe the label. The second half is the point: a driver that
# died between restarting the server and writing the state file would otherwise mislabel every
# number it recorded, silently.
arm_args() {
    case "$1" in
        base)      echo "" ;;
        map-k)     echo "--spec-type ngram-map-k" ;;
        map-k4v)   echo "--spec-type ngram-map-k4v" ;;
        mod)       echo "--spec-type ngram-mod" ;;
        simple)    echo "--spec-type ngram-simple" ;;
        *) echo "unknown arm $1" >&2; return 1 ;;
    esac
}
# What the server must log for us to believe it is that arm. The base arm's marker is the
# absence of any speculative type, so it gets the one line every run prints.
arm_require() {
    case "$1" in
        base)      echo "amp-preflight: n_parallel" ;;
        map-k)     echo "ngram-map-k" ;;
        map-k4v)   echo "ngram-map-k4v" ;;
        mod)       echo "ngram-mod" ;;
        simple)    echo "ngram-simple" ;;
    esac
}

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

no_strays() {
    local p
    p="$(pgrep -a amp-server 2>/dev/null || true)"
    if [ -n "$p" ]; then
        say "REFUSING TO START, an amp-server is already running:"
        say "$p"
        return 1
    fi
    return 0
}

wait_gone() {
    local i
    for i in $(seq 1 60); do
        pgrep -x amp-server >/dev/null 2>&1 || return 0
        sleep 1
    done
    return 1
}

mkdir -p "$WORK"
: > "$LOG"

no_strays || exit 1
[ -x "$BIN" ] || { say "FATAL: $BIN not found or not executable. Build first."; exit 1; }
[ -f "$MODEL" ] || { say "FATAL: model not found at $MODEL"; exit 1; }

say "sweep: pairs=$PAIRS reqs/session=$REQS ctx=$CTX port=$PORT"
say "model=$MODEL"
python3 scripts/gen_spec_prompts.py "$WORK" 2>&1 | tee -a "$LOG"
say "free RAM: $(free -g | awk 'NR==2{print $7" GiB"}')  free VRAM: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"

for workload in code prose; do
    prompt="$WORK/spec_${workload}.txt"
    [ -f "$prompt" ] || { say "FATAL: $prompt missing; run scripts/gen_spec_prompts.py"; exit 1; }

    for pair in $(seq 1 "$PAIRS"); do
        for arm in base map-k; do
            args="$(arm_args "$arm")"
            require="$(arm_require "$arm")"
            slog="$WORK/server_${workload}_${arm}_${pair}.log"

            say "--- $workload pair $pair/$PAIRS arm=$arm args=[$args] ---"

            rm -f "$slog"
            # Detached, so an OOM kill takes the server and not whatever is driving it.
            setsid nohup "$BIN" --model "$MODEL" --port "$PORT" -c "$CTX" $args \
                > "$slog" 2>&1 < /dev/null &
            sleep 1

            # Wait for the load, judged by the log rather than by /health. `/health` answers 200
            # about two seconds in, while the model is still being read off disk; a readiness
            # check that trusts it measures an empty server.
            loaded=0
            for i in $(seq 1 400); do
                if grep -qE "server is listening|load_tensors: offload [0-9]+/[0-9]+ tensors" "$slog" 2>/dev/null; then
                    loaded=1; break
                fi
                if grep -qiE "error|failed to allocate|out of memory|cudaMalloc" "$slog" 2>/dev/null; then
                    say "arm=$arm FAILED TO LOAD:"
                    grep -iE "error|failed to allocate|out of memory|cudaMalloc" "$slog" | head -5 | tee -a "$LOG"
                    break
                fi
                sleep 2
            done

            if [ "$loaded" = 1 ]; then
                python3 scripts/bench_spec.py \
                    --url "http://127.0.0.1:$PORT" \
                    --state "$WORK/spec_state.json" \
                    --log "$slog" \
                    --require "$require" \
                    --prompt-file "$prompt" \
                    --workload "$workload" \
                    --n "$REQS" \
                    --n-predict 128 \
                    --pair "$pair" \
                    2>&1 | tee -a "$LOG" &
                bench=$!
                wait "$bench"
                # The script writes the state file the driver owns; do it after, so the
                # corroboration check above is reading a log that already exists.
                echo "{\"arm\": \"$arm\"}" > "$WORK/spec_state.json"
            else
                say "arm=$arm did not come up; skipping this cell"
            fi

            say "stopping server"
            pkill -x amp-server 2>/dev/null || true
            wait_gone || { say "FATAL: a server would not die; refusing to start another"; exit 1; }
            sleep 3
        done
    done
done

say "sweep complete. summary:"
python3 scripts/summarise_spec.py "$WORK" 2>&1 | tee -a "$LOG"
