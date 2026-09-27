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
ARMS="${AMP_SPEC_ARMS:-base,map-k}"
WORKLOADS="${AMP_SPEC_WORKLOADS:-code,prose}"
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
        # Strata's configuration rather than llama.cpp's default. Strata's SuffixDrafter keys
        # on a trigram (WAYS=4 candidates, min_match 3, drafts up to 5). llama.cpp's default
        # is size_n=12, size_m=48: a 12-token EXACT key, which is why the first sweep drafted
        # nothing at all on the code workload and 48 tokens with zero accepted on prose.
        map-k3)    echo "--spec-type ngram-map-k --spec-ngram-map-k-size-n 3 --spec-ngram-map-k-size-m 16" ;;
        simple3)   echo "--spec-type ngram-simple --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 16" ;;
        # The last configuration worth testing. The 3-gram arm loses because a 10-token window
        # needs an accepted run of 3.7 and gets 2.75. A 5-token window raises the density of
        # chances (more passes, each shorter) while lowering the pass cost, so it is the only
        # remaining shape where a run-based drafter could pay. If this also loses, the idea is
        # exhausted for this model and no further n-gram tuning is worth spending on.
        map-k3-m5)  echo "--spec-type ngram-map-k --spec-ngram-map-k-size-n 3 --spec-ngram-map-k-size-m 5" ;;
        map-k4v)   echo "--spec-type ngram-map-k4v" ;;
        mod)       echo "--spec-type ngram-mod" ;;
        simple)    echo "--spec-type ngram-simple" ;;
        *) echo "unknown arm $1" >&2; return 1 ;;
    esac
}
# What the server must log for us to believe it is that arm, and what it must NOT log.
#
# Both halves matter, and the negative half is the only sound one for the baseline. The base arm
# is defined by the ABSENCE of speculation, so requiring a line that merely proves the server
# started proves nothing - a driver that failed to pass --spec-type would look identical. So:
#
#   base  -> forbid the speculation line
#   spec  -> require the exact implementation name
#
# The marker text is from this pinned llama.cpp at -lv 5, which is why every arm runs at -lv 5:
#   I spec common_specu: adding speculative implementation 'ngram-map-k'
#   I srv    load_model: speculative decoding context initialized
# At the default verbosity 3 the speculative configuration is not logged at all, which is how a
# whole sweep produced base numbers for an arm that was supposed to be measuring speculation.
SPEC_MARK="adding speculative implementation"
arm_require() {
    case "$1" in
        base)      echo "" ;;
        map-k)     echo "$SPEC_MARK 'ngram-map-k'" ;;
        map-k3)    echo "$SPEC_MARK 'ngram-map-k'" ;;
        map-k3-m5) echo "$SPEC_MARK 'ngram-map-k'" ;;
        simple3)   echo "$SPEC_MARK 'ngram-simple'" ;;
        map-k4v)   echo "$SPEC_MARK 'ngram-map-k4v'" ;;
        mod)       echo "$SPEC_MARK 'ngram-mod'" ;;
        simple)    echo "$SPEC_MARK 'ngram-simple'" ;;
    esac
}
arm_forbid() {
    case "$1" in
        base)      echo "$SPEC_MARK" ;;
        *)         echo "" ;;
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
say "arms=$ARMS  workloads=$WORKLOADS"
say "model=$MODEL"
python3 scripts/gen_spec_prompts.py "$WORK" 2>&1 | tee -a "$LOG"
say "free RAM: $(free -g | awk 'NR==2{print $7" GiB"}')  free VRAM: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"

# shellcheck disable=SC2206
ARMS_LIST=($(echo "$ARMS" | tr ',' ' '))
# shellcheck disable=SC2206
WORKLOADS_LIST=($(echo "$WORKLOADS" | tr ',' ' '))

for workload in "${WORKLOADS_LIST[@]}"; do
    prompt="$WORK/spec_${workload}.txt"
    [ -f "$prompt" ] || { say "FATAL: $prompt missing; run scripts/gen_spec_prompts.py"; exit 1; }

    for pair in $(seq 1 "$PAIRS"); do
        for arm in "${ARMS_LIST[@]}"; do
            args="$(arm_args "$arm")"
            require="$(arm_require "$arm")"
            forbid="$(arm_forbid "$arm")"
            slog="$WORK/server_${workload}_${arm}_${pair}.log"

            say "--- $workload pair $pair/$PAIRS arm=$arm args=[$args] ---"

            rm -f "$slog"
            # Detached, so an OOM kill takes the server and not whatever is driving it.
            # -lv 5 on every arm, including base: the speculative configuration is not logged
            # below it, so at the default verbosity the base arm cannot be told from a spec arm.
            setsid nohup "$BIN" --model "$MODEL" --port "$PORT" -c "$CTX" -lv 5 $args \
                > "$slog" 2>&1 < /dev/null &
            sleep 1

            # Wait for the load, judged by the log rather than by /health. `/health` answers 200
            # about two seconds in, while the model is still being read off disk; a readiness
            # check that trusts it measures an empty server.
            #
            # The marker has to be one THIS build actually prints. An earlier version waited for
            # "server is listening" and "load_tensors: offload N/M tensors", neither of which
            # exists in the pinned llama.cpp - it emits "llama_server: model loaded" and
            # "llama_server: listening on http://...". The wait then never fired and the sweep
            # sat out its full 800 s per cell against a server that had been ready in 12 s.
            # Grounded in the log we have on disk:
            #   0.12.443.447 I srv  llama_server: model loaded
            #   0.12.443.448 I srv  llama_server: listening on http://127.0.0.1:8081
            loaded=0
            for i in $(seq 1 400); do
                if grep -qE "llama_server: (model loaded|listening on http)" "$slog" 2>/dev/null; then
                    loaded=1
                    say "loaded after ~$((i * 2))s"
                    break
                fi
                if grep -qiE "error|failed to allocate|out of memory|cudaMalloc|assertion" "$slog" 2>/dev/null; then
                    say "arm=$arm FAILED TO LOAD:"
                    grep -iE "error|failed to allocate|out of memory|cudaMalloc|assertion" "$slog" | head -5 | tee -a "$LOG"
                    break
                fi
                # Say something if this is dragging, so a stuck wait is visible rather than silent.
                if [ $((i % 30)) -eq 0 ]; then
                    say "  ...still waiting for arm=$arm ($((i * 2))s); last log line:"
                    tail -1 "$slog" | tee -a "$LOG"
                fi
                sleep 2
            done

            if [ "$loaded" = 1 ]; then
                # BEFORE the benchmark, not after: bench_spec.py reads this file to learn which
                # arm it is measuring, and exits 2 if it is missing or stale. An earlier version
                # wrote it afterwards, on the reasoning that the log had to exist first - but
                # the log's existence is what `loaded=1` has just established, and writing the
                # label late meant every single cell died with "no state file".
                printf '{"arm": "%s", "pair": %s, "workload": "%s"}\n' \
                    "$arm" "$pair" "$workload" > "$WORK/spec_state.json"

                python3 scripts/bench_spec.py \
                    --url "http://127.0.0.1:$PORT" \
                    --state "$WORK/spec_state.json" \
                    --log "$slog" \
                    --require "$require" \
                    --forbid "$forbid" \
                    --prompt-file "$prompt" \
                    --workload "$workload" \
                    --n "$REQS" \
                    --n-predict 128 \
                    --pair "$pair" \
                    2>&1 | tee -a "$LOG"
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
