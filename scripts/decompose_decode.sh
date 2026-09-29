#!/usr/bin/env bash
# Decompose one decode phase into GPU duty cycle and CPU duty cycle.
#
# Cross-process A/B on this box swings 11-28 t/s on identical config, because the 12.19 GiB
# expert working set is only sometimes resident in an ~11 GiB page cache. That variance is
# larger than every effect we want to measure. So instead of comparing runs, instrument a
# single run: sample the GPU's own utilisation counter and the process's CPU time over the
# same wall-clock window and report both duty cycles.
#
# The ratio is the robust part. It does not care whether the page cache was warm, because
# both sides are measured over the same window of the same process.
set -u

MODEL="${MODEL:-/home/e0u/localhost/models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf}"
PROMPT="${PROMPT:-/tmp/opencode/p128k.txt}"
NPREDICT="${NPREDICT:-96}"
CTX="${CTX:-200000}"
# Sample period. Fine enough to see decode, coarse enough that nvidia-smi is not the thing
# being measured.
PERIOD="${PERIOD:-0.25}"

JSON=/tmp/opencode/decompose.json
LOG=/tmp/opencode/decompose.log

./build/bin/amp-infer --model "$MODEL" --prompt-file "$PROMPT" --ctx "$CTX" \
    --n-predict "$NPREDICT" --json >"$JSON" 2>"$LOG" &
PID=$!

SAMPLES=/tmp/opencode/decompose.samples
: >"$SAMPLES"

# One row per sample: elapsed seconds, GPU util percent, process CPU ticks (utime+stime,
# summed over threads, so busy can exceed wall time on a multicore process).
#
# utilization.memory is sampled alongside utilization.gpu because the two answer different
# questions. utilization.gpu is the fraction of time any kernel is resident, which stays
# high for a kernel that is latency-bound on tiny work. utilization.memory is the fraction
# of peak DRAM bandwidth achieved, which is the quantity that actually bounds this model.
# A high gpu number with a low memory number is a kernel-occupancy problem, not a
# bandwidth problem, and the two call for opposite fixes.
start=$(date +%s.%N)
while kill -0 "$PID" 2>/dev/null; do
    now=$(date +%s.%N)
    read -r util mutil < <(nvidia-smi \
        --query-gpu=utilization.gpu,utilization.memory \
        --format=csv,noheader,nounits 2>/dev/null | head -1 | tr ',' ' ')
    ticks=$(awk '{print $14+$15}' "/proc/$PID/stat" 2>/dev/null)
    if [ -n "$util" ] && [ -n "$ticks" ]; then
        printf '%s %s %s %s\n' "$now" "$util" "$mutil" "$ticks" >>"$SAMPLES"
    fi
    sleep "$PERIOD"
done
wait "$PID" 2>/dev/null

python3 - "$JSON" "$SAMPLES" "$start" <<'PY'
import json, sys

with open(sys.argv[1]) as f:
    d = json.load(f)

rows = []
with open(sys.argv[2]) as f:
    for line in f:
        parts = line.split()
        if len(parts) == 4:
            rows.append((float(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
start = float(sys.argv[3])

tps = d['decode_tps']
print(f"  prompt tokens    {d['prompt_tokens']}")
print(f"  decode           {tps:.2f} t/s  ({1000/tps:.1f} ms/token)")
print()

if len(rows) < 3:
    print("  too few samples to decompose")
    sys.exit(0)

# Analyse only the tail, after the load and prefill have finished and decode is steady.
# The head of the run is dominated by faulting the model in and is not representative.
tail = rows[max(1, int(len(rows) * 0.6)):]
span = tail[-1][0] - tail[0][0]
if span <= 0:
    print("  zero-length window")
    sys.exit(0)

utils = sorted(r[1] for r in tail)
mem = sorted(r[2] for r in tail)
cpu_ticks = tail[-1][3] - tail[0][3]

# 100 Hz USER_HZ, so ticks/100 is CPU seconds. nproc logical cores is the denominator.
import os
ncpu = os.cpu_count() or 8

med = lambda v: v[len(v)//2]

print(f"  sampled window   {span:.1f} s, {len(tail)} samples (tail 40% of run)")
print(f"  GPU resident     median {med(utils):3d}%   mean {sum(utils)/len(utils):5.1f}%")
print(f"  GPU bandwidth    median {med(mem):3d}% of peak   mean {sum(mem)/len(mem):5.1f}%")
print(f"  CPU duty cycle   {cpu_ticks/100.0:6.1f} CPU-s / {span:.1f} wall-s"
      f"  = {cpu_ticks/100.0/span:.2f} cores of {ncpu}"
      f"  ({cpu_ticks/100.0/span/ncpu*100:.0f}%)")
print()
busy = cpu_ticks/100.0/span
print(f"  cores busy       {busy:.2f} of {ncpu}")
if busy < 0.5:
    print("  -> the CPU is nearly idle. It is waiting on the GPU, not the other way round.")
print()
if med(utils) > 50 and med(mem) < 40:
    print("  -> GPU kernels are resident but DRAM is idle: the GPU is latency-bound on small")
    print("     work, not bandwidth-bound. Faster kernels, not more of them, is the lever.")
PY
