#!/bin/bash
#SBATCH --signal=B:USR1@300
# Array job submitted by scripts/launch.py, which passes partition, time, gres, array, etc.:
#   sbatch [options] slurm/run_array.sbatch <list file> <runs per task K>
# Task i runs lines [i*K, (i+1)*K) of the list (run config paths) concurrently on its GPU. Near the time
# limit (USR1) or on preemption (TERM) the runs get SIGTERM, save latest.pt and exit with code 3; the
# job then requeues itself if a run is unfinished, and train.py resumes from latest.pt.
LIST=${1:-$LIST}
K=${2:-${K:-1}}
source "${OPTSPLIT_CODE:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "$OPTSPLIT_CODE"

runs=()
pids=()

unfinished() {
    for cfg in "${runs[@]}"; do
        [ -e "$(dirname "$cfg")/done" ] || return 0
    done
    return 1
}

stop() {
    echo "signal received, stopping runs"
    kill -TERM "${pids[@]}" 2>/dev/null
    wait
    if unfinished; then scontrol requeue "$SLURM_JOB_ID"; fi
    exit 0
}
trap stop USR1 TERM

mapfile -t lines < "$LIST"
for cfg in "${lines[@]:$((SLURM_ARRAY_TASK_ID * K)):$K}"; do
    if [ -e "$(dirname "$cfg")/done" ]; then
        echo "skip (done): $cfg"
        continue
    fi
    echo "start: $cfg"
    python -m optsplit.train "$cfg" >> "$(dirname "$cfg")/train.out" 2>&1 &
    runs+=("$cfg")
    pids+=($!)
done

requeue=0
failed=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}"
    rc=$?
    echo "exit $rc: ${runs[$i]}"
    if [ $rc -eq 3 ]; then requeue=1; elif [ $rc -ne 0 ]; then failed=1; fi
done
if [ $requeue -eq 1 ]; then scontrol requeue "$SLURM_JOB_ID"; fi
exit $failed
