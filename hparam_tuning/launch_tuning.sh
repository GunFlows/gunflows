#!/bin/bash
#SBATCH --job-name=optuna_loop
#SBATCH --partition=private-dpnc-cpu
#SBATCH --time=168:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/shares/sanchezf/gundam_n_flow/GuNFlows/hparam_tuning/logs/loop_%j.log
#SBATCH --error=/home/shares/sanchezf/gundam_n_flow/GuNFlows/hparam_tuning/logs/loop_%j.log

set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 STUDY STAGES NWORKERS EXPERIMENT" >&2
    exit 1
fi

STUDY=$1        # study name
STAGES=$2       # number of stages
NWORKERS=$3     # number of workers in parallel per stage
EXPERIMENT=$4   # hydra experiment config name (e.g. demonstrator_100plus10_fakedata)

DIR=/home/shares/sanchezf/gundam_n_flow/GuNFlows/hparam_tuning
cd "$DIR" || exit 1

# Per-study DB/working directory
DB_BASE="$DIR/databases/$STUDY"

mkdir -p "$DIR/logs" "$DB_BASE/tmp_dbs"
TRIALS_PER_JOB=1   

for ((s=1; s<=STAGES; s++)); do
    STAGE="s${s}"

    JOBS=$NWORKERS
    MAXP=$NWORKERS
    ARRAY="0-$((JOBS-1))%$MAXP"

    echo "Launching stage ${STAGE}: STUDY=${STUDY}, JOBS=${JOBS}, TRIALS_PER_JOB=${TRIALS_PER_JOB}, MAXP=${MAXP}"

    JOBID=$(sbatch --array="$ARRAY" \
        --export=STUDY="$STUDY",STAGE="$STAGE",TRIALS_PER_JOB="$TRIALS_PER_JOB",EXPERIMENT="$EXPERIMENT" \
        "$DIR/run_array.sh" | awk '{print $4}')

    echo "Stage ${STAGE}: submitted array job ${JOBID}"

    # Wait for all workers of this stage to drop their flag in DB_BASE/tmp_dbs.
    # Also break if all SLURM tasks have exited (handles cancellations that
    # may not produce a done flag, e.g. SIGKILL before cleanup runs).
    while [ "$(ls "$DB_BASE"/tmp_dbs/done_${STUDY}_${STAGE}_*.flag 2>/dev/null | wc -l)" -lt "$JOBS" ]; do
        if ! squeue --job "$JOBID" --noheader 2>/dev/null | grep -q .; then
            echo "Stage ${STAGE}: all SLURM tasks for job ${JOBID} have exited (some may have been cancelled)."
            break
        fi
        sleep 20
    done

    # Merge stage DBs into master DB and run diagnostics (called inside merge_stage.py)
    flock "$DB_BASE/tmp_dbs/.merge.lock" python merge_stage.py "$STUDY" "$STAGE"

    # Clean flags for this stage
    rm -f "$DB_BASE"/tmp_dbs/done_${STUDY}_${STAGE}_*.flag
done
