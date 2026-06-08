#!/bin/bash
#SBATCH --job-name=gunflows_optuna
#SBATCH --partition=private-dpnc-gpu,shared-gpu
#SBATCH --exclude=gpu023 
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding   
#SBATCH --gpu-bind=single:1            
#SBATCH --ntasks=1    
#SBATCH --time=6:00:00
# Deliver SIGTERM to the *batch shell* (B:) 120s before the walltime kill,
# so _cleanup can stop the worker, let it salvage its partial loss, and copy
# the DB before SLURM sends SIGKILL.
#SBATCH --signal=B:TERM@120
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --output=/home/shares/sanchezf/gundam_n_flow/GuNFlows/hparam_tuning/logs/%x_%A_%a.log
#SBATCH --error=/home/shares/sanchezf/gundam_n_flow/GuNFlows/hparam_tuning/logs/%x_%A_%a.log
echo "Starting array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID} at $(date)"
DIR=/home/shares/sanchezf/gundam_n_flow/GuNFlows/hparam_tuning
cd "$DIR" || exit 1

DB_BASE="$DIR/databases/$STUDY"

choose_scratch () {
    if [[ -n "$SLURM_TMPDIR" && -w "$SLURM_TMPDIR" ]];          then echo "$SLURM_TMPDIR"
    elif [[ -d "$HOME/scratch" && -w "$HOME/scratch" ]];        then echo "$HOME/scratch"
    else mkdir -p "$DB_BASE/tmp_dbs/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
         && echo "$DB_BASE/tmp_dbs/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    fi
}
SCRATCH=$(choose_scratch)

MASTER_DB="$DB_BASE/${STUDY}.db"      # <— master lives under databases/STUDY
export TMPDB="$SCRATCH/${STUDY}_${STAGE}_${SLURM_ARRAY_TASK_ID}.db"

mkdir -p "$SCRATCH" || exit 1
if [[ -f "$MASTER_DB" ]]; then
    cp "$MASTER_DB" "$TMPDB"
else
    : > "$TMPDB"
fi
chmod 600 "$TMPDB"

export OPTUNA_STUDY_NAME=$STUDY
export OPTUNA_STORAGE="sqlite:///$TMPDB"
export TOTAL_TRIALS=$TRIALS_PER_JOB
export OPTUNA_EXPERIMENT=$EXPERIMENT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DONEFLAG="$DB_BASE/tmp_dbs/done_${STUDY}_${STAGE}_${SLURM_ARRAY_TASK_ID}.flag"

# _cleanup must be idempotent: EXIT fires after TERM if we call `exit` there.
_cleaned=0
_cleanup() {
    [[ $_cleaned -eq 1 ]] && return; _cleaned=1
    if [[ -n "${WORKER_PID:-}" ]]; then
        # On walltime SIGTERM the worker is still mid-trial. Forward TERM so its
        # signal handler stops the training subprocess, parses the partial
        # "Best loss", records the trial in TMPDB, and exits. (No-op on a normal
        # exit where the worker is already gone.)
        kill -TERM "$WORKER_PID" 2>/dev/null || true
        # Give it up to ~90s to drain output and commit the trial to sqlite,
        # then force-kill if it's wedged. Fits inside the 120s SIGTERM->SIGKILL
        # window from --signal=B:TERM@120.
        for _ in $(seq 1 90); do
            kill -0 "$WORKER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$WORKER_PID" 2>/dev/null || true
        wait "$WORKER_PID" 2>/dev/null || true
    fi
    cp "$TMPDB" "$DB_BASE/tmp_dbs/" 2>/dev/null || true
    touch "$DONEFLAG"
}
trap '_cleanup' EXIT
# Explicit TERM trap so bash waits for python to drain output + write DB
# before the shell exits (without this, bash dies immediately on SIGTERM).
trap '_cleanup; exit 143' TERM

python -u worker.py &
WORKER_PID=$!
wait $WORKER_PID
