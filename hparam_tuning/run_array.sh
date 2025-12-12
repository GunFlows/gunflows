#!/bin/bash
#SBATCH --job-name=gunflows_optuna
#SBATCH --partition=private-dpnc-gpu,shared-gpu
#SBATCH --exclude=gpu023 
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding   
#SBATCH --gpu-bind=single:1            
#SBATCH --ntasks=1    
#SBATCH --time=12:00:00
#SBATCH --mem=72G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/shares/sanchezf/gundam_n_flow/GuNFlows/hparam_tuning/logs/%x_%A_%a.log
#SBATCH --error=/home/shares/sanchezf/gundam_n_flow/GuNFlows/hparam_tuning/logs/%x_%A_%a.log
echo "Starting array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID} at $(date)"
DIR=/home/shares/sanchezf/gundam_n_flow/GuNFlows/hparam_tuning
cd "$DIR" || exit 1

# Per-study DB base directory (must match launch_tuning)
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DONEFLAG="$DB_BASE/tmp_dbs/done_${STUDY}_${STAGE}_${SLURM_ARRAY_TASK_ID}.flag"
trap 'touch "$DONEFLAG"' EXIT

python -u worker.py

# Copy this job's DB back into the stage tmp_dbs dir for merging
cp "$TMPDB" "$DB_BASE/tmp_dbs/"
