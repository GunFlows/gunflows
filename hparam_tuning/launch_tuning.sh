#!/bin/bash
#SBATCH --job-name=optuna_loop
#SBATCH --partition=shared-cpu
#SBATCH --time=6:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/shares/sanchezf/gundam_n_flow/GuNFlows/hparam_tuning/logs/loop_%j.log
#SBATCH --error=/home/shares/sanchezf/gundam_n_flow/GuNFlows/hparam_tuning/logs/loop_%j.log

STUDY=$1
STAGES=$2
JOBS=$3
TRIALS=$4
MAXP=$5

DIR=/home/shares/sanchezf/gundam_n_flow/GuNFlows/hparam_tuning
cd "$DIR" || exit 1

# Per-study DB base directory
DB_BASE="$DIR/databases/$STUDY"

mkdir -p "$DIR/logs" "$DB_BASE/tmp_dbs"

for ((s=1; s<=STAGES; s++)); do
    STAGE="s${s}"
    ARRAY="0-$(($JOBS-1))%$MAXP"

    JOBID=$(sbatch --array="$ARRAY" \
          --export=STUDY=$STUDY,STAGE=$STAGE,TRIALS_PER_JOB=$TRIALS \
          "$DIR/run_array.sh" | awk '{print $4}')

    # Wait for all array tasks of this stage to drop their flag in DB_BASE/tmp_dbs
    while [ $(ls "$DB_BASE"/tmp_dbs/done_${STUDY}_${STAGE}_*.flag 2>/dev/null | wc -l) -lt $JOBS ]; do
        sleep 20
    done

    flock "$DB_BASE/tmp_dbs/.merge.lock" python merge_stage.py "$STUDY" "$STAGE"
    rm -f "$DB_BASE"/tmp_dbs/done_${STUDY}_${STAGE}_*.flag
done
