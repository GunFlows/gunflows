#!/bin/env bash
#SBATCH --job-name=sample_fds
#SBATCH --partition=private-dpnc-cpu,shared-cpu
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16GB
#SBATCH --output=logs/sample_fds_%A_%a.out
#SBATCH --error=logs/sample_fds_%A_%a.err
#SBATCH --mail-type=ALL

index=${SLURM_ARRAY_TASK_ID}
time=$(date +%s%N)
seed=$(($index + $time))
CONFIG_FILE="output/FDSOA2021/fdsFit_Martini1pi.root"

OUTPUT_FOLDER="oa2022_fds_martini"
mkdir -p ${OUTPUT_FOLDER}
OUTPUT_FILE="${OUTPUT_FOLDER}/batch${index}.npz"
N=50000


SCRIPTARGS="-m gunflows.make_initial_dataset -o ${OUTPUT_FILE} -c ${CONFIG_FILE} -n ${N} -t 16 " #-s ${seed}

APPTAINER_OPTIONS="--nv --cleanenv \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH=/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows \
  --bind /home/shares/sanchezf/gundam_n_flow/GuNFlows_dev:/workspace/work/GuNFlows \
  --bind /srv/beegfs/scratch/groups/dpnc/neutrinos:/workspace/config \
  --bind /srv/beegfs/scratch/shares/sanchezf/gundam_n_flow/tmp_inputs/nextcloud/:/workspace/data \
  --pwd /workspace/config/GundamInputOA2021"

IMAGE_PATH="/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev/env/containers/ml_image2.sif"


echo "Starting job: " `date`

apptainer exec ${APPTAINER_OPTIONS} ${IMAGE_PATH} bash -c "source /workspace/work/GuNFlows/setup_nosubshell.sh; python ${SCRIPTARGS}" 

echo "Job done: " `date`
