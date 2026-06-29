#!/bin/bash
#SBATCH --job-name=sthr-dim-sweep
#SBATCH --partition=shared-gpu,shared-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=ALL

set -euo pipefail

module load apptainer 2>/dev/null || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

HOST_REPO="/home/shares/sanchezf/gundam_n_flow/GuNFlows"
HOST_CONFIG="/home/shares/sanchezf/gundam_n_flow/common_gundam_workspace"
HOST_DATA="/home/shares/sanchezf/gundam_n_flow/common_gundam_workspace/DATA"
GUNFLOWS_DEV="/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev"
SIF="/home/shares/sanchezf/gundam_n_flow/GuNFlows/env/containers/ml_image2.sif"

IN_CONTAINER_WORKDIR="/workspace/work/GuNFlows"
IN_CONTAINER_SETUP="${IN_CONTAINER_WORKDIR}/setup_nosubshell.sh"

# Pass device as first argument: cuda (default) or cpu
DEVICE="${1:-cuda}"

NDIMS_ARRAY=(20 30 60 100)
NDIMS="${NDIMS_ARRAY[${SLURM_ARRAY_TASK_ID}]}"
TRAINING_FOLDER="/workspace/config/Config100p10_noDet/dim_sweep/top1_last${NDIMS}dims"
SAVE_DIR="${TRAINING_FOLDER}/sample_throughput_${DEVICE}"

echo "Job started at $(date) — array task ${SLURM_ARRAY_TASK_ID}, last ${NDIMS} dims, device=${DEVICE}"

NV_FLAG=""
[[ "${DEVICE}" == "cuda" ]] && NV_FLAG="--nv"

srun --ntasks=1 apptainer exec ${NV_FLAG} \
  --env PYTHONNOUSERSITE=1 \
  --env OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  --env PYTHONPATH="/workspace/work/GuNFlows/env/python_packages:/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows" \
  --bind "${HOST_REPO}:${IN_CONTAINER_WORKDIR}" \
  --bind "${HOST_CONFIG}:/workspace/config" \
  --bind "${HOST_DATA}:/workspace/data" \
  --bind "${GUNFLOWS_DEV}:/workspace/gunflows_dev" \
  --pwd "${IN_CONTAINER_WORKDIR}" \
  "${SIF}" bash -c "source '${IN_CONTAINER_SETUP}' && \
                     HYDRA_FULL_ERROR=1 python -s -m gunflows.sample_throughput \
                     training_folder=${TRAINING_FOLDER} \
                     save_dir=${SAVE_DIR} \
                     device=${DEVICE}"

echo "Job ended at $(date)"
