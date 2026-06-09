#!/bin/bash
#SBATCH --job-name=ess
#SBATCH --partition=shared-gpu,private-dpnc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=16G
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:1,VramPerGpu:24G
#SBATCH --constraint=COMPUTE_TYPE_AMPERE
#SBATCH --output=logs/ess_%j.out
#SBATCH --error=logs/ess_%j.err
#SBATCH --mail-type=ALL

#set -euo pipefail

module load apptainer 2>/dev/null || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
   
HOST_REPO="/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev"
MATHIAS_REPO="/home/shares/sanchezf/gundam_n_flow/GuNFlows"
HOST_CONFIG="/home/shares/sanchezf/gundam_n_flow/common_gundam_workspace"
HOST_DATA="/home/shares/sanchezf/gundam_n_flow/ToyNDFit/DATA"
SIF="/home/shares/sanchezf/gundam_n_flow/GuNFlows/env/containers/ml_image2.sif"

IN_CONTAINER_WORKDIR="/workspace/work/GuNFlows"
IN_CONTAINER_SETUP="${IN_CONTAINER_WORKDIR}/setup_nosubshell.sh"

EXTRA_ARGS=""
if [ "$#" -gt 0 ]; then
  EXTRA_ARGS="$(printf ' %q' "$@")"
fi

echo "Job started at $(date)"

srun --ntasks=1 apptainer exec --nv \
  --env PYTHONNOUSERSITE=1 \
  --env OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  --env PYTHONPATH="/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows" \
  --bind "${HOST_REPO}:${IN_CONTAINER_WORKDIR}" \
  --bind "${MATHIAS_REPO}:/workspace/work/GuNFlows_M" \
  --bind "${HOST_CONFIG}:/workspace/config" \
  --bind "${HOST_DATA}:/workspace/data" \
  --pwd "${IN_CONTAINER_WORKDIR}" \
  "${SIF}" bash -lc "source '${IN_CONTAINER_SETUP}' && \
                     HYDRA_FULL_ERROR=1 python -s -m gunflows.effective_sample_size ${EXTRA_ARGS}"

echo "Job ended at $(date)"
