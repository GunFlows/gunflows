#!/bin/bash
#SBATCH --job-name=paper_plots
#SBATCH --partition=shared-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=COMPUTE_TYPE_AMPERE
#SBATCH --output=logs/paper_plots%j.out
#SBATCH --error=logs/paper_plots%j.err
#SBATCH --mail-type=ALL

set -euo pipefail

module load apptainer 2>/dev/null || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

HOST_REPO="/home/shares/sanchezf/gundam_n_flow/GuNFlows"
HOST_CONFIG="/home/shares/sanchezf/gundam_n_flow/common_gundam_workspace"
HOST_DATA="/home/shares/sanchezf/gundam_n_flow/common_gundam_workspace/DATA"
trained_models="/home/shares/sanchezf/gundam_n_flow/common_gundam_workspace/Config100p10_noDet"
SIF="/home/shares/sanchezf/gundam_n_flow/GuNFlows/env/containers/ml_image2.sif"
GUNFLOWS_DEV="/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev"

IN_CONTAINER_WORKDIR="/workspace/work/GuNFlows"
IN_CONTAINER_SETUP="${IN_CONTAINER_WORKDIR}/setup_nosubshell.sh"

EXTRA_ARGS=""
if [ "$#" -gt 0 ]; then
  EXTRA_ARGS="$(printf ' %q' "$@")"
fi

echo "Job started at $(date)"

srun --ntasks=1 apptainer exec --nv \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH="/workspace/work/GuNFlows/env/python_packages:/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows" \
  --bind "${HOST_REPO}:${IN_CONTAINER_WORKDIR}" \
  --bind "${HOST_CONFIG}:/workspace/config" \
  --bind "${HOST_DATA}:/workspace/data" \
  --bind "${GUNFLOWS_DEV}:/workspace/gunflows_dev" \
  --bind "${trained_models}:/workspace/trained_models" \
  --pwd "${IN_CONTAINER_WORKDIR}" \
  "${SIF}" bash -c "source '${IN_CONTAINER_SETUP}' && \
                     HYDRA_FULL_ERROR=1 python -s -m gunflows.make_paper_plots \
                     --config-path ${IN_CONTAINER_WORKDIR}/configs \
                     --config-name make_paper_plots \
                     ${EXTRA_ARGS}"

echo "Job ended at $(date)"
