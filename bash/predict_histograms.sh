#!/bin/bash
#SBATCH --job-name=sample_nf_mcmc_toy
#SBATCH --partition=shared-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=COMPUTE_TYPE_AMPERE
#SBATCH --output=logs/predict_hists%j.out
#SBATCH --error=logs/predict_hists%j.err
#SBATCH --mail-type=ALL

set -euo pipefail

module load apptainer 2>/dev/null || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

HOST_REPO="/home/shares/sanchezf/gundam_n_flow/GuNFlows"
# HOST_CONFIG="/srv/beegfs/scratch/groups/dpnc/neutrinos" # belong to OA config
# HOST_DATA="/srv/beegfs/scratch/shares/sanchezf/gundam_n_flow/tmp_inputs/nextcloud" # belong to OA config
HOST_CONFIG="/home/shares/sanchezf/gundam_n_flow/common_gundam_workspace"
# HOST_CONFIG="/home/shares/sanchezf/gundam_n_flow/common_gundam_workspace_2"
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
  --env PYTHONPATH="/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows" \
  --bind "${HOST_REPO}:${IN_CONTAINER_WORKDIR}" \
  --bind "${HOST_CONFIG}:/workspace/config" \
  --bind "${HOST_DATA}:/workspace/data" \
  --bind "${GUNFLOWS_DEV}:/workspace/gunflows_dev" \
  --bind "${trained_models}:/workspace/trained_models" \
  --pwd "${IN_CONTAINER_WORKDIR}" \
  "${SIF}" bash -c "source '${IN_CONTAINER_SETUP}' && \
                     HYDRA_FULL_ERROR=1 python -s -m gunflows.predict_histograms \
                     --config-path ${IN_CONTAINER_WORKDIR}/configs \
                     --config-name predict_histograms \
                     ${EXTRA_ARGS}"

echo "Job ended at $(date)"
