#!/bin/bash
#SBATCH --job-name=sample_nf_mcmc_toy
#SBATCH --partition=private-dpnc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=COMPUTE_TYPE_AMPERE
#SBATCH --output=logs/sample_nf_mcmc_toy_%j.out
#SBATCH --error=logs/sample_nf_mcmc_toy_%j.err
#SBATCH --mail-type=ALL

set -euo pipefail

module load apptainer 2>/dev/null || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

HOST_REPO="/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev"
# HOST_CONFIG="/srv/beegfs/scratch/groups/dpnc/neutrinos" # belong to OA config
# HOST_DATA="/srv/beegfs/scratch/shares/sanchezf/gundam_n_flow/tmp_inputs/nextcloud" # belong to OA config
# HOST_CONFIG="/home/shares/sanchezf/gundam_n_flow/ToyNDFit_dev"
HOST_CONFIG="/home/shares/sanchezf/gundam_n_flow/common_gundam_workspace_2"
HOST_DATA="/home/shares/sanchezf/gundam_n_flow/ToyNDFit/DATA"
trained_models="/home/shares/sanchezf/gundam_n_flow/trained_models"
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
  --env PYTHONPATH="/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows" \
  --bind "${HOST_REPO}:${IN_CONTAINER_WORKDIR}" \
  --bind "${HOST_CONFIG}:/workspace/config" \
  --bind "${HOST_DATA}:/workspace/data" \
  --bind "${trained_models}:/workspace/trained_models" \
  --pwd "${IN_CONTAINER_WORKDIR}" \
  "${SIF}" bash -lc "source '${IN_CONTAINER_SETUP}' && \
                     HYDRA_FULL_ERROR=1 python -s -m gunflows.sample_mcmc_toy \
                     --config-path ${IN_CONTAINER_WORKDIR}/configs \
                     --config-name sample_mcmc_nf_toyOA \
                     ${EXTRA_ARGS}"

echo "Job ended at $(date)"
