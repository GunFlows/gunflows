#!/bin/bash
#SBATCH --job-name=gunflows-mcmc
#SBATCH --partition=private-dpnc-cpu
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load apptainer 2>/dev/null || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

HOST_REPO="/home/shares/sanchezf/gundam_n_flow/GuNFlows"
HOST_CONFIG="/srv/beegfs/scratch/groups/dpnc/neutrinos"
HOST_DATA="/srv/beegfs/scratch/shares/sanchezf/gundam_n_flow/tmp_inputs/nextcloud"
SIF="/home/shares/sanchezf/gundam_n_flow/GuNFlows/env/containers/ml_image2.sif"

IN_CONTAINER_WORKDIR="/workspace/work/GuNFlows"
IN_CONTAINER_SETUP="${IN_CONTAINER_WORKDIR}/setup_nosubshell.sh"

EXTRA_ARGS=""
if [ "$#" -gt 0 ]; then
  EXTRA_ARGS="$(printf ' %q' "$@")"
fi

srun --ntasks=1 apptainer exec --nv \
  --env PYTHONNOUSERSITE=1 \
  --env DATA_DIR="/workspace/data/DatasetNFlowsOA2022/Asimov/allParameters" \
  --env PYTHONPATH="/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows" \
  --bind "${HOST_REPO}:${IN_CONTAINER_WORKDIR}" \
  --bind "${HOST_CONFIG}:/workspace/config" \
  --bind "${HOST_DATA}:/workspace/data" \
  --pwd "${IN_CONTAINER_WORKDIR}" \
  "${SIF}" bash -lc "source '${IN_CONTAINER_SETUP}' && \
                     HYDRA_FULL_ERROR=1 python -s -m gunflows.mcmc${EXTRA_ARGS}"
