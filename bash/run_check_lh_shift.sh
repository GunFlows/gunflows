#!/bin/bash
#SBATCH --job-name=check_lh_shift
#SBATCH --partition=shared-cpu,private-dpnc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/check_lh_shift_%j.out
#SBATCH --error=logs/check_lh_shift_%j.err

set -euo pipefail

module load apptainer 2>/dev/null || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

HOST_REPO="/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev"
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

srun --ntasks=1 apptainer exec \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH="/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows" \
  --bind "${HOST_REPO}:${IN_CONTAINER_WORKDIR}" \
  --bind "${HOST_CONFIG}:/workspace/config" \
  --bind "${HOST_DATA}:/workspace/data" \
  --bind "${trained_models}:/workspace/trained_models" \
  --pwd "${IN_CONTAINER_WORKDIR}" \
  "${SIF}" bash -lc "source '${IN_CONTAINER_SETUP}' && \
                     HYDRA_FULL_ERROR=1 python -s -m gunflows.check_lh_shift \
                     --config-path ${IN_CONTAINER_WORKDIR}/configs \
                     --config-name sample_mcmc_nf_toyOA \
                     hydra.run.dir=/workspace/trained_models/check_lh_shift \
                     hydra.output_subdir=null \
                     ${EXTRA_ARGS}"
