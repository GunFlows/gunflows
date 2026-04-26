#!/bin/bash
#SBATCH --job-name=gundam_mcmc_toy
#SBATCH --partition=private-dpnc-cpu,public-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev/demonstrator_scripts/logs/gundam_mcmc_%j.out
#SBATCH --error=/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev/demonstrator_scripts/logs/gundam_mcmc_%j.err

set -euo pipefail

module load apptainer 2>/dev/null || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

BASE="/home/shares/sanchezf/gundam_n_flow/ToyNDFit"
HOST_CONFIG="${BASE}/GundamWorkspace"
HOST_DATA="${BASE}/DATA"
HOST_OUT="${BASE}/outputs"
HOST_LOGS="${BASE}/logs"

HOST_GUNFLOWS="/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev"
SIF="/home/shares/sanchezf/gundam_n_flow/GuNFlows/env/containers/ml_image2.sif"

IN_CONFIG="/workspace/config"
IN_DATA="/workspace/data"
IN_OUT="/workspace/output"

IN_GUNFLOWS="/workspace/work/GuNFlows"
IN_SETUP="${IN_GUNFLOWS}/setup_nosubshell.sh"

BASE_CONFIG="configToyOA_60plus6.yaml"
MCMC_OVERRIDE="configMcmcToy.yaml"

SEED="6"

OUTROOT="Demo_mcmc_ToySeed_${SEED}_${SLURM_JOB_ID}.root"
#OUTROOT="asimov_mcmc_local.root"

EXTRA_ARGS=""
if [ "$#" -gt 0 ]; then
  EXTRA_ARGS="$(printf ' %q' "$@")"
fi

mkdir -p "${HOST_OUT}" "${HOST_LOGS}"

srun --ntasks=1 apptainer exec \
  --cleanenv \
  --env PYTHONNOUSERSITE=1 \
  --env DATASET_FOLDER="${IN_DATA}" \
  --env OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  --bind "${HOST_GUNFLOWS}:${IN_GUNFLOWS}" \
  --bind "${HOST_CONFIG}:${IN_CONFIG}" \
  --bind "${HOST_DATA}:${IN_DATA}:ro" \
  --bind "${HOST_OUT}:${IN_OUT}" \
  --pwd "${IN_CONFIG}" \
  "${SIF}" bash -lc "\
    set -euo pipefail; \
    source '${IN_SETUP}'; \
    gundamFitter \
      -c '${IN_CONFIG}/${BASE_CONFIG}' \
      -of '${IN_CONFIG}/${MCMC_OVERRIDE}' \
      --toy \
      -s 6 \
      -t 10 \
      -o '${IN_OUT}/${OUTROOT}' \
      ${EXTRA_ARGS} \
  "

echo
echo "Job ended at $(date)"
echo "Output ROOT: ${HOST_OUT}/${OUTROOT}"
