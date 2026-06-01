#!/bin/bash
#SBATCH --job-name=gundam-mcmc-toyfit
#SBATCH --partition=private-dpnc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/gundam_mcmc_toyfit_%j.out
#SBATCH --error=logs/gundam_mcmc_toyfit_%j.err
#SBATCH --mail-type=ALL

# Run gundamFitter in MCMC mode on the ToyNDFit fake-data (non-Asimov) config.
# Bindings mirror predict_histograms.sh so the LH is identical to the one the NF trained on.
# Output ROOT goes to common_gundam_workspace (consistent with run_toy_mcmc.sh convention).
#
# Env-var overrides (export before sbatch):
#   CYCLES=1  STEPS=20000  (defaults)
export CYCLES=5
export STEPS=1000000

set -euo pipefail

module load apptainer 2>/dev/null || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

HOST_GUNFLOWS="/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev"
HOST_CONFIG="/home/shares/sanchezf/gundam_n_flow/ToyNDFit"
HOST_DATA="/home/shares/sanchezf/gundam_n_flow/ToyNDFit/DATA"
HOST_OUT="/home/shares/sanchezf/gundam_n_flow/common_gundam_workspace"
SIF="/home/shares/sanchezf/gundam_n_flow/GuNFlows/env/containers/ml_image2.sif"

IN_GUNFLOWS="/workspace/work/GuNFlows"
IN_SETUP="${IN_GUNFLOWS}/setup_nosubshell.sh"
IN_CONFIG="/workspace/config/GundamWorkspace"
IN_DATA="/workspace/data"
IN_OUT="/workspace/output"

CYCLES="${CYCLES:-1}"
STEPS="${STEPS:-20000}"

OUTFILE_HOST="${HOST_OUT}/DemoMCMC_100p10_noasimov_${SLURM_JOB_ID}.root"
OUTFILE_CONT="${IN_OUT}/DemoMCMC_100p10_noasimov_${SLURM_JOB_ID}.root"

mkdir -p "${HOST_OUT}"

echo "Job started at $(date)"
echo "Cycles: ${CYCLES}  Steps: ${STEPS}"
echo "Output: ${OUTFILE_HOST}"

srun --ntasks=1 apptainer exec \
  --cleanenv \
  --env PYTHONNOUSERSITE=1 \
  --env DATASET_FOLDER="${IN_DATA}" \
  --env OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  --bind "${HOST_GUNFLOWS}:${IN_GUNFLOWS}" \
  --bind "${HOST_CONFIG}:/workspace/config" \
  --bind "${HOST_DATA}:${IN_DATA}:ro" \
  --bind "${HOST_OUT}:${IN_OUT}" \
  --pwd "${IN_CONFIG}" \
  "${SIF}" bash -lc "
    set -euo pipefail
    source '${IN_SETUP}'
    gundamFitter \
      -c '${IN_CONFIG}/configToyOA_100plus10.yaml' \
      -of '${IN_CONFIG}/configMcmcToy.yaml' \
      -t ${OMP_NUM_THREADS} \
      -o '${OUTFILE_CONT}' \
      -O '/fitterEngineConfig/minimizerConfig/cycles=${CYCLES}' \
      -O '/fitterEngineConfig/minimizerConfig/steps=${STEPS}'
  "

echo "Job ended at $(date)"
echo "Output ROOT: ${OUTFILE_HOST}"
