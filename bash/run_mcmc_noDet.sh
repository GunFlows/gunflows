#!/bin/bash
#SBATCH --job-name=mcmc-noDet
#SBATCH --partition=private-dpnc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=168:00:00
#SBATCH --output=logs/gundam_mcmc_noDet_%j.out
#SBATCH --error=logs/gundam_mcmc_noDet_%j.err
#SBATCH --mail-type=ALL

export CYCLES=4
export STEPS=500000

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

CONFIG_NAME="configToyOA_100plus10_noDet"
OUTFILE_HOST="${HOST_OUT}/DemoMCMC_${CONFIG_NAME}_${SLURM_JOB_ID}.root"
OUTFILE_CONT="${IN_OUT}/DemoMCMC_${CONFIG_NAME}_${SLURM_JOB_ID}.root"
mkdir -p "${HOST_OUT}"

echo "Job started at $(date) — Cycles: ${CYCLES}  Steps: ${STEPS}"
echo "Output: ${OUTFILE_HOST}"

srun --ntasks=1 apptainer exec   --cleanenv   --env PYTHONNOUSERSITE=1   --env OMP_NUM_THREADS="${OMP_NUM_THREADS}"   --bind "${HOST_GUNFLOWS}:${IN_GUNFLOWS}"   --bind "${HOST_CONFIG}:/workspace/config"   --bind "${HOST_DATA}:${IN_DATA}:ro"   --bind "${HOST_OUT}:${IN_OUT}"   --pwd "${IN_CONFIG}"   "${SIF}" bash -lc "
    set -euo pipefail
    source '${IN_SETUP}'
    gundamFitter       -c '${IN_CONFIG}/${CONFIG_NAME}.yaml'       -of '${IN_CONFIG}/configMcmcToy.yaml'       -t ${OMP_NUM_THREADS}       -o '${OUTFILE_CONT}'       -O '/fitterEngineConfig/minimizerConfig/cycles=${CYCLES}'       -O '/fitterEngineConfig/minimizerConfig/steps=${STEPS}'
  "

echo "Job ended at $(date)"
echo "Output ROOT: ${OUTFILE_HOST}"
