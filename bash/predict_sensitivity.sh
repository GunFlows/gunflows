#!/bin/bash
#SBATCH --job-name=pred_sensitivity
#SBATCH --partition=shared-gpu,private-dpnc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=COMPUTE_TYPE_AMPERE
#SBATCH --output=/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev/bash/logs/pred_sensitivity%j.out
#SBATCH --error=/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev/bash/logs/pred_sensitivity%j.err
#SBATCH --mail-type=ALL

set -euo pipefail

module load apptainer 2>/dev/null || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

HOST_REPO="/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev"
HOST_CONFIG="/home/shares/sanchezf/gundam_n_flow/ToyNDFit"
HOST_DATA="/home/shares/sanchezf/gundam_n_flow/ToyNDFit/DATA"
HOST_MODELS="/home/shares/sanchezf/gundam_n_flow/trained_models"
SIF="/home/shares/sanchezf/gundam_n_flow/GuNFlows/env/containers/ml_image2.sif"

# Container mount points
IN_WORKDIR="/workspace/work/GuNFlows"
IN_SETUP="${IN_WORKDIR}/setup_nosubshell.sh"

# ── Hydra overrides (use full in-container paths) ──────────────────────────
TRAINING_FOLDER="/workspace/work/GuNFlows/outputs/2026-05-28/17-30-29"
SAVE_DIR="${TRAINING_FOLDER}/predict_sensitivity"

OVERRIDES=(
  "training_folder=${TRAINING_FOLDER}"
  "save_dir=${SAVE_DIR}"
  "num_samples=300"
  "batch_size=256"
  "threads=${SLURM_CPUS_PER_TASK:-24}"
  "device=cuda"
  "save_every=50"
  "grid.sin2_theta23_n=51"
  "grid.dm2_32_n=51"
  "sensitivity_chunk_size=500"
)

echo "Job started at $(date)"
echo "Training folder : ${TRAINING_FOLDER}"
echo "Save dir        : ${SAVE_DIR}"

srun --ntasks=1 apptainer exec --nv \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH="${IN_WORKDIR}/src:${IN_WORKDIR}/src/normalizing-flows" \
  --bind "${HOST_REPO}:${IN_WORKDIR}" \
  --bind "${HOST_CONFIG}:/workspace/config" \
  --bind "${HOST_DATA}:/workspace/data" \
  --bind "${HOST_MODELS}:/workspace/trained_models" \
  --pwd "${IN_WORKDIR}" \
  "${SIF}" bash -c "source '${IN_SETUP}' && \
    HYDRA_FULL_ERROR=1 python -s -m gunflows.predict_sensitivity \
    --config-path ${IN_WORKDIR}/configs \
    --config-name predict_sensitivity \
    ${OVERRIDES[*]}"

echo "Job ended at $(date)"
