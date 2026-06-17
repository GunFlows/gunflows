#!/bin/bash
# CPU-only variant of run_sample_throughput.sh: run NF sampling + NF density
# evaluation + LH evaluation entirely on CPU (device=cpu), so every metric is in
# CPU-hours. No GPU is requested. Threads = allocated CPUs.
#SBATCH --job-name=sthr_cpu
#SBATCH --partition=shared-cpu,public-cpu,private-dpnc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sthr_cpu_%j.out
#SBATCH --error=logs/sthr_cpu_%j.err
#SBATCH --mail-type=ALL

#set -euo pipefail

module load apptainer 2>/dev/null || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

HOST_REPO="/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev"
MATHIAS_REPO="/home/shares/sanchezf/gundam_n_flow/GuNFlows"
GUNDAM_DEV="/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev"
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

# No --nv (CPU only); device=cpu forced at the end so it overrides any override.
srun --ntasks=1 apptainer exec \
  --env PYTHONNOUSERSITE=1 \
  --env OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  --env PYTHONPATH="/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows" \
  --bind "${HOST_REPO}:${IN_CONTAINER_WORKDIR}" \
  --bind "${GUNDAM_DEV}:/workspace/gunflows_dev" \
  --bind "${MATHIAS_REPO}:/workspace/work/GuNFlows_M" \
  --bind "${HOST_CONFIG}:/workspace/config" \
  --bind "${HOST_DATA}:/workspace/data" \
  --pwd "${IN_CONTAINER_WORKDIR}" \
  "${SIF}" bash -lc "source '${IN_CONTAINER_SETUP}' && \
                     HYDRA_FULL_ERROR=1 python -s -m gunflows.sample_throughput ${EXTRA_ARGS} device=cpu"

echo "Job ended at $(date)"
