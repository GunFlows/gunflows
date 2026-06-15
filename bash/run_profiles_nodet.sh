#!/bin/bash
# =============================================================================
#  run_profiles_nodet.sh -- NLL profile plots (gunflows.likelihood_sampler.lh_profiles)
#  for the noDet GUNDAM config, using the common_gundam_workspace mounts.
#  CPU only (GUNDAM); not GPU. Plots use the paper style (ylabel -log(L_p)).
#
#  Submit:  cd .../GuNFlows_dev/bash && sbatch run_profiles_nodet.sh [extra lh_profiles args]
#  Or run interactively:  srun ... bash run_profiles_nodet.sh   (or call the srun line below)
#
#  Defaults below scan all 110 params over +-3 sigma with 20 points. Extra args
#  are appended and OVERRIDE the defaults (argparse: last occurrence wins), e.g.:
#    sbatch run_profiles_nodet.sh -n 5            # fewer points
#    sbatch run_profiles_nodet.sh -x              # cross-section params only
#    sbatch run_profiles_nodet.sh -o /workspace/config/Config100p10_noDet/my_profiles
# =============================================================================
#SBATCH --job-name=profiles
#SBATCH --partition=shared-cpu,private-dpnc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/profiles_%j.out
#SBATCH --error=logs/profiles_%j.err
#SBATCH --mail-type=ALL

#set -euo pipefail

module load apptainer 2>/dev/null || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

DEV="/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev"
CGW="/home/shares/sanchezf/gundam_n_flow/common_gundam_workspace"
SIF="/home/shares/sanchezf/gundam_n_flow/GuNFlows/env/containers/ml_image2.sif"
IN_CONTAINER_WORKDIR="/workspace/work/GuNFlows"
IN_CONTAINER_SETUP="${IN_CONTAINER_WORKDIR}/setup_nosubshell.sh"

# defaults (overridable by passing the same flags as extra args)
CONFIG="/workspace/config/Config100p10_noDet/gundamFitter_configToyOA_100plus10_noDet.root"
OUTPUT="/workspace/config/Config100p10_noDet/lh_profiles"
SIGMAS=3
NPOINTS=20
THREADS="${SLURM_CPUS_PER_TASK:-8}"

EXTRA_ARGS=""
if [ "$#" -gt 0 ]; then
  EXTRA_ARGS="$(printf ' %q' "$@")"
fi

echo "Job started at $(date)"

# NB: lh_profiles loads the dataset from /workspace/data, and GUNDAM lives under
# /workspace/gunflows_dev (per setup_nosubshell.sh) -> both binds are required.
srun --ntasks=1 apptainer exec \
  --env PYTHONNOUSERSITE=1 \
  --env OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  --env PYTHONPATH="/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows" \
  --bind "${DEV}:${IN_CONTAINER_WORKDIR}" \
  --bind "${DEV}:/workspace/gunflows_dev" \
  --bind "${CGW}:/workspace/config" \
  --bind "${CGW}/DATA:/workspace/data" \
  --pwd "${IN_CONTAINER_WORKDIR}" \
  "${SIF}" bash -lc "source '${IN_CONTAINER_SETUP}' && \
                     HYDRA_FULL_ERROR=1 python3 -m gunflows.likelihood_sampler.lh_profiles \
                     -c ${CONFIG} -o ${OUTPUT} -s ${SIGMAS} -n ${NPOINTS} -t ${THREADS} ${EXTRA_ARGS}"

echo "Job ended at $(date)"
