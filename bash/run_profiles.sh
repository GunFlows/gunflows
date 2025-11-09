#!/bin/bash
#SBATCH --job-name=profile
#SBATCH --partition=private-dpnc-cpu,shared-cpu
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --output=logs/profile_%A.out
#SBATCH --error=logs/profile_%A.err
#SBATCH --mail-type=ALL

APPTAINER_OPTIONS="--nv --cleanenv \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH=/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows \
  --bind /home/shares/sanchezf/gundam_n_flow/GuNFlows_dev:/workspace/work/GuNFlows \
  --bind /srv/beegfs/scratch/groups/dpnc/neutrinos:/workspace/config \
  --bind /srv/beegfs/scratch/shares/sanchezf/gundam_n_flow/tmp_inputs/nextcloud/:/workspace/data \
  --pwd /workspace/config/GundamInputOA2021"

IMAGE_PATH="/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev/env/containers/ml_image2.sif"

IN_CONTAINER_WORKDIR="/workspace/work/GuNFlows"
IN_CONTAINER_SETUP="${IN_CONTAINER_WORKDIR}/setup_nosubshell.sh"

# data
srun --ntasks=1 apptainer exec ${APPTAINER_OPTIONS} ${IMAGE_PATH} bash -lc "source '${IN_CONTAINER_SETUP}' && \
                     HYDRA_FULL_ERROR=1 python3 -m gunflows.likelihood_sampler.lh_profiles \
                     -c /workspace/config/GundamInputOA2021/output/gundamFitter_configOa2021.root \
                     -o /workspace/config/GundamInputOA2021/oa2022_data_profiles \
                     -x \
                     -s 5 \
                     -n 30"

# # fds
# srun --ntasks=1 apptainer exec ${APPTAINER_OPTIONS} ${IMAGE_PATH} bash -lc "source '${IN_CONTAINER_SETUP}' && \
#                      HYDRA_FULL_ERROR=1 python3 -m gunflows.likelihood_sampler.lh_profiles \
#                      -c /workspace/config/GundamInputOA2021/output/FDSOA2021/fdsFit_Martini1pi.root \
#                      -o /workspace/config/GundamInputOA2021/fds_check/profiles \
#                      -x \
#                      -s 5 \
#                      -n 30"