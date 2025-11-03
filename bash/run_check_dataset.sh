#!/bin/bash
#SBATCH --job-name=check_dataset
#SBATCH --partition=private-dpnc-cpu,shared-cpu
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --output=logs/check_%A.out
#SBATCH --error=logs/check_%A.err

APPTAINER_OPTIONS="--nv --cleanenv \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH=/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows \
  --bind /home/shares/sanchezf/gundam_n_flow/GuNFlows:/workspace/work/GuNFlows \
  --bind /srv/beegfs/scratch/groups/dpnc/neutrinos:/workspace/config \
  --bind /srv/beegfs/scratch/shares/sanchezf/gundam_n_flow/tmp_inputs/nextcloud/:/workspace/data \
  --pwd /workspace/config/GundamInputOA2021"

IMAGE_PATH="/home/shares/sanchezf/gundam_n_flow/GuNFlows/env/containers/ml_image2.sif"

IN_CONTAINER_WORKDIR="/workspace/work/GuNFlows"
IN_CONTAINER_SETUP="${IN_CONTAINER_WORKDIR}/setup_nosubshell.sh"

srun --ntasks=1 apptainer exec ${APPTAINER_OPTIONS} ${IMAGE_PATH} bash -lc "source '${IN_CONTAINER_SETUP}' && \
                     HYDRA_FULL_ERROR=1 python3 -m gunflows.check_initial_dataset \
                     -f /workspace/config/GundamInputOA2021/oa2022_data/batch_1.npz \
                     -o /workspace/config/GundamInputOA2021/oa2022_data/oa2022_data_check_plots \
                     -x" 

# srun --ntasks=1 apptainer exec ${APPTAINER_OPTIONS} ${IMAGE_PATH} bash -lc "source '${IN_CONTAINER_SETUP}' && \
#                      HYDRA_FULL_ERROR=1 python3 -m gunflows.check_initial_dataset \
#                      -f /workspace/config/GundamInputOA2021/oa2022_asimov/batch.npz \
#                      -o /workspace/config/GundamInputOA2021/asimov_check \
#                      -x" 
                     
#!/bin/bash
#SBATCH --job-name=check_dataset
#SBATCH --partition=private-dpnc-cpu,shared-cpu
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --output=logs/check_%A.out
#SBATCH --error=logs/check_%A.err
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

srun --ntasks=1 apptainer exec ${APPTAINER_OPTIONS} ${IMAGE_PATH} bash -lc "source '${IN_CONTAINER_SETUP}' && \
                     HYDRA_FULL_ERROR=1 python3 -m gunflows.check_initial_dataset \
                     -f /workspace/config/GundamInputOA2021/oa2022_data/batch10.npz \
                     -o /workspace/config/GundamInputOA2021/oa2022_data_check_plots \
                     -x" 

# srun --ntasks=1 apptainer exec ${APPTAINER_OPTIONS} ${IMAGE_PATH} bash -lc "source '${IN_CONTAINER_SETUP}' && \
#                      HYDRA_FULL_ERROR=1 python3 -m gunflows.check_initial_dataset \
#                      -f /workspace/config/GundamInputOA2021/oa2022_asimov/batch.npz \
#                      -o /workspace/config/GundamInputOA2021/asimov_check \
#                      -x" 
                     