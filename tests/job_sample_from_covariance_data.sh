#!/bin/env bash
#SBATCH --job-name=sample_cov_data
#SBATCH --partition=shared-cpu
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=50
#SBATCH --output=sample_from_cov_data_%A_%a.out
#SBATCH --error=sample_from_cov_data_%A_%a.err
#SBATCH --mem=30000 # in MB
#SBATCH --mail-type=ALL

CONFIG_FILE="output/gundamFitter_configOa2021.root"
OUTPUT_FILE="oa2022_data.npz"
N=100000
SCRIPTARGS="-m gunflows.likelihood_sampler.generate_dataset_from_covariance -o ${OUTPUT_FILE} -c ${CONFIG_FILE} -n ${N}"

APPTAINER_OPTIONS="--nv --cleanenv \
  --env PYTHONNOUSERSITE=1 \
  --env DATA_DIR=/workspace/data/DatasetNFlowsOA2022/Asimov/allParameters \
  --env PYTHONPATH=/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows \
  --bind /home/shares/sanchezf/gundam_n_flow/GuNFlows:/workspace/work/GuNFlows \
  --bind /srv/beegfs/scratch/groups/dpnc/neutrinos:/workspace/config \
  --bind /srv/beegfs/scratch/shares/sanchezf/gundam_n_flow/tmp_inputs/nextcloud/:/workspace/data \
  --pwd /workspace/config/GundamInputOA2021"

IMAGE_PATH="/home/shares/sanchezf/gundam_n_flow/GuNFlows/env/containers/ml_image2.sif"

echo "Starting job: " `date`

srun apptainer exec ${APPTAINER_OPTIONS} ${IMAGE_PATH} bash -c "source /workspace/work/GuNFlows/setup_nosubshell.sh; python ${SCRIPTARGS}" 

echo "Job done: " `date`
