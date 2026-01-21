#!/bin/env bash
#SBATCH --job-name=sample_cov_asimov
#SBATCH --partition=private-dpnc-cpu,shared-cpu
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=160G
#SBATCH --output=logs/sample_from_cov_asimov_%A_%a.out
#SBATCH --error=logs/sample_from_cov_asimov_%A_%a.err
#SBATCH --mail-type=ALL

index=${SLURM_ARRAY_TASK_ID}
time=$(date +%s%N)
seed=$(($index + $time))
CONFIG_FILE="output/gundamFitter_configOa2021_With_allowEigenDecompWithBounds_shiftedOPPB_Asimov.root"
CONFIG_FOLDER_LOCAL="/srv/beegfs/scratch/groups/dpnc/neutrinos/GundamInputOA2021"

OUTPUT_FOLDER="oa2022_asimov"
mkdir -p ${CONFIG_FOLDER_LOCAL}/${OUTPUT_FOLDER}
OUTPUT_FILE="${OUTPUT_FOLDER}/batch${index}.npz"
N=1000


SCRIPTARGS="-m gunflows.make_initial_dataset -o ${OUTPUT_FILE} -c ${CONFIG_FILE} -a -n ${N} -t 16 -of override/shiftedOPPB.yaml" #-s ${seed}

APPTAINER_OPTIONS="--nv --cleanenv \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH=/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows \
  --bind /home/shares/sanchezf/gundam_n_flow/GuNFlows_dev:/workspace/work/GuNFlows \
  --bind /srv/beegfs/scratch/groups/dpnc/neutrinos:/workspace/config \
  --bind /srv/beegfs/scratch/shares/sanchezf/gundam_n_flow/tmp_inputs/nextcloud/:/workspace/data \
  --pwd /workspace/config/GundamInputOA2021"

IMAGE_PATH="/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev/env/containers/ml_image2.sif"


echo "Starting job: " `date`

apptainer exec ${APPTAINER_OPTIONS} ${IMAGE_PATH} bash -c "source /workspace/work/GuNFlows/setup_nosubshell.sh; python ${SCRIPTARGS}" 

echo "Job done: " `date`
