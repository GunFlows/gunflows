#!/bin/env bash
#SBATCH --job-name=sample_cov_asimov
#SBATCH --partition=private-dpnc-cpu,shared-cpu
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --output=logs/sample_from_cov_asimov_%A_%a.out
#SBATCH --error=logs/sample_from_cov_asimov_%A_%a.err
#SBATCH --mail-type=ALL

index=${SLURM_ARRAY_TASK_ID}
time=$(date +%s%N)
seed=$(($index + $time))

good_fits=(11 12 13 14 15 16 18 20 23 24 26 27 29 31 32 34 35 37 39 40 41 42 45 46 47 49)
# Pick a seed out of the above ones, the fit converged for them.
SEED=12

INPUT_IDENTIFIER="configToyOA_22plus3_Asimov" 
#INPUT_IDENTIFIER="configToyOA_100plus10_Asimov_ToyFit_${SEED}"
# this is the second part of the Gundam fitter output filename, without extension:
# gundamFitter_configToyOA_60plus6_Asimov.root -> configToyOA_60plus6_Asimov

CONFIG_FILE="output/gundamFitter_${INPUT_IDENTIFIER}.root"
CONFIG_FOLDER_LOCAL="/home/shares/sanchezf/gundam_n_flow/ToyNDFit/GundamWorkspace"

OUTPUT_FOLDER="datasets_sb_${INPUT_IDENTIFIER}"
mkdir -p ${CONFIG_FOLDER_LOCAL}/${OUTPUT_FOLDER}
OUTPUT_FILE="${OUTPUT_FOLDER}/batch${index}.npz"
N=50000


SCRIPTARGS="-m gunflows.make_initial_dataset -o ${OUTPUT_FILE} -c ${CONFIG_FILE} -n ${N} -t 16 "

APPTAINER_OPTIONS="--nv --cleanenv \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH=/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows \
  --bind /home/shares/sanchezf/gundam_n_flow/GuNFlows_dev:/workspace/work/GuNFlows \
  --bind /home/shares/sanchezf/gundam_n_flow/ToyNDFit/GundamWorkspace:/workspace/config \
  --bind /home/shares/sanchezf/gundam_n_flow/ToyNDFit/DATA:/workspace/data \
  --pwd /workspace/config/"

IMAGE_PATH="/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev/env/containers/ml_image2.sif"


echo "Starting job: " `date`

apptainer exec ${APPTAINER_OPTIONS} ${IMAGE_PATH} bash -c "source /workspace/work/GuNFlows/demonstrator_scripts/setup.sh; python ${SCRIPTARGS}" 

echo "Job done: " `date`
