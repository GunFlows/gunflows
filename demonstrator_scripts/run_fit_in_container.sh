#!/bin/env bash
#SBATCH --job-name=fit
#SBATCH --partition=private-dpnc-cpu,shared-cpu
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --output=logs/fit_%A_%a.out
#SBATCH --error=logs/fit_%A_%a.err
#SBATCH --mail-type=ALL


TOY_SEED=$SLURM_ARRAY_TASK_ID

# ARGUMENTS TO PASS TO gundamFitter
# SCRIPTARGS="-c configToyOA_100plus10.yaml -t 10 -a --toy ${TOY_SEED} -s ${TOY_SEED}"
#SCRIPTARGS="-c configToyOA_60plus6.yaml -t 10 -a"
# Reduced-binning + new (small-sigma) detector prior variant.
# Output is auto-named by gundam from the main-config name:
#   output/gundamFitter_configToyOA_100plus10_reduced.root
# so we do NOT pass -o and the previous run's output is not overwritten.
SCRIPTARGS="-c configToyOA_100plus10_reduced.yaml -t 10"
echo "Running Gundam Fit with arguments: ${SCRIPTARGS}" 

# CHECK AVAILABLE FITTER CONFIGS AT: 
# /home/shares/sanchezf/gundam_n_flow/ToyNDFit/GundamWorkspace

APPTAINER_OPTIONS="--nv --cleanenv \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH=/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows \
  --bind /home/shares/sanchezf/gundam_n_flow/GuNFlows_dev:/workspace/work/GuNFlows \
  --bind /home/shares/sanchezf/gundam_n_flow/ToyNDFit/GundamWorkspace:/workspace/config \
  --bind /home/shares/sanchezf/gundam_n_flow/ToyNDFit/DATA:/workspace/data \
  --pwd /workspace/config/"

IMAGE_PATH="/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev/env/containers/ml_image2.sif"

echo "Starting Gundam Fit: " `date`
echo 

apptainer exec ${APPTAINER_OPTIONS} ${IMAGE_PATH} ./run_fit.sh ${SCRIPTARGS}

echo
echo "Finished Gundam Fit: " `date`
