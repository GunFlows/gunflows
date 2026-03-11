#!/bin/env bash
#SBATCH --job-name=fit
#SBATCH --partition=private-dpnc-cpu,shared-cpu
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --output=logs/fit_%A.out
#SBATCH --error=logs/fit_%A.err
#SBATCH --mail-type=ALL


# ARGUMENTS TO PASS TO gundamFitter
SCRIPTARGS="-c configToyOA_100plus0.yaml -a -t 10 "
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
