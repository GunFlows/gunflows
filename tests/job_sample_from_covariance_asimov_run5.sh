#!/bin/env bash
#SBATCH --job-name=sample_cov_asimov_run5
#SBATCH --partition=private-dpnc-cpu,shared-cpu
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=12
#SBATCH --output=sample_from_cov_asimov_run5_%A_%a.out
#SBATCH --error=sample_from_cov_asimov_run5_%A_%a.err
#SBATCH --mem-per-cpu=8000 # in MB
#SBATCH --mail-type=ALL

index=${SLURM_ARRAY_TASK_ID}
time=$(date +%s%N)
seed=$(($index + $time))
CONFIG_FILE="output/Fitter/gundamFitter_configOa2021_With_allowEigenDecompWithBounds_Asimov.root"
OUTPUT_FILE="oa2022_asimovRun5/batch${index}.npz"
N=50000


SCRIPTARGS="-m gunflows.likelihood_sampler.test_sample_from_custom -o ${OUTPUT_FILE} -c ${CONFIG_FILE} -of override/onlyRun5.yaml -n ${N} -a -t 12 " #-s ${seed}

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
