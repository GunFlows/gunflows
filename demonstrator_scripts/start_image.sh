
APPTAINER_OPTIONS="--nv --cleanenv \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH=/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows \
  --bind /home/shares/sanchezf/gundam_n_flow/GuNFlows_dev:/workspace/work/GuNFlows \
  --bind /home/shares/sanchezf/gundam_n_flow/ToyNDFit/GundamWorkspace:/workspace/config \
  --bind /home/shares/sanchezf/gundam_n_flow/ToyNDFit/DATA:/workspace/data \
  --pwd /workspace/config/"

IMAGE_PATH="/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev/env/containers/ml_image2.sif"


echo "Opening shell in container "

apptainer shell ${APPTAINER_OPTIONS} ${IMAGE_PATH}
