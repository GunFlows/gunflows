#!/bin/bash
# =============================================================================
#  gaussian_ess.sh -- compute the Gaussian-surrogate ESS from the original
#  training dataset batches (no GPU, no likelihood calls). Writes a single-entry
#  ess_vs_epoch.json (epoch 0) that can be merged with the NF ESS jsons.
#
#  Usage:
#    ./gaussian_ess.sh <training_folder> [hydra overrides ...]
#
#  <training_folder> is the IN-CONTAINER path (under /workspace/config), e.g.:
#    ./gaussian_ess.sh /workspace/config/Config100p10_noDet/best_models_1day/top1
#  Optional overrides: save_dir=... max_batches=10
# =============================================================================
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <in-container training_folder> [hydra overrides ...]" >&2
  exit 1
fi
TF="$1"; shift

module load apptainer 2>/dev/null || true

HOST_REPO="/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev"
HOST_CONFIG="/home/shares/sanchezf/gundam_n_flow/common_gundam_workspace"
SIF="/home/shares/sanchezf/gundam_n_flow/GuNFlows/env/containers/ml_image2.sif"
IN_CONTAINER_WORKDIR="/workspace/work/GuNFlows"
IN_CONTAINER_SETUP="${IN_CONTAINER_WORKDIR}/setup_nosubshell.sh"

EXTRA_ARGS=""
if [ "$#" -gt 0 ]; then
  EXTRA_ARGS="$(printf ' %q' "$@")"
fi

apptainer exec \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH="/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows" \
  --bind "${HOST_REPO}:${IN_CONTAINER_WORKDIR}" \
  --bind "${HOST_CONFIG}:/workspace/config" \
  --pwd "${IN_CONTAINER_WORKDIR}" \
  "${SIF}" bash -lc "source '${IN_CONTAINER_SETUP}' >/dev/null 2>&1 && \
                     HYDRA_FULL_ERROR=1 python -s -m gunflows.gaussian_ess_from_dataset \
                     training_folder=${TF} ${EXTRA_ARGS}"
