#!/bin/bash
# =============================================================================
#  plot_ess.sh  --  (re)draw the ESS plots from a saved ess_vs_epoch.json,
#  in the container, with the paper style. No GPU / sbatch needed.
#
#  Usage:
#    ./plot_ess.sh <json_path> [hydra overrides ...]
#
#  <json_path> is the IN-CONTAINER path to ess_vs_epoch.json, i.e. under
#  /workspace/config (= common_gundam_workspace). Example:
#
#    ./plot_ess.sh \
#      /workspace/config/Config100p10_noDet/best_models_1day/top1/ess_5k/ess_vs_epoch.json
#
#  Multiple json files (epochs merged, FIRST file wins on duplicates) can be
#  passed as one comma-separated argument (no spaces):
#    ./plot_ess.sh /workspace/config/.../a/ess_vs_epoch.json,/workspace/config/.../b/ess_vs_epoch.json \
#      out_dir=/workspace/config/.../combined
#
#  Optional overrides (forwarded to Hydra), e.g.:
#    out_dir=/workspace/config/.../somewhere   fmt=pdf   usetex=true   label_fontsize=18
#
#  Defaults (see configs/plot_ess.yaml): paper_style=true, y_percent=true,
#  also_loglog=true, show_title=false, usetex=false (machine-independent).
# =============================================================================
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <in-container json_path> [hydra overrides ...]" >&2
  exit 1
fi
JSON="$1"; shift

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
                     HYDRA_FULL_ERROR=1 python -s -m gunflows.plot_ess_from_json \
                     json_path=${JSON} ${EXTRA_ARGS}"
