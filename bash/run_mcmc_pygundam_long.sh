#!/bin/bash
# =============================================================================
#  run_mcmc_pygundam_long.sh — long MCMC run on private-dpnc-cpu (7 days)
#  More threads per chain, xsec-only corner plots, improved beta ladder.
#
#  Env-overridable:
#    SEED      = random seed  (default 0)
#    N_CHAINS  = parallel workers  (default 4)
#    THREADS   = OMP threads per worker  (default 8)  → 32 CPUs total
#    MODE      = parallel_tempering | independent  (default parallel_tempering)
#
#  Launch several seeds:
#    for s in 0 1 2 3; do sbatch --export=ALL,SEED=$s run_mcmc_pygundam_long.sh; done
# =============================================================================
#SBATCH --job-name=mcmc-long
#SBATCH --partition=private-dpnc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # N_CHAINS=4 x THREADS=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/mcmc_long_%j.out
#SBATCH --error=logs/mcmc_long_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mathielbaz@gmail.com

: "${SEED:=0}"
: "${N_CHAINS:=4}"
: "${THREADS:=8}"
: "${MODE:=parallel_tempering}"
export SEED N_CHAINS THREADS MODE

set -euo pipefail
module load apptainer 2>/dev/null || true
export OMP_NUM_THREADS="${THREADS}"

HOST_GUNFLOWS="/home/shares/sanchezf/gundam_n_flow/GuNFlows"
GUNFLOWS_DEV="/home/shares/sanchezf/gundam_n_flow/GuNFlows_dev"
HOST_CONFIG="/home/shares/sanchezf/gundam_n_flow/common_gundam_workspace"
HOST_DATA="/home/shares/sanchezf/gundam_n_flow/common_gundam_workspace/DATA"
HOST_OUT="/home/shares/sanchezf/gundam_n_flow/common_gundam_workspace"
SIF="/home/shares/sanchezf/gundam_n_flow/GuNFlows/env/containers/ml_image2.sif"

IN_GUNFLOWS="/workspace/work/GuNFlows"
IN_SETUP="${IN_GUNFLOWS}/setup_nosubshell.sh"
IN_CONFIG="/workspace/config"
IN_DATA="/workspace/data"
IN_OUT="/workspace/output"

mkdir -p "${HOST_OUT}"

OUTFILE_CONT="${IN_OUT}/mcmc_long_seed${SEED}_${SLURM_JOB_ID:-local}.npz"
OUTFILE_HOST="${HOST_OUT}/mcmc_long_seed${SEED}_${SLURM_JOB_ID:-local}.npz"
PLOTS_DIR="${HOST_OUT}/plots_pt_mcmc_long_seed${SEED}_${SLURM_JOB_ID:-local}"

EXTRA_ARGS=""
if [ "$#" -gt 0 ]; then
  EXTRA_ARGS="$(printf ' %q' "$@")"
fi

echo "Job started at $(date) — Seed: ${SEED}  N_chains: ${N_CHAINS}  Threads/chain: ${THREADS}  Mode: ${MODE}"
echo "Output: ${OUTFILE_HOST}"

srun --ntasks=1 apptainer exec \
  --env PYTHONNOUSERSITE=1 \
  --env OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  --env PYTHONPATH="/workspace/work/GuNFlows/src:/workspace/work/GuNFlows/src/normalizing-flows" \
  --bind "${HOST_GUNFLOWS}:${IN_GUNFLOWS}" \
  --bind "${GUNFLOWS_DEV}:/workspace/gunflows_dev" \
  --bind "${HOST_CONFIG}:${IN_CONFIG}" \
  --bind "${HOST_DATA}:${IN_DATA}:ro" \
  --bind "${HOST_OUT}:${IN_OUT}" \
  --pwd "${IN_GUNFLOWS}" \
  "${SIF}" bash -c "
    set -euo pipefail
    source '${IN_SETUP}' 2>/dev/null
    export PYTHONPATH=/opt/root/lib:\$PYTHONPATH
    export LD_LIBRARY_PATH=/opt/root/lib:\$LD_LIBRARY_PATH
    HYDRA_FULL_ERROR=1 python3 -s -m gunflows.run_mcmc_pygundam \
      --config-path ${IN_GUNFLOWS}/configs \
      --config-name mcmc_pygundam \
      seed=${SEED} \
      n_chains=${N_CHAINS} \
      mode=${MODE} \
      likelihood.threads_per_chain=${THREADS} \
      beta_min=0.3 \
      n_steps=2000000 \
      burn_in=20000 \
      plot_every=50000 \
      xsec_only=true \
      save_dir=${IN_OUT} \
      out_file=mcmc_long_seed${SEED}_${SLURM_JOB_ID:-local}.npz \
      'hydra.run.dir=outputs/long_seed${SEED}_${SLURM_JOB_ID:-local}' \
      ${EXTRA_ARGS}
  "

echo "Job ended at $(date)"
echo "Output: ${OUTFILE_HOST}"

# Email the latest corner plot on job completion
LATEST_PLOT="\$(ls -t ${PLOTS_DIR}/corner_*.png 2>/dev/null | head -1)"
if [ -n "\${LATEST_PLOT}" ]; then
  echo "MCMC long job ${SLURM_JOB_ID} done. Seed=${SEED}. Output: ${OUTFILE_HOST}" | \
    s-nail -s "MCMC long job ${SLURM_JOB_ID} done — xsec corner plot" \
      -a "\${LATEST_PLOT}" \
      mathielbaz@gmail.com
  echo "Corner plot emailed: \${LATEST_PLOT}"
fi
