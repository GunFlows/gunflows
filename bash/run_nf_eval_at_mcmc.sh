#!/bin/bash
# Generic: submit with MCMC_CHK and OUT_PDF env vars, or use defaults.
#SBATCH --job-name=nf-eval
#SBATCH --partition=shared-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/nf_eval_%j.out
#SBATCH --error=logs/nf_eval_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mathielbaz@gmail.com

: "${MCMC_CHK:=/home/shares/sanchezf/gundam_n_flow/common_gundam_workspace/mcmc_long_seed0_9734293_chk0500000.npz}"
: "${OUT_PDF:=/home/shares/sanchezf/gundam_n_flow/common_gundam_workspace/mcmc_nf_full_seed0_chk0500000_nfeval.pdf}"
: "${EMAIL_SUBJECT:=[NF eval] Weight plot with NF log-prob at MCMC pts}"
: "${CORNER_PNG:=}"   # optional extra attachment (corner plot)

HOST_GUNFLOWS="/home/shares/sanchezf/gundam_n_flow/GuNFlows"
HOST_OUT="/home/shares/sanchezf/gundam_n_flow/common_gundam_workspace"
SIF="${HOST_GUNFLOWS}/env/containers/ml_image2.sif"
NF_DIR="${HOST_OUT}/Config100p10_noDet/17-30-29"

IN_GUNFLOWS="/workspace/work/GuNFlows"
IN_OUT="/workspace/output"
IN_NF_DIR="${IN_OUT}/Config100p10_noDet/17-30-29"

MCMC_CHK_CONT="${IN_OUT}/$(basename ${MCMC_CHK})"
OUT_PDF_CONT="${IN_OUT}/$(basename ${OUT_PDF})"

echo "Job started at $(date)"
echo "MCMC: ${MCMC_CHK}"
echo "Out:  ${OUT_PDF}"

module load apptainer 2>/dev/null || true

srun --ntasks=1 apptainer exec \
  --env PYTHONNOUSERSITE=1 \
  --env OMP_NUM_THREADS=8 \
  --env PYTHONPATH="${IN_GUNFLOWS}/src:${IN_GUNFLOWS}/src/normalizing-flows" \
  --bind "${HOST_GUNFLOWS}:${IN_GUNFLOWS}" \
  --bind "${HOST_OUT}:${IN_OUT}" \
  --pwd "${IN_GUNFLOWS}" \
  "${SIF}" bash -c "
    set -euo pipefail
    source '${IN_GUNFLOWS}/setup_nosubshell.sh' 2>/dev/null || true
    export PYTHONPATH=/opt/root/lib:\$PYTHONPATH
    export LD_LIBRARY_PATH=/opt/root/lib:\$LD_LIBRARY_PATH
    python3 src/gunflows/compare_mcmc_nf_marginals.py \
      --mcmc ${MCMC_CHK_CONT} \
      --nf ${IN_NF_DIR}/paper_plots_100k/cache/samp_nf.npy \
      --gauss ${IN_NF_DIR}/paper_plots_100k/cache/samp_gaussian.npy \
      --burnin 50000 --thin_to 100000 \
      --nf_model ${IN_NF_DIR} \
      --out ${OUT_PDF_CONT}
  "

echo "Done at $(date). Emailing..."
ATTACH_ARGS="-a ${OUT_PDF}"
[ -n "${CORNER_PNG}" ] && ATTACH_ARGS="${ATTACH_ARGS} -a ${CORNER_PNG}"
echo "${EMAIL_SUBJECT}" | s-nail -s "${EMAIL_SUBJECT}" ${ATTACH_ARGS} mathielbaz@gmail.com
echo "Email sent."
