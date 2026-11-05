#!/bin/bash
#SBATCH --job-name=ab_to_cd_keepab
#SBATCH --account=punim0477
#SBATCH --partition=gpu-a100-mig
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --chdir=/home/yitai/drosophila_vision/vision/simulation
#SBATCH --output=0_logs/%j.log
#SBATCH --error=0_logs/%j.log
set -euo pipefail
ITERATIONS="${1:?Expected iteration count}"
BASELINE="${2:-}"
PYTHON=/home/yitai/drosophila_vision/vision/.venv/bin/python
EXP=scratch/ab34381383_cd_transfer_keepab_20261105
RUN_NAME="${SLURM_JOB_ID}-ab34381383-cd-transfer-keepab-iter-${ITERATIONS}"
"${PYTHON}" "${EXP}/check_code.py"
echo "Job ${SLURM_JOB_ID}; iterations=${ITERATIONS}; run=${RUN_NAME}"
"${PYTHON}" -u run_init_from.py --config-path "${EXP}" \
  "run_name=${RUN_NAME}" "n_iter=${ITERATIONS}" \
  'tasks=[mbar]' 'contrasts=[bright]' \
  'gt_by_task.mbar=[T4a,T4b,T4c,T4d]' \
  'bar_directions=[right,left,up,down]' \
  cost_mid=0 'lrs=[0.1]' resume_adam=false \
  '~part_cost_scales' '+part_cost_scales={mbar:1,mbar_DSI:0}' \
  'figure_tasks=[mbar]' plot_right_only=false
"${PYTHON}" -u "${EXP}/analyze.py" "0_runs/hp_lp/${RUN_NAME}" "${ITERATIONS}" "${BASELINE}"
