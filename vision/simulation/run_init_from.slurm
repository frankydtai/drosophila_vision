#!/bin/bash
#SBATCH --job-name=run_init_from
#SBATCH --account=punim0477
#SBATCH --partition=gpu-a100-mig
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --chdir=/home/yitai/drosophila_vision/vision/simulation
#SBATCH --output=0_logs/%j.log
#SBATCH --error=0_logs/%j.log

# Repo on $HOME (scratch/punim0477/.../simulation is 0_logs-only).
REPO=/home/yitai/drosophila_vision/vision

set -euo pipefail

PYTHON="${REPO}/.venv/bin/python"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "GPU:    ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
"${PYTHON}" -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "Start:  $(date)"
echo "==== config ===="
"${PYTHON}" - <<'PY'
from pprint import pprint
import config as pd

params = {
    k: v
    for k, v in vars(pd).items()
    if k.isupper() and not k.startswith("_")
}
pprint(params, sort_dicts=False, width=120)
PY
echo "==== end config ===="
# init_from defaults to RUN_PATH; freeze all schema params except syn_strength_cell Mi4:T4a (indi).
"${PYTHON}" -u run_init_from.py \
  n_iter=100 \
  tasks=[sbar] \
  params.a_gt.mode=frozen \
  params.bias_gt.mode=frozen \
  params.a_in.mode=frozen \
  params.a_out.mode=frozen \
  params.e_leak.mode=frozen \
  params.v_th.mode=frozen \
  params.tau_lp.mode=frozen \
  params.tau_hp_rise.mode=frozen \
  params.tau_hp_fall.mode=frozen \
  params.a_h.mode=frozen \
  params.a_sti_mid.mode=frozen \
  params.a_sti_radius.mode=frozen \
  'params.syn_strength_cell.mode=frozen' \
  'params.syn_strength_cell.modes.indi=[Mi4:T4a]' \
  'params.syn_strength_cell.modes.fixed=[]' \
  '~part_cost_scales' \
  '+part_cost_scales={sbar:0,sbar_bright_T4a_mid+0:1,sbar_bright_T4a_mid-1:1,sbar_bright_T4a_mid-2:1,sbar_bright_T4a_mid+1:1,sbar_bright_T4a_mid+2:1}'
