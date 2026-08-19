#!/bin/bash
#SBATCH --job-name=run
#SBATCH --account=punim0477
#SBATCH --partition=gpu-a100-mig
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:10:00
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
# Sorted N_name layout: import_bootstrap via venv .pth; train via run.py (not -m training.driver).
"${PYTHON}" -u run.py n_iter=300

