#!/bin/bash
#SBATCH --job-name=run
#SBATCH --account=punim0477
#SBATCH --partition=gpu-h100
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --chdir=/home/yitai/drosophila_vision/vision/simulation
#SBATCH --output=0_logs/%j.out
#SBATCH --error=0_logs/%j.out

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
echo "==== param_defaults ===="
"${PYTHON}" - <<'PY'
from pprint import pprint
import param_defaults as pd

params = {
    k: v
    for k, v in vars(pd).items()
    if k.isupper() and not k.startswith("_")
}
pprint(params, sort_dicts=False, width=120)
PY
echo "==== end param_defaults ===="
# Sorted N_name layout: import_bootstrap via venv .pth; train via run.py (not -m training.driver).
"${PYTHON}" -u run.py --nofsteps 1000 --a-h init.L1,L2,L4,L5=0.5 --e-leak init.L1,L2=-20 --task spot_bright 

