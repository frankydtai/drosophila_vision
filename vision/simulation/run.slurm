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
#SBATCH --time=0:10:00
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
"${PYTHON}" -u run.py --nofsteps 500 --a-h init.L1,L2,L4,L5=0.5 --e-leak init.L1,L2,L3,Lai,Mi10,Mi15,Tm16,Tm5a,Tm5b,TmY11=0 init.R1-6,R7,R8,L4,T4a,T4b,T4c,T4d,Tm1,Tm2,Tm4,T5a,T5b,T5c,T5d,T2,T2a,T3,Lawf1,Lawf2,Dm3v,Mi14,Mi2,Tm21,Tm27,Tm5c,Tm5f,TmY14,TmY15,TmY5a,TmY9q,TmY9q__perp=-40 --task spot

