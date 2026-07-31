#!/bin/bash
#SBATCH --job-name=run
#SBATCH --account=punim0477
#SBATCH --partition=gpu-a100
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --chdir=/home/yitai/drosophila_vision/temporal_filtering/SimulationCode
#SBATCH --output=0_logs/training_%j.out
#SBATCH --error=0_logs/training_%j.err

# Repo on $HOME (scratch/punim0477/.../SimulationCode is 0_logs-only).
REPO=/home/yitai/drosophila_vision/temporal_filtering

set -euo pipefail

PYTHON="${REPO}/.venv/bin/python"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "GPU:    ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
"${PYTHON}" -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "Start:  $(date)"
# Sorted N_name layout: import_bootstrap via venv .pth; train via run.py (not -m training.driver).
"${PYTHON}" -u run.py --nofsteps 100 --tau-hp init=L1,L2,L4,L5:200 all:10000 --pulse-ms 150 --delta-ms 5
