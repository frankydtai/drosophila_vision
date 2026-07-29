#!/bin/bash
#SBATCH --job-name=train
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
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

# Repo on $HOME (scratch/punim0477/.../SimulationCode is logs-only).
REPO=/home/yitai/drosophila_vision/temporal_filtering

set -euo pipefail

PYTHON="${REPO}/.venv/bin/python"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "GPU:    ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
"${PYTHON}" -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "Start:  $(date)"
"${PYTHON}" -u -m training.train --nofsteps 1000 --tau-hp init=L1,L2,L4,L5:200 all:10000
