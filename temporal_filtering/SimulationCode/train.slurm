#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=punim0477
#SBATCH --partition=gpu-h100
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --chdir=/home/yitai/drosophila_vision/temporal_filtering/SimulationCode
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

# Repo on $HOME (scratch/punim0477/.../SimulationCode is logs-only).
REPO=/home/yitai/drosophila_vision/temporal_filtering

set -euo pipefail

source "${REPO}/.venv/bin/activate"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "GPU:    ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "Start:  $(date)"
python -u train.py --target spot_bright,moving_bar_bright \
  --model-type conductance --nofsteps 10000 --lrs 0.1 --shift --network right_min_neuron1_extent2 --cost-extent 1
echo "End:    $(date)"
