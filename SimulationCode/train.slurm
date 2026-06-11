#!/bin/bash
#SBATCH --job-name=borst_train
#SBATCH --account=punim0477
#SBATCH --partition=gpu-h100
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

set -euo pipefail

cd /scratch/punim0477/yitai/temporal_filtering/SimulationCode
source ../.venv/bin/activate

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "GPU:    ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "Start:  $(date)"
python -u run_training.py --nofruns 20 --nofsteps 10000 --fname training_with_Ih.npy
echo "End:    $(date)"
