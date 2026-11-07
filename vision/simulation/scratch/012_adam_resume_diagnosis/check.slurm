#!/bin/bash
#SBATCH --job-name=adam_diagnosis
#SBATCH --account=punim0477
#SBATCH --partition=gpu-a100-mig
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:15:00
#SBATCH --array=0-2
#SBATCH --chdir=/home/yitai/drosophila_vision/vision/simulation/scratch/012_adam_resume_diagnosis
#SBATCH --output=/home/yitai/drosophila_vision/vision/simulation/scratch/012_adam_resume_diagnosis/check-%A_%a.log
#SBATCH --error=/home/yitai/drosophila_vision/vision/simulation/scratch/012_adam_resume_diagnosis/check-%A_%a.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/matplotlib_$SLURM_ARRAY_TASK_ID"
export XDG_CACHE_HOME="$PWD/cache_$SLURM_ARRAY_TASK_ID"
/home/yitai/drosophila_vision/vision/.venv/bin/python -B -u worker.py "$SLURM_ARRAY_TASK_ID" cuda
