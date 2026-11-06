#!/bin/bash
#SBATCH --job-name=lr-only-gpu
#SBATCH --account=punim0477
#SBATCH --partition=gpu-a100-mig
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/yitai/drosophila_vision
#SBATCH --output=/home/yitai/drosophila_vision/vision/simulation/scratch/lr_dt_stability_20261106/gpu-%A_%a.log
set -euo pipefail
ulimit -c 0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/home/yitai/drosophila_vision/vision/.venv/bin/python -u /home/yitai/drosophila_vision/vision/simulation/scratch/lr_dt_stability_20261106/worker.py "${1:-${SLURM_ARRAY_TASK_ID}}" cuda "${2:-20}" "${3:-screen}"
