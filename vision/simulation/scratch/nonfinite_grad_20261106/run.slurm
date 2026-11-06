#!/bin/bash
#SBATCH --job-name=grad-diagnose
#SBATCH --account=punim0477
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=00:20:00
#SBATCH --chdir=/home/yitai/drosophila_vision
#SBATCH --output=/home/yitai/drosophila_vision/vision/simulation/scratch/nonfinite_grad_20261106/%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/home/yitai/drosophila_vision/vision/.venv/bin/python -u /home/yitai/drosophila_vision/vision/simulation/scratch/nonfinite_grad_20261106/diagnose.py "$@"
