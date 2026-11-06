#!/bin/bash
#SBATCH --job-name=native-init-ab
#SBATCH --account=punim0477
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --chdir=/home/yitai/drosophila_vision
#SBATCH --output=/home/yitai/drosophila_vision/vision/simulation/scratch/native_initialization_ab_20261106/job-%j.log
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORM_NAME=cpu
export PYTHONDONTWRITEBYTECODE=1
TASK_DIR=/home/yitai/drosophila_vision/vision/simulation/scratch/native_initialization_ab_20261106
/home/yitai/drosophila_vision/vision/.venv/bin/python -u "$TASK_DIR/run_a.py"
/scratch/punim0477/yitai/ripple_p1_bar_20261028/comp_neuro/bin/python -u "$TASK_DIR/run_b.py"
