#!/bin/bash
#SBATCH --job-name=redraw_34431163_gt
#SBATCH --account=punim0477
#SBATCH --partition=gpu-a100-mig
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:05:00
#SBATCH --chdir=/home/yitai/drosophila_vision/vision/simulation/scratch/011_redraw_34431163_gt
#SBATCH --output=/home/yitai/drosophila_vision/vision/simulation/scratch/011_redraw_34431163_gt/redraw-%j.log
#SBATCH --error=/home/yitai/drosophila_vision/vision/simulation/scratch/011_redraw_34431163_gt/redraw-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/matplotlib"
export XDG_CACHE_HOME="$PWD/cache"
export TMPDIR="$PWD"
/home/yitai/drosophila_vision/vision/.venv/bin/python -B -u redraw.py
