#!/bin/bash
#SBATCH --job-name=save_reset469101
#SBATCH --account=punim0477
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --chdir=/home/yitai/drosophila_vision/vision/simulation/scratch/013_save_reset469101
#SBATCH --output=/home/yitai/drosophila_vision/vision/simulation/scratch/013_save_reset469101/replay-%j.log
#SBATCH --error=/home/yitai/drosophila_vision/vision/simulation/scratch/013_save_reset469101/replay-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/matplotlib"
export XDG_CACHE_HOME="$PWD/cache"
export TMPDIR="$PWD"
/home/yitai/drosophila_vision/vision/.venv/bin/python -B -u replay.py
/home/yitai/drosophila_vision/vision/.venv/bin/python -B -u render.py
