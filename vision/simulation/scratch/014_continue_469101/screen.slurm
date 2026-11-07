#!/bin/bash
#SBATCH --job-name=hh_continue_screen
#SBATCH --account=punim0477
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:25:00
#SBATCH --array=0-3
#SBATCH --chdir=/home/yitai/drosophila_vision/vision/simulation/scratch/014_continue_469101
#SBATCH --output=/home/yitai/drosophila_vision/vision/simulation/scratch/014_continue_469101/screen-%A_%a.log
#SBATCH --error=/home/yitai/drosophila_vision/vision/simulation/scratch/014_continue_469101/screen-%A_%a.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/matplotlib_$SLURM_ARRAY_TASK_ID"
export XDG_CACHE_HOME="$PWD/cache_$SLURM_ARRAY_TASK_ID"
export TMPDIR="$PWD"
/home/yitai/drosophila_vision/vision/.venv/bin/python -B -u worker.py screen_cases.json "$SLURM_ARRAY_TASK_ID"
