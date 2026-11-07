#!/bin/bash
#SBATCH --job-name=hh_init_mbar
#SBATCH --account=punim0477
#SBATCH --partition=sapphire
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --chdir=/home/yitai/drosophila_vision/vision/simulation/scratch/020_init_from_scratch
#SBATCH --output=/home/yitai/drosophila_vision/vision/simulation/scratch/020_init_from_scratch/mbar-%A_%a.log
#SBATCH --error=/home/yitai/drosophila_vision/vision/simulation/scratch/020_init_from_scratch/mbar-%A_%a.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/matplotlib_mbar_$SLURM_ARRAY_TASK_ID"
export XDG_CACHE_HOME="$PWD/cache_mbar_$SLURM_ARRAY_TASK_ID"
export TMPDIR="$PWD"
/home/yitai/drosophila_vision/vision/.venv/bin/python -B -u worker.py mbar_cases.json "$SLURM_ARRAY_TASK_ID"
