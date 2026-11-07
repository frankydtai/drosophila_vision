#!/bin/bash
#SBATCH --job-name=hh_rule_init_cmp
#SBATCH --account=punim0477
#SBATCH --partition=sapphire
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --chdir=/home/yitai/drosophila_vision/vision/simulation/scratch/022_rule_init_nd_comparison
#SBATCH --output=/home/yitai/drosophila_vision/vision/simulation/scratch/022_rule_init_nd_comparison/compare-%A_%a.log
#SBATCH --error=/home/yitai/drosophila_vision/vision/simulation/scratch/022_rule_init_nd_comparison/compare-%A_%a.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/matplotlib_mbar_$SLURM_ARRAY_TASK_ID"
export XDG_CACHE_HOME="$PWD/cache_mbar_$SLURM_ARRAY_TASK_ID"
export TMPDIR="$PWD"
/home/yitai/drosophila_vision/vision/.venv/bin/python -B -u worker.py "$SLURM_ARRAY_TASK_ID"
