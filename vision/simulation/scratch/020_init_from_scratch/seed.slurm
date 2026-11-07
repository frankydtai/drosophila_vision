#!/bin/bash
#SBATCH --job-name=hh_init_seed
#SBATCH --account=punim0477
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:10:00
#SBATCH --chdir=/home/yitai/drosophila_vision/vision/simulation/scratch/020_init_from_scratch
#SBATCH --output=/home/yitai/drosophila_vision/vision/simulation/scratch/020_init_from_scratch/seed-%j.log
#SBATCH --error=/home/yitai/drosophila_vision/vision/simulation/scratch/020_init_from_scratch/seed-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/matplotlib_extend"
export XDG_CACHE_HOME="$PWD/cache_extend"
export TMPDIR="$PWD"
/home/yitai/drosophila_vision/vision/.venv/bin/python -B -u seed_refine.py
