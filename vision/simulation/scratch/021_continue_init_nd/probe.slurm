#!/bin/bash
#SBATCH --job-name=hh_nd_probe
#SBATCH --account=punim0477
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:10:00
#SBATCH --chdir=/home/yitai/drosophila_vision/vision/simulation/scratch/021_continue_init_nd
#SBATCH --output=/home/yitai/drosophila_vision/vision/simulation/scratch/021_continue_init_nd/probe-%j.log
#SBATCH --error=/home/yitai/drosophila_vision/vision/simulation/scratch/021_continue_init_nd/probe-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/matplotlib_mbar_0"
export XDG_CACHE_HOME="$PWD/cache_mbar_0"
export TMPDIR="$PWD"
/home/yitai/drosophila_vision/vision/.venv/bin/python -B -u early_probe.py
