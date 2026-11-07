#!/bin/bash
#SBATCH --job-name=hh_394150_render
#SBATCH --account=punim0477
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:10:00
#SBATCH --chdir=/home/yitai/drosophila_vision/vision/simulation/scratch/015_continue_394150
#SBATCH --output=/home/yitai/drosophila_vision/vision/simulation/scratch/015_continue_394150/render-%j.log
#SBATCH --error=/home/yitai/drosophila_vision/vision/simulation/scratch/015_continue_394150/render-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/matplotlib_render"
export XDG_CACHE_HOME="$PWD/cache_render"
export TMPDIR="$PWD"
/home/yitai/drosophila_vision/vision/.venv/bin/python -B -u render.py
