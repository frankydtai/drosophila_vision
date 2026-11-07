#!/bin/bash
#SBATCH --job-name=hh_nd_finalize
#SBATCH --account=punim0477
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --chdir=/home/yitai/drosophila_vision/vision/simulation/scratch/021_continue_init_nd
#SBATCH --output=/home/yitai/drosophila_vision/vision/simulation/scratch/021_continue_init_nd/finalize-%j.log
#SBATCH --error=/home/yitai/drosophila_vision/vision/simulation/scratch/021_continue_init_nd/finalize-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/matplotlib_render"
export XDG_CACHE_HOME="$PWD/cache_render"
export TMPDIR="$PWD"
test -s selected.json
/home/yitai/drosophila_vision/vision/.venv/bin/python -B -u render.py
/home/yitai/drosophila_vision/vision/.venv/bin/python -B -u compare_nd.py
/home/yitai/drosophila_vision/vision/.venv/bin/python -B -u export.py
