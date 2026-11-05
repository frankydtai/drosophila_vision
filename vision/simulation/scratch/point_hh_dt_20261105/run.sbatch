#!/bin/bash
#SBATCH --job-name=point-hh-dt
#SBATCH --account=punim0477
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:08:00
#SBATCH --chdir=/home/yitai/drosophila_vision/vision/simulation/scratch/point_hh_dt_20261105
#SBATCH --output=slurm-%j.log
set -euo pipefail
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/point-hh-dt-mpl-${SLURM_JOB_ID}
/data/scratch/projects/punim0477/yitai/ripple_p1_bar_20261028/comp_neuro/bin/python -u sweep.py \
  --ripple /home/yitai/Ripple \
  --output results-${SLURM_JOB_ID}
