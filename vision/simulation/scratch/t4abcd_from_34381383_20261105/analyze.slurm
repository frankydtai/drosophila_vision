#!/bin/bash
#SBATCH --job-name=t4abcd_analysis
#SBATCH --account=punim0477
#SBATCH --partition=gpu-a100-mig
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:10:00
#SBATCH --chdir=/home/yitai/drosophila_vision/vision/simulation
#SBATCH --output=0_logs/%j.log
#SBATCH --error=0_logs/%j.log
set -euo pipefail
/home/yitai/drosophila_vision/vision/.venv/bin/python -u scratch/t4abcd_from_34381383_20261105/analyze.py
