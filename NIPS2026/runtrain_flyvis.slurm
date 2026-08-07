#!/bin/bash
# Created based on MaskCycleGAN-VC/runtrain.slurm template
# Wednesday Apr 1, 2026

# Partition for the job:
#SBATCH --partition=gpu-h100

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="flyvis_train"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1
# The amount of memory in megabytes per node:
#SBATCH --mem=102400

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-9:0:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
export PYTHONPATH="flyvis"

./.venv/bin/python flyvis/flyvis_cli/training/train_single.py \
    ensemble_and_network_id=0001/000 \
    task_name=flow \
    train=true \
    task.n_iters=300000 \
    description='test_run' \
    delete_if_exists=true

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s