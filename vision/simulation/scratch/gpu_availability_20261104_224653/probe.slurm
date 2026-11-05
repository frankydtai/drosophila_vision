#!/bin/bash
set -eu
printf 'JOB_ID=%s PARTITION=%s NODE=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_PARTITION" "$(hostname)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
printf 'SLURM_JOB_GPUS=%s\n' "${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
