#!/bin/bash --login
#SBATCH -p gpuH_short
#SBATCH -G 1
#SBATCH -t 0-8
#SBATCH -n 8
#SBATCH -J grpo
#SBATCH -o ~/scratch/trl/logs/%x_%j.out
#SBATCH -e ~/scratch/trl/logs/%x_%j.err

echo "Job $SLURM_JOB_ID using $SLURM_NTASKS CPU core(s) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

module purge
module load libs/cuda

WORKDIR="\${WORKDIR:-\$HOME/scratch/trl}"


cd "\$WORKDIR"

source .venv/bin/activate

python scripts/posttrain_grpo.py --per_device_train_batch_size 2 --gradient_accumulation_steps 8
