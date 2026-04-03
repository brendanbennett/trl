#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -t 0-12
#SBATCH -n 12
#SBATCH -J grpo
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

echo "Job $SLURM_JOB_ID using $SLURM_NTASKS CPU core(s) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

module purge
module load libs/cuda

source ~/scratch/trl/.venv/bin/activate

python ~/scratch/trl/scripts/posttrain_grpo.py --per_device_train_batch_size 1 --gradient_accumulation_steps 16
