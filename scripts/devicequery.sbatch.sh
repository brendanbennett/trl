#!/bin/bash --login
#SBATCH -p gpuA    # GPU option
#SBATCH -G 1       # 1 GPU
#SBATCH -t 5       # Job will run for at most 5 minutes
#SBATCH -n 8       # (or --ntasks=) Optional number of cores. The amount of host RAM
                   # available to your job is affected by this setting.

echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

# Get the CUDA software libraries and applications 
module purge
module load libs/cuda

# Run the Nvidia app that reports GPU statistics
deviceQuery