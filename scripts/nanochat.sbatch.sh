#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -t 0-2
#SBATCH -n 8
#SBATCH -J nanochat            # job name for easier identification in squeue
#SBATCH -o logs/%x_%j.out      # stdout → logs/<jobname>_<jobid>.out
#SBATCH -e logs/%x_%j.err      # stderr → logs/<jobname>_<jobid>.err

# Experiment config
DEPTH=8
NUM_SHARDS_INIT=8
NUM_SHARDS_BACKGROUND=40
RUN_NAME="d${DEPTH}"
MODEL_TAG="d${DEPTH}"
DEVICE_BATCH_SIZE=32

CORE_METRIC_EVERY=999999
SAMPLE_EVERY=-1
SAVE_EVERY=-1

# Environment
VENV_PATH=~/scratch/nanochat/.venv

echo "Job $SLURM_JOB_ID using $SLURM_NTASKS CPU core(s) on $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Experiment: depth=$DEPTH shards=$NUM_SHARDS run=$RUN_NAME"

module purge
module load libs/cuda

source "$VENV_PATH/bin/activate"
which python
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Report reset ─────────────────────────────────────────────────────
python -m nanochat.report reset

# Tokenizer

# Download the first ~2B characters of pretraining dataset
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# look at dev/repackage_data_reference.py for details on how this data was prepared
python -m nanochat.dataset -n $NUM_SHARDS_INIT
# Immediately also kick off downloading more shards in the background while tokenizer trains
# Approximately 350 shards are needed for 10B tokens of data for pretraining.
# The maximum total number of shards available in the entire dataset is 1822.
python -m nanochat.dataset -n $NUM_SHARDS_BACKGROUND &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
python -m scripts.tok_train
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval


# ── Wait for dataset before training ─────────────────────────────────
echo "Waiting for dataset download (PID $DATASET_DOWNLOAD_PID)..."
wait "$DATASET_DOWNLOAD_PID" || { echo "Dataset download failed"; exit 1; }

# ── Model training ───────────────────────────────────────────────────
torchrun --standalone --nproc_per_node=gpu -m scripts.base_train -- \
    --depth="$DEPTH" \
    --run="$RUN_NAME" \
    --model-tag="$MODEL_TAG" \
    --core-metric-every="$CORE_METRIC_EVERY" \
    --sample-every="$SAMPLE_EVERY" \
    --save-every="$SAVE_EVERY" \
    --device-batch-size=$DEVICE_BATCH_SIZE

torchrun --standalone --nproc_per_node=gpu -m scripts.base_eval -- \
    --device-batch-size=$DEVICE_BATCH_SIZE

echo "Job $SLURM_JOB_ID finished at $(date)"