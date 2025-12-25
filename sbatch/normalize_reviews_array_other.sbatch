#!/bin/bash
#SBATCH --job-name=norm_reviews
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --output=logs/normalize_reviews_2/normalize_reviews_%A_%a.out
#SBATCH --error=logs/normalize_reviews_2/normalize_reviews_%A_%a.err

# Normalize ICLR reviews using vLLM with SLURM job array
# Each job processes 1/8 of the input parquet
#
# Usage:
#   # Step 1: Prepare input parquet (run once)
#   python -m lib.normalize_reviews --prepare-input
#
#   # Step 2: Submit job array
#   sbatch sbatch/normalize_reviews_array.sbatch

set -e  # Exit on error

cd /n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer
source .venv/bin/activate

echo "Job array task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Start Ray with unique port per task (9000 + task_id)
RAY_PORT=$((9000 + SLURM_ARRAY_TASK_ID))
RAY_TEMP_DIR="/tmp/ray_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RAY_TEMP_DIR"

echo "Starting Ray head node on port $RAY_PORT (temp: $RAY_TEMP_DIR)..."
ray start --head --port=$RAY_PORT --temp-dir="$RAY_TEMP_DIR" --disable-usage-stats --include-dashboard=false

# Read the actual GCS address from Ray's temp dir
sleep 2  # Give Ray time to write the address file
RAY_GCS_ADDRESS=$(cat "$RAY_TEMP_DIR/ray_current_cluster" 2>/dev/null || echo "")
if [ -z "$RAY_GCS_ADDRESS" ]; then
    # Fallback: construct from node IP
    RAY_GCS_ADDRESS="$(hostname -i):$RAY_PORT"
fi
echo "Ray GCS address: $RAY_GCS_ADDRESS"

# Cleanup function to stop Ray on exit
cleanup() {
    echo ""
    echo "Stopping Ray..."
    ray stop
    rm -rf "$RAY_TEMP_DIR"
    echo "Ray stopped."
}
trap cleanup EXIT

echo ""
echo "Starting batch $SLURM_ARRAY_TASK_ID normalization..."

python -u -m lib.normalize_reviews \
    --input-parquet data/normalized_reviews/reviews_input.parquet \
    --output-dir data/normalized_reviews_2 \
    --batch-idx $SLURM_ARRAY_TASK_ID \
    --num-batches 2 \
    --tensor-parallel 4 \
    --concurrency 2 \
    --n-completions 3 \
    --max-model-len 32768 \
    --max-tokens 16384 \
    --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --ray-address "$RAY_GCS_ADDRESS"

echo "Batch $SLURM_ARRAY_TASK_ID complete"
