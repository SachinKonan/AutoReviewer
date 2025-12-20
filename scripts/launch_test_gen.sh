#!/bin/bash
# Launch script for normalize_reviews.py preview mode
# Sets up Ray cluster and runs the normalization pipeline

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Default values
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
OUTPUT_PATH="${OUTPUT_PATH:-data/test_normalized_reviews/preview.parquet}"
PREVIEW_PER_YEAR="${PREVIEW_PER_YEAR:-5}"

echo "=========================================="
echo "Review Normalization - Preview Mode"
echo "=========================================="
echo "Project directory: $PROJECT_DIR"
echo "Model: $MODEL_NAME"
echo "Tensor parallel: $TENSOR_PARALLEL"
echo "Output path: $OUTPUT_PATH"
echo "Preview per year: $PREVIEW_PER_YEAR"
echo "=========================================="
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "Virtual environment activated"
echo ""

# Start Ray with unique port and temp dir based on SLURM_JOB_ID or PID
RAY_PORT=$((6379 + (${SLURM_JOB_ID:-$$} % 1000)))
RAY_TEMP_DIR="/tmp/ray_tmp/${SLURM_JOB_ID:-$$}"
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
echo ""

# Cleanup function to stop Ray on exit
cleanup() {
    echo ""
    echo "Stopping Ray..."
    ray stop
    echo "Ray stopped."
}
trap cleanup EXIT

# Run normalization
echo "=========================================="
echo "Starting Review Normalization"
echo "=========================================="
echo ""

python -u lib/normalize_reviews.py \
    --preview \
    --preview-per-year "$PREVIEW_PER_YEAR" \
    --output-path "$OUTPUT_PATH" \
    --model-name "$MODEL_NAME" \
    --tensor-parallel "$TENSOR_PARALLEL" \
    --ray-address "$RAY_GCS_ADDRESS"

echo ""
echo "=========================================="
echo "Normalization completed!"
echo "Output: $OUTPUT_PATH"
echo "=========================================="
