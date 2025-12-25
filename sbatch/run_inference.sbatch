#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Run zero-shot inference on ICLR test split
#
# Usage:
#   sbatch sbatch/run_inference.sbatch
#   sbatch sbatch/run_inference.sbatch --export=OUTPUT_TYPE=multiclass

cd /n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer
source .venv/bin/activate

mkdir -p logs outputs

# Default to binary if not specified
OUTPUT_TYPE=${OUTPUT_TYPE:-binary}

echo "Starting inference at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Output type: $OUTPUT_TYPE"

python scripts/run_inference.py \
    --output-type "$OUTPUT_TYPE" \
    --output-path "outputs/inference_${OUTPUT_TYPE}.json" \
    --batch-size 8

echo "Completed at $(date)"
