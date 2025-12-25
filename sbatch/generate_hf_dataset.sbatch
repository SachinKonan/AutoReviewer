#!/bin/bash
#SBATCH --job-name=gen_hf_ds
#SBATCH --output=logs/generate_hf_dataset_%j.out
#SBATCH --error=logs/generate_hf_dataset_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=6:00:00

# Generate HuggingFace Dataset from ICLR papers data
# Filters to "Clean %" submissions (~38,540 total)
#
# Usage:
#   mkdir -p logs
#   sbatch sbatch/generate_hf_dataset.sbatch

set -e

cd /n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer
source .venv/bin/activate

echo "Starting HuggingFace dataset generation..."
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"

python -u scripts/generate_hf_dataset.py --output data/hf_dataset_new

echo ""
echo "Dataset generation complete!"
echo "End time: $(date)"
echo "Output saved to: data/hf_dataset/"
