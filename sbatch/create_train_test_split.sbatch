#!/bin/bash
#SBATCH --job-name=train_test_split
#SBATCH --output=logs/train_test_split_%j.out
#SBATCH --error=logs/train_test_split_%j.out
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=all

echo "Starting train/test split creation..."
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "============================================================"

cd /n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer
source .venv/bin/activate

python -u scripts/create_train_test_split.py

echo "============================================================"
echo "Train/test split creation complete!"
echo "End time: $(date)"
