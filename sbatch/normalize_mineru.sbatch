#!/bin/bash
#SBATCH --job-name=normalize_mineru
#SBATCH --output=logs/normalize_mineru_%j.out
#SBATCH --error=logs/normalize_mineru_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# MinerU Normalization Pipeline
# Usage: sbatch sbatch/normalize_mineru.sbatch [optional args]
# Example: sbatch sbatch/normalize_mineru.sbatch --year 2024
# Example: sbatch sbatch/normalize_mineru.sbatch --limit 1000

cd /n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer
source .venv/bin/activate

mkdir -p logs

echo "Starting normalization at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"

python -m lib.normalize_mineru --workers 8 "$@"

echo "Completed at $(date)"
