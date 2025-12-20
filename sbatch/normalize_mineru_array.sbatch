#!/bin/bash
#SBATCH --job-name=norm_mineru
#SBATCH --output=logs/normalize/normalize_mineru_%A_%a.out
#SBATCH --error=logs/normalize/normalize_mineru_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-39

# MinerU Normalization Pipeline - Job Array Version
# Splits workload across 40 workers
#
# Usage: sbatch sbatch/normalize_mineru_array.sbatch [optional args]
# Example: sbatch sbatch/normalize_mineru_array.sbatch --year 2024
# Example: sbatch sbatch/normalize_mineru_array.sbatch --force
#
# Each array task processes 1/40th of the submissions

cd /n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer
source .venv/bin/activate

mkdir -p logs

echo "Starting normalization at $(date)"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"

python -m lib.normalize_mineru --workers 8 --task-id $SLURM_ARRAY_TASK_ID --num-tasks 40 "$@"

echo "Task $SLURM_ARRAY_TASK_ID completed at $(date)"
