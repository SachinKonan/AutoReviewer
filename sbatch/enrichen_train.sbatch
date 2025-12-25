#!/bin/bash
#SBATCH --job-name=enrich_train
#SBATCH --output=logs/enrichen/enrichen_train_%A_%a.out
#SBATCH --error=logs/enrichen/enrichen_train_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-19

# Enrich HF Dataset - Train Split
# Loads file contents and images from paths
# Splits workload across 10 workers
#
# Usage: sbatch sbatch/enrichen_train.sbatch

cd /n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer
source .venv/bin/activate

mkdir -p logs/enrichen

echo "Starting train enrichment at $(date)"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"

python scripts/enrichen_hf_dataset_subset.py \
    --split train \
    --task-id $SLURM_ARRAY_TASK_ID \
    --num-tasks 20

echo "Task $SLURM_ARRAY_TASK_ID completed at $(date)"
