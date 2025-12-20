#!/bin/bash
#SBATCH --job-name=pdf2img
#SBATCH --output=logs/pdf_to_images/pdf2img_%A_%a.out
#SBATCH --error=logs/pdf_to_images/pdf2img_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-39

# PDF to Images Pipeline - Job Array Version
# Splits workload across 40 workers
#
# Usage: sbatch sbatch/pdf_to_images_array.sbatch

cd /n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer
source .venv/bin/activate

mkdir -p logs/pdf_to_images

echo "Starting PDF to images at $(date)"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"

python lib/pdf_to_images.py --all --task-id $SLURM_ARRAY_TASK_ID --num-tasks 40

echo "Task $SLURM_ARRAY_TASK_ID completed at $(date)"
