#!/bin/bash
#SBATCH --job-name=mineru
#SBATCH --array=0-18
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=logs/mineru_%a_%j.log
#SBATCH --error=logs/mineru_%a_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"

cd /n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer
source .venv/bin/activate

echo "Python version: $(python --version)"
nvidia-smi

BATCH_ID=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
INPUT_DIR="data/mineru_batches/batch_${BATCH_ID}"
OUTPUT_DIR="data/full_run/md_mineru/batch_${BATCH_ID}"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

pdf_count=$(find "$INPUT_DIR" -maxdepth 1 -name "*.pdf" | wc -l)

echo "================================================================================"
echo "MinerU PDF to Markdown - Batch $BATCH_ID"
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Found $pdf_count PDFs (symlinks)"
echo "================================================================================"

# Run MinerU with vllm-engine backend (uses port 30000), max 14 pages
mineru -p "$INPUT_DIR" -o "$OUTPUT_DIR" -b vlm-vllm-engine --end 14

echo "Job finished at: $(date)"
