#!/bin/bash
#SBATCH --job-name=marker_2025p4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=96:00:00
#SBATCH --output=logs/marker_2025p4_%j.log
#SBATCH --error=logs/marker_2025p4_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"

cd /n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer

source /n/fs/vision-mix/jl0796/iclr_data/.venv/bin/activate

echo "Python version: $(python --version)"
nvidia-smi

INPUT_DIR="data/full_run/pdfs/2025p4"
OUTPUT_DIR="data/full_run/mds/2025md4"

export NUM_DEVICES=8
export NUM_WORKERS=3

echo "================================================================================"
echo "PDF to Markdown Conversion - 2025 Chunk 4"
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "================================================================================"

pdf_count=$(find "$INPUT_DIR" -maxdepth 1 -name "*.pdf" | wc -l)
echo "Found $pdf_count PDFs"

marker_chunk_convert "$INPUT_DIR" "$OUTPUT_DIR"

echo "Job finished at: $(date)"
