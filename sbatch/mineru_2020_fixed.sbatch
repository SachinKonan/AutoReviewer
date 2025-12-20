#!/bin/bash
#SBATCH --job-name=mineru_2020
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --output=logs/mineru_2020_fixed_%j.log
#SBATCH --error=logs/mineru_2020_fixed_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

cd /n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer
source .venv/bin/activate

nvidia-smi

INPUT_DIR="data/mineru_batches/batch_2020_fixed"
OUTPUT_DIR="data/full_run/md_mineru/batch_2020_fixed"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

pdf_count=$(find "$INPUT_DIR" -maxdepth 1 -name "*.pdf" | wc -l)

echo "========================================"
echo "MinerU - Fixed 2020 PDFs"
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "PDFs:   $pdf_count"
echo "========================================"

mineru -p "$INPUT_DIR" -o "$OUTPUT_DIR" -b vlm-vllm-engine --end 14

echo "Job finished at: $(date)"
