#!/bin/bash
#SBATCH --job-name=marker_convert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=96:00:00
#SBATCH --output=logs/marker_convert_%x_%j.log
#SBATCH --error=logs/marker_convert_%x_%j.err

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"

# Navigate to working directory
cd /n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer

# Use JL's venv which has marker installed
source /n/fs/vision-mix/jl0796/iclr_data/.venv/bin/activate

# Print Python and package versions
echo "Python version: $(python --version)"
echo "Marker location: $(which marker_chunk_convert)"

# Check GPU availability
nvidia-smi

# Get year from command line argument
YEAR=${1:-2025}

echo "Processing year: $YEAR"
echo "================================================================================"

# Run marker conversion: NUM_DEVICES=8, NUM_WORKERS=3
./sh/run_marker_convert.sh --year $YEAR --num-gpus $SLURM_GPUS_ON_NODE --num-workers 3

echo "Job finished at: $(date)"
