#!/bin/bash
#SBATCH --job-name=uv_sync
#SBATCH --output=logs/uv_sync_%j.out
#SBATCH --error=logs/uv_sync_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=400G
#SBATCH --time=24:00:00

set -e

module load cudatoolkit/12.6

export MAX_JOBS=10

cd /n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer
source .venv/bin/activate

uv sync

echo "Done"