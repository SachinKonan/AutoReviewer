#!/bin/bash
#SBATCH --job-name=generate_hf_dataset
#SBATCH --output=logs/generate_hf_dataset_%j.out
#SBATCH --error=logs/generate_hf_dataset_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=400G
#SBATCH --time=12:00:00

set -e

cd /n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer
source .venv/bin/activate

python scripts/generate_hf_dataset.py --years 2020 2021 2022 2023 2024 2025 2026 --max-workers 30 --output-dir data/full_run/raw_hf_dump_v5 --citations-db data/citations.db --upload --repo-id skonan/iclr-data-2020-2026

echo "Done"