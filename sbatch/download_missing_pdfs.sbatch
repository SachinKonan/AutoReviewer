#!/bin/bash
#SBATCH --job-name=download_missing_pdfs
#SBATCH --output=logs/download_missing_pdfs_%j.out
#SBATCH --error=logs/download_missing_pdfs_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

set -e

cd /n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer
source .venv/bin/activate

mkdir -p logs

echo "Starting download of missing PDFs (from data/missing_pdfs.json)"
echo "Start time: $(date)"
echo ""

# Run each year in background (uses missing_pdfs.json, not manifest)
for year in 2024 2025 2026; do
    echo "Starting year $year..."
    python lib/download_missing_pdfs.py --year $year > logs/download_missing_pdfs_${year}.log 2>&1 &
done

echo "All years started in background. Waiting for completion..."
wait

echo ""
echo "All downloads complete!"
echo "End time: $(date)"

# Print summary
echo ""
echo "=== LOGS ==="
for year in 2024 2025 2026; do
    echo "--- $year ---"
    tail -15 logs/download_missing_pdfs_${year}.log
done
