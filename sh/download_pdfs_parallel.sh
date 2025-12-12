#!/bin/bash
# Download PDFs for all years in parallel (locally)
# Each year runs as a background process

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"

# Activate virtual environment
source .venv/bin/activate

# First, regenerate the manifest to ensure it's up to date
echo "Regenerating PDF manifest..."
python lib/extract_pdf_manifest.py
echo ""

echo "Starting parallel PDF downloads for years 2025-2026..."
echo "Logs will be written to $LOG_DIR/"
echo ""

# Run each year in background
for year in 2025 2026; do
    echo "Starting year $year..."
    python lib/download_pdfs.py --year $year --skip-existing \
        > "$LOG_DIR/download_pdfs_${year}.log" 2>&1 &
    echo "  PID: $! -> $LOG_DIR/download_pdfs_${year}.log"
done

echo ""
echo "All jobs started. Monitor with:"
echo "  tail -f $LOG_DIR/download_pdfs_*.log"
echo ""
echo "Or check status with:"
echo "  jobs -l"
echo ""

# Wait for all background jobs
wait

echo "All downloads complete!"
echo ""

# Print summary
for year in 2025 2026; do
    echo "=== $year ==="
    tail -10 "$LOG_DIR/download_pdfs_${year}.log" | grep -E "(Success|Failed|Skipped|SUMMARY)"
    echo ""
done
