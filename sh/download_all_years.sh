#!/bin/bash
# Download OpenReview metadata for all years in parallel
# Each year runs as a background process

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"

# Activate virtual environment
source .venv/bin/activate

echo "Starting parallel downloads for years 2020-2026..."
echo "Logs will be written to $LOG_DIR/"
echo ""

# Run each year in background
for year in 2020 2021 2022 2023 2024 2025; do
    echo "Starting year $year..."
    python lib/download_metadata_v2.py --download --year $year \
        > "$LOG_DIR/download_${year}.log" 2>&1 &
    echo "  PID: $! -> $LOG_DIR/download_${year}.log"
done

echo ""
echo "All jobs started. Monitor with:"
echo "  tail -f $LOG_DIR/download_*.log"
echo ""
echo "Or check status with:"
echo "  jobs -l"
echo ""

# Wait for all background jobs
wait

echo "All downloads complete!"
