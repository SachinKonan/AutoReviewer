#!/bin/bash
# Convert PDFs to Markdown using marker_chunk_convert
# Usage: ./sh/run_marker_convert.sh --year 2024 [--num-gpus 4] [--num-workers 15]

set -e

# Defaults
NUM_GPUS=4
NUM_WORKERS=15
YEAR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --year)
            YEAR="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --year YEAR [--num-gpus N] [--num-workers N]"
            echo ""
            echo "Options:"
            echo "  --year        Year to process (required)"
            echo "  --num-gpus    Number of GPUs (default: 4)"
            echo "  --num-workers Number of workers (default: 15)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate year
if [[ -z "$YEAR" ]]; then
    echo "Error: --year is required"
    echo "Usage: $0 --year YEAR [--num-gpus N] [--num-workers N]"
    exit 1
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INPUT_DIR="$PROJECT_DIR/data/full_run/pdfs/$YEAR"
OUTPUT_DIR="$PROJECT_DIR/data/full_run/mds/$YEAR"

# Export for marker
export NUM_DEVICES=$NUM_GPUS
export NUM_WORKERS=$NUM_WORKERS

echo "================================================================================"
echo "PDF to Markdown Conversion using marker_chunk_convert"
echo "================================================================================"
echo "Year:        $YEAR"
echo "NUM_DEVICES: $NUM_DEVICES"
echo "NUM_WORKERS: $NUM_WORKERS"
echo "Input:       $INPUT_DIR"
echo "Output:      $OUTPUT_DIR"
echo "================================================================================"

# Check input directory exists
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Count PDFs
pdf_count=$(find "$INPUT_DIR" -maxdepth 1 -name "*.pdf" | wc -l)
echo "Found $pdf_count PDFs"
echo ""

# Run marker_chunk_convert
if marker_chunk_convert "$INPUT_DIR" "$OUTPUT_DIR"; then
    echo ""
    echo "================================================================================"
    echo "Completed! Output: $OUTPUT_DIR"
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo "Error processing (exit code: $?)"
    echo "================================================================================"
    exit 1
fi
