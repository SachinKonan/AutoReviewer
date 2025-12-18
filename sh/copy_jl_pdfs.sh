#!/bin/bash
# Copy PDFs from JL's directory to ours with [id]_.pdf naming convention
# Usage: ./sh/copy_jl_pdfs.sh --year 2026

set -e

YEAR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --year)
            YEAR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --year YEAR"
            echo "Copy PDFs from JL's directory to ours with [id]_.pdf naming"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$YEAR" ]]; then
    echo "Error: --year is required"
    exit 1
fi

SRC_DIR="/n/fs/vision-mix/jl0796/iclr_data/full_iclr_pdfs/ICLR/${YEAR}"
DST_DIR="/n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer/data/full_run/pdfs/${YEAR}"

if [[ ! -d "$SRC_DIR" ]]; then
    echo "Error: Source directory does not exist: $SRC_DIR"
    exit 1
fi

mkdir -p "$DST_DIR"

echo "Source: $SRC_DIR"
echo "Dest:   $DST_DIR"
echo ""

src_count=$(ls "$SRC_DIR"/*.pdf 2>/dev/null | wc -l)
dst_count=$(ls "$DST_DIR"/*.pdf 2>/dev/null | wc -l)
echo "Source PDFs: $src_count"
echo "Existing PDFs: $dst_count"
echo ""

copied=0
skipped=0

for src_file in "$SRC_DIR"/*.pdf; do
    filename=$(basename "$src_file")
    # Convert [id].pdf -> [id]_.pdf
    id="${filename%.pdf}"
    dst_file="$DST_DIR/${id}_.pdf"

    if [[ -f "$dst_file" ]]; then
        skipped=$((skipped + 1))
    else
        cp "$src_file" "$dst_file"
        copied=$((copied + 1))
    fi
done

echo ""
echo "Copied: $copied"
echo "Skipped: $skipped"
echo "Total: $(ls "$DST_DIR"/*.pdf | wc -l)"
