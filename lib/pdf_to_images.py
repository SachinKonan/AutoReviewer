#!/usr/bin/env python3
"""Convert redacted PDFs to page images at 200 DPI."""

import argparse
import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm


def pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 200) -> int:
    """Convert a PDF to page images.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save images
        dpi: Resolution in DPI (default 200)

    Returns:
        Number of pages converted
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate zoom factor for desired DPI (72 is default PDF DPI)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    doc = fitz.open(pdf_path)
    num_pages = len(doc)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix)
        # Page numbers start from 1
        output_path = output_dir / f"page_{i + 1}.png"
        pix.save(str(output_path))

    doc.close()
    return num_pages


def find_submission_dir(base_dir: Path, submission_id: str) -> Path:
    """Find submission directory by ID (searches across all years).

    Args:
        base_dir: Base directory containing data/full_run/normalized/
        submission_id: The submission ID to find

    Returns:
        Path to the submission directory

    Raises:
        ValueError: If submission not found
    """
    normalized_dir = base_dir / "data/full_run/normalized"
    for year_dir in sorted(normalized_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        sub_dir = year_dir / submission_id
        if sub_dir.exists():
            return sub_dir
    raise ValueError(f"Submission {submission_id} not found in {normalized_dir}")


def process_submission(sub_dir: Path, dpi: int = 200) -> int:
    """Process a single submission directory.

    Args:
        sub_dir: Path to submission directory
        dpi: Resolution in DPI

    Returns:
        Number of pages converted
    """
    submission_id = sub_dir.name
    pdf_path = sub_dir / f"{submission_id}.pdf"

    if not pdf_path.exists():
        print(f"Warning: PDF not found: {pdf_path}")
        return 0

    output_dir = sub_dir / "redacted_pdf_img_content"
    return pdf_to_images(pdf_path, output_dir, dpi)


def process_all(base_dir: Path, dpi: int = 200, task_id: int = None, num_tasks: int = None):
    """Process all normalized PDFs.

    Args:
        base_dir: Base directory containing data/full_run/normalized/
        dpi: Resolution in DPI
        task_id: Optional task ID for parallel processing (0-indexed)
        num_tasks: Total number of tasks for parallel processing
    """
    normalized_dir = base_dir / "data/full_run/normalized"

    # Find all submission directories
    sub_dirs = []
    for year_dir in sorted(normalized_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        for sub_dir in sorted(year_dir.iterdir()):
            if sub_dir.is_dir():
                sub_dirs.append(sub_dir)

    # Filter for task if specified
    if task_id is not None and num_tasks is not None:
        sub_dirs = [d for i, d in enumerate(sub_dirs) if i % num_tasks == task_id]

    print(f"Processing {len(sub_dirs)} submissions")

    for sub_dir in tqdm(sub_dirs, desc="Converting PDFs"):
        # Skip if already done
        output_dir = sub_dir / "redacted_pdf_img_content"
        if output_dir.exists() and any(output_dir.glob("page_*.png")):
            continue

        try:
            process_submission(sub_dir, dpi)
        except Exception as e:
            print(f"Error processing {sub_dir.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert redacted PDFs to page images")
    parser.add_argument("--base-dir", type=Path,
                        default=Path("/n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer"),
                        help="Base directory")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for output images")

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--submission-id", type=str, help="Process single submission by ID")
    mode_group.add_argument("--all", action="store_true", help="Process all submissions")

    # Parallel processing options
    parser.add_argument("--task-id", type=int, help="Task ID for parallel processing (0-indexed)")
    parser.add_argument("--num-tasks", type=int, help="Total number of tasks")

    args = parser.parse_args()

    if args.submission_id:
        # Single submission mode
        sub_dir = find_submission_dir(args.base_dir, args.submission_id)
        print(f"Found submission at: {sub_dir}")
        num_pages = process_submission(sub_dir, args.dpi)
        print(f"Converted {num_pages} pages to {sub_dir / 'redacted_pdf_img_content'}")
    else:
        # Batch mode
        process_all(args.base_dir, args.dpi, args.task_id, args.num_tasks)


if __name__ == "__main__":
    main()
