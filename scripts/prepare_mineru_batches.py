"""Prepare PDF batches for parallel MinerU processing (non-withdrawn only)."""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.submission import load_submissions_from_pickle
from lib.utils import (
    build_pdf_file_index,
    find_pdf_path,
    EXCLUDED_DECISIONS,
    DEFAULT_YEARS,
)


def prepare_batches(
    data_dir: Path = Path("data/full_run"),
    batch_dir: Path = Path("data/mineru_batches"),
    batch_size: int = 2200,
    years: list = None,
):
    """
    Create symlink batches for non-withdrawn PDFs only.

    Filters PDFs by:
    1. Loading submissions from pickle files
    2. Excluding Withdrawn and Desk Reject decisions
    3. Only including PDFs that match non-withdrawn submission IDs
    """
    if years is None:
        years = DEFAULT_YEARS

    pdf_dir = data_dir / "pdfs"

    # Build PDF index per year (sorted for binary search)
    print("Building PDF index...")
    pdf_index = build_pdf_file_index(pdf_dir, years)

    # Collect non-withdrawn PDFs across years
    all_pdfs = []
    for year in years:
        pkl_path = data_dir / f"get_all_notes_{year}.pickle"
        if not pkl_path.exists():
            print(f"  {year}: pickle not found, skipping")
            continue

        subs = load_submissions_from_pickle(pkl_path, year)
        year_count = 0

        for sub in subs:
            # Skip withdrawn/desk rejected
            if sub.decision in EXCLUDED_DECISIONS:
                continue

            # Find matching PDF
            pdf_path = find_pdf_path(sub.id, year, pdf_index, pdf_dir)
            if pdf_path and pdf_path.exists():
                all_pdfs.append(pdf_path)
                year_count += 1

        print(f"  {year}: {year_count} non-withdrawn PDFs")

    print(f"\nTotal non-withdrawn PDFs: {len(all_pdfs)}")

    # Create batch directories with symlinks
    batch_dir.mkdir(parents=True, exist_ok=True)

    batch_num = 0
    for i in range(0, len(all_pdfs), batch_size):
        batch_pdfs = all_pdfs[i : i + batch_size]
        batch_path = batch_dir / f"batch_{batch_num:02d}"
        batch_path.mkdir(exist_ok=True)

        # Create symlinks to original PDFs
        for pdf in batch_pdfs:
            link = batch_path / pdf.name
            if not link.exists():
                link.symlink_to(pdf.resolve())

        print(f"Batch {batch_num}: {len(batch_pdfs)} PDFs")
        batch_num += 1

    print(f"\nCreated {batch_num} batches in {batch_dir}")


if __name__ == "__main__":
    prepare_batches()
