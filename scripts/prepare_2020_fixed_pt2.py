"""Prepare batch directory for 2020 PDFs that need reprocessing.

Includes submissions where:
- Manifest has a download_id for 2020
- Existing MinerU output (any batch) has a DIFFERENT download_id
"""
import csv
import os
from pathlib import Path


def find_by_prefix(submission_id: str, directory: Path) -> Path | None:
    """Find PDF file that starts with submission_id prefix."""
    for pdf in directory.glob("*.pdf"):
        if pdf.stem.startswith(submission_id + "_") or pdf.stem == submission_id:
            return pdf
    return None


def find_mineru_output(submission_id: str, mineru_dir: Path) -> tuple[str, str] | None:
    """Find existing MinerU output for submission across all batches.

    Returns (batch_name, folder_name) or None if not found.
    """
    for batch_dir in mineru_dir.iterdir():
        if not batch_dir.is_dir() or not batch_dir.name.startswith("batch_"):
            continue
        # Skip pt2 itself
        if batch_dir.name == "batch_2020_fixed_pt2":
            continue
        # Use scandir for efficiency
        with os.scandir(batch_dir) as entries:
            for entry in entries:
                if entry.is_dir() and entry.name.startswith(submission_id + "_"):
                    return batch_dir.name, entry.name
    return None


def prepare_batch():
    manifest_path = Path("data/pdf_manifest.csv")
    pdf_dir = Path("data/full_run/pdfs/2020")
    mineru_dir = Path("data/full_run/md_mineru")
    pt2_batch = Path("data/mineru_batches/batch_2020_fixed_pt2")

    pt2_batch.mkdir(parents=True, exist_ok=True)

    # Clear existing symlinks
    for link in pt2_batch.glob("*.pdf"):
        link.unlink()

    # Load 2020 entries from manifest
    manifest_entries = {}
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['year'] == '2020':
                manifest_entries[row['submission_id']] = row.get('download_id', '')

    print(f"Found {len(manifest_entries)} 2020 entries in manifest")

    # Find submissions that need reprocessing
    created = 0
    for sub_id, manifest_dl_id in sorted(manifest_entries.items()):
        if not manifest_dl_id:
            continue

        # Find existing MinerU output
        result = find_mineru_output(sub_id, mineru_dir)
        if not result:
            continue

        batch_name, folder_name = result
        # Extract download_id from folder name (everything after submission_id_)
        existing_dl_id = folder_name[len(sub_id) + 1:]  # +1 for underscore

        # If download_ids differ, needs reprocessing
        if existing_dl_id != manifest_dl_id:
            pdf = find_by_prefix(sub_id, pdf_dir)
            if pdf:
                link = pt2_batch / pdf.name
                if not link.exists():
                    link.symlink_to(pdf.resolve())
                    print(f"  {pdf.name} (was {folder_name} in {batch_name})")
                    created += 1

    print(f"\nCreated {created} symlinks in {pt2_batch}")


if __name__ == "__main__":
    prepare_batch()
