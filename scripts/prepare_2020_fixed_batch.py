"""Prepare batch directory for fixed 2020 PDFs."""
import csv
from pathlib import Path


def find_by_prefix(submission_id: str, directory: Path) -> Path | None:
    """Find PDF file that starts with submission_id prefix."""
    for pdf in directory.glob("*.pdf"):
        if pdf.stem.startswith(submission_id + "_") or pdf.stem == submission_id:
            return pdf
    return None


def prepare_batch():
    pdf_dir = Path("data/full_run/pdfs")
    backup_dir = pdf_dir / "2020_old_download_id"
    new_dir = pdf_dir / "2020"
    batch_dir = Path("data/mineru_batches/batch_2020_fixed")
    manifest_path = Path("data/pdf_manifest.csv")

    batch_dir.mkdir(parents=True, exist_ok=True)

    # Load 2020 submission IDs from manifest
    sub_ids_2020 = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['year'] == '2020':
                sub_ids_2020.append(row['submission_id'])

    # Find which ones have backups (these are the changed ones)
    changed_ids = []
    for sub_id in sub_ids_2020:
        if find_by_prefix(sub_id, backup_dir):
            changed_ids.append(sub_id)

    print(f"Found {len(changed_ids)} changed submission IDs")

    # Create symlinks to new PDFs
    created = 0
    for sub_id in changed_ids:
        new_pdf = find_by_prefix(sub_id, new_dir)
        if new_pdf:
            link = batch_dir / new_pdf.name
            if not link.exists():
                link.symlink_to(new_pdf.resolve())
                created += 1

    print(f"Created {created} symlinks in {batch_dir}")


if __name__ == "__main__":
    prepare_batch()
