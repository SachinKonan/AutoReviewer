"""Merge MinerU batch outputs into year-based structure."""
import shutil
from pathlib import Path


def merge_outputs(
    batch_output_dir: Path = Path("data/full_run/md_mineru"),
    final_output_dir: Path = Path("data/full_run/mds"),
    pdf_base_dir: Path = Path("data/full_run/pdfs"),
):
    # Build PDF→year mapping
    pdf_to_year = {}
    for year_dir in pdf_base_dir.iterdir():
        if year_dir.is_dir() and year_dir.name.isdigit():
            for pdf in year_dir.glob("*.pdf"):
                pdf_to_year[pdf.stem] = year_dir.name

    print(f"Built mapping for {len(pdf_to_year)} PDFs")

    # Process each batch output
    copied = 0
    skipped = 0
    for batch_dir in sorted(batch_output_dir.glob("batch_*")):
        for md_folder in batch_dir.iterdir():
            if md_folder.is_dir():
                # Get year from PDF stem
                stem = md_folder.name
                year = pdf_to_year.get(stem)
                if year:
                    dest = final_output_dir / year / md_folder.name
                    if not dest.exists():
                        shutil.copytree(md_folder, dest)
                        print(f"Copied {md_folder.name} → {year}/")
                        copied += 1
                    else:
                        skipped += 1

    print(f"\nDone: {copied} copied, {skipped} skipped (already exist)")


if __name__ == "__main__":
    merge_outputs()
