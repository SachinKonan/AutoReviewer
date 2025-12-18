#!/usr/bin/env python3
"""Split remaining 2025 PDFs into 4 chunks with symlinks."""
from pathlib import Path

data_dir = Path("/n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer/data/full_run")
pdf_dir = data_dir / "pdfs/2025"
md_dir = data_dir / "mds/2025"

# Get all PDF basenames and completed MD folders
all_pdfs = sorted([f.stem for f in pdf_dir.glob("*.pdf")])
completed_mds = set(d.name for d in md_dir.iterdir() if d.is_dir() and (d / f"{d.name}.md").exists())

# Find remaining PDFs (no matching MD folder)
remaining = [p for p in all_pdfs if p not in completed_mds]
print(f"Total PDFs: {len(all_pdfs)}")
print(f"Completed MDs: {len(completed_mds)}")
print(f"Remaining: {len(remaining)}")

# Split into 4 chunks
chunk_size = len(remaining) // 4
chunks = [remaining[i*chunk_size:(i+1)*chunk_size] for i in range(4)]
chunks[3].extend(remaining[4*chunk_size:])  # Add remainder to last chunk

# Create symlink directories and output directories
for i, chunk in enumerate(chunks, 1):
    # Input dir with symlinks
    chunk_pdf_dir = data_dir / f"pdfs/2025p{i}"
    chunk_pdf_dir.mkdir(exist_ok=True)

    # Output dir
    chunk_md_dir = data_dir / f"mds/2025md{i}"
    chunk_md_dir.mkdir(exist_ok=True)

    # Create symlinks
    for pdf_name in chunk:
        src = pdf_dir / f"{pdf_name}.pdf"
        dst = chunk_pdf_dir / f"{pdf_name}.pdf"
        if not dst.exists():
            dst.symlink_to(src)

    print(f"Chunk {i}: {len(chunk)} PDFs -> {chunk_pdf_dir}")
