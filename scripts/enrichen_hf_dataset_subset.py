#!/usr/bin/env python3
"""
Enrich HF dataset by loading file contents and images from paths.

Transforms:
- original_md_path -> original_md_content (load file)
- clean_md_path -> clean_md_content (load file)
- images_in_clean_md (JSON dict) -> md_images (List[{name: str, image: Image}])
- clean_pdf_img_paths (JSON list) -> pdf_page_images (List[Image])

Images are stored as HuggingFace Image feature type (PIL-compatible).

Usage:
    # Process train split with 10 workers
    python enrichen_hf_dataset_subset.py --split train --task-id 0 --num-tasks 10

    # Process test split
    python enrichen_hf_dataset_subset.py --split test --task-id 0 --num-tasks 10
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, Features, Image, Sequence, Value, load_from_disk
from PIL import Image as PILImage


# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_INPUT = DATA_DIR / "iclr_2020_2025_80_20_split"
DEFAULT_OUTPUT = DATA_DIR / "iclr_2020_2025_80_20_split_rich"


def load_file_content(path: str) -> str:
    """Load text file content, return empty string if not found."""
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding='utf-8')
    except Exception:
        return ""


def load_pil_image(path: str) -> PILImage.Image | None:
    """Load image as PIL Image for HF Dataset Image feature."""
    if not path:
        return None
    try:
        return PILImage.open(path).convert('RGB')
    except Exception:
        return None


def enrich_row(example):
    """Transform a single row: load file contents and images."""
    # Load markdown content
    example['original_md_content'] = load_file_content(example.get('original_md_path', ''))
    example['clean_md_content'] = load_file_content(example.get('clean_md_path', ''))

    # Load images from clean_md as list of {name, image} dicts
    images_json = example.get('images_in_clean_md') or '{}'
    try:
        images_dict = json.loads(images_json)
    except Exception:
        images_dict = {}

    # Store as list of {name, image} structs for HF compatibility
    # This is functionally equivalent to Dict[str, Image]
    md_images = []
    for name, path in images_dict.items():
        img = load_pil_image(path)
        if img is not None:
            md_images.append({"name": name, "image": img})
    example['md_images'] = md_images

    # Load PDF page images as list of PIL Images
    pdf_json = example.get('clean_pdf_img_paths') or '[]'
    try:
        pdf_paths = json.loads(pdf_json)
    except Exception:
        pdf_paths = []

    pdf_images = []
    for path in pdf_paths:
        img = load_pil_image(path)
        if img is not None:
            pdf_images.append(img)
    example['pdf_page_images'] = pdf_images

    return example


def main():
    parser = argparse.ArgumentParser(description="Enrich HF dataset with file contents and images")
    parser.add_argument('--input-path', type=Path, default=DEFAULT_INPUT,
                        help="Input dataset path")
    parser.add_argument('--output-path', type=Path, default=DEFAULT_OUTPUT,
                        help="Output dataset path")
    parser.add_argument('--split', type=str, required=True, choices=['train', 'test'],
                        help="Which split to process")
    parser.add_argument('--task-id', type=int, required=True,
                        help="Task ID for sbatch array (0-indexed)")
    parser.add_argument('--num-tasks', type=int, default=10,
                        help="Total number of tasks")
    args = parser.parse_args()

    print("=" * 60)
    print("ENRICH HF DATASET")
    print("=" * 60)
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Split: {args.split}")
    print(f"Task: {args.task_id} / {args.num_tasks}")

    # Load dataset split
    print(f"\n1. Loading dataset split '{args.split}'...")
    ds_dict = load_from_disk(str(args.input_path))
    ds = ds_dict[args.split]
    print(f"   Total rows in {args.split}: {len(ds):,}")

    # Compute subset range (contiguous chunks)
    total = len(ds)
    chunk_size = total // args.num_tasks
    remainder = total % args.num_tasks

    if args.task_id < remainder:
        start = args.task_id * (chunk_size + 1)
        end = start + chunk_size + 1
    else:
        start = remainder * (chunk_size + 1) + (args.task_id - remainder) * chunk_size
        end = start + chunk_size

    # Select subset
    print(f"\n2. Selecting subset [{start}:{end}]...")
    subset = ds.select(range(start, end))
    print(f"   Subset size: {len(subset):,} rows")

    # Transform with progress
    print(f"\n3. Enriching rows...")
    enriched = subset.map(
        enrich_row,
        desc=f"Enriching {args.split}[{start}:{end}]",
        num_proc=1,
        writer_batch_size=50,  # Flush to disk frequently to reduce memory
    )

    # Cast image columns to HF Image type
    print(f"\n4. Casting image columns to HF Image type...")
    # md_images: List[{name: str, image: Image}] - functionally Dict[str, Image]
    md_images_feature = Sequence({"name": Value("string"), "image": Image()})
    enriched = enriched.cast_column('md_images', md_images_feature)
    # pdf_page_images: List[Image]
    enriched = enriched.cast_column('pdf_page_images', Sequence(Image()))

    # Save to output
    output_dir = args.output_path / args.split / f"shard_{args.task_id:02d}"
    print(f"\n5. Saving to {output_dir}...")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    enriched.save_to_disk(str(output_dir))

    print(f"\n   Done! Saved {len(enriched):,} rows")
    print("=" * 60)


if __name__ == "__main__":
    main()
