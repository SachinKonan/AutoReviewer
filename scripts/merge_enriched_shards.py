#!/usr/bin/env python3
"""
Merge enriched dataset shards into a final DatasetDict.

After running enrichen_train.sbatch and enrichen_test.sbatch,
this script combines the shards into a single dataset.

Usage:
    python scripts/merge_enriched_shards.py
    python scripts/merge_enriched_shards.py --input-path data/iclr_2020_2025_80_20_split_rich
"""

import argparse
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_from_disk


# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_INPUT = DATA_DIR / "iclr_2020_2025_80_20_split_rich"


def merge_shards(shards_dir: Path, num_shards: int = 10) -> None:
    """Load and concatenate all shards for a split."""
    shards = []
    missing = []

    for i in range(num_shards):
        shard_path = shards_dir / f"shard_{i:02d}"
        if shard_path.exists():
            print(f"  Loading {shard_path.name}...")
            shard = load_from_disk(str(shard_path))
            shards.append(shard)
            print(f"    {len(shard):,} rows")
        else:
            missing.append(i)

    if missing:
        print(f"  WARNING: Missing shards: {missing}")

    if not shards:
        return None

    print(f"  Concatenating {len(shards)} shards...")
    merged = concatenate_datasets(shards)
    print(f"  Total: {len(merged):,} rows")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge enriched dataset shards")
    parser.add_argument('--input-path', type=Path, default=DEFAULT_INPUT,
                        help="Path containing train/ and test/ shard directories")
    parser.add_argument('--num-shards', type=int, default=20,
                        help="Number of shards per split")
    args = parser.parse_args()

    print("=" * 60)
    print("MERGE ENRICHED SHARDS")
    print("=" * 60)
    print(f"Input: {args.input_path}")

    # Merge train
    print("\n1. Merging train shards...")
    train_dir = args.input_path / "train"
    if train_dir.exists():
        train_ds = merge_shards(train_dir, args.num_shards)
    else:
        print(f"  WARNING: {train_dir} not found")
        train_ds = None

    # Merge test
    print("\n2. Merging test shards...")
    test_dir = args.input_path / "test"
    if test_dir.exists():
        test_ds = merge_shards(test_dir, args.num_shards)
    else:
        print(f"  WARNING: {test_dir} not found")
        test_ds = None

    if not train_ds and not test_ds:
        print("\nERROR: No shards found!")
        return

    # Create DatasetDict
    print("\n3. Creating DatasetDict...")
    splits = {}
    if train_ds:
        splits["train"] = train_ds
    if test_ds:
        splits["test"] = test_ds

    dataset_dict = DatasetDict(splits)

    # Save merged dataset
    output_path = args.input_path / "merged"
    print(f"\n4. Saving to {output_path}...")
    dataset_dict.save_to_disk(str(output_path))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for split_name, ds in dataset_dict.items():
        print(f"  {split_name}: {len(ds):,} rows")
    print(f"\nOutput: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
