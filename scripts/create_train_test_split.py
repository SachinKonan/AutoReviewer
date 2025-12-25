#!/usr/bin/env python3
"""
Create a balanced 80/20 train/test split dataset from hf_dataset_new.

Features:
- 50/50 accept/reject balance per year (sample rejects to match accepts)
- 80% train, 20% test per year
- DeepReview-13K contamination prevention for 2024/2025:
  - Their test -> our test
  - Their train -> our train
- New columns: split, deepreview_split, deepreview_row_index
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, load_from_disk

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_PATH = DATA_DIR / "hf_dataset_new"
OUTPUT_PATH = DATA_DIR / "iclr_2020_2025_80_20_split"
DEEPREVIEW_PATH = Path(
    "/n/fs/vision-mix/sk7524/caches/.hf/hub/datasets--WestlakeNLP--DeepReview-13K"
    "/snapshots/3db597e1e789ce04af98c5eae9e9430341face23/data"
)

# Configuration
YEARS = list(range(2020, 2026))  # 2020-2025, exclude 2026
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def load_deepreview_with_row_indices() -> pd.DataFrame:
    """Load DeepReview-13K CSVs with row indices and labels."""
    files = [
        ("train.csv", "train"),
        ("test_2024.csv", "test"),
        ("test_2025.csv", "test"),
    ]

    dfs = []
    global_row_idx = 0

    for filename, label in files:
        filepath = DEEPREVIEW_PATH / filename
        if not filepath.exists():
            print(f"  Warning: {filepath} not found, skipping")
            continue

        # Only read id and year columns for efficiency
        df = pd.read_csv(filepath, usecols=["id", "year"], engine="pyarrow")
        df["label"] = label
        df["row_index"] = range(global_row_idx, global_row_idx + len(df))
        global_row_idx += len(df)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def build_deepreview_lookup(deepreview_df: pd.DataFrame) -> dict:
    """Build efficient lookup: submission_id -> (label, first_row_index)."""
    return (
        deepreview_df.groupby("id")
        .agg(label=("label", "first"), first_row_index=("row_index", "min"))
        .apply(lambda row: (row["label"], int(row["first_row_index"])), axis=1)
        .to_dict()
    )


def sample_and_split_year(
    year_df: pd.DataFrame,
    year: int,
    deepreview_lookup: dict,
    rng: np.random.Generator,
    complete_ids: set = None,
) -> tuple[list, list]:
    """
    Sample balanced accept/reject and split into train/test for one year.

    Args:
        complete_ids: Set of submission_ids that have complete data (normalized_reviews,
                      normalized_metareview, and all technical_indicators not None).
                      If provided, prefers these when sampling rejects.

    Returns:
        (train_ids, test_ids) - lists of submission_ids
    """
    accepts = year_df[year_df["decision"] == "accept"]["submission_id"].unique()
    rejects = year_df[year_df["decision"] == "reject"]["submission_id"].unique()

    # Sample rejects to match accepts (balanced 50/50)
    n_samples = len(accepts)
    if len(rejects) < n_samples:
        raise Exception(f"Not enough rejects for year {year}: {len(rejects)} < {n_samples}")

    # Prefer complete rejects if complete_ids is provided
    if complete_ids:
        complete_rejects = np.array([r for r in rejects if r in complete_ids])
        incomplete_rejects = np.array([r for r in rejects if r not in complete_ids])

        if len(complete_rejects) >= n_samples:
            # Enough complete rejects - sample only from them
            sampled_rejects = rng.choice(complete_rejects, n_samples, replace=False)
        else:
            # Take all complete, fill remainder from incomplete
            remaining = n_samples - len(complete_rejects)
            sampled_incomplete = rng.choice(incomplete_rejects, remaining, replace=False)
            sampled_rejects = np.concatenate([complete_rejects, sampled_incomplete])
    else:
        sampled_rejects = rng.choice(rejects, n_samples, replace=False)
    all_selected = list(accepts) + list(sampled_rejects)

    if year in {2024, 2025}:
        # we get all samples that are in deep-review train vs test vs not in deep-review atall
        dr_test = [s for s in all_selected if deepreview_lookup.get(s, (None,))[0] == "test"]
        dr_train = [s for s in all_selected if deepreview_lookup.get(s, (None,))[0] == "train"]
        not_in_dr = [s for s in all_selected if s not in deepreview_lookup]

        # target number of test-samples
        target_test = int((1 - TRAIN_RATIO) * len(all_selected))

        # start off with the train/test samples from deep-review
        our_test = dr_test.copy()
        our_train = dr_train.copy()

        # Fill remaining from not_in_dr
        rng.shuffle(not_in_dr)
        remaining_test = max(0, target_test - len(our_test))
        our_test += not_in_dr[:remaining_test]
        our_train += not_in_dr[remaining_test:]
    else:
        # Random 80/20 split
        rng.shuffle(all_selected)
        split_idx = int(TRAIN_RATIO * len(all_selected))
        our_train = all_selected[:split_idx]
        our_test = all_selected[split_idx:]

    return our_train, our_test


def main():
    print("=" * 60)
    print("CREATE TRAIN/TEST SPLIT DATASET")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    # Step 1: Load our dataset
    print("\n1. Loading our HF dataset...")
    our_ds = load_from_disk(str(INPUT_PATH))
    print(f"   Total rows: {len(our_ds):,}")

    # Extract metadata for splitting
    our_df = our_ds.select_columns([
        "submission_id", "year", "technical_indicators",
        "normalized_reviews", "normalized_metareview"
    ]).to_pandas()
    our_df["decision"] = our_df["technical_indicators"].apply(
        lambda x: json.loads(x)["binary_decision"]
    )
    print(f"   Unique submissions: {our_df['submission_id'].nunique():,}")

    # Build set of "complete" submission IDs (for reject sampling preference)
    # Complete means: normalized_reviews, normalized_metareview, and technical_indicators are all non-null
    complete_ids = set(
        our_df[
            our_df["normalized_reviews"].notna()
            & our_df["normalized_metareview"].notna()
            & our_df["technical_indicators"].notna()
        ]["submission_id"].unique()
    )
    print(f"   Complete submissions: {len(complete_ids):,}")

    # Step 2: Load DeepReview-13K
    print("\n2. Loading DeepReview-13K...")
    deepreview_df = load_deepreview_with_row_indices()
    print(f"   Total rows: {len(deepreview_df):,}")
    print(f"   Unique IDs: {deepreview_df['id'].nunique():,}")

    # Build lookup
    print("\n3. Building DeepReview lookup...")
    deepreview_lookup = build_deepreview_lookup(deepreview_df)
    print(f"   Lookup entries: {len(deepreview_lookup):,}")

    # Step 3: Per-year sampling and splitting
    print("\n4. Sampling and splitting per year...")
    all_train_ids = []
    all_test_ids = []

    for year in YEARS:
        year_df = our_df[our_df["year"] == year]
        if len(year_df) == 0:
            print(f"   {year}: No data, skipping")
            continue

        train_ids, test_ids = sample_and_split_year(year_df, year, deepreview_lookup, rng, complete_ids)
        all_train_ids.extend(train_ids)
        all_test_ids.extend(test_ids)

        # Stats
        n_accepts_train = sum(1 for s in train_ids if our_df[our_df["submission_id"] == s]["decision"].iloc[0] == "accept")
        n_accepts_test = sum(1 for s in test_ids if our_df[our_df["submission_id"] == s]["decision"].iloc[0] == "accept")
        print(
            f"   {year}: train={len(train_ids):,} (accept={n_accepts_train}, reject={len(train_ids)-n_accepts_train}), "
            f"test={len(test_ids):,} (accept={n_accepts_test}, reject={len(test_ids)-n_accepts_test})"
        )

    print(f"\n   Total: train={len(all_train_ids):,}, test={len(all_test_ids):,}")

    # Step 4: Create split assignments
    print("\n5. Creating split assignments...")
    split_assignment = {}
    for sid in all_train_ids:
        split_assignment[sid] = "train"
    for sid in all_test_ids:
        split_assignment[sid] = "test"

    # Step 5: Filter and add columns
    print("\n6. Filtering dataset and adding columns...")
    selected_ids = set(all_train_ids + all_test_ids)

    def add_columns(example):
        sid = example["submission_id"]
        dr_info = deepreview_lookup.get(sid)

        example["split"] = split_assignment.get(sid, "")
        # Use empty string instead of None to avoid type casting issues
        example["deepreview_split"] = dr_info[0] if dr_info else ""
        example["deepreview_row_index"] = dr_info[1] if dr_info else -1

        return example

    # Filter to selected submissions and years
    filtered_ds = our_ds.filter(
        lambda x: x["submission_id"] in selected_ids and x["year"] in YEARS
    )
    print(f"   Filtered to {len(filtered_ds):,} rows")

    # Add new columns with explicit feature types to avoid type inference issues
    new_features = filtered_ds.features.copy()
    new_features["split"] = Value("string")
    new_features["deepreview_split"] = Value("string")
    new_features["deepreview_row_index"] = Value("int64")
    filtered_ds = filtered_ds.map(add_columns, features=Features(new_features))

    # Step 6: Split into train/test
    print("\n7. Creating DatasetDict...")
    train_ds = filtered_ds.filter(lambda x: x["split"] == "train")
    test_ds = filtered_ds.filter(lambda x: x["split"] == "test")

    dataset_dict = DatasetDict({"train": train_ds, "test": test_ds})

    print(f"   Train: {len(train_ds):,} rows")
    print(f"   Test: {len(test_ds):,} rows")

    # Step 7: Save
    print(f"\n8. Saving to {OUTPUT_PATH}...")
    dataset_dict.save_to_disk(str(OUTPUT_PATH))
    print("   Done!")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for split_name, ds in dataset_dict.items():
        df = ds.to_pandas()
        print(f"\n{split_name.upper()}:")
        print(f"  Total: {len(df):,}")
        for year in sorted(df["year"].unique()):
            year_df = df[df["year"] == year]
            decisions = year_df["technical_indicators"].apply(
                lambda x: json.loads(x)["binary_decision"]
            ).value_counts()
            print(f"  {year}: {len(year_df):,} (accept={decisions.get('accept', 0)}, reject={decisions.get('reject', 0)})")


if __name__ == "__main__":
    main()
