#!/usr/bin/env python3
"""
Run zero-shot inference on ICLR test split.

Usage:
    # Dry run (10 samples)
    python scripts/run_inference.py --dry-run

    # Binary prediction on full test set
    python scripts/run_inference.py --output-type binary

    # Multiclass with custom output path
    python scripts/run_inference.py --output-type multiclass --output-path outputs/multiclass.json
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from datasets import load_from_disk
from tqdm import tqdm

# Import from lib.llamafactory
from lib.llamafactory import (
    InputModality,
    ModelType,
    OutputType,
    PredictorConfig,
    TrainingMode,
    UnifiedPredictor,
)
from lib.llamafactory.core.types import ReviewData, SubmissionData


def load_markdown_content(path: str) -> str:
    """Load markdown from file path."""
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""


def parse_decision(submission_json: str) -> Optional[str]:
    """Extract decision from submission JSON."""
    if not submission_json:
        return None
    try:
        data = json.loads(submission_json)
        return data.get("decision")
    except (json.JSONDecodeError, TypeError):
        return None


def row_to_submission(row) -> SubmissionData:
    """Convert HF dataset row to SubmissionData."""
    # Load markdown content from path
    clean_md = load_markdown_content(row.get("clean_md_path", ""))

    # Get decision from submission_json
    decision = parse_decision(row.get("submission_json"))

    # Parse normalized reviews if available
    reviews = []
    if row.get("normalized_reviews"):
        try:
            reviews_data = json.loads(row["normalized_reviews"])
            if isinstance(reviews_data, list):
                for r in reviews_data:
                    reviews.append(
                        ReviewData.from_normalized(r) if isinstance(r, dict) else ReviewData()
                    )
        except (json.JSONDecodeError, TypeError):
            pass

    return SubmissionData(
        submission_id=str(row.get("submission_id", "")),
        year=int(row.get("year", 0)),
        title=str(row.get("title", "")),
        abstract=str(row.get("no_github_abstract") or row.get("original_abstract", "")),
        decision=decision,
        clean_md=clean_md,
        reviews=reviews,
    )


def get_output_type(output_type_str: str) -> OutputType:
    """Map string to OutputType enum."""
    mapping = {
        "binary": OutputType.BINARY,
        "multiclass": OutputType.MULTICLASS,
        "rating": OutputType.MEAN_RATING,
    }
    return mapping.get(output_type_str.lower(), OutputType.BINARY)


def main():
    parser = argparse.ArgumentParser(description="Run inference on ICLR test split")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/iclr_2020_2025_80_20_split"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/inference_results.json"),
    )
    parser.add_argument(
        "--output-type",
        type=str,
        default="binary",
        choices=["binary", "multiclass", "rating"],
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run on 10 samples only",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--num-workers", type=int, default=16, help="Threads for loading markdown")
    args = parser.parse_args()

    # Dry run overrides max-samples
    if args.dry_run:
        args.max_samples = 10
        print("DRY RUN: Running on 10 samples only")

    print("=" * 60)
    print("ICLR INFERENCE")
    print("=" * 60)
    print(f"Dataset: {args.dataset_path}")
    print(f"Model: {args.model}")
    print(f"Output type: {args.output_type}")
    print(f"Max samples: {args.max_samples or 'all'}")

    # Load dataset
    print("\n1. Loading dataset...")
    ds_dict = load_from_disk(str(args.dataset_path))
    test_ds = ds_dict["test"]
    print(f"   Test split: {len(test_ds):,} rows")

    # Limit samples if specified
    if args.max_samples:
        test_ds = test_ds.select(range(min(args.max_samples, len(test_ds))))
        print(f"   Limited to: {len(test_ds):,} rows")

    # Convert to SubmissionData (parallel loading for I/O bound markdown reads)
    print(f"\n2. Converting to SubmissionData (loading markdown with {args.num_workers} workers)...")
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(row_to_submission, row): i for i, row in enumerate(test_ds)}
        results = [None] * len(futures)
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading"):
            idx = futures[future]
            results[idx] = future.result()
        submissions = results

    # Count valid markdown
    has_md = sum(1 for s in submissions if s.clean_md)
    print(f"   {has_md:,}/{len(submissions):,} have markdown content")

    # Create predictor
    print("\n3. Creating predictor...")
    predictor = UnifiedPredictor.create(
        input_modality=InputModality.TEXT_ONLY,
        output_type=get_output_type(args.output_type),
        training_mode=TrainingMode.ZERO_SHOT,
        config=PredictorConfig(
            model_name_or_path=args.model,
            model_type=ModelType.TEXT_ONLY,
            batch_size=args.batch_size,
        ),
        input_kwargs={
            "max_tokens": args.max_tokens,
            "include_markdown": True,
            "include_reviews": False,  # Focus on paper content
        },
    )

    # Run inference
    print("\n4. Running inference...")
    results = predictor.predict_with_evaluation(submissions)

    # Save results
    print("\n5. Saving results...")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config": {
            "model": args.model,
            "output_type": args.output_type,
            "max_tokens": args.max_tokens,
            "num_samples": len(submissions),
        },
        "metrics": results["metrics"],
        "predictions": [
            {
                "submission_id": s.submission_id,
                "year": s.year,
                "title": s.title,
                "ground_truth": s.decision,
                **pred,
            }
            for s, pred in zip(submissions, results["predictions"])
        ],
    }

    with open(args.output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for metric, value in results["metrics"].items():
        print(f"  {metric}: {value:.4f}")
    print(f"\nSaved to: {args.output_path}")


if __name__ == "__main__":
    main()
