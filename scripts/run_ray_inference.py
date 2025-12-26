#!/usr/bin/env python3
"""
Run inference using Ray Data + vLLM.

Streams from HuggingFace dataset, runs inference with vLLM, writes to parquet.
No data is fully materialized in memory.

Usage:
    # Run with Ray cluster
    python scripts/run_ray_inference.py \
        --dataset-path data/iclr_2020_2025_80_20_split \
        --output-path outputs/ray_inference.parquet \
        --ray-address auto

    # Dry run (10 samples, returns pandas DataFrame)
    python scripts/run_ray_inference.py --dry-run

    # Different output types
    python scripts/run_ray_inference.py --output-type multiclass
    python scripts/run_ray_inference.py --output-type rating
"""

import argparse
from pathlib import Path

from lib.llamafactory import (
    InputModality,
    OutputType,
    RayDataPredictor,
)


def get_input_modality(name: str) -> InputModality:
    """Map string to InputModality enum."""
    mapping = {
        "text_only": InputModality.TEXT_ONLY,
        "text_with_images": InputModality.TEXT_WITH_IMAGES,
        "images_only": InputModality.IMAGES_ONLY,
    }
    return mapping.get(name.lower(), InputModality.TEXT_ONLY)


def get_output_type(name: str) -> OutputType:
    """Map string to OutputType enum."""
    mapping = {
        "binary": OutputType.BINARY,
        "multiclass": OutputType.MULTICLASS,
        "rating": OutputType.MEAN_RATING,
    }
    return mapping.get(name.lower(), OutputType.BINARY)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference using Ray Data + vLLM"
    )

    # Data paths
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/iclr_2020_2025_80_20_split"),
        help="Path to HuggingFace dataset",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/ray_inference.parquet"),
        help="Path to output parquet file/directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--input-modality",
        type=str,
        default="text_only",
        choices=["text_only", "text_with_images", "images_only"],
        help="Input modality",
    )
    parser.add_argument(
        "--output-type",
        type=str,
        default="binary",
        choices=["binary", "multiclass", "rating"],
        help="Prediction output type",
    )

    # Ray configuration
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray cluster address (None for local)",
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Number of GPUs for model parallelism",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of vLLM replicas",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size",
    )

    # vLLM configuration
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum context length",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate",
    )

    # Input configuration
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=16000,
        help="Maximum tokens for input content",
    )
    parser.add_argument(
        "--include-reviews",
        action="store_true",
        default=False,
        help="Include reviewer feedback",
    )
    parser.add_argument(
        "--include-markdown",
        action="store_true",
        default=True,
        help="Include paper markdown content",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Maximum images (PDF pages for images_only, inline for text_with_images)",
    )

    # Testing
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run on 10 samples only (returns pandas DataFrame)",
    )
    parser.add_argument(
        "--visualize-input",
        action="store_true",
        help="Print 1 sample input (no inference)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RAY DATA + vLLM INFERENCE")
    print("=" * 60)
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_path}")
    print(f"Model: {args.model}")
    print(f"Input modality: {args.input_modality}")
    print(f"Output type: {args.output_type}")
    print(f"Ray address: {args.ray_address or 'local'}")

    # Create predictor
    predictor = RayDataPredictor(
        input_modality=get_input_modality(args.input_modality),
        output_type=get_output_type(args.output_type),
        model_name=args.model,
        ray_address=args.ray_address,
        tensor_parallel=args.tensor_parallel,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        max_input_tokens=args.max_input_tokens,
        include_reviews=args.include_reviews,
        include_markdown=args.include_markdown,
        max_images=args.max_images,
    )

    if args.visualize_input:
        print("\n*** VISUALIZE INPUT: 1 sample ***")
        from datasets import load_from_disk

        # Load 1 sample
        hf_ds = load_from_disk(str(args.dataset_path))
        row = hf_ds[args.split][0]

        # Build the preprocessed input (without Ray)
        submission = predictor._row_to_submission(row)
        content = predictor.input_formatter.format_content(submission)
        instruction = predictor.output_handler.get_instruction()
        prompt = f"{instruction}\n\n{content}"

        # Get images
        images = predictor.input_formatter.get_images(submission)

        # Get inline image mapping ONLY for TEXT_WITH_IMAGES modality
        images_in_md = None
        if predictor.input_modality == InputModality.TEXT_WITH_IMAGES:
            images_in_md = predictor._parse_images_in_md(row)

        # Build messages
        messages = predictor._build_messages(prompt, images, images_in_md)

        print("\n" + "=" * 60)
        print("SUBMISSION INFO")
        print("=" * 60)
        print(f"ID: {submission.submission_id}")
        print(f"Year: {submission.year}")
        print(f"Title: {submission.title}")
        print(f"Decision: {submission.decision}")

        print("\n" + "=" * 60)
        print("INPUT FORMATTER")
        print("=" * 60)
        print(f"Type: {type(predictor.input_formatter).__name__}")
        print(f"Modality: {predictor.input_modality}")

        print("\n" + "=" * 60)
        print("MESSAGES STRUCTURE")
        print("=" * 60)
        for i, msg in enumerate(messages):
            print(f"\n[Message {i}] role={msg['role']}")
            content = msg["content"]
            if isinstance(content, str):
                print(f"  Type: text ({len(content):,} chars)")
                print(f"  Preview: {content[:500]}...")
            elif isinstance(content, list):
                print(f"  Type: multimodal ({len(content)} parts)")
                for j, part in enumerate(content):
                    if part.get("type") == "text":
                        text = part.get("text", "")
                        print(f"    [{j}] text: {len(text):,} chars")
                        if j == 0:
                            print(f"        Preview: {text[:300]}...")
                    elif part.get("type") == "image":
                        print(f"    [{j}] image: {part.get('image', '')}")

        print("\n" + "=" * 60)
        print("FULL PROMPT")
        print("=" * 60)
        print(prompt)
        return

    if args.dry_run:
        print("\n*** DRY RUN: 10 samples ***")
        df = predictor.predict_to_pandas(
            dataset_path=str(args.dataset_path),
            split=args.split,
            max_samples=10,
        )
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(df[["submission_id", "ground_truth", "prediction", "parse_success"]])

        # Compute quick metrics
        success_rate = df["parse_success"].mean()
        accuracy = (df["prediction"] == df["ground_truth"]).mean()
        print(f"\nParse success rate: {success_rate:.2%}")
        print(f"Accuracy: {accuracy:.2%}")
    else:
        # Ensure output directory exists
        args.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Run streaming inference
        predictor.predict_from_hf_dataset(
            dataset_path=str(args.dataset_path),
            output_path=str(args.output_path),
            split=args.split,
        )

        print("\n" + "=" * 60)
        print(f"Results saved to: {args.output_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
