#!/usr/bin/env python3
"""
Create LlamaFactory datasets from HuggingFace dataset.

Converts our paper dataset to LlamaFactory ShareGPT format for SFT training.
Supports text-only, text-and-images, and images-only input modes.
Supports binary, multiclass, and rating output modes.

Usage:
    # Text-only input, binary output (creates both train and test splits)
    python scripts/create_alpaca_dataset_for_llama_factory.py \
        --dataset-path data/iclr_2020_2025_80_20_split \
        --input-mode text-only \
        --output-mode binary \
        --output-name iclr_text_only_binary

    # Text with images, multiclass output
    python scripts/create_alpaca_dataset_for_llama_factory.py \
        --dataset-path data/iclr_2020_2025_80_20_split \
        --input-mode text-and-images \
        --output-mode multiclass \
        --output-name iclr_text_images_multiclass \
        --split train \
        --add-to-llama-factory
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_from_disk
from tqdm import tqdm

# System prompt for all samples
SYSTEM_PROMPT = (
    "You are an ICLR Reviewer, tasked with reviewing this paper. "
    "Be critical, ICLR has an acceptance rate that is less than 50%."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create LlamaFactory datasets from HuggingFace dataset"
    )

    # Required arguments
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to HuggingFace dataset",
    )
    parser.add_argument(
        "--input-mode",
        choices=["text-only", "text-and-images", "images-only"],
        required=True,
        help="Input modality mode",
    )
    parser.add_argument(
        "--output-mode",
        choices=["binary", "multiclass", "rating"],
        required=True,
        help="Output prediction mode (binary=Accept/Reject, multiclass=Oral/Spotlight/Poster/Reject, rating=1-10)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        required=True,
        help="Base name for output files (e.g., 'iclr_text_only_binary')",
    )

    # Optional arguments
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to process (default: both train and test)",
    )
    parser.add_argument(
        "--llama-factory-path",
        type=Path,
        default=Path("/n/fs/vision-mix/sk7524/LLaMA-Factory"),
        help="Path to LlamaFactory installation",
    )
    parser.add_argument(
        "--add-to-llama-factory",
        action="store_true",
        default=False,
        help="Add dataset to LlamaFactory's dataset_info.json",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of worker threads (default: 8)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per split (for testing)",
    )
    parser.add_argument(
        "--max-content-chars",
        type=int,
        default=50000,
        help="Maximum characters for content (default: 50000)",
    )

    return parser.parse_args()


def get_instruction(output_mode: str) -> str:
    """Get instruction text based on output mode."""
    if output_mode == "binary":
        return (
            "Based on the paper information provided, predict whether this paper "
            "will be accepted or rejected at ICLR.\n\n"
            "Provide your response in the following format:\n"
            "Reasoning: [Your reasoning about the paper's quality, novelty, "
            "significance, and fit for the conference]\n"
            "Answer: [Either 'Accept' or 'Reject']"
        )
    elif output_mode == "multiclass":
        return (
            "Based on the paper information provided, predict the decision for this paper "
            "at ICLR.\n\n"
            "Provide your response in the following format:\n"
            "Reasoning: [Your reasoning about the paper's quality]\n"
            "Answer: [One of: 'Oral', 'Spotlight', 'Poster', 'Reject']"
        )
    else:  # rating
        return (
            "Based on the paper information provided, predict the average reviewer rating "
            "for this paper at ICLR.\n\n"
            "Provide your response in the following format:\n"
            "Reasoning: [Your reasoning about the paper's quality]\n"
            "Answer: [A number between 1 and 10]"
        )


def get_label(row: Dict[str, Any], output_mode: str) -> str:
    """Extract label from row based on output mode."""
    decision = None

    # Try submission_json first
    if row.get("submission_json"):
        try:
            data = json.loads(row["submission_json"])
            decision = data.get("decision", "")
        except (json.JSONDecodeError, TypeError):
            pass

    if output_mode == "binary":
        if not decision:
            return "Reject"
        decision_lower = decision.lower()
        if any(x in decision_lower for x in ["accept", "oral", "spotlight", "poster"]):
            return "Accept"
        return "Reject"

    elif output_mode == "multiclass":
        if not decision:
            return "Reject"
        decision_lower = decision.lower()
        if "oral" in decision_lower:
            return "Oral"
        elif "spotlight" in decision_lower:
            return "Spotlight"
        elif "poster" in decision_lower or "accept" in decision_lower:
            return "Poster"
        return "Reject"

    else:  # rating
        # Try to get mean rating from reviews
        mean_rating = None
        if row.get("normalized_reviews"):
            try:
                reviews = json.loads(row["normalized_reviews"])
                if isinstance(reviews, list):
                    ratings = []
                    for r in reviews:
                        if isinstance(r, str):
                            r = json.loads(r)
                        if isinstance(r, dict) and r.get("rating"):
                            ratings.append(float(r["rating"]))
                    if ratings:
                        mean_rating = sum(ratings) / len(ratings)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        if mean_rating is not None:
            return f"{mean_rating:.1f}"

        # Fallback: estimate from decision
        if not decision:
            return "5.0"
        decision_lower = decision.lower()
        if "oral" in decision_lower:
            return "8.0"
        elif "spotlight" in decision_lower:
            return "7.0"
        elif "poster" in decision_lower or "accept" in decision_lower:
            return "6.0"
        return "4.0"


def extract_page_num(path: str) -> int:
    """Extract page number from path like page_1.png."""
    match = re.search(r'page_(\d+)', str(path))
    return int(match.group(1)) if match else 0


def build_text_only_content(row: Dict[str, Any], max_chars: int) -> str:
    """Build text-only content (title + abstract + markdown)."""
    title = row.get("title", "")
    abstract = row.get("no_github_abstract") or row.get("original_abstract", "")

    parts = [f"# {title}", f"\n## Abstract\n{abstract}"]

    # Add markdown if available
    clean_md_path = row.get("clean_md_path")
    if clean_md_path:
        try:
            md_content = Path(clean_md_path).read_text(encoding="utf-8")
            # Calculate remaining budget
            current_len = sum(len(p) for p in parts)
            remaining = max(0, max_chars - current_len)
            if remaining > 1000:
                parts.append(f"\n## Paper Content\n{md_content[:remaining]}")
        except Exception:
            pass

    return "\n".join(parts)


def build_text_with_images_content(
    row: Dict[str, Any], max_chars: int
) -> Tuple[str, List[str]]:
    """Build content with inline images from markdown."""
    title = row.get("title", "")
    abstract = row.get("no_github_abstract") or row.get("original_abstract", "")

    parts = [f"# {title}", f"\n## Abstract\n{abstract}"]
    images: List[str] = []

    # Parse images_in_clean_md for the mapping
    images_map = {}
    if row.get("images_in_clean_md"):
        try:
            data = row["images_in_clean_md"]
            if isinstance(data, str):
                data = json.loads(data)
            if isinstance(data, dict):
                images_map = data
        except (json.JSONDecodeError, TypeError):
            pass

    # Read markdown and replace ![](images/X) with <image> tokens
    clean_md_path = row.get("clean_md_path")
    if clean_md_path and images_map:
        try:
            md_content = Path(clean_md_path).read_text(encoding="utf-8")

            # Pattern to match markdown images
            pattern = r'!\[[^\]]*\]\(images/([^)]+)\)'

            def replace_with_token(match):
                filename = match.group(1)
                if filename in images_map:
                    img_path = images_map[filename]
                    if Path(img_path).exists():
                        images.append(img_path)
                        return "<image>"
                return ""  # Remove unresolved image refs

            md_with_tokens = re.sub(pattern, replace_with_token, md_content)

            # Calculate remaining budget
            current_len = sum(len(p) for p in parts)
            remaining = max(0, max_chars - current_len)
            if remaining > 1000:
                parts.append(f"\n## Paper Content\n{md_with_tokens[:remaining]}")
        except Exception:
            pass

    return "\n".join(parts), images


def build_images_only_content(
    row: Dict[str, Any], max_images: int = 20
) -> Tuple[str, List[str]]:
    """Build content with PDF page images."""
    title = row.get("title", "")
    abstract = row.get("no_github_abstract") or row.get("original_abstract", "")

    # Get image paths
    img_paths = []
    if row.get("clean_pdf_img_paths"):
        try:
            data = row["clean_pdf_img_paths"]
            if isinstance(data, str):
                data = json.loads(data)
            if isinstance(data, list):
                img_paths = [p for p in data if p and Path(p).exists()]
        except (json.JSONDecodeError, TypeError):
            pass

    # Sort numerically and limit
    img_paths = sorted(img_paths, key=extract_page_num)[:max_images]

    # Build content with image tokens
    if img_paths:
        image_tokens = " ".join(["<image>"] * len(img_paths))
        content = f"# {title}\n\n## Abstract\n{abstract}\n\n## Paper Pages\n{image_tokens}"
    else:
        content = f"# {title}\n\n## Abstract\n{abstract}"

    return content, img_paths


def process_row(
    row: Dict[str, Any],
    input_mode: str,
    output_mode: str,
    instruction: str,
    max_chars: int,
) -> Optional[Dict[str, Any]]:
    """Convert a single row to LlamaFactory format."""
    try:
        # Build content and images based on input mode
        if input_mode == "text-only":
            content = build_text_only_content(row, max_chars)
            images = []
        elif input_mode == "text-and-images":
            content, images = build_text_with_images_content(row, max_chars)
        else:  # images-only
            content, images = build_images_only_content(row)

        # Build the user message
        user_content = f"{instruction}\n\n{content}"

        # Build the assistant response from ground truth
        label = get_label(row, output_mode)
        assistant_content = f"Answer: {label}"

        # Build messages with system prompt
        result = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        }

        if images:
            result["images"] = images

        return result

    except Exception as e:
        print(f"Error processing row {row.get('submission_id', 'unknown')}: {e}")
        return None


def process_split(
    ds,
    split_name: str,
    args: argparse.Namespace,
    instruction: str,
) -> Path:
    """Process a single split and save to JSON."""
    data = ds[split_name]

    if args.max_samples:
        data = data.select(range(min(args.max_samples, len(data))))

    print(f"\nProcessing {split_name} split: {len(data):,} samples")

    # Process with ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_row,
                row,
                args.input_mode,
                args.output_mode,
                instruction,
                args.max_content_chars,
            ): i
            for i, row in enumerate(data)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=split_name):
            result = future.result()
            if result:
                results.append(result)

    # Save to JSON
    output_dir = Path("data/llama_factory_jsons")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.output_name}_{split_name}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results):,} samples to {output_file}")
    return output_file


def add_to_dataset_info(
    args: argparse.Namespace,
    output_file: Path,
    dataset_name: str,
) -> None:
    """Add dataset entry to LlamaFactory dataset_info.json."""
    info_path = args.llama_factory_path / "data" / "dataset_info.json"

    if not info_path.exists():
        print(f"Warning: {info_path} not found, skipping registration")
        return

    with open(info_path) as f:
        info = json.load(f)

    # Build entry
    entry = {
        "file_name": str(output_file.absolute()),
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }

    # Add images column if needed
    if args.input_mode in ["text-and-images", "images-only"]:
        entry["columns"]["images"] = "images"

    info[dataset_name] = entry

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"Added '{dataset_name}' to {info_path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("CREATE LLAMAFACTORY DATASET")
    print("=" * 60)
    print(f"Dataset: {args.dataset_path}")
    print(f"Input mode: {args.input_mode}")
    print(f"Output mode: {args.output_mode}")
    print(f"Output name: {args.output_name}")
    print(f"Split: {args.split or 'train + test'}")
    print(f"Add to LlamaFactory: {args.add_to_llama_factory}")

    # Load dataset
    print(f"\nLoading dataset from {args.dataset_path}...")
    ds = load_from_disk(str(args.dataset_path))
    print(f"Available splits: {list(ds.keys())}")

    # Get instruction based on output mode
    instruction = get_instruction(args.output_mode)

    # Determine which splits to process
    if args.split:
        splits = [args.split]
    else:
        splits = ["train", "test"]

    # Process each split
    output_files = {}
    for split_name in splits:
        if split_name not in ds:
            print(f"Warning: split '{split_name}' not found, skipping")
            continue
        output_files[split_name] = process_split(ds, split_name, args, instruction)

    # Optionally add to LlamaFactory dataset_info.json
    if args.add_to_llama_factory:
        print("\nRegistering with LlamaFactory...")
        for split_name, output_file in output_files.items():
            dataset_name = f"{args.output_name}_{split_name}"
            add_to_dataset_info(args, output_file, dataset_name)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
