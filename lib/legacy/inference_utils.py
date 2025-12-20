#!/usr/bin/env python3
"""
Shared utilities for ICLR review prediction inference scripts.

This module contains shared code between:
- predict_iclr_reviews.py (vLLM local model inference)
- gemini_batch_submit.py (Gemini Batch API inference)
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd


# ============================================================================
# SHARED ARGUMENT PARSER
# ============================================================================

def create_base_parser(description: str) -> argparse.ArgumentParser:
    """Create base argument parser with shared arguments."""
    parser = argparse.ArgumentParser(description=description)

    # Required arguments
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base output directory for results')

    # Task configuration
    parser.add_argument('--indicator', type=int, default=7, choices=[1, 2, 3, 5, 6, 7],
                        help='Task indicator (default: 7 for binary accept/reject)')
    parser.add_argument('--version', type=str, default='v1',
                        help='Version suffix for output directory')

    # Model configuration
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name or path')
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature')

    # Prompting options
    parser.add_argument('--use_fewshot', action='store_true',
                        help='Enable few-shot prompting')
    parser.add_argument('--use_paper_reviews', action='store_true',
                        help='Use full paper MD content (incompatible with --use_fewshot)')

    # Sampling options
    parser.add_argument('--pct', type=float, default=1.0,
                        help='Percentage of data to sample per year (0.0-1.0)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Run with 50 samples for testing. Output to dry_run/ subdirectory.')

    # Prediction target
    parser.add_argument('--predict_col', type=str, default='Accepted',
                        help='Column to predict (default: Accepted). Only "Accepted" is supported.')

    return parser


def add_vllm_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add vLLM-specific arguments."""
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for vLLM inference')
    parser.add_argument('--concurrency', type=int, default=1,
                        help='Number of vLLM replicas/concurrency')
    parser.add_argument('--tensor_parallel', type=int, default=1,
                        help='Tensor parallel size (number of GPUs to split model across)')
    parser.add_argument('--ray_address', type=str, default=None,
                        help='Ray cluster address (e.g., ray://localhost:10001)')
    return parser


def add_gemini_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add Gemini batch-specific arguments."""
    parser.add_argument('--poll_interval', type=int, default=300,
                        help='Polling interval in seconds (default: 300 = 5 min)')
    parser.add_argument('--submit_only', action='store_true',
                        help='Submit job but do not wait for results')
    return parser


def validate_args(args) -> None:
    """Validate argument combinations."""
    if args.use_paper_reviews and args.use_fewshot:
        raise ValueError("--use_paper_reviews and --use_fewshot are mutually exclusive.")

    if hasattr(args, 'pct') and not 0.0 < args.pct <= 1.0:
        raise ValueError(f"--pct must be between 0.0 and 1.0 (got {args.pct})")

    if hasattr(args, 'predict_col') and args.predict_col != 'Accepted':
        raise ValueError(f"--predict_col only supports 'Accepted' (got '{args.predict_col}')")


# ============================================================================
# SHARED UTILITIES
# ============================================================================

def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for use in directory path."""
    # Replace slashes and special chars
    sanitized = model_name.replace("/", "-").replace("\\", "-")
    # Keep only alphanumeric, hyphens, underscores, dots
    sanitized = re.sub(r'[^a-zA-Z0-9\-_.]', '-', sanitized)
    # Convert to lowercase
    sanitized = sanitized.lower()
    # Remove consecutive hyphens
    sanitized = re.sub(r'-+', '-', sanitized)
    # Strip leading/trailing hyphens
    sanitized = sanitized.strip('-')
    return sanitized


def create_output_path(output_dir: str, indicator: int, model_name: str, version: str,
                       use_fewshot: bool = False, use_paper_reviews: bool = False,
                       pct: float = 1.0, dry_run: bool = False) -> str:
    """Create output path with auto-generated subdirectory."""
    sanitized_model = sanitize_model_name(model_name)

    # Build variant suffix
    if use_fewshot:
        variant = f"task{indicator}_{version}_fewshot"
    elif use_paper_reviews:
        variant = f"task{indicator}_{version}_paper"
    else:
        variant = f"task{indicator}_{version}"

    # Add _pct suffix if sampling is enabled
    if pct < 1.0:
        pct_str = f"{int(pct * 100)}"
        variant = f"{variant}_pct{pct_str}"

    # Add dry_run subdirectory if enabled
    if dry_run:
        output_path = os.path.join(output_dir, sanitized_model, "dry_run", variant)
    else:
        output_path = os.path.join(output_dir, sanitized_model, variant)

    return output_path


def extract_year_decisions(df: pd.DataFrame) -> Dict[int, List[str]]:
    """Extract unique paper decisions for each year from the dataset."""
    year_decisions = {}

    for year in sorted(df['year'].unique()):
        # Get non-null decisions for this year
        decisions = df[df['year'] == year]['str_paper_decision'].dropna().unique()
        # Convert to list and sort for consistency
        year_decisions[year] = sorted([str(d) for d in decisions if str(d) != 'nan'])

    return year_decisions


def calculate_paper_age(year: int, current_year: int = 2025) -> int:
    """Calculate how many years since publication."""
    # ICLR year corresponds to publication in year-1
    publication_year = year - 1
    return current_year - publication_year


def get_available_horizons(year: int) -> List[int]:
    """Get available citation horizons based on paper age."""
    age = calculate_paper_age(year)
    return list(range(1, min(age + 1, 7)))  # Max 6 horizons


# ============================================================================
# PAPER LOADING AND TRUNCATION
# ============================================================================

def load_and_truncate_paper(md_path: str, max_tokens: int = 18000) -> str:
    """Load MD file and truncate to max tokens (simple character-based truncation).

    Uses empirical ratio: 15k tokens ≈ 53k chars (3.53 chars/token).

    Args:
        md_path: Path to markdown file
        max_tokens: Maximum tokens to keep (default 18k)

    Returns:
        Truncated paper content as string
    """
    with open(md_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Character limit based on empirical ratio: 15k tokens ≈ 53k chars
    max_chars = int(max_tokens * 53e3 / 15e3)

    if len(content) <= max_chars:
        return content

    return content[:max_chars] + "\n\n[... truncated for length ...]"


# ============================================================================
# SHARED DATA LOADING
# ============================================================================

def load_holdout_examples(indicator: int, data_dir: str = None) -> tuple:
    """Load holdout examples for few-shot prompting.

    Args:
        indicator: Task indicator (1, 2, 3, 7, etc.)
        data_dir: Directory containing holdout_examples/ subdirectory.
                  If None, looks in current directory.

    Returns:
        tuple: (holdout_examples dict, holdout_ids set)
    """
    # Build holdout path - look in data_dir/holdout_examples/ if data_dir provided
    if data_dir:
        holdout_path = os.path.join(data_dir, "holdout_examples", f"task{indicator}.json")
    else:
        holdout_path = f"holdout_examples/task{indicator}.json"

    holdout_examples = {}
    holdout_ids = set()

    if os.path.exists(holdout_path):
        print(f"Loading holdout examples from {holdout_path}")
        with open(holdout_path, 'r') as f:
            holdout_examples = json.load(f)

        # Collect all holdout IDs to exclude
        for year, examples in holdout_examples.items():
            for ex in examples:
                holdout_ids.add(ex['id'])

        print(f"  Loaded {sum(len(v) for v in holdout_examples.values())} holdout examples")
        print(f"  Will exclude {len(holdout_ids)} IDs from dataset")
    else:
        print(f"Warning: Holdout file not found at {holdout_path}")

    return holdout_examples, holdout_ids


def load_dataset(args) -> pd.DataFrame:
    """Load and preprocess dataset based on arguments.

    Performs filtering in this order:
    1. Load dataset (with MD paths if use_paper_reviews)
    2. Filter to valid years for task
    3. Exclude holdout examples if use_fewshot
    4. Apply pct sampling per year
    5. Apply dry_run sampling (deterministic, seed=42)

    Returns:
        Preprocessed DataFrame ready for inference
    """
    # Load dataset
    df = pd.read_csv(args.dataset_path)
    print(f"Loaded {len(df)} rows from {args.dataset_path}")

    # Filter for papers with MD files if use_paper_reviews is enabled
    if args.use_paper_reviews:
        if 'md_path' not in df.columns:
            raise ValueError(
                "Dataset does not have 'md_path' column required for --use_paper_reviews.\n"
                "Use a dataset with MD paths (e.g., data/train.csv from the data pipeline)."
            )
        df_before = len(df)
        df = df[df['md_path'].notna()].copy()
        print(f"Filtered to papers with MD files: {df_before} -> {len(df)} rows")

    # Preserve original index - use existing column if present, otherwise use DataFrame index
    if 'original_index' not in df.columns:
        df['original_index'] = df.index.values

    # Filter to years < 2026 for tasks 1, 2, 7 (no accepts in 2026+)
    if args.indicator in [1, 2, 7]:
        df_before = len(df)
        df = df[df['year'] < 2026].copy()
        print(f"Filtered to years < 2026: {df_before} -> {len(df)} rows")

    # Handle few-shot: load holdout examples and exclude from dataset
    # Derive data directory from dataset path (for finding holdout_examples/)
    data_dir = os.path.dirname(args.dataset_path)

    holdout_examples = {}
    if args.use_fewshot:
        holdout_examples, holdout_ids = load_holdout_examples(args.indicator, data_dir=data_dir)

        if holdout_ids:
            df_before = len(df)
            df = df[~df['id'].isin(holdout_ids)].copy()
            print(f"Excluded holdout examples: {df_before} -> {len(df)} rows")
        else:
            print("Warning: No holdout examples found. Proceeding without few-shot.")
            args.use_fewshot = False

    # Sample percentage of data per year if pct < 1.0
    if args.pct < 1.0:
        print(f"\nSampling {args.pct * 100:.1f}% of data per year...")
        sampled_dfs = []
        for year in sorted(df['year'].unique()):
            df_year = df[df['year'] == year]
            n_samples = max(1, int(len(df_year) * args.pct))  # At least 1 sample
            df_year_sampled = df_year.sample(n=n_samples, random_state=42)
            sampled_dfs.append(df_year_sampled)
            print(f"  Year {year}: {len(df_year)} -> {len(df_year_sampled)} papers")
        df = pd.concat(sampled_dfs, ignore_index=True)
        print(f"Total after sampling: {len(df)} papers")

    # Dry run sampling - deterministic AFTER all other filtering
    if args.dry_run:
        n_samples = min(50, len(df))
        df = df.sample(n=n_samples, random_state=42).copy()
        print(f"Dry run: deterministically sampled {len(df)} rows (seed=42)")

    # Reset index and add tracking columns (original_index already preserved above)
    df = df.reset_index(drop=True)
    df['indicator'] = args.indicator
    df['uses_fewshot'] = args.use_fewshot

    # Add holdout examples to each row based on year if few-shot enabled
    df['_holdout_examples'] = None
    if args.use_fewshot and holdout_examples:
        def get_holdout_for_year(year):
            year_str = str(int(year))
            return holdout_examples.get(year_str, [])

        df['_holdout_examples'] = df['year'].apply(get_holdout_for_year)

    # Extract year decisions for task 1
    if args.indicator == 1:
        year_decisions = extract_year_decisions(df)
        df['_year_decisions'] = df['year'].map(year_decisions)

    return df


# ============================================================================
# PROMPT BUILDERS
# ============================================================================

def build_prompt_task7_critical(row: Dict[str, Any], fewshot_examples: List[Dict] = None,
                                 use_paper_reviews: bool = False, max_paper_tokens: int = 18000) -> str:
    """Task 7: Binary paper decision (Accept vs Reject).

    Supports both abstract-based and full paper-based prompts via use_paper_reviews flag.

    Args:
        row: Data row with title, abstract, year, and optionally md_path
        fewshot_examples: Optional few-shot examples (mutually exclusive with use_paper_reviews)
        use_paper_reviews: If True, use full paper MD content instead of abstract
        max_paper_tokens: Max tokens for paper content (default 18k)

    Returns:
        Formatted prompt string
    """
    title = row["title"]
    year = row.get("year", "Unknown")

    # Determine content source
    if use_paper_reviews:
        md_path = row.get("md_path")
        if not md_path or (isinstance(md_path, float) and np.isnan(md_path)):
            # Fallback to abstract if no md_path
            abstract = row.get("abstract", "")
            content_section = f"Abstract: {abstract}"
            content_description = "the paper's title and abstract"
        else:
            paper_content = load_and_truncate_paper(md_path, max_paper_tokens)
            content_section = f"Paper Content:\n{paper_content}"
            content_description = "the full paper content"
    else:
        abstract = row.get("abstract", "")
        content_section = f"Abstract: {abstract}"
        content_description = "the paper's title and abstract"

    # Format few-shot examples if available (only for non-paper mode)
    fewshot_str = ""
    if fewshot_examples and not use_paper_reviews:
        fewshot_str = "\n\nHere are some examples:\n\n"
        for i, ex in enumerate(fewshot_examples, 1):
            # Use Accepted column directly
            accepted = ex.get('Accepted', True)
            decision = 'Accept' if accepted else 'Reject'
            ex_year = ex.get('year', 'Unknown')
            fewshot_str += f"""Example {i}:
Year: ICLR {ex_year}
Title: {ex['title']}

Abstract: {ex['abstract']}

Answer: {decision}

"""

    prompt = f"""You are an expert reviewer for the ICLR conference. Based on {content_description}, predict whether it will be accepted or rejected.{fewshot_str}

Now predict for this paper:

Year: ICLR {year}
Title: {title}

{content_section}

Provide your response in the following format, try to be critical:
Reasoning: [Your reasoning about the paper's quality, novelty, and fit]
Answer: [Either "Accept" or "Reject"]"""

    return prompt


# ============================================================================
# OUTPUT PARSERS
# ============================================================================

def extract_reasoning_and_answer(text: str) -> Dict[str, str]:
    """Extract reasoning and answer from formatted output."""
    reasoning = None
    answer = None

    # Try to extract reasoning and answer
    reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=Answer:|$)', text, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r'Answer:\s*(.+?)$', text, re.DOTALL | re.IGNORECASE)

    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # If no "Answer:" label, use the whole text as answer
        answer = text.strip()

    return {"reasoning": reasoning, "answer": answer}


def parse_output_task7(text: str) -> Dict[str, Any]:
    """Parse binary decision (Accept vs Reject)."""
    # Extract reasoning and answer
    parsed = extract_reasoning_and_answer(text)
    answer_text = parsed["answer"] or ""

    # Try to match Accept or Reject
    decision = None

    # Try exact match first (case insensitive)
    if answer_text.lower() == 'accept':
        decision = 'Accept'
    elif answer_text.lower() == 'reject':
        decision = 'Reject'
    else:
        # Try substring match
        if 'accept' in answer_text.lower():
            decision = 'Accept'
        elif 'reject' in answer_text.lower():
            decision = 'Reject'

    return {"reasoning": parsed["reasoning"], "answer": decision}


def parse_output_task1(text: str, decisions: List[str]) -> Dict[str, Any]:
    """Parse paper decision output."""
    parsed = extract_reasoning_and_answer(text)
    answer_text = parsed["answer"] or ""

    if not decisions:
        return {"reasoning": parsed["reasoning"], "answer": None}

    # Try to match decision from answer
    decision = None

    # Try exact match first
    if answer_text in decisions:
        decision = answer_text
    else:
        # Try case-insensitive match
        for d in decisions:
            if answer_text.lower() == d.lower():
                decision = d
                break

        # Try substring match
        if decision is None:
            for d in decisions:
                if d.lower() in answer_text.lower() or answer_text.lower() in d.lower():
                    decision = d
                    break

    return {"reasoning": parsed["reasoning"], "answer": decision}


def parse_output_task2(text: str, expected_count: int) -> Dict[str, Any]:
    """Parse citation predictions."""
    parsed = extract_reasoning_and_answer(text)
    answer_text = parsed["answer"] or ""

    # Try to extract comma-separated numbers
    numbers = re.findall(r'\d+', answer_text)

    try:
        citations = [int(n) for n in numbers[:expected_count]]
        # Pad with None if not enough values
        while len(citations) < expected_count:
            citations.append(None)
        citations = citations[:expected_count]
    except:
        citations = [None] * expected_count

    return {"reasoning": parsed["reasoning"], "answer": citations}


def parse_output_task3(text: str) -> Dict[str, Any]:
    """Parse review rating [1-10]."""
    parsed = extract_reasoning_and_answer(text)
    answer_text = parsed["answer"] or ""

    # Extract first number
    rating = None
    match = re.search(r'(\d+(?:\.\d+)?)', answer_text)
    if match:
        try:
            rating = float(match.group(1))
            if not (1 <= rating <= 10):
                rating = None
        except:
            pass

    return {"reasoning": parsed["reasoning"], "answer": rating}


def parse_output_task5(text: str) -> Dict[str, Any]:
    """Parse impact percentile [0-1]."""
    parsed = extract_reasoning_and_answer(text)
    answer_text = parsed["answer"] or ""

    # Extract first decimal number
    percentile = None
    match = re.search(r'(\d+(?:\.\d+)?)', answer_text)
    if match:
        try:
            percentile = float(match.group(1))
            # Handle both 0-1 and 0-100 scales
            if percentile > 1.0:
                percentile = percentile / 100.0
            if not (0 <= percentile <= 1):
                percentile = None
        except:
            pass

    return {"reasoning": parsed["reasoning"], "answer": percentile}


def parse_output_task6(text: str) -> Dict[str, Any]:
    """Parse confidence-weighted quality [0-1]."""
    parsed = extract_reasoning_and_answer(text)
    answer_text = parsed["answer"] or ""

    # Extract first decimal number
    quality = None
    match = re.search(r'(\d+(?:\.\d+)?)', answer_text)
    if match:
        try:
            quality = float(match.group(1))
            # Handle both 0-1 and 0-100 scales
            if quality > 1.0:
                quality = quality / 100.0
            if not (0 <= quality <= 1):
                quality = None
        except:
            pass

    return {"reasoning": parsed["reasoning"], "answer": quality}


# Map indicator to parser function
PARSERS = {
    1: parse_output_task1,
    2: parse_output_task2,
    3: parse_output_task3,
    5: parse_output_task5,
    6: parse_output_task6,
    7: parse_output_task7,
}


def get_parser(indicator: int):
    """Get parser function for given task indicator."""
    return PARSERS.get(indicator, parse_output_task7)
