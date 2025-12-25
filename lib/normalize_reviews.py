#!/usr/bin/env python3
"""
Review normalization pipeline using vLLM with structured outputs.

Normalizes ICLR reviews (2020-2026) from year-specific formats to the
LLMUniversalReview schema using vLLM with guided JSON outputs.

Usage:
    python lib/normalize_reviews.py --data-dir data/full_run --output-dir data/full_run/normalized_reviews \
        --model-name meta-llama/Llama-3.1-8B-Instruct

    # Dry run with 50 samples
    python lib/normalize_reviews.py --data-dir data/full_run --model-name meta-llama/Llama-3.1-8B-Instruct --dry-run
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import pandas as pd
from pydantic import BaseModel

# Add parent to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.schemas import (
    LLMUniversalReview,
    LLMUniversalMetaReview,
    extract_reviews_from_submission,
    extract_meta_review_from_submission,
    get_review_schema_str,
    get_meta_schema_str,
    get_review_example,
    get_meta_example,
)
from lib.utils import build_normalized_index, load_raw_notes, EXCLUDED_DECISIONS, DEFAULT_YEARS
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
from lib.vllm_utils.transform import VLLMTransformer


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

REVIEW_PROMPT_TEMPLATE = """Extract and normalize this ICLR review into a structured JSON format. This is a structuring task
do not hallucinate details unpresent in the original. Leave fields unfilled if no details are present. Fill fields
if there is evidence in the original.

Original Review:
{review_json}

Output a JSON object with these fields:
{schema_fields}

Example output format:
{example_json}

JSON:"""


META_REVIEW_PROMPT_TEMPLATE = """Extract and normalize this ICLR review into a structured JSON format. This is a structuring task
do not hallucinate details unpresent in the original. Leave fields unfilled if no details are present. Fill fields
if there is evidence in the original.

Original Meta-Review:
{meta_json}

Output a JSON object with these fields:
{schema_fields}

Example output format:
{example_json}

JSON:"""


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_raw_notes(data_dir: Path, years: List[int]) -> Dict[str, Any]:
    """Load raw OpenReview notes from pickle files for multiple years.

    Returns:
        Dict mapping submission_id -> (note, year)
    """
    notes_by_id = {}

    for year in years:
        pkl_path = data_dir / f"get_all_notes_{year}.pickle"
        if not pkl_path.exists():
            print(f"  {year}: pickle not found, skipping")
            continue

        with open(pkl_path, 'rb') as f:
            notes = pickle.load(f)

        for note in notes:
            notes_by_id[note.id] = (note, year)

        print(f"  {year}: {len(notes)} notes loaded")

    return notes_by_id


def get_valid_submission_ids(normalized_dir: Path) -> set:
    """Get submission IDs that pass validation (have images).

    Uses build_normalized_index to check which papers have images.
    """
    index = build_normalized_index(normalized_dir)
    valid_ids = set()

    for sub_id, info in index.items():
        # Check if has images (image_count > 0)
        if info.get('image_count', 0) > 0:
            valid_ids.add(sub_id)

    return valid_ids


def get_md_path(normalized_dir: Path, year: int, submission_id: str) -> Optional[str]:
    """Get the markdown file path for a submission."""
    md_path = normalized_dir / str(year) / submission_id / f"{submission_id}.md"
    if md_path.exists():
        return str(md_path)
    return None


# =============================================================================
# PROMPT BUILDING
# =============================================================================

def build_review_prompts(
    submission_id: str,
    year: int,
    note: Any,
    md_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Build prompts for all reviews + meta from a submission.

    Args:
        submission_id: Paper ID
        year: Conference year
        note: Raw OpenReview Note object
        md_path: Path to markdown file (optional, included in output)

    Returns:
        List of prompt dicts with keys:
            - submission_id, year, md_path
            - type: 'review' or 'meta'
            - review_index: 0, 1, 2... for reviews, -1 for meta
            - original_json: Original review/meta as JSON string
            - prompt: The prompt text
    """
    prompts = []

    # Get cached schema strings
    review_schema_str = get_review_schema_str()
    review_example = get_review_example()
    meta_schema_str = get_meta_schema_str()
    meta_example = get_meta_example()

    # Extract reviews using schema parser
    try:
        reviews = extract_reviews_from_submission(year, note)
    except Exception as e:
        reviews = []

    # Extract meta-review
    try:
        meta = extract_meta_review_from_submission(year, note)
    except Exception:
        meta = None

    # Build prompts for reviews
    for idx, review in enumerate(reviews):
        review_json = review.model_dump_json(indent=2)
        prompt = REVIEW_PROMPT_TEMPLATE.format(
            review_json=review_json,
            schema_fields=review_schema_str,
            example_json=review_example
        )

        prompts.append({
            'submission_id': submission_id,
            'year': year,
            'md_path': md_path,
            'type': 'review',
            'review_index': idx,
            'original_json': review_json,
            'prompt': prompt,
        })

    # Build prompt for meta-review
    if meta is not None:
        meta_json = meta.model_dump_json(indent=2)
        prompt = META_REVIEW_PROMPT_TEMPLATE.format(
            meta_json=meta_json,
            schema_fields=meta_schema_str,
            example_json=meta_example
        )

        prompts.append({
            'submission_id': submission_id,
            'year': year,
            'md_path': md_path,
            'type': 'meta',
            'review_index': -1,
            'original_json': meta_json,
            'prompt': prompt,
        })

    return prompts


# =============================================================================
# REVIEW NORMALIZER
# =============================================================================

def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON from text by finding first { and last }.

    Args:
        text: Raw text that may contain JSON

    Returns:
        Extracted JSON string, or None if not found
    """
    first_brace = text.find('{')
    last_brace = text.rfind('}')

    if first_brace == -1 or last_brace == -1 or first_brace >= last_brace:
        return None

    return text[first_brace:last_brace + 1]


def extract_first_valid(
    completions: List[str],
    schema: type[BaseModel]
) -> Optional[BaseModel]:
    """Try each completion until one parses successfully.

    Args:
        completions: List of generated texts
        schema: Pydantic model class to validate against

    Returns:
        First successfully parsed model, or None
    """
    for text in completions:
        # Extract JSON from text
        json_str = extract_json_from_text(text)
        if json_str is None:
            continue

        try:
            # First try direct parsing
            return schema.model_validate_json(json_str)
        except Exception:
            pass

        # If direct parsing fails, check if LLM output schema-wrapped format
        # e.g., {"description": "...", "properties": {"summary": "...", ...}}
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and "properties" in data:
                # Unwrap the properties and try again
                return schema.model_validate(data["properties"])
        except Exception:
            continue
    return None


def generate_dummy_reviews(n_samples: int = 10) -> List[Dict[str, Any]]:
    """Generate dummy review and meta-review prompts for testing.

    Creates realistic-looking reviews with varied content for pipeline testing.

    Args:
        n_samples: Number of samples to generate (mix of reviews and meta)

    Returns:
        List of prompt dicts ready for the normalizer
    """
    # Sample review content variations
    review_templates = [
        {
            "rating": "6: marginally above the acceptance threshold",
            "confidence": "4: You are confident in your assessment",
            "summary": "This paper proposes a novel approach to neural network pruning using gradient-based importance scoring.",
            "strengths": "1. Novel pruning criterion based on gradient flow analysis\n2. Comprehensive experiments on ImageNet and CIFAR\n3. Clear theoretical motivation",
            "weaknesses": "1. Limited comparison with recent pruning methods\n2. Missing ablation on hyperparameter sensitivity\n3. No analysis on different architectures",
            "questions": "1. How does the method scale to larger models like ViT?\n2. Can you provide runtime comparisons?",
        },
        {
            "rating": "5: marginally below the acceptance threshold",
            "confidence": "3: You are fairly confident in your assessment",
            "summary": "The authors present a contrastive learning framework for time series representation.",
            "strengths": "1. Interesting use of temporal augmentations\n2. Good performance on downstream tasks\n3. Well-written paper",
            "weaknesses": "1. The novelty is somewhat limited - similar ideas exist in prior work\n2. Missing comparison with SimCLR adapted for time series\n3. Evaluation limited to classification tasks",
            "questions": "1. How does this compare to TS2Vec?\n2. What about forecasting tasks?",
        },
        {
            "rating": "8: accept, good paper",
            "confidence": "5: You are absolutely certain about your assessment",
            "summary": "A significant contribution to efficient transformers through sparse attention patterns.",
            "strengths": "1. Strong theoretical analysis of attention sparsity\n2. State-of-the-art results on long-range arena\n3. Practical implementation with good speedups\n4. Excellent ablation studies",
            "weaknesses": "1. Memory savings could be better characterized\n2. Limited analysis on decoder-only models",
            "questions": "1. Have you tested on language modeling tasks?",
        },
        {
            "rating": "3: reject, not good enough",
            "confidence": "4: You are confident in your assessment",
            "summary": "This paper attempts to improve GAN training stability through a new regularization term.",
            "strengths": "1. Clear presentation of the method\n2. Some improvement in FID scores",
            "weaknesses": "1. The theoretical justification is weak\n2. Experiments only on small datasets (CIFAR-10)\n3. Missing comparison with recent methods like StyleGAN3\n4. The regularization seems to slow down training significantly",
            "questions": "1. Can you provide results on larger datasets?\n2. What is the computational overhead?",
        },
        {
            "rating": "7: good paper, accept",
            "confidence": "4: You are confident in your assessment",
            "summary": "Novel self-supervised learning approach for medical imaging that leverages anatomical priors.",
            "strengths": "1. Clever use of anatomical structure for pretext tasks\n2. Strong results on multiple medical imaging benchmarks\n3. Good analysis of learned representations\n4. Practical value for low-data medical settings",
            "weaknesses": "1. Limited to specific imaging modalities (CT/MRI)\n2. Requires anatomical annotations which may not always be available",
            "questions": "1. How transferable are the representations across modalities?\n2. Can this work with X-ray images?",
        },
    ]

    meta_templates = [
        {
            "metareview": "The paper presents interesting ideas on neural network pruning. Reviewers appreciated the novel gradient-based criterion but raised concerns about limited comparisons with recent work. After discussion, the AC recommends acceptance given the strong rebuttal addressing computational costs.",
            "justification_for_why_not_higher_score": "The experimental scope is limited to vision models, and some reviewers felt the theoretical analysis could be deeper.",
            "justification_for_why_not_lower_score": "The core contribution is solid, experiments are comprehensive within scope, and the method shows practical value.",
        },
        {
            "metareview": "This submission on contrastive learning for time series received mixed reviews. While the approach is sound, reviewers noted significant overlap with existing methods. The authors' rebuttal clarified some distinctions but did not fully address novelty concerns.",
            "justification_for_why_not_higher_score": "Limited novelty compared to existing contrastive learning approaches adapted for time series.",
            "justification_for_why_not_lower_score": "The empirical results are solid and the paper is well-written.",
        },
        {
            "metareview": "Strong paper on efficient transformers with unanimous positive reviews. The theoretical analysis is rigorous and the practical improvements are significant. Minor concerns about decoder-only models were addressed in the rebuttal.",
            "justification_for_why_not_higher_score": "Some reviewers wanted to see results on larger language models.",
            "justification_for_why_not_lower_score": "The contribution is substantial with strong theory and empirical validation.",
        },
    ]

    # Get cached schema strings
    review_schema_str = get_review_schema_str()
    review_example = get_review_example()
    meta_schema_str = get_meta_schema_str()
    meta_example = get_meta_example()

    prompts = []
    for i in range(n_samples):
        year = 2024
        submission_id = f"test_submission_{i:03d}"

        # Alternate between reviews (70%) and meta-reviews (30%)
        if i % 10 < 7:
            # Generate review
            review_data = review_templates[i % len(review_templates)]
            review_json = json.dumps(review_data, indent=2)
            prompt = REVIEW_PROMPT_TEMPLATE.format(
                review_json=review_json,
                schema_fields=review_schema_str,
                example_json=review_example
            )

            prompts.append({
                'submission_id': submission_id,
                'year': year,
                'md_path': None,
                'type': 'review',
                'review_index': i % 4,
                'original_json': review_json,
                'prompt': prompt,
            })
        else:
            # Generate meta-review
            meta_data = meta_templates[i % len(meta_templates)]
            meta_json = json.dumps(meta_data, indent=2)
            prompt = META_REVIEW_PROMPT_TEMPLATE.format(
                meta_json=meta_json,
                schema_fields=meta_schema_str,
                example_json=meta_example
            )

            prompts.append({
                'submission_id': submission_id,
                'year': year,
                'md_path': None,
                'type': 'meta',
                'review_index': -1,
                'original_json': meta_json,
                'prompt': prompt,
            })

    return prompts


class ReviewNormalizer(VLLMTransformer):
    """Normalizes reviews and meta-reviews using schema-in-prompt approach.

    No guided decoding - schema is in the prompt, JSON extracted from response.
    """

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Build vLLM request from prompt row."""
        prompt = row['prompt']
        return {
            "messages": [{"role": "user", "content": prompt}],
        }

    def postprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Parse output by extracting JSON and validating with pydantic."""
        generated_text = row.get("generated_text", "")
        review_type = row.get("type", "review")

        # Handle multiple completions (if n > 1) - always store as list
        if isinstance(generated_text, list):
            completions = generated_text
        else:
            completions = [generated_text]

        # Select schema based on type
        if review_type == 'review':
            schema = LLMUniversalReview
        else:
            schema = LLMUniversalMetaReview

        # Try to parse first valid completion
        parsed = extract_first_valid(completions, schema)
        normalized_json = parsed.model_dump_json() if parsed else None

        # Calculate time_taken_llm from metrics
        metrics = row.get("metrics", {})
        if isinstance(metrics, dict) and "last_token_ts" in metrics and "arrival_time" in metrics:
            time_taken_llm = metrics["last_token_ts"] - metrics["arrival_time"]
        else:
            time_taken_llm = None

        # Convert empty metrics to None for parquet
        if isinstance(metrics, dict) and len(metrics) == 0:
            metrics = None

        return {
            # Original columns
            'submission_id': row.get('submission_id'),
            'year': row.get('year'),
            'md_path': row.get('md_path'),
            'type': row.get('type'),
            'review_index': row.get('review_index'),
            'original_json': row.get('original_json'),
            'prompt': row.get('prompt'),  # Include original prompt
            # Generated columns - store ALL completions as list
            'generated_texts': completions,
            'normalized_json': normalized_json,
            'parse_success': parsed is not None,
            # Metrics
            'num_generated_tokens': row.get('num_generated_tokens'),
            'time_taken_llm': time_taken_llm,
            'metrics': metrics,
        }


class ReviewNormalizerTest(ReviewNormalizer):
    """Test version of ReviewNormalizer with dummy data generation."""

    @classmethod
    def create_test_instance(
        cls,
        n_samples: int = 10,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        **kwargs
    ) -> "ReviewNormalizerTest":
        """Create a test normalizer with dummy data.

        Args:
            n_samples: Number of test samples to generate
            model_name: Model to use for inference
            **kwargs: Additional args passed to VLLMTransformer

        Returns:
            Configured ReviewNormalizerTest instance
        """
        prompts = generate_dummy_reviews(n_samples)
        df = pd.DataFrame(prompts)

        return cls(
            df=df,
            model_name=model_name,
            max_tokens=4096,
            n=1,  # Single completion for test
            **kwargs
        )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_test_pipeline(
    output_dir: Path = Path("data/test_normalized_reviews"),
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel: int = 1,
    concurrency: int = 1,
    batch_size: int = 32,
    max_model_len: int = 4096,
    ray_address: Optional[str] = None,
    n_samples: int = 10,
    verbose: bool = True,
):
    """Run test pipeline with dummy data.

    Args:
        output_dir: Output directory for parquet files
        model_name: HuggingFace model name
        tensor_parallel: GPUs to split model across
        concurrency: vLLM replicas
        batch_size: Batch size for inference
        max_model_len: Maximum context length
        ray_address: Ray cluster address
        n_samples: Number of test samples (default: 10)
        verbose: Print progress
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Running test pipeline with {n_samples} dummy samples...")
        print(f"Model: {model_name}")

    # Generate dummy data
    prompts = generate_dummy_reviews(n_samples)
    review_count = sum(1 for p in prompts if p['type'] == 'review')
    meta_count = sum(1 for p in prompts if p['type'] == 'meta')

    if verbose:
        print(f"Generated {len(prompts)} prompts:")
        print(f"  Reviews: {review_count}, Meta-reviews: {meta_count}")

    # Single normalizer for all prompts
    df = pd.DataFrame(prompts)
    normalizer = ReviewNormalizer(
        df=df,
        model_name=model_name,
        tensor_parallel=tensor_parallel,
        concurrency=concurrency,
        batch_size=batch_size,
        max_model_len=max_model_len,
        ray_address=ray_address,
        max_tokens=4096,
        n=1,
    )

    output_path = output_dir / "test_normalized.parquet"
    normalizer.transform_to_parquet(str(output_path))

    print(f"\nOutput written to: {output_path}")

    if verbose:
        result_df = pd.read_parquet(output_path)
        total = len(result_df)
        successful = result_df['parse_success'].sum()
        print(f"Total: {total}, Parsed: {successful} ({100*successful/total:.1f}%)")


def run_pipeline(
    data_dir: Path = Path("data/full_run"),
    output_dir: Path = None,
    output_filename: str = None,
    years: List[int] = None,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel: int = 1,
    concurrency: int = 1,
    batch_size: int = 32,
    max_model_len: int = 8192,
    ray_address: Optional[str] = None,
    n_completions: int = 2,  # Number of completions to generate
    limit: int = None,
    per_year_limit: int = None,
    dry_run: bool = False,
    verbose: bool = True,
):
    """Run the review normalization pipeline.

    Args:
        data_dir: Base data directory with pickle files and normalized/
        output_dir: Output directory for parquet files
        output_filename: Custom output filename (default: normalized_reviews.parquet)
        years: Years to process (default: 2020-2026)
        model_name: HuggingFace model name
        tensor_parallel: GPUs to split model across
        concurrency: vLLM replicas
        batch_size: Batch size for inference
        max_model_len: Maximum context length
        ray_address: Ray cluster address
        n_completions: Number of completions for retry logic
        limit: Limit total number of submissions to process
        per_year_limit: Limit submissions per year (for preview mode)
        dry_run: Use 50 samples for testing
        verbose: Print progress
    """
    if output_dir is None:
        output_dir = data_dir / "normalized_reviews"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    years = years or DEFAULT_YEARS
    normalized_dir = data_dir / "normalized"

    # Load raw notes from pickle
    if verbose:
        print("Loading raw notes from pickle files...")
    notes_by_id = load_all_raw_notes(data_dir, years)
    print(f"Total notes loaded: {len(notes_by_id)}")

    # Get valid submission IDs (those with images)
    if verbose:
        print("Getting valid submission IDs...")
    valid_ids = get_valid_submission_ids(normalized_dir)
    print(f"Valid submissions (with images): {len(valid_ids)}")

    # Filter to valid IDs
    valid_notes = {
        sub_id: (note, year)
        for sub_id, (note, year) in notes_by_id.items()
        if sub_id in valid_ids
    }
    print(f"Submissions with notes: {len(valid_notes)}")

    # Apply per-year limit (for preview mode)
    if per_year_limit:
        from collections import defaultdict
        by_year = defaultdict(list)
        for sub_id, (note, year) in valid_notes.items():
            by_year[year].append((sub_id, note, year))

        sampled = {}
        for year in sorted(by_year.keys()):
            items = by_year[year][:per_year_limit]
            for sub_id, note, yr in items:
                sampled[sub_id] = (note, yr)
            if verbose:
                print(f"  {year}: sampled {len(items)} submissions")

        valid_notes = sampled
        print(f"Per-year limit ({per_year_limit}): {len(valid_notes)} total submissions")

    # Apply total limit
    if limit:
        valid_notes = dict(list(valid_notes.items())[:limit])
        print(f"Limited to: {len(valid_notes)} submissions")

    # Build prompts for all reviews
    if verbose:
        print("Building prompts...")
    all_prompts = []

    for sub_id, (note, year) in valid_notes.items():
        md_path = get_md_path(normalized_dir, year, sub_id)
        prompts = build_review_prompts(sub_id, year, note, md_path)
        all_prompts.extend(prompts)

    print(f"Total prompts: {len(all_prompts)}")
    review_count = sum(1 for p in all_prompts if p['type'] == 'review')
    meta_count = sum(1 for p in all_prompts if p['type'] == 'meta')
    print(f"  Reviews: {review_count}, Meta-reviews: {meta_count}")

    # Dry run: sample 50
    if dry_run:
        import random
        random.seed(42)
        all_prompts = random.sample(all_prompts, min(50, len(all_prompts)))
        print(f"Dry run: sampled {len(all_prompts)} prompts")

    # Determine output path
    if output_filename:
        output_path = output_dir / output_filename
    elif dry_run:
        output_path = output_dir / "normalized_dry_run.parquet"
    else:
        output_path = output_dir / "normalized.parquet"

    # Single normalizer for all prompts
    df = pd.DataFrame(all_prompts)
    normalizer = ReviewNormalizer(
        df=df,
        model_name=model_name,
        tensor_parallel=tensor_parallel,
        concurrency=concurrency,
        batch_size=batch_size,
        max_model_len=max_model_len,
        ray_address=ray_address,
        max_tokens=4096,
        n=n_completions,
    )

    normalizer.transform_to_parquet(str(output_path))
    print(f"\nOutput written to: {output_path}")

    if verbose:
        result_df = pd.read_parquet(output_path)
        total = len(result_df)
        successful = result_df['parse_success'].sum()
        print(f"Total: {total}, Parsed: {successful} ({100*successful/total:.1f}%)")


# =============================================================================
# BATCH PROCESSING (PARQUET INPUT)
# =============================================================================

def prepare_input_parquet(
    data_dir: Path = Path("data/full_run"),
    output_path: Path = Path("data/normalized_reviews/reviews_input.parquet"),
    years: List[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Prepare and save all review prompts to parquet for batch processing.

    Loads pickle files, filters valid submissions, builds prompts, saves to parquet.
    This avoids repeated loading for parallel jobs.

    Args:
        data_dir: Base data directory with pickle files
        output_path: Output parquet path
        years: Years to process (default: all)
        verbose: Print progress

    Returns:
        DataFrame with all prompts
    """
    years = years or DEFAULT_YEARS
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build index of all submissions in normalized directory
    normalized_dir = data_dir / "normalized"
    norm_index = build_normalized_index(normalized_dir)
    valid_ids = set(norm_index.keys())

    if verbose:
        print(f"Found {len(valid_ids)} submissions in normalized index")

    # Load notes and build prompts
    all_prompts = []
    for year in years:
        pkl_path = data_dir / f"get_all_notes_{year}.pickle"
        if not pkl_path.exists():
            if verbose:
                print(f"  {year}: pickle not found, skipping")
            continue

        notes = load_raw_notes(pkl_path)
        year_count = 0
        year_prompts = 0
        for note in notes:
            if note.id not in valid_ids:
                continue
            md_path = normalized_dir / str(year) / note.id / f"{note.id}.md"
            md_path_str = str(md_path) if md_path.exists() else None
            prompts = build_review_prompts(note.id, year, note, md_path_str)
            all_prompts.extend(prompts)
            year_count += 1
            year_prompts += len(prompts)

        if verbose:
            print(f"  {year}: {year_count} submissions, {year_prompts} prompts")

    df = pd.DataFrame(all_prompts)
    df.to_parquet(output_path)

    if verbose:
        print(f"\nSaved {len(df)} prompts to {output_path}")
        print(f"  Reviews: {(df['type'] == 'review').sum()}")
        print(f"  Meta-reviews: {(df['type'] == 'meta').sum()}")

    return df


def run_from_parquet(
    input_parquet: Path,
    output_dir: Path,
    batch_idx: int = 0,
    num_batches: int = 1,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel: int = 2,
    concurrency: int = 1,
    batch_size: int = 32,
    max_model_len: int = 8192,
    max_tokens: int = 4096,
    ray_address: Optional[str] = None,
    n_completions: int = 2,
    verbose: bool = True,
):
    """Run normalization on a batch slice of the input parquet.

    Args:
        input_parquet: Path to reviews_input.parquet
        output_dir: Base output directory (batch_{idx} appended)
        batch_idx: Which batch (0 to num_batches-1)
        num_batches: Total number of batches
        model_name: HuggingFace model name
        tensor_parallel: GPUs per replica
        concurrency: Number of vLLM replicas
        batch_size: Batch size for inference
        max_model_len: Maximum context length
        ray_address: Ray cluster address
        n_completions: Completions per request
        verbose: Print progress
    """
    input_parquet = Path(input_parquet)
    output_dir = Path(output_dir)

    # Read parquet
    df = pd.read_parquet(input_parquet)

    # Slice for this batch
    total_rows = len(df)
    batch_size_rows = total_rows // num_batches
    start_idx = batch_idx * batch_size_rows
    # Last batch gets remaining rows
    end_idx = start_idx + batch_size_rows if batch_idx < num_batches - 1 else total_rows
    df_batch = df.iloc[start_idx:end_idx].copy()

    if verbose:
        print(f"Batch {batch_idx}/{num_batches}: rows {start_idx}-{end_idx} ({len(df_batch)} prompts)")

    # Output path
    batch_output_dir = output_dir / f"batch_{batch_idx}"
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = batch_output_dir / "normalized.parquet"

    # Resume: skip already-processed samples
    if output_path.exists():
        done_df = pd.read_parquet(output_path)
        done_keys = set(
            done_df['submission_id'] + '_' +
            done_df['type'] + '_' +
            done_df['review_index'].astype(str)
        )

        df_batch['_key'] = (
            df_batch['submission_id'] + '_' +
            df_batch['type'] + '_' +
            df_batch['review_index'].astype(str)
        )
        original_count = len(df_batch)
        df_batch = df_batch[~df_batch['_key'].isin(done_keys)].drop(columns=['_key'])

        if verbose:
            print(f"Resume: {len(done_keys)} already done, {len(df_batch)} remaining (of {original_count})")

        if len(df_batch) == 0:
            print("All samples already processed!")
            return

    # Run normalizer
    normalizer = ReviewNormalizer(
        df=df_batch,
        model_name=model_name,
        tensor_parallel=tensor_parallel,
        concurrency=concurrency,
        batch_size=batch_size,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        ray_address=ray_address,
        n=n_completions,
    )
    normalizer.transform_to_parquet(str(output_path))

    if verbose:
        result = pd.read_parquet(output_path)
        success_rate = result['parse_success'].mean() * 100
        print(f"Batch {batch_idx} complete: {success_rate:.1f}% parse success")
        print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Normalize ICLR reviews using vLLM")
    parser.add_argument("--data-dir", type=Path, default=Path("data/full_run"),
                        help="Base data directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: data_dir/normalized_reviews)")
    parser.add_argument("--year", type=int, default=None,
                        help="Process single year only")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of vLLM replicas")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Maximum context length")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Maximum output tokens")
    parser.add_argument("--ray-address", type=str, default=None,
                        help="Ray cluster address")
    parser.add_argument("--n-completions", type=int, default=2,
                        help="Number of completions per request")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of submissions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with 50 samples for testing")
    parser.add_argument("--test-run", action="store_true",
                        help="Run with 10 dummy samples (no data loading required)")
    parser.add_argument("--test-samples", type=int, default=10,
                        help="Number of samples for --test-run (default: 10)")
    parser.add_argument("--preview", action="store_true",
                        help="Preview mode: 5 submissions per year across all years")
    parser.add_argument("--preview-per-year", type=int, default=5,
                        help="Number of submissions per year for --preview (default: 5)")
    parser.add_argument("--output-path", type=Path, default=None,
                        help="Output parquet file path (for --preview mode)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")

    # Batch processing modes
    parser.add_argument("--prepare-input", action="store_true",
                        help="Prepare input parquet from pickle files (run once)")
    parser.add_argument("--input-parquet", type=Path, default=None,
                        help="Path to prepared input parquet (skip pickle loading)")
    parser.add_argument("--batch-idx", type=int, default=0,
                        help="Batch index for job array (0 to num-batches-1)")
    parser.add_argument("--num-batches", type=int, default=1,
                        help="Total number of batches")

    args = parser.parse_args()

    # Prepare input parquet mode - run once to create reviews_input.parquet
    if args.prepare_input:
        prepare_input_parquet(
            data_dir=args.data_dir,
            output_path=args.output_path or Path("data/normalized_reviews/reviews_input.parquet"),
            verbose=not args.quiet,
        )
        return

    # Batch processing mode - process a slice of the input parquet
    if args.input_parquet:
        output_dir = args.output_dir or Path("data/normalized_reviews")
        run_from_parquet(
            input_parquet=args.input_parquet,
            output_dir=output_dir,
            batch_idx=args.batch_idx,
            num_batches=args.num_batches,
            model_name=args.model_name,
            tensor_parallel=args.tensor_parallel,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            ray_address=args.ray_address,
            n_completions=args.n_completions,
            verbose=not args.quiet,
        )
        return

    # Test run mode - uses dummy data, no pickle loading needed
    if args.test_run:
        output_dir = args.output_dir or Path("data/test_normalized_reviews")
        run_test_pipeline(
            output_dir=output_dir,
            model_name=args.model_name,
            tensor_parallel=args.tensor_parallel,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
            max_model_len=args.max_model_len,
            ray_address=args.ray_address,
            n_samples=args.test_samples,
            verbose=not args.quiet,
        )
        return

    # Preview mode - sample per year across all years
    if args.preview:
        if not args.output_path:
            print("ERROR: --preview requires --output-path to specify the parquet output file")
            sys.exit(1)

        run_pipeline(
            data_dir=args.data_dir,
            output_dir=args.output_path.parent,
            output_filename=args.output_path.name,
            years=None,  # All years
            model_name=args.model_name,
            tensor_parallel=args.tensor_parallel,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
            max_model_len=args.max_model_len,
            ray_address=args.ray_address,
            n_completions=args.n_completions,
            per_year_limit=args.preview_per_year,
            verbose=not args.quiet,
        )
        return

    # Normal pipeline
    years = [args.year] if args.year else None

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        years=years,
        model_name=args.model_name,
        tensor_parallel=args.tensor_parallel,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        ray_address=args.ray_address,
        n_completions=args.n_completions,
        limit=args.limit,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
