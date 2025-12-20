#!/usr/bin/env python3
"""
ICLR Review Prediction using vLLM with Ray Data

Predicts various metrics for ICLR papers based on title and abstract:
1. Paper decision (year-specific options)
2. Citations at horizons [1,2,3,4,5,6] years
3. Review rating [1-10]
5. Impact percentile [0-1]
6. Confidence-weighted quality [0-1]
7. Binary accept/reject decision

NOTE: For Gemini models, use lib/gemini_batch_submit.py instead.

Usage (from NipsIclrData directory):
    # Zero-shot inference
    python lib/predict_iclr_reviews.py --dataset_path data/train.csv --output_dir parquet_out/ \\
        --model_name meta-llama/Llama-3.1-8B-Instruct --indicator 7

    # Few-shot inference
    python lib/predict_iclr_reviews.py --dataset_path data/train.csv --output_dir parquet_out/ \\
        --model_name meta-llama/Llama-3.1-8B-Instruct --indicator 7 --use_fewshot

    # Paper reviews mode
    python lib/predict_iclr_reviews.py --dataset_path data/train.csv --output_dir parquet_out/ \\
        --model_name meta-llama/Llama-3.1-8B-Instruct --indicator 7 --use_paper_reviews

    # Dry run (50 samples)
    python lib/predict_iclr_reviews.py --dataset_path data/train.csv --output_dir parquet_out/ \\
        --model_name meta-llama/Llama-3.1-8B-Instruct --indicator 7 --dry_run
"""

import os
import sys
from typing import Dict, Any, List

# Add lib directory to path for imports when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
import pandas as pd
from dotenv import load_dotenv

import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Import shared utilities
from inference_utils import (
    create_base_parser,
    add_vllm_args,
    validate_args,
    load_dataset,
    create_output_path,
    sanitize_model_name,
    extract_year_decisions,
    get_available_horizons,
    build_prompt_task7_critical,
    parse_output_task1,
    parse_output_task2,
    parse_output_task3,
    parse_output_task5,
    parse_output_task6,
    parse_output_task7,
)


# ============================================================================
# PROMPT BUILDERS (Tasks 1-6 - kept here for non-task7 support)
# ============================================================================

def format_fewshot_examples_task1(examples: List[Dict[str, Any]]) -> str:
    """Format few-shot examples for task 1 (paper decision)."""
    if not examples:
        return ""

    fewshot_str = "\n\nHere are some examples:\n\n"
    for i, ex in enumerate(examples, 1):
        decision = ex.get('paper_decision', 'Unknown')
        fewshot_str += f"""Example {i}:
Title: {ex['title']}

Abstract: {ex['abstract']}

Answer: {decision}

"""
    return fewshot_str


def build_prompt_task1(row: Dict[str, Any]) -> str:
    """Task 1: Predict paper decision."""
    year = row["year"]
    title = row["title"]
    abstract = row["abstract"]
    decisions = row.get("_year_decisions", [])
    fewshot_examples = row.get("_holdout_examples", [])

    if not decisions:
        return "No decisions available for this year"

    decision_list = "\n".join([f"- {d}" for d in decisions])
    fewshot_str = format_fewshot_examples_task1(fewshot_examples) if fewshot_examples else ""

    prompt = f"""You are an expert reviewer for the ICLR {year} conference. Based on the paper's title and abstract, predict the final decision.{fewshot_str}

Now predict for this paper:

Title: {title}

Abstract: {abstract}

Possible decisions:
{decision_list}

Provide your response in the following format:
Reasoning: [Your reasoning about the paper's quality, novelty, and fit]
Answer: [Exact decision string from the list above]"""

    return prompt


def build_prompt_task2(row: Dict[str, Any]) -> str:
    """Task 2: Predict citations at multiple horizons."""
    year = row["year"]
    title = row["title"]
    abstract = row["abstract"]
    horizons = get_available_horizons(year)
    fewshot_examples = row.get("_holdout_examples", [])

    if not horizons:
        return "No valid horizons"

    horizon_str = ", ".join([f"{h} year{'s' if h > 1 else ''}" for h in horizons])

    # Format few-shot examples if available
    from inference_utils import calculate_paper_age
    fewshot_str = ""
    if fewshot_examples:
        fewshot_str = "\n\nHere are some examples:\n\n"
        for i, ex in enumerate(fewshot_examples[:3], 1):
            ex_year = int(ex['year'])
            ex_age = calculate_paper_age(ex_year, current_year=2025)
            # We know citations at this horizon (years since publication)
            known_horizon = min(ex_age, 6)
            total_citations = int(ex.get('gs_citation_copy', 10))

            fewshot_str += f"""Example {i}:
Title: {ex['title']}

Abstract: {ex['abstract']}

Answer: {total_citations} citations at horizon {known_horizon}

"""

    prompt = f"""You are an expert at predicting research impact. Based on the paper's title and abstract, predict how many citations it will receive at different time horizons.{fewshot_str}

Now predict for this paper:

Title: {title}

Abstract: {abstract}

Publication year: {year - 1}

Predict the citation count at the following horizons: {horizon_str} after publication.

IMPORTANT: Provide citations PER YEAR (not cumulative sum). For example, if a paper gets 10 citations in year 1 and 15 MORE in year 2, respond with "10,15" not "10,25".

Provide your response in the following format:
Reasoning: [Your assessment of the paper's potential impact and why]
Answer: [Comma-separated integers (e.g., "5,12,25,45,67,89"), exactly {len(horizons)} numbers]"""

    return prompt


def build_prompt_task3(row: Dict[str, Any]) -> str:
    """Task 3: Predict review rating [1-10]."""
    title = row["title"]
    abstract = row["abstract"]
    fewshot_examples = row.get("_holdout_examples", [])

    # Format few-shot examples if available
    fewshot_str = ""
    if fewshot_examples:
        fewshot_str = "\n\nHere are some examples:\n\n"
        for i, ex in enumerate(fewshot_examples[:3], 1):
            # Use mean of extracted_scores
            scores_str = ex.get('extracted_scores', '[7.0]')
            try:
                scores = eval(scores_str) if isinstance(scores_str, str) else scores_str
                avg_score = np.mean([float(s) for s in scores if s is not None])
                avg_score = round(avg_score, 1)  # Round to 1 decimal
            except:
                avg_score = 7.0

            fewshot_str += f"""Example {i}:
Title: {ex['title']}

Abstract: {ex['abstract']}

Answer: {avg_score}

"""

    prompt = f"""You are an expert ICLR reviewer. Rate this paper's quality on a scale from 1-10.{fewshot_str}

Now rate this paper:

Title: {title}

Abstract: {abstract}

Rating scale:
1-2: Strong rejection (trivial or wrong)
3-4: Clear rejection (not good enough)
5: Marginally below acceptance threshold
6: Marginally above acceptance threshold
7-8: Good paper, clear accept
9-10: Outstanding paper, strong accept

Provide your response in the following format:
Reasoning: [Your assessment of the paper's quality, strengths, and weaknesses]
Answer: [Single number from 1-10]"""

    return prompt


def build_prompt_task5(row: Dict[str, Any]) -> str:
    """Task 5: Impact percentile [0-1]."""
    title = row["title"]
    abstract = row["abstract"]
    fewshot_examples = row.get("_holdout_examples", [])

    # Format few-shot examples if available
    fewshot_str = ""
    if fewshot_examples:
        fewshot_str = "\n\nHere are some examples:\n\n"
        for i, ex in enumerate(fewshot_examples[:3], 1):
            citations = float(ex.get('gs_citation_copy', 10))
            # Estimate percentile based on citations (rough heuristic)
            percentile = min(0.95, max(0.1, citations / 100))
            fewshot_str += f"""Example {i}:
Title: {ex['title']}

Abstract: {ex['abstract']}

Reasoning: This paper addresses {'a fundamental problem' if percentile > 0.7 else 'an interesting problem'} with {'novel insights' if percentile > 0.7 else 'a reasonable approach'}. Expected {'high' if percentile > 0.7 else 'moderate'} citation impact.

Answer: {percentile:.2f}

"""

    prompt = f"""You are an expert at assessing research impact. Based on this paper's title and abstract, estimate its citation impact percentile.{fewshot_str}

Now assess this paper:

Title: {title}

Abstract: {abstract}

If you consider 1000 papers in the same research area as this paper, what percentile would this paper's citations be at?

- 0.0 = bottom percentile (least cited)
- 0.5 = median
- 0.9 = top 10%
- 1.0 = top percentile (most cited)

Provide your response in the following format:
Reasoning: [Your assessment of why this paper will have high/low citation impact]
Answer: [Decimal number between 0.0 and 1.0 (e.g., "0.75")]"""

    return prompt


def build_prompt_task6(row: Dict[str, Any]) -> str:
    """Task 6: Confidence-weighted quality [0-1]."""
    title = row["title"]
    abstract = row["abstract"]
    fewshot_examples = row.get("_holdout_examples", [])

    # Format few-shot examples if available
    fewshot_str = ""
    if fewshot_examples:
        fewshot_str = "\n\nHere are some examples:\n\n"
        for i, ex in enumerate(fewshot_examples[:3], 1):
            scores_str = ex.get('extracted_scores', '[7.0]')
            try:
                scores = eval(scores_str) if isinstance(scores_str, str) else scores_str
                avg_score = np.mean([float(s) for s in scores if s is not None])
                # Normalize 1-10 scale to 0-1
                quality = (avg_score - 1) / 9.0
            except:
                quality = 0.65

            fewshot_str += f"""Example {i}:
Title: {ex['title']}

Abstract: {ex['abstract']}

Reasoning: This paper presents {'strong' if quality > 0.7 else 'solid'} work. Reviewers would likely {'agree on high quality' if quality > 0.7 else 'find it acceptable'}. Expected confidence-weighted score reflects {'strong accept' if quality > 0.7 else 'borderline accept'}.

Answer: {quality:.2f}

"""

    prompt = f"""You are an expert ICLR reviewer. Based on this paper's title and abstract, estimate what its confidence-weighted average review score would be.{fewshot_str}

Now assess this paper:

Title: {title}

Abstract: {abstract}

Assume 3-4 reviewers rate this paper, each with different confidence levels. Higher confidence reviewers' scores are weighted more heavily. Estimate the final confidence-weighted average quality score, normalized to 0-1 scale.

- 0.0 = worst possible quality
- 0.5 = borderline paper
- 1.0 = exceptional quality

Provide your response in the following format:
Reasoning: [Your assessment of the paper's quality and likely reviewer consensus]
Answer: [Decimal number between 0.0 and 1.0 (e.g., "0.65")]"""

    return prompt


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Load environment variables
    load_dotenv()

    ctx = ray.data.DataContext.get_current()
    ctx.verbose_stats_logs = True

    # Use shared argument parser
    parser = create_base_parser("ICLR Review Prediction with vLLM")
    parser = add_vllm_args(parser)
    args = parser.parse_args()

    # Check for Gemini models - not supported in this script
    if args.model_name.lower().startswith('gemini'):
        raise ValueError(
            "Gemini models are not supported in predict_iclr_reviews.py.\n"
            "Please use gemini_batch_submit.py for Gemini inference."
        )

    # Validate arguments
    validate_args(args)

    # Validate few-shot usage for certain tasks
    if args.use_fewshot and args.indicator in [5, 6]:
        print(f"ERROR: Task {args.indicator} does not support few-shot prompting")
        print("Please run without --use_fewshot flag")
        sys.exit(1)

    # Create output path
    output_path = create_output_path(
        args.output_dir, args.indicator, args.model_name, args.version,
        args.use_fewshot, args.use_paper_reviews, args.pct, args.dry_run
    )

    print(f"Starting ICLR prediction task {args.indicator}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}, Concurrency: {args.concurrency}, Tensor Parallel: {args.tensor_parallel}")
    print(f"Output path: {output_path}")
    if args.use_paper_reviews:
        print("Using full paper content (MD files) - will use tensor_parallel=2 if not overridden")
    if args.dry_run:
        print("DRY RUN: Using 50 samples for testing")

    # Initialize Ray
    runtime_env = {
        "env_vars": {
            "UV_OFFLINE": os.environ.get("UV_OFFLINE", "1"),
            "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", "/n/fs/vision-mix/sk7524/caches/.uv"),
            "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE", "1"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1"),
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        }
    }

    # Initialize Ray - connect to existing cluster or start local
    if args.ray_address:
        print(f"Connecting to Ray cluster at {args.ray_address}")
        ray.init(address=args.ray_address, ignore_reinit_error=True, runtime_env=runtime_env)
    else:
        print("Auto-detecting Ray cluster...")
        ray.init(address="auto", ignore_reinit_error=True, runtime_env=runtime_env)

    # Load dataset using shared loader
    print(f"Loading dataset from {args.dataset_path}")
    df = load_dataset(args)
    print(f"Final dataset size: {len(df)} rows")

    # Extract year decisions for task 1 (already done in load_dataset)
    if args.indicator == 1:
        year_decisions = extract_year_decisions(df)
        for year, decisions in sorted(year_decisions.items()):
            print(f"  Year {year}: {len(decisions)} decisions - {decisions}")

    # Convert to Ray Dataset
    ds = ray.data.from_pandas(df)

    # Select prompt builder based on indicator
    prompt_builders = {
        1: build_prompt_task1,
        2: build_prompt_task2,
        3: build_prompt_task3,
        5: build_prompt_task5,
        6: build_prompt_task6,
        7: lambda row: build_prompt_task7_critical(
            row,
            fewshot_examples=row.get('_holdout_examples'),
            use_paper_reviews=args.use_paper_reviews
        ),
    }

    prompt_builder = prompt_builders[args.indicator]

    # Configure vLLM processor
    print("Configuring vLLM processor...")

    # Set context length - increase if using paper reviews
    max_model_len = 32000 if args.use_paper_reviews else 4096

    # Set tensor parallel - default to 2 for paper reviews (larger context needs more GPU memory)
    tensor_parallel = args.tensor_parallel
    if args.use_paper_reviews and tensor_parallel == 1:
        tensor_parallel = 2
        print(f"Auto-setting tensor_parallel=2 for paper reviews mode")

    config = vLLMEngineProcessorConfig(
        model_source=args.model_name,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        engine_kwargs={
            "max_model_len": max_model_len,
            "trust_remote_code": True,
            "tensor_parallel_size": tensor_parallel,
        }
    )

    # Define preprocessing function (vLLM format)
    def preprocess_fn(row: Dict[str, Any]) -> Dict[str, Any]:
        prompt = prompt_builder(row)
        return {
            "messages": [{"role": "user", "content": prompt}],
            "sampling_params": {
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            },
        }

    # Define postprocessing function
    def postprocess_fn(row: Dict[str, Any]) -> Dict[str, Any]:
        generated = row.get("generated_text", "")

        # Parse based on task
        if args.indicator == 1:
            decisions = row.get("_year_decisions", [])
            extracted_decision = parse_output_task1(generated, decisions)

        elif args.indicator == 2:
            horizons = get_available_horizons(row["year"])
            extracted_decision = parse_output_task2(generated, len(horizons))

        elif args.indicator == 3:
            extracted_decision = parse_output_task3(generated)

        elif args.indicator == 5:
            extracted_decision = parse_output_task5(generated)

        elif args.indicator == 6:
            extracted_decision = parse_output_task6(generated)

        elif args.indicator == 7:
            extracted_decision = parse_output_task7(generated)

        else:
            extracted_decision = {"reasoning": None, "answer": None}

        # Calculate time_taken_llm from metrics if available
        metrics = row.get("metrics", {})
        if isinstance(metrics, dict) and "last_token_ts" in metrics and "arrival_time" in metrics:
            time_taken_llm = metrics["last_token_ts"] - metrics["arrival_time"]
        else:
            time_taken_llm = None

        # Convert empty metrics dict to None (Parquet can't write empty structs)
        if isinstance(metrics, dict) and len(metrics) == 0:
            metrics = None

        # Get prompt from messages
        messages = row.get("messages", [])
        prompt = messages[0]["content"] if messages and len(messages) > 0 else ""

        # Return only the specified columns
        return {
            "generated_text": generated,
            "extracted_decision": extracted_decision,
            "original_index": row.get("original_index"),
            "indicator": row.get("indicator"),
            "prompt": prompt,
            "num_generated_tokens": row.get("num_generated_tokens"),
            "metrics": metrics,
            "time_taken_llm": time_taken_llm,
        }

    # Build processor
    processor = build_llm_processor(
        config,
        preprocess=preprocess_fn,
        postprocess=postprocess_fn,
    )

    # Apply processor
    print("Running inference...")
    ds = processor(ds)

    # Write to parquet (streaming, no memory load)
    print(f"Writing results to {output_path}")
    output = ds.write_parquet(output_path)

    print("Done!")

    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
