"""
Test all output/prediction types:
1. BINARY - Accept/Reject
2. MULTICLASS - Reject/Accept(Poster)/Accept(Spotlight)/Accept(Oral)
3. CITATION_PERCENTILE - Predict citation percentile [0, 1]
4. MEAN_RATING - Predict mean reviewer rating

Usage:
    python experiments/test_output_types.py [--test TYPE]
"""

import argparse
from pathlib import Path
from lib.llamafactory import (
    ICLRDataLoader,
    UnifiedPredictor,
    PredictorConfig,
    InputModality,
    OutputType,
    TrainingMode,
    ModelType,
)

# Configuration
LOCAL_DATASET_PATH = "/n/fs/vision-mix/jl0796/iclr-reviews-2020-2026"
TEXT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_SAMPLES = 3  # Keep small for login node


def load_submissions():
    """Load a few submissions for testing."""
    loader = ICLRDataLoader()
    submissions = []
    for i, s in enumerate(loader.load_from_huggingface(
        dataset_name=LOCAL_DATASET_PATH,
        years=["2020"],
        load_images=False,
    )):
        submissions.append(s)
        if i >= NUM_SAMPLES - 1:
            break
    return submissions


def test_binary_output():
    """Test BINARY output type (Accept/Reject)."""
    print("\n" + "=" * 60)
    print("TEST: OutputType.BINARY - Accept/Reject Prediction")
    print("=" * 60)

    submissions = load_submissions()
    print(f"Loaded {len(submissions)} submissions")

    predictor = UnifiedPredictor.create(
        input_modality=InputModality.TEXT_ONLY,
        output_type=OutputType.BINARY,
        training_mode=TrainingMode.ZERO_SHOT,
        config=PredictorConfig(
            model_name_or_path=TEXT_MODEL,
            model_type=ModelType.TEXT_ONLY,
        ),
        input_kwargs={"include_reviews": True},
    )

    # Show instruction
    samples = predictor.format_submissions(submissions[:1], include_label=False)
    print(f"\nInstruction:\n{samples[0].instruction}")

    # Show ground truth
    for s in submissions:
        print(f"  {s.submission_id}: decision={s.decision} -> binary={s.get_binary_label()}")

    # Run prediction
    results = predictor.predict(submissions)
    print("\nPredictions:")
    for i, r in enumerate(results):
        print(f"  {r.get('submission_id')}: {r.get('prediction')} (parse_success={r.get('parse_success')})")
        print(f"    Raw: {r.get('raw', '')[:100]}...")


def test_multiclass_output():
    """Test MULTICLASS output type (Reject/Poster/Spotlight/Oral)."""
    print("\n" + "=" * 60)
    print("TEST: OutputType.MULTICLASS - Decision Tier Prediction")
    print("=" * 60)

    submissions = load_submissions()
    print(f"Loaded {len(submissions)} submissions")

    predictor = UnifiedPredictor.create(
        input_modality=InputModality.TEXT_ONLY,
        output_type=OutputType.MULTICLASS,
        training_mode=TrainingMode.ZERO_SHOT,
        config=PredictorConfig(
            model_name_or_path=TEXT_MODEL,
            model_type=ModelType.TEXT_ONLY,
        ),
        input_kwargs={"include_reviews": True},
    )

    # Show instruction
    samples = predictor.format_submissions(submissions[:1], include_label=False)
    print(f"\nInstruction:\n{samples[0].instruction}")

    # Show ground truth
    for s in submissions:
        print(f"  {s.submission_id}: decision={s.decision} -> multiclass={s.get_multiclass_label()}")

    # Run prediction
    results = predictor.predict(submissions)
    print("\nPredictions:")
    for i, r in enumerate(results):
        print(f"  {r.get('submission_id')}: {r.get('prediction')} (parse_success={r.get('parse_success')})")
        print(f"    Raw: {r.get('raw', '')[:100]}...")


def test_citation_percentile_output():
    """Test CITATION_PERCENTILE output type."""
    print("\n" + "=" * 60)
    print("TEST: OutputType.CITATION_PERCENTILE - Citation Prediction")
    print("=" * 60)

    submissions = load_submissions()
    print(f"Loaded {len(submissions)} submissions")

    predictor = UnifiedPredictor.create(
        input_modality=InputModality.TEXT_ONLY,
        output_type=OutputType.CITATION_PERCENTILE,
        training_mode=TrainingMode.ZERO_SHOT,
        config=PredictorConfig(
            model_name_or_path=TEXT_MODEL,
            model_type=ModelType.TEXT_ONLY,
        ),
        input_kwargs={"include_reviews": False},  # Predict from paper content only
    )

    # Show instruction
    samples = predictor.format_submissions(submissions[:1], include_label=False)
    print(f"\nInstruction:\n{samples[0].instruction}")

    # Show ground truth if available
    print("\nGround truth (if available):")
    for s in submissions:
        citation_pctl = s.labels.get("citation_percentile")
        print(f"  {s.submission_id}: citation_percentile={citation_pctl}")

    # Run prediction
    results = predictor.predict(submissions)
    print("\nPredictions:")
    for i, r in enumerate(results):
        print(f"  {r.get('submission_id')}: {r.get('prediction')} (parse_success={r.get('parse_success')})")
        print(f"    Raw: {r.get('raw', '')[:100]}...")


def test_mean_rating_output():
    """Test MEAN_RATING output type."""
    print("\n" + "=" * 60)
    print("TEST: OutputType.MEAN_RATING - Reviewer Rating Prediction")
    print("=" * 60)

    submissions = load_submissions()
    print(f"Loaded {len(submissions)} submissions")

    predictor = UnifiedPredictor.create(
        input_modality=InputModality.TEXT_ONLY,
        output_type=OutputType.MEAN_RATING,
        training_mode=TrainingMode.ZERO_SHOT,
        config=PredictorConfig(
            model_name_or_path=TEXT_MODEL,
            model_type=ModelType.TEXT_ONLY,
        ),
        input_kwargs={"include_reviews": False},  # Predict rating without seeing reviews
    )

    # Show instruction
    samples = predictor.format_submissions(submissions[:1], include_label=False)
    print(f"\nInstruction:\n{samples[0].instruction}")

    # Show ground truth
    print("\nGround truth (calculated from reviews):")
    for s in submissions:
        mean_rating = s.get_mean_rating()
        num_reviews = len(s.reviews)
        rating_str = f"{mean_rating:.2f}" if mean_rating else "N/A"
        print(f"  {s.submission_id}: mean_rating={rating_str} ({num_reviews} reviews)")

    # Run prediction
    results = predictor.predict(submissions)
    print("\nPredictions:")
    for i, r in enumerate(results):
        print(f"  {r.get('submission_id')}: {r.get('prediction')} (parse_success={r.get('parse_success')})")
        print(f"    Raw: {r.get('raw', '')[:100]}...")


def test_with_reasoning():
    """Test output with reasoning enabled."""
    print("\n" + "=" * 60)
    print("TEST: BINARY with reasoning enabled")
    print("=" * 60)

    submissions = load_submissions()
    print(f"Loaded {len(submissions)} submissions")

    predictor = UnifiedPredictor.create(
        input_modality=InputModality.TEXT_ONLY,
        output_type=OutputType.BINARY,
        training_mode=TrainingMode.ZERO_SHOT,
        config=PredictorConfig(
            model_name_or_path=TEXT_MODEL,
            model_type=ModelType.TEXT_ONLY,
        ),
        input_kwargs={"include_reviews": True},
        output_kwargs={"include_reasoning": True},  # Ask for reasoning
    )

    # Show instruction
    samples = predictor.format_submissions(submissions[:1], include_label=False)
    print(f"\nInstruction:\n{samples[0].instruction}")

    # Run prediction
    results = predictor.predict(submissions)
    print("\nPredictions with reasoning:")
    for i, r in enumerate(results):
        print(f"\n--- {r.get('submission_id')} ---")
        print(f"Prediction: {r.get('prediction')}")
        print(f"Reasoning: {r.get('reasoning', 'N/A')[:200]}...")
        print(f"Parse success: {r.get('parse_success')}")


def test_evaluation_metrics():
    """Test computing evaluation metrics on predictions."""
    print("\n" + "=" * 60)
    print("TEST: Evaluation Metrics")
    print("=" * 60)

    submissions = load_submissions()
    print(f"Loaded {len(submissions)} submissions")

    predictor = UnifiedPredictor.create(
        input_modality=InputModality.TEXT_ONLY,
        output_type=OutputType.BINARY,
        training_mode=TrainingMode.ZERO_SHOT,
        config=PredictorConfig(
            model_name_or_path=TEXT_MODEL,
            model_type=ModelType.TEXT_ONLY,
        ),
        input_kwargs={"include_reviews": True},
    )

    # Run prediction with evaluation
    eval_results = predictor.predict_with_evaluation(submissions)

    print(f"\nNum samples: {eval_results['num_samples']}")
    print(f"\nMetrics:")
    for key, value in eval_results['metrics'].items():
        print(f"  {key}: {value:.4f}")

    print(f"\nPrediction breakdown:")
    predictions = eval_results['predictions']
    accept_count = sum(1 for p in predictions if p.get('prediction') == 'Accept')
    reject_count = sum(1 for p in predictions if p.get('prediction') == 'Reject')
    none_count = sum(1 for p in predictions if p.get('prediction') is None)
    print(f"  Accept: {accept_count}")
    print(f"  Reject: {reject_count}")
    print(f"  Failed to parse: {none_count}")


def main():
    parser = argparse.ArgumentParser(description="Test output types")
    parser.add_argument(
        "--test",
        type=str,
        choices=["binary", "multiclass", "citation", "rating", "reasoning", "metrics", "all"],
        default="all",
        help="Which test to run",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("OUTPUT TYPE TESTS")
    print("=" * 60)
    print(f"Dataset: {LOCAL_DATASET_PATH}")
    print(f"Model: {TEXT_MODEL}")
    print(f"Samples per test: {NUM_SAMPLES}")

    if args.test in ["binary", "all"]:
        test_binary_output()

    if args.test in ["multiclass", "all"]:
        test_multiclass_output()

    if args.test in ["citation", "all"]:
        test_citation_percentile_output()

    if args.test in ["rating", "all"]:
        test_mean_rating_output()

    if args.test in ["reasoning", "all"]:
        test_with_reasoning()

    if args.test in ["metrics", "all"]:
        test_evaluation_metrics()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
