"""
Test all input modalities:
1. TEXT_ONLY variants (abstract, with reviews, full text, full context)
2. TEXT_WITH_IMAGES (requires VL model + images)
3. IMAGES_ONLY (requires VL model + images)

Usage:
    python experiments/test_input_modalities.py [--with-images]
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
VL_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
NUM_SAMPLES = 3  # Keep small for login node


def load_submissions(load_images: bool = False):
    """Load a few submissions for testing."""
    loader = ICLRDataLoader()
    submissions = []
    for i, s in enumerate(loader.load_from_huggingface(
        dataset_name=LOCAL_DATASET_PATH,
        years=["2020"],
        load_images=load_images,
    )):
        submissions.append(s)
        if i >= NUM_SAMPLES - 1:
            break
    return submissions


def test_text_only_abstract():
    """Test TEXT_ONLY with abstract only (no reviews, no markdown)."""
    print("\n" + "=" * 60)
    print("TEST: TEXT_ONLY - Abstract Only")
    print("=" * 60)

    submissions = load_submissions(load_images=False)
    print(f"Loaded {len(submissions)} submissions")

    predictor = UnifiedPredictor.create(
        input_modality=InputModality.TEXT_ONLY,
        output_type=OutputType.BINARY,
        training_mode=TrainingMode.ZERO_SHOT,
        config=PredictorConfig(
            model_name_or_path=TEXT_MODEL,
            model_type=ModelType.TEXT_ONLY,
        ),
        input_kwargs={
            "include_reviews": False,
            "include_markdown": False,
        },
    )

    # Show formatted sample
    samples = predictor.format_submissions(submissions[:1], include_label=False)
    print(f"\nSample prompt (first 500 chars):\n{samples[0].instruction[:200]}...")
    print(f"\nInput text (first 300 chars):\n{samples[0].input_text[:300]}...")

    # Run prediction
    results = predictor.predict(submissions)
    for i, r in enumerate(results):
        print(f"\n--- Result {i} ---")
        print(f"Prediction: {r.get('prediction')}")
        print(f"Raw (first 150 chars): {r.get('raw', '')[:150]}")


def test_text_only_with_reviews():
    """Test TEXT_ONLY with abstract + reviews."""
    print("\n" + "=" * 60)
    print("TEST: TEXT_ONLY - Abstract + Reviews")
    print("=" * 60)

    submissions = load_submissions(load_images=False)
    print(f"Loaded {len(submissions)} submissions")

    predictor = UnifiedPredictor.create(
        input_modality=InputModality.TEXT_ONLY,
        output_type=OutputType.BINARY,
        training_mode=TrainingMode.ZERO_SHOT,
        config=PredictorConfig(
            model_name_or_path=TEXT_MODEL,
            model_type=ModelType.TEXT_ONLY,
        ),
        input_kwargs={
            "include_reviews": True,
            "include_markdown": False,
        },
    )

    # Show formatted sample
    samples = predictor.format_submissions(submissions[:1], include_label=False)
    print(f"\nSample has reviews: {len(submissions[0].reviews)} reviews")
    print(f"Input text length: {len(samples[0].input_text)} chars")

    # Run prediction
    results = predictor.predict(submissions)
    for i, r in enumerate(results):
        print(f"\n--- Result {i} ---")
        print(f"Prediction: {r.get('prediction')}")
        print(f"Raw (first 150 chars): {r.get('raw', '')[:150]}")


def test_text_only_full_context():
    """Test TEXT_ONLY with abstract + reviews + markdown (full context)."""
    print("\n" + "=" * 60)
    print("TEST: TEXT_ONLY - Full Context (Abstract + Reviews + Markdown)")
    print("=" * 60)

    submissions = load_submissions(load_images=False)
    print(f"Loaded {len(submissions)} submissions")

    predictor = UnifiedPredictor.create(
        input_modality=InputModality.TEXT_ONLY,
        output_type=OutputType.BINARY,
        training_mode=TrainingMode.ZERO_SHOT,
        config=PredictorConfig(
            model_name_or_path=TEXT_MODEL,
            model_type=ModelType.TEXT_ONLY,
        ),
        input_kwargs={
            "include_reviews": True,
            "include_markdown": True,
            "max_tokens": 8000,  # Allow more context
        },
    )

    # Show formatted sample
    samples = predictor.format_submissions(submissions[:1], include_label=False)
    print(f"\nSample has markdown: {submissions[0].clean_md is not None}")
    print(f"Input text length: {len(samples[0].input_text)} chars")

    # Run prediction
    results = predictor.predict(submissions)
    for i, r in enumerate(results):
        print(f"\n--- Result {i} ---")
        print(f"Prediction: {r.get('prediction')}")
        print(f"Raw (first 150 chars): {r.get('raw', '')[:150]}")


def test_text_with_images():
    """Test TEXT_WITH_IMAGES modality (requires VL model and images)."""
    print("\n" + "=" * 60)
    print("TEST: TEXT_WITH_IMAGES - Abstract + PDF Page Images")
    print("=" * 60)

    submissions = load_submissions(load_images=True)
    print(f"Loaded {len(submissions)} submissions")

    # Check if any have images
    has_images = any(s.pdf_image_paths for s in submissions)
    if not has_images:
        print("WARNING: No submissions have images loaded!")
        print("Images may not exist at the paths specified in clean_pdf_img_paths")
        print("Skipping this test...")
        return

    predictor = UnifiedPredictor.create(
        input_modality=InputModality.TEXT_WITH_IMAGES,
        output_type=OutputType.BINARY,
        training_mode=TrainingMode.ZERO_SHOT,
        config=PredictorConfig(
            model_name_or_path=VL_MODEL,
            model_type=ModelType.VISION_LANGUAGE,
        ),
        input_kwargs={
            "include_reviews": False,
            "max_images": 5,  # Limit pages
        },
    )

    # Show formatted sample
    samples = predictor.format_submissions(submissions[:1], include_label=False)
    print(f"\nSample images: {samples[0].images}")
    print(f"Input text (first 300 chars):\n{samples[0].input_text[:300]}...")

    # Run prediction
    results = predictor.predict(submissions)
    for i, r in enumerate(results):
        print(f"\n--- Result {i} ---")
        print(f"Prediction: {r.get('prediction')}")
        print(f"Raw (first 150 chars): {r.get('raw', '')[:150]}")


def test_images_only():
    """Test IMAGES_ONLY modality (requires VL model and images)."""
    print("\n" + "=" * 60)
    print("TEST: IMAGES_ONLY - PDF Page Images Only (no text)")
    print("=" * 60)

    submissions = load_submissions(load_images=True)
    print(f"Loaded {len(submissions)} submissions")

    # Check if any have images
    has_images = any(s.pdf_image_paths for s in submissions)
    if not has_images:
        print("WARNING: No submissions have images loaded!")
        print("Images may not exist at the paths specified in clean_pdf_img_paths")
        print("Skipping this test...")
        return

    predictor = UnifiedPredictor.create(
        input_modality=InputModality.IMAGES_ONLY,
        output_type=OutputType.BINARY,
        training_mode=TrainingMode.ZERO_SHOT,
        config=PredictorConfig(
            model_name_or_path=VL_MODEL,
            model_type=ModelType.VISION_LANGUAGE,
        ),
        input_kwargs={
            "max_images": 8,  # Use more pages for images-only
        },
    )

    # Show formatted sample
    samples = predictor.format_submissions(submissions[:1], include_label=False)
    print(f"\nSample images: {samples[0].images}")
    print(f"Input text:\n{samples[0].input_text}")

    # Run prediction
    results = predictor.predict(submissions)
    for i, r in enumerate(results):
        print(f"\n--- Result {i} ---")
        print(f"Prediction: {r.get('prediction')}")
        print(f"Raw (first 150 chars): {r.get('raw', '')[:150]}")


def main():
    parser = argparse.ArgumentParser(description="Test input modalities")
    parser.add_argument(
        "--with-images",
        action="store_true",
        help="Run vision tests (requires VL model and image files)",
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["abstract", "reviews", "full", "text_images", "images_only", "all"],
        default="all",
        help="Which test to run",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("INPUT MODALITY TESTS")
    print("=" * 60)
    print(f"Dataset: {LOCAL_DATASET_PATH}")
    print(f"Text Model: {TEXT_MODEL}")
    print(f"VL Model: {VL_MODEL}")
    print(f"Samples per test: {NUM_SAMPLES}")

    # Text-only tests (always run)
    if args.test in ["abstract", "all"]:
        test_text_only_abstract()

    if args.test in ["reviews", "all"]:
        test_text_only_with_reviews()

    if args.test in ["full", "all"]:
        test_text_only_full_context()

    # Vision tests (only if --with-images)
    if args.with_images or args.test in ["text_images", "images_only"]:
        if args.test in ["text_images", "all"]:
            test_text_with_images()

        if args.test in ["images_only", "all"]:
            test_images_only()
    else:
        print("\n" + "=" * 60)
        print("SKIPPED: Vision tests (use --with-images to enable)")
        print("=" * 60)

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
