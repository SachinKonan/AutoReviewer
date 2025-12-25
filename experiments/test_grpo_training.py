"""
Test GRPO (Group Relative Policy Optimization) training functionality.

Tests:
1. Data preparation (prompt + ground truth format)
2. Reward function (custom and default)
3. Config generation
4. Training execution (requires easyr1 or llamafactory)
5. Inference with trained adapter

Usage:
    # Test data prep, reward functions, and config (no GPU needed)
    python experiments/test_grpo_training.py --test prep

    # Run actual training (requires GPU + easyr1/llamafactory)
    python experiments/test_grpo_training.py --test train

    # Test inference with adapter
    python experiments/test_grpo_training.py --test inference --adapter-path /path/to/adapter
"""

import argparse
import json
from pathlib import Path
from typing import Callable
from lib.llamafactory import (
    ICLRDataLoader,
    UnifiedPredictor,
    PredictorConfig,
    InputModality,
    OutputType,
    TrainingMode,
    TrainingConfig,
    ModelType,
)
from lib.llamafactory.training.grpo import GRPOMode
from lib.llamafactory.inputs.text_only import TextOnlyFormatter
from lib.llamafactory.outputs.binary import BinaryAcceptRejectHandler
from lib.llamafactory.outputs.rating import MeanRatingHandler

# Configuration
LOCAL_DATASET_PATH = "/n/fs/vision-mix/jl0796/iclr-reviews-2020-2026"
TEXT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = Path("outputs/grpo_test")
NUM_TRAIN_SAMPLES = 20
NUM_EVAL_SAMPLES = 5


def load_submissions(num_samples: int = 10):
    """Load submissions for training/testing."""
    loader = ICLRDataLoader()
    submissions = []
    for i, s in enumerate(loader.load_from_huggingface(
        dataset_name=LOCAL_DATASET_PATH,
        years=["2020"],
        load_images=False,
    )):
        submissions.append(s)
        if i >= num_samples - 1:
            break
    return submissions


def test_data_preparation():
    """Test data preparation for GRPO training."""
    print("\n" + "=" * 60)
    print("TEST: GRPO Data Preparation")
    print("=" * 60)

    # Load data
    submissions = load_submissions(NUM_TRAIN_SAMPLES)
    print(f"Loaded {len(submissions)} submissions")

    # Create components
    formatter = TextOnlyFormatter(include_reviews=True)
    handler = BinaryAcceptRejectHandler()
    config = TrainingConfig(
        model_name_or_path=TEXT_MODEL,
        model_type=ModelType.TEXT_ONLY,
        num_epochs=1,
        batch_size=2,
        grpo_group_size=4,
        grpo_kl_coeff=0.1,
    )

    # Create GRPO mode
    grpo_mode = GRPOMode(
        config=config,
        input_formatter=formatter,
        output_handler=handler,
    )

    # Format samples
    samples = grpo_mode.format_samples(submissions, include_label=True)
    print(f"\nFormatted {len(samples)} samples")

    # Show sample
    print("\n--- Sample 0 ---")
    print(f"Instruction (first 200 chars): {samples[0].instruction[:200]}...")
    print(f"Input (first 200 chars): {samples[0].input_text[:200]}...")
    print(f"Output (ground truth): {samples[0].output}")

    # Prepare data
    data_path = OUTPUT_DIR / "data"
    data_info = grpo_mode.prepare_data(samples, data_path)

    print(f"\n--- Data Info ---")
    for key, value in data_info.items():
        print(f"  {key}: {value}")

    # Check generated file
    print(f"\n--- Generated File ---")
    data_file = Path(data_info["data_file"])
    if data_file.exists():
        with open(data_file) as f:
            data = json.load(f)
        print(f"  {data_file.name}: {len(data)} samples")
        print(f"  First sample keys: {list(data[0].keys())}")
        print(f"  First prompt (first 100 chars): {data[0]['prompt'][:100]}...")
        print(f"  First ground_truth: {data[0]['ground_truth']}")

    return grpo_mode, data_info


def test_reward_functions():
    """Test reward function behavior."""
    print("\n" + "=" * 60)
    print("TEST: GRPO Reward Functions")
    print("=" * 60)

    # Test with binary handler
    print("\n--- Binary Reward Function ---")
    binary_handler = BinaryAcceptRejectHandler()
    config = TrainingConfig(model_name_or_path=TEXT_MODEL)
    formatter = TextOnlyFormatter()

    grpo_binary = GRPOMode(
        config=config,
        input_formatter=formatter,
        output_handler=binary_handler,
    )

    # Test cases
    test_cases = [
        ("Accept", "Accept", "Correct Accept"),
        ("Reject", "Reject", "Correct Reject"),
        ("Accept", "Reject", "Wrong prediction"),
        ("Based on the reviews, I think this should be Accept", "Accept", "Verbose correct"),
        ("This paper should be rejected", "Accept", "Verbose wrong"),
    ]

    for prediction, ground_truth, description in test_cases:
        reward = grpo_binary._default_reward(prediction, ground_truth)
        print(f"  {description}: reward={reward:.2f}")
        print(f"    pred='{prediction[:50]}...' gt='{ground_truth}'")

    # Test with rating handler
    print("\n--- Rating Reward Function ---")
    rating_handler = MeanRatingHandler()

    grpo_rating = GRPOMode(
        config=config,
        input_formatter=formatter,
        output_handler=rating_handler,
    )

    rating_cases = [
        ("7.5", "7.5", "Exact match"),
        ("7.0", "7.5", "Close prediction"),
        ("5.0", "8.0", "Far off prediction"),
        ("Answer: 7.5", "7.5", "Formatted answer"),
    ]

    for prediction, ground_truth, description in rating_cases:
        reward = grpo_rating._default_reward(prediction, ground_truth)
        print(f"  {description}: reward={reward:.2f}")
        print(f"    pred='{prediction}' gt='{ground_truth}'")

    # Test custom reward function
    print("\n--- Custom Reward Function ---")

    def custom_reward(prediction: str, ground_truth: str) -> float:
        """Custom reward that gives partial credit."""
        parsed = binary_handler.parse_prediction(prediction)
        pred = parsed.get("prediction")

        if pred == ground_truth:
            return 1.0
        elif pred is None:
            return -0.5  # Penalty for unparseable output
        else:
            return 0.0

    grpo_custom = GRPOMode(
        config=config,
        input_formatter=formatter,
        output_handler=binary_handler,
        reward_function=custom_reward,
    )

    custom_cases = [
        ("Accept", "Accept", "Correct"),
        ("Reject", "Accept", "Wrong"),
        ("I'm not sure...", "Accept", "Unparseable"),
    ]

    for prediction, ground_truth, description in custom_cases:
        reward = grpo_custom.reward_function(prediction, ground_truth)
        print(f"  {description}: reward={reward:.2f}")


def test_config_generation(grpo_mode=None, data_info=None):
    """Test config generation for GRPO training."""
    print("\n" + "=" * 60)
    print("TEST: GRPO Config Generation")
    print("=" * 60)

    if grpo_mode is None or data_info is None:
        grpo_mode, data_info = test_data_preparation()

    # Generate config
    config_path = grpo_mode.generate_config(data_info, OUTPUT_DIR)
    print(f"\nGenerated config: {config_path}")

    # Show config content
    if config_path.exists():
        with open(config_path) as f:
            content = f.read()
        print(f"\n--- Config Content ---\n{content}")

    return config_path


def test_training(config_path=None):
    """Test actual GRPO training."""
    print("\n" + "=" * 60)
    print("TEST: GRPO Training")
    print("=" * 60)

    if config_path is None:
        grpo_mode, data_info = test_data_preparation()
        config_path = test_config_generation(grpo_mode, data_info)

    # Check available training backends
    easyr1_available = False
    llamafactory_available = False

    try:
        from easyr1.train import train_grpo
        easyr1_available = True
        print("EasyR1 found")
    except ImportError:
        print("EasyR1 not installed")

    try:
        from llamafactory.train.tuner import run_exp
        llamafactory_available = True
        print("LlamaFactory found")
    except ImportError:
        print("LlamaFactory not installed")

    if not easyr1_available and not llamafactory_available:
        print("\nERROR: No training backend available")
        print("Install one of:")
        print("  - pip install easyr1")
        print("  - pip install llamafactory")
        return None

    # Create GRPO mode and run training
    submissions = load_submissions(NUM_TRAIN_SAMPLES)
    formatter = TextOnlyFormatter(include_reviews=True)
    handler = BinaryAcceptRejectHandler()
    config = TrainingConfig(
        model_name_or_path=TEXT_MODEL,
        model_type=ModelType.TEXT_ONLY,
        num_epochs=1,
        batch_size=2,
        grpo_group_size=4,
    )

    grpo_mode = GRPOMode(
        config=config,
        input_formatter=formatter,
        output_handler=handler,
    )

    print("\nStarting GRPO training...")
    result = grpo_mode.train(config_path)

    print(f"\n--- Training Result ---")
    for key, value in result.items():
        print(f"  {key}: {value}")

    return result


def test_inference(adapter_path: str = None):
    """Test inference with GRPO-trained adapter."""
    print("\n" + "=" * 60)
    print("TEST: GRPO Inference")
    print("=" * 60)

    if adapter_path is None:
        adapter_path = str(OUTPUT_DIR / "grpo_checkpoints")
        if not Path(adapter_path).exists():
            print(f"No adapter found at {adapter_path}")
            print("Run training first or specify --adapter-path")
            return

    print(f"Using adapter: {adapter_path}")

    # Load eval data
    submissions = load_submissions(NUM_EVAL_SAMPLES)
    print(f"Loaded {len(submissions)} eval submissions")

    # Create predictor with GRPO mode
    predictor = UnifiedPredictor.create(
        input_modality=InputModality.TEXT_ONLY,
        output_type=OutputType.BINARY,
        training_mode=TrainingMode.GRPO,
        config=PredictorConfig(
            model_name_or_path=TEXT_MODEL,
            model_type=ModelType.TEXT_ONLY,
            adapter_path=adapter_path,
        ),
        input_kwargs={"include_reviews": True},
    )

    # Run prediction
    results = predictor.predict(submissions)

    print("\n--- Predictions ---")
    for sub, res in zip(submissions, results):
        gt = sub.get_binary_label()
        pred = res.get("prediction")
        correct = "✓" if pred == gt else "✗"
        print(f"  {sub.submission_id}: pred={pred}, gt={gt} {correct}")

    correct = sum(1 for s, r in zip(submissions, results)
                  if r.get("prediction") == s.get_binary_label())
    print(f"\nAccuracy: {correct}/{len(submissions)} = {correct/len(submissions):.2%}")


def test_comparison_sft_vs_grpo():
    """Compare SFT and GRPO data formats."""
    print("\n" + "=" * 60)
    print("TEST: SFT vs GRPO Data Format Comparison")
    print("=" * 60)

    from lib.llamafactory.training.sft import SFTLoRAMode

    submissions = load_submissions(3)
    formatter = TextOnlyFormatter(include_reviews=True)
    handler = BinaryAcceptRejectHandler()
    config = TrainingConfig(model_name_or_path=TEXT_MODEL)

    # Create both modes
    sft_mode = SFTLoRAMode(config=config, input_formatter=formatter, output_handler=handler)
    grpo_mode = GRPOMode(config=config, input_formatter=formatter, output_handler=handler)

    # Format samples
    samples = sft_mode.format_samples(submissions, include_label=True)

    # Prepare data for both
    sft_data_info = sft_mode.prepare_data(samples, OUTPUT_DIR / "compare_sft")
    grpo_data_info = grpo_mode.prepare_data(samples, OUTPUT_DIR / "compare_grpo")

    # Load and compare
    print("\n--- SFT Data Format (ShareGPT) ---")
    with open(sft_data_info["data_file"]) as f:
        sft_data = json.load(f)
    print(f"Keys: {list(sft_data[0].keys())}")
    print(f"Sample: {json.dumps(sft_data[0], indent=2)[:500]}...")

    print("\n--- GRPO Data Format ---")
    with open(grpo_data_info["data_file"]) as f:
        grpo_data = json.load(f)
    print(f"Keys: {list(grpo_data[0].keys())}")
    print(f"Sample: {json.dumps(grpo_data[0], indent=2)[:500]}...")


def main():
    parser = argparse.ArgumentParser(description="Test GRPO training")
    parser.add_argument(
        "--test",
        type=str,
        choices=["prep", "reward", "config", "train", "inference", "compare", "all"],
        default="prep",
        help="Which test to run",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to trained adapter for inference test",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GRPO TRAINING TESTS")
    print("=" * 60)
    print(f"Dataset: {LOCAL_DATASET_PATH}")
    print(f"Model: {TEXT_MODEL}")
    print(f"Output: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grpo_mode = None
    data_info = None

    if args.test in ["prep", "all"]:
        grpo_mode, data_info = test_data_preparation()

    if args.test in ["reward", "all"]:
        test_reward_functions()

    if args.test in ["config", "all"]:
        if grpo_mode is None:
            grpo_mode, data_info = test_data_preparation()
        test_config_generation(grpo_mode, data_info)

    if args.test in ["train", "all"]:
        test_training()

    if args.test in ["inference", "all"]:
        test_inference(args.adapter_path)

    if args.test in ["compare", "all"]:
        test_comparison_sft_vs_grpo()

    print("\n" + "=" * 60)
    print("TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
