"""
Test SFT (Supervised Fine-Tuning) with LoRA functionality.

Tests:
1. Data preparation (formatting to LlamaFactory format)
2. Config generation (YAML for training)
3. Training execution (requires llamafactory installed)
4. Inference with trained adapter

Usage:
    # Test data prep and config only (no GPU needed)
    python experiments/test_sft_training.py --test prep

    # Run actual training (requires GPU + llamafactory)
    python experiments/test_sft_training.py --test train

    # Test inference with adapter
    python experiments/test_sft_training.py --test inference --adapter-path /path/to/adapter
"""

import argparse
import json
from pathlib import Path
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
from lib.llamafactory.training.sft import SFTLoRAMode
from lib.llamafactory.inputs.text_only import TextOnlyFormatter
from lib.llamafactory.outputs.binary import BinaryAcceptRejectHandler

# Configuration
LOCAL_DATASET_PATH = "/n/fs/vision-mix/jl0796/iclr-reviews-2020-2026"
TEXT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = Path("outputs/sft_test")
NUM_TRAIN_SAMPLES = 20  # Small for testing
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
    """Test data preparation for SFT training."""
    print("\n" + "=" * 60)
    print("TEST: SFT Data Preparation")
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
    )

    # Create SFT mode
    sft_mode = SFTLoRAMode(
        config=config,
        input_formatter=formatter,
        output_handler=handler,
    )

    # Format samples
    samples = sft_mode.format_samples(submissions, include_label=True)
    print(f"\nFormatted {len(samples)} samples")

    # Show sample
    print("\n--- Sample 0 ---")
    print(f"Instruction (first 200 chars): {samples[0].instruction[:200]}...")
    print(f"Input (first 200 chars): {samples[0].input_text[:200]}...")
    print(f"Output: {samples[0].output}")
    print(f"Images: {samples[0].images}")

    # Prepare data
    data_path = OUTPUT_DIR / "data"
    data_info = sft_mode.prepare_data(samples, data_path)

    print(f"\n--- Data Info ---")
    for key, value in data_info.items():
        print(f"  {key}: {value}")

    # Check generated files
    print(f"\n--- Generated Files ---")
    data_file = Path(data_info["data_file"])
    if data_file.exists():
        with open(data_file) as f:
            data = json.load(f)
        print(f"  {data_file.name}: {len(data)} samples")
        print(f"  First sample keys: {list(data[0].keys())}")

    info_file = Path(data_info["dataset_info_file"])
    if info_file.exists():
        with open(info_file) as f:
            info = json.load(f)
        print(f"  {info_file.name}: {json.dumps(info, indent=2)}")

    return sft_mode, data_info


def test_config_generation(sft_mode=None, data_info=None):
    """Test config generation for SFT training."""
    print("\n" + "=" * 60)
    print("TEST: SFT Config Generation")
    print("=" * 60)

    if sft_mode is None or data_info is None:
        sft_mode, data_info = test_data_preparation()

    # Generate config
    config_path = sft_mode.generate_config(data_info, OUTPUT_DIR)
    print(f"\nGenerated config: {config_path}")

    # Show config content
    if config_path.exists():
        with open(config_path) as f:
            content = f.read()
        print(f"\n--- Config Content ---\n{content}")

    return config_path


def test_training(config_path=None):
    """Test actual SFT training (requires llamafactory)."""
    print("\n" + "=" * 60)
    print("TEST: SFT Training")
    print("=" * 60)

    if config_path is None:
        sft_mode, data_info = test_data_preparation()
        config_path = test_config_generation(sft_mode, data_info)

    # Check if llamafactory is available
    try:
        from llamafactory.train.tuner import run_exp
        print("LlamaFactory found, proceeding with training...")
    except ImportError:
        print("ERROR: llamafactory not installed")
        print("Install with: pip install llamafactory")
        return None

    # Load submissions and create mode
    submissions = load_submissions(NUM_TRAIN_SAMPLES)
    formatter = TextOnlyFormatter(include_reviews=True)
    handler = BinaryAcceptRejectHandler()
    config = TrainingConfig(
        model_name_or_path=TEXT_MODEL,
        model_type=ModelType.TEXT_ONLY,
        num_epochs=1,
        batch_size=2,
        learning_rate=1e-4,
    )

    sft_mode = SFTLoRAMode(
        config=config,
        input_formatter=formatter,
        output_handler=handler,
    )

    # Run training
    print("\nStarting training...")
    result = sft_mode.train(config_path)

    print(f"\n--- Training Result ---")
    for key, value in result.items():
        print(f"  {key}: {value}")

    return result


def test_inference(adapter_path: str = None):
    """Test inference with SFT-trained adapter."""
    print("\n" + "=" * 60)
    print("TEST: SFT Inference")
    print("=" * 60)

    if adapter_path is None:
        # Try default checkpoint location
        adapter_path = str(OUTPUT_DIR / "checkpoints")
        if not Path(adapter_path).exists():
            print(f"No adapter found at {adapter_path}")
            print("Run training first or specify --adapter-path")
            return

    print(f"Using adapter: {adapter_path}")

    # Load eval data
    submissions = load_submissions(NUM_EVAL_SAMPLES)
    print(f"Loaded {len(submissions)} eval submissions")

    # Create predictor with SFT mode
    predictor = UnifiedPredictor.create(
        input_modality=InputModality.TEXT_ONLY,
        output_type=OutputType.BINARY,
        training_mode=TrainingMode.SFT_LORA,
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
    for i, (sub, res) in enumerate(zip(submissions, results)):
        gt = sub.get_binary_label()
        pred = res.get("prediction")
        correct = "✓" if pred == gt else "✗"
        print(f"  {sub.submission_id}: pred={pred}, gt={gt} {correct}")

    # Compute accuracy
    correct = sum(1 for s, r in zip(submissions, results)
                  if r.get("prediction") == s.get_binary_label())
    print(f"\nAccuracy: {correct}/{len(submissions)} = {correct/len(submissions):.2%}")


def test_full_pipeline():
    """Test the full SFT pipeline using UnifiedPredictor."""
    print("\n" + "=" * 60)
    print("TEST: Full SFT Pipeline via UnifiedPredictor")
    print("=" * 60)

    # This uses the run_pipeline method
    submissions = load_submissions(NUM_TRAIN_SAMPLES)
    train_subs = submissions[:15]
    eval_subs = submissions[15:]

    print(f"Train: {len(train_subs)}, Eval: {len(eval_subs)}")

    # Create predictor
    predictor = UnifiedPredictor.create(
        input_modality=InputModality.TEXT_ONLY,
        output_type=OutputType.BINARY,
        training_mode=TrainingMode.SFT_LORA,
        config=PredictorConfig(
            model_name_or_path=TEXT_MODEL,
            model_type=ModelType.TEXT_ONLY,
        ),
        input_kwargs={"include_reviews": True},
    )

    # Run full pipeline
    print("\nRunning full training pipeline...")
    result = predictor.training_mode.run_pipeline(
        train_submissions=train_subs,
        eval_submissions=eval_subs,
        output_dir=OUTPUT_DIR / "pipeline",
    )

    print(f"\n--- Pipeline Result ---")
    print(f"Mode: {result.get('mode')}")
    print(f"Data info: {result.get('data_info', {}).get('num_samples')} samples")
    if "train_result" in result:
        print(f"Train status: {result['train_result'].get('status')}")
    if "eval_metrics" in result:
        print(f"Eval metrics: {result['eval_metrics']}")


def main():
    parser = argparse.ArgumentParser(description="Test SFT training")
    parser.add_argument(
        "--test",
        type=str,
        choices=["prep", "config", "train", "inference", "pipeline", "all"],
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
    print("SFT TRAINING TESTS")
    print("=" * 60)
    print(f"Dataset: {LOCAL_DATASET_PATH}")
    print(f"Model: {TEXT_MODEL}")
    print(f"Output: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.test in ["prep", "all"]:
        sft_mode, data_info = test_data_preparation()

    if args.test in ["config", "all"]:
        if args.test == "config":
            sft_mode, data_info = test_data_preparation()
        test_config_generation(sft_mode, data_info)

    if args.test in ["train", "all"]:
        test_training()

    if args.test in ["inference", "all"]:
        test_inference(args.adapter_path)

    if args.test == "pipeline":
        test_full_pipeline()

    print("\n" + "=" * 60)
    print("TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
