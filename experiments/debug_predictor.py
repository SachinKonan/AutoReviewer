from lib.llamafactory import (
    ICLRDataLoader,
    UnifiedPredictor,
    PredictorConfig,
    InputModality,
    OutputType,
    TrainingMode,
    ModelType,
)
from lib.llamafactory.training.base import build_prompt

# Load a few submissions from all splits
loader = ICLRDataLoader()
subs = []
for i, s in enumerate(loader.load_from_huggingface(split="all", load_images=False)):
    subs.append(s)
    if i >= 4:
        break

print(f"Loaded {len(subs)} submissions (sample IDs: {[s.submission_id for s in subs]})")

# Create predictor but avoid model init; just format samples
predictor = UnifiedPredictor.create(
    input_modality=InputModality.TEXT_ONLY,
    output_type=OutputType.BINARY,
    training_mode=TrainingMode.ZERO_SHOT,
    config=PredictorConfig(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        model_type=ModelType.TEXT_ONLY,
    ),
)

samples = predictor.format_submissions(subs, include_label=False)
print(f"Formatted {len(samples)} samples")
for i, samp in enumerate(samples):
    print(f"--- Sample {i} ---")
    print("submission_id:", samp.submission_id)
    print("year:", samp.year)
    print("instruction non-empty:", bool(samp.instruction and samp.instruction.strip()))
    print("input_text non-empty:", bool(samp.input_text and samp.input_text.strip()))
    print("built prompt (first 300 chars):\n", build_prompt(samp)[:300])
