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

# 1. Load data from local path (limited to 10 for login node)
LOCAL_DATASET_PATH = "/n/fs/vision-mix/jl0796/iclr-reviews-2020-2026"
loader = ICLRDataLoader()
submissions = []
for i, s in enumerate(loader.load_from_huggingface(
    dataset_name=LOCAL_DATASET_PATH,
    years=["2020"],
    load_images=False,
)):
    submissions.append(s)
    if i >= 9:  # Limit to 10 inputs
        break

# 2. Create predictor
predictor = UnifiedPredictor.create(
    input_modality=InputModality.TEXT_ONLY,
    output_type=OutputType.BINARY,
    training_mode=TrainingMode.ZERO_SHOT,
    config=PredictorConfig(
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        model_type=ModelType.TEXT_ONLY,
    ),
)

# 3. Run predictions
results = predictor.predict(submissions)
for r in results:
    print("Raw:", r.get("raw", "")[:200])
    print("Prediction:", r.get("prediction", "")[:200])
    print("Reasoning:", r.get("reasoning", "")[:200])
    print("---")


# print(results)