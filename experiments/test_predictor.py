
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

# 1. Load data from HuggingFace (recommended)
loader = ICLRDataLoader()
splits = ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]
submissions = list(loader.load_from_huggingface(
    dataset_name="skonan/iclr-reviews-2020-2026",
    split=splits, 
    load_images=True,
))

# 2. Create predictor
predictor = UnifiedPredictor.create(
    input_modality=InputModality.TEXT_ONLY,
    output_type=OutputType.BINARY,
    training_mode=TrainingMode.ZERO_SHOT,
    config=PredictorConfig(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        model_type=ModelType.TEXT_ONLY,
    ),
)

# 3. Run predictions
results = predictor.predict(submissions)