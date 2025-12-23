# LlamaFactory Training & Prediction Infrastructure

A flexible, composable infrastructure for training and predicting paper outcomes using LlamaFactory with Qwen models.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Input Modalities](#input-modalities)
5. [Output Types](#output-types)
6. [Training Modes](#training-modes)
7. [Complete Examples](#complete-examples)
8. [Configuration Reference](#configuration-reference)

---

## Installation

The core infrastructure works without LlamaFactory for zero-shot inference.
For SFT/GRPO training, install LlamaFactory separately due to version conflicts.

### Step 1: Sync main dependencies

```bash
uv sync
```

### Step 2: Install LlamaFactory (for training only)

Due to version conflicts with numpy/peft, install LlamaFactory separately:

```bash
# Option A: Install with --no-deps and add compatible versions
pip install llamafactory==0.9.3 --no-deps
pip install trl>=0.8.6,<=0.9.6

# Option B: Use a separate virtual environment for training
uv venv .venv-train
source .venv-train/bin/activate
pip install llamafactory[torch,vllm]
```

### What works without LlamaFactory

- **Zero-shot prediction**: Uses vLLM directly
- **Data preparation**: Creates LlamaFactory-compatible JSON files
- **Config generation**: Creates YAML configs for `llamafactory-cli`

### What requires LlamaFactory

- **SFT training via Python API**: `SFTLoRAMode.train()`
- **GRPO training via Python API**: `GRPOMode.train()`

You can always use the generated configs with `llamafactory-cli` directly:

```bash
llamafactory-cli train outputs/sft_config.yaml
```

---

## Quick Start

```python
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
submissions = list(loader.load_from_huggingface(
    dataset_name="skonan/iclr-reviews-2020-2026",
    split="2024",  # or "all", "2020", "2021", etc.
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
```

---

## Core Concepts

The infrastructure is built around three composable components:

| Component | Purpose | Examples |
|-----------|---------|----------|
| **InputFormatter** | How to format paper content | Text-only, Text+Images, Images-only |
| **OutputHandler** | What to predict | Binary, Multi-class, Citation, Rating |
| **TrainingMode** | How to train/predict | Zero-shot, SFT, GRPO |

These can be freely combined. For example:
- Text-only input + Binary output + SFT training
- Images-only input + Citation percentile + GRPO training
- Text+Images input + Multi-class output + Zero-shot prediction

---

## Input Modalities

### 1. Text-Only (TEXT_ONLY)

Uses only textual content. Works with `Qwen/Qwen2.5-7B-Instruct`.

```python
from lib.llamafactory import (
    TextOnlyFormatter,
    AbstractOnlyFormatter,
    AbstractWithReviewsFormatter,
    FullTextFormatter,
    FullContextFormatter,
)

# Option A: Abstract only (minimal)
formatter = AbstractOnlyFormatter()

# Option B: Abstract + reviews (default)
formatter = TextOnlyFormatter(
    include_reviews=True,
    include_markdown=False,
    use_normalized_reviews=True,  # Use LLMUniversalReview format
)

# Option C: Abstract + reviews + full paper markdown
formatter = FullContextFormatter(max_tokens=32000)

# Option D: Full paper without reviews
formatter = FullTextFormatter(max_tokens=16000)
```

**Few-shot prompting:**
```python
formatter = TextOnlyFormatter(
    include_reviews=True,
    few_shot_examples=[
        {"title": "Paper A", "abstract": "...", "decision": "Accept"},
        {"title": "Paper B", "abstract": "...", "decision": "Reject"},
    ],
    few_shot_count=2,
)
```

### 2. Text with Images (TEXT_WITH_IMAGES)

Combines text with PDF page images. Requires `Qwen/Qwen2.5-VL-7B-Instruct`.

```python
from lib.llamafactory import (
    TextWithImagesFormatter,
    AbstractWithPageImagesFormatter,
    ReviewsWithPageImagesFormatter,
    KeyPagesFormatter,
)

# Option A: Abstract + all page images
formatter = TextWithImagesFormatter(
    include_reviews=True,
    max_images=20,
    image_position="after_abstract",  # or "after_reviews", "before_text"
)

# Option B: Abstract + page images (no reviews)
formatter = AbstractWithPageImagesFormatter(max_images=15)

# Option C: Key pages only (first 3 + last 2)
formatter = KeyPagesFormatter(first_pages=3, last_pages=2)
```

### 3. Images Only (IMAGES_ONLY)

Uses only PDF page images. Model extracts all info from images.

```python
from lib.llamafactory import ImagesOnlyFormatter, ImagesWithTitleFormatter

# Option A: Pure images
formatter = ImagesOnlyFormatter(max_images=20)

# Option B: Title + images
formatter = ImagesWithTitleFormatter(max_images=20)
```

---

## Output Types

### 1. Binary Accept/Reject (BINARY)

```python
from lib.llamafactory import BinaryAcceptRejectHandler

handler = BinaryAcceptRejectHandler(include_reasoning=True)

# Predictions return:
# {"prediction": "Accept" or "Reject", "reasoning": "...", "parse_success": True}
```

### 2. Multi-class Decision (MULTICLASS)

Predicts: Reject, Accept (Poster), Accept (Spotlight), Accept (Oral)

```python
from lib.llamafactory import MultiClassDecisionHandler

handler = MultiClassDecisionHandler(include_reasoning=True)

# Predictions return:
# {"prediction": "Accept (Poster)", "reasoning": "...", "parse_success": True}
```

### 3. Citation Percentile (CITATION_PERCENTILE)

Predicts citation impact as percentile [0.0, 1.0].

```python
from lib.llamafactory import CitationPercentileHandler

handler = CitationPercentileHandler(
    include_reasoning=True,
    discretize=False,  # Set True to round to bins
)

# Predictions return:
# {"prediction": 0.75, "reasoning": "...", "parse_success": True}
```

### 4. Mean Rating (MEAN_RATING)

Predicts expected mean reviewer rating.

```python
from lib.llamafactory import MeanRatingHandler

# Option A: Raw rating [1-10]
handler = MeanRatingHandler(as_percentile=False)

# Option B: Rating as percentile [0-1]
handler = MeanRatingHandler(as_percentile=True)

# Predictions return:
# {"prediction": 6.5, "reasoning": "...", "parse_success": True}
```

---

## Training Modes

### 1. Zero-Shot Prediction

No training required. Uses base model directly.

```python
from lib.llamafactory import ZeroShotMode, TrainingConfig

config = TrainingConfig(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    model_type=ModelType.TEXT_ONLY,
)

mode = ZeroShotMode(
    config=config,
    input_formatter=formatter,
    output_handler=handler,
    use_vllm=True,  # Use vLLM for faster inference
)

# Predict directly
samples = mode.format_samples(submissions, include_label=False)
predictions = mode.predict(samples, InferenceConfig(
    model_name_or_path=config.model_name_or_path,
    temperature=0.0,
    max_tokens=512,
))
```

### 2. Supervised Fine-Tuning (SFT) with LoRA

```python
from lib.llamafactory import SFTLoRAMode, TrainingConfig, DataFormat

config = TrainingConfig(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    model_type=ModelType.TEXT_ONLY,
    # Training hyperparameters
    learning_rate=1e-4,
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=8,
    # LoRA settings
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.05,
    # Output
    output_dir="outputs/sft_experiment",
)

mode = SFTLoRAMode(
    config=config,
    input_formatter=formatter,
    output_handler=handler,
    data_format=DataFormat.SHAREGPT,  # or DataFormat.ALPACA
)

# Prepare data
train_samples = mode.format_samples(train_submissions, include_label=True)
data_info = mode.prepare_data(train_samples, Path("data/llamafactory/train"))

# Generate config and train
config_path = mode.generate_config(data_info, Path("outputs/sft_experiment"))
mode.train(config_path)

# Inference with trained adapter
predictions = mode.predict(test_samples, InferenceConfig(
    model_name_or_path=config.model_name_or_path,
    adapter_path="outputs/sft_experiment/checkpoints",
))
```

### 3. GRPO (Group Relative Policy Optimization)

```python
from lib.llamafactory import GRPOMode, TrainingConfig

config = TrainingConfig(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    model_type=ModelType.TEXT_ONLY,
    learning_rate=1e-5,
    num_epochs=1,
    # GRPO-specific
    grpo_group_size=4,
    grpo_kl_coeff=0.1,
)

# Custom reward function (optional)
def custom_reward(prediction: str, ground_truth: str) -> float:
    # Return 1.0 for correct, 0.0 for incorrect
    parsed = handler.parse_prediction(prediction)
    return 1.0 if parsed["prediction"] == ground_truth else 0.0

mode = GRPOMode(
    config=config,
    input_formatter=formatter,
    output_handler=handler,
    reward_function=custom_reward,  # Optional, has sensible default
)

# Train
train_samples = mode.format_samples(train_submissions, include_label=True)
data_info = mode.prepare_data(train_samples, Path("data/llamafactory/grpo"))
config_path = mode.generate_config(data_info, Path("outputs/grpo_experiment"))
mode.train(config_path)
```

---

## Complete Examples

### Example 1: Zero-Shot Binary Classification with Text+Images

```python
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

# Load data
loader = ICLRDataLoader(
    data_dir=Path("data/full_run"),
    image_base_dir=Path("data/full_run/normalized"),
)
submissions = list(loader.load_from_csv(Path("data/iclr_data.csv")))

# Split data
train_submissions = [s for s in submissions if s.year < 2024]
test_submissions = [s for s in submissions if s.year >= 2024]

# Create predictor
predictor = UnifiedPredictor.create(
    input_modality=InputModality.TEXT_WITH_IMAGES,
    output_type=OutputType.BINARY,
    training_mode=TrainingMode.ZERO_SHOT,
    config=PredictorConfig(
        model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
        model_type=ModelType.VISION_LANGUAGE,
        temperature=0.0,
        max_tokens=512,
    ),
    input_kwargs={"include_reviews": True, "max_images": 10},
)

# Evaluate
results = predictor.predict_with_evaluation(test_submissions)
print(f"Accuracy: {results['metrics']['accuracy']:.3f}")
print(f"F1: {results['metrics']['f1']:.3f}")
```

### Example 2: SFT Training for Multi-class Prediction

```python
from pathlib import Path
from lib.llamafactory import (
    ICLRDataLoader,
    TextOnlyFormatter,
    MultiClassDecisionHandler,
    SFTLoRAMode,
    TrainingConfig,
    ModelType,
    InferenceConfig,
)

# Load data
loader = ICLRDataLoader(data_dir=Path("data/full_run"))
submissions = list(loader.load_from_csv(Path("data/iclr_data.csv")))

train_subs = [s for s in submissions if s.year < 2024]
test_subs = [s for s in submissions if s.year >= 2024]

# Configure components
formatter = TextOnlyFormatter(include_reviews=True)
handler = MultiClassDecisionHandler()

config = TrainingConfig(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    model_type=ModelType.TEXT_ONLY,
    learning_rate=1e-4,
    num_epochs=3,
    batch_size=4,
    lora_rank=16,
)

# Create training mode
sft = SFTLoRAMode(config, formatter, handler)

# Prepare and train
train_samples = sft.format_samples(train_subs, include_label=True)
data_info = sft.prepare_data(train_samples, Path("data/sft_multiclass"))
config_path = sft.generate_config(data_info, Path("outputs/sft_multiclass"))

print(f"Training config saved to: {config_path}")
print(f"Run: llamafactory-cli train {config_path}")

# Or train via Python API
sft.train(config_path)

# Evaluate
test_samples = sft.format_samples(test_subs, include_label=False)
predictions = sft.predict(test_samples, InferenceConfig(
    model_name_or_path=config.model_name_or_path,
    adapter_path="outputs/sft_multiclass/checkpoints",
))

# Compute metrics
ground_truths = [s.get_multiclass_label() for s in test_subs]
metrics = handler.compute_metrics(predictions, ground_truths)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Macro F1: {metrics['macro_f1']:.3f}")
```

### Example 3: GRPO Training for Citation Prediction

```python
from pathlib import Path
from lib.llamafactory import (
    ICLRDataLoader,
    TextWithImagesFormatter,
    CitationPercentileHandler,
    GRPOMode,
    TrainingConfig,
    ModelType,
)

# Load data with citation labels
loader = ICLRDataLoader(
    data_dir=Path("data/full_run"),
    image_base_dir=Path("data/full_run/normalized"),
)
submissions = list(loader.load_from_csv(Path("data/iclr_with_citations.csv")))

# Filter to submissions with citation data
submissions = [s for s in submissions if "citation_percentile" in s.labels]

# Configure
formatter = TextWithImagesFormatter(include_reviews=True, max_images=5)
handler = CitationPercentileHandler()

config = TrainingConfig(
    model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
    model_type=ModelType.VISION_LANGUAGE,
    learning_rate=5e-6,
    num_epochs=1,
    grpo_group_size=4,
    grpo_kl_coeff=0.05,
)

# Custom reward: closer predictions get higher rewards
def proximity_reward(prediction: str, ground_truth: str) -> float:
    parsed = handler.parse_prediction(prediction)
    pred = parsed.get("prediction", 0.5)
    gt = float(ground_truth.replace("Answer: ", ""))
    return max(0.0, 1.0 - abs(pred - gt) * 2)

grpo = GRPOMode(config, formatter, handler, reward_function=proximity_reward)

# Train
samples = grpo.format_samples(submissions, include_label=True)
data_info = grpo.prepare_data(samples, Path("data/grpo_citation"))
config_path = grpo.generate_config(data_info, Path("outputs/grpo_citation"))
grpo.train(config_path)
```

### Example 4: Batch Prediction with Progress

```python
from lib.llamafactory import UnifiedPredictor, BatchPredictor, PredictorConfig

predictor = UnifiedPredictor.create(
    input_modality=InputModality.TEXT_ONLY,
    output_type=OutputType.BINARY,
    training_mode=TrainingMode.ZERO_SHOT,
    config=PredictorConfig(model_name_or_path="Qwen/Qwen2.5-7B-Instruct"),
)

batch_predictor = BatchPredictor(
    predictor=predictor,
    batch_size=32,
    show_progress=True,
)

# Predict and save results
output_path = batch_predictor.predict_and_save(
    submissions=all_submissions,
    output_path=Path("results/predictions.json"),
    include_metrics=True,
)
print(f"Results saved to: {output_path}")
```

---

## Configuration Reference

### TrainingConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name_or_path` | required | Model name or path |
| `model_type` | `TEXT_ONLY` | `TEXT_ONLY` or `VISION_LANGUAGE` |
| `learning_rate` | `1e-4` | Learning rate |
| `num_epochs` | `3` | Number of training epochs |
| `batch_size` | `4` | Per-device batch size |
| `gradient_accumulation_steps` | `8` | Gradient accumulation |
| `lora_rank` | `8` | LoRA rank |
| `lora_alpha` | `16` | LoRA alpha |
| `lora_dropout` | `0.05` | LoRA dropout |
| `max_length` | `4096` | Maximum sequence length |
| `val_size` | `0.1` | Validation split ratio |
| `grpo_group_size` | `4` | GRPO group size |
| `grpo_kl_coeff` | `0.1` | GRPO KL coefficient |

### PredictorConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name_or_path` | required | Model name or path |
| `model_type` | `TEXT_ONLY` | Model type |
| `adapter_path` | `None` | Path to LoRA adapter |
| `temperature` | `0.0` | Sampling temperature |
| `max_tokens` | `512` | Maximum generated tokens |
| `batch_size` | `32` | Inference batch size |

### Model Names

| ModelType | Default Model |
|-----------|---------------|
| `TEXT_ONLY` | `Qwen/Qwen2.5-7B-Instruct` |
| `VISION_LANGUAGE` | `Qwen/Qwen2.5-VL-7B-Instruct` |

---

## CSV Schema

Expected columns in input CSV:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `submission_id` | str | Yes | Unique identifier |
| `year` | int | Yes | Conference year |
| `title` | str | Yes | Paper title |
| `original_abstract` | str | Yes | Paper abstract |
| `decision` | str | For training | Final decision |
| `clean_md` | str | Optional | Cleaned markdown content |
| `standardized_reviews` | JSON | Optional | Review data |
| `citation_percentile` | float | For citation prediction | Citation percentile |

---

## Extending the Infrastructure

### Adding a New Input Formatter

```python
from lib.llamafactory import InputFormatter, register_input_formatter, InputModality

@register_input_formatter(InputModality.TEXT_ONLY)  # or create new enum
class MyCustomFormatter(InputFormatter):
    modality = InputModality.TEXT_ONLY
    requires_vl_model = False

    def format_content(self, submission):
        # Custom formatting logic
        return f"Title: {submission.title}\nAbstract: {submission.abstract}"

    def get_images(self, submission):
        return None
```

### Adding a New Output Handler

```python
from lib.llamafactory import OutputHandler, register_output_handler, OutputType

@register_output_handler(OutputType.BINARY)  # or create new enum
class MyCustomHandler(OutputHandler):
    output_type = OutputType.BINARY

    def get_instruction(self):
        return "Predict accept or reject."

    def format_label(self, submission):
        return submission.get_binary_label()

    def parse_prediction(self, text):
        # Parse model output
        return {"prediction": "Accept", "raw": text, "parse_success": True}

    def compute_metrics(self, predictions, ground_truths):
        # Compute evaluation metrics
        return {"accuracy": 0.85}
```
