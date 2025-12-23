"""
LlamaFactory Training and Prediction Infrastructure.

A flexible, composable infrastructure for training and predicting
paper outcomes using LlamaFactory with Qwen models.

Example usage:
    from lib.llamafactory import (
        ICLRDataLoader,
        TextWithImagesFormatter,
        BinaryAcceptRejectHandler,
        SFTLoRAMode,
        UnifiedPredictor,
        PredictorConfig,
        TrainingConfig,
        InputModality,
        OutputType,
        TrainingMode,
        ModelType,
    )

    # Load data
    loader = ICLRDataLoader(data_dir=Path("data"))
    submissions = list(loader.load_from_csv(Path("data.csv")))

    # Create predictor
    predictor = UnifiedPredictor.create(
        input_modality=InputModality.TEXT_WITH_IMAGES,
        output_type=OutputType.BINARY,
        training_mode=TrainingMode.SFT_LORA,
        config=PredictorConfig(
            model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
            model_type=ModelType.VISION_LANGUAGE,
        ),
    )

    # Run predictions
    results = predictor.predict(submissions)
"""

# Core types and enums
from .core.types import (
    InputModality,
    OutputType,
    TrainingMode,
    ModelType,
    DataFormat,
    ReviewData,
    SubmissionData,
    FormattedSample,
    TrainingConfig,
    InferenceConfig,
    PredictorConfig,
)

# Registry functions
from .core.registry import (
    register_input_formatter,
    register_output_handler,
    register_training_mode,
    get_input_formatter,
    get_output_handler,
    get_training_mode,
    list_registered_components,
    create_input_formatter,
    create_output_handler,
    create_training_mode,
)

# Data loading and formatting
from .data.loaders import ICLRDataLoader
from .data.formatters import (
    AlpacaFormatter,
    ShareGPTFormatter,
    get_formatter,
    save_train_eval_split,
)

# Input formatters
from .inputs.base import InputFormatter
from .inputs.text_only import (
    TextOnlyFormatter,
    AbstractOnlyFormatter,
    AbstractWithReviewsFormatter,
    FullTextFormatter,
    FullContextFormatter,
)
from .inputs.text_with_images import (
    TextWithImagesFormatter,
    AbstractWithPageImagesFormatter,
    ReviewsWithPageImagesFormatter,
    KeyPagesFormatter,
)
from .inputs.images_only import (
    ImagesOnlyFormatter,
    ImagesWithTitleFormatter,
)

# Output handlers
from .outputs.base import OutputHandler
from .outputs.binary import BinaryAcceptRejectHandler
from .outputs.multiclass import MultiClassDecisionHandler
from .outputs.citation import CitationPercentileHandler
from .outputs.rating import MeanRatingHandler

# Training modes
from .training.base import TrainingModeBase
from .training.zero_shot import ZeroShotMode
from .training.sft import SFTLoRAMode
from .training.grpo import GRPOMode

# Config generation
from .configs.generator import (
    ConfigGenerator,
    generate_dataset_info,
    create_experiment_configs,
)

# Inference
from .inference.predictor import UnifiedPredictor, BatchPredictor

__all__ = [
    # Core types
    "InputModality",
    "OutputType",
    "TrainingMode",
    "ModelType",
    "DataFormat",
    "ReviewData",
    "SubmissionData",
    "FormattedSample",
    "TrainingConfig",
    "InferenceConfig",
    "PredictorConfig",
    # Registry
    "register_input_formatter",
    "register_output_handler",
    "register_training_mode",
    "get_input_formatter",
    "get_output_handler",
    "get_training_mode",
    "list_registered_components",
    "create_input_formatter",
    "create_output_handler",
    "create_training_mode",
    # Data
    "ICLRDataLoader",
    "AlpacaFormatter",
    "ShareGPTFormatter",
    "get_formatter",
    "save_train_eval_split",
    # Input formatters
    "InputFormatter",
    "TextOnlyFormatter",
    "AbstractOnlyFormatter",
    "AbstractWithReviewsFormatter",
    "FullTextFormatter",
    "FullContextFormatter",
    "TextWithImagesFormatter",
    "AbstractWithPageImagesFormatter",
    "ReviewsWithPageImagesFormatter",
    "KeyPagesFormatter",
    "ImagesOnlyFormatter",
    "ImagesWithTitleFormatter",
    # Output handlers
    "OutputHandler",
    "BinaryAcceptRejectHandler",
    "MultiClassDecisionHandler",
    "CitationPercentileHandler",
    "MeanRatingHandler",
    # Training modes
    "TrainingModeBase",
    "ZeroShotMode",
    "SFTLoRAMode",
    "GRPOMode",
    # Config
    "ConfigGenerator",
    "generate_dataset_info",
    "create_experiment_configs",
    # Inference
    "UnifiedPredictor",
    "BatchPredictor",
]
