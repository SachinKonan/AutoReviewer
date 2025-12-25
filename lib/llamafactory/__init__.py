"""
LlamaFactory Training and Prediction Infrastructure.

A flexible, composable infrastructure for training and predicting
paper outcomes using LlamaFactory with Qwen models.

Example usage:
    from lib.llamafactory import (
        RayDataPredictor,
        InputModality,
        OutputType,
    )

    # Create predictor (uses Ray Data + vLLM)
    predictor = RayDataPredictor(
        input_modality=InputModality.TEXT_ONLY,
        output_type=OutputType.BINARY,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        ray_address="auto",
        include_markdown=True,
    )

    # Run inference (streaming to parquet)
    predictor.predict_from_hf_dataset(
        dataset_path="data/iclr_split",
        output_path="outputs/predictions.parquet",
        split="test",
    )
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
from .inference.ray_predictor import RayDataPredictor

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
    "RayDataPredictor",
]
