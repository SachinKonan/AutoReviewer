"""
Type definitions for LlamaFactory training and prediction infrastructure.

This module defines enums, dataclasses, and type aliases used throughout
the llamafactory package.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class InputModality(Enum):
    """Input content types for model input."""
    TEXT_ONLY = auto()           # Abstract, reviews, markdown - no images
    TEXT_WITH_IMAGES = auto()    # Text content + PDF page images
    IMAGES_ONLY = auto()         # PDF page images only


class OutputType(Enum):
    """Prediction target types."""
    BINARY = auto()              # Accept/Reject
    MULTICLASS = auto()          # Accept/Reject/Oral/Spotlight
    CITATION_PERCENTILE = auto() # Citation count percentile [0, 1]
    MEAN_RATING = auto()         # Mean rating [1-10] or percentile


class TrainingMode(Enum):
    """Training/inference modes."""
    ZERO_SHOT = auto()           # No training, direct prediction
    SFT_LORA = auto()            # Supervised Fine-Tuning with LoRA
    GRPO = auto()                # Group Relative Policy Optimization


class ModelType(Enum):
    """Model architecture types."""
    TEXT_ONLY = auto()           # Qwen2.5-7B-Instruct
    VISION_LANGUAGE = auto()     # Qwen2.5-VL-7B-Instruct


class DataFormat(Enum):
    """LlamaFactory data formats."""
    ALPACA = auto()
    SHAREGPT = auto()


# Model name mappings
MODEL_NAME_MAP = {
    ModelType.TEXT_ONLY: "Qwen/Qwen2.5-7B-Instruct",
    ModelType.VISION_LANGUAGE: "Qwen/Qwen2.5-VL-7B-Instruct",
}

TEMPLATE_MAP = {
    ModelType.TEXT_ONLY: "qwen",
    ModelType.VISION_LANGUAGE: "qwen2_vl",
}


@dataclass
class ReviewData:
    """Container for normalized review attributes."""
    summary: Optional[str] = None
    strengths: Optional[str] = None
    weaknesses: Optional[str] = None
    questions: Optional[List[str]] = None
    rating: Optional[float] = None
    confidence: Optional[float] = None
    # Raw fields for fallback
    raw_review: Optional[Dict[str, Any]] = None

    @classmethod
    def from_normalized(cls, normalized_dict: Dict[str, Any]) -> "ReviewData":
        """Create ReviewData from LLMUniversalReview dict."""
        return cls(
            summary=normalized_dict.get("summary"),
            strengths=normalized_dict.get("strengths"),
            weaknesses=normalized_dict.get("weaknesses"),
            questions=normalized_dict.get("questions", []),
            rating=normalized_dict.get("rating"),
            confidence=normalized_dict.get("confidence"),
        )

    @classmethod
    def from_raw(cls, raw_dict: Dict[str, Any]) -> "ReviewData":
        """Create ReviewData from raw review dict."""
        # Try to extract rating from various field names
        rating = None
        for key in ["rating", "recommendation", "score"]:
            if key in raw_dict:
                val = raw_dict[key]
                if isinstance(val, (int, float)):
                    rating = float(val)
                elif isinstance(val, str):
                    # Extract numeric prefix (e.g., "5: marginally below...")
                    import re
                    match = re.match(r'^(\d+)', val.strip())
                    if match:
                        rating = float(match.group(1))
                break

        confidence = None
        if "confidence" in raw_dict:
            val = raw_dict["confidence"]
            if isinstance(val, (int, float)):
                confidence = float(val)
            elif isinstance(val, str):
                import re
                match = re.match(r'^(\d+)', val.strip())
                if match:
                    confidence = float(match.group(1))

        return cls(
            rating=rating,
            confidence=confidence,
            raw_review=raw_dict,
        )


@dataclass
class SubmissionData:
    """Container for submission data."""
    submission_id: str
    year: int
    title: str
    abstract: str
    decision: Optional[str] = None
    # Optional content
    clean_md: Optional[str] = None
    reviews: List[ReviewData] = field(default_factory=list)
    meta_review: Optional[Dict[str, Any]] = None
    # Image paths for VL models
    pdf_image_paths: Optional[List[Path]] = None
    # Ground truth labels for different tasks
    labels: Dict[str, Any] = field(default_factory=dict)

    def get_binary_label(self) -> str:
        """Get binary accept/reject label from decision."""
        if self.decision and "Accept" in self.decision:
            return "Accept"
        return "Reject"

    def get_multiclass_label(self) -> str:
        """Get multi-class decision label."""
        if not self.decision:
            return "Reject"
        decision = self.decision
        if "Oral" in decision:
            return "Accept (Oral)"
        if "Spotlight" in decision:
            return "Accept (Spotlight)"
        if "Poster" in decision or "Accept" in decision:
            return "Accept (Poster)"
        return "Reject"

    def get_mean_rating(self) -> Optional[float]:
        """Calculate mean rating from reviews."""
        ratings = [r.rating for r in self.reviews if r.rating is not None]
        if not ratings:
            return None
        return sum(ratings) / len(ratings)


@dataclass
class FormattedSample:
    """A formatted sample ready for LlamaFactory conversion."""
    # Core content
    instruction: str
    input_text: str
    output: Optional[str] = None  # None for inference
    # Multimodal content
    images: Optional[List[str]] = None
    # Metadata
    submission_id: str = ""
    year: int = 0
    output_type: Optional[OutputType] = None

    def to_alpaca(self) -> Dict[str, Any]:
        """Convert to Alpaca format."""
        item = {
            "instruction": self.instruction,
            "input": self.input_text,
            "output": self.output or "",
        }
        if self.images:
            item["images"] = self.images
        return item

    def to_sharegpt(self) -> Dict[str, Any]:
        """Convert to ShareGPT format."""
        conversations = [
            {
                "from": "human",
                "value": f"{self.instruction}\n\n{self.input_text}",
            },
        ]
        if self.output:
            conversations.append({
                "from": "gpt",
                "value": self.output,
            })

        item = {"conversations": conversations}
        if self.images:
            item["images"] = self.images
        return item


@dataclass
class TrainingConfig:
    """Configuration for training runs."""
    # Model settings
    model_name_or_path: str
    model_type: ModelType = ModelType.TEXT_ONLY

    # Training hyperparameters
    learning_rate: float = 1e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1

    # LoRA settings
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    # GRPO settings
    grpo_group_size: int = 4
    grpo_kl_coeff: float = 0.1

    # Data settings
    max_length: int = 4096
    val_size: float = 0.1

    # Output settings
    output_dir: str = "outputs"
    logging_steps: int = 10
    save_steps: int = 500

    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True

    def get_model_name(self) -> str:
        """Get default model name for model type if not specified."""
        if self.model_name_or_path:
            return self.model_name_or_path
        return MODEL_NAME_MAP.get(self.model_type, MODEL_NAME_MAP[ModelType.TEXT_ONLY])

    def get_template(self) -> str:
        """Get template name for model type."""
        return TEMPLATE_MAP.get(self.model_type, "qwen")


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    model_name_or_path: str
    adapter_path: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512
    batch_size: int = 32
    # vLLM settings
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9


@dataclass
class PredictorConfig:
    """Configuration for unified predictor."""
    model_name_or_path: str
    model_type: ModelType = ModelType.TEXT_ONLY
    adapter_path: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512
    batch_size: int = 32

    def to_inference_config(self) -> InferenceConfig:
        """Convert to InferenceConfig."""
        return InferenceConfig(
            model_name_or_path=self.model_name_or_path,
            adapter_path=self.adapter_path,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            batch_size=self.batch_size,
        )
