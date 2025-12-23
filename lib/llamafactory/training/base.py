"""
Base class for training modes.

Training modes define how to prepare data, configure training, and run inference
for different training strategies (zero-shot, SFT, GRPO).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..core.types import (
    FormattedSample,
    InferenceConfig,
    TrainingConfig,
    TrainingMode,
)

if TYPE_CHECKING:
    from ..inputs.base import InputFormatter
    from ..outputs.base import OutputHandler


class TrainingModeBase(ABC):
    """
    Abstract base class for training/inference modes.

    Subclasses implement specific training strategies:
    - ZeroShot: Direct model prediction without training
    - SFT: Supervised Fine-Tuning with LoRA adapters
    - GRPO: Group Relative Policy Optimization

    Attributes:
        mode: The training mode type
        requires_training: Whether this mode requires a training step
    """

    mode: TrainingMode = NotImplemented
    requires_training: bool = True

    def __init__(
        self,
        config: TrainingConfig,
        input_formatter: "InputFormatter",
        output_handler: "OutputHandler",
    ):
        """
        Initialize training mode.

        Args:
            config: Training configuration
            input_formatter: Input formatting strategy
            output_handler: Output handling strategy
        """
        self.config = config
        self.input_formatter = input_formatter
        self.output_handler = output_handler

    @abstractmethod
    def prepare_data(
        self,
        samples: List[FormattedSample],
        output_path: Path,
    ) -> Dict[str, Any]:
        """
        Prepare data in LlamaFactory format.

        Args:
            samples: Formatted training samples
            output_path: Path to save prepared data

        Returns:
            Dict with data paths and metadata
        """
        pass

    @abstractmethod
    def generate_config(
        self,
        data_info: Dict[str, Any],
        output_path: Path,
    ) -> Optional[Path]:
        """
        Generate LlamaFactory YAML config.

        Args:
            data_info: Data preparation output
            output_path: Path to save config

        Returns:
            Path to generated config file, or None if not applicable
        """
        pass

    @abstractmethod
    def train(
        self,
        config_path: Optional[Path] = None,
        resume_from: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Execute training.

        Args:
            config_path: Path to LlamaFactory config
            resume_from: Optional checkpoint to resume from

        Returns:
            Training results and metrics
        """
        pass

    @abstractmethod
    def predict(
        self,
        samples: List[FormattedSample],
        inference_config: InferenceConfig,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on samples.

        Args:
            samples: Formatted samples for prediction
            inference_config: Inference settings

        Returns:
            List of predictions (from output_handler.parse_prediction)
        """
        pass

    def format_samples(
        self,
        submissions: List,
        include_label: bool = True,
    ) -> List[FormattedSample]:
        """
        Format submissions into samples using the input formatter.

        Args:
            submissions: List of SubmissionData objects
            include_label: Whether to include ground truth labels

        Returns:
            List of FormattedSample objects
        """
        samples = []
        for submission in submissions:
            sample = self.input_formatter.format_sample(
                submission,
                self.output_handler,
                include_label=include_label,
            )
            samples.append(sample)
        return samples

    def run_pipeline(
        self,
        train_submissions: List,
        eval_submissions: Optional[List] = None,
        output_dir: Path = Path("outputs"),
    ) -> Dict[str, Any]:
        """
        Run the full training pipeline.

        Args:
            train_submissions: Training data
            eval_submissions: Optional evaluation data
            output_dir: Directory for outputs

        Returns:
            Dict with training results and evaluation metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {"mode": self.mode.name}

        # Prepare training data
        train_samples = self.format_samples(train_submissions, include_label=True)
        data_info = self.prepare_data(train_samples, output_dir / "data")
        results["data_info"] = data_info

        # Generate config and train
        if self.requires_training:
            config_path = self.generate_config(data_info, output_dir)
            train_result = self.train(config_path)
            results["train_result"] = train_result

        # Evaluate if we have eval data
        if eval_submissions:
            eval_samples = self.format_samples(eval_submissions, include_label=False)
            inference_config = InferenceConfig(
                model_name_or_path=self.config.model_name_or_path,
                adapter_path=str(output_dir / "checkpoints") if self.requires_training else None,
            )
            predictions = self.predict(eval_samples, inference_config)

            # Extract ground truths
            ground_truths = [s.output for s in self.format_samples(eval_submissions, include_label=True)]
            metrics = self.output_handler.compute_metrics(predictions, ground_truths)

            results["eval_metrics"] = metrics
            results["predictions"] = predictions

        return results


def build_prompt(sample: FormattedSample) -> str:
    """
    Build a prompt string from a FormattedSample.

    Args:
        sample: The formatted sample

    Returns:
        Combined prompt string
    """
    return f"{sample.instruction}\n\n{sample.input_text}"


def build_messages(
    sample: FormattedSample,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build chat messages from a FormattedSample.

    Args:
        sample: The formatted sample
        system_prompt: Optional system message

    Returns:
        List of message dicts for chat API
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = f"{sample.instruction}\n\n{sample.input_text}"
    messages.append({"role": "user", "content": user_content})

    return messages
