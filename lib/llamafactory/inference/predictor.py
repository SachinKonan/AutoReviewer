"""
Unified prediction interface for all training modes.

Provides a single interface for running predictions regardless of
the underlying training mode (zero-shot, SFT, GRPO).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.types import (
    FormattedSample,
    InputModality,
    OutputType,
    PredictorConfig,
    SubmissionData,
    TrainingConfig,
    TrainingMode,
    InferenceConfig,
)
from ..core.registry import (
    get_input_formatter,
    get_output_handler,
    get_training_mode,
)
from ..inputs.base import InputFormatter
from ..outputs.base import OutputHandler
from ..training.base import TrainingModeBase


class UnifiedPredictor:
    """
    Unified prediction interface for all modes.

    Combines input formatter, output handler, and training mode
    into a single interface for predictions.

    Example:
        predictor = UnifiedPredictor.create(
            input_modality=InputModality.TEXT_WITH_IMAGES,
            output_type=OutputType.BINARY,
            training_mode=TrainingMode.SFT_LORA,
            config=PredictorConfig(...),
        )
        predictions = predictor.predict(submissions)
    """

    def __init__(
        self,
        input_formatter: InputFormatter,
        output_handler: OutputHandler,
        training_mode: TrainingModeBase,
        config: PredictorConfig,
    ):
        """
        Initialize unified predictor.

        Args:
            input_formatter: Input formatting strategy
            output_handler: Output handling strategy
            training_mode: Training mode implementation
            config: Predictor configuration
        """
        self.input_formatter = input_formatter
        self.output_handler = output_handler
        self.training_mode = training_mode
        self.config = config

    @classmethod
    def create(
        cls,
        input_modality: InputModality,
        output_type: OutputType,
        training_mode: TrainingMode,
        config: PredictorConfig,
        input_kwargs: Optional[Dict[str, Any]] = None,
        output_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "UnifiedPredictor":
        """
        Factory method to create predictor from configuration.

        Args:
            input_modality: Type of input content
            output_type: Type of prediction output
            training_mode: Training/inference mode
            config: Predictor configuration
            input_kwargs: Additional arguments for input formatter
            output_kwargs: Additional arguments for output handler

        Returns:
            Configured UnifiedPredictor instance
        """
        input_kwargs = input_kwargs or {}
        output_kwargs = output_kwargs or {}

        # Get component classes from registry
        input_formatter_cls = get_input_formatter(input_modality)
        output_handler_cls = get_output_handler(output_type)
        training_mode_cls = get_training_mode(training_mode)

        # Instantiate components
        input_formatter = input_formatter_cls(**input_kwargs)
        output_handler = output_handler_cls(**output_kwargs)

        # Create training config from predictor config
        training_config = TrainingConfig(
            model_name_or_path=config.model_name_or_path,
            model_type=config.model_type,
        )

        training_mode_instance = training_mode_cls(
            config=training_config,
            input_formatter=input_formatter,
            output_handler=output_handler,
        )

        return cls(
            input_formatter=input_formatter,
            output_handler=output_handler,
            training_mode=training_mode_instance,
            config=config,
        )

    def _get_inference_config(self) -> InferenceConfig:
        """Convert predictor config to inference config."""
        return self.config.to_inference_config()

    def format_submissions(
        self,
        submissions: List[SubmissionData],
        include_label: bool = False,
    ) -> List[FormattedSample]:
        """
        Format submissions into samples.

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

    def predict(
        self,
        submissions: List[SubmissionData],
    ) -> List[Dict[str, Any]]:
        """
        Run prediction on submissions.

        Args:
            submissions: List of SubmissionData objects

        Returns:
            List of prediction dicts with 'prediction', 'reasoning', etc.
        """
        samples = self.format_submissions(submissions, include_label=False)
        inference_config = self._get_inference_config()
        predictions = self.training_mode.predict(samples, inference_config)
        return predictions

    def predict_with_evaluation(
        self,
        submissions: List[SubmissionData],
    ) -> Dict[str, Any]:
        """
        Run prediction and compute evaluation metrics.

        Args:
            submissions: List of SubmissionData with ground truth labels

        Returns:
            Dict with 'predictions', 'metrics', 'num_samples'
        """
        # Get predictions
        predictions = self.predict(submissions)

        # Extract ground truths
        labeled_samples = self.format_submissions(submissions, include_label=True)
        ground_truths = []
        for sample in labeled_samples:
            # Parse the expected output format
            parsed = self.output_handler.parse_prediction(sample.output or "")
            ground_truths.append(parsed.get("prediction", sample.output))

        # Compute metrics
        metrics = self.output_handler.compute_metrics(predictions, ground_truths)

        return {
            "predictions": predictions,
            "metrics": metrics,
            "num_samples": len(submissions),
        }

    def predict_single(
        self,
        submission: SubmissionData,
    ) -> Dict[str, Any]:
        """
        Predict on a single submission.

        Args:
            submission: Single SubmissionData object

        Returns:
            Prediction dict
        """
        results = self.predict([submission])
        return results[0] if results else {}


class BatchPredictor:
    """
    Batch prediction utilities for large-scale inference.

    Handles batching, progress tracking, and result aggregation.
    """

    def __init__(
        self,
        predictor: UnifiedPredictor,
        batch_size: int = 32,
        show_progress: bool = True,
    ):
        """
        Initialize batch predictor.

        Args:
            predictor: UnifiedPredictor instance
            batch_size: Number of samples per batch
            show_progress: Whether to show progress bar
        """
        self.predictor = predictor
        self.batch_size = batch_size
        self.show_progress = show_progress

    def predict_batched(
        self,
        submissions: List[SubmissionData],
    ) -> List[Dict[str, Any]]:
        """
        Run batched prediction.

        Args:
            submissions: List of SubmissionData objects

        Returns:
            List of all predictions
        """
        all_predictions = []
        num_batches = (len(submissions) + self.batch_size - 1) // self.batch_size

        iterator = range(0, len(submissions), self.batch_size)
        if self.show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=num_batches, desc="Predicting")
            except ImportError:
                pass

        for i in iterator:
            batch = submissions[i:i + self.batch_size]
            batch_predictions = self.predictor.predict(batch)
            all_predictions.extend(batch_predictions)

        return all_predictions

    def predict_and_save(
        self,
        submissions: List[SubmissionData],
        output_path: Path,
        include_metrics: bool = True,
    ) -> Path:
        """
        Run prediction and save results to file.

        Args:
            submissions: List of SubmissionData objects
            output_path: Path to save results
            include_metrics: Whether to compute and include metrics

        Returns:
            Path to saved results file
        """
        import json

        predictions = self.predict_batched(submissions)

        results = {
            "predictions": predictions,
            "num_samples": len(submissions),
            "config": {
                "model": self.predictor.config.model_name_or_path,
                "adapter": self.predictor.config.adapter_path,
                "input_modality": self.predictor.input_formatter.modality.name,
                "output_type": self.predictor.output_handler.output_type.name,
            },
        }

        if include_metrics:
            # Get ground truths if available
            labeled_samples = self.predictor.format_submissions(submissions, include_label=True)
            ground_truths = []
            for sample in labeled_samples:
                if sample.output:
                    parsed = self.predictor.output_handler.parse_prediction(sample.output)
                    ground_truths.append(parsed.get("prediction", sample.output))
                else:
                    ground_truths.append(None)

            # Only compute metrics if we have ground truths
            valid_pairs = [(p, g) for p, g in zip(predictions, ground_truths) if g is not None]
            if valid_pairs:
                valid_preds, valid_gts = zip(*valid_pairs)
                metrics = self.predictor.output_handler.compute_metrics(list(valid_preds), list(valid_gts))
                results["metrics"] = metrics

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return output_path
