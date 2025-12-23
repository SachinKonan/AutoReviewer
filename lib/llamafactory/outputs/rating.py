"""
Mean rating output handler.

Handles predictions for paper mean reviewer rating (regression task).
"""

import re
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.registry import register_output_handler
from ..core.types import OutputType, SubmissionData
from .base import OutputHandler


@register_output_handler(OutputType.MEAN_RATING)
class MeanRatingHandler(OutputHandler):
    """
    Handle mean rating predictions.

    Predicts the expected mean reviewer rating for a paper.
    Can output as raw rating [1-10] or as percentile [0-1].
    """

    output_type = OutputType.MEAN_RATING

    def __init__(
        self,
        as_percentile: bool = False,
        min_rating: float = 1.0,
        max_rating: float = 10.0,
        **kwargs,
    ):
        """
        Initialize mean rating handler.

        Args:
            as_percentile: Output as percentile [0,1] instead of raw rating
            min_rating: Minimum possible rating
            max_rating: Maximum possible rating
            **kwargs: Additional arguments for OutputHandler
        """
        super().__init__(**kwargs)
        self.as_percentile = as_percentile
        self.min_rating = min_rating
        self.max_rating = max_rating

    def _default_system_prompt(self) -> str:
        return (
            "You are an expert reviewer for the ICLR conference. "
            "Your task is to predict the mean reviewer rating a paper will receive "
            "based on the provided information."
        )

    def get_instruction(self) -> str:
        if self.as_percentile:
            instruction = (
                "Based on the paper information provided, predict the expected "
                "mean rating percentile.\n\n"
                "The percentile should be between 0.0 and 1.0, where:\n"
                "- 0.0 = lowest rated papers\n"
                "- 0.5 = median rating\n"
                "- 1.0 = highest rated papers\n\n"
            )

            if self.include_reasoning:
                instruction += (
                    "Provide your response in the following format:\n"
                    "Reasoning: [Your assessment of the paper's likely reception]\n"
                    "Answer: [Decimal number between 0.0 and 1.0]"
                )
            else:
                instruction += "Answer with a decimal number between 0.0 and 1.0."
        else:
            instruction = (
                "Based on the paper information provided, predict the expected "
                f"mean reviewer rating on a scale of {self.min_rating} to {self.max_rating}.\n\n"
                "Rating scale interpretation:\n"
                f"- {self.min_rating}-3: Strong reject\n"
                "- 4: Reject\n"
                "- 5: Marginally below acceptance threshold\n"
                "- 6: Marginally above acceptance threshold\n"
                "- 7: Good paper, accept\n"
                "- 8: Strong accept\n"
                f"- 9-{self.max_rating}: Outstanding paper\n\n"
            )

            if self.include_reasoning:
                instruction += (
                    "Provide your response in the following format:\n"
                    "Reasoning: [Your assessment of the paper's quality and likely reception]\n"
                    f"Answer: [Number between {self.min_rating} and {self.max_rating}]"
                )
            else:
                instruction += f"Answer with a number between {self.min_rating} and {self.max_rating}."

        return instruction

    def format_label(self, submission: SubmissionData) -> str:
        """Format ground truth label from submission."""
        # Try to get mean rating from labels or compute from reviews
        mean_rating = submission.labels.get("mean_rating")
        if mean_rating is None:
            mean_rating = submission.get_mean_rating()
        if mean_rating is None:
            mean_rating = 5.0  # Default to middle of scale

        if self.as_percentile:
            # Convert to percentile
            percentile = (mean_rating - self.min_rating) / (self.max_rating - self.min_rating)
            percentile = max(0.0, min(1.0, percentile))
            if self.include_reasoning:
                return f"Answer: {percentile:.2f}"
            return f"{percentile:.2f}"
        else:
            if self.include_reasoning:
                return f"Answer: {mean_rating:.1f}"
            return f"{mean_rating:.1f}"

    def parse_prediction(self, text: str) -> Dict[str, Any]:
        """Parse model output to extract rating prediction."""
        result = {
            "prediction": None,
            "reasoning": None,
            "raw": text,
            "parse_success": False,
        }

        # Extract reasoning
        result["reasoning"] = self.extract_reasoning(text)

        # Try to find numeric answer
        patterns = [
            r'Answer:\s*(\d+\.?\d*)',
            r'Rating:\s*(\d+\.?\d*)',
            r'Prediction:\s*(\d+\.?\d*)',
            r'Mean rating:\s*(\d+\.?\d*)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))

                    if self.as_percentile:
                        # Expect value in [0, 1]
                        if value > 1.0:
                            if value <= 100:
                                value = value / 100.0
                            elif value <= self.max_rating:
                                # Likely a raw rating, convert to percentile
                                value = (value - self.min_rating) / (self.max_rating - self.min_rating)
                        value = max(0.0, min(1.0, value))
                    else:
                        # Expect value in rating scale
                        if value <= 1.0:
                            # Likely a percentile, convert to rating
                            value = value * (self.max_rating - self.min_rating) + self.min_rating
                        value = max(self.min_rating, min(self.max_rating, value))

                    result["prediction"] = value
                    result["parse_success"] = True
                    break
                except ValueError:
                    continue

        # If no structured answer, try to find any number in reasonable range
        if not result["parse_success"]:
            if self.as_percentile:
                decimals = re.findall(r'(?<!\d)0\.\d+(?!\d)', text)
                if decimals:
                    result["prediction"] = float(decimals[-1])
                    result["parse_success"] = True
            else:
                numbers = re.findall(r'(?<!\d)([1-9]|10)(?:\.\d+)?(?!\d)', text)
                if numbers:
                    result["prediction"] = float(numbers[-1])
                    result["parse_success"] = True

        return result

    def compute_metrics(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Any],
    ) -> Dict[str, float]:
        """Compute regression metrics for rating prediction."""
        from scipy.stats import pearsonr, spearmanr

        y_pred = []
        y_true = []

        default_val = 0.5 if self.as_percentile else 5.0

        for pred, gt in zip(predictions, ground_truths):
            pred_val = pred.get("prediction")
            if pred_val is None:
                pred_val = default_val

            y_pred.append(pred_val)
            y_true.append(float(gt))

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        # Compute metrics
        mse = np.mean((y_pred - y_true) ** 2)
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(mse)

        # Correlation metrics
        try:
            spearman, _ = spearmanr(y_true, y_pred)
        except Exception:
            spearman = 0.0

        try:
            pearson, _ = pearsonr(y_true, y_pred)
        except Exception:
            pearson = 0.0

        metrics = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "spearman": spearman,
            "pearson": pearson,
        }

        # Additional metrics based on mode
        if self.as_percentile:
            # Quartile accuracy
            pred_quartile = (y_pred * 4).astype(int).clip(0, 3)
            true_quartile = (y_true * 4).astype(int).clip(0, 3)
            metrics["quartile_accuracy"] = np.mean(pred_quartile == true_quartile)
        else:
            # Acceptance prediction accuracy (rating >= 6 usually means accept)
            threshold = 5.5
            pred_accept = y_pred >= threshold
            true_accept = y_true >= threshold
            metrics["accept_accuracy"] = np.mean(pred_accept == true_accept)

            # Within 1 point accuracy
            metrics["within_1_accuracy"] = np.mean(np.abs(y_pred - y_true) <= 1.0)

        # Parse success rate
        parse_successes = sum(1 for p in predictions if p.get("parse_success", False))
        metrics["parse_success_rate"] = parse_successes / len(predictions) if predictions else 0

        return metrics

    def get_label_choices(self) -> Optional[List[str]]:
        """Return None as this is a regression task."""
        return None
