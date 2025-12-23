"""
Citation percentile output handler.

Handles predictions for paper citation impact percentile (regression task).
"""

import re
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.registry import register_output_handler
from ..core.types import OutputType, SubmissionData
from .base import OutputHandler


@register_output_handler(OutputType.CITATION_PERCENTILE)
class CitationPercentileHandler(OutputHandler):
    """
    Handle citation percentile predictions.

    Predicts the expected citation percentile of a paper
    compared to other papers in the same venue/year.

    Output is a float in [0, 1]:
    - 0.0 = bottom percentile (least cited)
    - 0.5 = median citations
    - 0.9 = top 10%
    - 1.0 = top percentile (most cited)
    """

    output_type = OutputType.CITATION_PERCENTILE

    def __init__(
        self,
        discretize: bool = False,
        num_bins: int = 10,
        **kwargs,
    ):
        """
        Initialize citation percentile handler.

        Args:
            discretize: Whether to discretize output into bins
            num_bins: Number of bins if discretizing
            **kwargs: Additional arguments for OutputHandler
        """
        super().__init__(**kwargs)
        self.discretize = discretize
        self.num_bins = num_bins

    def _default_system_prompt(self) -> str:
        return (
            "You are an expert at predicting research impact. "
            "Your task is to estimate the citation percentile a paper will achieve "
            "compared to other papers in the same venue and year."
        )

    def get_instruction(self) -> str:
        instruction = (
            "Based on the paper information provided, predict its citation impact percentile.\n\n"
            "The percentile should be between 0.0 and 1.0, where:\n"
            "- 0.0 = bottom percentile (least cited papers)\n"
            "- 0.25 = bottom quartile\n"
            "- 0.5 = median citations\n"
            "- 0.75 = top quartile\n"
            "- 0.9 = top 10%\n"
            "- 1.0 = top percentile (most cited papers)\n\n"
        )

        if self.include_reasoning:
            instruction += (
                "Provide your response in the following format:\n"
                "Reasoning: [Your assessment of the paper's potential impact, "
                "considering novelty, significance, clarity, and field]\n"
                "Answer: [Decimal number between 0.0 and 1.0, e.g., 0.75]"
            )
        else:
            instruction += "Answer with a decimal number between 0.0 and 1.0."

        return instruction

    def format_label(self, submission: SubmissionData) -> str:
        """Format ground truth label from submission labels."""
        percentile = submission.labels.get("citation_percentile", 0.5)

        if self.discretize:
            # Round to bin center
            bin_size = 1.0 / self.num_bins
            bin_idx = min(int(percentile / bin_size), self.num_bins - 1)
            percentile = (bin_idx + 0.5) * bin_size

        if self.include_reasoning:
            return f"Answer: {percentile:.2f}"
        return f"{percentile:.2f}"

    def parse_prediction(self, text: str) -> Dict[str, Any]:
        """Parse model output to extract percentile prediction."""
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
            r'Prediction:\s*(\d+\.?\d*)',
            r'Percentile:\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*(?:percentile|%|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))

                    # Handle different scales
                    if value > 1.0:
                        if value <= 100:
                            value = value / 100.0  # Percentage to decimal
                        else:
                            continue  # Not a valid percentile

                    # Clamp to [0, 1]
                    value = max(0.0, min(1.0, value))

                    result["prediction"] = value
                    result["parse_success"] = True
                    break
                except ValueError:
                    continue

        # If no structured answer found, try to find any decimal
        if not result["parse_success"]:
            decimals = re.findall(r'(?<!\d)0\.\d+(?!\d)', text)
            if decimals:
                result["prediction"] = float(decimals[-1])  # Use last decimal found
                result["parse_success"] = True

        return result

    def compute_metrics(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Any],
    ) -> Dict[str, float]:
        """Compute regression metrics for percentile prediction."""
        from scipy.stats import pearsonr, spearmanr

        y_pred = []
        y_true = []

        for pred, gt in zip(predictions, ground_truths):
            pred_val = pred.get("prediction")
            if pred_val is None:
                pred_val = 0.5  # Default to median for failed parses

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

        # Binned accuracy (how often we get the correct quartile)
        pred_quartile = (y_pred * 4).astype(int).clip(0, 3)
        true_quartile = (y_true * 4).astype(int).clip(0, 3)
        metrics["quartile_accuracy"] = np.mean(pred_quartile == true_quartile)

        # Parse success rate
        parse_successes = sum(1 for p in predictions if p.get("parse_success", False))
        metrics["parse_success_rate"] = parse_successes / len(predictions) if predictions else 0

        return metrics

    def get_label_choices(self) -> Optional[List[str]]:
        """Return None as this is a regression task."""
        return None
