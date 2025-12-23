"""
Binary accept/reject output handler.

Handles predictions for binary paper acceptance decisions.
"""

import re
from typing import Any, Dict, List, Optional

from ..core.registry import register_output_handler
from ..core.types import OutputType, SubmissionData
from .base import OutputHandler


@register_output_handler(OutputType.BINARY)
class BinaryAcceptRejectHandler(OutputHandler):
    """
    Handle binary accept/reject predictions.

    Predicts whether a paper will be accepted or rejected,
    without distinguishing between acceptance tiers.
    """

    output_type = OutputType.BINARY
    LABELS = ["Accept", "Reject"]

    def _default_system_prompt(self) -> str:
        return (
            "You are an expert reviewer for the ICLR conference. "
            "Your task is to predict whether a paper will be accepted or rejected "
            "based on the provided information."
        )

    def get_instruction(self) -> str:
        instruction = (
            "Based on the paper information provided, predict whether this paper "
            "will be accepted or rejected at ICLR.\n\n"
        )

        if self.include_reasoning:
            instruction += (
                "Provide your response in the following format:\n"
                "Reasoning: [Your reasoning about the paper's quality, novelty, "
                "significance, and fit for the conference]\n"
                "Answer: [Either 'Accept' or 'Reject']"
            )
        else:
            instruction += "Answer with either 'Accept' or 'Reject'."

        return instruction

    def format_label(self, submission: SubmissionData) -> str:
        """Format ground truth label from submission decision."""
        label = submission.get_binary_label()

        if self.include_reasoning:
            # For training, we might want to include minimal reasoning
            return f"Answer: {label}"
        return label

    def parse_prediction(self, text: str) -> Dict[str, Any]:
        """Parse model output to extract accept/reject prediction."""
        result = {
            "prediction": None,
            "reasoning": None,
            "raw": text,
            "parse_success": False,
        }

        # Extract reasoning
        result["reasoning"] = self.extract_reasoning(text)

        # Extract answer - try multiple patterns
        patterns = [
            r'Answer:\s*(Accept|Reject)',
            r'Prediction:\s*(Accept|Reject)',
            r'\b(Accept|Reject)\b(?:\s*$|\s*\.)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip().title()
                if answer in self.LABELS:
                    result["prediction"] = answer
                    result["parse_success"] = True
                    break

        # Last resort: look for accept/reject anywhere
        if not result["parse_success"]:
            text_lower = text.lower()
            if "accept" in text_lower and "reject" not in text_lower:
                result["prediction"] = "Accept"
                result["parse_success"] = True
            elif "reject" in text_lower and "accept" not in text_lower:
                result["prediction"] = "Reject"
                result["parse_success"] = True

        return result

    def compute_metrics(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Any],
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        # Convert to binary
        y_pred = []
        y_true = []

        for pred, gt in zip(predictions, ground_truths):
            pred_val = pred.get("prediction")
            if pred_val is None:
                pred_val = "Reject"  # Default to reject for failed parses

            y_pred.append(1 if pred_val == "Accept" else 0)
            y_true.append(1 if gt == "Accept" else 0)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        # Add parse success rate
        parse_successes = sum(1 for p in predictions if p.get("parse_success", False))
        metrics["parse_success_rate"] = parse_successes / len(predictions) if predictions else 0

        return metrics

    def get_label_choices(self) -> List[str]:
        return self.LABELS
