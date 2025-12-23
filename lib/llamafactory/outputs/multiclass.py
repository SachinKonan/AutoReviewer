"""
Multi-class decision output handler.

Handles predictions for paper decisions with multiple acceptance tiers:
Reject, Accept (Poster), Accept (Spotlight), Accept (Oral).
"""

import re
from typing import Any, Dict, List, Optional

from ..core.registry import register_output_handler
from ..core.types import OutputType, SubmissionData
from .base import OutputHandler


@register_output_handler(OutputType.MULTICLASS)
class MultiClassDecisionHandler(OutputHandler):
    """
    Handle multi-class decision predictions.

    Predicts the specific acceptance tier or rejection:
    - Reject
    - Accept (Poster)
    - Accept (Spotlight)
    - Accept (Oral)
    """

    output_type = OutputType.MULTICLASS

    # Valid decision classes in order of prestige
    CLASSES = [
        "Reject",
        "Accept (Poster)",
        "Accept (Spotlight)",
        "Accept (Oral)",
    ]

    # Short forms for matching
    CLASS_ALIASES = {
        "reject": "Reject",
        "poster": "Accept (Poster)",
        "accept poster": "Accept (Poster)",
        "accept (poster)": "Accept (Poster)",
        "spotlight": "Accept (Spotlight)",
        "accept spotlight": "Accept (Spotlight)",
        "accept (spotlight)": "Accept (Spotlight)",
        "oral": "Accept (Oral)",
        "accept oral": "Accept (Oral)",
        "accept (oral)": "Accept (Oral)",
        "accept": "Accept (Poster)",  # Default accept to poster
    }

    def _default_system_prompt(self) -> str:
        return (
            "You are an expert reviewer for the ICLR conference. "
            "Your task is to predict the final decision for a paper, "
            "including the specific acceptance category if accepted."
        )

    def get_instruction(self) -> str:
        classes_str = ", ".join(f"'{c}'" for c in self.CLASSES)

        instruction = (
            "Based on the paper information provided, predict the final decision "
            "for this ICLR submission.\n\n"
            f"Possible decisions: {classes_str}\n\n"
        )

        if self.include_reasoning:
            instruction += (
                "Provide your response in the following format:\n"
                "Reasoning: [Your reasoning about the paper's quality, novelty, "
                "significance, and expected tier]\n"
                "Answer: [Exact decision from the list above]"
            )
        else:
            instruction += f"Answer with one of: {classes_str}"

        return instruction

    def format_label(self, submission: SubmissionData) -> str:
        """Format ground truth label from submission decision."""
        label = submission.get_multiclass_label()

        if self.include_reasoning:
            return f"Answer: {label}"
        return label

    def parse_prediction(self, text: str) -> Dict[str, Any]:
        """Parse model output to extract multi-class prediction."""
        result = {
            "prediction": None,
            "reasoning": None,
            "raw": text,
            "parse_success": False,
        }

        # Extract reasoning
        result["reasoning"] = self.extract_reasoning(text)

        # Try to find answer in structured format
        answer_patterns = [
            r'Answer:\s*(.+?)(?:\n|$)',
            r'Prediction:\s*(.+?)(?:\n|$)',
            r'Decision:\s*(.+?)(?:\n|$)',
        ]

        answer_text = None
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer_text = match.group(1).strip()
                break

        if not answer_text:
            # Use the whole text for matching
            answer_text = text

        # Normalize and match
        answer_lower = answer_text.lower()

        # Try exact class matches first
        for cls in self.CLASSES:
            if cls.lower() in answer_lower:
                result["prediction"] = cls
                result["parse_success"] = True
                return result

        # Try aliases
        for alias, cls in self.CLASS_ALIASES.items():
            if alias in answer_lower:
                result["prediction"] = cls
                result["parse_success"] = True
                return result

        # Default: if parsing fails, try to at least determine accept/reject
        if "oral" in answer_lower:
            result["prediction"] = "Accept (Oral)"
            result["parse_success"] = True
        elif "spotlight" in answer_lower:
            result["prediction"] = "Accept (Spotlight)"
            result["parse_success"] = True
        elif "poster" in answer_lower or "accept" in answer_lower:
            result["prediction"] = "Accept (Poster)"
            result["parse_success"] = True
        elif "reject" in answer_lower:
            result["prediction"] = "Reject"
            result["parse_success"] = True

        return result

    def compute_metrics(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Any],
    ) -> Dict[str, float]:
        """Compute multi-class classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            classification_report,
        )

        # Map classes to indices
        class_to_idx = {c: i for i, c in enumerate(self.CLASSES)}

        y_pred = []
        y_true = []

        for pred, gt in zip(predictions, ground_truths):
            pred_val = pred.get("prediction")
            if pred_val is None or pred_val not in class_to_idx:
                pred_val = "Reject"  # Default for failed parses

            y_pred.append(class_to_idx.get(pred_val, 0))
            y_true.append(class_to_idx.get(gt, 0))

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        # Per-class accuracy
        for cls in self.CLASSES:
            cls_idx = class_to_idx[cls]
            cls_true = [1 if y == cls_idx else 0 for y in y_true]
            cls_pred = [1 if y == cls_idx else 0 for y in y_pred]
            if sum(cls_true) > 0:
                metrics[f"recall_{cls.replace(' ', '_').replace('(', '').replace(')', '')}"] = (
                    sum(1 for t, p in zip(cls_true, cls_pred) if t == 1 and p == 1) / sum(cls_true)
                )

        # Binary accept/reject accuracy
        binary_pred = [0 if p == 0 else 1 for p in y_pred]
        binary_true = [0 if t == 0 else 1 for t in y_true]
        metrics["binary_accuracy"] = accuracy_score(binary_true, binary_pred)

        # Parse success rate
        parse_successes = sum(1 for p in predictions if p.get("parse_success", False))
        metrics["parse_success_rate"] = parse_successes / len(predictions) if predictions else 0

        return metrics

    def get_label_choices(self) -> List[str]:
        return self.CLASSES
