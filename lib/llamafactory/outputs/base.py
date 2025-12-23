"""
Base class for output handlers.

Output handlers define how to format task instructions, parse model predictions,
and compute evaluation metrics for different prediction targets.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..core.types import OutputType, SubmissionData


class OutputHandler(ABC):
    """
    Abstract base class for handling prediction outputs.

    Subclasses define how to:
    - Generate task-specific instructions
    - Format ground truth labels for training
    - Parse model predictions
    - Compute evaluation metrics

    Attributes:
        output_type: The output type this handler handles
    """

    output_type: OutputType = NotImplemented

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        output_format: str = "text",
        include_reasoning: bool = True,
    ):
        """
        Initialize output handler.

        Args:
            system_prompt: Optional system prompt override
            output_format: Expected output format ("text" or "json")
            include_reasoning: Whether to request reasoning in output
        """
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.output_format = output_format
        self.include_reasoning = include_reasoning

    @abstractmethod
    def _default_system_prompt(self) -> str:
        """Get default system prompt for this output type."""
        pass

    @abstractmethod
    def get_instruction(self) -> str:
        """
        Get task instruction for the model.

        Returns:
            Instruction text describing what the model should predict
        """
        pass

    @abstractmethod
    def format_label(self, submission: SubmissionData) -> str:
        """
        Format ground truth label for training.

        Args:
            submission: Submission data with labels/decision

        Returns:
            Formatted label string for model output
        """
        pass

    @abstractmethod
    def parse_prediction(self, text: str) -> Dict[str, Any]:
        """
        Parse model output into structured prediction.

        Args:
            text: Raw model output text

        Returns:
            Dict with:
                - 'prediction': The parsed prediction value
                - 'reasoning': Optional extracted reasoning
                - 'confidence': Optional confidence score
                - 'raw': Original text
                - 'parse_success': Whether parsing succeeded
        """
        pass

    @abstractmethod
    def compute_metrics(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Any],
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            predictions: List of parsed predictions (from parse_prediction)
            ground_truths: List of ground truth values

        Returns:
            Dict of metric names to values
        """
        pass

    def get_label_choices(self) -> Optional[List[str]]:
        """
        Get valid label choices for classification tasks.

        Returns:
            List of valid labels, or None for regression tasks
        """
        return None

    def get_reasoning_prompt(self) -> str:
        """Get the reasoning request part of the instruction."""
        if self.include_reasoning:
            return "Reasoning: [Your analysis and reasoning]\n"
        return ""

    def extract_reasoning(self, text: str) -> Optional[str]:
        """
        Extract reasoning from model output.

        Args:
            text: Raw model output

        Returns:
            Extracted reasoning text or None
        """
        import re

        # Try to find reasoning section
        patterns = [
            r'Reasoning:\s*(.+?)(?=Answer:|Prediction:|$)',
            r'Analysis:\s*(.+?)(?=Answer:|Prediction:|$)',
            r'Explanation:\s*(.+?)(?=Answer:|Prediction:|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None
