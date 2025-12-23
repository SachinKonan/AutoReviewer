"""
Base class for input formatters.

Input formatters define how to extract and format content from submission data
for different input modalities (text-only, text+images, images-only).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..core.types import (
    FormattedSample,
    InputModality,
    ReviewData,
    SubmissionData,
)

if TYPE_CHECKING:
    from ..outputs.base import OutputHandler


class InputFormatter(ABC):
    """
    Abstract base class for formatting input content.

    Subclasses define how to extract and format content from submission data
    for different input modalities (text-only, text+images, images-only).

    Attributes:
        modality: The input modality this formatter handles
        requires_vl_model: Whether this formatter requires a vision-language model
    """

    modality: InputModality = NotImplemented
    requires_vl_model: bool = False

    def __init__(
        self,
        max_tokens: int = 4096,
        include_reviews: bool = True,
        include_markdown: bool = False,
        use_normalized_reviews: bool = True,
        max_images: int = 20,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        few_shot_count: int = 0,
    ):
        """
        Initialize input formatter.

        Args:
            max_tokens: Maximum tokens for text content (approximate)
            include_reviews: Whether to include reviewer feedback
            include_markdown: Whether to include paper markdown content
            use_normalized_reviews: Use LLM-normalized reviews if available
            max_images: Maximum number of PDF page images to include
            few_shot_examples: Optional list of few-shot example dicts
            few_shot_count: Number of few-shot examples to include (0 = none)
        """
        self.max_tokens = max_tokens
        self.include_reviews = include_reviews
        self.include_markdown = include_markdown
        self.use_normalized_reviews = use_normalized_reviews
        self.max_images = max_images
        self.few_shot_examples = few_shot_examples or []
        self.few_shot_count = few_shot_count

    @abstractmethod
    def format_content(self, submission: SubmissionData) -> str:
        """
        Format the main content for the prompt.

        Args:
            submission: Submission data container

        Returns:
            Formatted content string (may include <image> tokens for VL models)
        """
        pass

    @abstractmethod
    def get_images(self, submission: SubmissionData) -> Optional[List[str]]:
        """
        Get image paths for multimodal input.

        Args:
            submission: Submission data container

        Returns:
            List of image file paths, or None for text-only
        """
        pass

    def format_sample(
        self,
        submission: SubmissionData,
        output_handler: "OutputHandler",
        include_label: bool = True,
    ) -> FormattedSample:
        """
        Create a complete formatted sample.

        Args:
            submission: Submission data
            output_handler: Handler for formatting output
            include_label: Whether to include ground truth (training vs inference)

        Returns:
            FormattedSample ready for LlamaFactory conversion
        """
        instruction = output_handler.get_instruction()

        # Add few-shot examples if configured
        if self.few_shot_count > 0 and self.few_shot_examples:
            few_shot_text = self._format_few_shot_examples(output_handler)
            instruction = f"{instruction}\n\n{few_shot_text}"

        content = self.format_content(submission)
        images = self.get_images(submission)

        output = None
        if include_label:
            output = output_handler.format_label(submission)

        return FormattedSample(
            instruction=instruction,
            input_text=content,
            output=output,
            images=images,
            submission_id=submission.submission_id,
            year=submission.year,
            output_type=output_handler.output_type,
        )

    def _format_few_shot_examples(
        self,
        output_handler: "OutputHandler",
    ) -> str:
        """Format few-shot examples for the prompt."""
        examples = self.few_shot_examples[:self.few_shot_count]
        if not examples:
            return ""

        parts = ["Here are some examples:\n"]
        for i, example in enumerate(examples, 1):
            # Build example text
            example_text = f"Example {i}:\n"
            if "title" in example:
                example_text += f"Title: {example['title']}\n"
            if "abstract" in example:
                example_text += f"Abstract: {example['abstract']}\n"
            if "decision" in example:
                example_text += f"Output: Answer: {example['decision']}\n"
            elif "output" in example:
                example_text += f"Output: {example['output']}\n"
            parts.append(example_text)

        parts.append("\nNow analyze the following paper:\n")
        return "\n".join(parts)

    def _truncate_text(self, text: str, max_chars: Optional[int] = None) -> str:
        """
        Truncate text to approximate token limit.

        Args:
            text: Text to truncate
            max_chars: Maximum characters (defaults to max_tokens * 3.5)

        Returns:
            Truncated text with indicator if truncated
        """
        if max_chars is None:
            max_chars = int(self.max_tokens * 3.5)  # ~3.5 chars per token

        if len(text) <= max_chars:
            return text

        return text[:max_chars] + "\n\n[... truncated for length ...]"

    def _format_reviews(self, reviews: List[ReviewData]) -> str:
        """
        Format reviews into readable text.

        Args:
            reviews: List of ReviewData objects

        Returns:
            Formatted review text
        """
        if not reviews:
            return ""

        review_texts = []
        for i, review in enumerate(reviews, 1):
            parts = [f"### Review {i}"]

            if self.use_normalized_reviews and review.summary:
                # Use normalized format
                if review.summary:
                    parts.append(f"**Summary:** {review.summary}")
                if review.strengths:
                    parts.append(f"**Strengths:** {review.strengths}")
                if review.weaknesses:
                    parts.append(f"**Weaknesses:** {review.weaknesses}")
                if review.questions:
                    questions_text = "; ".join(review.questions)
                    parts.append(f"**Questions:** {questions_text}")
            elif review.raw_review:
                # Fallback to raw review format
                raw = review.raw_review
                # Try common field names
                for key in ["summary_of_the_paper", "summary", "review", "main_review"]:
                    if key in raw and raw[key]:
                        parts.append(f"**Review:** {raw[key]}")
                        break
                for key in ["strength_and_weaknesses", "strengths", "weaknesses"]:
                    if key in raw and raw[key]:
                        parts.append(f"**{key.replace('_', ' ').title()}:** {raw[key]}")

            # Add rating if available
            if review.rating is not None:
                parts.append(f"**Rating:** {review.rating}")
            if review.confidence is not None:
                parts.append(f"**Confidence:** {review.confidence}")

            review_texts.append("\n".join(parts))

        return "\n\n".join(review_texts)

    def _format_meta_review(self, meta_review: Optional[Dict[str, Any]]) -> str:
        """
        Format meta-review into readable text.

        Args:
            meta_review: Meta-review dictionary

        Returns:
            Formatted meta-review text
        """
        if not meta_review:
            return ""

        parts = ["### Meta-Review"]

        # Try common field names
        for key in ["metareview", "comment", "metareview_summary_strengths_weaknesses"]:
            if key in meta_review and meta_review[key]:
                parts.append(meta_review[key])
                break

        if "justification_for_why_not_higher_score" in meta_review:
            parts.append(f"**Why not higher:** {meta_review['justification_for_why_not_higher_score']}")
        if "justification_for_why_not_lower_score" in meta_review:
            parts.append(f"**Why not lower:** {meta_review['justification_for_why_not_lower_score']}")

        return "\n".join(parts) if len(parts) > 1 else ""
