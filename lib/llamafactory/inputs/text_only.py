"""
Text-only input formatters.

These formatters work with text content only (abstract, reviews, markdown)
and do not require a vision-language model.
"""

from typing import List, Optional

from ..core.registry import register_input_formatter
from ..core.types import InputModality, SubmissionData
from .base import InputFormatter


@register_input_formatter(InputModality.TEXT_ONLY)
class TextOnlyFormatter(InputFormatter):
    """
    Format text-only inputs (abstract, reviews, markdown).

    This is the default formatter that works with any text-only model.
    It can include various combinations of content based on configuration.
    """

    modality = InputModality.TEXT_ONLY
    requires_vl_model = False

    def format_content(self, submission: SubmissionData) -> str:
        """Format submission content as text."""
        parts = []

        # Title and abstract (always included)
        parts.append(f"# {submission.title}")
        parts.append(f"\n## Abstract\n{submission.abstract}")

        # Reviews if available and requested
        if self.include_reviews and submission.reviews:
            parts.append("\n## Reviews")
            reviews_text = self._format_reviews(submission.reviews)
            parts.append(reviews_text)

        # Meta-review if available
        if self.include_reviews and submission.meta_review:
            meta_text = self._format_meta_review(submission.meta_review)
            if meta_text:
                parts.append(f"\n{meta_text}")

        # Markdown content if requested
        if self.include_markdown and submission.clean_md:
            # Allocate remaining token budget to markdown
            used_chars = sum(len(p) for p in parts)
            remaining = max(0, int(self.max_tokens * 3.5) - used_chars)
            if remaining > 1000:  # Only include if meaningful amount
                truncated_md = self._truncate_text(submission.clean_md, remaining)
                parts.append(f"\n## Paper Content\n{truncated_md}")

        return "\n".join(parts)

    def get_images(self, submission: SubmissionData) -> Optional[List[str]]:
        """Text-only formatter returns no images."""
        return None


class AbstractOnlyFormatter(TextOnlyFormatter):
    """
    Format with only title and abstract.

    Minimal formatter useful for quick predictions or when
    detailed review information is not available.
    """

    def __init__(self, **kwargs):
        # Override defaults to exclude reviews and markdown
        kwargs.setdefault("include_reviews", False)
        kwargs.setdefault("include_markdown", False)
        super().__init__(**kwargs)


class AbstractWithReviewsFormatter(TextOnlyFormatter):
    """
    Format with abstract and normalized reviews.

    Standard formatter that includes reviewer feedback but not
    the full paper content.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("include_reviews", True)
        kwargs.setdefault("include_markdown", False)
        super().__init__(**kwargs)


class FullTextFormatter(TextOnlyFormatter):
    """
    Format with full paper markdown content.

    Uses larger context window to include full paper text.
    Excludes reviews by default to focus on paper content.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("include_reviews", False)
        kwargs.setdefault("include_markdown", True)
        kwargs.setdefault("max_tokens", 16000)
        super().__init__(**kwargs)


class FullContextFormatter(TextOnlyFormatter):
    """
    Format with full context: abstract, reviews, and markdown.

    Maximum information formatter for models with large context windows.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("include_reviews", True)
        kwargs.setdefault("include_markdown", True)
        kwargs.setdefault("max_tokens", 32000)
        super().__init__(**kwargs)
