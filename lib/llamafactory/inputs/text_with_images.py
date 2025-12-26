"""
Text with images input formatter.

This formatter combines text content with PDF page images for
vision-language models like Qwen2.5-VL.
"""

from typing import List, Optional

from ..core.registry import register_input_formatter
from ..core.types import InputModality, SubmissionData
from .base import InputFormatter


@register_input_formatter(InputModality.TEXT_WITH_IMAGES)
class TextWithImagesFormatter(InputFormatter):
    """
    Format text content with PDF page images.

    Requires a vision-language model (e.g., Qwen2.5-VL-7B-Instruct).
    Combines textual information (abstract, reviews) with visual
    paper content from PDF page images.
    """

    modality = InputModality.TEXT_WITH_IMAGES
    requires_vl_model = True

    def __init__(
        self,
        image_position: str = "after_abstract",
        **kwargs,
    ):
        """
        Initialize text with images formatter.

        Args:
            image_position: Where to place image tokens
                - "after_abstract": After abstract, before reviews
                - "after_reviews": After all text content
                - "before_text": At the beginning
            **kwargs: Additional arguments for InputFormatter
        """
        super().__init__(**kwargs)
        self.image_position = image_position

    def format_content(self, submission: SubmissionData) -> str:
        """Format submission content with image placeholders."""
        parts = []

        # Count images we'll include
        num_images = 0
        if submission.pdf_image_paths:
            num_images = min(len(submission.pdf_image_paths), self.max_images)

        # Image token string
        image_tokens = ""
        if num_images > 0:
            image_tokens = " ".join(["<image>"] * num_images)

        # Images at beginning
        if self.image_position == "before_text" and image_tokens:
            parts.append(f"## Paper Pages\n{image_tokens}")

        # Title and abstract
        parts.append(f"# {submission.title}")
        parts.append(f"\n## Abstract\n{submission.abstract}")

        # Images after abstract
        if self.image_position == "after_abstract" and image_tokens:
            parts.append(f"\n## Paper Pages\n{image_tokens}")

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

        # Images after reviews
        if self.image_position == "after_reviews" and image_tokens:
            parts.append(f"\n## Paper Pages\n{image_tokens}")

        return "\n".join(parts)

    def get_images(self, submission: SubmissionData) -> Optional[List[str]]:
        """Get PDF page image paths."""
        if not submission.pdf_image_paths:
            return None

        paths = submission.pdf_image_paths[:self.max_images]
        return [str(p) for p in paths]


class AbstractWithPageImagesFormatter(TextWithImagesFormatter):
    """
    Format with abstract and PDF page images (no reviews).

    Useful for predictions based on paper content without
    reviewer influence.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("include_reviews", False)
        kwargs.setdefault("image_position", "after_abstract")
        super().__init__(**kwargs)


class ReviewsWithPageImagesFormatter(TextWithImagesFormatter):
    """
    Format with abstract, reviews, and PDF page images.

    Full multimodal context for comprehensive predictions.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("include_reviews", True)
        kwargs.setdefault("image_position", "after_abstract")
        super().__init__(**kwargs)


class KeyPagesFormatter(TextWithImagesFormatter):
    """
    Format with only key pages (first few and last few).

    Optimizes image count by focusing on title page, intro,
    and conclusion/references.
    """

    def __init__(
        self,
        first_pages: int = 3,
        last_pages: int = 2,
        **kwargs,
    ):
        """
        Initialize key pages formatter.

        Args:
            first_pages: Number of pages from the beginning
            last_pages: Number of pages from the end
            **kwargs: Additional arguments for TextWithImagesFormatter
        """
        # Set max_images to sum of first and last
        kwargs.setdefault("max_images", first_pages + last_pages)
        super().__init__(**kwargs)
        self.first_pages = first_pages
        self.last_pages = last_pages

    def get_images(self, submission: SubmissionData) -> Optional[List[str]]:
        """Get first and last page images."""
        if not submission.pdf_image_paths:
            return None

        paths = submission.pdf_image_paths
        if len(paths) <= self.first_pages + self.last_pages:
            # Paper is short enough to include all pages
            return [str(p) for p in paths]

        # Select first N and last M pages
        selected = list(paths[:self.first_pages])
        selected.extend(paths[-self.last_pages:])

        return [str(p) for p in selected]


class MarkdownWithInlineImagesFormatter(InputFormatter):
    """
    Format with markdown content containing inline image references.

    Uses clean_md (markdown with ![](images/...) references) which are
    resolved to actual images at inference time. Images are interleaved
    with text based on their position in the markdown.

    This requires a vision-language model and the images_in_clean_md
    column in the dataset to resolve image references to paths.
    """

    modality = InputModality.TEXT_WITH_IMAGES
    requires_vl_model = True

    def format_content(self, submission: SubmissionData) -> str:
        """Format submission content with markdown (preserving image refs)."""
        parts = []

        # Title and abstract
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

        # Include markdown content WITH image references preserved
        if submission.clean_md:
            # Allocate remaining token budget to markdown
            used_chars = sum(len(p) for p in parts)
            remaining = max(0, int(self.max_tokens * 3.5) - used_chars)
            if remaining > 1000:
                truncated_md = self._truncate_text(submission.clean_md, remaining)
                parts.append(f"\n## Paper Content\n{truncated_md}")

        return "\n".join(parts)

    def get_images(self, submission: SubmissionData) -> Optional[List[str]]:
        """
        Returns None - images are resolved inline from markdown references.

        The RayDataPredictor._build_inline_image_messages method handles
        parsing ![](images/...) references and resolving them using the
        images_in_clean_md mapping.
        """
        return None
