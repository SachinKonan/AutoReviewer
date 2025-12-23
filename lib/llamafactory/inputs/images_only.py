"""
Images-only input formatter.

This formatter uses only PDF page images without explicit text content,
relying on the vision-language model to extract information from images.
"""

from typing import List, Optional

from ..core.registry import register_input_formatter
from ..core.types import InputModality, SubmissionData
from .base import InputFormatter


@register_input_formatter(InputModality.IMAGES_ONLY)
class ImagesOnlyFormatter(InputFormatter):
    """
    Format with only PDF page images (no explicit text).

    Requires a vision-language model. The model must extract
    all information (title, abstract, content) from the images.

    This tests the model's ability to understand papers purely
    from visual content.
    """

    modality = InputModality.IMAGES_ONLY
    requires_vl_model = True

    def __init__(
        self,
        include_page_context: bool = True,
        **kwargs,
    ):
        """
        Initialize images-only formatter.

        Args:
            include_page_context: Whether to mention these are paper pages
            **kwargs: Additional arguments for InputFormatter
        """
        # Images-only doesn't use reviews or markdown
        kwargs["include_reviews"] = False
        kwargs["include_markdown"] = False
        super().__init__(**kwargs)
        self.include_page_context = include_page_context

    def format_content(self, submission: SubmissionData) -> str:
        """Format with only image placeholders and minimal context."""
        if not submission.pdf_image_paths:
            raise ValueError(
                f"ImagesOnlyFormatter requires images but submission "
                f"{submission.submission_id} has no PDF image paths"
            )

        num_images = min(len(submission.pdf_image_paths), self.max_images)
        image_tokens = " ".join(["<image>"] * num_images)

        if self.include_page_context:
            return (
                f"The following {num_images} images show pages from an "
                f"academic paper submitted to ICLR {submission.year}:\n\n"
                f"{image_tokens}"
            )
        else:
            return image_tokens

    def get_images(self, submission: SubmissionData) -> Optional[List[str]]:
        """Get PDF page image paths."""
        if not submission.pdf_image_paths:
            return None

        paths = submission.pdf_image_paths[:self.max_images]
        return [str(p) for p in paths]


class ImagesWithTitleFormatter(ImagesOnlyFormatter):
    """
    Images with only the paper title provided.

    A middle ground between pure images and text+images,
    providing minimal textual context.
    """

    def format_content(self, submission: SubmissionData) -> str:
        """Format with title and image placeholders."""
        if not submission.pdf_image_paths:
            raise ValueError(
                f"ImagesWithTitleFormatter requires images but submission "
                f"{submission.submission_id} has no PDF image paths"
            )

        num_images = min(len(submission.pdf_image_paths), self.max_images)
        image_tokens = " ".join(["<image>"] * num_images)

        return (
            f"# {submission.title}\n\n"
            f"The following images show pages from this paper:\n\n"
            f"{image_tokens}"
        )


class FiguresOnlyFormatter(InputFormatter):
    """
    Format with only figure/diagram images extracted from the paper.

    This requires pre-extracted figure images rather than full pages.
    Useful for analyzing visual content quality.

    Note: This formatter expects figure images to be stored separately
    from page images, potentially in a 'figures' subdirectory.
    """

    modality = InputModality.IMAGES_ONLY
    requires_vl_model = True

    def __init__(
        self,
        figures_subdir: str = "figures",
        **kwargs,
    ):
        """
        Initialize figures-only formatter.

        Args:
            figures_subdir: Subdirectory name for figure images
            **kwargs: Additional arguments for InputFormatter
        """
        kwargs["include_reviews"] = False
        kwargs["include_markdown"] = False
        super().__init__(**kwargs)
        self.figures_subdir = figures_subdir

    def format_content(self, submission: SubmissionData) -> str:
        """Format with title, abstract, and figure placeholders."""
        images = self.get_images(submission)
        if not images:
            # Fall back to abstract only if no figures
            return (
                f"# {submission.title}\n\n"
                f"## Abstract\n{submission.abstract}\n\n"
                f"(No figures available for this paper)"
            )

        num_images = len(images)
        image_tokens = " ".join(["<image>"] * num_images)

        return (
            f"# {submission.title}\n\n"
            f"## Abstract\n{submission.abstract}\n\n"
            f"## Figures and Diagrams\n"
            f"The following {num_images} images show figures from this paper:\n\n"
            f"{image_tokens}"
        )

    def get_images(self, submission: SubmissionData) -> Optional[List[str]]:
        """Get figure image paths from figures subdirectory."""
        if not submission.pdf_image_paths:
            return None

        # Try to find figures directory
        # Assuming pdf_image_paths[0] is like .../submission_id/redacted_pdf_img_content/page_1.png
        if submission.pdf_image_paths:
            first_path = submission.pdf_image_paths[0]
            parent = first_path.parent.parent  # Go up to submission directory
            figures_dir = parent / self.figures_subdir

            if figures_dir.exists():
                figure_paths = sorted(figures_dir.glob("*.png"))
                if figure_paths:
                    return [str(p) for p in figure_paths[:self.max_images]]

        # Fall back to None if no figures found
        return None
