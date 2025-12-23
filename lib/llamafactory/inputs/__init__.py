"""Input formatters for different modalities."""

from .base import InputFormatter
from .text_only import (
    TextOnlyFormatter,
    AbstractOnlyFormatter,
    AbstractWithReviewsFormatter,
    FullTextFormatter,
    FullContextFormatter,
)
from .text_with_images import (
    TextWithImagesFormatter,
    AbstractWithPageImagesFormatter,
    ReviewsWithPageImagesFormatter,
    KeyPagesFormatter,
)
from .images_only import (
    ImagesOnlyFormatter,
    ImagesWithTitleFormatter,
    FiguresOnlyFormatter,
)

__all__ = [
    # Base
    "InputFormatter",
    # Text-only
    "TextOnlyFormatter",
    "AbstractOnlyFormatter",
    "AbstractWithReviewsFormatter",
    "FullTextFormatter",
    "FullContextFormatter",
    # Text with images
    "TextWithImagesFormatter",
    "AbstractWithPageImagesFormatter",
    "ReviewsWithPageImagesFormatter",
    "KeyPagesFormatter",
    # Images only
    "ImagesOnlyFormatter",
    "ImagesWithTitleFormatter",
    "FiguresOnlyFormatter",
]
