"""
Pydantic schemas for ICLR OpenReview data (2020-2026).

Schema Evolution:
- Reviews changed structure significantly across years
- Meta-reviews evolved from simple (2020-2022) to complex (2023+)
- 2025-2026 switched from string to integer scores
- 2020-2023: Meta-review IS the Decision note (/-/Decision invitation)
- 2024+: Meta-review is SEPARATE (/-/Meta_Review invitation)
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Any, Dict, Type, Union
import re


def _parse_score_prefix(value: Union[str, int]) -> int:
    """
    Extract integer prefix from score strings.

    Examples:
        "5: marginally below the acceptance threshold" -> 5
        "3 good" -> 3
        "3: You are fairly confident..." -> 3
        5 -> 5 (already int)
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.match(r'^(\d+)', value.strip())
        if match:
            return int(match.group(1))
    raise ValueError(f"Cannot parse score from: {value}")


# =============================================================================
# RATING SCALES (documented for reference)
# =============================================================================
# 2020: 1: Reject, 3: Weak Reject, 6: Weak Accept, 8: Accept
# 2021: 1-10 scale ("1: Trivial or wrong" to "10: Top 5%")
# 2022-2024: 1, 3, 5, 6, 8, 10 scale (recommendation/rating as string)
# 2025-2026: Same scale but as integers
#
# Confidence: 1-5 scale (all years except 2020 which has none)
# Soundness/Presentation/Contribution (2024+): 1-4 scale


# =============================================================================
# SUBMISSION SCHEMA (consistent across years)
# =============================================================================
class Submission(BaseModel):
    """Paper submission metadata from OpenReview."""
    # Required fields
    id: str
    title: str
    abstract: str
    decision: str  # Final decision text (e.g., 'Accept (Poster)', 'Reject')
    # Optional fields
    authors: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    venue: Optional[str] = None  # Decision venue (e.g., "ICLR 2024 poster")
    pdf_url: Optional[str] = None
    # Timestamps
    created_date: Optional[int] = None  # Unix timestamp
    modified_date: Optional[int] = None
    # Additional metadata
    tldr: Optional[str] = None
    primary_area: Optional[str] = None


# =============================================================================
# REVIEW SCHEMAS BY YEAR
# =============================================================================
# All review schemas include response count fields computed at parse time:
#   - number_of_author_responses: int
#   - number_of_reviewer_responses_to_author: int

class Review2020(BaseModel):
    """
    ICLR 2020 Review Schema

    Rating scale: 1: Reject, 3: Weak Reject, 6: Weak Accept, 8: Accept
    Unique fields: experience_assessment, review_assessment:* fields
    """
    rating: str = Field(..., description="Rating with description (e.g., '6: Weak Accept')")
    review: str = Field(..., description="Main review text")
    title: str = Field(default="", description="Review title (e.g., 'Official Blind Review #3')")
    experience_assessment: str = Field(..., description="Reviewer's expertise in the area")
    review_assessment_derivations: str = Field(
        ...,
        alias="review_assessment:_checking_correctness_of_derivations_and_theory",
        description="How thoroughly derivations were checked"
    )
    review_assessment_experiments: str = Field(
        ...,
        alias="review_assessment:_checking_correctness_of_experiments",
        description="How thoroughly experiments were checked"
    )
    review_assessment_reading: str = Field(
        ...,
        alias="review_assessment:_thoroughness_in_paper_reading",
        description="How thoroughly paper was read"
    )
    # Response counts (computed from Official_Comment replies)
    number_of_author_responses: int = Field(default=0, description="Author replies to this review")
    number_of_reviewer_responses_to_author: int = Field(default=0, description="Reviewer replies to author")

    class Config:
        populate_by_name = True


class Review2021(BaseModel):
    """
    ICLR 2021 Review Schema

    Rating scale: 1-10 ("1: Trivial or wrong" to "10: Top 5% of accepted papers")
    Confidence scale: 1-5 (string with description)
    """
    rating: str = Field(..., description="Rating 1-10 with description")
    confidence: str = Field(..., description="Confidence 1-5 with description")
    review: str = Field(..., description="Main review text")
    title: str = Field(default="", description="Review title")
    # Response counts
    number_of_author_responses: int = Field(default=0)
    number_of_reviewer_responses_to_author: int = Field(default=0)


class Review2022(BaseModel):
    """
    ICLR 2022 Review Schema

    Recommendation scale: 1, 3, 5, 6, 8, 10 (string with description)
    Introduced: structured summary, correctness, novelty scores
    """
    recommendation: str = Field(..., description="Recommendation score (e.g., '6: marginally above...')")
    confidence: str = Field(..., description="Confidence 1-5 with description")
    summary_of_the_paper: str = Field(..., description="Paper summary by reviewer")
    main_review: str = Field(..., description="Main review with strengths/weaknesses")
    summary_of_the_review: str = Field(..., description="TL;DR of the review")
    correctness: str = Field(..., description="Correctness assessment 1-4")
    technical_novelty_and_significance: str = Field(..., description="Technical novelty 1-4")
    empirical_novelty_and_significance: str = Field(..., description="Empirical novelty 1-4")
    flag_for_ethics_review: List[str] = Field(default_factory=list, description="Ethics flags")
    details_of_ethics_concerns: Optional[str] = Field(None, description="Ethics concern details")
    # Response counts
    number_of_author_responses: int = Field(default=0)
    number_of_reviewer_responses_to_author: int = Field(default=0)


class Review2023(BaseModel):
    """
    ICLR 2023 Review Schema

    Similar to 2022 but:
    - 'strength_and_weaknesses' replaces 'main_review'
    - Added 'clarity,_quality,_novelty_and_reproducibility' field
    """
    recommendation: str
    confidence: str
    summary_of_the_paper: str
    strength_and_weaknesses: str = Field(..., description="Combined strengths and weaknesses section")
    clarity_quality_novelty_reproducibility: str = Field(
        ...,
        alias="clarity,_quality,_novelty_and_reproducibility",
        description="Clarity, quality, novelty, reproducibility assessment"
    )
    summary_of_the_review: str
    correctness: str
    technical_novelty_and_significance: str
    empirical_novelty_and_significance: str
    flag_for_ethics_review: List[str] = Field(default_factory=list)
    details_of_ethics_concerns: Optional[str] = None
    # Response counts
    number_of_author_responses: int = Field(default=0)
    number_of_reviewer_responses_to_author: int = Field(default=0)

    class Config:
        populate_by_name = True


class Review2024_2025_2026(BaseModel):
    """
    ICLR 2024-2026 Review Schema

    New structure with explicit sections and sub-scores.
    2024: Raw scores are strings (e.g., "5: marginally below...", "3 good") - parsed via validators
    2025-2026: Scores are already integers
    """
    rating: int = Field(..., ge=0, le=10, description="Rating 0-10")
    confidence: int = Field(..., ge=1, le=5, description="Confidence 1-5")
    summary: str = Field(..., description="Paper summary")
    strengths: str = Field(..., description="Paper strengths")
    weaknesses: str = Field(..., description="Paper weaknesses")
    questions: str = Field(..., description="Questions for authors")
    soundness: int = Field(..., ge=1, le=4, description="Soundness 1-4")
    presentation: int = Field(..., ge=1, le=4, description="Presentation 1-4")
    contribution: int = Field(..., ge=1, le=4, description="Contribution 1-4")
    flag_for_ethics_review: List[str] = Field(default_factory=list)
    details_of_ethics_concerns: Optional[str] = None
    code_of_conduct: str = Field(default="Yes", description="Code of conduct acknowledgment")
    # Response counts
    number_of_author_responses: int = Field(default=0)
    number_of_reviewer_responses_to_author: int = Field(default=0)

    # Validators to parse string scores to int (for 2024 data)
    @field_validator('rating', 'confidence', 'soundness', 'presentation', 'contribution', mode='before')
    @classmethod
    def parse_score(cls, v):
        return _parse_score_prefix(v)


# =============================================================================
# META-REVIEW SCHEMAS BY YEAR
# =============================================================================
# 2020-2023: Meta-review IS the Decision note (contains decision + AC's assessment)
# 2024+: Meta_Review is a SEPARATE invitation from Decision

class MetaReview2020_2022(BaseModel):
    """
    ICLR 2020-2022 Meta-Review Schema

    Invitation: /-/Decision (meta-review IS the decision note)
    Contains both the decision and AC's justification comment.
    The 'comment' field IS the meta-review text.
    """
    decision: str = Field(..., description="Decision (e.g., 'Accept (Poster)', 'Reject')")
    comment: str = Field(default="", description="AC meta-review/justification")
    title: str = Field(default="Paper Decision", description="Note title")


class MetaReview2023(BaseModel):
    """
    ICLR 2023 Meta-Review Schema

    Invitation: /-/Decision (meta-review IS the decision note)
    Complex structure with detailed AC justifications.
    Field: 'metareview:_summary,_strengths_and_weaknesses'
    """
    decision: str
    metareview_summary_strengths_weaknesses: str = Field(
        ...,
        alias="metareview:_summary,_strengths_and_weaknesses",
        description="AC summary of paper strengths and weaknesses"
    )
    justification_for_why_not_higher_score: str = Field(
        ..., description="Why not a higher score"
    )
    justification_for_why_not_lower_score: str = Field(
        ..., description="Why not a lower score"
    )
    summary_of_AC_reviewer_meeting: Optional[str] = Field(
        None,
        alias="summary_of_AC-reviewer_meeting",
        description="Summary of AC-reviewer discussion"
    )
    note_from_PC: Optional[str] = Field(None, description="Note from Program Chair")
    title: str = Field(default="Paper Decision")

    class Config:
        populate_by_name = True


class MetaReview2024(BaseModel):
    """
    ICLR 2024 Meta-Review Schema

    Invitation: /-/Meta_Review (SEPARATE from /-/Decision)
    Decision is in a separate note; this contains only AC's assessment.
    """
    metareview: str = Field(..., description="AC metareview text")
    justification_for_why_not_higher_score: str = Field(default="", description="Why not higher")
    justification_for_why_not_lower_score: str = Field(default="", description="Why not lower")


class MetaReview2025(BaseModel):
    """
    ICLR 2025 Meta-Review Schema

    Invitation: /-/Meta_Review (SEPARATE from /-/Decision)
    Simplified from 2024: no justification fields, added discussion comments.
    """
    metareview: str = Field(default="", description="AC metareview text")
    additional_comments_on_reviewer_discussion: Optional[str] = Field(
        None, description="AC comments on reviewer discussion"
    )


# =============================================================================
# YEAR-TO-SCHEMA MAPPING
# =============================================================================
REVIEW_SCHEMA_BY_YEAR: Dict[int, Type[BaseModel]] = {
    2020: Review2020,
    2021: Review2021,
    2022: Review2022,
    2023: Review2023,
    2024: Review2024_2025_2026,
    2025: Review2024_2025_2026,
    2026: Review2024_2025_2026,
}

# Meta-review schemas by year
# 2020-2023: Meta-review IS the Decision note (/-/Decision invitation)
# 2024+: Meta-review is SEPARATE (/-/Meta_Review invitation)
META_REVIEW_SCHEMA_BY_YEAR: Dict[int, Type[BaseModel]] = {
    2020: MetaReview2020_2022,
    2021: MetaReview2020_2022,
    2022: MetaReview2020_2022,
    2023: MetaReview2023,
    2024: MetaReview2024,
    2025: MetaReview2025,
    2026: MetaReview2025,  # Assuming same as 2025
}


# =============================================================================
# PARSER FUNCTIONS
# =============================================================================

def _get_value(val: Any) -> Any:
    """Extract value from OpenReview API v2 format (dict with 'value' key)."""
    if isinstance(val, dict) and 'value' in val:
        return val['value']
    return val


def _extract_reviewer_id(signatures: List[str]) -> Optional[str]:
    """
    Extract reviewer identifier from signatures.

    Examples:
        ['ICLR.cc/2020/Conference/Paper1130/AnonReviewer3'] -> 'AnonReviewer3'
        ['ICLR.cc/2024/Conference/Submission1909/Reviewer_AG4r'] -> 'Reviewer_AG4r'
    """
    if not signatures:
        return None
    sig = signatures[0]
    parts = sig.split('/')
    for part in reversed(parts):
        if 'Reviewer' in part or 'AnonReviewer' in part:
            return part
    return None


def compute_response_counts(
    review_id: str,
    review_signatures: List[str],
    all_comments: List[Dict]
) -> tuple[int, int]:
    """
    Compute author response and reviewer follow-up counts for a review.

    Args:
        review_id: The ID of the review
        review_signatures: Signatures of the review (to identify the reviewer)
        all_comments: List of all Official_Comment replies from the submission

    Returns:
        Tuple of (number_of_author_responses, number_of_reviewer_responses_to_author)

    Comment chain structure:
        Review (ID: ryl4xqCRKH, sig: AnonReviewer3)
          └── Author Comment (replyto: ryl4xqCRKH, sig: Authors) <- author_response
                └── Reviewer Comment (replyto: author_comment_id, sig: AnonReviewer3) <- reviewer_response
    """
    reviewer_id = _extract_reviewer_id(review_signatures)

    # Build comment lookup by ID
    comment_by_id = {c.get('id', ''): c for c in all_comments}

    # Find author responses to this review
    author_response_ids = set()
    for comment in all_comments:
        sig = str(comment.get('signatures', []))
        replyto = comment.get('replyto', '')

        # Is this an author comment replying to this review?
        if 'Authors' in sig and replyto == review_id:
            author_response_ids.add(comment.get('id', ''))

    number_of_author_responses = len(author_response_ids)

    # Find reviewer responses to author comments
    number_of_reviewer_responses = 0
    if reviewer_id and author_response_ids:
        for comment in all_comments:
            sig = str(comment.get('signatures', []))
            replyto = comment.get('replyto', '')

            # Is this the same reviewer replying to an author response?
            if reviewer_id in sig and replyto in author_response_ids:
                number_of_reviewer_responses += 1

    return number_of_author_responses, number_of_reviewer_responses


def parse_submission(year: int, submission, decision: str) -> Submission:
    """
    Parse OpenReview submission object to Submission schema.

    Args:
        year: Conference year
        submission: OpenReview submission object
        decision: Decision string (extracted from meta-review for 2020-2023,
                  or from separate Decision note for 2024+)
    """
    content = submission.content
    return Submission(
        id=submission.id,
        title=_get_value(content.get('title', '')),
        abstract=_get_value(content.get('abstract', '')),
        decision=decision,
        authors=_get_value(content.get('authors')),
        keywords=_get_value(content.get('keywords')),
        venue=getattr(submission, 'venue', None),
        pdf_url=_get_value(content.get('pdf')),
        created_date=getattr(submission, 'cdate', None),
        modified_date=getattr(submission, 'mdate', None),
        tldr=_get_value(content.get('TLDR') or content.get('tldr')),
        primary_area=_get_value(content.get('primary_area')),
    )


def parse_review(
    year: int,
    review_content: dict,
    review_id: str = "",
    review_signatures: List[str] = None,
    all_comments: List[Dict] = None
) -> BaseModel:
    """
    Parse review content dict to year-appropriate Review schema.

    Args:
        year: Conference year
        review_content: The content dict from the review
        review_id: ID of the review (for computing response counts)
        review_signatures: Signatures of the review (for identifying reviewer)
        all_comments: All Official_Comments from submission (for response counts)
    """
    schema = REVIEW_SCHEMA_BY_YEAR[year]
    # Normalize content values (handle API v2 format)
    normalized = {k: _get_value(v) for k, v in review_content.items()}

    # Compute response counts if we have the necessary data
    if review_id and all_comments is not None:
        author_responses, reviewer_responses = compute_response_counts(
            review_id,
            review_signatures or [],
            all_comments
        )
        normalized['number_of_author_responses'] = author_responses
        normalized['number_of_reviewer_responses_to_author'] = reviewer_responses

    return schema.model_validate(normalized)


def parse_meta_review(year: int, meta_review_content: dict) -> BaseModel:
    """
    Parse meta-review content dict to year-appropriate MetaReview schema.

    2020-2023: Meta-review IS the Decision note (/-/Decision invitation)
    2024+: Meta-review is SEPARATE (/-/Meta_Review invitation)
    """
    schema = META_REVIEW_SCHEMA_BY_YEAR[year]
    normalized = {k: _get_value(v) for k, v in meta_review_content.items()}
    return schema.model_validate(normalized)


def extract_reviews_from_submission(year: int, submission) -> List[BaseModel]:
    """Extract all reviews from a submission's replies, including response counts."""
    reviews = []
    if not hasattr(submission, 'details') or not submission.details:
        return reviews

    replies_key = 'directReplies' if 'directReplies' in submission.details else 'replies'
    replies = submission.details.get(replies_key, [])

    # Collect all Official_Comments for response counting
    all_comments = []
    for reply in replies:
        inv = reply.get('invitation', '') or str(reply.get('invitations', []))
        if 'Official_Comment' in inv:
            all_comments.append(reply)

    # Parse reviews with response counts
    for reply in replies:
        inv = reply.get('invitation', '') or str(reply.get('invitations', []))
        if 'Official_Review' in inv:
            content = reply.get('content', {})
            review_id = reply.get('id', '')
            review_signatures = reply.get('signatures', [])
            try:
                reviews.append(parse_review(
                    year,
                    content,
                    review_id=review_id,
                    review_signatures=review_signatures,
                    all_comments=all_comments
                ))
            except Exception:
                pass  # Skip malformed reviews
    return reviews


def extract_meta_review_from_submission(year: int, submission) -> Optional[BaseModel]:
    """
    Extract meta-review from a submission's replies.

    2020-2023: Looks for /-/Decision invitation (meta-review IS decision note)
    2024+: Looks for /-/Meta_Review invitation (separate from decision)
    """
    if not hasattr(submission, 'details') or not submission.details:
        return None

    replies_key = 'directReplies' if 'directReplies' in submission.details else 'replies'
    replies = submission.details.get(replies_key, [])

    for reply in replies:
        inv = reply.get('invitation', '') or str(reply.get('invitations', []))

        # 2020-2023: Meta-review is in the Decision note
        if year <= 2023 and 'Decision' in inv:
            content = reply.get('content', {})
            return parse_meta_review(year, content)

        # 2024+: Meta-review is separate from Decision
        if year >= 2024 and 'Meta_Review' in inv and 'Decision' not in inv:
            content = reply.get('content', {})
            return parse_meta_review(year, content)

    return None


def extract_decision_from_submission(year: int, submission) -> Optional[str]:
    """
    Extract decision string from a submission's replies.

    2020-2023: Decision is in the meta-review note (/-/Decision)
    2024+: Decision is in a separate /-/Decision note

    Returns the decision string or None if not found.
    """
    if not hasattr(submission, 'details') or not submission.details:
        return None

    replies_key = 'directReplies' if 'directReplies' in submission.details else 'replies'
    replies = submission.details.get(replies_key, [])

    for reply in replies:
        inv = reply.get('invitation', '') or str(reply.get('invitations', []))
        if 'Decision' in inv:
            content = reply.get('content', {})
            return _get_value(content.get('decision'))

    return None
