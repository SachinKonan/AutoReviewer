"""
Submission dataclass and loader for normalized OpenReview data.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pickle


@dataclass
class Submission:
    # Core identifiers
    id: str
    forum: str
    number: int
    conference: str  # "ICLR"
    year: int

    # Content
    title: str
    abstract: str
    tldr: str
    keywords: list[str]
    primary_area: str

    # Authors
    authors: list[str]
    author_ids: list[str]

    # PDF
    pdf_path: str  # local path
    pdf_url: str   # OpenReview URL

    # Decision (required)
    decision: str  # Accept/Reject/Withdrawn/Unknown
    venue: str

    # Timestamps
    creation_date: datetime
    modification_date: datetime


def _get_content_value(content: dict, key: str, default=""):
    """Extract value from content, handling API v1 vs v2 differences."""
    val = content.get(key)
    if val is None:
        return default
    if isinstance(val, dict):
        return val.get('value', default)
    return val


def _get_tldr(content: dict) -> str:
    """Get TLDR from various field names."""
    for key in ['TLDR', 'TL;DR', 'one-sentence_summary']:
        val = _get_content_value(content, key)
        if val:
            return val
    return ""


def _get_primary_area(content: dict) -> str:
    """Get primary area from various field names."""
    for key in ['primary_area', 'Please_choose_the_closest_area_that_your_submission_falls_into']:
        val = _get_content_value(content, key)
        if val:
            return val
    return ""


def _get_decision_from_details(details: dict, year: int) -> str:
    """Extract decision from details.replies."""
    if not details:
        return "Unknown"

    replies = details.get('replies', [])

    for reply in replies:
        # API v1 (2020-2023)
        if year <= 2023:
            invitation = reply.get('invitation', '')
            if 'Decision' in invitation:
                decision = reply.get('content', {}).get('decision', '')
                if decision:
                    return _normalize_decision(decision)
        # API v2 (2024+)
        else:
            invitations = reply.get('invitations', [])
            if any('Decision' in inv for inv in invitations):
                dec = reply.get('content', {}).get('decision', {})
                decision = dec.get('value') if isinstance(dec, dict) else dec
                if decision:
                    return _normalize_decision(decision)

    return "Unknown"


def _normalize_decision(decision: str) -> str:
    """Normalize decision to: Withdrawn | Desk Reject | Reject | Accept (Poster) | Accept (Spotlight) | Accept (Oral)"""
    decision_lower = decision.lower()

    # Check for oral/talk first (most specific)
    if 'oral' in decision_lower or 'talk' in decision_lower or 'notable-top-5%' in decision_lower:
        return "Accept (Oral)"
    # Then spotlight
    elif 'spotlight' in decision_lower or 'notable-top-25%' in decision_lower:
        return "Accept (Spotlight)"
    # Then poster (generic accept)
    elif 'poster' in decision_lower or ('accept' in decision_lower and 'reject' not in decision_lower):
        return "Accept (Poster)"
    # Desk reject
    elif 'desk' in decision_lower and 'reject' in decision_lower:
        return "Desk Reject"
    # Regular reject
    elif 'reject' in decision_lower:
        return "Reject"
    # Withdrawn
    elif 'withdraw' in decision_lower:
        return "Withdrawn"

    return "Unknown"


def _get_decision_from_venue(venue: str) -> str:
    """Fallback: infer decision from venue string."""
    if not venue:
        return "Unknown"
    venue_lower = venue.lower()

    # Check for oral first
    if 'oral' in venue_lower or 'notable top 5%' in venue_lower:
        return "Accept (Oral)"
    # Then spotlight
    elif 'spotlight' in venue_lower or 'notable top 25%' in venue_lower:
        return "Accept (Spotlight)"
    # Then poster
    elif 'poster' in venue_lower:
        return "Accept (Poster)"
    # Desk reject
    elif 'desk reject' in venue_lower:
        return "Desk Reject"
    # Regular reject
    elif 'reject' in venue_lower:
        return "Reject"
    # Withdrawn
    elif 'withdraw' in venue_lower:
        return "Withdrawn"

    return "Unknown"


def load_submission(note, year: int, pdf_base_path: Path = None) -> Submission:
    """
    Convert an OpenReview Note object to a Submission.

    Args:
        note: OpenReview Note object (v1 or v2)
        year: Conference year
        pdf_base_path: Base path for PDF files (e.g., data/full_run/pdfs)
    """
    content = note.content

    # Get decision - try details first, then venue
    decision = _get_decision_from_details(note.details, year)
    venue = _get_content_value(content, 'venue', "")

    if decision == "Unknown" and venue:
        decision = _get_decision_from_venue(venue)

    # PDF paths
    pdf_url_path = _get_content_value(content, 'pdf', "")
    pdf_url = f"https://openreview.net{pdf_url_path}" if pdf_url_path else ""

    # Local PDF path
    pdf_path = ""
    if pdf_base_path:
        # Check for downloaded PDF (format: {id}_{download_id}.pdf or {id}_.pdf)
        year_dir = pdf_base_path / str(year)
        if year_dir.exists():
            for f in year_dir.iterdir():
                if f.name.startswith(note.id) and f.suffix == '.pdf':
                    pdf_path = str(f)
                    break

    # Timestamps
    cdate = datetime.fromtimestamp(note.cdate / 1000) if note.cdate else datetime.min
    mdate = datetime.fromtimestamp(note.mdate / 1000) if note.mdate else datetime.min

    return Submission(
        id=note.id,
        forum=note.forum,
        number=note.number or 0,
        conference="ICLR",
        year=year,
        title=_get_content_value(content, 'title', ""),
        abstract=_get_content_value(content, 'abstract', ""),
        tldr=_get_tldr(content),
        keywords=_get_content_value(content, 'keywords', []) or [],
        primary_area=_get_primary_area(content),
        authors=_get_content_value(content, 'authors', []) or [],
        author_ids=_get_content_value(content, 'authorids', []) or [],
        pdf_path=pdf_path,
        pdf_url=pdf_url,
        decision=decision,
        venue=venue,
        creation_date=cdate,
        modification_date=mdate,
    )


def load_submissions_from_pickle(
    pickle_path: str | Path,
    year: int,
    pdf_base_path: str | Path = None
) -> list[Submission]:
    """
    Load all submissions from a pickle file.

    Args:
        pickle_path: Path to get_all_notes_{year}.pickle
        year: Conference year
        pdf_base_path: Base path for PDFs (optional)

    Returns:
        List of Submission objects
    """
    pickle_path = Path(pickle_path)
    pdf_base_path = Path(pdf_base_path) if pdf_base_path else None

    with open(pickle_path, 'rb') as f:
        notes = pickle.load(f)

    return [load_submission(note, year, pdf_base_path) for note in notes]


def load_all_submissions(
    data_dir: str | Path = "data/full_run",
    years: list[int] = None
) -> list[Submission]:
    """
    Load submissions for all years.

    Args:
        data_dir: Directory containing pickle files and pdfs/
        years: List of years to load (default: 2020-2026)

    Returns:
        List of all Submission objects
    """
    data_dir = Path(data_dir)
    years = years or [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    pdf_base = data_dir / "pdfs"

    all_submissions = []
    for year in years:
        pickle_path = data_dir / f"get_all_notes_{year}.pickle"
        if pickle_path.exists():
            subs = load_submissions_from_pickle(pickle_path, year, pdf_base)
            all_submissions.extend(subs)

    return all_submissions


if __name__ == "__main__":
    # Test loading
    subs = load_all_submissions()
    print(f"Loaded {len(subs)} submissions")

    # Stats
    from collections import Counter
    by_year = Counter(s.year for s in subs)
    by_decision = Counter(s.decision for s in subs)

    print("\nBy year:")
    for year in sorted(by_year):
        print(f"  {year}: {by_year[year]}")

    print("\nBy decision:")
    for dec, count in by_decision.most_common():
        print(f"  {dec}: {count}")

    # Sample
    print("\nSample submission:")
    s = subs[0]
    print(f"  id: {s.id}")
    print(f"  title: {s.title[:60]}...")
    print(f"  decision: {s.decision}")
    print(f"  tldr: {s.tldr[:60]}..." if s.tldr else "  tldr: (none)")
