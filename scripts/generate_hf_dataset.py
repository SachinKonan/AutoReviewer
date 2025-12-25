#!/usr/bin/env python3
"""
Generate HuggingFace Dataset from ICLR papers data.

Creates a dataset with paper PDFs, markdown, images, reviews, and metadata.
Filters to valid submissions (has abstract, valid images, not withdrawn/rejected).

Usage:
    python generate_hf_dataset.py
    python generate_hf_dataset.py --year 2024  # Single year
    python generate_hf_dataset.py --dry-run    # Test on 10 samples

================================================================================
DATASET STATISTICS (Images Column)
================================================================================

Pipeline breakdown showing submissions at each stage:

Year   Total   Review   HasPDF  MinerU  ValidMD  HasAbsRef  Redacted  Images  VldRev
------------------------------------------------------------------------------------
2020   2213    2213     2213    2211    2211     2193       2193      2192    2192
2021   2594    2594     2594    2593    2593     2505       2504      2504    2504
2022   2617    2617     2617    2617    2617     2597       2597      2597    2597
2023   3792    3792     3792    3792    3792     3769       3769      3769    3769
2024   7404    5780     5780    5779    5779     5748       5746      5746    5746
2025   11672   8727     8727    8727    8727     8303       8294      8294    8293
2026   19605   15948    15948   15946   15946    15146      15140     15140   0
------------------------------------------------------------------------------------
Total  49897   41671    41671   41665   41665    40261      40243     40242   25101

Column definitions:
- Total: All submissions in OpenReview
- Review: Reviewable (excludes Withdrawn, Desk Reject)
- HasPDF: Has downloadable PDF
- MinerU: Successfully processed by MinerU PDF parser
- ValidMD: MinerU output has valid markdown
- HasAbsRef: Has abstract and references sections detected
- Redacted: Successfully redacted (removed identifying info)
- Images: Has valid page images (count matches page_end + 1) (THIS DATASET)
- VldRev: Has valid reviews (>=1 review + meta-review, 2026 has no reviews yet)

This dataset contains the "Images" column: ~40,242 submissions across 2020-2026.

================================================================================
SCHEMA
================================================================================

Each row contains:

IDENTIFIERS:
- submission_id: str          OpenReview submission ID
- year: int                   Conference year (2020-2026)
- openreview_link: str        https://openreview.net/forum?id={submission_id}
- pdf_download_link: str      https://openreview.net{pdf_path} (correct version)
- title: str                  Paper title

ABSTRACTS:
- original_abstract: str      Original abstract from OpenReview
- no_github_abstract: str     Cleaned abstract (GitHub links removed)

PAPER CONTENT:
- original_md_path: str       Path to raw markdown from MinerU parser
- clean_md_path: str          Path to cleaned/normalized markdown
- images_in_clean_md: str     JSON dict[filename, full_path] - images referenced in clean_md
- clean_pdf_img_paths: str    JSON list[full_path] - redacted PDF page images

REVIEWS:
- original_reviews: str       JSON list of original review dicts from OpenReview
- normalized_reviews: str     JSON list of normalized review JSONs (null if incomplete)
- original_metareview: str    JSON of original meta-review (null for 2026)
- normalized_metareview: str  JSON of normalized meta-review (null if not available)

METADATA:
- technical_indicators: str   JSON with decision info:
                              {
                                "binary_decision": "accept" | "reject",
                                "specific_decision": "reject" | "poster" | "spotlight" | "oral",
                                "citations": int,
                                "ratings": [int, ...]  # from original reviews
                              }
- metadata_of_changes: str    JSON of _meta.json (what was modified during normalization)
- submission_json: str        JSON of full Submission dataclass
- content_list_json: str      JSON of MinerU content_list.json

LOCAL PATHS (all full absolute paths):
- _pdf_path: str              Path to PDF file
- _mineru_path: str           Path to MinerU output directory
- _normalized_path: str       Path to normalized output directory

================================================================================
NOTES
================================================================================

- 2026 submissions have no reviews/meta-reviews yet (normalized_reviews = null)
- normalized_reviews is only filled if ALL reviews for a submission are normalized
- normalized_metareview can be null independently
- All image/path fields contain full absolute paths
- Ratings extracted using year-specific logic:
  - 2020, 2021, 2024-2026: 'rating' field
  - 2022, 2023: 'recommendation' field
"""

import sys
from pathlib import Path

# Add parent directory to path for imports when running from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict

import pandas as pd
from tqdm import tqdm

from lib.schemas import (
    _parse_score_prefix,
    extract_meta_review_from_submission,
    extract_reviews_from_submission,
)
from lib.submission import load_submission
from lib.citations import load_citations
from lib.utils import (
    build_mineru_index,
    build_normalized_index,
    find_mineru_by_prefix,
    load_raw_notes,
    DEFAULT_YEARS,
    EXCLUDED_DECISIONS,
)


# =============================================================================
# ABSTRACT CLEANING
# =============================================================================

# Pattern to match git-related URLs (github, gitlab, etc.)
GIT_URL_PATTERN = re.compile(r'https:\s*//.*git', re.IGNORECASE)


def remove_git_sentences(abstract: str) -> str:
    """Remove sentences containing any git-related URL (github, gitlab, etc.)."""
    if not abstract:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', abstract)
    filtered = [s for s in sentences if not GIT_URL_PATTERN.search(s)]
    return ' '.join(filtered).strip()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pdf_manifest(manifest_path: Path = None) -> dict:
    """Load PDF manifest into dict keyed by submission_id."""
    if manifest_path is None:
        manifest_path = Path("data/pdf_manifest.csv")

    manifest = {}
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest[row['submission_id']] = {
                'pdf_path': row['pdf_path'],
                'pdf_source': row['pdf_source'],
                'download_id': row['download_id'],
            }
    return manifest


def load_normalized_reviews(reviews_dir: Path = None) -> dict:
    """Load normalized reviews grouped by submission_id -> {(type, idx): json}."""
    if reviews_dir is None:
        reviews_dir = Path("data/normalized_reviews")

    by_sub = defaultdict(dict)
    for batch_dir in reviews_dir.glob("batch_*"):
        parquet_path = batch_dir / "normalized.parquet"
        if parquet_path.exists() and parquet_path.is_dir():
            # Ray Data saves as directory with multiple part files
            for part_file in parquet_path.glob("*.parquet"):
                df = pd.read_parquet(part_file)
                for _, row in df.iterrows():
                    if row['normalized_json']:  # Only non-null
                        by_sub[row['submission_id']][(row['type'], row['review_index'])] = row['normalized_json']
        elif parquet_path.exists() and parquet_path.is_file():
            # Single parquet file
            df = pd.read_parquet(parquet_path)
            for _, row in df.iterrows():
                if row['normalized_json']:  # Only non-null
                    by_sub[row['submission_id']][(row['type'], row['review_index'])] = row['normalized_json']
    return dict(by_sub)


# =============================================================================
# HELPERS
# =============================================================================

def get_rating(review, year: int) -> int:
    """Extract integer rating using existing _parse_score_prefix."""
    if year in (2020, 2021, 2024, 2025, 2026):
        raw = getattr(review, 'rating', None)
    else:  # 2022, 2023
        raw = getattr(review, 'recommendation', None)

    if raw is None:
        return 0
    try:
        return _parse_score_prefix(raw)
    except (ValueError, TypeError):
        return 0


def build_technical_indicators(submission, original_reviews, year: int, id_lookup: dict) -> dict:
    """Build technical indicators dict."""
    decision = submission.decision

    # Binary decision
    binary = "accept" if "Accept" in decision else "reject"

    # Specific decision
    if "Oral" in decision:
        specific = "oral"
    elif "Spotlight" in decision:
        specific = "spotlight"
    elif "Accept" in decision:
        specific = "poster"
    else:
        specific = "reject"

    # Ratings from reviews
    ratings = [get_rating(r, year) for r in original_reviews]

    # Citation count from Google Scholar (0 if not found)
    citations = id_lookup.get(submission.id, 0)

    return {
        "binary_decision": binary,
        "specific_decision": specific,
        "citations": citations,
        "ratings": ratings,
    }


def get_image_paths(img_dir: Path) -> dict:
    """Get image paths as dict[filename, full_path_str]."""
    images = {}
    if img_dir.exists() and img_dir.is_dir():
        # Follow symlink if needed
        real_dir = img_dir.resolve() if img_dir.is_symlink() else img_dir
        if real_dir.exists():
            for f in sorted(real_dir.iterdir()):
                if f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    images[f.name] = str(f.resolve())
    return images


def get_image_path_list(img_dir: Path) -> list:
    """Get sorted list of full image paths."""
    paths = []
    if img_dir.exists() and img_dir.is_dir():
        # Follow symlink if needed
        real_dir = img_dir.resolve() if img_dir.is_symlink() else img_dir
        if real_dir.exists():
            for f in sorted(real_dir.iterdir()):
                if f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    paths.append(str(f.resolve()))
    return paths


# =============================================================================
# ROW BUILDING
# =============================================================================

def build_row(
    submission_id: str,
    year: int,
    note,
    mineru_path: Path,
    norm_reviews_by_sub: dict,
    pdf_manifest: dict,
    data_dir: Path,
    id_lookup: dict,
) -> dict:
    """Build a single dataset row for a submission."""
    normalized_path = data_dir / "full_run" / "normalized" / str(year) / submission_id

    # Load meta.json
    meta_path = normalized_path / f"{submission_id}_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    # Markdown paths (store paths instead of content for compactness)
    clean_md_file = normalized_path / f"{submission_id}.md"
    clean_md_path_str = str(clean_md_file.resolve()) if clean_md_file.exists() else ""

    original_md_path_str = ""
    if mineru_path:
        # mineru_path is content_list.json: .../folder_name/vlm/folder_name_content_list.json
        folder_name = mineru_path.parent.parent.name  # The folder containing vlm/
        md_file = mineru_path.parent / f"{folder_name}.md"  # .../vlm/folder_name.md
        if md_file.exists():
            original_md_path_str = str(md_file.resolve())

    # Image paths (full paths)
    images_in_clean_md = get_image_paths(normalized_path / "images")
    clean_pdf_img_paths = get_image_path_list(normalized_path / "redacted_pdf_img_content")

    # Original reviews from pickle
    original_reviews = extract_reviews_from_submission(year, note)
    original_reviews_json = [r.model_dump() for r in original_reviews]

    # Original meta review
    original_meta = None
    if year < 2026:
        try:
            original_meta = extract_meta_review_from_submission(year, note)
        except Exception:
            pass

    # Normalized reviews: only fill if ALL reviews are non-null
    sub_norm = norm_reviews_by_sub.get(submission_id, {})
    reviews_list = [sub_norm.get(('review', i)) for i in range(len(original_reviews))]
    normalized_reviews = reviews_list if (reviews_list and all(r is not None for r in reviews_list)) else None
    normalized_metareview = sub_norm.get(('meta', -1))  # Meta reviews have index -1

    # Submission dataclass
    submission = load_submission(note, year)

    # Technical indicators
    tech_indicators = build_technical_indicators(submission, original_reviews, year, id_lookup)

    # PDF info from manifest
    pdf_info = pdf_manifest.get(submission_id, {})
    pdf_path = pdf_info.get('pdf_path', '')

    # Content list JSON
    content_list_path = normalized_path / f"{submission_id}_content_list.json"
    content_list = []
    if content_list_path.exists():
        try:
            with open(content_list_path) as f:
                content_list = json.load(f)
        except Exception:
            pass

    return {
        "submission_id": submission_id,
        "year": year,
        "openreview_link": f"https://openreview.net/forum?id={submission_id}",
        "pdf_download_link": f"https://openreview.net{pdf_path}" if pdf_path else "",
        "title": submission.title,
        "original_abstract": submission.abstract,
        "no_github_abstract": remove_git_sentences(submission.abstract),
        "original_md_path": original_md_path_str,
        "clean_md_path": clean_md_path_str,
        "images_in_clean_md": json.dumps(images_in_clean_md),  # JSON dict[filename, full_path]
        "clean_pdf_img_paths": json.dumps(clean_pdf_img_paths),  # JSON list[full_path]
        "original_reviews": json.dumps(original_reviews_json),
        "normalized_reviews": json.dumps(normalized_reviews) if normalized_reviews else None,
        "original_metareview": json.dumps(original_meta.model_dump()) if original_meta else None,
        "normalized_metareview": json.dumps(normalized_metareview) if normalized_metareview else None,
        "technical_indicators": json.dumps(tech_indicators),
        "metadata_of_changes": json.dumps(meta),
        "submission_json": json.dumps(asdict(submission), default=str),
        "content_list_json": json.dumps(content_list),
        "_pdf_path": str(Path(meta.get('original_pdf_path', '')).resolve()) if meta.get('original_pdf_path') else "",
        "_mineru_path": str(mineru_path.resolve()) if mineru_path else "",
        "_normalized_path": str(normalized_path.resolve()),
    }


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_year(
    year: int,
    notes_by_id: dict,
    normalized_index: dict,
    mineru_index: dict,
    mineru_sorted_keys: list,
    fixed_keys: list,
    fixed_index: dict,
    fixed_pt2_keys: list,
    fixed_pt2_index: dict,
    norm_reviews_by_sub: dict,
    pdf_manifest: dict,
    data_dir: Path,
    id_lookup: dict,
    limit: int = None,
    num_workers: int = 32,
) -> list:
    """Process all submissions for a given year with threading."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Get submissions for this year
    year_notes = [(note_id, note) for note_id, (note, note_year) in notes_by_id.items() if note_year == year]

    if limit:
        year_notes = year_notes[:limit]

    # Pre-filter submissions (fast, in-memory checks)
    valid_submissions = []
    for submission_id, note in tqdm(year_notes, desc=f"Year {year} filter"):
        # Check if normalized exists
        if submission_id not in normalized_index:
            continue

        norm_info = normalized_index[submission_id]
        meta = norm_info.get('meta')
        if not meta:
            continue

        # Filter criteria
        # 1. has_abstract
        if not meta.get('has_abstract', False):
            continue

        # 2. Valid images
        page_end = meta.get('page_end', 0)
        image_count = norm_info.get('image_count', 0)
        if image_count != page_end + 1 or page_end < 0:
            continue

        # 3. Not excluded decision
        submission = load_submission(note, year)
        if submission.decision in EXCLUDED_DECISIONS:
            continue

        # Find MinerU path
        mineru_result = find_mineru_by_prefix(
            submission_id,
            mineru_sorted_keys,
            mineru_index,
            year=year,
            fixed_keys=fixed_keys,
            fixed_index=fixed_index,
            fixed_pt2_keys=fixed_pt2_keys,
            fixed_pt2_index=fixed_pt2_index,
        )
        mineru_path = mineru_result[1] if mineru_result else None

        valid_submissions.append((submission_id, note, mineru_path))

    # Process submissions in parallel (I/O-heavy build_row)
    def process_submission(args):
        submission_id, note, mineru_path = args
        try:
            return build_row(
                submission_id,
                year,
                note,
                mineru_path,
                norm_reviews_by_sub,
                pdf_manifest,
                data_dir,
                id_lookup,
            )
        except Exception as e:
            print(f"Error processing {submission_id}: {e}")
            return None

    rows = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_submission, args): args for args in valid_submissions}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Year {year} build"):
            result = future.result()
            if result:
                rows.append(result)

    return rows


def create_dataset(
    output_dir: Path,
    years: list = None,
    data_dir: Path = None,
    limit: int = None,
):
    """Create the full dataset as a single combined Dataset.

    Processes all years and saves as one dataset to disk.
    """
    from datasets import Dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if data_dir is None:
        data_dir = Path("data")
    if years is None:
        years = DEFAULT_YEARS

    print("Loading indices...")

    # Build MinerU index
    mineru_dir = data_dir / "full_run" / "md_mineru"
    mineru_index, fixed_index, fixed_pt2_index = build_mineru_index(mineru_dir)
    mineru_sorted_keys = sorted(mineru_index.keys())
    fixed_keys = sorted(fixed_index.keys())
    fixed_pt2_keys = sorted(fixed_pt2_index.keys())
    print(f"  MinerU: {len(mineru_index)} folders")

    # Build normalized index
    normalized_index = build_normalized_index(data_dir / "full_run" / "normalized")
    print(f"  Normalized: {len(normalized_index)} submissions")

    # Load notes
    notes_by_id = {}
    for year in years:
        pkl_path = data_dir / "full_run" / f"get_all_notes_{year}.pickle"
        if pkl_path.exists():
            notes = load_raw_notes(pkl_path)
            for note in notes:
                notes_by_id[note.id] = (note, year)
    print(f"  Notes: {len(notes_by_id)} submissions")

    # Load PDF manifest
    pdf_manifest = load_pdf_manifest(data_dir / "pdf_manifest.csv")
    print(f"  PDF manifest: {len(pdf_manifest)} entries")

    # Load normalized reviews
    norm_reviews_by_sub = load_normalized_reviews(data_dir / "normalized_reviews_2")
    print(f"  Normalized reviews: {len(norm_reviews_by_sub)} submissions")

    # Load citations (ID lookup)
    citations_db = data_dir / "citations.db"
    id_lookup, _ = load_citations(citations_db)
    print(f"  Citations: {len(id_lookup)} papers")

    # Process all years and combine into one dataset
    all_rows = []
    year_counts = {}
    for year in years:
        year_rows = process_year(
            year,
            notes_by_id,
            normalized_index,
            mineru_index,
            mineru_sorted_keys,
            fixed_keys,
            fixed_index,
            fixed_pt2_keys,
            fixed_pt2_index,
            norm_reviews_by_sub,
            pdf_manifest,
            data_dir,
            id_lookup=id_lookup,
            limit=limit,
        )
        print(f"  {year}: {len(year_rows)} rows")
        year_counts[year] = len(year_rows)
        all_rows.extend(year_rows)

    print(f"\nTotal: {len(all_rows)} rows")

    # Create and save single combined dataset
    print(f"\nSaving dataset to {output_dir}...")
    dataset = Dataset.from_list(all_rows)
    dataset.save_to_disk(output_dir)
    print(f"Done!")

    return len(all_rows), year_counts


def main():
    parser = argparse.ArgumentParser(description='Generate HuggingFace Dataset from ICLR papers')
    parser.add_argument('--year', type=int, help='Process specific year only')
    parser.add_argument('--dry-run', action='store_true', help='Test on 10 samples per year')
    parser.add_argument('--output', type=str, default='data/hf_dataset', help='Output directory')
    parser.add_argument('--upload', action='store_true', help='Upload to HuggingFace after generation')
    parser.add_argument('--upload-only', action='store_true', help='Skip generation, just upload existing dataset')
    parser.add_argument('--repo-id', type=str, help='HuggingFace repo ID (e.g., username/iclr-reviews-2020-2026)')
    args = parser.parse_args()

    # Validate upload arguments
    if args.upload or args.upload_only:
        if not args.repo_id:
            parser.error("--repo-id is required for upload")

    years = [args.year] if args.year else DEFAULT_YEARS
    limit = 10 if args.dry_run else None
    output_path = Path(args.output)

    # Upload-only mode: load existing dataset
    if args.upload_only:
        from datasets import Dataset
        print("=" * 60)
        print("LOADING EXISTING DATASET")
        print(f"Path: {output_path}")
        print("=" * 60)
        ds = Dataset.load_from_disk(output_path)
        total = len(ds)
        # Count by year
        year_counts = {}
        for row in ds:
            y = row['year']
            year_counts[y] = year_counts.get(y, 0) + 1
        del ds
        print(f"Loaded {total:,} total rows")
    else:
        # Generate dataset
        print("=" * 60)
        print("GENERATING HUGGINGFACE DATASET")
        print(f"Years: {years}")
        print(f"Limit: {limit if limit else 'None'}")
        print(f"Output: {args.output}")
        print("=" * 60)

        total, year_counts = create_dataset(output_dir=args.output, years=years, limit=limit)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    for year in sorted(year_counts.keys()):
        count = year_counts[year]
        print(f"  {year}: {count:,} rows")
    print(f"  -----")
    print(f"  Total: {sum(year_counts.values()):,} rows")
    print("=" * 60)

    # Upload to HuggingFace
    if args.upload or args.upload_only:
        from datasets import Dataset, Features, Value

        print("\n" + "=" * 60)
        print(f"UPLOADING TO HUGGINGFACE: {args.repo_id}")
        print("=" * 60)

        # Define the canonical schema with all nullable string fields
        canonical_features = Features({
            'submission_id': Value('string'),
            'year': Value('int64'),
            'openreview_link': Value('string'),
            'pdf_download_link': Value('string'),
            'title': Value('string'),
            'original_abstract': Value('string'),
            'no_github_abstract': Value('string'),
            'original_md_path': Value('string'),
            'clean_md_path': Value('string'),
            'images_in_clean_md': Value('string'),
            'clean_pdf_img_paths': Value('string'),
            'original_reviews': Value('string'),
            'normalized_reviews': Value('string'),
            'original_metareview': Value('string'),
            'normalized_metareview': Value('string'),
            'technical_indicators': Value('string'),
            'metadata_of_changes': Value('string'),
            'submission_json': Value('string'),
            'content_list_json': Value('string'),
            '_pdf_path': Value('string'),
            '_mineru_path': Value('string'),
            '_normalized_path': Value('string'),
        })

        # Upload as single dataset
        ds = Dataset.load_from_disk(output_path)
        ds = ds.cast(canonical_features)

        print(f"  Uploading {len(ds):,} rows...")
        ds.push_to_hub(args.repo_id)

        print(f"\nUploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == '__main__':
    main()
