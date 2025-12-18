"""Generate HuggingFace parquet dataset from ICLR review data."""

import json
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import ray
from tqdm import tqdm

from lib.submission import load_submissions_from_pickle, Submission
from lib.utils import (
    load_raw_notes,
    build_md_folder_index,
    build_pdf_file_index,
    find_md_path,
    find_pdf_path,
    has_complete_md,
    EXCLUDED_DECISIONS,
    DEFAULT_YEARS,
)
from lib.schemas import (
    extract_reviews_from_submission,
    extract_meta_review_from_submission,
)
from lib.normalize import (
    clean_md_text,
    extract_sections,
    remove_urls_from_abstract,
)
from lib.citations import load_citations, add_citations_to_df

MAX_REVIEWS = 12


def extract_section_breakdown(clean_md: str) -> Dict[str, str]:
    """
    Extract sections from cleaned MD as dict mapping title -> content.

    Args:
        clean_md: Cleaned markdown text (output of clean_md_text)

    Returns:
        Dict mapping section title (e.g., "INTRODUCTION") to content
    """
    sections = extract_sections(clean_md)
    return {s.title: s.content.strip() for s in sections}


def is_complete_paper(
    sub: Submission,
    year: int,
    md_index: Dict[int, List[str]],
    md_dir: Path
) -> bool:
    """
    Check if paper qualifies as "complete" for the dataset.

    Complete papers must have:
    - Decision not in EXCLUDED_DECISIONS (Withdrawn, Desk Reject)
    - MD file exists with Abstract and References sections

    Args:
        sub: Submission object
        year: Conference year
        md_index: Pre-built MD folder index
        md_dir: Base MD directory

    Returns:
        True if paper is complete
    """
    # Check decision
    if sub.decision in EXCLUDED_DECISIONS:
        return False

    # Check MD has Abstract + References
    return has_complete_md(sub.id, year, md_index, md_dir)


def process_submission(
    note,  # Raw OpenReview note (from pickle)
    sub: Submission,
    year: int,
    md_index: Dict[int, List[str]],
    md_dir: Path,
    pdf_index: Dict[int, List[str]],
    pdf_dir: Path,
) -> Optional[Dict[str, Any]]:
    """
    Process a single submission into a dataset row.

    Args:
        note: Raw OpenReview note object (for review extraction)
        sub: Submission object (normalized)
        year: Conference year
        md_index: Pre-built MD folder index
        md_dir: Base MD directory
        pdf_index: Pre-built PDF file index
        pdf_dir: Base PDF directory

    Returns:
        Dict with submission, reviews, meta_review, md content, paths, or None if failed
    """
    # Get MD file path
    md_path_obj, _ = find_md_path(sub.id, year, md_index, md_dir)
    if not md_path_obj:
        return None

    try:
        raw_md = md_path_obj.read_text(encoding='utf-8')
    except Exception:
        return None

    # Clean MD
    clean_md = clean_md_text(raw_md)
    if clean_md is None:
        return None

    # Get PDF file path
    pdf_path_obj = find_pdf_path(sub.id, year, pdf_index, pdf_dir)

    # Extract reviews and meta-review from raw note
    reviews = extract_reviews_from_submission(year, note)
    meta_review = extract_meta_review_from_submission(year, note)

    # Build submission dict from Submission dataclass
    # Handle datetime.min (used as default) - convert to None
    created_ts = None
    if sub.creation_date and sub.creation_date > datetime(1970, 1, 1):
        created_ts = int(sub.creation_date.timestamp())
    modified_ts = None
    if sub.modification_date and sub.modification_date > datetime(1970, 1, 1):
        modified_ts = int(sub.modification_date.timestamp())

    submission_dict = {
        'id': sub.id,
        'title': sub.title,
        'abstract': remove_urls_from_abstract(sub.abstract),  # Anonymized (sentences with URLs removed)
        '_original_abstract': sub.abstract,  # Preserve original for reference
        'decision': sub.decision,
        'authors': sub.authors,
        'keywords': sub.keywords,
        'venue': sub.venue,
        'pdf_url': sub.pdf_url,
        'created_date': created_ts,
        'modified_date': modified_ts,
        'tldr': sub.tldr,
        'primary_area': sub.primary_area,
    }

    # Build row
    row = {
        "submission": submission_dict,
        "meta_review": meta_review.model_dump() if meta_review else None,
        "raw_md": raw_md,
        "clean_md": clean_md,
        "clean_md_sections": extract_section_breakdown(clean_md),
        "md_path": str(md_path_obj),
        "pdf_path": str(pdf_path_obj) if pdf_path_obj else None,
    }

    # Add reviews (pad to MAX_REVIEWS with None)
    for i in range(MAX_REVIEWS):
        review = reviews[i] if i < len(reviews) else None
        row[f"review_{i+1}"] = review.model_dump() if review else None

    return row


# Global variables for worker processes (set by initializer)
_worker_md_index = None
_worker_md_dir = None
_worker_pdf_index = None
_worker_pdf_dir = None


def _init_worker(md_index, md_dir, pdf_index, pdf_dir):
    """Initialize worker process with shared data."""
    global _worker_md_index, _worker_md_dir, _worker_pdf_index, _worker_pdf_dir
    _worker_md_index = md_index
    _worker_md_dir = md_dir
    _worker_pdf_index = pdf_index
    _worker_pdf_dir = pdf_dir


def _process_single(args: Tuple) -> Optional[Dict[str, Any]]:
    """Worker function for ProcessPoolExecutor."""
    note, sub, year = args
    return process_submission(
        note, sub, year,
        _worker_md_index, _worker_md_dir,
        _worker_pdf_index, _worker_pdf_dir
    )


def generate_year_dataset(
    year: int,
    data_dir: Path,
    md_dir: Path,
    pdf_dir: Path,
    output_dir: Path,
    md_index: Dict[int, List[str]],
    pdf_index: Dict[int, List[str]],
    verbose: bool = True,
    max_workers: int = 8,
    chunk_size: int = 1000,
    id_lookup: Optional[Dict[str, int]] = None,
    title_lookup: Optional[Dict[str, int]] = None,
) -> int:
    """
    Generate parquet for a single year.

    Args:
        year: Conference year (2020-2026)
        data_dir: Base data directory (contains get_all_notes_*.pickle)
        md_dir: Base MD directory (contains mds/{year}/)
        pdf_dir: Base PDF directory (contains pdfs/{year}/)
        output_dir: Output directory for parquet files
        md_index: Pre-built MD folder index
        pdf_index: Pre-built PDF file index
        verbose: Print progress
        max_workers: Number of threads for parallel processing
        chunk_size: Write to disk every N rows (default 1000)

    Returns:
        Number of complete papers processed
    """
    pkl_file = data_dir / f"get_all_notes_{year}.pickle"
    year_md_dir = md_dir / str(year)

    if not pkl_file.exists():
        if verbose:
            print(f"Skipping {year}: pickle file not found")
        return 0

    if not year_md_dir.exists():
        if verbose:
            print(f"Skipping {year}: MD directory not found")
        return 0

    # Load raw notes (for review extraction) and submissions (normalized)
    notes = load_raw_notes(pkl_file)
    subs = load_submissions_from_pickle(pkl_file, year)

    # Filter to complete papers first
    complete_pairs = [
        (note, sub) for note, sub in zip(notes, subs)
        if is_complete_paper(sub, year, md_index, md_dir)
    ]
    skipped = len(notes) - len(complete_pairs)

    if not complete_pairs:
        if verbose:
            print(f"No complete papers found for {year}")
        return 0

    # Prepare args for parallel processing (only pass lightweight args, indexes via initializer)
    args_list = [(note, sub, year) for note, sub in complete_pairs]

    # Create output directory for this year's chunks
    year_output_dir = output_dir / str(year)
    year_output_dir.mkdir(parents=True, exist_ok=True)

    # Clear any existing chunks
    for old_chunk in year_output_dir.glob("*.parquet"):
        old_chunk.unlink()

    # Process in parallel with ProcessPoolExecutor (CPU-bound due to regex in clean_md_text)
    # Use initializer to pass large indexes once per worker, not with each task
    rows = []
    total_saved = 0
    chunk_num = 0

    def _save_chunk():
        nonlocal rows, total_saved, chunk_num
        if rows:
            df = pd.DataFrame(rows)
            # Add citations if lookups provided
            if id_lookup is not None and title_lookup is not None:
                df = add_citations_to_df(df, id_lookup, title_lookup)
            chunk_path = year_output_dir / f"part_{chunk_num:04d}.parquet"
            df.to_parquet(chunk_path, index=False)
            total_saved += len(rows)
            chunk_num += 1
            rows = []

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(md_index, md_dir, pdf_index, pdf_dir)
    ) as executor:
        futures = {executor.submit(_process_single, args): args for args in args_list}

        if verbose:
            futures_iter = tqdm(as_completed(futures), total=len(futures), desc=f"Processing {year}")
        else:
            futures_iter = as_completed(futures)

        for future in futures_iter:
            try:
                row = future.result()
                if row:
                    rows.append(row)
                    # Write chunk when we hit chunk_size
                    if len(rows) >= chunk_size:
                        _save_chunk()
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
                if verbose:
                    print(f"  Error processing submission: {e}")

    # Save any remaining rows
    _save_chunk()

    if total_saved == 0:
        if verbose:
            print(f"No complete papers found for {year}")
        return 0

    if verbose:
        print(f"  Saved {total_saved} papers to {year_output_dir}/ ({chunk_num} chunks)")
        print(f"  Skipped {skipped} incomplete/failed papers")

    return total_saved


def process_submission_ray(item: dict) -> Optional[Dict[str, Any]]:
    """Ray-compatible wrapper for process_submission.

    Converts nested dicts to JSON strings for Arrow compatibility.
    """
    result = process_submission(
        note=item["note"],
        sub=item["sub"],
        year=item["year"],
        md_index=item["md_index"],
        md_dir=Path(item["md_dir"]),
        pdf_index=item["pdf_index"],
        pdf_dir=Path(item["pdf_dir"]),
    )

    if result is None:
        return None

    # Convert nested dicts to JSON strings for Arrow compatibility
    # This allows Ray to merge blocks without schema conflicts
    # Use json.dumps for all - None becomes "null" string for consistent typing
    result["submission"] = json.dumps(result["submission"])
    result["clean_md_sections"] = json.dumps(result["clean_md_sections"])
    result["meta_review"] = json.dumps(result["meta_review"])  # None -> "null"

    # Convert review dicts to JSON (None -> "null" for consistent schema)
    for i in range(1, 13):
        key = f"review_{i}"
        result[key] = json.dumps(result[key])  # None -> "null"

    return result


def merge_parquet_shards(shard_dir: Path, output_file: Path):
    """Merge Ray parquet shards into single file."""
    # Use pandas for merging - handles schema differences in dict columns better
    dfs = [pd.read_parquet(f) for f in sorted(shard_dir.glob("*.parquet"))]
    if not dfs:
        return
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_parquet(output_file, index=False)
    # Clean up shards directory
    shutil.rmtree(shard_dir)


def generate_year_dataset_ray(
    year: int,
    data_dir: Path,
    md_dir: Path,
    pdf_dir: Path,
    output_dir: Path,
    md_index: Dict[int, List[str]],
    pdf_index: Dict[int, List[str]],
    verbose: bool = True,
    num_cpus: int = 16,
    id_lookup: Optional[Dict[str, int]] = None,
    title_lookup: Optional[Dict[str, int]] = None,
) -> int:
    """
    Generate parquet for a single year using Ray Data.

    Args:
        year: Conference year (2020-2026)
        data_dir: Base data directory (contains get_all_notes_*.pickle)
        md_dir: Base MD directory (contains mds/{year}/)
        pdf_dir: Base PDF directory (contains pdfs/{year}/)
        output_dir: Output directory for parquet files
        md_index: Pre-built MD folder index
        pdf_index: Pre-built PDF file index
        verbose: Print progress
        num_cpus: Number of CPUs/partitions for parallel processing

    Returns:
        Number of complete papers processed
    """
    pkl_file = data_dir / f"get_all_notes_{year}.pickle"
    year_md_dir = md_dir / str(year)

    if not pkl_file.exists():
        if verbose:
            print(f"Skipping {year}: pickle file not found")
        return 0

    if not year_md_dir.exists():
        if verbose:
            print(f"Skipping {year}: MD directory not found")
        return 0

    # Initialize Ray if not already
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Load raw notes and submissions
    if verbose:
        print(f"Loading {year} data...")
    notes = load_raw_notes(pkl_file)
    subs = load_submissions_from_pickle(pkl_file, year)

    # Filter to complete papers and prepare items for Ray
    complete_items = [
        {
            "note": note,
            "sub": sub,
            "year": year,
            "md_index": md_index,
            "md_dir": str(md_dir),
            "pdf_index": pdf_index,
            "pdf_dir": str(pdf_dir),
        }
        for note, sub in zip(notes, subs)
        if is_complete_paper(sub, year, md_index, md_dir)
    ]
    skipped = len(notes) - len(complete_items)

    if not complete_items:
        if verbose:
            print(f"No complete papers found for {year}")
        return 0

    if verbose:
        print(f"Processing {len(complete_items)} papers with Ray ({num_cpus} partitions)...")

    # Create Ray Dataset
    ds = ray.data.from_items(complete_items)

    # Map processing function (automatically parallelized)
    ds = ds.map(process_submission_ray)

    # Filter out None results
    ds = ds.filter(lambda x: x is not None)

    # Repartition for parallel writes
    ds = ds.repartition(num_cpus)

    # Write parquet shards in parallel
    output_dir.mkdir(parents=True, exist_ok=True)
    year_output_dir = output_dir / str(year)
    if year_output_dir.exists():
        shutil.rmtree(year_output_dir)

    ds.write_parquet(str(year_output_dir))

    # Apply citations to each shard if lookups provided
    parquet_files = list(year_output_dir.glob("*.parquet"))
    if id_lookup is not None and title_lookup is not None:
        if verbose:
            print(f"  Adding citations to {len(parquet_files)} shards...")
        for f in parquet_files:
            df = pd.read_parquet(f)
            # Parse JSON strings back to dicts for citation lookup
            df['submission'] = df['submission'].apply(json.loads)
            df = add_citations_to_df(df, id_lookup, title_lookup)
            # Convert back to JSON for consistent format
            df['submission'] = df['submission'].apply(json.dumps)
            df.to_parquet(f, index=False)

    # Count rows from shards
    count = sum(len(pd.read_parquet(f)) for f in parquet_files)

    if verbose:
        print(f"  Saved {count} papers to {year_output_dir}/ ({len(parquet_files)} shards)")
        print(f"  Skipped {skipped} incomplete/failed papers")

    return count


def generate_all_datasets_ray(
    data_dir: str | Path = "data/full_run",
    md_dir: str | Path = None,
    pdf_dir: str | Path = None,
    output_dir: str | Path = None,
    years: List[int] = None,
    verbose: bool = True,
    num_cpus: int = 16,
    citations_db_path: str | Path = None,
) -> Dict[int, int]:
    """
    Generate parquet datasets for all years using Ray.

    Args:
        data_dir: Base data directory
        md_dir: MD directory (default: data_dir/mds)
        pdf_dir: PDF directory (default: data_dir/pdfs)
        output_dir: Output directory (default: data_dir/raw_hf_dump)
        years: Years to process (default: 2020-2026)
        verbose: Print progress
        num_cpus: Number of CPUs/partitions for parallel processing
        citations_db_path: Path to citations.db (optional, adds google_scholar_citations)

    Returns:
        Dict mapping year -> number of complete papers
    """
    data_dir = Path(data_dir)
    md_dir = Path(md_dir) if md_dir else data_dir / "mds"
    pdf_dir = Path(pdf_dir) if pdf_dir else data_dir / "pdfs"
    output_dir = Path(output_dir) if output_dir else data_dir / "raw_hf_dump"
    citations_db_path = Path(citations_db_path) if citations_db_path else None
    years = years or DEFAULT_YEARS

    if verbose:
        print(f"Generating HuggingFace datasets (Ray)")
        print(f"  Data directory: {data_dir}")
        print(f"  MD directory: {md_dir}")
        print(f"  PDF directory: {pdf_dir}")
        print(f"  Output directory: {output_dir}")
        print(f"  Years: {years}")
        print(f"  Num CPUs: {num_cpus}")
        print(f"  Citations DB: {citations_db_path}")
        print()

    # Load citation lookups once (if DB provided)
    id_lookup, title_lookup = None, None
    if citations_db_path and citations_db_path.exists():
        if verbose:
            print("Loading citations...")
        id_lookup, title_lookup = load_citations(citations_db_path)
        print()

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=num_cpus)

    # Pre-build MD folder index
    if verbose:
        print("Building MD folder index...")
    md_index = build_md_folder_index(md_dir, years)

    # Pre-build PDF file index
    if verbose:
        print("Building PDF file index...")
    pdf_index = build_pdf_file_index(pdf_dir, years)

    stats = {}
    for year in years:
        count = generate_year_dataset_ray(
            year, data_dir, md_dir, pdf_dir, output_dir, md_index, pdf_index,
            verbose=verbose, num_cpus=num_cpus,
            id_lookup=id_lookup, title_lookup=title_lookup
        )
        stats[year] = count
        if verbose:
            print(f"  {year}: {count} complete papers\n")

    # Print summary
    if verbose:
        print("=" * 60)
        print("Summary:")
        total = sum(stats.values())
        for year, count in sorted(stats.items()):
            print(f"  {year}: {count:,} papers")
        print(f"  Total: {total:,} papers")
        print("=" * 60)

    return stats


def generate_all_datasets(
    data_dir: str | Path = "data/full_run",
    md_dir: str | Path = None,
    pdf_dir: str | Path = None,
    output_dir: str | Path = None,
    years: List[int] = None,
    verbose: bool = True,
    max_workers: int = 16,
    citations_db_path: str | Path = None,
) -> Dict[int, int]:
    """
    Generate parquet datasets for all years.

    Args:
        data_dir: Base data directory
        md_dir: MD directory (default: data_dir/mds)
        pdf_dir: PDF directory (default: data_dir/pdfs)
        output_dir: Output directory (default: data_dir/raw_hf_dump)
        years: Years to process (default: 2020-2026)
        verbose: Print progress
        max_workers: Number of threads for parallel processing
        citations_db_path: Path to citations.db (optional, adds google_scholar_citations)

    Returns:
        Dict mapping year -> number of complete papers
    """
    data_dir = Path(data_dir)
    md_dir = Path(md_dir) if md_dir else data_dir / "mds"
    pdf_dir = Path(pdf_dir) if pdf_dir else data_dir / "pdfs"
    output_dir = Path(output_dir) if output_dir else data_dir / "raw_hf_dump"
    citations_db_path = Path(citations_db_path) if citations_db_path else None
    years = years or DEFAULT_YEARS

    if verbose:
        print(f"Generating HuggingFace datasets")
        print(f"  Data directory: {data_dir}")
        print(f"  MD directory: {md_dir}")
        print(f"  PDF directory: {pdf_dir}")
        print(f"  Output directory: {output_dir}")
        print(f"  Years: {years}")
        print(f"  Max workers: {max_workers}")
        print(f"  Citations DB: {citations_db_path}")
        print()

    # Load citation lookups once (if DB provided)
    id_lookup, title_lookup = None, None
    if citations_db_path and citations_db_path.exists():
        if verbose:
            print("Loading citations...")
        id_lookup, title_lookup = load_citations(citations_db_path)
        print()

    # Pre-build MD folder index (using consolidated function from utils)
    if verbose:
        print("Building MD folder index...")
    md_index = build_md_folder_index(md_dir, years)

    # Pre-build PDF file index
    if verbose:
        print("Building PDF file index...")
    pdf_index = build_pdf_file_index(pdf_dir, years)

    stats = {}
    for year in years:
        count = generate_year_dataset(
            year, data_dir, md_dir, pdf_dir, output_dir, md_index, pdf_index,
            verbose=verbose, max_workers=max_workers,
            id_lookup=id_lookup, title_lookup=title_lookup
        )
        stats[year] = count
        if verbose:
            print(f"  {year}: {count} complete papers\n")

    # Print summary
    if verbose:
        print("=" * 60)
        print("Summary:")
        total = sum(stats.values())
        for year, count in sorted(stats.items()):
            print(f"  {year}: {count:,} papers")
        print(f"  Total: {total:,} papers")
        print("=" * 60)

    return stats


if __name__ == "__main__":
    generate_all_datasets()
