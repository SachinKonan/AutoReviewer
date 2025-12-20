"""
Utility functions for AutoReviewer.
"""

import bisect
import os
import re
from pathlib import Path

from tqdm import tqdm

from lib.normalize import ABSTRACT_PATTERN, REFERENCES_PATTERN
from lib.schemas import extract_reviews_from_submission, extract_meta_review_from_submission


def has_github_link(text: str) -> bool:
    """Check if text contains a GitHub link."""
    if not text:
        return False
    # Convert to lowercase and look for github.com
    return 'github.com/' in text.lower()


def _find_by_prefix(sorted_list: list[str], prefix: str) -> bool:
    """Check if any item in sorted list starts with prefix using binary search."""
    i = bisect.bisect_left(sorted_list, prefix)
    return i < len(sorted_list) and sorted_list[i].startswith(prefix)


def _get_matching_item(sorted_list: list[str], prefix: str) -> str | None:
    """Get the first item in sorted list that starts with prefix using binary search."""
    i = bisect.bisect_left(sorted_list, prefix)
    if i < len(sorted_list) and sorted_list[i].startswith(prefix):
        return sorted_list[i]
    return None


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_raw_notes(pickle_path: str | Path) -> list:
    """
    Load raw OpenReview notes from pickle file.

    Args:
        pickle_path: Path to get_all_notes_{year}.pickle

    Returns:
        List of raw OpenReview Note objects
    """
    import pickle
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# FILE INDEX FUNCTIONS
# =============================================================================

def build_pdf_file_index(pdf_dir: Path, years: list[int]) -> dict[int, list[str]]:
    """
    Build sorted list of PDF file stems per year for binary search.

    Args:
        pdf_dir: Base PDF directory (e.g., data/full_run/pdfs)
        years: List of years to index

    Returns:
        Dict mapping year -> sorted list of PDF stems (e.g., "abc123_xyz")
    """
    pdf_files = {}
    for year in years:
        year_pdf_dir = pdf_dir / str(year)
        if year_pdf_dir.exists():
            pdf_files[year] = sorted([
                f.stem for f in year_pdf_dir.iterdir() if f.suffix == '.pdf'
            ])
        else:
            pdf_files[year] = []
    return pdf_files


def build_md_folder_index(md_dir: Path, years: list[int]) -> dict[int, list[str]]:
    """
    Build sorted list of MD folder names per year for binary search.

    Folders are only included if they contain a .md file with matching name.

    Args:
        md_dir: Base MD directory (e.g., data/full_run/mds)
        years: List of years to index

    Returns:
        Dict mapping year -> sorted list of folder names (e.g., "abc123_xyz")
    """
    md_folders = {}
    for year in years:
        year_md_dir = md_dir / str(year)
        if year_md_dir.exists():
            folders = []
            for d in year_md_dir.iterdir():
                if d.is_dir():
                    # MD file has same name as folder
                    md_file = d / f"{d.name}.md"
                    if md_file.exists():
                        folders.append(d.name)
            md_folders[year] = sorted(folders)
        else:
            md_folders[year] = []
    return md_folders


# =============================================================================
# FILE PATH FINDER FUNCTIONS
# =============================================================================

def find_pdf_path(
    submission_id: str,
    year: int,
    pdf_index: dict[int, list[str]],
    pdf_dir: Path
) -> Path | None:
    """
    Find PDF file path for a submission.

    PDF files are named {submission_id}_{download_id}.pdf

    Args:
        submission_id: OpenReview submission ID
        year: Conference year
        pdf_index: Pre-built index from build_pdf_file_index()
        pdf_dir: Base PDF directory

    Returns:
        Path to PDF file or None if not found
    """
    year_pdfs = pdf_index.get(year, [])
    pdf_stem = _get_matching_item(year_pdfs, submission_id)
    if pdf_stem:
        return pdf_dir / str(year) / f"{pdf_stem}.pdf"
    return None


def find_md_path(
    submission_id: str,
    year: int,
    md_index: dict[int, list[str]],
    md_dir: Path
) -> tuple[Path | None, str | None]:
    """
    Find MD file path and folder name for a submission.

    MD files are stored as {folder_name}/{folder_name}.md where
    folder_name = {submission_id}_{suffix}

    Args:
        submission_id: OpenReview submission ID
        year: Conference year
        md_index: Pre-built index from build_md_folder_index()
        md_dir: Base MD directory

    Returns:
        Tuple of (Path to MD file, folder name) or (None, None) if not found
    """
    year_mds = md_index.get(year, [])
    md_folder = _get_matching_item(year_mds, submission_id)
    if md_folder:
        md_path = md_dir / str(year) / md_folder / f"{md_folder}.md"
        if md_path.exists():
            return md_path, md_folder
    return None, None


def has_complete_md(
    submission_id: str,
    year: int,
    md_index: dict[int, list[str]],
    md_dir: Path
) -> bool:
    """
    Check if submission has a complete MD file (with Abstract and References).

    Args:
        submission_id: OpenReview submission ID
        year: Conference year
        md_index: Pre-built index from build_md_folder_index()
        md_dir: Base MD directory

    Returns:
        True if MD exists with Abstract and References sections
    """
    md_path, _ = find_md_path(submission_id, year, md_index, md_dir)
    if not md_path:
        return False

    try:
        md_text = md_path.read_text(encoding='utf-8')
        has_abs = ABSTRACT_PATTERN.search(md_text) is not None
        has_refs = REFERENCES_PATTERN.search(md_text) is not None
        return has_abs and has_refs
    except Exception:
        return False


# =============================================================================
# CONSTANTS
# =============================================================================

EXCLUDED_DECISIONS = {'Withdrawn', 'Desk Reject'}
DEFAULT_YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]


# =============================================================================
# COVERAGE TABLES
# =============================================================================

def get_pdf_coverage_table(
    data_dir: str | Path = "data/full_run",
    md_dir: str | Path = "data/full_run/mds",
    years: list[int] = None
) -> None:
    """Print PDF and MD coverage table with decision breakdown.

    Args:
        data_dir: Directory containing pickle files and pdfs/
        md_dir: Directory containing markdown files (mds/{year}/{id}/)
        years: List of years to include (default: 2020-2026)
    """
    from lib.submission import load_submissions_from_pickle

    data_dir = Path(data_dir)
    md_dir = Path(md_dir)
    years = years or [2020, 2021, 2022, 2023, 2024, 2025, 2026]

    # Build sorted PDF filename list per year (for prefix matching)
    pdf_files = {}
    for year in years:
        pdf_dir = data_dir / 'pdfs' / str(year)
        if pdf_dir.exists():
            pdf_files[year] = sorted([f.stem for f in pdf_dir.iterdir() if f.suffix == '.pdf'])

    # Build sorted MD folder list per year (folders that contain a .md file)
    md_folders = {}
    for year in years:
        year_md_dir = md_dir / str(year)
        if year_md_dir.exists():
            folders = []
            for d in year_md_dir.iterdir():
                if d.is_dir():
                    md_file = d / f"{d.name}.md"
                    if md_file.exists():
                        folders.append(d.name)
            md_folders[year] = sorted(folders)

    # Load submissions and compute stats using ground-truth IDs from pickle
    rows = []
    for year in tqdm(years, desc="Years", position=0):
        pickle_path = data_dir / f'get_all_notes_{year}.pickle'
        if not pickle_path.exists():
            continue

        subs = load_submissions_from_pickle(pickle_path, year)
        year_pdfs = pdf_files.get(year, [])
        year_mds = md_folders.get(year, [])
        year_md_dir = md_dir / str(year)

        for s in tqdm(subs, desc=f"  {year}", position=1, leave=False):
            # Use submission ID as prefix to find matching PDF/MD
            has_pdf = _find_by_prefix(year_pdfs, s.id)
            has_md_file = _find_by_prefix(year_mds, s.id)
            has_gh = has_github_link(s.abstract)

            # Check for Abstract and References in MD file
            has_ref_abs_md = False
            if has_md_file:
                md_folder = _get_matching_item(year_mds, s.id)
                if md_folder:
                    md_path = year_md_dir / md_folder / f"{md_folder}.md"
                    try:
                        md_text = md_path.read_text(encoding='utf-8')
                        has_abs = ABSTRACT_PATTERN.search(md_text) is not None
                        has_refs = REFERENCES_PATTERN.search(md_text) is not None
                        has_ref_abs_md = has_abs and has_refs
                    except:
                        pass

            rows.append({
                'year': year,
                'id': s.id,
                'decision': s.decision,
                'has_pdf': has_pdf,
                'has_md': has_md_file,
                'has_github': has_gh,
                'has_ref_abs_md': has_ref_abs_md
            })

    # Build summary
    summary = []
    for year in years:
        year_rows = [r for r in rows if r['year'] == year]
        if not year_rows:
            continue

        total = len(year_rows)
        with_pdf = sum(1 for r in year_rows if r['has_pdf'])

        # Count by decision
        dec_counts = {}
        for r in year_rows:
            dec = r['decision']
            if dec not in dec_counts:
                dec_counts[dec] = {'count': 0, 'with_pdf': 0}
            dec_counts[dec]['count'] += 1
            if r['has_pdf']:
                dec_counts[dec]['with_pdf'] += 1

        # Excluded decisions: Withdrawn and Desk Reject (no PDFs available for these)
        excluded = {'Withdrawn', 'Desk Reject'}
        wdrn = dec_counts.get('Withdrawn', {}).get('count', 0)
        desk = dec_counts.get('Desk Reject', {}).get('count', 0)
        reviewable = total - wdrn - desk  # Papers that went through review (have PDFs)
        pct_reviewable = f"{100*reviewable/total:.1f}" if total > 0 else "N/A"
        # Count PDFs for reviewable papers only
        pdf_reviewable = sum(1 for r in year_rows if r['has_pdf'] and r['decision'] not in excluded)
        pct_pdf_reviewable = f"{100*pdf_reviewable/reviewable:.1f}" if reviewable > 0 else "N/A"
        # Count MDs for reviewable papers only
        md_reviewable = sum(1 for r in year_rows if r['has_md'] and r['decision'] not in excluded)
        pct_md_reviewable = f"{100*md_reviewable/reviewable:.1f}" if reviewable > 0 else "N/A"
        # Count papers with both PDF and MD for reviewable
        both_reviewable = sum(1 for r in year_rows if r['has_pdf'] and r['has_md'] and r['decision'] not in excluded)
        pct_both_reviewable = f"{100*both_reviewable/reviewable:.1f}" if reviewable > 0 else "N/A"
        # Count papers with GitHub link in abstract (reviewable with PDF)
        gh_reviewable = sum(1 for r in year_rows if r['has_github'] and r['has_pdf'] and r['decision'] not in excluded)
        pct_gh_reviewable = f"{100*gh_reviewable/pdf_reviewable:.1f}" if pdf_reviewable > 0 else "N/A"
        # Count papers with both Abstract and References in MD (reviewable)
        ref_abs_md_reviewable = sum(1 for r in year_rows if r['has_ref_abs_md'] and r['decision'] not in excluded)
        pct_ref_abs_md = f"{100*ref_abs_md_reviewable/reviewable:.1f}" if reviewable > 0 else "N/A"

        summary.append({
            'Year': year,
            'Total': total,
            'Reviewable': reviewable,  # Excludes Withdrawn + Desk Reject
            '%Reviewable': pct_reviewable,
            'WithPdf': pdf_reviewable,
            '%WithPdf': pct_pdf_reviewable,
            'WithMd': md_reviewable,
            '%WithMd': pct_md_reviewable,
            'WithBoth': both_reviewable,
            '%WithBoth': pct_both_reviewable,
            'RefAbsMd': ref_abs_md_reviewable,
            '%RefAbsMd': pct_ref_abs_md,
            'WithGH': gh_reviewable,
            '%WithGH': pct_gh_reviewable,
            'Wdrn': wdrn,
            'Desk': desk,
            'Rej': dec_counts.get('Reject', {}).get('count', 0),
            'Post': dec_counts.get('Accept (Poster)', {}).get('count', 0),
            'Spot': dec_counts.get('Accept (Spotlight)', {}).get('count', 0),
            'Oral': dec_counts.get('Accept (Oral)', {}).get('count', 0),
            'Unk': dec_counts.get('Unknown', {}).get('count', 0),
        })

    # Print table
    # Reviewable = Total - Withdrawn - Desk Reject (papers that have PDFs available)
    print(f"{'Year':<6} {'Total':<7} {'Review':<8} {'%Rev':<6} {'WithPdf':<8} {'%Pdf':<6} {'WithMd':<7} {'%Md':<6} {'Both':<6} {'%Both':<6} {'RefAbs':<7} {'%RA':<6} {'WithGH':<7} {'%GH':<6} {'Wdrn':<5} {'Desk':<5} {'Rej':<6} {'Post':<6} {'Spot':<5} {'Oral':<5} {'Unk':<6}")
    print("-" * 155)

    totals = {'Total': 0, 'Reviewable': 0, 'WithPdf': 0, 'WithMd': 0, 'WithBoth': 0, 'RefAbsMd': 0, 'WithGH': 0, 'Wdrn': 0, 'Desk': 0, 'Rej': 0, 'Post': 0, 'Spot': 0, 'Oral': 0, 'Unk': 0}
    for r in summary:
        print(f"{r['Year']:<6} {r['Total']:<7} {r['Reviewable']:<8} {r['%Reviewable']:<6} {r['WithPdf']:<8} {r['%WithPdf']:<6} {r['WithMd']:<7} {r['%WithMd']:<6} {r['WithBoth']:<6} {r['%WithBoth']:<6} {r['RefAbsMd']:<7} {r['%RefAbsMd']:<6} {r['WithGH']:<7} {r['%WithGH']:<6} {r['Wdrn']:<5} {r['Desk']:<5} {r['Rej']:<6} {r['Post']:<6} {r['Spot']:<5} {r['Oral']:<5} {r['Unk']:<6}")
        for k in totals:
            totals[k] += r[k] if k in r else 0

    print("-" * 155)
    pct_reviewable = f"{100*totals['Reviewable']/totals['Total']:.1f}" if totals['Total'] > 0 else "0"
    pct_pdf = f"{100*totals['WithPdf']/totals['Reviewable']:.1f}" if totals['Reviewable'] > 0 else "0"
    pct_md = f"{100*totals['WithMd']/totals['Reviewable']:.1f}" if totals['Reviewable'] > 0 else "0"
    pct_both = f"{100*totals['WithBoth']/totals['Reviewable']:.1f}" if totals['Reviewable'] > 0 else "0"
    pct_ref_abs = f"{100*totals['RefAbsMd']/totals['Reviewable']:.1f}" if totals['Reviewable'] > 0 else "0"
    pct_gh = f"{100*totals['WithGH']/totals['WithPdf']:.1f}" if totals['WithPdf'] > 0 else "0"
    print(f"{'Total':<6} {totals['Total']:<7} {totals['Reviewable']:<8} {pct_reviewable:<6} {totals['WithPdf']:<8} {pct_pdf:<6} {totals['WithMd']:<7} {pct_md:<6} {totals['WithBoth']:<6} {pct_both:<6} {totals['RefAbsMd']:<7} {pct_ref_abs:<6} {totals['WithGH']:<7} {pct_gh:<6} {totals['Wdrn']:<5} {totals['Desk']:<5} {totals['Rej']:<6} {totals['Post']:<6} {totals['Spot']:<5} {totals['Oral']:<5} {totals['Unk']:<6}")


def get_complete_papers_table(
    data_dir: str | Path = "data/full_run",
    md_dir: str | Path = "data/full_run/mds",
    years: list[int] = None
) -> None:
    """Print table of complete papers (has pickle + PDF + MD + not withdrawn) with decision breakdown.

    Args:
        data_dir: Directory containing pickle files and pdfs/
        md_dir: Directory containing markdown files (mds/{year}/{id}/)
        years: List of years to include (default: 2020-2026)
    """
    from lib.submission import load_submissions_from_pickle

    data_dir = Path(data_dir)
    md_dir = Path(md_dir)
    years = years or [2020, 2021, 2022, 2023, 2024, 2025, 2026]

    # Build sorted PDF filename list per year (for prefix matching)
    pdf_files = {}
    for year in years:
        pdf_dir = data_dir / 'pdfs' / str(year)
        if pdf_dir.exists():
            pdf_files[year] = sorted([f.stem for f in pdf_dir.iterdir() if f.suffix == '.pdf'])

    # Build sorted MD folder list per year (folders that contain a .md file)
    md_folders = {}
    for year in years:
        year_md_dir = md_dir / str(year)
        if year_md_dir.exists():
            folders = []
            for d in year_md_dir.iterdir():
                if d.is_dir():
                    md_file = d / f"{d.name}.md"
                    if md_file.exists():
                        folders.append(d.name)
            md_folders[year] = sorted(folders)

    # Load submissions and filter to complete papers using ground-truth IDs
    rows = []
    for year in years:
        pickle_path = data_dir / f'get_all_notes_{year}.pickle'
        if not pickle_path.exists():
            continue

        subs = load_submissions_from_pickle(pickle_path, year)
        year_pdfs = pdf_files.get(year, [])
        year_mds = md_folders.get(year, [])

        # Excluded decisions: Withdrawn and Desk Reject (no PDFs available)
        excluded = {'Withdrawn', 'Desk Reject'}
        for s in subs:
            has_pdf = _find_by_prefix(year_pdfs, s.id)
            has_md_file = _find_by_prefix(year_mds, s.id)
            # Only include complete papers: has PDF + MD + reviewable (not withdrawn/desk reject)
            if has_pdf and has_md_file and s.decision not in excluded:
                rows.append({
                    'year': year,
                    'id': s.id,
                    'decision': s.decision,
                })

    # Build summary by year
    summary = []
    for year in years:
        year_rows = [r for r in rows if r['year'] == year]
        if not year_rows:
            continue

        total = len(year_rows)

        # Count by decision
        dec_counts = {}
        for r in year_rows:
            dec = r['decision']
            dec_counts[dec] = dec_counts.get(dec, 0) + 1

        rej = dec_counts.get('Reject', 0)
        post = dec_counts.get('Accept (Poster)', 0)
        spot = dec_counts.get('Accept (Spotlight)', 0)
        oral = dec_counts.get('Accept (Oral)', 0)
        unk = dec_counts.get('Unknown', 0)

        summary.append({
            'Year': year,
            'Complete': total,
            'Rej': rej,
            '%Rej': f"{100*rej/total:.1f}" if total > 0 else "0",
            'Post': post,
            '%Post': f"{100*post/total:.1f}" if total > 0 else "0",
            'Spot': spot,
            '%Spot': f"{100*spot/total:.1f}" if total > 0 else "0",
            'Oral': oral,
            '%Oral': f"{100*oral/total:.1f}" if total > 0 else "0",
            'Unk': unk,
            '%Unk': f"{100*unk/total:.1f}" if total > 0 else "0",
        })

    # Print table (excludes Withdrawn and Desk Reject - no PDFs available for those)
    print(f"{'Year':<6} {'Complete':<9} {'Rej':<5} {'%Rej':<6} {'Post':<5} {'%Post':<6} {'Spot':<5} {'%Spot':<6} {'Oral':<5} {'%Oral':<6} {'Unk':<5} {'%Unk':<6}")
    print("-" * 85)

    totals = {'Complete': 0, 'Rej': 0, 'Post': 0, 'Spot': 0, 'Oral': 0, 'Unk': 0}
    for r in summary:
        print(f"{r['Year']:<6} {r['Complete']:<9} {r['Rej']:<5} {r['%Rej']:<6} {r['Post']:<5} {r['%Post']:<6} {r['Spot']:<5} {r['%Spot']:<6} {r['Oral']:<5} {r['%Oral']:<6} {r['Unk']:<5} {r['%Unk']:<6}")
        for k in totals:
            totals[k] += r[k]

    print("-" * 85)
    t = totals['Complete']
    if t > 0:
        print(f"{'Total':<6} {t:<9} {totals['Rej']:<5} {100*totals['Rej']/t:.1f}{'':3} {totals['Post']:<5} {100*totals['Post']/t:.1f}{'':3} {totals['Spot']:<5} {100*totals['Spot']/t:.1f}{'':3} {totals['Oral']:<5} {100*totals['Oral']/t:.1f}{'':3} {totals['Unk']:<5} {100*totals['Unk']/t:.1f}{'':3}")
    else:
        print("No complete papers found.")


def get_coverage_stats(
    data_dir: str | Path = "data/full_run",
    md_dir: str | Path = "data/full_run/mds",
    years: list[int] = None
) -> dict:
    """Get coverage statistics as a dict for use in README generation.

    Returns dict with:
        - years: list of year dicts with all stats
        - totals: aggregated totals

    This is the programmatic version of get_pdf_coverage_table().
    """
    from lib.submission import load_submissions_from_pickle

    data_dir = Path(data_dir)
    md_dir = Path(md_dir)
    years = years or [2020, 2021, 2022, 2023, 2024, 2025, 2026]

    # Build sorted MD folder list per year
    md_folders = {}
    for year in years:
        year_md_dir = md_dir / str(year)
        if year_md_dir.exists():
            folders = []
            for d in year_md_dir.iterdir():
                if d.is_dir():
                    md_file = d / f"{d.name}.md"
                    if md_file.exists():
                        folders.append(d.name)
            md_folders[year] = sorted(folders)

    # Load submissions and compute stats
    rows = []
    for year in years:
        pickle_path = data_dir / f'get_all_notes_{year}.pickle'
        if not pickle_path.exists():
            continue

        subs = load_submissions_from_pickle(pickle_path, year)
        year_mds = md_folders.get(year, [])
        year_md_dir = md_dir / str(year)

        for s in subs:
            has_md_file = _find_by_prefix(year_mds, s.id)

            # Check for Abstract and References in MD file
            has_ref_abs_md = False
            if has_md_file:
                md_folder = _get_matching_item(year_mds, s.id)
                if md_folder:
                    md_path = year_md_dir / md_folder / f"{md_folder}.md"
                    try:
                        md_text = md_path.read_text(encoding='utf-8')
                        has_abs = ABSTRACT_PATTERN.search(md_text) is not None
                        has_refs = REFERENCES_PATTERN.search(md_text) is not None
                        has_ref_abs_md = has_abs and has_refs
                    except:
                        pass

            rows.append({
                'year': year,
                'id': s.id,
                'decision': s.decision,
                'has_md': has_md_file,
                'has_ref_abs_md': has_ref_abs_md
            })

    # Build summary
    summary = []
    excluded = {'Withdrawn', 'Desk Reject'}

    for year in years:
        year_rows = [r for r in rows if r['year'] == year]
        if not year_rows:
            continue

        total = len(year_rows)

        # Count by decision
        dec_counts = {}
        for r in year_rows:
            dec = r['decision']
            dec_counts[dec] = dec_counts.get(dec, 0) + 1

        wdrn = dec_counts.get('Withdrawn', 0)
        desk = dec_counts.get('Desk Reject', 0)
        reviewable = total - wdrn - desk

        # Complete = has MD with Abstract + References and reviewable
        complete = sum(1 for r in year_rows if r['has_ref_abs_md'] and r['decision'] not in excluded)

        summary.append({
            'year': year,
            'total': total,
            'reviewable': reviewable,
            'complete': complete,
            'pct_complete': round(100 * complete / reviewable, 1) if reviewable > 0 else 0,
            'withdrawn': wdrn,
            'desk_reject': desk,
            'reject': dec_counts.get('Reject', 0),
            'poster': dec_counts.get('Accept (Poster)', 0),
            'spotlight': dec_counts.get('Accept (Spotlight)', 0),
            'oral': dec_counts.get('Accept (Oral)', 0),
        })

    # Compute totals
    totals = {
        'total': sum(s['total'] for s in summary),
        'reviewable': sum(s['reviewable'] for s in summary),
        'complete': sum(s['complete'] for s in summary),
        'withdrawn': sum(s['withdrawn'] for s in summary),
        'desk_reject': sum(s['desk_reject'] for s in summary),
        'reject': sum(s['reject'] for s in summary),
        'poster': sum(s['poster'] for s in summary),
        'spotlight': sum(s['spotlight'] for s in summary),
        'oral': sum(s['oral'] for s in summary),
    }
    totals['pct_complete'] = round(100 * totals['complete'] / totals['reviewable'], 1) if totals['reviewable'] > 0 else 0

    return {'years': summary, 'totals': totals}


# =============================================================================
# PIPELINE STATISTICS (Full pipeline from submissions to images)
# =============================================================================

def build_mineru_index(mineru_dir: Path) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    """Build index mapping folder name -> content_list.json path across all batches.

    Uses os.scandir() for fast directory traversal.

    Returns:
        (index, fixed_index, fixed_pt2_index):
        - index: all batches
        - fixed_index: batch_2020_fixed only
        - fixed_pt2_index: batch_2020_fixed_pt2 only (highest priority for 2020)
    """
    index = {}
    fixed_index = {}
    fixed_pt2_index = {}

    for batch_dir in sorted(mineru_dir.glob("batch_*")):
        batch_name = batch_dir.name
        with os.scandir(batch_dir) as entries:
            for entry in entries:
                if entry.is_dir():
                    folder_name = entry.name
                    content_list_path = Path(entry.path) / "vlm" / f"{folder_name}_content_list.json"
                    if content_list_path.exists() and content_list_path.stat().st_size > 10:
                        index[folder_name] = content_list_path
                        if batch_name == "batch_2020_fixed":
                            fixed_index[folder_name] = content_list_path
                        elif batch_name == "batch_2020_fixed_pt2":
                            fixed_pt2_index[folder_name] = content_list_path

    return index, fixed_index, fixed_pt2_index


def build_normalized_index(normalized_dir: Path) -> dict[str, dict]:
    """Build index of normalized outputs using os.scandir().

    Returns:
        Dict mapping submission_id -> {
            'pdf': bool,
            'meta_path': Path|None,
            'image_count': int,
            'meta': dict|None  # Loaded meta.json fields
        }
    """
    import json
    index = {}

    if not normalized_dir.exists():
        return index

    with os.scandir(normalized_dir) as year_entries:
        for year_entry in year_entries:
            if not year_entry.is_dir():
                continue
            with os.scandir(year_entry.path) as sub_entries:
                for sub_entry in sub_entries:
                    if not sub_entry.is_dir():
                        continue
                    sub_id = sub_entry.name
                    sub_path = Path(sub_entry.path)
                    pdf_path = sub_path / f"{sub_id}.pdf"
                    meta_path = sub_path / f"{sub_id}_meta.json"
                    img_dir = sub_path / "redacted_pdf_img_content"

                    # Count images using scandir (fast)
                    image_count = 0
                    if img_dir.exists():
                        with os.scandir(img_dir) as img_entries:
                            for img_entry in img_entries:
                                if img_entry.name.startswith("page_") and img_entry.name.endswith(".png"):
                                    image_count += 1

                    # Load meta.json if exists
                    meta_data = None
                    if meta_path.exists():
                        try:
                            with open(meta_path) as f:
                                m = json.load(f)
                            meta_data = {
                                'has_abstract': m.get('has_abstract', False),
                                'has_references': m.get('has_references', False),
                                'page_end': m.get('page_end', 0),
                                'original_pdf_path': m.get('original_pdf_path'),
                                'anonymized_abstract': m.get('anonymized_abstract', ''),
                                'removed_before_intro_count': m.get('removed_before_intro_count', 0),
                                'removed_footnotes_count': m.get('removed_footnotes_count', 0),
                                'removed_reproducibility_count': m.get('removed_reproducibility_count', 0),
                                'removed_acknowledgments_count': m.get('removed_acknowledgments_count', 0),
                                'removed_github_in_abstract': m.get('removed_github_in_abstract', False),
                                'headers_whiteout_count': m.get('headers_whiteout_count', 0),
                            }
                        except Exception:
                            pass

                    index[sub_id] = {
                        'pdf': pdf_path.exists(),
                        'meta_path': meta_path if meta_path.exists() else None,
                        'image_count': image_count,
                        'meta': meta_data
                    }

    return index


def _mineru_prefix_lookup(submission_id: str, sorted_keys: list[str], index: dict[str, Path]) -> tuple[str, Path] | None:
    """Binary search for submission_id prefix in sorted keys."""
    i = bisect.bisect_left(sorted_keys, submission_id)
    if i < len(sorted_keys) and sorted_keys[i].startswith(submission_id):
        folder_name = sorted_keys[i]
        return folder_name, index[folder_name]
    return None


def find_mineru_by_prefix(
    submission_id: str,
    sorted_keys: list[str],
    index: dict[str, Path],
    year: int = None,
    fixed_keys: list[str] = None,
    fixed_index: dict[str, Path] = None,
    fixed_pt2_keys: list[str] = None,
    fixed_pt2_index: dict[str, Path] = None,
) -> tuple[str, Path] | None:
    """Find MinerU content_list by submission ID prefix using binary search.

    For year=2020, checks indices in priority order:
    1. fixed_pt2_index (batch_2020_fixed_pt2) - highest priority
    2. fixed_index (batch_2020_fixed)
    3. main index (all batches)
    """
    if year == 2020:
        if fixed_pt2_keys and fixed_pt2_index:
            result = _mineru_prefix_lookup(submission_id, fixed_pt2_keys, fixed_pt2_index)
            if result:
                return result
        if fixed_keys and fixed_index:
            result = _mineru_prefix_lookup(submission_id, fixed_keys, fixed_index)
            if result:
                return result

    return _mineru_prefix_lookup(submission_id, sorted_keys, index)


def get_pipeline_stats(
    data_dir: str | Path = "data/full_run",
    years: list[int] = None,
    show_drops: int = 0
) -> None:
    """Print comprehensive pipeline statistics table.

    Tracks: submissions -> reviewable -> PDFs -> MinerU -> valid MinerU -> valid paper -> redacted -> images

    Uses fast os.scandir() indexing for all directory operations.

    Args:
        data_dir: Directory containing pickle files, pdfs/, md_mineru/, normalized/
        years: List of years to include (default: 2020-2026)
        show_drops: Number of examples per year of papers that dropped from ValidMD to HasAbsRef (0 = disabled)
    """
    import json
    from collections import defaultdict
    from lib.submission import load_submissions_from_pickle

    data_dir = Path(data_dir)
    years = years or DEFAULT_YEARS

    print("Building indices...")

    # 1. Build PDF index (by year, sorted stems for binary search)
    pdf_index = build_pdf_file_index(data_dir / "pdfs", years)
    print(f"  PDF index: {sum(len(v) for v in pdf_index.values())} files")

    # 2. Build MinerU index
    mineru_dir = data_dir / "md_mineru"
    mineru_index, fixed_index, fixed_pt2_index = build_mineru_index(mineru_dir)
    sorted_keys = sorted(mineru_index.keys())
    fixed_keys = sorted(fixed_index.keys())
    fixed_pt2_keys = sorted(fixed_pt2_index.keys())
    print(f"  MinerU index: {len(mineru_index)} folders")

    # 3. Build normalized index
    normalized_index = build_normalized_index(data_dir / "normalized")
    print(f"  Normalized index: {len(normalized_index)} submissions")

    # 4. Build raw notes index for review counting
    notes_by_id = {}
    for year in years:
        pkl_path = data_dir / f"get_all_notes_{year}.pickle"
        if pkl_path.exists():
            notes = load_raw_notes(pkl_path)
            for note in notes:
                notes_by_id[note.id] = (note, year)
    print(f"  Notes index: {len(notes_by_id)} submissions")

    # 5. Iterate through submissions and collect stats
    rows = []
    drops = defaultdict(list)  # Track drops from ValidMD -> HasAbsRef per year

    for year in tqdm(years, desc="Processing years"):
        pickle_path = data_dir / f'get_all_notes_{year}.pickle'
        if not pickle_path.exists():
            continue

        subs = load_submissions_from_pickle(pickle_path, year)
        year_pdfs = pdf_index.get(year, [])

        for s in subs:
            is_reviewable = s.decision not in EXCLUDED_DECISIONS

            # PDF exists?
            has_pdf = _find_by_prefix(year_pdfs, s.id)

            # MinerU exists?
            mineru_result = find_mineru_by_prefix(
                s.id, sorted_keys, mineru_index,
                year=year,
                fixed_keys=fixed_keys, fixed_index=fixed_index,
                fixed_pt2_keys=fixed_pt2_keys, fixed_pt2_index=fixed_pt2_index
            )
            has_mineru = mineru_result is not None

            # Valid MinerU (md file size > 100 bytes)?
            valid_mineru = False
            mineru_md_path = None
            if mineru_result:
                folder_name, content_list_path = mineru_result
                mineru_md_path = content_list_path.parent / f"{folder_name}.md"
                if mineru_md_path.exists():
                    try:
                        valid_mineru = mineru_md_path.stat().st_size > 100
                    except:
                        pass

            # Normalized outputs?
            norm_info = normalized_index.get(s.id, {})
            has_redacted = norm_info.get('pdf', False)
            image_count = norm_info.get('image_count', 0)
            meta = norm_info.get('meta')  # Pre-loaded meta.json fields

            # Valid paper (from meta: page_end > 0, has_abstract, has_references)?
            valid_paper = False
            has_images = False
            is_clean = False
            drop_reason = None
            paper_meta = None  # For manipulation stats

            if meta:
                page_end = meta.get('page_end', 0)
                has_abs = meta.get('has_abstract', False)
                has_refs = meta.get('has_references', False)
                valid_paper = page_end > 0 and has_abs and has_refs

                # Images valid only if count matches page_end + 1 (page_end is 0-indexed)
                has_images = image_count == page_end + 1 and page_end >= 0

                # Clean: no github in abstract (removed_github_in_abstract == False) AND has_abstract
                no_github = not meta.get('removed_github_in_abstract', False)
                is_clean = has_abs and no_github

                # Store meta for manipulation stats
                paper_meta = meta

            # Check for valid reviews (at least 1 review + meta) - only for papers with images
            valid_reviews = False
            num_reviews = 0
            review_has_meta = False

            if s.id in notes_by_id:
                note, note_year = notes_by_id[s.id]
                try:
                    reviews = extract_reviews_from_submission(note_year, note)
                    num_reviews = len(reviews)
                except Exception:
                    pass
                try:
                    meta_review = extract_meta_review_from_submission(note_year, note)
                    review_has_meta = meta_review is not None
                except Exception:
                    pass
                valid_reviews = num_reviews >= 1 and review_has_meta

            # Track drop reason if valid_mineru but not valid_paper
            if valid_mineru and not valid_paper and is_reviewable:
                reasons = []
                if page_end == 0:
                    reasons.append('page_end=0')
                if not has_abs:
                    reasons.append('no_abstract')
                if not has_refs:
                    reasons.append('no_references')
                drop_reason = ', '.join(reasons) if reasons else 'unknown'

            # Collect drop examples
            if drop_reason and mineru_md_path:
                drops[year].append({
                    'id': s.id,
                    'reason': drop_reason,
                    'md_path': str(mineru_md_path),
                })

            rows.append({
                'year': year,
                'id': s.id,
                'reviewable': is_reviewable,
                'has_pdf': has_pdf,
                'has_mineru': has_mineru,
                'valid_mineru': valid_mineru,
                'valid_paper': valid_paper,
                'has_redacted': has_redacted,
                'has_images': has_images,
                'is_clean': is_clean,
                'valid_reviews': valid_reviews,
                'num_reviews': num_reviews,
                'has_meta_review': review_has_meta,
                'meta': paper_meta,  # For manipulation table
            })

    # 5. Build and print summary table
    summary = []
    manipulation_stats = defaultdict(lambda: defaultdict(int))  # year -> stat_name -> count

    for year in years:
        year_rows = [r for r in rows if r['year'] == year]
        if not year_rows:
            continue

        total = len(year_rows)
        reviewable = sum(1 for r in year_rows if r['reviewable'])
        # Each stage is a cumulative subset of the previous
        has_pdf = sum(1 for r in year_rows if r['reviewable'] and r['has_pdf'])
        has_mineru = sum(1 for r in year_rows if r['reviewable'] and r['has_pdf'] and r['has_mineru'])
        valid_mineru = sum(1 for r in year_rows if r['reviewable'] and r['has_pdf'] and r['has_mineru'] and r['valid_mineru'])
        valid_paper = sum(1 for r in year_rows if r['reviewable'] and r['has_pdf'] and r['has_mineru'] and r['valid_mineru'] and r['valid_paper'])
        has_redacted = sum(1 for r in year_rows if r['reviewable'] and r['has_pdf'] and r['has_mineru'] and r['valid_mineru'] and r['valid_paper'] and r['has_redacted'])
        has_images = sum(1 for r in year_rows if r['reviewable'] and r['has_pdf'] and r['has_mineru'] and r['valid_mineru'] and r['valid_paper'] and r['has_redacted'] and r['has_images'])
        clean = sum(1 for r in year_rows if r['reviewable'] and r['has_pdf'] and r['has_mineru'] and r['valid_mineru'] and r['valid_paper'] and r['has_redacted'] and r['has_images'] and r['is_clean'])
        valid_revs = sum(1 for r in year_rows if r['reviewable'] and r['has_pdf'] and r['has_mineru'] and r['valid_mineru'] and r['valid_paper'] and r['has_redacted'] and r['has_images'] and r['valid_reviews'])

        # Count total reviews and meta-reviews for papers with images
        total_reviews_year = sum(r['num_reviews'] for r in year_rows if r['reviewable'] and r['has_pdf'] and r['has_mineru'] and r['valid_mineru'] and r['valid_paper'] and r['has_redacted'] and r['has_images'])
        total_metas_year = sum(1 for r in year_rows if r['reviewable'] and r['has_pdf'] and r['has_mineru'] and r['valid_mineru'] and r['valid_paper'] and r['has_redacted'] and r['has_images'] and r['has_meta_review'])

        # Collect manipulation stats for papers with images
        for r in year_rows:
            if r['reviewable'] and r['has_pdf'] and r['has_mineru'] and r['valid_mineru'] and r['valid_paper'] and r['has_redacted'] and r['has_images'] and r['meta']:
                m = r['meta']
                if m.get('removed_before_intro_count', 0) > 0:
                    manipulation_stats[year]['before_intro'] += 1
                if m.get('removed_footnotes_count', 0) > 0:
                    manipulation_stats[year]['footnotes'] += 1
                if m.get('removed_reproducibility_count', 0) > 0:
                    manipulation_stats[year]['reproducibility'] += 1
                if m.get('removed_acknowledgments_count', 0) > 0:
                    manipulation_stats[year]['acknowledgments'] += 1
                if m.get('removed_github_in_abstract', False):
                    manipulation_stats[year]['github_abstract'] += 1
                if m.get('headers_whiteout_count', 0) > 0:
                    manipulation_stats[year]['headers'] += 1
                manipulation_stats[year]['total'] += 1

        summary.append({
            'year': year,
            'total': total,
            'reviewable': reviewable,
            'has_pdf': has_pdf,
            'has_mineru': has_mineru,
            'valid_mineru': valid_mineru,
            'valid_paper': valid_paper,
            'has_redacted': has_redacted,
            'has_images': has_images,
            'clean': clean,
            'valid_revs': valid_revs,
            'total_reviews': total_reviews_year,
            'total_metas': total_metas_year,
        })

    # Print table
    print()
    print(f"{'Year':<6} {'Total':<7} {'Review':<8} {'HasPDF':<7} {'%':<5} {'MinerU':<7} {'%':<5} {'ValidMD':<8} {'%':<5} {'HasAbsRef':<9} {'%':<5} {'Redacted':<9} {'%':<5} {'Images':<7} {'%':<5} {'Clean':<6} {'%':<5} {'VldRev':<6} {'%':<5}")
    print("-" * 150)

    totals = {k: 0 for k in ['total', 'reviewable', 'has_pdf', 'has_mineru', 'valid_mineru', 'valid_paper', 'has_redacted', 'has_images', 'clean', 'valid_revs', 'total_reviews', 'total_metas']}
    for r in summary:
        rev = r['reviewable']
        pct_pdf = f"{100*r['has_pdf']/rev:.1f}" if rev > 0 else "0"
        pct_min = f"{100*r['has_mineru']/rev:.1f}" if rev > 0 else "0"
        pct_val = f"{100*r['valid_mineru']/rev:.1f}" if rev > 0 else "0"
        pct_pap = f"{100*r['valid_paper']/rev:.1f}" if rev > 0 else "0"
        pct_red = f"{100*r['has_redacted']/rev:.1f}" if rev > 0 else "0"
        pct_img = f"{100*r['has_images']/rev:.1f}" if rev > 0 else "0"
        pct_cln = f"{100*r['clean']/rev:.1f}" if rev > 0 else "0"
        pct_vr = f"{100*r['valid_revs']/rev:.1f}" if rev > 0 else "0"

        print(f"{r['year']:<6} {r['total']:<7} {r['reviewable']:<8} {r['has_pdf']:<7} {pct_pdf:<5} {r['has_mineru']:<7} {pct_min:<5} {r['valid_mineru']:<8} {pct_val:<5} {r['valid_paper']:<9} {pct_pap:<5} {r['has_redacted']:<9} {pct_red:<5} {r['has_images']:<7} {pct_img:<5} {r['clean']:<6} {pct_cln:<5} {r['valid_revs']:<6} {pct_vr:<5}")

        for k in totals:
            totals[k] += r[k]

    print("-" * 150)
    rev = totals['reviewable']
    pct_pdf = f"{100*totals['has_pdf']/rev:.1f}" if rev > 0 else "0"
    pct_min = f"{100*totals['has_mineru']/rev:.1f}" if rev > 0 else "0"
    pct_val = f"{100*totals['valid_mineru']/rev:.1f}" if rev > 0 else "0"
    pct_pap = f"{100*totals['valid_paper']/rev:.1f}" if rev > 0 else "0"
    pct_red = f"{100*totals['has_redacted']/rev:.1f}" if rev > 0 else "0"
    pct_img = f"{100*totals['has_images']/rev:.1f}" if rev > 0 else "0"
    pct_cln = f"{100*totals['clean']/rev:.1f}" if rev > 0 else "0"
    pct_vr = f"{100*totals['valid_revs']/rev:.1f}" if rev > 0 else "0"

    print(f"{'Total':<6} {totals['total']:<7} {totals['reviewable']:<8} {totals['has_pdf']:<7} {pct_pdf:<5} {totals['has_mineru']:<7} {pct_min:<5} {totals['valid_mineru']:<8} {pct_val:<5} {totals['valid_paper']:<9} {pct_pap:<5} {totals['has_redacted']:<9} {pct_red:<5} {totals['has_images']:<7} {pct_img:<5} {totals['clean']:<6} {pct_cln:<5} {totals['valid_revs']:<6} {pct_vr:<5}")

    # Review statistics summary
    print(f"\n=== Review Statistics (papers with valid images) ===")
    print(f"Papers with images: {totals['has_images']:,}")
    print(f"Papers with valid reviews (>=1 review + meta): {totals['valid_revs']:,}")
    print(f"Total reviews: {totals['total_reviews']:,}")
    print(f"Total meta-reviews: {totals['total_metas']:,}")
    avg_reviews = totals['total_reviews'] / totals['has_images'] if totals['has_images'] > 0 else 0
    print(f"Average reviews per paper: {avg_reviews:.2f}")

    # Print drop examples if requested
    if show_drops > 0:
        print(f"\n=== Papers dropped from ValidMD -> HasAbsRef (up to {show_drops} per year) ===")
        for year in sorted(drops.keys()):
            year_drops = drops[year]
            print(f"\n{year} ({len(year_drops)} total drops):")
            for item in year_drops[:show_drops]:
                print(f"  {item['id']}: {item['reason']}")
                print(f"    md: {item['md_path']}")

    # Print manipulation stats table (count of papers with each manipulation, from has_images subset)
    print(f"\n=== Manipulation Stats (papers with valid images) ===")
    print(f"{'Year':<6} {'Total':<7} {'BeforeIntro':<12} {'Footnotes':<10} {'Repro':<8} {'Ack':<8} {'GitHub':<8} {'Headers':<8}")
    print("-" * 80)

    manip_totals = defaultdict(int)
    for year in years:
        if year not in manipulation_stats:
            continue
        s = manipulation_stats[year]
        print(f"{year:<6} {s['total']:<7} {s['before_intro']:<12} {s['footnotes']:<10} {s['reproducibility']:<8} {s['acknowledgments']:<8} {s['github_abstract']:<8} {s['headers']:<8}")
        for k in ['total', 'before_intro', 'footnotes', 'reproducibility', 'acknowledgments', 'github_abstract', 'headers']:
            manip_totals[k] += s[k]

    print("-" * 80)
    print(f"{'Total':<6} {manip_totals['total']:<7} {manip_totals['before_intro']:<12} {manip_totals['footnotes']:<10} {manip_totals['reproducibility']:<8} {manip_totals['acknowledgments']:<8} {manip_totals['github_abstract']:<8} {manip_totals['headers']:<8}")


if __name__ == "__main__":
    print("=== Pipeline Statistics ===")
    get_pipeline_stats()
    print("\n=== Coverage Table (Review = Total - Withdrawn - Desk Reject) ===")
    get_pdf_coverage_table()
    print("\n=== Complete Papers (PDF + MD + Reviewable) ===")
    get_complete_papers_table()
