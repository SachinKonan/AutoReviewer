"""
Utility functions for AutoReviewer.
"""

import bisect
import os
import re
from pathlib import Path

from tqdm import tqdm

from lib.normalize import ABSTRACT_PATTERN, REFERENCES_PATTERN


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


if __name__ == "__main__":
    print("=== Coverage Table (Review = Total - Withdrawn - Desk Reject) ===")
    get_pdf_coverage_table()
    print("\n=== Complete Papers (PDF + MD + Reviewable) ===")
    get_complete_papers_table()
