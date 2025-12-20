"""Analyze MinerU conversion statistics with content criteria."""
import bisect
import csv
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.submission import load_submissions_from_pickle
from lib.utils import (
    build_pdf_file_index,
    find_pdf_path,
    EXCLUDED_DECISIONS,
    DEFAULT_YEARS,
)

# Regex patterns
ABSTRACT_START = re.compile(r'^#\s*ABSTRACT', re.IGNORECASE | re.MULTILINE)
REPROD_HEADER = re.compile(r'^#.*reprod', re.IGNORECASE | re.MULTILINE)
GITHUB_PATTERN = re.compile(r'github', re.IGNORECASE)
ANON_PATTERN = re.compile(r'anonymous', re.IGNORECASE)

def classify_decision(decision: str) -> str:
    """Classify decision as 'Accept', 'Reject', or 'Undecided'."""
    if decision.startswith('Accept'):
        return 'Accept'
    elif decision == 'Reject':
        return 'Reject'
    else:
        return 'Undecided'


def build_mineru_index(mineru_dir: Path) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    """Build index mapping folder name -> markdown file path across all batches.

    Returns:
        (index, fixed_index, fixed_pt2_index):
        - index: all batches (batch_00 to batch_18)
        - fixed_index: batch_2020_fixed only
        - fixed_pt2_index: batch_2020_fixed_pt2 only (highest priority for 2020)
    """
    index = {}
    fixed_index = {}
    fixed_pt2_index = {}

    for batch_dir in sorted(mineru_dir.glob("batch_*")):
        batch_name = batch_dir.name
        # Use scandir for efficiency on NFS - caches is_dir() result
        with os.scandir(batch_dir) as entries:
            for entry in entries:
                if entry.is_dir():
                    # Construct md path directly: vlm/{folder_name}.md
                    folder_name = entry.name
                    md_path = Path(entry.path) / "vlm" / f"{folder_name}.md"
                    if md_path.exists():
                        index[folder_name] = md_path
                        if batch_name == "batch_2020_fixed":
                            fixed_index[folder_name] = md_path
                        elif batch_name == "batch_2020_fixed_pt2":
                            fixed_pt2_index[folder_name] = md_path

    return index, fixed_index, fixed_pt2_index


def _prefix_lookup(submission_id: str, sorted_keys: list[str], index: dict[str, Path]) -> Path | None:
    """Binary search for submission_id prefix in sorted keys."""
    i = bisect.bisect_left(sorted_keys, submission_id)
    if i < len(sorted_keys) and sorted_keys[i].startswith(submission_id):
        return index[sorted_keys[i]]
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
) -> Path | None:
    """Find MinerU markdown by submission ID prefix using binary search.

    For year=2020, checks indices in priority order:
    1. fixed_pt2_index (batch_2020_fixed_pt2) - highest priority
    2. fixed_index (batch_2020_fixed)
    3. main index (all batches)
    """
    if year == 2020:
        # Check pt2 first (highest priority)
        if fixed_pt2_keys and fixed_pt2_index:
            result = _prefix_lookup(submission_id, fixed_pt2_keys, fixed_pt2_index)
            if result:
                return result
        # Check fixed next
        if fixed_keys and fixed_index:
            result = _prefix_lookup(submission_id, fixed_keys, fixed_index)
            if result:
                return result

    # Fall back to main index
    return _prefix_lookup(submission_id, sorted_keys, index)


def extract_abstract(md_text: str) -> str:
    """Extract abstract section from markdown."""
    match = ABSTRACT_START.search(md_text)
    if not match:
        return ""
    start = match.end()
    # Find next section header (# followed by number or # INTRODUCTION etc.)
    next_header = re.search(r'^#\s+(?:\d|[A-Z])', md_text[start:], re.MULTILINE)
    if next_header:
        return md_text[start : start + next_header.start()]
    return md_text[start : start + 2000]


def analyze_paper(md_path: Path) -> tuple[bool, bool, bool]:
    """
    Analyze paper content for criteria.

    Returns:
        (has_github_in_abstract, has_reproducibility_header, no_anonymous_authors)
    """
    try:
        text = md_path.read_text(encoding='utf-8')
    except Exception:
        return (False, False, False)

    abstract = extract_abstract(text)
    has_github = bool(GITHUB_PATTERN.search(abstract))
    has_reprod = bool(REPROD_HEADER.search(text))
    no_anon = not bool(ANON_PATTERN.search(text))

    return (has_github, has_reprod, no_anon)


def get_subset_key(c1: bool, c2: bool, c3: bool) -> str:
    """Convert criteria booleans to subset key."""
    key = ""
    if c1:
        key += "1"
    if c2:
        key += "2"
    if c3:
        key += "3"
    return key if key else "none"


def make_empty_counts(subset_keys: list[str]) -> dict:
    """Create empty counts dict with example tracking."""
    result = {'count': 0, 'pdfs': 0, 'mineru': 0}
    for k in subset_keys:
        result[k] = {'n': 0, 'ex': None}  # n=count, ex=example submission ID
    return result


def analyze_stats(data_dir: Path = Path("data/full_run")):
    """Run the full analysis and print results table."""
    pdf_dir = data_dir / "pdfs"
    mineru_dir = data_dir / "md_mineru"

    print("Building PDF index...")
    pdf_index = build_pdf_file_index(pdf_dir, DEFAULT_YEARS)

    print("Building MinerU index...")
    mineru_index, fixed_index, fixed_pt2_index = build_mineru_index(mineru_dir)
    mineru_keys = sorted(mineru_index.keys())
    fixed_keys = sorted(fixed_index.keys())
    fixed_pt2_keys = sorted(fixed_pt2_index.keys())
    print(f"  Found {len(mineru_index)} MinerU conversions")
    print(f"  Found {len(fixed_index)} in batch_2020_fixed")
    print(f"  Found {len(fixed_pt2_index)} in batch_2020_fixed_pt2 (highest priority for 2020)")

    # Subset keys for counting
    subset_keys = ['1', '2', '3', '12', '13', '23', '123', 'none']
    decision_types = ['Accept', 'Reject', 'Undecided']

    # Results: year -> decision_type -> counts
    results = []

    # CSV rows: (submission_id, md_path, pdf_path, year, violation)
    csv_rows = []

    # Totals by decision type
    totals = {dt: make_empty_counts(subset_keys) for dt in decision_types}
    totals['All'] = make_empty_counts(subset_keys)

    print("\nAnalyzing papers by year...")
    for year in DEFAULT_YEARS:
        pkl_path = data_dir / f"get_all_notes_{year}.pickle"
        if not pkl_path.exists():
            print(f"  {year}: pickle not found, skipping")
            continue

        print(f"  {year}: loading submissions...")
        subs = load_submissions_from_pickle(pkl_path, year)
        non_wd = [s for s in subs if s.decision not in EXCLUDED_DECISIONS]

        # Initialize year counts by decision type
        year_counts = {dt: make_empty_counts(subset_keys) for dt in decision_types}

        for s in non_wd:
            dec_type = classify_decision(s.decision)

            year_counts[dec_type]['count'] += 1

            # Check PDF
            pdf_path = find_pdf_path(s.id, year, pdf_index, pdf_dir)
            if pdf_path:
                year_counts[dec_type]['pdfs'] += 1

            # Check MinerU + analyze (prefer batch_2020_fixed for 2020)
            md_path = find_mineru_by_prefix(
                s.id, mineru_keys, mineru_index,
                year=year,
                fixed_keys=fixed_keys, fixed_index=fixed_index,
                fixed_pt2_keys=fixed_pt2_keys, fixed_pt2_index=fixed_pt2_index,
            )
            if md_path:
                year_counts[dec_type]['mineru'] += 1
                c1, c2, c3 = analyze_paper(md_path)
                key = get_subset_key(c1, c2, c3)
                year_counts[dec_type][key]['n'] += 1
                # Store example if we don't have one yet for this year/decision
                if year_counts[dec_type][key]['ex'] is None:
                    year_counts[dec_type][key]['ex'] = s.id
                # Collect CSV row
                csv_rows.append({
                    'submission_id': s.id,
                    'md_path': str(md_path),
                    'pdf_path': str(pdf_path) if pdf_path else '',
                    'year': year,
                    'decision': dec_type,
                    'violation': key,
                })

        # Store results
        results.append({'year': year, 'counts': year_counts})

        # Update totals
        for dt in decision_types:
            totals[dt]['count'] += year_counts[dt]['count']
            totals[dt]['pdfs'] += year_counts[dt]['pdfs']
            totals[dt]['mineru'] += year_counts[dt]['mineru']
            totals['All']['count'] += year_counts[dt]['count']
            totals['All']['pdfs'] += year_counts[dt]['pdfs']
            totals['All']['mineru'] += year_counts[dt]['mineru']
            for k in subset_keys:
                totals[dt][k]['n'] += year_counts[dt][k]['n']
                totals['All'][k]['n'] += year_counts[dt][k]['n']
                # Keep first example found
                if totals[dt][k]['ex'] is None and year_counts[dt][k]['ex']:
                    totals[dt][k]['ex'] = year_counts[dt][k]['ex']
                if totals['All'][k]['ex'] is None and year_counts[dt][k]['ex']:
                    totals['All'][k]['ex'] = year_counts[dt][k]['ex']

        acc = year_counts['Accept']
        rej = year_counts['Reject']
        print(f"  {year}: Accept={acc['count']} (MinerU={acc['mineru']}), Reject={rej['count']} (MinerU={rej['mineru']})")

    # Helper to format cell with count and example
    def fmt_cell(cell_data: dict, width: int = 20) -> str:
        n = cell_data['n']
        ex = cell_data['ex']
        if ex:
            return f"{n} ({ex[:11]})"[:width].rjust(width)
        else:
            return str(n).rjust(width)

    # Print table
    print("\n" + "=" * 220)
    print("MinerU Statistics Analysis (by Year and Decision)")
    print("=" * 220)
    print("\nCriteria:")
    print("  1 = Has 'github' (case-insensitive) in abstract")
    print("  2 = Has reproducibility header (# ... reprod ...)")
    print("  3 = Does NOT contain 'anonymous'")
    print()

    # Header
    header = f"{'Year':<6} {'Dec':<7} {'Count':>7} {'PDFs':>7} {'MinerU':>7}"
    header += f" {'none':>20} {'1':>20} {'2':>20} {'3':>20} {'12':>20} {'13':>20} {'23':>20} {'123':>20}"
    print(header)
    print("-" * 220)

    # Data rows
    for entry in results:
        year = entry['year']
        for dt in decision_types:
            c = entry['counts'][dt]
            line = f"{year:<6} {dt:<7} {c['count']:>7} {c['pdfs']:>7} {c['mineru']:>7}"
            for key in ['none', '1', '2', '3', '12', '13', '23', '123']:
                line += f" {fmt_cell(c[key])}"
            print(line)
        print()  # Blank line between years

    # Total rows
    print("-" * 220)
    for dt in decision_types:
        c = totals[dt]
        line = f"{'Total':<6} {dt:<7} {c['count']:>7} {c['pdfs']:>7} {c['mineru']:>7}"
        for key in ['none', '1', '2', '3', '12', '13', '23', '123']:
            line += f" {fmt_cell(c[key])}"
        print(line)

    # Grand total
    c = totals['All']
    line = f"{'Total':<6} {'All':<7} {c['count']:>7} {c['pdfs']:>7} {c['mineru']:>7}"
    for key in ['none', '1', '2', '3', '12', '13', '23', '123']:
        line += f" {fmt_cell(c[key])}"
    print(line)
    print("=" * 220)

    # Summary
    print("\nSummary:")
    for dt in ['Accept', 'Reject', 'Undecided', 'All']:
        t = totals[dt]
        if t['count'] > 0:
            print(f"\n  {dt}:")
            print(f"    Total: {t['count']}")
            print(f"    With PDF: {t['pdfs']} ({100*t['pdfs']/t['count']:.1f}%)")
            print(f"    With MinerU: {t['mineru']} ({100*t['mineru']/t['count']:.1f}%)")

            if t['mineru'] > 0:
                has_github = t['1']['n'] + t['12']['n'] + t['13']['n'] + t['123']['n']
                has_reprod = t['2']['n'] + t['12']['n'] + t['23']['n'] + t['123']['n']
                no_anon = t['3']['n'] + t['13']['n'] + t['23']['n'] + t['123']['n']
                print(f"    Among MinerU-converted:")
                print(f"      Has github in abstract: {has_github} ({100*has_github/t['mineru']:.1f}%)")
                print(f"      Has reproducibility header: {has_reprod} ({100*has_reprod/t['mineru']:.1f}%)")
                print(f"      No 'anonymous': {no_anon} ({100*no_anon/t['mineru']:.1f}%)")

    # Write CSV
    csv_path = data_dir / "mineru_violations.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['submission_id', 'md_path', 'pdf_path', 'year', 'decision', 'violation'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV written to: {csv_path} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    analyze_stats()
