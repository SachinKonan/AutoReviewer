#!/usr/bin/env python3
"""
Download missing PDFs from OpenReview using the missing_pdfs.json file.

Uses submission IDs directly (no manifest needed).

Downloads to: data/full_run/pdfs/{year}/{submission_id}_.pdf

Usage:
    python lib/download_missing_pdfs.py --year 2024
    python lib/download_missing_pdfs.py --year 2025
    python lib/download_missing_pdfs.py --year 2026
    python lib/download_missing_pdfs.py  # all years
"""

import argparse
import json
import time
from pathlib import Path

import openreview


def pdf_exists(submission_id: str, pdf_dir: Path) -> bool:
    """Check if a PDF for this submission already exists (using startswith)."""
    if not pdf_dir.exists():
        return False
    return any(f.stem.startswith(submission_id) for f in pdf_dir.glob("*.pdf"))


def download_pdf_v1(client, submission_id, output_file):
    """Download PDF for API v1 (2020-2023)."""
    try:
        pdf_bytes = client.get_pdf(submission_id)
        with open(output_file, 'wb') as f:
            f.write(pdf_bytes)
        return True, len(pdf_bytes)
    except Exception as e:
        return False, str(e)


def download_pdf_v2(client, submission_id, output_file):
    """Download PDF for API v2 (2024+)."""
    try:
        pdf_bytes = client.get_pdf(submission_id)
        with open(output_file, 'wb') as f:
            f.write(pdf_bytes)
        return True, len(pdf_bytes)
    except Exception as e:
        return False, str(e)


def download_year(year: int, missing_ids: list, data_dir: Path, limit: int = None):
    """Download missing PDFs for a single year."""
    output_dir = data_dir / 'full_run' / 'pdfs' / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize client based on year
    if year <= 2023:
        client = openreview.Client(baseurl='https://api.openreview.net')
        api_version = 'v1'
        download_fn = download_pdf_v1
    else:
        client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
        api_version = 'v2'
        download_fn = download_pdf_v2

    print("=" * 80)
    print(f"DOWNLOADING MISSING PDFs FOR {year}")
    print(f"API: {api_version}")
    print(f"Output: {output_dir}")
    print(f"Missing: {len(missing_ids)}")
    print("=" * 80)

    if limit:
        missing_ids = missing_ids[:limit]
        print(f"Limited to {limit} entries")

    success_count = 0
    fail_count = 0
    skip_count = 0

    for i, submission_id in enumerate(missing_ids):
        # Check if already exists (might have been downloaded with different suffix)
        if pdf_exists(submission_id, output_dir):
            skip_count += 1
            continue

        # Filename: {submission_id}_.pdf (no download_id suffix)
        filename = f"{submission_id}_.pdf"
        output_file = output_dir / filename

        success, result = download_fn(client, submission_id, output_file)

        if success:
            success_count += 1
            print(f"[{i+1}/{len(missing_ids)}] {submission_id}: {result} bytes")
        else:
            fail_count += 1
            print(f"[{i+1}/{len(missing_ids)}] {submission_id}: FAILED - {result}")

        # Rate limiting
        time.sleep(0.5)

    print()
    print("=" * 80)
    print(f"SUMMARY for {year}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Skipped: {skip_count}")
    print("=" * 80)

    return success_count, fail_count, skip_count


def main():
    parser = argparse.ArgumentParser(description='Download missing PDFs from OpenReview')
    parser.add_argument('--year', type=int, default=None, help='Year to download (default: all)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of downloads per year')
    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent.parent  # AutoReviewer/
    data_dir = script_dir / 'data'
    missing_file = data_dir / 'missing_pdfs.json'

    if not missing_file.exists():
        print(f"Missing file not found: {missing_file}")
        print("Run the missing PDF detection script first.")
        return

    # Load missing PDFs
    with open(missing_file, 'r') as f:
        missing_by_year = json.load(f)

    # Filter to requested year(s)
    if args.year:
        years = [args.year]
    else:
        years = sorted(int(y) for y in missing_by_year.keys())

    total_success = 0
    total_fail = 0
    total_skip = 0

    for year in years:
        year_str = str(year)
        if year_str not in missing_by_year:
            print(f"No missing PDFs for {year}")
            continue

        missing_ids = missing_by_year[year_str]
        if not missing_ids:
            print(f"No missing PDFs for {year}")
            continue

        success, fail, skip = download_year(year, missing_ids, data_dir, args.limit)
        total_success += success
        total_fail += fail
        total_skip += skip

    print()
    print("=" * 80)
    print("TOTAL SUMMARY")
    print(f"  Success: {total_success}")
    print(f"  Failed: {total_fail}")
    print(f"  Skipped: {total_skip}")
    print("=" * 80)


if __name__ == '__main__':
    main()
