#!/usr/bin/env python3
"""
Download PDFs from OpenReview using the manifest.

Downloads to: data/full_run/pdfs/{year}/{submission_id}_{download_id}.pdf

Usage:
    python lib/download_pdfs.py --year 2020
    python lib/download_pdfs.py --year 2024
    python lib/download_pdfs.py --year 2025
"""

import argparse
import csv
import time
from pathlib import Path

import openreview


def download_pdf_v1(client, submission_id, download_id, output_file):
    """Download PDF for API v1 (2020-2023).

    Uses get_pdf(revision_id, is_reference=True) if download_id provided,
    otherwise get_pdf(submission_id).
    """
    try:
        if download_id:
            pdf_bytes = client.get_pdf(download_id, is_reference=True)
        else:
            pdf_bytes = client.get_pdf(submission_id)

        with open(output_file, 'wb') as f:
            f.write(pdf_bytes)
        return True, len(pdf_bytes)
    except Exception as e:
        return False, str(e)


def download_pdf_v2_note(client, submission_id, output_file):
    """Download PDF for API v2 using note ID (2024, or 2025 fallback)."""
    try:
        pdf_bytes = client.get_pdf(submission_id)
        with open(output_file, 'wb') as f:
            f.write(pdf_bytes)
        return True, len(pdf_bytes)
    except Exception as e:
        return False, str(e)


def download_pdf_v2_edit(client, edit_id, output_file):
    """Download PDF for API v2 edit (2025 rebuttal).

    Uses session.get to /notes/edits/attachment endpoint.
    """
    try:
        url = client.baseurl + '/notes/edits/attachment'
        params = {'id': edit_id, 'name': 'pdf'}
        headers = client.headers.copy()
        headers['content-type'] = 'application/pdf'

        response = client.session.get(url, params=params, headers=headers)

        if response.status_code == 200 and len(response.content) > 1000:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            return True, len(response.content)
        else:
            return False, f"Status {response.status_code}, size {len(response.content)}"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Download PDFs from OpenReview')
    parser.add_argument('--year', type=int, required=True, help='Year to download')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of downloads')
    parser.add_argument('--skip-existing', action='store_true', help='Skip existing files')
    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent.parent  # AutoReviewer/
    data_dir = script_dir / 'data'
    manifest_file = data_dir / 'pdf_manifest.csv'
    output_dir = data_dir / 'full_run' / 'pdfs' / str(args.year)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize clients
    if args.year <= 2023:
        client = openreview.Client(baseurl='https://api.openreview.net')
        api_version = 'v1'
    else:
        client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
        api_version = 'v2'

    print("=" * 80)
    print(f"DOWNLOADING PDFs FOR {args.year}")
    print(f"API: {api_version}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # Load manifest
    with open(manifest_file, 'r') as f:
        reader = csv.DictReader(f)
        entries = [row for row in reader if int(row['year']) == args.year]

    print(f"Found {len(entries)} entries for {args.year}")

    if args.limit:
        entries = entries[:args.limit]
        print(f"Limited to {args.limit} entries")

    # Download
    success_count = 0
    fail_count = 0
    skip_count = 0

    for i, entry in enumerate(entries):
        submission_id = entry['submission_id']
        download_id = entry['download_id']
        pdf_source = entry['pdf_source']

        # Filename: {submission_id}_{download_id}.pdf
        filename = f"{submission_id}_{download_id}.pdf" if download_id else f"{submission_id}_.pdf"
        output_file = output_dir / filename

        if args.skip_existing and output_file.exists():
            skip_count += 1
            continue

        # Download based on year and source
        if args.year <= 2023:
            success, result = download_pdf_v1(client, submission_id, download_id, output_file)
        elif args.year in (2024, 2026):
            # 2024 and 2026: use note's PDF directly
            success, result = download_pdf_v2_note(client, submission_id, output_file)
        else:  # 2025
            if pdf_source == 'rebuttal_edit' and download_id:
                success, result = download_pdf_v2_edit(client, download_id, output_file)
            else:
                success, result = download_pdf_v2_note(client, submission_id, output_file)

        if success:
            success_count += 1
            print(f"[{i+1}/{len(entries)}] {filename}: {result} bytes")
        else:
            fail_count += 1
            print(f"[{i+1}/{len(entries)}] {filename}: FAILED - {result}")

        # Rate limiting - be gentle
        time.sleep(0.5)

    print()
    print("=" * 80)
    print(f"SUMMARY for {args.year}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Skipped: {skip_count}")
    print("=" * 80)


if __name__ == '__main__':
    main()
