#!/usr/bin/env python3
"""
Extract pre-camera-ready PDF manifest from OpenReview pickle data.

Generates a CSV with:
- submission_id
- year
- decision (Accept/Reject, excludes Withdrawn)
- pdf_path (pre-camera-ready PDF path)

Usage:
    python lib/extract_pdf_manifest.py
    python lib/extract_pdf_manifest.py --dry-run  # Use dry_run data
"""

import argparse
import pickle
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import csv

import pandas as pd


def get_value(field):
    """Extract value from API v2 nested dict format."""
    if isinstance(field, dict):
        return field.get('value', field)
    return field


def extract_decision(submission, api_version: str) -> str:
    """Extract decision from submission replies.

    Returns: 'Accept', 'Reject', 'Withdrawn', or 'Unknown'
    """
    replies = submission.details.get('replies', [])

    for reply in replies:
        if api_version == 'API 1 (Legacy)':
            inv = reply.get('invitation', '')
            if 'Decision' in inv:
                decision = reply.get('content', {}).get('decision', 'Unknown')
                return normalize_decision(decision)
        else:
            invs = reply.get('invitations', [])
            if any('Decision' in inv for inv in invs):
                content = reply.get('content', {})
                decision = content.get('decision', {})
                if isinstance(decision, dict):
                    decision = decision.get('value', 'Unknown')
                return normalize_decision(decision or 'Unknown')

    return 'Unknown'


def normalize_decision(decision: str) -> str:
    """Normalize decision string to Accept/Reject/Withdrawn/Unknown."""
    decision_lower = decision.lower()

    if 'withdraw' in decision_lower:
        return 'Withdrawn'
    elif 'accept' in decision_lower:
        return 'Accept'
    elif 'reject' in decision_lower:
        return 'Reject'
    else:
        return 'Unknown'


def extract_pdf_path_v1(submission, revisions: list, conf_year: int) -> tuple:
    """Extract pre-camera-ready PDF path for API v1 (2020-2023).

    Uses year-based heuristic: ICLR {year} -> last PDF in {year-1}

    Returns: (pdf_path, source, download_id) where:
        - pdf_path: path to PDF
        - source: 'revision' or 'note'
        - download_id: revision_id for get_pdf(id, is_reference=True), or None
    """
    cutoff_year = conf_year - 1

    # Get revisions with real PDF paths, sorted by time
    revs_with_pdf = []
    for r in revisions:
        pdf = r.content.get('pdf', '')
        if pdf and pdf.startswith('/pdf/'):
            ts = datetime.fromtimestamp(r.tcdate / 1000) if r.tcdate else None
            if ts:
                revs_with_pdf.append((r, ts, pdf))

    # Filter to cutoff year and get last one
    pre_camera = [(r, ts, pdf) for r, ts, pdf in revs_with_pdf if ts.year == cutoff_year]

    if pre_camera:
        # Sort by timestamp and get last
        pre_camera.sort(key=lambda x: x[1])
        revision = pre_camera[-1][0]
        return pre_camera[-1][2], 'revision', revision.id

    # Fallback to note's PDF
    pdf = submission.content.get('pdf', '')
    return pdf, 'note', None


def extract_pdf_path_v2_2024(submission) -> tuple:
    """Extract PDF path for API v2 2024.

    2024 doesn't have rebuttal revisions stored, use note's PDF directly.

    Returns: (pdf_path, source, download_id) where download_id is None (use submission_id)
    """
    pdf = submission.content.get('pdf', {})
    pdf_path = get_value(pdf)
    return pdf_path, 'note', None


def extract_pdf_path_v2_2025(submission, edits: list) -> tuple:
    """Extract pre-camera-ready PDF path for API v2 2025.

    Uses last Rebuttal_Revision edit's PDF, or note's PDF if none.

    Returns: (pdf_path, source, download_id) where:
        - pdf_path: path to PDF
        - source: 'rebuttal_edit' or 'note'
        - download_id: edit_id for session.get to /notes/edits/attachment, or None
    """
    # Filter to rebuttal edits
    rebuttal_edits = []
    for e in edits:
        inv = e.invitation or ''
        if 'Rebuttal' in inv:
            # Check if edit has PDF
            if hasattr(e, 'note') and e.note:
                content = e.note.content if hasattr(e.note, 'content') else {}
                pdf = content.get('pdf', {})
                pdf_path = get_value(pdf)
                if pdf_path and pdf_path.startswith('/pdf/'):
                    rebuttal_edits.append((e, pdf_path))

    if rebuttal_edits:
        # Sort by tcdate and get last
        rebuttal_edits.sort(key=lambda x: x[0].tcdate or 0)
        edit = rebuttal_edits[-1][0]
        return rebuttal_edits[-1][1], 'rebuttal_edit', edit.id

    # Fallback to note's PDF
    pdf = submission.content.get('pdf', {})
    pdf_path = get_value(pdf)
    return pdf_path, 'note', None


def main():
    parser = argparse.ArgumentParser(description='Extract PDF manifest from OpenReview data')
    parser.add_argument('--dry-run', action='store_true', help='Use dry_run data instead of full_run')
    parser.add_argument('--year', type=int, help='Process specific year only')
    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent.parent  # AutoReviewer/
    base_data_dir = script_dir / 'data'

    if args.dry_run:
        pickle_dir = base_data_dir / 'dry_run'
    else:
        pickle_dir = base_data_dir / 'full_run'

    output_file = base_data_dir / 'pdf_manifest.csv'

    # Load API versions
    api_versions = pd.read_csv(base_data_dir / 'api_version_detection.csv')

    years = [args.year] if args.year else [2020, 2021, 2022, 2023, 2024, 2025, 2026]

    print("=" * 80)
    print("EXTRACTING PDF MANIFEST")
    print(f"Source: {pickle_dir}")
    print(f"Output: {output_file}")
    print("=" * 80)

    manifest = []
    stats = defaultdict(lambda: defaultdict(int))

    for year in years:
        submissions_file = pickle_dir / f'get_all_notes_{year}.pickle'
        revisions_file = pickle_dir / f'get_revisions_{year}.pickle'

        if not submissions_file.exists():
            print(f"\n{year}: Skipping (no data)")
            continue

        print(f"\n--- {year} ---")

        # Load data
        with open(submissions_file, 'rb') as f:
            submissions = pickle.load(f)

        revisions = {}
        if revisions_file.exists():
            with open(revisions_file, 'rb') as f:
                revisions = pickle.load(f)

        api_row = api_versions[api_versions['year'] == year]
        if api_row.empty:
            print(f"  Warning: No API version info for {year}")
            continue
        api_version = api_row['api_version'].values[0]

        for sub in submissions:
            # Extract decision
            decision = extract_decision(sub, api_version)

            # Skip withdrawn and unknown (except 2026 where we include Unknown)
            if decision == 'Withdrawn':
                stats[year]['skipped_withdrawn'] += 1
                continue
            if decision == 'Unknown' and year != 2026:
                stats[year]['skipped_unknown'] += 1
                continue

            # Extract PDF path based on year/API
            sub_revisions = revisions.get(sub.id, [])

            if year <= 2023:
                pdf_path, source, download_id = extract_pdf_path_v1(sub, sub_revisions, year)
            elif year in (2024, 2026):
                # 2024 and 2026: use note's PDF directly (no rebuttals stored / no decisions yet)
                pdf_path, source, download_id = extract_pdf_path_v2_2024(sub)
            else:  # 2025
                pdf_path, source, download_id = extract_pdf_path_v2_2025(sub, sub_revisions)

            # Extract PDF hash from path
            pdf_hash = ''
            if pdf_path and pdf_path.startswith('/pdf/'):
                pdf_hash = pdf_path.split('/')[-1].replace('.pdf', '')

            # Get title for reference
            title = get_value(sub.content.get('title', 'N/A'))

            manifest.append({
                'submission_id': sub.id,
                'year': year,
                'decision': decision,
                'pdf_path': pdf_path,
                'pdf_hash': pdf_hash,
                'pdf_source': source,
                'download_id': download_id or '',  # revision_id, edit_id, or empty
                'title': title[:100] if title else ''
            })

            stats[year][decision] += 1
            stats[year][f'source_{source}'] += 1

        # Print stats for year
        print(f"  Accept: {stats[year].get('Accept', 0)}")
        print(f"  Reject: {stats[year].get('Reject', 0)}")
        print(f"  Skipped (withdrawn): {stats[year].get('skipped_withdrawn', 0)}")
        print(f"  Skipped (unknown): {stats[year].get('skipped_unknown', 0)}")
        print(f"  PDF from revision: {stats[year].get('source_revision', 0)}")
        print(f"  PDF from note: {stats[year].get('source_note', 0)}")
        print(f"  PDF from rebuttal_edit: {stats[year].get('source_rebuttal_edit', 0)}")

    # Write manifest
    print(f"\n--- Writing manifest ---")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['submission_id', 'year', 'decision', 'pdf_path', 'pdf_hash', 'pdf_source', 'download_id', 'title'])
        writer.writeheader()
        writer.writerows(manifest)

    print(f"Wrote {len(manifest)} entries to {output_file}")

    # Summary
    print(f"\n--- Summary ---")
    total_accept = sum(stats[y].get('Accept', 0) for y in years)
    total_reject = sum(stats[y].get('Reject', 0) for y in years)
    print(f"Total Accept: {total_accept}")
    print(f"Total Reject: {total_reject}")
    print(f"Total papers: {len(manifest)}")


if __name__ == '__main__':
    main()
