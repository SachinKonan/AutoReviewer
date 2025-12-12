#!/usr/bin/env python3
"""
Download and analyze OpenReview metadata for ICLR 2020-2026.

Saves raw API responses as pickle files for inspection.

Usage:
    # Download all data
    python lib/download_metadata_v2.py --download

    # Analyze existing pickles
    python lib/download_metadata_v2.py --analyze

    # Both
    python lib/download_metadata_v2.py --download --analyze

    # Single year
    python lib/download_metadata_v2.py --download --year 2024

    # Dry run (5 papers per year)
    python lib/download_metadata_v2.py --download --dry-run
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

import openreview
import pandas as pd
from tqdm import tqdm


# Years to process
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]


def get_api_clients():
    """Initialize OpenReview API clients."""
    client_v1 = openreview.Client(baseurl='https://api.openreview.net')
    client_v2 = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
    return client_v1, client_v2


def download_submissions(year: int, client_v1, client_v2, api_versions_df, dry_run: bool = False) -> List:
    """Download all submissions with full replies tree for a year.

    Args:
        year: Conference year
        client_v1: API v1 client
        client_v2: API v2 client
        api_versions_df: DataFrame with API version info
        dry_run: If True, only fetch 5 submissions

    Returns:
        List of submission Note objects with replies in details
    """
    row = api_versions_df[api_versions_df['year'] == year].iloc[0]
    venue_id = row['venue_id']
    api_version = row['api_version']

    print(f"  Fetching {year} submissions ({api_version})...")

    if api_version == 'API 1 (Legacy)':
        submissions = client_v1.get_all_notes(
            invitation=f'{venue_id}/-/Blind_Submission',
            details='replies'  # Full tree of all replies
        )
        if dry_run:
            submissions = submissions[:5]
    else:
        submissions = list(client_v2.get_all_notes(
            invitation=f'{venue_id}/-/Submission',
            details='replies'  # Full tree of all replies
        ))
        if dry_run:
            submissions = submissions[:5]

    print(f"  Got {len(submissions)} submissions")
    return submissions


def download_revisions(year: int, submissions: List, client_v1, client_v2,
                       api_version: str) -> Dict[str, List]:
    """Download revisions for each submission.

    Args:
        year: Conference year
        submissions: List of submission Note objects
        client_v1: API v1 client
        client_v2: API v2 client
        api_version: 'API 1 (Legacy)' or 'API 2 (New)'

    Returns:
        Dict mapping submission_id -> List of revision objects
    """
    revisions = {}

    print(f"  Fetching revisions for {len(submissions)} submissions...")

    for sub in tqdm(submissions, desc=f"  {year} revisions"):
        sub_id = sub.id
        try:
            if api_version == 'API 1 (Legacy)':
                refs = client_v1.get_all_references(referent=sub_id, original=True)
                refs = sorted(refs, key=lambda x: x.tcdate or 0)
            else:
                edits = client_v2.get_note_edits(note_id=sub_id)
                refs = sorted(edits, key=lambda x: x.tcdate or 0)

            revisions[sub_id] = refs
        except Exception as e:
            print(f"    Error fetching revisions for {sub_id}: {e}")
            revisions[sub_id] = []

    return revisions


def extract_decision(submission, api_version: str) -> str:
    """Extract decision from submission replies.

    Args:
        submission: Note object with details['replies']
        api_version: 'API 1 (Legacy)' or 'API 2 (New)'

    Returns:
        Decision string (e.g., 'Accept', 'Reject') or 'Unknown'
    """
    replies = submission.details.get('replies', [])

    for reply in replies:
        if api_version == 'API 1 (Legacy)':
            inv = reply.get('invitation', '')
            if 'Decision' in inv:
                return reply.get('content', {}).get('decision', 'Unknown')
        else:
            invs = reply.get('invitations', [])
            if any('Decision' in inv for inv in invs):
                content = reply.get('content', {})
                decision = content.get('decision', {})
                if isinstance(decision, dict):
                    return decision.get('value', 'Unknown')
                return decision or 'Unknown'

    return 'Unknown'


def get_value(field):
    """Extract value from API v2 nested dict format."""
    if isinstance(field, dict):
        return field.get('value', field)
    return field


def analyze_metadata(pickle_dir: Path, base_data_dir: Path, years: List[int]):
    """Load pickles and analyze metadata across years.

    Args:
        pickle_dir: Directory containing pickle files (dry_run or full_run)
        base_data_dir: Base data directory containing api_version_detection.csv
        years: List of years to analyze

    Prints:
    - Accept/reject counts with revision stats
    - Common fields across all years
    - Fields that differ by year
    """
    print("\n" + "=" * 80)
    print("METADATA ANALYSIS")
    print(f"Data directory: {pickle_dir}")
    print("=" * 80)

    # Load API versions for decision extraction
    api_versions = pd.read_csv(base_data_dir / 'api_version_detection.csv')

    all_content_keys = {}  # year -> set of content keys
    all_reply_types = {}   # year -> set of reply invitation types

    for year in years:
        submissions_file = pickle_dir / f'get_all_notes_{year}.pickle'
        revisions_file = pickle_dir / f'get_revisions_{year}.pickle'

        if not submissions_file.exists():
            print(f"\n{year}: No data (missing {submissions_file.name})")
            continue

        print(f"\n{'=' * 40}")
        print(f"YEAR {year}")
        print(f"{'=' * 40}")

        # Load data
        with open(submissions_file, 'rb') as f:
            submissions = pickle.load(f)

        revisions = {}
        if revisions_file.exists():
            with open(revisions_file, 'rb') as f:
                revisions = pickle.load(f)

        api_version = api_versions[api_versions['year'] == year]['api_version'].values[0]

        # Count decisions
        decision_counts = defaultdict(int)
        revision_counts = defaultdict(list)

        for sub in submissions:
            decision = extract_decision(sub, api_version)
            decision_counts[decision] += 1

            rev_count = len(revisions.get(sub.id, []))
            revision_counts[decision].append(rev_count)

        print(f"\nSubmissions: {len(submissions)}")
        print(f"Decisions:")
        for decision, count in sorted(decision_counts.items()):
            rev_list = revision_counts[decision]
            avg_rev = sum(rev_list) / len(rev_list) if rev_list else 0
            print(f"  {decision}: {count} papers, avg {avg_rev:.1f} revisions")

        # Collect content keys
        content_keys = set()
        for sub in submissions:
            content_keys.update(sub.content.keys())
        all_content_keys[year] = content_keys
        print(f"\nContent fields ({len(content_keys)}): {sorted(content_keys)}")

        # Collect reply types
        reply_types = set()
        for sub in submissions:
            for reply in sub.details.get('replies', []):
                if api_version == 'API 1 (Legacy)':
                    inv = reply.get('invitation', '')
                    reply_types.add(inv.split('/')[-1])
                else:
                    for inv in reply.get('invitations', []):
                        reply_types.add(inv.split('/')[-1])
        all_reply_types[year] = reply_types
        print(f"Reply types ({len(reply_types)}): {sorted(reply_types)}")

        # Sample: print first 3 submissions with revisions
        print(f"\nSample submissions:")
        for sub in submissions[:3]:
            decision = extract_decision(sub, api_version)
            rev_count = len(revisions.get(sub.id, []))
            title = get_value(sub.content.get('title', 'N/A'))
            if len(title) > 50:
                title = title[:50] + '...'
            print(f"  {sub.id}: {decision} | {rev_count} revs | {title}")

    # Compare fields across years
    if len(all_content_keys) > 1:
        print(f"\n{'=' * 40}")
        print("FIELD COMPARISON ACROSS YEARS")
        print(f"{'=' * 40}")

        all_keys = set()
        for keys in all_content_keys.values():
            all_keys.update(keys)

        # Common fields (in all years)
        common_keys = all_keys.copy()
        for keys in all_content_keys.values():
            common_keys &= keys
        print(f"\nCommon fields (all years): {sorted(common_keys)}")

        # Fields that vary by year
        print(f"\nFields by year:")
        for key in sorted(all_keys - common_keys):
            present_years = [y for y, keys in all_content_keys.items() if key in keys]
            print(f"  {key}: {present_years}")


def main():
    parser = argparse.ArgumentParser(description='Download and analyze OpenReview metadata')
    parser.add_argument('--download', action='store_true', help='Download data from API')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing pickle files')
    parser.add_argument('--year', type=int, default=None, help='Process specific year only')
    parser.add_argument('--dry-run', action='store_true', help='Only fetch 5 submissions per year')
    parser.add_argument('--skip-revisions', action='store_true', help='Skip revision download')
    args = parser.parse_args()

    if not args.download and not args.analyze:
        print("Please specify --download and/or --analyze")
        return

    # Setup paths
    script_dir = Path(__file__).parent.parent  # AutoReviewer/
    base_data_dir = script_dir / 'data'

    # Output directory based on run mode
    if args.dry_run:
        output_dir = base_data_dir / 'dry_run'
    else:
        output_dir = base_data_dir / 'full_run'
    output_dir.mkdir(parents=True, exist_ok=True)

    years = [args.year] if args.year else YEARS

    # Load API version config (always from base data dir)
    api_versions = pd.read_csv(base_data_dir / 'api_version_detection.csv')
    api_versions = api_versions[api_versions['status'] == 'Found']

    if args.download:
        print("=" * 80)
        print("DOWNLOADING DATA")
        print(f"Output directory: {output_dir}")
        print("=" * 80)

        client_v1, client_v2 = get_api_clients()
        print("Connected to OpenReview APIs")

        for year in years:
            print(f"\n--- {year} ---")

            # Step 1: Download submissions with replies
            submissions_file = output_dir / f'get_all_notes_{year}.pickle'
            submissions = download_submissions(
                year, client_v1, client_v2, api_versions, dry_run=args.dry_run
            )
            with open(submissions_file, 'wb') as f:
                pickle.dump(submissions, f)
            print(f"  Saved to {submissions_file}")

            # Step 2: Download revisions
            if not args.skip_revisions:
                api_version = api_versions[api_versions['year'] == year]['api_version'].values[0]
                revisions = download_revisions(
                    year, submissions, client_v1, client_v2, api_version
                )
                revisions_file = output_dir / f'get_revisions_{year}.pickle'
                with open(revisions_file, 'wb') as f:
                    pickle.dump(revisions, f)
                print(f"  Saved to {revisions_file}")

    if args.analyze:
        analyze_metadata(output_dir, base_data_dir, years)


if __name__ == '__main__':
    main()
