#!/usr/bin/env python3
"""
Download ICLR metadata from OpenReview API (2020-2026)

Features:
- Downloads paper metadata with reviews, decisions, and rebuttals
- Tracks all revisions with timestamps
- Identifies the latest revision BEFORE camera-ready (Final Decision Date)
- Handles both API v1 (Legacy) and API v2 (New)

Usage:
    # Dry run (5 papers per year)
    python lib/download_metadata.py --dry-run

    # Full extraction
    python lib/download_metadata.py

    # Specific year
    python lib/download_metadata.py --year 2024
"""

import openreview
import pandas as pd
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Column names for output CSV (same as extract_all_conferences.py + revisions)
COLUMNS = [
    'conference',
    'year',
    'submission_id',
    'title',
    'abstract',
    'keywords',
    'authors',
    'publish_date',
    'last_modified_date',
    'supplemental_link',
    'primary_area',
    'paper_decision',
    'meta_review',
    'author_main_response',
    'reviewer_response_json',
    # New column for revision tracking
    'revisions_json',           # List of all revisions with metadata
]

# Conference timeline information
# Source: https://iclr.cc/Conferences/{year}/CallForPapers
CONFERENCE_TIMELINES = {
    2020: {
        'submission_date': '2019-09-25',
        'reviews_release_date': '2019-12-04',
        'discussion_ends_date': '2019-11-15',
        'final_decision_date': '2019-12-19',
        'paper_min_length': None,
        'paper_max_length_submission': 10,
        'paper_max_length_camera_ready': 10,
    },
    2021: {
        'submission_date': '2020-10-02',
        'reviews_release_date': '2020-11-10',
        'discussion_ends_date': '2020-11-24',
        'final_decision_date': '2021-01-14',
        'paper_min_length': None,
        'paper_max_length_submission': 8,
        'paper_max_length_camera_ready': 9,
    },
    2022: {
        'submission_date': '2021-10-05',
        'reviews_release_date': '2021-11-08',
        'discussion_ends_date': '2021-11-22',
        'final_decision_date': '2022-01-24',
        'paper_min_length': None,
        'paper_max_length_submission': 9,
        'paper_max_length_camera_ready': 9,
    },
    2023: {
        'submission_date': '2022-10-28',
        'reviews_release_date': '2022-11-04',
        'discussion_ends_date': None,  # Not provided
        'final_decision_date': '2023-01-20',
        'paper_min_length': None,
        'paper_max_length_submission': 9,
        'paper_max_length_camera_ready': 9,
    },
    2024: {
        'submission_date': '2023-09-28',
        'reviews_release_date': '2023-11-10',
        'discussion_ends_date': '2023-11-22',
        'final_decision_date': '2024-01-15',
        'paper_min_length': None,
        'paper_max_length_submission': 9,
        'paper_max_length_camera_ready': 9,
        'revision_tracking_unavailable': True,  # Special flag
    },
    2025: {
        'submission_date': '2024-10-01',
        'reviews_release_date': '2024-10-12',
        'discussion_ends_date': '2024-11-27',
        'final_decision_date': '2025-01-22',
        'paper_min_length': 6,
        'paper_max_length_submission': 10,
        'paper_max_length_camera_ready': 10,
    },
    2026: {
        # 2026 dates not yet available - use placeholders
        'submission_date': None,
        'reviews_release_date': None,
        'discussion_ends_date': None,
        'final_decision_date': None,
        'paper_min_length': None,
        'paper_max_length_submission': None,
        'paper_max_length_camera_ready': None,
    },
}


def get_value_api2(field):
    """Extract value from API 2 nested dict format"""
    if isinstance(field, dict):
        return field.get('value', '')
    return field


def get_revisions_api1(client, submission_id, venue_id):
    """Get all revisions for a submission using API v1

    Returns list of dicts with revision info sorted by tcdate (true creation date):
    - revision_number
    - tcdate: when this revision was actually created
    - cdate: original note creation date
    - note_id
    - title
    - pdf
    """
    revisions = []
    try:
        # Get all references with original=True to get full revision history
        references = client.get_all_references(referent=submission_id, original=True)

        # Sort by tcdate (true creation date)
        references = sorted(references, key=lambda x: x.tcdate or 0)

        for i, ref in enumerate(references):
            rev_info = {
                'revision_number': i + 1,
                'tcdate': datetime.fromtimestamp(ref.tcdate / 1000).isoformat() if ref.tcdate else None,
                'cdate': datetime.fromtimestamp(ref.cdate / 1000).isoformat() if ref.cdate else None,
                'note_id': ref.id,
                'title': ref.content.get('title', ''),
                'pdf': ref.content.get('pdf', ''),
                'invitation': getattr(ref, 'invitation', ''),
            }
            revisions.append(rev_info)
    except Exception as e:
        # Some submissions may not have revision history accessible
        pass

    return revisions


def get_revisions_api2(client, submission_id, venue_id):
    """Get all revisions for a submission using API v2

    Returns list of dicts with revision info sorted by tcdate:
    - revision_number
    - tcdate: when this revision was created
    - cdate: creation date
    - edit_id
    - note_id
    - title
    - pdf
    - invitation: e.g. Rebuttal_Revision, Camera_Ready_Revision
    """
    revisions = []
    try:
        # Get note edits/versions
        edits = client.get_note_edits(note_id=submission_id)

        # Sort by tcdate (true creation date)
        edits = sorted(edits, key=lambda x: x.tcdate or 0)

        for i, edit in enumerate(edits):
            content = edit.note.content if hasattr(edit.note, 'content') else {}

            # Extract PDF (API v2 uses nested value format)
            pdf = ''
            if content.get('pdf'):
                pdf = get_value_api2(content['pdf'])

            rev_info = {
                'revision_number': i + 1,
                'tcdate': datetime.fromtimestamp(edit.tcdate / 1000).isoformat() if edit.tcdate else None,
                'cdate': datetime.fromtimestamp(edit.cdate / 1000).isoformat() if edit.cdate else None,
                'edit_id': edit.id,
                'note_id': edit.note.id if hasattr(edit.note, 'id') else submission_id,
                'title': get_value_api2(content.get('title', '')),
                'pdf': pdf,
                'invitation': edit.invitation if hasattr(edit, 'invitation') else '',
            }
            revisions.append(rev_info)
    except Exception as e:
        pass

    return revisions


def extract_api1_submission(submission, conference, year, csv_writer, csv_file,
                            client, venue_id, fetch_revisions=True):
    """Extract data from API 1 submission and write to CSV"""
    timeline = CONFERENCE_TIMELINES.get(year, {})

    # Basic submission fields
    row = {
        'conference': conference,
        'year': year,
        'submission_id': submission.id,
        'title': submission.content.get('title', ''),
        'abstract': submission.content.get('abstract', ''),
        'keywords': json.dumps(submission.content.get('keywords', [])),
        'authors': json.dumps(submission.content.get('authors', [])),
        'publish_date': datetime.fromtimestamp(submission.cdate/1000).isoformat() if hasattr(submission, 'cdate') and submission.cdate else '',
        'last_modified_date': datetime.fromtimestamp(submission.mdate/1000).isoformat() if hasattr(submission, 'mdate') and submission.mdate else '',
        'supplemental_link': submission.content.get('supplementary_material', ''),
        'primary_area': submission.content.get('Please_choose_the_closest_area_that_your_submission_falls_into', ''),
    }

    # Get replies from details
    direct_replies = submission.details.get('directReplies', [])

    # Extract reviews
    reviews = []
    for reply in direct_replies:
        if reply.get('invitation', '').endswith('Official_Review'):
            review_note = openreview.Note.from_json(reply)
            reviews.append({
                'id': review_note.id,
                'content': dict(review_note.content)
            })
    row['reviewer_response_json'] = json.dumps(reviews)

    # Extract decision
    decision_text = ''
    meta_review_text = ''
    for reply in direct_replies:
        if 'Decision' in reply.get('invitation', ''):
            decision_note = openreview.Note.from_json(reply)
            decision_text = decision_note.content.get('decision', '')
            meta_review_text = decision_note.content.get('metareview:_summary,_strengths_and_weaknesses',
                                                         decision_note.content.get('metareview', ''))
            break
    row['paper_decision'] = decision_text
    row['meta_review'] = meta_review_text

    # Extract author response/rebuttal
    author_response = {}
    for reply in direct_replies:
        inv = reply.get('invitation', '')
        if 'Rebuttal' in inv or 'Author' in inv:
            rebuttal_note = openreview.Note.from_json(reply)
            author_response = dict(rebuttal_note.content)
            break
    row['author_main_response'] = json.dumps(author_response)

    # Get revisions
    if fetch_revisions and not timeline.get('revision_tracking_unavailable'):
        revisions = get_revisions_api1(client, submission.id, venue_id)
        row['revisions_json'] = json.dumps(revisions)
    else:
        row['revisions_json'] = json.dumps([])

    # Write row immediately
    csv_writer.writerow(row)
    csv_file.flush()
    return row


def extract_api2_submission(submission, conference, year, csv_writer, csv_file,
                            client, venue_id, fetch_revisions=True):
    """Extract data from API 2 submission and write to CSV"""
    timeline = CONFERENCE_TIMELINES.get(year, {})

    # Basic submission fields
    row = {
        'conference': conference,
        'year': year,
        'submission_id': submission.id,
        'title': get_value_api2(submission.content.get('title', '')),
        'abstract': get_value_api2(submission.content.get('abstract', '')),
        'keywords': json.dumps(get_value_api2(submission.content.get('keywords', []))),
        'authors': json.dumps(get_value_api2(submission.content.get('authors', []))),
        'publish_date': datetime.fromtimestamp(submission.cdate/1000).isoformat() if submission.cdate else '',
        'last_modified_date': datetime.fromtimestamp(submission.mdate/1000).isoformat() if submission.mdate else '',
        'supplemental_link': get_value_api2(submission.content.get('supplementary_material', '')),
        'primary_area': get_value_api2(submission.content.get('primary_area', '')),
    }

    # Get replies from details
    replies = submission.details.get('replies', [])

    # Extract reviews
    reviews = []
    for reply in replies:
        if any('Official_Review' in inv for inv in reply.get('invitations', [])):
            review_content = {}
            for key, value in reply.get('content', {}).items():
                review_content[key] = get_value_api2(value)
            reviews.append({
                'id': reply.get('id', ''),
                'content': review_content
            })
    row['reviewer_response_json'] = json.dumps(reviews)

    # Extract decision
    decision_text = ''
    for reply in replies:
        if any('Decision' in inv for inv in reply.get('invitations', [])):
            decision_text = get_value_api2(reply.get('content', {}).get('decision', ''))
            break
    row['paper_decision'] = decision_text

    # Extract meta-review
    meta_review_text = ''
    for reply in replies:
        if any('Meta_Review' in inv for inv in reply.get('invitations', [])):
            meta_review_text = get_value_api2(reply.get('content', {}).get('metareview', ''))
            break
    row['meta_review'] = meta_review_text

    # Extract author response/rebuttal
    author_response = {}
    for reply in replies:
        if any('Rebuttal' in inv for inv in reply.get('invitations', [])):
            rebuttal_content = {}
            for key, value in reply.get('content', {}).items():
                rebuttal_content[key] = get_value_api2(value)
            author_response = rebuttal_content
            break
    row['author_main_response'] = json.dumps(author_response)

    # Get revisions
    if fetch_revisions and not timeline.get('revision_tracking_unavailable'):
        revisions = get_revisions_api2(client, submission.id, venue_id)
        row['revisions_json'] = json.dumps(revisions)
    else:
        row['revisions_json'] = json.dumps([])

    # Write row immediately
    csv_writer.writerow(row)
    csv_file.flush()
    return row


def main():
    parser = argparse.ArgumentParser(description='Download ICLR metadata from OpenReview')
    parser.add_argument('--dry-run', action='store_true',
                       help='Limit to 5 submissions per year for testing')
    parser.add_argument('--year', type=int, default=None,
                       help='Extract only specific year (e.g., 2024)')
    parser.add_argument('--no-revisions', action='store_true',
                       help='Skip fetching revision history (faster)')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for CSV files')
    args = parser.parse_args()

    # Resolve paths relative to AutoReviewer directory
    script_dir = Path(__file__).parent.parent  # AutoReviewer/
    data_dir = script_dir / args.output_dir
    data_dir.mkdir(exist_ok=True)

    api_versions_file = data_dir / 'api_version_detection.csv'

    print("=" * 80)
    print("ICLR Metadata Download")
    print("=" * 80)
    print(f"Mode: {'DRY RUN (5 per year)' if args.dry_run else 'FULL EXTRACTION'}")
    print(f"Revisions: {'DISABLED' if args.no_revisions else 'ENABLED'}")
    if args.year:
        print(f"Year filter: {args.year}")
    print(f"Start time: {datetime.now()}")
    print()

    # Read API version detection
    print("Loading API version configuration...")
    api_versions = pd.read_csv(api_versions_file)
    api_versions = api_versions[api_versions['status'] == 'Found']

    # Filter by year if specified
    if args.year:
        api_versions = api_versions[api_versions['year'] == args.year]

    print(f"  Loaded {len(api_versions)} conference configurations")
    print()

    # Initialize clients
    print("Initializing OpenReview clients...")
    client_v1 = openreview.Client(baseurl='https://api.openreview.net')
    client_v2 = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
    print("  Connected to both APIs")
    print()

    # Create output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_year{args.year}" if args.year else ""
    output_file = data_dir / f"iclr_metadata_{'dryrun' if args.dry_run else 'full'}{suffix}_{timestamp}.csv"

    # Open CSV file for streaming writes
    with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=COLUMNS)
        csv_writer.writeheader()
        csv_file.flush()

        total_submissions = 0

        # Process each conference/year
        for _, row in api_versions.iterrows():
            conference = row['conference']
            year = row['year']
            venue_id = row['venue_id']
            api_version = row['api_version']

            print(f"\n{'='*80}")
            print(f"{conference} {year} ({api_version})")
            timeline = CONFERENCE_TIMELINES.get(year, {})
            if timeline.get('final_decision_date'):
                print(f"  Final decision date: {timeline['final_decision_date']}")
            if timeline.get('revision_tracking_unavailable'):
                print(f"  NOTE: Revision tracking unavailable for this year")
            print(f"{'='*80}")

            try:
                if api_version == 'API 1 (Legacy)':
                    invitation = f'{venue_id}/-/Blind_Submission'
                    print(f"Fetching submissions with details='directReplies'...")

                    submissions = client_v1.get_all_notes(
                        invitation=invitation,
                        details='directReplies'
                    )

                    if args.dry_run:
                        submissions = submissions[:5]

                    print(f"Processing {len(submissions) if isinstance(submissions, list) else '?'} submissions...")
                    for sub in tqdm(submissions, desc=f"{conference} {year}"):
                        extract_api1_submission(
                            sub, conference, year, csv_writer, csv_file,
                            client_v1, venue_id, fetch_revisions=not args.no_revisions
                        )
                        total_submissions += 1

                else:  # API 2
                    print(f"Fetching submissions with details='replies'...")

                    submissions = client_v2.get_all_notes(
                        invitation=f'{venue_id}/-/Submission',
                        details='replies'
                    )

                    submissions = list(submissions)
                    if args.dry_run:
                        submissions = submissions[:5]

                    print(f"Processing {len(submissions)} submissions...")
                    for sub in tqdm(submissions, desc=f"{conference} {year}"):
                        extract_api2_submission(
                            sub, conference, year, csv_writer, csv_file,
                            client_v2, venue_id, fetch_revisions=not args.no_revisions
                        )
                        total_submissions += 1

                print(f"  Completed {conference} {year}")

            except Exception as e:
                print(f"  ERROR processing {conference} {year}: {str(e)}")
                import traceback
                traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Total submissions extracted: {total_submissions}")
    print(f"Output file: {output_file}")
    print(f"End time: {datetime.now()}")
    print("=" * 80)


if __name__ == '__main__':
    main()
