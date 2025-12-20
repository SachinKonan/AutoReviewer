#!/usr/bin/env python3
"""
Fix ICLR 2020 PDF manifest to use pre-decision date cutoff.

Problem: Current manifest uses year-based filtering (ts.year == 2019), but the
final decision date was 2019-12-19, so revisions from 12/20-12/31 could be
camera-ready versions.

Fix: Use ts < 2019-12-20 as cutoff to ensure pre-decision PDFs.

Usage:
    python scripts/fix_2020_pdf_manifest.py --dry-run   # Show what would change
    python scripts/fix_2020_pdf_manifest.py --apply     # Apply the fix
"""

import argparse
import csv
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path

import openreview


# Cutoff: Dec 17 to include only through Dec 16 (some camera-readys posted Dec 17-19)
CUTOFF_DATE = datetime(2019, 12, 17)


def extract_pdf_path_fixed(submission, revisions: list) -> tuple:
    """Extract pre-camera-ready PDF using date cutoff instead of year.

    Returns: (pdf_path, source, download_id)
    """
    revs_with_pdf = []
    for r in revisions:
        pdf = r.content.get('pdf', '')
        if pdf and pdf.startswith('/pdf/'):
            ts = datetime.fromtimestamp(r.tcdate / 1000) if r.tcdate else None
            if ts:
                revs_with_pdf.append((r, ts, pdf))

    # Filter to BEFORE cutoff date (not just year)
    pre_camera = [(r, ts, pdf) for r, ts, pdf in revs_with_pdf if ts < CUTOFF_DATE]

    if pre_camera:
        pre_camera.sort(key=lambda x: x[1])
        revision = pre_camera[-1][0]
        return pre_camera[-1][2], 'revision', revision.id

    # Fallback to note's PDF
    pdf = submission.content.get('pdf', '')
    return pdf, 'note', None


def extract_decision(submission) -> str:
    """Extract decision from submission replies (API v1)."""
    replies = submission.details.get('replies', [])

    for reply in replies:
        inv = reply.get('invitation', '')
        if 'Decision' in inv:
            decision = reply.get('content', {}).get('decision', 'Unknown')
            return normalize_decision(decision)

    return 'Unknown'


def normalize_decision(decision: str) -> str:
    """Normalize decision string."""
    decision_lower = decision.lower()
    if 'withdraw' in decision_lower:
        return 'Withdrawn'
    elif 'accept' in decision_lower:
        return 'Accept'
    elif 'reject' in decision_lower:
        return 'Reject'
    return 'Unknown'


def load_old_manifest(manifest_path: Path) -> dict:
    """Load current manifest and return 2020 entries as dict by submission_id."""
    entries = {}
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['year'] == '2020':
                entries[row['submission_id']] = row
    return entries


def regenerate_2020_entries(pickle_dir: Path) -> list:
    """Regenerate 2020 manifest entries using date cutoff."""
    submissions_file = pickle_dir / 'get_all_notes_2020.pickle'
    revisions_file = pickle_dir / 'get_revisions_2020.pickle'

    with open(submissions_file, 'rb') as f:
        submissions = pickle.load(f)

    revisions = {}
    if revisions_file.exists():
        with open(revisions_file, 'rb') as f:
            revisions = pickle.load(f)

    entries = []
    for sub in submissions:
        decision = extract_decision(sub)

        # Skip withdrawn
        if decision == 'Withdrawn':
            continue

        sub_revisions = revisions.get(sub.id, [])
        pdf_path, source, download_id = extract_pdf_path_fixed(sub, sub_revisions)

        # Extract PDF hash
        pdf_hash = ''
        if pdf_path and pdf_path.startswith('/pdf/'):
            pdf_hash = pdf_path.split('/')[-1].replace('.pdf', '')

        title = sub.content.get('title', 'N/A')
        if isinstance(title, dict):
            title = title.get('value', 'N/A')

        entries.append({
            'submission_id': sub.id,
            'year': '2020',
            'decision': decision,
            'pdf_path': pdf_path,
            'pdf_hash': pdf_hash,
            'pdf_source': source,
            'download_id': download_id or '',
            'title': title[:100] if title else ''
        })

    return entries


def find_changed_entries(old_entries: dict, new_entries: list) -> list:
    """Find entries where download_id changed."""
    changed = []
    for entry in new_entries:
        sub_id = entry['submission_id']
        if sub_id in old_entries:
            old_dl_id = old_entries[sub_id].get('download_id', '')
            new_dl_id = entry.get('download_id', '')
            if old_dl_id != new_dl_id:
                changed.append({
                    'submission_id': sub_id,
                    'old_download_id': old_dl_id,
                    'new_download_id': new_dl_id,
                    'decision': entry['decision'],
                    'title': entry['title'],
                    **entry
                })
    return changed


def backup_old_pdfs(changed: list, pdf_dir: Path, dry_run: bool = False):
    """Move old PDFs to backup directory."""
    backup_dir = pdf_dir / '2020_old_download_id'

    if not dry_run:
        backup_dir.mkdir(exist_ok=True)

    moved = 0
    for entry in changed:
        old_dl_id = entry['old_download_id']
        if not old_dl_id:
            continue

        old_name = f"{entry['submission_id']}_{old_dl_id}.pdf"
        old_path = pdf_dir / '2020' / old_name

        if old_path.exists():
            if dry_run:
                print(f"  Would move: {old_name}")
            else:
                shutil.move(str(old_path), str(backup_dir / old_name))
                print(f"  Moved: {old_name}")
            moved += 1

    return moved


def download_new_pdfs(changed: list, pdf_dir: Path, dry_run: bool = False):
    """Download PDFs with new download_ids."""
    if dry_run:
        print(f"  Would download {len(changed)} PDFs")
        return 0

    # Initialize API v1 client
    client = openreview.Client(baseurl='https://api.openreview.net')
    output_dir = pdf_dir / '2020'

    success = 0
    for i, entry in enumerate(changed):
        sub_id = entry['submission_id']
        dl_id = entry['new_download_id']

        filename = f"{sub_id}_{dl_id}.pdf" if dl_id else f"{sub_id}_.pdf"
        output_file = output_dir / filename

        try:
            if dl_id:
                pdf_bytes = client.get_pdf(dl_id, is_reference=True)
            else:
                pdf_bytes = client.get_pdf(sub_id)

            with open(output_file, 'wb') as f:
                f.write(pdf_bytes)
            success += 1
            print(f"  [{i+1}/{len(changed)}] Downloaded: {filename} ({len(pdf_bytes)} bytes)")
        except Exception as e:
            print(f"  [{i+1}/{len(changed)}] FAILED: {filename} - {e}")

        time.sleep(0.5)  # Rate limiting

    return success


def update_manifest(manifest_path: Path, new_2020_entries: list, dry_run: bool = False):
    """Update manifest with corrected 2020 entries."""
    # Read all entries
    all_entries = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row['year'] != '2020':
                all_entries.append(row)

    # Add new 2020 entries
    all_entries.extend(new_2020_entries)

    # Sort by year, submission_id
    all_entries.sort(key=lambda x: (int(x['year']), x['submission_id']))

    if dry_run:
        print(f"  Would update manifest with {len(new_2020_entries)} 2020 entries")
        return

    # Write back
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_entries)

    print(f"  Updated manifest with {len(new_2020_entries)} 2020 entries")


def main():
    parser = argparse.ArgumentParser(description='Fix ICLR 2020 PDF manifest')
    parser.add_argument('--dry-run', action='store_true', help='Show what would change')
    parser.add_argument('--apply', action='store_true', help='Apply the fix')
    args = parser.parse_args()

    if not args.dry_run and not args.apply:
        print("Please specify --dry-run or --apply")
        return

    # Setup paths
    script_dir = Path(__file__).parent.parent  # AutoReviewer/
    data_dir = script_dir / 'data'
    pickle_dir = data_dir / 'full_run'
    manifest_path = data_dir / 'pdf_manifest.csv'
    pdf_dir = data_dir / 'full_run' / 'pdfs'

    print("=" * 80)
    print("FIX ICLR 2020 PDF MANIFEST")
    print(f"Cutoff date: {CUTOFF_DATE} (include through Dec 16 only)")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'APPLY'}")
    print("=" * 80)

    # Step 1: Load current manifest
    print("\n1. Loading current manifest...")
    old_entries = load_old_manifest(manifest_path)
    print(f"   Found {len(old_entries)} 2020 entries")

    # Step 2: Regenerate with date cutoff
    print("\n2. Regenerating 2020 entries with date cutoff...")
    new_entries = regenerate_2020_entries(pickle_dir)
    print(f"   Generated {len(new_entries)} entries")

    # Step 3: Find changes
    print("\n3. Comparing old vs new download_ids...")
    changed = find_changed_entries(old_entries, new_entries)
    print(f"   Found {len(changed)} entries with different download_id")

    if not changed:
        print("\n   No changes needed!")
        return

    # Show sample changes
    print("\n   Sample changes:")
    for entry in changed[:5]:
        print(f"     {entry['submission_id']}: {entry['old_download_id']} -> {entry['new_download_id']}")
    if len(changed) > 5:
        print(f"     ... and {len(changed) - 5} more")

    # Step 4: Backup old PDFs
    print("\n4. Backing up old PDFs...")
    moved = backup_old_pdfs(changed, pdf_dir, dry_run=args.dry_run)
    print(f"   {'Would move' if args.dry_run else 'Moved'} {moved} PDFs to 2020_old_download_id/")

    # Step 5: Download new PDFs
    print("\n5. Downloading new PDFs...")
    downloaded = download_new_pdfs(changed, pdf_dir, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"   Downloaded {downloaded}/{len(changed)} PDFs")

    # Step 6: Update manifest
    print("\n6. Updating manifest...")
    update_manifest(manifest_path, new_entries, dry_run=args.dry_run)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
