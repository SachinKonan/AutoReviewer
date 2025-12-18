#!/usr/bin/env python3
"""
CLI script to generate HuggingFace dataset from ICLR review data.

Usage:
    # Generate parquet files for all years
    python scripts/generate_hf_dataset.py

    # Generate for specific years
    python scripts/generate_hf_dataset.py --years 2024 2025

    # Generate and upload to HuggingFace
    python scripts/generate_hf_dataset.py --upload --repo-id username/iclr-reviews-2020-2026

    # Just upload existing parquet files
    python scripts/generate_hf_dataset.py --upload-only --repo-id username/iclr-reviews-2020-2026
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from lib.hf_dataset import generate_all_datasets, generate_all_datasets_ray
from lib.hf_upload import upload_to_huggingface, save_readme_locally
from lib.utils import get_coverage_stats

# Fixed submission IDs for md_viewer examples (consistent across regenerations)
MD_VIEWER_IDS = {
    ("2020", "accept"): "ryxz8CVYDH",
    ("2020", "reject"): "ryxnJlSKvr",
    ("2021", "accept"): "zeFrfgyZln",
    ("2021", "reject"): "zsKWh2pRSBK",
    ("2022", "accept"): "zzk231Ms1Ih",
    ("2022", "reject"): "zhynF6JnC4q",
    ("2023", "accept"): "zzqBoIFOQ1",
    ("2023", "reject"): "zzL_5WoI3I",
    ("2024", "accept"): "zyBJodMrn5",
    ("2024", "reject"): "zrxlSviRqC",
    ("2025", "accept"): "zxg6601zoc",
    ("2025", "reject"): "zzR1Uskhj0",
    ("2026", "unknown"): "zyCjizqOxB",
}


def export_md_viewer_examples(
    parquet_dir: Path,
    output_dir: Path,
    years: list[int] = None,
    verbose: bool = True
) -> None:
    """
    Export example md files from parquet to md_viewer directory.

    Uses fixed submission IDs from MD_VIEWER_IDS for consistency across regenerations.

    Args:
        parquet_dir: Directory containing parquet files (year.parquet or year/*.parquet)
        output_dir: Output directory for md files (e.g., data/md_viewer)
        years: Years to export (default: all years in MD_VIEWER_IDS)
        verbose: Print progress
    """
    import shutil

    # Get years from MD_VIEWER_IDS if not specified
    if years is None:
        years = sorted(set(int(y) for y, _ in MD_VIEWER_IDS.keys()))

    if verbose:
        print(f"Exporting md_viewer examples for years: {years}")

    # Clean existing year directories
    for year in years:
        year_output = output_dir / str(year)
        if year_output.exists():
            shutil.rmtree(year_output)
            if verbose:
                print(f"  Cleaned {year_output}")

    # Export fixed IDs from parquet
    for (year_str, decision), sub_id in sorted(MD_VIEWER_IDS.items()):
        year = int(year_str)
        if year not in years:
            continue

        # Find parquet files for this year
        year_parquet_dir = parquet_dir / year_str
        parquet_path = parquet_dir / f"{year_str}.parquet"

        chunks = []
        if year_parquet_dir.exists() and year_parquet_dir.is_dir():
            chunks = sorted(year_parquet_dir.glob("*.parquet"))
        elif parquet_path.exists():
            chunks = [parquet_path]

        if not chunks:
            if verbose:
                print(f"  {year}/{decision}/{sub_id}: parquet not found")
            continue

        # Search for the submission ID in parquet chunks
        found = False
        for chunk in chunks:
            df = pd.read_parquet(chunk)
            for _, row in df.iterrows():
                if row['submission']['id'] == sub_id:
                    # Found - write to output
                    dest_dir = output_dir / year_str / decision
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    (dest_dir / f"{sub_id}_raw.md").write_text(row['raw_md'])
                    (dest_dir / f"{sub_id}_clean.md").write_text(row['clean_md'])
                    if verbose:
                        print(f"  {year}/{decision}/{sub_id}")
                    found = True
                    break
            if found:
                break

        if not found and verbose:
            print(f"  {year}/{decision}/{sub_id}: NOT FOUND")


def main():
    parser = argparse.ArgumentParser(
        description="Generate HuggingFace dataset from ICLR review data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/full_run"),
        help="Base data directory containing pickle files and mds/ (default: data/full_run)"
    )

    parser.add_argument(
        "--md-dir",
        type=Path,
        default=None,
        help="MD directory (default: data_dir/mds)"
    )

    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="PDF directory (default: data_dir/pdfs)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for parquet files (default: data_dir/raw_hf_dump)"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Number of processes for parallel processing (default: 16)"
    )

    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="Years to process (default: 2020-2026)"
    )

    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to HuggingFace after generation"
    )

    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip generation, only upload existing parquet files"
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace repo ID (e.g., username/iclr-reviews-2020-2026)"
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the HuggingFace dataset private"
    )

    parser.add_argument(
        "--save-readme",
        type=Path,
        default=None,
        help="Save README.md locally to this path (for review before upload)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    parser.add_argument(
        "--use-ray",
        action="store_true",
        help="Use Ray for parallel processing (faster for large datasets)"
    )

    parser.add_argument(
        "--num-cpus",
        type=int,
        default=16,
        help="Number of CPUs/partitions for Ray (default: 16)"
    )

    parser.add_argument(
        "--export-md-viewer",
        type=Path,
        default=None,
        help="Export example md files to this directory (e.g., data/md_viewer)"
    )

    parser.add_argument(
        "--citations-db",
        type=Path,
        default=Path("data/citations.db"),
        help="Path to citations.db for Google Scholar citations (default: data/citations.db)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.upload or args.upload_only:
        if not args.repo_id:
            parser.error("--repo-id is required for upload")

    md_dir = args.md_dir or args.data_dir / "mds"
    pdf_dir = args.pdf_dir or args.data_dir / "pdfs"
    output_dir = args.output_dir or args.data_dir / "raw_hf_dump"
    years = args.years or list(range(2020, 2027))
    verbose = not args.quiet

    # Generate datasets
    if not args.upload_only:
        if verbose:
            print("=" * 60)
            print("Generating HuggingFace Dataset" + (" (Ray)" if args.use_ray else ""))
            print("=" * 60)

        if args.use_ray:
            stats = generate_all_datasets_ray(
                data_dir=args.data_dir,
                md_dir=md_dir,
                pdf_dir=pdf_dir,
                output_dir=output_dir,
                years=years,
                verbose=verbose,
                num_cpus=args.num_cpus,
                citations_db_path=args.citations_db,
            )
        else:
            stats = generate_all_datasets(
                data_dir=args.data_dir,
                md_dir=md_dir,
                pdf_dir=pdf_dir,
                output_dir=output_dir,
                years=years,
                verbose=verbose,
                max_workers=args.max_workers,
                citations_db_path=args.citations_db,
            )
    else:
        # Load stats from existing parquet files (supports both single file and directory of chunks)
        if verbose:
            print("Loading stats from existing parquet files...")

        stats = {}
        for year in years:
            year_dir = output_dir / str(year)
            parquet_path = output_dir / f"{year}.parquet"

            if year_dir.exists() and year_dir.is_dir():
                # Directory of chunks
                chunks = list(year_dir.glob("*.parquet"))
                if chunks:
                    total = sum(len(pd.read_parquet(f)) for f in chunks)
                    stats[year] = total
                    if verbose:
                        print(f"  {year}: {total:,} papers ({len(chunks)} chunks)")
                else:
                    if verbose:
                        print(f"  {year}: no parquet files in directory")
            elif parquet_path.exists():
                # Single file (legacy format)
                df = pd.read_parquet(parquet_path)
                stats[year] = len(df)
                if verbose:
                    print(f"  {year}: {len(df):,} papers")
            else:
                if verbose:
                    print(f"  {year}: parquet file not found")

    # Save stats to JSON
    stats_path = output_dir / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    if verbose:
        print(f"\nSaved stats to {stats_path}")

    # Get full coverage stats for README (from utils.py)
    if verbose:
        print("\nComputing coverage statistics...")
    coverage_stats = get_coverage_stats(
        data_dir=args.data_dir,
        md_dir=md_dir,
        years=years
    )

    # Save coverage stats to JSON
    coverage_stats_path = output_dir / "coverage_stats.json"
    with open(coverage_stats_path, 'w') as f:
        json.dump(coverage_stats, f, indent=2)
    if verbose:
        print(f"Saved coverage stats to {coverage_stats_path}")

    # Save README locally
    if args.save_readme:
        username = args.repo_id.split("/")[0] if args.repo_id and "/" in args.repo_id else "username"
        save_readme_locally(
            stats,
            args.save_readme,
            coverage_stats=coverage_stats,
            username=username
        )

    # Upload to HuggingFace
    if args.upload or args.upload_only:
        if verbose:
            print("\n" + "=" * 60)
            print("Uploading to HuggingFace")
            print("=" * 60)

        upload_to_huggingface(
            repo_id=args.repo_id,
            data_dir=output_dir,
            stats=stats,
            coverage_stats=coverage_stats,
            private=args.private
        )

    # Export md_viewer examples
    if args.export_md_viewer:
        if verbose:
            print("\n" + "=" * 60)
            print("Exporting md_viewer examples")
            print("=" * 60)

        export_md_viewer_examples(
            parquet_dir=output_dir,
            output_dir=args.export_md_viewer,
            years=years,
            verbose=verbose
        )

    if verbose:
        print("\nDone!")


if __name__ == "__main__":
    main()
