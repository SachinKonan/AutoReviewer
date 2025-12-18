"""Upload dataset to HuggingFace."""

from pathlib import Path
from textwrap import dedent
from typing import Dict, Optional


def generate_readme(
    stats: Dict[int, int],
    coverage_stats: Optional[Dict] = None,
    username: str = "username",
    year_range: Optional[str] = None
) -> str:
    """
    Generate README.md content for HuggingFace.

    Args:
        stats: Dict mapping year -> number of complete papers in dataset
        coverage_stats: Optional full coverage stats from lib.utils.get_coverage_stats()
        username: HuggingFace username for example code
        year_range: Custom year range string (e.g., "2020-2025"), auto-detected from stats if None

    Returns:
        README content as string
    """
    # Auto-detect year range from stats if not provided
    if year_range is None:
        years = sorted(stats.keys())
        year_range = f"{min(years)}-{max(years)}" if years else "2020-2025"
    # Build coverage table
    if coverage_stats:
        # Use detailed stats from utils.get_coverage_stats()
        coverage_rows = []
        for year_data in coverage_stats['years']:
            year = year_data['year']
            complete = stats.get(year, 0)  # Use actual dataset count
            reviewable = year_data['reviewable']
            pct = round(100 * complete / reviewable, 1) if reviewable > 0 else 0
            reject = year_data['reject']
            poster = year_data['poster']
            spotlight = year_data['spotlight']
            oral = year_data['oral']
            coverage_rows.append(
                f"| {year} | {reviewable:,} | {complete:,} | {pct}% | {reject:,} | {poster:,} | {spotlight:,} | {oral:,} |"
            )

        totals = coverage_stats['totals']
        total_complete = sum(stats.values())
        total_reviewable = totals['reviewable']
        total_pct = round(100 * total_complete / total_reviewable, 1) if total_reviewable > 0 else 0

        coverage_table = "\n".join([
            "| Year | Reviewable | Complete | % | Reject | Poster | Spotlight | Oral |",
            "|------|-----------|----------|---|--------|--------|-----------|------|",
            *coverage_rows,
            f"| **Total** | **{total_reviewable:,}** | **{total_complete:,}** | **{total_pct}%** | **{totals['reject']:,}** | **{totals['poster']:,}** | **{totals['spotlight']:,}** | **{totals['oral']:,}** |"
        ])

        coverage_notes = """
**Notes:**
- **Reviewable** = Total submissions minus Withdrawn and Desk Rejected
- **Complete** = Has MD file with detected Abstract and References sections
- Decision counts are for reviewable papers only
"""
    else:
        # Simple table from just stats dict
        coverage_rows = []
        total = 0
        for year in sorted(stats.keys()):
            count = stats[year]
            total += count
            coverage_rows.append(f"| {year} | {count:,} |")

        coverage_table = "\n".join([
            "| Year | Complete Papers |",
            "|------|-----------------|",
            *coverage_rows,
            f"| **Total** | **{total:,}** |"
        ])
        coverage_notes = ""

    # Build configs for each year
    years = sorted(stats.keys())
    configs_yaml = "\n".join([
        f'  - config_name: "{year}"\n    data_files: "{year}.parquet"'
        for year in years
    ])

    readme = dedent(f"""
---
license: apache-2.0
task_categories:
  - text-generation
  - text-classification
language:
  - en
pretty_name: ICLR Reviews {year_range}
size_categories:
  - 10K<n<100K
tags:
  - peer-review
  - scientific-papers
  - iclr
  - openreview
configs:
{configs_yaml}
default: "{years[-1] if years else 2025}"
---

# ICLR Reviews Dataset ({year_range})

Peer reviews, meta-reviews, and paper content from ICLR conferences ({year_range}).

## Dataset Description

This dataset contains **complete papers** from ICLR {year_range} with:
- Submission metadata (title, abstract, authors, decision)
- Up to 12 reviewer assessments with response counts
- Area Chair meta-reviews
- Raw and cleaned markdown paper content
- Section-level breakdown

## Filtering Criteria

**Complete papers** must have:
- PDF available on OpenReview
- Markdown conversion successful
- Both Abstract and References sections detected
- Not Withdrawn or Desk Rejected
- Has a final decision

## Abstract Normalization

The `abstract` field is normalized for anonymization:
- **Sentences containing URLs** are removed entirely (not just the URL)
- This catches GitHub links, project pages, code repositories, etc.
- Original abstract preserved in `_original_abstract` field

This enables blind review analysis without leaking author identity through URLs.

## Coverage Statistics

{coverage_table}
{coverage_notes}
## Schema

Each row contains:

| Field | Type | Description |
|-------|------|-------------|
| `submission` | dict | Paper metadata (id, title, abstract, decision, authors, etc.) |
| `review_1` to `review_12` | dict/None | Reviewer assessments (None if fewer reviews) |
| `meta_review` | dict/None | Area Chair assessment |
| `raw_md` | str | Raw markdown from PDF conversion |
| `clean_md` | str | Cleaned markdown (Introduction → References) |
| `clean_md_sections` | dict | Mapping of section titles to content |
| `md_path` | str | Local path to markdown file |
| `pdf_path` | str/None | Local path to PDF file (None if not found) |

### Submission Fields

```python
{{
    "id": "abc123xyz",           # OpenReview ID
    "title": "Paper Title",
    "abstract": "Abstract text...",
    "decision": "Accept (Poster)",
    "authors": ["Author One", "Author Two"],
    "keywords": ["machine learning", "nlp"],
    "venue": "ICLR 2024 poster",
    "pdf_url": "/pdf/abc123xyz.pdf",
    "created_date": 1699574400,  # Unix timestamp
    "modified_date": 1699574400,
    "tldr": "Short summary...",
    "primary_area": "machine learning",
    "google_scholar_citations": 42,  # Google Scholar citation count (None if not found)
}}
```

### Review Fields

Review schemas vary by year due to OpenReview changes:

| Year | Rating Field | Key Content Fields |
|------|--------------|-------------------|
| 2020 | `rating` (str) | `review`, `experience_assessment`, `review_assessment_*` |
| 2021 | `rating` (str) | `review`, `confidence` |
| 2022 | `recommendation` (str) | `main_review`, `summary_of_the_paper`, `correctness`, `novelty` |
| 2023 | `recommendation` (str) | `strength_and_weaknesses`, `summary_of_the_paper`, `correctness`, `novelty` |
| 2024-2026 | `rating` (int) | `summary`, `strengths`, `weaknesses`, `soundness`, `presentation`, `contribution` |

All reviews include:
- `number_of_author_responses`: Author replies to this review
- `number_of_reviewer_responses_to_author`: Reviewer follow-ups

### Rating Scales

| Year | Rating Scale | Confidence | Sub-scores |
|------|--------------|------------|------------|
| 2020 | 1, 3, 6, 8 (str) | N/A | N/A |
| 2021 | 1-10 (str) | 1-5 (str) | N/A |
| 2022-2023 | 1,3,5,6,8,10 (str) | 1-5 (str) | correctness, novelty (str 1-4) |
| 2024-2026 | 1-10 (int) | 1-5 (int) | soundness, presentation, contribution (int 1-4) |

## Markdown Normalization

The `clean_md` field contains normalized markdown produced from the raw PDF conversion. The normalization pipeline:

### 1. Validation
- Paper must have both **Abstract** and **References** sections detected
- Papers failing validation are excluded from the dataset

### 2. Content Clipping
- **Start**: First section header AFTER Abstract (typically "Introduction")
- **End**: End of References section (before Appendix/Supplementary)
- This removes: title, authors, abstract, appendices, supplementary material

### 3. Section Removal
- **Acknowledgements**: Removed entirely (to preserve anonymity for blind review analysis)
- **Reproducibility**: Removed entirely (often contains author-identifying information)

### 4. Artifact Cleaning
- **Line numbers**: Removed (e.g., `**054 055 056**` remnants from submitted PDFs)
- **Standalone number lines**: Removed (bare PDF line numbers like `327`, `337 338`)
- **Page anchors**: Lines containing `<span id="page-X">...<sup>` removed entirely
- **Code/GitHub refs**: Entire sentences containing `code...https://github...` removed (author code)
- **Footnotes**: Removed except those referencing figures
- **Dagger markers**: Removed (†, ‡) except figure references

### 5. Header Normalization
- All headers normalized to single `#` level
- Titles converted to UPPERCASE
- Span tags and bold markers removed
- Example: `## 3.1 **<span>Methods</span>**` → `# 3.1 METHODS`

### 6. Whitespace Normalization
- Multiple blank lines collapsed to single blank line
- Trailing whitespace stripped

### Section Breakdown

The `clean_md_sections` field provides a dict mapping normalized section titles to content:

```python
{{
    "INTRODUCTION": "Section content...",
    "RELATED WORK": "Section content...",
    "METHODS": "Section content...",
    "EXPERIMENTS": "Section content...",
    "CONCLUSION": "Section content...",
    "REFERENCES": "Reference list..."
}}
```

Note: Section titles vary by paper. Common sections include INTRODUCTION, RELATED WORK, METHOD/METHODS, EXPERIMENTS, RESULTS, DISCUSSION, CONCLUSION, REFERENCES.

## Usage

```python
from datasets import load_dataset

# Load a specific year (as a config/subset)
ds = load_dataset("{username}/iclr-data-{year_range}", "2024")

# Load default (most recent year)
ds = load_dataset("{username}/iclr-data-{year_range}")

# Access data
for row in ds["train"]:
    print(row["submission"]["title"])
    print(row["submission"]["decision"])

    # Access reviews
    if row["review_1"]:
        print(row["review_1"]["rating"])

    # Access sections
    intro = row["clean_md_sections"].get("INTRODUCTION", "")
    print(intro[:500])
```

## Data Source

Data extracted from [OpenReview](https://openreview.net/) using the OpenReview API.
Paper PDFs converted to markdown using [Marker](https://github.com/VikParuchuri/marker).

## License

Apache 2.0

## Citation

If you use this dataset, please cite:

```bibtex
@misc{{iclr-data-{year_range},
  title={{ICLR Reviews Dataset {year_range}}},
  author={{OpenReview Community}},
  year={{2024}},
  howpublished={{HuggingFace Datasets}},
  url={{https://huggingface.co/datasets/{username}/iclr-data-{year_range}}}
}}
```
""").strip()

    return readme


def upload_to_huggingface(
    repo_id: str,
    data_dir: Path,
    stats: Dict[int, int],
    coverage_stats: Optional[Dict] = None,
    private: bool = False
):
    """
    Upload parquet files and README to HuggingFace.

    Args:
        repo_id: HuggingFace repo ID (e.g., "username/iclr-reviews-2020-2026")
        data_dir: Directory containing parquet files
        stats: Dict mapping year -> number of papers
        coverage_stats: Optional full coverage stats from lib.utils.get_coverage_stats()
        private: Whether to make the repo private

    Returns:
        URL of the created dataset
    """
    from huggingface_hub import HfApi

    api = HfApi()

    # Create repo if needed
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)

    # Extract username from repo_id
    username = repo_id.split("/")[0] if "/" in repo_id else "username"

    # Upload README
    readme = generate_readme(stats, coverage_stats=coverage_stats, username=username)
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded README.md to {repo_id}")

    # Upload parquet files
    import pandas as pd
    import tempfile

    for year in sorted(stats.keys()):
        parquet_path = data_dir / f"{year}.parquet"
        year_dir = data_dir / str(year)

        if parquet_path.exists():
            # Single file format
            api.upload_file(
                path_or_fileobj=str(parquet_path),
                path_in_repo=f"{year}.parquet",
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"Uploaded {year}.parquet ({stats[year]:,} papers)")
        elif year_dir.exists() and year_dir.is_dir():
            # Chunked directory format - merge and upload
            chunks = sorted(year_dir.glob("*.parquet"))
            if chunks:
                print(f"Merging {len(chunks)} chunks for {year}...")
                dfs = [pd.read_parquet(f) for f in chunks]
                merged = pd.concat(dfs, ignore_index=True)

                # Write to temp file and upload
                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                    merged.to_parquet(tmp.name, index=False)
                    api.upload_file(
                        path_or_fileobj=tmp.name,
                        path_in_repo=f"{year}.parquet",
                        repo_id=repo_id,
                        repo_type="dataset",
                    )
                print(f"Uploaded {year}.parquet ({len(merged):,} papers)")
            else:
                print(f"Skipped {year}: no parquet files in directory")
        else:
            print(f"Skipped {year}: parquet file not found")

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"\nDataset available at: {url}")
    return url


def save_readme_locally(
    stats: Dict[int, int],
    output_path: Path,
    coverage_stats: Optional[Dict] = None,
    username: str = "username"
):
    """
    Save README locally for review before upload.

    Args:
        stats: Dict mapping year -> number of papers
        output_path: Path to save README.md
        coverage_stats: Optional full coverage stats from lib.utils.get_coverage_stats()
        username: HuggingFace username for example code
    """
    readme = generate_readme(stats, coverage_stats=coverage_stats, username=username)
    output_path.write_text(readme, encoding="utf-8")
    print(f"Saved README to {output_path}")
