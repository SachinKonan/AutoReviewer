"""
MinerU content_list.json normalization pipeline.

Processes MinerU-converted papers:
1. Build index with 2020 batch priority (pt2 > fixed > other)
2. Load submissions from pickle (use abstract directly)
3. Filter content: remove before first header, after references, footnotes
4. Standardize headers: uppercase text_level=1 only
5. Generate: filtered JSON, markdown, chopped PDF, metadata
6. Symlink images directory
"""

import argparse
import bisect
import copy
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pymupdf as fitz  # PyMuPDF for true redaction
from tqdm import tqdm

# Add parent to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.submission import load_submissions_from_pickle
from lib.utils import build_pdf_file_index, find_pdf_path, EXCLUDED_DECISIONS, DEFAULT_YEARS


# =============================================================================
# CONSTANTS AND PATTERNS
# =============================================================================

GITHUB_URL_PATTERN = re.compile(r'https?://github\.com[^\s\)\]\"\'<>]*', re.IGNORECASE)
GITHUB_SIMPLE_PATTERN = re.compile(r'github', re.IGNORECASE)
REFERENCES_PATTERN = re.compile(r'^r\s*e\s*f\s*e\s*r\s*e\s*n\s*c\s*e\s*s?$', re.IGNORECASE)
ABSTRACT_PATTERN = re.compile(r'^a\s*b\s*s\s*t\s*r\s*a\s*c\s*t$', re.IGNORECASE)

# Patterns for sections to remove (between body and references)
REPRODUCIBILITY_PATTERN = re.compile(r'repro', re.IGNORECASE)
ACKNOWLEDGMENTS_PATTERN = re.compile(r'acknowledg', re.IGNORECASE)

# Pattern for sentences containing https://github (removes entire sentence)
# Uses non-greedy matching and stops URL at period+space boundary
GITHUB_SENTENCE_PATTERN = re.compile(
    r'(?<=\.)\s*[^.]*?https?://github\.com[^\s\)\]\"\'<>]*[^.]*?\.'
    r'|^[^.]*?https?://github\.com[^\s\)\]\"\'<>]*[^.]*?\.',
    re.IGNORECASE
)


# =============================================================================
# INDEX BUILDING (from analyze_mineru_stats.py)
# =============================================================================

def build_mineru_index(mineru_dir: Path) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    """Build index mapping folder name -> content_list.json path across all batches.

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
        with os.scandir(batch_dir) as entries:
            for entry in entries:
                if entry.is_dir():
                    folder_name = entry.name
                    content_list_path = Path(entry.path) / "vlm" / f"{folder_name}_content_list.json"
                    if content_list_path.exists() and content_list_path.stat().st_size > 10:
                        index[folder_name] = content_list_path
                        if batch_name == "batch_2020_fixed":
                            fixed_index[folder_name] = content_list_path
                        elif batch_name == "batch_2020_fixed_pt2":
                            fixed_pt2_index[folder_name] = content_list_path

    return index, fixed_index, fixed_pt2_index


def _prefix_lookup(submission_id: str, sorted_keys: list[str], index: dict[str, Path]) -> tuple[str, Path] | None:
    """Binary search for submission_id prefix in sorted keys."""
    i = bisect.bisect_left(sorted_keys, submission_id)
    if i < len(sorted_keys) and sorted_keys[i].startswith(submission_id):
        folder_name = sorted_keys[i]
        return folder_name, index[folder_name]
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
) -> tuple[str, Path] | None:
    """Find MinerU content_list by submission ID prefix using binary search.

    For year=2020, checks indices in priority order:
    1. fixed_pt2_index (batch_2020_fixed_pt2) - highest priority
    2. fixed_index (batch_2020_fixed)
    3. main index (all batches)

    Returns (folder_name, content_list_path) or None.
    """
    if year == 2020:
        if fixed_pt2_keys and fixed_pt2_index:
            result = _prefix_lookup(submission_id, fixed_pt2_keys, fixed_pt2_index)
            if result:
                return result
        if fixed_keys and fixed_index:
            result = _prefix_lookup(submission_id, fixed_keys, fixed_index)
            if result:
                return result

    return _prefix_lookup(submission_id, sorted_keys, index)


# =============================================================================
# CONTENT ANALYSIS
# =============================================================================

def is_section_header(item: dict) -> bool:
    """Check if content item is a section header (text_level > 0)."""
    return item.get('type') == 'text' and item.get('text_level', 0) > 0


def is_abstract_header(item: dict) -> bool:
    """Check if item is an abstract section header."""
    if not is_section_header(item):
        return False
    text = item.get('text', '').strip()
    return bool(ABSTRACT_PATTERN.match(text))


def is_references_header(item: dict) -> bool:
    """Check if item is a references section header."""
    if not is_section_header(item):
        return False
    text = item.get('text', '').strip()
    return bool(REFERENCES_PATTERN.match(text))


def has_github_link(text: str) -> bool:
    """Check if text contains github link or mention."""
    return bool(GITHUB_URL_PATTERN.search(text)) or bool(GITHUB_SIMPLE_PATTERN.search(text))


def anonymize_abstract(abstract: str) -> tuple[str, bool]:
    """Remove github links from abstract text.

    Removes entire sentences containing https://github.

    Returns:
        (anonymized_text, had_github): Tuple of cleaned text and whether github was found
    """
    if not abstract:
        return "", False

    # Split into sentences (split on period/exclamation/question followed by space or end)
    sentences = re.split(r'(?<=[.!?])\s+', abstract)

    # Filter out sentences containing github URLs
    filtered = [s for s in sentences if 'https://github' not in s.lower()]

    # Rejoin
    result = ' '.join(filtered)

    # Clean up whitespace
    result = re.sub(r'\s+', ' ', result).strip()

    had_github = len(filtered) < len(sentences)
    return result, had_github


# =============================================================================
# CONTENT FILTERING
# =============================================================================

def filter_content_list(content_list: list[dict]) -> tuple[list[dict], dict]:
    """
    Filter content list according to normalization rules.

    Rules:
    1. Find ABSTRACT header (text_level=1)
    2. Use bbox analysis to find body start: abstract text is indented (larger left_x)
       vs body text (smaller left_x). Body starts when left_x drops significantly.
    3. Remove all content before body start (title, authors, abstract, etc.)
    4. Remove all content after references section
    5. Remove first-page footnotes (page_idx == 0)
    6. Remove any footnote containing github
    7. Uppercase text_level=1 headers
    8. Track type="header" items for whiteout

    Returns:
        (filtered_list, metadata_dict)
    """
    metadata = {
        'abstract_idx': -1,
        'intro_idx': -1,  # First section after abstract (usually INTRODUCTION)
        'references_idx': -1,
        'references_end_idx': -1,
        'page_end': -1,
        'removed_before_intro': [],  # Renamed: now removes up to intro, not just abstract
        'removed_footnotes': [],
        'removed_after_refs': [],
        'removed_reproducibility': [],  # Reproducibility section items
        'removed_acknowledgments': [],  # Acknowledgments section items
        'headers_to_whiteout': [],  # type="header" items to white out
        'has_abstract': False,
        'has_references': False,
    }

    if not content_list:
        return [], metadata

    # Find ABSTRACT header (text_level=1)
    abstract_idx = -1
    for i, item in enumerate(content_list):
        if is_abstract_header(item):
            abstract_idx = i
            metadata['abstract_idx'] = i
            metadata['has_abstract'] = True
            break

    if abstract_idx < 0:
        # No abstract found - use first text_level=1 header as fallback
        for i, item in enumerate(content_list):
            if item.get('text_level') == 1:
                abstract_idx = i
                metadata['abstract_idx'] = i
                break

    if abstract_idx < 0:
        return [], metadata

    # Find body start using bbox analysis
    # Abstract text is indented (larger left_x ~228-233) vs body text (~169-174)
    intro_idx = abstract_idx + 1  # Default: start after abstract header

    if abstract_idx + 1 < len(content_list):
        abstract_text_item = content_list[abstract_idx + 1]
        abstract_bbox = abstract_text_item.get('bbox', [])

        if abstract_bbox:
            abstract_left_x = abstract_bbox[0]
            threshold = 30  # Body text is ~55-60 pixels further left

            for i in range(abstract_idx + 2, len(content_list)):
                item = content_list[i]
                bbox = item.get('bbox', [])

                if bbox:
                    left_x = bbox[0]
                    # Body text has smaller left_x (further left on page)
                    if abstract_left_x - left_x > threshold:
                        intro_idx = i
                        break

                # Also check for explicit section header
                if item.get('text_level') == 1:
                    intro_idx = i
                    break
        else:
            # No bbox on abstract text - fall back to two-hop
            intro_idx = abstract_idx + 2 if abstract_idx + 2 < len(content_list) else abstract_idx + 1

    metadata['intro_idx'] = intro_idx

    # Find REFERENCES header
    references_idx = -1
    for i, item in enumerate(content_list):
        if is_references_header(item):
            references_idx = i
            metadata['references_idx'] = i
            metadata['has_references'] = True
            break

    # Find end of references by locating last ref_text item
    references_end_idx = len(content_list)
    if references_idx >= 0:
        # Find the last occurrence of type="list", sub_type="ref_text"
        last_ref_idx = references_idx
        for i in range(references_idx + 1, len(content_list)):
            item = content_list[i]
            if item.get('type') == 'list' and item.get('sub_type') == 'ref_text':
                last_ref_idx = i

        # page_end is the page of the last reference
        metadata['page_end'] = content_list[last_ref_idx].get('page_idx', 0)
        # references_end_idx is everything after the last ref
        references_end_idx = last_ref_idx + 1
        metadata['references_end_idx'] = references_end_idx
    else:
        # No references found - use last page
        metadata['page_end'] = max(item.get('page_idx', 0) for item in content_list)

    # Find reproducibility and acknowledgments sections (between intro and references)
    # These appear after the main body but before references
    reproducibility_start = -1
    reproducibility_end = -1
    acknowledgments_start = -1
    acknowledgments_end = -1
    search_end = references_idx if references_idx > 0 else len(content_list)

    for i in range(intro_idx, search_end):
        item = content_list[i]
        if item.get('text_level') == 1:
            text = item.get('text', '').strip()
            if REPRODUCIBILITY_PATTERN.search(text):
                reproducibility_start = i
                # Find end: next text_level=1 or references
                for j in range(i + 1, search_end):
                    if content_list[j].get('text_level') == 1:
                        reproducibility_end = j
                        break
                else:
                    reproducibility_end = search_end
            elif ACKNOWLEDGMENTS_PATTERN.search(text):
                acknowledgments_start = i
                # Find end: next text_level=1 or references
                for j in range(i + 1, search_end):
                    if content_list[j].get('text_level') == 1:
                        acknowledgments_end = j
                        break
                else:
                    acknowledgments_end = search_end

    # Build filtered list
    filtered = []

    for i, item in enumerate(content_list):
        item_type = item.get('type', '')

        # Track type="header" items for whiteout (but don't remove from content)
        if item_type == 'header':
            metadata['headers_to_whiteout'].append({
                'index': i,
                'bbox': item.get('bbox'),
                'page_idx': item.get('page_idx'),
                'type': item_type,
            })

        # Skip all content before INTRO (title, authors, abstract header, abstract text, etc.)
        if i < intro_idx:
            metadata['removed_before_intro'].append({
                'index': i,
                'bbox': item.get('bbox'),
                'page_idx': item.get('page_idx'),
                'type': item_type,
            })
            continue

        # Skip content after references end (if references found)
        if references_idx >= 0 and i >= references_end_idx:
            metadata['removed_after_refs'].append({
                'index': i,
                'bbox': item.get('bbox'),
                'page_idx': item.get('page_idx'),
                'type': item_type,
            })
            continue

        # Skip reproducibility section
        if reproducibility_start >= 0 and reproducibility_start <= i < reproducibility_end:
            metadata['removed_reproducibility'].append({
                'index': i,
                'bbox': item.get('bbox'),
                'page_idx': item.get('page_idx'),
                'type': item_type,
            })
            continue

        # Skip acknowledgments section
        if acknowledgments_start >= 0 and acknowledgments_start <= i < acknowledgments_end:
            metadata['removed_acknowledgments'].append({
                'index': i,
                'bbox': item.get('bbox'),
                'page_idx': item.get('page_idx'),
                'type': item_type,
            })
            continue

        # Skip first-page footnotes OR github-containing footnotes
        if item_type in ('page_footnote', 'footer'):
            page_idx = item.get('page_idx', 0)
            text = item.get('text', '')
            if page_idx == 0 or has_github_link(text):
                metadata['removed_footnotes'].append({
                    'index': i,
                    'bbox': item.get('bbox'),
                    'page_idx': page_idx,
                    'text': text[:100] if text else '',
                })
                continue

        # Skip type="header" items (they're tracked but not included in output)
        if item_type == 'header':
            continue

        # Copy item (will modify headers)
        item_copy = copy.deepcopy(item)

        # Uppercase text_level=1 headers only
        if item_copy.get('text_level') == 1:
            item_copy['text'] = item_copy.get('text', '').upper()

        filtered.append(item_copy)

    return filtered, metadata


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def generate_markdown(content_list: list[dict]) -> str:
    """Generate markdown from filtered content list."""
    md_parts = []

    for item in content_list:
        item_type = item.get('type', '')
        text = item.get('text', '')
        text_level = item.get('text_level', 0)

        if item_type == 'text':
            if text_level > 0:
                # Section header
                md_parts.append(f"\n{'#' * text_level} {text}\n")
            else:
                # Regular paragraph
                md_parts.append(f"\n{text}\n")

        elif item_type == 'image':
            img_path = item.get('img_path', '')
            captions = item.get('image_caption', [])
            caption = ' '.join(captions) if captions else ''
            if img_path:
                md_parts.append(f"\n![{caption}]({img_path})\n")

        elif item_type == 'table':
            html = item.get('table_body', '')
            captions = item.get('table_caption', [])
            caption = ' '.join(captions) if captions else ''
            if html:
                if caption:
                    md_parts.append(f"\n{caption}\n")
                md_parts.append(f"\n{html}\n")
            elif item.get('img_path'):
                md_parts.append(f"\n![{caption}]({item['img_path']})\n")

        elif item_type == 'equation':
            latex = item.get('text', '')
            if latex:
                md_parts.append(f"\n$$\n{latex}\n$$\n")

        elif item_type == 'list':
            list_items = item.get('list_items', [])
            for li in list_items:
                md_parts.append(f"- {li}\n")

        elif item_type == 'code':
            code_body = item.get('code_body', '')
            lang = item.get('guess_lang', '')
            if code_body:
                md_parts.append(f"\n```{lang}\n{code_body}\n```\n")

    return '\n'.join(md_parts)


def load_page_sizes_from_middle_json(middle_json_path: Path) -> dict[int, tuple[float, float]]:
    """Load page sizes from middle.json for coordinate conversion."""
    try:
        with open(middle_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {i: tuple(page['page_size']) for i, page in enumerate(data['pdf_info'])}
    except Exception:
        return {}


def normalized_to_pdf_points(bbox: list, page_size: tuple) -> list:
    """Convert normalized 0-1000 bbox to PDF points.

    content_list.json bboxes use normalized 0-1000 range:
    - Formula: normalized = (pdf_points * 1000) / page_dimension
    - Reverse: pdf_points = (normalized * page_dimension) / 1000
    """
    x0, y0, x1, y1 = bbox
    page_width, page_height = page_size
    return [
        x0 * page_width / 1000,
        y0 * page_height / 1000,
        x1 * page_width / 1000,
        y1 * page_height / 1000,
    ]


def process_pdf(
    pdf_path: Path,
    middle_json_path: Path,
    output_path: Path,
    items_to_whiteout: list[dict],
    page_end: int,
) -> bool:
    """
    Process PDF: redact specified regions + trim pages after references.

    1. Redact: content before intro, footnotes, headers, after refs on kept pages
    2. Page trim: keep only pages 0 through page_end (inclusive)

    Uses PyMuPDF (fitz) for TRUE redaction - actually removes content, not just overlay.
    Converts bboxes from normalized 0-1000 range to PDF points.
    """
    try:
        # Load page sizes from middle.json
        page_sizes = load_page_sizes_from_middle_json(middle_json_path)

        # Group redaction items by page
        redact_by_page = defaultdict(list)
        for item in items_to_whiteout:
            bbox = item.get('bbox')
            page_idx = item.get('page_idx', 0)
            if bbox and len(bbox) >= 4:
                redact_by_page[page_idx].append(bbox)

        # Open PDF with PyMuPDF
        doc = fitz.open(str(pdf_path))

        # Process pages for redaction (only pages we're keeping)
        max_page = min(len(doc), page_end + 1)

        for page_idx in range(max_page):
            if page_idx not in redact_by_page:
                continue

            page = doc[page_idx]
            page_rect = page.rect
            page_width = page_rect.width
            page_height = page_rect.height

            # Get page size for coordinate conversion (prefer middle.json)
            page_size = page_sizes.get(page_idx, (page_width, page_height))

            for bbox in redact_by_page[page_idx]:
                # Convert normalized 0-1000 to PDF points
                x0, y0, x1, y1 = normalized_to_pdf_points(bbox, page_size)

                # PyMuPDF uses top-left origin like content_list, no Y-flip needed
                # But we need to ensure coords are within page bounds
                rect = fitz.Rect(x0, y0, x1, y1)

                # Add redaction annotation (white fill)
                page.add_redact_annot(rect, fill=(1, 1, 1))  # White fill

            # Apply all redactions on this page (actually removes content)
            page.apply_redactions()

        # Delete pages after page_end
        if len(doc) > page_end + 1:
            doc.delete_pages(from_page=page_end + 1, to_page=len(doc) - 1)

        # Save the modified PDF
        doc.save(str(output_path), garbage=4, deflate=True)
        doc.close()

        return True

    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# MAIN NORMALIZATION
# =============================================================================

def normalize_paper(
    submission_id: str,
    year: int,
    content_list_path: Path,
    pdf_path: Path,
    abstract_from_pickle: str,
    output_dir: Path,
) -> dict:
    """Normalize a single paper."""
    result = {
        'submission_id': submission_id,
        'year': year,
        'success': False,
        'error': None,
    }

    try:
        # Load content list
        with open(content_list_path, 'r', encoding='utf-8') as f:
            content_list = json.load(f)

        if not content_list:
            result['error'] = 'Empty content list'
            return result

        # Filter content
        filtered_list, metadata = filter_content_list(content_list)

        if not filtered_list:
            result['error'] = 'No content after filtering'
            return result

        # Create output directory (remove existing if present)
        paper_dir = output_dir / str(year) / submission_id
        if paper_dir.exists():
            shutil.rmtree(paper_dir)
        paper_dir.mkdir(parents=True, exist_ok=True)

        # Create symlink to images
        src_images = content_list_path.parent / "images"
        dst_images = paper_dir / "images"
        if src_images.exists() and not dst_images.exists():
            dst_images.symlink_to(src_images.resolve())

        # Write filtered JSON
        json_path = paper_dir / f"{submission_id}_content_list.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_list, f, indent=2, ensure_ascii=False)

        # Generate and write markdown
        markdown = generate_markdown(filtered_list)
        md_path = paper_dir / f"{submission_id}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown)

        # Process PDF: whiteout + page trim
        if pdf_path and pdf_path.exists():
            # Construct middle.json path (same directory as content_list)
            folder_name = content_list_path.parent.name
            middle_json_path = content_list_path.parent / f"{folder_name}_middle.json"

            # White out: before intro + footnotes + type="header" + after refs (on kept pages)
            # Filter after_refs to only include items on pages that won't be trimmed
            page_end = metadata['page_end']
            after_refs_on_kept_pages = [
                item for item in metadata['removed_after_refs']
                if item.get('page_idx', 0) <= page_end
            ]

            items_to_whiteout = (
                metadata['removed_before_intro'] +
                metadata['removed_footnotes'] +
                metadata['removed_reproducibility'] +
                metadata['removed_acknowledgments'] +
                metadata['headers_to_whiteout'] +
                after_refs_on_kept_pages
            )

            processed_pdf_path = paper_dir / f"{submission_id}.pdf"
            process_pdf(
                pdf_path,
                middle_json_path,
                processed_pdf_path,
                items_to_whiteout,
                metadata['page_end'],  # Trim pages after this
            )

        # Write metadata
        anonymized_abstract, had_github_in_abstract = anonymize_abstract(abstract_from_pickle)
        meta = {
            'submission_id': submission_id,
            'year': year,
            # Paths
            'original_pdf_path': str(pdf_path) if pdf_path else None,
            'mineru_dir': str(content_list_path.parent),
            'output_dir': str(paper_dir),
            # Content info
            'has_abstract': metadata['has_abstract'],
            'has_references': metadata['has_references'],
            'abstract_text': abstract_from_pickle,
            'anonymized_abstract': anonymized_abstract,
            'page_end': metadata['page_end'],
            # Removal counts
            'removed_before_intro_count': len(metadata['removed_before_intro']),
            'removed_footnotes_count': len(metadata['removed_footnotes']),
            'removed_after_refs_pages': len(metadata['removed_after_refs']),
            'removed_reproducibility_count': len(metadata['removed_reproducibility']),
            'removed_acknowledgments_count': len(metadata['removed_acknowledgments']),
            'removed_github_in_abstract': had_github_in_abstract,
            'headers_whiteout_count': len(metadata['headers_to_whiteout']),
        }
        meta_path = paper_dir / f"{submission_id}_meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        result['success'] = True
        result['has_abstract'] = metadata['has_abstract']
        result['has_references'] = metadata['has_references']

    except Exception as e:
        result['error'] = str(e)

    return result


def run_pipeline(
    data_dir: Path = Path("data/full_run"),
    output_dir: Path = None,
    years: list[int] = None,
    limit: int = None,
    submission_id: str = None,
    verbose: bool = True,
    workers: int = 1,
    task_id: int = None,
    num_tasks: int = None,
):
    """Run the normalization pipeline.

    Args:
        workers: Number of parallel workers. If > 1, uses ThreadPoolExecutor.
        task_id: Task ID for job array splitting (0-indexed).
        num_tasks: Total number of tasks in job array.
    """
    if output_dir is None:
        output_dir = data_dir / "normalized"

    years = years or DEFAULT_YEARS
    mineru_dir = data_dir / "md_mineru"
    pdf_dir = data_dir / "pdfs"

    # Build indices
    if verbose:
        print("Building MinerU index...")
    mineru_index, fixed_index, fixed_pt2_index = build_mineru_index(mineru_dir)
    mineru_keys = sorted(mineru_index.keys())
    fixed_keys = sorted(fixed_index.keys())
    fixed_pt2_keys = sorted(fixed_pt2_index.keys())

    if verbose:
        print(f"  Total MinerU outputs: {len(mineru_index)}")
        print(f"  batch_2020_fixed: {len(fixed_index)}")
        print(f"  batch_2020_fixed_pt2: {len(fixed_pt2_index)}")

    # Build PDF index
    if verbose:
        print("Building PDF index...")
    pdf_index = build_pdf_file_index(pdf_dir, years)

    # Load submissions from pickle
    submissions_by_id = {}
    for year in years:
        pkl_path = data_dir / f"get_all_notes_{year}.pickle"
        if not pkl_path.exists():
            if verbose:
                print(f"  {year}: pickle not found, skipping")
            continue

        subs = load_submissions_from_pickle(pkl_path, year)
        non_excluded = [s for s in subs if s.decision not in EXCLUDED_DECISIONS]
        for s in non_excluded:
            submissions_by_id[s.id] = s

        if verbose:
            print(f"  {year}: {len(non_excluded)} submissions loaded")

    if verbose:
        print(f"Total submissions: {len(submissions_by_id)}")

    # If single submission requested, find it
    if submission_id:
        if submission_id not in submissions_by_id:
            print(f"Submission {submission_id} not found in pickle")
            return []

        sub = submissions_by_id[submission_id]
        result = find_mineru_by_prefix(
            submission_id, mineru_keys, mineru_index,
            year=sub.year,
            fixed_keys=fixed_keys, fixed_index=fixed_index,
            fixed_pt2_keys=fixed_pt2_keys, fixed_pt2_index=fixed_pt2_index,
        )
        if not result:
            print(f"MinerU output not found for {submission_id}")
            return []

        folder_name, content_list_path = result
        pdf_path = find_pdf_path(submission_id, sub.year, pdf_index, pdf_dir)

        if verbose:
            print(f"\nProcessing single submission: {submission_id}")
            print(f"  Year: {sub.year}")
            print(f"  Content list: {content_list_path}")
            print(f"  PDF: {pdf_path}")

        res = normalize_paper(
            submission_id=submission_id,
            year=sub.year,
            content_list_path=content_list_path,
            pdf_path=pdf_path,
            abstract_from_pickle=sub.abstract,
            output_dir=output_dir,
        )

        if res['success']:
            print(f"  Success! Output: {output_dir / str(sub.year) / submission_id}")
        else:
            print(f"  Failed: {res['error']}")

        return [res]

    # Process all submissions
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Iterate over submissions
    subs_to_process = list(submissions_by_id.values())
    if limit:
        subs_to_process = subs_to_process[:limit]

    # Split by task_id for job array (contiguous ranges)
    if task_id is not None and num_tasks is not None:
        total = len(subs_to_process)
        chunk_size = total // num_tasks
        remainder = total % num_tasks
        # First 'remainder' tasks get chunk_size+1, rest get chunk_size
        if task_id < remainder:
            start = task_id * (chunk_size + 1)
            end = start + chunk_size + 1
        else:
            start = remainder * (chunk_size + 1) + (task_id - remainder) * chunk_size
            end = start + chunk_size
        subs_to_process = subs_to_process[start:end]
        if verbose:
            print(f"Task {task_id}/{num_tasks}: processing {len(subs_to_process)} submissions (indices {start}-{end-1})")

    # Worker function for parallel processing
    def process_one(sub):
        """Process a single submission. Returns (result, content_list_path) or None."""
        result = find_mineru_by_prefix(
            sub.id, mineru_keys, mineru_index,
            year=sub.year,
            fixed_keys=fixed_keys, fixed_index=fixed_index,
            fixed_pt2_keys=fixed_pt2_keys, fixed_pt2_index=fixed_pt2_index,
        )
        if not result:
            return None

        folder_name, content_list_path = result
        pdf_path = find_pdf_path(sub.id, sub.year, pdf_index, pdf_dir)

        res = normalize_paper(
            submission_id=sub.id,
            year=sub.year,
            content_list_path=content_list_path,
            pdf_path=pdf_path,
            abstract_from_pickle=sub.abstract,
            output_dir=output_dir,
        )
        return (res, content_list_path)

    if workers > 1:
        # Parallel processing with ThreadPoolExecutor
        if verbose:
            print(f"Processing with {workers} workers...")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one, sub): sub for sub in subs_to_process}
            iterator = tqdm(as_completed(futures), total=len(futures), desc="Normalizing") if verbose else as_completed(futures)

            for future in iterator:
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    sub = futures[future]
                    if verbose:
                        print(f"Error processing {sub.id}: {e}")
    else:
        # Sequential processing
        iterator = tqdm(subs_to_process, desc="Normalizing") if verbose else subs_to_process

        for sub in iterator:
            result = process_one(sub)
            if result:
                results.append(result)

    # Print warnings to stderr (captured by sbatch logs)
    for res, content_list_path in results:
        if res.get('error'):
            print(f"WARNING: {res['year']}/{res['submission_id']}: {res['error']}", file=sys.stderr)
        elif not res.get('has_abstract') or not res.get('has_references'):
            reasons = []
            if not res.get('has_abstract'):
                reasons.append('no_abstract')
            if not res.get('has_references'):
                reasons.append('no_references')
            print(f"WARNING: {res['year']}/{res['submission_id']}: {','.join(reasons)}", file=sys.stderr)

    # Print summary
    if verbose:
        # results is list of (res, content_list_path) tuples
        successful = sum(1 for res, _ in results if res.get('success'))
        with_abstract = sum(1 for res, _ in results if res.get('has_abstract'))
        with_refs = sum(1 for res, _ in results if res.get('has_references'))
        errors = [(res, path) for res, path in results if res.get('error')]

        print(f"\nNormalization complete:")
        print(f"  Processed: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  With abstract: {with_abstract}")
        print(f"  With references: {with_refs}")

        if errors:
            print(f"\nSample errors ({len(errors)} total):")
            for res, _ in errors[:5]:
                print(f"  {res['submission_id']}: {res['error']}")

    return [res for res, _ in results]


def main():
    parser = argparse.ArgumentParser(description="Normalize MinerU content_list.json files")
    parser.add_argument("--data-dir", type=Path, default=Path("data/full_run"),
                        help="Base data directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: data_dir/normalized)")
    parser.add_argument("--year", type=int, default=None,
                        help="Process single year only")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of papers to process")
    parser.add_argument("--submission", type=str, default=None,
                        help="Process single submission ID")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Task ID for job array (0-indexed)")
    parser.add_argument("--num-tasks", type=int, default=None,
                        help="Total number of tasks in job array")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    years = [args.year] if args.year else None

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        years=years,
        limit=args.limit,
        submission_id=args.submission,
        verbose=not args.quiet,
        workers=args.workers,
        task_id=args.task_id,
        num_tasks=args.num_tasks,
    )


if __name__ == "__main__":
    main()
