"""
MD file parser for extracting sections from marker-converted markdown files.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Section:
    level: int          # Number of # (1-6)
    number: str         # "1", "1.2", "1.2.3" or None
    title: str          # Section title (cleaned)
    start: int          # Start position in text
    end: int            # End position (start of next section or EOF)
    content: str        # Text content of section


# =============================================================================
# REGEX PATTERNS
# =============================================================================

# ABSTRACT_PATTERN: Matches abstract section headers
# - Optional # prefix (1-6): Some marker outputs have "ABSTRACT" without #
# - Optional <span> tags: Marker sometimes wraps in spans
# - Optional ** bold markers
# - Spaced letters "A B S T R A C T": OCR artifacts from some PDFs
# Examples matched:
#   "# Abstract", "## **ABSTRACT**", "ABSTRACT", "# <span>Abstract</span>"
#   "# A B S T R A C T" (OCR spacing issues)
ABSTRACT_PATTERN = re.compile(
    r'^(#{1,6})?\s*(?:<[^>]+>\s*)*\**\s*a\s*b\s*s\s*t\s*r\s*a\s*c\s*t\s*\**\s*(?:<[^>]+>\s*)*$',
    re.IGNORECASE | re.MULTILINE
)

# HEADER_PATTERN: Matches any markdown header (requires # prefix)
# - Group 1: The # symbols (1-6)
# - Group 2: The header text
# Examples: "# Introduction", "## 1.1 Methods", "### Results"
HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

# NUMBERED_SECTION_PATTERN: Matches numbered section headers
# - Group 1: # symbols, Group 2: section number, Group 3: title
# Examples: "# 1 Introduction", "## 3.2 Experiments"
NUMBERED_SECTION_PATTERN = re.compile(r'^(#{1,6})\s+(\d+(?:\.\d+)*)\s*(.*)$', re.MULTILINE)

# SPAN_CLEANUP: Removes <span>...</span> tags from text
# Used to clean headers that have formatting spans
SPAN_CLEANUP = re.compile(r'<span[^>]*>.*?</span>', re.IGNORECASE)

# BOLD_CLEANUP: Removes markdown bold/italic markers (* or **)
BOLD_CLEANUP = re.compile(r'\*+')

# BOLD_LINE_NUMBER_PATTERN: Matches bold line number remnants anywhere in text
# These are PDF line numbers that appear as **digits and spaces**
# Examples: "**054 055 056**", "**750 751**", "**808 809**"
# Removed globally (not just standalone lines) to catch inline occurrences
BOLD_LINE_NUMBER_PATTERN = re.compile(r'\*\*[\d\s]+\*\*')

# SPAN_SUP_LINE_PATTERN: Matches ENTIRE LINE containing <span...>...<sup>
# These are page anchors with footnote refs - delete the whole line
# Examples: "some text <span id="page-5-1"></span><sup>1</sup> https://github.com/..."
# Efficiency: O(n) - no nested quantifiers, .* stops at newlines in MULTILINE mode
SPAN_SUP_LINE_PATTERN = re.compile(r'^.*<span[^>]*>.*<sup>.*$', re.MULTILINE | re.IGNORECASE)

# SPAN_PAGE_PATTERN: Matches standalone <span id="page-..."></span> tags
# Examples: "<span id="page-8-4"></span>", "<span id="page-1-0"></span>"
SPAN_PAGE_PATTERN = re.compile(r'<span id="page-[^"]*"></span>', re.IGNORECASE)

# STANDALONE_NUMBER_LINE_PATTERN: Matches lines that are just numbers (PDF line numbers)
# Examples: "327", "337 338", "332333334", "054 055 056"
# Matches: line with only digits and whitespace (at least one digit)
# Efficiency: O(n) - single character class, no nested quantifiers
STANDALONE_NUMBER_LINE_PATTERN = re.compile(r'^\s*\d[\d\s]*\s*$', re.MULTILINE)

# REPRODUCIBILITY_PATTERN: Matches reproducibility section headers
# Examples: "# Reproducibility", "## Reproducibility Statement", "# REPRODUCIBILITY"
REPRODUCIBILITY_PATTERN = re.compile(
    r'^(#{1,6})\s*(?:<[^>]+>\s*)*\**\s*reproducibility',
    re.IGNORECASE | re.MULTILINE
)

# FOOTNOTE_PATTERN: Matches footnote definition lines
# These are lines starting with <sup>X</sup> where X is NOT "(" (to exclude figure labels)
# Note: In clean_md_text, we also preserve footnotes containing "fig" (figure references)
# Examples matched:
#   "<sup>1</sup> https://example.com"
#   "<sup>*</sup> Additional details..."
#   "<sup>†</sup>Corresponding author"
# Examples NOT matched (figure labels):
#   "<sup>(</sup>a) Results for..."
# Examples preserved (contain "fig"):
#   "<sup>1</sup> See Figure 3 for details"
FOOTNOTE_PATTERN = re.compile(r'^\s*<sup>[^(].*</sup>\s*.+$', re.MULTILINE)

# DAGGER_FOOTNOTE_PATTERN: Matches dagger/double-dagger footnote lines
# Note: In clean_md_text, we also preserve footnotes containing "fig" (figure references)
# Examples: "† denotes unreleased models", "‡ Equal contribution"
DAGGER_FOOTNOTE_PATTERN = re.compile(r'^\s*[†‡]\s+.+$', re.MULTILINE)


def clean_header(text: str) -> str:
    """Clean a header string by removing span tags, bold markers, etc."""
    text = SPAN_CLEANUP.sub('', text)
    text = BOLD_CLEANUP.sub('', text)
    return text.strip()


def find_abstract(md_text: str) -> Optional[tuple[str, int, int]]:
    """
    Find abstract section in markdown text.

    Returns:
        (abstract_text, start_pos, end_pos) or None if not found.
    """
    # Strategy 1: Find explicit abstract header
    match = ABSTRACT_PATTERN.search(md_text)

    if match:
        start = match.end()
        # Find next header to determine end of abstract
        next_header = HEADER_PATTERN.search(md_text, start)
        if next_header:
            end = next_header.start()
        else:
            end = len(md_text)

        abstract_text = md_text[start:end].strip()
        return (abstract_text, match.start(), end)

    # Strategy 2: Look for text between author section and first numbered section (1 Introduction)
    # Find first numbered section like "# 1 " or "## 1 " (section 1)
    section_1_pattern = re.compile(r'^(#{1,6})\s+1\s+', re.MULTILINE)
    section_1_match = section_1_pattern.search(md_text)

    if section_1_match:
        # Look for substantial text before section 1
        # Skip the title (first # header) and find content before section 1
        first_header = HEADER_PATTERN.search(md_text)
        if first_header:
            # Find all headers before section 1
            headers_before = list(HEADER_PATTERN.finditer(md_text, 0, section_1_match.start()))

            if len(headers_before) >= 1:
                # Get text after last author/title header but before section 1
                last_header_before = headers_before[-1]
                potential_abstract_start = last_header_before.end()
                potential_abstract_end = section_1_match.start()

                # Check if there's substantial content (> 200 chars)
                potential_text = md_text[potential_abstract_start:potential_abstract_end].strip()
                if len(potential_text) > 200:
                    return (potential_text, potential_abstract_start, potential_abstract_end)

    return None


def extract_sections(md_text: str) -> list[Section]:
    """
    Extract all sections from markdown text.

    Returns:
        List of Section objects with their content.
    """
    sections = []

    # Find all headers
    headers = list(HEADER_PATTERN.finditer(md_text))

    for i, match in enumerate(headers):
        level = len(match.group(1))  # Number of #
        raw_title = match.group(2)

        # Check if it's a numbered section
        num_match = re.match(r'^(\d+(?:\.\d+)*)\s*(.*)', raw_title)
        if num_match:
            number = num_match.group(1)
            title = clean_header(num_match.group(2))
        else:
            number = None
            title = clean_header(raw_title)

        start = match.end()

        # End is start of next header or EOF
        if i + 1 < len(headers):
            end = headers[i + 1].start()
        else:
            end = len(md_text)

        content = md_text[start:end].strip()

        sections.append(Section(
            level=level,
            number=number,
            title=title,
            start=match.start(),
            end=end,
            content=content
        ))

    return sections


def parse_md_file(md_path: Path) -> dict:
    """
    Parse a single MD file and return structured data.

    Returns:
        {
            'path': Path,
            'abstract': str or None,
            'abstract_start': int or None,
            'abstract_end': int or None,
            'sections': list[Section],
            'has_abstract': bool,
            'has_intro': bool,
            'error': str or None,
        }
    """
    result = {
        'path': md_path,
        'abstract': None,
        'abstract_start': None,
        'abstract_end': None,
        'sections': [],
        'has_abstract': False,
        'has_intro': False,
        'error': None,
    }

    try:
        md_text = md_path.read_text(encoding='utf-8')
    except Exception as e:
        result['error'] = str(e)
        return result

    # Find abstract
    abstract_result = find_abstract(md_text)
    if abstract_result:
        result['abstract'], result['abstract_start'], result['abstract_end'] = abstract_result
        result['has_abstract'] = True

    # Extract sections
    sections = extract_sections(md_text)
    result['sections'] = sections

    # Check for introduction
    for sec in sections:
        title_lower = sec.title.lower()
        if 'intro' in title_lower or sec.number == '1':
            result['has_intro'] = True
            break

    return result


def validate_md_files(
    md_dir: str | Path,
    year: int,
    verbose: bool = True
) -> dict:
    """
    Validate all MD files for a year, print those missing abstract.

    Args:
        md_dir: Base directory containing mds/{year}/{id}/
        year: Year to validate
        verbose: Print detailed output

    Returns:
        {
            'total': int,
            'with_abstract': int,
            'with_intro': int,
            'missing_abstract': list[str],
            'errors': list[tuple[str, str]],
        }
    """
    md_dir = Path(md_dir)
    year_dir = md_dir / str(year)

    if not year_dir.exists():
        print(f"Directory not found: {year_dir}")
        return {'total': 0, 'with_abstract': 0, 'with_intro': 0, 'missing_abstract': [], 'errors': []}

    stats = {
        'total': 0,
        'with_abstract': 0,
        'with_intro': 0,
        'missing_abstract': [],
        'errors': [],
    }

    # Iterate through subdirectories
    for sub_dir in sorted(year_dir.iterdir()):
        if not sub_dir.is_dir():
            continue

        # Find the .md file
        md_file = sub_dir / f"{sub_dir.name}.md"
        if not md_file.exists():
            continue

        stats['total'] += 1

        result = parse_md_file(md_file)

        if result['error']:
            stats['errors'].append((sub_dir.name, result['error']))
            continue

        if result['has_abstract']:
            stats['with_abstract'] += 1
        else:
            stats['missing_abstract'].append(sub_dir.name)

        if result['has_intro']:
            stats['with_intro'] += 1

    # Print summary
    if verbose:
        pct_abstract = 100 * stats['with_abstract'] / stats['total'] if stats['total'] > 0 else 0
        pct_intro = 100 * stats['with_intro'] / stats['total'] if stats['total'] > 0 else 0

        print(f"\nYear {year}: {stats['total']} files")
        print(f"  With abstract: {stats['with_abstract']} ({pct_abstract:.1f}%)")
        print(f"  With intro:    {stats['with_intro']} ({pct_intro:.1f}%)")

        if stats['missing_abstract']:
            print(f"\n  Missing abstract ({len(stats['missing_abstract'])}):")
            for sub_id in stats['missing_abstract'][:20]:  # Show first 20
                print(f"    - {sub_id}")
            if len(stats['missing_abstract']) > 20:
                print(f"    ... and {len(stats['missing_abstract']) - 20} more")

        if stats['errors']:
            print(f"\n  Errors ({len(stats['errors'])}):")
            for sub_id, err in stats['errors'][:10]:
                print(f"    - {sub_id}: {err}")

    return stats


def validate_all_years(
    md_dir: str | Path = "data/full_run/mds",
    years: list[int] = None
) -> dict:
    """Validate MD files across all years."""
    md_dir = Path(md_dir)
    years = years or [2020, 2021, 2022, 2023, 2024, 2025, 2026]

    all_stats = {}
    for year in years:
        print(f"\n{'='*60}")
        stats = validate_md_files(md_dir, year)
        all_stats[year] = stats

    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")

    total_files = sum(s['total'] for s in all_stats.values())
    total_abstract = sum(s['with_abstract'] for s in all_stats.values())
    total_missing = sum(len(s['missing_abstract']) for s in all_stats.values())

    print(f"Total files:      {total_files}")
    print(f"With abstract:    {total_abstract} ({100*total_abstract/total_files:.1f}%)" if total_files > 0 else "")
    print(f"Missing abstract: {total_missing}")

    return all_stats


def get_preamble(md_text: str) -> str:
    """
    Get all content up until the section AFTER abstract.
    This is the title, authors, abstract - before the main body (e.g., Introduction).
    """
    # Find abstract first
    abstract_result = find_abstract(md_text)

    if abstract_result:
        _, _, abstract_end = abstract_result
        # Find next header after abstract end
        next_header = HEADER_PATTERN.search(md_text, abstract_end)
        if next_header:
            return md_text[:next_header.start()]
        return md_text[:abstract_end]

    # Fallback: find section 1 and return everything before it
    section_1_pattern = re.compile(r'^(#{1,6})\s+1\s+', re.MULTILINE)
    section_1_match = section_1_pattern.search(md_text)
    if section_1_match:
        return md_text[:section_1_match.start()]

    # No structure found, return first 2000 chars
    return md_text[:2000]


def find_acknowledgements_in_preamble(
    md_dir: str | Path,
    years: list[int] = None,
    verbose: bool = True
) -> list[dict]:
    """
    Find 'Acknowledgement' SECTION HEADERS in the preamble (before main body).
    This shouldn't exist in double-blind submissions.

    Returns list of dicts with file info and matched text.
    """
    md_dir = Path(md_dir)
    years = years or [2020, 2021, 2022, 2023, 2024, 2025, 2026]

    # Pattern for acknowledgement section headers
    # Matches: # Acknowledgement, ## Acknowledgments, ### ACKNOWLEDGEMENT, etc.
    ack_header_pattern = re.compile(
        r'^(#{1,6})\s*(?:<[^>]+>\s*)*\**\s*a\s*c\s*k\s*n\s*o\s*w\s*l\s*e\s*d\s*g',
        re.IGNORECASE | re.MULTILINE
    )

    results = []
    total_checked = 0

    for year in years:
        year_dir = md_dir / str(year)
        if not year_dir.exists():
            continue

        year_count = 0

        for sub_dir in sorted(year_dir.iterdir()):
            if not sub_dir.is_dir():
                continue

            md_file = sub_dir / f"{sub_dir.name}.md"
            if not md_file.exists():
                continue

            total_checked += 1

            try:
                md_text = md_file.read_text(encoding='utf-8')
            except:
                continue

            preamble = get_preamble(md_text)

            # Search for acknowledgement headers
            matches = list(ack_header_pattern.finditer(preamble))
            if matches:
                year_count += 1
                # Get the full line for each match
                contexts = []
                for m in matches:
                    # Find end of line
                    line_end = preamble.find('\n', m.start())
                    if line_end == -1:
                        line_end = len(preamble)
                    line = preamble[m.start():line_end].strip()
                    contexts.append(line)

                results.append({
                    'year': year,
                    'sub_id': sub_dir.name,
                    'path': str(md_file),
                    'matches': len(matches),
                    'contexts': contexts
                })

        if verbose and year_count > 0:
            print(f"Year {year}: {year_count} files with 'Acknowledgement' header in preamble")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Total checked: {total_checked}")
        print(f"With Acknowledgement headers in preamble: {len(results)}")
        print(f"{'='*60}\n")

        for r in results:
            print(f"\n{r['year']}/{r['sub_id']}:")
            for ctx in r['contexts']:
                print(f"  {ctx}")

    return results


# ACKNOWLEDGEMENT_PATTERN: Matches acknowledgement section headers
# - Requires # prefix (1-6)
# - Optional <span> tags and ** bold markers
# - Handles spelling variations: acknowledgement/acknowledgment/acknowledgements
# - Spaced letters for OCR artifacts
# Examples: "# Acknowledgements", "## ACKNOWLEDGMENT", "### **Acknowledgements**"
ACKNOWLEDGEMENT_PATTERN = re.compile(
    r'^(#{1,6})\s*(?:<[^>]+>\s*)*\**\s*a\s*c\s*k\s*n\s*o\s*w\s*l\s*e\s*d\s*g[ements]*\s*\**\s*(?:<[^>]+>\s*)*$',
    re.IGNORECASE | re.MULTILINE
)

# ACKNOWLEDGEMENT_BROAD_PATTERN: Broader match for acknowledgement headers
# Used for removal - matches any header starting with "acknowledg..."
# Less strict than ACKNOWLEDGEMENT_PATTERN to catch edge cases
ACKNOWLEDGEMENT_BROAD_PATTERN = re.compile(
    r'^(#{1,6})\s*(?:<[^>]+>\s*)*\**\s*a\s*c\s*k\s*n\s*o\s*w\s*l\s*e\s*d\s*g',
    re.IGNORECASE | re.MULTILINE
)

# REFERENCES_PATTERN: Matches references section headers
# - Optional # prefix: Some marker outputs have "REFERENCES" without #
# - Optional <span> tags and ** bold markers
# - Matches both "Reference" and "References"
# - Spaced letters for OCR artifacts
# Examples: "# References", "## REFERENCES", "REFERENCES", "# R E F E R E N C E S"
REFERENCES_PATTERN = re.compile(
    r'^(#{1,6})?\s*(?:<[^>]+>\s*)*\**\s*r\s*e\s*f\s*e\s*r\s*e\s*n\s*c\s*e\s*s?\s*\**\s*(?:<[^>]+>\s*)*$',
    re.IGNORECASE | re.MULTILINE
)


def remove_sentences_containing(text: str, pattern: re.Pattern, collapse_whitespace: bool = True) -> str:
    """
    Remove sentences containing matches of the given pattern.

    Finds pattern matches and removes from previous period to next period.

    Args:
        text: Input text
        pattern: Compiled regex pattern to search for
        collapse_whitespace: If True, collapse whitespace after removal (for abstracts)

    Returns:
        Text with matching sentences removed
    """
    result = text

    # Keep removing matches until none left
    while True:
        match = pattern.search(result)
        if not match:
            break

        match_start = match.start()
        match_end = match.end()

        # Find previous period (or start)
        prev_period = result.rfind('.', 0, match_start)
        if prev_period == -1:
            sentence_start = 0
        else:
            sentence_start = prev_period + 1

        # Find next period (or end)
        next_period = result.find('.', match_end)
        if next_period == -1:
            sentence_end = len(result)
        else:
            sentence_end = next_period + 1

        # Remove the sentence
        result = result[:sentence_start] + result[sentence_end:]

    # Clean up extra whitespace (optional)
    if collapse_whitespace:
        result = re.sub(r'\s+', ' ', result).strip()

    return result


# Default URL pattern for abstracts
_URL_PATTERN = re.compile(r'https?://[^\s\)\]\"\'<>]+[^\s\)\]\"\'<>\.,;:!?]', re.IGNORECASE)

# Pattern for sentences containing code+github (removes full sentence from period to period)
CODE_GITHUB_SENTENCE_PATTERN = re.compile(
    r'(?<=\.)\s*[^.]*code[^.]*https?://github[^\s\)]*[^.]*\.?'
    r'|^[^.]*code[^.]*https?://github[^\s\)]*[^.]*\.?',
    re.IGNORECASE
)


def remove_urls_from_abstract(abstract_text: str) -> str:
    """
    Remove sentences containing URLs from abstract text.

    Finds any https://... and removes from previous period to next period.
    """
    result = remove_sentences_containing(abstract_text, _URL_PATTERN, collapse_whitespace=False)

    # Also check for simple http(s):// that might be left (malformed URLs)
    simple_pattern = re.compile(r'https?://', re.IGNORECASE)
    result = remove_sentences_containing(result, simple_pattern, collapse_whitespace=False)

    # Clean up extra whitespace
    result = re.sub(r'\s+', ' ', result).strip()

    return result


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text:
    - Strip leading/trailing whitespace from each line
    - Collapse multiple blank lines into single blank line
    - Remove trailing whitespace
    """
    lines = text.split('\n')
    # Strip each line
    lines = [line.rstrip() for line in lines]

    # Collapse multiple blank lines
    result = []
    prev_blank = False
    for line in lines:
        is_blank = line.strip() == ''
        if is_blank:
            if not prev_blank:
                result.append('')
            prev_blank = True
        else:
            result.append(line)
            prev_blank = False

    return '\n'.join(result).strip()


def _add_header_prefix(match: re.Match) -> str:
    """Add # prefix to headerless ABSTRACT/REFERENCES lines."""
    if match.group(1):  # Already has # prefix
        return match.group(0)
    return f"# {match.group(0).strip()}"


def clean_md_text(md_text: str) -> Optional[str]:
    """
    Clean markdown text for training data.

    Pipeline:
    1. Pre-process: Ensure ABSTRACT/REFERENCES have # prefix for consistent parsing
    2. Validate: Must have both Abstract and References
    3. Clip: From first section AFTER abstract → end of References
    4. Remove: Acknowledgements section (if present)
    5. Clean: Remove line number remnants and footnotes
    6. Normalize: Headers → # SECTION_TITLE, strip extra whitespace

    Returns:
        Cleaned text (Introduction → ... → References) or None if missing abstract/refs
    """
    # Step 1: Pre-process - add # to headerless ABSTRACT and REFERENCES
    md_text = ABSTRACT_PATTERN.sub(_add_header_prefix, md_text)
    md_text = REFERENCES_PATTERN.sub(_add_header_prefix, md_text)

    # Step 2: Validate - must have both abstract and references
    abstract_result = find_abstract(md_text)
    refs_match = REFERENCES_PATTERN.search(md_text)
    if not abstract_result or not refs_match:
        return None

    # Step 3: Clip content
    # Start: first header AFTER abstract (e.g., Introduction)
    _, _, abstract_end = abstract_result
    body_start = HEADER_PATTERN.search(md_text, abstract_end)
    if not body_start:
        return None

    # End: next header after References (Appendix, etc.) or EOF
    after_refs = md_text[refs_match.end():]
    next_after_refs = HEADER_PATTERN.search(after_refs)
    if next_after_refs:
        refs_end = refs_match.end() + next_after_refs.start()
    else:
        refs_end = len(md_text)

    md_text = md_text[body_start.start():refs_end].rstrip()

    # Step 4: Remove acknowledgements section (if present)
    ack_match = ACKNOWLEDGEMENT_BROAD_PATTERN.search(md_text)
    if ack_match:
        ack_level = len(ack_match.group(1))
        ack_start = ack_match.start()

        # Find end: next header of same or higher level
        ack_end = len(md_text)
        for m in HEADER_PATTERN.finditer(md_text[ack_match.end():]):
            if len(m.group(1)) <= ack_level:
                ack_end = ack_match.end() + m.start()
                break

        md_text = md_text[:ack_start] + md_text[ack_end:]

    # Step 4b: Remove reproducibility section (if present)
    repro_match = REPRODUCIBILITY_PATTERN.search(md_text)
    if repro_match:
        repro_level = len(repro_match.group(1))
        repro_start = repro_match.start()

        # Find end: next header of same or higher level
        repro_end = len(md_text)
        for m in HEADER_PATTERN.finditer(md_text[repro_match.end():]):
            if len(m.group(1)) <= repro_level:
                repro_end = repro_match.end() + m.start()
                break

        md_text = md_text[:repro_start] + md_text[repro_end:]

    # Step 5: Clean artifacts - remove line numbers, footnotes, and author code refs
    md_text = BOLD_LINE_NUMBER_PATTERN.sub('', md_text)          # "**054 055**" remnants
    md_text = SPAN_SUP_LINE_PATTERN.sub('', md_text)             # Delete lines with <span...><sup>
    md_text = SPAN_PAGE_PATTERN.sub('', md_text)                 # Remove <span id="page-..."></span>
    md_text = STANDALONE_NUMBER_LINE_PATTERN.sub('', md_text)    # Bare number lines like "327"
    md_text = CODE_GITHUB_SENTENCE_PATTERN.sub('', md_text)        # Remove sentences with "code...github"
    # Remove footnotes only if they don't mention figures
    md_text = FOOTNOTE_PATTERN.sub(
        lambda m: m.group(0) if 'fig' in m.group(0).lower() else '', md_text
    )
    md_text = DAGGER_FOOTNOTE_PATTERN.sub(
        lambda m: m.group(0) if 'fig' in m.group(0).lower() else '', md_text
    )
    # Collapse excessive newlines (more than 2 in a row → 2)
    md_text = re.sub(r'\n{3,}', '\n\n', md_text)

    # Step 6: Normalize headers and whitespace
    headers = list(HEADER_PATTERN.finditer(md_text))
    if not headers:
        return normalize_whitespace(md_text)

    chunks = []
    for i, match in enumerate(headers):
        # Normalize header: # SECTION_TITLE (single #, uppercase, no spans/bold)
        clean_title = clean_header(match.group(2)).upper()
        normalized_header = f"# {clean_title}"

        # Get content until next header
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(md_text)
        content = normalize_whitespace(md_text[start:end])

        # Combine header + content
        chunks.append(f"{normalized_header}\n\n{content}" if content else normalized_header)

    return '\n\n'.join(chunks)


def remove_acknowledgements(md_text: str) -> tuple[str, bool]:
    """
    Remove acknowledgement section(s) from markdown text.

    Returns:
        (cleaned_text, was_removed) - the text with ack section removed, and whether one was found
    """
    match = ACKNOWLEDGEMENT_PATTERN.search(md_text)
    if not match:
        return md_text, False

    # Find the end of the acknowledgement section (next header of same or higher level, or EOF)
    ack_level = len(match.group(1))  # Number of #
    start = match.start()

    # Look for next header of same or higher level (fewer or equal #)
    remaining = md_text[match.end():]
    next_header = None
    for m in HEADER_PATTERN.finditer(remaining):
        header_level = len(m.group(1))
        if header_level <= ack_level:
            next_header = m
            break

    if next_header:
        end = match.end() + next_header.start()
    else:
        # No next header found - ack goes to end of document
        end = len(md_text)

    # Remove the section
    cleaned = md_text[:start] + md_text[end:]

    # Recursively remove any additional acknowledgement sections
    cleaned, _ = remove_acknowledgements(cleaned)

    return cleaned, True


def find_references(md_text: str) -> Optional[tuple[int, int]]:
    """
    Find references section in markdown text.

    Returns:
        (start_pos, end_pos) or None if not found
    """
    match = REFERENCES_PATTERN.search(md_text)
    if not match:
        return None

    # Find end (next header of same or higher level, or EOF)
    ref_level = len(match.group(1))
    remaining = md_text[match.end():]

    next_header = None
    for m in HEADER_PATTERN.finditer(remaining):
        header_level = len(m.group(1))
        if header_level <= ref_level:
            next_header = m
            break

    if next_header:
        end = match.end() + next_header.start()
    else:
        end = len(md_text)

    return (match.start(), end)


def has_references(md_text: str) -> bool:
    """Check if markdown text has a references section."""
    return REFERENCES_PATTERN.search(md_text) is not None


def validate_references(
    md_dir: str | Path,
    years: list[int] = None,
    verbose: bool = True
) -> dict:
    """
    Count papers with References section by year.

    Returns:
        {year: {'total': int, 'with_refs': int, 'missing_refs': list[str]}}
    """
    md_dir = Path(md_dir)
    years = years or [2020, 2021, 2022, 2023, 2024, 2025, 2026]

    results = {}

    for year in years:
        year_dir = md_dir / str(year)
        if not year_dir.exists():
            continue

        stats = {'total': 0, 'with_refs': 0, 'missing_refs': []}

        for sub_dir in sorted(year_dir.iterdir()):
            if not sub_dir.is_dir():
                continue

            md_file = sub_dir / f"{sub_dir.name}.md"
            if not md_file.exists():
                continue

            stats['total'] += 1

            try:
                md_text = md_file.read_text(encoding='utf-8')
            except:
                continue

            if has_references(md_text):
                stats['with_refs'] += 1
            else:
                stats['missing_refs'].append(sub_dir.name)

        results[year] = stats

        if verbose and stats['total'] > 0:
            pct = 100 * stats['with_refs'] / stats['total']
            print(f"Year {year}: {stats['with_refs']}/{stats['total']} ({pct:.1f}%) have References section")

    if verbose:
        print("\n" + "=" * 60)
        total_files = sum(s['total'] for s in results.values())
        total_refs = sum(s['with_refs'] for s in results.values())
        pct = 100 * total_refs / total_files if total_files > 0 else 0
        print(f"Overall: {total_refs}/{total_files} ({pct:.1f}%) have References section")
        print("=" * 60)

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'ack':
            find_acknowledgements_in_preamble('data/full_run/mds')
        elif sys.argv[1] == 'refs':
            validate_references('data/full_run/mds')
    else:
        validate_all_years()
