"""Citation lookup from Google Scholar data."""

import re
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# Global lookup dicts (computed once, reused across all years)
_id_lookup: Optional[Dict[str, int]] = None
_title_lookup: Optional[Dict[str, int]] = None


def normalize_title(title: str) -> str:
    """Normalize title: lowercase, remove non-alphanumeric."""
    return re.sub(r'[^a-z0-9]', '', str(title).lower())


def load_citations(db_path: Path, conference: str = 'iclr') -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Load citations from DB and build lookup dicts (computed once).

    Args:
        db_path: Path to citations.db SQLite database
        conference: Conference to filter by (default: 'iclr')

    Returns:
        (id_lookup, title_lookup) - dicts mapping id/normalized_title to gs_citation
    """
    global _id_lookup, _title_lookup

    if _id_lookup is not None and _title_lookup is not None:
        return _id_lookup, _title_lookup

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT id, normalized_title, gs_citation FROM papers WHERE conference = ?",
        conn, params=(conference,)
    )
    conn.close()

    # Build lookup dicts once (expensive operations)
    _id_lookup = df.set_index('id')['gs_citation'].to_dict()
    _title_lookup = df.set_index('normalized_title')['gs_citation'].to_dict()

    print(f"Loaded {len(df):,} citations into memory ({len(_id_lookup):,} by ID, {len(_title_lookup):,} by title)")

    return _id_lookup, _title_lookup


def add_citations_to_df(
    df: pd.DataFrame,
    id_lookup: Dict[str, int],
    title_lookup: Dict[str, int]
) -> pd.DataFrame:
    """
    Add google_scholar_citations to each submission dict using vectorized lookup.

    The df has a 'submission' column containing dicts with 'id' and 'title'.

    Strategy:
    1. Extract id/title from submission dicts to flat columns
    2. Match on 'id' first (most reliable)
    3. Fall back to normalized_title for non-matches
    4. If no match or gs_citation < 0: None
    5. Add citation back into submission dict

    Args:
        df: DataFrame with 'submission' column containing dicts
        id_lookup: Pre-computed dict mapping paper ID to citation count
        title_lookup: Pre-computed dict mapping normalized title to citation count

    Returns:
        DataFrame with google_scholar_citations added to each submission dict
    """
    # Extract id and title from nested submission dict
    df['_sub_id'] = df['submission'].apply(lambda x: x.get('id'))
    df['_sub_title'] = df['submission'].apply(lambda x: x.get('title', ''))
    df['_normalized_title'] = df['_sub_title'].apply(normalize_title)

    # Step 1: Match on ID (using pre-computed lookup)
    df['_citation_by_id'] = df['_sub_id'].map(id_lookup)

    # Step 2: Match on normalized_title for rows without ID match
    df['_citation_by_title'] = df['_normalized_title'].map(title_lookup)

    # Combine: prefer ID match, fall back to title match
    df['_citation'] = df['_citation_by_id'].fillna(df['_citation_by_title'])

    # Clean up: None if no match or < 0
    df.loc[df['_citation'] < 0, '_citation'] = None

    # Add citation into submission dict
    def add_citation(row):
        sub = row['submission'].copy()
        citation = row['_citation']
        sub['google_scholar_citations'] = int(citation) if pd.notna(citation) else None
        return sub

    df['submission'] = df.apply(add_citation, axis=1)

    # Drop helper columns
    df = df.drop(columns=['_sub_id', '_sub_title', '_normalized_title',
                          '_citation_by_id', '_citation_by_title', '_citation'])

    return df
