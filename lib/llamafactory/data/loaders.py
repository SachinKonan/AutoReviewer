"""
Data loaders for ICLR paper data from CSV and parquet sources.

This module provides utilities to load submission data from various formats
and convert them to SubmissionData objects for use with the training pipeline.
"""

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any

import pandas as pd

from ..core.types import ReviewData, SubmissionData


class ICLRDataLoader:
    """
    Load ICLR paper data from CSV or parquet sources.

    Supports loading from:
    - CSV files with the standard schema
    - Parquet files organized by year
    - Pre-normalized reviews from parquet

    Args:
        data_dir: Base directory for data files
        normalized_reviews_path: Optional path to normalized reviews parquet
        image_base_dir: Optional base directory for PDF page images
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        normalized_reviews_path: Optional[Path] = None,
        image_base_dir: Optional[Path] = None,
    ):
        self.data_dir = Path(data_dir) if data_dir else None
        self.normalized_reviews_path = Path(normalized_reviews_path) if normalized_reviews_path else None
        self.image_base_dir = Path(image_base_dir) if image_base_dir else None

        # Cache for normalized reviews
        self._normalized_reviews: Dict[str, List[ReviewData]] = {}
        self._loaded_normalized = False

    def _load_normalized_reviews(self) -> None:
        """Load pre-normalized reviews from parquet."""
        if self._loaded_normalized:
            return
        if not self.normalized_reviews_path or not self.normalized_reviews_path.exists():
            self._loaded_normalized = True
            return

        df = pd.read_parquet(self.normalized_reviews_path)
        for _, row in df.iterrows():
            sub_id = str(row.get("submission_id", ""))
            if not sub_id:
                continue

            if sub_id not in self._normalized_reviews:
                self._normalized_reviews[sub_id] = []

            # Check if parse was successful and we have normalized JSON
            if row.get("parse_success", False) and row.get("normalized_json"):
                try:
                    review_data = json.loads(row["normalized_json"])
                    self._normalized_reviews[sub_id].append(
                        ReviewData.from_normalized(review_data)
                    )
                except (json.JSONDecodeError, TypeError):
                    pass

        self._loaded_normalized = True

    def _get_image_paths(
        self,
        submission_id: str,
        year: int,
    ) -> Optional[List[Path]]:
        """Get PDF page image paths for a submission."""
        if not self.image_base_dir:
            return None

        # Try different path patterns
        patterns = [
            self.image_base_dir / str(year) / submission_id / "redacted_pdf_img_content",
            self.image_base_dir / f"normalized/{year}/{submission_id}/redacted_pdf_img_content",
        ]

        for img_dir in patterns:
            if img_dir.exists():
                images = sorted(img_dir.glob("page_*.png"))
                if images:
                    return images

        return None

    def _parse_reviews_from_json(
        self,
        reviews_json: Any,
    ) -> List[ReviewData]:
        """Parse reviews from JSON field."""
        if not reviews_json:
            return []

        if isinstance(reviews_json, str):
            try:
                reviews_json = json.loads(reviews_json)
            except json.JSONDecodeError:
                return []

        if not isinstance(reviews_json, list):
            return []

        reviews = []
        for review_dict in reviews_json:
            if isinstance(review_dict, dict):
                reviews.append(ReviewData.from_raw(review_dict))
        return reviews

    def _parse_labels(self, row: pd.Series) -> Dict[str, Any]:
        """Extract label fields from a row."""
        labels = {}

        # Citation percentile
        for key in ["citation_percentile", "citation_pctl", "citations_percentile"]:
            if key in row and pd.notna(row[key]):
                labels["citation_percentile"] = float(row[key])
                break

        # Mean rating
        for key in ["mean_rating", "avg_rating", "rating"]:
            if key in row and pd.notna(row[key]):
                labels["mean_rating"] = float(row[key])
                break

        return labels

    def load_from_csv(
        self,
        csv_path: Path,
        load_images: bool = True,
    ) -> Iterator[SubmissionData]:
        """
        Load submissions from a CSV file.

        Expected CSV columns (from schema):
        - submission_id: int
        - year: int
        - title: str
        - original_abstract or no_github_abstract: str
        - clean_md: str (optional)
        - standardized_reviews: JSON (optional)
        - _reviews_jsonable: JSON (optional)

        Args:
            csv_path: Path to CSV file
            load_images: Whether to load image paths

        Yields:
            SubmissionData objects
        """
        self._load_normalized_reviews()

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            submission_id = str(row.get("submission_id", ""))
            year = int(row.get("year", 0))

            # Get abstract (try multiple columns)
            abstract = ""
            for col in ["original_abstract", "no_github_abstract", "abstract"]:
                if col in row and pd.notna(row[col]):
                    abstract = str(row[col])
                    break

            # Get title
            title = str(row.get("title", ""))

            # Get decision
            decision = None
            if "decision" in row and pd.notna(row["decision"]):
                decision = str(row["decision"])

            # Get clean markdown
            clean_md = None
            if "clean_md" in row and pd.notna(row["clean_md"]):
                clean_md = str(row["clean_md"])

            # Get reviews - prefer normalized, fall back to raw
            reviews = self._normalized_reviews.get(submission_id, [])
            if not reviews:
                # Try standardized_reviews or _reviews_jsonable
                for col in ["standardized_reviews", "_reviews_jsonable"]:
                    if col in row and pd.notna(row[col]):
                        reviews = self._parse_reviews_from_json(row[col])
                        if reviews:
                            break

            # Get image paths
            image_paths = None
            if load_images:
                image_paths = self._get_image_paths(submission_id, year)

            # Parse labels
            labels = self._parse_labels(row)

            yield SubmissionData(
                submission_id=submission_id,
                year=year,
                title=title,
                abstract=abstract,
                decision=decision,
                clean_md=clean_md,
                reviews=reviews,
                pdf_image_paths=image_paths,
                labels=labels,
            )

    def load_from_parquet(
        self,
        parquet_path: Path,
        years: Optional[List[int]] = None,
        load_images: bool = True,
    ) -> Iterator[SubmissionData]:
        """
        Load submissions from parquet files.

        Supports both single parquet files and directories organized by year.

        Args:
            parquet_path: Path to parquet file or directory
            years: Optional list of years to load (for directories)
            load_images: Whether to load image paths

        Yields:
            SubmissionData objects
        """
        self._load_normalized_reviews()

        parquet_path = Path(parquet_path)

        if parquet_path.is_file():
            # Single parquet file
            parquet_files = [parquet_path]
        else:
            # Directory - find all parquet files
            years = years or [2020, 2021, 2022, 2023, 2024, 2025, 2026]
            parquet_files = []
            for year in years:
                year_dir = parquet_path / str(year)
                if year_dir.exists():
                    parquet_files.extend(year_dir.glob("*.parquet"))
            # Also check root directory
            parquet_files.extend(parquet_path.glob("*.parquet"))

        for pq_file in parquet_files:
            df = pd.read_parquet(pq_file)

            for _, row in df.iterrows():
                # Handle different column structures
                if "submission" in row and pd.notna(row["submission"]):
                    # HuggingFace dataset format with JSON submission
                    try:
                        if isinstance(row["submission"], str):
                            sub_dict = json.loads(row["submission"])
                        else:
                            sub_dict = row["submission"]
                        submission_id = str(sub_dict.get("id", ""))
                        title = sub_dict.get("title", "")
                        abstract = sub_dict.get("abstract", "")
                        decision = sub_dict.get("decision")
                        year = sub_dict.get("year", 0) or int(row.get("year", 0))
                    except (json.JSONDecodeError, TypeError):
                        continue
                else:
                    # Direct columns
                    submission_id = str(row.get("submission_id", row.get("id", "")))
                    title = str(row.get("title", ""))
                    abstract = str(row.get("abstract", row.get("original_abstract", "")))
                    decision = row.get("decision") if pd.notna(row.get("decision")) else None
                    year = int(row.get("year", 0))

                if not submission_id:
                    continue

                # Get clean markdown
                clean_md = None
                if "clean_md" in row and pd.notna(row["clean_md"]):
                    clean_md = str(row["clean_md"])

                # Get reviews
                reviews = self._normalized_reviews.get(submission_id, [])

                # Try to get reviews from row if not in normalized cache
                if not reviews:
                    for i in range(1, 13):  # review_1 to review_12
                        col = f"review_{i}"
                        if col in row and pd.notna(row[col]):
                            try:
                                if isinstance(row[col], str):
                                    review_dict = json.loads(row[col])
                                else:
                                    review_dict = row[col]
                                if review_dict:
                                    reviews.append(ReviewData.from_raw(review_dict))
                            except (json.JSONDecodeError, TypeError):
                                pass

                # Get image paths
                image_paths = None
                if load_images:
                    image_paths = self._get_image_paths(submission_id, year)

                # Parse labels
                labels = self._parse_labels(row)

                yield SubmissionData(
                    submission_id=submission_id,
                    year=year,
                    title=title,
                    abstract=abstract,
                    decision=decision,
                    clean_md=clean_md,
                    reviews=reviews,
                    pdf_image_paths=image_paths,
                    labels=labels,
                )

    def load_from_huggingface(
        self,
        dataset_name: str = "skonan/iclr-reviews-2020-2026",
        split: Optional[str] = None,
        years: Optional[List[str]] = None,
        load_images: bool = True,
        use_normalized_reviews: bool = True,
    ) -> Iterator[SubmissionData]:
        """
        Load submissions from HuggingFace dataset or local path.

        Dataset: https://huggingface.co/datasets/skonan/iclr-reviews-2020-2026

        Args:
            dataset_name: HuggingFace dataset name or local path to downloaded dataset
            split: Dataset split ('2020', '2021', ..., '2026', 'all', or None for all)
            years: Filter by years (alternative to split) - list of year strings like ["2020", "2021"]
            load_images: Whether to load image paths from clean_pdf_img_paths
            use_normalized_reviews: Use normalized_reviews if available

        Yields:
            SubmissionData objects
        """
        from datasets import load_dataset

        # Check if dataset_name is a local path
        dataset_path = Path(dataset_name)
        is_local = dataset_path.exists() and dataset_path.is_dir()

        # Determine which splits to load
        # Note: Cannot load without split or with split='all' due to HF dataset configuration
        # Must explicitly specify year splits
        if split == "all" or (split is None and years is None):
            # Load all year splits explicitly
            splits_to_load = ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]
        elif years:
            # Load specific years (already strings)
            splits_to_load = years
        else:
            # Load specific split
            splits_to_load = [split] if split else []
        print("Splits to load: ", splits_to_load)

        # Load datasets for each split
        datasets = []
        for split_name in splits_to_load:
            try:
                print(f"Loading {split_name} from {'local path' if is_local else 'HuggingFace'}")
                if is_local:
                    # Load from local path - pass the directory path
                    ds = load_dataset(str(dataset_path), split=split_name)
                else:
                    # Load from HuggingFace Hub
                    ds = load_dataset(dataset_name, split=split_name)
                datasets.append(ds)
                print(f"Loaded {split_name} of length {len(ds)}\n")
            except Exception as e:
                # Skip splits that fail to load
                print(f"Exception loading {split_name}: {e}")
                pass

        for dataset in datasets:
            print(f"DEBUG: Processing dataset, type={type(dataset)}, len={len(dataset) if hasattr(dataset, '__len__') else '?'}")
            for row in dataset:
                print(f"DEBUG: Got row with keys: {list(row.keys())[:5]}")
                submission_id = str(row.get("submission_id", ""))
                year = int(row.get("year", 0))
                title = str(row.get("title", ""))

                # Get abstract - prefer no_github_abstract (cleaned)
                abstract = row.get("no_github_abstract") or row.get("original_abstract") or ""

                # Get decision from submission_json
                decision = None
                submission_json = row.get("submission_json")
                if submission_json:
                    try:
                        sub_data = json.loads(submission_json) if isinstance(submission_json, str) else submission_json
                        decision = sub_data.get("decision")
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Get clean markdown
                clean_md = row.get("clean_md")

                # Get reviews
                reviews = []
                if use_normalized_reviews and row.get("normalized_reviews"):
                    reviews = self._parse_reviews_from_json(row["normalized_reviews"])
                if not reviews and row.get("original_reviews"):
                    reviews = self._parse_reviews_from_json(row["original_reviews"])

                # Get image paths from clean_pdf_img_paths
                image_paths = None
                if load_images and row.get("clean_pdf_img_paths"):
                    try:
                        paths_data = row["clean_pdf_img_paths"]
                        if isinstance(paths_data, str):
                            paths_list = json.loads(paths_data)
                        else:
                            paths_list = paths_data

                        if paths_list:
                            image_paths = [Path(p) for p in paths_list if p]
                            # Filter to only existing paths
                            image_paths = [p for p in image_paths if p.exists()]
                            if not image_paths:
                                image_paths = None
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Parse meta review for additional context
                meta_review = None
                if row.get("normalized_metareview"):
                    try:
                        meta_review = json.loads(row["normalized_metareview"]) if isinstance(row["normalized_metareview"], str) else row["normalized_metareview"]
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Build labels
                labels = {}
                if decision:
                    labels["decision"] = decision

                yield SubmissionData(
                    submission_id=submission_id,
                    year=year,
                    title=title,
                    abstract=abstract,
                    decision=decision,
                    clean_md=clean_md,
                    reviews=reviews,
                    meta_review=meta_review,
                    pdf_image_paths=image_paths,
                    labels=labels,
                )

    def load_all(
        self,
        source_path: Optional[Path] = None,
        years: Optional[List[int]] = None,
        load_images: bool = True,
        from_huggingface: bool = False,
        hf_dataset: str = "skonan/iclr-reviews-2020-2026",
        hf_split: Optional[str] = None,
    ) -> List[SubmissionData]:
        """
        Load all submissions from a source (auto-detect format).

        Args:
            source_path: Path to CSV, parquet file, or directory (ignored if from_huggingface=True)
            years: Optional years filter
            load_images: Whether to load image paths
            from_huggingface: If True, load from HuggingFace dataset
            hf_dataset: HuggingFace dataset name
            hf_split: HuggingFace split to load

        Returns:
            List of SubmissionData objects
        """
        if from_huggingface:
            return list(self.load_from_huggingface(
                dataset_name=hf_dataset,
                split=hf_split,
                years=years,
                load_images=load_images,
            ))

        if source_path is None:
            raise ValueError("source_path required when not loading from HuggingFace")

        source_path = Path(source_path)

        if source_path.suffix == ".csv":
            return list(self.load_from_csv(source_path, load_images))
        elif source_path.suffix == ".parquet" or source_path.is_dir():
            return list(self.load_from_parquet(source_path, years, load_images))
        else:
            raise ValueError(f"Unknown source format: {source_path}")
