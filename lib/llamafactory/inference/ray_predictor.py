"""
Ray Data + vLLM predictor for streaming inference.

Replaces UnifiedPredictor with a streaming architecture that:
- Uses Ray Data for lazy loading
- Uses vLLM for efficient inference
- Streams results to parquet (no memory materialization)
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor

from ..core.registry import get_input_formatter, get_output_handler
from ..core.types import (
    InputModality,
    OutputType,
    ReviewData,
    SubmissionData,
)


class RayDataPredictor:
    """
    Ray Data + vLLM predictor for streaming inference.

    Uses Ray Data for lazy data loading and vLLM for efficient inference.
    Supports both text-only and vision-language models.

    Example:
        predictor = RayDataPredictor(
            input_modality=InputModality.TEXT_ONLY,
            output_type=OutputType.BINARY,
            model_name="Qwen/Qwen2.5-7B-Instruct",
            ray_address="auto",
        )
        predictor.predict_from_hf_dataset(
            dataset_path="data/iclr_split",
            output_path="outputs/predictions.parquet",
        )
    """

    # Fixed sampling parameters (matching VLLMTransformer)
    TEMPERATURE = 0.7
    TOP_P = 0.8
    TOP_K = 20
    MIN_P = 0.0

    def __init__(
        self,
        input_modality: InputModality,
        output_type: OutputType,
        model_name: str,
        ray_address: Optional[str] = None,
        # vLLM params
        tensor_parallel: int = 1,
        concurrency: int = 1,
        batch_size: int = 32,
        max_model_len: int = 8192,
        max_tokens: int = 4096,
        n: int = 1,
        # Input formatter params
        max_input_tokens: int = 16000,
        include_reviews: bool = True,
        include_markdown: bool = False,
        max_images: int = 20,
        # Output handler params
        include_reasoning: bool = True,
    ):
        """
        Initialize Ray Data predictor.

        Args:
            input_modality: Input content type
                - TEXT_ONLY: title + abstract + clean_md (text only)
                - TEXT_WITH_IMAGES: title + abstract + clean_md with inline images
                - IMAGES_ONLY: PDF page images
            output_type: Prediction output type (BINARY, MULTICLASS, MEAN_RATING)
            model_name: HuggingFace model name or path
            ray_address: Ray cluster address (None for local)
            tensor_parallel: Number of GPUs for model parallelism
            concurrency: Number of vLLM replicas
            batch_size: Inference batch size
            max_model_len: Maximum context length
            max_tokens: Maximum tokens to generate
            n: Number of completions per request
            max_input_tokens: Maximum tokens for input content
            include_reviews: Whether to include reviewer feedback
            include_markdown: Whether to include paper markdown (TEXT_ONLY)
            max_images: Maximum images (PDF pages for IMAGES_ONLY)
            include_reasoning: Whether to request reasoning in output
        """
        self.input_modality = input_modality
        self.output_type = output_type
        self.model_name = model_name
        self.ray_address = ray_address

        # vLLM config
        self.tensor_parallel = tensor_parallel
        self.concurrency = concurrency
        self.batch_size = batch_size
        self.max_model_len = max_model_len
        self.max_tokens = max_tokens
        self.n = n

        # Create input formatter based on modality
        if input_modality == InputModality.TEXT_WITH_IMAGES:
            # TEXT_WITH_IMAGES: Use markdown with inline image references
            from ..inputs.text_with_images import MarkdownWithInlineImagesFormatter
            self.input_formatter = MarkdownWithInlineImagesFormatter(
                max_tokens=max_input_tokens,
                include_reviews=include_reviews,
            )
        elif input_modality == InputModality.IMAGES_ONLY:
            # IMAGES_ONLY: Use PDF page images
            from ..inputs.images_only import ImagesOnlyFormatter
            self.input_formatter = ImagesOnlyFormatter(
                max_images=max_images,
            )
        else:
            # TEXT_ONLY: Use text-only formatter
            from ..inputs.text_only import FullContextFormatter
            self.input_formatter = FullContextFormatter(
                max_tokens=max_input_tokens,
                include_reviews=include_reviews,
                include_markdown=include_markdown,
            )

        # Create output handler
        output_handler_cls = get_output_handler(output_type)
        self.output_handler = output_handler_cls(
            include_reasoning=include_reasoning,
        )

        # Check if VL model needed
        self.is_vl_model = self.input_formatter.requires_vl_model

    def _extract_page_num(self, path: Path) -> int:
        """Extract page number from path like page_1.png, page_10.png, etc."""
        match = re.search(r'page_(\d+)', str(path))
        return int(match.group(1)) if match else 0

    def _init_ray(self) -> None:
        """Initialize Ray connection."""
        env_vars = {
            "HOME": os.environ.get("HOME", ""),
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "HF_HUB_OFFLINE": "0",
        }
        for var in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME"]:
            if var in os.environ:
                env_vars[var] = os.environ[var]

        runtime_env = {"env_vars": env_vars}

        if self.ray_address:
            print(f"Connecting to Ray cluster at {self.ray_address}")
            ray.init(address=self.ray_address, ignore_reinit_error=True, runtime_env=runtime_env)
        else:
            print("Starting local Ray cluster")
            ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

    def _row_to_submission(self, row: Dict[str, Any]) -> SubmissionData:
        """Convert dataset row to SubmissionData (lazy loading)."""
        # Load markdown content on-demand
        clean_md = ""
        if row.get("clean_md_path"):
            try:
                clean_md = Path(row["clean_md_path"]).read_text(encoding="utf-8")
            except Exception:
                pass

        # Parse decision from submission_json
        decision = None
        if row.get("submission_json"):
            try:
                data = json.loads(row["submission_json"])
                decision = data.get("decision")
            except (json.JSONDecodeError, TypeError):
                pass

        # Parse reviews - handles nested JSON (list of JSON strings)
        reviews = []
        if row.get("normalized_reviews"):
            try:
                reviews_data = json.loads(row["normalized_reviews"])
                if isinstance(reviews_data, list):
                    for r in reviews_data:
                        # Each item might be a JSON string or a dict
                        if isinstance(r, str):
                            try:
                                r = json.loads(r)
                            except (json.JSONDecodeError, TypeError):
                                continue
                        if isinstance(r, dict):
                            reviews.append(ReviewData.from_normalized(r))
            except (json.JSONDecodeError, TypeError):
                pass

        # Get image paths - handles both clean_pdf_img_paths (JSON list) and pdf_image_dir
        pdf_image_paths = []
        if row.get("clean_pdf_img_paths"):
            try:
                # Parse JSON string to list of paths
                paths_data = row["clean_pdf_img_paths"]
                if isinstance(paths_data, str):
                    paths_data = json.loads(paths_data)
                if isinstance(paths_data, list):
                    pdf_image_paths = [Path(p) for p in paths_data if p]
                    # Sort by page number (page_1, page_2, ..., page_10, page_11)
                    pdf_image_paths = sorted(pdf_image_paths, key=self._extract_page_num)
            except (json.JSONDecodeError, TypeError):
                pass
        elif row.get("pdf_image_dir"):
            img_dir = Path(row["pdf_image_dir"])
            if img_dir.exists():
                # Sort by page number numerically
                pdf_image_paths = sorted(img_dir.glob("page_*.png"), key=self._extract_page_num)

        return SubmissionData(
            submission_id=str(row.get("submission_id", "")),
            year=int(row.get("year", 0)),
            title=str(row.get("title", "")),
            abstract=str(row.get("no_github_abstract") or row.get("original_abstract", "")),
            decision=decision,
            clean_md=clean_md,
            reviews=reviews,
            pdf_image_paths=pdf_image_paths,
        )

    def _parse_decision(self, row: Dict[str, Any]) -> Optional[str]:
        """Extract decision from row."""
        if row.get("submission_json"):
            try:
                data = json.loads(row["submission_json"])
                return data.get("decision")
            except (json.JSONDecodeError, TypeError):
                pass
        return None

    def _build_messages(
        self,
        prompt: str,
        image_paths: Optional[List[str]],
        images_in_md: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Build messages for vLLM, handling images for VL models.

        For TEXT_WITH_IMAGES modality, parses markdown for ![](images/...)
        patterns and inserts images inline (interleaved with text).

        Args:
            prompt: The text prompt (may contain markdown image refs)
            image_paths: List of image paths (for non-inline images)
            images_in_md: Mapping from image filename to full path
        """
        if not image_paths and not images_in_md:
            return [{"role": "user", "content": prompt}]

        # Check if we should do inline image insertion
        if images_in_md and "![" in prompt:
            return self._build_inline_image_messages(prompt, images_in_md)

        # Fallback: all images at the start, then text
        content = []
        if image_paths:
            for path in image_paths:
                content.append({"type": "image", "image": f"file://{path}"})
        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]

    def _build_inline_image_messages(
        self,
        prompt: str,
        images_in_md: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Build messages with inline images parsed from markdown.

        Parses ![](images/filename.ext) or ![...](images/filename.ext) patterns
        and replaces them with actual image content, creating interleaved
        text/image content.
        """
        # Pattern to match markdown images: ![alt](images/filename.ext)
        # Also matches ![](images/...) with empty alt text
        pattern = r'!\[[^\]]*\]\(images/([^)]+)\)'

        content = []
        last_end = 0

        for match in re.finditer(pattern, prompt):
            # Add text before this image
            text_before = prompt[last_end:match.start()].strip()
            if text_before:
                content.append({"type": "text", "text": text_before})

            # Get the image filename and look up full path
            image_filename = match.group(1)
            image_path = images_in_md.get(image_filename)

            if image_path and Path(image_path).exists():
                content.append({"type": "image", "image": f"file://{image_path}"})
            else:
                # Image not found, keep the markdown reference as text
                content.append({"type": "text", "text": match.group(0)})

            last_end = match.end()

        # Add remaining text after the last image
        text_after = prompt[last_end:].strip()
        if text_after:
            content.append({"type": "text", "text": text_after})

        # If no content was added (no images found), just return text
        if not content:
            return [{"role": "user", "content": prompt}]

        return [{"role": "user", "content": content}]

    def _parse_images_in_md(self, row: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Parse images_in_clean_md column to get filename->path mapping."""
        if not row.get("images_in_clean_md"):
            return None

        try:
            data = row["images_in_clean_md"]
            if isinstance(data, str):
                data = json.loads(data)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dataset row to vLLM messages format."""
        # Build SubmissionData from row
        submission = self._row_to_submission(row)

        # Format content using InputFormatter
        content = self.input_formatter.format_content(submission)
        instruction = self.output_handler.get_instruction()
        prompt = f"{instruction}\n\n{content}"

        # Get image paths (not loaded - just paths)
        images = self.input_formatter.get_images(submission)

        # Get inline image mapping ONLY for TEXT_WITH_IMAGES modality
        images_in_md = None
        if self.input_modality == InputModality.TEXT_WITH_IMAGES:
            images_in_md = self._parse_images_in_md(row)

        # Build messages (with inline image support for TEXT_WITH_IMAGES)
        messages = self._build_messages(prompt, images, images_in_md)

        return {
            "messages": messages,
            # Pass through for postprocess
            "submission_id": row.get("submission_id"),
            "year": row.get("year"),
            "title": row.get("title"),
            "decision": self._parse_decision(row),
        }

    def postprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Parse vLLM output and extract prediction."""
        generated_text = row.get("generated_text", "")

        # Handle multiple completions
        if isinstance(generated_text, list):
            generated_text = generated_text[0] if generated_text else ""

        # Use OutputHandler to parse
        parsed = self.output_handler.parse_prediction(generated_text)

        # Extract metrics
        metrics = row.get("metrics", {})
        time_taken = None
        if isinstance(metrics, dict):
            arrival = metrics.get("arrival_time")
            last_token = metrics.get("last_token_ts")
            if arrival is not None and last_token is not None:
                time_taken = last_token - arrival

        return {
            "submission_id": row.get("submission_id"),
            "year": row.get("year"),
            "title": row.get("title"),
            "ground_truth": row.get("decision"),
            "prediction": parsed.get("prediction"),
            "reasoning": parsed.get("reasoning"),
            "raw_output": generated_text,
            "parse_success": parsed.get("parse_success", False),
            "num_generated_tokens": row.get("num_generated_tokens"),
            "time_taken": time_taken,
        }

    def _build_processor(self):
        """Build the vLLM processor."""
        config = vLLMEngineProcessorConfig(
            model_source=self.model_name,
            concurrency=self.concurrency,
            batch_size=self.batch_size,
            engine_kwargs={
                "max_model_len": self.max_model_len,
                "trust_remote_code": True,
                "tensor_parallel_size": self.tensor_parallel,
            }
        )

        # Build sampling params
        sampling_params_base = {
            "temperature": self.TEMPERATURE,
            "top_p": self.TOP_P,
            "top_k": self.TOP_K,
            "min_p": self.MIN_P,
            "max_tokens": self.max_tokens,
            "n": self.n,
        }

        # Wrap preprocess to include sampling params
        def preprocess_fn(row: Dict[str, Any]) -> Dict[str, Any]:
            result = self.preprocess(row)
            result["sampling_params"] = sampling_params_base.copy()
            return result

        return build_llm_processor(
            config,
            preprocess=preprocess_fn,
            postprocess=self.postprocess,
        )

    def predict_from_parquet(
        self,
        input_path: str,
        output_path: str,
    ) -> None:
        """
        Stream from parquet, run inference, write to parquet.

        Args:
            input_path: Path to input parquet file/directory
            output_path: Path to output parquet file/directory
        """
        self._init_ray()

        ctx = ray.data.DataContext.get_current()
        ctx.verbose_stats_logs = True

        print(f"Reading from {input_path}")
        ds = ray.data.read_parquet(input_path)

        print("Building vLLM processor...")
        processor = self._build_processor()

        print("Running inference...")
        ds = processor(ds)

        print(f"Writing results to {output_path}")
        ds.write_parquet(output_path)
        print("Done!")

    def predict_from_hf_dataset(
        self,
        dataset_path: str,
        output_path: str,
        split: str = "test",
    ) -> None:
        """
        Stream from HuggingFace dataset, run inference, write to parquet.

        Args:
            dataset_path: Path to HuggingFace dataset
            output_path: Path to output parquet file/directory
            split: Dataset split to use
        """
        from datasets import load_from_disk

        self._init_ray()

        ctx = ray.data.DataContext.get_current()
        ctx.verbose_stats_logs = True

        print(f"Loading HuggingFace dataset from {dataset_path}")
        hf_ds = load_from_disk(dataset_path)

        if split not in hf_ds:
            available = list(hf_ds.keys())
            raise ValueError(f"Split '{split}' not found. Available: {available}")

        hf_split = hf_ds[split]
        print(f"Split '{split}': {len(hf_split):,} rows")

        print("Converting to Ray Dataset...")
        ds = ray.data.from_huggingface(hf_split)

        print("Building vLLM processor...")
        processor = self._build_processor()

        print("Running inference...")
        ds = processor(ds)

        print(f"Writing results to {output_path}")
        ds.write_parquet(output_path)
        print("Done!")

    def predict_to_pandas(
        self,
        dataset_path: str,
        split: str = "test",
        max_samples: Optional[int] = None,
    ):
        """
        Run inference and return results as pandas DataFrame.

        Warning: Loads all results into memory. Use predict_from_hf_dataset
        for large datasets.

        Args:
            dataset_path: Path to HuggingFace dataset
            split: Dataset split to use
            max_samples: Maximum samples to process (for testing)

        Returns:
            pandas DataFrame with predictions
        """
        from datasets import load_from_disk

        self._init_ray()

        print(f"Loading HuggingFace dataset from {dataset_path}")
        hf_ds = load_from_disk(dataset_path)
        hf_split = hf_ds[split]

        if max_samples:
            hf_split = hf_split.select(range(min(max_samples, len(hf_split))))
            print(f"Limited to {len(hf_split)} samples")

        ds = ray.data.from_huggingface(hf_split)

        print("Building vLLM processor...")
        processor = self._build_processor()

        print("Running inference...")
        ds = processor(ds)

        print("Collecting results to pandas...")
        return ds.to_pandas()
