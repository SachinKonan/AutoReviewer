"""
Base class for vLLM-based transformations using Ray Data.

Subclasses implement preprocess/postprocess for specific use cases.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd
import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor


class VLLMTransformer(ABC):
    """Base class for vLLM-based transformations.

    Subclasses must implement:
        - preprocess(row): Transform input row to vLLM format
        - postprocess(row): Transform vLLM output to final format

    Sampling parameters are fixed (temp=0.7, top_p=0.8, top_k=20, min_p=0).
    Only `n` (number of completions) is user-configurable.

    Usage:
        class MyTransformer(VLLMTransformer):
            def preprocess(self, row):
                return {
                    "messages": [{"role": "user", "content": row["text"]}],
                }

            def postprocess(self, row):
                return {"output": row["generated_text"]}

        transformer = MyTransformer(df, model_name="meta-llama/Llama-3.1-8B-Instruct", n=3)
        result_df = transformer.transform()
    """

    # Fixed sampling parameters (not user-configurable)
    TEMPERATURE = 0.7
    TOP_P = 0.8
    TOP_K = 20
    MIN_P = 0.0

    def __init__(
        self,
        df: pd.DataFrame,
        model_name: str,
        tensor_parallel: int = 1,
        concurrency: int = 1,
        batch_size: int = 32,
        max_model_len: int = 8192,
        ray_address: Optional[str] = None,
        max_tokens: int = 4096,
        n: int = 2,
        guided_json: Optional[dict] = None,
    ):
        """Initialize the transformer.

        Args:
            df: Input DataFrame with rows to transform
            model_name: HuggingFace model name or path
            tensor_parallel: Number of GPUs to split model across
            concurrency: Number of vLLM replicas
            batch_size: Batch size for inference
            max_model_len: Maximum context length
            ray_address: Ray cluster address (None for auto-detect)
            max_tokens: Maximum tokens to generate per request
            n: Number of completions to generate per request
            guided_json: JSON schema for structured output (optional)

        Note:
            Sampling parameters are fixed: temp=0.7, top_p=0.8, top_k=20, min_p=0
        """
        self.df = df
        self.model_name = model_name
        self.tensor_parallel = tensor_parallel
        self.concurrency = concurrency
        self.batch_size = batch_size
        self.max_model_len = max_model_len
        self.ray_address = ray_address
        self.max_tokens = max_tokens
        self.n = n
        self.guided_json = guided_json

    @abstractmethod
    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Transform input row to vLLM format.

        Args:
            row: Input row as dictionary

        Returns:
            Dictionary with:
                - messages: List of message dicts [{"role": "user", "content": "..."}]
                - sampling_params: Dict with temperature, max_tokens, etc.
        """
        pass

    @abstractmethod
    def postprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Transform vLLM output row to final format.

        Args:
            row: Output row with:
                - generated_text: Generated text (or list if n > 1)
                - metrics: Timing metrics dict
                - num_generated_tokens: Token count
                - messages: Original messages from preprocess

        Returns:
            Dictionary with final output columns
        """
        pass

    def _init_ray(self) -> None:
        """Initialize Ray connection."""
        # Pass through environment vars to workers
        env_vars = {
            "HOME": os.environ.get("HOME", ""),
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            # Explicitly disable offline mode (override parent shell)
            "HF_HUB_OFFLINE": "0",
        }
        # Pass through HF cache dirs if set
        for var in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME"]:
            if var in os.environ:
                env_vars[var] = os.environ[var]

        runtime_env = {
            "env_vars": env_vars,
        }

        if self.ray_address:
            print(f"Connecting to Ray cluster at {self.ray_address}")
            ray.init(address=self.ray_address, ignore_reinit_error=True, runtime_env=runtime_env)
        else:
            raise ValueError("ray_address is required - start Ray with 'ray start --head' and pass --ray-address")

    def _build_processor(self):
        """Build the vLLM processor with config."""
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

        # Build sampling params dict with fixed values
        sampling_params_base = {
            "temperature": self.TEMPERATURE,
            "top_p": self.TOP_P,
            "top_k": self.TOP_K,
            "min_p": self.MIN_P,
            "max_tokens": self.max_tokens,
            "n": self.n,
        }
        if self.guided_json:
            sampling_params_base["guided_json"] = self.guided_json

        # Wrap preprocess to include default sampling params
        def preprocess_fn(row: Dict[str, Any]) -> Dict[str, Any]:
            result = self.preprocess(row)
            # Merge default sampling params with any custom ones from preprocess
            if "sampling_params" not in result:
                result["sampling_params"] = {}
            merged_params = {**sampling_params_base, **result["sampling_params"]}
            result["sampling_params"] = merged_params
            return result

        return build_llm_processor(
            config,
            preprocess=preprocess_fn,
            postprocess=self.postprocess,
        )

    def transform(self) -> pd.DataFrame:
        """Run the full transformation pipeline.

        Returns:
            DataFrame with transformed results
        """
        # Initialize Ray
        self._init_ray()

        # Enable verbose stats for debugging
        ctx = ray.data.DataContext.get_current()
        ctx.verbose_stats_logs = True

        # Convert to Ray Dataset
        print(f"Processing {len(self.df)} rows...")
        ds = ray.data.from_pandas(self.df)

        # Build and apply processor
        print("Building vLLM processor...")
        processor = self._build_processor()

        print("Running inference...")
        ds = processor(ds)

        # Convert back to pandas
        print("Collecting results...")
        result_df = ds.to_pandas()

        return result_df

    def transform_to_parquet(self, output_path: str) -> None:
        """Run transformation and write directly to parquet (streaming).

        Args:
            output_path: Path to output parquet file/directory
        """
        # Initialize Ray
        self._init_ray()

        # Enable verbose stats
        ctx = ray.data.DataContext.get_current()
        ctx.verbose_stats_logs = True

        # Convert to Ray Dataset
        print(f"Processing {len(self.df)} rows...")
        ds = ray.data.from_pandas(self.df)

        # Build and apply processor
        print("Building vLLM processor...")
        processor = self._build_processor()

        print("Running inference...")
        ds = processor(ds)

        # Write directly to parquet (streaming, no memory load)
        print(f"Writing results to {output_path}")
        ds.write_parquet(output_path)
        print("Done!")
