"""
Zero-shot prediction mode.

Runs inference directly on the base model without any training.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.registry import register_training_mode
from ..core.types import (
    FormattedSample,
    InferenceConfig,
    TrainingConfig,
    TrainingMode,
)
from .base import TrainingModeBase, build_prompt


@register_training_mode(TrainingMode.ZERO_SHOT)
class ZeroShotMode(TrainingModeBase):
    """
    Zero-shot prediction without any training.

    Uses the base model directly for predictions. Supports both
    vLLM and HuggingFace transformers backends.
    """

    mode = TrainingMode.ZERO_SHOT
    requires_training = False

    def __init__(
        self,
        config: TrainingConfig,
        input_formatter: "InputFormatter",
        output_handler: "OutputHandler",
        use_vllm: bool = True,
    ):
        """
        Initialize zero-shot mode.

        Args:
            config: Training configuration (model settings)
            input_formatter: Input formatting strategy
            output_handler: Output handling strategy
            use_vllm: Whether to use vLLM for inference (faster)
        """
        super().__init__(config, input_formatter, output_handler)
        self.use_vllm = use_vllm
        self._model = None
        self._tokenizer = None

    def prepare_data(
        self,
        samples: List[FormattedSample],
        output_path: Path,
    ) -> Dict[str, Any]:
        """No data preparation needed for zero-shot."""
        return {
            "mode": "zero_shot",
            "num_samples": len(samples),
            "requires_training": False,
        }

    def generate_config(
        self,
        data_info: Dict[str, Any],
        output_path: Path,
    ) -> Optional[Path]:
        """No config needed for zero-shot."""
        return None

    def train(
        self,
        config_path: Optional[Path] = None,
        resume_from: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """No training for zero-shot."""
        return {
            "status": "skipped",
            "mode": "zero_shot",
            "message": "Zero-shot mode does not require training",
        }

    def predict(
        self,
        samples: List[FormattedSample],
        inference_config: InferenceConfig,
    ) -> List[Dict[str, Any]]:
        """Run zero-shot inference."""
        if self.use_vllm:
            return self._predict_vllm(samples, inference_config)
        else:
            return self._predict_transformers(samples, inference_config)

    def _predict_vllm(
        self,
        samples: List[FormattedSample],
        inference_config: InferenceConfig,
    ) -> List[Dict[str, Any]]:
        """Run inference using vLLM."""
        from vllm import LLM, SamplingParams

        # Initialize model if needed
        model_name = inference_config.model_name_or_path or self.config.get_model_name()

        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=inference_config.tensor_parallel_size,
            gpu_memory_utilization=inference_config.gpu_memory_utilization,
        )

        sampling_params = SamplingParams(
            temperature=inference_config.temperature,
            max_tokens=inference_config.max_tokens,
            stop=None,
        )

        # Build prompts
        prompts = [build_prompt(sample) for sample in samples]

        # Run inference
        outputs = llm.generate(prompts, sampling_params)

        # Parse results
        results = []
        for sample, output in zip(samples, outputs):
            generated_text = output.outputs[0].text
            parsed = self.output_handler.parse_prediction(generated_text)
            parsed["submission_id"] = sample.submission_id
            parsed["year"] = sample.year
            results.append(parsed)

        return results

    def _predict_transformers(
        self,
        samples: List[FormattedSample],
        inference_config: InferenceConfig,
    ) -> List[Dict[str, Any]]:
        """Run inference using HuggingFace transformers."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = inference_config.model_name_or_path or self.config.get_model_name()

        # Load model and tokenizer
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        results = []
        batch_size = inference_config.batch_size

        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]
            prompts = [build_prompt(s) for s in batch_samples]

            # Tokenize
            inputs = self._tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            ).to(self._model.device)

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=inference_config.max_tokens,
                    temperature=inference_config.temperature if inference_config.temperature > 0 else None,
                    do_sample=inference_config.temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            # Decode and parse
            for j, (sample, output) in enumerate(zip(batch_samples, outputs)):
                # Get only the generated tokens
                input_len = inputs["input_ids"][j].shape[0]
                generated_ids = output[input_len:]
                generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

                parsed = self.output_handler.parse_prediction(generated_text)
                parsed["submission_id"] = sample.submission_id
                parsed["year"] = sample.year
                results.append(parsed)

        return results

    def predict_single(
        self,
        sample: FormattedSample,
        inference_config: Optional[InferenceConfig] = None,
    ) -> Dict[str, Any]:
        """
        Run inference on a single sample.

        Args:
            sample: Single FormattedSample
            inference_config: Optional inference settings

        Returns:
            Parsed prediction dict
        """
        if inference_config is None:
            inference_config = InferenceConfig(
                model_name_or_path=self.config.get_model_name(),
            )

        results = self.predict([sample], inference_config)
        return results[0] if results else {}
