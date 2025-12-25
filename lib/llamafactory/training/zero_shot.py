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
    ModelType,
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

    def _is_vision_model(self, model_name: str) -> bool:
        """Check if model is a vision-language model."""
        vl_indicators = ["vl", "vision", "VL", "Vision"]
        return any(ind in model_name for ind in vl_indicators)

    def predict(
        self,
        samples: List[FormattedSample],
        inference_config: InferenceConfig,
    ) -> List[Dict[str, Any]]:
        """Run zero-shot inference."""
        model_name = inference_config.model_name_or_path or self.config.get_model_name()
        is_vl = (
            self.config.model_type == ModelType.VISION_LANGUAGE
            or self._is_vision_model(model_name)
        )

        # vLLM has compatibility issues with some VL models (flash attention)
        # Fall back to transformers for VL models
        use_vllm = self.use_vllm and not is_vl

        if is_vl and self.use_vllm:
            print("Note: Using transformers backend for VL model (vLLM has compatibility issues)")

        if use_vllm:
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
        from transformers import AutoTokenizer

        # Initialize model if needed
        model_name = inference_config.model_name_or_path or self.config.get_model_name()

        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=inference_config.tensor_parallel_size,
            gpu_memory_utilization=inference_config.gpu_memory_utilization,
        )

        # Load tokenizer for chat template
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        sampling_params = SamplingParams(
            temperature=inference_config.temperature,
            max_tokens=inference_config.max_tokens,
            stop=None,
        )

        # Build prompts with chat template
        prompts = []
        for sample in samples:
            messages = [{"role": "user", "content": build_prompt(sample)}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(text)

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
        model_name = inference_config.model_name_or_path or self.config.get_model_name()
        is_vl = (
            self.config.model_type == ModelType.VISION_LANGUAGE
            or self._is_vision_model(model_name)
        )

        if is_vl:
            return self._predict_transformers_vl(samples, inference_config)
        else:
            return self._predict_transformers_text(samples, inference_config)

    def _predict_transformers_text(
        self,
        samples: List[FormattedSample],
        inference_config: InferenceConfig,
    ) -> List[Dict[str, Any]]:
        """Run inference using HuggingFace transformers for text-only models."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = inference_config.model_name_or_path or self.config.get_model_name()

        # Load model and tokenizer
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            # Ensure pad token is set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
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

            # Build chat messages and apply chat template for instruct models
            batch_inputs = []
            for s in batch_samples:
                messages = [
                    {"role": "user", "content": build_prompt(s)}
                ]
                text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                batch_inputs.append(text)

            # Tokenize
            inputs = self._tokenizer(
                batch_inputs,
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

    def _predict_transformers_vl(
        self,
        samples: List[FormattedSample],
        inference_config: InferenceConfig,
    ) -> List[Dict[str, Any]]:
        """Run inference using HuggingFace transformers for vision-language models."""
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        model_name = inference_config.model_name_or_path or self.config.get_model_name()

        # Load model and processor (VL models use processor instead of tokenizer)
        if self._model is None:
            print(f"Loading VL model: {model_name}")
            self._tokenizer = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            print("VL model loaded successfully")

        results = []

        # Process one sample at a time for VL models (images make batching complex)
        for sample in samples:
            # Build message with images
            content = []

            # Add images if present
            if sample.images:
                for img_path in sample.images:
                    content.append({
                        "type": "image",
                        "image": f"file://{img_path}",
                    })

            # Add text
            content.append({
                "type": "text",
                "text": build_prompt(sample),
            })

            messages = [{"role": "user", "content": content}]

            # Apply chat template
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Process vision info (handles image loading and preprocessing)
            image_inputs, video_inputs = process_vision_info(messages)

            # Tokenize with images
            inputs = self._tokenizer(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self._model.device)

            # Generate
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=inference_config.max_tokens,
                    temperature=inference_config.temperature if inference_config.temperature > 0 else None,
                    do_sample=inference_config.temperature > 0,
                )

            # Decode - get only generated tokens
            generated_ids = [
                output_ids[0][len(inputs.input_ids[0]):]
            ]
            generated_text = self._tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

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
