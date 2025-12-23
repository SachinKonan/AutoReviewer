"""
Group Relative Policy Optimization (GRPO) training mode.

Uses EasyR1 for GRPO training with custom reward functions.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from ..core.registry import register_training_mode
from ..core.types import (
    FormattedSample,
    InferenceConfig,
    TrainingConfig,
    TrainingMode,
)
from .base import TrainingModeBase, build_prompt

if TYPE_CHECKING:
    from ..inputs.base import InputFormatter
    from ..outputs.base import OutputHandler


@register_training_mode(TrainingMode.GRPO)
class GRPOMode(TrainingModeBase):
    """
    Group Relative Policy Optimization training.

    Uses EasyR1 library for GRPO training with custom reward functions.
    GRPO is effective for tasks with clear reward signals.
    """

    mode = TrainingMode.GRPO
    requires_training = True

    def __init__(
        self,
        config: TrainingConfig,
        input_formatter: "InputFormatter",
        output_handler: "OutputHandler",
        reward_function: Optional[Callable[[str, str], float]] = None,
    ):
        super().__init__(config, input_formatter, output_handler)
        self.reward_function = reward_function or self._default_reward

    def _default_reward(self, prediction: str, ground_truth: str) -> float:
        """Default accuracy-based reward function."""
        parsed = self.output_handler.parse_prediction(prediction)
        pred_value = parsed.get("prediction")

        if self.output_handler.get_label_choices() is not None:
            return 1.0 if pred_value == ground_truth else 0.0

        try:
            pred_float = float(pred_value) if pred_value is not None else 0.5
            gt_float = float(ground_truth)
            error = abs(pred_float - gt_float)
            return max(0.0, 1.0 - error)
        except (ValueError, TypeError):
            return 0.0

    def prepare_data(
        self,
        samples: List[FormattedSample],
        output_path: Path,
    ) -> Dict[str, Any]:
        """Prepare data for GRPO training."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        grpo_data = []
        for sample in samples:
            item = {
                "prompt": f"{sample.instruction}\n\n{sample.input_text}",
                "ground_truth": sample.output,
            }
            if sample.images:
                item["images"] = sample.images
            grpo_data.append(item)

        data_file = output_path / "grpo_train.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(grpo_data, f, indent=2, ensure_ascii=False)

        has_images = any(s.images for s in samples)

        return {
            "data_file": str(data_file),
            "dataset_dir": str(output_path),
            "num_samples": len(samples),
            "has_images": has_images,
        }

    def generate_config(
        self,
        data_info: Dict[str, Any],
        output_path: Path,
    ) -> Path:
        """Generate GRPO training configuration."""
        import yaml

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        config = {
            "model_name_or_path": self.config.get_model_name(),
            "template": self.config.get_template(),
            "stage": "grpo",
            "do_train": True,
            "finetuning_type": "lora",
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "lora_target": "all",
            "grpo_group_size": self.config.grpo_group_size,
            "kl_coeff": self.config.grpo_kl_coeff,
            "dataset": "grpo_train",
            "dataset_dir": data_info["dataset_dir"],
            "cutoff_len": self.config.max_length,
            "per_device_train_batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "num_train_epochs": self.config.num_epochs,
            "bf16": self.config.bf16,
            "output_dir": str(output_path / "grpo_checkpoints"),
            "logging_steps": self.config.logging_steps,
            "save_steps": self.config.save_steps,
        }

        if data_info.get("has_images"):
            config["visual_inputs"] = True

        config_path = output_path / "grpo_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        return config_path

    def train(
        self,
        config_path: Optional[Path] = None,
        resume_from: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Execute GRPO training."""
        if config_path is None:
            raise ValueError("config_path required for GRPO training")

        try:
            from easyr1.train import train_grpo
            import yaml

            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)

            if resume_from:
                config_dict["resume_from_checkpoint"] = str(resume_from)

            def reward_fn(prompts, responses, ground_truths):
                return [self.reward_function(r, gt) for r, gt in zip(responses, ground_truths)]

            train_grpo(config=config_dict, reward_fn=reward_fn)

            return {"status": "success", "method": "easyr1", "config_path": str(config_path)}

        except ImportError:
            return self._train_fallback(config_path, resume_from)

    def _train_fallback(self, config_path: Path, resume_from: Optional[Path]) -> Dict[str, Any]:
        """Fallback training method."""
        try:
            from llamafactory.train.tuner import run_exp
            import yaml

            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)

            config_dict["stage"] = "ppo"
            if resume_from:
                config_dict["resume_from_checkpoint"] = str(resume_from)

            run_exp(config_dict)
            return {"status": "success", "method": "llamafactory_ppo"}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def predict(
        self,
        samples: List[FormattedSample],
        inference_config: InferenceConfig,
    ) -> List[Dict[str, Any]]:
        """Run inference with GRPO-trained model."""
        from vllm import LLM, SamplingParams

        model_name = inference_config.model_name_or_path or self.config.get_model_name()
        enable_lora = inference_config.adapter_path is not None

        llm = LLM(
            model=model_name,
            enable_lora=enable_lora,
            trust_remote_code=True,
            tensor_parallel_size=inference_config.tensor_parallel_size,
            gpu_memory_utilization=inference_config.gpu_memory_utilization,
        )

        sampling_params = SamplingParams(
            temperature=inference_config.temperature,
            max_tokens=inference_config.max_tokens,
        )

        prompts = [build_prompt(sample) for sample in samples]

        lora_request = None
        if enable_lora and inference_config.adapter_path:
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest("grpo_adapter", 1, inference_config.adapter_path)

        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request) if lora_request else llm.generate(prompts, sampling_params)

        results = []
        for sample, output in zip(samples, outputs):
            parsed = self.output_handler.parse_prediction(output.outputs[0].text)
            parsed["submission_id"] = sample.submission_id
            parsed["year"] = sample.year
            results.append(parsed)

        return results
