"""
Supervised Fine-Tuning (SFT) with LoRA mode.

Trains the model using supervised learning with LoRA adapters
via the LlamaFactory Python API.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.registry import register_training_mode
from ..core.types import (
    DataFormat,
    FormattedSample,
    InferenceConfig,
    TrainingConfig,
    TrainingMode,
)
from ..data.formatters import ShareGPTFormatter, AlpacaFormatter
from .base import TrainingModeBase, build_prompt


@register_training_mode(TrainingMode.SFT_LORA)
class SFTLoRAMode(TrainingModeBase):
    """
    Supervised Fine-Tuning with LoRA adapters.

    Uses LlamaFactory's Python API for training and supports
    both Alpaca and ShareGPT data formats.
    """

    mode = TrainingMode.SFT_LORA
    requires_training = True

    def __init__(
        self,
        config: TrainingConfig,
        input_formatter: "InputFormatter",
        output_handler: "OutputHandler",
        data_format: DataFormat = DataFormat.SHAREGPT,
    ):
        """
        Initialize SFT mode.

        Args:
            config: Training configuration
            input_formatter: Input formatting strategy
            output_handler: Output handling strategy
            data_format: Data format for training (Alpaca or ShareGPT)
        """
        super().__init__(config, input_formatter, output_handler)
        self.data_format = data_format

        # Select formatter
        if data_format == DataFormat.ALPACA:
            self.data_formatter = AlpacaFormatter()
        else:
            self.data_formatter = ShareGPTFormatter()

    def prepare_data(
        self,
        samples: List[FormattedSample],
        output_path: Path,
    ) -> Dict[str, Any]:
        """
        Prepare data in LlamaFactory format.

        Args:
            samples: Formatted training samples
            output_path: Directory to save data files

        Returns:
            Dict with data paths and metadata
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Format and save training data
        formatted_data = self.data_formatter.format_samples(samples)

        data_file = output_path / "train.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)

        # Check if we have images
        has_images = any(s.images for s in samples)

        # Generate dataset_info.json
        dataset_info = self.data_formatter.generate_dataset_info(
            dataset_name="iclr_train",
            file_name="train.json",
            has_images=has_images,
        )

        info_file = output_path / "dataset_info.json"
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)

        return {
            "data_file": str(data_file),
            "dataset_info_file": str(info_file),
            "dataset_dir": str(output_path),
            "dataset_name": "iclr_train",
            "num_samples": len(samples),
            "has_images": has_images,
            "format": self.data_format.name,
        }

    def generate_config(
        self,
        data_info: Dict[str, Any],
        output_path: Path,
    ) -> Path:
        """
        Generate LlamaFactory training configuration.

        Args:
            data_info: Output from prepare_data
            output_path: Directory to save config

        Returns:
            Path to generated YAML config file
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build config dict
        config = {
            # Model
            "model_name_or_path": self.config.get_model_name(),
            "template": self.config.get_template(),

            # Training method
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "lora",

            # LoRA settings
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "lora_target": "all",

            # Dataset
            "dataset": data_info["dataset_name"],
            "dataset_dir": data_info["dataset_dir"],
            "cutoff_len": self.config.max_length,
            "preprocessing_num_workers": 16,
            "overwrite_cache": True,

            # Training hyperparameters
            "per_device_train_batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "num_train_epochs": self.config.num_epochs,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": self.config.warmup_ratio,

            # Hardware optimization
            "bf16": self.config.bf16,
            "gradient_checkpointing": self.config.gradient_checkpointing,

            # Evaluation
            "val_size": self.config.val_size,
            "per_device_eval_batch_size": self.config.batch_size,
            "eval_strategy": "steps",
            "eval_steps": self.config.save_steps,

            # Output
            "output_dir": str(output_path / "checkpoints"),
            "logging_steps": self.config.logging_steps,
            "save_steps": self.config.save_steps,
            "save_total_limit": 3,
            "plot_loss": True,
            "overwrite_output_dir": True,
        }

        # Handle VL model settings
        if data_info.get("has_images", False):
            config["visual_inputs"] = True

        # Save config as YAML
        import yaml
        config_path = output_path / "sft_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return config_path

    def train(
        self,
        config_path: Optional[Path] = None,
        resume_from: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Execute SFT training using LlamaFactory Python API.

        Args:
            config_path: Path to training config YAML
            resume_from: Optional checkpoint to resume from

        Returns:
            Training results dict
        """
        if config_path is None:
            raise ValueError("config_path is required for SFT training")

        try:
            # Import LlamaFactory
            from llamafactory.train.tuner import run_exp

            # Load config
            import yaml
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)

            # Add resume checkpoint if provided
            if resume_from:
                config_dict["resume_from_checkpoint"] = str(resume_from)

            # Run training
            run_exp(config_dict)

            return {
                "status": "success",
                "config_path": str(config_path),
                "output_dir": config_dict.get("output_dir"),
            }

        except ImportError:
            # Fallback to CLI if Python API not available
            return self._train_cli(config_path, resume_from)

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "config_path": str(config_path),
            }

    def _train_cli(
        self,
        config_path: Path,
        resume_from: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Fallback to CLI-based training."""
        import subprocess

        cmd = ["llamafactory-cli", "train", str(config_path)]
        if resume_from:
            cmd.extend(["--resume_from_checkpoint", str(resume_from)])

        result = subprocess.run(cmd, capture_output=True, text=True)

        return {
            "status": "success" if result.returncode == 0 else "error",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "config_path": str(config_path),
        }

    def predict(
        self,
        samples: List[FormattedSample],
        inference_config: InferenceConfig,
    ) -> List[Dict[str, Any]]:
        """
        Run inference with trained LoRA adapter.

        Args:
            samples: Formatted samples for prediction
            inference_config: Inference settings (including adapter_path)

        Returns:
            List of parsed predictions
        """
        from vllm import LLM, SamplingParams

        model_name = inference_config.model_name_or_path or self.config.get_model_name()

        # Check if we have an adapter
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

        # Build prompts
        prompts = [build_prompt(sample) for sample in samples]

        # Setup LoRA request if using adapter
        lora_request = None
        if enable_lora and inference_config.adapter_path:
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest(
                "iclr_adapter",
                1,
                inference_config.adapter_path,
            )

        # Run inference
        if lora_request:
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        else:
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

    def export_adapter(
        self,
        checkpoint_dir: Path,
        output_path: Path,
        merge_weights: bool = False,
    ) -> Path:
        """
        Export trained LoRA adapter.

        Args:
            checkpoint_dir: Directory with training checkpoints
            output_path: Where to save the exported adapter
            merge_weights: Whether to merge LoRA weights into base model

        Returns:
            Path to exported adapter/model
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if merge_weights:
            # Merge LoRA weights into base model
            try:
                from llamafactory.train.tuner import export_model

                export_model({
                    "model_name_or_path": self.config.get_model_name(),
                    "adapter_name_or_path": str(checkpoint_dir),
                    "template": self.config.get_template(),
                    "finetuning_type": "lora",
                    "export_dir": str(output_path),
                    "export_size": 2,
                    "export_legacy_format": False,
                })
            except ImportError:
                # Copy adapter files directly
                import shutil
                shutil.copytree(checkpoint_dir, output_path, dirs_exist_ok=True)
        else:
            # Just copy the adapter
            import shutil
            shutil.copytree(checkpoint_dir, output_path, dirs_exist_ok=True)

        return output_path
