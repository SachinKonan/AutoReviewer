"""
YAML configuration generator for LlamaFactory training.

Provides utilities to generate training configurations for different
model types and training modes.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..core.types import ModelType, TrainingConfig, TrainingMode, TEMPLATE_MAP, MODEL_NAME_MAP


class ConfigGenerator:
    """Generate LlamaFactory YAML configurations."""

    # Default LoRA target modules by model family
    LORA_TARGETS = {
        "qwen": "all",
        "qwen2_vl": "all",
        "llama": ["q_proj", "v_proj", "k_proj", "o_proj"],
    }

    def __init__(self, base_config: TrainingConfig):
        """
        Initialize config generator.

        Args:
            base_config: Base training configuration
        """
        self.base_config = base_config

    def _get_base_config(self) -> Dict[str, Any]:
        """Get base configuration common to all training modes."""
        return {
            "model_name_or_path": self.base_config.get_model_name(),
            "template": self.base_config.get_template(),
            "cutoff_len": self.base_config.max_length,
            "preprocessing_num_workers": 16,
            "per_device_train_batch_size": self.base_config.batch_size,
            "gradient_accumulation_steps": self.base_config.gradient_accumulation_steps,
            "learning_rate": self.base_config.learning_rate,
            "num_train_epochs": self.base_config.num_epochs,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": self.base_config.warmup_ratio,
            "bf16": self.base_config.bf16,
            "gradient_checkpointing": self.base_config.gradient_checkpointing,
            "logging_steps": self.base_config.logging_steps,
            "save_steps": self.base_config.save_steps,
            "save_total_limit": 3,
            "overwrite_output_dir": True,
        }

    def _get_lora_config(self) -> Dict[str, Any]:
        """Get LoRA-specific configuration."""
        template = self.base_config.get_template()
        lora_target = self.LORA_TARGETS.get(template, "all")

        return {
            "finetuning_type": "lora",
            "lora_rank": self.base_config.lora_rank,
            "lora_alpha": self.base_config.lora_alpha,
            "lora_dropout": self.base_config.lora_dropout,
            "lora_target": lora_target,
        }

    def generate_sft_config(
        self,
        dataset_name: str,
        dataset_dir: Path,
        output_dir: Path,
        has_images: bool = False,
        eval_dataset: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate SFT training configuration.

        Args:
            dataset_name: Name of the dataset
            dataset_dir: Directory containing dataset
            output_dir: Output directory for checkpoints
            has_images: Whether dataset has images
            eval_dataset: Optional evaluation dataset name

        Returns:
            Complete SFT config dict
        """
        config = self._get_base_config()
        config.update(self._get_lora_config())

        config.update({
            "stage": "sft",
            "do_train": True,
            "dataset": dataset_name,
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "plot_loss": True,
        })

        # Evaluation settings
        if eval_dataset or self.base_config.val_size > 0:
            config.update({
                "val_size": self.base_config.val_size,
                "per_device_eval_batch_size": self.base_config.batch_size,
                "eval_strategy": "steps",
                "eval_steps": self.base_config.save_steps,
            })
            if eval_dataset:
                config["eval_dataset"] = eval_dataset

        # Vision-language settings
        if has_images:
            config["visual_inputs"] = True

        return config

    def generate_grpo_config(
        self,
        dataset_name: str,
        dataset_dir: Path,
        output_dir: Path,
        has_images: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate GRPO training configuration.

        Args:
            dataset_name: Name of the dataset
            dataset_dir: Directory containing dataset
            output_dir: Output directory for checkpoints
            has_images: Whether dataset has images

        Returns:
            Complete GRPO config dict
        """
        config = self._get_base_config()
        config.update(self._get_lora_config())

        config.update({
            "stage": "grpo",
            "do_train": True,
            "dataset": dataset_name,
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "grpo_group_size": self.base_config.grpo_group_size,
            "kl_coeff": self.base_config.grpo_kl_coeff,
        })

        if has_images:
            config["visual_inputs"] = True

        return config

    def generate_inference_config(
        self,
        adapter_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate inference configuration.

        Args:
            adapter_path: Optional path to LoRA adapter

        Returns:
            Inference config dict
        """
        config = {
            "model_name_or_path": self.base_config.get_model_name(),
            "template": self.base_config.get_template(),
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 512,
        }

        if adapter_path:
            config["adapter_name_or_path"] = str(adapter_path)
            config["finetuning_type"] = "lora"

        return config

    def save_config(
        self,
        config: Dict[str, Any],
        output_path: Path,
    ) -> Path:
        """
        Save configuration to YAML file.

        Args:
            config: Configuration dict
            output_path: Path to save file

        Returns:
            Path to saved config file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return output_path


def generate_dataset_info(
    datasets: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> Path:
    """
    Generate dataset_info.json file.

    Args:
        datasets: Dict mapping dataset names to their configurations
        output_path: Path to save file

    Returns:
        Path to saved file
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(datasets, f, indent=2)

    return output_path


def create_experiment_configs(
    base_config: TrainingConfig,
    experiment_name: str,
    dataset_dir: Path,
    output_base: Path,
    training_modes: list = None,
    has_images: bool = False,
) -> Dict[str, Path]:
    """
    Create configuration files for multiple training modes.

    Args:
        base_config: Base training configuration
        experiment_name: Name for the experiment
        dataset_dir: Directory containing prepared data
        output_base: Base output directory
        training_modes: List of TrainingMode enums to generate configs for
        has_images: Whether data includes images

    Returns:
        Dict mapping mode names to config file paths
    """
    if training_modes is None:
        training_modes = [TrainingMode.SFT_LORA]

    generator = ConfigGenerator(base_config)
    configs = {}

    for mode in training_modes:
        mode_name = mode.name.lower()
        output_dir = output_base / experiment_name / mode_name

        if mode == TrainingMode.SFT_LORA:
            config = generator.generate_sft_config(
                dataset_name=f"{experiment_name}_train",
                dataset_dir=dataset_dir,
                output_dir=output_dir / "checkpoints",
                has_images=has_images,
            )
        elif mode == TrainingMode.GRPO:
            config = generator.generate_grpo_config(
                dataset_name=f"{experiment_name}_train",
                dataset_dir=dataset_dir,
                output_dir=output_dir / "checkpoints",
                has_images=has_images,
            )
        else:
            continue

        config_path = generator.save_config(config, output_dir / f"{mode_name}_config.yaml")
        configs[mode_name] = config_path

    return configs
