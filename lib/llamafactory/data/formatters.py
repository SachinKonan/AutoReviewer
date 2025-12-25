"""
Data formatters for LlamaFactory training formats.

This module provides formatters to convert FormattedSample objects to
LlamaFactory-compatible formats (Alpaca, ShareGPT) and generate the
required dataset_info.json configuration.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.types import DataFormat, FormattedSample


class BaseDataFormatter:
    """Base class for data formatters."""

    format_type: DataFormat = NotImplemented

    def format_samples(self, samples: List[FormattedSample]) -> List[Dict[str, Any]]:
        """
        Convert FormattedSamples to format-specific dicts.

        Args:
            samples: List of FormattedSample objects

        Returns:
            List of formatted dictionaries
        """
        raise NotImplementedError

    def generate_dataset_info(
        self,
        dataset_name: str,
        file_name: str,
        has_images: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate dataset_info.json entry for this format.

        Args:
            dataset_name: Name for the dataset
            file_name: Name of the data file
            has_images: Whether dataset includes images

        Returns:
            Dictionary for dataset_info.json
        """
        raise NotImplementedError

    def save_dataset(
        self,
        samples: List[FormattedSample],
        output_dir: Path,
        dataset_name: str = "train",
        split_name: str = "train",
    ) -> Dict[str, Any]:
        """
        Save formatted dataset to disk.

        Args:
            samples: List of FormattedSample objects
            output_dir: Directory to save files
            dataset_name: Name for the dataset
            split_name: Name for the split (train, eval, etc.)

        Returns:
            Dict with file paths and metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Format and save data
        formatted = self.format_samples(samples)
        data_file = output_dir / f"{split_name}.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(formatted, f, indent=2, ensure_ascii=False)

        # Check if any samples have images
        has_images = any(s.images for s in samples)

        # Generate and save dataset_info
        dataset_info = self.generate_dataset_info(
            dataset_name=dataset_name,
            file_name=data_file.name,
            has_images=has_images,
        )
        info_file = output_dir / "dataset_info.json"

        # Merge with existing dataset_info if present
        existing_info = {}
        if info_file.exists():
            with open(info_file, "r") as f:
                existing_info = json.load(f)
        existing_info.update(dataset_info)

        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(existing_info, f, indent=2, ensure_ascii=False)

        return {
            "data_file": str(data_file),
            "dataset_info_file": str(info_file),
            "dataset_name": dataset_name,
            "num_samples": len(samples),
            "has_images": has_images,
            "format": self.format_type.name,
        }


class AlpacaFormatter(BaseDataFormatter):
    """
    Format samples in Alpaca format for LlamaFactory.

    Alpaca format:
    {
        "instruction": "task instruction",
        "input": "input content",
        "output": "expected output",
        "images": ["path1.png", "path2.png"]  # optional
    }
    """

    format_type = DataFormat.ALPACA

    def format_samples(self, samples: List[FormattedSample]) -> List[Dict[str, Any]]:
        """Convert FormattedSamples to Alpaca format."""
        formatted = []
        for sample in samples:
            item = {
                "instruction": sample.instruction,
                "input": sample.input_text,
                "output": sample.output or "",
            }
            if sample.images:
                item["images"] = sample.images
            formatted.append(item)
        return formatted

    def generate_dataset_info(
        self,
        dataset_name: str,
        file_name: str,
        has_images: bool = False,
    ) -> Dict[str, Any]:
        """Generate dataset_info.json entry for Alpaca format."""
        info = {
            dataset_name: {
                "file_name": file_name,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                },
            }
        }
        if has_images:
            info[dataset_name]["columns"]["images"] = "images"
        return info


class ShareGPTFormatter(BaseDataFormatter):
    """
    Format samples in ShareGPT format for LlamaFactory.

    ShareGPT format:
    {
        "conversations": [
            {"from": "human", "value": "instruction + input"},
            {"from": "gpt", "value": "output"}
        ],
        "images": ["path1.png", "path2.png"]  # optional
    }
    """

    format_type = DataFormat.SHAREGPT

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        include_system: bool = False,
    ):
        """
        Initialize ShareGPT formatter.

        Args:
            system_prompt: Optional system message to include
            include_system: Whether to include system message
        """
        self.system_prompt = system_prompt
        self.include_system = include_system

    def format_samples(self, samples: List[FormattedSample]) -> List[Dict[str, Any]]:
        """Convert FormattedSamples to ShareGPT format."""
        formatted = []
        for sample in samples:
            conversations = []

            # Optional system message
            if self.include_system and self.system_prompt:
                conversations.append({
                    "from": "system",
                    "value": self.system_prompt,
                })

            # User message (instruction + input)
            user_content = sample.instruction
            if sample.input_text:
                user_content = f"{sample.instruction}\n\n{sample.input_text}"
            conversations.append({
                "from": "human",
                "value": user_content,
            })

            # Assistant response
            if sample.output:
                conversations.append({
                    "from": "gpt",
                    "value": sample.output,
                })

            item = {"conversations": conversations}
            if sample.images:
                item["images"] = sample.images
            formatted.append(item)
        return formatted

    def generate_dataset_info(
        self,
        dataset_name: str,
        file_name: str,
        has_images: bool = False,
    ) -> Dict[str, Any]:
        """Generate dataset_info.json entry for ShareGPT format."""
        info = {
            dataset_name: {
                "file_name": file_name,
                "formatting": "sharegpt",
                "columns": {
                    "messages": "conversations",
                },
                "tags": {
                    "role_tag": "from",
                    "content_tag": "value",
                    "user_tag": "human",
                    "assistant_tag": "gpt",
                },
            }
        }

        if self.include_system:
            info[dataset_name]["tags"]["system_tag"] = "system"

        if has_images:
            info[dataset_name]["columns"]["images"] = "images"

        return info


def get_formatter(format_type: DataFormat) -> BaseDataFormatter:
    """
    Factory function to get a formatter by type.

    Args:
        format_type: The data format to use

    Returns:
        A BaseDataFormatter instance
    """
    if format_type == DataFormat.ALPACA:
        return AlpacaFormatter()
    elif format_type == DataFormat.SHAREGPT:
        return ShareGPTFormatter()
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def save_train_eval_split(
    train_samples: List[FormattedSample],
    eval_samples: List[FormattedSample],
    output_dir: Path,
    format_type: DataFormat = DataFormat.SHAREGPT,
    dataset_name: str = "iclr",
) -> Dict[str, Any]:
    """
    Save training and evaluation datasets.

    Args:
        train_samples: Training samples
        eval_samples: Evaluation samples
        output_dir: Directory to save files
        format_type: Data format to use
        dataset_name: Base name for the dataset

    Returns:
        Dict with paths and metadata for both splits
    """
    formatter = get_formatter(format_type)
    output_dir = Path(output_dir)

    # Save training data
    train_result = formatter.save_dataset(
        samples=train_samples,
        output_dir=output_dir,
        dataset_name=f"{dataset_name}_train",
        split_name="train",
    )

    # Save evaluation data
    eval_result = formatter.save_dataset(
        samples=eval_samples,
        output_dir=output_dir,
        dataset_name=f"{dataset_name}_eval",
        split_name="eval",
    )

    return {
        "train": train_result,
        "eval": eval_result,
        "output_dir": str(output_dir),
    }
