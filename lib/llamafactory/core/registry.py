"""
Component registry for extensible input formatters, output handlers, and training modes.

This module provides a registration system that allows new components to be added
without modifying existing code. Components are registered by their enum type
and can be retrieved via factory functions.
"""

from typing import TYPE_CHECKING, Dict, Type, TypeVar, Callable, Any

from .types import InputModality, OutputType, TrainingMode

if TYPE_CHECKING:
    from ..inputs.base import InputFormatter
    from ..outputs.base import OutputHandler
    from ..training.base import TrainingModeBase

# Type variables for generic registry
T = TypeVar("T")

# Component registries
_INPUT_FORMATTER_REGISTRY: Dict[InputModality, Type["InputFormatter"]] = {}
_OUTPUT_HANDLER_REGISTRY: Dict[OutputType, Type["OutputHandler"]] = {}
_TRAINING_MODE_REGISTRY: Dict[TrainingMode, Type["TrainingModeBase"]] = {}


def register_input_formatter(modality: InputModality) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register an input formatter class.

    Usage:
        @register_input_formatter(InputModality.TEXT_ONLY)
        class TextOnlyFormatter(InputFormatter):
            ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        _INPUT_FORMATTER_REGISTRY[modality] = cls
        return cls
    return decorator


def register_output_handler(output_type: OutputType) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register an output handler class.

    Usage:
        @register_output_handler(OutputType.BINARY)
        class BinaryAcceptRejectHandler(OutputHandler):
            ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        _OUTPUT_HANDLER_REGISTRY[output_type] = cls
        return cls
    return decorator


def register_training_mode(mode: TrainingMode) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a training mode class.

    Usage:
        @register_training_mode(TrainingMode.SFT_LORA)
        class SFTLoRAMode(TrainingModeBase):
            ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        _TRAINING_MODE_REGISTRY[mode] = cls
        return cls
    return decorator


def get_input_formatter(modality: InputModality) -> Type["InputFormatter"]:
    """
    Get input formatter class by modality.

    Args:
        modality: The input modality type

    Returns:
        The registered InputFormatter class

    Raises:
        KeyError: If no formatter is registered for the modality
    """
    if modality not in _INPUT_FORMATTER_REGISTRY:
        available = list(_INPUT_FORMATTER_REGISTRY.keys())
        raise KeyError(
            f"No input formatter registered for {modality}. "
            f"Available: {available}"
        )
    return _INPUT_FORMATTER_REGISTRY[modality]


def get_output_handler(output_type: OutputType) -> Type["OutputHandler"]:
    """
    Get output handler class by output type.

    Args:
        output_type: The output type

    Returns:
        The registered OutputHandler class

    Raises:
        KeyError: If no handler is registered for the output type
    """
    if output_type not in _OUTPUT_HANDLER_REGISTRY:
        available = list(_OUTPUT_HANDLER_REGISTRY.keys())
        raise KeyError(
            f"No output handler registered for {output_type}. "
            f"Available: {available}"
        )
    return _OUTPUT_HANDLER_REGISTRY[output_type]


def get_training_mode(mode: TrainingMode) -> Type["TrainingModeBase"]:
    """
    Get training mode class by mode type.

    Args:
        mode: The training mode type

    Returns:
        The registered TrainingModeBase class

    Raises:
        KeyError: If no training mode is registered for the mode
    """
    if mode not in _TRAINING_MODE_REGISTRY:
        available = list(_TRAINING_MODE_REGISTRY.keys())
        raise KeyError(
            f"No training mode registered for {mode}. "
            f"Available: {available}"
        )
    return _TRAINING_MODE_REGISTRY[mode]


def list_registered_components() -> Dict[str, list]:
    """
    List all registered components.

    Returns:
        Dict with keys 'input_formatters', 'output_handlers', 'training_modes'
        containing lists of registered enum values.
    """
    return {
        "input_formatters": list(_INPUT_FORMATTER_REGISTRY.keys()),
        "output_handlers": list(_OUTPUT_HANDLER_REGISTRY.keys()),
        "training_modes": list(_TRAINING_MODE_REGISTRY.keys()),
    }


def create_input_formatter(
    modality: InputModality,
    **kwargs: Any,
) -> "InputFormatter":
    """
    Factory function to create an input formatter instance.

    Args:
        modality: The input modality type
        **kwargs: Arguments to pass to the formatter constructor

    Returns:
        An InputFormatter instance
    """
    formatter_cls = get_input_formatter(modality)
    return formatter_cls(**kwargs)


def create_output_handler(
    output_type: OutputType,
    **kwargs: Any,
) -> "OutputHandler":
    """
    Factory function to create an output handler instance.

    Args:
        output_type: The output type
        **kwargs: Arguments to pass to the handler constructor

    Returns:
        An OutputHandler instance
    """
    handler_cls = get_output_handler(output_type)
    return handler_cls(**kwargs)


def create_training_mode(
    mode: TrainingMode,
    **kwargs: Any,
) -> "TrainingModeBase":
    """
    Factory function to create a training mode instance.

    Args:
        mode: The training mode type
        **kwargs: Arguments to pass to the training mode constructor

    Returns:
        A TrainingModeBase instance
    """
    mode_cls = get_training_mode(mode)
    return mode_cls(**kwargs)
