"""Custom reduce functions registry for embedding segment reduction.

This module provides a registry pattern for combiner functions used in
embedding segment reduction. It allows for easy extension with new
combiners without modifying core code.

Example:
    Register a custom combiner::

        from recis.nn.functional.custom_reduce_functions import register_combiner


        @register_combiner("my_custom")
        def my_custom_combiner(emb, weight, reverse_index, offsets, combiner_kwargs):
            # Custom reduction logic
            return result

"""

from typing import Callable, Dict, Optional

import torch

from recis.nn.functional.embedding_ops import ragged_embedding_segment_reduce
from recis.utils.logger import Logger


logger = Logger(__name__)


class ReduceFunctionRegistry:
    """Singleton registry for combiner functions used in embedding segment reduction.

    This class provides a centralized registry pattern with singleton behavior,
    ensuring only one instance of the registry exists throughout the application.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry: Dict[str, Callable] = {}
        return cls._instance

    def register(self, name: str, func: Callable) -> None:
        """Register a combiner function.

        Args:
            name: The name to register the combiner under.
            func: The combiner function to register.

        Raises:
            ValueError: If a combiner with the same name is already registered.
        """
        if name in self._registry:
            raise ValueError(f"Combiner '{name}' is already registered")
        self._registry[name] = func
        logger.info(f"Registered combiner '{name}'")

    def get(self, name: str) -> Callable:
        """Get a registered combiner function by name.

        Args:
            name: The name of the combiner to retrieve.

        Returns:
            The registered combiner function.

        Raises:
            ValueError: If the combiner is not found in the registry.
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(
                f"Combiner '{name}' not found. Available combiners: {available}"
            )
        return self._registry[name]

    def list_combiners(self) -> list:
        """List all registered combiner names.

        Returns:
            List of registered combiner names.
        """
        return list(self._registry.keys())


# Global singleton instance - created after class definition
_COMBINER_REGISTRY_INSTANCE = ReduceFunctionRegistry()


def register_combiner(name: str):
    """Decorator to register a combiner function.

    Args:
        name: The name to register the combiner under.

    Returns:
        Decorator function that registers the combiner.

    Raises:
        ValueError: If a combiner with the same name is already registered.

    Example:
        @register_combiner("sum")
        def combiner_sum(...):
            pass
    """

    def decorator(func: Callable) -> Callable:
        _COMBINER_REGISTRY_INSTANCE.register(name, func)
        return func

    return decorator


def get_combiner(name: str) -> Callable:
    """Get a registered combiner function by name.

    Args:
        name: The name of the combiner to retrieve.

    Returns:
        The registered combiner function.

    Raises:
        ValueError: If the combiner is not found in the registry.
    """
    return _COMBINER_REGISTRY_INSTANCE.get(name)


@register_combiner("sum")
def combiner_sum(
    emb: torch.Tensor = None,
    weight: torch.Tensor = None,
    reverse_index: torch.Tensor = None,
    offsets: torch.Tensor = None,
    combiner_kwargs: Optional[Dict] = None,
) -> torch.Tensor:
    """Sum combiner for embedding reduction.

    Performs element-wise sum within each segment.

    Args:
        emb: Input embedding tensor.
        weight: Optional weight tensor for weighted reduction.
        reverse_index: Index tensor for reordering.
        offsets: Offset tensor defining segment boundaries.
        combiner_kwargs: Additional arguments (unused for sum).

    Returns:
        Reduced embedding tensor with shape [num_segments, emb_dim].
    """
    return ragged_embedding_segment_reduce(
        emb, weight, reverse_index, offsets, "sum", combiner_kwargs
    )


@register_combiner("mean")
def combiner_mean(
    emb: torch.Tensor = None,
    weight: torch.Tensor = None,
    reverse_index: torch.Tensor = None,
    offsets: torch.Tensor = None,
    combiner_kwargs: Optional[Dict] = None,
) -> torch.Tensor:
    """Mean combiner for embedding reduction.

    Performs element-wise mean within each segment.

    Args:
        emb: Input embedding tensor.
        weight: Optional weight tensor for weighted reduction.
        reverse_index: Index tensor for reordering.
        offsets: Offset tensor defining segment boundaries.
        combiner_kwargs: Additional arguments (unused for mean).

    Returns:
        Reduced embedding tensor with shape [num_segments, emb_dim].
    """
    return ragged_embedding_segment_reduce(
        emb, weight, reverse_index, offsets, "mean", combiner_kwargs
    )


@register_combiner("tile")
def combiner_tile(
    emb: torch.Tensor = None,
    weight: torch.Tensor = None,
    reverse_index: torch.Tensor = None,
    offsets: torch.Tensor = None,
    combiner_kwargs: Optional[Dict] = None,
) -> torch.Tensor:
    """Tile combiner for embedding reduction.

    Tiles embeddings within each segment according to tile_len.

    Args:
        emb: Input embedding tensor.
        weight: Optional weight tensor for weighted reduction.
        reverse_index: Index tensor for reordering.
        offsets: Offset tensor defining segment boundaries.
        combiner_kwargs: Dictionary containing 'tile_len' and 'bs' parameters.

    Returns:
        Tiled embedding tensor.
    """
    return ragged_embedding_segment_reduce(
        emb, weight, reverse_index, offsets, "tile", combiner_kwargs
    )
