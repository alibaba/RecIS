"""Compute real (valid) sequence length for ragged tensors.

This module provides functionality to compute the real sequence length
for ragged tensors by counting positions where both item and seller
features are non-zero.
"""

import torch

from recis.ragged.tensor import RaggedTensor
from recis.utils.logger import Logger


logger = Logger(__name__)


def compute_real_length(
    item_ragged: RaggedTensor,
    seller_ragged: RaggedTensor,
) -> torch.Tensor:
    """Compute real (valid) sequence length for ragged tensors.

    For each batch sample, counts the number of sequence positions where
    both item and seller features are non-zero (valid). This is useful
    for determining the actual usable length of sequences after masking
    out invalid entries.

    Args:
        item_ragged (RaggedTensor): Item feature ragged tensor.
            - For numeric types: 3D RaggedTensor with shape [batch, seq, value]
            - For string types: 4D RaggedTensor (int8) with shape [batch, seq, string, char]
        seller_ragged (RaggedTensor): Seller feature ragged tensor.
            Must have the same structure as item_ragged.

    Returns:
        torch.Tensor: Real lengths tensor of shape [batch_size], where each
            element represents the count of valid sequence positions for
            that batch sample.

    Raises:
        ValueError: If item_ragged and seller_ragged have different batch sizes.
        ValueError: If the ragged tensors have unsupported dimensions.

    Example:
        >>> from recis.ragged.tensor import RaggedTensor
        >>> from recis.nn.functional.compute_real_length import compute_real_length
        >>> # Create 3D ragged tensors for item and seller
        >>> item_values = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
        >>> item_offsets = [
        ...     torch.tensor([0, 2, 4]),  # batch offsets: 2 samples
        ...     torch.tensor([0, 1, 3, 4, 6]),  # sequence offsets: 4 positions
        ... ]
        >>> item_ragged = RaggedTensor(item_values, item_offsets)
        >>> seller_values = torch.tensor([10, 0, 30, 40, 0, 60], dtype=torch.int64)
        >>> seller_offsets = [torch.tensor([0, 2, 4]), torch.tensor([0, 1, 3, 4, 6])]
        >>> seller_ragged = RaggedTensor(seller_values, seller_offsets)
        >>> real_lengths = compute_real_length(item_ragged, seller_ragged)
        >>> print(real_lengths)  # tensor([1., 2.])

    Note:
        - For numeric types (3D tensor): A position is valid if both item and
          seller values at that position are non-zero.
        - For string types (4D tensor, stored as int8): A position is valid if
          both item and seller strings at that position are non-empty and
          not equal to "0".
        - Both ragged tensors must have the same batch size and compatible
          sequence structures.
    """
    # Validate inputs
    if not isinstance(item_ragged, RaggedTensor):
        raise TypeError(f"item_ragged must be RaggedTensor, got {type(item_ragged)}")
    if not isinstance(seller_ragged, RaggedTensor):
        raise TypeError(
            f"seller_ragged must be RaggedTensor, got {type(seller_ragged)}"
        )

    # Get offsets
    item_offsets = item_ragged.offsets()
    seller_offsets = seller_ragged.offsets()

    # Determine number of ragged dimensions
    num_ragged_dim = len(item_offsets)

    # Validate dimensions
    if num_ragged_dim not in [2, 3]:
        raise ValueError(
            f"Unsupported ragged dimension: {num_ragged_dim}. "
            "Only 3D (num_ragged_dim=2) and 4D (num_ragged_dim=3) "
            "ragged tensors are supported."
        )

    # Validate batch sizes match
    item_batch_size = item_offsets[0].size(0) - 1
    seller_batch_size = seller_offsets[0].size(0) - 1
    if item_batch_size != seller_batch_size:
        raise ValueError(
            f"Batch sizes do not match: item has {item_batch_size}, "
            f"seller has {seller_batch_size}"
        )

    # Call the C++/CUDA operator
    real_lengths = torch.ops.recis.compute_real_length(
        item_ragged.values(),
        item_offsets,
        seller_ragged.values(),
        seller_offsets,
        num_ragged_dim,
    )

    return real_lengths
