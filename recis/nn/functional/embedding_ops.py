from typing import List, Optional, Union

import torch


__ALL__ = [
    "ragged_embedding_segment_reduce",
    "sparse_embedding_segment_reduce",
    "merge_offsets",
    "ids_encode",
]

from recis.utils.logger import Logger


logger = Logger(__name__)


def segment_sum_sparse(
    data, weight, indices, segment_ids, num_segments, backend="recis"
):
    """Perform segment-wise sum reduction on sparse data.

    This function computes the sum of data elements within each segment,
    optionally weighted, using either RecIS custom operations or PyTorch
    native operations for flexibility and performance comparison.

    Args:
        data (torch.Tensor): Input data tensor of shape (N, D) where N is
            the number of data points and D is the feature dimension.
        weight (torch.Tensor or None): Optional weight tensor of shape (M,)
            where M is the number of indices. If None, uniform weights are used.
        indices (torch.Tensor): Index tensor of shape (M,) mapping to data elements.
        segment_ids (torch.Tensor): Segment ID tensor of shape (M,) indicating
            which segment each element belongs to.
        num_segments (int): Total number of segments in the output.
        backend (str, optional): Backend to use for computation. Options are
            "recis" (custom CUDA kernels) or "torch" (PyTorch native).
            Defaults to "recis".

    Returns:
        torch.Tensor: Segment-wise sum tensor of shape (num_segments, D).

    Raises:
        RuntimeError: If backend is not "recis" or "torch".

    Example:
        >>> data = torch.randn(10, 5)
        >>> weight = torch.ones(6)
        >>> indices = torch.tensor([0, 2, 4, 6, 8, 9])
        >>> segment_ids = torch.tensor([0, 0, 1, 1, 2, 2])
        >>> result = segment_sum_sparse(data, weight, indices, segment_ids, 3)
        >>> print(result.shape)  # torch.Size([3, 5])
    """
    if backend == "recis":
        out = torch.ops.recis.segment_sum(
            data, weight, indices, segment_ids, num_segments
        )
    elif backend == "torch":
        src_tensor = data[indices]
        if weight is not None:
            src_tensor = src_tensor * torch.unsqueeze(weight, dim=-1)
        out = torch.zeros(
            [num_segments, data.shape[-1]], dtype=src_tensor.dtype, device=data.device
        )
        out.index_add_(0, segment_ids, src_tensor)
    else:
        raise RuntimeError(
            f"Segment sum sparse only support ['recis'|'torch'] backend, got: {backend}"
        )
    return out


def weight_norm_sparse(data, weight, segment_ids, num_segments, backend="recis"):
    """Compute normalized weights for sparse segment operations.

    This function calculates normalized weights for each element based on
    the segment it belongs to, enabling proper mean computation in segment
    reduction operations.

    Args:
        data (torch.Tensor): Input data tensor for reference (used for device/dtype).
        weight (torch.Tensor or None): Original weight tensor. If None,
            uniform weights are assumed.
        segment_ids (torch.Tensor): Segment ID tensor indicating which segment
            each element belongs to.
        num_segments (int): Total number of segments.
        backend (str, optional): Backend to use for computation. Options are
            "recis" or "torch". Defaults to "recis".

    Returns:
        torch.Tensor: Normalized weight tensor where each weight is divided
            by the total count of elements in its segment.

    Raises:
        RuntimeError: If backend is not "recis" or "torch".

    Example:
        >>> data = torch.randn(10, 5)
        >>> weight = torch.ones(6)
        >>> segment_ids = torch.tensor([0, 0, 1, 1, 2, 2])
        >>> norm_weight = weight_norm_sparse(data, weight, segment_ids, 3)
        >>> print(norm_weight)  # Each weight divided by segment count
    """
    if backend == "recis":
        weight_norm = torch.ops.recis.segment_mean(
            data, weight, segment_ids, num_segments
        )
    elif backend == "torch":
        if weight is None:
            weight = torch.ones(
                [segment_ids.numel()], dtype=data.dtype, device=data.device
            )
        weight_count = torch.zeros(
            [num_segments], dtype=weight.dtype, device=weight.device
        )
        weight_count.index_add_(
            0, segment_ids, torch.ones_like(segment_ids, dtype=weight.dtype)
        )
        weight_mask = (weight_count <= 0).to(
            device=weight.device, dtype=weight_count.dtype
        )
        weight_count = weight_count + weight_mask
        norm = weight_count[segment_ids]
        weight_norm = weight / norm
    else:
        raise RuntimeError(
            f"Weight norm sparse only support ['recis'|'torch'] backend, got: {backend}"
        )
    return weight_norm


class EmbeddingSegmentReduceSparse(torch.autograd.Function):
    """PyTorch autograd Function for sparse embedding segment reduction.

    This class implements forward and backward passes for segment-based
    reduction operations on sparse embeddings, supporting different combiners
    and automatic gradient computation for training.

    The function is designed to handle sparse embedding lookups followed by
    segment-wise aggregation, which is common in recommendation systems for
    processing variable-length feature sequences.

    Example:
        >>> unique_emb = torch.randn(100, 64, requires_grad=True)
        >>> weight = torch.ones(200)
        >>> reverse_indices = torch.randint(0, 100, (200,))
        >>> segment_ids = torch.randint(0, 10, (200,))
        >>> result = EmbeddingSegmentReduceSparse.apply(
        ...     unique_emb, weight, reverse_indices, segment_ids, 10, "mean", "recis"
        ... )
    """

    @staticmethod
    def forward(
        ctx,
        unique_emb: torch.Tensor,
        weight: torch.Tensor,
        reverse_indices: torch.Tensor,
        segment_ids: torch.Tensor,
        num_segments: torch.int64,
        combiner: str,
        backend: str,
    ):
        """Forward pass for sparse embedding segment reduction.

        Args:
            ctx: PyTorch autograd context for saving tensors.
            unique_emb (torch.Tensor): Unique embedding tensor of shape (U, D).
            weight (torch.Tensor): Weight tensor of shape (N,).
            reverse_indices (torch.Tensor): Index mapping tensor of shape (N,).
            segment_ids (torch.Tensor): Segment ID tensor of shape (N,).
            num_segments (int): Number of output segments.
            combiner (str): Reduction combiner ("sum" or "mean").
            backend (str): Backend to use ("recis", "torch", or "auto").

        Returns:
            torch.Tensor: Reduced embedding tensor of shape (num_segments, D).
        """
        if backend == "auto":
            fwd_backend = "recis"
            bwd_backend = "torch"
        else:
            fwd_backend = backend
            bwd_backend = backend
        ctx.backend = bwd_backend
        ctx.unique_size = unique_emb.size(0)
        if combiner == "mean":
            weight = weight_norm_sparse(
                unique_emb, weight, segment_ids, num_segments, backend=fwd_backend
            )
        ctx.save_for_backward(weight, reverse_indices, segment_ids)
        emb = segment_sum_sparse(
            unique_emb,
            weight,
            reverse_indices,
            segment_ids,
            num_segments,
            backend=fwd_backend,
        )
        return emb

    def backward(ctx, grad):
        """Backward pass for sparse embedding segment reduction.

        Args:
            ctx: PyTorch autograd context with saved tensors.
            grad (torch.Tensor): Gradient tensor from upstream.

        Returns:
            tuple: Gradients for input tensors. Only unique_emb gets gradients,
                others are None.
        """
        weight, reverse_indices, segment_ids = ctx.saved_tensors
        unique_size = ctx.unique_size
        backend = ctx.backend
        unique_emb_grad = segment_sum_sparse(
            grad, weight, segment_ids, reverse_indices, unique_size, backend=backend
        )
        return unique_emb_grad, None, None, None, None, None, None


class EmbeddingSegmentReduceRagged(torch.autograd.Function):
    """PyTorch autograd Function for ragged embedding segment reduction.

    This class supports forward and backward computation for embedding reduction
    on ragged tensors using different combiners like 'sum', 'mean', and 'tile'.
    It handles custom CUDA kernels internally for efficient computation.

    Ragged tensors are useful for handling variable-length sequences in
    recommendation systems, such as user interaction histories or item
    feature lists of different lengths.

    Example:
        >>> unique_emb = torch.randn(10, 5, requires_grad=True)
        >>> weight = torch.randn(20)
        >>> reverse_indices = torch.randint(0, 10, (20,))
        >>> offsets = torch.tensor([0, 5, 10, 15, 20])
        >>> result = EmbeddingSegmentReduceRagged.apply(
        ...     unique_emb, weight, reverse_indices, offsets, "sum", {}
        ... )
    """

    @staticmethod
    def forward(
        ctx,
        unique_emb: torch.Tensor,
        weight: torch.Tensor,
        reverse_indices: torch.Tensor,
        offsets: torch.Tensor,
        combiner: str,
        combiner_kwargs: Optional[dict] = None,
    ):
        """Forward pass for ragged embedding segment reduction.

        Args:
            ctx: PyTorch autograd context for saving values.
            unique_emb (torch.Tensor): Unique embedding tensor of shape (U, D)
                where U is the number of unique embeddings and D is the dimensionality.
            weight (torch.Tensor or None): Optional weight tensor for weighted reductions.
            reverse_indices (torch.Tensor): Mapping from input indices to unique embeddings.
            offsets (torch.Tensor): Offsets defining segments in the input data.
            combiner (str): Reduction operation combiner. Supported values are
                "sum", "mean", and "tile".
            combiner_kwargs (dict): Additional arguments for specific combiners.

        Returns:
            torch.Tensor: Resulting reduced embeddings based on the specified combiner.
        """
        ctx.combiner = combiner
        ctx.combiner_kwargs = combiner_kwargs
        ctx.unique_size = unique_emb.size(0)
        if combiner == "tile":
            if combiner_kwargs is None:
                combiner_kwargs = {}
            left_pad = torch.concat(combiner_kwargs["left_pad"], dim=0).cuda(
                non_blocking=True
            )
            emb, batch_tile_len = torch.ops.recis.ragged_tile(
                combiner_kwargs["bs"],
                combiner_kwargs["tile_len"],
                reverse_indices,
                offsets,
                unique_emb,
                left_pad,
            )
            ctx.save_for_backward(
                weight, reverse_indices, offsets, batch_tile_len, left_pad
            )
            batch_info = (
                unique_emb.shape[0],
                len(combiner_kwargs["bs"]),
                max(combiner_kwargs["bs"]),
                min(combiner_kwargs["tile_len"]),
            )
            ctx.batch_info = batch_info
        else:
            emb = torch.ops.recis.segment_reduce_forward(
                unique_emb, weight, reverse_indices, offsets, combiner
            )
            ctx.save_for_backward(weight, reverse_indices, offsets)
        return emb

    def backward(ctx, grad):
        """Backward pass for ragged embedding segment reduction.

        Args:
            ctx: PyTorch autograd context with saved tensors.
            grad (torch.Tensor): Gradient tensor from upstream.

        Returns:
            tuple: Gradients for input tensors. Only unique_emb gets gradients,
                others are None.

        Raises:
            NotImplementedError: If combiner is "tile" (not yet supported).
        """
        unique_size = ctx.unique_size
        if ctx.combiner == "tile":
            weight, reverse_indices, offsets, batch_tile_len, left_pad = (
                ctx.saved_tensors
            )
            batch_info = ctx.batch_info
            unique_emb_grad = torch.ops.recis.ragged_tile_back(
                batch_tile_len, batch_info, reverse_indices, offsets, grad, left_pad
            )
        else:
            weight, reverse_indices, offsets = ctx.saved_tensors
            unique_emb_grad = torch.ops.recis.segment_reduce_backward(
                grad.clone(),
                weight,
                reverse_indices,
                offsets,
                unique_size,
                ctx.combiner,
            )
        return unique_emb_grad, None, None, None, None, None


def ragged_embedding_segment_reduce(
    unique_emb: torch.Tensor,
    weight: torch.Tensor,
    reverse_indices: torch.Tensor,
    offsets: torch.Tensor,
    combiner: str,
    combiner_kwargs: Optional[dict] = None,
):
    """Perform segment reduction on ragged embeddings with provided weights.

    This function performs efficient segment-based reduction operations on
    ragged embeddings, which are commonly used in recommendation systems
    for aggregating variable-length sequences like user interaction histories.

    The function automatically selects between sparse and ragged processing
    modes based on the data characteristics for optimal performance.

    Args:
        unique_emb (torch.Tensor): Unique embeddings tensor of shape (U, D)
            where U is the number of unique embeddings and D is the embedding dimension.
        weight (torch.Tensor): Weights tensor of shape (N,) where N is the
            number of input indices. Used for weighted aggregation.
        reverse_indices (torch.Tensor): Index mapping tensor of shape (N,)
            that maps input positions to unique embedding indices.
        offsets (torch.Tensor): Segment offset tensor of shape (S+1,) where
            S is the number of segments. Defines segment boundaries.
        combiner (str): Reduction combiner. Supported options:
            - "sum": Element-wise sum within each segment
            - "mean": Element-wise mean within each segment
            - "tile": Tile embeddings (experimental)
        combiner_kwargs (dict, optional): Additional arguments for specific
            combiners. Defaults to {}.

    Returns:
        torch.Tensor: Reduced embeddings tensor of shape (S, D) where S is
            the number of segments.

    Raises:
        AssertionError: If combiner is not one of "sum", "mean", or "tile".

    Example:
        >>> # Create sample data
        >>> unique_emb = torch.randn(100, 64, requires_grad=True)
        >>> weight = torch.ones(500)
        >>> reverse_indices = torch.randint(0, 100, (500,))
        >>> offsets = torch.tensor([0, 50, 150, 300, 500])  # 4 segments
        >>> # Perform mean reduction
        >>> result = ragged_embedding_segment_reduce(
        ...     unique_emb, weight, reverse_indices, offsets, "mean"
        ... )
        >>> print(result.shape)  # torch.Size([4, 64])
        >>> # Perform sum reduction
        >>> result_sum = ragged_embedding_segment_reduce(
        ...     unique_emb, weight, reverse_indices, offsets, "sum"
        ... )

    Note:
        - The function automatically converts offset and index tensors to int64
        - Sparse mode is used when the data density is high (> 5 elements per segment)
        - Ragged mode is used for sparser data or when using "tile" combiner
        - All tensors should be on the same device for optimal performance
    """
    if combiner_kwargs is None:
        combiner_kwargs = {}
    assert combiner in ["sum", "mean", "tile"], f"combiner {combiner} is not supported"
    if offsets.dtype != torch.int64:
        offsets = offsets.to(torch.int64)
    if reverse_indices.dtype != torch.int64:
        reverse_indices = reverse_indices.to(torch.int64)
    sparse_mode = True
    if combiner == "tile" or reverse_indices.numel() / offsets.numel() > 5:
        sparse_mode = False

    if sparse_mode:
        num_segments = offsets.numel() - 1
        segment_ids = torch.ops.recis.gen_segment_indices_by_offset(offsets)
        return EmbeddingSegmentReduceSparse.apply(
            unique_emb,
            weight,
            reverse_indices,
            segment_ids,
            num_segments,
            combiner,
            "auto",
        )
    else:
        return EmbeddingSegmentReduceRagged.apply(
            unique_emb, weight, reverse_indices, offsets, combiner, combiner_kwargs
        )


def sparse_embedding_segment_reduce(
    unique_emb: torch.Tensor,
    weight: torch.Tensor,
    reverse_indices: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    combiner: str,
):
    """Perform segment reduction on sparse embeddings.

    This function provides a high-level interface for sparse embedding
    segment reduction operations, commonly used in recommendation systems
    for aggregating embeddings within defined segments.

    Args:
        unique_emb (torch.Tensor): Unique embeddings tensor of shape (U, D).
        weight (torch.Tensor): Weights tensor of shape (N,).
        reverse_indices (torch.Tensor): Index mapping tensor of shape (N,).
        segment_ids (torch.Tensor): Segment ID tensor of shape (N,).
        num_segments (int): Total number of segments in the output.
        combiner (str): Reduction combiner ("sum" or "mean").

    Returns:
        torch.Tensor: Reduced embeddings tensor of shape (num_segments, D).

    Example:
        >>> unique_emb = torch.randn(50, 32, requires_grad=True)
        >>> weight = torch.ones(100)
        >>> reverse_indices = torch.randint(0, 50, (100,))
        >>> segment_ids = torch.randint(0, 5, (100,))
        >>> result = sparse_embedding_segment_reduce(
        ...     unique_emb, weight, reverse_indices, segment_ids, 5, "mean"
        ... )
    """
    assert combiner in ["sum", "mean"], f"combiner {combiner} is not supported"
    assert weight is not None, "none weight is not supported"
    return EmbeddingSegmentReduceSparse.apply(
        unique_emb, weight, reverse_indices, segment_ids, num_segments, combiner, "auto"
    )


def merge_offsets(offsets: List[torch.Tensor]) -> torch.Tensor:
    """Merge multiple offset tensors into a single offset tensor.

    This utility function combines multiple offset tensors from different
    ragged dimensions into a single flattened offset tensor, useful for
    multi-dimensional ragged tensor operations.

    Args:
        offsets (List[torch.Tensor]): List of offset tensors to merge.
            Each tensor should be 1D with int32 or int64 dtype.

    Returns:
        torch.Tensor: Merged offset tensor.

    Example:
        >>> offset1 = torch.tensor([0, 2, 5, 7])
        >>> offset2 = torch.tensor([0, 3, 6])
        >>> merged = merge_offsets([offset1, offset2])
    """
    return torch.ops.recis.merge_offsets(offsets)


def ids_partition(ids: torch.Tensor, max_partition_num: int, world_size: int):
    """Partitions IDs into multiple segments across distributed devices.

    Args:
        ids (torch.Tensor): Input tensor of IDs to be partitioned.
        max_partition_num (int): Maximum number of partitions.
        world_size (int): Number of devices or partitions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Unique IDs, segment sizes, and reverse indices.

    Example:
        >>> ids = torch.randint(0, 100, (1000,))
        >>> unique_ids, segment_size, reverse_indice = ids_partition(
        ...     ids, max_partition_num=16, world_size=4
        ... )
    """
    """
    unique_ids, segment_size, reverse_indice = torch.ops.recis.ids_partition(
        ids, max_partition_num, world_size
    )
    """
    unique_ids, segment_size, reverse_indice = torch.ops.recis.ids_partition(
        ids, world_size
    )
    return unique_ids, segment_size, reverse_indice


def ids_encode(ids_list: List[torch.Tensor], table_ids: Union[torch.Tensor, list]):
    """Encode a list of ID tensors by applying table IDs as offsets.

    This function encodes multiple ID tensors by adding corresponding table
    ID offsets, which is useful for managing multiple embedding tables in
    recommendation systems.

    Args:
        ids_list (List[torch.Tensor]): List of ID tensors to encode.
        table_ids (Union[torch.Tensor, list]): Table IDs used for encoding.
            Can be a list or tensor of int64 values.

    Returns:
        torch.Tensor: Encoded ID tensor with table offsets applied.

    Raises:
        AssertionError: If ids_list is not a list or if tensors are not
            on the same device.

    Example:
        >>> ids_list = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        >>> table_ids = [0, 100]  # Add 0 to first table, 100 to second
        >>> encoded_ids = ids_encode(ids_list, table_ids)
        >>> print(encoded_ids)  # [1, 2, 3, 104, 105, 106]
    """
    assert isinstance(ids_list, list), "ids_list must be a list"
    for ids in ids_list:
        assert isinstance(ids, torch.Tensor), "ids must be a tensor"
        assert ids.device == ids_list[0].device, (
            f"ids must be on the same device, {ids.device} != {ids_list[0].device}"
        )
    if isinstance(table_ids, list):
        table_ids = torch.tensor(
            table_ids, dtype=torch.int64, device=ids_list[0].device
        )
    return torch.ops.recis.ids_encode(ids_list, table_ids)
