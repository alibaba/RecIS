from typing import List, Tuple, Union

import torch


__ALL__ = [
    "fused_bucketize_gpu",
    "fused_uint64_mod_gpu",
    "fused_multi_hash",
    "fused_int64_to_string_int8",
]


def _check_device_all(tensors: List[torch.Tensor], device_type: str) -> None:
    """Checks that all tensors are on the specified device.

    Args:
        tensors (List[torch.Tensor]): List of tensors to check.
        device_type (str): Expected device type (e.g., 'cuda', 'cpu').
    """
    for t in tensors:
        assert t.device.type == device_type, (
            f"tensors must be on {device_type}, but got {t.device.type}"
        )


def _check_dtype_all(tensors: List[torch.Tensor], dtype: torch.dtype) -> None:
    """Checks that all tensors have the specified data type.

    Args:
        tensors (List[torch.Tensor]): List of tensors to check.
        dtype (torch.dtype): Expected data type.
    """
    for t in tensors:
        assert t.dtype == dtype, f"tensors must be {dtype}, but got {t.dtype}"


def fused_bucketize_gpu(
    values: List[torch.Tensor], boundaries: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GPU-accelerated bucketization operation. Maps each value in `values` to a bucket index based on the corresponding `boundaries`.

    Args:
        values (List[torch.Tensor]): List of input tensors containing float values to be bucketized. Must be on CUDA.
        boundaries (List[torch.Tensor]): List of boundary tensors for bucket definitions. Each tensor must be sorted and on CUDA.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - **bucket_indices**: Tensor of bucket indices for each value.
            - **offsets**: Auxiliary tensor representing offsets for merging buckets.

    Raises:
        AssertionError: If input conditions are not met.

    Example:
        >>> values = [torch.tensor([1.2, 3.5, 0.8], device='cuda'),
        >>>           torch.tensor([2.1, 4.3, 1.9], device='cuda')]
        >>> boundaries = [torch.tensor([1.0, 2.0, 3.0], device='cuda'),
        >>>               torch.tensor([3.0, 4.0, 5.0], device='cuda')]
        >>> indices, offsets = fused_bucketize_gpu(values, boundaries)
    """
    assert len(values) == len(boundaries), (
        "values and boundaries must have the same length"
    )
    _check_device_all(values, "cuda")
    _check_dtype_all(values, torch.float)
    _check_dtype_all(boundaries, torch.float)
    _check_device_all(boundaries, "cuda")

    return torch.ops.recis.fused_bucketized(values, boundaries)


def fused_uint64_mod_gpu(
    values: List[torch.Tensor], mods: Union[List, torch.Tensor]
) -> torch.Tensor:
    """GPU-accelerated unsigned 64-bit integer modulo operation.

    Args:
        values (List[torch.Tensor]): List of tensors containing int64 values. Must be on CUDA.
        mods (Union[List, torch.Tensor]): Modulo values. Can be a list or tensor of int64 values.

    Returns:
        torch.Tensor: Result tensor where each element is `(value % mod)` using unsigned interpretation.

    Raises:
        AssertionError: If input conditions are not met.

    Example:
        >>> values = [torch.tensor([10, 20, 30], dtype=torch.int64, device='cuda'),
        >>>           torch.tensor([40, 50, 60], dtype=torch.int64, device='cuda')]
        >>> mods = [3, 5]
        >>> result = fused_uint64_mod_gpu(values, mods)
    """
    _check_device_all(values, "cuda")
    _check_dtype_all(values, torch.int64)
    if isinstance(mods, list):
        mods = torch.tensor(mods, dtype=torch.int64, device=values[0].device)
    return torch.ops.recis.fused_uint64_mod(values, mods)


def fused_ids_encode_gpu(
    ids_list: List[torch.Tensor], table_ids: Union[torch.Tensor, list]
):
    """Encodes a list of ID tensors by applying table IDs as an offset.

    Args:
        ids_list (List[torch.Tensor]): List of ID tensors to encode.
        table_ids (Union[torch.Tensor, list]): Table IDs used for encoding; can be a list or tensor.

    Returns:
        torch.Tensor: Encoded ID tensor.

    Raises:
        AssertionError: If `ids_list` is not a list or if tensors in `ids_list` are not on the same device.

    Example:
        >>> ids_list = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        >>> table_ids = [0, 1]
        >>> encoded_ids = ids_encode(ids_list, table_ids)
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


def fused_multi_hash(
    inputs: List[torch.Tensor],
    muls: List[torch.Tensor],
    primes: List[torch.Tensor],
    bucket_nums: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Fused multi hash.
    """
    assert len(inputs) == len(muls) == len(primes) == len(bucket_nums)
    assert len(inputs) > 0
    device = inputs[0].device
    _check_device_all(inputs, device.type)
    _check_dtype_all(inputs, torch.int64)
    _check_dtype_all(muls, torch.int64)
    _check_dtype_all(primes, torch.int64)
    _check_dtype_all(bucket_nums, torch.int64)
    return torch.ops.recis.fused_multi_hash(inputs, muls, primes, bucket_nums)


def fused_int64_to_string_int8(
    inputs: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Fused operation to convert multiple 1D int64 tensors to string representation as int8 tensors with offsets.

    This function converts each int64 value in each input tensor to its string representation,
    then converts each character to its ASCII code (int8). It returns both the flattened int8
    tensors and offsets arrays indicating the length of each string for each input.

    Args:
        inputs (List[torch.Tensor]): List of 1D int64 tensors to convert. All tensors must be on the same device.

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor]]: A tuple containing:
            - **outputs**: List of 1D int8 tensors containing ASCII codes of all characters for each input.
            - **offsets**: List of 1D int64 tensors containing the length of each string for each input.

    Raises:
        AssertionError: If inputs are not 1D, not int64 type, or not on the same device.

    Example:
        >>> inputs = [
        >>>     torch.tensor([123, -456], dtype=torch.int64),
        >>>     torch.tensor([0, 789], dtype=torch.int64)
        >>> ]
        >>> outputs, offsets = fused_int64_to_string_int8(inputs)
        >>> # outputs[0]: tensor([49, 50, 51, 45, 52, 53, 54], dtype=torch.int8)
        >>> # offsets[0]: tensor([3, 4], dtype=torch.int64)
        >>> # outputs[1]: tensor([48, 55, 56, 57], dtype=torch.int8)
        >>> # offsets[1]: tensor([1, 3], dtype=torch.int64)
        >>> # "123" -> [49, 50, 51], "-456" -> [45, 52, 53, 54]
        >>> # "0" -> [48], "789" -> [55, 56, 57]
    """
    assert isinstance(inputs, list) and len(inputs) > 0, (
        "inputs must be a non-empty list"
    )

    device = inputs[0].device
    for i, input_tensor in enumerate(inputs):
        assert input_tensor.dtype == torch.int64, (
            f"All inputs must be int64, but inputs[{i}] has dtype {input_tensor.dtype}"
        )
        assert input_tensor.dim() == 1, (
            f"All inputs must be 1-dimensional, but inputs[{i}] is {input_tensor.dim()}D"
        )
        assert input_tensor.device == device, (
            f"All inputs must be on the same device, but inputs[{i}] is on {input_tensor.device} "
            f"while inputs[0] is on {device}"
        )

    return torch.ops.recis.fused_int64_to_string_int8(inputs)
