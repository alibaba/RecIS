import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class RaggedPadInfo:
    """Information about padding operations applied to ragged tensors.

    This dataclass stores metadata about padding and dropping operations
    that have been applied to ragged tensors, enabling proper reconstruction
    and understanding of the tensor's transformation history.

    Attributes:
        drop_nums (List[torch.Tensor], optional): Number of elements dropped
            from each dimension. Defaults to None.
        pad_nums (List[torch.Tensor], optional): Number of elements padded
            to each dimension. Defaults to None.
        drop_sides (torch.Tensor, optional): Sides from which elements were
            dropped (e.g., 'left', 'right'). Defaults to None.
        pad_sides (torch.Tensor, optional): Sides to which elements were
            padded (e.g., 'left', 'right'). Defaults to None.

    Example:
        >>> pad_info = RaggedPadInfo(
        ...     drop_nums=[torch.tensor([1, 0, 2])],
        ...     pad_nums=[torch.tensor([0, 1, 0])],
        ...     drop_sides=torch.tensor([0, 1, 0]),  # 0=left, 1=right
        ...     pad_sides=torch.tensor([1, 0, 1]),
        ... )
    """

    drop_nums: List[torch.Tensor] = None
    pad_nums: List[torch.Tensor] = None
    drop_sides: torch.Tensor = None
    pad_sides: torch.Tensor = None


class RaggedTensor:
    """Multi-dimensional tensor with variable-length dimensions.

    RaggedTensor represents tensors where one or more dimensions can have
    variable lengths. This is particularly useful for handling sequences
    of different lengths in batch processing scenarios common in
    recommendation systems and natural language processing.

    The tensor is stored in a flattened format with offset arrays that
    indicate the boundaries of each sequence in each dimension.

    Args:
        values (torch.Tensor): Flattened tensor containing all values.
        offsets (List[torch.Tensor]): List of offset tensors defining
            boundaries for each ragged dimension. Can also be a single
            tensor which will be converted to a list.
        weight (torch.Tensor, optional): Optional weight tensor with same
            shape as values. Defaults to None.
        dense_shape (tuple, optional): Shape of the equivalent dense tensor.
            If None, will be inferred from the data. Defaults to None.

    Attributes:
        _values (torch.Tensor): Internal storage for tensor values.
        _offsets (List[torch.Tensor]): Internal storage for offset arrays.
        _weight (torch.Tensor): Internal storage for weights.
        _dense_shape (tuple): Shape information for the tensor.
        _pad_info (RaggedPadInfo): Padding operation metadata.

    Example:
        >>> import torch
        >>> from recis.ragged.tensor import RaggedTensor
        >>> # Create ragged tensor manually
        >>> values = torch.tensor([1, 2, 3, 4, 5, 6])
        >>> offsets = [torch.tensor([0, 2, 5, 6])]  # 3 sequences: [1,2], [3,4,5], [6]
        >>> ragged = RaggedTensor(values, offsets)
        >>> print(ragged.shape)  # (3, 3) - 3 sequences, max length 3
        >>> # Create from dense tensor
        >>> dense = torch.tensor([[1, 2, 0], [3, 4, 5], [6, 0, 0]])
        >>> ragged = RaggedTensor.from_dense(dense)
        >>> # Convert to different formats
        >>> dense_reconstructed = ragged.to_dense()
        >>> sparse = ragged.to_sparse()
    """

    def __init__(
        self,
        values: torch.Tensor,
        offsets: List[torch.Tensor],
        weight: Optional[torch.Tensor] = None,
        dense_shape: Optional[tuple] = None,
    ):
        """Initialize a RaggedTensor.

        Args:
            values (torch.Tensor): Flattened tensor containing all values.
            offsets (List[torch.Tensor]): Offset tensors defining boundaries.
            weight (torch.Tensor, optional): Optional weight tensor. Defaults to None.
            dense_shape (tuple, optional): Dense tensor shape. Defaults to None.
        """
        self._values = values
        if isinstance(offsets, torch.Tensor):
            offsets = [offsets]
        self._offsets = offsets
        self._weight = weight
        self._dense_shape = dense_shape
        if self._dense_shape is None:
            self._dense_shape = (-1,) * (len(offsets) + 1)
            self._dense_shape = self.real_shape()
        self._dense_shape = tuple(self._dense_shape)
        self._pad_info = None

    def set_dim_by_rank(self, dim, rank):
        """Set the size of a specific dimension by rank.

        Args:
            dim (int): The dimension size to set.
            rank (int): The rank (dimension index) to modify.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> ragged.set_dim_by_rank(10, 1)  # Set dimension 1 to size 10
        """
        dense_shape = list(self._dense_shape)
        dense_shape[rank] = dim
        self._dense_shape = tuple(dense_shape)

    @property
    def dim(self):
        """Get the number of dimensions.

        Returns:
            int: Number of dimensions in the tensor.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> print(ragged.dim)  # 2 for 2D ragged tensor
        """
        return len(self._dense_shape)

    @property
    def max_length(self):
        """Get the maximum length in the last dimension.

        Returns:
            int: Maximum sequence length in the last dimension.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> print(ragged.max_length)  # Maximum sequence length
        """
        return self.get_shape(-1)

    @property
    def pad_info(self):
        """Get padding information.

        Returns:
            RaggedPadInfo: Padding operation metadata, or None if no padding applied.
        """
        return self._pad_info

    def set_pad_info(self, pad_info: RaggedPadInfo):
        """Set padding information for the tensor.

        Args:
            pad_info (RaggedPadInfo): Padding operation metadata to store.

        Example:
            >>> pad_info = RaggedPadInfo(drop_nums=[torch.tensor([1, 0])])
            >>> ragged.set_pad_info(pad_info)
        """
        self._pad_info = pad_info

    def real_shape(self, start=0, end=None):
        """Get the real shape of the tensor within a dimension range.

        Computes the actual shape by resolving any -1 placeholders with
        the real dimension sizes.

        Args:
            start (int, optional): Starting dimension index. Defaults to 0.
            end (int, optional): Ending dimension index. If None, uses all
                dimensions from start. Defaults to None.

        Returns:
            tuple: Real shape of the tensor in the specified range.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> print(ragged.real_shape())  # (3, 3) - full shape
            >>> print(ragged.real_shape(0, 1))  # (3,) - first dimension only
        """
        if end is None:
            end = len(self._dense_shape)
        if end < 0:
            end = len(self._dense_shape) + end
        real_shape = []
        for i in range(start, end):
            s = self._dense_shape[i]
            if s == -1:
                s = self.get_shape(i)
            real_shape.append(s)
        return tuple(real_shape)

    def get_shape(self, dim):
        """Get the size of a specific dimension.

        Args:
            dim (int): Dimension index to query.

        Returns:
            int: Size of the specified dimension.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> print(ragged.get_shape(0))  # Number of sequences
            >>> print(ragged.get_shape(1))  # Maximum sequence length
        """
        if self._dense_shape[dim] != -1:
            return self._dense_shape[dim]
        elif dim == 0:
            return self._offsets[0].numel() - 1
        elif self._offsets[dim - 1].numel() < 2:
            return 0
        else:
            return (
                (self._offsets[dim - 1][1:] - self._offsets[dim - 1][:-1]).max().item()
            )

    def to_dense(self, default_value: float = 0.0):
        """Convert ragged tensor to dense tensor format.

        Args:
            default_value (float, optional): Value to use for padding shorter
                sequences. Defaults to 0.0.

        Returns:
            torch.Tensor: Dense tensor with padding applied to make all
                sequences the same length.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> dense = ragged.to_dense(default_value=-1.0)
            >>> print(dense)  # Dense tensor with -1.0 padding
        """
        return torch.ops.recis.ragged_to_dense(
            self._values, self._offsets, default_value
        )

    @staticmethod
    def from_numpy(values: np.ndarray, offsets: List[np.ndarray]):
        """Create RaggedTensor from NumPy arrays.

        Args:
            values (np.ndarray): NumPy array containing flattened values.
            offsets (List[np.ndarray]): List of NumPy arrays containing offsets.
                Can also be a single array which will be converted to a list.

        Returns:
            RaggedTensor: New RaggedTensor instance created from NumPy data.

        Example:
            >>> import numpy as np
            >>> values = np.array([1, 2, 3, 4, 5])
            >>> offsets = [np.array([0, 2, 5])]
            >>> ragged = RaggedTensor.from_numpy(values, offsets)
        """
        values = torch.from_numpy(values)
        if isinstance(offsets, np.ndarray):
            offsets = [offsets]
        offsets = [torch.from_numpy(offset) for offset in offsets]
        return RaggedTensor(values, offsets)

    @staticmethod
    def from_dense(
        dense: torch.Tensor, check_invalid: bool = False, invalid_value: int = 0
    ):
        """Create RaggedTensor from dense tensor.

        Automatically detects padding (zero values) and creates an efficient
        ragged representation by removing trailing zeros.

        Args:
            dense (torch.Tensor): Dense tensor to convert. Zero values at the
                end of sequences are treated as padding.

        Returns:
            RaggedTensor: New RaggedTensor instance with padding removed.

        Example:
            >>> dense = torch.tensor([[1, 2, 0], [3, 4, 5], [6, 0, 0]])
            >>> ragged = RaggedTensor.from_dense(dense)
            >>> print(ragged.values())  # [1, 2, 3, 4, 5, 6]
        """
        invalid_tensor = torch.tensor([invalid_value], dtype=dense.dtype)
        values, offsets = torch.ops.recis.dense_to_ragged(
            dense, check_invalid, invalid_tensor
        )
        return RaggedTensor(values, offsets, dense_shape=dense.shape)

    def to_sparse(self):
        """Convert ragged tensor to sparse tensor format.

        Returns:
            torch.sparse.Tensor: Sparse tensor representation where non-zero
                values are stored with their indices.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> sparse = ragged.to_sparse()
            >>> print(sparse.indices())  # Indices of non-zero values
            >>> print(sparse.values())  # Non-zero values
        """
        return torch.ops.recis.ragged_to_sparse(self._values, self._offsets)

    @property
    def device(self):
        """Get the device of the tensor.

        Returns:
            torch.device: Device where the tensor is stored.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> print(ragged.device)  # cuda:0 or cpu
        """
        return self._values.device

    @property
    def shape(self):
        """Get the shape of the tensor.

        Returns:
            tuple: Shape tuple indicating dimensions of the tensor.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> print(ragged.shape)  # (batch_size, max_length)
        """
        return self.real_shape()

    def size(self):
        """Get the size (shape) of the tensor.

        Returns:
            tuple: Shape tuple indicating dimensions of the tensor.

        Note:
            This method is provided for compatibility with PyTorch tensors.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> print(ragged.size())  # Same as ragged.shape
        """
        return self.real_shape()

    def values(self):
        """Get the flattened values tensor.

        Returns:
            torch.Tensor: Flattened tensor containing all values.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> vals = ragged.values()
            >>> print(vals)  # [1, 2, 3, 4, 5, 6, ...]
        """
        return self._values

    def set_values(self, values: torch.Tensor):
        """Set new values for the tensor.

        Args:
            values (torch.Tensor): New flattened values tensor. Must have
                the same number of elements as the current values.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> new_values = torch.tensor([10, 20, 30, 40, 50, 60])
            >>> ragged.set_values(new_values)
        """
        self._values = values

    def offsets(self):
        """Get the offset tensors.

        Returns:
            List[torch.Tensor]: List of offset tensors defining sequence boundaries.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> offs = ragged.offsets()
            >>> print(offs[0])  # [0, 2, 5, 6] - boundaries for sequences
        """
        return self._offsets

    def weight(self):
        """Get the weight tensor.

        Returns:
            torch.Tensor or None: Weight tensor if available, None otherwise.

        Example:
            >>> ragged = RaggedTensor(values, offsets, weight=weights)
            >>> w = ragged.weight()
            >>> print(w)  # Weight values corresponding to each element
        """
        return self._weight

    def set_weight(self, weight: torch.Tensor):
        """Set weight tensor for the ragged tensor.

        Args:
            weight (torch.Tensor): Weight tensor with same shape as values.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> weights = torch.ones_like(ragged.values())
            >>> ragged.set_weight(weights)
        """
        self._weight = weight

    @property
    def dtype(self):
        """Get the data type of the tensor values.

        Returns:
            torch.dtype: Data type of the values tensor.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> print(ragged.dtype)  # torch.float32, torch.int64, etc.
        """
        return self._values.dtype

    def pin_memory(self):
        """Create a copy of the tensor in pinned memory.

        Pinned memory enables faster CPU-GPU transfers.

        Returns:
            RaggedTensor: New RaggedTensor instance with tensors in pinned memory.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> pinned_ragged = ragged.pin_memory()
            >>> # Use pinned_ragged for faster GPU transfers
        """
        return RaggedTensor(
            self._values.pin_memory(),
            [offset.pin_memory() for offset in self._offsets],
            self._weight.pin_memory() if self._weight is not None else None,
            dense_shape=self._dense_shape,
        )

    def to(self, *args, **kwargs):
        """Move tensor to specified device or convert to specified dtype.

        Args:
            *args: Positional arguments passed to torch.Tensor.to().
            **kwargs: Keyword arguments passed to torch.Tensor.to().

        Returns:
            RaggedTensor: New RaggedTensor instance on the specified device/dtype.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> gpu_ragged = ragged.to("cuda")
            >>> float_ragged = ragged.to(torch.float32)
            >>> gpu_float_ragged = ragged.to("cuda", torch.float32)
        """
        return RaggedTensor(
            self._values.to(*args, **kwargs),
            [offset.to(*args, **kwargs) for offset in self._offsets],
            self._weight.to(*args, **kwargs) if self._weight is not None else None,
            dense_shape=self._dense_shape,
        )

    def cuda(self):
        """Move tensor to CUDA device.

        Returns:
            RaggedTensor: New RaggedTensor instance on CUDA device.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> gpu_ragged = ragged.cuda()
            >>> print(gpu_ragged.device)  # cuda:0
        """
        return RaggedTensor(
            self._values.cuda(),
            [offset.cuda() for offset in self._offsets],
            self._weight.cuda() if self._weight is not None else None,
            dense_shape=self._dense_shape,
        )

    def clone(self):
        """Create a shallow copy of the ragged tensor.

        The returned tensor shares the same underlying data but is a separate
        object that can be modified independently.

        Returns:
            RaggedTensor: Shallow copy of the ragged tensor.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> cloned = ragged.clone()
            >>> # cloned shares data with ragged but is a separate object
        """
        return RaggedTensor(
            self._values,
            list(self._offsets),
            self._weight if self._weight is not None else None,
            dense_shape=self._dense_shape,
        )

    @property
    def _nested(self):
        if len(self._offsets) > 1:
            return RaggedTensor(
                values=self._values, offsets=self._offsets[1:], weight=self._weight
            )
        else:
            return self._values

    @classmethod
    def from_offsets(
        cls,
        values: torch.Tensor,
        offsets: Union[torch.Tensor, List[torch.Tensor]],
        weight: Optional[torch.Tensor] = None,
    ):
        if isinstance(values, RaggedTensor):
            return cls(
                values=values.values(),
                offsets=[offsets] + values.offsets(),
                weight=values.weight(),
            )
        return cls(values, offsets=offsets, weight=weight)

    def __str__(self) -> str:
        """Return string representation of the ragged tensor.

        Returns:
            str: Detailed string representation showing all tensor properties.

        Example:
            >>> ragged = RaggedTensor(values, offsets)
            >>> print(ragged)
            RaggedTensor(
              values=tensor([1, 2, 3, 4, 5, 6]),
              offsets=[tensor([0, 2, 5, 6])],
              weight=None,
              dense_shape=(3, 3),
              dtype=torch.int64,
              device=cpu
            )
        """
        s = (
            f"RaggedTensor(\n"
            f"  values={self._values},\n"
            f"  offsets={self._offsets},\n"
            f"  weight={self._weight},\n"
            f"  dense_shape={self._dense_shape},\n"
            f"  dtype={self._values.dtype},\n"
            f"  device={self._values.device}\n"
            f")"
        )
        return s

    def __repr__(self) -> str:
        """Return detailed string representation of the ragged tensor.

        This method is called when the object is displayed in containers
        like lists, dictionaries, or when using repr().

        Returns:
            str: Detailed string representation showing all tensor properties.

        Example:
            >>> data = {"a": RaggedTensor(values, offsets)}
            >>> print(data)  # Will show detailed RaggedTensor content
        """
        return self.__str__()

    def __deepcopy__(self, memo):
        """Create a deep copy of the ragged tensor.

        This method enables compatibility with Python's copy.deepcopy() function.
        All tensor data and metadata are deeply copied.

        Args:
            memo (dict): Memo dictionary used by copy.deepcopy() to avoid
                infinite recursion and optimize copying of shared objects.

        Returns:
            RaggedTensor: Deep copy of the ragged tensor with all data copied.

        Example:
            >>> import copy
            >>> original = RaggedTensor(values, offsets)
            >>> deep_copied = copy.deepcopy(original)
            >>> # Modifying deep_copied won't affect original
            >>> deep_copied._values[0] = 999
            >>> print(original._values[0])  # Original value unchanged
        """
        # Deep copy all tensor data
        new_values = self._values.clone()
        new_offsets = [offset.clone() for offset in self._offsets]
        new_weight = self._weight.clone() if self._weight is not None else None

        # Deep copy pad_info if it exists
        new_pad_info = None
        if self._pad_info is not None:
            new_pad_info = copy.deepcopy(self._pad_info, memo)

        # Create new RaggedTensor with deep copied data
        new_ragged = RaggedTensor(
            values=new_values,
            offsets=new_offsets,
            weight=new_weight,
            dense_shape=self._dense_shape,  # tuple is immutable, safe to share
        )
        new_ragged._pad_info = new_pad_info

        return new_ragged

    def __getitem__(self, key):
        """Support advanced indexing for RaggedTensor.

        Supports standard Python indexing syntax including integers, slices,
        None (newaxis), and Ellipsis. Indexing into inner ragged dimensions
        with integers or tensors is not allowed.

        Args:
            key: Indexing key. Can be an int, slice, tuple of keys, or Ellipsis.

        Returns:
            A new RaggedTensor representing the indexed result.

        Example:
            >>> import torch
            >>> from recis.ragged.tensor import RaggedTensor
            >>> values = torch.tensor([1, 2, 3, 4, 5, 6])
            >>> offsets = [torch.tensor([0, 2, 5, 6])]  # Sequences: [1,2], [3,4,5], [6]
            >>> ragged = RaggedTensor(values, offsets)
            >>> # Select first sequence
            >>> first_seq = ragged[0]  # Returns dense tensor [1, 2]
            >>> # Slice outer dimension
            >>> subset = ragged[1:3]  # Returns RaggedTensor with sequences [3,4,5], [6]
            >>> # Add new axis
            >>> expanded = ragged[None, :]  # Shape becomes (1, 3, 3)
        """
        if not isinstance(key, (list, tuple)):
            key = (key,)
        if Ellipsis in key:
            key = self._expand_ellipsis(key)
        return self._getitem_impl(key)

    def _getitem_impl(self, keys: Tuple) -> "RaggedTensor":
        """Internal implementation of __getitem__.

        Processes a normalized tuple of indexing keys and dispatches to
        appropriate slicing methods based on the key type and position.

        Args:
            keys: Normalized tuple of indexing objects (int, slice, None, etc.).

        Returns:
            A new RaggedTensor representing the indexed result.

        Raises:
            IndexError: If negative step slicing is used or if inner ragged
                dimensions are indexed with integers or tensors.
            TypeError: If an unsupported index type is provided.
        """
        if not keys:
            return self

        # Disallow negative step slicing
        if any(
            isinstance(k, slice) and k.step is not None and k.step < 0 for k in keys
        ):
            raise IndexError("Negative step slicing is not supported")

        # Disallow indexing into inner ragged dimensions with int or tensor
        row_key = keys[0]
        remaining_keys = keys[1:]
        if any(isinstance(item, (int, torch.Tensor)) for item in remaining_keys):
            raise IndexError("Indexing into inner ragged dimensions is not allowed")

        # Handle newaxis (None) at outer dimension
        if row_key is None:
            offset = self._offsets[0]
            new_offset = torch.tensor(
                [0, len(offset) - 1], dtype=offset.dtype, device=offset.device
            )
            inner_result = self._getitem_impl(remaining_keys)
            return RaggedTensor.from_offsets(inner_result, new_offset)

        # Handle integer indexing (single row)
        if isinstance(row_key, int):
            return self._slice_row(row_key, remaining_keys)

        # Handle slice indexing (multiple rows)
        if isinstance(row_key, slice):
            sliced_rt = self._slice_row_axis(row_key)
            return (
                sliced_rt._slice_inner_axis(remaining_keys)
                if remaining_keys
                else sliced_rt
            )

        raise TypeError(f"Unsupported index type: {type(row_key)}")

    def _slice_row_axis(self, row_slice: slice) -> "RaggedTensor":
        """Slice along the outermost (batch) dimension.

        This method handles slicing of the top-level sequences without
        modifying inner ragged structures.

        Args:
            row_slice: Slice object for the outer dimension.

        Returns:
            A new RaggedTensor containing the selected rows.

        Raises:
            NotImplementedError: If step != 1 (non-contiguous slicing).
        """
        offset = self._offsets[0]
        num_rows = len(offset) - 1
        start, stop, step = row_slice.indices(num_rows)

        if self._is_full_slice(row_slice):
            return self

        # Optimized path for contiguous slicing (step=1)
        if step == 1:
            if stop <= start:
                return self._empty_like_same_ndim()

            sliced_offset = offset[start : stop + 1]
            layer_start = sliced_offset[0]
            layer_stop = sliced_offset[-1]
            fixed_offset = sliced_offset - layer_start
            nested_sliced = self._nested[layer_start:layer_stop]

            return RaggedTensor.from_offsets(nested_sliced, fixed_offset)
        else:
            indices = torch.arange(start, stop, step, device=offset.device)
            return self.__gather_rows(indices)

    def _slice_inner_axis(self, keys: Tuple) -> "RaggedTensor":
        """Apply slicing to inner ragged dimensions.

        After outer rows have been selected, this method applies slicing
        to each row's internal structure. Each row is sliced independently
        according to its actual length.

        Args:
            keys: Remaining indexing keys for inner dimensions.

        Returns:
            A new RaggedTensor with inner dimensions sliced.

        Raises:
            IndexError: If attempting to index inner dimensions with non-slice objects.
        """
        if not keys:
            return self

        col_key = keys[0]
        remaining = keys[1:]

        # Handle newaxis in inner dimensions
        if col_key is None:
            inner = self._slice_inner_axis(remaining)
            new_dim_offset = torch.arange(
                len(inner._offsets[0]),
                dtype=inner._offsets[0].dtype,
                device=inner._offsets[0].device,
            )
            return RaggedTensor.from_offsets(inner, new_dim_offset)

        assert isinstance(col_key, slice), (
            "Indexing into inner ragged dimension is not allowed."
        )

        offset = self._offsets[0]
        inner_lengths = offset[1:] - offset[:-1]
        rel_start, rel_stop, step = self._resolve_ragged_slice(col_key, inner_lengths)
        abs_start = offset[:-1] + rel_start
        abs_stop = offset[:-1] + rel_stop

        return self._ragged_range_gather(abs_start, abs_stop, step, remaining=remaining)

    def _ragged_range_gather(
        self,
        starts: torch.Tensor,
        stops: torch.Tensor,
        step: int,
        remaining: Optional[Tuple] = None,
    ) -> "RaggedTensor":
        """Gather ragged ranges and reconstruct as a new RaggedTensor.

        Given per-row start/stop positions and a step size, extract elements
        and reassemble them into a properly structured RaggedTensor.

        Args:
            starts: Absolute start indices for each row.
            stops: Absolute stop indices (exclusive) for each row.
            step: Step size for strided extraction.
            remaining: Additional indexing keys for deeper dimensions.

        Returns:
            A new RaggedTensor containing the gathered data.
        """
        diff = stops - starts
        diff = torch.clamp_min(diff, 0)
        lengths = (diff + step - 1) // step
        total_elements = lengths.sum().item()

        if total_elements == 0:
            offset = torch.zeros(
                len(starts) + 1, dtype=starts.dtype, device=starts.device
            )
            empty_nested = self._nested[:0]
            return RaggedTensor.from_offsets(empty_nested, offset)

        # Generate gather indices vectorized
        row_ids = torch.repeat_interleave(
            torch.arange(len(lengths), device=lengths.device), lengths
        )
        length_cumsum = torch.cumsum(lengths, dim=0)
        global_starts = torch.cat(
            [torch.tensor([0], device=lengths.device), length_cumsum[:-1]]
        )
        global_arange = torch.arange(total_elements, device=lengths.device)
        local_indices = global_arange - global_starts[row_ids]
        gather_indices = starts[row_ids] + local_indices * step

        gathered_nested = self.__internal_gather(self._nested, gather_indices)

        # Handle weights (only at leaf level)
        gathered_weight = None
        if self._weight is not None and not isinstance(self._nested, RaggedTensor):
            gathered_weight = self._weight[gather_indices]

        # Build new offsets
        new_offset = torch.cat(
            [torch.zeros(1, device=lengths.device, dtype=lengths.dtype), length_cumsum]
        )

        # Recurse on remaining keys if needed
        if remaining:
            if not isinstance(gathered_nested, RaggedTensor):
                if len(remaining) == 1 and remaining[0] is None:
                    gathered_nested = RaggedTensor.from_offsets(
                        gathered_nested,
                        torch.arange(gathered_nested.numel() + 1),
                        weight=gathered_weight,
                    )
                else:
                    raise IndexError("Cannot apply inner slicing to dense tensor")
            else:
                gathered_nested = gathered_nested._slice_inner_axis(remaining)

        return RaggedTensor.from_offsets(
            gathered_nested, new_offset, weight=gathered_weight
        )

    def __internal_gather(self, target, indices):
        """Private method to safely gather from tensor or RaggedTensor."""
        if isinstance(target, RaggedTensor):
            return target.__gather_rows(indices)
        else:
            return target[indices]

    def __gather_rows(self, indices: torch.Tensor):
        """Private method to gather rows by index.

        Used internally for strided slicing or recursive indexing.
        """
        if len(indices) == 0:
            return self._empty_like_same_ndim()

        offset = self._offsets[0]
        start = offset[indices]
        stop = offset[indices + 1]
        return self._ragged_range_gather(start, stop, step=1)

    def _resolve_ragged_slice(
        self, s: slice, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Resolve a slice into per-row relative start/stop positions.

        Handles negative indices and clamps to valid range [0, length].

        Args:
            s: Slice object to parse.
            lengths: Length of each row.

        Returns:
            Tuple of (relative_starts, relative_stops, step).
        """
        step = s.step if s.step is not None else 1

        if s.start is None:
            r_start = torch.zeros_like(lengths)
        else:
            val = s.start
            if val < 0:
                r_start = torch.clamp_min(lengths + val, 0)
            else:
                r_start = torch.clamp_max(
                    torch.tensor(val, device=lengths.device), lengths
                )

        if s.stop is None:
            r_stop = lengths
        else:
            val = s.stop
            if val < 0:
                r_stop = torch.clamp_min(lengths + val, 0)
            else:
                r_stop = torch.clamp_max(
                    torch.tensor(val, device=lengths.device), lengths
                )

        return r_start, r_stop, step

    def _slice_row(
        self, idx: int, remaining: Tuple
    ) -> Union[torch.Tensor, "RaggedTensor"]:
        """Extract a single row by index and apply remaining slicing.

        Converts negative indices to positive and validates bounds before
        extracting the corresponding segment from the flattened values.

        Args:
            idx: Row index (supports negative indexing).
            remaining: Additional indexing keys for inner dimensions.

        Returns:
            The extracted row, possibly further sliced by remaining keys.

        Raises:
            IndexError: If index is out of bounds.
        """
        offset = self._offsets[0]
        num_rows = len(offset) - 1
        if idx < 0:
            idx += num_rows
        if idx < 0 or idx >= num_rows:
            raise IndexError(f"Index {idx} out of bounds for {num_rows} rows")

        start, stop = offset[idx], offset[idx + 1]
        row = self._nested[start:stop]
        return row[remaining] if remaining else row

    def _empty_like_same_ndim(self):
        """Create an empty RaggedTensor with the same number of dimensions."""
        return RaggedTensor.from_offsets(
            values=torch.empty(
                (0,), dtype=self._values.dtype, device=self._values.device
            ),
            offsets=[
                torch.tensor(
                    [0], device=self._offsets[0].device, dtype=self._offsets[0].dtype
                )
            ]
            + self._offsets[1:],
        )

    def _is_full_slice(self, s: slice) -> bool:
        """Check if a slice represents the full range with default step.

        A full slice is defined as `slice(None)`, `slice(None, None)`,
        or `slice(None, None, 1)`.

        Args:
            s: Slice object to check.

        Returns:
            True if the slice selects all elements with step 1, False otherwise.
        """
        return s.start is None and s.stop is None and (s.step is None or s.step == 1)

    def _expand_ellipsis(self, key: Tuple) -> Tuple:
        """Expand Ellipsis (...) into appropriate number of full slices.

        Replaces the Ellipsis with enough `slice(None)` objects to match
        the tensor's number of dimensions.

        Args:
            key: Tuple containing exactly one Ellipsis.

        Returns:
            Expanded tuple with Ellipsis replaced by full slices.

        Raises:
            AssertionError: If multiple Ellipses are present.
        """
        explicit_dims = sum(1 for k in key if k is not Ellipsis and k is not None)
        expand_count = self.dim - explicit_dims
        assert key.count(Ellipsis) == 1, "Multiple ellipses are not allowed"

        expanded_key = []
        for k in key:
            if k is Ellipsis:
                expanded_key.extend([slice(None)] * max(0, expand_count))
            else:
                expanded_key.append(k)
        return tuple(expanded_key)
