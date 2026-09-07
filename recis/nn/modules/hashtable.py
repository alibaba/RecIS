import json
import math
import os
from typing import List, Optional, Tuple

import torch

from recis.common.singleton import SingletonMeta
from recis.nn.hashtable_hook import AdmitHook, FilterHook
from recis.nn.initializers import ConstantInitializer
from recis.nn.modules.hashtable_hook_impl import HashtableHookFactory, ReadOnlyHookImpl
from recis.utils.logger import Logger


logger = Logger(__name__)


class Slice:
    """Partitioning configuration for distributed hash table storage.

    This class defines how the hash table's key space is partitioned
    across different workers in a distributed setting.

    Attributes:
        slice_begin (int): Starting index of the slice.
        slice_end (int): Ending index of the slice (exclusive).
        slice_size (int): Total size of the key space.

    Example:
    .. code-block:: python
        # Create a slice for worker 0 out of 4 workers
        slice_config = Slice(0, 16384, 65536)

    """

    def __init__(self, slice_beg, slice_end, slice_size) -> None:
        """Initialize slice configuration.

        Args:
            slice_beg (int): Starting index of the slice.
            slice_end (int): Ending index of the slice (exclusive).
            slice_size (int): Total size of the key space.
        """
        self.slice_begin = slice_beg
        self.slice_end = slice_end
        self.slice_size = slice_size


def gen_slice(shard_index=0, shard_num=1, slice_size=65536):
    """Generate slice configuration for distributed hash table partitioning.

    This function creates a Slice object that defines how to partition
    the hash table's key space across multiple workers. It ensures
    balanced distribution with proper handling of remainder keys.

    Args:
        shard_index (int, optional): Index of the current shard/worker.
            Defaults to 0.
        shard_num (int, optional): Total number of shards/workers.
            Defaults to 1.
        slice_size (int, optional): Total size of the key space.
            Defaults to 65536.

    Returns:
        Slice: Slice configuration for the specified shard.

    Example:

    .. code-block:: python

        # Generate slice for worker 1 out of 4 workers
        slice_config = gen_slice(shard_index=1, shard_num=4, slice_size=65536)
        print(
            f"Worker 1 handles keys from {slice_config.slice_begin} "
            f"to {slice_config.slice_end}"
        )

    """
    shard_slice_size = slice_size // shard_num
    shard_slice_sizes = [shard_slice_size] * shard_num
    remain = slice_size % shard_num
    shard_slice_sizes = [
        size + 1 if i < remain else size for i, size in enumerate(shard_slice_sizes)
    ]
    slice_infos = []
    beg = 0
    for size in shard_slice_sizes:
        end = beg + size
        slice_infos.append((beg, end))
        beg = end
    slice_info = slice_infos[shard_index]

    return Slice(slice_info[0], slice_info[1], slice_size)


_default_slice = Slice(0, 65536, 65536)


class HashTable(torch.nn.Module):
    """Distributed hash table for sparse parameter storage and lookup.

    This module provides a distributed hash table implementation that supports
    dynamic sparse parameter storage, efficient lookup operations, and gradient
    computation. It's designed for large-scale sparse learning scenarios where
    the feature vocabulary can grow dynamically.

    Key features:
        - Dynamic feature admission and eviction
        - Distributed storage across multiple workers
        - Efficient gradient computation and aggregation
        - Support for various initialization strategies
        - Hook-based filtering and admission control

    Example:
        Basic usage:

    .. code-block:: python

        import torch
        from recis.nn.modules.hashtable import HashTable

        # Create hash table
        hashtable = HashTable(
            embedding_shape=[64],
            block_size=1024,
            dtype=torch.float32,
            device=torch.device("cuda"),
            name="user_embedding",
        )

        # Lookup embeddings
        ids = torch.tensor([1, 2, 3, 100, 1000])
        embeddings = hashtable(ids)  # Shape: [5, 64]


        Advanced usage with hooks:

    .. code-block:: python

        from recis.nn.hashtable_hook import FrequencyFilterHook

        # Create hash table with filtering
        filter_hook = FrequencyFilterHook(min_frequency=5)
        hashtable = HashTable(
            embedding_shape=[128],
            block_size=2048,
            filter_hook=filter_hook,
            grad_reduce_by="id",
        )

        # Create hash table for distributed training with gradient sum
        hashtable_sum = HashTable(
            embedding_shape=[128],
            block_size=2048,
            grad_reduce_by="worker_sum",  # Use gradient sum instead of average
        )

    """

    def __init__(
        self,
        embedding_shape: List,
        block_size: int = 5,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        coalesced: bool = False,
        children: Optional[List[str]] = None,
        slice: Slice = _default_slice,
        initializer=None,
        name: str = "hashtable",
        grad_reduce_by: str = "worker",
        hdmp_group_size: Optional[int] = None,
        hdmp_group_reduce_by: Optional[str] = None,
        filter_hook: Optional[FilterHook] = None,
        use_pinned_memory: bool = False,
    ):
        """Initialize hash table module.

        Args:
            embedding_shape (List[int]): Shape of embedding vectors.
            block_size (int, optional): Number of embeddings per block. Defaults to 5.
            dtype (torch.dtype, optional): Data type. Defaults to torch.float32.
            device (torch.device, optional): Computation device. Defaults to CPU.
            coalesced (bool, optional): Use coalesced operations. Defaults to False.
            children (Optional[List[str]], optional): Child table names. Defaults to None.
            slice (Slice, optional): Partitioning config. Defaults to _default_slice.
            initializer (Initializer, optional): Initializer. Defaults to None.
            name (str, optional): Table name. Defaults to "hashtable".
            grad_reduce_by (str, optional): Gradient reduction method. Defaults to "worker".
                Options:
                - "worker": Average gradients across workers
                - "worker_sum": Sum gradients across workers (for SparseAdagradSum)
                - "hdmp_group_sum": Sum HDMP group-reduced gradients and store
                  group-reduced gradient squares (for SparseAdagradSum)
                - "id": Average gradients by feature ID
            hdmp_group_size (int, optional): Worker count in each HDMP source group.
            hdmp_group_reduce_by (str, optional): Reduction method inside each HDMP
                source group. Options: "id", "worker", "worker_sum".
            filter_hook (Optional[FilterHook], optional): Filter hook. Defaults to None.
            use_pinned_memory (bool, optional): Whether to use pinned memory for
                CPU intermediate tensors to accelerate H2D/D2H transfers.
                Defaults to False. Set to True to enable pinned memory.

        Raises:
            AssertionError: If grad_reduce_by is not "id", "worker", "worker_sum",
                or "hdmp_group_sum".
        """
        super().__init__()
        if initializer is None:
            self._initializer = ConstantInitializer(init_val=0)
        else:
            self._initializer = initializer

        if children is None:
            children = [name]
        self._device = device
        assert grad_reduce_by in ["id", "worker", "worker_sum", "hdmp_group_sum"]
        if grad_reduce_by == "hdmp_group_sum":
            assert hdmp_group_size is not None and hdmp_group_size > 0
            assert hdmp_group_reduce_by in ["id", "worker", "worker_sum"]
        self._grad_reduce_by = grad_reduce_by
        self._hdmp_group_size = hdmp_group_size
        self._hdmp_group_reduce_by = hdmp_group_reduce_by
        self._initializer.set_shape([block_size] + embedding_shape)
        self._initializer.set_dtype(dtype)
        self._initializer.build()
        self._dtype = dtype
        self._name = name

        for child in children:
            info_str = json.dumps(
                dict(
                    shape=embedding_shape,
                    dtype=str(dtype),
                    initializer=str(self._initializer),
                )
            )
            HashtableRegister().register(child, info_str, self)

        self._hashtable_impl = torch.ops.recis.make_hashtable(
            block_size,
            embedding_shape,
            dtype,
            device,
            coalesced,
            children,
            self._initializer.impl(),
            slice.slice_begin,
            slice.slice_end,
            slice.slice_size,
            use_pinned_memory,
        )
        self._backward_holder = torch.tensor([0.0], requires_grad=True)
        self._worker_num = int(os.environ.get("WORLD_SIZE", 1))

        def state_dict_hook(
            self: HashTable, state_dict: dict, prefix: str, local_metadata
        ):
            state_dict[self._name] = self._hashtable_impl

        self._register_state_dict_hook(state_dict_hook)

        # TODO (sunhechen.shc) support more filter hook
        if filter_hook is not None:
            self._filter_hook_impl = HashtableHookFactory().create_filter_hook(
                self, filter_hook
            )
        else:
            self._filter_hook_impl = torch.nn.Identity()

    @classmethod
    def clear_child(cls, child) -> None:
        """Clear child hashtable."""
        HashtableRegister().get_ht_by_child_name(child)._hashtable_impl.clear(child)

    def forward(
        self,
        ids: torch.Tensor,
        admit_hook: AdmitHook = None,
        source_group: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform embedding lookup for given feature IDs.

        This method looks up embeddings for the provided feature IDs,
        handling deduplication, gradient computation, and optional
        feature admission control.

        Args:
            ids (torch.Tensor): Feature IDs to lookup. Shape: [N] where N
                is the number of features.
            admit_hook (AdmitHook, optional): Hook for controlling feature
                admission. Defaults to None.

        Returns:
            torch.Tensor: Looked up embeddings. Shape: [N, embedding_dim]
                where embedding_dim is determined by embedding_shape.

        Example:

        .. code-block:: python

            # Basic lookup
            ids = torch.tensor([1, 2, 3, 2, 1])  # Note: duplicates
            embeddings = hashtable(ids)  # Shape: [5, embedding_dim]

            # With admission hook
            from recis.nn.hashtable_hook import FrequencyAdmitHook

            admit_hook = FrequencyAdmitHook(min_frequency=3)
            embeddings = hashtable(ids, admit_hook)

        """
        admit_hook_impl = (
            HashtableHookFactory().create_admit_hook(self, admit_hook)
            if admit_hook
            else None
        )
        ids, index = ids.unique(return_inverse=True)
        index = index.to("cuda", non_blocking=True)
        if self.training and self._dtype not in (torch.int8, torch.int32, torch.int64):
            emb_idx, embedding = HashTableLookupHelpFunction.apply(
                ids, self._hashtable_impl, self._backward_holder, admit_hook_impl
            )
            if self._grad_reduce_by == "id":
                embedding = GradIDMeanFunction.apply(embedding, index)
            elif self._grad_reduce_by == "worker_sum":
                embedding = GradWorkerSumFunction.apply(embedding, index, self, emb_idx)
            elif self._grad_reduce_by == "hdmp_group_sum":
                if source_group is None:
                    raise RuntimeError(
                        "source_group is required when grad_reduce_by='hdmp_group_sum'"
                    )
                embedding = GradHDMPGroupSumFunction.apply(
                    embedding,
                    index,
                    source_group,
                    self,
                    emb_idx,
                    self._hdmp_group_size,
                    math.ceil(self._worker_num / self._hdmp_group_size),
                    self._hdmp_group_reduce_by,
                )
            else:
                slice_num = torch.scalar_tensor(self._worker_num)
                embedding = GradWorkerMeanFunction.apply(embedding, index, slice_num)
            self._filter_hook_impl(emb_idx)
        else:
            ids = ids.detach()
            _, embedding = self._hashtable_impl.embedding_lookup(ids, True)
            embedding = embedding.to(device="cuda", non_blocking=True)
            embedding = torch.ops.recis.gather(index, embedding)
        return embedding

    def initializer(self):
        """Get the embedding initializer.

        Returns:
            Initializer: The initializer used for new embeddings.
        """
        return self._initializer

    @property
    def device(self):
        """Get the computation device.

        Returns:
            torch.device: The device used for computation.
        """
        return self._device

    @property
    def coalesce(self):
        """Check if coalesced operations are enabled.

        Returns:
            bool: True if coalesced operations are enabled.
        """
        return self._hashtable_impl.children_info().is_coalesce()

    @property
    def children_hashtable(self):
        """Get the list of child hash tables.

        Returns:
            List[str]: Names of child hash tables.
        """
        return self._hashtable_impl.children_info().children()

    def accept_grad(self, grad_index, grad) -> None:
        """Accept gradients for specific embedding indices.

        Args:
            grad_index (torch.Tensor): Indices of embeddings to update.
            grad (torch.Tensor): Gradient values for the embeddings.
        """
        self._hashtable_impl.accept_grad(grad_index, grad)

    def grad(self) -> torch.Tensor:
        """Get accumulated gradients.

        Returns:
            torch.Tensor: Accumulated gradients.
        """
        return self._hashtable_impl.grad()

    def clear_grad(self) -> None:
        """Clear accumulated gradients."""
        self._hashtable_impl.clear_grad()

    def insert(self, ids, embeddings) -> None:
        """Insert embeddings for specific IDs.

        Args:
            ids (torch.Tensor): Feature IDs to insert.
            embeddings (torch.Tensor): Embedding values to insert.
        """
        self._hashtable_impl.insert(ids, embeddings)

    def reset(self) -> None:
        """Resets the hashtable to its initial factory state. clears all IDs/embeddings and physically releases underlying memory resources."""
        self._hashtable_impl.reset()

    def clear(self, child=None) -> None:
        """
        Performs a logical clear of stored embeddings or specific child table IDs.

        Preserves underlying memory capacity for fast reuse.
        To physically free memory, use the `reset()` method.

        Args:
            child (str, optional):
                The name of the child table/group to clear IDs from.

                * If None (default), the operation is applied to the
                    entire coalesced hashtable.
                * If a string is provided, the operation is scoped to the
                    specified child-table.
        """
        self._hashtable_impl.clear(child)

    def ids(self, child=None) -> torch.Tensor:
        """
        Get the feature IDs currently stored in the table or a specific child-table.

        Args:
            child (str, optional):
                The name of the child table/group to retrieve IDs from.

                * If None (default), returns IDs from the entire
                    coalesced table.
                * If a string is provided, returns IDs only from the
                    specified child-table.

        Returns:
            torch.Tensor: All feature IDs currently stored (or in the child-table).
        """
        return self._hashtable_impl.ids(child)

    def ids_map(self, child=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the mapping between feature IDs and their internal storage indices
        for the entire table or a specific child-table.

        Args:
            child (str, optional):
                The name of the child table/group to retrieve the mapping from.

                * If None (default), returns the mapping for the entire
                    coalesced table.
                * If a string is provided, returns the mapping for the
                    specified child-table only.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                A tuple containing: (feature_ids, internal_indices).
        """
        return self._hashtable_impl.ids_map(child)

    def embeddings(self, child=None) -> torch.Tensor:
        """
        Retrieves all embedding values currently stored in the table or a specific child-table.

        Args:
            child (str, optional):
                The name of the child table/group to retrieve embeddings from.

                * If None (default), returns embeddings from the entire
                    coalesced table.
                * If a string is provided, returns embeddings only from the
                    specified child-table.

        Returns:
            torch.Tensor: All embedding values currently stored (or in the child-table).
        """
        return self._hashtable_impl.embs(child)

    def embeddings_map(self, child=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the mapping between internal storage indices and their corresponding embedding.

        Args:
            child (str, optional):
                The name of the child table/group to retrieve the mapping from.

                * If None (default), returns the mapping for the entire
                    coalesced table.
                * If a string is provided, returns the mapping for the
                    specified child-table only.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                A tuple containing: (internal_indices, embeddings).
        """
        return self._hashtable_impl.embs_map(child)

    def ids_embeddings(self, child=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the mapping between feature IDs and their associated embedding.

        Args:
            child (str, optional):
                The name of the child table/group to retrieve the mapping from.

                * If None (default), returns the mapping for the entire
                    coalesced table.
                * If a string is provided, returns the mapping for the
                    specified child-table only.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                A tuple containing: (feature_ids, embeddings).
        """
        return self._hashtable_impl.ids_embs(child)

    def snap_shot(self, child=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Captures a complete snapshot of the table's state, including the full
        ID-to-Index-to-Embedding relationship.

        This method retrieves all three primary components: feature IDs, internal
        storage indices, and the embedding vectors themselves.

        Args:
            child (str, optional):
                The name of the child table/group to retrieve the snapshot from.

                * If None (default), captures the snapshot of the entire
                    coalesced table.
                * If a string is provided, captures the snapshot for the
                    specified child-table only.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                A tuple containing: (feature_ids, internal_indices, embeddings).
        """
        return self._hashtable_impl.snap_shot(child)

    def raw_embeddings(self) -> torch.Tensor:
        """Get a copy of the underlying embedding buffer that backs the table.

        Returns:
            torch.Tensor: a copy of the underlying embedding buffer that backs the table.
        Note:
            To retrieve only valid embeddings, use `embeddings()` or `embeddings_map()`.
        """
        return self._hashtable_impl.slot_group().slot_by_name("embedding").value()

    def allocator_id_info(self):
        return self._hashtable_impl.allocator_id_info()

    def id_info(self):
        return self._hashtable_impl.id_info()

    def slot_group(self):
        """Get the slot group for advanced operations.

        Returns:
            SlotGroup: The slot group containing all storage slots.
        """
        return self._hashtable_impl.slot_group()

    def children_info(self):
        """Get information about child hash tables.

        Returns:
            ChildrenInfo: Information about child hash tables.
        """
        return self._hashtable_impl.children_info()

    def __str__(self) -> str:
        """String representation of the hash table.

        Returns:
            str: String representation including the table name.
        """
        return f"HashTable_{self._name}"

    def __repr__(self) -> str:
        """Detailed string representation of the hash table.

        Returns:
            str: Detailed string representation.
        """
        return self.__str__()


class HashTableLookupHelpFunction(torch.autograd.Function):
    """Autograd function for hash table embedding lookup with gradient support.

    This function provides the forward and backward passes for embedding
    lookup operations, handling gradient computation and admission hooks.
    """

    @staticmethod
    def forward(
        ctx,
        ids: torch.Tensor,
        hashtable: object,
        backward_holder: torch.Tensor,
        admit_hook_impl,
    ) -> torch.Tensor:
        """Forward pass for embedding lookup.

        Args:
            ctx: Autograd context for storing information.
            ids (torch.Tensor): Feature IDs to lookup.
            hashtable (torch.classes.recis.HashtableImpl): Hash table implementation.
            backward_holder (torch.Tensor): Tensor for gradient computation.
            admit_hook_impl: Implementation of admission hook.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (indices, embeddings)

        Raises:
            AssertionError: If admit_hook_impl is not None or ReadOnlyHookImpl.
        """
        assert admit_hook_impl is None or isinstance(
            admit_hook_impl, ReadOnlyHookImpl
        ), f"admit hook only support ReadOnlyHook yet, but got: {admit_hook_impl}"
        ids = ids.detach()
        index, embedding = hashtable.embedding_lookup(ids, admit_hook_impl is not None)
        ctx.save_for_backward(index)
        ctx.hashtable = hashtable
        return index.to(device="cuda", non_blocking=True), embedding.to(
            device="cuda", non_blocking=True
        )

    @staticmethod
    def backward(ctx, grad_output_index, grad_output_emb) -> torch.Tensor:
        """Backward pass for embedding lookup.

        This function handles the gradient computation for the hash table
        lookup operation. It aggregates gradients across duplicate IDs and
        passes them to the hash table for storage.

        Args:
            ctx: Autograd context containing forward pass information.
            grad_output_index: Gradient for indices (unused).
            grad_output_emb (torch.Tensor): Gradient for embeddings.

        Returns:
            Tuple: Gradients for all inputs (most are None).
        """
        (index,) = ctx.saved_tensors
        hashtable = ctx.hashtable

        if (
            hasattr(grad_output_emb, "_grad_already_handled")
            and grad_output_emb._grad_already_handled
        ):
            return (None, None, None, None)

        hashtable.accept_grad(
            index.to(device="cuda", non_blocking=True),
            grad_output_emb.to(device="cuda", non_blocking=True),
        )
        return (None, None, None, None)


class GradIDMeanFunction(torch.autograd.Function):
    """Autograd function for gradient aggregation by feature ID.

    This function handles gradient computation when using ID-based
    gradient reduction, ensuring proper gradient flow for duplicate IDs.
    """

    @staticmethod
    def forward(ctx, embedding, index):
        """Forward pass for ID-based gradient aggregation.

        Args:
            ctx: Autograd context.
            embedding (torch.Tensor): Input embeddings.
            index (torch.Tensor): Index mapping for gathering.

        Returns:
            torch.Tensor: Gathered embeddings.
        """
        ctx.save_for_backward(index)
        return torch.ops.recis.gather(index, embedding)

    @staticmethod
    def backward(ctx, grad_outputs):
        """Backward pass for ID-based gradient aggregation.

        Args:
            ctx: Autograd context.
            grad_outputs (torch.Tensor): Output gradients.

        Returns:
            Tuple[torch.Tensor, None]: (reduced_gradients, None)
        """
        grad_outputs = grad_outputs.to(device="cuda", non_blocking=True)
        (index,) = ctx.saved_tensors
        if index.numel() == 0:
            return (
                torch.zeros(
                    [0] + list(grad_outputs.shape)[1:], device=grad_outputs.device
                ),
                None,
            )
        shape = [index.max() + 1] + list(grad_outputs.shape)[1:]
        reduce_grad = torch.zeros(shape, device=grad_outputs.device)
        reduce_grad.index_reduce_(0, index, grad_outputs, "mean", include_self=False)
        return reduce_grad, None


class GradWorkerMeanFunction(torch.autograd.Function):
    """Autograd function for gradient aggregation by worker.

    This function handles gradient computation when using worker-based
    gradient reduction, distributing gradients across multiple workers.
    """

    @staticmethod
    def forward(ctx, embedding, index, slice_num):
        """Forward pass for worker-based gradient aggregation.

        Args:
            ctx: Autograd context.
            embedding (torch.Tensor): Input embeddings.
            index (torch.Tensor): Index mapping for gathering.
            slice_num (torch.Tensor): Number of worker slices.

        Returns:
            torch.Tensor: Gathered embeddings.
        """
        ctx.save_for_backward(index, slice_num)
        ctx.grad_shape = embedding.shape
        return torch.ops.recis.gather(index, embedding)

    @staticmethod
    def backward(ctx, grad_outputs):
        """Backward pass for worker-based gradient aggregation.

        Args:
            ctx: Autograd context.
            grad_outputs (torch.Tensor): Output gradients.

        Returns:
            Tuple[torch.Tensor, None, None]: (reduced_gradients, None, None)
        """
        grad_outputs = grad_outputs.to(device="cuda", non_blocking=True)
        (index, slice_num) = ctx.saved_tensors
        grad_shape = ctx.grad_shape
        if index.numel() == 0:
            return (
                torch.zeros(
                    [0] + list(grad_outputs.shape)[1:], device=grad_outputs.device
                ),
                None,
                None,
            )
        grad_outputs = grad_outputs / slice_num
        reduce_grad = torch.zeros(
            grad_shape,
            dtype=grad_outputs.dtype,
            device=grad_outputs.device,
        )
        reduce_grad.index_add_(0, index, grad_outputs)
        return reduce_grad, None, None


class GradWorkerSumFunction(torch.autograd.Function):
    """Autograd function for gradient sum aggregation by worker.

    This function handles gradient computation when using worker-based
    gradient sum reduction, accumulating gradients across multiple workers
    without averaging. It also computes gradient squared sums for optimizer
    state updates.

    Note:
        This function returns a special tensor with a flag attribute that
        tells HashTableLookupHelpFunction.backward to skip gradient storage,
        as GradWorkerSumFunction.backward handles it directly.
    """

    @staticmethod
    def forward(ctx, embedding, index, hashtable, embedding_index):
        """Forward pass for worker-based gradient sum aggregation.

        Args:
            ctx: Autograd context.
            embedding (torch.Tensor): Input embeddings.
            index (torch.Tensor): Index mapping for gathering.
            hashtable: The HashTable instance to store gradients.

        Returns:
            torch.Tensor: Gathered embeddings.
        """
        ctx.save_for_backward(index, embedding_index)
        ctx.hashtable = hashtable
        return torch.ops.recis.gather(index, embedding)

    @staticmethod
    def backward(ctx, grad_outputs):
        """Backward pass for worker-based gradient sum aggregation.

        Computes both gradient sum and gradient squared sum, storing them
        directly to the hashtable. Returns grad_sum tensor with a flag
        to indicate that gradient has been handled.

        Args:
            ctx: Autograd context.
            grad_outputs (torch.Tensor): Output gradients.

        Returns:
            Tuple[torch.Tensor, None, None]: (grad_sum, None, None)
        """
        grad_outputs = grad_outputs.cuda()
        (index, embedding_index) = ctx.saved_tensors
        hashtable = ctx.hashtable

        if index.numel() == 0:
            grad_sum = torch.zeros(
                [0] + list(grad_outputs.shape)[1:], device=grad_outputs.device
            )
            grad_sq_sum = torch.zeros(
                [0] + list(grad_outputs.shape)[1:], device=grad_outputs.device
            )
            # Store empty gradients to hashtable
            empty_index = torch.zeros([0], dtype=torch.long, device=grad_outputs.device)
            hashtable._hashtable_impl.accept_grad(empty_index, grad_sum)
            hashtable._hashtable_impl.accept_grad_sq(empty_index, grad_sq_sum)
            # Mark gradient as handled
            grad_sum._grad_already_handled = True
            return grad_sum, None, None, None

        # Get unique indices for aggregation
        index_unique, index_reverse = torch.unique(
            index.view((-1,)), return_inverse=True, sorted=False
        )

        # Compute gradient sum
        grad_sum = torch.zeros(
            [index_unique.numel()] + list(grad_outputs.shape)[1:],
            dtype=grad_outputs.dtype,
            device=grad_outputs.device,
        )
        grad_sum.index_add_(0, index_reverse, grad_outputs)

        # Compute gradient squared sum
        grad_sq_outputs = grad_outputs * grad_outputs
        grad_sq_sum = torch.zeros(
            [index_unique.numel()] + list(grad_outputs.shape)[1:],
            dtype=grad_outputs.dtype,
            device=grad_outputs.device,
        )
        grad_sq_sum.index_add_(0, index_reverse, grad_sq_outputs)

        # Store gradients to hashtable
        index_cuda = embedding_index.to(device="cuda")
        hashtable._hashtable_impl.accept_grad(index_cuda, grad_sum)
        hashtable._hashtable_impl.accept_grad_sq(index_cuda, grad_sq_sum)

        # Mark gradient as handled so HashTableLookupHelpFunction.backward skips it
        grad_sum._grad_already_handled = True
        return grad_sum, None, None, None


class GradHDMPGroupSumFunction(torch.autograd.Function):
    """Aggregate sparse grads by HDMP source group for SparseAdagradSum.

    The incoming gradients are aligned with the pre-unique request sequence.
    ``index`` maps each request row to the unique embedding row, while
    ``source_group`` maps each request row to its 32-worker HDMP group.
    """

    @staticmethod
    def forward(
        ctx,
        embedding,
        index,
        source_group,
        hashtable,
        embedding_index,
        group_size,
        num_groups,
        group_reduce_by,
    ):
        ctx.save_for_backward(index, source_group, embedding_index)
        ctx.hashtable = hashtable
        ctx.group_size = group_size
        ctx.num_groups = num_groups
        ctx.group_reduce_by = group_reduce_by
        return torch.ops.recis.gather(index, embedding)

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.cuda()
        index, source_group, embedding_index = ctx.saved_tensors
        hashtable = ctx.hashtable

        if index.numel() == 0:
            grad_sum = torch.zeros(
                [0] + list(grad_outputs.shape)[1:], device=grad_outputs.device
            )
            grad_sq_sum = torch.zeros(
                [0] + list(grad_outputs.shape)[1:], device=grad_outputs.device
            )
            empty_index = torch.zeros([0], dtype=torch.long, device=grad_outputs.device)
            hashtable._hashtable_impl.accept_grad(empty_index, grad_sum)
            hashtable._hashtable_impl.accept_grad_sq(empty_index, grad_sq_sum)
            grad_sum._grad_already_handled = True
            return grad_sum, None, None, None, None, None, None, None

        index = index.to(device=grad_outputs.device, dtype=torch.long)
        source_group = source_group.to(device=grad_outputs.device, dtype=torch.long)
        source_group = source_group.view(-1)
        index = index.view(-1)

        if source_group.numel() != index.numel():
            raise RuntimeError(
                "source_group must align with the original request sequence: "
                f"source_group={source_group.numel()}, index={index.numel()}"
            )

        num_unique = int(embedding_index.numel())
        num_groups = int(ctx.num_groups)
        group_size = int(ctx.group_size)
        group_reduce_by = ctx.group_reduce_by

        if source_group.numel() > 0:
            max_group = int(source_group.max().item())
            if max_group >= num_groups:
                raise RuntimeError(
                    f"source_group contains group {max_group}, "
                    f"but num_groups={num_groups}"
                )

        flat_group_index = source_group * num_unique + index
        grouped_shape = [num_groups * num_unique] + list(grad_outputs.shape)[1:]
        group_grad = torch.zeros(
            grouped_shape, dtype=grad_outputs.dtype, device=grad_outputs.device
        )
        group_grad.index_add_(0, flat_group_index, grad_outputs)

        if group_reduce_by == "id":
            counts = torch.zeros(
                [num_groups * num_unique],
                dtype=grad_outputs.dtype,
                device=grad_outputs.device,
            )
            ones = torch.ones(
                [flat_group_index.numel()],
                dtype=grad_outputs.dtype,
                device=grad_outputs.device,
            )
            counts.index_add_(0, flat_group_index, ones)
            counts = counts.clamp_min_(1).view(
                [num_groups * num_unique] + [1] * (grad_outputs.dim() - 1)
            )
            group_grad = group_grad / counts
        elif group_reduce_by == "worker":
            group_grad = group_grad / group_size

        group_grad = group_grad.view(
            [num_groups, num_unique] + list(grad_outputs.shape)[1:]
        )
        grad_sum = group_grad.sum(dim=0)
        grad_sq_sum = (group_grad * group_grad).sum(dim=0)

        index_cuda = embedding_index.to(device=grad_outputs.device)
        hashtable._hashtable_impl.accept_grad(index_cuda, grad_sum)
        hashtable._hashtable_impl.accept_grad_sq(index_cuda, grad_sq_sum)

        grad_sum._grad_already_handled = True
        return grad_sum, None, None, None, None, None, None, None


def is_hashtable(obj):
    """Check if an object is a hash table.

    Args:
        obj: Object to check.

    Returns:
        bool: True if the object is a hash table, False otherwise.
    """
    return hasattr(obj, "hashtable_tag")


def split_sparse_dense_state_dict(state_dict: dict) -> Tuple[dict, dict]:
    """Split state dictionary into sparse and dense parameters.

    This function separates hash table parameters (sparse) from regular
    tensor parameters (dense) in a model's state dictionary.

    Args:
        state_dict (dict): State dictionary from model.state_dict().
            Format: {"parameter_name": parameter_value}.

    Returns:
        Tuple[dict, dict]: (sparse_state_dict, dense_state_dict)
            - sparse_state_dict: Dictionary containing hash table parameters
            - dense_state_dict: Dictionary containing regular tensor parameters

    Example:
    .. code-block:: python
        model = MyModel()  # Contains both hash tables and regular layers
        state_dict = model.state_dict()

        sparse_params, dense_params = split_sparse_dense_state_dict(state_dict)

        print(f"Sparse parameters: {list(sparse_params.keys())}")
        print(f"Dense parameters: {list(dense_params.keys())}")

    """
    sparse_state_dict = {}
    remove_key = set()
    for key in state_dict:
        value = state_dict[key]
        if value is not None:
            if is_hashtable(value):
                sparse_state_dict[key] = value
                remove_key.add(key)
    for key in remove_key:
        del state_dict[key]
    return sparse_state_dict, state_dict


def filter_out_sparse_param(model: torch.nn.Module) -> dict:
    """Extract sparse parameters from a PyTorch model.

    This function extracts all hash table parameters from a model,
    which is useful for separate handling of sparse parameters in
    distributed training scenarios.

    Args:
        model (torch.nn.Module): PyTorch model containing hash tables.

    Returns:
        dict: Dictionary containing only sparse (hash table) parameters.

    Example:
    .. code-block:: python

        from recis.nn.modules.hashtable import filter_out_sparse_param

        # Separate parameters
        sparse_params = filter_out_sparse_param(model)

        # Create different optimizers
        from recis.optim import SparseAdamW
        from torch.optim import AdamW

        sparse_optimizer = SparseAdamW(sparse_params, lr=0.001)
        dense_optimizer = AdamW(model.parameters(), lr=0.001)

    """
    state_dict = model.state_dict()
    sparse_state_dict, _ = split_sparse_dense_state_dict(state_dict)
    return sparse_state_dict


class HashtableRegister(metaclass=SingletonMeta):
    """Singleton registry for managing hash table instances.

    This class provides a centralized registry for tracking hash table
    instances across the application, ensuring proper management and
    avoiding naming conflicts.

    Attributes:
        _hashtables (dict): Dictionary mapping hash table names to their
            configuration information.

    Example:
    .. code-block:: python

        # Register a hash table (usually done automatically)
        register = HashtableRegister()
        register.register("user_embedding", '{"shape": [64], "dtype": "float32"}')

        # The registry is a singleton, so all instances are the same
        register2 = HashtableRegister()
        assert register is register2  # True

    """

    def __init__(self) -> None:
        """Initialize the hash table registry."""
        self._hashtables = {}
        self._child_name_to_ht = {}

    def get_ht_by_child_name(self, name: str) -> HashTable:
        return self._child_name_to_ht[name]

    def register(self, name: str, info: str, ht: HashTable):
        """Register a hash table with the given name and configuration.

        Args:
            name (str): Unique name for the hash table.
            info (str): JSON string containing hash table configuration.

        Raises:
            ValueError: If a hash table with the same name is already registered
                with different configuration.

        Example:

        .. code-block:: python

            register = HashtableRegister()

            # Register a new hash table
            config = '{"shape": [128], "dtype": "float32", "initializer": "constant"}'
            register.register("item_embedding", config)

            # This would raise ValueError due to duplicate name
            # register.register("item_embedding", different_config)

        """
        if name in self._hashtables:
            raise ValueError(
                f"Duplicate hashtable shard name: {name}, before: {self._hashtables[name]}, now: {info}"
            )
        self._hashtables[name] = info
        self._child_name_to_ht[name] = ht
