import json
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
        filter_hook: Optional[FilterHook] = None,
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
            grad_reduce_by (str, optional): Gradient reduction. Defaults to "worker".
            filter_hook (Optional[FilterHook], optional): Filter hook. Defaults to None.

        Raises:
            AssertionError: If grad_reduce_by is not "id" or "worker".
        """
        super().__init__()
        if initializer is None:
            self._initializer = ConstantInitializer(init_val=0)
        else:
            self._initializer = initializer

        if children is None:
            children = [name]
        self._device = device
        assert grad_reduce_by in ["id", "worker"]
        self._grad_reduce_by = grad_reduce_by
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

    def forward(self, ids: torch.Tensor, admit_hook: AdmitHook = None) -> torch.Tensor:
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
            else:
                slice_num = torch.scalar_tensor(self._worker_num)
                embedding = GradWorkerMeanFunction.apply(embedding, index, slice_num)
            self._filter_hook_impl(emb_idx)
        else:
            ids = ids.detach()
            _, embedding = self._hashtable_impl.embedding_lookup(ids, True)
            embedding = embedding.cuda()
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

    def grad(self, acc_step=1) -> torch.Tensor:
        """Get accumulated gradients.

        Args:
            acc_step (int, optional): Accumulation step. Defaults to 1.

        Returns:
            torch.Tensor: Accumulated gradients.
        """
        return self._hashtable_impl.grad(acc_step)

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
        return index.to(device="cuda"), embedding.to(device="cuda")

    @staticmethod
    def backward(ctx, grad_output_index, grad_output_emb) -> torch.Tensor:
        """Backward pass for embedding lookup.

        Args:
            ctx: Autograd context containing forward pass information.
            grad_output_index: Gradient for indices (unused).
            grad_output_emb (torch.Tensor): Gradient for embeddings.

        Returns:
            Tuple: Gradients for all inputs (most are None).
        """
        (index,) = ctx.saved_tensors
        ctx.hashtable.accept_grad(
            index.to(device="cuda"),
            grad_output_emb.to(device="cuda"),
        )
        return (None, None, None, None, None)


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
        grad_outputs = grad_outputs.cuda()
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
        grad_outputs = grad_outputs.cuda()
        (index, slice_num) = ctx.saved_tensors
        if index.numel() == 0:
            return (
                torch.zeros(
                    [0] + list(grad_outputs.shape)[1:], device=grad_outputs.device
                ),
                None,
                None,
            )
        grad_outputs = grad_outputs / slice_num
        index_unique, index_reverse = torch.unique(
            index.view((-1,)), return_inverse=True, sorted=False
        )
        reduce_grad = torch.zeros(
            [index_unique.numel()] + list(grad_outputs.shape)[1:],
            dtype=grad_outputs.dtype,
            device=grad_outputs.device,
        )
        reduce_grad.index_add_(0, index_reverse, grad_outputs)
        return reduce_grad, None, None


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
