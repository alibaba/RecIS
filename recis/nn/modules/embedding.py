import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
import torch.distributed as dist

from recis.monitor.monitor_reporter import (
    EMB_BYTES_NAME,
    ID_SIZE_A2A_TIME_NAME,
    ID_SIZE_NAME,
    REDUCE_EMB_BYTES_NAME,
    UNIQUE_ID_SIZE_NAME,
    MonitorReporter,
)
from recis.nn.functional.embedding_ops import ids_partition
from recis.nn.functional.reduce_function_registry import (
    _COMBINER_REGISTRY_INSTANCE,
    get_combiner,
)
from recis.nn.hashtable_hook import AdmitHook, FilterHook
from recis.nn.initializers import ConstantInitializer, Initializer
from recis.nn.modules.hashtable import HashTable, gen_slice
from recis.ragged.tensor import RaggedTensor
from recis.utils.logger import Logger


logger = Logger(__name__)


class EmbeddingExchange(torch.autograd.Function):
    """Autograd function for distributed embedding exchange across workers.

    This function handles the forward and backward passes for exchanging
    embeddings between different workers in a distributed setting using
    all-to-all communication patterns.
    """

    @staticmethod
    def forward(
        ctx,
        embedding: torch.Tensor,
        parts: List[int],
        parts_reverse: List[int],
        pg: dist.ProcessGroup = None,
    ):
        """Forward pass for embedding exchange.

        Args:
            ctx: Autograd context for storing information for backward pass.
            embedding (torch.Tensor): Input embedding tensor to exchange.
            parts (List[int]): List of partition sizes for input splitting.
            parts_reverse (List[int]): List of partition sizes for output splitting.
            pg (dist.ProcessGroup, optional): Process group for communication.
                Defaults to None (uses default group).

        Returns:
            Tuple containing:
                - torch.Tensor: Exchanged embedding tensor
                - object: Async operation handle for synchronization
                - List[int]: Original partial shape for reconstruction
        """
        ctx.pg = pg
        ctx.parts = parts
        ctx.parts_reverse = parts_reverse
        ctx.origin_partial_shape = list(embedding.shape)
        ctx.origin_partial_shape[0] = -1
        ctx.block_size = None
        if ctx.pg is None:
            ctx.pg = dist.distributed_c10d._get_default_group()
        # exchange embedding
        block_size = math.prod(embedding.shape[1:])
        embedding = embedding.view(-1)
        ctx.block_size = block_size
        input_split_sizes = [block_size * part for part in parts]
        output_split_sizes = [block_size * part for part in parts_reverse]
        output_embedding = torch.empty(
            (sum(output_split_sizes)), dtype=embedding.dtype, device=embedding.device
        )
        emb_await = dist.all_to_all_single(
            output=output_embedding,
            input=embedding,
            input_split_sizes=input_split_sizes,
            output_split_sizes=output_split_sizes,
            group=pg,
            async_op=True,
        )
        return output_embedding, emb_await, ctx.origin_partial_shape

    @staticmethod
    def backward(ctx, grad_output, _1, _2):
        """Backward pass for embedding exchange.

        Args:
            ctx: Autograd context containing forward pass information.
            grad_output (torch.Tensor): Gradient from the next layer.
            _1: Unused gradient argument.
            _2: Unused gradient argument.

        Returns:
            Tuple containing:
                - torch.Tensor: Gradient for embedding input
                - None: No gradient for parts
                - None: No gradient for parts_reverse
                - None: No gradient for pg
        """
        parts = ctx.parts
        reverse_parts = ctx.parts_reverse
        block_size = ctx.block_size
        input_split_sizes = [part * block_size for part in reverse_parts]
        output_split_sizes = [part * block_size for part in parts]
        output_grad = torch.empty(
            (sum(output_split_sizes)),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        dist.all_to_all_single(
            output=output_grad,
            input=grad_output.view(-1),
            input_split_sizes=input_split_sizes,
            output_split_sizes=output_split_sizes,
            group=ctx.pg,
        )
        output = output_grad.view(ctx.origin_partial_shape)
        return output, None, None, None


@dataclass
class ExchangeIDsResults:
    """Data class for storing ID exchange results.

    Attributes:
        ids (torch.Tensor): Exchanged ID tensor.
        parts (List[int]): Original partition sizes.
        parts_reverse (List[int]): Reverse partition sizes.
        reverse_index (torch.Tensor): Index for reversing the exchange.
        offsets (torch.Tensor): Offset tensor for ragged operations.
        ids_await (Optional[object]): Async operation handle. Defaults to None.
    """

    ids: torch.Tensor
    parts: List[int]
    parts_reverse: List[int]
    reverse_index: torch.Tensor
    offsets: torch.Tensor
    ids_await: Optional[object] = None


@dataclass
class NoReduceEmbedding:
    """A2A 完成但未做 segment reduce 的 embedding 结果。

    当 ``EmbeddingOption.no_reduce=True`` 时，对应 feature 在 ``EmbeddingEngine``
    forward 的输出里就是这个 dataclass，用户可以拿到去做自定义聚合（attention、
    MoE、序列建模等）。

    原始输入的 dense shape 不在这里返回——调用方一般可以从 ``offsets``
    （batch 维 = ``offsets.numel() - 1``）或自己持有的输入张量推出。

    Attributes:
        emb (torch.Tensor): all-to-all 后的 unique embedding，形状 ``[N_unique, dim]``。
        reverse_index (torch.Tensor): 把原始 ids 映射到 ``emb`` 行的索引，
            ``emb[reverse_index]`` 即每个原始 id 对应的 embedding。
        offsets (torch.Tensor): 原始输入的 segment 边界，形状 ``[batch_size+1]``。
    """

    emb: torch.Tensor
    reverse_index: torch.Tensor
    offsets: torch.Tensor


@dataclass
class ExchangeEmbResults:
    """Data class for storing embedding exchange results.

    Attributes:
        emb (torch.Tensor): Exchanged embedding tensor.
        reverse_index (torch.Tensor): Index for reversing the exchange.
        emb_shape (List[int]): Original embedding shape.
        offsets (torch.Tensor): Offset tensor for ragged operations.
        emb_await (Optional[object]): Async operation handle. Defaults to None.
    """

    emb: torch.Tensor
    reverse_index: torch.Tensor
    emb_shape: list[int]
    offsets: torch.Tensor
    emb_await: Optional[object] = None


@dataclass
class EmbeddingOption:
    """Configuration class for dynamic embedding parameters.

    This class encapsulates all configuration options for dynamic embedding,
    including dimension settings, device placement, training options, and
    distributed communication parameters.

    Attributes:
        embedding_dim (int): Dimension of embedding vectors. Defaults to 16.
        block_size (int): Block size for hash table storage. Defaults to 10240.
        dtype (torch.dtype): Data type for embeddings. Defaults to torch.float32.
        device (torch.device): Device for computation. Defaults to CPU.
        trainable (bool): Whether embeddings are trainable. Defaults to True.
        pg (dist.ProcessGroup): Process group for distributed training. Defaults to None.
        max_partition_num (int): Maximum partition number. Defaults to 65536.
        shared_name (str): Shared name for embedding table. Defaults to "embedding".
        children (List[str]): List of child embedding names. Defaults to empty list.
        coalesced (Optional[bool]): Whether to use coalesced operations. Defaults to False.
        initializer (Optional[Initializer]): Embedding initializer. Defaults to None.
        use_weight (Optional[bool]): Whether to use weights. Defaults to True.
        combiner (Optional[str]): Combiner type ("sum", "mean", "tile") or function registered with `recis.nn.functional.reduce_function_registry.register_combiner` Defaults to "sum".
        combiner_kwargs (Optional[dict]): Additional combiner arguments. Defaults to None.
        grad_reduce_by (Optional[str]): Gradient reduction strategy. Defaults to "worker".
        filter_hook (Optional[FilterHook]): Filter hook for feature filtering. Defaults to None.
        admit_hook (Optional[AdmitHook]): Admit hook for feature admission. Defaults to None.

    Example:

    .. code-block:: python

        from recis.nn.modules.embedding import EmbeddingOption

        # Basic configuration
        emb_opt = EmbeddingOption(
            embedding_dim=128,
            block_size=2048,
            dtype=torch.float32,
            trainable=True,
            combiner="mean",
        )

        # Advanced configuration with hooks
        emb_opt = EmbeddingOption(
            embedding_dim=64,
            combiner="tile",
            combiner_kwargs={"tile_len": 10},
            filter_hook=my_filter_hook,
            admit_hook=my_admit_hook,
        )

    """

    embedding_dim: int = 16
    block_size: int = 10240
    dtype: torch.dtype = torch.float32
    device: torch.device = torch.device("cpu")
    trainable: bool = True
    pg: dist.ProcessGroup = None
    max_partition_num: int = 65536
    shared_name: str = "embedding"
    children: List[str] = field(default_factory=list)
    coalesced: Optional[bool] = False
    initializer: Optional[Initializer] = None
    use_weight: Optional[bool] = True
    combiner: Optional[str] = "sum"
    combiner_kwargs: Optional[dict] = None
    grad_reduce_by: Optional[str] = "worker"
    filter_hook: Optional[FilterHook] = None
    admit_hook: Optional[AdmitHook] = None
    # Convert embeddings of int8 type to fp16; otherwise, convert them to fp32
    fp16_enabled: bool = False
    # If True, EmbeddingEngine returns the post-all-to-all embedding directly
    # (wrapped in NoReduceEmbedding) without segment reduction or per-feature split.
    # The feature must have a unique shared_name (cannot share a hashtable).
    no_reduce: bool = False

    def __post_init__(self):
        """Post-initialization validation and setup.

        Raises:
            AssertionError: If combiner is not in supported types.
            RuntimeError: If tile combiner is used without proper configuration.
        """
        if not self.children:
            self.children = [self.shared_name]
        if self.initializer is None:
            self.initializer = ConstantInitializer(init_val=0)
        if self.fp16_enabled:
            assert self.dtype in (torch.int8,), "only int8 emb can set fp16_enabled"
        if self.no_reduce:
            return
        assert self.combiner in _COMBINER_REGISTRY_INSTANCE.list_combiners(), (
            f"Combiner '{self.combiner}' not registered. Available combiners: {list(_COMBINER_REGISTRY_INSTANCE.list_combiners())}"
        )
        if self.combiner == "tile":
            if self.combiner_kwargs is None:
                raise RuntimeError("combiner_kwargs must be set when combiner is tile.")
            if "tile_len" not in self.combiner_kwargs:
                raise RuntimeError(
                    "tile_len must be in combiner_kwargs when combiner is tile."
                )

    def coalesced_info(self):
        """Get coalesced configuration information.

        Returns:
            str: JSON string containing coalesced configuration.
        """
        info = {
            "dim": self.embedding_dim,
            "dtype": str(self.dtype),
            "device": str(self.device.type),
            "initializer": str(self.initializer),
            "grad_reduce_by": self.grad_reduce_by,
            "filter_hook": str(self.filter_hook),
        }
        return json.dumps(info)

    def runtime_info(self):
        """Get runtime configuration information.

        Returns:
            str: JSON string containing runtime configuration.
        """
        info = {
            "combiner": self.combiner,
            "use_weight": self.use_weight,
            "trainable": self.trainable,
            "admit_hook": str(self.admit_hook),
            "fp16_enabled": self.fp16_enabled,
        }
        return json.dumps(info)


class DynamicEmbedding(torch.nn.Module):
    """Dynamic embedding module for distributed sparse feature learning.

    This module provides a distributed dynamic embedding table that can
    automatically handle feature admission, eviction, and cross-worker
    communication. It supports both dense tensors and ragged tensors
    for flexible sparse feature handling.

    The module uses a hash table backend for efficient sparse storage
    and supports various combiners (sum, mean, tile) for aggregating
    multiple embeddings per sample.

    Args:
        emb_opt (EmbeddingOption): Configuration options for the embedding.
        pg (dist.ProcessGroup, optional): Process group for distributed
            communication. Defaults to None (uses default group).

    Example:
        Basic usage:

    .. code-block:: python

        from recis.nn import DynamicEmbedding, EmbeddingOption
        from recis.nn.initializers import TruncNormalInitializer

        # Configure embedding options
        emb_opt = EmbeddingOption(
            embedding_dim=64,
            shared_name="user_embedding",
            combiner="sum",
            initializer=TruncNormalInitializer(std=0.01)
        )

        # Create dynamic embedding
        embedding = DynamicEmbedding(emb_opt)

        # Forward propagation
        ids = torch.LongTensor([1, 2, 3, 4])
        emb_output = embedding(ids)


        Advanced usage with ragged tensors:

    .. code-block:: python

        from recis.ragged.tensor import RaggedTensor

        # Create ragged tensor for variable-length sequences
        values = torch.tensor([1, 2, 3, 4, 5, 6, 7])
        offsets = torch.tensor([0, 3, 5, 7])  # Batch boundaries
        ragged_ids = RaggedTensor(values, offsets)

        # Forward pass
        embeddings = embedding(ragged_ids)

    """

    def __init__(
        self,
        emb_opt: EmbeddingOption,
        pg: dist.ProcessGroup = None,
        use_pinned_memory: bool = False,
    ):
        """Initialize dynamic embedding module.

        Args:
            emb_opt (EmbeddingOption): Configuration options for the embedding.
            pg (dist.ProcessGroup, optional): Process group for distributed
                communication. Defaults to None.
            use_pinned_memory (bool, optional): Whether to use pinned memory for
                CPU intermediate tensors to accelerate H2D/D2H transfers.
                Defaults to False. Set to True to enable pinned memory.
        """
        super().__init__()
        self._emb_opt = emb_opt
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))
        self._rank = int(os.environ.get("RANK", 0))
        if pg is None:
            self._pg = dist.distributed_c10d._get_default_group()
        self._cpu_device = torch.device("cpu")
        self._gpu_device = torch.device(int(os.environ.get("LOCAL_RANK", 0)))
        self._hashtable = HashTable(
            embedding_shape=[self._emb_opt.embedding_dim],
            block_size=self._emb_opt.block_size,
            dtype=self._emb_opt.dtype,
            device=self._emb_opt.device,
            initializer=self._emb_opt.initializer,
            children=self._emb_opt.children,
            name=self._emb_opt.shared_name,
            coalesced=self._emb_opt.coalesced,
            slice=gen_slice(shard_index=self._rank, shard_num=self._world_size),
            grad_reduce_by=self._emb_opt.grad_reduce_by,
            filter_hook=self._emb_opt.filter_hook,
            use_pinned_memory=use_pinned_memory,
        )

    @property
    def name(self):
        return self._emb_opt.shared_name

    @property
    def info(self):
        return f"{self._emb_opt.shared_name}_{self._emb_opt.embedding_dim}_{self._emb_opt.dtype}"

    def deal_with_tensor(self, input_tensor: Union[torch.Tensor, RaggedTensor]):
        """Process input tensor to extract values, offsets, weights, and shape.

        This method handles both dense tensors and ragged tensors, extracting
        the necessary components for embedding lookup and aggregation.

        Args:
            input_tensor (Union[torch.Tensor, RaggedTensor]): Input tensor
                containing feature IDs. Can be either a dense tensor or
                a ragged tensor for variable-length sequences.

        Returns:
            Tuple containing:
                - torch.Tensor: Flattened values tensor
                - torch.Tensor: Offsets tensor for segment operations
                - Optional[torch.Tensor]: Weights tensor (None for dense tensors)
                - Tuple: Original shape for output reconstruction

        Raises:
            RuntimeError: If RaggedTensor is not properly padded.
            TypeError: If input tensor type is not supported or is sparse.
        """
        if isinstance(input_tensor, RaggedTensor):
            val = input_tensor.values()
            weight = input_tensor.weight()
            offsets = input_tensor.offsets()[-1]
            shape = input_tensor.real_shape(0, -1)
            if not math.prod(shape) == (offsets.numel() - 1):
                raise RuntimeError(
                    f"RaggedTensor must pad before lookup, got: {input_tensor}"
                )
        elif isinstance(input_tensor, torch.Tensor):
            if input_tensor.is_sparse:
                raise TypeError("RaggedDynamicEmbedding doesn't support sparse ids")
            else:
                val = input_tensor.view(-1)
                bs = input_tensor.shape[0]
                fea_dim = input_tensor.shape[1]
                offsets = torch.arange(bs + 1, device=val.device) * (fea_dim)
                weight = None
                shape = input_tensor.shape[:-1]
        else:
            raise TypeError(
                f"RaggedDynamicEmbedding only support tensor but get {type(input_tensor)}"
            )
        return val, offsets, weight, shape

    def insert(self, input_ids, input_emb):
        """Insert embeddings for specific IDs, Embs to hashtable."""
        bs = input_ids.shape[0]
        input_ids = input_ids.reshape(bs, -1)
        input_emb = input_emb.reshape(bs, -1)

        ids, offsets, _, _ = self.deal_with_tensor(input_ids)
        # exchange ids
        ids_exchange_result: ExchangeIDsResults = self.exchange_ids(ids, offsets)
        ids_exchange_result.ids_await.wait()

        _, unique_to_sorted = torch.sort(ids_exchange_result.reverse_index)
        input_emb = input_emb[unique_to_sorted]

        output_embs, emb_await, _ = EmbeddingExchange.apply(
            input_emb,
            ids_exchange_result.parts,
            ids_exchange_result.parts_reverse,
            self._pg,
        )
        emb_await.wait()
        output_embs = output_embs.reshape(-1, self._emb_opt.embedding_dim)
        self._hashtable.insert(ids_exchange_result.ids, output_embs)

    def forward(
        self,
        input_ids: Union[torch.Tensor, RaggedTensor],
        input_weights=None,
    ):
        """Forward pass of dynamic embedding lookup.

        Performs distributed embedding lookup with the following steps:
        1. Process input tensor to extract IDs, offsets, and weights
        2. Exchange IDs across workers for distributed lookup
        3. Perform embedding lookup and exchange results back
        4. Aggregate embeddings using the specified combiner

        Args:
            input_ids (Union[torch.Tensor, RaggedTensor]): Input feature IDs.
                For dense tensors, shape should be [batch_size, num_features].
                For ragged tensors, supports variable-length sequences.
            input_weights (torch.Tensor, optional): Weights for weighted
                aggregation. Only used when input_ids is a dense tensor.
                Defaults to None.

        Returns:
            torch.Tensor: Aggregated embeddings with shape:
                - For sum/mean combiner: [batch_size, embedding_dim]
                - For tile combiner: [batch_size, tile_len * embedding_dim]

        Example:

        .. code-block:: python

            # Dense tensor input
            ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
            embeddings = embedding(ids)  # Shape: [2, embedding_dim]

            # With weights
            weights = torch.tensor([[0.5, 1.0, 0.8], [1.2, 0.9, 1.1]])
            weighted_embs = embedding(ids, weights)

            # Ragged tensor input
            ragged_ids = RaggedTensor(values, offsets)
            ragged_embs = embedding(ragged_ids)

        """
        # deal with tensor or ragged tensor
        ids, offsets, weight, shape = self.deal_with_tensor(input_ids)
        if input_ids is not RaggedTensor:
            weight = input_weights
        # exchange ids
        ids_exchange_result: ExchangeIDsResults = self.exchange_ids(ids, offsets)
        # lookup && exchange emb
        emb_exchange_result: ExchangeEmbResults = self.lookup_exchange_emb(
            ids_exchange_result, self._emb_opt.admit_hook
        )
        # reduce by segment
        combiner_kwargs = {}
        if self._emb_opt.combiner == "tile":
            combiner_kwargs["tile_len"] = [self._emb_opt.combiner_kwargs["tile_len"]]
            combiner_kwargs["bs"] = [ids.offsets.numel() - 1]
        emb = self.emb_reduce(
            emb_exchange_result, weight, self._emb_opt.combiner, combiner_kwargs
        )
        if self._emb_opt.combiner == "tile":
            emb = emb.view(
                -1, self._emb_opt.combiner_kwargs["tile_len"] * emb.shape[-1]
            )
        else:
            out_shape = shape + (emb.shape[-1],)
            emb = emb.view(out_shape)
        if not self._emb_opt.trainable:
            emb = emb.detach()
        return emb

    def exchange_ids(self, ids: torch.Tensor, offsets: torch.Tensor):
        """Exchange feature IDs across workers for distributed lookup.

        This method partitions the input IDs and exchanges them across workers
        using all-to-all communication, enabling each worker to lookup its
        assigned portion of the embedding table.

        Args:
            ids (torch.Tensor): Flattened feature IDs tensor.
            offsets (torch.Tensor): Offsets tensor for segment operations.

        Returns:
            ExchangeIDsResults: Data class containing exchanged IDs and
                metadata needed for the reverse operation.
        """
        MonitorReporter.report_size(ID_SIZE_NAME, ids, {"recis_ht_name": self.info})
        ids, ids_parts, ids_reverse_index = ids_partition(
            ids, self._emb_opt.max_partition_num, self._world_size
        )
        MonitorReporter.report_size(
            UNIQUE_ID_SIZE_NAME, ids, {"recis_ht_name": self.info}
        )
        # sync all to all: exchange parts num
        ids_parts_reverse = torch.empty_like(ids_parts)
        with MonitorReporter.report_time(
            ID_SIZE_A2A_TIME_NAME, {"recis_ht_name": self.info}
        ):
            dist.all_to_all_single(ids_parts_reverse, ids_parts, group=self._pg)
        ids_parts_reverse = ids_parts_reverse.to(device="cpu")
        ids_parts = ids_parts.to(device="cpu")
        ids_parts_reverse = ids_parts_reverse.tolist()
        ids_parts = ids_parts.tolist()
        output_ids = torch.empty(
            size=[sum(ids_parts_reverse)], dtype=ids.dtype, device=ids.device
        )
        # async all to all: exhange parts ids
        ids_await = dist.all_to_all_single(
            output_ids,
            ids,
            output_split_sizes=ids_parts_reverse,
            input_split_sizes=ids_parts,
            async_op=True,
        )

        return ExchangeIDsResults(
            ids=output_ids,
            ids_await=ids_await,
            parts=ids_parts,
            parts_reverse=ids_parts_reverse,
            reverse_index=ids_reverse_index,
            offsets=offsets,
        )

    def lookup_exchange_emb(
        self, ids_exchange_result: ExchangeIDsResults, admit_hook=None
    ):
        """Perform embedding lookup and exchange results back to original workers.

        This method waits for ID exchange to complete, performs embedding lookup
        using the hash table, and then exchanges the embeddings back to their
        original workers.

        Args:
            ids_exchange_result (ExchangeIDsResults): Results from ID exchange
                containing the IDs to lookup and exchange metadata.
            admit_hook (AdmitHook, optional): Hook for feature admission control.
                Defaults to None.

        Returns:
            ExchangeEmbResults: Data class containing exchanged embeddings and
                metadata needed for aggregation.
        """
        ids_exchange_result.ids_await.wait()
        embedding = self._hashtable(ids_exchange_result.ids, admit_hook)
        embedding_async, emb_await, emb_shape = EmbeddingExchange.apply(
            embedding,
            ids_exchange_result.parts_reverse,
            ids_exchange_result.parts,
            self._pg,
        )
        return ExchangeEmbResults(
            emb=embedding_async,
            emb_await=emb_await,
            reverse_index=ids_exchange_result.reverse_index,
            emb_shape=emb_shape,
            offsets=ids_exchange_result.offsets,
        )

    def wait_exchange_emb(self, emb_exchange_result):
        """Wait for embedding exchange to complete and reshape embeddings.

        Args:
            emb_exchange_result (ExchangeEmbResults): Results from embedding exchange.

        Returns:
            torch.Tensor: Reshaped embedding tensor.
        """
        emb_exchange_result.emb_await.wait()
        emb = emb_exchange_result.emb.view(emb_exchange_result.emb_shape)
        return emb

    def emb_reduce(
        self, emb_exchange_result, weight, combiner, combiner_kwargs, fp16_enable=False
    ):
        """Aggregate embeddings using registry-based combiner selection.

        This method waits for embedding exchange to complete and then performs
        segment-wise reduction using a combiner function selected from the registry.

        Args:
            emb_exchange_result (ExchangeEmbResults): Results from embedding exchange.
            weight (torch.Tensor, optional): Weights for weighted aggregation.
            combiner (str): Combiner type (must be registered in COMBINER_REGISTRY).
            combiner_kwargs (dict): Additional arguments for the combiner.
            fp16_enable (bool): Enable fp16 for int embedding.

        Returns:
            torch.Tensor: Aggregated embeddings.
        """
        emb = self.wait_exchange_emb(emb_exchange_result)
        if emb.dtype in (torch.int8,):
            if fp16_enable:
                emb = emb.to(torch.float16)
            else:
                emb = emb.to(torch.float32)
        MonitorReporter.report_bytes(EMB_BYTES_NAME, emb, {"recis_ht_name": self.info})

        # Get combiner function from registry and execute
        combiner_func = get_combiner(combiner)
        emb = combiner_func(
            emb=emb,
            weight=weight,
            reverse_index=emb_exchange_result.reverse_index,
            offsets=emb_exchange_result.offsets,
            combiner_kwargs=combiner_kwargs,
        )

        MonitorReporter.report_bytes(
            REDUCE_EMB_BYTES_NAME, emb, {"recis_ht_name": self.info}
        )
        return emb
