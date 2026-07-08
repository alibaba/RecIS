import copy
import hashlib
import math
from collections import defaultdict

import torch
from torch import nn

from recis.monitor.monitor_reporter import (
    EMB_ENGINE_NAME,
    SPARSE_FWD_NAME,
    MonitorReporter,
)
from recis.nn.modules.embedding import (
    DynamicEmbedding,
    EmbeddingOption,
    NoReduceEmbedding,
)
from recis.ragged.tensor import RaggedTensor
from recis.utils.logger import Logger


logger = Logger(__name__)


class HashTableCoalescedGroup:
    """Groups embedding options with similar configurations for coalesced operations.

    This class manages multiple embedding options that share similar configurations,
    enabling them to be processed together efficiently. It handles the mapping
    between feature names and their corresponding embedding configurations.

    Attributes:
        _emb_opt (EmbeddingOption): Base embedding option for the group.
        _fea_to_runtime (dict): Mapping from feature names to runtime info.
        _fea_to_encode_id (dict): Mapping from feature names to encode IDs.
        _fea_to_info (dict): Mapping from feature names to detailed info.
        _runtime_info (set): Set of unique runtime information strings.
        _encode_idx (int): Current encode index counter.
        _children (dict): Mapping from shared names to encode IDs.
        _children_info (dict): Additional children information.
        _name (str): Name of the coalesced group.

    Example:

    .. code-block:: python

        group = HashTableCoalescedGroup("user_group")

        # Add embedding options to the group
        user_opt = EmbeddingOption(embedding_dim=64, shared_name="user_emb")
        group.add_option("user_id", user_opt)

        profile_opt = EmbeddingOption(embedding_dim=64, shared_name="profile_emb")
        group.add_option("user_profile", profile_opt)

        # Get embedding info for the group
        group_emb_info = group.embedding_info()

    """

    def __init__(self, name):
        """Initialize coalesced group.

        Args:
            name (str): Name of the coalesced group.
        """
        self._emb_opt = None
        self._fea_to_runtime = {}
        self._fea_to_encode_id = {}
        self._fea_to_info = {}
        self._runtime_info = set()
        self._encode_idx = -1
        self._children = {}
        self._children_info = {}
        self._name = name

    def add_option(self, fea_name, emb_opt):
        """Add an embedding option to the group.

        Args:
            fea_name (str): Name of the feature.
            emb_opt (EmbeddingOption): Embedding option to add.
        """
        if self._emb_opt is None:
            self._emb_opt = copy.deepcopy(emb_opt)
        if emb_opt.shared_name not in self._children:
            self._encode_idx += 1
            self._children[emb_opt.shared_name] = self._encode_idx
        encode_id = self._children[emb_opt.shared_name]
        combiner = emb_opt.combiner
        dim = emb_opt.embedding_dim
        use_weight = emb_opt.use_weight
        combiner_kwargs = emb_opt.combiner_kwargs
        admit_hook = emb_opt.admit_hook
        fp16_enabled = emb_opt.fp16_enabled
        no_reduce = emb_opt.no_reduce
        # A no_reduce feature must own its RuntimeGroupFeature (cannot share
        # the per-sample all-to-all batch with any other feature), so we tag
        # the runtime_key with the feature name to make it unique.
        runtime_key = emb_opt.runtime_info()
        if no_reduce:
            runtime_key = f"{runtime_key}|no_reduce={fea_name}"
        if runtime_key not in self._runtime_info:
            self._runtime_info.add(runtime_key)
        self._fea_to_runtime[fea_name] = runtime_key
        self._fea_to_encode_id[fea_name] = encode_id
        self._fea_to_info[fea_name] = {
            "combiner": combiner,
            "dim": dim,
            "use_weight": use_weight,
            "combiner_kwargs": combiner_kwargs,
            "admit_hook": admit_hook,
            "fp16_enabled": fp16_enabled,
            "no_reduce": no_reduce,
        }

    def runtime_info(self, fea_name):
        """Get runtime information for a feature.

        Args:
            fea_name (str): Name of the feature.

        Returns:
            str: Runtime information string.
        """
        return self._fea_to_runtime[fea_name]

    def encode_id(self, fea_name):
        """Get encode ID for a feature.

        Args:
            fea_name (str): Name of the feature.

        Returns:
            int: Encode ID for the feature.
        """
        return self._fea_to_encode_id[fea_name]

    def children_info(self, fea_name):
        """Get detailed information for a feature.

        Args:
            fea_name (str): Name of the feature.

        Returns:
            dict: Dictionary containing feature configuration details.
        """
        return self._fea_to_info[fea_name]

    def embedding_info(self):
        """Get embedding option for the entire group.

        Returns:
            EmbeddingOption: Embedding option configured for coalesced operations.

        Raises:
            RuntimeError: If no options have been added to the group.
        """
        if self._emb_opt is None:
            raise RuntimeError("HashTableCoalescedGroup has not build any option")
        return EmbeddingOption(
            embedding_dim=self._emb_opt.embedding_dim,
            dtype=self._emb_opt.dtype,
            device=self._emb_opt.device,
            shared_name=self._name,
            children=[
                k for k, v in sorted(self._children.items(), key=lambda item: item[1])
            ],
            coalesced=True,
            initializer=self._emb_opt.initializer,
            grad_reduce_by=self._emb_opt.grad_reduce_by,
            fp16_enabled=self._emb_opt.fp16_enabled,
            filter_hook=self._emb_opt.filter_hook,
        )


class RuntimeGroupFeature:
    """Groups features with the same runtime characteristics for efficient processing.

    This class collects features that have the same combiner, dimension, and other
    runtime properties, enabling them to be processed together in a single operation.
    It handles the coalescing of multiple feature tensors into unified representations.

    Attributes:
        _dim (int): Embedding dimension.
        _combiner (str): Combiner type ("sum", "mean", "tile") or function registered with `recis.nn.functional.reduce_function_registry.register_combiner`.
        _admit_hook: Admission hook for feature control.
        _use_weight (bool): Whether to use weights.
        _offset_dtype (torch.dtype): Data type for offsets.
        _names (List[str]): List of feature names.
        _encode_ids (List[int]): List of encode IDs.
        _ids (List[torch.Tensor]): List of feature ID tensors.
        _weights (List[torch.Tensor]): List of weight tensors.
        _offsets (List[torch.Tensor]): List of offset tensors.
        _max_sizes (List[int]): List of maximum sizes.
        _split_sizes (List[int]): List of split sizes.
        _shapes (List[tuple]): List of output shapes.
        _coalesced_ids (torch.Tensor): Coalesced ID tensor.
        _coalesced_weights (torch.Tensor): Coalesced weight tensor.
        _coalesced_offsets (torch.Tensor): Coalesced offset tensor.
        _combiner_kwargs (dict): Additional combiner arguments.

    Example:

    .. code-block:: python

        # Create runtime group for sum combiner
        group = RuntimeGroupFeature(
            dim=64,
            combiner="sum",
            use_weight=True,
            offset_dtype=torch.int32,
            admit_hook=None,
        )

        # Add features to the group
        user_ids = torch.tensor([[1, 2], [3, 4]])
        group.add_fea("user_id", user_ids, encode_id=0, combiner_kwargs={})

        item_ids = torch.tensor([[10, 20], [30, 40]])
        group.add_fea("item_id", item_ids, encode_id=1, combiner_kwargs={})

        # Coalesce features for efficient processing
        group.coalesce()

        # Access coalesced tensors
        coalesced_ids = group.coalesced_ids()
        coalesced_offsets = group.coalesced_offsets()

    """

    def __init__(
        self,
        dim,
        combiner,
        use_weight,
        offset_dtype,
        admit_hook,
        fp16_enabled,
        no_reduce=False,
        **kwargs,
    ):
        """Initialize runtime group feature.

        Args:
            dim (int): Embedding dimension.
            combiner (str): Combiner type ("sum", "mean", "tile") or function registered with `recis.nn.functional.reduce_function_registry.register_combiner`.
            use_weight (bool): Whether to use weights.
            offset_dtype (torch.dtype): Data type for offsets.
            admit_hook: Admission hook for feature control.
            fp16_enabled: (bool): Whether use fp16 for int emb.
            no_reduce (bool): Whether to skip segment reduction and per-feature
                split. When True, this group must contain exactly one feature.
            **kwargs: Additional keyword arguments.
        """
        self._dim = dim
        self._combiner = combiner
        self._admit_hook = admit_hook
        self._use_weight = use_weight
        self._offset_dtype = offset_dtype
        self._fp16_enabled = fp16_enabled
        self._no_reduce = no_reduce
        # feature names
        self._names = []
        # coalesced ids
        self._encode_ids = []
        # coalesced ragged tensor values
        self._ids = []
        # coalesced ragged tensor weights
        self._weights = []
        # coalesced ragged tensor offsets
        self._offsets = []
        # coalesced ragged tensor max size
        self._max_sizes = []

        # coalesced ragged tensor split size
        self._split_sizes = []
        # coalesced ragged tensor shapes
        self._shapes = []
        self._coalesced_ids = None
        self._coalesced_weights = None
        self._coalesced_offsets = None
        self._combiner_kwargs = {"bs": [], "tile_len": [], "left_pad": []}

    def combiner(self):
        """Get the combiner type.

        Returns:
            str: Combiner type ("sum", "mean", "tile").
        """
        return self._combiner

    def admit_hook(self):
        """Get the admission hook.

        Returns:
            AdmitHook: Admission hook for feature control.
        """
        return self._admit_hook

    def fp16_enabled(self):
        return self._fp16_enabled

    def no_reduce(self):
        """Whether this group skips segment reduction and per-feature split."""
        return self._no_reduce

    def names(self):
        """Get the list of feature names.

        Returns:
            List[str]: List of feature names in the group.
        """
        return self._names

    def shapes(self):
        """Get the list of output shapes.

        Returns:
            List[tuple]: List of output shapes for each feature.
        """
        return self._shapes

    def coalesced_ids(self):
        """Get the coalesced ID tensor.

        Returns:
            torch.Tensor: Coalesced ID tensor containing all feature IDs.
        """
        return self._coalesced_ids

    def coalesced_weights(self):
        """Get the coalesced weight tensor.

        Returns:
            torch.Tensor: Coalesced weight tensor, or None if weights not used.
        """
        return self._coalesced_weights

    def coalesced_offsets(self):
        """Get the coalesced offset tensor.

        Returns:
            torch.Tensor: Coalesced offset tensor for segment operations.
        """
        return self._coalesced_offsets

    def combiner_kwargs(self):
        """Get additional combiner arguments.

        Returns:
            dict: Dictionary containing combiner-specific arguments.
        """
        return self._combiner_kwargs

    def split_size(self):
        """Get the list of split sizes.

        Returns:
            List[int]: List of split sizes for each feature.
        """
        return self._split_sizes

    def clear_ids(self):
        """Clear coalesced ID and offset tensors to free memory."""
        self._coalesced_ids = None
        self._coalesced_offsets = None

    def _format_tensor(self, input_tensor, combiner, dim, use_weight, combiner_kwargs):
        """Format input tensor for processing.

        This method processes both dense tensors and ragged tensors, extracting
        the necessary components for embedding lookup and aggregation.

        Args:
            input_tensor (Union[torch.Tensor, RaggedTensor]): Input tensor.
            combiner (str): Combiner type.
            dim (int): Embedding dimension.
            use_weight (bool): Whether to use weights.
            combiner_kwargs (dict): Additional combiner arguments.

        Returns:
            Tuple containing:
                - torch.Tensor: Values tensor
                - torch.Tensor: Weight tensor (or None)
                - torch.Tensor: Offsets tensor
                - int: Maximum size
                - int: Split size
                - tuple: Dense shape
                - dict: Updated combiner kwargs

        Raises:
            RuntimeError: If RaggedTensor is not properly padded.
            TypeError: If input tensor type is not supported.
        """
        combiner_kwargs = copy.copy(combiner_kwargs)
        if isinstance(input_tensor, RaggedTensor):
            val = input_tensor.values()
            weight = input_tensor.weight()
            offsets = input_tensor.offsets()[-1]
            max_size = val.numel()
            split_size = math.prod(input_tensor.real_shape(0, -1))
            if not split_size == (offsets.numel() - 1):
                raise RuntimeError(
                    f"RaggedTensor must pad before lookup, got: {input_tensor}"
                )
            if combiner == "tile":
                split_size *= combiner_kwargs["tile_len"]
                shape = (
                    input_tensor.real_shape(0, 1)[0],
                    combiner_kwargs["tile_len"],
                    dim,
                )
                combiner_kwargs["bs"] = input_tensor.shape[0]
            else:
                shape = input_tensor.real_shape(0, -1) + (dim,)
        elif isinstance(input_tensor, torch.Tensor):
            if input_tensor.dim() > 2:
                input_tensor_tmp = input_tensor.view(-1, input_tensor.shape[-1])
            else:
                input_tensor_tmp = input_tensor
            if input_tensor.is_sparse:
                raise TypeError("EmbeddingEngine doesn't support sparse ids")
            val = input_tensor.view(-1)
            weight = None
            bs = input_tensor_tmp.shape[0]
            fea_dim = input_tensor_tmp.shape[1]
            offsets = torch.arange(
                0,
                (bs + 1) * fea_dim,
                step=fea_dim,
                dtype=self._offset_dtype,
                device=val.device,
            )
            max_size = val.numel()
            split_size = math.prod(input_tensor.shape[:-1])
            if combiner == "tile":
                split_size *= combiner_kwargs["tile_len"]
                shape = (input_tensor.shape[0], dim * combiner_kwargs["tile_len"])
                combiner_kwargs["bs"] = input_tensor.shape[0]
            else:
                shape = input_tensor.shape[:-1] + (dim,)
        else:
            raise TypeError(
                f"EmbeddingEngine only support tensor but get {type(input_tensor)}"
            )
        if not use_weight:
            weight = None
        else:
            if weight is None:
                if self._fp16_enabled:
                    weight = torch.ones_like(val, dtype=torch.float16)
                else:
                    weight = torch.ones_like(val, dtype=torch.float32)
        offsets = offsets.to(self._offset_dtype)

        return val, weight, offsets, max_size, split_size, shape, combiner_kwargs

    def add_fea(self, fea_name, fea, encode_id, combiner_kwargs):
        """Add a feature to the runtime group.

        Args:
            fea_name (str): Name of the feature.
            fea (Union[torch.Tensor, RaggedTensor]): Feature tensor.
            encode_id (int): Encode ID for the feature.
            combiner_kwargs (dict): Additional combiner arguments.
        """
        self._names.append(fea_name)
        self._encode_ids.append(encode_id)
        ids, weight, offset, max_size, split_size, dense_shape, combiner_kwargs = (
            self._format_tensor(
                fea, self._combiner, self._dim, self._use_weight, combiner_kwargs
            )
        )
        self._ids.append(ids)
        self._weights.append(weight)
        self._offsets.append(offset)
        self._max_sizes.append(max_size)
        self._split_sizes.append(split_size)
        self._shapes.append(dense_shape)
        if self._combiner == "tile":
            self._combiner_kwargs["bs"].append(combiner_kwargs["bs"])
            self._combiner_kwargs["tile_len"].append(combiner_kwargs["tile_len"])
            if isinstance(fea, RaggedTensor) and fea.pad_info is not None:
                self._combiner_kwargs["left_pad"].append(fea.pad_info.pad_sides.view(1))
            else:
                self._combiner_kwargs["left_pad"].append(
                    torch.zeros([1], dtype=torch.bool, device=fea.device)
                )

    def clear_child(self):
        """Clear child data to free memory."""
        # coalesced ids
        self._encode_ids = []
        # coalesced ragged tensor values
        self._ids = []
        # coalesced ragged tensor offsets
        self._offsets = []
        # coalesced ragged tensor max size
        self._max_sizes = []
        # coalesced ragged tensor weights
        self._weights = []

    def coalesce(self):
        """Coalesce all features in the group into unified tensors.

        This method combines all individual feature tensors into coalesced
        representations that can be processed efficiently in a single operation.
        """
        merge_id = torch.ops.recis.ids_encode(
            self._ids,
            torch.tensor(
                self._encode_ids, dtype=torch.int64, device=self._ids[0].device
            ),
        )
        merge_offset = torch.ops.recis.merge_offsets(
            self._offsets,
            torch.tensor(self._max_sizes, dtype=self._offsets[0].dtype, device="cpu"),
        )
        merge_weight = (
            None if self._weights[0] is None else torch.cat(self._weights, dim=0)
        )
        self._coalesced_ids = merge_id
        self._coalesced_weights = merge_weight
        self._coalesced_offsets = merge_offset
        self.clear_child()


class EmbeddingEngine(nn.Module):
    """Embedding engine for efficient batch processing of multiple embeddings.

    This module provides a high-level interface for managing and processing
    multiple embedding tables efficiently. It automatically groups embeddings
    with similar configurations and processes them using coalesced operations
    to minimize communication overhead in distributed training.

    Key features:
        - Automatic grouping of similar embedding configurations
        - Coalesced operations for improved performance
        - Support for mixed tensor types (dense and ragged)
        - Flexible feature routing (embedding vs. pass-through)
        - Memory-efficient processing pipeline

    Args:
        emb_options (dict[str, EmbeddingOption]): Dictionary mapping feature
            names to their embedding options.

    Example:
        Multi-embedding scenario:

    .. code-block:: python

        import torch
        from recis.nn.modules.embedding import EmbeddingOption
        from recis.nn.modules.embedding_engine import EmbeddingEngine

        # Define embedding options for different features
        emb_options = {
            "user_id": EmbeddingOption(
                embedding_dim=128,
                shared_name="user_embedding",
                combiner="sum",
                trainable=True,
            ),
            "item_id": EmbeddingOption(
                embedding_dim=128,
                shared_name="item_embedding",
                combiner="sum",
                trainable=True,
            ),
            "category": EmbeddingOption(
                embedding_dim=64,
                shared_name="category_embedding",
                combiner="mean",
                trainable=True,
            ),
        }

        # Create embedding engine
        engine = EmbeddingEngine(emb_options)

        # Prepare input features
        batch_size = 32
        features = {
            "user_id": torch.randint(0, 10000, (batch_size, 1)),
            "item_id": torch.randint(0, 50000, (batch_size, 5)),  # Multi-hot
            "category": torch.randint(0, 100, (batch_size, 1)),
            "other_feature": torch.randn(batch_size, 10),  # Pass-through
        }

        # Forward pass
        embeddings = engine(features)

        # Results contain embeddings for configured features
        # and pass-through for non-embedding features
        print(embeddings["user_id"].shape)  # [32, 128]
        print(embeddings["item_id"].shape)  # [32, 128] (summed)
        print(embeddings["category"].shape)  # [32, 64]
        print(embeddings["other_feature"].shape)  # [32, 10] (pass-through)


        Advanced usage with ragged tensors:

    .. code-block:: python

        from recis.ragged.tensor import RaggedTensor

        # Variable-length sequences
        values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        offsets = torch.tensor([0, 3, 5, 8])  # Batch boundaries
        ragged_features = RaggedTensor(values, offsets)

        features = {
            "sequence_ids": ragged_features,
            "user_id": torch.tensor([[100], [200], [300]]),
        }

        embeddings = engine(features)

    """

    def __init__(self, emb_options: dict[str, EmbeddingOption]):
        """Initialize embedding engine with multiple embedding options.

        Args:
            emb_options (dict[str, EmbeddingOption]): Dictionary mapping feature
                names to their embedding configurations.

        Raises:
            RuntimeError: If embedding options have conflicting configurations
                for the same shared name.
        """
        super().__init__()
        self._ht = nn.ModuleDict()
        self._fea_group = {}
        self._fea_to_ht = {}
        self._fea_to_group = {}
        self._emb_opts = emb_options
        self._offset_dtype = torch.int32
        tmp_ht_to_coalesced = {}  # check hashtable
        for fea_name, emb_opt in emb_options.items():
            ht_name = f"CoalescedHashtable_{hashlib.sha256(emb_opt.coalesced_info().encode()).hexdigest()}"
            if emb_opt.shared_name not in tmp_ht_to_coalesced:
                tmp_ht_to_coalesced[emb_opt.shared_name] = ht_name
            elif not tmp_ht_to_coalesced[emb_opt.shared_name] == ht_name:
                raise RuntimeError(
                    f"Create embedding failed, emb sahred name already created by info: {self._fea_group[ht_name]._emb_opt.coalesced_info()}, current: {emb_opt.coalesced_info()}"
                )
            if ht_name not in self._fea_group:
                self._fea_group[ht_name] = HashTableCoalescedGroup(ht_name)
            self._fea_group[ht_name].add_option(fea_name, emb_opt)
            self._fea_to_ht[fea_name] = ht_name
            self._fea_to_group[fea_name] = self._fea_group[ht_name]

        for ht_name, fea_group in self._fea_group.items():
            self._ht[ht_name] = DynamicEmbedding(fea_group.embedding_info())
            logger.info(
                f"ht name: {ht_name}, coalesced info: {fea_group.embedding_info().coalesced_info()}, children: {fea_group.embedding_info().children}"
            )

    @MonitorReporter.report_time_wrapper(
        SPARSE_FWD_NAME, {"recis_emb_phase": EMB_ENGINE_NAME}
    )
    def forward(self, input_features: dict[str, torch.Tensor]):
        """Forward pass for batch embedding processing.

        This method processes multiple features efficiently by:
        1. Grouping features by their runtime characteristics
        2. Performing coalesced ID exchange across workers
        3. Looking up embeddings in batches
        4. Reducing embeddings using specified combiners
        5. Splitting results back to individual features

        Args:
            input_features (dict[str, torch.Tensor]): Dictionary mapping feature
                names to their input tensors. Features not in embedding options
                will be passed through unchanged.

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping feature names to their
                processed outputs. Embedding features return embedding tensors,
                while non-embedding features are passed through.

        Example:

        .. code-block:: python

            features = {
                "user_id": torch.tensor([[1, 2], [3, 4]]),
                "item_id": torch.tensor([[10], [20]]),
                "raw_feature": torch.randn(2, 5),  # Pass-through
            }

            outputs = engine(features)
            # outputs["user_id"]: embedding tensor [2, embedding_dim]
            # outputs["item_id"]: embedding tensor [2, embedding_dim]
            # outputs["raw_feature"]: original tensor [2, 5]

        """
        group_features, direct_outs = self.group_features(input_features)
        group_exchange_ids = self.group_exchange_ids(group_features)
        group_exchange_embs = self.group_exchange_embs(
            group_exchange_ids, group_features
        )
        del group_exchange_ids
        group_embs = self.group_reduce(group_exchange_embs, group_features)
        del group_exchange_embs
        emb_outs = self.split_group_embs(group_embs, group_features)
        del group_embs, group_features
        direct_outs = self.format_direct_out(direct_outs)
        emb_outs.update(direct_outs)
        return emb_outs

    def format_direct_out(self, ori_outs):
        """Format direct output features (non-embedding features).

        Args:
            ori_outs (dict): Dictionary of original output features.

        Returns:
            dict: Dictionary of formatted output features.

        Raises:
            TypeError: If sparse tensors are encountered.
        """
        outs = {}
        for k, v in ori_outs.items():
            if isinstance(v, RaggedTensor) and (not v._values.dtype == torch.int8):
                outs[k] = v.to_dense()
            elif isinstance(v, torch.Tensor):
                if v.is_sparse:
                    raise TypeError("EmbeddingEngine doesn't support sparse tensor")
                outs[k] = v
            else:
                outs[k] = v
        return outs

    def group_features(self, input_dict: dict[str, torch.Tensor]):
        """Group input features by their runtime characteristics.

        This method separates embedding features from pass-through features
        and groups embedding features by their runtime properties (combiner,
        dimension, etc.) for efficient batch processing.

        Args:
            input_dict (dict[str, torch.Tensor]): Input feature dictionary.

        Returns:
            Tuple containing:
                - dict: Grouped features for embedding processing
                - dict: Direct output features (pass-through)
        """
        group_features = defaultdict(dict)
        direct_out = {}
        for fea_name, fea_tensor in input_dict.items():
            if fea_name not in self._fea_to_ht:
                direct_out[fea_name] = fea_tensor
            else:
                ht_name = self._fea_to_ht[fea_name]
                runtime_info = self._fea_to_group[fea_name].runtime_info(fea_name)
                if runtime_info not in group_features[ht_name]:
                    group_features[ht_name][runtime_info] = RuntimeGroupFeature(
                        **(self._fea_to_group[fea_name].children_info(fea_name)),
                        offset_dtype=self._offset_dtype,
                    )
                group_features[ht_name][runtime_info].add_fea(
                    fea_name,
                    fea_tensor,
                    self._fea_to_group[fea_name].encode_id(fea_name),
                    self._fea_to_group[fea_name].children_info(fea_name)[
                        "combiner_kwargs"
                    ],
                )
        return group_features, direct_out

    def group_exchange_ids(self, group_features):
        """Exchange feature IDs across workers for distributed lookup.

        Args:
            group_features (dict): Grouped features for processing.

        Returns:
            dict: Dictionary containing exchange ID results for each group.
        """
        group_exchange_ids = defaultdict(dict)
        for ht_name, group_fea in group_features.items():
            ht = self._ht[ht_name]
            for run_name, run_fea in group_fea.items():
                run_fea.coalesce()
                group_exchange_ids[ht_name][run_name] = ht.exchange_ids(
                    run_fea.coalesced_ids(), run_fea.coalesced_offsets()
                )
                run_fea.clear_ids()
        return group_exchange_ids

    def group_exchange_embs(self, group_exchange_ids, group_features):
        """Perform embedding lookup and exchange results back.

        Args:
            group_exchange_ids (dict): Exchange ID results from previous step.
            group_features (dict): Grouped features for processing.

        Returns:
            dict: Dictionary containing exchange embedding results for each group.
        """
        group_exchange_embs = defaultdict(dict)
        for ht_name, exchange_ids in group_exchange_ids.items():
            ht = self._ht[ht_name]
            for run_name, run_exchange_ids in exchange_ids.items():
                group_exchange_embs[ht_name][run_name] = ht.lookup_exchange_emb(
                    run_exchange_ids,
                    group_features[ht_name][run_name].admit_hook(),
                )
        return group_exchange_embs

    def group_reduce(self, group_exchange_embs, group_features):
        """Reduce embeddings using specified combiners.

        For groups whose feature is configured with ``no_reduce=True``, this
        method skips the segment reduction and instead waits on the embedding
        all-to-all and returns the raw post-a2a tensor. The caller is expected
        to detect and forward these as :class:`NoReduceEmbedding` in
        ``split_group_embs``.

        Args:
            group_exchange_embs (dict): Exchange embedding results.
            group_features (dict): Grouped features for processing.

        Returns:
            dict: Dictionary containing reduced embeddings (or raw post-a2a
                tensors for no_reduce groups) for each group.
        """
        group_embs = defaultdict(dict)
        for ht_name, exchange_emb in group_exchange_embs.items():
            group_fea = group_features[ht_name]
            ht = self._ht[ht_name]
            for run_name in exchange_emb.keys():
                run_fea = group_fea[run_name]
                if run_fea.no_reduce():
                    exch = exchange_emb[run_name]
                    emb = ht.wait_exchange_emb(exch)
                    group_embs[ht_name][run_name] = NoReduceEmbedding(
                        emb=emb,
                        reverse_index=exch.reverse_index,
                        offsets=exch.offsets,
                    )
                else:
                    group_embs[ht_name][run_name] = ht.emb_reduce(
                        exchange_emb[run_name],
                        run_fea.coalesced_weights(),
                        run_fea.combiner(),
                        run_fea.combiner_kwargs(),
                        run_fea.fp16_enabled(),
                    )
        return group_embs

    def split_group_embs(self, group_embs, group_features):
        """Split group embeddings back to individual feature embeddings.

        For no_reduce groups (which always contain exactly one feature) the
        :class:`NoReduceEmbedding` is forwarded as-is, with no splitting or
        reshape applied to the post-a2a tensor.

        Args:
            group_embs (dict): Reduced group embeddings.
            group_features (dict): Grouped features for processing.

        Returns:
            dict[str, Union[torch.Tensor, NoReduceEmbedding]]: Dictionary mapping
                feature names to their individual outputs.
        """
        emb_outs = {}
        for ht_name, group_emb in group_embs.items():
            group_fea = group_features[ht_name]
            for run_name, run_emb in group_emb.items():
                run_fea = group_fea[run_name]
                if run_fea.no_reduce():
                    name = run_fea.names()[0]
                    if not self._emb_opts[name].trainable:
                        run_emb.emb = run_emb.emb.detach()
                    emb_outs[name] = run_emb
                    continue
                emb_list = list(
                    torch.split(run_emb, run_fea.split_size(), dim=0)
                )

                for name, emb, out_shape in zip(
                    *(
                        run_fea.names(),
                        emb_list,
                        run_fea.shapes(),
                    )
                ):
                    if not self._emb_opts[name].trainable:
                        emb = emb.detach()
                    emb_outs[name] = emb.view(out_shape)
        return emb_outs
