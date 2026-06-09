from collections import defaultdict
from typing import List, Optional

import torch

from recis.common.singleton import SingletonMeta
from recis.nn.functional.fused_ops import (
    fused_bucketize_gpu,
    fused_int64_to_string_int8,
    fused_multi_hash,
    fused_number_mask,
    fused_string_mask,
    fused_uint64_mod_gpu,
)
from recis.nn.functional.hash_ops import (
    djb2hash,
    ev_farmhash,
    farmhash,
    murmurhash,
    sdbmhash,
)
from recis.nn.functional.ragged_ops import (
    fused_ragged_cutoff_2D,
    fused_ragged_cutoff_3D,
)
from recis.ragged.tensor import RaggedPadInfo, RaggedTensor
from recis.utils.logger import Logger

from .op import (
    Bucketize,
    DataValueProcessor,
    Hash,
    IDMultiHash,
    Mask,
    Mod,
    SequenceTruncate,
    StringMultiHash,
)


logger = Logger(__name__)


class FusedOpFactory(metaclass=SingletonMeta):
    """Factory for registering and managing fused operations.

    This singleton factory manages the mapping between regular operations
    and their fused counterparts, enabling automatic fusion optimization
    during feature processing.

    Attributes:
        _ops (dict): Mapping of operation class names to operation classes.
        _fused_ops (dict): Mapping of operation class names to fused operation classes.
    """

    def __init__(self) -> None:
        """Initialize the factory with empty operation mappings."""
        self._ops = {}
        self._fused_ops = {}

    @staticmethod
    def register(op_class: type, fused_op_class: type):
        """Register a mapping between an operation and its fused implementation.

        Args:
            op_class (type): The original operation class.
            fused_op_class (type): The fused implementation class.

        Raises:
            ValueError: If the operation class is already registered.
        """
        logger.info(
            f"FusedOpFactory register {op_class.__name__} -> {fused_op_class.__name__}"
        )
        factory = FusedOpFactory()
        if op_class.__name__ in factory._ops:
            raise ValueError(f"OP '{op_class}' already exists.")
        factory._ops[op_class.__name__] = op_class
        factory._fused_ops[op_class.__name__] = fused_op_class

    @staticmethod
    def contains(op_class: object):
        factory = FusedOpFactory()
        return type(op_class).__name__ in factory._ops

    @staticmethod
    def get_fused_op(op_class: object):
        factory = FusedOpFactory()
        return factory._fused_ops[type(op_class).__name__]


class _FusedOP:
    """Base class for all fused operations.

    Fused operations batch multiple similar operations together to improve
    performance by reducing kernel launch overhead and memory access patterns.
    """

    def __init__(self):
        """Initialize the fused operation with an empty operation list."""
        self._ops = []

    def add_op(self, op):
        """Add an operation to the fused operation batch.

        Args:
            op: The operation to add to the batch.
        """
        self._ops.append(op)

    def process(self, inputs: List):
        """Process a batch of inputs using the fused operation.

        Args:
            inputs (List): List of input tensors to process.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Make the fused operation callable.

        Returns:
            The result of calling process() with the given arguments.
        """
        return self.process(*args, **kwargs)


class FusedBoundaryOP(_FusedOP):
    """Fused implementation for boundary (bucketization) operations.

    This class batches multiple boundary operations together, precomputing
    boundaries on GPU for efficient processing.

    Attributes:
        _cached_boundaries (List[torch.Tensor]): Precomputed boundaries on GPU.
    """

    def __init__(self, ops: Optional[List[Bucketize]] = None):
        """Initialize the fused boundary operation.

        Args:
            ops (Optional[List[Bucketize]]): Initial list of boundary operations.
        """
        super().__init__()
        self._ops = ops if ops else []
        for op in self._ops:
            assert isinstance(op, Bucketize)
        self._cached_boundaries = None
        self._precompute_boundaries()

    def _precompute_boundaries(self):
        if self._ops:
            self._cached_boundaries = [op.boundary.cuda() for op in self._ops]

    def add_op(self, op):
        super().add_op(op)
        self._precompute_boundaries()

    def process(self, inputs: List):
        """Process inputs using fused bucketization.

        Args:
            inputs (List): List of input tensors to bucketize.

        Returns:
            List: List of bucketized output tensors.
        """
        outputs = []
        data_processors = [DataValueProcessor(data) for data in inputs]
        # TODO(yuhuan.zh) force change bucketize inputs to float32
        fused_inputs = [dp.get_input_value().float() for dp in data_processors]
        fused_outputs = fused_bucketize_gpu(fused_inputs, self._cached_boundaries)
        for dp, output in zip(data_processors, fused_outputs):
            outputs.append(dp.get_output_value(output))
        return outputs


class FusedModOP(_FusedOP):
    """Fused implementation for unsigned 64-bit modulo operations.

    This class batches multiple modulo operations together for efficient
    GPU processing.

    Attributes:
        _cached_mods (List[int]): Cached modulo values for batch processing.
    """

    def __init__(self, ops: Optional[List[Mod]] = None):
        """Initialize the fused modulo operation.

        Args:
            ops (Optional[List[Mod]]): Initial list of modulo operations.
        """
        super().__init__()
        self._ops = ops if ops else []
        for op in self._ops:
            assert isinstance(op, Mod)
        self._cached_mods = [op.mod_value for op in self._ops]

    def add_op(self, op: Mod):
        super().add_op(op)
        self._cached_mods.append(op.mod_value)

    def process(self, inputs: List):
        """Process inputs using fused modulo operations.

        Args:
            inputs (List): List of input tensors for modulo operations.

        Returns:
            List: List of output tensors after modulo operations.
        """
        outputs = []
        data_processors = [DataValueProcessor(data) for data in inputs]
        fused_inputs = [dp.get_input_value() for dp in data_processors]
        fused_outputs = fused_uint64_mod_gpu(fused_inputs, self._cached_mods)
        for dp, output in zip(data_processors, fused_outputs):
            outputs.append(dp.get_output_value(output))
        return outputs


class FusedHashOP(_FusedOP):
    """Fused implementation for hash operations.

    This class groups hash operations by hash type (farm, murmur) and
    processes them in batches for improved performance.

    Attributes:
        _hash_func (dict): Mapping of hash type names to hash functions.
        _hash_type_to_indices (defaultdict): Grouping of operations by hash type.
    """

    def __init__(self, ops: Optional[List[Hash]] = None):
        """Initialize the fused hash operation.

        Args:
            ops (Optional[List[Hash]]): Initial list of hash operations.
        """
        super().__init__()
        self._ops = ops if ops else []
        for op in self._ops:
            assert isinstance(op, Hash)
        self._hash_func = {
            "farm": farmhash,
            "ev_farm": ev_farmhash,
            "murmur": murmurhash,
        }
        self._precompute_grouping()

    def _precompute_grouping(self):
        self._hash_type_to_indices = defaultdict(list)
        for i, op in enumerate(self._ops):
            self._hash_type_to_indices[op.hash_type].append(i)

    def add_op(self, op: Hash):
        super().add_op(op)
        self._precompute_grouping()

    def process(self, inputs: List):
        """Process inputs using fused hash operations grouped by hash type.

        Args:
            inputs (List): List of RaggedTensor inputs for hashing.

        Returns:
            List: List of RaggedTensor outputs with hashed values.
        """
        outputs = [None] * len(inputs)
        for hash_type, indices in self._hash_type_to_indices.items():
            fused_inputs = []
            fused_offsets = []

            for i in indices:
                data = inputs[i]
                fused_inputs.append(data.values())
                fused_offsets.append(data.offsets()[-1].int())

            splits = fused_offsets
            sub_outputs = self._hash_func[hash_type](fused_inputs, splits)

            for j, i in enumerate(indices):
                outputs[i] = RaggedTensor(
                    values=sub_outputs[j],
                    offsets=inputs[i].offsets()[:-1],
                    weight=inputs[i].weight(),
                    dense_shape=inputs[i]._dense_shape[:-1],
                )
        return outputs


class FusedCutoffOP(_FusedOP):
    """Fused implementation for sequence processing (cutoff) operations.

    This class groups sequence processing operations by data type and dimensions,
    then processes them in batches for improved performance.

    Attributes:
        _cache_groups_tensor (dict): Cached tensors for each operation group.
        _groups (defaultdict): Grouping of operations by (dtype, n_dims).
        _group_params (dict): Parameters for each operation group.
    """

    def __init__(self, ops: Optional[List[SequenceTruncate]] = None):
        """Initialize the fused cutoff operation.

        Args:
            ops (Optional[List[SequenceTruncate]]): Initial list of sequence operations.
        """
        super().__init__()
        self._ops = ops if ops else []
        for op in self._ops:
            assert isinstance(op, SequenceTruncate)
        self._cache_groups_tensor = {}
        self._precompute_groups_and_params()

    def _precompute_groups_and_params(self):
        self._groups = defaultdict(list)
        self._group_params = {}
        for i, op in enumerate(self._ops):
            group_key = (op.dtype, op.n_dims)
            self._groups[group_key].append(
                {
                    "op": op,
                    "index": i,
                    "seq_len": op.seq_len,
                    "is_left_truncate": op.truncate_side == "left",
                    "is_left_padding": op.pad_side == "left",
                }
            )

        for group_key, group_items in self._groups.items():
            keep_lengths_list = [item["seq_len"] for item in group_items]
            drop_sides_list = [item["is_left_truncate"] for item in group_items]
            pad_sides_list = [item["is_left_padding"] for item in group_items]

            self._group_params[group_key] = {
                "keep_lengths_list": keep_lengths_list,
                "drop_sides_list": drop_sides_list,
                "pad_sides_list": pad_sides_list,
                "items": group_items,
            }

    def add_op(self, op: SequenceTruncate):
        super().add_op(op)
        self._precompute_groups_and_params()

    def process(self, inputs: List):
        """Process inputs using fused sequence cutoff operations.

        Args:
            inputs (List): List of RaggedTensor inputs for sequence processing.

        Returns:
            List: List of processed RaggedTensor outputs with modified sequences.
        """
        outputs = [None] * len(inputs)

        for key, params in self._group_params.items():
            dtype, n_dims = key
            group_items = params["items"]
            fused_values = []
            fused_offsets = []

            for item in group_items:
                data = inputs[item["index"]]
                fused_values.append(data.values())

                if n_dims == 2:
                    fused_offsets.append(data.offsets()[-1])
                else:
                    fused_offsets.append(data.offsets())

            device = inputs[group_items[0]["index"]].device
            if not self._cache_groups_tensor.get(key, None):
                keep_lengths = torch.tensor(
                    params["keep_lengths_list"], dtype=torch.int32, device=device
                )
                drop_sides = torch.tensor(
                    params["drop_sides_list"], dtype=torch.bool, device=device
                )
                pad_sides = torch.tensor(
                    params["pad_sides_list"], dtype=torch.bool, device=device
                )
                self._cache_groups_tensor[key] = {
                    "keep_lengths": keep_lengths,
                    "drop_sides": drop_sides,
                    "pad_sides": pad_sides,
                }
            else:
                keep_lengths = self._cache_groups_tensor[key]["keep_lengths"]
                drop_sides = self._cache_groups_tensor[key]["drop_sides"]
                pad_sides = self._cache_groups_tensor[key]["pad_sides"]

            if n_dims == 2:
                fused_outputs = fused_ragged_cutoff_2D(
                    fused_values, fused_offsets, keep_lengths, drop_sides, pad_sides
                )
                (
                    output_values,
                    output_offsets,
                    drop_nums,
                    pad_nums,
                    drop_sides,
                    pad_sides,
                ) = fused_outputs
            else:
                fused_outputs = fused_ragged_cutoff_3D(
                    fused_values, fused_offsets, keep_lengths, drop_sides, pad_sides
                )
                output_values, output_offsets = fused_outputs
            for j, item in enumerate(group_items):
                original_index = item["index"]
                original_data = inputs[original_index]
                op = self._ops[original_index]

                new_dense_shape = list(original_data._dense_shape)
                new_dense_shape[1] = op.seq_len
                outputs[original_index] = RaggedTensor(
                    values=output_values[j],
                    offsets=output_offsets[j],
                    weight=original_data.weight(),
                    dense_shape=new_dense_shape,
                )
                if n_dims == 2:
                    outputs[original_index].set_pad_info(
                        RaggedPadInfo(
                            drop_nums[j], pad_nums[j], drop_sides[j], pad_sides[j]
                        )
                    )
        return outputs


class FusedMultiHashOP(_FusedOP):
    """Fused implementation for multi-hash operations.

    This class batches multiple multi-hash operations together, managing
    the parameters and device placement for efficient GPU processing.

    Attributes:
        fused_multi_muls (List[torch.Tensor]): Multiplier tensors for each operation.
        fused_multi_primes (List[torch.Tensor]): Prime number tensors for each operation.
        fused_num_buckets (List[torch.Tensor]): Bucket count tensors for each operation.
        fused_bucket_lens (List[int]): Number of hash functions for each operation.
        _copy_to_device (bool): Flag indicating if tensors have been moved to device.
    """

    def __init__(self, ops: Optional[List[IDMultiHash]] = None):
        """Initialize the fused multi-hash operation.

        Args:
            ops (Optional[List[IDMultiHash]]): Initial list of multi-hash operations.
        """
        super().__init__()
        self._ops = ops if ops else []
        for op in self._ops:
            assert isinstance(op, IDMultiHash)

        self._precompute_grouping()
        self._copy_to_device = False

    def _precompute_grouping(self):
        self.fused_multi_muls = []
        self.fused_multi_primes = []
        self.fused_num_buckets = []
        self.fused_bucket_lens = []
        self.fused_multi_prefix = []
        for op in self._ops:
            self.fused_multi_muls.append(op.multi_muls)
            self.fused_multi_primes.append(op.multi_primes)
            self.fused_num_buckets.append(op.num_buckets)
            self.fused_bucket_lens.append(op.bucket_lens)
            self.fused_multi_prefix.append(op.prefix)

    def add_op(self, op):
        super().add_op(op)
        self._precompute_grouping()

    def process(self, inputs: List):
        """Process inputs using fused multi-hash operations.

        Args:
            inputs (List): List of input tensors for multi-hash operations.

        Returns:
            List[dict]: List of dictionaries containing multiple hash results
                       for each input, with keys like 'multi_hash_0', 'multi_hash_1', etc.
        """
        if not self._ops or not inputs:
            return []

        outputs = []
        inputs_tensors = [data.values() for data in inputs]

        if self._copy_to_device is False:
            device = inputs[0].device
            self.fused_multi_muls = [
                multi_muls.to(device) for multi_muls in self.fused_multi_muls
            ]
            self.fused_multi_primes = [
                multi_primes.to(device) for multi_primes in self.fused_multi_primes
            ]
            self.fused_num_buckets = [
                num_buckets.to(device) for num_buckets in self.fused_num_buckets
            ]
            self._copy_to_device = True

        fused_results = fused_multi_hash(
            inputs_tensors,
            self.fused_multi_muls,
            self.fused_multi_primes,
            self.fused_num_buckets,
        )

        for i, data in enumerate(inputs):
            return_data = dict()
            prefix = self.fused_multi_prefix[i]
            start_idx = sum(self.fused_bucket_lens[:i])
            for j in range(self.fused_bucket_lens[i]):
                result_idx = start_idx + j
                if isinstance(data, RaggedTensor):
                    return_data[f"{prefix}_{j}"] = RaggedTensor(
                        fused_results[result_idx], data.offsets(), data.weight()
                    )
                else:
                    return_data[f"{prefix}_{j}"] = fused_results[result_idx]
            outputs.append(return_data)

        return outputs


class FusedStringMultiHashOP(_FusedOP):
    """Fused implementation for string multi-hash operations.

    This class batches multiple string multi-hash operations together,
    applying different hash algorithms (farm, murmur, sdbm, djb2) to
    each input and managing device placement for efficient GPU processing.

    Attributes:
        fused_num_buckets (List[torch.Tensor]): Bucket counts for each operation.
        fused_hash_types (List[List[str]]): Hash types for each operation.
        fused_prefix (List[str]): Prefix strings for each operation.
        fused_bucket_lens (List[int]): Number of hash functions for each operation.
        _copy_to_device (bool): Flag indicating if tensors have been moved to device.
    """

    def __init__(self, ops: Optional[List[StringMultiHash]] = None):
        """Initialize the fused string multi-hash operation.

        Args:
            ops (Optional[List[StringMultiHash]]): Initial list of string multi-hash operations.
        """
        super().__init__()
        self._ops = ops if ops else []
        for op in self._ops:
            assert isinstance(op, StringMultiHash)

        self._precompute_grouping()
        self._copy_to_device = False

    def _precompute_grouping(self):
        """Precompute grouping parameters for all operations."""
        self.fused_num_buckets = []
        self.fused_hash_types = []
        self.fused_prefix = []
        self.fused_bucket_lens = []

        for op in self._ops:
            self.fused_num_buckets.append(op.num_buckets)
            self.fused_hash_types.append(op.hash_types)
            self.fused_prefix.append(op.prefix)
            self.fused_bucket_lens.append(op.bucket_lens)

    def add_op(self, op: StringMultiHash):
        """Add an operation to the fused batch.

        Args:
            op (StringMultiHash): The string multi-hash operation to add.
        """
        super().add_op(op)
        self._precompute_grouping()

    def process(self, inputs: List):
        """Process inputs using fused string multi-hash operations.

        Args:
            inputs (List): List of RaggedTensor inputs for string multi-hash operations.

        Returns:
            List[dict]: List of dictionaries containing multiple hash results
                       for each input, with keys like '{prefix}_farm', '{prefix}_murmur', etc.
        """
        if not self._ops or not inputs:
            return []

        outputs = []

        # Move bucket tensors to device if needed
        if self._copy_to_device is False and inputs:
            device = inputs[0].device
            self.fused_num_buckets = [
                buckets.to(device)
                if isinstance(buckets, torch.Tensor)
                else torch.tensor(buckets, dtype=torch.int64, device=device)
                for buckets in self.fused_num_buckets
            ]
            self._copy_to_device = True

        # Validate all inputs are RaggedTensor

        for data in inputs:
            assert isinstance(data, RaggedTensor), (
                "StringMultiHash only supports RaggedTensor inputs"
            )

        str_data, str_offsets = fused_int64_to_string_int8(
            [data.values() for data in inputs]
        )

        # Group operations by hash type for batch processing
        hash_type_groups = defaultdict(list)
        for i, data in enumerate(inputs):
            hash_types = self.fused_hash_types[i]
            bucket_lens = self.fused_bucket_lens[i]

            for j in range(bucket_lens):
                hash_type = hash_types[j]
                hash_type_groups[hash_type].append(
                    {
                        "input_idx": i,
                        "hash_idx": j,
                        "data": (str_data[i], str_offsets[i]),
                        "num_bucket": self.fused_num_buckets[i][j],
                    }
                )

        # Process each hash type group in batch
        hash_results = {}
        for hash_type, group_items in hash_type_groups.items():
            # Prepare batch inputs
            batch_values = [item["data"][0] for item in group_items]
            batch_splits = [item["data"][1] for item in group_items]

            # Apply hash function in batch
            if hash_type == "farm":
                batch_hashed = farmhash(batch_values, batch_splits)
            elif hash_type in ["mur", "murmur"]:
                batch_hashed = murmurhash(batch_values, batch_splits)
            elif hash_type == "sdbm":
                batch_hashed = sdbmhash(batch_values, batch_splits)
            elif hash_type == "djb2":
                batch_hashed = djb2hash(batch_values, batch_splits)
            else:
                raise ValueError(f"Unsupported hash_type: {hash_type}")

            # Apply modulo operation in batch
            batch_mod_values = [
                item["num_bucket"].item()
                if isinstance(item["num_bucket"], torch.Tensor)
                else item["num_bucket"]
                for item in group_items
            ]
            batch_modded = fused_uint64_mod_gpu(batch_hashed, batch_mod_values)

            # Store results
            for idx, item in enumerate(group_items):
                key = (item["input_idx"], item["hash_idx"], hash_type)
                hash_results[key] = batch_modded[idx]

        # Construct output dictionaries
        for i, data in enumerate(inputs):
            return_data = dict()
            prefix = self.fused_prefix[i]
            hash_types = self.fused_hash_types[i]
            bucket_lens = self.fused_bucket_lens[i]

            for j in range(bucket_lens):
                hash_type = hash_types[j]
                key = (i, j, hash_type)
                modded_values = hash_results[key]

                # Create output RaggedTensor
                return_data[f"{prefix}_{j}"] = RaggedTensor(
                    values=modded_values,
                    offsets=data.offsets(),
                    weight=data.weight(),
                    dense_shape=data._dense_shape,
                )

            outputs.append(return_data)

        return outputs


class FusedMaskOP(_FusedOP):
    """Fused implementation for mask operations.

    This class batches multiple Mask operations together, grouping by dtype
    and processing each group in a single fused kernel call for improved performance.

    Supports:
    - String mask (dtype="string"): For int8-encoded string data
    - Number mask (dtype=torch.float32/float64/int32/int64): For numeric data

    Attributes:
        _groups (defaultdict): Grouping of operations by dtype.
        _group_params (dict): Parameters for each operation group.
    """

    def __init__(self, ops: Optional[List[Mask]] = None):
        """Initialize the fused mask operation.

        Args:
            ops (Optional[List[Mask]]): Initial list of mask operations.
        """
        super().__init__()
        self._ops = ops if ops else []
        for op in self._ops:
            assert isinstance(op, Mask)
        self._precompute_groups_and_params()

    def _precompute_groups_and_params(self):
        """Precompute grouping and parameters for all operations."""
        self._groups = defaultdict(list)
        self._group_params = {}

        for i, op in enumerate(self._ops):
            group_key = op.dtype
            self._groups[group_key].append(
                {
                    "op": op,
                    "index": i,
                    "masks": op.masks,
                }
            )

        for group_key, group_items in self._groups.items():
            masks_list = [item["masks"] for item in group_items]
            self._group_params[group_key] = {
                "masks_list": masks_list,
                "items": group_items,
            }

    def add_op(self, op: Mask):
        """Add a Mask operation to the fused batch.

        Args:
            op (Mask): The mask operation to add.
        """
        super().add_op(op)
        self._precompute_groups_and_params()

    def process(self, inputs: List):
        """Process inputs using fused mask operations grouped by dtype.

        Args:
            inputs (List): List of input data (RaggedTensor or torch.Tensor).

        Returns:
            List: List of output tensors with mask results
                (0.0 for matched, 1.0 for unmatched).
        """
        if not self._ops or not inputs:
            return []

        outputs = [None] * len(inputs)

        for group_key, params in self._group_params.items():
            dtype = group_key
            group_items = params["items"]
            masks_list = params["masks_list"]

            if dtype == torch.int8:
                # Process string mask group
                fused_values = []
                fused_offsets = []
                for item in group_items:
                    data = inputs[item["index"]]
                    assert isinstance(data, RaggedTensor), (
                        "String mask only supports RaggedTensor inputs"
                    )
                    fused_values.append(data.values())
                    fused_offsets.append(data.offsets()[-1])

                mask_results = fused_string_mask(
                    fused_values, fused_offsets, masks_list
                )

                for j, item in enumerate(group_items):
                    original_index = item["index"]
                    original_data = inputs[original_index]
                    outputs[original_index] = RaggedTensor(
                        values=mask_results[j],
                        offsets=original_data.offsets()[:-1],
                        weight=original_data.weight(),
                        dense_shape=original_data._dense_shape[:-1],
                    )
            else:
                # Process number mask group (float32/float64/int32/int64)
                fused_values = []
                for item in group_items:
                    data = inputs[item["index"]]
                    if isinstance(data, RaggedTensor):
                        fused_values.append(data.values())
                    else:
                        fused_values.append(data)

                # Convert masks to float for comparison
                float_masks_list = [[float(m) for m in masks] for masks in masks_list]
                mask_results = fused_number_mask(fused_values, float_masks_list)

                for j, item in enumerate(group_items):
                    original_index = item["index"]
                    original_data = inputs[original_index]
                    if isinstance(original_data, RaggedTensor):
                        outputs[original_index] = RaggedTensor(
                            values=mask_results[j],
                            offsets=original_data.offsets(),
                            weight=original_data.weight(),
                            dense_shape=original_data._dense_shape,
                        )
                    else:
                        outputs[original_index] = mask_results[j]

        return outputs


# Register fused operation implementations with the factory
FusedOpFactory.register(Bucketize, FusedBoundaryOP)
FusedOpFactory.register(Mod, FusedModOP)
FusedOpFactory.register(Hash, FusedHashOP)
FusedOpFactory.register(SequenceTruncate, FusedCutoffOP)
FusedOpFactory.register(IDMultiHash, FusedMultiHashOP)
FusedOpFactory.register(StringMultiHash, FusedStringMultiHashOP)
FusedOpFactory.register(Mask, FusedMaskOP)
