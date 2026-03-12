import hashlib
import math
from typing import Dict, List, Union

import torch
from torch import nn

from recis.nn.functional.array_ops import bucketize, bucketize_mod
from recis.nn.functional.fused_ops import fused_int64_to_string_int8, fused_multi_hash
from recis.nn.functional.hash_ops import (
    djb2hash as djb2hash_gpu,
    farmhash as farmhash_gpu,
    murmurhash as murmurhash_gpu,
    sdbmhash as sdbmhash_gpu,
)
from recis.nn.functional.ragged_ops import (
    feature_cross_ragged,
    fused_ragged_cutoff_2D,
    fused_ragged_cutoff_3D,
)
from recis.ragged.tensor import RaggedTensor


class DataValueProcessor:
    """Utility class for processing different data types in feature operations.

    This class provides a unified interface for extracting values from and
    reconstructing various tensor types including RaggedTensor, sparse tensors,
    and dense tensors.

    Attributes:
        _data (Union[RaggedTensor, torch.Tensor]): The input data to process.
    """

    def __init__(self, data: Union[RaggedTensor, torch.Tensor]):
        """Initialize the data value processor.

        Args:
            data (Union[RaggedTensor, torch.Tensor]): Input data to process.
        """
        self._data = data

    def get_input_value(self):
        """Extract the underlying values from the input data.

        Returns:
            torch.Tensor: The extracted values tensor suitable for processing.

        Raises:
            ValueError: If the data type is not supported.
        """
        if isinstance(self._data, RaggedTensor):
            return self._data.values()
        elif isinstance(self._data, torch.Tensor):
            if self._data.is_sparse:
                return self._data._values()
            return self._data
        else:
            raise ValueError(f"Unsupported data type: {type(self._data)}")

    def get_output_value(self, result):
        """Reconstruct the output data with processed values.

        Args:
            result (torch.Tensor): The processed values tensor.

        Returns:
            Union[RaggedTensor, torch.Tensor]: The reconstructed output data
                                             maintaining the original structure.

        Raises:
            ValueError: If the data type is not supported.
        """
        if isinstance(self._data, RaggedTensor):
            self._data = RaggedTensor(
                values=result,
                offsets=self._data.offsets(),
                weight=self._data.weight(),
                dense_shape=self._data._dense_shape,
            )
            return self._data
        elif isinstance(self._data, torch.Tensor):
            if self._data.is_sparse:
                return torch.sparse_coo_tensor(
                    indices=self._data._indices(),
                    values=result,
                    size=self._data.shape,
                )
            else:
                return result.reshape(self._data.shape)
        else:
            raise ValueError(f"Unsupported data type: {type(self._data)}")


class _OP(nn.Module):
    """Base class for all feature processing operations.

    This abstract base class defines the interface that all feature operations
    must implement, including dependency management and hash computation for
    caching and optimization purposes.
    """

    def __init__(self):
        """Initialize the base operation."""
        super().__init__()

    def get_hash(self) -> int:
        """Compute a hash value for this operation based on its configuration.

        The hash is used for caching, deduplication, and optimization purposes.
        Operations with identical configurations will have the same hash.

        Returns:
            int: A signed 64-bit hash value representing this operation.
        """
        class_name = self.__class__.__name__
        config_dict = self._get_config()
        config_str = f"{class_name}:{str(sorted(config_dict.items()))}"
        hash_bytes = hashlib.sha256(config_str.encode("utf-8")).digest()
        hash_value = int.from_bytes(hash_bytes[:8], byteorder="big", signed=True)

        return hash_value

    def _get_config(self) -> Dict:
        """Get the configuration dictionary for this operation.

        Returns:
            Dict: Configuration parameters that define this operation.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


class SelectField(_OP):
    """Data input operation for extracting fields from input dictionaries.

     This operation serves as the entry point for feature pipelines, extracting
     specific fields from input data dictionaries and optionally applying
     sequence processing operations.

     Examples:

    .. code-block:: python

       from recis.features.op import SelectField

       # ID Feature
       data_input = SelectField("user_id")

       # Sequence Feature
       data_input_sequence = SelectField("user_seq", dim=2)

    """

    def __init__(
        self,
        name: str,
        dtype=torch.long,
        dim=None,
        from_dict=True,
    ):
        """Initialize the data input operation.

        Args:
            name (str): Name of the input field to extract.
            dtype (torch.dtype): Expected data type. Defaults to torch.long.
            dim (int, optional): Dimension specification for RaggedTensor.
            from_dict (bool): Whether to extract from dictionary. Defaults to True.
        """
        super().__init__()
        self._name = name
        self._dtype = dtype
        self._dim = dim
        self._from_dict = from_dict

    def forward(self, data):
        """Extract and process input data.

        Args:
            data (Union[dict, torch.Tensor, RaggedTensor]): Input data to process.

        Returns:
            Union[torch.Tensor, RaggedTensor]: Processed input data.
        """
        if self._from_dict:
            assert isinstance(data, dict), f"data must be a dict, fn:{self._name}"
            x = data[self._name]
        else:
            x = data
        if isinstance(data, RaggedTensor) and self._dim and (self._dim > 0):
            x.set_dim_by_rank(self._dim, -1)
        return x

    def _get_config(self) -> Dict:
        return {
            "name": self._name,
            "dtype": str(self._dtype),
            "dim": self._dim,
            "from_dict": self._from_dict,
        }


class SelectFields(_OP):
    """Multi-field data input operation for processing multiple inputs simultaneously.

    This operation applies multiple SelectField operations and returns their results
    as a list, enabling parallel processing of multiple input fields.

    Attributes:
        input_list (List[SelectField]): List of SelectField operations to apply.

    Examples:

    .. code-block:: python

        from recis.features.op import SelectFields

        multi_input = SelectFields(["user_id", "item_id", "category_id"])

    """

    def __init__(self, input_list: List[SelectField]):
        """Initialize the multi-data input operation.

        Args:
            input_list (List[SelectField]): List of SelectField operations to execute.
        """
        super().__init__()
        self.input_list = input_list

    def forward(self, data):
        """Process input data through multiple SelectField operations.

        Args:
            data: Input data to be processed by all SelectField operations.

        Returns:
            List: List of results from each SelectField operation.
        """
        ret = []
        for input_op in self.input_list:
            ret.append(input_op(data))
        return ret

    def _get_config(self) -> Dict:
        input_hashes = [input_op.get_hash() for input_op in self.input_list]
        return {
            "input_hashes": input_hashes,
        }


class Mod(_OP):
    """Unsigned 64-bit integer modulo operation.

    This operation applies modulo arithmetic to input values, treating them
    as unsigned 64-bit integers. Commonly used for hash bucketing and
    ID space reduction.

    Attributes:
        mod (int): The modulo value to apply.

    Examples:

    .. code-block:: python

        from recis.features.op import Mod

        mod_op = Mod(mod_value=1000000)
    """

    def __init__(self, mod_value):
        """Initialize the modulo operation.

        Args:
            mod_value (int): The modulo value for the operation.
        """
        super().__init__()
        self.mod_value = mod_value

    def forward(self, x: Union[RaggedTensor, torch.Tensor]):
        """Apply modulo operation to input data.

        Args:
            x (Union[RaggedTensor, torch.Tensor]): Input tensor data.

        Returns:
            Union[RaggedTensor, torch.Tensor]: Output with modulo applied to values.
        """
        data = DataValueProcessor(x)
        input_data = data.get_input_value()
        result = bucketize_mod(input_data, self.mod_value)
        return data.get_output_value(result)

    def _get_config(self) -> Dict:
        return {
            "mod": self.mod_value,
        }


class Bucketize(_OP):
    """Bucketize-based bucketing operation for continuous value discretization.

    This operation maps continuous values to discrete bucket indices based on
    predefined boundary values. Values are assigned to buckets according to
    which boundaries they fall between.

    Attributes:
        boundary (torch.Tensor): Sorted tensor of boundary values defining buckets.

    Examples:

    .. code-block:: python

        from recis.features.op import Bucketize

        # age bucketing
        age_boundary = Bucketize(
            boundary=[18, 25, 35, 45, 55, 65],
        )

        # inputs: [20, 30, 40, 50, 60]
        # outputs: [1, 2, 3, 4, 5]  (bucket ID)
    """

    def __init__(self, boundary):
        """Initialize the boundary operation.

        Args:
            boundary (Union[List[float], torch.Tensor]): Bucketize values for bucketing.
                                                        Must be sorted in ascending order.
        """
        super().__init__()
        if isinstance(boundary, list):
            self.boundary = torch.tensor(boundary, dtype=torch.float)
        else:
            self.boundary = boundary

    def forward(self, x: Union[RaggedTensor, torch.Tensor]):
        """Apply boundary-based bucketing to input data.

        Args:
            x (Union[RaggedTensor, torch.Tensor]): Input tensor with continuous values.

        Returns:
            Union[RaggedTensor, torch.Tensor]: Output with bucket indices.
        """
        data = DataValueProcessor(x)
        input_data = data.get_input_value()
        result = bucketize(input_data, self.boundary)
        return data.get_output_value(result)

    def _get_config(self) -> Dict:
        if isinstance(self.boundary, torch.Tensor):
            boundary_list = self.boundary.tolist()
        else:
            boundary_list = self.boundary
        return {
            "boundary": boundary_list,
        }


class FeatureCross(_OP):
    """Feature crossing operation for generating interaction features.

    This operation creates cross features by combining two RaggedTensor inputs,
    generating new features that capture interactions between the original features.

    Examples:

    .. code-block:: python

        from recis.features.op import FeatureCross

        cross_op = FeatureCross()
    """

    def __init__(self):
        """Initialize the feature cross operation."""
        super().__init__()

    def forward(self, data: List[RaggedTensor]):
        """Create cross features from two RaggedTensor inputs.

        Args:
            data (List[RaggedTensor]): List containing exactly two RaggedTensor inputs.

        Returns:
            RaggedTensor: Cross feature tensor combining the input features.

        Raises:
            AssertionError: If inputs are not RaggedTensors or if there aren't exactly two inputs.
        """
        x, y = data

        assert isinstance(x, RaggedTensor) and isinstance(y, RaggedTensor)
        val, offsets, weight = feature_cross_ragged(
            x.values(),
            x.offsets()[-1],
            y.values(),
            y.offsets()[-1],
            x.weight(),
            y.weight(),
        )
        ret = RaggedTensor(
            values=val, offsets=offsets, weight=weight, dense_shape=x._dense_shape
        )
        return ret

    def _get_config(self) -> Dict:
        return {}


class SequenceTruncate(_OP):
    """Sequence processing operation for truncation.

    This operation handles sequence data by applying truncation
    to ensure sequences meet specified length requirements. Supports both 2D
    and 3D sequence data with configurable truncation sides.

    Examples:

    .. code-block:: python

        from recis.features.op import SequenceTruncate

        SequenceTruncate(
            seq_len=20,
            truncate=True,
            truncate_side="right",
            check_length=False,
            n_dims=3,
            dtype=torch.int64,
        )
    """

    CUT_FUNC_MAP = {
        2: fused_ragged_cutoff_2D,
        3: fused_ragged_cutoff_3D,
    }

    def __init__(
        self,
        seq_len=64,
        check_length=True,
        truncate=True,
        truncate_side="left",
        pad_side="right",
        n_dims=2,
        dtype=torch.long,
    ):
        """Initialize the sequence processing operation.

        Args:
            seq_len (int): Target sequence length. Defaults to 64.
            check_length (bool): Whether to validate sequence length. Defaults to True.
            truncate (bool): Whether to apply truncation. Defaults to True.
            truncate_side (str): Truncation side ("left" or "right"). Defaults to "left".
            pad_side (str): Padding side ("left" or "right"). Defaults to "right".
            n_dims (int): Number of input dimensions (2 or 3). Defaults to 2.
            dtype (torch.dtype): Data type of sequences. Defaults to torch.long.

        Raises:
            AssertionError: If n_dims is not 2 or 3.
        """
        super().__init__()
        assert n_dims in [2, 3]
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.check_length = check_length
        self.truncate = truncate
        self.truncate_side = truncate_side
        self.pad_side = pad_side
        self.cutoff_func = self.CUT_FUNC_MAP[n_dims]
        self.drop_sides = torch.BoolTensor([self.truncate_side == "left"])
        self.pad_sides = torch.BoolTensor([self.pad_side == "left"])
        self.keep_lengths = torch.IntTensor([self.seq_len])
        self.dtype = dtype
        self.keep_lengths = torch.IntTensor([self.seq_len])
        self.dtype = dtype

    def forward(self, x: Union[RaggedTensor, torch.Tensor]):
        """Process sequences with truncation and padding.

        Args:
            x (Union[RaggedTensor, torch.Tensor]): Input sequence data.

        Returns:
            RaggedTensor: Processed sequence data with target length.

        Raises:
            AssertionError: If check_length is True and sequence exceeds target length.
        """
        if self.check_length:
            assert x.shape[1] <= self.seq_len, (
                f"sequence length must be less than {self.seq_len}"
            )
        if self.truncate:
            values = [x.values()]
            offsets = [x.offsets()[-1]] if self.n_dims == 2 else [x.offsets()]
            self.drop_sides = self.drop_sides.to(device=x.device)
            self.pad_sides = self.pad_sides.to(device=x.device)
            self.keep_lengths = self.keep_lengths.to(device=x.device)
            output = self.cutoff_func(
                values, offsets, self.keep_lengths, self.drop_sides, self.pad_sides
            )
            values = output[0]
            offsets = output[1]
            dense_shape = list(x._dense_shape)
            dense_shape[1] = self.seq_len
            return RaggedTensor(
                values=values[0],
                offsets=offsets[0],
                weight=x.weight(),
                dense_shape=dense_shape,
            )

    def _get_config(self) -> Dict:
        return {
            "seq_len": self.seq_len,
            "check_length": self.check_length,
            "truncate": self.truncate,
            "truncate_side": self.truncate_side,
            "n_dims": self.n_dims,
            "dtype": str(self.dtype),
        }


def _is_prime(x):
    for n in range(int(math.sqrt(x) + 1e-6), 1, -1):
        if x % n == 0:
            return False
    return True


def _find_prime_lower_than(x):
    for n in range(x, 0, -1):
        if _is_prime(n):
            return n
    return 11


class IDMultiHash(_OP):
    """Multi-hash operation for generating multiple hash values.

    This operation applies multiple hash functions with different parameters
    to generate several hash values from a single input, useful for techniques
    like feature hashing and locality-sensitive hashing.

    Examples:

    .. code-block:: python

        from recis.features.op import IDMultiHash

        multi_hash = IDMultiHash(num_buckets=[20000, 20000, 10000, 500])
    """

    multi_muls = [1, 3, 5, 7]
    multi_mods = [29, 47, 67, 83]

    def __init__(self, num_buckets: List[int], prefix: str = "multi_hash"):
        """Initialize the multi-hash operation.

        Args:
            num_buckets (List[int]): List of bucket counts for each hash function.
                                   Must contain at least one element.

        Raises:
            AssertionError: If num_buckets is empty.
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.prefix = prefix
        self.bucket_lens = len(self.num_buckets)
        assert len(self.num_buckets) > 0, "num_buckets must be a list of 4 integers"
        multi_primes = [
            _find_prime_lower_than(IDMultiHash.multi_mods[i] * self.num_buckets[i])
            for i in range(self.bucket_lens)
        ]
        self.multi_muls = torch.tensor(
            IDMultiHash.multi_muls[: self.bucket_lens], dtype=torch.int64
        )
        self.multi_primes = torch.tensor(
            multi_primes[: self.bucket_lens], dtype=torch.int64
        )
        self.num_buckets = torch.tensor(self.num_buckets, dtype=torch.int64)

    def forward(self, x: Union[RaggedTensor, torch.Tensor]):
        """Apply multi-hash operation to input data.

        Args:
            x (Union[RaggedTensor, torch.Tensor]): Input data to hash.

        Returns:
            dict: Dictionary with keys 'multi_hash_0', 'multi_hash_1', etc.,
                 containing the results of each hash function.
        """
        if isinstance(x, RaggedTensor):
            self.multi_primes = self.multi_primes.to(device=x.device)
            self.multi_muls = self.multi_muls.to(device=x.device)
            self.num_buckets = self.num_buckets.to(device=x.device)
            new_values = fused_multi_hash(
                [x.values()],
                [self.multi_muls],
                [self.multi_primes],
                [self.num_buckets],
            )
            return_data = dict()
            for i in range(self.bucket_lens):
                return_data[f"{self.prefix}_{i}"] = RaggedTensor(
                    new_values[i], x.offsets(), x.weight()
                )
            return return_data

        else:
            new_tensors = fused_multi_hash(
                [x],
                [self.multi_muls],
                [self.multi_primes],
                [self.num_buckets],
            )
            return_data = dict()
            for i in range(self.bucket_lens):
                return_data[f"{self.prefix}_{i}"] = new_tensors[i]
            return return_data

    def _get_config(self) -> Dict:
        return {
            "num_buckets": self.num_buckets,
            "multi_muls": IDMultiHash.multi_muls,
            "multi_mods": IDMultiHash.multi_mods,
        }


class Hash(_OP):
    """Hash operation for applying hash functions to sequence data.

    This operation applies either FarmHash or MurmurHash algorithms to
    RaggedTensor data, commonly used for feature hashing and dimensionality
    reduction in recommendation systems.

    Attributes:
        hash_type (str): Type of hash function ("farm" or "murmur").

    Examples:

    .. code-block:: python

        from recis.features.op import Hash

        # Farm Hash
        hash_op = Hash(hash_type="farm")

        # Murmur Hash
        murmur_hash = Hash(hash_type="murmur")
    """

    def __init__(self, hash_type: str):
        """Initialize the hash operation.

        Args:
            hash_type (str): Hash algorithm to use ("farm" or "murmur").

        Raises:
            AssertionError: If hash_type is not "farm" or "murmur".
        """
        super().__init__()
        assert hash_type in ["farm", "murmur"]
        self.hash_type = hash_type
        self.hash_type

    def forward(self, x: Union[RaggedTensor, torch.Tensor]):
        """Apply hash function to input RaggedTensor.

        Args:
            x (Union[RaggedTensor, torch.Tensor]): Input RaggedTensor to hash.

        Returns:
            RaggedTensor: Hashed output with reduced dimensionality.
        """
        assert isinstance(x, RaggedTensor)
        if self.hash_type == "farm":
            output = farmhash_gpu([x.values()], [x.offsets()[-1].int()])
        else:
            output = murmurhash_gpu([x.values()], [x.offsets()[-1].int()])
        return RaggedTensor(
            values=output[0],
            offsets=x.offsets()[:-1],
            weight=x.weight(),
            dense_shape=x._dense_shape[:-1],
        )

    def _get_config(self) -> Dict:
        return {
            "hash_type": self.hash_type,
        }


class StringMultiHash(_OP):
    def __init__(self, num_buckets: List[int], prefix: str = "str_multi_hash"):
        super().__init__()
        self.num_buckets = num_buckets
        self.prefix = prefix
        self.hash_types = ["farm", "mur", "sdbm", "djb2"]
        self.bucket_lens = len(self.num_buckets)
        assert len(self.num_buckets) > 0, "num_buckets must be a list of 4 integers"
        self.num_buckets = torch.tensor(self.num_buckets, dtype=torch.int64)

    def hash(self, x: torch.Tensor, split: torch.Tensor, hash_type: str):
        if hash_type == "farm":
            output = farmhash_gpu([x], [split])
        elif hash_type in ["mur", "murmur"]:
            output = murmurhash_gpu([x], [split])
        elif hash_type == "sdbm":
            output = sdbmhash_gpu([x], [split])
        elif hash_type == "djb2":
            output = djb2hash_gpu([x], [split])
        else:
            raise ValueError(f"hash_type {hash_type} not supported")
        return output[0]

    def forward(self, x: Union[RaggedTensor, torch.Tensor]):
        if isinstance(x, RaggedTensor):
            return_data = dict()
            ids = x.values()
            str_ids, offsets = fused_int64_to_string_int8([ids])
            for i in range(self.bucket_lens):
                new_values = self.hash(str_ids, offsets, self.hash_types[i])
                new_values = bucketize_mod(new_values, self.num_buckets[i])
                return_data[f"{self.prefix}_{i}"] = RaggedTensor(
                    values=new_values,
                    offsets=x.offsets(),
                    weight=x.weight(),
                    dense_shape=x._dense_shape,
                )
            return return_data

        else:
            raise NotImplementedError

    def _get_config(self) -> Dict:
        return {
            "num_buckets": self.num_buckets,
            "hash_types": self.hash_types,
        }
