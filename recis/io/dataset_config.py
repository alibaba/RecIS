from dataclasses import dataclass
from typing import Optional


@dataclass
class FeatureIOConf:
    """Configuration dataclass for feature I/O operations.

    This dataclass defines the configuration parameters needed for feature
    input/output operations in the RecIS system. It specifies how features
    should be processed during data loading, including format, hashing,
    and transformation settings.

    Key Features:
        - Support for both fixed-length and variable-length features
        - Configurable hash functions for string feature processing
        - Optional integer transformation for compatibility
        - Flexible dimension specification for multi-dimensional features

    Attributes:
        name (str): Name of the feature for identification and processing.
        varlen (bool): Whether the feature has variable length format.
            If True, the feature is treated as a sparse/variable-length feature.
            If False, the feature is treated as a dense/fixed-length feature.
            Defaults to False.
        hash_type (Optional[str]): Type of hash function to apply to string values.
            Common values include "farm", "murmur", etc. If None, no hashing
            is applied. Defaults to None.
        hash_bucket_size (int): Size of the hash bucket for hash operations.
            Only used when hash_type is specified. A value of 0 means no
            bucketing is applied. Defaults to 0.
        trans_int (bool): Whether to transform string values to integers.
            This is typically used for compatibility with downstream processing
            that expects integer inputs. Defaults to False.
        dim (int): Dimension of the feature values. For scalar features,
            this should be 1. For vector features, this specifies the
            vector dimension. Defaults to 1.
    """

    name: str
    varlen: bool = False
    hash_type: Optional[str] = None
    hash_bucket_size: int = 0
    trans_int: bool = False
    dim: int = 1
    seq_length: Optional[int] = None
    fill_seq: bool = False
