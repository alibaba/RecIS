import copy
import json
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional

import torch

from recis.fg.utils import dict_lower_case
from recis.io.dataset_config import FeatureIOConf
from recis.utils.logger import Logger


FG_FEATURE_KEY = "features"
FEATURE_NAME_KEY = "feature_name"
SEQUENCE_NAME_KEY = "sequence_name"
SEQUENCE_LENGTH_KEY = "sequence_length"

value_type_key = "value_type"
hash_bucket_key = "hash_bucket_size"
hash_type_key = "hash_type"
shared_name_key = "shared_name"
feature_type_key = "feature_type"
value_dim_key = "value_dimension"
boundaries_key = "boundaries"
mask_value_key = "mask_out"
compress_strategy_key = "compress_strategy"
combiner_key = "combiner"
emb_dim_key = "embedding_dimension"
trainable_key = "trainable"
emb_device_key = "emb_device"
emb_type_key = "emb_type"
admit_hook_key = "admit_hook"
filter_hook_key = "filter_hook"

gen_key_type_key = "gen_key_type"
gen_value_type_key = "gen_val_type"
from_feature_key = "from_feature"

HASH_TYPE_MAP = {"farmhash": "farm", "murmur": "murmur", "ev_hash": "ev_farm"}
VALUE_TYPE_MAP = {"string": torch.int8, "double": torch.float32, "integer": torch.int64}
EMB_TYPE_MAP = {
    "float": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "int8": torch.int8,
}

logger = Logger(__name__)


class IdTransformType(IntEnum):
    """Enumeration of ID transformation types for feature processing.

    This enum defines the different ways feature IDs can be transformed
    during the feature processing pipeline. Each transformation type
    corresponds to a specific operation applied to input feature values.

    Attributes:
        RAW (int): No transformation, use raw values as-is.
        BUCKETIZE (int): Apply bucketization based on predefined boundaries.
        HASH (int): Apply hash function to convert values to integers.
        MOD (int): Apply modulo operation for bucket assignment.
        MASK (int): Apply mask operation (not currently supported).
        MULTIHASH (int): Apply multiple hash functions for advanced embedding.
        HASH_MULTIHASH (int): Combine hash and multi-hash transformations.
        MOD_MULTIHASH (int): Combine modulo and multi-hash transformations.
    """

    RAW = 1
    BUCKETIZE = 2
    HASH = 3
    MOD = 4
    MASK = 5
    MULTIHASH = 6
    HASH_MULTIHASH = 7
    MOD_MULTIHASH = 8


class EmbTransformType(IntEnum):
    """Enumeration of embedding transformation types.

    This enum defines how features are transformed into embeddings
    after ID transformation. It determines the final representation
    format for features in the embedding space.

    Attributes:
        RAW (int): No embedding transformation, use raw values.
        LOOKUP (int): Standard embedding lookup from embedding tables.
        MULTIHASH_LOOKUP (int): Multi-hash embedding lookup for advanced strategies.
    """

    RAW = 1
    LOOKUP = 2
    MULTIHASH_LOOKUP = 3


@dataclass
class FGConf:
    """Configuration dataclass for feature generation settings.

    This dataclass holds all configuration parameters needed for feature
    generation, including transformation settings, embedding parameters,
    and various feature-specific options.

    Attributes:
        name (str): Name of the feature.
        is_sparse (bool): Whether the feature is sparse.
        gen_key_type (str): Type of key generation.
        gen_value_type (str): Type of value generation.
        need_hash (bool): Whether hashing is needed. Defaults to False.
        hash_type (str): Type of hash function to use. Defaults to "farm".
        hash_bucket_size (int): Size of hash buckets. Defaults to 0.
        is_seq (bool): Whether this is a sequence feature. Defaults to False.
        seq_length (int): Length of sequences. Defaults to 0.
        value_dimension (int): Dimension of feature values. Defaults to 1.
        combiner (str): Combiner strategy for embeddings. Defaults to "mean".
        embedding_dim (Optional[int]): Embedding dimension. Defaults to None.
        value_type (Optional[str]): Type of feature values. Defaults to None.
        boundaries (Optional[List[float]]): Boundaries for bucketization. Defaults to None.
        compress_strategy (Optional[str]): Compression strategy. Defaults to None.
        shared_name (Optional[str]): Shared embedding name. Defaults to None.
        from_feature (Optional[str]): Source feature for copying. Defaults to None.
        emb_device (Optional[str]): Device for embeddings. Defaults to None.
        emb_type (Optional[str]): Embedding data type. Defaults to None.
        trainable (bool): Whether embeddings are trainable. Defaults to True.
        admit_hook (Optional[Dict[str, str]]): Admission hook configuration. Defaults to None.
        filter_hook (Optional[Dict[str, str]]): Filter hook configuration. Defaults to None.
    """

    name: str
    is_sparse: bool
    gen_key_type: str
    gen_value_type: str
    need_hash: bool = False
    hash_type: str = "farm"
    hash_bucket_size: int = 0
    is_seq: bool = False
    seq_length: int = 0
    value_dimension: int = 1
    combiner: str = "mean"
    embedding_dim: Optional[int] = None
    value_type: Optional[str] = None
    boundaries: Optional[List[float]] = None
    mask_value: Optional[List[str]] = None
    compress_strategy: Optional[str] = None
    shared_name: Optional[str] = None
    from_feature: Optional[str] = None
    emb_device: Optional[str] = None
    emb_type: Optional[str] = None
    trainable: bool = True
    admit_hook: Optional[Dict[str, str]] = None
    filter_hook: Optional[Dict[str, str]] = None


@dataclass
class FeatureEmbConf:
    """Configuration dataclass for feature embedding settings.

    This dataclass contains all parameters needed to configure feature
    embeddings, including transformation types, dimensions, and various
    embedding-specific options.

    Attributes:
        io_name (str): Name used for I/O operations.
        out_name (str): Output name for the feature.
        id_transform_type (IdTransformType): Type of ID transformation.
        emb_transform_type (EmbTransformType): Type of embedding transformation.
        hash_type (str): Hash function type. Defaults to "farm".
        hash_bucket_size (int): Hash bucket size. Defaults to 0.
        seq_length (int): Sequence length for sequence features. Defaults to 0.
        raw_dim (Optional[int]): Raw dimension of the feature. Defaults to None.
        combiner (str): Embedding combiner strategy. Defaults to "mean".
        embedding_dim (Optional[int]): Embedding dimension. Defaults to None.
        dtype (Optional[torch.dtype]): Data type for the feature. Defaults to None.
        boundaries (Optional[List[float]]): Bucketization boundaries. Defaults to None.
        compress_strategy (Optional[str]): Compression strategy. Defaults to None.
        shared_name (Optional[str]): Shared embedding name. Defaults to None.
        emb_device (Optional[str]): Embedding device. Defaults to None.
        emb_type (Optional[torch.dtype]): Embedding data type. Defaults to None.
        trainable (bool): Whether embedding is trainable. Defaults to True.
        admit_hook (Optional[Dict[str, str]]): Admission hook config. Defaults to None.
        filter_hook (Optional[Dict[str, str]]): Filter hook config. Defaults to None.
    """

    io_name: str
    out_name: str
    id_transform_type: IdTransformType
    emb_transform_type: EmbTransformType
    hash_type: str = "farm"
    hash_bucket_size: int = 0
    seq_length: int = 0
    raw_dim: Optional[int] = None
    combiner: str = "mean"
    embedding_dim: Optional[int] = None
    dtype: Optional["torch.dtype"] = None
    boundaries: Optional[List[float]] = None
    mask_value: Optional[List[str]] = None
    compress_strategy: Optional[str] = None
    shared_name: Optional[str] = None
    emb_device: Optional[str] = None
    emb_type: Optional["torch.dtype"] = None
    trainable: bool = True
    admit_hook: Optional[Dict[str, str]] = None
    filter_hook: Optional[Dict[str, str]] = None
    fill_seq: bool = False


class FGParser:
    """Feature Generation configuration parser and processor.

    The FGParser class is responsible for parsing feature generation configuration
    files, processing feature definitions, and creating structured configurations
    for the feature generation pipeline. It handles both regular and sequence
    features, applies various transformations, and manages feature filtering
    based on model configuration.

    Key Features:
        - Parse JSON configuration files for feature definitions
        - Filter features based on model configuration requirements
        - Handle sequence features with proper length and structure
        - Support feature copying and inheritance
        - Generate I/O and embedding configurations
        - Validate and transform feature parameters

    Attributes:
        already_hashed (bool): Whether input features are already hashed.
        hash_in_io (bool): Whether to perform hashing in I/O layer.
        mc_parser: Model configuration parser instance.
        devel_mode (bool): Whether development mode is enabled.
        multihash_conf_ (dict): Multi-hash configuration dictionary.
        fg_conf (list): Parsed feature generation configuration.
        parsed_conf_ (list): Processed feature configurations.
        io_conf_ (dict): I/O configuration dictionary.
        emb_conf_ (dict): Embedding configuration dictionary.
    """

    def __init__(
        self,
        conf_file_path,
        mc_parser,
        already_hashed=False,
        hash_in_io=False,
        lower_case=False,
        devel_mode=False,
        io_name_lower_case=False,
    ):
        """Initialize the FG Parser.

        Args:
            conf_file_path (str): Path to the feature generation configuration file.
            mc_parser: Model configuration parser instance.
            already_hashed (bool, optional): Whether features are already hashed.
                Defaults to False.
            hash_in_io (bool, optional): Whether to hash in I/O layer.
                Defaults to False.
            lower_case (bool, optional): Whether to convert keys to lowercase.
                Defaults to False.
            devel_mode (bool, optional): Whether to enable development mode.
                Defaults to False.
        """
        self.already_hashed = already_hashed
        self.hash_in_io = hash_in_io
        self.mc_parser = mc_parser
        self.devel_mode = devel_mode
        self.multihash_conf_ = {}
        self.fg_path = conf_file_path
        self.lower_case = lower_case
        self.io_name_lower_case = io_name_lower_case
        self.fg_conf = self._init_fg(conf_file_path, lower_case)
        self.parsed_conf_ = self._parse_feature_conf()
        self.io_conf_ = self._init_io_conf()
        self.emb_conf_ = self._init_emb_conf()

    @property
    def feature_blocks(self):
        """Get feature blocks from the model configuration parser.

        Returns:
            dict: Dictionary mapping block names to feature lists.
        """
        return self.mc_parser.feature_blocks

    @property
    def io_configs(self):
        """Get I/O configurations for all features.

        Returns:
            dict: Dictionary mapping feature names to I/O configurations.
        """
        return self.io_conf_

    @property
    def emb_configs(self):
        """Get embedding configurations for all features.

        Returns:
            dict: Dictionary mapping feature names to embedding configurations.
        """
        return self.emb_conf_

    @property
    def seq_block_names(self):
        """Get sequence block names from the model configuration parser.

        Returns:
            list: List of sequence block names.
        """
        return self.mc_parser.seq_block_names

    @property
    def multihash_conf(self):
        """Get multi-hash configuration dictionary.

        Returns:
            dict: Multi-hash configuration settings.
        """
        return self.multihash_conf_

    def get_mc_conf(self):
        """Get mc configuration dictionary.

        Returns:
            dict: mc configuration settings.
        """

        return self.mc_parser.mc_conf

    def get_fg_conf(self):
        """Get fg configuration dictionary.

        Returns:
            dict: fg configuration settings.
        """
        return self._load_fg_conf(self.fg_path, self.lower_case)

    def _init_fg(self, fg_path, lower_case):
        """Initialize feature generation configuration.

        Args:
            fg_path (str): Path to the FG configuration file.
            lower_case (bool): Whether to convert keys to lowercase.

        Returns:
            list: Processed and filtered feature configuration list.
        """
        fg = self._load_fg_conf(fg_path, lower_case)
        fg = fg[FG_FEATURE_KEY]
        self._build_mc(fg)
        fg = self._filter_fg(fg)
        return fg

    def _load_fg_conf(self, fg_path, lower_case):
        """Load feature generation configuration from file.

        Args:
            fg_path (str): Path to the configuration file.
            lower_case (bool): Whether to convert keys to lowercase.

        Returns:
            list: Raw feature configuration list.
        """
        with open(fg_path) as f:
            fg = json.load(f)
        fg = dict_lower_case(fg, lower_case)
        return fg

    def _build_mc(self, fg):
        """Build model configuration from feature generation config.

        Args:
            fg (list): Feature generation configuration list.
        """
        candidate_seq_blocks = {}
        for fea_conf in fg:
            if self._is_seq(fea_conf):
                candidate_seq_blocks[fea_conf[SEQUENCE_NAME_KEY]] = fea_conf[
                    FEATURE_NAME_KEY
                ]
        self.mc_parser.init_blocks(candidate_seq_blocks)

    def _filter_fg(self, fg):
        """Filter feature configuration based on model configuration.

        Args:
            fg (list): Raw feature configuration list.

        Returns:
            list: Filtered feature configuration list.
        """
        filter_fg = []
        for fea_conf in fg:
            if self._is_seq(fea_conf):
                fea_name = fea_conf[SEQUENCE_NAME_KEY]
            else:
                fea_name = fea_conf[FEATURE_NAME_KEY]
            if self.mc_parser.has_fea(fea_name):
                if self._is_seq(fea_conf):
                    seq_fg = copy.deepcopy(fea_conf)
                    seq_fg[FG_FEATURE_KEY] = []
                    for seq_fea_conf in fea_conf[FG_FEATURE_KEY]:
                        seq_fea_name = seq_fea_conf[FEATURE_NAME_KEY]
                        if self.mc_parser.has_seq_fea(fea_name, seq_fea_name):
                            seq_fg[FG_FEATURE_KEY].append(seq_fea_conf)
                    filter_fg.append(seq_fg)
                else:
                    filter_fg.append(fea_conf)
        return filter_fg

    def _is_feature_copy(self, fea_conf):
        """Check if a feature configuration is a copy of another feature.

        Args:
            fea_conf (dict): Feature configuration dictionary.

        Returns:
            tuple: (is_copy: bool, copy_name: str or None)
        """
        copy_name = fea_conf.get(from_feature_key, None)
        is_copy = copy_name is not None
        return is_copy, copy_name

    def _is_seq(self, fea_conf):
        """Check if a feature configuration represents a sequence feature.

        Args:
            fea_conf (dict): Feature configuration dictionary.

        Returns:
            bool: True if it's a sequence feature, False otherwise.
        """
        return SEQUENCE_NAME_KEY in fea_conf

    def get_seq_len(self, fea_name):
        """Get sequence length for a sequence feature.

        Args:
            fea_name (str): Name of the sequence feature.

        Returns:
            int: Sequence length of the feature.

        Raises:
            RuntimeError: If the feature is not a sequence feature.
        """
        if fea_name in self.emb_conf_:
            return self.emb_conf_[fea_name].seq_length
        else:
            raise RuntimeError(f"feature: {fea_name} is not a seq feature")

    def _parse_feature_conf(self):
        """Parse all feature configurations into structured format.

        Returns:
            list: List of parsed feature configuration objects.
        """
        parsed_conf = []
        for fea_conf in self.fg_conf:
            if self._is_seq(fea_conf):
                seq_len = fea_conf[SEQUENCE_LENGTH_KEY]
                seq_name = fea_conf[SEQUENCE_NAME_KEY]
                for sub_fea_conf in fea_conf[FG_FEATURE_KEY]:
                    fc = self._parse_fg(
                        sub_fea_conf, seq_len=seq_len, seq_prefix=seq_name
                    )
                    parsed_conf.append(fc)
            else:
                fc = self._parse_fg(fea_conf, seq_len=0)
                parsed_conf.append(fc)
        return parsed_conf

    def _parse_fg(self, fea_conf, seq_len=0, seq_prefix=""):
        """Parse a single feature configuration.

        Args:
            fea_conf (dict): Feature configuration dictionary.
            seq_len (int, optional): Sequence length. Defaults to 0.
            seq_prefix (str, optional): Sequence name prefix. Defaults to "".

        Returns:
            FGConf: Parsed feature configuration object.
        """
        name = fea_conf[FEATURE_NAME_KEY]
        raw_name = name
        if seq_len > 0:
            name = seq_prefix + "_" + name
        is_sparse = fea_conf[feature_type_key].lower() != "raw_feature"
        gen_key_type = fea_conf[gen_key_type_key]
        gen_val_type = fea_conf[gen_value_type_key]
        # TODO(yuhuan.zh) maybe no need hash?
        need_hash = fea_conf[value_type_key].lower() == "string"
        hash_type = fea_conf.get(hash_type_key, "farmhash")
        hash_bucket_size = fea_conf.get(hash_bucket_key, 0)
        # deal with ev_hash
        if gen_key_type == "ev_hash":
            gen_key_type = "hash"
            hash_type = "ev_hash"
            hash_bucket_size = 0
        if gen_key_type == "ev_hash_mod":
            gen_key_type = "hash"
            hash_type = "ev_hash"
        hash_type = HASH_TYPE_MAP[hash_type]
        is_seq = seq_len > 0
        seq_length = seq_len
        value_dimension = fea_conf.get(value_dim_key, 1)
        combiner = fea_conf.get(combiner_key, "mean")
        embedding_dim = fea_conf.get(emb_dim_key, None)
        value_type = fea_conf[value_type_key].lower()
        boundaries = fea_conf.get(boundaries_key, None)
        boundaries = (
            boundaries
            if boundaries is None
            else list(map(float, boundaries.split(",")))
        )
        mask_value = str(fea_conf.get(mask_value_key, "0"))
        mask_value = mask_value if mask_value is None else list(mask_value.split(";"))
        compress_strategy = fea_conf.get(compress_strategy_key, None)
        # add to multihash configs
        if compress_strategy is not None:
            self.multihash_conf_[name] = compress_strategy
        shared_name = fea_conf.get(shared_name_key, raw_name)
        _, from_feature = self._is_feature_copy(fea_conf)
        emb_device = fea_conf.get(emb_device_key, None)
        emb_type = fea_conf.get(emb_type_key, None)
        trainable = fea_conf.get(trainable_key, True)
        admit_hook = fea_conf.get(admit_hook_key, None)
        filter_hook = fea_conf.get(filter_hook_key, None)
        return FGConf(
            name,
            is_sparse,
            gen_key_type,
            gen_val_type,
            need_hash,
            hash_type,
            hash_bucket_size,
            is_seq=is_seq,
            seq_length=seq_length,
            value_dimension=value_dimension,
            combiner=combiner,
            embedding_dim=embedding_dim,
            value_type=value_type,
            boundaries=boundaries,
            mask_value=mask_value,
            compress_strategy=compress_strategy,
            shared_name=shared_name,
            from_feature=from_feature,
            emb_device=emb_device,
            emb_type=emb_type,
            trainable=trainable,
            admit_hook=admit_hook,
            filter_hook=filter_hook,
        )

    def _parse_emb_type(self, fea_conf):
        """Parse embedding transformation type from feature configuration.

        Args:
            fea_conf (FGConf): Feature configuration object.

        Returns:
            EmbTransformType: Parsed embedding transformation type.

        Raises:
            RuntimeError: If the gen_value_type is not supported.
        """
        trans_t = None
        if fea_conf.gen_value_type == "lookup":
            trans_t = EmbTransformType.LOOKUP
        elif fea_conf.gen_value_type == "multihash_lookup":
            trans_t = EmbTransformType.MULTIHASH_LOOKUP
        elif fea_conf.gen_value_type == "idle":
            trans_t = EmbTransformType.RAW
        else:
            raise RuntimeError(
                f"Not support gen_value_type: {fea_conf.gen_value_type} in feature {fea_conf}"
            )
        return trans_t

    def _parse_id_type(self, fea_conf):
        """Parse ID transformation type from feature configuration.

        This method determines the appropriate ID transformation type based on
        the feature's gen_key_type and the parser's configuration settings
        (already_hashed, hash_in_io, devel_mode).

        Args:
            fea_conf (FGConf): Feature configuration object.

        Returns:
            IdTransformType: Parsed ID transformation type.

        Raises:
            NotImplementedError: If mask type is used in non-development mode.
            RuntimeError: If the gen_key_type is not supported.
        """
        trans_t = None
        if fea_conf.gen_key_type == "idle":
            trans_t = IdTransformType.RAW
        elif fea_conf.gen_key_type == "boundary":
            trans_t = IdTransformType.BUCKETIZE
        elif fea_conf.gen_key_type == "hash":
            if self.already_hashed:
                if fea_conf.hash_bucket_size > 0:
                    trans_t = IdTransformType.MOD
                else:
                    trans_t = IdTransformType.RAW
            elif self.hash_in_io:
                trans_t = IdTransformType.RAW
            else:
                trans_t = IdTransformType.HASH
        elif fea_conf.gen_key_type == "mask":
            if self.already_hashed or self.hash_in_io:
                raise ValueError(
                    "Not support gen_key_type mask, when already_hashed or hash_in_io is True"
                )
            trans_t = IdTransformType.MASK
        elif fea_conf.gen_key_type == "multihash":
            if self.already_hashed:
                if fea_conf.hash_bucket_size > 0:
                    trans_t = IdTransformType.MOD_MULTIHASH
                else:
                    trans_t = IdTransformType.MULTIHASH
            elif self.hash_in_io:
                trans_t = IdTransformType.MULTIHASH
            else:
                trans_t = IdTransformType.HASH_MULTIHASH
        else:
            raise RuntimeError(
                f"Not support gen_key_type: {fea_conf.gen_key_type} in feature {fea_conf}"
            )
        return trans_t

    def _parse_dtype_dim(self, fea_conf):
        """Parse data type and dimension from feature configuration.

        This method determines the appropriate PyTorch data type and dimension
        for a feature based on its value type and configuration settings.

        Args:
            fea_conf (FGConf): Feature configuration object.

        Returns:
            tuple: A tuple containing (dtype: torch.dtype, dim: int or None).

        Raises:
            NotImplementedError: If string type feature uses idle or mask
                gen_key_type in non-development mode.
        """
        dtype = VALUE_TYPE_MAP[fea_conf.value_type]
        dim = fea_conf.value_dimension
        if fea_conf.value_type == "string":
            dim = None
            if self.already_hashed or self.hash_in_io:
                dtype = torch.int64
            elif fea_conf.gen_key_type == "mask":
                dim = fea_conf.value_dimension
            # TODO(yuhuan.zh) support string input raw feature
            if fea_conf.gen_key_type in ["idle"]:
                dim = fea_conf.value_dimension
                if self.devel_mode:
                    logger.warning(
                        f"String type feature: {fea_conf} not support idle yet, maybe get wrong value"
                    )
                else:
                    raise NotImplementedError(
                        f"String type feature: {fea_conf} not support idle yet."
                    )
        return dtype, dim

    def _init_emb_conf(self):
        """Initialize embedding configurations for all parsed features.

        This method creates FeatureEmbConf objects for each parsed feature
        configuration, determining the appropriate transformation types,
        data types, and other embedding parameters.

        Returns:
            OrderedDict: Dictionary mapping feature names to FeatureEmbConf objects.
        """
        emb_conf = OrderedDict()
        for fea_conf in self.parsed_conf_:
            id_type = self._parse_id_type(fea_conf)
            emb_type = self._parse_emb_type(fea_conf)
            dtype, dim = self._parse_dtype_dim(fea_conf)
            io_name = (
                fea_conf.name
                if fea_conf.from_feature is None
                else fea_conf.from_feature
            )
            io_name = io_name.lower() if self.io_name_lower_case else io_name
            is_float = dtype in (torch.float32, torch.float64)
            ec = FeatureEmbConf(
                io_name=io_name,
                out_name=fea_conf.name,
                id_transform_type=id_type,
                emb_transform_type=emb_type,
                embedding_dim=fea_conf.embedding_dim,
                raw_dim=dim,
                shared_name=fea_conf.shared_name,
                hash_bucket_size=fea_conf.hash_bucket_size,
                hash_type=fea_conf.hash_type,
                boundaries=fea_conf.boundaries,
                mask_value=fea_conf.mask_value,
                compress_strategy=fea_conf.compress_strategy,
                combiner=fea_conf.combiner,
                seq_length=fea_conf.seq_length,
                dtype=dtype,
                emb_device=fea_conf.emb_device,
                emb_type=EMB_TYPE_MAP[fea_conf.emb_type]
                if fea_conf.emb_type is not None
                else None,
                trainable=fea_conf.trainable,
                admit_hook=fea_conf.admit_hook,
                filter_hook=fea_conf.filter_hook,
                fill_seq=(fea_conf.seq_length > 0 and is_float),
            )
            emb_conf[fea_conf.name] = ec
        return emb_conf

    def _init_io_conf(self):
        """Initialize I/O configurations for all parsed features.

        This method creates FeatureIOConf objects for each parsed feature
        configuration, determining the appropriate I/O parameters such as
        variable length format, hash settings, and dimensions.

        Returns:
            OrderedDict: Dictionary mapping feature names to FeatureIOConf objects.
        """
        io_conf = OrderedDict()
        for fea_conf in self.parsed_conf_:
            real_name = (
                fea_conf.name
                if fea_conf.from_feature is None
                else fea_conf.from_feature
            )
            real_name = real_name.lower() if self.io_name_lower_case else real_name
            varlen = self._is_io_sparse(fea_conf)
            hash_type, trans_int, hash_bucket = self._get_io_hash_args(fea_conf)
            dtype, _ = self._parse_dtype_dim(fea_conf)
            is_float = dtype in (torch.float32, torch.float64)
            seq_length = fea_conf.seq_length if fea_conf.seq_length else None
            fc = FeatureIOConf(
                name=real_name,
                varlen=varlen,
                hash_type=hash_type,
                hash_bucket_size=hash_bucket,
                trans_int=trans_int,
                dim=fea_conf.value_dimension,
                seq_length=seq_length,
                fill_seq=(seq_length is not None and is_float),
            )
            io_conf[real_name] = fc
        return io_conf

    def _is_io_sparse(self, conf):
        """Determine if a feature should use sparse I/O format.

        Args:
            conf (FGConf): Feature configuration object.

        Returns:
            bool: True if the feature should use sparse format, False otherwise.
        """
        # cannot convert bucketize features to sparse
        sparse_format = conf.is_sparse or conf.is_seq
        return sparse_format

    def _get_io_hash_args(self, conf):
        """Get I/O hash arguments for a feature configuration.

        This method determines the appropriate hash settings for I/O operations
        based on the feature configuration and parser settings.

        Args:
            conf (FGConf): Feature configuration object.

        Returns:
            tuple: A tuple containing (hash_type: str or None, trans_int: bool,
                   hash_bucket: int).
        """
        need_hash = conf.need_hash
        hash_bucket = conf.hash_bucket_size
        hash_type = conf.hash_type if need_hash else None
        trans_int = False
        if self.already_hashed:
            hash_type = None
            hash_bucket = 0
            trans_int = False
        elif need_hash and (not self.hash_in_io):
            hash_type = None
            hash_bucket = 0
            trans_int = True
        return hash_type, trans_int, hash_bucket
