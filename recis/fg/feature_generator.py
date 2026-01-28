from collections import OrderedDict

import torch

from recis.features.feature import Feature
from recis.features.op import (
    Bucketize,
    Hash,
    IDMultiHash,
    Mod,
    SelectField,
    SequenceTruncate,
    StringMultiHash,
)
from recis.fg.fg_parser import EmbTransformType, FGParser, IdTransformType
from recis.fg.mc_parser import MCParser
from recis.fg.shape_manager import ShapeManager
from recis.fg.utils import (
    get_multihash_name,
    get_multihash_shared_name,
    parse_multihash,
)
from recis.io.dataset_base import DatasetBase
from recis.nn.hashtable_hook import AdmitHook, FilterHook
from recis.nn.initializers import (
    ConstantInitializer,
    NormalInitializer,
    UniformInitializer,
    XavierNormalInitializer,
    XavierUniformInitializer,
)
from recis.nn.modules.embedding import EmbeddingOption
from recis.utils.logger import Logger


logger = Logger(__name__)

INITIALIZER_MAPPING = {
    "constant": ConstantInitializer,
    "uniform": UniformInitializer,
    "normal": NormalInitializer,
    "xavier_normal": XavierNormalInitializer,
    "xavier_uniform": XavierUniformInitializer,
}
INITIALIZER_DEFAULT_KWARGS = {
    "constant": {"init_val": 0.0},
    "uniform": {"a": -2e-5, "b": 2e-5},
    "normal": {"mean": 0.0, "std": 1e-5},
    "xavier_normal": {"gain": 1.0},
    "xavier_uniform": {"gain": 1.0},
}


class FG:
    """Feature Generator for managing feature configurations and embeddings.

    The FG class serves as the main interface for feature generation in the RecIS
    system. It manages feature parsing, shape inference, embedding configurations,
    and provides utilities for building feature pipelines with proper initialization
    and device management.

    Key Features:
        - Feature configuration parsing and validation
        - Automatic shape inference for features and blocks
        - Embedding configuration management with multiple initializers
        - Support for both hash table and bucket embeddings
        - Multi-hash feature support for advanced embedding strategies
        - Integration with dataset I/O operations

    Attributes:
        fg_parser (FGParser): Parser for feature configuration files.
        shape_manager (ShapeManager): Manager for feature and block shapes.
        use_coalesce (bool): Whether to use coalesced operations for efficiency.
        grad_reduce_by (str): Gradient reduction strategy ("worker" or other).
        embedding_initializer: Initializer class for embedding parameters.
        emb_default_class (str): Default embedding class ("hash_table" or "bucket_emb").
        emb_default_device (str): Default device for embeddings ("cpu" or "cuda").
        emb_default_type (torch.dtype): Default data type for embeddings.
        init_kwargs (dict): Keyword arguments for embedding initialization.
        _labels (dict): Dictionary storing label configurations.
        _ids (set): Set of ID feature names.
    """

    def __init__(
        self,
        fg_parser: FGParser,
        shape_manager: ShapeManager,
        use_coalesce=True,
        grad_reduce_by="worker",
        initializer="uniform",
        init_kwargs=None,
        emb_default_class="hash_table",
        emb_default_device="cuda",
        emb_default_type=torch.float32,
    ):
        """Initialize the Feature Generator.

        Args:
            fg_parser (FGParser): Parser for feature configuration files.
            shape_manager (ShapeManager): Manager for feature and block shapes.
            use_coalesce (bool, optional): Whether to use coalesced operations.
                Defaults to True.
            grad_reduce_by (str, optional): Gradient reduction strategy.
                Defaults to "worker".
            initializer (str, optional): Embedding initializer type. Must be one of
                "constant", "uniform", "normal", "xavier_normal", "xavier_uniform".
                Defaults to "uniform".
            init_kwargs (dict, optional): Custom initialization parameters.
                If None, uses default parameters for the specified initializer.
            emb_default_class (str, optional): Default embedding class.
                Must be "hash_table" or "bucket_emb". Defaults to "hash_table".
            emb_default_device (str, optional): Default device for embeddings.
                Must be "cpu" or "cuda". Defaults to "cuda".
            emb_default_type (torch.dtype, optional): Default data type for embeddings.
                Defaults to torch.float32.

        Raises:
            ValueError: If emb_default_class is not "hash_table" or "bucket_emb".
            ValueError: If emb_default_device is not "cpu" or "cuda".
            NotImplementedError: If bucket embedding is selected (not yet implemented).
        """
        self.fg_parser = fg_parser
        self.shape_manager = shape_manager
        self.use_coalesce = use_coalesce
        self.grad_reduce_by = grad_reduce_by
        self.embedding_initializer = INITIALIZER_MAPPING[initializer]
        if emb_default_class not in ["hash_table", "bucket_emb"]:
            raise ValueError(
                f"emb_default_class must be one of `hash_table|bucket_emb` got {emb_default_class}"
            )
        self.emb_default_class = emb_default_class
        if emb_default_device not in ["cpu", "cuda"]:
            raise ValueError(
                f"emb_default_device must be one of `cpu|cuda` got {emb_default_device}"
            )
        self.emb_default_device = emb_default_device
        self.emb_default_type = emb_default_type
        # TODO(yuhuan.zh) enable bucket embedding
        if not self.emb_default_class == "hash_table":
            raise NotImplementedError("Bucketize Embedding not impletened yet.")
        if init_kwargs is None:
            init_kwargs = INITIALIZER_DEFAULT_KWARGS[initializer]
        self.init_kwargs = init_kwargs
        self._labels = dict()
        self._ids = set()

    @property
    def feature_blocks(self):
        """Get feature blocks from the parser.

        Returns:
            dict: Dictionary mapping block names to feature lists.
        """
        return self.fg_parser.feature_blocks

    @property
    def seq_block_names(self):
        """Get sequence block names from the parser.

        Returns:
            list: List of sequence block names.
        """
        return self.fg_parser.seq_block_names

    @property
    def sample_ids(self):
        """Get list of sample ID feature names.

        Returns:
            list: List of ID feature names.
        """
        return list(self._ids)

    @property
    def labels(self):
        """Get list of label names.

        Returns:
            list: List of label names.
        """
        return list(self._labels)

    @property
    def feature_shapes(self):
        """Get feature shapes from the shape manager.

        Returns:
            dict: Dictionary mapping feature names to their shapes.
        """
        return self.shape_manager.feature_shapes

    @property
    def block_shapes(self):
        """Get block shapes from the shape manager.

        Returns:
            dict: Dictionary mapping block names to their shapes.
        """
        return self.shape_manager.block_shapes

    def get_mc_conf(self):
        """Get mc configuration dictionary.

        Returns:
            dict: mc configuration settings.
        """

        return self.fg_parser.get_mc_conf()

    def get_fg_conf(self):
        """Get fg configuration dictionary.

        Returns:
            dict: fg configuration settings.
        """
        return self.fg_parser.get_fg_conf()

    def is_seq_block(self, block_name):
        """Check if a block is a sequence block.

        Args:
            block_name (str): Name of the block to check.

        Returns:
            bool: True if the block is a sequence block, False otherwise.

        Raises:
            RuntimeError: If the block name is not found in feature blocks.
        """
        if block_name not in self.feature_blocks:
            raise RuntimeError(f"block name: {block_name} not used in mc, please check")
        return block_name in self.seq_block_names

    @property
    def multihash_conf(self):
        """Get multi-hash configuration from the parser.

        Returns:
            dict: Multi-hash configuration dictionary.
        """
        return self.fg_parser.multihash_conf

    def get_block_seq_len(self, block_name):
        """Get sequence length for a sequence block.

        Args:
            block_name (str): Name of the sequence block.

        Returns:
            int: Sequence length of the block.
        """
        feature_name = self.feature_blocks[block_name][0]
        return self.fg_parser.get_seq_len(feature_name)

    def add_label(self, label_name, dim=1, default_value=0.0):
        """Add a label configuration.

        Args:
            label_name (str): Name of the label.
            dim (int, optional): Dimension of the label. Defaults to 1.
            default_value (float, optional): Default value for the label.
                Defaults to 0.0.
        """
        self._labels[label_name] = (dim, default_value)

    def add_id(self, id_name):
        """Add an ID feature name.

        Args:
            id_name (str): Name of the ID feature.
        """
        self._ids.add(id_name)

    def get_shape(self, name):
        """Get shape for a feature or block by name.

        Args:
            name (str): Name of the feature or block.

        Returns:
            list: Shape of the specified feature or block.
        """
        return self.shape_manager.get_shape(name)

    def has_shape_context(self, context_name):
        """Check if a shape context exists.

        Args:
            context_name (str): Name of the shape context.

        Returns:
            bool: True if the context exists, False otherwise.
        """
        return self.shape_manager.has_shape_context(context_name)

    def regist_shape_context(self, context_name):
        """Register a new shape context.

        Args:
            context_name (str): Name of the shape context to register.
        """
        self.shape_manager.regist_shape_context(context_name)

    def set_context_shape(self, context_name, name, shape):
        """Set shape for a specific name within a context.

        Args:
            context_name (str): Name of the shape context.
            name (str): Name within the context.
            shape (list): Shape to set.
        """
        self.shape_manager.set_context_shape(context_name, name, shape)

    def get_context_shapes(self, context_name):
        """Get all shapes within a context.

        Args:
            context_name (str): Name of the shape context.

        Returns:
            dict: Dictionary mapping names to shapes within the context.
        """
        return self.shape_manager.get_context_shapes(context_name)

    def get_context_shape(self, context_name, name):
        """Get shape for a specific name within a context.

        Args:
            context_name (str): Name of the shape context.
            name (str): Name within the context.

        Returns:
            list: Shape of the specified name within the context.
        """
        return self.shape_manager.get_context_shape(context_name, name)

    def get_feature_shape(self, fea_name):
        """Get shape for a specific feature.

        Args:
            fea_name (str): Name of the feature.

        Returns:
            list: Shape of the specified feature.
        """
        return self.shape_manager.get_feature_shape(fea_name)

    def get_block_shape(self, block_name):
        """Get shape for a specific block.

        Args:
            block_name (str): Name of the block.

        Returns:
            list: Shape of the specified block.
        """
        return self.shape_manager.get_block_shape(block_name)

    def add_io_features(self, dataset: DatasetBase):
        """Add I/O features to a dataset based on parser configurations.

        This method configures the dataset with features from the parser's I/O
        configurations, adds label features with their dimensions and default values,
        and adds variable-length ID features.

        Args:
            dataset (DatasetBase): Dataset to configure with features.
        """
        dataset.parse_from(self.fg_parser.io_configs.values())
        for label_name, label_conf in self._labels.items():
            dataset.fixedlen_feature(
                label_name, default_value=[label_conf[1]] * label_conf[0]
            )
        for id_name in self._ids:
            dataset.varlen_feature(id_name)

    def get_emb_confs(self):
        """Generate embedding configurations for all features.

        This method processes all embedding configurations from the parser and
        creates EmbeddingOption objects with appropriate settings for device,
        data type, initializer, and hooks.

        Returns:
            OrderedDict: Dictionary mapping embedding names to EmbeddingOption objects.

        Raises:
            RuntimeError: If an unsupported transform configuration is encountered.
        """
        emb_dict = OrderedDict()
        for conf in self.fg_parser.emb_configs.values():
            if conf.emb_transform_type == EmbTransformType.RAW:
                continue
            elif conf.emb_transform_type == EmbTransformType.LOOKUP:
                device = conf.emb_device or self.emb_default_device
                device = (
                    torch.device("cuda") if device == "cuda" else torch.device("cpu")
                )
                dtype = (
                    self.emb_default_type if conf.emb_type is None else conf.emb_type
                )
                # TODO(yuhuan.zh) enable bucket_emb when hash_bucket_size > 0
                emb_dict[conf.out_name] = EmbeddingOption(
                    embedding_dim=conf.embedding_dim,
                    shared_name=conf.shared_name,
                    combiner=conf.combiner,
                    initializer=self.embedding_initializer(**self.init_kwargs),
                    grad_reduce_by=self.grad_reduce_by,
                    use_weight=False,
                    device=device,
                    dtype=dtype,
                    trainable=conf.trainable,
                    admit_hook=None
                    if conf.admit_hook is None
                    else AdmitHook(**conf.admit_hook),
                    filter_hook=None
                    if conf.filter_hook is None
                    else FilterHook(**conf.filter_hook),
                )
            elif conf.emb_transform_type == EmbTransformType.MULTIHASH_LOOKUP:
                prefix, buckets, _, shared = parse_multihash(conf.compress_strategy)
                mh_num = len(buckets)
                device = conf.emb_device or self.emb_default_device
                device = (
                    torch.device("cuda") if device == "cuda" else torch.device("cpu")
                )
                dtype = self.emb_default_type
                for i in range(mh_num):
                    # TODO(yuhuan.zh) enable bucket_emb when hash_bucket_size > 0
                    out_name = get_multihash_name(conf.out_name, prefix, i)
                    if shared:
                        shared_name = conf.shared_name
                    else:
                        shared_name = get_multihash_shared_name(
                            conf.shared_name, prefix, i
                        )
                    emb_dict[out_name] = EmbeddingOption(
                        embedding_dim=conf.embedding_dim,
                        shared_name=shared_name,
                        combiner=conf.combiner,
                        initializer=self.embedding_initializer(**self.init_kwargs),
                        grad_reduce_by=self.grad_reduce_by,
                        use_weight=False,
                        device=device,
                        dtype=dtype,
                        trainable=conf.trainable,
                        admit_hook=None
                        if conf.admit_hook is None
                        else AdmitHook(**conf.admit_hook),
                        filter_hook=None
                        if conf.filter_hook is None
                        else FilterHook(**conf.filter_hook),
                    )
            else:
                raise RuntimeError(f"Not support transform config: {conf}")
        return emb_dict

    def get_feature_confs(self):
        """Generate feature configurations for all features.

        This method processes all embedding configurations from the parser and
        creates Feature objects with appropriate operations based on the
        transformation types (bucketize, hash, mod, etc.).

        Returns:
            list: List of Feature objects with configured operations.

        Raises:
            RuntimeError: If an unsupported ID transform type is encountered.
        """
        feature_confs = []
        for conf in self.fg_parser.emb_configs.values():
            fea_conf = Feature(conf.out_name).add_op(
                SelectField(conf.io_name, dim=conf.raw_dim)
            )
            dtype = conf.dtype
            if conf.id_transform_type in [
                IdTransformType.RAW,
                IdTransformType.MULTIHASH,
            ]:
                pass
            elif conf.id_transform_type == IdTransformType.BUCKETIZE:
                fea_conf = fea_conf.add_op(Bucketize(conf.boundaries))
                dtype = torch.int64
            elif conf.id_transform_type in [
                IdTransformType.HASH,
                IdTransformType.HASH_MULTIHASH,
            ]:
                fea_conf = fea_conf.add_op(Hash(conf.hash_type))
                dtype = torch.int64
                if conf.hash_bucket_size > 0:
                    fea_conf = fea_conf.add_op(Mod(conf.hash_bucket_size))
            elif conf.id_transform_type in [
                IdTransformType.MOD,
                IdTransformType.MOD_MULTIHASH,
            ]:
                fea_conf = fea_conf.add_op(Mod(conf.hash_bucket_size))
            else:
                raise RuntimeError(f"Not support transform config: {conf}")
            if conf.seq_length:
                fea_conf = fea_conf.add_op(
                    SequenceTruncate(
                        seq_len=conf.seq_length,
                        truncate=True,
                        truncate_side="right",
                        check_length=False,
                        n_dims=3,
                        dtype=dtype,
                    )
                )
            if conf.id_transform_type in [
                IdTransformType.MULTIHASH,
                IdTransformType.HASH_MULTIHASH,
                IdTransformType.MOD_MULTIHASH,
            ]:
                prefix, num_buckets, _, _ = parse_multihash(conf.compress_strategy)
                if prefix == "mh":
                    fea_conf = fea_conf.add_op(op=StringMultiHash(num_buckets, prefix))
                elif prefix == "yx":
                    fea_conf = fea_conf.add_op(IDMultiHash(num_buckets, prefix))
                else:
                    raise NotImplementedError(f"multihash type {prefix} is not support")

            feature_confs.append(fea_conf)
        return feature_confs


def build_fg(
    fg_conf_path,
    mc_conf_path=None,
    mc_config=None,
    fg_parser_class=FGParser,
    mc_parser_class=MCParser,
    fg_class=FG,
    shape_manager_class=ShapeManager,
    uses_columns=None,
    lower_case=False,
    with_seq_prefix=False,
    already_hashed=False,
    hash_in_io=False,
    devel_mode=False,
    **kwargs,
):
    """Build a complete Feature Generator with all necessary components.

    This factory function creates and initializes all components needed for
    feature generation: MC parser, FG parser, shape manager, and the main
    FG instance. It provides a convenient way to set up the entire feature
    generation pipeline with proper configuration.

    Args:
        fg_conf_path (str): Path to the feature generation configuration file.
        mc_conf_path (str, optional): Path to the MC configuration file.
            Either this or mc_config must be provided.
        mc_config (dict, optional): MC configuration dictionary.
            Either this or mc_conf_path must be provided.
        fg_parser_class (type, optional): FGParser class to use.
            Defaults to FGParser.
        mc_parser_class (type, optional): MCParser class to use.
            Defaults to MCParser.
        fg_class (type, optional): FG class to use. Defaults to FG.
        shape_manager_class (type, optional): ShapeManager class to use.
            Defaults to ShapeManager.
        uses_columns (list, optional): List of column names to use.
            If None, uses all columns.
        lower_case (bool, optional): Whether to convert configuration keys
            to lowercase. Defaults to False.
        with_seq_prefix (bool, optional): Whether the feature name already has sequence block name as prefix.
            Defaults to False.
        already_hashed (bool, optional): Whether features are already hashed.
            Defaults to False.
        hash_in_io (bool, optional): Whether to perform hashing in I/O layer.
            Defaults to False.
        devel_mode (bool, optional): Whether to enable development mode.
            Defaults to False.
        **kwargs: Additional keyword arguments passed to the FG constructor.

    Returns:
        FG: Configured Feature Generator instance ready for use.

    Example:
        .. code-block:: python

            # Build FG with file paths
            fg = build_fg(
                fg_conf_path="features.json",
                mc_conf_path="model_config.json",
                initializer="xavier_uniform",
                emb_default_device="cuda",
            )

            # Build FG with configuration dictionary
            fg = build_fg(
                fg_conf_path="features.json",
                mc_config={"block1": ["feature1", "feature2"]},
                uses_columns=["block1"],
            )
    """
    mc_parser = mc_parser_class(
        mc_config_path=mc_conf_path,
        mc_config=mc_config,
        uses_columns=uses_columns,
        lower_case=lower_case,
        with_seq_prefix=with_seq_prefix,
    )
    fg_parser = fg_parser_class(
        fg_conf_path,
        mc_parser,
        already_hashed=already_hashed,
        hash_in_io=hash_in_io,
        lower_case=lower_case,
        devel_mode=devel_mode,
    )
    shape_manager = shape_manager_class(fg_parser)
    fg = fg_class(fg_parser, shape_manager, **kwargs)
    return fg
