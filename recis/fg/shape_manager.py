import copy

from recis.fg.fg_parser import EmbTransformType
from recis.fg.utils import parse_multihash


class ShapeManager:
    """Shape manager for inferring and managing feature and block dimensions.

    The ShapeManager class is responsible for automatically inferring the shapes
    of features and blocks based on their configurations, and providing utilities
    to manage shape contexts for different processing stages. It handles various
    embedding transformation types and ensures proper shape compatibility for
    feature concatenation within blocks.

    Key Features:
        - Automatic shape inference for features and blocks
        - Support for sequence features with variable lengths
        - Multi-hash embedding shape calculation
        - Shape context management for different processing stages
        - Shape compatibility validation for block concatenation

    Attributes:
        fg_parser: Feature generation parser instance.
        feature_shape (dict): Dictionary mapping feature names to their shapes.
        block_shape (dict): Dictionary mapping block names to their shapes.
        shape_context (dict): Dictionary storing shape contexts for different stages.
    """

    def __init__(self, fg_parser):
        """Initialize the Shape Manager.

        Args:
            fg_parser: Feature generation parser instance containing
                embedding configurations and feature definitions.
        """
        self.fg_parser = fg_parser
        self.feature_shape = {}
        self.block_shape = {}
        self.shape_context = {}
        self._infer_shape()

    @property
    def feature_shapes(self):
        """Get all feature shapes.

        Returns:
            dict: Dictionary mapping feature names to their shape lists.
        """
        return self.feature_shape

    @property
    def block_shapes(self):
        """Get all block shapes.

        Returns:
            dict: Dictionary mapping block names to their shape lists.
        """
        return self.block_shape

    def get_shape(self, name):
        """Get shape for a feature or block by name.

        This method first checks if the name corresponds to a block,
        then checks if it's a feature. It provides a unified interface
        for retrieving shapes regardless of the entity type.

        Args:
            name (str): Name of the feature or block.

        Returns:
            list: Shape of the specified feature or block.

        Raises:
            RuntimeError: If the name is not found in either blocks or features.
        """
        if name in self.block_shape:
            return self.get_block_shape(name)
        elif name in self.feature_shape:
            return self.get_feature_shape(name)
        else:
            raise RuntimeError(f"name: {name} not used in mc, please check.")

    def get_feature_shape(self, fea_name):
        """Get shape for a specific feature.

        Args:
            fea_name (str): Name of the feature.

        Returns:
            list: Shape of the specified feature.

        Raises:
            RuntimeError: If the feature name is not found in the configuration.
        """
        if fea_name not in self.feature_shape:
            raise RuntimeError(
                f"feature name: {fea_name} not used in mc, please check."
            )
        return self.feature_shape[fea_name]

    def get_block_shape(self, block_name):
        """Get shape for a specific block.

        Args:
            block_name (str): Name of the block.

        Returns:
            list: Shape of the specified block.

        Raises:
            RuntimeError: If the block name is not found in the configuration.
        """
        if block_name not in self.block_shape:
            raise RuntimeError(
                f"block name: {block_name} not used in mc, please check."
            )
        return self.block_shape[block_name]

    def has_shape_context(self, context_name):
        """Check if a shape context exists.

        Args:
            context_name (str): Name of the shape context.

        Returns:
            bool: True if the context exists, False otherwise.
        """
        return context_name in self.shape_context

    def regist_shape_context(self, context_name):
        """Register a new shape context.

        Args:
            context_name (str): Name of the shape context to register.

        Raises:
            RuntimeError: If the context name is already registered.
        """
        if context_name in self.shape_context:
            raise RuntimeError(f"context : {context_name} already registed.")
        self.shape_context[context_name] = {}

    def set_context_shape(self, context_name, name, shape):
        """Set shape for a specific name within a context.

        Args:
            context_name (str): Name of the shape context.
            name (str): Name within the context.
            shape (list): Shape to set.

        Raises:
            RuntimeError: If the context name is not registered.
        """
        if context_name not in self.shape_context:
            raise RuntimeError(f"context name: {context_name} not registed")
        self.shape_context[context_name][name] = shape

    def get_context_shape(self, context_name, name):
        """Get shape for a specific name within a context.

        Args:
            context_name (str): Name of the shape context.
            name (str): Name within the context.

        Returns:
            list: Shape of the specified name within the context.

        Raises:
            RuntimeError: If the context name is not registered or the name
                is not found within the context.
        """
        if context_name not in self.shape_context:
            raise RuntimeError(f"context name: {context_name} not registed.")
        if name not in self.shape_context[context_name]:
            raise RuntimeError(f"name: {name} not in context: {context_name}")
        return self.shape_context[context_name][name]

    def get_context_shapes(self, context_name):
        """Get all shapes within a context.

        Args:
            context_name (str): Name of the shape context.

        Returns:
            dict: Dictionary mapping names to shapes within the context.

        Raises:
            RuntimeError: If the context name is not registered.
        """
        if context_name not in self.shape_context:
            raise RuntimeError(f"context name: {context_name} not registed.")
        return self.shape_context[context_name]

    def _infer_shape(self):
        """Infer shapes for all features and blocks.

        This method orchestrates the shape inference process by first
        inferring individual feature shapes, then inferring block shapes
        based on the features they contain.
        """
        self._infer_feature_shape()
        self._infer_block_shape()

    def _infer_feature_shape(self):
        """Infer shapes for all individual features.

        This method processes each embedding configuration to determine
        the appropriate shape based on the embedding transformation type,
        sequence length, and embedding dimensions.

        Raises:
            RuntimeError: If an unsupported embedding transform type is encountered.
        """
        for conf in self.fg_parser.emb_configs.values():
            shape = [-1]
            if conf.seq_length:
                shape.append(conf.seq_length)
            if conf.emb_transform_type == EmbTransformType.RAW:
                shape.append(conf.raw_dim)
            elif conf.emb_transform_type == EmbTransformType.LOOKUP:
                shape.append(conf.embedding_dim)
            elif conf.emb_transform_type == EmbTransformType.MULTIHASH_LOOKUP:
                _, buckets, mh_combiner, _ = parse_multihash(conf.compress_strategy)
                real_dim = conf.embedding_dim
                if mh_combiner == "concat":
                    real_dim *= len(buckets)
                shape.append(real_dim)
            else:
                raise RuntimeError(f"Not support emb transform type: {conf}")
            self.feature_shape[conf.out_name] = shape

    def _infer_block_shape(self):
        """Infer shapes for all feature blocks.

        This method processes each feature block to determine the combined
        shape when features are concatenated. It validates that all features
        within a block have compatible shapes for concatenation.

        Raises:
            RuntimeError: If a block has no features or if features within
                a block have incompatible shapes for concatenation.
        """
        for block_name, feas in self.fg_parser.feature_blocks.items():
            if len(feas) == 0:
                raise RuntimeError(
                    f"mc config error, block: {block_name} has no features"
                )
            self.block_shape[block_name] = copy.deepcopy(
                self.get_feature_shape(feas[0])
            )
            for fea in feas[1:]:
                cur_shape = self.get_feature_shape(fea)
                check_ok = self._check_block_shape(
                    cur_shape, self.block_shape[block_name]
                )
                if not check_ok:
                    raise RuntimeError(
                        f"features: {feas} in block name: {block_name} cannot concat, current feature shape [{fea}: {cur_shape}], block current shape: [{self.block_shape[block_name]}]"
                    )
                self.block_shape[block_name][-1] = (
                    self.block_shape[block_name][-1] + cur_shape[-1]
                )

    def _check_block_shape(self, cur_shape, dst_shape):
        """Check if a feature shape is compatible with the current block shape.

        This method validates that a feature can be concatenated with existing
        features in a block by checking dimension compatibility.

        Args:
            cur_shape (list): Shape of the current feature.
            dst_shape (list): Current shape of the block.

        Returns:
            bool: True if shapes are compatible for concatenation, False otherwise.
        """
        if not len(cur_shape) == len(dst_shape):
            return False
        for i in range(1, len(cur_shape) - 1):
            if not cur_shape[i] == dst_shape[i]:
                return False
        return True
