import torch
import torch.nn as nn

from recis.features.feature_engine import FeatureEngine
from recis.fg.utils import get_multihash_name, parse_multihash
from recis.nn.modules.embedding_engine import EmbeddingEngine


class BlockBuilder(nn.Module):
    """Block builder for organizing and combining feature embeddings.

    The BlockBuilder class is responsible for processing feature embeddings
    and organizing them into blocks for model consumption. It handles multi-hash
    feature combination, block concatenation, and sequence block separation.

    Key Features:
        - Multi-hash feature combination with configurable combiners
        - Feature block concatenation for dense representations
        - Sequence block separation for specialized processing
        - Automatic data type conversion for compatibility

    Attributes:
        feature_blocks (dict): Dictionary mapping block names to feature lists.
        multihash_conf (dict): Configuration for multi-hash feature processing.
        sequence_blocks (list): List of sequence block names for separation.
    """

    def __init__(self, feature_blocks, multihash_conf=None, sequence_blocks=None):
        """Initialize the Block Builder.

        Args:
            feature_blocks (dict): Dictionary mapping block names to lists of
                feature names that should be concatenated together.
            multihash_conf (dict, optional): Multi-hash configuration dictionary
                mapping feature names to their multi-hash configurations.
                Defaults to None.
            sequence_blocks (list, optional): List of sequence block names that
                should be separated from regular blocks. Defaults to None.
        """
        super().__init__()
        self.feature_blocks = feature_blocks
        self.multihash_conf = self._parse_multihash(multihash_conf)
        self.sequence_blocks = sequence_blocks

    def _parse_multihash(self, mh_confs):
        out_confs = {}
        for fea_name, conf in mh_confs.items():
            prefix, buckets, combiner, _ = parse_multihash(conf)
            num = len(buckets)
            sub_names = [get_multihash_name(fea_name, prefix, i) for i in range(num)]
            out_confs[fea_name] = {"sub_name": sub_names, "combiner": combiner}
        return out_confs

    def _multihash_combiner(self, mh_inputs, combiner="concat"):
        if combiner == "concat":
            return torch.cat(mh_inputs, dim=-1)
        elif combiner == "mean":
            return torch.mean(torch.stack(mh_inputs, dim=0), dim=0)
        elif combiner == "sum":
            return torch.sum(torch.stack(mh_inputs, dim=0), dim=0)
        else:
            raise NotImplementedError(
                f"Multihash combiner only support combiner yet, got: {combiner}"
            )

    def combine_compress_emb(self, samples):
        """Combine multi-hash embeddings into single feature representations.

        This method processes samples to combine multi-hash sub-embeddings
        into their corresponding main feature embeddings using the configured
        combiner strategies.

        Args:
            samples (dict): Dictionary mapping feature names to their embedding
                tensors, including multi-hash sub-features.

        Returns:
            dict: Updated samples dictionary with multi-hash features combined
                and sub-features removed.
        """
        if self.multihash_conf is None:
            return samples
        for mh_fea, mh_conf in self.multihash_conf.items():
            mh_embs = []
            for sub_name in mh_conf["sub_name"]:
                mh_embs.append(samples.pop(sub_name))
            samples[mh_fea] = self._multihash_combiner(mh_embs, mh_conf["combiner"])
        return samples

    def block_concat(self, samples):
        """Concatenate features within each block to create block representations.

        This method groups features according to the feature_blocks configuration
        and concatenates them along the last dimension to create dense block
        representations. It also handles data type conversion from double to float.

        Args:
            samples (dict): Dictionary mapping feature names to their embedding
                tensors.

        Returns:
            dict: Dictionary mapping block names to their concatenated feature
                tensors.
        """
        block_results = {}
        for block_name, feas in self.feature_blocks.items():
            block_results[block_name] = []
            for fea in feas:
                if (
                    isinstance(samples[fea], torch.Tensor)
                    and samples[fea].dtype == torch.double
                ):
                    samples[fea] = samples[fea].float()
                value = samples[fea]
                block_results[block_name].append(value)
            block_results[block_name] = torch.cat(block_results[block_name], dim=-1)
        return block_results

    def split_sequence_block(self, samples):
        """Split sequence blocks from regular blocks for specialized processing.

        This method separates sequence blocks from regular blocks, allowing
        them to be processed differently in downstream model components.

        Args:
            samples (dict): Dictionary mapping block names to their concatenated
                feature tensors.

        Returns:
            tuple or dict: If sequence_blocks is configured, returns a tuple of
                (regular_samples, sequence_samples). Otherwise, returns the
                original samples dictionary.
        """
        if self.sequence_blocks is None:
            return samples
        seq_samples = {}
        for seq_block_name in self.sequence_blocks:
            seq_samples[seq_block_name] = samples.pop(seq_block_name)
        return samples, seq_samples

    def forward(self, samples):
        """Process samples through the complete block building pipeline.

        This method orchestrates the complete block building process: combining
        multi-hash embeddings, concatenating features within blocks, and
        separating sequence blocks.

        Args:
            samples (dict): Dictionary mapping feature names to their embedding
                tensors.

        Returns:
            tuple or dict: Processed samples with features organized into blocks.
                If sequence blocks are configured, returns (regular_blocks,
                sequence_blocks), otherwise returns block dictionary.
        """
        samples = self.combine_compress_emb(samples)
        samples = self.block_concat(samples)
        samples = self.split_sequence_block(samples)
        return samples


class RecISModel(nn.Module):
    """Complete RecIS model integrating feature processing and embedding engines.

    The RecISModel class provides a complete end-to-end model for the RecIS
    (Recommendation Intelligence System) framework. It integrates feature
    processing, embedding lookup, and block organization into a unified
    model architecture.

    Key Features:
        - Integrated feature processing pipeline with FeatureEngine
        - Embedding lookup and management with EmbeddingEngine
        - Flexible block organization with configurable BlockBuilder
        - Support for sequence features and multi-hash embeddings
        - Automatic separation of labels and sample IDs

    Attributes:
        feature_engine (FeatureEngine): Engine for processing raw features.
        embedding_engine (EmbeddingEngine): Engine for embedding lookup operations.
        block_builder (BlockBuilder): Builder for organizing features into blocks.
        sample_ids (list): List of sample ID field names.
        labels (list): List of label field names.

    Example:
        .. code-block:: python

            from recis.fg.feature_generator import build_fg

            # Build feature generator
            fg = build_fg(
                fg_conf_path="features.json", mc_conf_path="model_config.json"
            )

            # Create model from feature generator
            model = RecISModel.from_fg(fg, split_seq=True)

            # Forward pass
            block_features, sample_ids, labels = model(input_samples)
    """

    def __init__(
        self,
        feature_confs,
        emb_confs,
        feature_blocks,
        sample_ids,
        labels,
        sequence_blocks=None,
        multihash_conf=None,
        block_builder_class=BlockBuilder,
    ):
        """Initialize the RecIS Model.

        Args:
            feature_confs (list): List of feature configuration objects for
                the FeatureEngine.
            emb_confs (dict): Dictionary of embedding configurations for the
                EmbeddingEngine.
            feature_blocks (dict): Dictionary mapping block names to feature
                lists for the BlockBuilder.
            sample_ids (list): List of sample ID field names to extract.
            labels (list): List of label field names to extract.
            sequence_blocks (list, optional): List of sequence block names.
                Defaults to None.
            multihash_conf (dict, optional): Multi-hash configuration dictionary.
                Defaults to None.
            block_builder_class (type, optional): Class to use for block building.
                Defaults to BlockBuilder.
        """
        super().__init__()
        self.feature_engine = FeatureEngine(feature_list=feature_confs)
        self.embedding_engine = EmbeddingEngine(emb_confs)
        self.block_builder = block_builder_class(
            feature_blocks,
            multihash_conf=multihash_conf,
            sequence_blocks=sequence_blocks,
        )
        self.sample_ids = sample_ids
        self.labels = labels

    @staticmethod
    def from_fg(fg, split_seq=False, block_builder_class=BlockBuilder):
        """Create a RecISModel instance from a feature generator.

        This factory method provides a convenient way to create a RecISModel
        from a configured feature generator (FG) object, automatically
        extracting all necessary configurations.

        Args:
            fg: Feature generator object containing all model configurations.
            split_seq (bool, optional): Whether to split sequence blocks from
                regular blocks. Defaults to False.
            block_builder_class (type, optional): Class to use for block building.
                Defaults to BlockBuilder.

        Returns:
            RecISModel: Configured RecISModel instance ready for training or
                inference.

        Example:
            .. code-block:: python

                # Create model with sequence splitting
                model = RecISModel.from_fg(fg, split_seq=True)

                # Create model with custom block builder
                model = RecISModel.from_fg(
                    fg, split_seq=True, block_builder_class=CustomBlockBuilder
                )
        """
        return RecISModel(
            fg.get_feature_confs(),
            fg.get_emb_confs(),
            fg.feature_blocks,
            fg.sample_ids,
            fg.labels,
            sequence_blocks=fg.seq_block_names if split_seq else None,
            multihash_conf=fg.multihash_conf,
            block_builder_class=block_builder_class,
        )

    def forward(self, samples: dict):
        """Process input samples through the complete RecIS pipeline.

        This method orchestrates the complete model forward pass: extracting
        labels and sample IDs, processing features through the feature engine,
        looking up embeddings, and organizing features into blocks.

        Args:
            samples (dict): Dictionary containing raw input samples with
                features, labels, and sample IDs.

        Returns:
            tuple: A tuple containing:
                - block_features: Processed feature blocks ready for downstream
                  model components
                - sample_ids (dict): Extracted sample ID values
                - labels (dict): Extracted label values

        Example:
            .. code-block:: python

                input_samples = {
                    "user_id": user_tensor,
                    "item_id": item_tensor,
                    "click": label_tensor,
                    "sample_id": id_tensor,
                }

                block_features, sample_ids, labels = model(input_samples)
        """
        sample_ids = {}
        labels = {}
        for label in self.labels:
            labels[label] = samples.pop(label)
        for sample_id in self.sample_ids:
            sample_ids[sample_id] = samples.pop(sample_id)
        samples = self.feature_engine(samples)
        samples = self.embedding_engine(samples)
        block_features = self.block_builder(samples)
        return block_features, sample_ids, labels
