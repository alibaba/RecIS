from collections import defaultdict
from typing import Dict, List, Union

from torch import nn

from recis.features.feature import Feature
from recis.features.fused_op_impl import FusedOpFactory
from recis.features.op import _OP
from recis.monitor.monitor_reporter import (
    FEA_ENGINE_NAME,
    SPARSE_FWD_NAME,
    MonitorReporter,
)
from recis.utils.logger import Logger


logger = Logger(__name__)


class RunContext:
    """Execution context for managing data flow during feature processing.

    The RunContext manages input data, intermediate results, and output data
    during feature processing execution. It provides a centralized way to
    store and retrieve data at different stages of the pipeline.

    Attributes:
        _data_buffer (dict): Storage for intermediate and output data.
        _input_data (dict): Original input data provided to the engine.
        _remain_no_use_data (bool): Whether to keep unused input data in results.
    """

    def __init__(self, input_data, remain_no_use_data: bool = True):
        self._data_buffer = {}
        self._input_data = input_data
        self._remain_no_use_data = remain_no_use_data

    def get(self, key):
        if key not in self._data_buffer:
            return self._input_data
        return self._data_buffer[key]

    def set(self, key, value):
        self._data_buffer[key] = value

    @property
    def results(self):
        if self._remain_no_use_data:
            for k, v in self._input_data.items():
                if k not in self._data_buffer:
                    self._data_buffer[k] = v
        return self._data_buffer


class FuseExecutionGroup:
    """Execution group for batching fuseable operations together.

    This class groups operations of the same type that can be fused together
    for improved performance. It manages the fused operation instance and
    coordinates the execution of multiple similar operations in a single batch.

    Attributes:
        _feature_names (List[str]): Names of features in this execution group.
        _fused_op: The fused operation instance that processes all operations.
    """

    def __init__(self, op: _OP = None):
        self._feature_names = []
        self._fused_op = FusedOpFactory.get_fused_op(op)
        self._fused_op = self._fused_op(ops=[])

    def add_op(self, feature_name: str, op: _OP):
        self._feature_names.append(feature_name)
        self._fused_op.add_op(op)

    def run(self, data: RunContext):
        # fetch inputs
        inputs = [data.get(name) for name in self._feature_names]
        outputs = self._fused_op(inputs)
        for name, output in zip(self._feature_names, outputs):
            data.set(name, output)
        return data


class NonFuseExecutionGroup:
    """Execution group for operations that cannot be fused.

    This class handles operations that must be executed individually,
    either because they don't have fused implementations or because
    fusion would not provide performance benefits.

    Attributes:
        _ops (List[_OP]): List of operations to execute individually.
        _feature_names (List[str]): Names of features corresponding to operations.
    """

    def __init__(self):
        self._ops: List[_OP] = []
        self._feature_names = []

    def add_op(self, feature_name: str, op: _OP):
        self._ops.append(op)
        self._feature_names.append(feature_name)

    def run(self, data: RunContext):
        for op, feature_name in zip(self._ops, self._feature_names):
            output = op(data.get(feature_name))
            data.set(feature_name, output)
        return data


class ExecutionStep:
    """A single step in the feature processing pipeline.

    An execution step groups operations that can be executed in parallel,
    organizing them into fused and non-fused execution groups for optimal
    performance.

    Attributes:
        execution_groups (Dict): Dictionary mapping group signatures to
                               execution group instances.
        _fuseable_ops (List): List of operations that support fusion.
        _non_fuseable_tag (str): Tag for grouping non-fuseable operations.
    """

    _fuseable_ops = []
    _non_fuseable_tag = "_non_fuseable_tag"

    def __init__(self):
        self.execution_groups: Dict[
            str, Union[FuseExecutionGroup, NonFuseExecutionGroup]
        ] = {}

    def _add_fuseable_op(self, feature_name: str, op: _OP):
        signature = type(op).__name__
        if signature not in self.execution_groups:
            self.execution_groups[signature] = FuseExecutionGroup(op)
        self.execution_groups[signature].add_op(feature_name, op)

    def _add_non_fuseable_op(self, feature_name: str, op: _OP):
        if self._non_fuseable_tag not in self.execution_groups:
            self.execution_groups[self._non_fuseable_tag] = NonFuseExecutionGroup()
        self.execution_groups[self._non_fuseable_tag].add_op(feature_name, op)

    def add_op(self, feature_name: str, op: _OP):
        if self._fuseable(op):
            self._add_fuseable_op(feature_name, op)
        else:
            self._add_non_fuseable_op(feature_name, op)
        return self

    def _fuseable(self, op: _OP):
        return FusedOpFactory.contains(op)

    def run(self, data: RunContext):
        for execution_group in self.execution_groups.values():
            data = execution_group.run(data)
        return data


class FeatureEngine(nn.Module):
    """Main feature processing engine with automatic optimization.

    The FeatureEngine manages a collection of features and their processing
    pipelines. It automatically optimizes execution through operation fusion,
    deduplication of identical features, and efficient step-by-step processing.

    Key Features:
        - Automatic operation fusion for improved performance
        - Feature deduplication based on hash values
        - Step-by-step execution with dependency management
        - Support for both fused and individual operation execution

    Example:

    .. code-block:: python

        from recis.features.feature import Feature
        from recis.features.op import SelectField, Hash, Mod

        # simple feature
        user_feature = Feature("user_id").\\
            add_op(SelectField("user_id")).\\
            add_op(Mod(10000))

        # sequence feature
        seq_feature = Feature("seq_item_id").\\
            SequenceTruncate(seq_len=20,
                             truncate=True,
                             truncate_side="right",
                             check_length=False,
                             n_dims=3,
                             dtype=torch.int64).\\
            add_op(Mod(10000))

    """

    def __init__(self, feature_list: List[Union[Feature]]):
        """Initialize the feature engine with a list of features.

        The engine automatically deduplicates identical features based on their
        hash values and compiles the features into optimized execution steps.

        Args:
            feature_list (List[Feature]): List of features to process.
        """
        super().__init__()
        self._data_cache_map = defaultdict(dict)
        self._features = nn.ModuleDict()

        self._hash_cache_map = {}
        hash_to_features = defaultdict(list)
        for feature in feature_list:
            feature_hash = feature.get_hash()
            hash_to_features[feature_hash].append(feature)

        for feature_hash, features in hash_to_features.items():
            primary_feature = features[0]
            self._features[primary_feature.name] = primary_feature
            for duplicate_feature in features[1:]:
                self._hash_cache_map[duplicate_feature.name] = primary_feature.name

        self.execution_steps = self._compile(list(self._features.values()))

    def _get_feature_by_hash(self, feature_hash: int) -> str:
        return self._feature_hash_cache.get(feature_hash)

    def _check_compiled(self, feature: Feature):
        if feature.compiled:
            raise ValueError(f"feature {feature.name} has been compiled")
        else:
            feature.compiled_(True)

    def _add_to_step(
        self, feature: Feature, step: ExecutionStep, step_idx: int
    ) -> bool:
        if len(feature.ops) <= step_idx:
            return True
        step.add_op(feature.name, feature.ops[step_idx])
        return False

    def _compile(self, feature_list: List[Feature]):
        for feature in feature_list:
            self._check_compiled(feature)
            if feature.name not in self._data_cache_map:
                self._data_cache_map[feature.name] = None
            else:
                raise ValueError(f"Duplicate feature name: {feature.name}")

        execution_steps = []
        step_idx = 0
        while True:
            exec_step = ExecutionStep()
            all_done = True
            for feature in feature_list:
                all_done = self._add_to_step(feature, exec_step, step_idx) and all_done
            if all_done:
                break
            step_idx += 1
            execution_steps.append(exec_step)
        return execution_steps

    @MonitorReporter.report_time_wrapper(
        SPARSE_FWD_NAME, {"recis_emb_phase": FEA_ENGINE_NAME}
    )
    def forward(self, data: Dict, remain_no_use_data: bool = True) -> Dict:
        """Process input data through all compiled feature pipelines.

        Executes all features through their compiled execution steps, applying
        automatic operation fusion and managing data flow between steps.

        Args:
            data (Dict): Input data dictionary with feature names as keys.
            remain_no_use_data (bool): Whether to include unused input data
                                     in the output. Defaults to True.

        Returns:
            Dict: Processed output data with feature results and optionally
                 unused input data.

        Raises:
            AssertionError: If input data is not a dictionary.
        """
        assert isinstance(data, dict), "data must be a dict"
        run_context = RunContext(data, remain_no_use_data)
        for step in self.execution_steps:
            run_context = step.run(run_context)

        results = run_context.results
        for feature_name in list(results.keys()):
            if isinstance(results[feature_name], dict):
                dict_value = results.pop(feature_name)
                for sub_name, sub_value in dict_value.items():
                    results[f"{feature_name}_{sub_name}"] = sub_value
        for feature_name, primary_feature in self._hash_cache_map.items():
            if isinstance(results[primary_feature], dict):
                for sub_name, sub_re in results[primary_feature].items():
                    results[f"{feature_name}_{sub_name}"] = sub_re
            else:
                results[feature_name] = results[primary_feature]
        return results
