"""GpuShard 数据结构与构建逻辑（统一表示，每特征一个 entry）。

每个 rank 持有 1/world_size 的负样本数据，存储在 GPU 上。数据按 unit_id 排序。
所有特征（dense 和 ragged）统一为 entry，通过 flat_bytes 抽象 dtype 和列数差异。
"""

import torch

from recis.ragged.tensor import RaggedTensor


class GpuShard:
    """GPU 分片数据，统一表示。

    每个特征（dense 或 ragged）对应一个 value entry。
    有权重的 ragged 特征额外有一个 weight entry。
    所有 entry 通过 flat_bytes 统一处理，无 dtype 分支。

    Entry 布局:
        [0..num_features-1]: 特征 values（dense + ragged 混合，按 build 顺序）
        [num_features..num_entries-1]: ragged weights（仅有权重的特征）

    Attributes:
        sorted_ids: [num_items] int64, 升序。
        value_vec: list[Tensor], 所有 entry 的 value 张量。
        feature_names: list[str], 特征名（仅 values，不含 weights）。
        feature_dtypes: list[torch.dtype], 特征 value 的 dtype。
        is_ragged: list[bool], 是否为 ragged 特征。
        feature_shapes: list[tuple], dense 特征的 shape 后缀（ragged 为 ()）。
        weight_entry_idx: list[Optional[int]], 每个特征的 weight entry 索引（无权重为 None）。
        weight_dtypes: list[Optional[torch.dtype]], 每个特征的 weight dtype。
        num_features: int, 特征数（values entry 数）。
        num_entries: int, 总 entry 数（values + weights）。
        num_ragged: int, ragged 特征数。
    """

    def __init__(
        self,
        sorted_ids: torch.Tensor,
        value_vec: list,
        offsets_vec: list,
        flat_bytes_vec_list: list,
        feature_names: list,
        feature_dtypes: list,
        is_ragged: list,
        feature_shapes: list,
        weight_entry_idx: list,
        weight_dtypes: list,
        num_features: int,
        num_ragged: int,
        num_items: int,
        shard_id: int,
    ):
        self.sorted_ids = sorted_ids
        self.value_vec = value_vec
        self.offsets_vec = offsets_vec
        self.flat_bytes_vec_list = flat_bytes_vec_list
        self.feature_names = feature_names
        self.feature_dtypes = feature_dtypes
        self.is_ragged = is_ragged
        self.feature_shapes = feature_shapes
        self.weight_entry_idx = weight_entry_idx
        self.weight_dtypes = weight_dtypes
        self.num_features = num_features
        self.num_ragged = num_ragged
        self.num_entries = len(value_vec)
        self.num_items = num_items
        self.shard_id = shard_id

        self._dense_offset = offsets_vec[0] if offsets_vec else None

        self.flat_offsets = None
        self.offset_indices = None
        self.flat_bytes_vec = None
        self.value_ptrs = None

    @property
    def has_weights(self):
        return any(w is not None for w in self.weight_entry_idx)

    @property
    def num_dense(self):
        return self.num_features - self.num_ragged

    def __repr__(self):
        return (
            f"GpuShard(shard_id={self.shard_id}, num_items={self.num_items}, "
            f"num_features={self.num_features}, num_ragged={self.num_ragged}, "
            f"num_entries={self.num_entries}, has_weights={self.has_weights})"
        )

    def to_device(self, device: str) -> "GpuShard":
        """移到 GPU，构建 kernel 需要的 flat_offsets/offset_indices/value_ptrs。"""
        self.sorted_ids = self.sorted_ids.to(device)
        self.value_vec = [v.to(device) for v in self.value_vec]
        self._dense_offset = self._dense_offset.to(device)
        self.offsets_vec = [
            o.to(device) if o is not None else None for o in self.offsets_vec
        ]

        # 堆叠唯一 offsets（dense 共享一个，每个 ragged 各一个）
        unique_offsets = [self._dense_offset]
        ragged_offset_rows = {}
        ragged_count = 0
        for f in range(self.num_features):
            if self.is_ragged[f]:
                unique_offsets.append(self.offsets_vec[f])
                ragged_offset_rows[f] = ragged_count
                ragged_count += 1

        self.flat_offsets = torch.stack(
            unique_offsets, dim=0
        )  # [1+num_ragged, num_items+1]

        # offset_indices: 每个 entry → flat_offsets 的行索引
        offset_indices_list = []
        for f in range(self.num_features):
            if self.is_ragged[f]:
                offset_indices_list.append(1 + ragged_offset_rows[f])
            else:
                offset_indices_list.append(0)  # dense 共享 row 0

        for f in range(self.num_features):
            if self.weight_entry_idx[f] is not None:
                offset_indices_list.append(1 + ragged_offset_rows[f])

        self.offset_indices = torch.tensor(
            offset_indices_list, dtype=torch.int64, device=device
        )
        self.flat_bytes_vec = torch.tensor(
            self.flat_bytes_vec_list, dtype=torch.int64, device=device
        )

        ptrs_cpu = torch.tensor(
            [v.data_ptr() for v in self.value_vec], dtype=torch.int64
        )
        self.value_ptrs = ptrs_cpu.to(device)

        self.offsets_vec = None
        self._dense_offset = None
        return self

    @property
    def ragged_names(self):
        return [
            self.feature_names[f] for f in range(self.num_features) if self.is_ragged[f]
        ]


def _categorize_features(batch: dict) -> dict:
    """将特征分为 dense / multi_value_ragged。"""
    categories = {"dense": [], "multi_value_ragged": []}
    for name, val in sorted(batch.items()):
        if name.startswith("_bench"):
            continue
        if isinstance(val, torch.Tensor):
            categories["dense"].append((name, val))
        elif isinstance(val, RaggedTensor):
            categories["multi_value_ragged"].append((name, val))
        else:
            raise TypeError(f"Unsupported feature type for '{name}': {type(val)}")
    return categories


def _select_ragged_rows(ragged: RaggedTensor, indices: torch.Tensor) -> tuple:
    """从 RaggedTensor 中选取指定行并重排，返回 (values, offsets)。"""
    old_offsets = ragged.offsets()[0]
    begins = old_offsets[indices]
    ends = old_offsets[indices + 1]
    lengths = ends - begins

    new_offsets = torch.zeros(len(indices) + 1, dtype=torch.int64)
    new_offsets[1:] = lengths.cumsum(0)

    inner_offsets = (
        torch.arange(lengths.max().item())
        if lengths.max() > 0
        else torch.zeros(0, dtype=torch.int64)
    )
    gather_idx = begins.unsqueeze(1) + inner_offsets.unsqueeze(0)
    gather_mask = inner_offsets.unsqueeze(0) < lengths.unsqueeze(1)
    gather_idx = gather_idx[gather_mask]

    new_values = ragged.values()[gather_idx]
    return new_values, new_offsets


def build_cpu_shard(batch: dict, shard_id: int) -> GpuShard:
    """从 neg table 数据构建 CPU 分片。每特征一个 entry，不按 dtype 拼接。

    Args:
        batch: neg table 的特征字典。
        shard_id: 本分片的 ID。

    Returns:
        GpuShard 实例，全部数据在 CPU 上。
    """
    if "unit_id" not in batch:
        raise ValueError("batch must contain 'unit_id' feature for lookup")

    # 1. 提取 unit_id 并排序
    unit_id_ragged = batch["unit_id"]
    if isinstance(unit_id_ragged, RaggedTensor):
        unit_ids = unit_id_ragged.values()
    else:
        unit_ids = unit_id_ragged

    num_items = unit_ids.numel()
    sorted_order = unit_ids.argsort()
    sorted_ids = unit_ids[sorted_order]

    # 2. 分类特征
    categories = _categorize_features(batch)

    # 3. 构建统一表示
    dense_offset = torch.arange(num_items + 1, dtype=torch.int64)

    value_vec = []
    offsets_vec = []
    flat_bytes_vec_list = []
    feature_names = []
    feature_dtypes = []
    is_ragged = []
    feature_shapes = []
    weight_entry_idx = []
    weight_dtypes = []
    num_ragged = 0

    # Dense features: 每个特征一个 entry
    for name, val in categories["dense"]:
        sorted_val = val[sorted_order]
        shape_suffix = tuple(val.shape[1:])
        elem_size = val.element_size()

        # 确保 2D: [N] → [N, 1]，[N, K] 保持
        if sorted_val.dim() == 1:
            stored = sorted_val.unsqueeze(1)
            flat_bytes = elem_size
        else:
            stored = sorted_val
            flat_bytes = sorted_val.shape[1] * elem_size

        value_vec.append(stored)
        offsets_vec.append(dense_offset)
        flat_bytes_vec_list.append(flat_bytes)
        feature_names.append(name)
        feature_dtypes.append(val.dtype)
        is_ragged.append(False)
        feature_shapes.append(shape_suffix)
        weight_entry_idx.append(None)
        weight_dtypes.append(None)

    # Ragged features: 每个特征 values 一个 entry
    ragged_feature_indices = []
    for name, ragged in categories["multi_value_ragged"]:
        vals, offs = _select_ragged_rows(ragged, sorted_order)

        value_vec.append(vals)
        offsets_vec.append(offs)
        flat_bytes_vec_list.append(vals.element_size())
        feature_names.append(name)
        feature_dtypes.append(vals.dtype)
        is_ragged.append(True)
        feature_shapes.append(())
        weight_entry_idx.append(None)
        weight_dtypes.append(None)
        ragged_feature_indices.append(num_ragged)
        num_ragged += 1

    # Weight entries: 仅有权重的 ragged 特征，追加在所有 value entries 之后
    for idx, (name, ragged) in enumerate(categories["multi_value_ragged"]):
        if ragged.weight() is not None:
            w_vals, _ = _select_ragged_rows(
                type(ragged)(ragged.weight(), ragged.offsets()), sorted_order
            )
            f_idx = ragged_feature_indices[idx]
            corresponding_offset = offsets_vec[len(categories["dense"]) + f_idx]

            weight_entry = len(value_vec)
            value_vec.append(w_vals)
            offsets_vec.append(corresponding_offset)
            flat_bytes_vec_list.append(w_vals.element_size())
            weight_entry_idx[f_idx + len(categories["dense"])] = weight_entry
            weight_dtypes[f_idx + len(categories["dense"])] = ragged.weight().dtype

    num_features = len(feature_names)

    return GpuShard(
        sorted_ids=sorted_ids,
        value_vec=value_vec,
        offsets_vec=offsets_vec,
        flat_bytes_vec_list=flat_bytes_vec_list,
        feature_names=feature_names,
        feature_dtypes=feature_dtypes,
        is_ragged=is_ragged,
        feature_shapes=feature_shapes,
        weight_entry_idx=weight_entry_idx,
        weight_dtypes=weight_dtypes,
        num_features=num_features,
        num_ragged=num_ragged,
        num_items=num_items,
        shard_id=shard_id,
    )
