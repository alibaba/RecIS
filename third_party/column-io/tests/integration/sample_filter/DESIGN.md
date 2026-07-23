# `sample_filter` 行级黑名单过滤设计文档

> append-only;每次迭代新增一个 `## V<N>` section,不修改既有内容。
> 关联代码:
> - `cc/column-io/dataset/map_dataset_sample_filter.{h,cc}`
> - `column_io/dataset/dataset.py` (class `MapDatasetSampleFilter` + registry `"sample_filter"`)
> - `cc/column-io/py_interface/interface.cc` (pybind 绑定 `_MapDataSetSampleFilter`)
> 关联测试: `tests/integration/sample_filter/packer_dataset_sample_filter_test.py`

---

## V1 (2026-05-30)

### 1. 背景与需求

需求源自 DingTalk 文档 [基于sample\_id解析的样本过滤](https://alidocs.dingtalk.com/i/nodes/YMyQA2dXW7gYo6Mzc5dYqM0oWzlwrZgb)
(作者: 震熊, 2026-01)。在 lake/odps 训练数据流里需要根据 `sample_id` 字符
串字段做行级黑名单过滤。

`sample_id` 由生产侧约定为如下格式:
```
"<prefix>\x01<k1>:<v1>,<k2>:<v2>,...,<kN>:<vN>"
```
示例:
```
"DIANTAO\x01pvid:7565c9216c7e170069fcb4282a4fb242,entityId:76062744470,productType:H,timestamp:1778168872,pid:431451_1007,subProductType:200^22001^1^11,nickname:许是大宝吧,position:null,pctr:0.022397,old_pctr:0.026613"
```

用户传入 `filter_dict`(`dict[str, list[str]]`):
```python
filter_dict = {
    "pid": ["431451_1007", "mm_1296839164_12431047901_13412341"],
    "subProductType": ["200^22001^1^11"],
}
```
**语义**:解析 sample_id 后,若**任一** `filter_dict` 的 key 的解析值 in 该
key 对应的 list,则丢弃此样本(denylist + OR-across-keys)。

### 2. 关键设计决策

#### 2.1 复用 Packer 的 `_sample_group_id` drop 路径,不自己重写列存

column 模式下 batch 是 CSR-ragged + indicator 的复合结构;**单独 drop 一行**
需要重写每个 ragged 列的 values/splits 和每个 `_indicator_*` 列。这部分逻辑
Packer 已经在 `do_classify=true` 路径下实现过(`packer.cc:444-558`
`GetNextForGroup`):
- `packer.cc:459-461` 直接 `continue` 跳过 `group_id < 0` 的行
- Python 端 `dataset.py:670-672` 当上游 schema 出现 `_sample_group_id`
  自动开启 `do_classify=True`

复用方案:本 dataset 只 inject 一个 `_sample_group_id` int64 列(`-1=drop,
0=keep`),实际的行 drop + ragged/indicator 重写交给下游 `.pack()`。代价是
**调用方必须 `.pack()`,否则过滤无效**——这与 K2RankCanXi 是同一套设计。

#### 2.2 实现层选择:C++ FilterDataset(而非 Python wrapper)

理由:
- sample_id 解析是 batch-level 热路径(每行都要 split + lookup)。Python 解
  析受 GIL 影响,而 C++ 端 absl::StrSplit + unordered_set lookup 都是
  O(token_count) 无锁的。
- Pybind 边界已经成熟,新增一个 map dataset 的接入成本(~3 处 wiring)
  低于实现一个高性能 Python row filter。

#### 2.3 sample_id 列名硬编码 `"sample_id"`

与现有 `map_dataset_k2_rank.cc:68` 的约定一致。生产侧字段名稳定,无需暴
露 column 名参数膨胀 API。

#### 2.4 解析鲁棒性:`MaxSplits(':', 1)` 而非 `StrSplit(':')`

K2 baseline `map_dataset_k2_rank.cc:165` 用 `StrSplit(token, ':')` +
`keys.size() != 2` 判断,会**静默丢弃**值本身含 `:` 的字段(例如嵌套
timestamp)。

对于黑名单这是**false-negative**(漏掉一次 drop),是危险方向(因为本应过滤
的脏数据进入了训练集)。V1 改用 `absl::MaxSplits(':', 1)` 保留完整 value。

详见 `cc/column-io/dataset/map_dataset_sample_filter.cc::ClassifySample` 注
释。

### 3. API Contract

```python
from column_io.dataset import dataset as dataset_io

ds = dataset_io.Dataset.from_lake_source(paths, ...)
ds = ds.map(name="sample_filter", kargs={
    "filter_dict": {
        "pid": ["431451_1007", "mm_1296839164_12431047901_13412341"],
        "subProductType": ["200^22001^1^11"],
    }
})
ds = ds.pack(batch_size, drop_remainder=True, ...)  # 必须 .pack()
```

- input.schema 必须含 `"sample_id"` 列,否则 Python 层抛 `ValueError`
- 输出 schema 多一个 `_sample_group_id` 列(int64, shape 与 sample_id 同)
- 空 filter_dict → no-op,Python 侧 logger.warning,C++ 侧 fast-path 全填 0

### 4. 生产者格式契约(producer-side, V1 假设)

- sample_id **必须**以 `\x01` 分隔 prefix 与 KV 区
- KV 区用 `,` 分隔条目,每个条目用第一个 `:` 分隔 key 与 value
  - value 内可以含 `:`(V1 已支持,见 §2.4)
  - value 内**不能**含 `,`(V1 不支持转义)
- 格式异常(无 `\x01` / 无 `:` 等)→ 该行默认**保留**(denylist 安全侧)

如生产侧已经/将要使用更复杂的 escaping,需在 V2 升级 parser。

### 5. 边界条件矩阵(V1 行为)

| 场景 | V1 行为 | 检查位置 |
|------|--------|---------|
| `filter_dict` 为空 | no-op,全 0 | C++ ClassifyBatch fast-path + Python warning |
| `sample_id` 列缺失 | ValueError fail-fast | Python `__init__` |
| 整批被 filter 殆尽 | safe,Packer 拉下一批 | Packer 原生 |
| filter_dict key 不在 sample_id 中 | 不参与判定 → 保留 | C++ ClassifySample 自然行为 |
| 多 key | OR(任一命中即 drop)+ 短路 | C++ ClassifySample 循环内 return |
| 用户忘记 `.pack()` | 过滤静默无效;`_sample_group_id` 列穿透下游 | Python warning + 本文档强调 |
| value 含 `:` | 保留完整 value(`MaxSplits(':', 1)`) | C++ ClassifySample |
| value 含 `,` | 不支持(生产者契约) | DESIGN.md §4 |
| sample_id 无 `\x01` | 视为格式异常,默认保留 | C++ ClassifySample |

### 6. Checkpoint(Save/Restore)

- `filter_dict_` 是构造期 const 数据,**故意不进 checkpoint**——上游代码重启
  时会同样的参数重建 Dataset
- 仅 `input_impl_` 走 SaveInput/RestoreInput 标准协议
- 与 `map_dataset_k2_rank_canxi.cc:79-94` 同构

### 7. 性能特征

- 构造期一次性把 `vector<string>` 转 `unordered_set<string>`,运行时 O(1)
  查找
- 每行解析:O(token_count) split + 短路。典型 sample_id 10 个 KV 条目左
  右,单行开销 ≈ 数百 ns
- 无并行(单线程内顺序解析);ClassifySample 无锁(filter_dict_ 在 ctor 之
  后只读)。如未来需要并行,可 wrap 在 ParallelDataset 之内

### 8. V1 暂未覆盖 / 待 V2 起讨论

- Row mode(`ODPS_DATASET_ROW_MODE=1`)下不工作。row mode 走
  `_OdpsOpenStorageRowDataset` 路径,不经过 Packer。如有 row 端过滤需
  求,需在 row reader 内做或者在 Python 端做 list 过滤。
- 不支持 AND-across-keys 语义。如需,在 C++ ClassifySample 内改为收集匹
  配数 == filter_dict_.size() 才 drop。
- 不支持 NOT 语义(白名单)。用户场景已确认为黑名单。
- value 不支持 escape(`\,` 等);如生产侧引入,需升级 parser。
- filter_dict 不可热更新(构造期固定)。

### 9. 测试覆盖

集成测试 `packer_dataset_sample_filter_test.py`(同目录):
1. **test_filter_drops_denylisted_rows**:从 baseline 拉真实样本,挑 N 个
   pid 加入 denylist,验证不再出现
2. **test_empty_filter_dict_is_noop**:空 filter_dict 时输出 == baseline
3. **test_multi_key_or_semantics**:同时给 `pid` + `subProductType`,确认
   任一命中均 drop
4. **test_missing_sample_id_raises**:构造一个无 sample_id 的源 → 期望
   ValueError

C++ unit test:V1 跳过(现有 gtest fixture 不适配,工作量 > 收益)。
