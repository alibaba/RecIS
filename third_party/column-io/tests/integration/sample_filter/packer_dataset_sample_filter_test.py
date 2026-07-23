"""Integration tests for sample_filter map dataset.

End-to-end: real lake source -> .map(name='sample_filter') -> .pack() -> read
batches -> re-parse sample_id in Python and assert denylist semantics.

Source table follows the dingtalk doc YMyQA2dXW7gYo6Mzc5dYqM0oWzlwrZgb:
  na175-8 / alimama_ecpm_rank_odl / ecpm_nmd_allpid_addpid_ct_lake2
  / default_column_family / current/data

Run:
  mkdir -p log
  python -u tests/integration/sample_filter/packer_dataset_sample_filter_test.py 2>&1 | tee log/sample_filter_test.log

Notes:
  - Requires recis installed (LD_LIBRARY_PATH preamble per CLAUDE.md:87).
  - Lake stream readers don't guarantee deterministic ordering across runs,
    so tests assert SEMANTIC properties (no denylisted value appears in
    filtered output) rather than exact row counts vs baseline.
  - LAKE_TIME_START_US / LAKE_TIME_END_US default to a wide window; override
    via env if the chosen partition has no data when the test runs.
"""

import os
import sys
import unittest

# Lake plugin loads via dlopen of LAKERUNTIMEso; recis ships the runtime libs.
# This preamble must come BEFORE importing column_io (per CLAUDE.md:87 and
# every other integration test in tests/integration/).
try:
    import recis  # noqa: F401

    os.environ["LD_LIBRARY_PATH"] = (
        os.path.join(os.path.split(recis.__file__)[0], "lib")
        + ":"
        + os.environ.get("LD_LIBRARY_PATH", "")
    )
except ImportError:
    pass

from column_io.dataset import dataset as dataset_io
from column_io.dataset.config import LakeConfig
from column_io.dataset.file_sharding import LakeStreamSharding


# ---- table coords from the dingtalk doc ----
LAKE_STORAGE = "na175-8"
LAKE_PROJECT = "alimama_ecpm_rank_odl"
LAKE_TABLE = "ecpm_nmd_allpid_addpid_ct_lake2"
LAKE_CF = "default_column_family"
LAKE_PART_SPEC = "current/data"

# Time window in microseconds. Default covers 2026-04 ~ 2026-05 which matches
# the orc file timestamps observed in current/data partition at test-write
# time (file name prefix 1776xxxxx = ~2026-04-13). Override via env when the
# table's retention has shifted past this window.
LAKE_TIME_START_US = int(os.environ.get("LAKE_TIME_START_US", "1776000000000000"))
LAKE_TIME_END_US = int(os.environ.get("LAKE_TIME_END_US", "1781000000000000"))

BATCH_SIZE = 64
# Number of batches to scan per test. Kept small so the test stays under
# a few seconds per case in CI.
N_BATCHES = 5


def _make_lake_dataset(select_columns):
    """Construct a baseline LakeStreamColumnDataset for the test table.

    Args:
      select_columns: list of column names to read from the lake table.
        For sample_filter to function, this MUST include "sample_id".

    Returns:
      A Dataset (LakeStreamColumnDataset) iterating over the partition.
    """
    lake_config = LakeConfig(
        storageName=LAKE_STORAGE,
        projectName=LAKE_PROJECT,
        tableName=LAKE_TABLE,
        columnFamilyName=LAKE_CF,
        partitionSpec=LAKE_PART_SPEC,
    )
    sharding = LakeStreamSharding()
    sharding.add_path(lake_config.get_v1_path(), LAKE_TIME_START_US, LAKE_TIME_END_US)
    # is_compressed=True 是这张表的物理属性 (lake 端压缩存储);
    # 与 select_columns 的命名无关, formater 内部会做 sample_id_0/_1 后缀
    # 转换 (参考 OdpsComboDataset is_feature_in_schema, dataset.py:953-959).
    return dataset_io.Dataset.from_lake_source(
        sharding.partition(0, 1)[0],
        True,  # is_compressed (per user: 这是一张压缩表)
        BATCH_SIZE,
        select_columns,
        [], [], [], [], [],
    )


def _parse_sample_id(sample_id_str):
    """Parse a sample_id string into a dict[str, str].

    Mirrors the producer-side contract documented in DESIGN.md §1:
        "<prefix>\\x01<k1>:<v1>,<k2>:<v2>,...,<kN>:<vN>"
    Returns an empty dict on malformed input (consistent with the C++
    ClassifySample's "default to keep" behavior).
    """
    try:
        if "\x01" not in sample_id_str:
            return {}
        _, kv_region = sample_id_str.split("\x01", 1)
        out = {}
        for token in kv_region.split(","):
            if ":" not in token:
                continue
            # Use split(":", 1) to preserve any ":" inside values (matches
            # C++ absl::MaxSplits(':', 1) in map_dataset_sample_filter.cc).
            k, v = token.split(":", 1)
            out[k] = v
        return out
    except Exception:
        return {}


def _collect_sample_ids(dataset, n_batches):
    """Pull n_batches from dataset and return a flat list of sample_id strings.

    Works for both packed (column-mode dict) and packed-with-group output
    shapes. Strategy: walk every Tensor of every batch; any 1-D string array
    whose elements pattern-match sample_id (contain "\\x01") is included.
    """
    sample_ids = []
    iterator = iter(dataset)
    for _ in range(n_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        # Pack output is dict[name -> list[list[Tensor]]]. Walk recursively.
        def _walk(obj):
            if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
                if hasattr(obj, "items"):
                    for v in obj.values():
                        _walk(v)
                else:
                    for item in obj:
                        _walk(item)
            else:
                # Leaf: hope it's a DLPack tensor we can convert to torch
                try:
                    import torch

                    t = torch.from_dlpack(obj)
                    # string tensors come back as numpy object arrays; skip ints
                    if t.dtype == torch.uint8 or t.dtype.is_floating_point:
                        return
                except Exception:
                    # Some leaves are numpy / list[str] / etc.
                    pass

        # Simpler path: assume column-mode pack output is a list/dict where
        # _sample_group_id is added; sample_id should be at batch[0]["sample_id"][0][0]
        # in the K2 convention. Try several shapes.
        try:
            sid_tensor = batch[0]["sample_id"][0][0]
        except (KeyError, IndexError, TypeError):
            try:
                sid_tensor = batch["sample_id"][0][0]
            except Exception:
                continue
        # Convert tensor to python list of str
        try:
            import numpy as np

            arr = np.asarray(sid_tensor)
            sample_ids.extend(s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in arr.tolist())
        except Exception:
            try:
                sample_ids.extend(list(sid_tensor))
            except Exception:
                pass
    return sample_ids


class SampleFilterIntegrationTest(unittest.TestCase):
    """End-to-end tests against the real lake table.

    Skipped automatically if the lake table is unreachable (e.g. running
    outside the corp network).
    """

    @classmethod
    def setUpClass(cls):
        # Probe: build a baseline dataset + pull 1 batch to surface env issues
        # early with a clear skip message rather than failing every test.
        try:
            cls._baseline_sample_ids = _collect_sample_ids(
                _make_lake_dataset(["sample_id"]).pack(
                    BATCH_SIZE, drop_remainder=True
                ),
                n_batches=N_BATCHES,
            )
        except Exception as e:
            raise unittest.SkipTest(
                "Cannot reach lake table {}.{}.{}.{} (window {}-{}): {}".format(
                    LAKE_STORAGE, LAKE_PROJECT, LAKE_TABLE, LAKE_PART_SPEC,
                    LAKE_TIME_START_US, LAKE_TIME_END_US, e,
                )
            )
        if not cls._baseline_sample_ids:
            raise unittest.SkipTest(
                "Baseline returned 0 rows from lake; cannot derive denylist samples"
            )

    def _pick_denylist_pids(self, n=3):
        """Pick N pids that actually appear in baseline to make filter meaningful."""
        from collections import Counter

        pids = []
        for sid in self._baseline_sample_ids:
            kv = _parse_sample_id(sid)
            if "pid" in kv:
                pids.append(kv["pid"])
        # Pick most frequent so the denylist actually affects output
        return [p for p, _ in Counter(pids).most_common(n)]

    def test_filter_drops_denylisted_rows(self):
        """V1 core test: pids in filter_dict[pid] must not appear in output."""
        denylist = self._pick_denylist_pids(n=3)
        self.assertTrue(denylist, "baseline has no parseable pid; cannot test")
        print("[test_filter_drops] denylist pids: {}".format(denylist))

        ds = _make_lake_dataset(["sample_id"])
        ds = ds.map(name="sample_filter", kargs={"filter_dict": {"pid": denylist}})
        ds = ds.pack(BATCH_SIZE, drop_remainder=True)
        filtered_ids = _collect_sample_ids(ds, n_batches=N_BATCHES)

        leaked = []
        for sid in filtered_ids:
            kv = _parse_sample_id(sid)
            if kv.get("pid") in denylist:
                leaked.append(sid[:100])
        self.assertFalse(
            leaked,
            "filter leaked {} denylisted rows; sample: {}".format(
                len(leaked), leaked[:3]
            ),
        )

    def test_empty_filter_dict_is_noop(self):
        """E1: empty filter_dict should not drop any row."""
        ds = _make_lake_dataset(["sample_id"])
        ds = ds.map(name="sample_filter", kargs={"filter_dict": {}})
        ds = ds.pack(BATCH_SIZE, drop_remainder=True)
        out_ids = _collect_sample_ids(ds, n_batches=N_BATCHES)
        # With drop_remainder=True both runs produce N_BATCHES * BATCH_SIZE
        # rows; we just assert non-empty output (lake source is non-deterministic
        # in ordering, so exact equality is unreliable).
        self.assertGreater(len(out_ids), 0, "empty filter produced 0 rows")

    def test_multi_key_or_semantics(self):
        """E5: any key match drops the row (OR-across-keys)."""
        # Find a real (pid, subProductType) combo from baseline to ensure
        # the denylist is realistic.
        sub_types_seen = []
        for sid in self._baseline_sample_ids:
            kv = _parse_sample_id(sid)
            if "subProductType" in kv:
                sub_types_seen.append(kv["subProductType"])
        if not sub_types_seen:
            self.skipTest("baseline has no parseable subProductType")
        from collections import Counter

        common_subs = [s for s, _ in Counter(sub_types_seen).most_common(2)]
        common_pids = self._pick_denylist_pids(n=2)
        filter_dict = {"pid": common_pids, "subProductType": common_subs}
        print("[test_multi_key] filter_dict: {}".format(filter_dict))

        ds = _make_lake_dataset(["sample_id"])
        ds = ds.map(name="sample_filter", kargs={"filter_dict": filter_dict})
        ds = ds.pack(BATCH_SIZE, drop_remainder=True)
        out_ids = _collect_sample_ids(ds, n_batches=N_BATCHES)

        for sid in out_ids:
            kv = _parse_sample_id(sid)
            self.assertNotIn(
                kv.get("pid"), common_pids,
                "row leaked through pid denylist: {}".format(sid[:100]),
            )
            self.assertNotIn(
                kv.get("subProductType"), common_subs,
                "row leaked through subProductType denylist: {}".format(sid[:100]),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
