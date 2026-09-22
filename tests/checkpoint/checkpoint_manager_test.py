"""recis.framework.checkpoint_manager 子方法的单元测试。

覆盖: filter step 兼容、rank-0 状态保存、checkpoint 索引更新、
旧 checkpoint 淘汰及本地路径解析。

运行:
    python -m pytest tests/checkpoint/checkpoint_manager_test.py -v
"""

import hashlib
import json
import os
import sys
from collections import OrderedDict


# 仅在 recis.so 不存在时设置 BUILD_DOCUMENT（本地开发环境）。
# CI 环境中 .so 必须正常加载，以保证 torch.classes 注册成功。
_recis_so = os.path.join(
    os.path.dirname(__file__), "..", "..", "recis", "lib", "recis.so"
)
if not os.path.exists(os.path.abspath(_recis_so)):
    os.environ["BUILD_DOCUMENT"] = "1"

import shutil  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402


def _mock_if_missing(module_name, mock_obj=None):
    try:
        __import__(module_name)
    except (ImportError, ModuleNotFoundError):
        sys.modules[module_name] = mock_obj or MagicMock()


_mock_if_missing("recis.info", MagicMock(is_internal_enabled=lambda: False))
_mock_if_missing("column_io")
_mock_if_missing("column_io.dataset")
_mock_if_missing("column_io.dataset.log_util")

import torch  # noqa: E402

from recis.framework.checkpoint_compat import (  # noqa: E402
    collect_filter_global_step_names,
    use_child_filter_global_step_names,
)
from recis.framework.checkpoint_manager import ExtraFields, Saver  # noqa: E402
from recis.framework.model_bank import (  # noqa: E402
    MBC,
    DensePatternMatcher,
    ModelBankEntry,
    ModelBankParser,
    parse_dense_oname,
)


class _FakeEmbeddingOption:
    def __init__(self, children, use_sparse_grad_group=False):
        self.children = children
        self.use_sparse_grad_group = use_sparse_grad_group

    def coalesced_info(self):
        grad_reduce_by = "group_sum" if self.use_sparse_grad_group else "worker"
        group_size = 32 if self.use_sparse_grad_group else None
        group_reduce_by = "worker_sum" if self.use_sparse_grad_group else None
        return (
            '{"dim": 8, "dtype": "torch.float32", "device": "cpu", '
            '"initializer": "ConstantInitializer_0", '
            f'"grad_reduce_by": "{grad_reduce_by}", '
            f'"sparse_grad_group_size": {json.dumps(group_size)}, '
            '"sparse_grad_group_reduce_by": '
            f"{json.dumps(group_reduce_by)}, "
            '"filter_hook": "GlobalStepFilter"}'
        )


class _FakeFilterHook(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_global_step", torch.tensor([17], dtype=torch.int64))


class _FakeHashTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._filter_hook_impl = _FakeFilterHook()


class _FakeDynamicEmbedding(torch.nn.Module):
    def __init__(self, option):
        super().__init__()
        self._emb_opt = option
        self._hashtable = _FakeHashTable()


class _FakeEmbeddingModel(torch.nn.Module):
    def __init__(self, use_sparse_grad_group=False):
        super().__init__()
        option = _FakeEmbeddingOption(
            ["child_a", "child_b"],
            use_sparse_grad_group=use_sparse_grad_group,
        )
        current_hash = hashlib.sha256(option.coalesced_info().encode()).hexdigest()
        self.embeddings = torch.nn.ModuleDict(
            {f"CoalescedHashtable_{current_hash}": _FakeDynamicEmbedding(option)}
        )


class TestFilterGlobalStepCheckpointCompatibility(unittest.TestCase):
    def setUp(self):
        self.model = _FakeEmbeddingModel()
        self.groups = collect_filter_global_step_names(self.model)

    def test_save_uses_child_names_only(self):
        dense_state = OrderedDict(self.model.state_dict())

        use_child_filter_global_step_names(dense_state, self.groups)

        self.assertEqual(len(self.groups), 1)
        group = self.groups[0]
        self.assertNotIn(group.runtime_name, dense_state)
        self.assertEqual(
            set(dense_state),
            {"child_a@filter_global_step", "child_b@filter_global_step"},
        )
        self.assertEqual(dense_state["child_a@filter_global_step"].item(), 17)
        self.assertEqual(dense_state["child_b@filter_global_step"].item(), 17)

    @patch("recis.framework.checkpoint_manager.load_pt_file")
    def test_loads_legacy_sparse_grad_group_hash_names(self, load_pt_file_mock):
        group = self.groups[0]
        self.assertEqual(len(group.legacy_names), 3)
        for expected_step, old_hash_name in enumerate(group.legacy_names[1:], 29):
            with self.subTest(old_hash_name=old_hash_name):
                load_pt_file_mock.return_value = (
                    {old_hash_name: torch.tensor([expected_step], dtype=torch.int64)},
                    False,
                )
                saver = object.__new__(Saver)
                saver._model = self.model
                saver._dense_name_aliases = dict.fromkeys(
                    group.child_names, group.legacy_names
                )
                saver._dense_name_to_runtime = dict.fromkeys(
                    group.child_names, group.runtime_name
                )
                model_bank_conf = {name: {} for name in group.child_names}

                Saver._load_dense_model(saver, "/old-checkpoint", model_bank_conf)

                loaded_step = self.model.embeddings[
                    group.runtime_name.split(".")[1]
                ]._hashtable._filter_hook_impl._global_step
                self.assertEqual(loaded_step.item(), expected_step)

    def test_collects_previous_grouping_schema_hash_name(self):
        model = _FakeEmbeddingModel(use_sparse_grad_group=True)
        group = collect_filter_global_step_names(model)[0]
        option = next(iter(model.embeddings.values()))._emb_opt
        legacy_fields = {
            "sparse_grad_group_size": "hdmp_group_size",
            "sparse_grad_group_reduce_by": "hdmp_group_reduce_by",
        }
        legacy_info = {
            legacy_fields.get(name, name): value
            for name, value in json.loads(option.coalesced_info()).items()
        }
        legacy_info["grad_reduce_by"] = "hdmp_group_sum"
        legacy_hash = hashlib.sha256(json.dumps(legacy_info).encode()).hexdigest()

        self.assertTrue(
            any(
                f"CoalescedHashtable_{legacy_hash}" in name
                for name in group.legacy_names
            )
        )


class TestModelBankOptimizerSelection(unittest.TestCase):
    def test_unresolved_dense_buffer_does_not_reassign_optimizer(self):
        dense_parameter = "dense.weight"
        dense_buffer = "child_a@filter_global_step"
        optimizer = ExtraFields.recis_dense_optim
        model_names = {dense_parameter, dense_buffer, optimizer}
        parser = object.__new__(ModelBankParser)
        parser._model_names = set(model_names)
        parser._original_model_names = set(model_names)
        parser._dense_model_names = {dense_parameter, dense_buffer}
        parser._dense_parameter_names = {dense_parameter}
        parser._dense_name_to_runtime = {}
        parser._dense_runtime_to_names = {}
        parser._sparse_model_names = set()
        parser._extra_fields = ExtraFields
        parser._dense_pattern_matcher = DensePatternMatcher()
        parser._dense_oname = {}
        parser._sparse_oname = {}
        parser._get_dst_names = MagicMock(
            side_effect=lambda path, _: (
                set(),
                {dense_parameter} if path == "/resume" else set(),
                {optimizer},
            )
        )
        external = ModelBankEntry(
            path="/external",
            load={"*"},
            exclude={optimizer},
            ignore_error=True,
        )
        resume = ModelBankEntry(
            path="/resume",
            load={"*"},
            ignore_error=True,
        )

        parsed = parser._travel_model_bank_reversely([external, resume])

        self.assertEqual(parsed[dense_parameter][MBC.LOAD], "/resume")
        self.assertEqual(parsed[optimizer][MBC.LOAD], "/resume")
        self.assertNotIn(dense_buffer, parsed)


class TestModelBankFilterGlobalStepSelection(unittest.TestCase):
    def test_high_priority_child_resolves_shared_filter_step(self):
        child_a = "child_a@filter_global_step"
        child_b = "child_b@filter_global_step"
        runtime_name = "embeddings.CoalescedHashtable_hash._global_step"
        model_names = {child_a, child_b}
        parser = object.__new__(ModelBankParser)
        parser._model_names = set(model_names)
        parser._original_model_names = set(model_names)
        parser._dense_model_names = set(model_names)
        parser._dense_parameter_names = set()
        parser._dense_name_to_runtime = {
            child_a: runtime_name,
            child_b: runtime_name,
        }
        parser._dense_runtime_to_names = {runtime_name: set(model_names)}
        parser._sparse_model_names = set()
        parser._extra_fields = ExtraFields
        parser._dense_pattern_matcher = DensePatternMatcher()
        parser._dense_oname = {}
        parser._sparse_oname = {}
        parser._get_dst_names = MagicMock(
            side_effect=lambda path, _: (
                set(),
                {child_a} if path == "/resume" else {child_b},
                set(),
            )
        )
        base = ModelBankEntry(path="/base", load={"*"}, ignore_error=True)
        resume = ModelBankEntry(path="/resume", load={"*"}, ignore_error=True)

        parsed = parser._travel_model_bank_reversely([base, resume])

        self.assertEqual(parsed[child_a][MBC.LOAD], "/resume")
        self.assertNotIn(child_b, parsed)

    def test_filter_step_supports_table_oname(self):
        oname_success = [0]

        dense_oname = parse_dense_oname(
            DensePatternMatcher(),
            [{"new@*": "old@*"}],
            {"new@filter_global_step"},
            {"old@filter_global_step"},
            False,
            oname_success,
        )

        self.assertEqual(
            dense_oname,
            {"new@filter_global_step": "old@filter_global_step"},
        )
        self.assertEqual(oname_success, [1])


class TestSaveDeviceSynchronization(unittest.TestCase):
    """Tests CUDA synchronization without requiring checkpoint data."""

    def _make_saver_stub(self):
        stub = MagicMock()
        stub._shard_num = 1
        stub._shard_id = 1
        stub._sparse_state_dict = {}
        stub._io_state = {}
        fs = MagicMock()
        fs.exists.return_value = True
        stub._resolve_save_context.return_value = ("/tmp/ckpt", fs)
        return stub

    @patch("recis.framework.checkpoint_manager.torch.cuda.synchronize")
    @patch("recis.framework.checkpoint_manager.torch.cuda.is_initialized")
    def test_save_skips_cuda_sync_before_initialization(
        self, is_initialized, synchronize
    ):
        is_initialized.return_value = False
        sync_func = MagicMock()

        Saver.save(self._make_saver_stub(), "ckpt", sync_func=sync_func)

        synchronize.assert_not_called()
        sync_func.assert_called_once_with()

    @patch("recis.framework.checkpoint_manager.torch.cuda.synchronize")
    @patch("recis.framework.checkpoint_manager.torch.cuda.is_initialized")
    def test_save_synchronizes_initialized_cuda(self, is_initialized, synchronize):
        is_initialized.return_value = True
        sync_func = MagicMock()

        Saver.save(self._make_saver_stub(), "ckpt", sync_func=sync_func)

        synchronize.assert_called_once_with()
        sync_func.assert_called_once_with()


class TestUpdateCkptIndex(unittest.TestCase):
    """测试 _update_ckpt_index 方法：ckpt 版本列表与索引文件的更新逻辑。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _make_saver_stub(self):
        stub = MagicMock()
        stub._output_dir = self.tmpdir
        stub._checkpoint_file = "checkpoint"
        stub._checkpoint_version_list = []
        return stub

    def test_old_protocol_creates_index_file(self):
        """老协议下首次保存 ckpt 时，创建 checkpoint 索引文件。"""
        stub = self._make_saver_stub()
        fs = MagicMock()
        fs.exists.return_value = False
        written = {}

        def fake_open(path, mode):
            m = MagicMock()
            if mode == "w":
                m.__enter__ = lambda s: m
                m.__exit__ = lambda s, *a: None
                m.write = lambda data: written.update({"data": data})
            return m

        fs.open = fake_open

        Saver._update_ckpt_index(stub, "ckpt-1", "/path/ckpt-1", fs)

        self.assertEqual(written["data"], "ckpt-1\n")
        self.assertEqual(stub._checkpoint_version_list, ["ckpt-1"])

    def test_old_protocol_appends_to_existing(self):
        """老协议下追加保存 ckpt 时，在已有索引文件末尾追加记录。"""
        stub = self._make_saver_stub()
        fs = MagicMock()
        fs.exists.return_value = True
        written = {}

        def fake_open(path, mode):
            m = MagicMock()
            m.__enter__ = lambda s: m
            m.__exit__ = lambda s, *a: None
            if mode == "r":
                m.read = lambda: "ckpt-0\n"
            elif mode == "w":
                m.write = lambda data: written.update({"data": data})
            return m

        fs.open = fake_open

        Saver._update_ckpt_index(stub, "ckpt-1", "/path/ckpt-1", fs)

        self.assertEqual(written["data"], "ckpt-0\nckpt-1\n")



class TestEvictOldCkpt(unittest.TestCase):
    """测试 _evict_old_ckpt 方法的本地 checkpoint 淘汰逻辑。"""

    def _make_saver_stub(self):
        stub = MagicMock()
        stub._output_dir = "/output"
        stub._checkpoint_file = "checkpoint"
        stub._checkpoint_version_list = ["ckpt-old", "ckpt-new"]
        return stub






    def test_removes_dir_and_updates_index(self):
        """淘汰旧 checkpoint 时删除目录并更新索引文件。"""
        stub = self._make_saver_stub()
        fs = MagicMock()

        read_content = "ckpt-old\nckpt-new\n"
        written = {}

        def fake_open(path, mode):
            m = MagicMock()
            m.__enter__ = lambda s: m
            m.__exit__ = lambda s, *a: None
            if mode == "r":
                m.read = lambda: read_content
            elif mode == "w":
                lines = []
                m.write = lambda data: lines.append(data)
                m._lines = lines
                written["lines"] = lines
            return m

        fs.open = fake_open

        Saver._evict_old_ckpt(stub, "ckpt-old", "/output/ckpt-new", fs)

        fs.rm.assert_called_once_with("/output/ckpt-old/", recursive=True)
        self.assertEqual(written["lines"], ["ckpt-new\n"])
        self.assertEqual(stub._checkpoint_version_list, ["ckpt-new"])


class TestSaveRank0States(unittest.TestCase):
    """测试 _save_rank0_states 方法：rank-0 的状态落盘与空索引补写逻辑。"""

    def _make_saver_stub(self):
        stub = MagicMock()
        stub._dense_state_dict = {}
        stub._extra_save_dict = {}
        stub._shard_num = 4
        stub.save_dense_params = MagicMock()
        stub._save_generic = lambda self_unused, v: v
        return stub

    def test_writes_empty_index_when_sparse_empty(self):
        """sparse 参数为空时，补写空的 index 和 tensorkey.json 文件。"""
        stub = self._make_saver_stub()
        fs = MagicMock()
        fs.exists.return_value = False
        written_files = {}

        def fake_open(path, mode):
            m = MagicMock()
            m.__enter__ = lambda s: m
            m.__exit__ = lambda s, *a: None
            content = []
            m.write = lambda data: content.append(data)
            written_files[path] = (mode, content, m)
            return m

        fs.open = fake_open

        Saver._save_rank0_states(stub, "/ckpt/path", fs, {})

        self.assertIn("/ckpt/path/index", written_files)
        self.assertIn("/ckpt/path/tensorkey.json", written_files)

    def test_skips_index_when_exists(self):
        """index 文件已存在时，跳过补写操作。"""
        stub = self._make_saver_stub()
        fs = MagicMock()
        fs.exists.return_value = True

        Saver._save_rank0_states(stub, "/ckpt/path", fs, {})

        fs.open.assert_not_called()

    def test_saves_dense_when_present(self):
        """存在 dense 参数时，正确保存 model.pt 文件。"""
        stub = self._make_saver_stub()
        stub._dense_state_dict = {"layer.weight": MagicMock()}
        fs = MagicMock()
        fs.exists.return_value = True

        Saver._save_rank0_states(stub, "/ckpt/path", fs, {})

        stub.save_dense_params.assert_called_once_with(
            "/ckpt/path", stub._dense_state_dict, fs=fs
        )

    @patch("torch.save")
    def test_saves_extra_and_io_state_count(self, mock_torch_save):
        """保存 extra 参数（如 global_step）和 io_state_count 分片计数文件。"""
        stub = self._make_saver_stub()
        stub._extra_save_dict = {"global_step": MagicMock()}
        stub._save_generic = MagicMock(return_value="serialized")
        fs = MagicMock()
        fs.exists.return_value = True
        written_data = {}

        def fake_open(path, mode):
            m = MagicMock()
            m.__enter__ = lambda s: m
            m.__exit__ = lambda s, *a: None
            m.write = lambda data: written_data.update({path: data})
            return m

        fs.open = fake_open
        io_states = {"train_io": {"offset": 100}}

        Saver._save_rank0_states(stub, "/ckpt/path", fs, io_states)

        self.assertIn("/ckpt/path/io_state_count", written_data)
        self.assertEqual(written_data["/ckpt/path/io_state_count"], "4")
        mock_torch_save.assert_called_once()






class TestLoadPathResolution(unittest.TestCase):
    """测试 load() 的显式路径和 checkpoint 索引解析。"""

    def _make_saver_stub(self):
        stub = MagicMock()
        stub._output_dir = "/output"
        stub._checkpoint_file = "checkpoint"
        stub.load_by_config = MagicMock()
        stub._shard_id = 0
        return stub

    def test_direct_path_uses_literal(self):
        """direct_path=True 时按字面路径加载，不走任何解析逻辑。"""
        stub = self._make_saver_stub()

        Saver.load(
            stub,
            ckpt_path="/explicit/path",
            direct_path=True,
            model_bank_conf={"*": {}},
        )

        stub.load_by_config.assert_called_once_with("/explicit/path", 0, {"*": {}})

    def test_direct_path_empty_returns_early(self):
        """direct_path=True 但 ckpt_path 为空时提前返回，不调用 load_by_config。"""
        stub = self._make_saver_stub()

        Saver.load(stub, ckpt_path=None, direct_path=True)

        stub.load_by_config.assert_not_called()





if __name__ == "__main__":
    unittest.main()
