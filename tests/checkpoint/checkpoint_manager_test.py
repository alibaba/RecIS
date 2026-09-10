"""recis.framework.checkpoint_manager 子方法的单元测试。

覆盖: _save_rank0_states, _update_ckpt_index, _evict_old_ckpt,
_register_ckpt, _maybe_inject_mos_resume_entry, load() 路径解析。

运行:
    python -m pytest tests/checkpoint/checkpoint_manager_test.py -v
"""

import hashlib
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


# openlm_hub 未安装时 mock 整个包
# 注：若已安装但缺少 CkptAction（版本不兼容），下方有第二层兜底
try:
    import openlm_hub  # noqa: F401
except ImportError:
    _mock_openlm_hub = MagicMock()
    _mock_openlm_hub.error = MagicMock()
    _mock_openlm_hub.error.MosCkptNotFoundError = type(
        "MosCkptNotFoundError", (Exception,), {}
    )
    _mock_openlm_hub.constants = MagicMock()
    _mock_openlm_hub.constants.CkptAction = MagicMock()
    _mock_openlm_hub.constants.CkptAction.WRITE = "WRITE"

    sys.modules["openlm_hub"] = _mock_openlm_hub
    sys.modules["openlm_hub.openlm_api"] = MagicMock()
    sys.modules["openlm_hub.constants"] = _mock_openlm_hub.constants
    sys.modules["openlm_hub.error"] = _mock_openlm_hub.error
    sys.modules["openlm_hub.utils"] = MagicMock()
    sys.modules["openlm_hub.utils.storage"] = MagicMock()

# 第二层兜底：openlm_hub 已安装但 constants.py 缺少 CkptAction（版本过旧）时，
# 往已加载的 constants 模块注入兼容 mock
try:
    from openlm_hub.constants import CkptAction  # noqa: F401
except ImportError:
    import openlm_hub.constants

    _CkptActionMock = MagicMock()
    _CkptActionMock.WRITE = "WRITE"
    openlm_hub.constants.CkptAction = _CkptActionMock


def _mock_if_missing(module_name, mock_obj=None):
    try:
        __import__(module_name)
    except (ImportError, ModuleNotFoundError):
        sys.modules[module_name] = mock_obj or MagicMock()


_mock_if_missing("recis.framework.metrics", MagicMock(get_mos_metrics=dict))
_mock_if_missing("recis.info", MagicMock(is_internal_enabled=lambda: False))
_mock_if_missing("pangudfs_client")
_mock_if_missing("pangudfs_client.common")
_mock_if_missing("pangudfs_client.common.exception")
_mock_if_missing("pangudfs_client.common.exception.exceptions")
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
    def __init__(self, children):
        self.children = children

    def coalesced_info(self):
        return (
            '{"dim": 8, "dtype": "torch.float32", "device": "cpu", '
            '"initializer": "ConstantInitializer_0", '
            '"grad_reduce_by": "worker", "hdmp_group_size": null, '
            '"hdmp_group_reduce_by": null, '
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
    def __init__(self, children):
        super().__init__()
        self._emb_opt = _FakeEmbeddingOption(children)
        self._hashtable = _FakeHashTable()


class _FakeEmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        option = _FakeEmbeddingOption(["child_a", "child_b"])
        current_hash = hashlib.sha256(option.coalesced_info().encode()).hexdigest()
        self.embeddings = torch.nn.ModuleDict(
            {
                f"CoalescedHashtable_{current_hash}": _FakeDynamicEmbedding(
                    option.children
                )
            }
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
    def test_loads_pre_hdmp_hash_name(self, load_pt_file_mock):
        group = self.groups[0]
        self.assertEqual(len(group.legacy_names), 2)
        old_hash_name = group.legacy_names[1]
        load_pt_file_mock.return_value = (
            {old_hash_name: torch.tensor([29], dtype=torch.int64)},
            False,
        )
        saver = object.__new__(Saver)
        saver._model = self.model
        saver._dense_name_aliases = dict.fromkeys(group.child_names, group.legacy_names)
        saver._dense_name_to_runtime = dict.fromkeys(
            group.child_names, group.runtime_name
        )
        model_bank_conf = {name: {} for name in group.child_names}

        Saver._load_dense_model(saver, "/old-checkpoint", model_bank_conf)

        loaded_step = self.model.embeddings[
            group.runtime_name.split(".")[1]
        ]._hashtable._filter_hook_impl._global_step
        self.assertEqual(loaded_step.item(), 29)


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

    def _make_saver_stub(self, is_openlm_hub=False):
        stub = MagicMock()
        stub._is_openlm_hub_ckpt = is_openlm_hub
        stub._output_dir = self.tmpdir
        stub._checkpoint_file = "checkpoint"
        stub._checkpoint_version_list = []
        stub.openlm_hub_helper = MagicMock() if is_openlm_hub else None
        return stub

    def test_old_protocol_creates_index_file(self):
        """老协议下首次保存 ckpt 时，创建 checkpoint 索引文件。"""
        stub = self._make_saver_stub(is_openlm_hub=False)
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
        stub = self._make_saver_stub(is_openlm_hub=False)
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

    def test_openlm_hub_skips_index_file(self):
        """openlm_hub 模式下跳过索引文件，仅缓存写入路径供淘汰时使用。"""
        stub = self._make_saver_stub(is_openlm_hub=True)
        fs = MagicMock()

        Saver._update_ckpt_index(stub, "ckpt-1", "/write/ckpt-1", fs)

        fs.open.assert_not_called()
        self.assertEqual(stub._checkpoint_version_list, ["ckpt-1"])
        stub.openlm_hub_helper.cache_write_path.assert_called_once_with(
            "ckpt-1", "/write/ckpt-1"
        )


class TestEvictOldCkpt(unittest.TestCase):
    """测试 _evict_old_ckpt 方法：旧 ckpt 淘汰与 MOS 注销逻辑。"""

    class FakePanguException(Exception):
        def __init__(self, pangu_err_no):
            self.pangu_err_no = pangu_err_no
            super().__init__(f"pangu errno={pangu_err_no}")

    def _make_saver_stub(self, is_openlm_hub=False):
        stub = MagicMock()
        stub._is_openlm_hub_ckpt = is_openlm_hub
        stub._output_dir = "/output"
        stub._checkpoint_file = "checkpoint"
        stub._checkpoint_version_list = ["ckpt-old", "ckpt-new"]
        stub._mos = MagicMock() if not is_openlm_hub else None
        stub.openlm_hub_helper = MagicMock() if is_openlm_hub else None
        return stub

    def test_openlm_hub_mode_pops_and_deletes(self):
        """openlm_hub 模式下淘汰旧 ckpt：弹出写入路径、删除文件、注销 MOS 记录。"""
        stub = self._make_saver_stub(is_openlm_hub=True)
        stub.openlm_hub_helper.pop_write_path.return_value = "/write/ckpt-old"
        fs = MagicMock()

        Saver._evict_old_ckpt(stub, "ckpt-old", "/write/ckpt-new", fs)

        stub.openlm_hub_helper.pop_write_path.assert_called_once_with("ckpt-old")
        fs.rm.assert_called_once_with("/write/ckpt-old/", recursive=True)
        stub.openlm_hub_helper.delete.assert_called_once_with("ckpt-old")
        self.assertEqual(stub._checkpoint_version_list, ["ckpt-new"])

    def test_openlm_hub_mode_no_write_path_skips_rm(self):
        """openlm_hub 模式下无写入路径时，跳过文件删除仅注销 MOS 记录。"""
        stub = self._make_saver_stub(is_openlm_hub=True)
        stub.openlm_hub_helper.pop_write_path.return_value = None
        fs = MagicMock()

        Saver._evict_old_ckpt(stub, "ckpt-old", "/path", fs)

        fs.rm.assert_not_called()
        stub.openlm_hub_helper.delete.assert_called_once_with("ckpt-old")

    def test_openlm_hub_mode_ignores_file_not_found(self):
        """旧 ckpt 已不存在时，继续更新版本列表并注销 MOS 记录。"""
        stub = self._make_saver_stub(is_openlm_hub=True)
        stub.openlm_hub_helper.pop_write_path.return_value = "/write/ckpt-old"
        fs = MagicMock()
        fs.rm.side_effect = FileNotFoundError("already removed")

        Saver._evict_old_ckpt(stub, "ckpt-old", "/write/ckpt-new", fs)

        stub.openlm_hub_helper.delete.assert_called_once_with("ckpt-old")
        self.assertEqual(stub._checkpoint_version_list, ["ckpt-new"])

    def test_openlm_hub_mode_ignores_pangu_not_found(self):
        """Pangu errno=2 与 FileNotFoundError 一样按删除成功处理。"""
        stub = self._make_saver_stub(is_openlm_hub=True)
        stub.openlm_hub_helper.pop_write_path.return_value = "/write/ckpt-old"
        fs = MagicMock()
        fs.rm.side_effect = self.FakePanguException(2)

        with patch(
            "recis.framework.checkpoint_manager.PanguException",
            self.FakePanguException,
        ):
            Saver._evict_old_ckpt(stub, "ckpt-old", "/write/ckpt-new", fs)

        stub.openlm_hub_helper.delete.assert_called_once_with("ckpt-old")
        self.assertEqual(stub._checkpoint_version_list, ["ckpt-new"])

    def test_openlm_hub_mode_reraises_other_pangu_errors(self):
        """权限、IO、只读等 Pangu 错误不能被误吞。"""
        for pangu_err_no in (3, 5, 11):
            with self.subTest(pangu_err_no=pangu_err_no):
                stub = self._make_saver_stub(is_openlm_hub=True)
                stub.openlm_hub_helper.pop_write_path.return_value = "/write/ckpt-old"
                fs = MagicMock()
                fs.rm.side_effect = self.FakePanguException(pangu_err_no)

                with patch(
                    "recis.framework.checkpoint_manager.PanguException",
                    self.FakePanguException,
                ):
                    with self.assertRaises(self.FakePanguException):
                        Saver._evict_old_ckpt(stub, "ckpt-old", "/write/ckpt-new", fs)

                stub.openlm_hub_helper.delete.assert_not_called()
                self.assertEqual(
                    stub._checkpoint_version_list, ["ckpt-old", "ckpt-new"]
                )

    def test_old_protocol_removes_dir_and_updates_index(self):
        """老协议下淘汰旧 ckpt：删目录、更新索引文件、调 MOS ckpt_update 注销。"""
        stub = self._make_saver_stub(is_openlm_hub=False)
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
        stub._mos.ckpt_update.assert_called_once_with(
            ckpt_id="ckpt-old", path="/output/ckpt-new", is_delete=True
        )


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


class TestRegisterCkpt(unittest.TestCase):
    """测试 _register_ckpt 方法：openlm_hub 和老协议两种注册路径。"""

    def _make_saver_stub(self, is_openlm_hub=False):
        stub = MagicMock()
        stub.openlm_hub_helper = MagicMock() if is_openlm_hub else None
        stub._mos = MagicMock()
        return stub

    def test_openlm_hub_mode(self):
        """openlm_hub 模式下通过 helper 注册 ckpt 并上报 metrics。"""
        stub = self._make_saver_stub(is_openlm_hub=True)
        cfm = MagicMock()

        Saver._register_ckpt(stub, cfm, "ckpt-10", "/path", "step", "10")

        stub.openlm_hub_helper.register_and_report.assert_called_once_with(
            cfm, "ckpt-10", labels=["step=10"]
        )
        self.assertEqual(stub._mos.last_ckpt_id, "ckpt-10")

    def test_openlm_hub_no_labels(self):
        """openlm_hub 模式无 label 时以空列表正常注册。"""
        stub = self._make_saver_stub(is_openlm_hub=True)
        cfm = MagicMock()

        Saver._register_ckpt(stub, cfm, "ckpt-5", "/path", None, None)

        stub.openlm_hub_helper.register_and_report.assert_called_once_with(
            cfm, "ckpt-5", labels=[]
        )

    def test_old_protocol_with_mos(self):
        """老协议下通过 MOS ckpt_update 注册 ckpt。"""
        stub = self._make_saver_stub(is_openlm_hub=False)

        Saver._register_ckpt(stub, None, "ckpt-10", "/path/ckpt-10", "step", "10")

        stub._mos.ckpt_update.assert_called_once_with(
            ckpt_id="ckpt-10",
            path="/path/ckpt-10",
            label_key="step",
            label_value="10",
        )

    def test_no_mos_no_helper_does_nothing(self):
        """无 MOS 也无 helper 时，_register_ckpt 不做任何事。"""
        stub = MagicMock()
        stub.openlm_hub_helper = None
        stub._mos = None

        Saver._register_ckpt(stub, None, "ckpt-1", "/path", None, None)


class TestMaybeInjectMosResumeEntry(unittest.TestCase):
    """测试 _maybe_inject_mos_resume_entry 方法：自动断点续训条目注入。"""

    def _make_saver_stub(self, is_openlm_hub=False):
        stub = MagicMock()
        stub._is_openlm_hub_ckpt = is_openlm_hub
        stub.openlm_hub_helper = MagicMock() if is_openlm_hub else None
        if is_openlm_hub:
            stub.openlm_hub_helper.version_uri = "model.proj.name/version=v1"
        return stub

    def test_not_openlm_hub_returns_unchanged(self):
        """非 openlm_hub 模式，model_bank 列表原样返回。"""
        stub = self._make_saver_stub(is_openlm_hub=False)
        original = [{"path": "/existing"}]

        result = Saver._maybe_inject_mos_resume_entry(stub, original)

        self.assertEqual(result, original)

    def test_openlm_hub_no_ckpt_returns_unchanged(self):
        """openlm_hub 模式但无已有 ckpt 时，model_bank 列表原样返回。"""
        stub = self._make_saver_stub(is_openlm_hub=True)
        stub.openlm_hub_helper.resolve_latest_resume.return_value = None
        original = [{"path": "/existing"}]

        result = Saver._maybe_inject_mos_resume_entry(stub, original)

        self.assertEqual(result, original)

    def test_openlm_hub_found_ckpt_appends_entry(self):
        """找到已有 ckpt 时，在 model_bank 末尾追加一条续训条目。"""
        stub = self._make_saver_stub(is_openlm_hub=True)
        stub.openlm_hub_helper.resolve_latest_resume.return_value = (
            "/data/read/ckpt-latest",
            "xpfs://cluster/data/ckpt-latest",
        )
        original = [{"path": "/existing"}]

        result = Saver._maybe_inject_mos_resume_entry(stub, original)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], {"path": "/existing"})
        injected = result[1]
        self.assertEqual(injected["path"], "/data/read/ckpt-latest")
        self.assertEqual(injected["load"], {"*"})
        self.assertTrue(injected["ignore_error"])

    def test_cross_app_tag(self):
        """跨应用 ckpt 的场景下，也正常追加续训条目。"""
        stub = self._make_saver_stub(is_openlm_hub=True)
        stub.openlm_hub_helper.resolve_latest_resume.return_value = (
            "/data/read/ckpt-latest",
            "xpfs://cluster/other_app/data/ckpt-latest",
        )
        os.environ["HIPPO_APP"] = "my_app"

        try:
            result = Saver._maybe_inject_mos_resume_entry(stub, [])
            self.assertEqual(len(result), 1)
        finally:
            del os.environ["HIPPO_APP"]


class TestLoadPathResolution(unittest.TestCase):
    """测试 load() 方法的路径解析分支：字面路径/openlm_hub/老协议索引。"""

    def _make_saver_stub(self, is_openlm_hub=False):
        stub = MagicMock()
        stub._is_openlm_hub_ckpt = is_openlm_hub
        stub.openlm_hub_helper = MagicMock() if is_openlm_hub else None
        stub._output_dir = "/output"
        stub._checkpoint_file = "checkpoint"
        stub.load_by_config = MagicMock()
        stub._shard_id = 0
        return stub

    def test_direct_path_uses_literal(self):
        """direct_path=True 时按字面路径加载，不走任何解析逻辑。"""
        stub = self._make_saver_stub(is_openlm_hub=False)

        Saver.load(
            stub,
            ckpt_path="/explicit/path",
            direct_path=True,
            model_bank_conf={"*": {}},
        )

        stub.load_by_config.assert_called_once_with("/explicit/path", 0, {"*": {}})

    def test_direct_path_empty_returns_early(self):
        """direct_path=True 但 ckpt_path 为空时提前返回，不调用 load_by_config。"""
        stub = self._make_saver_stub(is_openlm_hub=False)

        Saver.load(stub, ckpt_path=None, direct_path=True)

        stub.load_by_config.assert_not_called()

    def test_openlm_hub_with_ckpt_path_uses_literal(self):
        """openlm_hub 模式下传入 ckpt_path 时直接使用字面路径，不走 MOS 解析。"""
        stub = self._make_saver_stub(is_openlm_hub=True)

        Saver.load(stub, ckpt_path="/explicit/path", model_bank_conf={"*": {}})

        stub.load_by_config.assert_called_once_with("/explicit/path", 0, {"*": {}})
        stub.openlm_hub_helper.resolve_load_path.assert_not_called()

    def test_openlm_hub_no_path_resolves_via_helper(self):
        """openlm_hub 模式下不传 ckpt_path 时，通过 helper.resolve_load_path 解析。"""
        stub = self._make_saver_stub(is_openlm_hub=True)
        stub.openlm_hub_helper.resolve_load_path.return_value = "/resolved/ckpt-5"

        Saver.load(stub, ckpt_id="ckpt-5", model_bank_conf={"*": {}})

        stub.openlm_hub_helper.resolve_load_path.assert_called_once_with("ckpt-5")
        stub.load_by_config.assert_called_once_with("/resolved/ckpt-5", 0, {"*": {}})

    def test_openlm_hub_not_found_returns_early(self):
        """openlm_hub 模式 helper 找不到 ckpt 时提前返回，不调用 load_by_config。"""
        stub = self._make_saver_stub(is_openlm_hub=True)
        stub.openlm_hub_helper.resolve_load_path.return_value = None

        Saver.load(stub, model_bank_conf={"*": {}})

        stub.load_by_config.assert_not_called()


if __name__ == "__main__":
    unittest.main()
