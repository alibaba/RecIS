"""recis.framework.checkpoint_manager 子方法的单元测试。

覆盖: _save_rank0_states, _update_ckpt_index, _evict_old_ckpt,
_register_ckpt, _maybe_inject_mos_resume_entry, load() 路径解析。

运行:
    python -m pytest tests/checkpoint/checkpoint_manager_test.py -v
"""

import os
import sys


# 仅在 recis.so 不存在时设置 BUILD_DOCUMENT（本地开发环境）。
# CI 环境中 .so 必须正常加载，以保证 torch.classes 注册成功。
_recis_so = os.path.join(os.path.dirname(__file__), "..", "..", "recis", "lib", "recis.so")
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

from recis.framework.checkpoint_manager import Saver  # noqa: E402


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

        Saver.load(stub, ckpt_path="/explicit/path", direct_path=True, model_bank_conf={"*": {}})

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
