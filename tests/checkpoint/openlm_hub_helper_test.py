"""OpenlmHubHelper 及 ckpt 管理相关逻辑的单元测试。

所有 openlm_hub 依赖在未安装时自动 mock —— 无需 GPU、分布式或 MOS 服务。

运行:
    python -m pytest tests/checkpoint/openlm_hub_helper_test.py -v
"""

import os
import sys


# 仅在 recis.so 不存在时设置 BUILD_DOCUMENT（本地开发环境）。
# CI 环境中 .so 必须正常加载，以保证 torch.classes 注册成功。
_recis_so = os.path.join(os.path.dirname(__file__), "..", "..", "recis", "lib", "recis.so")
if not os.path.exists(os.path.abspath(_recis_so)):
    os.environ["BUILD_DOCUMENT"] = "1"

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

# 仅在内部依赖不可用时 mock，避免本地/CI 环境差异报错
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

from recis.utils.openlm_hub_helper import OpenlmHubHelper  # noqa: E402


class TestOpenlmHubHelperInit(unittest.TestCase):
    """测试 OpenlmHubHelper 初始化：基本属性赋值与缓存初始状态。"""
    def test_basic_attrs(self):
        """初始化后 version_uri 和 user_id 正确赋值。"""
        helper = OpenlmHubHelper("model.proj.name/version=v1", "user123")
        self.assertEqual(helper.version_uri, "model.proj.name/version=v1")
        self.assertEqual(helper.user_id, "user123")

    def test_cache_initially_empty(self):
        """初始化后 _ckpt_path_by_id 缓存为空。"""
        helper = OpenlmHubHelper("model.proj.name/version=v1", None)
        self.assertIsNone(helper.pop_write_path("nonexistent"))


class TestCacheWritePath(unittest.TestCase):
    """测试 ckpt 写入路径缓存（_ckpt_path_by_id）的存取逻辑。"""
    def setUp(self):
        self.helper = OpenlmHubHelper("model.proj.name/version=v1", "user1")

    def test_cache_and_pop(self):
        """缓存写入路径后可正确取出。"""
        self.helper.cache_write_path("ckpt-100", "/data/write/ckpt-100")
        self.assertEqual(
            self.helper.pop_write_path("ckpt-100"), "/data/write/ckpt-100"
        )

    def test_pop_removes_entry(self):
        """pop 取出后条目被移除，再次 pop 返回 None。"""
        self.helper.cache_write_path("ckpt-100", "/data/write/ckpt-100")
        self.helper.pop_write_path("ckpt-100")
        self.assertIsNone(self.helper.pop_write_path("ckpt-100"))

    def test_pop_nonexistent_returns_none(self):
        """pop 不存在的 key 时返回 None。"""
        self.assertIsNone(self.helper.pop_write_path("ckpt-999"))

    def test_multiple_entries(self):
        """多个 ckpt 的写入路径缓存互不干扰。"""
        self.helper.cache_write_path("ckpt-1", "/path/1")
        self.helper.cache_write_path("ckpt-2", "/path/2")
        self.assertEqual(self.helper.pop_write_path("ckpt-1"), "/path/1")
        self.assertEqual(self.helper.pop_write_path("ckpt-2"), "/path/2")


@patch("recis.utils.openlm_hub_helper.MosCkptFileManager")
@patch("recis.utils.openlm_hub_helper.get_ckpt_access_path")
class TestGetSaveContext(unittest.TestCase):
    """测试 get_save_context 方法：创建写入上下文并修正 EROFS WRITE 路径。"""
    def setUp(self):
        self.helper = OpenlmHubHelper("model.proj.name/version=v1", "user1")

    def test_returns_tuple(self, mock_access_path, mock_cfm_cls):
        """返回正确三元组 (cfm, path, fs)，且 cfm.path 被重写为 WRITE 路径。"""
        mock_cfm = MagicMock()
        mock_cfm.ckpt_physical_path = "xpfs://cluster/data/ckpt-100"
        mock_cfm.get_fs.return_value = MagicMock()
        mock_cfm_cls.return_value = mock_cfm
        mock_access_path.return_value = "/data/write/ckpt-100"

        cfm, path, fs = self.helper.get_save_context("ckpt-100")

        self.assertEqual(cfm, mock_cfm)
        self.assertEqual(path, "/data/write/ckpt-100")
        self.assertEqual(cfm.path, "/data/write/ckpt-100")
        mock_cfm_cls.assert_called_once_with(
            "model.proj.name/version=v1/ckpt_id=ckpt-100", mode="w"
        )

    def test_overwrites_cfm_path(self, mock_access_path, mock_cfm_cls):
        """cfm.path 为 READ 路径时被强制改写为 WRITE 路径，规避 EROFS。"""
        mock_cfm = MagicMock()
        mock_cfm.ckpt_physical_path = "xpfs://cluster/read/ckpt-1"
        mock_cfm.path = "/read/ckpt-1"
        mock_cfm.get_fs.return_value = MagicMock()
        mock_cfm_cls.return_value = mock_cfm
        mock_access_path.return_value = "/write/ckpt-1"

        cfm, path, fs = self.helper.get_save_context("ckpt-1")

        self.assertEqual(cfm.path, "/write/ckpt-1")


@patch("recis.utils.openlm_hub_helper.MosCkptFileManager")
class TestResolveLoadPath(unittest.TestCase):
    """测试 resolve_load_path 方法：通过 MOS 解析 ckpt 读取路径。"""
    def setUp(self):
        self.helper = OpenlmHubHelper("model.proj.name/version=v1", "user1")

    def test_latest_ckpt(self, mock_cfm_cls):
        """ckpt_id=None 时取最新 ckpt 的 READ 路径。"""
        mock_cfm = MagicMock()
        mock_cfm.path = "/data/read/latest-ckpt"
        mock_cfm_cls.return_value = mock_cfm

        result = self.helper.resolve_load_path()

        self.assertEqual(result, "/data/read/latest-ckpt")
        mock_cfm_cls.assert_called_once_with(
            "model.proj.name/version=v1", mode="r"
        )

    def test_specific_ckpt_id(self, mock_cfm_cls):
        """传入 ckpt_id 时解析对应 ckpt 的 READ 路径。"""
        mock_cfm = MagicMock()
        mock_cfm.path = "/data/read/ckpt-50"
        mock_cfm_cls.return_value = mock_cfm

        self.helper.resolve_load_path("ckpt-50")

        mock_cfm_cls.assert_called_once_with(
            "model.proj.name/version=v1/ckpt_id=ckpt-50", mode="r"
        )

    def test_not_found_returns_none(self, mock_cfm_cls):
        """ckpt 不存在时返回 None。"""
        from openlm_hub.error import MosCkptNotFoundError

        mock_cfm_cls.side_effect = MosCkptNotFoundError("not found")

        result = self.helper.resolve_load_path("missing")
        self.assertIsNone(result)


@patch("recis.utils.openlm_hub_helper.MosCkptFileManager")
class TestResolveLatestResume(unittest.TestCase):
    """测试 resolve_latest_resume 方法：查 MOS 最新 ckpt 用于断点续训。"""
    def setUp(self):
        self.helper = OpenlmHubHelper("model.proj.name/version=v1", "user1")

    def test_found(self, mock_cfm_cls):
        """找到最新 ckpt 时返回 (resume_path, ckpt_physical_path) 二元组。"""
        mock_cfm = MagicMock()
        mock_cfm.path = "/data/read/ckpt-latest"
        mock_cfm.ckpt_physical_path = "xpfs://cluster/data/ckpt-latest"
        mock_cfm_cls.return_value = mock_cfm

        result = self.helper.resolve_latest_resume()

        self.assertEqual(result, ("/data/read/ckpt-latest", "xpfs://cluster/data/ckpt-latest"))

    def test_not_found_returns_none(self, mock_cfm_cls):
        """无已注册 ckpt 时返回 None。"""
        from openlm_hub.error import MosCkptNotFoundError

        mock_cfm_cls.side_effect = MosCkptNotFoundError("empty")

        result = self.helper.resolve_latest_resume()
        self.assertIsNone(result)


@patch("recis.utils.openlm_hub_helper.add_or_update_ckpt_metrics")
@patch("recis.utils.openlm_hub_helper.get_mos_metrics")
class TestRegisterAndReport(unittest.TestCase):
    """测试 register_and_report 方法：ckpt 注册与 MOS metrics 上报。"""
    def setUp(self):
        self.helper = OpenlmHubHelper("model.proj.name/version=v1", "user1")

    def test_calls_register_and_metrics(self, mock_metrics, mock_update):
        """注册 ckpt 并上报 metrics，验证 MOS URI 和 MODE.TRAIN 前缀正确。"""
        mock_metrics.return_value = {"loss": 0.5}
        mock_cfm = MagicMock()

        self.helper.register_and_report(mock_cfm, "ckpt-10", labels=["step=10"])

        mock_cfm.register_ckpt.assert_called_once_with(labels=["step=10"])
        mock_update.assert_called_once()
        call_kwargs = mock_update.call_args[1]
        self.assertEqual(call_kwargs["mos_ckpt_uri"], "model.proj.name/version=v1/ckpt_id=ckpt-10")
        self.assertEqual(call_kwargs["user_id"], "user1")
        self.assertIn("MODE.TRAIN.loss", call_kwargs["metrics"])
        self.assertIn("MODE.TRAIN.ckpt_id", call_kwargs["metrics"])

    def test_empty_labels(self, mock_metrics, mock_update):
        """无 label 时以空列表调用 register_ckpt。"""
        mock_metrics.return_value = {}
        mock_cfm = MagicMock()

        self.helper.register_and_report(mock_cfm, "ckpt-1")

        mock_cfm.register_ckpt.assert_called_once_with(labels=[])


@patch("recis.utils.openlm_hub_helper.delete_ckpt")
class TestDelete(unittest.TestCase):
    """测试 delete 方法：从 MOS 注销已注册 ckpt。"""
    def setUp(self):
        self.helper = OpenlmHubHelper("model.proj.name/version=v1", "user1")

    def test_calls_delete_ckpt(self, mock_delete):
        """调用 delete 时传入正确的 MOS URI 和 user_id。"""
        self.helper.delete("ckpt-old")

        mock_delete.assert_called_once_with(
            mos_ckpt_uri="model.proj.name/version=v1/ckpt_id=ckpt-old",
            user_id="user1",
        )


class TestFormatPhysicalPath(unittest.TestCase):
    """测试 mos.py 中 format_physical_path 的路径前缀拆分逻辑。"""

    def setUp(self):
        import recis.utils.mos as mos_module
        self._mos_module = mos_module
        self._mock_access = MagicMock()
        mos_module.get_ckpt_access_path = self._mock_access

    def test_non_xpfs_returns_empty_prefix(self):
        """非 xpfs 路径（本地路径）返回空前缀，原始路径不变。"""
        prefix, local_path = self._mos_module.format_physical_path("/data/local/path")
        self.assertEqual(prefix, "")
        self.assertEqual(local_path, "/data/local/path")
        self._mock_access.assert_not_called()

    def test_xpfs_standard(self):
        """标准 xpfs 路径正确拆分为前缀和本地路径。"""
        self._mock_access.return_value = "/data/ckpt/model-v1"
        prefix, local_path = self._mos_module.format_physical_path(
            "xpfs://ks03-xpfs-0/data/ckpt/model-v1"
        )
        self.assertEqual(prefix, "xpfs://ks03-xpfs-0")
        self.assertEqual(local_path, "/data/ckpt/model-v1")

    def test_xpfs_rw_split_prefix_empty(self):
        """xpfs 读写分离场景下，WRITE 前缀被替换为 READ 路径后前缀置空。"""
        self._mock_access.return_value = "/read-base/ckpt/model-v1"
        prefix, local_path = self._mos_module.format_physical_path(
            "xpfs://cluster/write-base/ckpt/model-v1"
        )
        self.assertEqual(prefix, "")
        self.assertEqual(local_path, "/read-base/ckpt/model-v1")

    def test_dfs_path(self):
        """dfs 协议路径返回空前缀，原始路径原样返回。"""
        prefix, local_path = self._mos_module.format_physical_path("dfs://cluster/output/ckpt")
        self.assertEqual(prefix, "")
        self.assertEqual(local_path, "dfs://cluster/output/ckpt")
        self._mock_access.assert_not_called()


if __name__ == "__main__":
    unittest.main()
