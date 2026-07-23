# tests/unit/open_storage_utils_test.py
# -*- coding: utf8 -*-
from unittest.mock import MagicMock, patch
import sys
import os
import pytest

# ── 阻断 C 扩展导入链 ──────────────────────────────────────
_mock_py_interface = MagicMock()
_mock_interface = MagicMock()
sys.modules.setdefault("column_io.lib.py_interface", _mock_py_interface)
sys.modules.setdefault("column_io.lib.interface", _mock_interface)

# ── 模拟 ODPS 访问凭证环境变量，防止 HashableSessionStruct 初始化失败 ──
MOCK_CREDS_ENV = {
    "ENCODED_ODPS_ACCESS_ID": "",
    "ENCODED_ODPS_ACCESS_KEY": "",
    "access_id": "mock_id",
    "access_key": "mock_key"
}
class TestHashableSessionStruct:
    """测试 HashableSessionStruct 类的新增逻辑"""

    @patch.dict(os.environ, {"HALO_WORKER_DOCKER_IMAGE": "registry/halo:worker-v1.2_accelerated"}, clear=False)
    def test_new_fields_initialization(self):
        """验证新增字段 halo_worker_docker_image_tag 的正确提取"""
        from column_io.dataset.open_storage_utils import HashableSessionStruct
        
        session = HashableSessionStruct(
            project="p", table="t", logical_partition="ds=1", 
            physical_partitions=["ds=1"], select_columns=["c1"]
        )
        
        # 验证后缀 _accelerated 被正确去除
        assert session.halo_worker_docker_image_tag == "worker-v1.2"
        assert session.halo_worker_docker_image == "registry/halo:worker-v1.2_accelerated"

    @patch.dict(os.environ, {"HALO_WORKER_DOCKER_IMAGE": "registry/halo:worker-v1.2_accelerated"}, clear=False)
    def test_hash_id_includes_new_params(self):
        """验证 get_hash_id 包含了 tunnel_endpoint, odps_sdk_version 等新参数"""
        from column_io.dataset.open_storage_utils import HashableSessionStruct
        
        session = HashableSessionStruct(
            project="p", table="t", logical_partition="ds=1", 
            physical_partitions=["ds=1"], select_columns=["c1"]
        )
        
        hash_id = session.get_hash_id()
        assert len(hash_id) == 64  # SHA256 hex length
        # 验证 to_str_extended 能正常工作且不报错
        extended_str = session.to_str_extended()
        assert "tunnel endpoint" in extended_str.lower()

    @patch("column_io.dataset.open_storage_utils.get_odps_partition_modified_timestamp")
    def test_partition_timestamp_caching(self, mock_get_ts):
        """验证分区修改时间戳的缓存逻辑"""
        from column_io.dataset.open_storage_utils import HashableSessionStruct
        
        mock_get_ts.return_value = "1700000000"
        session = HashableSessionStruct(
            project="p", table="t", logical_partition="ds=1", 
            physical_partitions=["ds=1"], select_columns=["c1"]
        )
        
        ts1 = session.get_partition_modified_timestamp()
        ts2 = session.get_partition_modified_timestamp()
        
        assert ts1 == "1700000000"
        assert ts1 == ts2
        mock_get_ts.assert_called_once() # 确保只调用了一次底层函数

class TestPartitionUtil:
    """测试 partition_util 中的工具函数"""

    def test_get_odps_partition_modified_timestamp_unreachable(self):
        """测试 ODPS 不可达时的 fallback 逻辑"""
        from column_io.dataset.partition_util import get_odps_partition_modified_timestamp
        import time
        
        current_time = int(time.time())
        timestamp = get_odps_partition_modified_timestamp(
            is_odps_endpoint_reachable=False,
            access_id="id", access_key="key",
            project_name="p", table_name="t",
            partition_name="ds=1", odps_endpoint="http://test"
        )
        
        assert isinstance(timestamp, str)
        assert abs(int(timestamp) - current_time) < 5

class TestMetricStatus:
    """测试新增的状态码"""
    def test_new_status_codes(self):
        from column_io.dataset.metric_util import MetricStatus
        assert hasattr(MetricStatus, 'PATITION_MODIDIFIED')
        assert isinstance(MetricStatus.PATITION_MODIDIFIED, int)
        assert hasattr(MetricStatus, 'FORCE_RECREATE_SESSION')
        assert isinstance(MetricStatus.FORCE_RECREATE_SESSION, int)

class TestGetSessionCacheFromRemote:
    """测试 get_session_cache_from_remote 的主要分支"""

    def _build_session_and_job(self):
        """构造测试用的 session_struct 和 job_info（mock）"""
        from column_io.dataset.open_storage_utils import HashableSessionStruct

        session = HashableSessionStruct(
            project="p", table="t", logical_partition="ds=1",
            physical_partitions=["ds=1"], select_columns=["c1"]
        )
        # 固定 hash_id 与 partition_modified_timestamp，避免外部依赖
        session.get_hash_id = MagicMock(return_value="h" * 64)
        session.get_partition_modified_timestamp = MagicMock(
            return_value=str(int(__import__("time").time() * 1000) - 3600 * 1000)
        )

        job_info = MagicMock()
        job_info._task_id = "task-1"
        job_info._app_id = "app-1"
        job_info._rank = 0
        return session, job_info

    def _make_resp(self, json_data=None, raise_http=False, raise_json=False, text=""):
        """构造一个伪 requests.Response"""
        resp = MagicMock()
        resp.encoding = "utf-8"
        resp.apparent_encoding = "utf-8"
        resp.text = text
        if raise_http:
            import requests
            resp.raise_for_status.side_effect = requests.exceptions.HTTPError("http err")
        else:
            resp.raise_for_status.return_value = None
        if raise_json:
            import requests
            resp.json.side_effect = requests.exceptions.JSONDecodeError("e", "doc", 0)
        else:
            resp.json.return_value = json_data or {}
        return resp

    @patch("column_io.dataset.open_storage_utils.IS_NEBULA_OPEN_STORAGE_CACHE_SERVER_REACHABLE", False)
    def test_service_unreachable(self):
        from column_io.dataset.open_storage_utils import get_session_cache_from_remote
        from column_io.dataset.metric_util import MetricStatus

        session, job_info = self._build_session_and_job()
        status, msg = get_session_cache_from_remote(job_info, session, "row")

        assert status == MetricStatus.REQUEST_ERROR
        assert "not reachable" in msg

    @patch("column_io.dataset.open_storage_utils.IS_NEBULA_OPEN_STORAGE_CACHE_SERVER_REACHABLE", True)
    @patch("column_io.dataset.open_storage_utils.requests.get")
    def test_http_error(self, mock_get):
        from column_io.dataset.open_storage_utils import get_session_cache_from_remote
        from column_io.dataset.metric_util import MetricStatus

        mock_get.return_value = self._make_resp(raise_http=True)
        session, job_info = self._build_session_and_job()
        status, msg = get_session_cache_from_remote(job_info, session, "row")

        assert status == MetricStatus.REQUEST_ERROR
        assert "fail in http get status" in msg

    @patch("column_io.dataset.open_storage_utils.IS_NEBULA_OPEN_STORAGE_CACHE_SERVER_REACHABLE", True)
    @patch("column_io.dataset.open_storage_utils.requests.get")
    def test_request_exception(self, mock_get):
        import requests
        from column_io.dataset.open_storage_utils import get_session_cache_from_remote
        from column_io.dataset.metric_util import MetricStatus

        mock_get.side_effect = requests.exceptions.ConnectionError("conn refused")
        session, job_info = self._build_session_and_job()
        status, msg = get_session_cache_from_remote(job_info, session, "row")

        assert status == MetricStatus.REQUEST_ERROR
        assert "fail in http get conn" in msg

    @patch("column_io.dataset.open_storage_utils.IS_NEBULA_OPEN_STORAGE_CACHE_SERVER_REACHABLE", True)
    @patch("column_io.dataset.open_storage_utils.requests.get")
    def test_json_decode_error(self, mock_get):
        from column_io.dataset.open_storage_utils import get_session_cache_from_remote
        from column_io.dataset.metric_util import MetricStatus

        mock_get.return_value = self._make_resp(raise_json=True, text="<html>bad</html>")
        session, job_info = self._build_session_and_job()
        status, msg = get_session_cache_from_remote(job_info, session, "row")

        assert status == MetricStatus.JSON_ERROR
        assert "fail in get json parsing" in msg

    @patch("column_io.dataset.open_storage_utils.IS_NEBULA_OPEN_STORAGE_CACHE_SERVER_REACHABLE", True)
    @patch("column_io.dataset.open_storage_utils.requests.get")
    def test_resp_field_missing(self, mock_get):
        from column_io.dataset.open_storage_utils import get_session_cache_from_remote
        from column_io.dataset.metric_util import MetricStatus

        # is_ok=False
        mock_get.return_value = self._make_resp(json_data={"is_ok": False})
        session, job_info = self._build_session_and_job()
        status, msg = get_session_cache_from_remote(job_info, session, "row")

        assert status == MetricStatus.FIELD_ERROR
        assert "is_ok or session_id False" in msg

    @patch("column_io.dataset.open_storage_utils.IS_NEBULA_OPEN_STORAGE_CACHE_SERVER_REACHABLE", True)
    @patch("column_io.dataset.open_storage_utils.requests.get")
    def test_session_expired(self, mock_get):
        import time
        from column_io.dataset.open_storage_utils import get_session_cache_from_remote
        from column_io.dataset.metric_util import MetricStatus

        # expiration_time 给一个已过期的 ms 时间戳
        expired_ms = int(time.time() * 1000) - 10 * 1000
        mock_get.return_value = self._make_resp(json_data={
            "is_ok": True,
            "session_id": "sid-001",
            "expiration_time": expired_ms,
        })
        session, job_info = self._build_session_and_job()
        status, msg = get_session_cache_from_remote(job_info, session, "row")

        assert status == MetricStatus.OUTDATE_ERROR
        assert "session_id expired" in msg

    @patch("column_io.dataset.open_storage_utils.is_session_creator", return_value=False)
    @patch("column_io.dataset.open_storage_utils.IS_NEBULA_OPEN_STORAGE_CACHE_SERVER_REACHABLE", True)
    @patch("column_io.dataset.open_storage_utils.requests.get")
    def test_waiting_for_creator(self, mock_get, _mock_is_creator):
        import time
        from column_io.dataset.open_storage_utils import get_session_cache_from_remote
        from column_io.dataset.metric_util import MetricStatus

        valid_ms = int(time.time() * 1000) + 7 * 24 * 3600 * 1000
        mock_get.return_value = self._make_resp(json_data={
            "is_ok": True,
            "session_id": "sid-001",
            "expiration_time": valid_ms,
            "session_create": int(time.time()) - 3600,
            "app_id": "other-app",  # 与 job_info._app_id="app-1" 不一致
        })
        session, job_info = self._build_session_and_job()
        status, msg = get_session_cache_from_remote(job_info, session, "row")

        assert status == MetricStatus.WAITING
        assert "wait session creator" in msg

    @patch("column_io.dataset.open_storage_utils.is_session_creator", return_value=True)
    @patch("column_io.dataset.open_storage_utils.IS_NEBULA_OPEN_STORAGE_CACHE_SERVER_REACHABLE", True)
    @patch("column_io.dataset.open_storage_utils.requests.get")
    def test_partition_modified(self, mock_get, _mock_is_creator):
        import time
        from column_io.dataset.open_storage_utils import get_session_cache_from_remote
        from column_io.dataset.metric_util import MetricStatus

        now = int(time.time())
        valid_ms = (now + 7 * 24 * 3600) * 1000
        # session_create 较早，partition_modified_timestamp 较新
        session, job_info = self._build_session_and_job()
        session.get_partition_modified_timestamp = MagicMock(return_value=str(now))

        mock_get.return_value = self._make_resp(json_data={
            "is_ok": True,
            "session_id": "sid-001",
            "expiration_time": valid_ms,
            "session_create": now - 3600,         # 早于分区修改时间
            "app_id": "other-app",                # 不同于 job_info._app_id
        })
        status, msg = get_session_cache_from_remote(job_info, session, "row")

        assert status == MetricStatus.PATITION_MODIDIFIED
        assert "odps partition modified" in msg

    @patch("column_io.dataset.open_storage_utils.force_recreate_session", return_value=True)
    @patch("column_io.dataset.open_storage_utils.is_session_creator", return_value=True)
    @patch("column_io.dataset.open_storage_utils.IS_NEBULA_OPEN_STORAGE_CACHE_SERVER_REACHABLE", True)
    @patch("column_io.dataset.open_storage_utils.requests.get")
    def test_force_recreate_session(self, mock_get, _mock_is_creator, _mock_force):
        import time
        from column_io.dataset.open_storage_utils import get_session_cache_from_remote
        from column_io.dataset.metric_util import MetricStatus

        now = int(time.time())
        valid_ms = (now + 7 * 24 * 3600) * 1000
        session, job_info = self._build_session_and_job()
        # 让分区时间戳早于 session_create，绕过 PATITION_MODIDIFIED 分支
        session.get_partition_modified_timestamp = MagicMock(return_value=str(now - 7200))

        mock_get.return_value = self._make_resp(json_data={
            "is_ok": True,
            "session_id": "sid-001",
            "expiration_time": valid_ms,
            "session_create": now - 3600,
            "app_id": "other-app",
        })
        status, msg = get_session_cache_from_remote(job_info, session, "row")

        assert status == MetricStatus.FORCE_RECREATE_SESSION
        assert "force_recreate_session" in msg

    @patch("column_io.dataset.open_storage_utils.force_recreate_session", return_value=False)
    @patch("column_io.dataset.open_storage_utils.is_session_creator", return_value=True)
    @patch("column_io.dataset.open_storage_utils.IS_NEBULA_OPEN_STORAGE_CACHE_SERVER_REACHABLE", True)
    @patch("column_io.dataset.open_storage_utils.requests.get")
    def test_success(self, mock_get, _mock_is_creator, _mock_force):
        import time
        from column_io.dataset.open_storage_utils import (
            get_session_cache_from_remote, MemSessionCache4Refresh,
        )
        from column_io.dataset.metric_util import MetricStatus

        now = int(time.time())
        valid_ms = (now + 7 * 24 * 3600) * 1000
        session, job_info = self._build_session_and_job()
        session.get_partition_modified_timestamp = MagicMock(return_value=str(now - 7200))

        mock_get.return_value = self._make_resp(json_data={
            "is_ok": True,
            "session_id": "sid-success",
            "expiration_time": valid_ms,
            "session_create": now - 3600,
            "app_id": job_info._app_id,  # 同一个 app
        })
        status, session_id = get_session_cache_from_remote(job_info, session, "row")

        assert status == MetricStatus.SUCCESS
        assert session_id == "sid-success"
        # 写入了内存缓存
        assert session in MemSessionCache4Refresh
        assert MemSessionCache4Refresh[session].session_id == "sid-success"

    @patch("column_io.dataset.open_storage_utils.force_recreate_session", return_value=False)
    @patch("column_io.dataset.open_storage_utils.is_session_creator", return_value=True)
    @patch("column_io.dataset.open_storage_utils.IS_NEBULA_OPEN_STORAGE_CACHE_SERVER_REACHABLE", True)
    @patch("column_io.dataset.open_storage_utils.requests.get")
    def test_success_with_second_timestamp(self, mock_get, _mock_is_creator, _mock_force):
        """expiration_time 是秒级时间戳时，函数会自动 *1000，仍应判定为有效"""
        import time
        from column_io.dataset.open_storage_utils import get_session_cache_from_remote
        from column_io.dataset.metric_util import MetricStatus

        now = int(time.time())
        valid_sec = now + 7 * 24 * 3600  # 秒级
        session, job_info = self._build_session_and_job()
        session.get_partition_modified_timestamp = MagicMock(return_value=str(now - 7200))

        mock_get.return_value = self._make_resp(json_data={
            "is_ok": True,
            "session_id": "sid-sec",
            "expiration_time": valid_sec,
            "session_create": now - 3600,
            "app_id": job_info._app_id,
        })
        status, session_id = get_session_cache_from_remote(job_info, session, "row")

        assert status == MetricStatus.SUCCESS
        assert session_id == "sid-sec"