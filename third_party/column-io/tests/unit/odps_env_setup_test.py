# tests/unit/odps_env_setup_test.py
# -*- coding: utf8 -*-
from unittest.mock import MagicMock, patch, call
import os,sys
import pytest

# ── 阻断 C 扩展导入链 ──────────────────────────────────────
# py_interface 是 C 扩展，在纯 Python UT 环境中不存在。
# 必须在 patch() 触发 import column_io.dataset.odps_env_setup 之前，
# 将整条依赖链中涉及 C 扩展的模块注入 sys.modules。
_mock_py_interface = MagicMock()
_mock_interface = MagicMock()
sys.modules.setdefault("column_io.lib.py_interface", _mock_py_interface)
sys.modules.setdefault("column_io.lib.interface", _mock_interface)


MODULE_PATH = "column_io.dataset.odps_env_setup"
MOCK_PATHS = ["odps://test_project/tables/test_table/ds=20251201"]
MOCK_ENV = {
    "NOTEBOOK_CONTAINER": "0",
    "ENCODED_ODPS_ACCESS_ID": "",
    "ENCODED_ODPS_ACCESS_KEY": "",
    "access_id": "mock_access_id",
    "access_key": "mock_access_key",
}
DEMO_PATCH = {
    f"{MODULE_PATH}._odps_table_path_parse": MagicMock(
        return_value=[("test_project", "test_table", "ds=20251201")]
    ),
    f"{MODULE_PATH}.is_column_io": MagicMock(return_value=True),
    f"{MODULE_PATH}.get_app_config": MagicMock(return_value=({}, None)),
    f"{MODULE_PATH}.IS_ODPS_ENDPOINT_REACHABLE": False,
    f"{MODULE_PATH}.dump_debug_access_info_batch": MagicMock(),
    f"{MODULE_PATH}.check_auth_and_data": MagicMock(),
    f"{MODULE_PATH}.get_and_register_read_session_list": MagicMock(return_value={1, 2}),
    f"{MODULE_PATH}.create_and_post_read_session_list": MagicMock(),
    f"{MODULE_PATH}.try_start_refresh_session_thread": MagicMock(),
    f"{MODULE_PATH}.try_start_refresh_halo_metrics": MagicMock(),
    f"{MODULE_PATH}.is_session_creator": MagicMock(return_value=True),
    f"{MODULE_PATH}.is_notebook": MagicMock(return_value=False),
    f"{MODULE_PATH}.decode": MagicMock(side_effect=lambda x: x),
}

@patch.dict(os.environ, MOCK_ENV, clear=False)
def test_if_create_session_when_setting_params_true():
    """验证 init_odps_open_storage_session 接口的 is_create_session 参数生效逻辑."""

    with patch(f"{MODULE_PATH}._odps_table_path_parse", return_value=[("test_project", "test_table", "ds=20251201")]), \
         patch(f"{MODULE_PATH}.is_column_io", return_value=False), \
         patch(f"{MODULE_PATH}.get_app_config", return_value=({}, None)), \
         patch(f"{MODULE_PATH}.IS_ODPS_ENDPOINT_REACHABLE", False), \
         patch(f"{MODULE_PATH}.dump_debug_access_info_batch"), \
         patch(f"{MODULE_PATH}.check_auth_and_data"), \
         patch(f"{MODULE_PATH}.get_and_register_read_session_list", return_value={1, 2}) as mock_get_session , \
         patch(f"{MODULE_PATH}.create_and_post_read_session_list") as mock_create_session, \
         patch(f"{MODULE_PATH}.try_start_refresh_session_thread") as mock_refresh_thread, \
         patch(f"{MODULE_PATH}.try_start_refresh_halo_metrics") as mock_refresh_halo, \
         patch(f"{MODULE_PATH}.is_notebook", return_value=False), \
         patch(f"{MODULE_PATH}.decode", side_effect=lambda x: x):

        from column_io.dataset.odps_env_setup import init_odps_open_storage_session
        from column_io.dataset.open_storage_utils import is_session_creator

        init_odps_open_storage_session(
            paths=MOCK_PATHS,
            is_create_session = True,
        )

        # is_create_session=True 覆盖了 is_session_creator() 的返回值
        assert is_session_creator() == True, "is_create_session=True should override is_session_creator to return True"

        # 不论哪种模式哪种角色 都首先调用 get 方法
        mock_get_session.assert_called_once()
        
        # is_create_session = True 且 mock_get_session 返回非空待鉴权列表时, 必须主动create session
        mock_create_session.assert_called_once()

        # NOTEBOOK 模式下不维护鉴权, 否则需要启动后台线程维护 session 有效期
        mock_refresh_thread.assert_called_once()
