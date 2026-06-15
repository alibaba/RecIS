from typing import Optional

from recis.framework.metrics import get_mos_metrics
from recis.utils.logger import Logger


logger = Logger(__name__)

try:
    from openlm_hub import MosCkptFileManager, add_or_update_ckpt_metrics, delete_ckpt
    from openlm_hub.constants import CkptAction
    from openlm_hub.error import MosCkptNotFoundError
    from openlm_hub.utils.storage import get_ckpt_access_path

    OPENLM_HUB_CKPT_AVAILABLE = True
except ImportError:
    OPENLM_HUB_CKPT_AVAILABLE = False
    try:
        import openlm_hub  # noqa: F401

        logger.warning(
            "请尽快升级: 'openlm-hub>=0.1.13'! "
            "当前openlm_hub 版本过低, 不支持标准 ckpt 读写分离模式, "
            "将退回老协议. 老协议在多云环境下ckpt同步有问题, "
            "写pangu易受到存储水位影响! "
        )
    except ImportError:
        pass

MOS_URI_PREFIX = "model"


class OpenlmHubHelper:
    """标准 openlm_hub 用法的统一入口.

    把 MosCkptFileManager / delete_ckpt / get_ckpt_access_path 等
    openlm_hub 直接调用集中在这里, checkpoint_manager 只通过本类操作 ckpt.
    """

    def __init__(self, version_uri: str, user_id: Optional[str]):
        self.version_uri = version_uri
        self.user_id = user_id
        # 记 ckpt_id -> save 时拿到的 WRITE 路径.
        # save 完成那一刻写入, max_keep 删旧 ckpt 时按 ckpt_id 查 WRITE 路径来 rm.
        # 避免再用 MosCkptFileManager(mode='r') 拿 READ 路径 (只读挂载) rm 报
        # EROFS (Read-Only File System) 错误.
        self._ckpt_path_by_id = {}

    def get_save_context(self, ckpt_id: str):
        """创建 ckpt 写入上下文, 含 EROFS WRITE 路径修正.

        Returns:
            tuple: (ckpt_file_manager, ckpt_path, fs)
        """
        cfm = MosCkptFileManager(
            f"{self.version_uri}/ckpt_id={ckpt_id}", mode="w"
        )
        # MosCkptFileManager bug 兜底: 已注册 ckpt 即使 mode='w' 也返 READ 路径,
        # r/w 分离存储下是只读 mount, 写会 EROFS. 强制改写为 WRITE 路径.
        cfm.path = get_ckpt_access_path(
            cfm.ckpt_physical_path, CkptAction.WRITE
        )
        return cfm, cfm.path, cfm.get_fs()

    def resolve_load_path(self, ckpt_id: Optional[str] = None) -> Optional[str]:
        """标准 openlm_hub 用法, 通过 openlm_hub 解析 ckpt 读取路径.
        ckpt_id=None 取最新, 找不到返 None."""
        mos_uri = (
            self.version_uri
            if ckpt_id is None
            else f"{self.version_uri}/ckpt_id={ckpt_id}"
        )
        try:
            cfm = MosCkptFileManager(mos_uri, mode="r")
            return cfm.path
        except MosCkptNotFoundError:
            logger.warning(f"MOS ckpt not found: {mos_uri}")
            return None

    def resolve_latest_resume(self):
        """查 MOS 最新 ckpt, 返回 (resume_path, ckpt_physical_path).

        用于 auto-resume: 把最新 ckpt 路径注入 model_bank 实现断点续训.
        没有已注册 ckpt 时返回 None.
        """
        try:
            cfm = MosCkptFileManager(self.version_uri, mode="r")
            return cfm.path, cfm.ckpt_physical_path
        except MosCkptNotFoundError:
            return None

    def register_and_report(self, ckpt_file_manager, ckpt_id: str, labels=None):
        """注册 ckpt 到 MOS 并上报 metrics."""
        ckpt_file_manager.register_ckpt(labels=labels or [])
        mos_metrics = get_mos_metrics()
        mos_metrics["ckpt_id"] = ckpt_id
        report_data = {}
        for k, v in mos_metrics.items():
            report_data[f"MODE.TRAIN.{k}"] = f"{v}"
        ckpt_uri = f"{self.version_uri}/ckpt_id={ckpt_id}"
        add_or_update_ckpt_metrics(
            mos_ckpt_uri=ckpt_uri, metrics=report_data, user_id=self.user_id
        )
        logger.info(f"Report mos metrics to uri: {ckpt_uri}")
        logger.warning(f"Report mos metrics: {report_data}")

    def delete(self, ckpt_id: str):
        """从 MOS 注销已注册的 ckpt."""
        # URI 必须和 register_ckpt 时一致, 否则 MOS 找不到记录
        delete_ckpt(
            mos_ckpt_uri=f"{self.version_uri}/ckpt_id={ckpt_id}",
            user_id=self.user_id,
        )

    def cache_write_path(self, ckpt_id: str, path: str):
        self._ckpt_path_by_id[ckpt_id] = path

    def pop_write_path(self, ckpt_id: str) -> Optional[str]:
        return self._ckpt_path_by_id.pop(ckpt_id, None)
