import os

from recis.info import is_internal_enabled


if is_internal_enabled():
    X_CLUSTER_TUNNEL_ENDPOINT = "http://dt.xcluster.odps.aliyun-inc.com"
    DOMESTIC_ODPS_ENDPOINT = "http://service.odps.aliyun-inc.com/api"


def is_external_cluster():
    """Check if running in an external cluster environment.

    Returns:
        bool: True if running in external cluster, False otherwise.
    """
    is_external = os.environ.get("IS_EXTERNAL_CLUSTER", None)
    return str(is_external).lower() == "true"
