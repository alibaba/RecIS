import os
from typing import Dict, Tuple

from recis.info import is_internal_enabled


if is_internal_enabled():
    X_CLUSTER_TUNNEL_ENDPOINT = "http://dt.xcluster.odps.aliyun-inc.com"
    DOMESTIC_ODPS_ENDPOINT = "http://service.odps.aliyun-inc.com/api"

    # Oversea or special external-tannei clusters mapping
    SPEC_CLUSTER_ODPS_ENDPOINT_MAPPING = {
        "asi_ap-northeast-2_gmt_core_01": "http://service.ali-ap-northeast-2.odps.aliyun-inc.com/api",
        "asi_ap-northeast-2_core_01": "http://service.ali-ap-northeast-2.odps.aliyun-inc.com/api",
    }

    def _get_current_cluster_name() -> str:
        """Get the name of the current cluster."""
        return os.environ.get("CALCULATE_CLUSTER", "")

    def is_external_cluster():
        """Check if running in an external cluster environment.

        Returns:
            bool: True if running in external cluster, False otherwise.
        """
        is_external = os.environ.get("IS_EXTERNAL_CLUSTER", None)
        return str(is_external).lower() == "true"

    def get_odps_access_info(config: Dict) -> Tuple[list, dict]:
        """
        Retrieves ODPS access parameters (Args & Kwargs).
        Priority Logic:
        1. [Highest] Special External-Tannei Clusters (e.g., asi_ap-northeast-2_*):
            Uses the regionalized Service Endpoint.
        2. [Medium] Standard External Clusters (IS_EXTERNAL_CLUSTER=true):
            Uses the generic Tunnel Endpoint.
        3. [Lowest] Default Internal Clusters:
            Uses the Endpoint specified in Config, or falls back to the domestic default Service Endpoint.
        Args:
            config (Dict): Contains access_id, access_key, project, end_point (optional).
        Returns:
            Tuple[list, dict]: (odps_args, odps_kwargs)
                - odps_args: [access_id, access_key, project]
                - odps_kwargs: {"endpoint": url} or {"tunnel_endpoint": url}
        """
        cluster_name = _get_current_cluster_name()
        is_normal_external = is_external_cluster()
        is_special_external_tannei = cluster_name in SPEC_CLUSTER_ODPS_ENDPOINT_MAPPING

        odps_args = [config["access_id"], config["access_key"], config["project"]]
        odps_kwargs = {}

        if is_special_external_tannei:
            primary_endpoint = (
                config.get("end_point")
                or SPEC_CLUSTER_ODPS_ENDPOINT_MAPPING[cluster_name]
            )
            odps_kwargs["endpoint"] = primary_endpoint
            need_create_table_or_partition = True
        elif is_normal_external:
            odps_kwargs["tunnel_endpoint"] = X_CLUSTER_TUNNEL_ENDPOINT
            need_create_table_or_partition = False
        else:
            final_endpoint = config.get("end_point") or DOMESTIC_ODPS_ENDPOINT
            odps_kwargs["endpoint"] = final_endpoint
            need_create_table_or_partition = True

        if not is_normal_external:
            user_provided_tunnel_ep = config.get("tunnel_endpoint")
            if user_provided_tunnel_ep:
                odps_kwargs["tunnel_endpoint"] = user_provided_tunnel_ep

        return odps_args, odps_kwargs, need_create_table_or_partition
