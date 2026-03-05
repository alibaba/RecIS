import os
import re
import time
from typing import Optional

from openlm_hub import add_or_update_ckpt_metrics

from recis.framework.metrics import get_mos_metrics
from recis.utils.logger import Logger


if not os.environ.get("BUILD_DOCUMENT", None) == "1":
    from openlm_hub.openlm_api import (
        add_or_sync_mos_ckpt,
        delete_ckpt,
        get_mos_ckpt_info,
        get_or_create_mos_version,
    )

logger = Logger(__name__)


def format_physical_path(path: str):
    prefix = ""
    local_path = path
    if path.startswith("xpfs://"):
        # "xpfs://xxx/data/.../" -> "/data/.../"
        index = path.find("/data/")
        prefix, local_path = path[:index], path[index:]
    return prefix, local_path


def parse_uri(uri, auto_version=False):
    """Parse and validate MOS URI format.

    This function parses a MOS URI string and extracts its components including
    project, model name, version, and checkpoint ID. It validates the URI format
    and optionally generates automatic version timestamps.

    Args:
        uri (str): MOS URI string in format:
            - "model.project.model_name" (model level)
            - "model.project.model_name/version=version_id" (version level)
            - "model.project.model_name/version=version_id/ckpt_id=checkpoint_id" (checkpoint level)
        auto_version (bool, optional): If True, automatically generates a timestamp-based
            version when URI is at model level. Defaults to False.

    Returns:
        dict: Dictionary containing parsed URI components:
            - uri (str): Original or modified URI
            - project (str): Project name
            - model_name (str): Model name
            - version (str or None): Version identifier
            - ckpt_id (str or None): Checkpoint identifier
            - level (str): URI level ('model', 'version', or 'ckpt_id')

    Raises:
        ValueError: If URI format is invalid.

    Example:
        >>> # Parse model-level URI
        >>> info = parse_uri("model.recommendation.user_embedding")
        >>> print(info["level"])  # 'model'
        >>> print(info["project"])  # 'recommendation'
        >>> # Parse version-level URI
        >>> info = parse_uri("model.recommendation.user_embedding/version=v1.0")
        >>> print(info["level"])  # 'version'
        >>> print(info["version"])  # 'v1.0'
        >>> # Parse checkpoint-level URI
        >>> uri = "model.recommendation.user_embedding/version=v1.0/ckpt_id=step_1000"
        >>> info = parse_uri(uri)
        >>> print(info["level"])  # 'ckpt_id'
        >>> print(info["ckpt_id"])  # 'step_1000'
        >>> # Auto-generate version
        >>> info = parse_uri("model.recommendation.user_embedding", auto_version=True)
        >>> print(info["version"])  # '20231201120000' (timestamp)
    """
    pattern = r"model\.\S+\.\S+(/version=\S+)?(/ckpt_id=\S+)?"
    if not re.match(pattern, uri):
        raise ValueError(f"[MOS] uri is invalid : {uri}")
    uri_partition = uri.split("/")
    level = "model"
    project, model_name = uri_partition[0].split(".")[1:]
    version = None
    ckpt_id = None
    if len(uri_partition) > 1:
        version = uri_partition[1].replace("version=", "")
        level = "version"
    if len(uri_partition) > 2:
        ckpt_id = uri_partition[2].replace("ckpt_id=", "")
        level = "ckpt_id"
    # version will be automatically generated when auto_version is True
    if level == "model" and auto_version:
        version = time.strftime("%Y%m%d%H%M%S", time.localtime())
        uri = os.path.join(uri, f"version={version}")
    return dict(
        uri=uri,
        project=project,
        model_name=model_name,
        version=version,
        ckpt_id=ckpt_id,
        level=level,
    )


def render_uri_to_model_bank_path(uri, user_id):
    """Convert MOS URI to physical model bank path.

    This function resolves a MOS URI to its corresponding physical path in the
    model bank storage system. It handles version and checkpoint level URIs,
    with support for latest checkpoint resolution.

    Args:
        uri (str): MOS URI string (version or checkpoint level).
        user_id (str): User ID for MOS authentication.

    Returns:
        tuple: A tuple containing:
            - physical_path (str): Physical file system path to the model/checkpoint
            - full_uri (str): Complete URI including resolved checkpoint information

    Raises:
        ValueError: If URI is model-level only, version/checkpoint doesn't exist,
            or no checkpoints are found.

    Example:
        >>> user_id = "user123"
        >>> uri = "model.project.model_name/version=v1.0"
        >>> path, full_uri = render_uri_to_model_bank_path(uri, user_id)
        >>> print(path)  # '/path/to/model/storage/v1.0'
        >>>  specific checkpoint
        >>> uri = "model.project.model_name/version=v1.0/ckpt_id=step_1000"
        >>> path, full_uri = render_uri_to_model_bank_path(uri, user_id)
        >>> print(path)  # '/path/to/model/storage/v1.0/step_1000'
    """
    uri_info = parse_uri(uri, auto_version=False)
    uri = uri_info["uri"]
    full_uri = None
    physical_path = None
    if uri_info["level"] == "model":
        raise ValueError("[XDL_MOS] mos model_bank path not support only model")
    if uri_info["level"] == "version":
        uri = os.path.join(uri, "ckpt_id=$max_pt")
    ckpt = get_mos_ckpt_info(user_id=user_id, mos_ckpt_uri=uri)
    physical_path = ckpt.physical_path
    full_uri = ckpt.uri
    return physical_path, full_uri


def render_uri_to_output_dir(uri):
    """Convert MOS URI to output directory path for model saving.

    This function renders a MOS URI to an output directory path suitable for
    saving model checkpoints. It automatically generates a version if not present
    and creates or updates the corresponding MOS version entry.

    Args:
        uri (str): MOS URI string. If version is not specified, one will be
            automatically generated using current timestamp.

    Returns:
        str: Physical path to the output directory where models can be saved.

    Raises:
        ValueError: If URI specifies checkpoint ID (not allowed for output dirs)
            or if output directory cannot be obtained.

    Example:
        >>> # URI without version (auto-generated)
        >>> uri = "model.project.model_name"
        >>> output_dir = render_uri_to_output_dir(uri)
        >>> print(output_dir)  # '/path/to/output/20231201120000'
        >>> # URI with specific version
        >>> uri = "model.project.model_name/version=v1.0"
        >>> output_dir = render_uri_to_output_dir(uri)
        >>> print(output_dir)  # '/path/to/output/v1.0'

    Note:
        - Automatically retries MOS operations up to 3 times on failure
        - Uses CALCULATE_CLUSTER environment variable for cluster specification
        - Creates or updates MOS version entry as needed
    """
    uri_info = parse_uri(uri, auto_version=True)
    user_id = os.environ.get("USER_ID", None)
    if uri_info["level"] == "ckpt_id":
        # output dir should be a directory, so configuring ckpt id is not allowed
        raise ValueError("[MOS] nebula_mos not support ckpt_id")

    version_info = get_or_create_mos_version(user_id, uri_info["uri"])
    real_physical_path = version_info.physical_path
    if len(real_physical_path) == 0:
        raise ValueError(f"[MOS] can not get output_dir by uri : {uri}")
    return real_physical_path


class Mos:
    """MOS client wrapper for model management.

    This class provides a high-level interface for interacting with
    Mos system. It handles URI parsing, path resolution, and
    checkpoint management operations.

    The Mos class supports both model bank access (for loading existing models)
    and output directory creation (for saving new models), with automatic
    version management and checkpoint operations.

    Args:
        uri (str): MOS URI string specifying the model, version, and optionally checkpoint.
        model_bank_path (bool, optional): If True, resolve to model bank path for
            loading. If False, resolve to output directory for saving. Defaults to False.

    Attributes:
        uri (str): Original MOS URI
        uri_info (dict): Parsed URI components
        version_uri (str): Version-level URI for checkpoint operations
        user_id (str): User ID from USER_ID environment variable
        real_physical_path (str): Resolved physical file system path

    Raises:
        ValueError: If URI doesn't start with "model." or is otherwise invalid.

    Example:
        >>> import os
        >>> os.environ["USER_ID"] = "user123"
        >>> # Create MOS client for saving models
        >>> mos_saver = Mos("model.project.model_name/version=v1.0")
        >>> save_path = mos_saver.real_physical_path
        >>> print(f"Save models to: {save_path}")
        >>> # Update checkpoint after training
        >>> mos_saver.ckpt_update("epoch_10", "/local/path/to/checkpoint")
        >>> # Create MOS client for loading models
        >>> mos_loader = Mos(
        ...     "model.project.model_name/version=v1.0", model_bank_path=True
        ... )
        >>> load_path = mos_loader.real_physical_path
        >>> print(f"Load models from: {load_path}")
        >>> # Delete a checkpoint
        >>> mos_saver.ckpt_update("old_checkpoint", "", is_delete=True)

    Note:
        - Requires USER_ID environment variable to be set
        - Automatically handles version creation for output directories
        - Supports retry logic for robust checkpoint operations
    """

    def __init__(self, uri, model_bank_path=False):
        """Initialize MOS client with URI and path resolution mode.

        Args:
            uri (str): MOS URI string.
            model_bank_path (bool, optional): Path resolution mode. Defaults to False.

        Raises:
            ValueError: If URI format is invalid.
        """
        if not uri.startswith("model."):
            raise ValueError(f"[MOS] uri is invalid : {uri}")
        self.uri = uri
        self.uri_info = parse_uri(uri)
        self.version_uri = "model.{}.{}/version={}".format(
            self.uri_info["project"],
            self.uri_info["model_name"],
            self.uri_info["version"],
        )
        self.user_id = os.environ.get("USER_ID", None)
        if model_bank_path:
            self.real_physical_path, _ = render_uri_to_model_bank_path(
                uri, self.user_id
            )
        else:
            self.real_physical_path = render_uri_to_output_dir(uri)
        self.storage_prefix, self.real_physical_path = format_physical_path(
            self.real_physical_path
        )
        self.last_ckpt_id = self.uri_info["ckpt_id"]

    def ckpt_update(
        self,
        ckpt_id,
        path,
        is_delete=False,
        label_key: Optional[str] = None,
        label_value: Optional[str] = None,
    ):
        """Create, update, or delete a model checkpoint.

        This method manages individual checkpoints within a model version,
        supporting creation, updates, and deletion operations with automatic
        retry logic for reliability.

        Args:
            ckpt_id (str): Unique identifier for the checkpoint.
            path (str): Local file system path to the checkpoint data.
                Ignored when is_delete=True.
            is_delete (bool, optional): If True, delete the checkpoint.
                If False, create or update it. Defaults to False.
            label_key (str): Key for the label when saving to MOS. Defaults to None.
            label_value (str): Value for the label when saving to MOS. Defaults to None.
        Raises:
            Exception: If checkpoint operation fails after 3 retry attempts.

        Example:
            >>> mos_client = Mos("model.project.model_name/version=v1.0")
            >>> # Create/update checkpoint
            >>> mos_client.ckpt_update("epoch_5", "/path/to/checkpoint_epoch_5")
            >>> mos_client.ckpt_update("best_model", "/path/to/best_checkpoint")
            >>> # Delete checkpoint
            >>> mos_client.ckpt_update("old_checkpoint", "", is_delete=True)

        Note:
            - Automatically retries operations up to 3 times with 5-second delays
            - Logs checkpoint deletion operations as warnings
            - Uses the version_uri for checkpoint operations
        """
        path = self.storage_prefix + path
        ckpt_labels = []
        if label_key is not None and label_value is not None:
            assert isinstance(label_key, str) and isinstance(label_value, str)
            assert not is_delete, "[MOS] label can not be set when ckpt is deleted"
            logger.info(
                f"[MOS] set label '{label_key}={label_value}' for ckpt_id={ckpt_id}"
            )
            ckpt_labels.append(f"{label_key}={label_value}")

        ckpt_uri = f"{self.version_uri}/ckpt_id={ckpt_id}"
        if is_delete:
            delete_ckpt(mos_ckpt_uri=ckpt_uri, user_id=self.user_id)
            logger.warning(f"[MOS] delete ckpt_id = {ckpt_id}")
        else:
            add_or_sync_mos_ckpt(
                user_id=self.user_id,
                mos_ckpt_uri=ckpt_uri,
                physical_path=path,
                ckpt_labels=ckpt_labels,
            )
        self.last_ckpt_id = ckpt_id
        self.report_mos_metrics(is_train=True)

    def report_mos_metrics(self, is_train=True):
        mos_metrics = get_mos_metrics()
        mos_metrics["ckpt_id"] = self.last_ckpt_id
        mode = "MODE.TRAIN" if is_train else "MODE.EVAL"
        report_data = {}
        for k, v in mos_metrics.items():
            report_data[f"{mode}.{k}"] = f"{v}"
        ckpt_uri = f"{self.version_uri}/ckpt_id={self.last_ckpt_id}"
        add_or_update_ckpt_metrics(
            mos_ckpt_uri=ckpt_uri, metrics=report_data, user_id=self.user_id
        )
        logger.info(f"Report mos metrics to uri: {ckpt_uri}")
        logger.warning(f"Report mos metrics: {report_data}")
