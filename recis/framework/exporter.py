import json
import os
import time

import torch
import torch.distributed as dist

from recis.framework.filesystem import get_file_system
from recis.info import is_internal_enabled
from recis.nn.modules.hashtable import filter_out_sparse_param
from recis.serialize import Loader, Saver
from recis.utils.logger import Logger
try:
    from recis.utils.torch_fx_tool.ExportTorchFxTool import ExportTorchFxTool
except ImportError:
    ExportTorchFxTool = None


if is_internal_enabled() and not os.environ.get("BUILD_DOCUMENT", None) == "1":
    from pangudfs_client.common.exception.exceptions import PanguException

    from recis.utils.mos import Mos
else:
    PanguException = None
    Mos = None

logger = Logger(__name__)
TMP_EXPORT_LOCAL_PATH = "./__tmp_export_path__/"


class Exporter:
    """Model exporter for RecIS framework with support for sparse and dense models.

    The Exporter class handles the export process for trained RecIS models,
    managing both sparse embedding tables and dense neural network components.
    It supports distributed export across multiple workers and handles various
    storage backends including local filesystem and cloud storage.

    Key Features:
        - Separate export of sparse and dense model components
        - Distributed export with automatic file partitioning
        - Support for multiple storage backends (local, cloud)
        - Configuration export for feature generation and model compilation
        - Automatic model preparation and state loading

    Attributes:
        rank (int): Current worker rank in distributed setup.
        shard_num (int): Total number of workers in distributed setup.
        model: Complete model containing both sparse and dense components.
        sparse_model: Sparse embedding component of the model.
        dense_model: Dense neural network component of the model.
        dataset: Dataset used for model tracing during export.
        ckpt_dir (str): Directory containing model checkpoints.
        export_dir (str): Target directory for exported model files.
        dense_optimizer: Optional optimizer for dense model components.
        export_model_name (str): Name for the exported model.
        export_outputs: Specification of model output nodes.
        filter_sparse_opt (bool): Whether to filter sparse optimization parameters.
        fg_conf (dict): Feature generation configuration.
        mc_conf (dict): Model compilation configuration.
        fx_tool (ExportTorchFxTool): Tool for exporting TorchFX models.
    """

    def __init__(
        self,
        model,
        sparse_model_name,
        dense_model_name,
        dataset,
        ckpt_dir,
        export_dir,
        dense_optimizer=None,
        export_folder_name="fx_user_model",
        export_model_name="user_model",
        export_outputs=None,
        fg=None,
        fg_conf_or_path=None,
        mc_conf_or_path=None,
        filter_sparse_opt=False,
        add_subfolder=False,
        need_export_sparse=True,
    ):
        """Initialize the model exporter with configuration parameters.

        Args:
            model: Complete RecIS model containing sparse and dense components.
            sparse_model_name (str): Name of the sparse model submodule.
            dense_model_name (str): Name of the dense model submodule.
            dataset: Dataset for model tracing during export process.
            ckpt_dir (str): Directory path containing model checkpoints.
            export_dir (str): Target directory for exported model files.
            dense_optimizer: Optional optimizer for dense model components.
            export_folder_name (str, optional): Name of the export folder.
                Defaults to "fx_user_model".
            export_model_name (str, optional): Name for the exported model.
                Defaults to "user_model".
            export_outputs: Specification of model output nodes for export.
            fg: Feature generator instance for configuration extraction.
            fg_conf_or_path: Feature generation configuration dict or file path.
            mc_conf_or_path: Model compilation configuration dict or file path.
            filter_sparse_opt (bool, optional): Whether to filter sparse
                optimization parameters. Defaults to False.

        Raises:
            AssertionError: If neither fg nor fg_conf_or_path is provided.
            AssertionError: If neither fg nor mc_conf_or_path is provided.
            AssertionError: If MOS is required but not available for model paths.
        """
        self.rank = int(os.environ.get("RANK", 0))
        self.shard_num = int(os.environ.get("WORLD_SIZE", 1))
        self.model = model
        self.sparse_model = model.get_submodule(sparse_model_name)
        self.sparse_model_name = sparse_model_name
        self.dense_model = model.get_submodule(dense_model_name)
        self.dense_model_name = dense_model_name
        self.dataset = dataset
        if ckpt_dir.startswith("model"):
            assert Mos is not None, "Cannot import mos, check interneal version."
            ckpt_dir = Mos(ckpt_dir, True).real_physical_path
        self.ckpt_dir = ckpt_dir
        if export_dir.startswith("model"):
            assert Mos is not None, "Cannot import mos, check interneal version."
            if add_subfolder:
                export_dir = Mos(export_dir, False).real_physical_path
            else:
                export_dir = Mos(export_dir, True).real_physical_path

        # add subfolder for export_dir
        if add_subfolder and not export_dir.endswith("data"):
            current_time = time.strftime("%Y%m%d", time.localtime(time.time()))
            current_time = current_time + "000000"
            export_dir = os.path.join(export_dir, current_time, "data")

        self.export_dir = export_dir

        self.dense_optimizer = dense_optimizer
        self.export_model_name = export_model_name
        self.export_outputs = export_outputs
        self.filter_sparse_opt = filter_sparse_opt
        self.need_export_sparse = need_export_sparse

        assert fg is not None or fg_conf_or_path is not None, (
            "one of fg or fg_config must be not None"
        )
        assert fg is not None or mc_conf_or_path is not None, (
            "one of fg or mc_config must be not None"
        )
        if fg_conf_or_path is not None:
            if not isinstance(fg_conf_or_path, dict):
                with open(fg_conf_or_path) as f:
                    fg_conf_or_path = json.load(f)
            self.fg_conf = fg_conf_or_path
        else:
            self.fg_conf = fg.get_fg_conf()
        if mc_conf_or_path is not None:
            if not isinstance(mc_conf_or_path, dict):
                with open(mc_conf_or_path) as f:
                    mc_conf_or_path = json.load(f)
            self.mc_conf = mc_conf_or_path
        else:
            self.mc_conf = fg.get_mc_conf()

        if ExportTorchFxTool is not None:
            self.fx_tool = ExportTorchFxTool(
                fx_folder=os.path.join(TMP_EXPORT_LOCAL_PATH, export_folder_name),
                model_name=self.export_model_name,
            )
            self.fx_tool.set_output_nodes_name(export_outputs)
        else:
            self.fx_tool = None

    def export(self):
        """Execute the complete model export process.

        This method orchestrates the entire export workflow, including model
        preparation, sparse component export, dense component export, and
        metadata export. The process is designed to work in distributed
        environments with automatic work partitioning.

        The export process consists of:
            1. Model preparation and checkpoint loading
            2. Sparse model component export
            3. Dense model component export with TorchFX
            4. Configuration metadata export

        Note:
            This method should be called on all workers in a distributed setup.
            File operations are automatically partitioned based on worker rank.
        """
        self.prepare_model()
        if self.need_export_sparse:
            self.export_sparse()
        self.export_dense()
        self.export_meta()

    def prepare_model(self):
        """Prepare the model for export by loading checkpoints and creating directories.

        This method handles the initial setup required for model export:
            - Creates the export directory if it doesn't exist
            - Loads dense model state from checkpoint files
            - Optionally loads sparse model parameters if filtering is enabled

        The method supports both local filesystem and cloud storage backends,
        automatically handling path resolution for different storage types.

        Raises:
            PanguException: If directory creation fails due to permission issues.
        """
        fs = get_file_system(self.export_dir)
        if not fs.exists(self.export_dir):
            try:
                fs.makedirs(self.export_dir + "/", exist_ok=True)
            except PanguException as e:
                if e.pangu_err_no == 7:
                    pass
        # load dense model
        pt_file = os.path.join(self.ckpt_dir, "model.pt")
        fs = get_file_system(pt_file)
        with fs.open(pt_file, "rb") as f:
            state_dict = torch.load(f=f)
        state_dict = {
            k.replace(f"{self.dense_model_name}.", ""): v for k, v in state_dict.items()
        }
        incompatible_keys = self.dense_model.load_state_dict(state_dict, strict=False)
        if incompatible_keys.missing_keys:
            logger.warning("⚠️ Missing keys (not loaded from checkpoint):")
            for key in incompatible_keys.missing_keys:
                logger.warning(f"  - {key}")
        # maybe load sparse model
        if self.need_export_sparse and self.filter_sparse_opt:
            sparse_params = filter_out_sparse_param(self.sparse_model)
            loader = Loader(self.ckpt_dir, hashtables=sparse_params, tensors={})
            loader.load()

    def export_dense(self):
        """Export the dense model component using TorchFX compilation.

        This method handles the export of the dense neural network component:
            1. Processes a sample batch through the sparse model to get dense inputs
            2. Uses TorchFX to trace and export the dense model
            3. Copies the exported model files to the target directory

        The dense model is exported in a format suitable for deployment,
        with optimizations applied through the TorchFX compilation process.
        Only the rank 0 worker performs the final file copying to avoid conflicts.

        Note:
            The method requires at least one batch from the dataset for model tracing.
            The sparse model must be properly initialized before calling this method.
        """
        # export dense to local tmp dir
        iterator = iter(self.dataset)
        stop_flag, data = next(iterator)
        dense_data = self.sparse_model(data)
        if self.dense_optimizer:
            self.dense_optimizer.zero_grad()
        if self.fx_tool is None:
            raise RuntimeError(
                "ExportTorchFxTool is not available in open-source version. "
                "Model export requires the internal torch_fx_tool submodule."
            )
        self.fx_tool.export_fx_model(self.dense_model, dense_data[0], self.mc_conf)
        dist.barrier()
        if self.rank == 0:
            # copy local tmp dir to dst dir
            fs = get_file_system(self.export_dir)
            fs.put(TMP_EXPORT_LOCAL_PATH, self.export_dir, recursive=True)

    def export_sparse(self):
        """Export the sparse model component with distributed file handling.

        This method manages the export of sparse embedding tables and related
        parameters. It supports two modes of operation:

        1. Direct file copying mode (filter_sparse_opt=False):
           - Identifies all sparse parameter files (safetensors, json)
           - Distributes files across workers for parallel copying
           - Each worker handles a subset of files based on rank

        2. Filtered export mode (filter_sparse_opt=True):
           - Extracts only relevant sparse parameters from the model
           - Uses the Saver class for optimized sparse parameter serialization
           - Automatically handles distributed saving across workers

        The method ensures efficient distribution of work across multiple workers
        while maintaining data consistency and avoiding file conflicts.

        Note:
            File distribution is based on worker rank to ensure balanced workload.
            All workers participate in the export process simultaneously.
        """
        if not self.filter_sparse_opt:
            # get all sparse files (ends with `safetensors` or `json`)
            candidate_files = []
            fs = get_file_system(self.ckpt_dir)
            for file_name in fs.ls(self.ckpt_dir, detail=False):
                if file_name.endswith(("safetensors", "json")):
                    candidate_files.append(file_name)
            file_to_copy = []
            # partition files
            for i, file_name in enumerate(candidate_files):
                if i % self.shard_num == self.rank:
                    file_to_copy.append(file_name)
            # copy
            for file_name in file_to_copy:
                fs.copy(file_name, self.export_dir)
        else:
            # save sparse files
            sparse_params = filter_out_sparse_param(self.sparse_model)
            saver = Saver(
                shard_index=self.rank,
                shard_num=self.shard_num,
                parallel=1,
                hashtables=sparse_params,
                tensors={},
                path=self.export_dir,
            )
            saver.save()

    def export_meta(self):
        """Export model metadata and configuration files.

        This method exports the configuration metadata required for model
        deployment and inference. Currently exports the feature generation
        configuration as a JSON file.

        The metadata includes:
            - Feature generation configuration (fg.json)
            - Model compilation settings (future extension)
            - Deployment-specific parameters (future extension)

        Only the rank 0 worker performs metadata export to avoid file conflicts
        and ensure consistency across the distributed export process.

        Note:
            Additional metadata types can be added by extending this method.
            The exported configurations must match the training setup exactly.
        """
        if self.rank == 0:
            fs = get_file_system(self.export_dir)
            # fg config
            with fs.open(os.path.join(self.export_dir, "fg.json"), "w") as out_f:
                json.dump(self.fg_conf, out_f)
