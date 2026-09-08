import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import torch.testing._internal.common_utils as common

from recis.framework.checkpoint_manager import ExtraFields, Saver, SaverOptions
from recis.framework.filesystem import get_file_system
from recis.framework.model_bank import load_pt_file
from recis.nn.modules.embedding import EmbeddingOption
from recis.nn.modules.embedding_engine import EmbeddingEngine
from recis.nn.modules.hashtable import filter_out_sparse_param
from recis.optim.sparse_adamw_tf import SparseAdamWTF
from recis.serialize.checkpoint_reader import CheckpointReader
from recis.utils.logger import Logger


logger = Logger(__name__)


class TestLoadPtFileDeviceSafety(unittest.TestCase):
    """Tests that model-bank inspection never restores tensors onto CUDA."""

    @patch("recis.framework.model_bank.torch.load")
    def test_pt_checkpoint_is_loaded_on_cpu(self, torch_load):
        expected = {"dense.weight": MagicMock()}
        torch_load.return_value = expected
        fs = MagicMock()
        fs.exists.side_effect = lambda path: path.endswith("model.pt")
        file_obj = MagicMock()
        fs.open.return_value.__enter__.return_value = file_obj

        data, from_pickle = load_pt_file("/checkpoint", "model", fs=fs)

        self.assertIs(data, expected)
        self.assertFalse(from_pickle)
        torch_load.assert_called_once_with(
            f=file_obj, map_location="cpu", weights_only=False
        )


class Model(torch.nn.Module):
    def __init__(self, shard_idx=0, shard_num=1):
        super().__init__()
        self.shard_idx = shard_idx
        self.shard_num = shard_num
        emb_options = {
            "table_1": EmbeddingOption(
                embedding_dim=1024,
                shared_name="table_1",
                combiner="sum",
                coalesced=True,
                device=torch.device("cuda"),
            ),
            "table_2": EmbeddingOption(
                embedding_dim=1024,
                shared_name="table_2",
                combiner="sum",
                coalesced=True,
                device=torch.device("cuda"),
            ),
        }

        self.embedding_engine = EmbeddingEngine(emb_options)

        self.dense1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 1),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 1),
        )

    def forward(self, x, y):
        embedding_results = self.embedding_engine(
            {
                "table_1": x,
                "table_2": y,
            }
        )

        table_1_emb = embedding_results["table_1"]
        table_2_emb = embedding_results["table_2"]

        return self.dense1(table_1_emb + table_2_emb) + self.dense2(table_2_emb)


class DenseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.densex = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 1),
        )
        self.densey = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.densex(x) + self.densey(x)


class TestModelBank(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(common.find_free_port())
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo"
        )

    @classmethod
    def tearDownClass(cls):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def setUp(self):
        # 创建临时目录
        self.tmpdir = tempfile.mkdtemp()

        self.model = Model(shard_idx=0, shard_num=1)
        self.model = self.model.to("cuda")
        self.sparse_param = filter_out_sparse_param(self.model)
        self.sparse_optim = SparseAdamWTF(param_dict=self.sparse_param, lr=0.001)
        self.dense_optim = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        print("in init", self.tmpdir)

        saver_option = SaverOptions(
            model=self.model,
            sparse_optim=self.sparse_optim,
            output_dir=self.tmpdir,
            model_bank=None,
            max_keep=20,
            concurrency=1,
            params_not_save=None,
            save_filter_fn=None,
        )
        self.epoch = torch.scalar_tensor(0, dtype=torch.int64).cuda()
        self.global_step = torch.scalar_tensor(0, dtype=torch.int64).cuda()
        self.saver = Saver(saver_option)
        self.saver.register_for_checkpointing(
            ExtraFields.recis_dense_optim, self.dense_optim
        )
        self.saver.register_for_checkpointing(ExtraFields.train_epoch, self.epoch)
        self.saver.register_for_checkpointing(ExtraFields.global_step, self.global_step)
        self._save_model()

    def tearDown(self):
        # 清理临时目录
        if hasattr(self, "tmpdir") and os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def _save_model(self):
        for _ in range(10):
            self.epoch.add_(1)
            self.global_step.add_(1)

            self.sparse_optim.zero_grad()
            self.dense_optim.zero_grad()
            emb = self.model(
                torch.arange(16, device="cuda").unsqueeze(1),
                torch.arange(16, device="cuda").unsqueeze(1),
            )
            loss = torch.sum(emb)
            loss.backward()
            self.sparse_optim.step()
            self.dense_optim.step()
            self.saver.save(ckpt_id=f"ckpt_{self.global_step.item()}")

    def _check_dense_oname(self, path):
        pt_file = os.path.join(path, "model.pt")
        fs = get_file_system(pt_file)
        with fs.open(pt_file, "rb") as f:
            state_dict = torch.load(f, weights_only=False)
        for name, param in self.model.named_parameters():
            if name.startswith("dense"):
                oname = ""
                if "dense1" in name:
                    oname = name.replace("dense1", "densex")
                elif "dense2" in name:
                    oname = name.replace("dense2", "densey")
                self.assertTrue(
                    torch.allclose(state_dict[oname].to("cuda"), param.to("cuda"))
                )

    def _check_dense_optim(self, path):
        fs = get_file_system(os.path.join(path, "extra.pt"))
        with fs.open(os.path.join(path, "extra.pt"), "rb") as f:
            extra_data = torch.load(f, weights_only=False)
        tmp_state_dict = self.dense_optim.state_dict()

        optim_key = (
            ExtraFields.recis_dense_optim
            if ExtraFields.recis_dense_optim in extra_data
            else ExtraFields.prev_optim
        )

        for cnt, value in enumerate(extra_data[optim_key]["state"].values()):
            for key, val in value.items():
                self.assertTrue(
                    torch.allclose(
                        val.to("cuda"),
                        tmp_state_dict["state"][cnt][key].to("cuda"),
                    )
                )

        self.assertEqual(
            tmp_state_dict["param_groups"],
            extra_data[optim_key]["param_groups"],
        )

    def _check_sparse_optim(self, path):
        reader = CheckpointReader(path)
        tensor_names = [
            "sparse_adamw_tf_beta2",
            "sparse_adamw_tf_beta1",
            "sparse_adamw_tf_step",
        ]
        for name in tensor_names:
            self.assertEqual(
                reader.read_tensor(name).to("cuda"),
                self.sparse_optim.state_dict()[name].to("cuda"),
            )

    def _check_sparse_oname(self, path, src_tables=None, dst_tables=None):
        if src_tables is None:
            src_tables = []
        if dst_tables is None:
            dst_tables = []
        reader = CheckpointReader(path)
        for src_table, dst_table in zip(src_tables, dst_tables):
            ht_name = self.model.embedding_engine._fea_to_ht[dst_table]
            dynamic_emb = self.model.embedding_engine._ht[ht_name]
            hashtable = dynamic_emb._hashtable

            hashtable_ids, index = hashtable.ids_map(child=dst_table)

            id_mask = (1 << 52) - 1
            decode_ids = torch.bitwise_and(hashtable_ids, id_mask)

            self.assertTrue(
                torch.allclose(
                    torch.sort(reader.read_tensor(f"{src_table}@id").to("cuda"))[0],
                    torch.sort(decode_ids.to("cuda"))[0],
                )
            )
            self.assertTrue(
                torch.allclose(
                    reader.read_tensor(f"{src_table}@embedding").to("cuda"),
                    hashtable.embeddings_map(child=dst_table)[1].to("cuda"),
                )
            )
            self.assertTrue(
                torch.allclose(
                    reader.read_tensor(f"{src_table}@sparse_adamw_tf_exp_avg").to(
                        "cuda"
                    ),
                    hashtable._hashtable_impl.slot_group()
                    .slot_by_name("sparse_adamw_tf_exp_avg")
                    .value()[index]
                    .to("cuda"),
                )
            )
            self.assertTrue(
                torch.allclose(
                    reader.read_tensor(f"{src_table}@sparse_adamw_tf_exp_avg_sq").to(
                        "cuda"
                    ),
                    hashtable._hashtable_impl.slot_group()
                    .slot_by_name("sparse_adamw_tf_exp_avg_sq")
                    .value()[index]
                    .to("cuda"),
                )
            )

    def _check_sparse_model(
        self,
        path,
        slot_names=(
            "table_1",
            "table_2",
        ),
        table_names=("table_1", "table_2"),
    ):
        reader = CheckpointReader(path)
        for table_name in table_names:
            ht_name = self.model.embedding_engine._fea_to_ht[table_name]
            dynamic_emb = self.model.embedding_engine._ht[ht_name]
            hashtable = dynamic_emb._hashtable
            hashtable_ids, index = hashtable.ids_map(child=table_name)
            id_mask = (1 << 52) - 1
            decode_ids = torch.bitwise_and(hashtable_ids, id_mask)
            self.assertTrue(
                torch.allclose(
                    torch.sort(reader.read_tensor(f"{table_name}@id").to("cuda"))[0],
                    torch.sort(decode_ids.to("cuda"))[0],
                )
            )

            _, embeddings = hashtable.embeddings_map(child=table_name)
            self.assertTrue(
                torch.allclose(
                    reader.read_tensor(f"{table_name}@embedding").to("cuda"),
                    embeddings.to("cuda"),
                )
            )

            for slot_name in slot_names:
                if slot_name != table_name:
                    continue
                self.assertTrue(
                    torch.allclose(
                        reader.read_tensor(f"{slot_name}@sparse_adamw_tf_exp_avg").to(
                            "cuda"
                        ),
                        hashtable._hashtable_impl.slot_group()
                        .slot_by_name("sparse_adamw_tf_exp_avg")
                        .value()[index]
                        .to("cuda"),
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        reader.read_tensor(
                            f"{slot_name}@sparse_adamw_tf_exp_avg_sq"
                        ).to("cuda"),
                        hashtable._hashtable_impl.slot_group()
                        .slot_by_name("sparse_adamw_tf_exp_avg_sq")
                        .value()[index]
                        .to("cuda"),
                    )
                )

    def _check_dense_model(self, path):
        pt_file = os.path.join(path, "model.pt")
        fs = get_file_system(pt_file)
        with fs.open(pt_file, "rb") as f:
            state_dict = torch.load(f, weights_only=False)
        for name, param in self.model.named_parameters():
            if name.startswith("dense"):
                self.assertTrue(
                    torch.allclose(state_dict[name].to("cuda"), param.to("cuda"))
                )

    def _run_dense_oname_model(self, path):
        if os.path.exists(os.path.join(path, "model.pt")):
            os.remove(os.path.join(path, "model.pt"))

        model = DenseModel().to("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        for _ in range(10):
            optimizer.zero_grad()
            x = torch.randn((16, 1024), device="cuda")
            y = model(x)
            loss = torch.sum(y)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), os.path.join(path, "model.pt"))

    def test_model_bank(self):
        # 使用 self.tmpdir 替代硬编码的路径
        ckpt_1 = os.path.join(self.tmpdir, "ckpt_1")
        ckpt_2 = os.path.join(self.tmpdir, "ckpt_2")
        ckpt_3 = os.path.join(self.tmpdir, "ckpt_3")
        ckpt_4 = os.path.join(self.tmpdir, "ckpt_4")
        ckpt_6 = os.path.join(self.tmpdir, "ckpt_6")
        ckpt_10 = os.path.join(self.tmpdir, "ckpt_10")

        self.saver.output_dir = ""

        # Test 1: Load all
        simple_bank = [
            {
                "path": ckpt_1,
                "load": ["*"],
                "exclude": ["io_state"],
                "is_dynamic": False,
            }
        ]
        self.saver._init_model_bank(simple_bank)
        self.saver.restore()
        self._check_sparse_model(ckpt_1)
        self._check_dense_model(ckpt_1)
        self._check_dense_optim(ckpt_1)
        self._check_sparse_optim(ckpt_1)

        # Test 2: Load only sparse
        sparse_bank = [
            {
                "path": ckpt_2,
                "load": ["table*"],
            }
        ]
        self.saver._init_model_bank(sparse_bank)
        self.saver.restore()
        self._check_sparse_model(ckpt_2)

        # Test 3: Load only dense
        dense_bank = [
            {
                "path": ckpt_4,
                "load": ["table_2*"],
            },
            {
                "path": ckpt_3,
                "load": ["dense*"],
            },
            {
                "path": ckpt_4,
                "load": ["table_1*"],
            },
        ]
        self.saver._init_model_bank(dense_bank)
        self.saver.restore()
        self._check_dense_model(ckpt_3)
        self._check_sparse_model(
            ckpt_4,
            slot_names=["table_1", "table_2"],
            table_names=["table_2", "table_1"],
        )

        # Test 5: Split load from different checkpoints
        split_bank = [
            {
                "path": ckpt_3,
                "load": ["*"],
                "exclude": ["table_1@*avg*", "io_state"],
            },
            {
                "path": ckpt_4,
                "load": ["table_1*"],
                "exclude": ["table_1@*avg*", "io_state"],
                "is_dynamic": False,
                "hashtable_clear": True,
            },
            {
                "path": ckpt_4,
                "load": ["table_2*"],
                "exclude": ["io_state"],
                "is_dynamic": False,
                "hashtable_clear": True,
            },
            {
                "path": ckpt_6,
                "load": ["dense*"],
                "exclude": ["io_state"],
            },
        ]
        self.saver._init_model_bank(split_bank)
        self.saver.restore()
        self._check_dense_model(ckpt_6)
        self._check_dense_optim(ckpt_6)
        self._check_sparse_model(
            ckpt_4, slot_names=["table_2"], table_names=["table_2"]
        )
        self._check_sparse_model(
            ckpt_4, slot_names=["table_none"], table_names=["table_1"]
        )
        self._check_sparse_optim(ckpt_3)

        # Test 6: is dynamic test
        dynamic_bank = [
            {
                "path": ckpt_10,
                "load": ["table*"],
                "exclude": ["io_state"],
                "is_dynamic": True,
                "ignore_error": True,
            },
        ]
        self.saver._init_model_bank(dynamic_bank)
        self.saver.restore()
        self._check_sparse_model(ckpt_10)

        # Test 7: oname test
        self._run_dense_oname_model(ckpt_10)
        oname_bank = [
            {
                "path": ckpt_10,
                "load": ["table_1*", "table_2*", "dense*"],
                "exclude": ["io_state"],
                "oname": [
                    {"table_1@id": "table_2@id"},
                    {"table_1@embedding": "table_2@embedding"},
                    {"table_2@id": "table_1@id"},
                    {"table_2@embedding": "table_1@embedding"},
                    {"dense1*": "densex*"},
                    {"dense2*": "densey*"},
                ],
            }
        ]
        self.saver._init_model_bank(oname_bank)
        self.saver.restore()
        self._check_sparse_oname(
            ckpt_10,
            src_tables=["table_1", "table_2"],
            dst_tables=["table_2", "table_1"],
        )
        self._check_dense_oname(ckpt_10)
        self._check_dense_optim(ckpt_10)


if __name__ == "__main__":
    unittest.main()
