import datetime
import os
import shutil
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
)
from packaging import version

from recis.framework.checkpoint_manager import ExtraFields, Saver, SaverOptions
from recis.framework.filesystem import get_file_system
from recis.nn.modules.hashtable import HashTable, filter_out_sparse_param, gen_slice
from recis.optim.named_optimizer import wrapped_named_optimizer
from recis.optim.sparse_adamw_tf import SparseAdamWTF
from recis.utils.logger import Logger


logger = Logger(__name__)


class NormalModel(nn.Module):
    def __init__(self, shard_idx=0, shard_num=1):
        super().__init__()
        self.shard_idx = shard_idx
        self.shard_num = shard_num
        self.table_1 = HashTable(
            [1024], name="table_n_1", slice=gen_slice(shard_idx, shard_num)
        )
        self.table_2 = HashTable(
            [1024], name="table_n_2", slice=gen_slice(shard_idx, shard_num)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 1),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.dense1(self.table_1(x) + self.table_2(x)) + self.dense2(
            self.table_2(x)
        )


class LargeModel(nn.Module):
    def __init__(self, shard_idx=0, shard_num=1):
        super().__init__()
        self.shard_idx = shard_idx
        self.shard_num = shard_num
        self.table_1 = HashTable(
            [1024], name="table_h_1", slice=gen_slice(shard_idx, shard_num)
        )
        self.table_2 = HashTable(
            [1024], name="table_h_2", slice=gen_slice(shard_idx, shard_num)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 1),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 1),
        )
        self.dense3 = nn.Linear(1, 1)

    def forward(self, x):
        return self.dense1(self.table_1(x) + self.table_2(x)) + self.dense3(
            self.dense2(self.table_2(x))
        )


class SmallModel(nn.Module):
    def __init__(self, shard_idx=0, shard_num=1):
        super().__init__()
        self.shard_idx = shard_idx
        self.shard_num = shard_num
        self.table_1 = HashTable(
            [1024], name="table_s_1", slice=gen_slice(shard_idx, shard_num)
        )
        self.table_2 = HashTable(
            [1024], name="table_s_2", slice=gen_slice(shard_idx, shard_num)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.dense1(self.table_1(x) + self.table_2(x))


class DiffModel(nn.Module):
    def __init__(self, shard_idx=0, shard_num=1):
        super().__init__()
        self.shard_idx = shard_idx
        self.shard_num = shard_num
        self.table_1 = HashTable(
            [1024], name="table_d_1", slice=gen_slice(shard_idx, shard_num)
        )
        self.table_2 = HashTable(
            [1024], name="table_d_2", slice=gen_slice(shard_idx, shard_num)
        )

        self.dense1 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Linear(256, 1),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(1024, 784),
            nn.Linear(784, 1),
        )

    def forward(self, x):
        return self.dense1(self.table_1(x) + self.table_2(x)) + self.dense2(
            self.table_2(x)
        )


def save_model(model, sparse_optim, dense_optim, tmpdir, ckpt_id):
    saver_option = SaverOptions(
        model,
        sparse_optim,
        "",
        None,
        20,
        1,
        None,
    )
    epoch = torch.scalar_tensor(0, dtype=torch.int64).cuda()
    global_step = torch.scalar_tensor(0, dtype=torch.int64).cuda()

    saver = Saver(saver_option)
    saver.register_for_checkpointing(ExtraFields.recis_dense_optim, dense_optim)
    saver.register_for_checkpointing(ExtraFields.train_epoch, epoch)
    saver.register_for_checkpointing(ExtraFields.global_step, global_step)

    epoch.add_(1)
    global_step.add_(1)
    sparse_optim.zero_grad()
    dense_optim.zero_grad()
    ids = torch.arange(100)
    emb = model(ids)
    loss = torch.sum(emb)
    loss.backward()
    sparse_optim.step()
    dense_optim.step()
    saver.output_dir = tmpdir
    saver.save(ckpt_id=ckpt_id)


class Test(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.tmpdir):
            shutil.rmtree(cls.tmpdir)

    @classmethod
    def setUpClass(cls):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=1800))
        cls.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs, init_kwargs],
            gradient_accumulation_steps=1,
        )

        normal_model = NormalModel(shard_idx=0, shard_num=1)
        cls.normal_model = normal_model.to("cuda")
        cls.sparse_param = filter_out_sparse_param(cls.normal_model)
        cls.sparse_optim = SparseAdamWTF(cls.sparse_param, lr=0.001)

        large_model = LargeModel(shard_idx=0, shard_num=1)
        cls.large_model = large_model.to("cuda")
        cls.sparse_param = filter_out_sparse_param(cls.large_model)
        cls.sparse_optim = SparseAdamWTF(cls.sparse_param, lr=0.001)

        small_model = SmallModel(shard_idx=0, shard_num=1)
        cls.small_model = small_model.to("cuda")
        cls.sparse_param = filter_out_sparse_param(cls.small_model)
        cls.sparse_optim = SparseAdamWTF(cls.sparse_param, lr=0.001)

        diff_model = DiffModel(shard_idx=0, shard_num=1)
        cls.diff_model = diff_model.to("cuda")
        cls.sparse_param = filter_out_sparse_param(cls.diff_model)
        cls.sparse_optim = SparseAdamWTF(cls.sparse_param, lr=0.001)

        cls.normal_model = cls.accelerator.prepare(cls.normal_model)
        cls.large_model = cls.accelerator.prepare(cls.large_model)
        cls.small_model = cls.accelerator.prepare(cls.small_model)
        cls.diff_model = cls.accelerator.prepare(cls.diff_model)

        cls.tmpdir = tempfile.mkdtemp()

        dense_optim_1 = wrapped_named_optimizer(optim.AdamW)(
            cls.normal_model.named_parameters()
        )
        dense_optim_1 = cls.accelerator.prepare(dense_optim_1)
        save_model(
            cls.normal_model, cls.sparse_optim, dense_optim_1, cls.tmpdir, "ckpt_1"
        )

        dense_optim_2 = optim.AdamW(cls.normal_model.parameters())
        dense_optim_2 = cls.accelerator.prepare(dense_optim_2)
        save_model(
            cls.normal_model, cls.sparse_optim, dense_optim_2, cls.tmpdir, "ckpt_2"
        )

    def _check_optim_by_extra(self, optimizer, path):
        fs = get_file_system(os.path.join(path, "extra.pt"))
        with fs.open(os.path.join(path, "extra.pt"), "rb") as f:
            extra_data = torch.load(f, weights_only=False)
        tmp_state_dict = optimizer.state_dict()

        optim_key = (
            ExtraFields.recis_dense_optim
            if ExtraFields.recis_dense_optim in extra_data
            else ExtraFields.prev_optim
        )

        for top_key, value in extra_data[optim_key]["state"].items():
            real_key = top_key
            if top_key not in tmp_state_dict["state"]:
                real_key = optimizer.optimizer.index_param_map.get(top_key, "")
            if real_key == "":
                continue

            for key, val in value.items():
                self.assertTrue(
                    torch.allclose(
                        val,
                        tmp_state_dict["state"][real_key][key],
                    )
                )

        self.assertEqual(
            len(tmp_state_dict["param_groups"]),
            len(extra_data[optim_key]["param_groups"]),
        )

        for group1, group2 in zip(
            tmp_state_dict["param_groups"], extra_data[optim_key]["param_groups"]
        ):
            for key in group1:
                if key == "params" or key == "param_names":
                    continue
                self.assertEqual(
                    group1[key],
                    group2[key],
                )

    def _load_model(self, model, sparse_optim, dense_optim, model_bank, ckpt_path):
        saver_option = SaverOptions(
            model,
            sparse_optim,
            "",
            None,
            20,
            1,
            None,
        )
        epoch = torch.scalar_tensor(0, dtype=torch.int64).cuda()
        global_step = torch.scalar_tensor(0, dtype=torch.int64).cuda()
        saver = Saver(saver_option)
        saver.register_for_checkpointing(ExtraFields.recis_dense_optim, dense_optim)
        saver.register_for_checkpointing(ExtraFields.train_epoch, epoch)
        saver.register_for_checkpointing(ExtraFields.global_step, global_step)

        ckpt_1 = os.path.join(self.tmpdir, ckpt_path)
        logger.info("        ||")
        logger.info("        ||")
        logger.info("        ||")
        logger.info("        ||")
        logger.info("        \\/  ")
        saver._init_model_bank(model_bank)
        saver.restore()
        self._check_optim_by_extra(dense_optim, ckpt_1)

        sparse_optim.zero_grad()
        dense_optim.zero_grad()
        ids = torch.arange(100)
        emb = model(ids)
        loss = torch.sum(emb)
        loss.backward()
        sparse_optim.step()
        dense_optim.step()

    @unittest.skipIf(
        version.parse(torch.__version__) < version.parse("2.6.0"),
        "Requires PyTorch >= 2.6.0",
    )
    def test_normal_normal(self):
        dense_optim_2 = wrapped_named_optimizer(optim.AdamW)(
            self.normal_model.named_parameters()
        )
        dense_optim_2 = self.accelerator.prepare(dense_optim_2)
        dense_optim_2.named_optimizer = True

        model_bank = [
            {
                "path": os.path.join(self.tmpdir, "ckpt_1"),
                "load": ["*"],
                "exclude": ["io_state"],
                "is_dynamic": False,
            }
        ]
        self._load_model(
            self.normal_model, self.sparse_optim, dense_optim_2, model_bank, "ckpt_1"
        )

        model_bank = [
            {
                "path": os.path.join(self.tmpdir, "ckpt_2"),
                "load": ["*"],
                "exclude": ["io_state"],
                "is_dynamic": False,
            }
        ]
        self._load_model(
            self.normal_model, self.sparse_optim, dense_optim_2, model_bank, "ckpt_2"
        )

    @unittest.skipIf(
        version.parse(torch.__version__) < version.parse("2.6.0"),
        "Requires PyTorch >= 2.6.0",
    )
    def test_normal_large(self):
        dense_optim_2 = wrapped_named_optimizer(optim.AdamW)(
            self.large_model.named_parameters()
        )
        dense_optim_2 = self.accelerator.prepare(dense_optim_2)
        dense_optim_2.named_optimizer = True

        model_bank = [
            {
                "path": os.path.join(self.tmpdir, "ckpt_1"),
                "load": ["*"],
                "exclude": ["io_state"],
                "oname": [
                    {"table_h_1*": "table_n_1*"},
                    {"table_h_2*": "table_n_2*"},
                ],
                "is_dynamic": False,
            }
        ]
        self._load_model(
            self.large_model, self.sparse_optim, dense_optim_2, model_bank, "ckpt_1"
        )

        model_bank = [
            {
                "path": os.path.join(self.tmpdir, "ckpt_2"),
                "load": ["*"],
                "exclude": ["io_state"],
                "oname": [
                    {"table_h_1*": "table_n_1*"},
                    {"table_h_2*": "table_n_2*"},
                ],
                "is_dynamic": False,
            }
        ]
        self._load_model(
            self.large_model, self.sparse_optim, dense_optim_2, model_bank, "ckpt_2"
        )

    @unittest.skipIf(
        version.parse(torch.__version__) < version.parse("2.6.0"),
        "Requires PyTorch >= 2.6.0",
    )
    def test_normal_small(self):
        dense_optim_2 = wrapped_named_optimizer(optim.AdamW)(
            self.small_model.named_parameters()
        )
        dense_optim_2 = self.accelerator.prepare(dense_optim_2)
        dense_optim_2.named_optimizer = True

        model_bank = [
            {
                "path": os.path.join(self.tmpdir, "ckpt_1"),
                "load": ["*"],
                "exclude": ["io_state"],
                "oname": [
                    {"table_s_1*": "table_n_1*"},
                    {"table_s_2*": "table_n_2*"},
                ],
                "is_dynamic": False,
            }
        ]
        self._load_model(
            self.small_model, self.sparse_optim, dense_optim_2, model_bank, "ckpt_1"
        )

        model_bank = [
            {
                "path": os.path.join(self.tmpdir, "ckpt_2"),
                "load": ["*"],
                "exclude": ["io_state"],
                "oname": [
                    {"table_s_1*": "table_n_1*"},
                    {"table_s_2*": "table_n_2*"},
                ],
                "is_dynamic": False,
            }
        ]
        self._load_model(
            self.small_model, self.sparse_optim, dense_optim_2, model_bank, "ckpt_2"
        )

    @unittest.skipIf(
        version.parse(torch.__version__) < version.parse("2.6.0"),
        "Requires PyTorch >= 2.6.0",
    )
    def test_normal_diff(self):
        dense_optim_2 = wrapped_named_optimizer(optim.AdamW)(
            self.diff_model.named_parameters()
        )
        dense_optim_2 = self.accelerator.prepare(dense_optim_2)
        dense_optim_2.named_optimizer = True

        model_bank = [
            {
                "path": os.path.join(self.tmpdir, "ckpt_1"),
                "load": ["*"],
                "exclude": ["io_state"],
                "is_dynamic": False,
            }
        ]
        with self.assertRaises(RuntimeError) as cm:
            self._load_model(
                self.diff_model, self.sparse_optim, dense_optim_2, model_bank, "ckpt_1"
            )
        self.assertIn("size mismatch", str(cm.exception))

        model_bank = [
            {
                "path": os.path.join(self.tmpdir, "ckpt_2"),
                "load": ["*"],
                "exclude": ["io_state"],
                "is_dynamic": False,
            }
        ]
        with self.assertRaises(RuntimeError) as cm:
            self._load_model(
                self.diff_model, self.sparse_optim, dense_optim_2, model_bank, "ckpt_2"
            )
        self.assertIn("size mismatch", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
