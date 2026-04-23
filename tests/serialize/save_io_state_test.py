import os
import shutil
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.testing._internal.common_utils as common
from torch.utils.data import IterableDataset

from recis.framework.checkpoint_manager import ExtraFields, Saver, SaverOptions
from recis.io.dataset_base import DatasetBase
from recis.nn.modules.embedding import EmbeddingOption
from recis.nn.modules.embedding_engine import EmbeddingEngine
from recis.nn.modules.hashtable import filter_out_sparse_param
from recis.optim.named_optimizer import wrapped_named_optimizer
from recis.optim.sparse_adamw_tf import SparseAdamWTF


EMB_DIM = 64
TOTAL_STEPS = 20
PARTIAL_STEPS = 7
# 目标总步数：续训结束后 global_step 须等于 train_step
train_step = TOTAL_STEPS


class DemoModel(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        emb_options = {
            "table_1": EmbeddingOption(
                embedding_dim=EMB_DIM,
                shared_name="table_1",
                combiner="sum",
                coalesced=True,
                device=device,
            ),
            "table_2": EmbeddingOption(
                embedding_dim=EMB_DIM,
                shared_name="table_2",
                combiner="sum",
                coalesced=True,
                device=device,
            ),
        }
        self.embedding_engine = EmbeddingEngine(emb_options)
        self.dense1 = nn.Sequential(
            nn.Linear(EMB_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(EMB_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
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


class _EmbSeqIter:
    def __init__(
        self,
        total_steps: int,
        batch: int,
        seq_len: int,
        vocab: int,
        device: torch.device,
        seed: int,
    ):
        self._total_steps = total_steps
        self._batch = batch
        self._seq_len = seq_len
        self._vocab = vocab
        self._device = device
        self._seed = seed
        self._step = 0

    def serialize(self):
        return {
            "step": self._step,
            "total": self._total_steps,
            "batch": self._batch,
            "seq_len": self._seq_len,
            "vocab": self._vocab,
            "seed": self._seed,
        }

    def deserialize(self, state: dict) -> None:
        self._step = int(state["step"])
        self._total_steps = int(state["total"])
        self._batch = int(state["batch"])
        self._seq_len = int(state["seq_len"])
        self._vocab = int(state["vocab"])
        self._seed = int(state["seed"])

    def __next__(self):
        if self._step >= self._total_steps:
            raise StopIteration
        x = torch.randint(
            0,
            self._vocab,
            (self._batch, self._seq_len),
            device=self._device,
        )
        y = torch.randint(
            0,
            self._vocab,
            (self._batch, self._seq_len),
            device=self._device,
        )
        self._step += 1
        return x, y


class EmbSeqIterableDataset(DatasetBase):
    def __init__(
        self,
        total_steps: int,
        batch: int,
        seq_len: int,
        vocab: int,
        device: torch.device,
        seed: int = 20250414,
    ):
        io_device = "cuda" if device.type == "cuda" else "cpu"
        super().__init__(
            batch_size=batch,
            device=io_device,
            save_interval=1,
            worker_num=1,
            worker_idx=0,
        )
        self._total_steps = total_steps
        self._torch_device = device
        self._seq_len = seq_len
        self._vocab = vocab
        self._seed = seed

    def _shard_path(self, sub_id, sub_num) -> None:
        return

    def _build_dataset(self):
        self._shard_paths = []
        sub_id, sub_num = self._get_sub_info()
        self._shard_path(sub_id, sub_num)
        inner = _EmbSeqTorchDataset(self)
        self._dataset = self._create_state_dataset(inner, sub_id, sub_num)
        self._state_dataset_flush_handle = self._dataset


class _EmbSeqTorchDataset(IterableDataset):
    def __init__(self, outer: EmbSeqIterableDataset):
        self._outer = outer

    def __iter__(self):
        o = self._outer
        return _EmbSeqIter(
            o._total_steps,
            o._batch_size,
            o._seq_len,
            o._vocab,
            o._torch_device,
            o._seed,
        )


class DummyEvalIo:
    _worker_num = 1

    def _flush_io_state(self) -> None:
        return

    def dump_io_state(self):
        return {}

    def load_io_state(self, io_states) -> None:
        return


def _run_train_steps(model, sparse_optim, dense_optim, epoch, global_step, it, n: int):
    for _ in range(n):
        x, y = next(it)
        epoch.add_(1)
        global_step.add_(1)
        sparse_optim.zero_grad()
        dense_optim.zero_grad()
        loss = model(x, y).sum()
        loss.backward()
        sparse_optim.step()
        dense_optim.step()


class TestResumeCheckpointIo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("This demo expects CUDA (same as model_bank_test).")

        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(common.find_free_port()))
        dist.init_process_group(backend="nccl")

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def setUp(self):
        self.device = torch.device("cuda")
        self.tmpdir = tempfile.mkdtemp(prefix="recis_resume_demo_")

        model = DemoModel(self.device).to(self.device)
        sparse_param = filter_out_sparse_param(model)
        sparse_optim = SparseAdamWTF(param_dict=sparse_param, lr=0.001)
        dense_optim = wrapped_named_optimizer(torch.optim.AdamW)(
            model.named_parameters(), lr=0.001
        )
        epoch = torch.zeros((), dtype=torch.int64, device=self.device)
        global_step = torch.zeros((), dtype=torch.int64, device=self.device)

        saver_opt = SaverOptions(
            model=model,
            sparse_optim=sparse_optim,
            output_dir=self.tmpdir,
            model_bank=None,
            max_keep=5,
            concurrency=1,
        )
        saver = Saver(saver_opt)
        saver.register_for_checkpointing(ExtraFields.recis_dense_optim, dense_optim)
        saver.register_for_checkpointing(ExtraFields.train_epoch, epoch)
        saver.register_for_checkpointing(ExtraFields.global_step, global_step)

        train_ds = EmbSeqIterableDataset(
            TOTAL_STEPS, batch=4, seq_len=3, vocab=256, device=self.device
        )
        saver.register_io_state(ExtraFields.train_io, train_ds)
        saver.register_io_state(ExtraFields.eval_io, DummyEvalIo())

        self.model = model
        self.sparse_optim = sparse_optim
        self.dense_optim = dense_optim
        self.epoch = epoch
        self.global_step = global_step
        self.saver = saver
        self.train_ds = train_ds

    def tearDown(self):
        if hasattr(self, "tmpdir") and os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def test_resume_checkpoint_io(self):
        it = iter(self.train_ds)
        _run_train_steps(
            self.model,
            self.sparse_optim,
            self.dense_optim,
            self.epoch,
            self.global_step,
            it,
            PARTIAL_STEPS,
        )

        self.saver.save(ckpt_id="PARTIAL_STEPS")
        ckpt_dir = os.path.join(self.tmpdir, "PARTIAL_STEPS")

        self.assertEqual(int(self.global_step.item()), PARTIAL_STEPS)

        io_path = os.path.join(ckpt_dir, "io_state_0.pt")
        try:
            saved_io = torch.load(io_path, map_location="cpu", weights_only=False)
        except TypeError:
            saved_io = torch.load(io_path, map_location="cpu")

        self.assertEqual(
            saved_io["train_io"][0]["step"],
            PARTIAL_STEPS,
            f"checkpoint IO must reflect consumed batches, got {saved_io['train_io']!r}",
        )

        self.saver.output_dir = ""
        simple_bank = [
            {
                "path": ckpt_dir,
                "load": ["*"],
                "is_dynamic": False,
            }
        ]
        self.saver._init_model_bank(simple_bank)
        self.saver.restore()
        self.saver.output_dir = self.tmpdir

        self.assertEqual(int(self.global_step.item()), PARTIAL_STEPS)

        try:
            io_for_resume = torch.load(io_path, map_location="cpu", weights_only=False)
        except TypeError:
            io_for_resume = torch.load(io_path, map_location="cpu")
        resume_begin_step = int(io_for_resume["train_io"][0]["step"])
        self.assertEqual(resume_begin_step, PARTIAL_STEPS)
        self.assertEqual(resume_begin_step, int(self.global_step.item()))

        it2 = iter(self.train_ds)
        remaining = train_step - resume_begin_step
        _run_train_steps(
            self.model,
            self.sparse_optim,
            self.dense_optim,
            self.epoch,
            self.global_step,
            it2,
            remaining,
        )

        self.assertEqual(int(self.global_step.item()), train_step)

        self.saver.save(ckpt_id=f"ckpt_{self.global_step.item()}")
        ckpt_dir = os.path.join(self.tmpdir, f"ckpt_{self.global_step.item()}")
        io_path = os.path.join(ckpt_dir, "io_state_0.pt")
        try:
            saved_io = torch.load(io_path, map_location="cpu", weights_only=False)
        except TypeError:
            saved_io = torch.load(io_path, map_location="cpu")

        self.assertEqual(
            saved_io["train_io"][0]["step"],
            train_step,
            'checkpoint IO must reflect consumed batches, saved_io["train_io"][0]["step"] != train_step',
        )

        with self.assertRaises(StopIteration):
            next(it2)

        print(
            "resume OK:",
            f"partial={PARTIAL_STEPS}",
            f"train_step={train_step}",
            f"final_global_step={int(self.global_step.item())}",
            f"ckpt_dir={ckpt_dir}",
        )


if __name__ == "__main__":
    unittest.main()
