import os
import random
import shutil
import unittest
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from recis.framework.checkpoint_manager import ExtraFields, Saver, SaverOptions
from recis.nn.modules.hashtable import HashTable, filter_out_sparse_param
from recis.optim.sparse_adagrad import SparseAdagrad
from recis.serialize.checkpoint_reader import CheckpointReader


def set_all_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        from torch.backends import flags_frozen

        if not flags_frozen():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def compare_snapshot(
    mapping1: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    mapping2: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    ids1, indices1, global_embs1, global_states1 = mapping1
    ids2, indices2, global_embs2, global_states2 = mapping2

    if ids1.size(0) != ids2.size(0):
        return False
    if ids1.size(0) == 0:
        return True
    sorted_ids1, sort_idx1 = torch.sort(ids1)
    sorted_ids2, sort_idx2 = torch.sort(ids2)
    if not torch.equal(sorted_ids1, sorted_ids2):
        return False
    emb_check = torch.allclose(
        global_embs1[sort_idx1],
        global_embs2[sort_idx2],
        rtol=rtol,
        atol=atol,
    )
    state_check = torch.allclose(
        global_states1[sort_idx1],
        global_states2[sort_idx2],
        rtol=rtol,
        atol=atol,
    )
    return emb_check and state_check


SEED = 42


class RecisModel(torch.nn.Module):
    def __init__(self, emb_size, num_classes, name, device) -> None:
        super().__init__()
        self._emb = HashTable(
            [emb_size], block_size=1024, device=torch.device(device), name=name
        )
        self.classifier = nn.Linear(
            emb_size, num_classes, bias=False, dtype=torch.float32, device=device
        )
        self.device = device
        self.loss_ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, ids, torch=False):
        embeds = self._emb(ids)
        embeds = embeds.to(self.device)
        logits = self.classifier(embeds)
        return embeds, logits

    def loss(self, logits, labels):
        return self.loss_ce(logits, labels)

    def ids_embeddings(self):
        return self._emb.ids_embeddings()

    def gather_embs(self, ids):
        return self._emb(ids)

    @property
    def recis_emb(self):
        return self._emb

    @property
    def linear(self):
        return self.classifier


class TorchModel(torch.nn.Module):
    def __init__(self, emb_size, num_classes, device) -> None:
        super().__init__()
        self._torch_emb = nn.Embedding(
            num_embeddings=100000,
            embedding_dim=emb_size,
            sparse=True,
            device=device,
        )
        self._torch_emb.weight.data.fill_(0.0)
        set_all_seeds(SEED)
        self.classifier = nn.Linear(
            emb_size, num_classes, bias=False, dtype=torch.float32, device=device
        )
        self.loss_ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, ids):
        embeds = self._torch_emb(ids)
        logits = self.classifier(embeds)
        return embeds, logits

    @property
    def torch_emb(self):
        return self._torch_emb

    @property
    def linear(self):
        return self.classifier

    def loss(self, logits, labels):
        return self.loss_ce(logits, labels)


class Test(unittest.TestCase):
    DEVICE = None
    CKPT_DIR = None

    @classmethod
    def setUpClass(cls):
        cls.DEVICE = os.getenv("TEST_DEVICE", "cpu")
        cls.CKPT_DIR = os.getenv("CKPT_DIR", "./tmpdir")
        os.makedirs(cls.CKPT_DIR, exist_ok=True)

    def setUp(self):
        set_all_seeds(SEED)
        self.lr = 0.01
        self.lr_decay = 0.001
        self.initial_accumulator_value = 0.03
        self.emb_size = 53
        self.num_classes = 20
        self.recis_model = RecisModel(
            emb_size=self.emb_size,
            num_classes=self.num_classes,
            name=f"test_{self._testMethodName}",
            device=self.DEVICE,
        )
        self.recis_model = self.recis_model.train()
        recis_sparse_param = filter_out_sparse_param(self.recis_model)
        self.recis_sparse_optim = SparseAdagrad(
            param_dict=recis_sparse_param,
            lr=self.lr,
            lr_decay=self.lr_decay,
            initial_accumulator_value=self.initial_accumulator_value,
        )

        self.torch_model = TorchModel(
            emb_size=self.emb_size, num_classes=self.num_classes, device=self.DEVICE
        )
        self.torch_model = self.torch_model.train()
        self.torch_optim = torch.optim.Adagrad(
            self.torch_model.torch_emb.parameters(),
            lr_decay=self.lr_decay,
            initial_accumulator_value=self.initial_accumulator_value,
        )

        # check linear
        self.assertTrue(
            torch.equal(self.recis_model.linear.weight, self.torch_model.linear.weight)
        )

        self.labels = torch.randint(low=0, high=20, size=(100000,), device=self.DEVICE)
        self.ids = [
            torch.arange(start=0, end=3, step=1, device=self.DEVICE),
            torch.arange(start=1, end=10, step=1, device=self.DEVICE),
            torch.randperm(100, device=self.DEVICE)[:5],
        ]
        for i in range(20):
            self.ids.append(torch.randperm(100000 - i, device=self.DEVICE)[: 1000 + i])
        self.steps = 100

        saver_option = SaverOptions(
            self.recis_model,
            self.recis_sparse_optim,
            self.CKPT_DIR,
            None,
            20,
            1,
            None,
        )
        self.saver = Saver(saver_option)
        self.global_step = torch.scalar_tensor(self.steps, dtype=torch.int64).cuda()
        self.saver.register_for_checkpointing(ExtraFields.global_step, self.global_step)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.CKPT_DIR)

    def test_sparse_adagrad(self):
        for step in range(self.steps):
            ids = self.ids[step % len(self.ids)]
            er, logits = self.recis_model(ids)
            target_labels = self.labels[ids]
            loss = self.recis_model.loss(logits, target_labels)
            loss.backward()
            self.recis_sparse_optim.step()
            self.recis_sparse_optim.zero_grad()

        self.recis_model = self.recis_model.eval()
        for step in range(self.steps):
            ids = self.ids[step % len(self.ids)]
            er, logits = self.recis_model(ids)

        for step in range(self.steps):
            ids = self.ids[step % len(self.ids)]
            et, logits = self.torch_model(ids)
            target_labels = self.labels[ids]
            loss = self.torch_model.loss(logits, target_labels)
            loss.backward()
            self.torch_optim.step()
            self.torch_optim.zero_grad()

        recis_ids, recis_index, recis_emb = self.recis_model.recis_emb.snap_shot()
        torch_valid_emb = self.torch_model.torch_emb(recis_ids)
        self.assertTrue(torch.allclose(recis_emb.cuda(), torch_valid_emb.cuda()))
        recis_state_sum_slot = (
            self.recis_model.recis_emb.slot_group()
            .slot_by_name("sparse_adagrad_state_sum")
            .value()
        )
        recis_state_sum = recis_state_sum_slot[recis_index]

        torch_state_sum = self.torch_optim.state[self.torch_model.torch_emb.weight][
            "sum"
        ]
        torch_state_sum = torch_state_sum[recis_ids]

        self.assertTrue(torch.allclose(recis_state_sum, torch_state_sum))

        ## Test Optimizer Save and Load
        self.saver.save(ckpt_id=f"ckpt_{self.global_step.item()}")
        save_ids, save_index, save_emb = self.recis_model.recis_emb.snap_shot()
        save_state_sum = (
            self.recis_model.recis_emb.slot_group()
            .slot_by_name("sparse_adagrad_state_sum")
            .value()[save_index]
        )

        self.recis_model.recis_emb.clear()

        # load and check sparse model and optimizer state
        ckpt_path = f"./{self.CKPT_DIR}/ckpt_{self.global_step.item()}"
        simple_bank = [
            {
                "path": ckpt_path,
                "load": ["*"],
                "exclude": ["io_state"],
                "is_dynamic": False,
            }
        ]
        self.saver._init_model_bank(simple_bank)
        self.recis_sparse_optim.reset_state_dict()
        self.saver.restore()

        reader = CheckpointReader(ckpt_path)
        load_sparse_adagrad_step = reader.read_tensor("sparse_adagrad_step")
        save_sparse_adagrad_step = self.recis_sparse_optim.state_dict()[
            "sparse_adagrad_step"
        ]
        self.assertTrue(
            torch.equal(
                load_sparse_adagrad_step.cuda(), save_sparse_adagrad_step.cuda()
            )
        )

        load_ids, load_index, load_emb = self.recis_model.recis_emb.snap_shot()
        load_state_sum = (
            self.recis_model.recis_emb.slot_group()
            .slot_by_name("sparse_adagrad_state_sum")
            .value()[load_index]
        )
        self.assertTrue(
            compare_snapshot(
                (save_ids, save_index, save_emb, save_state_sum),
                (load_ids, load_index, load_emb, load_state_sum),
            )
        )

    def test_weight_decay_uses_adjusted_gradient(self):
        weight_decay = 0.1
        model = RecisModel(
            emb_size=self.emb_size,
            num_classes=self.num_classes,
            name="test_weight_decay",
            device=self.DEVICE,
        ).train()
        optimizer = SparseAdagrad(
            param_dict=filter_out_sparse_param(model),
            lr=self.lr,
            lr_decay=self.lr_decay,
            initial_accumulator_value=self.initial_accumulator_value,
            weight_decay=weight_decay,
            save_update_info_interval=1,
        )
        ids = self.ids[1]

        for _ in range(2):
            _, logits = model(ids)
            loss = model.loss(logits, self.labels[ids])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        update_info = optimizer.get_step_update_info()

        def value_with_suffix(suffix):
            return next(
                value for key, value in update_info.items() if key.endswith(suffix)
            )

        state_sum_before = value_with_suffix("_updated_state_sum")
        params_before = value_with_suffix("_updated_params_before")
        params_after = value_with_suffix("_updated_params_after")
        raw_grad = value_with_suffix("_updated_grad")
        saved_step = value_with_suffix("_last_saved_step").item()
        adjusted_grad = raw_grad + weight_decay * params_before
        state_sum_after = state_sum_before + adjusted_grad.square()
        effective_lr = self.lr / (1 + (saved_step - 1) * self.lr_decay)
        expected_params_after = params_before - effective_lr * adjusted_grad / (
            state_sum_after.sqrt() + optimizer._eps
        )

        torch.testing.assert_close(params_after, expected_params_after)


if __name__ == "__main__":
    unittest.main()
