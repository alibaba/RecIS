import os
import random
import shutil
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.testing._internal.common_utils as common
from accelerate import Accelerator

from recis.nn.initializers import ConstantInitializer
from recis.nn.modules.embedding import DynamicEmbedding, EmbeddingOption
from recis.nn.modules.embedding_engine import EmbeddingEngine
from recis.nn.modules.hashtable import filter_out_sparse_param
from recis.optim.sparse_adamw_tf import SparseAdamWTF


def compare_pt_files(dir_path, prefix_a="1_", prefix_b="5_"):
    """Compare pt files with two different prefixes (same name after prefix).
    Returns dict: {name: max_abs_diff}.
    Dense files: load as tensor, compute max absolute difference.
    Sparse files: load as dict, compare the 'values' key's tensor.
    """
    import glob

    files_a = {
        os.path.basename(f)
        for f in glob.glob(os.path.join(dir_path, f"{prefix_a}*.pt"))
    }
    files_b = {
        os.path.basename(f)
        for f in glob.glob(os.path.join(dir_path, f"{prefix_b}*.pt"))
    }

    names_a = {f[len(prefix_a) :]: f for f in files_a}
    names_b = {f[len(prefix_b) :]: f for f in files_b}

    common = set(names_a.keys()) & set(names_b.keys())

    results = {}
    for name in sorted(common):
        path_a = os.path.join(dir_path, names_a[name])
        path_b = os.path.join(dir_path, names_b[name])

        is_sparse = "sparse" in name.lower() or "CoalescedHashtable" in name

        data_a = torch.load(path_a, map_location="cpu", weights_only=False)
        data_b = torch.load(path_b, map_location="cpu", weights_only=False)

        if is_sparse:
            t_a = data_a["values"]
            t_b = data_b["values"]
            max_diff = (t_a - t_b).abs().max().item()
        else:
            max_diff = (data_a - data_b).abs().max().item()

        results[name] = max_diff

    return results


def sparse_grad_stacked_by_sorted_fid(hashtable, sparse_grad: torch.Tensor):
    """sparse_grad: coalesced COO from hashtable.grad(...). Returns (ids_sorted, vals[N,D])."""
    s = sparse_grad.coalesce()
    rows = s.indices()[0]
    vals = s.values()
    slot_to_vec = {int(r.item()): vals[k] for k, r in enumerate(rows)}
    fea_ids, internal = hashtable.ids_map()
    fea_ids = fea_ids.view(-1)
    internal = internal.view(-1)
    pairs = []
    for fid, slot in zip(fea_ids.tolist(), internal.tolist()):
        slot = int(slot)
        if slot in slot_to_vec:
            pairs.append((int(fid), slot_to_vec[slot]))
    pairs.sort(key=lambda x: x[0])
    if not pairs:
        return torch.empty(0, dtype=torch.long), torch.empty(
            0, vals.shape[-1], device=vals.device, dtype=vals.dtype
        )
    ids_sorted = torch.tensor(
        [p[0] for p in pairs], dtype=torch.long, device=vals.device
    )
    vals_stacked = torch.stack([p[1] for p in pairs], dim=0)
    return ids_sorted, vals_stacked


def save_sparse_grad_for_compare(path: str, hashtable, sparse_grad: torch.Tensor):
    ids_sorted, vals = sparse_grad_stacked_by_sorted_fid(hashtable, sparse_grad)
    payload = {
        "ids": ids_sorted.cpu(),
        "values": vals.detach().cpu().float(),
    }
    torch.save(payload, path)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_options = {
            "table_1": EmbeddingOption(
                embedding_dim=3,
                shared_name="table_1",
                combiner="sum",
                coalesced=True,
                device=torch.device("cuda"),
                initializer=ConstantInitializer(0.5),
            ),
            "table_2": EmbeddingOption(
                embedding_dim=3,
                shared_name="table_2",
                combiner="sum",
                coalesced=True,
                device=torch.device("cuda"),
                initializer=ConstantInitializer(0.5),
            ),
        }

        self.embedding_engine = EmbeddingEngine(emb_options)

        self.dense1 = nn.Sequential(
            nn.Linear(3, 7),
            nn.ReLU(),
            nn.Linear(7, 1),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(3, 7),
            nn.ReLU(),
            nn.Linear(7, 1),
        )

    def reset_parameters(self):
        for ht in self.embedding_engine._ht.values():
            ht._hashtable.reset()
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0.15)
                nn.init.constant_(m.bias, 0.11)

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


def dump_grad(epoch, model, save_dir):
    for name, param in model.named_parameters():
        if param.grad is not None:
            fname = str(epoch) + "_" + name + ".pt"
            torch.save(param.grad, os.path.join(save_dir, fname))

    for name, param in model.named_modules():
        if not isinstance(param, DynamicEmbedding):
            continue
        data = param._hashtable.grad().coalesce()
        fname = str(epoch) + "_" + name + ".pt"
        save_sparse_grad_for_compare(
            os.path.join(save_dir, fname), param._hashtable, data
        )


def get_optimizer(model: nn.Module):
    dense_lr = 0.001
    sparse_lr = 0.001
    sparse_param = filter_out_sparse_param(model)

    dense_opt = torch.optim.AdamW(model.parameters(), lr=dense_lr, weight_decay=1e-6)
    sparse_opt = SparseAdamWTF(
        sparse_param,
        lr=sparse_lr,
        weight_decay=1e-6,
    )
    return dense_opt, sparse_opt


class GradientAccumulationTest(unittest.TestCase):
    """Verify gradient consistency across different gradient_accumulation_steps."""

    @classmethod
    def setUpClass(cls):
        cls.save_dir = tempfile.mkdtemp(prefix="grad_accu_test_")
        # Model can only be created once due to global registration, so create it here
        if not torch.cuda.is_available():
            return
        # env:// rendezvous requires these before init_process_group
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(common.find_free_port()))
        torch.distributed.init_process_group(backend="nccl")
        cls.model = Model()
        cls.model = cls.model.cuda()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.save_dir):
            shutil.rmtree(cls.save_dir)
        if torch.cuda.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def _run_train(self, accumulation_steps, seed=42):
        # Reset model parameters instead of reloading state_dict
        self.model.reset_parameters()

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        dense_opt, sparse_opt = get_optimizer(self.model)

        accelerator = Accelerator(
            gradient_accumulation_steps=accumulation_steps,
        )
        model, dense_opt = accelerator.prepare(self.model, dense_opt)
        sparse_opt.set_grad_accum_steps(accumulation_steps)

        random_input1 = torch.arange(0, 30, dtype=torch.long).reshape(10, 3).cuda()
        random_input2 = torch.arange(0, 30, dtype=torch.long).reshape(10, 3).cuda()

        dense_opt.zero_grad()
        sparse_opt.zero_grad()

        batch_size = 10 // accumulation_steps

        for i in range(accumulation_steps):
            with accelerator.accumulate(model):
                loss = model(
                    random_input1[i * batch_size : (i + 1) * batch_size],
                    random_input2[i * batch_size : (i + 1) * batch_size],
                )
                loss = loss.mean()
                accelerator.backward(loss)
                dense_opt.step()
                sparse_opt.step()

        dump_grad(accumulation_steps, model, self.save_dir)

    def test_grad_accu_1_vs_5(self):
        self._run_train(accumulation_steps=1)
        self._run_train(accumulation_steps=5)

        results = compare_pt_files(self.save_dir, prefix_a="1_", prefix_b="5_")

        print("\n" + "=" * 60)
        print(f"{'parameter name':<80} {'max_abs_diff':>15}")
        print("-" * 95)
        for name, diff in sorted(results.items()):
            print(f"{name:<80} {diff:>15.8e}")
        print("=" * 60)

        for name, diff in results.items():
            self.assertLess(
                diff, 1e-5, f"Gradient difference too large: {name} = {diff}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
