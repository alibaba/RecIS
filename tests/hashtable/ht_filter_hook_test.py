import os
import random
import shutil
import unittest
from collections import defaultdict
from typing import List, Optional, Union
from unittest import mock

import torch
import torch.distributed as dist
import torch.testing._internal.common_utils as common

from recis.common.singleton import SingletonMeta
from recis.hooks.filter_hook import HashTableFilterHook
from recis.nn.functional import fused_ops
from recis.nn.hashtable_hook import AdmitHook, FilterHook
from recis.nn.initializers import ConstantInitializer
from recis.nn.modules.embedding import EmbeddingOption
from recis.nn.modules.embedding_engine import EmbeddingEngine
from recis.nn.modules.hashtable import HashtableRegister, split_sparse_dense_state_dict
from recis.nn.modules.hashtable_hook_impl import HashtableHookFactory
from recis.optim import SparseAdamWTF
from recis.ragged.tensor import RaggedTensor
from recis.serialize import Loader as SLoader, Saver as SSaver


def ts_equal(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    t1 = t1.cuda()
    t2 = t2.cuda()
    if t1.shape != t2.shape:
        print(
            f"Tensors have different shapes: t1.shape={t1.shape}, t2.shape={t2.shape}"
        )
        return False
    sorted_t1, _ = torch.sort(t1)
    sorted_t2, _ = torch.sort(t2)
    is_equal = torch.equal(sorted_t1, sorted_t2)
    if not is_equal:
        print(
            f"Tensors do not contain the same set of elements, {sorted_t1}, {sorted_t2}"
        )
        for i in range(sorted_t1.shape[0]):
            if sorted_t1[i] != sorted_t2[i]:
                print(
                    f"  - At index {i}: t1 has value {sorted_t1[i].item()}, but t2 has value {sorted_t2[i].item()}"
                )
    return is_equal


def gen_2d_ragged_tensor(
    batch_size: int,
    max_row_length: int,
    must_have_vals: Union[int, List[int], None] = None,
    must_not_have_vals: Union[int, List[int], None] = None,
    min_val: int = 0,
    max_val: int = 99999999,
    is_ragged: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.int64,
    seed: Optional[int] = None,
):
    table = set()
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    if must_have_vals is not None and not isinstance(must_have_vals, list):
        must_have_vals = [must_have_vals]
    if must_not_have_vals is not None and not isinstance(must_not_have_vals, list):
        must_not_have_vals = [must_not_have_vals]

    offsets = [0]
    all_flat_values = []

    for i in range(batch_size):
        row_len = random.randint(0, max_row_length) if is_ragged else max_row_length
        if row_len > 0:
            vals = []
            while len(vals) < row_len:
                new_val = random.randint(min_val, max_val)
                if must_not_have_vals is None or new_val not in must_not_have_vals:
                    vals.append(new_val)
                    if new_val not in table:
                        table.add(new_val)
            all_flat_values.extend(vals)

        offsets.append(offsets[-1] + row_len)

    if not all_flat_values:
        print("No values were generated. Consider adjusting parameters.")
        values = torch.empty(0, dtype=dtype, device=device)
        offsets_tensor = torch.tensor([0], dtype=torch.int32, device=device)
        return values, offsets_tensor

    if must_have_vals is not None:
        if len(must_have_vals) > len(all_flat_values):
            raise ValueError(
                f"The number of values to include [{len(must_have_vals)}], exceeds the total number of generated elements [{len(all_flat_values)}]."
            )
        for random_index, val_to_add in enumerate(must_have_vals):
            tmp = all_flat_values[random_index]
            table.discard(tmp)
            all_flat_values[random_index] = val_to_add
            table.add(val_to_add)

    values = torch.tensor(all_flat_values, dtype=dtype, device=device)
    offsets_tensor = torch.tensor(offsets, dtype=torch.int32, device=device)

    return RaggedTensor(values, offsets_tensor), table


def contains_all_values(my_tensor: torch.Tensor, values_to_check: torch.Tensor) -> bool:
    unique_main = torch.unique(my_tensor)
    unique_to_check = torch.unique(values_to_check)
    bool_mask = torch.isin(unique_to_check, unique_main)
    return torch.all(bool_mask).item()


def exclude_all_values(my_tensor: torch.Tensor, values_to_check: torch.Tensor) -> bool:
    isin_mask = torch.isin(my_tensor, values_to_check)
    found_any = torch.any(isin_mask)
    return not found_any


def check_emb_rows(tensor, indices, x):
    mask = torch.zeros(tensor.size(0), dtype=torch.bool, device=tensor.device)
    mask[torch.as_tensor(indices, device=tensor.device)] = True
    rows = tensor[mask]
    other_rows = tensor[~mask]
    x_tensor = torch.as_tensor(x, device=tensor.device)
    if x_tensor.dim() == 0:
        return (
            torch.all(other_rows == x_tensor).item()
            and not torch.any(rows == x_tensor).item()
        )
    else:
        return (
            torch.all(other_rows == x_tensor.unsqueeze(0)).item()
            and torch.any(rows == x_tensor.unsqueeze(0)).item()
        )


def nested_dict():
    return defaultdict(nested_dict)


class HashTableFilterHookTest(unittest.TestCase):
    DEVICE = None
    CKPT_DIR = None

    @classmethod
    def setUpClass(cls):
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(common.find_free_port())
        dist.init_process_group()
        cls.DEVICE = os.getenv("TEST_DEVICE", "cuda")
        cls.CKPT_DIR = os.getenv("CKPT_DIR", "./unittest_ckpt_cuda")
        os.makedirs(cls.CKPT_DIR, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        shutil.rmtree(cls.CKPT_DIR)

    def setIds(self):
        self.ids = {
            "step_0": {
                "fea_1": RaggedTensor(
                    torch.tensor(
                        [12, 13, 14, 15, 16], dtype=torch.int64, device="cuda"
                    ),
                    torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64, device="cuda"),
                ),
                "fea_2": RaggedTensor(
                    torch.tensor(
                        [27, 28, 29, 30, 31, 32, 33], dtype=torch.int64, device="cuda"
                    ),
                    torch.tensor([0, 1, 2, 3, 4, 7], dtype=torch.int64, device="cuda"),
                ),
                "fea_3": RaggedTensor(
                    torch.tensor([12, 99], dtype=torch.int64, device="cuda"),
                    torch.tensor([0, 1, 2, 2, 2, 2], dtype=torch.int64, device="cuda"),
                ),
                "fea_4": RaggedTensor(
                    torch.tensor([100, 101, 102], dtype=torch.int64, device="cuda"),
                    torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda"),
                ),
                "fea_6": RaggedTensor(
                    torch.tensor([200, 201, 202], dtype=torch.int64, device="cuda"),
                    torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda"),
                ),
            },
            "step_1": {
                "fea_1": RaggedTensor(
                    torch.tensor(
                        [12, 13, 14, 15, 16], dtype=torch.int64, device="cuda"
                    ),
                    torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64, device="cuda"),
                ),
                "fea_2": RaggedTensor(
                    torch.tensor(
                        [27, 28, 29, 30, 31, 32, 33], dtype=torch.int64, device="cuda"
                    ),
                    torch.tensor([0, 1, 2, 3, 4, 7], dtype=torch.int64, device="cuda"),
                ),
                "fea_3": RaggedTensor(
                    torch.tensor([12, 99], dtype=torch.int64, device="cuda"),
                    torch.tensor([0, 1, 2, 2, 2, 2], dtype=torch.int64, device="cuda"),
                ),
                "fea_4": RaggedTensor(
                    torch.tensor([100, 101, 102], dtype=torch.int64, device="cuda"),
                    torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda"),
                ),
                "fea_6": RaggedTensor(
                    torch.tensor([200, 201, 202], dtype=torch.int64, device="cuda"),
                    torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda"),
                ),
            },
            "step_2": {
                "fea_1": RaggedTensor(
                    torch.tensor(
                        [12, 13, 14, 15, 16], dtype=torch.int64, device="cuda"
                    ),
                    torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64, device="cuda"),
                ),
                "fea_2": RaggedTensor(
                    torch.tensor(
                        [27, 28, 29, 30, 31, 32, 33], dtype=torch.int64, device="cuda"
                    ),
                    torch.tensor([0, 1, 2, 3, 4, 7], dtype=torch.int64, device="cuda"),
                ),
                "fea_3": RaggedTensor(
                    torch.tensor([12, 99], dtype=torch.int64, device="cuda"),
                    torch.tensor([0, 1, 2, 2, 2, 2], dtype=torch.int64, device="cuda"),
                ),
                "fea_4": RaggedTensor(
                    torch.tensor([100, 101, 102], dtype=torch.int64, device="cuda"),
                    torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda"),
                ),
                "fea_6": RaggedTensor(
                    torch.tensor([200, 201, 202], dtype=torch.int64, device="cuda"),
                    torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda"),
                ),
            },
            "step_3": {
                "fea_1": RaggedTensor(
                    torch.tensor([12, 13, 16], dtype=torch.int64, device="cuda"),
                    torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda"),
                ),
                "fea_2": RaggedTensor(
                    torch.tensor([777777, 28], dtype=torch.int64, device="cuda"),
                    torch.tensor([0, 1, 2, 2], dtype=torch.int64, device="cuda"),
                ),
                "fea_6": RaggedTensor(
                    torch.tensor([200, 201, 202], dtype=torch.int64, device="cuda"),
                    torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda"),
                ),
            },
            "step_4": {
                "fea_3": RaggedTensor(
                    torch.tensor([12, 13, 16, 999], dtype=torch.int64, device="cuda"),
                    torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device="cuda"),
                ),
                "fea_6": RaggedTensor(
                    torch.tensor([131, 202, 203], dtype=torch.int64, device="cuda"),
                    torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda"),
                ),
            },
            "step_5": {
                "fea_1": RaggedTensor(
                    torch.tensor([999], dtype=torch.int64, device="cuda"),
                    torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
                ),
                "fea_2": RaggedTensor(
                    torch.tensor([12], dtype=torch.int64, device="cuda"),
                    torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
                ),
                "fea_6": RaggedTensor(
                    torch.tensor(
                        [999, 200, 201, 202], dtype=torch.int64, device="cuda"
                    ),
                    torch.tensor([0, 4], dtype=torch.int64, device="cuda"),
                ),
            },
            "step_6": {
                "fea_2": RaggedTensor(
                    torch.tensor([1111], dtype=torch.int64, device="cuda"),
                    torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
                ),
            },
            "step_7": {
                "fea_2": RaggedTensor(
                    torch.tensor([1111], dtype=torch.int64, device="cuda"),
                    torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
                ),
            },
        }

        self.record = {}
        for i in range(8, 68):
            rt5, re5 = gen_2d_ragged_tensor(
                batch_size=5000,
                max_row_length=20,
                is_ragged=False,
                device="cuda",
            )
            for e in re5:
                self.record[e] = i
            rt7, re7 = gen_2d_ragged_tensor(
                batch_size=5000,
                max_row_length=20,
                is_ragged=False,
                device="cuda",
            )
            for e in re7:
                if e in self.record:
                    # print(f"{e} in record!, update step = {i}")
                    self.record[e] = i
                # else:
                # print(f"{e} not in record!, at step = {i}")
            self.ids[f"step_{i}"] = {"fea_5": rt5, "fea_7": rt7}

        # print(f"!!!!!adasd self.record = {self.record}")

        # max(8 + 50 + 1, 7 + 10 * 6) = 67 (The first time when id is filtered of fea 5 hashtable)
        # 67 - 50 = 17 (The latest step of id to be filterd)
        self.invalid_ids = [key for key, value in self.record.items() if value <= 17]
        self.invalid_ids_numel = len(self.invalid_ids)
        self.valid_ids = [key for key, value in self.record.items() if value > 17]
        self.valid_ids_numel = len(self.valid_ids)

        print(f"self.invalid_ids_numel = {self.invalid_ids_numel}")
        for i in range(68, 69):
            tmp_val = list(range(100000000, 100000000 + self.invalid_ids_numel + 1))
            rt, re = gen_2d_ragged_tensor(
                batch_size=self.invalid_ids_numel + 1,
                max_row_length=1,
                must_have_vals=tmp_val,
                is_ragged=False,
                device="cuda",
            )
            self.ids[f"step_{i}"] = {"fea_5": rt}

    def setFeas(self):
        group_a_filter_step = 1
        group_b_filter_step = 3
        group_c_filter_step = 50

        self.group_a = nested_dict()
        self.group_b = nested_dict()
        self.group_c = nested_dict()

        self.group_a["ht1"]["emb_opt"] = {
            "fea_1": EmbeddingOption(
                embedding_dim=8,
                shared_name="ht1",
                combiner="sum",
                initializer=ConstantInitializer(init_val=3.0),
                device=torch.device(self.DEVICE),
                filter_hook=FilterHook(
                    "GlobalStepFilter", {"filter_step": group_a_filter_step}
                ),  # create step filter hook
            ),
            "fea_3": EmbeddingOption(
                embedding_dim=8,
                shared_name="ht1",
                combiner="mean",
                initializer=ConstantInitializer(init_val=3.0),
                device=torch.device(self.DEVICE),
                filter_hook=FilterHook(
                    "GlobalStepFilter", {"filter_step": group_a_filter_step}
                ),
                trainable=False,
            ),
            "fea_6": EmbeddingOption(
                embedding_dim=8,
                shared_name="ht1",
                combiner="mean",
                initializer=ConstantInitializer(init_val=3.0),
                device=torch.device(self.DEVICE),
                filter_hook=FilterHook(
                    "GlobalStepFilter", {"filter_step": group_a_filter_step}
                ),
                admit_hook=AdmitHook(
                    "ReadOnly"
                ),  # id from feature 6 is not admitted to ht1
                trainable=True,
            ),
        }
        self.group_a["ht2"]["emb_opt"] = {
            "fea_2": EmbeddingOption(
                embedding_dim=8,
                shared_name="ht2",
                combiner="mean",
                initializer=ConstantInitializer(init_val=3.0),
                device=torch.device(self.DEVICE),
                filter_hook=FilterHook(
                    "GlobalStepFilter", {"filter_step": group_a_filter_step}
                ),
            )
        }

        self.group_b["ht3"]["emb_opt"] = {
            "fea_4": EmbeddingOption(
                embedding_dim=8,
                shared_name="ht3",
                combiner="sum",
                initializer=ConstantInitializer(init_val=3.0),
                device=torch.device(self.DEVICE),
                filter_hook=FilterHook(
                    "GlobalStepFilter", {"filter_step": group_b_filter_step}
                ),
            )
        }

        self.group_c["ht4"]["emb_opt"] = {
            "fea_5": EmbeddingOption(
                embedding_dim=2,
                shared_name="ht4",
                combiner="sum",
                initializer=ConstantInitializer(init_val=8.0),
                device=torch.device(self.DEVICE),
                filter_hook=FilterHook(
                    "GlobalStepFilter", {"filter_step": group_c_filter_step}
                ),
            ),
            "fea_7": EmbeddingOption(
                embedding_dim=2,
                shared_name="ht4",
                combiner="sum",
                initializer=ConstantInitializer(init_val=8.0),
                device=torch.device(self.DEVICE),
                filter_hook=FilterHook(
                    "GlobalStepFilter",
                    {"filter_step": group_c_filter_step},
                ),
                admit_hook=AdmitHook(
                    "ReadOnly"
                ),  # id from feature 7 is not admitted to ht7
            ),
        }

    def setEmbeddingEngine(self):
        from collections import ChainMap

        ee_init = dict(
            ChainMap(
                self.group_a["ht1"]["emb_opt"],
                self.group_a["ht2"]["emb_opt"],
                self.group_b["ht3"]["emb_opt"],
                self.group_c["ht4"]["emb_opt"],
            )
        )
        self.ee = EmbeddingEngine(ee_init)

    def setHtInfo(self):
        self.tables = nested_dict()
        group_to_fea = {
            "group_a": "fea_1",
            "group_b": "fea_4",
            "group_c": "fea_5",
        }
        for group_name, fea_name in group_to_fea.items():
            ht_key = self.ee._fea_to_ht[fea_name]
            self.tables[group_name]["ht"] = self.ee._ht[ht_key]._hashtable
        for group_name in group_to_fea:
            ht = self.tables[group_name]["ht"]
            filter_slot = ht.slot_group().slot_by_name("step_filter")
            self.tables[group_name]["filter_slot"] = filter_slot
            self.tables[group_name]["emb_slot"] = ht.slot_group().slot_by_name(
                "embedding"
            )
            self.tables[group_name]["sparse_adamw_tf_exp_avg_slot"] = (
                ht.slot_group().slot_by_name("sparse_adamw_tf_exp_avg")
            )
            self.tables[group_name]["sparse_adamw_tf_exp_avg_sq_slot"] = (
                ht.slot_group().slot_by_name("sparse_adamw_tf_exp_avg_sq")
            )

    def setEncodeIds(self):
        self.encode_ids = defaultdict(str)
        self.feas = ["fea_1", "fea_2", "fea_3", "fea_4", "fea_5", "fea_6", "fea_7"]
        for fea in self.feas:
            self.encode_ids[fea] = self.ee._fea_to_group[fea].encode_id(fea)

    def setUp(self):
        # Each test owns its tables and hooks, even when xdist reuses a worker.
        # addCleanup also restores the previous instances if setup fails.
        isolated_instances = mock.patch.dict(SingletonMeta._instances)
        isolated_instances.start()
        self.addCleanup(isolated_instances.stop)
        SingletonMeta._instances.pop(HashtableRegister, None)
        factory = HashtableHookFactory()
        # Keep implementations registered at module import; isolate table instances.
        for attribute in ("_filters", "_admits"):
            isolated_hooks = mock.patch.object(factory, attribute, defaultdict(dict))
            isolated_hooks.start()
            self.addCleanup(isolated_hooks.stop)
        exec_step = 1
        self.setIds()
        self.setFeas()
        self.setEmbeddingEngine()
        self.sparse_state, self.dense_state = split_sparse_dense_state_dict(
            self.ee.state_dict()
        )
        self.opt = SparseAdamWTF(self.sparse_state)
        self.setHtInfo()
        self.setEncodeIds()
        self.hook = HashTableFilterHook(exec_step)

    def testFilterHook(self):
        for step in range(3):
            step_id = self.ids[f"step_{step}"]
            out = self.ee(step_id)
            (out["fea_1"].sum() + out["fea_2"].sum() + out["fea_6"].sum()).backward()
            self.opt.step()
            self.opt.zero_grad()
            self.hook.after_step(global_step=step)  # run filter

            ids, mmap = self.tables["group_a"]["ht"].ids_map()
            self.assertTrue(
                ts_equal(
                    self.tables["group_a"]["ht"].ids(),
                    fused_ops.fused_ids_encode_gpu(
                        [
                            step_id["fea_1"].values(),
                            step_id["fea_2"].values(),
                            torch.tensor([99], dtype=torch.int64, device="cuda"),
                        ],
                        [
                            self.encode_ids["fea_1"],
                            self.encode_ids["fea_2"],
                            self.encode_ids["fea_3"],
                        ],
                    ),
                )
            )
            self.assertTrue(
                ts_equal(
                    self.tables["group_b"]["ht"].ids(),
                    torch.tensor([100, 101, 102]).cuda(),
                )
            )
            self.assertTrue(
                (self.tables["group_a"]["filter_slot"].value()[:13].view(-1) == step)
                .all()
                .item()
            )
            self.assertTrue(
                (self.tables["group_b"]["filter_slot"].value()[:3].view(-1) == step)
                .all()
                .item()
            )
            self.assertTrue(mmap.max().item() == 12)

        for step in range(3, 4):
            step_id = self.ids[f"step_{step}"]
            out = self.ee(step_id)
            (out["fea_1"].sum() + out["fea_2"].sum() + out["fea_6"].sum()).backward()
            self.opt.step()
            self.opt.zero_grad()
            self.hook.after_step(global_step=step)  # run filter.

            # self.assertTrue(ts_equal(mmap, torch.tensor([4, 3, 0, 10, 13]))). only for gpu
            ids, mmap = self.tables["group_a"]["ht"].ids_map()
            self.assertTrue(mmap.max().item() <= 13)
            self.assertTrue(
                ts_equal(
                    self.tables["group_a"]["ht"].ids(),
                    fused_ops.fused_ids_encode_gpu(
                        [step_id["fea_1"].values(), step_id["fea_2"].values()],
                        [self.encode_ids["fea_1"], self.encode_ids["fea_2"]],
                    ),
                )
            )
            self.assertTrue(
                ts_equal(
                    self.tables["group_b"]["ht"].ids(),
                    torch.tensor([100, 101, 102]).cuda(),
                )
            )
            self.assertTrue(
                ts_equal(
                    self.tables["group_a"]["filter_slot"].value()[:14].view(-1),
                    torch.tensor([3] * 5 + [2] * 9, dtype=torch.int64),
                )
            )
            self.assertTrue(
                (self.tables["group_b"]["filter_slot"].value()[:3].view(-1) == 2)
                .all()
                .item()
            )

        for step in range(4, 5):
            step_id = self.ids[f"step_{step}"]
            out = self.ee(step_id)
            (out["fea_3"].sum() + out["fea_6"].sum()).backward()
            self.opt.step()
            self.opt.zero_grad()

            self.hook.after_step(global_step=step)  # run filter hook

            ids, mmap = self.tables["group_a"]["ht"].ids_map()
            self.assertTrue(mmap.max().item() <= 12)
            self.assertTrue(
                ts_equal(
                    self.tables["group_a"]["ht"].ids(),
                    fused_ops.fused_ids_encode_gpu(
                        [step_id["fea_3"].values()], [self.encode_ids["fea_3"]]
                    ),
                )
            )
            self.assertTrue(
                ts_equal(
                    self.tables["group_b"]["ht"].ids(),
                    torch.tensor([100, 101, 102]).cuda(),
                )
            )
            self.assertTrue(
                ts_equal(
                    self.tables["group_a"]["filter_slot"].value()[:14].view(-1),
                    torch.tensor([4] * 4 + [3] * 2 + [2] * 8, dtype=torch.int64),
                )
            )
            self.assertTrue(
                (self.tables["group_b"]["filter_slot"].value()[:3].view(-1) == 2)
                .all()
                .item()
            )

        for step in range(5, 6):
            step_id = self.ids[f"step_{step}"]
            out = self.ee(step_id)
            (out["fea_1"].sum() + out["fea_2"].sum() + out["fea_6"].sum()).backward()
            self.opt.step()
            self.opt.zero_grad()

            ids, mmap = self.tables["group_a"]["ht"].ids_map()
            self.hook.after_step(global_step=step)  # run hook
            self.assertTrue(mmap.max().item() <= 13)
            self.assertTrue(
                ts_equal(
                    self.tables["group_a"]["ht"].ids(),
                    fused_ops.fused_ids_encode_gpu(
                        [step_id["fea_1"].values(), step_id["fea_2"].values()],
                        [self.encode_ids["fea_1"], self.encode_ids["fea_2"]],
                    ),
                )
            )
            self.assertTrue(
                ts_equal(self.tables["group_b"]["ht"].ids(), torch.tensor([]))
            )
            self.assertTrue(
                (self.tables["group_b"]["filter_slot"].value()[:3].view(-1) == 2)
                .all()
                .item()
            )
        # check embedding and optimizer slot
        group_a_ids, group_a_mmap = self.tables["group_a"]["ht"].ids_map()
        group_b_ids, group_b_mmap = self.tables["group_b"]["ht"].ids_map()
        self.assertTrue(
            check_emb_rows(
                self.tables["group_a"]["emb_slot"].value()[:14], group_a_mmap, 3
            )
        )
        self.assertTrue(
            check_emb_rows(
                self.tables["group_a"]["sparse_adamw_tf_exp_avg_slot"].value()[:14],
                group_a_mmap,
                0,
            )
        )
        self.assertTrue(
            check_emb_rows(
                self.tables["group_a"]["sparse_adamw_tf_exp_avg_sq_slot"].value()[:14],
                group_a_mmap,
                0,
            )
        )
        self.assertTrue(
            check_emb_rows(
                self.tables["group_b"]["emb_slot"].value()[:3], group_b_mmap, 3
            )
        )
        self.assertTrue(
            check_emb_rows(
                self.tables["group_b"]["sparse_adamw_tf_exp_avg_slot"].value()[:3],
                group_b_mmap,
                0,
            )
        )
        self.assertTrue(
            check_emb_rows(
                self.tables["group_b"]["sparse_adamw_tf_exp_avg_sq_slot"].value()[:3],
                group_b_mmap,
                0,
            )
        )

        saver = SSaver(
            shard_index=0,
            shard_num=1,
            parallel=1,
            hashtables=self.sparse_state,
            tensors=self.dense_state,
            path=self.CKPT_DIR,
        )
        saver.save()
        filter_save_gstep = {}
        for hook_name, hooks in HashtableHookFactory().get_filters().items():
            if hook_name != "GlobalStepFilter":
                continue
            for ht, ft in hooks.items():
                filter_save_gstep[ht] = ft._global_step
                ft._global_step.fill_(0)
        self.tables["group_a"]["filter_slot"].values()[0].fill_(0)
        self.tables["group_a"]["emb_slot"].values()[0].fill_(3)
        self.tables["group_a"]["sparse_adamw_tf_exp_avg_slot"].values()[0].fill_(0)
        self.tables["group_a"]["sparse_adamw_tf_exp_avg_sq_slot"].values()[0].fill_(0)
        self.tables["group_b"]["filter_slot"].values()[0].fill_(0)
        self.tables["group_b"]["emb_slot"].values()[0].fill_(3)
        self.tables["group_b"]["sparse_adamw_tf_exp_avg_slot"].values()[0].fill_(0)
        self.tables["group_b"]["sparse_adamw_tf_exp_avg_sq_slot"].values()[0].fill_(0)

        loader = SLoader(
            self.CKPT_DIR,
            hashtables=self.sparse_state,
            tensors=self.dense_state,
            parallel=1,
        )
        loader.load()
        for hook_name, hooks in HashtableHookFactory().get_filters().items():
            if hook_name != "GlobalStepFilter":
                continue
            for ht, ft in hooks.items():
                self.assertTrue(filter_save_gstep[ht], ft._global_step)

        group_a_ids, group_a_mmap = self.tables["group_a"]["ht"].ids_map()
        group_b_ids, group_b_mmap = self.tables["group_b"]["ht"].ids_map()

        self.assertTrue(ts_equal(group_a_ids, torch.tensor([12, 4503599627371495])))
        self.assertTrue(
            check_emb_rows(
                self.tables["group_a"]["emb_slot"].value()[:14], group_a_mmap, 3
            )
        )
        self.assertTrue(
            check_emb_rows(
                self.tables["group_a"]["sparse_adamw_tf_exp_avg_slot"].value()[:14],
                group_a_mmap,
                0,
            )
        )
        self.assertTrue(
            check_emb_rows(
                self.tables["group_a"]["sparse_adamw_tf_exp_avg_sq_slot"].value()[:14],
                group_a_mmap,
                0,
            )
        )
        self.assertTrue(
            check_emb_rows(
                self.tables["group_b"]["emb_slot"].value()[:3], group_b_mmap, 3
            )
        )
        self.assertTrue(
            check_emb_rows(
                self.tables["group_b"]["sparse_adamw_tf_exp_avg_slot"].value()[:3],
                group_b_mmap,
                0,
            )
        )
        self.assertTrue(
            check_emb_rows(
                self.tables["group_b"]["sparse_adamw_tf_exp_avg_sq_slot"].value()[:3],
                group_b_mmap,
                0,
            )
        )

        for step in range(6, 8):
            step_id = self.ids[f"step_{step}"]
            out = self.ee(step_id)
            out["fea_2"].sum().backward()
            self.opt.step()
            self.opt.zero_grad()
            ids, mmap = self.tables["group_a"]["ht"].ids_map()
            if step == 6:
                self.assertTrue(
                    ts_equal(ids, torch.tensor([12, 1111, 4503599627371495]))
                )
            else:
                self.assertTrue(ts_equal(ids, torch.tensor([1111])))
            self.hook.after_step(global_step=step)  # run hook

        self.hook.reset_filter_interval(10)
        last_step_id_numel = 0
        after_filter_numel = 0
        for step in range(8, 69):
            step_id = self.ids[f"step_{step}"]
            out = self.ee(step_id)
            if step != 68:
                (out["fea_5"] + out["fea_7"]).sum().backward()
            else:
                out["fea_5"].sum().backward()
            self.opt.step()
            self.opt.zero_grad()
            self.hook.after_step(global_step=step)

            ids, mmap = self.tables["group_c"]["ht"].ids_map()
            last_step_id_numel = after_filter_numel
            after_filter_numel = ids.numel()
            if step == 67:
                self.assertTrue(
                    exclude_all_values(
                        ids.cuda(), torch.tensor(self.invalid_ids, device="cuda")
                    )
                )
                self.assertTrue(
                    ts_equal(ids.cuda(), torch.tensor(self.valid_ids, device="cuda"))
                )

                self.assertTrue(
                    check_emb_rows(
                        self.tables["group_c"]["sparse_adamw_tf_exp_avg_slot"].value()[
                            : mmap.max().item() + 1
                        ],
                        mmap,
                        0,
                    )
                )
            elif step == 68:
                insert_numel = self.ids[f"step_{step}"]["fea_5"].values().numel()
                self.assertTrue(mmap.max().item() == ids.numel() - 1)
                self.assertTrue(ids.numel() == last_step_id_numel + insert_numel)

        saver.save()


if __name__ == "__main__":
    unittest.main()
