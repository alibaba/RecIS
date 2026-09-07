import unittest
import uuid

import torch

from recis.nn.modules.hashtable import HashTable
from recis.optim.sparse_adagrad_sum import SparseAdagradSum


class SparseAdagradSumTest(unittest.TestCase):
    def test_rejects_weight_decay(self):
        with self.assertRaisesRegex(ValueError, "does not support weight_decay"):
            SparseAdagradSum({}, weight_decay=1e-4)

    def test_uses_external_grad_sq_for_state_update(self):
        if not torch.cuda.is_available():
            self.skipTest("SparseAdagradSum test requires cuda")

        lr = 0.1
        initial_accumulator_value = 0.5
        eps = 0.0
        table = HashTable(
            [3],
            block_size=8,
            device=torch.device("cuda"),
            name=f"sparse_adagrad_sum_{uuid.uuid4().hex}",
        )
        optimizer = SparseAdagradSum(
            {"table": table._hashtable_impl},
            lr=lr,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
        )

        ids = torch.tensor([11, 23], dtype=torch.long, device="cuda")
        params_before = torch.zeros([2, 3], dtype=torch.float32, device="cuda")
        table.insert(ids, params_before)
        snap_ids, snap_index, _ = table.snap_shot()
        order = torch.argsort(snap_ids)
        snap_index = snap_index[order].to(device="cuda", dtype=torch.long)

        grad = torch.tensor(
            [[3.0, 4.0, 0.0], [1.0, 2.0, 2.0]],
            dtype=torch.float32,
            device="cuda",
        )
        grad_sq = torch.tensor(
            [[5.0, 8.0, 1.0], [4.0, 9.0, 16.0]],
            dtype=torch.float32,
            device="cuda",
        )
        table._hashtable_impl.accept_grad(snap_index, grad)
        table._hashtable_impl.accept_grad_sq(snap_index, grad_sq)

        optimizer.step()

        result_ids, result_index, result_params = table.snap_shot()
        result_order = torch.argsort(result_ids)
        result_index = result_index[result_order]
        result_params = result_params[result_order]
        expected_state_sum = initial_accumulator_value + grad_sq
        expected_params = params_before - lr * grad / torch.sqrt(expected_state_sum)
        state_sum = (
            table.slot_group()
            .slot_by_name("sparse_adagrad_state_sum")
            .value()[result_index]
        )

        self.assertTrue(torch.allclose(state_sum, expected_state_sum))
        self.assertTrue(torch.allclose(result_params, expected_params))


if __name__ == "__main__":
    unittest.main()
