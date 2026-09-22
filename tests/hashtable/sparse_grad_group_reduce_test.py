import unittest

import torch

from recis.nn.functional.sparse_grad_group_ops import (
    _sparse_grad_group_reduce,
    _sparse_grad_group_reduce_compact,
    _sparse_grad_group_reduce_dense,
)


class SparseGradGroupReduceTest(unittest.TestCase):
    def setUp(self):
        self.index = torch.tensor([1, 1, 2, 1, 3, 1, 1, 2], dtype=torch.long)
        self.source_group = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.long)
        self.grad_outputs = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0],
                [13.0, 14.0],
                [15.0, 16.0],
            ],
            dtype=torch.float32,
        )
        self.num_unique = 4
        self.num_groups = 3
        self.group_size = 2

    def test_compact_matches_dense_for_all_reduce_modes(self):
        for group_reduce_by in ["id", "worker", "worker_sum"]:
            with self.subTest(group_reduce_by=group_reduce_by):
                dense = _sparse_grad_group_reduce_dense(
                    self.index,
                    self.source_group,
                    self.grad_outputs,
                    self.num_unique,
                    self.num_groups,
                    self.group_size,
                    group_reduce_by,
                )
                compact = _sparse_grad_group_reduce_compact(
                    self.index,
                    self.source_group,
                    self.grad_outputs,
                    self.num_unique,
                    self.group_size,
                    group_reduce_by,
                )
                self.assertTrue(torch.allclose(compact[0], dense[0]))
                self.assertTrue(torch.allclose(compact[1], dense[1]))

    def test_chunk_compact_matches_dense_for_all_reduce_modes(self):
        for group_reduce_by in ["id", "worker", "worker_sum"]:
            for chunk_groups in [1, 2]:
                with self.subTest(
                    group_reduce_by=group_reduce_by,
                    chunk_groups=chunk_groups,
                ):
                    dense = _sparse_grad_group_reduce(
                        self.index,
                        self.source_group,
                        self.grad_outputs,
                        self.num_unique,
                        self.num_groups,
                        self.group_size,
                        group_reduce_by,
                        "dense",
                        chunk_groups,
                    )
                    chunk_compact = _sparse_grad_group_reduce(
                        self.index,
                        self.source_group,
                        self.grad_outputs,
                        self.num_unique,
                        self.num_groups,
                        self.group_size,
                        group_reduce_by,
                        "chunk_compact",
                        chunk_groups,
                    )
                    self.assertTrue(torch.allclose(chunk_compact[0], dense[0]))
                    self.assertTrue(torch.allclose(chunk_compact[1], dense[1]))

    def test_id_reduce_semantics(self):
        grad_sum, grad_sq_sum = _sparse_grad_group_reduce_compact(
            self.index,
            self.source_group,
            self.grad_outputs,
            self.num_unique,
            self.group_size,
            "id",
        )

        expected_group0 = torch.tensor([2.0, 3.0])
        expected_group1 = torch.tensor([7.0, 8.0])
        expected_group2 = torch.tensor([12.0, 13.0])
        expected_sum = expected_group0 + expected_group1 + expected_group2
        expected_sq_sum = (
            expected_group0 * expected_group0
            + expected_group1 * expected_group1
            + expected_group2 * expected_group2
        )

        self.assertTrue(torch.allclose(grad_sum[1], expected_sum))
        self.assertTrue(torch.allclose(grad_sq_sum[1], expected_sq_sum))


if __name__ == "__main__":
    unittest.main()
