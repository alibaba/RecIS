import unittest

import torch

from recis.nn.modules.hashtable import HashTable


def ts_equal(
    t1: torch.Tensor, t2: torch.Tensor, float_rtol=1e-5, float_atol=1e-8
) -> bool:
    t1 = t1.cuda()
    t2 = t2.cuda()
    if t1.shape != t2.shape:
        print(
            f"Tensors have different shapes: t1.shape={t1.shape}, t2.shape={t2.shape}"
        )
        return False
    if t1.dtype != t2.dtype:
        print(f"Tensors have different dtype: t1.shape={t1.dtype}, t2.shape={t2.dtype}")
        return False

    sorted_t1, _ = torch.sort(t1)
    sorted_t2, _ = torch.sort(t2)
    if sorted_t1.dtype.is_floating_point:
        is_equal = torch.allclose(
            sorted_t1, sorted_t2, rtol=float_rtol, atol=float_atol
        )
    else:
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


class HashTableGradAccumulateTest(unittest.TestCase):
    def test_grad(self):
        ht = HashTable(embedding_shape=[4], dtype=torch.float32)
        grad_index = torch.LongTensor([1, 2, 3])
        grad = torch.ones([3, 4], dtype=torch.float32)
        ht.accept_grad(grad_index, grad)
        grad_index = torch.LongTensor([1, 1, 1])
        grad = torch.ones([3, 4], dtype=torch.float32) * 2
        ht.accept_grad(grad_index, grad)
        grad_index = torch.LongTensor([2, 2, 2])
        grad = torch.ones([3, 4], dtype=torch.float32) * 3
        ht.accept_grad(grad_index, grad)
        ga_mean = ht.grad()
        indices = ga_mean.coalesce().indices()
        indices_true = torch.LongTensor([[1, 2, 3]])
        self.assertTrue(ts_equal(indices, indices_true))
        values = ga_mean.coalesce().values()
        values_true = torch.tensor(
            [
                [7, 7, 7, 7],
                [10, 10, 10, 10],
                [1, 1, 1, 1],
            ]
        )
        self.assertTrue(ts_equal(values, values_true * 1.0))


if __name__ == "__main__":
    unittest.main()
