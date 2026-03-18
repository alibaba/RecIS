import random
import unittest

import torch

from recis.nn.functional.ragged_ops import dense_to_ragged
from recis.ragged.tensor import RaggedTensor


def get_invaild_data(device="cpu", dtype=torch.float32):
    data = (
        torch.Tensor([[1, 0, 3, 0, 0], [6, 7, 8, 0, 0]])
        .to(device=device)
        .to(dtype=dtype)
    )
    values = torch.Tensor([1, 0, 3, 6, 7, 8]).to(device=device).to(dtype=dtype)
    offsets = torch.tensor([0, 3, 6], device=device).to(torch.int)
    return data, values, offsets


def get_data(device="cpu", dtype=torch.float32):
    data = (
        torch.Tensor([[1, 2, 3, 0, 0], [6, 7, 8, 0, 0]])
        .to(device=device)
        .to(dtype=dtype)
    )
    values = torch.Tensor([1, 2, 3, 6, 7, 8]).to(device=device).to(dtype=dtype)
    offsets = torch.tensor([0, 3, 6], device=device).to(torch.int)
    return data, values, offsets


def get_random_data(min_shape, max_shape, start=0, dtype=torch.int64):
    shape = tuple(
        random.randint(min_dim, max_dim)
        for min_dim, max_dim in zip(min_shape, max_shape)
    )
    total_elements = 1
    for dim in shape:
        total_elements *= dim

    return torch.arange(
        start, start + total_elements, dtype=dtype, device="cuda"
    ).reshape(shape)


class TestDenseToRagged(unittest.TestCase):
    def test_dense_to_ragged_check_invalid(self):
        check_invalid = True
        invalid_value = 0
        for data_invalid in [True, False]:
            for device in ["cpu", "cuda"]:
                for dtype in [torch.int32, torch.float32, torch.int64]:
                    with self.subTest(
                        device=device,
                        dtype=dtype,
                        check_invalid=check_invalid,
                        data_invalid=data_invalid,
                    ):
                        if data_invalid:
                            data, values, offsets = get_invaild_data(device, dtype)
                        else:
                            data, values, offsets = get_data(device, dtype)
                        values_ret, offsets_ret = dense_to_ragged(
                            data, check_invalid, invalid_value
                        )
                        self.assertTrue(torch.equal(values, values_ret))
                        self.assertTrue(torch.equal(offsets, offsets_ret[0]))

    def test_dense_to_ragged_no_check_invalid(self):
        check_invalid = False
        invalid_value = 0
        for i in range(10):
            tensor = get_random_data((1, 1, 1), (100, 100, 100))
            val, offsets = dense_to_ragged(tensor, check_invalid, invalid_value)
            ragged = RaggedTensor(val, offsets)
            ragged_dense = ragged.to_dense()
            self.assertTrue(torch.equal(tensor, ragged_dense))

        for i in range(10):
            tensor = get_random_data((1, 1), (100, 100))
            val, offsets = dense_to_ragged(tensor, check_invalid, invalid_value)
            ragged = RaggedTensor(val, offsets)
            ragged_dense = ragged.to_dense()
            self.assertTrue(torch.equal(tensor, ragged_dense))


if __name__ == "__main__":
    unittest.main()
