import unittest

import numpy as np
import torch

from recis.utils.logger import Logger


logger = Logger(__name__)

_FLOAT_PACK_ALIGNMENT = 16
_LEFT_SENTINEL = -12345.0
_RIGHT_SENTINEL = 12345.0


def make_data(N, shape, device="cuda"):
    np.random.seed(0)
    params = [
        torch.from_numpy(np.random.randn(*shape).astype(np.float32)).to(device=device)
        for _ in range(N)
    ]
    grads = [
        torch.from_numpy(np.random.randn(*shape).astype(np.float32)).to(device=device)
        for _ in range(N)
    ]
    avg = [
        torch.from_numpy(np.random.randn(*shape).astype(np.float32)).to(device=device)
        for _ in range(N)
    ]
    avg_sq = [
        torch.from_numpy(np.abs(np.random.randn(*shape).astype(np.float32))).to(
            device=device
        )
        for _ in range(N)
    ]
    state_steps = [torch.scalar_tensor(i) for i in range(N)]
    return params, grads, avg, avg_sq, state_steps


def _make_cuda_view(values, offset):
    """Copy values into a contiguous CUDA view with guarded storage."""
    storage = torch.empty(values.numel() + offset + 1, device="cuda")
    storage[:offset].fill_(_LEFT_SENTINEL)
    storage[offset : offset + values.numel()].copy_(values)
    storage[offset + values.numel() :].fill_(_RIGHT_SENTINEL)
    return storage[offset : offset + values.numel()], storage


class FusedAdamWTest(unittest.TestCase):
    def _assert_storage_guards_unchanged(self, tensors, storages, offsets):
        for tensor, storage, offset in zip(tensors, storages, offsets):
            torch.testing.assert_close(
                storage[:offset],
                torch.full_like(storage[:offset], _LEFT_SENTINEL),
            )
            torch.testing.assert_close(
                storage[offset + tensor.numel() :],
                torch.full_like(storage[offset + tensor.numel() :], _RIGHT_SENTINEL),
            )

    def _make_guarded_cuda_inputs(self, sizes, offsets):
        tensors = {name: [] for name in offsets}
        storages = {name: [] for name in offsets}
        for tensor_index, size in enumerate(sizes):
            values = {
                "params": torch.linspace(
                    -0.5 + tensor_index,
                    0.5 + tensor_index,
                    size,
                    device="cuda",
                ),
                "grads": torch.linspace(0.4, -0.3, size, device="cuda"),
                "avg": torch.linspace(-0.2, 0.3, size, device="cuda"),
                "avg_sq": torch.linspace(0.1, 0.9, size, device="cuda"),
            }
            for name in offsets:
                view, storage = _make_cuda_view(
                    values[name], offsets[name][tensor_index]
                )
                tensors[name].append(view)
                storages[name].append(storage)
        return tensors, storages

    def _assert_fused_cuda_matches_reference(self, sizes, offsets, num_steps):
        tensors, storages = self._make_guarded_cuda_inputs(sizes, offsets)
        params = tensors["params"]
        grads = tensors["grads"]
        avg = tensors["avg"]
        avg_sq = tensors["avg_sq"]
        state_steps = [torch.scalar_tensor(float(i)) for i in range(len(sizes))]

        for name, tensor_list in tensors.items():
            for tensor, offset in zip(tensor_list, offsets[name]):
                self.assertTrue(tensor.is_contiguous())
                expected_remainder = offset * tensor.element_size()
                self.assertEqual(
                    tensor.data_ptr() % _FLOAT_PACK_ALIGNMENT,
                    expected_remainder % _FLOAT_PACK_ALIGNMENT,
                )

        expected_params = [tensor.clone() for tensor in params]
        expected_grads = [tensor.clone() for tensor in grads]
        expected_avg = [tensor.clone() for tensor in avg]
        expected_avg_sq = [tensor.clone() for tensor in avg_sq]
        expected_steps = [tensor.clone() for tensor in state_steps]
        original_grad_storages = [storage.clone() for storage in storages["grads"]]

        weight_decay = 0.02
        lr = 0.001
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        for _ in range(num_steps):
            for param, grad, exp_avg, exp_avg_sq, step in zip(
                expected_params,
                expected_grads,
                expected_avg,
                expected_avg_sq,
                expected_steps,
            ):
                step.add_(1)
                param.mul_(1.0 - weight_decay)
                torch.ops.recis.adam_tf_apply(
                    param,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    step.item(),
                    lr,
                    beta1,
                    beta2,
                    eps,
                )

            torch.ops.recis.fused_adamw_tf_apply(
                params,
                grads,
                avg,
                avg_sq,
                state_steps,
                weight_decay,
                lr,
                beta1,
                beta2,
                eps,
            )
        torch.cuda.synchronize()

        for actual, expected in zip(params, expected_params):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(avg, expected_avg):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(avg_sq, expected_avg_sq):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(state_steps, expected_steps):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(grads, expected_grads):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(storages["grads"], original_grad_storages):
            torch.testing.assert_close(actual, expected)
        for name in ("params", "avg", "avg_sq"):
            self._assert_storage_guards_unchanged(
                tensors[name], storages[name], offsets[name]
            )

    def test_adamw_tf_cuda(self):
        for device in ["cpu", "cuda"]:
            logger.info(f"Testing fused_adamw_tf_apply on {device}")
            N = 10
            shape = (10,)
            params, grads, avg, avg_sq, state_steps = make_data(N, shape, device=device)
            lr_scalar = 0.001
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            weight_decay = 0.02

            for i in range(N):
                param, grad, exp_avg, exp_avg_sq = (
                    params[i],
                    grads[i],
                    avg[i],
                    avg_sq[i],
                )
                param.mul_(1.0 - weight_decay)
                step = state_steps[i] + 1
                torch.ops.recis.adam_tf_apply(
                    param,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    step.item(),
                    lr_scalar,
                    beta1,
                    beta2,
                    eps,
                )

            params_2, grads_2, avg_2, avg_sq_2, state_steps_2 = make_data(
                N, shape, device=device
            )
            torch.ops.recis.fused_adamw_tf_apply(
                params_2,
                grads_2,
                avg_2,
                avg_sq_2,
                state_steps_2,
                weight_decay,
                lr_scalar,
                beta1,
                beta2,
                eps,
            )

            for i in range(N):
                param, param_2 = params[i], params_2[i]
                self.assertTrue(torch.allclose(param, param_2))

            for i in range(N):
                exp_avg, exp_avg_2 = avg[i], avg_2[i]
                self.assertTrue(torch.allclose(exp_avg, exp_avg_2))

            for i in range(N):
                exp_avg_sq, exp_avg_sq_2 = avg_sq[i], avg_sq_2[i]
                self.assertTrue(torch.allclose(exp_avg_sq, exp_avg_sq_2))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_adamw_tf_cuda_handles_alignment_and_pack_boundaries(self):
        sizes = [1, 3, 4, 5, 8, 9, 11, 17]
        offsets = {
            "params": [1, 0, 0, 0, 0, 1, 0, 0],
            "grads": [0, 1, 0, 0, 0, 1, 2, 3],
            "avg": [0, 0, 1, 0, 0, 1, 0, 0],
            "avg_sq": [0, 0, 0, 1, 0, 1, 0, 0],
        }

        self._assert_fused_cuda_matches_reference(sizes, offsets, num_steps=3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_adamw_tf_cuda_pack_processing_crosses_thread_stride(self):
        multiprocessor_count = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        # The kernel caps its grid at eight 256-thread blocks per SM. The
        # extra packs force the first threads to take a second stride.
        thread_count_at_grid_cap = multiprocessor_count * 8 * 256
        size = thread_count_at_grid_cap * 4 + 5
        offsets = {
            "params": [0],
            "grads": [0],
            "avg": [0],
            "avg_sq": [0],
        }

        self._assert_fused_cuda_matches_reference([size], offsets, num_steps=2)


if __name__ == "__main__":
    unittest.main()
