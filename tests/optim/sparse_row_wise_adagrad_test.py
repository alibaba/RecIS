import itertools
import tempfile
import unittest
from importlib import metadata
from unittest import mock

import torch

from recis.framework.checkpoint_manager import Saver, SaverOptions
from recis.nn.modules.hashtable import HashTable, filter_out_sparse_param
from recis.optim.sparse_row_wise_adagrad import SparseRowWiseAdagrad
from recis.optim.sparse_row_wise_adagrad_sum import SparseRowWiseAdagradSum


try:
    from torchrec.optim.rowwise_adagrad import RowWiseAdagrad as TorchRecRowWiseAdagrad

    _TORCHREC_VERSION = metadata.version("torchrec")
    _TORCHREC_ERROR = ""
except Exception as error:
    TorchRecRowWiseAdagrad = None
    _TORCHREC_VERSION = ""
    _TORCHREC_ERROR = str(error)


_STATE_SUM_NAME = "sparse_row_wise_adagrad_state_sum"
_SUM_STATE_SUM_NAME = "sparse_row_wise_adagrad_sum_state_sum"
_LEGACY_STEP_NAME = "sparse_row_wise_adagrad_step"
_SUM_LEGACY_STEP_NAME = "sparse_row_wise_adagrad_sum_step"
_STEP_PREFIX = "sparse_row_wise_adagrad_"
_STEP_SUFFIX = "_step"
_TABLE_COUNTER = itertools.count()


def _torchrec_1_3_available() -> bool:
    return TorchRecRowWiseAdagrad is not None and _TORCHREC_VERSION.startswith("1.3.")


class SparseRowWiseAdagradTest(unittest.TestCase):
    def _make_table(
        self,
        embedding_shape,
        *,
        label: str,
        block_size: int = 16,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        name = f"row_wise_adagrad_{label}_{next(_TABLE_COUNTER)}"
        table = HashTable(
            embedding_shape,
            block_size=block_size,
            dtype=dtype,
            device=torch.device(device),
            name=name,
        )
        return name, table

    def _take_step(self, table, optimizer, ids, gradients=None) -> None:
        ids = torch.as_tensor(ids, dtype=torch.long, device=table.device)
        output = table(ids)
        if gradients is None:
            gradients = torch.ones_like(output)
        else:
            gradients = gradients.to(device=output.device, dtype=output.dtype)
        (output * gradients).sum().backward()
        optimizer.step()
        optimizer.zero_grad()

    def _read_embeddings(self, table, ids) -> torch.Tensor:
        ids = torch.as_tensor(ids, dtype=torch.long, device=table.device)
        was_training = table.training
        table.eval()
        try:
            with torch.no_grad():
                return table(ids).detach()
        finally:
            table.train(was_training)

    def _row_indices(self, table, ids) -> torch.Tensor:
        table_ids, table_indices, _ = table.snap_shot()
        index_by_id = dict(
            zip(
                table_ids.detach().cpu().tolist(),
                table_indices.detach().cpu().tolist(),
            )
        )
        return torch.tensor(
            [index_by_id[int(row_id)] for row_id in ids],
            dtype=torch.long,
            device=table.device,
        )

    def _read_state_rows(
        self, table, ids, state_sum_name=_STATE_SUM_NAME
    ) -> torch.Tensor:
        state = table.slot_group().slot_by_name(state_sum_name).value()
        indices = self._row_indices(table, ids).to(state.device)
        return state.index_select(0, indices)

    def _step_values(self, state_dict) -> dict:
        return {
            key: int(value.item())
            for key, value in state_dict.items()
            if key == _LEGACY_STEP_NAME
            or (key.startswith(_STEP_PREFIX) and key.endswith(_STEP_SUFFIX))
        }

    def _run_formula_parity_test(self, device: str) -> None:
        dimension = 7
        ids = torch.tensor([2, 5, 2, 9], device=device)
        gradients = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                [2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
                [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
                [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0],
            ]
        )
        _, table = self._make_table(
            [dimension],
            block_size=2,
            device=device,
            label=f"formula_{device}",
        )
        optimizer = SparseRowWiseAdagrad(
            filter_out_sparse_param(table),
            lr=0.1,
            lr_decay=0.02,
            initial_accumulator_value=0.3,
            eps=1e-4,
            weight_decay=0.07,
        )

        unique_ids = [2, 5, 9]
        expected_grad = torch.stack(
            (gradients[0] + gradients[2], gradients[1], gradients[3])
        )
        expected_state = torch.full((3, 1), 0.3)
        expected_embedding = torch.zeros((3, dimension))
        for step in range(1, 4):
            self._take_step(table, optimizer, ids, gradients)

            adjusted_grad = expected_grad + 0.07 * expected_embedding
            expected_state += adjusted_grad.square().mean(dim=1, keepdim=True)
            effective_lr = 0.1 / (1 + (step - 1) * 0.02)
            expected_embedding -= (
                effective_lr * adjusted_grad / (expected_state.sqrt() + 1e-4)
            )

        torch.testing.assert_close(
            self._read_embeddings(table, unique_ids).cpu(),
            expected_embedding,
            rtol=2e-5,
            atol=2e-6,
        )
        actual_state = self._read_state_rows(table, unique_ids).cpu()
        torch.testing.assert_close(
            actual_state,
            expected_state,
            rtol=2e-5,
            atol=2e-6,
        )
        self.assertEqual(tuple(actual_state.shape), (3, 1))
        self.assertEqual(
            self._step_values(optimizer.state_dict()),
            {_LEGACY_STEP_NAME: 3},
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_sum_uses_grad_sq_for_row_state(self) -> None:
        _, table = self._make_table(
            [3],
            block_size=2,
            device="cuda",
            label="sum_manual_grad_sq",
        )
        ids = torch.tensor([7], device=table.device)
        table.insert(ids, torch.zeros((1, 3), device=table.device))
        table_ids, table_indices, _ = table.snap_shot()
        row_by_id = dict(
            zip(table_ids.detach().cpu().tolist(), table_indices.cpu().tolist())
        )
        row_index = torch.tensor([row_by_id[7]], dtype=torch.long, device="cuda")

        optimizer = SparseRowWiseAdagradSum(
            filter_out_sparse_param(table),
            lr=0.1,
            initial_accumulator_value=0.5,
            eps=0.0,
        )
        grad = torch.tensor([[3.0, 4.0, 0.0]], device="cuda")
        grad_sq = torch.tensor([[5.0, 8.0, 1.0]], device="cuda")

        table.accept_grad(row_index, grad)
        table._hashtable_impl.accept_grad_sq(row_index, grad_sq)
        optimizer.step()
        optimizer.zero_grad()

        expected_state = torch.tensor([[0.5 + (5.0 + 8.0 + 1.0) / 3.0]])
        expected_embedding = -0.1 * grad.cpu() / expected_state.sqrt()
        actual_state = self._read_state_rows(table, [7], _SUM_STATE_SUM_NAME).cpu()
        torch.testing.assert_close(actual_state, expected_state)
        torch.testing.assert_close(
            self._read_embeddings(table, [7]).cpu(), expected_embedding
        )
        self.assertEqual(
            self._step_values(optimizer.state_dict()),
            {_SUM_LEGACY_STEP_NAME: 1},
        )

    def test_sum_rejects_missing_grad_sq(self) -> None:
        _, table = self._make_table([3], label="sum_missing_grad_sq")
        optimizer = SparseRowWiseAdagradSum(filter_out_sparse_param(table))
        table._hashtable_impl.accept_grad(torch.tensor([0]), torch.ones((1, 3)))

        with self.assertRaisesRegex(RuntimeError, "requires grad_sq"):
            optimizer.step()

    def test_sum_zero_grad_respects_accumulation_boundary(self) -> None:
        optimizer = SparseRowWiseAdagradSum.__new__(SparseRowWiseAdagradSum)
        optimizer._grad_accum_steps = 2
        optimizer._local_step = 1
        optimizer._imp = mock.Mock()

        optimizer.zero_grad()
        optimizer._imp.zero_grad.assert_not_called()
        optimizer._imp.zero_grad_sq.assert_not_called()

        optimizer._local_step = 2
        optimizer.zero_grad()
        optimizer._imp.zero_grad.assert_called_once_with(None)
        optimizer._imp.zero_grad_sq.assert_called_once_with()

    @unittest.skipUnless(torch.cuda.is_available(), "RecIS lookup requires CUDA")
    def test_cpu_formula_parity_across_blocks(self) -> None:
        self._run_formula_parity_test("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_formula_parity_across_blocks(self) -> None:
        self._run_formula_parity_test("cuda")

    @unittest.skipUnless(
        _torchrec_1_3_available() and torch.cuda.is_available(),
        f"TorchRec 1.3 oracle is unavailable: {_TORCHREC_ERROR or _TORCHREC_VERSION}",
    )
    def test_matches_torchrec_1_3_for_active_rows(self) -> None:
        dimension = 7
        recis_ids = torch.tensor([2, 5, 2, 9], device="cuda")
        torchrec_ids = torch.tensor([0, 1, 0, 2], device="cuda")
        gradients = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                [2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
                [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
                [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0],
            ],
            device="cuda",
        )
        _, table = self._make_table(
            [dimension], block_size=2, device="cuda", label="torchrec_oracle"
        )
        recis_optimizer = SparseRowWiseAdagrad(
            filter_out_sparse_param(table),
            lr=0.1,
            lr_decay=0.02,
            initial_accumulator_value=0.3,
            eps=1e-4,
            weight_decay=0.07,
        )
        weight = torch.nn.Parameter(torch.zeros((3, dimension), device="cuda"))
        torchrec_optimizer = TorchRecRowWiseAdagrad(
            [weight],
            lr=0.1,
            lr_decay=0.02,
            initial_accumulator_value=0.3,
            eps=1e-4,
            weight_decay=0.07,
        )

        for _ in range(3):
            self._take_step(table, recis_optimizer, recis_ids, gradients)
            oracle_output = weight.index_select(0, torchrec_ids)
            (oracle_output * gradients).sum().backward()
            torchrec_optimizer.step()
            torchrec_optimizer.zero_grad()

        torch.testing.assert_close(
            self._read_embeddings(table, [2, 5, 9]),
            weight,
            rtol=2e-5,
            atol=2e-6,
        )
        torch.testing.assert_close(
            self._read_state_rows(table, [2, 5, 9]),
            torchrec_optimizer.state[weight]["sum"],
            rtol=2e-5,
            atol=2e-6,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "RecIS lookup requires CUDA")
    def test_state_round_trip_preserves_independent_table_steps(self) -> None:
        first_name, first_table = self._make_table([3], label="state_first")
        optimizer = SparseRowWiseAdagrad(
            filter_out_sparse_param(first_table), lr=0.1, lr_decay=0.1
        )
        self._take_step(first_table, optimizer, [1])
        self._take_step(first_table, optimizer, [1])

        second_name, second_table = self._make_table([3], label="state_second")
        optimizer.add_params(filter_out_sparse_param(second_table))
        self._take_step(second_table, optimizer, [2])

        first_step_name = f"{_STEP_PREFIX}{first_name}{_STEP_SUFFIX}"
        second_step_name = f"{_STEP_PREFIX}{second_name}{_STEP_SUFFIX}"
        expected_steps = {first_step_name: 2, second_step_name: 1}
        state_dict = optimizer.state_dict()
        self.assertIn(first_name, state_dict)
        self.assertIn(second_name, state_dict)
        self.assertEqual(self._step_values(state_dict), expected_steps)
        saved_state = {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in state_dict.items()
        }

        optimizer.reset_state_dict()
        self.assertEqual(
            self._step_values(optimizer.state_dict()),
            {first_step_name: 0, second_step_name: 0},
        )
        optimizer.load_state_dict(saved_state)
        self.assertEqual(self._step_values(optimizer.state_dict()), expected_steps)

        self._take_step(second_table, optimizer, [2])
        expected_second_embedding = -0.1 - 0.1 / (1 + 0.1) / (2.0**0.5 + 1e-10)
        torch.testing.assert_close(
            self._read_embeddings(second_table, [2]).cpu(),
            torch.full((1, 3), expected_second_embedding),
        )
        self.assertEqual(
            self._step_values(optimizer.state_dict()),
            {first_step_name: 2, second_step_name: 2},
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_saver_round_trip_with_multiple_tables(self) -> None:
        first_name, first_table = self._make_table(
            [3], block_size=2, device="cuda", label="save_first"
        )
        model = torch.nn.Module()
        model.add_module("first_table", first_table)
        optimizer = SparseRowWiseAdagrad(
            filter_out_sparse_param(model), lr=0.1, lr_decay=0.1
        )

        second_name, second_table = self._make_table(
            [3], block_size=2, device="cuda", label="save_second"
        )
        model.add_module("second_table", second_table)
        optimizer.add_params(filter_out_sparse_param(second_table))

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            saver = Saver(
                SaverOptions(
                    model=model,
                    sparse_optim=optimizer,
                    output_dir=checkpoint_dir,
                    max_keep=2,
                    concurrency=1,
                )
            )

            self._take_step(first_table, optimizer, [1])
            self._take_step(first_table, optimizer, [1])
            self._take_step(second_table, optimizer, [2])

            expected_first_embedding = self._read_embeddings(first_table, [1]).clone()
            expected_first_state = self._read_state_rows(first_table, [1]).clone()
            expected_second_embedding = self._read_embeddings(second_table, [2]).clone()
            expected_second_state = self._read_state_rows(second_table, [2]).clone()
            expected_steps = {
                f"{_STEP_PREFIX}{first_name}{_STEP_SUFFIX}": 2,
                f"{_STEP_PREFIX}{second_name}{_STEP_SUFFIX}": 1,
            }

            saver.save("row_wise_multi_table")
            first_table.clear()
            second_table.clear()
            optimizer.reset_state_dict()

            checkpoint_path = f"{checkpoint_dir}/row_wise_multi_table"
            saver._init_model_bank(
                [
                    {
                        "path": checkpoint_path,
                        "load": ["*"],
                        "exclude": ["io_state"],
                        "is_dynamic": False,
                    }
                ]
            )
            saver.restore()

            torch.testing.assert_close(
                self._read_embeddings(first_table, [1]),
                expected_first_embedding,
            )
            torch.testing.assert_close(
                self._read_state_rows(first_table, [1]),
                expected_first_state,
            )
            torch.testing.assert_close(
                self._read_embeddings(second_table, [2]),
                expected_second_embedding,
            )
            torch.testing.assert_close(
                self._read_state_rows(second_table, [2]),
                expected_second_state,
            )
            self.assertEqual(self._step_values(optimizer.state_dict()), expected_steps)

    def test_single_table_loads_legacy_step(self) -> None:
        table_name, table = self._make_table([3], label="legacy_state")
        optimizer = SparseRowWiseAdagrad(filter_out_sparse_param(table))
        state_dict = optimizer.state_dict()
        self.assertEqual(self._step_values(state_dict), {_LEGACY_STEP_NAME: 0})
        live_step = state_dict[_LEGACY_STEP_NAME]
        loaded_state = state_dict.copy()
        loaded_state[_LEGACY_STEP_NAME] = torch.tensor([7], dtype=torch.int64)

        optimizer.load_state_dict(loaded_state)

        self.assertEqual(live_step.item(), 7)
        self.assertEqual(
            self._step_values(optimizer.state_dict()),
            {_LEGACY_STEP_NAME: 7},
        )

        ambiguous_state = optimizer.state_dict()
        ambiguous_state[f"{_STEP_PREFIX}{table_name}{_STEP_SUFFIX}"] = torch.tensor(
            [8], dtype=torch.int64
        )
        with self.assertRaises(RuntimeError):
            optimizer.load_state_dict(ambiguous_state)

        for invalid_step in (-0.5, 1.5, float("nan"), float("inf")):
            invalid_state = optimizer.state_dict()
            invalid_state[_LEGACY_STEP_NAME] = torch.tensor([invalid_step])
            with self.subTest(invalid_step=invalid_step):
                with self.assertRaises(RuntimeError):
                    optimizer.load_state_dict(invalid_state)

    def test_direct_load_rejects_foreign_hashtable_state(self) -> None:
        table_name, table = self._make_table([3], label="load_target")
        optimizer = SparseRowWiseAdagrad(filter_out_sparse_param(table))
        state_dict = optimizer.state_dict()

        foreign_name, foreign_table = self._make_table([3], label="load_foreign")
        state_dict[table_name] = filter_out_sparse_param(foreign_table)[foreign_name]

        with self.assertRaises(RuntimeError):
            optimizer.load_state_dict(state_dict)

    @unittest.skipUnless(torch.cuda.is_available(), "RecIS lookup requires CUDA")
    def test_existing_rows_keep_weight_decay_sparse(self) -> None:
        _, table = self._make_table([3], block_size=2, label="existing_rows")
        ids = torch.tensor([1, 2])
        table.insert(ids, torch.ones((2, 3)))
        optimizer = SparseRowWiseAdagrad(
            filter_out_sparse_param(table),
            lr=0.2,
            initial_accumulator_value=0.5,
            eps=1e-4,
            weight_decay=0.1,
        )

        self._take_step(table, optimizer, [1], torch.zeros((1, 3)))

        expected_active_state = torch.tensor([[0.51]])
        expected_active = torch.full((3,), 1.0 - 0.2 * 0.1 / (0.51**0.5 + 1e-4))
        actual = self._read_embeddings(table, [1, 2]).cpu()
        torch.testing.assert_close(actual[0], expected_active)
        torch.testing.assert_close(actual[1], torch.ones(3))
        torch.testing.assert_close(
            self._read_state_rows(table, [1, 2]).cpu(),
            torch.cat((expected_active_state, torch.tensor([[0.5]]))),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "RecIS lookup requires CUDA")
    def test_add_params_initializes_preallocated_blocks(self) -> None:
        _, primary = self._make_table([3], block_size=2, label="add_primary")
        optimizer = SparseRowWiseAdagrad(filter_out_sparse_param(primary), lr=0.1)

        _, added = self._make_table([3], block_size=2, label="add_existing")
        ids = torch.tensor([10, 11, 12])
        added.insert(ids, torch.zeros((3, 3)))
        optimizer.add_params(filter_out_sparse_param(added))
        self._take_step(added, optimizer, ids, torch.full((3, 3), 2.0))

        expected_state = torch.full((3, 1), 4.0)
        expected_embedding = torch.full((3, 3), -0.1)
        torch.testing.assert_close(
            self._read_embeddings(added, ids).cpu(), expected_embedding
        )
        torch.testing.assert_close(
            self._read_state_rows(added, ids).cpu(), expected_state
        )

    def test_reused_state_slot_updates_new_row_initializer(self) -> None:
        _, table = self._make_table([3], block_size=1, label="reused_slot")
        SparseRowWiseAdagrad(
            filter_out_sparse_param(table), initial_accumulator_value=0.25
        )
        table.insert(torch.tensor([1]), torch.zeros((1, 3)))

        SparseRowWiseAdagrad(
            filter_out_sparse_param(table), initial_accumulator_value=0.75
        )
        table.insert(torch.tensor([2]), torch.zeros((1, 3)))

        torch.testing.assert_close(
            self._read_state_rows(table, [1, 2]).cpu(),
            torch.tensor([[0.25], [0.75]]),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "RecIS lookup requires CUDA")
    def test_multidimensional_embedding_uses_full_row_mean(self) -> None:
        _, table = self._make_table([2, 3], block_size=2, label="multidimensional")
        optimizer = SparseRowWiseAdagrad(
            filter_out_sparse_param(table),
            lr=0.1,
            initial_accumulator_value=0.2,
            eps=1e-4,
        )
        ids = torch.tensor([4, 8])
        gradients = torch.arange(1.0, 13.0).reshape(2, 2, 3)

        table.insert(ids, torch.zeros_like(gradients))
        row_indices = self._row_indices(table, ids)
        table.accept_grad(
            row_indices,
            gradients.to(device=table.device),
        )
        optimizer.step()
        optimizer.zero_grad()

        expected_state = 0.2 + gradients.square().reshape(2, -1).mean(
            dim=1, keepdim=True
        )
        expected_embedding = (
            -0.1 * gradients / (expected_state.sqrt().reshape(2, 1, 1) + 1e-4)
        )
        torch.testing.assert_close(
            table.raw_embeddings().index_select(0, row_indices).cpu(),
            expected_embedding,
        )
        torch.testing.assert_close(
            self._read_state_rows(table, ids).cpu(), expected_state
        )

    @unittest.skipUnless(torch.cuda.is_available(), "RecIS lookup requires CUDA")
    def test_empty_batch_on_unallocated_table(self) -> None:
        _, table = self._make_table([4], label="empty_batch")
        optimizer = SparseRowWiseAdagrad(filter_out_sparse_param(table))

        self._take_step(table, optimizer, torch.empty(0, dtype=torch.long))

        self.assertEqual(table.ids().numel(), 0)
        self.assertEqual(
            self._step_values(optimizer.state_dict()),
            {_LEGACY_STEP_NAME: 1},
        )

    def test_default_learning_rate_matches_sparse_adagrad(self) -> None:
        _, table = self._make_table([4], label="default_lr")
        optimizer = SparseRowWiseAdagrad(filter_out_sparse_param(table))

        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)
        self.assertEqual(optimizer._lr, 1e-3)
        self.assertFalse(optimizer._maximize)

    def test_step_runs_closure_with_grad_and_returns_loss(self) -> None:
        _, table = self._make_table([4], label="closure")
        optimizer = SparseRowWiseAdagrad(filter_out_sparse_param(table))
        closure_calls = []

        def closure():
            closure_calls.append(torch.is_grad_enabled())
            return torch.tensor(3.0, requires_grad=True)

        loss = optimizer.step(closure)

        self.assertEqual(closure_calls, [True])
        self.assertEqual(loss.item(), 3.0)

    @unittest.skipUnless(torch.cuda.is_available(), "RecIS lookup requires CUDA")
    def test_maximize_reverses_the_gradient_update(self) -> None:
        _, table = self._make_table([4], label="maximize")
        initial_embedding = torch.arange(1.0, 5.0).reshape(1, 4)
        table.insert(torch.tensor([1]), initial_embedding)
        optimizer = SparseRowWiseAdagrad(
            filter_out_sparse_param(table),
            lr=0.1,
            initial_accumulator_value=0.5,
            eps=0,
            weight_decay=0.25,
            maximize=True,
        )

        self._take_step(table, optimizer, [1], torch.ones((1, 4)))

        adjusted_grad = -torch.ones((1, 4)) + 0.25 * initial_embedding
        expected_state = 0.5 + adjusted_grad.square().mean(dim=1, keepdim=True)
        expected_embedding = initial_embedding - (
            0.1 * adjusted_grad / expected_state.sqrt()
        )
        torch.testing.assert_close(
            self._read_embeddings(table, [1]).cpu(),
            expected_embedding,
        )
        torch.testing.assert_close(
            self._read_state_rows(table, [1]).cpu(),
            expected_state,
        )

    def test_rejects_invalid_hyperparameters(self) -> None:
        invalid_values = {
            "lr": -0.1,
            "lr_decay": -0.1,
            "initial_accumulator_value": -0.1,
            "eps": -0.1,
            "weight_decay": -0.1,
        }
        for argument, value in invalid_values.items():
            for invalid_value in (value, float("nan")):
                with self.subTest(argument=argument, value=invalid_value):
                    _, table = self._make_table([4], label=f"invalid_{argument}")
                    with self.assertRaises(ValueError):
                        SparseRowWiseAdagrad(
                            filter_out_sparse_param(table),
                            **{argument: invalid_value},
                        )

    def test_rejects_empty_parameter_dict(self) -> None:
        with self.assertRaises(ValueError):
            SparseRowWiseAdagrad({})

    def test_add_params_rejects_existing_parameter_name(self) -> None:
        _, table = self._make_table([4], label="duplicate_param")
        params = filter_out_sparse_param(table)
        optimizer = SparseRowWiseAdagrad(params)

        with self.assertRaises(RuntimeError):
            optimizer.add_params(params)

    def _run_low_precision_zero_gradient_test(
        self, device: str, dtype: torch.dtype
    ) -> None:
        _, table = self._make_table(
            [4],
            block_size=2,
            device=device,
            dtype=dtype,
            label=f"zero_{device}_{dtype}",
        )
        optimizer = SparseRowWiseAdagrad(filter_out_sparse_param(table))
        ids = torch.tensor([1], dtype=torch.long, device=table.device)
        initial_embedding = torch.zeros((1, 4), dtype=dtype, device=table.device)
        table.insert(ids, initial_embedding)
        row_indices = torch.zeros(1, dtype=torch.long, device=table.device)
        table.accept_grad(row_indices, torch.zeros_like(initial_embedding))
        optimizer.step()
        optimizer.zero_grad()

        embedding = table.raw_embeddings().index_select(0, row_indices)
        state = (
            table.slot_group()
            .slot_by_name(_STATE_SUM_NAME)
            .value()
            .index_select(0, row_indices)
        )
        self.assertTrue(torch.isfinite(embedding).all())
        self.assertTrue(torch.isfinite(state).all())
        torch.testing.assert_close(embedding, torch.zeros_like(embedding))
        torch.testing.assert_close(state, torch.zeros_like(state))

    @unittest.skipUnless(torch.cuda.is_available(), "RecIS lookup requires CUDA")
    def test_cpu_float16_zero_gradient_is_finite(self) -> None:
        self._run_low_precision_zero_gradient_test("cpu", torch.float16)

    @unittest.skipUnless(torch.cuda.is_available(), "RecIS lookup requires CUDA")
    def test_cpu_bfloat16_zero_gradient_is_finite(self) -> None:
        self._run_low_precision_zero_gradient_test("cpu", torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_float16_zero_gradient_is_finite(self) -> None:
        self._run_low_precision_zero_gradient_test("cuda", torch.float16)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "CUDA BF16 support is required",
    )
    def test_cuda_bfloat16_zero_gradient_is_finite(self) -> None:
        self._run_low_precision_zero_gradient_test("cuda", torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
