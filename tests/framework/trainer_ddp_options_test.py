import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from recis.framework.trainer import Trainer, TrainingArguments


class TrainerDdpOptionsTest(unittest.TestCase):
    def _construct_trainer(self, *args, **kwargs):
        ddp_kwargs_factory = MagicMock(return_value=object())
        accelerator = SimpleNamespace(prepare=lambda *values: values)
        scalar = MagicMock()
        scalar.cuda.return_value = scalar

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "recis.framework.trainer.DistributedDataParallelKwargs",
                    ddp_kwargs_factory,
                )
            )
            stack.enter_context(
                patch(
                    "recis.framework.trainer.InitProcessGroupKwargs",
                    return_value=object(),
                )
            )
            stack.enter_context(
                patch(
                    "recis.framework.trainer.Accelerator",
                    return_value=accelerator,
                )
            )
            stack.enter_context(
                patch(
                    "recis.framework.trainer.setup_pipeline_prefetch",
                    return_value=(None, 0),
                )
            )
            stack.enter_context(patch.object(Trainer, "_setup_sparse_forward_notify"))
            init_saver = stack.enter_context(
                patch.object(Trainer, "init_saver", return_value=object())
            )
            stack.enter_context(patch.object(Trainer, "init_hooks"))
            stack.enter_context(
                patch("recis.framework.trainer.MonitorReporter.report_forward")
            )
            stack.enter_context(
                patch(
                    "recis.framework.trainer.torch.scalar_tensor",
                    return_value=scalar,
                )
            )
            trainer = Trainer(*args, **kwargs)

        return trainer, ddp_kwargs_factory.call_args.kwargs, init_saver.call_args

    def test_ddp_options_keep_existing_defaults(self):
        _, ddp_options, _ = self._construct_trainer(
            model=object(),
            args=TrainingArguments(),
            dense_optimizers=(object(), None),
        )

        self.assertEqual(
            ddp_options,
            {
                "find_unused_parameters": True,
                "broadcast_buffers": True,
                "gradient_as_bucket_view": False,
                "bucket_cap_mb": 25,
                "static_graph": False,
            },
        )

    def test_ddp_options_pass_explicit_values_together(self):
        _, ddp_options, _ = self._construct_trainer(
            model=object(),
            args=TrainingArguments(),
            dense_optimizers=(object(), None),
            ddp_find_unused_parameters=False,
            ddp_broadcast_buffers=False,
            ddp_gradient_as_bucket_view=True,
            ddp_bucket_cap_mb=4,
            ddp_static_graph=True,
        )

        self.assertEqual(
            ddp_options,
            {
                "find_unused_parameters": False,
                "broadcast_buffers": False,
                "gradient_as_bucket_view": True,
                "bucket_cap_mb": 4,
                "static_graph": True,
            },
        )

    def test_saver_keeps_its_existing_positional_slot(self):
        model = object()
        training_args = TrainingArguments()
        saver = object()
        _, ddp_options, init_saver_call = self._construct_trainer(
            model,
            training_args,
            None,
            None,
            None,
            (object(), None),
            None,
            False,
            True,
            False,
            saver,
        )

        self.assertEqual(
            ddp_options,
            {
                "find_unused_parameters": True,
                "broadcast_buffers": False,
                "gradient_as_bucket_view": False,
                "bucket_cap_mb": 25,
                "static_graph": False,
            },
        )
        self.assertEqual(init_saver_call.args, (model, training_args, saver))


if __name__ == "__main__":
    unittest.main()
