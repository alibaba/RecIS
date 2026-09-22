import torch

from recis.optim.sparse_optim import SparseOptimizer


class SparseRowWiseAdagradSum(SparseOptimizer):
    """Sparse row-wise Adagrad using externally supplied gradient-square sums.

    This optimizer is intended for grouped sparse-gradient training. HashTable
    backward with ``grad_reduce_by="group_sum"`` provides both ``Grad()`` and
    ``GradSq()``:

    - ``Grad()`` is ``sum_k(group_grad_k)`` and is used for parameter updates.
    - ``GradSq()`` is ``sum_k(group_grad_k ** 2)`` and is used for the row-wise
      accumulator update.

    The row-wise accumulator remains one scalar per embedding row:

    ``row_state_sum += mean_dim(GradSq())``.

    Args:
        param_dict (dict): Mapping from parameter names to RecIS HashTables.
        lr (float, optional): Learning rate. Defaults to ``1e-3``.
        lr_decay (float, optional): Learning-rate decay. Defaults to ``0``.
        initial_accumulator_value (float, optional): Initial value for each
            row accumulator. Defaults to ``0``.
        eps (float, optional): Term added to the denominator for numerical
            stability. Defaults to ``1e-10``.
        weight_decay (float, optional): Not supported. Must be ``0``.
        maximize (bool, optional): Maximize the objective instead of minimizing
            it. Defaults to ``False``.
    """

    def __init__(
        self,
        param_dict: dict,
        lr: float = 1e-3,
        lr_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
        weight_decay: float = 0,
        *,
        maximize: bool = False,
    ) -> None:
        if not param_dict:
            raise ValueError("optimizer got an empty parameter list")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= lr_decay:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if weight_decay != 0:
            raise ValueError(
                "SparseRowWiseAdagradSum does not support weight_decay; "
                f"got {weight_decay}"
            )
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                f"Invalid initial_accumulator_value value: {initial_accumulator_value}"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        super().__init__(lr=lr)
        self._lr = lr
        self._lr_decay = lr_decay
        self._initial_accumulator_value = initial_accumulator_value
        self._eps = eps
        self._weight_decay = weight_decay
        self._maximize = maximize
        self._imp = torch.classes.recis.SparseRowWiseAdagradSum.make(
            param_dict,
            self._lr,
            self._lr_decay,
            self._initial_accumulator_value,
            self._eps,
            self._weight_decay,
            self._maximize,
        )

    def step(self, closure=None):
        """Perform an optimizer step and return the optional closure loss."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        super().step()
        return loss

    def zero_grad(self):
        """Clear both gradients and gradient-square buffers."""
        assert self._grad_accum_steps > 0
        if self._local_step % self._grad_accum_steps == 0:
            self._imp.zero_grad(None)
            self._imp.zero_grad_sq()
