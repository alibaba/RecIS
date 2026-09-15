import torch

from recis.optim.sparse_optim import SparseOptimizer


class SparseRowWiseAdagrad(SparseOptimizer):
    """Sparse row-wise Adagrad for RecIS ``HashTable`` parameters.

    The optimizer stores one accumulator per embedding row. For every active
    row, it adds the mean squared adjusted gradient to the accumulator and uses
    the resulting scalar denominator for every element in that row.

    Unlike dense TorchRec row-wise Adagrad, this sparse implementation updates
    only rows present in the current gradient. Consequently, ``weight_decay``
    is also applied only to active rows; inactive rows and their accumulators
    remain unchanged.

    Row accumulators are HashTable slots. Use the RecIS Saver/Loader path to
    restore them across different HashTable instances; direct
    ``load_state_dict`` restores tensor state only for the same table objects.
    Sparse optimizer checkpoints do not store hyperparameters, so resume with
    the same ``lr``, ``lr_decay``, ``eps``, ``weight_decay``, and ``maximize``
    values used when the checkpoint was written. Complete any ``add_params``
    calls before constructing a RecIS Saver.

    Args:
        param_dict (dict): Mapping from parameter names to RecIS HashTables.
        lr (float, optional): Learning rate. Defaults to ``1e-3``.
        lr_decay (float, optional): Learning-rate decay. Defaults to ``0``.
        initial_accumulator_value (float, optional): Initial value for each
            row accumulator. Defaults to ``0``.
        eps (float, optional): Term added to the denominator for numerical
            stability. Defaults to ``1e-10``.
        weight_decay (float, optional): L2 penalty applied to active rows.
            Defaults to ``0``.
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
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
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
        self._imp = torch.classes.recis.SparseRowWiseAdagrad.make(
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
