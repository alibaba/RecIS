import torch

from recis.optim.sparse_optim import SparseOptimizer


class SparseAdagrad(SparseOptimizer):
    """Sparse Adagrad optimizer for efficient sparse parameter optimization.

    This class implements the Adagrad optimization algorithm specifically optimized
    for sparse parameters in recommendation systems. It extends the SparseOptimizer
    base class and uses RecIS's C++ implementation for maximum performance.

    The Adagrad algorithm provides adaptive learning rates by scaling updates based
    on the historical sum of squared gradients. For sparse parameters, this implementation
    only  updates parameters that have received gradients, making it highly efficient
    for large embedding tables where only a small fraction of parameters are active in
    each training step.

    Mathematical formulation:

    .. math::

        state_sum_{t} = state_sum_{t-1} + g_t^2 \n
        lr = lr / (1 + (step - 1) * lr_decay) \n
        θ_t = θ_{t-1} - lr * (g_t / (√state_sum_t + ε))

    Where:
        - θ: parameters
        - g: gradients
        - ε: eps
        - lr: learning rate
        - lr_decay: learning rate decay
        - state_sum: historical sum of squared gradients

    Example:
        Creating and using SparseAdagrad:

    .. code-block:: python

        # Initialize with custom hyperparameters
        optimizer = SparseAdagrad(
            param_dict=sparse_parameters,
            lr=0.001,  # Learning rate
            eps=1e-8,  # Numerical stability
        )

        # Training with gradient accumulation
        optimizer.set_grad_accum_steps(4)

        for batch in dataloader:
            loss = model(batch) / 4  # Scale for accumulation
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
    """

    def __init__(
        self,
        param_dict: dict,
        lr=1e-3,
        lr_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
        weight_decay: float = 0,
    ) -> None:
        """Initialize SparseAdam optimizer with specified hyperparameters.

        Args:
            param_dict (dict): Dictionary of sparse parameters to optimize.
                Keys are parameter names, values are parameter tensors (typically HashTables).
            lr (float, optional): Learning rate. Defaults to 1e-2.
            lr (float, Tensor, optional): learning rate (default: 1e-2)
            lr_decay (float, optional): learning rate decay (default: 0)
            initial_accumulator_value (float, optional): initial value of the sum of squares of gradients (default: 0)
            eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-10)

        Note:
            The SparseAdagrad implemented here operates on RecIS HashTables.
            Algorithmically, it is equivalent to the torch.optim.Adagrad optimizer
            applied to parameters within a standard torch.nn.Embedding layer.
        """
        super().__init__(lr=lr)

        # Store hyperparameters
        self._lr = lr
        self._lr_decay = lr_decay
        self._eps = eps
        self._weight_decay = weight_decay
        self._initial_accumulator_value = initial_accumulator_value

        # Create the underlying C++ optimizer implementation
        print(f"param_dict is {param_dict}")
        self._imp = torch.classes.recis.SparseAdagrad.make(
            param_dict,
            self._lr,
            self._lr_decay,
            self._initial_accumulator_value,
            self._eps,
            self._weight_decay,
        )
