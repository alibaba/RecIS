from torch.optim.optimizer import Optimizer

from recis.nn.modules.hashtable import split_sparse_dense_state_dict


class SparseOptimizer(Optimizer):
    """Base class for all sparse optimizers in RecIS.

    This class provides the common interface and functionality for sparse parameter
    optimization in recommendation systems. It extends PyTorch's Optimizer class
    to provide specialized support for HashTable parameters, gradient accumulation,
    and state management.

    The SparseOptimizer is designed to work with RecIS's C++ optimizer implementations
    through the _imp attribute, which provides the actual optimization algorithms
    optimized for sparse parameter updates.

    Attributes:
        _imp: The underlying C++ optimizer implementation.
        _local_step (int): Current local step counter for gradient accumulation.
        _grad_accum_steps (int): Number of steps to accumulate gradients before updating.

    Example:
        Basic usage pattern for subclasses:

    .. code-block:: python
        class MyOptimizer(SparseOptimizer):
            def __init__(self, param_dict, lr=0.01):
                super().__init__(lr=lr)
                # Create C++ implementation
                self._imp = torch.classes.recis.MyOptimizer.make(param_dict, lr)


        # Usage
        optimizer = MyOptimizer(sparse_params, lr=0.001)

        # Training loop with gradient accumulation
        optimizer.set_grad_accum_steps(4)
        for i, batch in enumerate(dataloader):
            loss = model(batch) / 4  # Scale loss for accumulation
            loss.backward()

            optimizer.step()  # Only updates every 4 steps
            optimizer.zero_grad()  # Only zeros every 4 steps


        State management:

    .. code-block:: python
        # Save optimizer state
        state = optimizer.state_dict()
        torch.save(state, "optimizer_state.pt")

        # Load optimizer state
        state = torch.load("optimizer_state.pt")
        optimizer.load_state_dict(state)

    """

    def __init__(self, lr=0.01) -> None:
        """Initialize SparseOptimizer with basic configuration.

        Args:
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.01.

        Note:
            Subclasses should call this constructor and then initialize their
            specific _imp attribute with the actual optimizer implementation.
        """
        self._imp = None
        self._local_step = 0
        self._grad_accum_steps = 1

        lr_param_groups = [{"initial_lr": lr, "lr": lr, "params": []}]
        super().__init__(lr_param_groups, defaults={})

    def add_params(self, params: dict):
        self._imp.add_parameters(params)

    def filter_param_dict(self, param_dict):
        new_param_dict = {}
        values = []
        for key, value in param_dict.items():
            if value in values:
                continue
            new_param_dict[key] = value
            values.append(value)
        return new_param_dict

    def step(self):
        """Perform a single optimization step with gradient accumulation support.

        This method implements gradient accumulation by only performing the actual
        parameter update every _grad_accum_steps steps. It maintains an internal
        step counter and delegates the actual optimization to the underlying
        C++ implementation.

        Note:
            When gradient accumulation is enabled (_grad_accum_steps > 1), this
            method only performs the actual parameter update every _grad_accum_steps
            calls. The learning rate is automatically handled by the implementation.
        """
        assert self._grad_accum_steps > 0
        self._local_step += 1
        # Only update parameters when accumulation steps are reached
        if self._local_step % self._grad_accum_steps == 0:
            lr = self.param_groups[0]["lr"]
            self._imp.set_lr(lr)
            self._imp.step()

    def zero_grad(self):
        """Clear gradients with gradient accumulation support.

        This method clears parameter gradients, but only when gradient accumulation
        steps are completed. This ensures that gradients are properly accumulated
        across multiple forward passes before being cleared.

        Note:
            When gradient accumulation is enabled, this method only clears
            gradients every _grad_accum_steps calls, synchronized with the
            step() method.
        """
        assert self._grad_accum_steps > 0

        # Only clear gradients when accumulation steps are reached
        if self._local_step % self._grad_accum_steps == 0:
            self._imp.zero_grad(None)

    def state_dict(self) -> dict:
        """Return the optimizer state as a dictionary.

        This method returns the complete state of the optimizer, including
        both HashTable parameters and step counters, which can be saved
        for checkpointing and later restored.

        Returns:
            dict: Dictionary containing the optimizer state with:
                - HashTable parameters and their optimization states
                - Step counters and other internal state variables
        """
        hashtables, steps = self._imp.state_dict()
        ret = {}
        ret.update(hashtables)
        ret.update(steps)
        return ret

    def load_state_dict(self, state_dict: dict):
        """Load optimizer state from a dictionary.

        This method restores the optimizer state from a previously saved
        state dictionary, enabling resumption of training from checkpoints.

        Args:
            state_dict (dict): Dictionary containing the optimizer state,
                typically obtained from a previous state_dict() call.
        """
        local_state_dict = state_dict.copy()
        hashtables, steps = split_sparse_dense_state_dict(local_state_dict)
        self._imp.load_state_dict(hashtables, steps)

    def set_grad_accum_steps(self, steps: int):
        """Set the number of gradient accumulation steps.

        This method configures gradient accumulation, which allows training
        with effectively larger batch sizes by accumulating gradients over
        multiple forward passes before updating parameters.

        Args:
            steps (int): Number of steps to accumulate gradients before
                performing a parameter update. Must be positive.
        """
        self._grad_accum_steps = steps
        self._imp.set_grad_accum_steps(steps)

    def reset_state_dict(self):
        """reset optimizer state from a dictionary.

        This method reset the optimizer state of itself, mainly
        for the tests of loading state of sparse optimizer
        """
        self._imp.reset_state_dict()
