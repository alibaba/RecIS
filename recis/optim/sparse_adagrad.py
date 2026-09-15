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

        adjusted_grad_t = g_t + weight_decay * θ_{t-1} \n
        state_sum_{t} = state_sum_{t-1} + adjusted_grad_t^2 \n
        lr = lr / (1 + (step - 1) * lr_decay) \n
        θ_t = θ_{t-1} - lr * (adjusted_grad_t / (√state_sum_t + ε))

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
        eps=1e-10,
        weight_decay: float = 0,
        save_update_info_interval: int = 0,
    ) -> None:
        """Initialize SparseAdagrad optimizer with specified hyperparameters.

        Args:
            param_dict (dict): Dictionary of sparse parameters to optimize.
                Keys are parameter names, values are parameter tensors (typically HashTables).
            lr (float, optional): Learning rate. Defaults to 1e-2.
            lr (float, Tensor, optional): learning rate (default: 1e-2)
            lr_decay (float, optional): learning rate decay (default: 0)
            initial_accumulator_value (float, optional): initial value of the sum of squares of gradients (default: 0)
            eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-10)
            weight_decay (float, optional): weight decay (L2 penalty). Defaults to 0.
            save_update_info_interval (int, optional): Interval for saving update information.
                Defaults to 0 (never save). Set to 1 to save every step, 2 to save every 2 steps, etc.

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
        self._save_update_info_interval = save_update_info_interval

        self._imp = torch.classes.recis.SparseAdagrad.make(
            param_dict,
            self._lr,
            self._lr_decay,
            self._initial_accumulator_value,
            self._eps,
            self._weight_decay,
            self._save_update_info_interval,
        )

    def get_step_update_info(self) -> dict:
        """Get update information from the last optimizer step.

        This method returns a dictionary containing information about which
        parameters were updated in the last step, including the indices that
        were updated, their corresponding state_sum values, parameter values
        before and after the update, and gradient values.

        Returns:
            dict: A dictionary where keys are parameter names with suffixes:
                - "{param_name}_updated_indices": Tensor of indices that were updated
                - "{param_name}_updated_state_sum": Tensor of state_sum values for updated indices (before update)
                - "{param_name}_updated_params_before": Tensor of parameter values before update
                - "{param_name}_updated_params_after": Tensor of parameter values after update
                - "{param_name}_updated_grad": Tensor of gradient values used for update
                - "{param_name}_last_saved_step": The step number when this update info was last saved
                If update info was not saved for the last step, or the step had
                no updated sparse rows, this method returns an empty dictionary.

        Note:
            Update info collection is disabled by default. When enabled, it
            allocates tensors for the active sparse rows in the saved step and
            can increase memory usage.

        Example:
            .. code-block:: python

                optimizer.step()
                update_info = optimizer.get_step_update_info()

                for key, value in update_info.items():
                    print(f"{key}: shape={value.shape}, dtype={value.dtype}")

                # Access specific information
                indices = update_info["hashtable_name_updated_indices"]
                state_sum_before = update_info["hashtable_name_updated_state_sum"]
                params_before = update_info["hashtable_name_updated_params_before"]
                params_after = update_info["hashtable_name_updated_params_after"]
                grad = update_info["hashtable_name_updated_grad"]
                last_saved_step = update_info["hashtable_name_last_saved_step"]

                # Clear update info if needed
                optimizer.clear_step_update_info()
        """
        update_info = self._imp.get_step_update_info()
        return dict(update_info)

    def clear_step_update_info(self) -> None:
        """Clear the update information stored from the last step.

        This method clears the stored update information to free memory
        and prepare for the next step. Call this method after you have
        processed the update information if you no longer need it.

        Example:
            .. code-block:: python

                optimizer.step()
                update_info = optimizer.get_step_update_info()
                # Process update_info...
                optimizer.clear_step_update_info()
        """
        self._imp.clear_step_update_info()

    def set_save_update_info_interval(self, interval: int) -> None:
        """Set the interval for saving update information.

        Args:
            interval (int): Interval for saving update information. Must be >= 0.
                A value of 0 means never save (default), 1 means save every step,
                2 means save every 2 steps, etc.

        Example:
            .. code-block:: python

                # Save update info every 100 steps to reduce overhead
                optimizer.set_save_update_info_interval(100)

                # Never save update info
                optimizer.set_save_update_info_interval(0)
        """
        self._imp.set_save_update_info_interval(interval)
        self._save_update_info_interval = interval

    def save_update_info_interval(self) -> int:
        """Get the current interval for saving update information.

        Returns:
            int: The current save interval. 0 means never save.

        Example:
            .. code-block:: python

                interval = optimizer.save_update_info_interval()
                if interval == 0:
                    print("Update info is not being saved")
                else:
                    print(f"Update info is saved every {interval} steps")
        """
        return self._imp.save_update_info_interval()
