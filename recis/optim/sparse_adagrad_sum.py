import torch

from recis.optim.sparse_optim import SparseOptimizer


class SparseAdagradSum(SparseOptimizer):
    """Sparse Adagrad optimizer with gradient sum (not average) for distributed training.

    This class implements the Adagrad optimization algorithm specifically optimized
    for sparse parameters in recommendation systems. Unlike standard SparseAdagrad,
    this version uses gradient SUM across workers instead of gradient AVERAGE,
    making it suitable for distributed training scenarios where gradients should
    be accumulated rather than averaged.

    Key differences from SparseAdagrad:
        - Gradient update uses SUM of gradients from all workers (not average)
        - State sum update uses SUM of squared gradients from all workers (not average of squared gradients)
        - weight_decay is not supported

    Mathematical formulation:

    .. math::

        state_sum_{t} = state_sum_{t-1} + \\sum_{w} g_{w}^2 \n
        lr = lr / (1 + (step - 1) * lr_decay) \n
        θ_t = θ_{t-1} - lr * (\\sum_{w} g_{w} / (√state_sum_t + ε))

    Where:
        - θ: parameters
        - g_w: gradient from worker w
        - \\sum_{w}: sum over all workers
        - ε: eps
        - lr: learning rate
        - lr_decay: learning rate decay
        - state_sum: historical sum of squared gradients

    Example:
        Creating and using SparseAdagradSum:

    .. code-block:: python

        # Initialize with custom hyperparameters
        optimizer = SparseAdagradSum(
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
        """Initialize SparseAdagradSum optimizer with specified hyperparameters.

        Args:
            param_dict (dict): Dictionary of sparse parameters to optimize.
                Keys are parameter names, values are parameter tensors (typically HashTables).
            lr (float, optional): Learning rate. Defaults to 1e-3.
            lr_decay (float, optional): Learning rate decay. Defaults to 0.
            initial_accumulator_value (float, optional): Initial value of the sum of squares of gradients. Defaults to 0.
            eps (float, optional): Term added to the denominator to improve numerical stability. Defaults to 1e-10.
            weight_decay (float, optional): Unsupported for SparseAdagradSum.
                Must be 0.
            save_update_info_interval (int, optional): Interval for saving update information.
                Defaults to 0 (never save). Set to 1 to save every step, 2 to save every 2 steps, etc.

        Note:
            The SparseAdagradSum implemented here operates on RecIS HashTables.
            It uses gradient SUM across workers instead of gradient AVERAGE,
            which is different from the standard SparseAdagrad behavior.
        """
        super().__init__(lr=lr)
        if weight_decay != 0:
            raise ValueError("SparseAdagradSum does not support weight_decay")

        # Store hyperparameters
        self._lr = lr
        self._lr_decay = lr_decay
        self._eps = eps
        self._weight_decay = weight_decay
        self._initial_accumulator_value = initial_accumulator_value
        self._save_update_info_interval = save_update_info_interval

        self._imp = torch.classes.recis.SparseAdagradSum.make(
            param_dict,
            self._lr,
            self._lr_decay,
            self._initial_accumulator_value,
            self._eps,
            self._save_update_info_interval,
        )

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
            self._imp.zero_grad_sq()

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
