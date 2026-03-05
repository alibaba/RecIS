from typing import Optional, Union

import torch


class MeanMetric:
    """Computes the weighted or unweighted mean of values over multiple updates.

    This class provides a stateful metric calculator that accumulates values and their
    optional weights across multiple update calls, then computes the mean. It supports
    both weighted and unweighted averaging, with automatic broadcasting of weights to
    match value shapes when necessary.

    Examples:
        Typical usage example for computing mean metrics:

        >>> metric = MeanMetric()
        >>> metric.update([1.0, 2.0, 3.0])
        >>> metric.compute()
        tensor(2.0)
        >>> metric.update([4.0, 5.0], weights=[0.5, 1.5])
        >>> metric.compute()
        tensor(2.8333)

    Attributes:
        dtype: The data type used for internal computations and storage.
        total: Accumulated sum of weighted values.
        count: Accumulated sum of weights.
    """

    def __init__(self, dtype: torch.dtype = torch.float32):
        """Initializes the MeanMetric with specified data type.

        Args:
            dtype: The torch data type to use for internal tensors. Defaults to torch.float32.

        Note:
            The metric is automatically reset upon initialization to prepare for accumulation.
        """
        self.dtype = dtype
        self.reset()

    def reset(self):
        """Resets the metric state to initial values.

        Clears all accumulated values and weights, preparing the metric for a new
        computation cycle. This is useful when starting a new epoch or evaluation phase.
        """
        self.total = torch.tensor(0.0, dtype=self.dtype)
        self.count = torch.tensor(0.0, dtype=self.dtype)

    def update(
        self,
        values: Union[torch.Tensor, list, float],
        weights: Optional[Union[torch.Tensor, list, float]] = None,
    ):
        """Updates the metric state with new values and optional weights.

        Args:
            values: The values to accumulate. Can be a tensor, list, or scalar float.
            weights: Optional weights for each value. If None, uniform weights of 1.0
                are used. Can be a tensor, list, or scalar float. If provided, will be
                broadcast to match the shape of values if necessary.

        Raises:
            ValueError: If weights cannot be broadcast to match the shape of values.

        Note:
            All inputs are converted to tensors with the metric's dtype. The weighted
            values and weights are accumulated using detached tensors to prevent
            gradient tracking.
        """
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=self.dtype)
        else:
            values = values.to(self.dtype).to(device="cpu")

        if weights is None:
            weights = torch.ones_like(values, dtype=self.dtype)
        else:
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, dtype=self.dtype)
            else:
                weights = weights.to(self.dtype).to(device="cpu")

            if weights.dim() == 0 or weights.shape != values.shape:
                weights = torch.broadcast_to(weights, values.shape)
        weighted_values = values * weights
        self.total += torch.sum(weighted_values).detach()
        self.count += torch.sum(weights).detach()

    def compute(self) -> torch.Tensor:
        """Computes and returns the current mean value.

        Returns:
            A tensor containing the weighted mean of all accumulated values. Returns
            0.0 if no values have been accumulated (count is zero).

        Note:
            This method does not reset the metric state. Call reset() explicitly if
            you want to start a new accumulation cycle.
        """
        if self.count == 0:
            return torch.tensor(0.0, dtype=self.dtype)
        return self.total / self.count

    def __call__(
        self,
        values: Union[torch.Tensor, list, float],
        weights: Optional[Union[torch.Tensor, list, float]] = None,
    ) -> torch.Tensor:
        """Updates the metric and immediately returns the computed mean.

        This convenience method combines update() and compute() in a single call,
        allowing the metric to be used as a callable object.

        Args:
            values: The values to accumulate. Can be a tensor, list, or scalar float.
            weights: Optional weights for each value. If None, uniform weights of 1.0
                are used.

        Returns:
            A tensor containing the updated weighted mean after incorporating the
            new values.

        Raises:
            ValueError: If weights cannot be broadcast to match the shape of values.
        """
        self.update(values, weights)
        return self.compute()
