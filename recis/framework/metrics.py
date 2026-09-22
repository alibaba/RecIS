"""Global metrics management for training and evaluation.

This module provides a simple global registry for metrics that can be accessed
and updated throughout the training process. It maintains a global dictionary
of metrics that can be shared across different components of the framework.
"""

GLOBAL_METRICS = {}


def get_log_metrics():
    """Get the global metrics dictionary.

    Returns:
        dict: The global metrics dictionary containing all registered metrics.

    Example:
        >>> metrics = get_log_metrics()
        >>> print(metrics)
        {'accuracy': 0.95, 'loss': 0.05}
    """
    return GLOBAL_METRICS


def add_metric(name, metric):
    """Add or update a metric in the global metrics registry.

    Args:
        name (str): The name of the metric to add or update.
        metric: The metric value to store. Can be any type (float, int, tensor, etc.).

    Example:
        >>> add_metric("accuracy", 0.95)
        >>> add_metric("loss", 0.05)
        >>> add_metric("learning_rate", 0.001)
    """
    global GLOBAL_METRICS
    GLOBAL_METRICS[name] = metric
