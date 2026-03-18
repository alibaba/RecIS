import logging
import os
import sys


class SingleLevelFilter(logging.Filter):
    """Custom logging filter that filters records based on log level.

    This filter can either pass through only records of a specific level
    or reject records of a specific level, depending on the reject parameter.

    Args:
        passlevel (int): The log level to filter on (e.g., logging.INFO).
        reject (bool): If True, reject records of passlevel. If False,
            only pass records of passlevel.

    Example:
        >>> # Create filter that only passes INFO level messages
        >>> info_filter = SingleLevelFilter(logging.INFO, False)
        >>> handler.addFilter(info_filter)
        >>> # Create filter that rejects INFO level messages
        >>> no_info_filter = SingleLevelFilter(logging.INFO, True)
        >>> handler.addFilter(no_info_filter)
    """

    def __init__(self, passlevel, reject):
        """Initialize the filter with specified level and reject behavior.

        Args:
            passlevel (int): The log level to filter on.
            reject (bool): Whether to reject (True) or accept (False) the level.
        """
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        """Filter log records based on level and reject setting.

        Args:
            record (logging.LogRecord): The log record to filter.

        Returns:
            bool: True if the record should be processed, False otherwise.
        """
        if self.reject:
            return record.levelno != self.passlevel
        else:
            return record.levelno == self.passlevel


class Logger:
    """Distributed-training-friendly logger for RecIS framework.

    This logger provides a unified interface for logging with support for
    distributed training scenarios. It automatically configures separate
    handlers for stdout and stderr, with appropriate filtering to ensure
    INFO messages go to stdout and other levels go to stderr.

    Features:
        - Automatic handler configuration with level-based routing
        - Rank-aware logging methods for distributed training
        - Consistent formatting across all messages
        - Support for all standard logging levels

    Args:
        name (str, optional): Name for the logger, typically __file__ or __name__.
            Defaults to __file__.
        level (int, optional): Minimum logging level. Defaults to logging.DEBUG.

    Example:
        >>> from recis.utils.logger import Logger
        >>> # Create logger for current module
        >>> logger = Logger(__name__)
        >>> # Basic logging
        >>> logger.info("Training started")
        >>> logger.warning("Learning rate is high")
        >>> logger.error("Failed to load checkpoint")
        >>> # Distributed training - only rank 0 logs
        >>> logger.info_rank0("Epoch completed")  # Only logs on rank 0
        >>> logger.warning_rank0("Memory usage high")  # Only logs on rank 0
        >>> # Regular logging (all ranks)
        >>> logger.info("Processing batch")  # Logs on all ranks
    """

    def __init__(self, name=__file__, level=logging.DEBUG):
        """Initialize the logger with configured handlers and formatting.

        Sets up stdout handler for INFO messages and stderr handler for
        all other levels, with consistent formatting.

        Args:
            name (str, optional): Logger name. Defaults to __file__.
            level (int, optional): Minimum log level. Defaults to logging.DEBUG.
        """
        stdout_handler = logging.StreamHandler(sys.stdout)
        stderr_handler = logging.StreamHandler(sys.stderr)
        stdout_handler.addFilter(SingleLevelFilter(logging.INFO, False))
        stderr_handler.addFilter(SingleLevelFilter(logging.INFO, True))
        stderr_handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        )
        logger = logging.getLogger(name)
        logger.setLevel(level)
        stdout_handler.setFormatter(formatter)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
        logger.addHandler(stderr_handler)
        logger.propagate = False
        self._logger = logger

    def debug(self, *args, **kwargs):
        """Log a debug message.
        Args:
            *args: Positional arguments passed to logger.debug().
            **kwargs: Keyword arguments passed to logger.debug().
        Example:
            >>> logger.debug("Processing batch %d", batch_idx)
            >>> logger.debug("Training loss: %.4f", loss.item())
        """
        self._logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        """Log an info message.

        Args:
            *args: Positional arguments passed to logger.info().
            **kwargs: Keyword arguments passed to logger.info().

        Example:
            >>> logger.info("Processing batch %d", batch_idx)
            >>> logger.info("Training loss: %.4f", loss.item())
        """
        self._logger.info(*args, **kwargs)

    def info_rank0(self, *args, **kwargs):
        """Log an info message only on rank 0 (distributed training).

        This method checks the RANK environment variable and only logs
        if the current process is rank 0. Useful for avoiding duplicate
        logs in distributed training scenarios.

        Args:
            *args: Positional arguments passed to logger.info().
            **kwargs: Keyword arguments passed to logger.info().

        Example:
            >>> # Only rank 0 will log this message
            >>> logger.info_rank0("Epoch %d completed", epoch)
            >>> logger.info_rank0("Saving checkpoint to %s", checkpoint_path)
        """
        if os.environ.get("RANK", "0") == "0":
            self._logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        """Log a warning message.

        Args:
            *args: Positional arguments passed to logger.warning().
            **kwargs: Keyword arguments passed to logger.warning().

        Example:
            >>> logger.warning("Learning rate is very high: %.6f", lr)
            >>> logger.warning("Memory usage: %.1f%%", memory_percent)
        """
        self._logger.warning(*args, **kwargs)

    def warning_rank0(self, *args, **kwargs):
        """Log a warning message only on rank 0 (distributed training).

        Args:
            *args: Positional arguments passed to logger.warning().
            **kwargs: Keyword arguments passed to logger.warning().

        Example:
            >>> # Only rank 0 will log this warning
            >>> logger.warning_rank0("Model convergence is slow")
        """
        if os.environ.get("RANK", "0") == "0":
            self._logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        """Log an error message.

        Args:
            *args: Positional arguments passed to logger.error().
            **kwargs: Keyword arguments passed to logger.error().

        Example:
            >>> logger.error("Failed to load model from %s", model_path)
            >>> logger.error("CUDA out of memory: %s", str(e))
        """
        self._logger.error(*args, **kwargs)

    def error_rank0(self, *args, **kwargs):
        """Log an error message only on rank 0 (distributed training).

        Args:
            *args: Positional arguments passed to logger.error().
            **kwargs: Keyword arguments passed to logger.error().

        Example:
            >>> # Only rank 0 will log this error
            >>> logger.error_rank0("Training failed: %s", error_msg)
        """
        if os.environ.get("RANK", "0") == "0":
            self._logger.error(*args, **kwargs)
