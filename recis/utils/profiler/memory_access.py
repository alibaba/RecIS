"""Memory access tracking utilities for custom operators.

This module provides tools to track memory access patterns of custom operators
and class methods, useful for performance analysis and optimization.
"""

import functools
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


class AccessContext:
    """Memory access statistics context.

    This class holds the statistics for a single tracking session,
    including total bytes accessed, execution time, and per-operation breakdowns.

    Attributes:
        name: Name of the tracking context.
        total_bytes: Total bytes accessed during this context.
        total_time_ms: Total execution time in milliseconds.
        op_stats: Per-operation statistics (count, bytes, time_ms).
        execution_details: List of each execution record with index, op_name, bytes, time_ms.
    """

    def __init__(self, name: str):
        """Initialize the access context.

        Args:
            name: Name identifier for this tracking context.
        """
        self.name = name
        self.total_bytes = 0
        self.total_time_ms = 0.0
        self.op_stats: Dict[str, Dict[str, Any]] = {}
        self.execution_details: List[Dict[str, Any]] = []  # 每次执行的详细信息
        self._execution_index = 0  # 执行序号计数器
        # (op_name, bytes, start_event, end_event) awaiting a time. Timing an op
        # with CUDA events and reading them later avoids the full-device
        # synchronize that measuring inline would need.
        self._pending: List[Tuple[str, int, Any, Any]] = []

    def add_access(self, op_name: str, bytes_count: int, time_ms: float):
        """Add a memory access record with execution time.

        Args:
            op_name: Name of the operation.
            bytes_count: Number of bytes accessed.
            time_ms: Execution time in milliseconds.
        """
        self.total_bytes += bytes_count
        self.total_time_ms += time_ms

        if op_name not in self.op_stats:
            self.op_stats[op_name] = {"count": 0, "bytes": 0, "time_ms": 0.0}

        self.op_stats[op_name]["count"] += 1
        self.op_stats[op_name]["bytes"] += bytes_count
        self.op_stats[op_name]["time_ms"] += time_ms

        # 记录每次执行的详细信息
        self.execution_details.append(
            {
                "index": self._execution_index,
                "op_name": op_name,
                "bytes": bytes_count,
                "time_ms": time_ms,
            }
        )
        self._execution_index += 1

    def add_pending(self, op_name: str, bytes_count: int, start_ev, end_ev):
        """Record an access whose time is not readable yet.

        elapsed_time() blocks until the events complete, so calling it here would
        reintroduce the very stall this avoids. The pair is kept and resolved in
        resolve(), after a single synchronize at the end of the context.
        """
        self._pending.append((op_name, bytes_count, start_ev, end_ev))

    def resolve(self):
        """Turn pending CUDA event pairs into times. Costs one synchronize."""
        if not self._pending:
            return
        torch.cuda.synchronize()
        for op_name, bytes_count, start_ev, end_ev in self._pending:
            try:
                self.add_access(op_name, bytes_count, start_ev.elapsed_time(end_ev))
            except Exception:
                pass
        self._pending = []

    def summary(self) -> dict:
        """Get a summary of the tracking results.

        Returns:
            Dictionary containing global statistics, per-operation stats, and execution details.
        """
        # Calculate per-operation bandwidth (GB/s)
        ops_summary = {}
        for op_name, stats in self.op_stats.items():
            time_s = stats["time_ms"] / 1000.0
            bandwidth_gbps = (
                (stats["bytes"] / (1024**3)) / time_s if time_s > 0 else 0.0
            )
            ops_summary[op_name] = {
                "count": stats["count"],
                "bytes": stats["bytes"],
                "time_ms": stats["time_ms"],
                "bandwidth_gbps": round(bandwidth_gbps, 2),
            }

        # Calculate global bandwidth
        total_time_s = self.total_time_ms / 1000.0
        avg_bandwidth_gbps = (
            (self.total_bytes / (1024**3)) / total_time_s if total_time_s > 0 else 0.0
        )

        return {
            "name": self.name,
            # Global statistics
            "total_bytes": self.total_bytes,
            "total_time_ms": round(self.total_time_ms, 3),
            "avg_bandwidth_gbps": round(avg_bandwidth_gbps, 2),
            # Per-operation statistics
            "ops": ops_summary,
            # Execution details (each execution record)
            "execution_details": self.execution_details,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MemoryAccessTracker:
    """Memory access tracking system.

    This class provides a centralized system for tracking memory access
    across custom operators and class methods. It supports:

    - Registration of memory access calculation formulas
    - Dynamic patch/unpatch of operators (zero overhead when disabled)
    - Context-based tracking with automatic lifecycle management

    Key features:
    - When NOT tracking: operators run at full speed (no wrapper overhead)
    - When tracking: operators are patched with tracking wrappers
    - Automatic patch on context enter, unpatch on context exit

    Example:
        >>> from recis.utils.profiler.memory_access import MemoryAccessTracker
        >>> # Use context manager for automatic tracking
        >>> with MemoryAccessTracker.context("forward_pass") as ctx:
        ...     output = model(input)
        >>> print(ctx.summary())
    """

    _formulas: Dict[str, Callable] = {}  # op_name -> calculation formula
    _current_context: Optional[AccessContext] = None

    # Patch management
    _patch_info: Dict[
        str, Tuple[Any, str, Callable]
    ] = {}  # op_name -> (module/class, func_name, original_func)
    _wrapped_funcs: Dict[str, Callable] = {}  # op_name -> wrapped_func
    _is_patched: bool = False

    # ==================== Formula Registration ====================

    @classmethod
    def register_formula(cls, op_name: str):
        """Decorator: Register a memory access calculation formula."""

        def decorator(formula_func: Callable):
            cls._formulas[op_name] = formula_func
            return formula_func

        return decorator

    @classmethod
    def register_formula_direct(cls, op_name: str, formula_func: Callable):
        """Directly register a memory access calculation formula."""
        cls._formulas[op_name] = formula_func

    @classmethod
    def get_formula(cls, op_name: str) -> Optional[Callable]:
        """Get the registered formula for an operation."""
        return cls._formulas.get(op_name)

    @classmethod
    def list_registered_ops(cls) -> List[str]:
        """List all registered operation names."""
        return list(cls._formulas.keys())

    # ==================== Core Wrapping Logic ====================

    @classmethod
    def _create_wrapper(cls, op_name: str, original_func: Callable) -> Callable:
        """Create a wrapped function with memory access tracking."""

        @functools.wraps(original_func)
        def wrapped(*args, **kwargs):
            ctx = cls._current_context
            formula = cls.get_formula(op_name) if ctx is not None else None
            if formula is None:
                return original_func(*args, **kwargs)

            # CUDA events instead of perf_counter plus a synchronize. The old form
            # drained the whole device after every tracked op, so the time it
            # reported covered unrelated queued work, the bandwidth came out too
            # low, and any profiler running in the same window saw its launch-gap
            # measurement destroyed. Events are read later, in one go.
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            output = original_func(*args, **kwargs)
            end_ev.record()
            try:
                access_bytes = formula(args, kwargs, output)
                ctx.add_pending(op_name, access_bytes, start_ev, end_ev)
            except Exception:
                pass
            return output

        return wrapped

    # ==================== Registration for Patch ====================

    @classmethod
    def register_operator(
        cls, module_or_class: Any, func_name: str, op_name: Optional[str] = None
    ):
        """Register an operator for patching (without actual patching)."""
        if op_name is None:
            op_name = func_name

        if hasattr(module_or_class, func_name):
            original_func = getattr(module_or_class, func_name)
            cls._patch_info[op_name] = (module_or_class, func_name, original_func)
            wrapped = cls._create_wrapper(op_name, original_func)
            cls._wrapped_funcs[op_name] = wrapped

    # ==================== Legacy Wrapping Methods ====================

    @classmethod
    def wrap_module_function(
        cls, module, func_name: str, op_name: Optional[str] = None
    ):
        """Register a function in a module for tracking."""
        if op_name is None:
            op_name = func_name

        if hasattr(module, func_name):
            original_func = getattr(module, func_name)
            cls._patch_info[op_name] = (module, func_name, original_func)
            wrapped = cls._create_wrapper(op_name, original_func)
            cls._wrapped_funcs[op_name] = wrapped

    @classmethod
    def wrap_class_method(
        cls, class_obj, method_name: str, op_name: Optional[str] = None
    ):
        """Register a method in a class for tracking."""
        if op_name is None:
            op_name = f"{class_obj.__name__}.{method_name}"

        if hasattr(class_obj, method_name):
            original_method = getattr(class_obj, method_name)
            cls._patch_info[op_name] = (class_obj, method_name, original_method)
            wrapped = cls._create_wrapper(op_name, original_method)
            cls._wrapped_funcs[op_name] = wrapped

    @classmethod
    def wrap_class_methods(
        cls, class_obj, method_names: List[str], op_names: Optional[List[str]] = None
    ):
        """Register multiple methods in a class for tracking."""
        if op_names is None:
            op_names = [f"{class_obj.__name__}.{m}" for m in method_names]

        for method_name, op_name in zip(method_names, op_names):
            cls.wrap_class_method(class_obj, method_name, op_name)

    # ==================== Decorator Interface ====================

    @classmethod
    def wrap_decorator(cls, op_name: Optional[str] = None):
        """Decorator factory: Returns a decorator for wrapping functions/methods."""

        def decorator(func: Callable) -> Callable:
            actual_op_name = op_name or func.__name__
            return cls._create_wrapper(actual_op_name, func)

        return decorator

    # ==================== Patch Management ====================

    @classmethod
    def patch_all(cls):
        """Patch all registered operators."""
        if cls._is_patched:
            return

        for op_name, (
            module_or_class,
            func_name,
            original_func,
        ) in cls._patch_info.items():
            wrapped = cls._wrapped_funcs.get(op_name)
            if wrapped is None:
                wrapped = cls._create_wrapper(op_name, original_func)
                cls._wrapped_funcs[op_name] = wrapped
            setattr(module_or_class, func_name, wrapped)

        cls._is_patched = True

    @classmethod
    def unpatch_all(cls):
        """Restore all original operators."""
        if not cls._is_patched:
            return

        for (
            module_or_class,
            func_name,
            original_func,
        ) in cls._patch_info.values():
            setattr(module_or_class, func_name, original_func)

        cls._is_patched = False

    @classmethod
    def is_patched(cls) -> bool:
        """Check if operators are currently patched."""
        return cls._is_patched

    # ==================== Context Management ====================

    @classmethod
    def start_context(cls, name: str = "default") -> AccessContext:
        """Start a tracking context, automatically patching operators."""
        cls.patch_all()
        ctx = AccessContext(name)
        cls._current_context = ctx
        return ctx

    @classmethod
    def end_context(cls) -> Optional[AccessContext]:
        """End the current tracking context, automatically unpatching operators."""
        ctx = cls._current_context
        cls._current_context = None
        cls.unpatch_all()
        if ctx is not None:
            ctx.resolve()
        return ctx

    @classmethod
    @contextmanager
    def context(cls, name: str = "default"):
        """Context manager for automatic tracking lifecycle."""
        ctx = cls.start_context(name)
        try:
            yield ctx
        finally:
            cls.end_context()

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if tracking is currently enabled."""
        return cls._current_context is not None

    @classmethod
    def get_current_context(cls) -> Optional[AccessContext]:
        """Get the current tracking context."""
        return cls._current_context


# Convenience alias
memory_access_context = MemoryAccessTracker.context
