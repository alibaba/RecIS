"""Pipeline prefetch helpers for overlapping data prep with train compute.

These utilities are independent of the Trainer loop itself: they resolve a
prefetch callable and wrap iterators with a side-CUDA-stream transform.
Raw batches are fetched on the main thread (inside ``__next__``, before
forward); only the transform runs in the worker thread, released by
``notify_prefetch`` after forward so it overlaps with backward/optim.
"""

from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Tuple, Union

from torch import nn

from recis.io.cuda_prefetch_dataset import StreamPriority, cuda_prefetch_iterator
from recis.utils.data_utils import copy_data_to_device
from recis.utils.logger import Logger


logger = Logger(__name__)


# Positions in the train step where notify_prefetch releases the worker to
# start transforming the next batch (i.e. what the transform overlaps with):
#   before_forward       — overlaps forward + backward + optim_step
#   after_sparse_forward — released by a forward hook right after the
#                          discovered prefetch module (e.g. RecISModel)
#                          finishes; overlaps dense forward + backward +
#                          optim_step
#   before_backward      — overlaps backward + optim_step
#   before_optim_step    — overlaps optim_step only
PREFETCH_BEFORE_FORWARD = "before_forward"
PREFETCH_AFTER_SPARSE_FORWARD = "after_sparse_forward"
PREFETCH_BEFORE_BACKWARD = "before_backward"
PREFETCH_BEFORE_OPTIM_STEP = "before_optim_step"
PREFETCH_NOTIFY_POSITIONS = (
    PREFETCH_BEFORE_FORWARD,
    PREFETCH_AFTER_SPARSE_FORWARD,
    PREFETCH_BEFORE_BACKWARD,
    PREFETCH_BEFORE_OPTIM_STEP,
)


@dataclass
class PrefetchArguments:
    """Configuration for pipeline prefetch on a side CUDA stream.

    Attributes:
        enable_pipeline_prefetch (bool | int): Enable pipeline prefetch.
            When an int ``> 0`` is given, it is also used as the prefetch
            buffer size. Defaults to False.
        pipeline_prefetch_fn (Optional[Callable]): User data transform
            ``data -> data``. When set, it is used instead of the
            auto-discovered ``model.prefetch_step`` after pipeline prefetch
            is enabled.
        stream_priority (int | StreamPriority): CUDA priority of the
            prefetch side stream. Lower value = higher GPU scheduling
            priority (HIGH=-2, DEFAULT=-1, LOW=0). Defaults to
            ``StreamPriority.LOW`` (same as the default stream).
        notify_position (str): Where in the train step the next-batch
            transform is released. One of ``"before_forward"``,
            ``"after_sparse_forward"``, ``"before_backward"``,
            ``"before_optim_step"``. ``"after_sparse_forward"`` notifies via
            a forward hook on the discovered prefetch module (e.g.
            RecISModel) so the transform overlaps the dense forward as
            well; it requires auto-discovery (not ``pipeline_prefetch_fn``).
            Defaults to ``"before_backward"``.
        enable_thread_prefetch (bool): When True (default), the transform
            runs in a background worker thread. When False, it runs inline
            on the main thread at the notify position: no GIL contention
            with the train loop, but the transform's Python part is
            serialised into it (GPU work still overlaps via the side
            stream).
        fetch_data_in_thread (bool): Where raw batches are pulled from the
            dataset. False (default) fetches them inline on the main thread
            in ``__next__``, right before forward. True moves the fetch into
            the prefetch worker thread, where it is gated to run after the
            notify — so its Python work (dataset IO plus the ragged convert)
            overlaps backward instead of forward, at the cost of contending
            for the GIL with the train loop. Requires
            ``enable_thread_prefetch``.
    """

    enable_pipeline_prefetch: Union[bool, int] = False
    pipeline_prefetch_fn: Optional[Callable] = None
    stream_priority: int = StreamPriority.LOW
    notify_position: str = PREFETCH_BEFORE_BACKWARD
    enable_thread_prefetch: bool = True
    fetch_data_in_thread: bool = False

    def __post_init__(self):
        if self.notify_position not in PREFETCH_NOTIFY_POSITIONS:
            raise ValueError(
                f"notify_position must be one of {PREFETCH_NOTIFY_POSITIONS}, "
                f"got '{self.notify_position}'"
            )
        if self.fetch_data_in_thread and not self.enable_thread_prefetch:
            raise ValueError(
                "fetch_data_in_thread requires enable_thread_prefetch: "
                "inline mode has no worker thread to fetch data on."
            )


def resolve_prefetch_model(model: Optional[nn.Module], _visited: Optional[set] = None):
    """Recursively find a submodule that exposes ``prefetch_step``.

    Search order:
        1. the model itself
        2. direct child named ``sparse`` (common convention)
        3. other ``named_children`` / ``module`` (DDP wrappers), DFS
    """
    if model is None:
        return None
    if _visited is None:
        _visited = set()
    model_id = id(model)
    if model_id in _visited:
        return None
    _visited.add(model_id)

    if hasattr(model, "prefetch_step") and callable(model.prefetch_step):
        return model

    # Prefer the conventional sparse child before a blind DFS.
    children = list(model.named_children())
    children.sort(key=lambda item: 0 if item[0] == "sparse" else 1)
    if hasattr(model, "module") and not any(name == "module" for name, _ in children):
        children.append(("module", model.module))

    for _, child in children:
        if not isinstance(child, nn.Module):
            continue
        found = resolve_prefetch_model(child, _visited)
        if found is not None:
            return found
    return None


def make_pipeline_prefetch_transform(
    prefetch_fn: Callable, data_to_cuda: bool
) -> Callable:
    """Build ``(stop_flag, data) -> (stop_flag, data)`` from a data-level fn.

    ``prefetch_fn`` receives the batch ``data`` and returns the transformed
    batch. Stop-flag handling and optional H2D copy stay here.
    """

    def transform(item):
        stop_flag, data = item
        if stop_flag:
            return (stop_flag, data)
        if data_to_cuda:
            data = copy_data_to_device(data, "cuda", non_blocking=True)
        data = prefetch_fn(data)
        return (stop_flag, data)

    return transform


def setup_pipeline_prefetch(
    model: Optional[nn.Module],
    data_to_cuda: bool,
    enable_pipeline_prefetch: Union[bool, int] = False,
    pipeline_prefetch_fn: Optional[Callable] = None,
) -> Tuple[Optional[Callable], int]:
    """Resolve the prefetch transform and buffer size.

    Priority:
        1. ``pipeline_prefetch_fn`` (user data fn; skips model.prefetch_step)
        2. auto-discovered ``prefetch_step`` on model or any submodule
           (prefers child named ``sparse``)

    Returns:
        tuple: ``(transform_or_none, buffer_size)``
    """
    buffer_size = 1
    enabled = bool(enable_pipeline_prefetch)
    if isinstance(enable_pipeline_prefetch, int):
        enabled = enable_pipeline_prefetch > 0
        if enable_pipeline_prefetch > 0:
            buffer_size = enable_pipeline_prefetch

    if not enabled:
        return None, buffer_size

    if pipeline_prefetch_fn is not None:
        transform = make_pipeline_prefetch_transform(pipeline_prefetch_fn, data_to_cuda)
        logger.info(
            "Pipeline prefetch enabled with user pipeline_prefetch_fn "
            f"(buffer_size={buffer_size})"
        )
        return transform, buffer_size

    prefetch_model = resolve_prefetch_model(model)
    if prefetch_model is None:
        raise TypeError(
            "enable_pipeline_prefetch requires a submodule that implements "
            "prefetch_step() (e.g. RecISModel), or pass pipeline_prefetch_fn. "
            f"Got: {type(model).__name__ if model is not None else None}"
        )
    transform = make_pipeline_prefetch_transform(
        prefetch_model.prefetch_step, data_to_cuda
    )
    logger.info(
        "Pipeline prefetch enabled "
        f"(buffer_size={buffer_size}, "
        f"prefetch_model={type(prefetch_model).__name__})"
    )
    return transform, buffer_size


def wrap_with_prefetch(
    iterator: Optional[Iterator],
    transform: Optional[Callable],
    buffer_size: int = 1,
    lazy_start: bool = False,
    stream_priority: int = StreamPriority.LOW,
    enable_thread: bool = True,
    fetch_in_thread: bool = False,
) -> Optional[Iterator]:
    """Wrap an iterator with CUDA-stream prefetch when a transform is set.

    Args:
        stream_priority (int | StreamPriority): CUDA priority of the side
            stream. Lower value = higher GPU scheduling priority. Defaults
            to ``StreamPriority.LOW`` (same as the default stream).
        enable_thread (bool): Run the transform in a worker thread (True,
            default) or inline on the consuming thread (False).
        fetch_in_thread (bool): Pull raw batches in the worker thread instead
            of inline in ``__next__``. Requires ``enable_thread``.
    """
    if transform is None or iterator is None:
        return iterator
    return cuda_prefetch_iterator(
        iterator,
        transform,
        buffer_size=buffer_size,
        lazy_start=lazy_start,
        stream_priority=stream_priority,
        enable_thread=enable_thread,
        fetch_in_thread=fetch_in_thread,
    )


def notify_prefetch(prefetch_iter) -> None:
    """Release the next-batch transform on the prefetch worker."""
    if prefetch_iter is not None and hasattr(prefetch_iter, "notify"):
        prefetch_iter.notify()
