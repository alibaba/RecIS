import collections
from dataclasses import fields, is_dataclass
from enum import IntEnum, auto


class CompareOp(IntEnum):
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    EQ = auto()


def copy_data_to_device(data, device, *args, **kwargs):
    """Recursively copies data to a specified PyTorch device.

    This function handles various data structures and copies them to the target
    device while preserving their original structure and type. It supports
    tensors, collections, dataclasses, and nested structures.

    Args:
        data: The data structure to copy to device. Can be any of:
            - torch.Tensor: Will be moved to device using .to()
            - list/tuple: Each element will be recursively copied
            - dict/Mapping: Each value will be recursively copied
            - namedtuple: Will be reconstructed with copied fields
            - dataclass: Fields will be recursively copied
            - Any object with .to() method: Will use .to() method
            - Other types: Returned as-is
        device (torch.device): The target device to copy data to.
        *args: Additional positional arguments passed to the .to() method.
        **kwargs: Additional keyword arguments passed to the .to() method.

    Returns:
        The data structure copied to the specified device, maintaining the
        original structure and type.

    Example:
        >>> import torch
        >>> from recis.utils.data_utils import copy_data_to_device
        >>> # Copy tensor to GPU
        >>> tensor = torch.tensor([1, 2, 3])
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> tensor_gpu = copy_data_to_device(tensor, device)
        >>> # Copy batch dictionary to GPU
        >>> batch = {
        ...     "user_id": torch.tensor([1, 2, 3]),
        ...     "item_id": torch.tensor([4, 5, 6]),
        ...     "labels": torch.tensor([0, 1, 0]),
        ... }
        >>> batch_gpu = copy_data_to_device(batch, device)
        >>> # Copy nested structure
        >>> nested_data = {
        ...     "features": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
        ...     "metadata": {"batch_size": 2, "sequence_length": 10},
        ... }
        >>> nested_gpu = copy_data_to_device(nested_data, device)

    Note:
        - This function preserves the exact type of input collections
        - For dataclasses, both init and non-init fields are handled
        - Objects without .to() method are returned unchanged
        - The function is recursive and handles arbitrarily nested structures
    """
    # Redundant isinstance(data, tuple) check is required here to make pyre happy
    if isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return type(data)(
            **copy_data_to_device(data._asdict(), device, *args, **kwargs)
        )
    elif isinstance(data, (list, tuple)):
        return type(data)(copy_data_to_device(e, device, *args, **kwargs) for e in data)
    elif isinstance(data, collections.defaultdict):
        return type(data)(
            data.default_factory,
            {
                k: copy_data_to_device(v, device, *args, **kwargs)
                for k, v in data.items()
            },
        )
    elif isinstance(data, collections.abc.Mapping):
        return type(data)(
            {
                k: copy_data_to_device(v, device, *args, **kwargs)
                for k, v in data.items()
            }
        )
    elif hasattr(data, "to"):
        return data.to(device, *args, **kwargs)
    elif is_dataclass(data) and not isinstance(data, type):
        new_data_class = type(data)(
            **{
                field.name: copy_data_to_device(
                    getattr(data, field.name), device, *args, **kwargs
                )
                for field in fields(data)
                if field.init
            }
        )
        for field in fields(data):
            if not field.init:
                setattr(
                    new_data_class,
                    field.name,
                    copy_data_to_device(
                        getattr(data, field.name), device, *args, **kwargs
                    ),
                )
        return new_data_class
    return data
