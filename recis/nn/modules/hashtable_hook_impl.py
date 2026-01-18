from collections import defaultdict
from typing import Any, Dict, Optional, Tuple, Type

import torch

from recis.common.singleton import SingletonMeta
from recis.nn.hashtable_hook import AdmitHook, FilterHook
from recis.utils.logger import Logger
from recis.utils.data_utils import CompareOp


logger = Logger(__name__)


class HashtableHookFactory(metaclass=SingletonMeta):
    """Singleton factory for managing hash table hook implementations.

    This factory provides centralized management of filter and admission hooks
    for hash tables. It handles registration, creation, and lifecycle management
    of hook implementations, ensuring consistent behavior across the application.

    The factory supports two types of hooks:
        - Filter hooks: Control which features should be removed from the table
        - Admission hooks: Control which features should be admitted to the table

    Attributes:
        _registed_filter (Dict[str, Type[BaseFilterHookImpl]]): Registered filter hook classes.
        _registed_admit (Dict[str, Type[BaseAdmitHookImpl]]): Registered admission hook classes.
        _filters (Dict[str, Dict[Any, BaseFilterHookImpl]]): Created filter hook instances.
        _admits (Dict[str, Dict[Any, BaseAdmitHookImpl]]): Created admission hook instances.

    Example:
        Register and use custom hooks:

    .. code-block:: python
        # Get factory instance (singleton)
        factory = HashtableHookFactory()


        # Register custom filter hook
        class MyCustomFilter(BaseFilterHookImpl):
            def filter(self, ids, index):
                # Custom filtering logic
                return filtered_ids, filtered_index


        factory.regist_filter("MyCustomFilter", MyCustomFilter)

        # Create hook for hash table
        filter_hook = FilterHook(name="MyCustomFilter", params={})
        hook_impl = factory.create_filter_hook(hashtable, filter_hook)

    """

    def __init__(self) -> None:
        """Initialize the hook factory with empty registries."""
        self._registed_filter: Dict[str, Type[BaseFilterHookImpl]] = {}
        self._registed_admit: Dict[str, Type[BaseAdmitHookImpl]] = {}
        self._filters: Dict[str, Dict[Any, BaseFilterHookImpl]] = defaultdict(dict)
        self._admits: Dict[str, Dict[Any, BaseAdmitHookImpl]] = defaultdict(dict)

    def get_filters(self) -> Dict:
        """Get all created filter hook instances.

        Returns:
            Dict: Dictionary mapping hook names to their instances.
        """
        return self._filters

    def get_admits(self) -> Dict:
        """Get all created admission hook instances.

        Returns:
            Dict: Dictionary mapping hook names to their instances.
        """
        return self._admits

    def create_filter_hook(self, ht: Any, hook: FilterHook):
        """Create a filter hook implementation for a hash table.

        Args:
            ht (Any): Hash table instance to attach the hook to.
            hook (FilterHook): Filter hook configuration.

        Returns:
            BaseFilterHookImpl: Created filter hook implementation.

        Raises:
            ValueError: If hook name is not registered or already exists for the table.
        """
        return self._check_and_create_hook(
            ht, hook, self._registed_filter, self._filters, False
        )

    def create_admit_hook(self, ht: Any, hook: AdmitHook):
        """Create an admission hook implementation for a hash table.

        Args:
            ht (Any): Hash table instance to attach the hook to.
            hook (AdmitHook): Admission hook configuration.

        Returns:
            BaseAdmitHookImpl: Created admission hook implementation.

        Raises:
            ValueError: If hook name is not registered.
        """
        return self._check_and_create_hook(
            ht, hook, self._registed_admit, self._admits, True
        )

    def _check_and_create_hook(
        self,
        ht: Any,
        hook: FilterHook | AdmitHook,
        registed_hooks: Dict[str, type],
        created_hooks: Dict[str, Any],
        reuse: bool = False,
    ):
        """Internal method to check and create hook implementations.

        Args:
            ht (Any): Hash table instance.
            hook (FilterHook | AdmitHook): Hook configuration.
            registed_hooks (Dict[str, type]): Registry of hook classes.
            created_hooks (Dict[str, Any]): Dictionary of created hooks.
            reuse (bool, optional): Whether to reuse existing hooks. Defaults to False.

        Returns:
            Hook implementation instance.

        Raises:
            ValueError: If hook name is not registered or already exists (when reuse=False).
        """
        if hook.name not in registed_hooks:
            raise ValueError(
                f"Hook implementation named '{hook.name}' not registered. "
                f"Registered hooks: {list(registed_hooks.keys())}"
            )
        if ht in created_hooks[hook.name]:
            if reuse:
                return created_hooks[hook.name][ht]
            else:
                raise ValueError(f"Duplicate '{hook.name}' hooks for hashtable: {ht}")
        else:
            hook_impl_class = registed_hooks[hook.name]
            if isinstance(hook, FilterHook):
                hook_impl = hook_impl_class(ht, **hook.params)
            else:
                hook_impl = hook_impl_class(**hook.params)
            created_hooks[hook.name][ht] = hook_impl
            return hook_impl

    def regist_filter(self, name: str, filter_hook_impl: Any) -> None:
        """Register a filter hook implementation class.

        Args:
            name (str): Unique name for the filter hook.
            filter_hook_impl (Type[BaseFilterHookImpl]): Filter hook implementation class.

        Raises:
            ValueError: If a hook with the same name is already registered.

        Example:

        .. code-block:: python
            class MyFilter(BaseFilterHookImpl):
                def filter(self, ids, index):
                    return filtered_ids, filtered_index


            factory.regist_filter("MyFilter", MyFilter)

        """
        if name in self._registed_filter:
            raise ValueError(f"Filter hook {name} has been registered!")
        self._registed_filter[name] = filter_hook_impl

    def regist_admit(self, name: str, admit_hook_impl: Any) -> None:
        """Register an admission hook implementation class.

        Args:
            name (str): Unique name for the admission hook.
            admit_hook_impl (Type[BaseAdmitHookImpl]): Admission hook implementation class.

        Raises:
            ValueError: If a hook with the same name is already registered.
        """
        if name in self._registed_admit:
            raise ValueError(f"Admit hook {name} has been registered!")
        self._registed_admit[name] = admit_hook_impl


class BaseAdmitHookImpl:
    """Base class for admission hook implementations.

    Admission hooks control which features are allowed to be stored in the
    hash table. They can implement various strategies such as frequency-based
    admission, probability-based sampling, or custom business logic.

    This base class provides the interface that all admission hooks must
    implement. Subclasses should override the necessary methods to provide
    specific admission control logic.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the admission hook.

        Args:
            **kwargs: Additional keyword arguments for hook configuration.
        """


class ReadOnlyHookImpl(BaseAdmitHookImpl):
    """Read-only admission hook implementation.

    This hook allows read-only access to existing features in the hash table
    without admitting new features. It's useful for inference scenarios where
    you want to prevent the vocabulary from growing.

    Example:
    .. code-block:: python
        # Create read-only hook
        admit_hook = AdmitHook(name="ReadOnly", params={})

        # Use with hash table
        hashtable = HashTable(
            embedding_shape=[64],
            admit_hook=admit_hook,  # Only existing features can be accessed
        )

        # During inference, new feature IDs will not be added to the table
        embeddings = hashtable(feature_ids)

    """

    def __init__(self, **kwargs) -> None:
        """Initialize the read-only admission hook.

        Args:
            **kwargs: Additional keyword arguments (unused).
        """
        super().__init__()


class BaseFilterHookImpl(torch.nn.Module):
    """Base class for filter hook implementations.

    Filter hooks control which features should be removed from the hash table
    based on various criteria such as staleness, frequency, or custom business
    logic. They help manage memory usage and maintain table quality by removing
    irrelevant or outdated features.

    This base class provides the common interface and functionality that all
    filter hooks must implement. Subclasses should override the `filter` method
    to provide specific filtering logic.

    Args:
        ht (Any): Hash table instance to attach the hook to.
        name (Optional[str], optional): Name of the filter hook. Defaults to None.
        **kwargs: Additional keyword arguments for hook configuration.

    Attributes:
        _ht: Reference to the hash table implementation.
        _name (str): Name of the filter hook.

    Example:
        Implement a custom filter hook:

    .. code-block:: python
        class FrequencyFilterHook(BaseFilterHookImpl):
            def __init__(self, ht, name="frequency_filter", min_frequency=10):
                super().__init__(ht, name)
                self.min_frequency = min_frequency
                self.frequency_counter = {}

            def update(self, index):
                # Update frequency counters when features are accessed
                for idx in index:
                    self.frequency_counter[idx.item()] = (
                        self.frequency_counter.get(idx.item(), 0) + 1
                    )

            def filter(self, ids, index):
                # Filter out features with low frequency
                mask = torch.tensor(
                    [
                        self.frequency_counter.get(idx.item(), 0) < self.min_frequency
                        for idx in index
                    ]
                )
                return torch.masked_select(ids, mask), torch.masked_select(index, mask)

    """

    def __init__(self, ht: Any, name: Optional[str] = None, **kwargs) -> None:
        """Initialize the filter hook.

        Args:
            ht (Any): Hash table instance to attach the hook to.
            name (Optional[str], optional): Name of the filter hook. Defaults to None.
            **kwargs: Additional keyword arguments for hook configuration.
        """
        super().__init__()
        self._ht = ht._hashtable_impl
        self._name = name

    def update(self, index: torch.Tensor) -> None:
        """Update hook state when features are accessed.

        This method is called whenever features are looked up in the hash table.
        Subclasses can override this method to track feature usage statistics
        or update internal state.

        Args:
            index (torch.Tensor): Indices of features that were accessed.
        """

    def do_filter(self) -> None:
        """Execute the filtering operation.

        This method retrieves all features from the hash table, applies the
        filtering logic, and removes the filtered features. It also calls
        the after_filter callback for logging or cleanup.
        """
        ids, index = self._ht.ids_map(None)
        delete_ids, delete_index = self.filter(ids, index)
        self._ht.delete(delete_ids, delete_index, self._name)
        self.after_filter(ids, delete_ids)

    def filter(
        self, ids: torch.Tensor, index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply filtering logic to determine which features to remove.

        This method must be implemented by subclasses to provide specific
        filtering criteria. It should return the IDs and indices of features
        that should be removed from the hash table.

        Args:
            ids (torch.Tensor): All feature IDs in the hash table.
            index (torch.Tensor): Corresponding internal indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (filtered_ids, filtered_indices)
                representing features to be removed.
        """

    def forward(self, index: torch.Tensor) -> None:
        """Forward pass for the filter hook.

        This method is called during hash table operations to update the
        hook's internal state based on feature access patterns.

        Args:
            index (torch.Tensor): Indices of features being accessed.
        """
        self.update(index)

    def after_filter(self, ids: torch.Tensor, delete_ids: torch.Tensor) -> None:
        """Callback method called after filtering is complete.

        This method can be overridden by subclasses to perform cleanup,
        logging, or other post-filtering operations.

        Args:
            ids (torch.Tensor): All feature IDs before filtering.
            delete_ids (torch.Tensor): Feature IDs that were removed.
        """

    def __str__(self) -> str:
        """String representation of the filter hook.

        Returns:
            str: String representation including hash table reference and name.
        """
        return (
            f"LookupFilterHookImplBase ht=HT@{id(self._ht):x}, name is {self._name!r}"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the filter hook.

        Returns:
            str: Detailed string representation.
        """
        return self.__str__()


class GlobalStepFilterHookImpl(BaseFilterHookImpl):
    """Global step-based filter hook implementation.

    This filter hook removes features that haven't been accessed for a specified
    number of global steps. It's useful for removing stale features in online
    learning scenarios where the feature distribution changes over time.

    An entry is considered stale if:
    current_global_step - last_update_step > filter_step

    Args:
        ht (Any): Hash table instance to attach the hook to.
        name (str, optional): Name of the filter hook. Defaults to "step_filter".
        filter_step (int, optional): Number of steps after which features are
            considered stale. Defaults to 100.
        **kwargs: Additional keyword arguments.

    Attributes:
        _global_step (torch.Tensor): Current global step counter.
        _filter_step (int): Step threshold for filtering stale features.

    Example:
        Basic usage:

    .. code-block:: python
        from recis.nn.hashtable_hook import FilterHook

        # Create filter hook that removes features inactive for 200 steps
        filter_hook = FilterHook(name="GlobalStepFilter", params={"filter_step": 200})

        # Use with hash table
        hashtable = HashTable(embedding_shape=[64], filter_hook=filter_hook)

        # During training, update the global step
        for step in range(1000):
            # ... training logic ...

            hook_impl = hashtable._filter_hook_impl
            if hasattr(hook_impl, "update_step"):
                hook_impl.update_step()

            # Perform filtering (this should be done less frequently)
            if step % 100 == 0:
                hook_impl.do_filter()


        Advanced usage with custom step management:

    .. code-block:: python
        # Create hook with custom configuration
        filter_hook = FilterHook(
            name="GlobalStepFilter",
            params={"filter_step": 50, "name": "custom_step_filter"},
        )

        hashtable = HashTable(embedding_shape=[128], filter_hook=filter_hook)

        # Access the hook implementation for manual control
        hook_impl = hashtable._filter_hook_impl

        # Manual step updates
        hook_impl.update_step(5)  # Increment by 5 steps

        # Check current step
        current_step = hook_impl._global_step.item()
        print(f"Current global step: {current_step}")

    """

    def __init__(
        self,
        ht: Any,
        name: str = "step_filter",
        filter_step: int = 100,
        **kwargs,
    ) -> None:
        """Initialize the global step filter hook.

        Args:
            ht (Any): Hash table instance to attach the hook to.
            name (str, optional): Name of the filter hook. Defaults to "step_filter".
            filter_step (int, optional): Number of steps after which features are
                considered stale. Must be > 0. Defaults to 100.
            **kwargs: Additional keyword arguments.

        Raises:
            AssertionError: If filter_step is not greater than 0.
        """
        super().__init__(ht, name)
        self.register_buffer(
            "_global_step", torch.tensor([0], dtype=torch.int64, device=ht.device)
        )
        assert filter_step > 0, (
            f"GlobalStepFilterHookImpl: filter_step must > 0, but get {filter_step}"
        )
        self._filter_step = filter_step
        self._ht.append_step_filter_slot(self._name, self._global_step)

    def update(self, index: torch.Tensor) -> None:
        """Update the last access step for the given feature indices.

        This method records the current global step as the last access time
        for the specified features, preventing them from being filtered out
        in subsequent filtering operations.

        Args:
            index (torch.Tensor): Indices of features being accessed.
        """
        self._ht.update_slot(self._name, index, self._global_step)

    def update_step(self, step_size: int = 1) -> None:
        """Update the global step counter.

        This method should be called periodically during training to advance
        the global step counter. Features that haven't been accessed recently
        will become eligible for filtering.

        Args:
            step_size (int, optional): Number of steps to increment. Defaults to 1.

        Example:

        .. code-block:: python
            # Update step counter during training loop
            for epoch in range(num_epochs):
                for batch in dataloader:
                    # ... training logic ...

                    # Update global step every batch
                    hook_impl.update_step()

                    # Or update by multiple steps
                    # hook_impl.update_step(batch_size)

        """
        with torch.no_grad():
            self._global_step += step_size

    def filter(
        self, ids: torch.Tensor, index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter out stale features based on global step threshold.

        This method identifies features that haven't been accessed for more
        than `filter_step` global steps and marks them for removal.

        Args:
            ids (torch.Tensor): All feature IDs in the hash table.
            index (torch.Tensor): Corresponding internal indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (stale_ids, stale_indices)
                representing features to be removed due to staleness.
        """
        filter_slot = self._ht.get_slot(self._name)
        filter_values = filter_slot.values()
        mask = torch.ops.recis.block_filter(
            index,
            filter_values,
            filter_slot.block_size(),
            self._global_step - self._filter_step,
            CompareOp.LT,
        )
        return torch.masked_select(ids, mask), torch.masked_select(index, mask)

    def after_filter(self, ids: torch.Tensor, delete_ids: torch.Tensor) -> None:
        """Log filtering results after filtering is complete.

        This method provides detailed logging about the filtering operation,
        including the number of features removed and the current state of
        the hash table.

        Args:
            ids (torch.Tensor): All feature IDs before filtering.
            delete_ids (torch.Tensor): Feature IDs that were removed.
        """
        logger.info(
            f"GlobalStepFilterHook [{self._name!r}] applied for ht=HT@{id(self._ht):x} at global step {self._global_step.item()} with filter_step {self._filter_step}. "
            f"Hashtable emb count reduced from {ids.numel()} to {ids.numel() - delete_ids.numel()}. "
            f"({delete_ids.numel()} emb erased). "
        )

    def __str__(self) -> str:
        """String representation of the global step filter hook.

        Returns:
            str: String representation including current step, filter threshold,
                and hash table reference.
        """
        return f"GlobalStepFilterHookImpl(global_step={self._global_step.item()}, filter_step={self._filter_step}, slot_name={self._name!r}, ht=HT@{id(self._ht):x})"

    def __repr__(self) -> str:
        """Detailed string representation of the global step filter hook.

        Returns:
            str: Detailed string representation.
        """
        return self.__str__()


# Register built-in hook implementations
HashtableHookFactory().regist_admit("ReadOnly", ReadOnlyHookImpl)
HashtableHookFactory().regist_filter("GlobalStepFilter", GlobalStepFilterHookImpl)
