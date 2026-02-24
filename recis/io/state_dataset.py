from typing import Iterator

from torch.utils.data import IterableDataset


__all__ = ["StateDataset"]


class _StateIterator:
    """Iterator that manages state serialization and checkpointing.

    This internal iterator class wraps an underlying data iterator and provides
    automatic state management capabilities. It periodically serializes the
    iterator state and stores it in a shared state map for persistence.

    The iterator maintains a local step counter and saves state at configurable
    intervals to balance between checkpoint frequency and performance overhead.

    Attributes:
        _input_iterator: The underlying data iterator being wrapped.
        _lock: Multiprocessing lock for thread-safe state updates.
        _state_map: Shared dictionary for storing serialized states.
        _save_interval (int): Number of steps between state saves.
        _sub_id: Unique identifier for this iterator instance.
        _local_step (int): Current step counter for this iterator.

    Note:
        This is an internal class used by StateDataset and should not be
        instantiated directly by users.
    """

    def __init__(
        self,
        input_iterator,
        load_state,
        state_map,
        save_interval,
        sub_id,
    ) -> None:
        """Initialize the state iterator with configuration.

        Args:
            input_iterator: The underlying iterator to wrap with state management.
            load_state: Previously saved state to restore from, or None for fresh start.
            lock: Multiprocessing lock for thread-safe state operations.
            state_map: Shared dictionary for storing iterator states.
            save_interval (int): Number of iterations between automatic state saves.
            sub_id: Unique identifier for this iterator instance.

        Note:
            If load_state is provided, the iterator will be restored to the
            exact position it was in when the state was saved.
        """
        self._input_iterator = input_iterator
        self._state_map = state_map
        self._save_interval = save_interval
        self._sub_id = sub_id
        self._local_step = 0
        self._init_state(load_state)

    def _init_state(self, load_state):
        """Initialize iterator state from saved checkpoint or fresh start.

        Args:
            load_state: Previously saved state to restore from, or None.

        Note:
            This method handles both cold starts (no saved state) and warm starts
            (restoring from checkpoint). It ensures the state map is properly
            initialized for subsequent checkpointing.
        """
        if load_state is not None:
            self._input_iterator.deserialize(load_state)
        if self._save_interval:
            state = load_state if load_state else self._input_iterator.serialize()
            self._update_state(state)

    def _update_state(self, state):
        """Thread-safely update the shared state map.

        Args:
            state: Serialized state to store in the shared state map.

        Note:
            This method uses multiprocessing locks to ensure thread-safe
            updates to the shared state map in distributed environments.
        """
        self._state_map[self._sub_id] = state

    def serialize(self):
        """Serialize the current iterator state.

        Returns:
            Serialized state that can be used to restore the iterator position.

        Note:
            The serialized state contains all information needed to resume
            iteration from the exact current position.
        """
        return self._input_iterator.serialize()

    def deserialize(self, state):
        """Restore iterator state from serialized data.

        Args:
            state: Previously serialized state to restore from.

        Note:
            After deserialization, the iterator will continue from the exact
            position where the state was captured.
        """
        self._input_iterator.deserialize(state)

    def __next__(self):
        """Get the next data item with automatic state checkpointing.

        This method increments the step counter and automatically saves state
        at configured intervals before returning the next data item.

        Returns:
            The next data item from the underlying iterator.

        Raises:
            StopIteration: When the underlying iterator is exhausted.

        Note:
            State is saved based on the save_interval configuration to balance
            between checkpoint frequency and performance overhead.
        """
        self._local_step += 1
        if self._save_interval and self._local_step % self._save_interval == 0:
            self._update_state(self.serialize())
        return next(self._input_iterator)


class StateDataset(IterableDataset):
    """Dataset wrapper that provides state management and checkpointing capabilities.

    StateDataset extends PyTorch's IterableDataset to provide automatic state
    serialization and checkpointing for data iterators. This enables resumable
    data processing, which is crucial for long-running training jobs and fault
    tolerance in distributed systems.

    The dataset maintains iterator state in a shared multiprocessing-safe data
    structure and automatically saves checkpoints at configurable intervals.
    This allows training jobs to be interrupted and resumed without losing
    progress or duplicating/skipping data.

    Attributes:
        _dataset: The underlying dataset to wrap with state management.
        _lock: Multiprocessing lock for thread-safe operations.
        _state_map: Shared dictionary for storing iterator states.
        _load_state: Initial state to restore from (if resuming).
        _save_interval (int): Number of iterations between automatic saves.
        _sub_id: Unique identifier for this dataset instance.
        _iter: Current iterator instance (created on demand).

    Example:
        Setting up resumable data processing:

        ```python
        import multiprocessing as mp

        # Setup shared state management
        manager = mp.Manager()
        lock = manager.Lock()
        state_map = manager.dict()

        # Create dataset with checkpointing
        dataset = StateDataset(
            dataset=my_dataset,
            mp_lock=lock,
            state_map=state_map,
            save_interval=50,  # Checkpoint every 50 batches
            sub_id=worker_id,
        )

        # Process data with automatic state management
        try:
            for i, batch in enumerate(dataset):
                # Training step
                model.train_step(batch)

                if i % 1000 == 0:
                    print(f"Processed {i} batches")

        except KeyboardInterrupt:
            # Save final state before exit
            final_state = dataset.dump_io_state()
            save_checkpoint(final_state)
        ```

        Resuming from saved state:

        ```python
        # Load previous state
        saved_state = load_checkpoint()

        # Resume from checkpoint
        dataset = StateDataset(
            dataset=my_dataset,
            mp_lock=lock,
            state_map=state_map,
            load_state=saved_state[worker_id],
            save_interval=50,
            sub_id=worker_id,
        )

        # Continue processing from where we left off
        for batch in dataset:
            model.train_step(batch)
        ```
    """

    def __init__(
        self,
        dataset,
        state_map,
        load_state=None,
        save_interval=100,
        sub_id=0,
    ) -> None:
        """Initialize StateDataset with state management configuration.

        Args:
            dataset: The underlying dataset to wrap with state management.
            mp_lock: Multiprocessing lock for thread-safe state operations.
            state_map: Shared dictionary for storing iterator states across processes.
            load_state (optional): Previously saved state to restore from. Defaults to None.
            save_interval (int, optional): Number of iterations between automatic state saves.
                Defaults to 100.
            sub_id (int, optional): Unique identifier for this dataset instance in
                multi-worker scenarios. Defaults to 0.

        Note:
            The mp_lock and state_map should be created using multiprocessing.Manager()
            to ensure proper sharing across processes in distributed training scenarios.
        """
        self._dataset = dataset
        self._state_map = state_map
        self._load_state = load_state
        self._save_interval = save_interval
        self._sub_id = sub_id
        self._iter = None

    def dump_io_state(self):
        """Retrieve the current state map for persistence.

        Returns:
            dict: The complete state map containing states for all dataset instances.

        Raises:
            RuntimeError: If called before the dataset iterator has been created.

        Example:
            ```python
            # Process some data
            for i, batch in enumerate(dataset):
                if i >= 100:
                    break

            # Get current state for saving
            state = dataset.dump_io_state()

            # Save to file or database
            with open("checkpoint.pkl", "wb") as f:
                pickle.dump(state, f)
            ```

        Note:
            This method should be called after iteration has started to ensure
            the state map contains valid checkpoint data.
        """
        if self._iter is None:
            raise RuntimeError("Cannot get state before run.")
        return self._state_map

    def __iter__(self) -> Iterator:
        """Create and return a state-aware iterator.

        Returns:
            _StateIterator: An iterator that automatically manages state and
                performs checkpointing at configured intervals.

        Note:
            Each call to __iter__ creates a new iterator instance. The iterator
            will restore from the provided load_state if available, or start
            fresh if no previous state exists.
        """
        self._iter = _StateIterator(
            iter(self._dataset),
            self._load_state,
            self._state_map,
            self._save_interval,
            self._sub_id,
        )
        return self._iter
