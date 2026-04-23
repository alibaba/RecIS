from typing import Iterator

from torch.utils.data import IterableDataset


__all__ = ["StateDataset"]


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
        state_map = manager.dict()

        # Create dataset with checkpointing
        dataset = StateDataset(
            dataset=my_dataset,
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
            state_map: Shared dictionary for storing iterator states across processes.
            load_state (optional): Previously saved state to restore from. Defaults to None.
            save_interval (int, optional): Number of iterations between automatic state saves.
                Defaults to 100.
            sub_id (int, optional): Unique identifier for this dataset instance in
                multi-worker scenarios. Defaults to 0.

        Note:
            The state_map should be created using multiprocessing.Manager() when sharing
            across processes in distributed training scenarios.
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
        """Create and return the underlying dataset iterator with IO state primed.

        Returns:
            Iterator: The object returned by ``iter(self._dataset)``. It must implement
            ``serialize`` and ``deserialize`` for checkpointing (same contract as before).

        Note:
            Each call to ``__iter__`` creates a new iterator. The iterator restores from
            ``load_state`` when provided, otherwise starts fresh. When ``save_interval``
            is non-zero, an initial entry is written to ``state_map`` for this ``sub_id``.
        """
        it = iter(self._dataset)
        if self._load_state is not None:
            it.deserialize(self._load_state)
        if self._save_interval:
            state = self._load_state if self._load_state else it.serialize()
            self._state_map[self._sub_id] = state
        self._iter = it
        return it

    def flush_io_state(self) -> None:
        """Persist the live iterator position into ``state_map`` for this worker."""
        if self._iter is None:
            return
        self._state_map[self._sub_id] = self._iter.serialize()
