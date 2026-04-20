from typing import Iterator

from torch.utils.data import IterableDataset

from recis.monitor.monitor_reporter import DS_END_LATENCY, MonitorReporter


__all__ = ["WrapEndDataset"]


class _WrapEndIterator:
    """Iterator that wraps data items with end-of-stream detection.

    This internal iterator class wraps an underlying data iterator and adds
    metadata to each item indicating whether the stream has ended. It handles
    the StopIteration exception gracefully and continues to signal the end
    condition to consumers.

    Attributes:
        _dataset (WrapEndDataset): Reference to the parent WrapEndDataset.
        _input_iterator (Iterator): The underlying data iterator.
        _should_stop (bool): Flag indicating whether the stream has ended.

    Note:
        This is an internal class used by WrapEndDataset and should not be
        instantiated directly by users.
    """

    def __init__(self, dataset, input_iterator) -> None:
        """Initialize the wrap end iterator.

        Args:
            dataset (WrapEndDataset): Reference to the parent WrapEndDataset instance.
            input_iterator (Iterator): The underlying iterator to wrap with end detection.
        """
        self._dataset = dataset
        self._input_iterator = input_iterator
        self._should_stop = False

    @MonitorReporter.report_time_wrapper(DS_END_LATENCY, tag=None)
    def __next__(self):
        """Get the next data item with end-of-stream metadata.

        This method retrieves the next item from the underlying iterator and
        wraps it with metadata indicating whether the stream has ended.

        Returns:
            tuple: A tuple (is_end, data) where:
                - is_end (bool): True if the stream has ended, False otherwise
                - data: The actual data item, or None if stream has ended

        Note:
            Once the underlying iterator is exhausted, this method will continue
            to return (True, None) for all subsequent calls, allowing consumers
            to handle the end condition appropriately.

        Example:
            ```python
            iterator = iter(wrapped_dataset)

            while True:
                is_end, data = next(iterator)
                if is_end:
                    if data is None:
                        print("Stream completely exhausted")
                        break
                    else:
                        print("Last item before end")
                        process_batch(data)
                else:
                    process_batch(data)
            ```
        """
        if not self._should_stop:
            try:
                ret = next(self._input_iterator)
                return (self._should_stop, ret)
            except StopIteration:
                self._should_stop = True
                return (self._should_stop, None)
        else:
            return (self._should_stop, None)


class WrapEndDataset(IterableDataset):
    """Dataset wrapper that adds end-of-stream detection to data items.

    WrapEndDataset extends PyTorch's IterableDataset to provide end-of-stream
    detection capabilities. It wraps each data item with metadata indicating
    whether the underlying iterator has been exhausted, which is useful for
    streaming data scenarios where consumers need to handle end conditions.

    The dataset transforms regular data items into tuples containing both the
    end-of-stream flag and the actual data, allowing downstream components to
    make informed decisions about data processing completion.

    Attributes:
        _dataset: The underlying dataset to wrap with end detection.

    Example:
        Using WrapEndDataset for streaming data processing:

        ```python
        # Create wrapped dataset
        dataset = WrapEndDataset(streaming_dataset)

        # Process with end detection
        processed_count = 0
        for is_end, batch in dataset:
            if not is_end:
                # Process normal batch
                model.train_step(batch)
                processed_count += 1
            else:
                # Handle end condition
                if batch is not None:
                    # Process final batch
                    model.train_step(batch)
                    processed_count += 1

                print(f"Processing complete. Processed {processed_count} batches")
                break
        ```

        Integration with training loops:

        ```python
        wrapped_dataset = WrapEndDataset(train_dataset)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            batch_count = 0

            for is_end, batch in wrapped_dataset:
                if not is_end:
                    loss = model.train_step(batch)
                    epoch_loss += loss
                    batch_count += 1
                else:
                    if batch is not None:
                        loss = model.train_step(batch)
                        epoch_loss += loss
                        batch_count += 1
                    break

            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            print(f"Epoch {epoch}: Average loss = {avg_loss}")
        ```
    """

    def __init__(self, dataset) -> None:
        """Initialize WrapEndDataset with the underlying dataset.

        Args:
            dataset: The underlying dataset to wrap with end-of-stream detection.
                This can be any iterable object including other datasets.

        Example:
            ```python
            # Wrap any iterable dataset
            original_dataset = MyDataset()
            wrapped_dataset = WrapEndDataset(original_dataset)

            # Now each item includes end-of-stream metadata
            for is_end, data in wrapped_dataset:
                if is_end:
                    handle_end_condition(data)
                else:
                    process_data(data)
            ```
        """
        self._dataset = dataset

    def __iter__(self) -> Iterator:
        """Create and return an iterator with end-of-stream detection.

        Returns:
            _WrapEndIterator: An iterator that wraps each data item with
                end-of-stream metadata.

        Note:
            Each call to __iter__ creates a new iterator instance, allowing
            multiple concurrent iterations over the same wrapped dataset.
        """
        return _WrapEndIterator(self, iter(self._dataset))
