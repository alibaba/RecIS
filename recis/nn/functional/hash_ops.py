import torch


def farmhash(inputs, splits):
    """Apply Google's FarmHash algorithm to multiple input tensors.

    FarmHash is a family of hash functions developed by Google that provides
    fast, high-quality hashing with good distribution properties. This function
    applies FarmHash to multiple input tensors with configurable split patterns,
    making it suitable for distributed hashing scenarios in recommendation systems.

    Args:
        inputs (List[torch.Tensor]): List of input tensors to be hashed.
            Each tensor should have dtype int8 and contain byte data to be hashed.
            All tensors should be on the same device.
        splits (List[int]): List of split configurations for each input tensor.
            Each integer specifies how the corresponding input tensor should be
            split or processed during hashing. The length should match the
            number of input tensors.

    Returns:
        List[torch.Tensor]: List of hash result tensors, one for each input tensor.
            The exact shape and content depend on the split configuration and
            the underlying FarmHash implementation.

    Example:
        >>> import torch
        >>> from recis.nn.functional.hash_ops import farmhash
        >>> # Create sample int8 input tensors
        >>> input1 = torch.randint(0, 256, (50,), dtype=torch.int8)
        >>> input2 = torch.randint(0, 256, (100,), dtype=torch.int8)
        >>> inputs = [input1, input2]
        >>> # Define split configurations
        >>> splits = [5, 10]  # Split first tensor into 5 parts, second into 10
        >>> # Apply FarmHash
        >>> hash_results = farmhash(inputs, splits)
        >>> print(f"Number of results: {len(hash_results)}")
        >>> # Use in feature hashing pipeline
        >>> user_features = torch.randint(0, 256, (64,), dtype=torch.int8)
        >>> item_features = torch.randint(0, 256, (32,), dtype=torch.int8)
        >>> feature_hashes = farmhash([user_features, item_features], [8, 4])

    Note:
        - Input tensors must have dtype int8 for proper byte-level hashing
        - FarmHash provides better hash quality compared to simpler hash functions
        - Commonly used for feature hashing and consistent hashing in distributed systems
        - The split parameter controls how input data is partitioned for hashing
        - All computations are performed on GPU for optimal performance
    """
    return torch.ops.recis.fused_hash(inputs, splits, "farm")


def ev_farmhash(inputs, splits):
    return torch.ops.recis.fused_hash(inputs, splits, "ev_farm")


def murmurhash(inputs, splits):
    """Apply MurmurHash algorithm to multiple input tensors.

    MurmurHash is a non-cryptographic hash function that provides good
    distribution and performance characteristics. This function applies
    MurmurHash to multiple input tensors with configurable split patterns,
    making it suitable for general-purpose hashing in recommendation systems.

    Args:
        inputs (List[torch.Tensor]): List of input tensors to be hashed.
            Each tensor should have dtype int8 and contain byte data to be hashed.
            All tensors should be on the same device.
        splits (List[int]): List of split configurations for each input tensor.
            Each integer specifies how the corresponding input tensor should be
            split or processed during hashing. The length should match the
            number of input tensors.

    Returns:
        List[torch.Tensor]: List of hash result tensors, one for each input tensor.
            The exact shape and content depend on the split configuration and
            the underlying MurmurHash implementation.

    Example:
        >>> import torch
        >>> from recis.nn.functional.hash_ops import murmurhash
        >>> # Create sample int8 input tensors
        >>> user_ids = torch.randint(0, 256, (32,), dtype=torch.int8)
        >>> item_ids = torch.randint(0, 256, (64,), dtype=torch.int8)
        >>> inputs = [user_ids, item_ids]
        >>> # Define split configurations
        >>> splits = [4, 8]  # Different split patterns for each input
        >>> # Apply MurmurHash
        >>> hash_results = murmurhash(inputs, splits)
        >>> print(f"Hash results: {len(hash_results)} tensors")
        >>> # Use for consistent hashing across multiple features
        >>> feature_tensors = [
        ...     torch.randint(0, 256, (16,), dtype=torch.int8),
        ...     torch.randint(0, 256, (24,), dtype=torch.int8),
        ...     torch.randint(0, 256, (32,), dtype=torch.int8),
        ... ]
        >>> split_configs = [2, 3, 4]
        >>> multi_hash = murmurhash(feature_tensors, split_configs)

    Note:
        - Input tensors must have dtype int8 for proper byte-level hashing
        - MurmurHash provides good performance and distribution for general use cases
        - Suitable for hash tables, bloom filters, and general hash-based algorithms
        - The split parameter allows flexible partitioning of input data
        - Fused operation enables efficient processing of multiple tensors
        - All computations are GPU-accelerated for high throughput
    """
    return torch.ops.recis.fused_hash(inputs, splits, "murmur")


def sdbmhash(inputs, splits):
    """Apply SDBM hash algorithm to multiple input tensors.

    SDBM is a simple hash function that provides reasonable distribution
    for general-purpose hashing. This function applies SDBM hash to multiple
    input tensors with configurable split patterns.

    Args:
        inputs (List[torch.Tensor]): List of input tensors to be hashed.
            Each tensor should have dtype int8 and contain byte data to be hashed.
            All tensors should be on the same device.
        splits (List[int]): List of split configurations for each input tensor.
            Each integer specifies how the corresponding input tensor should be
            split or processed during hashing. The length should match the
            number of input tensors.

    Returns:
        List[torch.Tensor]: List of hash result tensors, one for each input tensor.

    Note:
        - Input tensors must have dtype int8 for proper byte-level hashing
        - SDBM provides simple and fast hashing for general use cases
        - All computations are GPU-accelerated for high throughput
    """
    return torch.ops.recis.fused_hash(inputs, splits, "sdbm")


def djb2hash(inputs, splits):
    """Apply DJB2 hash algorithm to multiple input tensors.

    DJB2 is a simple hash function created by Daniel J. Bernstein that
    provides good distribution for string hashing. This function applies
    DJB2 hash to multiple input tensors with configurable split patterns.

    Args:
        inputs (List[torch.Tensor]): List of input tensors to be hashed.
            Each tensor should have dtype int8 and contain byte data to be hashed.
            All tensors should be on the same device.
        splits (List[int]): List of split configurations for each input tensor.
            Each integer specifies how the corresponding input tensor should be
            split or processed during hashing. The length should match the
            number of input tensors.

    Returns:
        List[torch.Tensor]: List of hash result tensors, one for each input tensor.

    Note:
        - Input tensors must have dtype int8 for proper byte-level hashing
        - DJB2 is particularly effective for string hashing
        - All computations are GPU-accelerated for high throughput
    """
    return torch.ops.recis.fused_hash(inputs, splits, "djb2")
