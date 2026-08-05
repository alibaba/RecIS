"""Test script for memory access tracking functionality.

Usage:
    python -m tests.utils.test_memory_access
"""

import torch

from recis.utils import MemoryAccessTracker, setup_memory_access_tracking


def test_fused_bucketized():
    """Test memory access tracking for fused_bucketized operator."""
    print("\n" + "=" * 60)
    print("Testing fused_bucketized operator")
    print("=" * 60)

    # Create test data
    values_list = [
        torch.tensor([1.0, 2.5, 3.0, 4.5], dtype=torch.float32, device="cuda"),
        torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float32, device="cuda"),
    ]
    boundaries_list = [
        torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32, device="cuda"),
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="cuda"),
    ]

    # Track memory access
    with MemoryAccessTracker.context("fused_bucketized_test") as ctx:
        result = torch.ops.recis.fused_bucketized(values_list, boundaries_list)

    print(f"Result type: {type(result)}")
    print(f"Result shapes: {[r.shape for r in result]}")
    print("\nMemory access summary:")
    summary = ctx.summary()
    print(f"  Total bytes: {summary['total_bytes']}")
    print(f"  Total MB: {summary['total_bytes'] / (1024**2):.4f}")
    print(f"  Operations: {summary['ops']}")

    return ctx


def test_hashtable_forward():
    """Test memory access tracking for HashTable.forward method."""
    print("\n" + "=" * 60)
    print("Testing HashTable.forward method")
    print("=" * 60)

    try:
        from recis.nn.modules.hashtable import HashTable

        # Create hash table
        hashtable = HashTable(
            embedding_shape=[32],
            block_size=100,
            dtype=torch.float32,
            device="cuda",
        )

        # Create test IDs
        ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64, device="cuda")

        # Track memory access
        with MemoryAccessTracker.context("hashtable_forward_test") as ctx:
            embeddings = hashtable(ids)

        print(f"Output shape: {embeddings.shape}")
        print("\nMemory access summary:")
        summary = ctx.summary()
        print(f"  Total bytes: {summary['total_bytes']}")
        print(f"  Total MB: {summary['total_bytes'] / (1024**2):.4f}")
        print(f"  Operations: {summary['ops']}")

        return ctx

    except Exception as e:
        print(f"Error testing HashTable: {e}")
        return None


def test_combined_tracking():
    """Test combined memory access tracking for multiple operations."""
    print("\n" + "=" * 60)
    print("Testing combined tracking")
    print("=" * 60)

    try:
        from recis.nn.modules.hashtable import HashTable

        # Create hash table
        hashtable = HashTable(
            embedding_shape=[16],
            block_size=50,
            dtype=torch.float32,
            device="cuda",
        )

        # Track multiple operations in one context
        with MemoryAccessTracker.context("combined_test") as ctx:
            # Operation 1: fused_bucketized
            values_list = [
                torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="cuda"),
            ]
            boundaries_list = [
                torch.tensor([1.5, 2.5], dtype=torch.float32, device="cuda"),
            ]
            torch.ops.recis.fused_bucketized(values_list, boundaries_list)

            # Operation 2: hashtable forward
            ids = torch.tensor([10, 20, 30], dtype=torch.int64, device="cuda")
            hashtable(ids)

            # Check intermediate stats
            print(f"Intermediate total bytes: {ctx.total_bytes}")

        print("\nFinal memory access summary:")
        summary = ctx.summary()
        print(f"  Total bytes: {summary['total_bytes']}")
        print(f"  Total MB: {summary['total_bytes'] / (1024**2):.4f}")
        print(f"  Total GB: {summary['total_bytes'] / (1024**3):.6f}")
        print("  Operations breakdown:")
        for op_name, stats in summary["ops"].items():
            print(f"    {op_name}: {stats['count']} calls, {stats['bytes']} bytes")

        return ctx

    except Exception as e:
        print(f"Error in combined test: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Run all tests."""
    print("=" * 60)
    print("Memory Access Tracking Test Suite")
    print("=" * 60)

    # Initialize memory access tracking
    print("\nInitializing memory access tracking...")
    setup_memory_access_tracking()

    # List registered operations
    print(f"Registered operations: {MemoryAccessTracker.list_registered_ops()}")

    # Run tests
    if torch.cuda.is_available():
        test_fused_bucketized()
        test_hashtable_forward()
        test_combined_tracking()
    else:
        print("\nCUDA not available, skipping GPU tests")
        print("Note: fused_bucketized requires CUDA")

    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
