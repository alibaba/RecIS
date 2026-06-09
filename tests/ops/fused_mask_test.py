"""Unit tests for fused_string_mask and fused_number_mask operations."""

import unittest

import torch

from recis.nn.functional.fused_ops import fused_number_mask, fused_string_mask


class TestFusedStringMask(unittest.TestCase):
    """Test cases for fused_string_mask operation."""

    def test_basic_string_mask(self):
        """Test basic string mask functionality."""
        # Prepare input: two strings "abc" and "de" encoded as int8
        # "abc" = [97, 98, 99], "de" = [100, 101]
        inputs = [torch.tensor([97, 98, 99, 100, 101], dtype=torch.int8, device="cuda")]
        input_offsets = [torch.tensor([0, 3, 5], dtype=torch.int64, device="cuda")]
        masks = [["abc"]]  # Only "abc" should be masked (output 0.0)

        results = fused_string_mask(inputs, input_offsets, masks)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].shape, torch.Size([2]))
        # "abc" matches mask -> 0.0, "de" does not match -> 1.0
        self.assertTrue(
            torch.allclose(results[0], torch.tensor([0.0, 1.0], device="cuda"))
        )

    def test_multiple_masks(self):
        """Test with multiple mask strings."""
        # Strings: "cat", "dog", "bird"
        # "cat" = [99, 97, 116], "dog" = [100, 111, 103], "bird" = [98, 105, 114, 100]
        inputs = [
            torch.tensor(
                [99, 97, 116, 100, 111, 103, 98, 105, 114, 100],
                dtype=torch.int8,
                device="cuda",
            )
        ]
        input_offsets = [torch.tensor([0, 3, 6, 10], dtype=torch.int64, device="cuda")]
        masks = [["cat", "bird"]]  # "cat" and "bird" should be masked

        results = fused_string_mask(inputs, input_offsets, masks)

        # "cat" -> 0.0, "dog" -> 1.0, "bird" -> 0.0
        self.assertTrue(
            torch.allclose(results[0], torch.tensor([0.0, 1.0, 0.0], device="cuda"))
        )

    def test_multiple_inputs(self):
        """Test with multiple input tensors, each containing multiple strings."""
        # Input 1: "hello" + "world" = [104, 101, 108, 108, 111, 119, 111, 114, 108, 100]
        # Input 2: "cat" + "dog" + "bird" = [99, 97, 116, 100, 111, 103, 98, 105, 114, 100]
        inputs = [
            torch.tensor(
                [104, 101, 108, 108, 111, 119, 111, 114, 108, 100],
                dtype=torch.int8,
                device="cuda",
            ),
            torch.tensor(
                [99, 97, 116, 100, 111, 103, 98, 105, 114, 100],
                dtype=torch.int8,
                device="cuda",
            ),
        ]
        input_offsets = [
            torch.tensor(
                [0, 5, 10], dtype=torch.int64, device="cuda"
            ),  # "hello", "world"
            torch.tensor(
                [0, 3, 6, 10], dtype=torch.int64, device="cuda"
            ),  # "cat", "dog", "bird"
        ]
        masks = [
            ["hello"],  # Only "hello" is masked in first input
            ["dog", "bird"],  # "dog" and "bird" are masked in second input
        ]

        results = fused_string_mask(inputs, input_offsets, masks)

        self.assertEqual(len(results), 2)
        # Input 1: "hello" -> 0.0 (matched), "world" -> 1.0 (not matched)
        self.assertTrue(
            torch.allclose(results[0], torch.tensor([0.0, 1.0], device="cuda"))
        )
        # Input 2: "cat" -> 1.0, "dog" -> 0.0, "bird" -> 0.0
        self.assertTrue(
            torch.allclose(results[1], torch.tensor([1.0, 0.0, 0.0], device="cuda"))
        )

    def test_empty_mask(self):
        """Test with empty mask list - all strings should not be masked."""
        inputs = [torch.tensor([97, 98, 99], dtype=torch.int8, device="cuda")]
        input_offsets = [torch.tensor([0, 3], dtype=torch.int64, device="cuda")]
        masks = [[]]  # No masks

        results = fused_string_mask(inputs, input_offsets, masks)

        self.assertTrue(torch.allclose(results[0], torch.tensor([1.0], device="cuda")))

    def test_no_match(self):
        """Test when no string matches the mask."""
        inputs = [torch.tensor([97, 98, 99], dtype=torch.int8, device="cuda")]
        input_offsets = [torch.tensor([0, 3], dtype=torch.int64, device="cuda")]
        masks = [["xyz"]]

        results = fused_string_mask(inputs, input_offsets, masks)

        self.assertTrue(torch.allclose(results[0], torch.tensor([1.0], device="cuda")))


class TestFusedNumberMask(unittest.TestCase):
    """Test cases for fused_number_mask operation."""

    def test_float32_mask(self):
        """Test float32 tensor masking."""
        inputs = [
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32, device="cuda")
        ]
        masks = [[1.0, 3.0]]  # Mask 1.0 and 3.0

        results = fused_number_mask(inputs, masks)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].shape, torch.Size([5]))
        expected = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0], device="cuda")
        self.assertTrue(torch.allclose(results[0], expected))

    def test_float64_mask(self):
        """Test float64 tensor masking."""
        inputs = [torch.tensor([1.5, 2.5, 3.5], dtype=torch.float64, device="cuda")]
        masks = [[2.5]]

        results = fused_number_mask(inputs, masks)

        expected = torch.tensor([1.0, 0.0, 1.0], device="cuda")
        self.assertTrue(torch.allclose(results[0], expected))

    def test_int32_mask(self):
        """Test int32 tensor masking."""
        inputs = [torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32, device="cuda")]
        masks = [[20.0, 40.0]]

        results = fused_number_mask(inputs, masks)

        expected = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0], device="cuda")
        self.assertTrue(torch.allclose(results[0], expected))

    def test_int64_mask(self):
        """Test int64 tensor masking."""
        inputs = [torch.tensor([100, 200, 300], dtype=torch.int64, device="cuda")]
        masks = [[100.0, 300.0]]

        results = fused_number_mask(inputs, masks)

        expected = torch.tensor([0.0, 1.0, 0.0], device="cuda")
        self.assertTrue(torch.allclose(results[0], expected))

    def test_multiple_inputs_same_dtype(self):
        """Test with multiple inputs of the same dtype."""
        inputs = [
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="cuda"),
            torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32, device="cuda"),
        ]
        masks = [[1.0], [5.0]]

        results = fused_number_mask(inputs, masks)

        self.assertEqual(len(results), 2)
        self.assertTrue(
            torch.allclose(results[0], torch.tensor([0.0, 1.0, 1.0], device="cuda"))
        )
        self.assertTrue(
            torch.allclose(results[1], torch.tensor([1.0, 0.0, 1.0], device="cuda"))
        )

    def test_empty_mask(self):
        """Test with empty mask list - all values should not be masked."""
        inputs = [torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="cuda")]
        masks = [[]]

        results = fused_number_mask(inputs, masks)

        expected = torch.tensor([1.0, 1.0, 1.0], device="cuda")
        self.assertTrue(torch.allclose(results[0], expected))

    def test_no_match(self):
        """Test when no value matches the mask."""
        inputs = [torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="cuda")]
        masks = [[99.0, 100.0]]

        results = fused_number_mask(inputs, masks)

        expected = torch.tensor([1.0, 1.0, 1.0], device="cuda")
        self.assertTrue(torch.allclose(results[0], expected))

    def test_negative_values(self):
        """Test with negative values."""
        inputs = [torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32, device="cuda")]
        masks = [[-1.0, 0.0]]

        results = fused_number_mask(inputs, masks)

        expected = torch.tensor([0.0, 0.0, 1.0], device="cuda")
        self.assertTrue(torch.allclose(results[0], expected))

    def test_large_tensor(self):
        """Test with a larger tensor."""
        values = torch.randn(1000, dtype=torch.float32, device="cuda")
        values[100] = 0.0
        values[500] = 0.0
        inputs = [values]
        masks = [[0.0]]

        results = fused_number_mask(inputs, masks)

        self.assertEqual(results[0].sum().item(), 998.0)  # 998 values are 1.0
        self.assertEqual(results[0][100].item(), 0.0)
        self.assertEqual(results[0][500].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
