import unittest

import torch

from recis.nn.functional.fused_ops import fused_int64_to_string_int8


class TestFusedInt64ToStringInt8(unittest.TestCase):
    def test_fused_int64_to_string_int8_single(self):
        """Test fused conversion with single tensor."""
        inputs = [torch.tensor([123], dtype=torch.int64).cuda()]
        outputs, offsets = fused_int64_to_string_int8(inputs)

        # "123" -> [49, 50, 51]
        expected_output = torch.tensor([49, 50, 51], dtype=torch.int8).cuda()
        expected_offsets = torch.tensor([0, 3], dtype=torch.int64).cuda()

        self.assertTrue(len(outputs) == 1)
        self.assertTrue(len(offsets) == 1)
        self.assertTrue(
            torch.equal(outputs[0], expected_output),
            f"Expected {expected_output}, got {outputs[0]}",
        )
        self.assertTrue(
            torch.equal(offsets[0], expected_offsets),
            f"Expected {expected_offsets}, got {offsets[0]}",
        )

    def test_fused_int64_to_string_int8_multiple_tensors(self):
        """Test fused conversion with multiple tensors."""
        inputs = [
            torch.tensor([123, -456], dtype=torch.int64).cuda(),
            torch.tensor([0, 789], dtype=torch.int64).cuda(),
            torch.tensor([100], dtype=torch.int64).cuda(),
        ]
        outputs, offsets = fused_int64_to_string_int8(inputs)

        self.assertTrue(len(outputs) == 3)
        self.assertTrue(len(offsets) == 3)

        # First tensor: "123" -> [49, 50, 51], "-456" -> [45, 52, 53, 54]
        expected_output_0 = torch.tensor(
            [49, 50, 51, 45, 52, 53, 54], dtype=torch.int8
        ).cuda()
        expected_offsets_0 = torch.tensor([0, 3, 7], dtype=torch.int64).cuda()
        self.assertTrue(
            torch.equal(outputs[0], expected_output_0),
            f"Expected {expected_output_0}, got {outputs[0]}",
        )
        self.assertTrue(
            torch.equal(offsets[0], expected_offsets_0),
            f"Expected {expected_offsets_0}, got {offsets[0]}",
        )

        # Second tensor: "0" -> [48], "789" -> [55, 56, 57]
        expected_output_1 = torch.tensor([48, 55, 56, 57], dtype=torch.int8).cuda()
        expected_offsets_1 = torch.tensor([0, 1, 4], dtype=torch.int64).cuda()
        self.assertTrue(
            torch.equal(outputs[1], expected_output_1),
            f"Expected {expected_output_1}, got {outputs[1]}",
        )
        self.assertTrue(
            torch.equal(offsets[1], expected_offsets_1),
            f"Expected {expected_offsets_1}, got {offsets[1]}",
        )

        # Third tensor: "100" -> [49, 48, 48]
        expected_output_2 = torch.tensor([49, 48, 48], dtype=torch.int8).cuda()
        expected_offsets_2 = torch.tensor([0, 3], dtype=torch.int64).cuda()
        self.assertTrue(
            torch.equal(outputs[2], expected_output_2),
            f"Expected {expected_output_2}, got {outputs[2]}",
        )
        self.assertTrue(
            torch.equal(offsets[2], expected_offsets_2),
            f"Expected {expected_offsets_2}, got {offsets[2]}",
        )

    def test_fused_int64_to_string_int8_negative(self):
        """Test fused conversion with negative numbers."""
        inputs = [
            torch.tensor([-456], dtype=torch.int64).cuda(),
            torch.tensor([-1, -999], dtype=torch.int64).cuda(),
        ]
        outputs, offsets = fused_int64_to_string_int8(inputs)

        # First tensor: "-456" -> [45, 52, 53, 54]
        expected_output_0 = torch.tensor([45, 52, 53, 54], dtype=torch.int8).cuda()
        expected_offsets_0 = torch.tensor([0, 4], dtype=torch.int64).cuda()
        self.assertTrue(
            torch.equal(outputs[0], expected_output_0),
            f"Expected {expected_output_0}, got {outputs[0]}",
        )
        self.assertTrue(
            torch.equal(offsets[0], expected_offsets_0),
            f"Expected {expected_offsets_0}, got {offsets[0]}",
        )

        # Second tensor: "-1" -> [45, 49], "-999" -> [45, 57, 57, 57]
        expected_output_1 = torch.tensor(
            [45, 49, 45, 57, 57, 57], dtype=torch.int8
        ).cuda()
        expected_offsets_1 = torch.tensor([0, 2, 6], dtype=torch.int64).cuda()
        self.assertTrue(
            torch.equal(outputs[1], expected_output_1),
            f"Expected {expected_output_1}, got {outputs[1]}",
        )
        self.assertTrue(
            torch.equal(offsets[1], expected_offsets_1),
            f"Expected {expected_offsets_1}, got {offsets[1]}",
        )

    def test_fused_int64_to_string_int8_zero(self):
        """Test fused conversion with zeros."""
        inputs = [
            torch.tensor([0], dtype=torch.int64).cuda(),
            torch.tensor([0, 0, 0], dtype=torch.int64).cuda(),
        ]
        outputs, offsets = fused_int64_to_string_int8(inputs)

        # First tensor: "0" -> [48]
        expected_output_0 = torch.tensor([48], dtype=torch.int8).cuda()
        expected_offsets_0 = torch.tensor([0, 1], dtype=torch.int64).cuda()
        self.assertTrue(
            torch.equal(outputs[0], expected_output_0),
            f"Expected {expected_output_0}, got {outputs[0]}",
        )
        self.assertTrue(
            torch.equal(offsets[0], expected_offsets_0),
            f"Expected {expected_offsets_0}, got {offsets[0]}",
        )

        # Second tensor: "0", "0", "0" -> [48, 48, 48]
        expected_output_1 = torch.tensor([48, 48, 48], dtype=torch.int8).cuda()
        expected_offsets_1 = torch.tensor([0, 1, 2, 3], dtype=torch.int64).cuda()
        self.assertTrue(
            torch.equal(outputs[1], expected_output_1),
            f"Expected {expected_output_1}, got {outputs[1]}",
        )
        self.assertTrue(
            torch.equal(offsets[1], expected_offsets_1),
            f"Expected {expected_offsets_1}, got {offsets[1]}",
        )


if __name__ == "__main__":
    unittest.main()
