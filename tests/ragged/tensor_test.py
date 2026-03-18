import copy

import pytest
import torch

from recis.ragged.tensor import RaggedPadInfo, RaggedTensor


class TestRaggedTensor:
    """Test cases for RaggedTensor class."""

    def test_basic_creation(self):
        """Test basic RaggedTensor creation."""
        values = torch.tensor([1, 2, 3, 4, 5, 6])
        offsets = [torch.tensor([0, 2, 5, 6])]  # 3 sequences: [1,2], [3,4,5], [6]

        ragged = RaggedTensor(values, offsets)

        assert ragged.shape == (3, 3)  # 3 sequences, max length 3
        assert ragged.dim == 2
        assert ragged.max_length == 3
        assert ragged.dtype == torch.int64
        assert ragged.device.type == "cpu"

    def test_from_dense(self):
        """Test creating RaggedTensor from dense tensor."""
        dense = torch.tensor([[1, 2, 3, 0, 0], [6, 7, 8, 0, 0]])
        ragged = RaggedTensor.from_dense(dense)

        rows = dense.shape[0]
        cols = dense.shape[1]
        assert torch.equal(ragged.values(), dense.view(-1))
        assert torch.equal(
            ragged.offsets()[0],
            torch.arange(0, (rows + 1) * cols, cols, device="cpu").to(torch.int),
        )

    def test_to_dense(self):
        """Test converting RaggedTensor to dense tensor."""
        values = torch.tensor([1, 2, 3, 4, 5, 6])
        offsets = [torch.tensor([0, 2, 5, 6])]
        ragged = RaggedTensor(values, offsets)

        dense = ragged.to_dense()
        expected = torch.tensor([[1, 2, 0], [3, 4, 5], [6, 0, 0]])

        assert torch.equal(dense, expected)

    def test_to_dense_with_custom_padding(self):
        """Test converting to dense with custom padding value."""
        values = torch.tensor([1, 2, 3, 4, 5, 6])
        offsets = [torch.tensor([0, 2, 5, 6])]
        ragged = RaggedTensor(values, offsets)

        dense = ragged.to_dense(default_value=-1.0)
        expected = torch.tensor([[1, 2, -1], [3, 4, 5], [6, -1, -1]])

        assert torch.equal(dense, expected)

    def test_device_operations(self):
        """Test device operations (cuda if available)."""
        values = torch.tensor([1, 2, 3, 4, 5, 6])
        offsets = [torch.tensor([0, 2, 5, 6])]
        ragged = RaggedTensor(values, offsets)

        if torch.cuda.is_available():
            cuda_ragged = ragged.cuda()
            assert cuda_ragged.device.type == "cuda"

            # Test to() method
            cpu_ragged = cuda_ragged.to("cpu")
            assert cpu_ragged.device.type == "cpu"
        else:
            # Test to() method on CPU
            cpu_ragged = ragged.to("cpu")
            assert cpu_ragged.device.type == "cpu"

    def test_clone_shallow_copy(self):
        """Test shallow copy behavior of clone()."""
        values = torch.tensor([1, 2, 3, 4, 5, 6])
        offsets = [torch.tensor([0, 2, 5, 6])]
        ragged = RaggedTensor(values, offsets)

        cloned = ragged.clone()

        # Should be different objects
        assert cloned is not ragged

        # But should share the same underlying data (shallow copy)
        assert torch.equal(cloned.values(), ragged.values())
        assert torch.equal(cloned.offsets()[0], ragged.offsets()[0])

        # Modifying cloned values should affect original (shallow copy behavior)
        cloned._values[0] = 999
        assert ragged._values[0] == 999

    def test_deepcopy(self):
        """Test deep copy behavior with copy.deepcopy()."""
        values = torch.tensor([1, 2, 3, 4, 5, 6])
        offsets = [torch.tensor([0, 2, 5, 6])]
        ragged = RaggedTensor(values, offsets)

        deep_copied = copy.deepcopy(ragged)

        # Should be different objects
        assert deep_copied is not ragged

        # Should have same values initially
        assert torch.equal(deep_copied.values(), ragged.values())
        assert torch.equal(deep_copied.offsets()[0], ragged.offsets()[0])

        # Modifying deep copied values should NOT affect original
        deep_copied._values[0] = 999
        assert ragged._values[0] == 1  # Original unchanged

    def test_pad_info(self):
        """Test padding information functionality."""
        values = torch.tensor([1, 2, 3, 4, 5, 6])
        offsets = [torch.tensor([0, 2, 5, 6])]
        ragged = RaggedTensor(values, offsets)

        # Initially no pad info
        assert ragged.pad_info is None

        # Set pad info
        pad_info = RaggedPadInfo(
            drop_nums=[torch.tensor([1, 0, 2])],
            pad_nums=[torch.tensor([0, 1, 0])],
            drop_sides=torch.tensor([0, 1, 0]),
            pad_sides=torch.tensor([1, 0, 1]),
        )
        ragged.set_pad_info(pad_info)

        assert ragged.pad_info is not None
        assert ragged.pad_info.drop_nums[0].tolist() == [1, 0, 2]

    def test_weight_functionality(self):
        """Test weight tensor functionality."""
        values = torch.tensor([1, 2, 3, 4, 5, 6])
        offsets = [torch.tensor([0, 2, 5, 6])]
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        ragged = RaggedTensor(values, offsets, weight=weights)

        assert ragged.weight() is not None
        assert torch.equal(ragged.weight(), weights)

        # Test setting new weight
        new_weights = torch.ones_like(weights)
        ragged.set_weight(new_weights)
        assert torch.equal(ragged.weight(), new_weights)

    def test_string_representation(self):
        """Test string representation methods."""
        values = torch.tensor([1, 2, 3, 4, 5, 6])
        offsets = [torch.tensor([0, 2, 5, 6])]
        ragged = RaggedTensor(values, offsets)

        # Test __str__ and __repr__
        str_repr = str(ragged)
        repr_repr = repr(ragged)

        assert "RaggedTensor" in str_repr
        assert "values=" in str_repr
        assert "offsets=" in str_repr
        assert "dense_shape=" in str_repr
        assert str_repr == repr_repr  # Should be the same

    def test_dictionary_representation(self):
        """Test that RaggedTensor displays properly in dictionaries."""
        values = torch.tensor([1, 2, 3, 4, 5, 6])
        offsets = [torch.tensor([0, 2, 5, 6])]
        ragged = RaggedTensor(values, offsets)

        data = {"tensor": ragged}
        data_str = str(data)

        # Should show detailed RaggedTensor content, not just object reference
        assert "RaggedTensor" in data_str
        assert "values=" in data_str
        assert "offsets=" in data_str

    def test_complex_deepcopy(self):
        """Test deepcopy with complex nested structures."""
        values = torch.tensor([1, 2, 3, 4, 5, 6])
        offsets = [torch.tensor([0, 2, 5, 6])]
        ragged = RaggedTensor(values, offsets)

        # Create complex nested structure
        complex_data = {
            "tensor": ragged,
            "nested": {"ragged": ragged, "list": [ragged, ragged]},
            "list": [ragged, {"inner": ragged}],
        }

        deep_copied = copy.deepcopy(complex_data)

        # All RaggedTensors should be independent
        deep_copied["tensor"]._values[0] = 999
        assert complex_data["tensor"]._values[0] == 1  # Original unchanged
        assert complex_data["nested"]["ragged"]._values[0] == 1  # Original unchanged
        assert complex_data["list"][0]._values[0] == 1  # Original unchanged

    def test_from_numpy(self):
        """Test creating RaggedTensor from NumPy arrays."""
        import numpy as np

        values = np.array([1, 2, 3, 4, 5, 6])
        offsets = [np.array([0, 2, 5, 6])]

        ragged = RaggedTensor.from_numpy(values, offsets)

        assert isinstance(ragged.values(), torch.Tensor)
        assert isinstance(ragged.offsets()[0], torch.Tensor)
        assert torch.equal(ragged.values(), torch.tensor([1, 2, 3, 4, 5, 6]))

    def test_shape_operations(self):
        """Test shape-related operations."""
        values = torch.tensor([1, 2, 3, 4, 5, 6])
        offsets = [torch.tensor([0, 2, 5, 6])]
        ragged = RaggedTensor(values, offsets)

        # Test real_shape
        assert ragged.real_shape() == (3, 3)
        assert ragged.real_shape(0, 1) == (3,)
        assert ragged.real_shape(1, 2) == (3,)

        # Test get_shape
        assert ragged.get_shape(0) == 3  # Number of sequences
        assert ragged.get_shape(1) == 3  # Max sequence length

        # Test set_dim_by_rank
        ragged.set_dim_by_rank(10, 1)
        assert ragged._dense_shape[1] == 10

    def test_pin_memory(self):
        """Test pin_memory functionality."""
        values = torch.tensor([1, 2, 3, 4, 5, 6])
        offsets = [torch.tensor([0, 2, 5, 6])]
        ragged = RaggedTensor(values, offsets)

        pinned = ragged.pin_memory()

        assert pinned.values().is_pinned()
        assert pinned.offsets()[0].is_pinned()
        assert pinned is not ragged  # Should be a new object

    def test_slice_2D(self):
        """Test __getitem__ for 2D ragged tensor"""
        values = torch.tensor([1, 2, 3, 4, 5, 6])
        offsets = torch.tensor([0, 2, 5, 6])
        ragged = RaggedTensor(values, offsets)
        ragged_copy = copy.deepcopy(ragged)

        # testing indexing the axis 0. 2D -> 1D, ragged -> dense
        assert ragged[0].shape == torch.Size([2])
        assert torch.equal(ragged[-1].to_dense(), torch.tensor([6]))
        # testing slice the axis 0
        assert torch.equal(ragged[1:3].to_dense(), ragged.to_dense()[1:3])
        # testing slice the axis 0 and 1
        assert torch.equal(ragged[1:3, 0:1].to_dense(), ragged.to_dense()[1:3, 0:1])
        assert torch.equal(ragged[0:8, 2:5].to_dense(), ragged.to_dense()[0:8, 2:5])
        assert torch.equal(
            ragged[0:8:1, 0:3:2].to_dense(), ragged.to_dense()[0:8:1, 0:3:2]
        )

        # testing slice the axis 0 and 1 with neg
        assert torch.equal(
            ragged[-3:-1:1, -2::1].to_dense(), torch.tensor([[1, 2], [4, 5]])
        ), f"ragged[-3:-1:1, -2::1].to_dense() is {ragged[-3:-1:1, -2::1].to_dense()}"

        # testing indexing the axis 1
        with pytest.raises(IndexError):
            ragged[0:5, 1]
        # testing add new axis
        assert torch.equal(ragged[None].to_dense(), ragged.to_dense()[None])
        assert torch.equal(ragged[:, None].to_dense(), ragged.to_dense()[:, None])
        assert torch.equal(ragged[:, :, None].to_dense(), ragged.to_dense()[:, :, None])

        assert ragged_copy is not ragged
        assert torch.equal(ragged_copy.to_dense(), ragged.to_dense())

    def test_slice_3D(self):
        values = torch.tensor(
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=torch.int64
        )
        offsets = [
            torch.tensor([0, 3, 4, 6], dtype=torch.int32),
            torch.tensor([0, 1, 4, 4, 8, 8, 10], dtype=torch.int32),
        ]
        ragged = RaggedTensor(values, offsets)
        ragged_copy = copy.deepcopy(ragged)

        # testing index the axis 0. 3D ragged tensor -> 2D ragged tensor
        assert torch.equal(ragged[1].to_dense(), torch.tensor([[14, 15, 16, 17]]))
        # testing slice the axis 0
        assert torch.equal(
            ragged[1:4:1].to_dense(),
            torch.tensor(
                [[[14, 15, 16, 17], [0, 0, 0, 0]], [[0, 0, 0, 0], [18, 19, 0, 0]]]
            ),
        )
        # testing index and slice the axis 0, 1 and 2
        assert torch.equal(ragged[2, :, :1].to_dense(), torch.tensor([[0], [18]]))
        assert torch.equal(ragged[..., :2].to_dense(), ragged.to_dense()[..., :2])
        assert torch.equal(
            ragged[1:4, 1:8, 1:2].to_dense(), torch.tensor([[[0]], [[19]]])
        )
        assert torch.equal(
            ragged[-3:-1, ::, -2:-1].to_dense(),
            torch.tensor([[[0], [12], [0]], [[16], [0], [0]]]),
        )
        assert (
            ragged[1:1, 1:8, 1:2].values().numel() == 0
            and ragged[1:1, 1:8, 1:2].dim == 3
        )

        # testing indexing the axis 1
        with pytest.raises(IndexError):
            ragged[0:5, 1]
        # testing add new axis
        assert torch.equal(ragged[None].to_dense(), ragged.to_dense()[None])
        assert torch.equal(ragged[:, None].to_dense(), ragged.to_dense()[:, None])
        assert torch.equal(ragged[:, :, None].to_dense(), ragged.to_dense()[:, :, None])
        assert torch.equal(
            ragged[:, :, :, None].to_dense(), ragged.to_dense()[:, :, :, None]
        )

        assert ragged_copy is not ragged
        assert torch.equal(ragged_copy.to_dense(), ragged.to_dense())


if __name__ == "__main__":
    pytest.main([__file__])
