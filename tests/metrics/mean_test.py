import unittest

import torch

from recis.metrics.mean import MeanMetric


class TestMeanMetric(unittest.TestCase):
    def test_basic_mean(self):
        """Test basic mean calculation without weights."""
        metric = MeanMetric()
        metric.update([1.0, 2.0, 3.0])
        result = metric.compute()
        self.assertAlmostEqual(float(result), 2.0, places=4)

    def test_weighted_mean(self):
        """Test weighted mean calculation."""
        metric = MeanMetric()
        metric.update([1.0, 2.0, 3.0], weights=[1.0, 2.0, 1.0])
        result = metric.compute()
        # (1*1 + 2*2 + 3*1) / (1 + 2 + 1) = 8 / 4 = 2.0
        self.assertAlmostEqual(float(result), 2.0, places=4)

    def test_multiple_updates(self):
        """Test accumulation across multiple updates."""
        metric = MeanMetric()
        metric.update([1.0, 2.0, 3.0])
        metric.update([4.0, 5.0])
        result = metric.compute()
        # (1 + 2 + 3 + 4 + 5) / 5 = 15 / 5 = 3.0
        self.assertAlmostEqual(float(result), 3.0, places=4)

    def test_multiple_updates_with_weights(self):
        """Test accumulation with weights across multiple updates."""
        metric = MeanMetric()
        metric.update([1.0, 2.0, 3.0])
        metric.update([4.0, 5.0], weights=[0.5, 1.5])
        result = metric.compute()
        # (1 + 2 + 3 + 4*0.5 + 5*1.5) / (1 + 1 + 1 + 0.5 + 1.5) = 15.5 / 5 = 3.1
        self.assertAlmostEqual(float(result), 3.1, places=4)

    def test_reset(self):
        """Test reset functionality."""
        metric = MeanMetric()
        metric.update([1.0, 2.0, 3.0])
        metric.reset()
        metric.update([4.0, 5.0, 6.0])
        result = metric.compute()
        self.assertAlmostEqual(float(result), 5.0, places=4)

    def test_callable(self):
        """Test using metric as a callable."""
        metric = MeanMetric()
        result = metric([1.0, 2.0, 3.0])
        self.assertAlmostEqual(float(result), 2.0, places=4)

    def test_tensor_input(self):
        """Test with tensor input."""
        metric = MeanMetric()
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        metric.update(values)
        result = metric.compute()
        self.assertAlmostEqual(float(result), 2.5, places=4)

    def test_tensor_input_with_weights(self):
        """Test with tensor input and weights."""
        metric = MeanMetric()
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        weights = torch.tensor([1.0, 1.0, 2.0, 2.0])
        metric.update(values, weights)
        result = metric.compute()
        # (1*1 + 2*1 + 3*2 + 4*2) / (1 + 1 + 2 + 2) = 17 / 6 = 2.8333
        self.assertAlmostEqual(float(result), 2.8333, places=4)

    def test_scalar_input(self):
        """Test with scalar input."""
        metric = MeanMetric()
        metric.update(5.0)
        metric.update(10.0)
        result = metric.compute()
        self.assertAlmostEqual(float(result), 7.5, places=4)

    def test_scalar_weight(self):
        """Test with scalar weight that broadcasts."""
        metric = MeanMetric()
        metric.update([1.0, 2.0, 3.0], weights=2.0)
        result = metric.compute()
        # All values weighted by 2.0, mean should be same as unweighted
        self.assertAlmostEqual(float(result), 2.0, places=4)

    def test_empty_metric(self):
        """Test compute on empty metric returns zero."""
        metric = MeanMetric()
        result = metric.compute()
        self.assertAlmostEqual(float(result), 0.0, places=4)

    def test_broadcast_weights(self):
        """Test weight broadcasting to match value shape."""
        metric = MeanMetric()
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        weights = torch.tensor([1.0, 2.0])
        metric.update(values, weights)
        result = metric.compute()
        # weights broadcast to [[1.0, 2.0], [1.0, 2.0]]
        # (1*1 + 2*2 + 3*1 + 4*2) / (1 + 2 + 1 + 2) = 16 / 6 = 2.6667
        self.assertAlmostEqual(float(result), 2.6667, places=4)

    def test_dtype_float64(self):
        """Test with float64 dtype."""
        metric = MeanMetric(dtype=torch.float64)
        metric.update([1.0, 2.0, 3.0])
        result = metric.compute()
        self.assertEqual(result.dtype, torch.float64)
        self.assertAlmostEqual(float(result), 2.0, places=4)


if __name__ == "__main__":
    unittest.main()
