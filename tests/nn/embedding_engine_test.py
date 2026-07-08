import os
import unittest

import torch
import torch.testing._internal.common_utils as common

from recis.nn.initializers import TruncNormalInitializer
from recis.nn.modules.embedding import EmbeddingOption, NoReduceEmbedding
from recis.nn.modules.embedding_engine import EmbeddingEngine
from recis.ragged.tensor import RaggedTensor


class EmbeddingEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(common.find_free_port())
        os.environ["RANK"] = "0"
        torch.distributed.init_process_group()

    def test_embedding_engine(self):
        id1 = torch.tensor(
            [5, 16385, 1, 32800, 16385], dtype=torch.int64, device="cuda"
        )
        row_splits_1 = torch.tensor([0, 1, 3, 5], dtype=torch.int64, device="cuda")
        rt1 = RaggedTensor(id1, row_splits_1)
        emb_opt1 = EmbeddingOption(
            embedding_dim=8,
            shared_name="ht2",
            combiner="sum",
            initializer=TruncNormalInitializer(mean=0, std=0.01),
        )

        id2 = torch.tensor(
            [9155707040084860980, 11, 12, 20, 21, 30, 31, 32, 33],
            dtype=torch.int64,
            device="cuda",
        )
        row_splits_2 = torch.tensor([0, 3, 5, 9], dtype=torch.int64, device="cuda")
        rt2 = RaggedTensor(id2, row_splits_2)
        emb_opt2 = EmbeddingOption(
            embedding_dim=8,
            shared_name="ht1",
            combiner="mean",
            initializer=TruncNormalInitializer(mean=0, std=0.01),
        )
        ee = EmbeddingEngine({"fea1": emb_opt1, "fea2": emb_opt2})
        out = ee({"fea1": rt1, "fea2": rt2})
        print(out)

    def _make_no_reduce_inputs(self):
        seq_ids = torch.tensor(
            [10, 20, 30, 40, 50, 11, 21, 31, 12, 22, 32, 42], dtype=torch.int64,
            device="cuda",
        )
        seq_offsets = torch.tensor([0, 5, 8, 12], dtype=torch.int64, device="cuda")
        return RaggedTensor(seq_ids, seq_offsets)

    def test_no_reduce_forward(self):
        rt_normal = RaggedTensor(
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64, device="cuda"),
            torch.tensor([0, 2, 3, 5], dtype=torch.int64, device="cuda"),
        )
        rt_seq = self._make_no_reduce_inputs()
        emb_opt_normal = EmbeddingOption(
            embedding_dim=8,
            shared_name="user_id_table",
            combiner="sum",
            initializer=TruncNormalInitializer(mean=0, std=0.01),
        )
        emb_opt_seq = EmbeddingOption(
            embedding_dim=8,
            shared_name="item_table",
            no_reduce=True,
            initializer=TruncNormalInitializer(mean=0, std=0.01),
        )
        ee = EmbeddingEngine({"normal": emb_opt_normal, "seq": emb_opt_seq})
        out = ee({"normal": rt_normal, "seq": rt_seq})

        self.assertIsInstance(out["normal"], torch.Tensor)
        self.assertEqual(out["normal"].shape, (3, 8))

        nr = out["seq"]
        self.assertIsInstance(nr, NoReduceEmbedding)
        self.assertEqual(nr.emb.dim(), 2)
        self.assertEqual(nr.emb.shape[1], 8)
        # offsets propagated through unchanged (values match the input).
        self.assertTrue(torch.equal(nr.offsets.cpu(), rt_seq.offsets()[-1].cpu()))
        # reverse_index lets us recover per-id embeddings.
        per_id = nr.emb[nr.reverse_index]
        self.assertEqual(per_id.shape, (rt_seq.values().numel(), 8))

    def test_no_reduce_backward(self):
        rt_seq = self._make_no_reduce_inputs()
        emb_opt = EmbeddingOption(
            embedding_dim=8,
            shared_name="bw_table",
            no_reduce=True,
            initializer=TruncNormalInitializer(mean=0, std=0.01),
        )
        ee = EmbeddingEngine({"seq": emb_opt})
        out = ee({"seq": rt_seq})

        nr = out["seq"]
        self.assertTrue(nr.emb.requires_grad)
        self.assertIsNotNone(nr.emb.grad_fn)
        # Should run end-to-end through EmbeddingExchange's backward a2a.
        nr.emb.mean().backward()

    def test_no_reduce_not_trainable(self):
        rt_seq = self._make_no_reduce_inputs()
        emb_opt = EmbeddingOption(
            embedding_dim=8,
            shared_name="frozen_table",
            no_reduce=True,
            trainable=False,
            initializer=TruncNormalInitializer(mean=0, std=0.01),
        )
        ee = EmbeddingEngine({"seq": emb_opt})
        out = ee({"seq": rt_seq})

        nr = out["seq"]
        self.assertIsInstance(nr, NoReduceEmbedding)
        self.assertFalse(nr.emb.requires_grad)

    @classmethod
    def tearDownClass(cls):
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
