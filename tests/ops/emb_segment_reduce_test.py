import unittest

import torch


def equal(a: torch.Tensor, b: torch.Tensor):
    return torch.allclose(a, b, rtol=1e-06, atol=1e-06)


class EmbeddingSegmentReduce(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        unique_emb: torch.Tensor,
        weight: torch.Tensor,
        reverse_indices: torch.Tensor,
        offsets: torch.Tensor,
        mode: str,
    ):
        ctx.mode = mode
        ctx.unique_size = unique_emb.size(0)
        ctx.save_for_backward(weight, reverse_indices, offsets)
        emb = torch.ops.recis.segment_reduce_forward(
            unique_emb, weight, reverse_indices, offsets, mode
        )
        return emb

    def backward(ctx, grad):
        weight, reverse_indices, offsets = ctx.saved_tensors
        unique_size = ctx.unique_size
        unique_emb_grad = torch.ops.recis.segment_reduce_backward(
            grad.clone(), weight, reverse_indices, offsets, unique_size, ctx.mode
        )
        return unique_emb_grad, None, None, None, None


def embedding_segment_reduce(
    unique_emb: torch.Tensor,
    weight: torch.Tensor,
    reverse_indices: torch.Tensor,
    offsets: torch.Tensor,
    mode: str,
):
    return EmbeddingSegmentReduce.apply(
        unique_emb, weight, reverse_indices, offsets, mode
    )


class MergeOffsetsTest(unittest.TestCase):
    def test_merge_offsets(self):
        offsets_list = [torch.tensor([0, 2, 5]).cuda(), torch.tensor([0, 3, 6]).cuda()]
        max_value = torch.tensor([5, 6])
        ans = torch.tensor([0, 2, 5, 8, 11]).cuda()
        ret = torch.ops.recis.merge_offsets(offsets_list, max_value)
        self.assertTrue(torch.equal(ret, ans))

    def test_empty_merge_offsets(self):
        offsets_list = [torch.tensor([0, 0, 0]).cuda(), torch.tensor([0, 0, 0]).cuda()]
        max_value = torch.tensor([0, 0])
        ans = torch.tensor([0] * 5).cuda()
        ret = torch.ops.recis.merge_offsets(offsets_list, max_value)
        self.assertTrue(torch.equal(ret, ans))


class SegmentReduceTest(unittest.TestCase):
    def test_segment_reduce_forward_and_backward(self):
        emb_dim = 64
        unique_emb = torch.randn([6, emb_dim], dtype=torch.float).cuda()
        reverse_indices = torch.tensor([0, 1, 2, 0, 1, 2]).cuda()
        offsets = torch.tensor([0, 3, 6]).cuda()
        weight = torch.randn([6], dtype=torch.float).cuda()
        for mode in ["mean", "sum", "tile"]:
            with self.subTest(mode=mode):
                w = torch.nn.Parameter(unique_emb)
                w.requires_grad = True
                w.retain_grad()
                w_2 = torch.nn.Parameter(unique_emb)
                w_2.requires_grad = True
                w_2.retain_grad()

                ret = embedding_segment_reduce(
                    w_2, weight, reverse_indices, offsets, mode
                )
                emb = w[reverse_indices] * weight.view(-1, 1)

                if mode == "sum":
                    emb = emb.view(-1, 3, emb_dim)
                    emb = emb.sum(dim=1)
                elif mode == "mean":
                    emb = emb.view(-1, 3, emb_dim)
                    emb = emb.sum(dim=1)
                    weight_sum = weight.view(-1, 3, 1).sum(dim=1)
                    emb = emb / weight_sum
                self.assertTrue(equal(ret.cpu(), emb.cpu()))
                ret = ret.sum()
                ret.backward()
                ans = emb.sum()
                ans.backward()
                self.assertTrue(equal(w.grad.cpu(), w_2.grad.cpu()))

    def test_empty_segment_reduce_forward_and_backward(self):
        emb_dim = 64
        unique_emb = torch.randn([0, emb_dim], dtype=torch.float).cuda()
        reverse_indices = torch.tensor([], dtype=torch.int64).cuda()
        batch_size = 4
        offsets = torch.tensor([0, 0, 0, 0, 0]).cuda()
        weight = torch.randn([0], dtype=torch.float).cuda()
        for mode in ["mean", "sum", "tile"]:
            with self.subTest(mode=mode):
                w = torch.nn.Parameter(unique_emb)
                w.requires_grad = True
                w.retain_grad()
                ret = embedding_segment_reduce(
                    w, weight, reverse_indices, offsets, mode
                )
                if mode == "tile":
                    self.assertTrue(equal(ret.cpu(), torch.zeros([0, emb_dim])))
                else:
                    self.assertTrue(
                        equal(ret.cpu(), torch.zeros([batch_size, emb_dim]))
                    )
                ret = ret.sum()
                ret.backward()
                self.assertTrue(equal(w.grad.cpu(), torch.zeros([0, emb_dim])))


class GenSegmentIndicesTest(unittest.TestCase):
    def test_gen_segment_indices(self):
        offsets = torch.tensor([0, 2, 5]).cuda()
        ret = torch.ops.recis.gen_segment_indices_by_offset(offsets, 5)
        ans = torch.tensor([0, 0, 1, 1, 1]).cuda()
        self.assertTrue(torch.equal(ret, ans))

    def test_empty_gen_segment_indices(self):
        offsets = torch.tensor([0]).cuda()
        ret = torch.ops.recis.gen_segment_indices_by_offset(offsets, 0)
        ans = torch.tensor([]).cuda()
        self.assertTrue(torch.equal(ret, ans))


if __name__ == "__main__":
    unittest.main()
