import unittest

import torch

from recis.nn.functional.ragged_ops import index_select_ragged_by_padded_index
from recis.ragged.tensor import RaggedPadInfo, RaggedTensor


class TestRagged(unittest.TestCase):
    def ragged_index_select(
        self,
        drop_side,
        pad_side,
        drop_num,
        pad_num,
        values,
        offset,
        indicator,
        topk_index,
    ):
        gatherd_pad_num = pad_num[indicator] * pad_side
        gatherd_dop_num = drop_num[indicator] * drop_side
        gatherd_offset_beg = offset[:-1][indicator]
        gatherd_offset_end = offset[1:][indicator]
        topk_index = (
            topk_index
            - gatherd_pad_num.view(-1, 1)
            + gatherd_dop_num.view(-1, 1)
            + gatherd_offset_beg.view(-1, 1)
        )
        topk_index[topk_index < gatherd_offset_beg.view(-1, 1)] = -1
        topk_index[topk_index >= gatherd_offset_end.view(-1, 1)] = -1
        valid_mask = topk_index != -1
        value = values[topk_index]
        offset = torch.arange(
            0, topk_index.numel() + 1, topk_index.size(1), device=topk_index.device
        )
        value = value * valid_mask
        return RaggedTensor(value.flatten(), [offset]), valid_mask.flatten()

    def test_ragged_topk_index_cutoff(self):
        values, offsets = [], []
        seq_length = 4
        values.append(torch.arange(12).cuda())
        offsets.append(torch.tensor([0, 2, 5, 6, 12]).cuda())  # [2, 3, 1, 6]
        values.append(torch.arange(13).cuda())
        offsets.append(torch.tensor([0, 3, 6, 7, 13]).cuda())
        values.append(torch.arange(11).cuda())
        offsets.append(torch.tensor([0, 2, 5, 9, 11]).cuda())
        r_tensors = [RaggedTensor(v, [o]) for v, o in zip(values, offsets)]
        drop_side = torch.tensor(True).cuda()  # left
        pad_side = torch.tensor(True).cuda()  # left
        drop_num = torch.clamp(
            (offsets[0][1:] - offsets[0][:-1]) - seq_length, min=0, max=seq_length
        )
        pad_num = torch.clamp(
            seq_length - (offsets[0][1:] - offsets[0][:-1]), min=0, max=seq_length
        )
        pad_info = RaggedPadInfo(drop_num, pad_num, drop_side, pad_side)
        indicator = torch.tensor([0, 0, 1, 1, 2, 3]).cuda()
        topk_index = torch.tensor(
            [[0, 3], [1, 2], [2, 3], [1, 3], [0, 3], [1, 3]]
        ).cuda()
        gatherd_tensors, gatherd_masks = index_select_ragged_by_padded_index(
            pad_info, topk_index, indicator, r_tensors
        )
        for idx, (v, o) in enumerate(zip(values, offsets)):
            target_r_tensor, target_mask = self.ragged_index_select(
                drop_side, pad_side, drop_num, pad_num, v, o, indicator, topk_index
            )
            self.assertTrue(
                torch.equal(target_r_tensor.values(), gatherd_tensors[idx].values())
            )
            self.assertTrue(
                torch.equal(
                    target_r_tensor.offsets()[0], gatherd_tensors[idx].offsets()[0]
                )
            )
            self.assertTrue(torch.equal(target_mask, gatherd_masks[idx]))

    def test_ragged_topk_index_cutoff_cuda(self):
        values, offsets = [], []
        seq_length = 4
        values.append(torch.arange(12).cuda())
        offsets.append(torch.tensor([0, 2, 5, 6, 12]).cuda())  # [2, 3, 1, 6]
        values.append(torch.arange(13).cuda())
        offsets.append(torch.tensor([0, 3, 6, 7, 13]).cuda())
        values.append(torch.arange(11).cuda())
        offsets.append(torch.tensor([0, 2, 5, 9, 11]).cuda())
        r_tensors = [
            RaggedTensor(v.cuda(), [o.cuda()]) for v, o in zip(values, offsets)
        ]
        drop_side = torch.tensor(True).cuda()  # left
        pad_side = torch.tensor(True).cuda()  # left
        drop_num = torch.clamp(
            (offsets[0][1:] - offsets[0][:-1]) - seq_length, min=0, max=seq_length
        )
        pad_num = torch.clamp(
            seq_length - (offsets[0][1:] - offsets[0][:-1]), min=0, max=seq_length
        )
        pad_info = RaggedPadInfo(
            drop_num.cuda(), pad_num.cuda(), drop_side.cuda(), pad_side.cuda()
        )
        indicator = torch.tensor([0, 0, 1, 1, 2, 3]).cuda()
        topk_index = torch.tensor(
            [[0, 3], [1, 2], [2, 3], [1, 3], [0, 3], [1, 3]]
        ).cuda()
        gatherd_tensors, gatherd_masks = index_select_ragged_by_padded_index(
            pad_info, topk_index, indicator, r_tensors
        )
        for idx, (v, o) in enumerate(zip(values, offsets)):
            target_r_tensor, target_mask = self.ragged_index_select(
                drop_side, pad_side, drop_num, pad_num, v, o, indicator, topk_index
            )
            self.assertTrue(
                torch.equal(target_r_tensor.values(), gatherd_tensors[idx].values())
            )
            self.assertTrue(
                torch.equal(
                    target_r_tensor.offsets()[0], gatherd_tensors[idx].offsets()[0]
                )
            )
            self.assertTrue(torch.equal(target_mask, gatherd_masks[idx]))


if __name__ == "__main__":
    unittest.main()
