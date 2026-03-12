import random
import unittest

import torch

import recis  # noqa: F401 - required for side effects (e.g., operator registration)


class TestRaggedTile(unittest.TestCase):
    def test_ragged_tile_fwd(self):
        batch_size_one = 10
        batch_size_two = 100
        seq_len_one = 20
        seq_len_two = 160
        left_pad = torch.tensor([False, True], dtype=torch.bool).cuda()
        dim = 1
        seq_size = torch.concat(
            [
                torch.randint(0, seq_len_one, (batch_size_one,)),
                torch.randint(0, seq_len_two, (batch_size_two,)),
            ]
        )
        seq_size[random.randint(0, batch_size_one)] = 0
        offset = torch.zeros(seq_size.numel() + 1, dtype=seq_size.dtype)
        offset.narrow(0, 1, seq_size.numel()).copy_(torch.cumsum(seq_size, 0))
        ids = torch.randint(0, 50, (offset[-1].item(),))
        unique_ids, reverse_index = torch.unique(ids, return_inverse=True, sorted=True)
        pad_num_one = seq_len_one - seq_size[:batch_size_one]
        unique_emb = torch.tile(unique_ids.view(-1, 1), (1, dim)).to(torch.float32)
        emb, _ = torch.ops.recis.ragged_tile(
            [batch_size_one, batch_size_two],
            [seq_len_one, seq_len_two],
            reverse_index.cuda(),
            offset.cuda(),
            unique_emb.cuda(),
            left_pad,
        )
        fea_emb_one, fea_emb_two = torch.split(
            emb, [batch_size_one * seq_len_one, batch_size_two * seq_len_two]
        )
        fea_emb_one = fea_emb_one.reshape(batch_size_one, seq_len_one, dim)
        fea_emb_two = fea_emb_two.reshape(batch_size_two, seq_len_two, dim)
        print(fea_emb_one.shape, fea_emb_two.shape)
        offset_one = offset[: batch_size_one + 1]
        fea_emb_one_row_index = torch.repeat_interleave(
            torch.arange(batch_size_one), offset_one[1:] - offset_one[:-1]
        )
        fea_emb_one_col_index = torch.arange(offset_one[-1]) - torch.repeat_interleave(
            offset_one[:-1], offset_one[1:] - offset_one[:-1]
        )
        fea_emb_one_col_index = (
            fea_emb_one_col_index
            + torch.repeat_interleave(pad_num_one, offset_one[1:] - offset_one[:-1])
            * left_pad[0].cpu()
        )
        fea_emb_one_fake = torch.zeros((batch_size_one, seq_len_one, dim))
        fea_emb_one_fake[fea_emb_one_row_index, fea_emb_one_col_index, :] = unique_emb[
            reverse_index
        ][: offset_one[batch_size_one]]
        self.assertEqual((fea_emb_one - fea_emb_one_fake.cuda()).abs().sum().item(), 0)

    def test_ragged_tile_bwd(self):
        batch_size_one = 10
        batch_size_two = 100
        seq_len_one = 16
        seq_len_two = 32
        dim = 128
        left_pad = torch.tensor([False, True], dtype=torch.bool).cuda()
        seq_size = torch.concat(
            [
                torch.randint(1, seq_len_one, (batch_size_one,)),
                torch.randint(0, seq_len_two, (batch_size_two,)),
            ]
        )
        offset = torch.zeros(seq_size.numel() + 1, dtype=seq_size.dtype)
        offset.narrow(0, 1, seq_size.numel()).copy_(torch.cumsum(seq_size, 0))
        ids = torch.randint(1, 50, (offset[-1].item(),))
        unique_ids, reverse_index = torch.unique(ids, return_inverse=True, sorted=True)
        pad_num_one = seq_len_one - seq_size[:batch_size_one]
        pad_num_two = seq_len_two - seq_size[batch_size_one:]
        unique_emb = torch.tile(unique_ids.view(-1, 1), (1, dim)).to(torch.float32)
        unique_emb[0, :] = 1
        emb, batch_tile_len = torch.ops.recis.ragged_tile(
            [batch_size_one, batch_size_two],
            [seq_len_one, seq_len_two],
            reverse_index.cuda(),
            offset.cuda(),
            unique_emb.cuda(),
            left_pad,
        )
        batch_info = (
            unique_ids.numel(),
            2,
            max(batch_size_one, batch_size_two),
            min(seq_len_one, seq_len_two),
        )
        unique_grad = torch.ops.recis.ragged_tile_back(
            batch_tile_len,
            batch_info,
            reverse_index.cuda(),
            offset.cuda(),
            emb,
            left_pad,
        )
        emb_one, emb_two = emb.split(
            [batch_size_one * seq_len_one, batch_size_two * seq_len_two]
        )
        emb_one = emb_one.view(batch_size_one, seq_len_one, dim)
        emb_two = emb_two.view(batch_size_two, seq_len_two, dim)
        offset_one = offset[: batch_size_one + 1]
        offset_two = offset[batch_size_one:] - offset[batch_size_one]
        fea_one_row_index = torch.repeat_interleave(
            torch.arange(offset_one.numel() - 1), offset_one[1:] - offset_one[:-1]
        )
        fea_one_col_index = torch.arange(offset_one[-1]) - torch.repeat_interleave(
            offset_one[:-1], offset_one[1:] - offset_one[:-1]
        )
        fea_one_col_index = fea_one_col_index + torch.repeat_interleave(
            pad_num_one * left_pad[0].cpu(), offset_one[1:] - offset_one[:-1]
        )
        emb_one_flat = emb_one[fea_one_row_index, fea_one_col_index]
        assert emb_one_flat.size(0) == offset_one[-1].item()
        fea_two_row_index = torch.repeat_interleave(
            torch.arange(offset_two.numel() - 1), offset_two[1:] - offset_two[:-1]
        )
        fea_two_col_index = torch.arange(offset_two[-1]) - torch.repeat_interleave(
            offset_two[:-1], offset_two[1:] - offset_two[:-1]
        )
        fea_two_col_index = fea_two_col_index + torch.repeat_interleave(
            pad_num_two * left_pad[1].cpu(), offset_two[1:] - offset_two[:-1]
        )
        emb_two_flat = emb_two[fea_two_row_index, fea_two_col_index]
        assert emb_two_flat.size(0) == offset_two[-1].item()
        emb_flat = torch.concat([emb_one_flat, emb_two_flat])
        assert emb_flat.size(0) == ids.numel()
        fea_one_index_map = (
            torch.ones([batch_size_one, seq_len_one], dtype=torch.int64) * -1
        )
        fea_one_index_map[fea_one_row_index, fea_one_col_index] = reverse_index[
            : offset_one[-1]
        ]
        fea_two_index_map = (
            torch.ones([batch_size_two, seq_len_two], dtype=torch.int64) * -1
        )
        fea_two_index_map[fea_two_row_index, fea_two_col_index] = reverse_index[
            offset_one[-1] :
        ]
        unique_grad_target = torch.zeros(unique_ids.numel(), dim).cuda()
        unique_grad_target = torch.index_add(
            unique_grad_target, 0, reverse_index.cuda(), emb_flat.cuda()
        )
        self.assertEqual((unique_grad - unique_grad_target).abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
