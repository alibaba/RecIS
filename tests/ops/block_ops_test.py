import unittest

import torch


class TestGather(unittest.TestCase):
    def setUp(self):
        self.block_size = 10240
        self.block_dim = 8
        self.block_num = 10
        self.gather_num = 50000
        self.blocks = [
            torch.empty((self.block_size, self.block_dim), device="cuda").uniform_()
            for _ in range(self.block_num)
        ]
        self.block = torch.empty(
            (self.block_size * self.block_num, self.block_dim), device="cuda"
        ).uniform_()
        self.index = torch.arange(self.gather_num, device="cuda")

    def test_block_gather(self):
        ret = torch.ops.recis.block_gather(
            self.index, self.blocks, self.block_size, -1, False
        )
        ans = []
        last_num = self.gather_num
        cur_index = 0
        while last_num > 0:
            real_num = min(last_num, self.block_size)
            ans.append(self.blocks[cur_index][0:real_num])
            last_num -= real_num
            cur_index += 1
        ans = torch.cat(ans)
        self.assertTrue(torch.equal(ret, ans))

    def test_gather(self):
        ret = torch.ops.recis.gather(self.index, self.block)
        ans = self.block[self.index]
        self.assertTrue(torch.equal(ret, ans))

    def test_int8_small_dimension_gather(self):
        block_size = 4
        ids = torch.tensor([0, 3, 4, 7], device="cuda")
        for embedding_dim in (1, 2):
            blocks = [
                torch.randint(
                    -128,
                    128,
                    (block_size, embedding_dim),
                    dtype=torch.int8,
                    device="cuda",
                )
                for _ in range(2)
            ]
            block = torch.cat(blocks)

            gathered = torch.ops.recis.gather(ids, block)
            block_gathered = torch.ops.recis.block_gather(
                ids, blocks, block_size, -1, False
            )

            torch.testing.assert_close(gathered, block[ids])
            torch.testing.assert_close(block_gathered, block[ids])


class TestInsert(unittest.TestCase):
    def setUp(self):
        self.block_size = 10240
        self.block_dim = 8
        self.block_num = 10
        self.insert_num = 50000
        self.blocks = [
            torch.empty((self.block_size, self.block_dim), device="cuda").uniform_()
            for _ in range(self.block_num)
        ]
        self.index = torch.arange(self.insert_num, device="cuda")

    def test_block_insert(self):
        emb = (
            torch.empty((self.insert_num, self.block_dim), device="cuda").uniform_()
            * 1000
        )
        torch.ops.recis.block_insert(self.index, emb, self.blocks, self.block_size)
        last_num = 0
        cur_index = 0
        while last_num < self.insert_num:
            real_num = min(self.insert_num - last_num, self.block_size)
            self.assertTrue(
                torch.equal(
                    self.blocks[cur_index][0:real_num],
                    emb[last_num : last_num + real_num],
                )
            )
            last_num += real_num
            cur_index += 1

    def test_int8_small_dimension_block_insert(self):
        block_size = 4
        ids = torch.tensor([0, 3, 4, 7], device="cuda")
        for embedding_dim in (1, 2):
            blocks = [
                torch.zeros(
                    (block_size, embedding_dim),
                    dtype=torch.int8,
                    device="cuda",
                )
                for _ in range(2)
            ]
            embedding = torch.randint(
                -128,
                128,
                (ids.numel(), embedding_dim),
                dtype=torch.int8,
                device="cuda",
            )

            torch.ops.recis.block_insert(ids, embedding, blocks, block_size)

            inserted = torch.cat(blocks)[ids]
            torch.testing.assert_close(inserted, embedding)


class TestApplyAdamw(unittest.TestCase):
    def setUp(self):
        self.block_size = 10240
        self.block_dim = 8
        self.block_num = 10
        self.num = 10000
        self.blocks = [
            torch.empty((self.block_size, self.block_dim), device="cuda").uniform_()
            for _ in range(self.block_num)
        ]
        self.cpu_blocks = [self.blocks[i].clone().cpu() for i in range(self.block_num)]
        self.exp_avgs = [
            torch.empty((self.block_size, self.block_dim), device="cuda").uniform_()
            for _ in range(self.block_num)
        ]
        self.cpu_exp_avgs = [
            self.exp_avgs[i].clone().cpu() for i in range(self.block_num)
        ]
        self.exp_avg_sqs = [
            torch.empty((self.block_size, self.block_dim), device="cuda").uniform_()
            for _ in range(self.block_num)
        ]
        self.cpu_exp_avg_sqs = [
            self.exp_avg_sqs[i].clone().cpu() for i in range(self.block_num)
        ]
        self.index = torch.arange(self.num, device="cuda")
        self.beta1 = torch.tensor([0.99], device="cuda")
        self.cpu_beta1 = torch.tensor([0.99], device="cpu")
        self.beta2 = torch.tensor([0.9], device="cuda")
        self.cpu_beta2 = torch.tensor([0.9], device="cpu")
        self.step = torch.tensor([10], device="cuda")
        self.cpu_step = torch.tensor([10], device="cpu")
        self.lr = 0.001
        self.b1 = 0.99
        self.b2 = 0.9
        self.weight_decay = 0.1
        self.eps = 1e-5

    def test_block_apply_adamw(self):
        grad = torch.ones((self.num, self.block_dim), device="cuda")
        torch.ops.recis.block_apply_adamw(
            self.index,
            grad,
            self.blocks,
            self.beta1,
            self.beta2,
            self.step,
            self.exp_avgs,
            self.exp_avg_sqs,
            self.lr,
            self.b1,
            self.b2,
            self.weight_decay,
            self.eps,
            self.block_size,
        )
        cpu_grad = torch.ones((self.num, self.block_dim), device="cpu")
        torch.ops.recis.block_apply_adamw(
            self.index.cpu(),
            cpu_grad,
            self.cpu_blocks,
            self.cpu_beta1,
            self.cpu_beta2,
            self.cpu_step,
            self.cpu_exp_avgs,
            self.cpu_exp_avg_sqs,
            self.lr,
            self.b1,
            self.b2,
            self.weight_decay,
            self.eps,
            self.block_size,
        )
        for i in range(self.block_num):
            torch.testing.assert_close(self.blocks[i], self.cpu_blocks[i].cuda())
            torch.testing.assert_close(self.exp_avgs[i], self.cpu_exp_avgs[i].cuda())
            torch.testing.assert_close(
                self.exp_avg_sqs[i], self.cpu_exp_avg_sqs[i].cuda()
            )


if __name__ == "__main__":
    unittest.main()
