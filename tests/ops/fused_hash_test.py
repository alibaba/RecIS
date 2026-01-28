import unittest

import torch

from recis.nn.functional import hash_ops


# import mmh3


def murmurhash64_py(data_bytes: bytes, seed: int = 0) -> int:
    m = 0xC6A4A7935BD1E995
    r = 47
    n = len(data_bytes)
    h = seed ^ (n * m)

    i = 0
    while n >= 8:
        k = int.from_bytes(data_bytes[i : i + 8], "little", signed=False)
        i += 8
        n -= 8

        k = (k * m) & 0xFFFFFFFFFFFFFFFF
        k ^= k >> r
        k = (k * m) & 0xFFFFFFFFFFFFFFFF

        h ^= k
        h = (h * m) & 0xFFFFFFFFFFFFFFFF

    if n > 0:
        tail_bytes = data_bytes[i:]
        if n >= 7:
            h ^= tail_bytes[6] << 48
        if n >= 6:
            h ^= tail_bytes[5] << 40
        if n >= 5:
            h ^= tail_bytes[4] << 32
        if n >= 4:
            h ^= tail_bytes[3] << 24
        if n >= 3:
            h ^= tail_bytes[2] << 16
        if n >= 2:
            h ^= tail_bytes[1] << 8
        if n >= 1:
            h ^= tail_bytes[0]
            h = (h * m) & 0xFFFFFFFFFFFFFFFF

    h ^= h >> r
    h = (h * m) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> r

    if h >= (1 << 63):
        return h - (1 << 64)
    else:
        return h


def djb2(data: bytes):
    hash_val = 5381
    mask = 0xFFFFFFFFFFFFFFFF
    for byte in data:
        hash_val = ((hash_val << 5) + hash_val) + byte
        hash_val = hash_val & mask
    return hash_val


def sdbm(s: bytes) -> int:
    byte_str = s
    # byte_str = s.encode('utf-8')
    hash_val = 0
    mask = 0xFFFFFFFFFFFFFFFF
    for byte in byte_str:
        hash_val = byte + (hash_val << 6) + (hash_val << 16) - hash_val
        hash_val = hash_val & mask
    return hash_val


def murmurhash(func, input, offset):
    inp_bytes = input.cpu().numpy().tobytes()
    offs_list = offset.cpu().numpy().tolist()
    num_segments = len(offs_list) - 1
    segment_hashes = []
    for i in range(num_segments):
        start = offs_list[i]
        end = offs_list[i + 1]
        segment_bytes = inp_bytes[start:end]
        hash_val = func(segment_bytes)
        segment_hashes.append(hash_val)
    return torch.LongTensor(segment_hashes)


def fused_murmurhash_py(func, inputs, offsets):
    outputs = []
    for inp, offs in zip(inputs, offsets):
        rt = murmurhash(func, inp, offs)
        outputs.append(rt)
    return outputs


class FusedHashTest(unittest.TestCase):
    def test_fused_farmhash(self):
        inputs = [
            torch.tensor([0, 1, 2, 1, 2, 3], dtype=torch.int8).cuda(),
            torch.tensor([0, 5, 5, 5, 6, 7], dtype=torch.int8).cuda(),
        ]
        offsets = [
            torch.tensor([0, 3, 6], dtype=torch.int32).cuda(),
            torch.tensor([0, 3, 6], dtype=torch.int32).cuda(),
        ]
        outputs = hash_ops.farmhash(inputs, offsets)
        res = [
            torch.LongTensor([-7736835464683084646, -5546720800891377920]),
            torch.LongTensor([-4442022684725028445, -909183002746659612]),
        ]
        for output, ans in zip(outputs, res):
            self.assertTrue(torch.equal(output.cpu(), ans))

    def test_fused_murmurhash(self):
        inputs = [
            torch.tensor([0, 1, 2, 1, 2, 3], dtype=torch.int8).cuda(),
            torch.tensor([0, 5, 5, 5, 6, 7], dtype=torch.int8).cuda(),
        ]
        offsets = [
            torch.tensor([0, 3, 6], dtype=torch.int32).cuda(),
            torch.tensor([0, 3, 6], dtype=torch.int32).cuda(),
        ]
        outputs = hash_ops.murmurhash(inputs, offsets)
        res = [
            torch.LongTensor([-4003751876240412087, -1042754718167355375]),
            torch.LongTensor([-5541886947548579749, 4434088491466393040]),
        ]
        for output, ans in zip(outputs, res):
            self.assertTrue(torch.equal(output.cpu(), ans))

    def test_fused_djb2hash(self):
        inputs = [
            torch.tensor([0, 1, 2, 1, 2, 3], dtype=torch.int8).cuda(),
            torch.tensor([0, 5, 5, 5, 6, 7], dtype=torch.int8).cuda(),
        ]
        offsets = [
            torch.tensor([0, 3, 6], dtype=torch.int32).cuda(),
            torch.tensor([0, 3, 6], dtype=torch.int32).cuda(),
        ]
        y = hash_ops.djb2hash(inputs, offsets)
        ref = fused_murmurhash_py(djb2, inputs, offsets)

        for output, ans in zip(y, ref):
            self.assertTrue(torch.equal(output.cpu(), ans))

    def test_fused_sdbmhash(self):
        inputs = [
            torch.tensor([0, 1, 2, 1, 2, 3], dtype=torch.int8).cuda(),
            torch.tensor([0, 5, 5, 5, 6, 7], dtype=torch.int8).cuda(),
        ]
        offsets = [
            torch.tensor([0, 3, 6], dtype=torch.int32).cuda(),
            torch.tensor([0, 3, 6], dtype=torch.int32).cuda(),
        ]
        y = hash_ops.sdbmhash(inputs, offsets)
        ref = fused_murmurhash_py(sdbm, inputs, offsets)
        for output, ans in zip(y, ref):
            self.assertTrue(torch.equal(output.cpu(), ans))


if __name__ == "__main__":
    unittest.main()
