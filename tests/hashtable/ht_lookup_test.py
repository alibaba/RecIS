import os
import unittest

import torch

from recis.nn.hashtable_hook import AdmitHook
from recis.nn.modules.hashtable import HashTable, split_sparse_dense_state_dict
from recis.optim import SparseAdam


# Keep this in sync with cuco::detail::XXHash_32<int64_t>.  The regression
# below uses it only to construct deterministic collisions through the public
# Python HashTable API; the lookup itself runs the production GPU code.
_U32_MASK = 0xFFFFFFFF
_XXH32_PRIME2 = 0x85EBCA77
_XXH32_PRIME3 = 0xC2B2AE3D
_XXH32_PRIME4 = 0x27D4EB2F
_XXH32_PRIME5 = 0x165667B1


def _rotl32(value, bits):
    return ((value << bits) | (value >> (32 - bits))) & _U32_MASK


def _xxhash32_int64(key):
    """Return cuco's default XXH32 result for an int64 key."""
    key &= 0xFFFFFFFFFFFFFFFF
    value = (_XXH32_PRIME5 + 8) & _U32_MASK
    for word in (key & _U32_MASK, (key >> 32) & _U32_MASK):
        value = (value + word * _XXH32_PRIME3) & _U32_MASK
        value = (_rotl32(value, 17) * _XXH32_PRIME4) & _U32_MASK
    value ^= value >> 15
    value = (value * _XXH32_PRIME2) & _U32_MASK
    value ^= value >> 13
    value = (value * _XXH32_PRIME3) & _U32_MASK
    value ^= value >> 16
    return value & _U32_MASK


class GPUHashtableTest(unittest.TestCase):
    DEVICE = None

    @classmethod
    def setUpClass(cls):
        cls.DEVICE = os.getenv("TEST_DEVICE", "cuda")

    def setUp(self):
        self.ids_num = 2048
        self.emb_dim = 128
        self.block_size = 1024
        self.dtype = torch.float32

    def test_embedding_lookup(self):
        ht = HashTable(
            embedding_shape=[self.emb_dim],
            block_size=self.block_size,
            dtype=self.dtype,
            device=torch.device(self.DEVICE),
            name="gpu_ht",
        )

        # init hashtable
        ids = torch.arange(self.ids_num, device=self.DEVICE)
        emb_r = ht(ids)
        exp_r = torch.zeros_like(emb_r)
        self.assertTrue((exp_r.cuda() == emb_r).all())

        # insert datas
        ids_beg = self.ids_num
        ids = torch.arange(ids_beg, ids_beg + self.ids_num, device=self.DEVICE)
        emb = torch.tile(ids.reshape([-1, 1]), [1, self.emb_dim])
        ht._hashtable_impl.insert(ids, emb.type(self.dtype))
        emb_r = ht(ids)
        exp_r = emb
        self.assertTrue((exp_r.cuda() == emb_r).all())

        # missing key
        ids_beg += self.ids_num
        ids = torch.arange(
            ids_beg - self.ids_num, ids_beg + self.ids_num, device=self.DEVICE
        )
        emb_r = ht(ids)
        exp_r = torch.concat([emb, torch.zeros_like(emb)], 0)
        self.assertTrue((exp_r.cuda() == emb_r).all())

        ids_r, _ = torch.sort(ht.ids())
        ids_exp = torch.arange(ids_beg + self.ids_num, device=self.DEVICE)
        self.assertTrue((ids_r == ids_exp).all())

    def test_embedding_lookup_readonly(self):
        ht = HashTable(
            embedding_shape=[self.emb_dim],
            block_size=self.block_size,
            dtype=self.dtype,
            device=torch.device(self.DEVICE),
            name="gpu_ht_ro",
        )
        sparse_state, _ = split_sparse_dense_state_dict(ht.state_dict())
        opt = SparseAdam(sparse_state)

        ro_hook = AdmitHook("ReadOnly")
        # init hashtable
        ids = torch.arange(self.ids_num, device=self.DEVICE)
        ht(ids, admit_hook=ro_hook)

        # insert datas
        ids_beg = self.ids_num
        ids = torch.arange(ids_beg, ids_beg + self.ids_num, device=self.DEVICE)
        emb = torch.tile(ids.reshape([-1, 1]), [1, self.emb_dim])
        ht._hashtable_impl.insert(ids, emb.type(self.dtype))
        emb = ht(ids, admit_hook=ro_hook)
        emb.sum().backward()
        opt.step()
        opt.zero_grad()

        # missing key
        ids_beg += self.ids_num
        ids = torch.arange(
            ids_beg - self.ids_num, ids_beg + self.ids_num, device=self.DEVICE
        )
        emb = ht(ids, admit_hook=ro_hook)
        emb.sum().backward()
        opt.step()
        opt.zero_grad()

        ids_r, _ = torch.sort(ht.ids())
        ids_exp = torch.arange(self.ids_num, 2 * self.ids_num, device=self.DEVICE)
        self.assertTrue((ids_r == ids_exp).all())

    def test_embedding_lookup_eval(self):
        ht = HashTable(
            embedding_shape=[self.emb_dim],
            block_size=self.block_size,
            dtype=self.dtype,
            device=torch.device(self.DEVICE),
            name="gpu_ht_eval",
        )
        ht.eval()

        # init hashtable
        ids = torch.arange(self.ids_num, device=self.DEVICE)
        ht(ids)

        # insert datas
        ids_beg = self.ids_num
        ids = torch.arange(ids_beg, ids_beg + self.ids_num, device=self.DEVICE)
        emb = torch.tile(ids.reshape([-1, 1]), [1, self.emb_dim])
        ht._hashtable_impl.insert(ids, emb.type(self.dtype))
        ht(ids)

        # missing key
        ids_beg += self.ids_num
        ids = torch.arange(
            ids_beg - self.ids_num, ids_beg + self.ids_num, device=self.DEVICE
        )
        ht(ids)

        ids_r, _ = torch.sort(ht.ids())
        ids_exp = torch.arange(self.ids_num, 2 * self.ids_num, device=self.DEVICE)
        self.assertTrue((ids_r == ids_exp).all())

    def test_cg_lookup_after_scalar_rehash_with_uint32_max_hash(self):
        """Regression for CG lanes 1--3 wrapping after an XXH32 max value.

        The old CG initial-slot expression evaluated ``hash + lane`` as a
        uint32 before taking the capacity modulus.  For the production key
        below, XXH32 is 0xffffffff, so lanes 1--3 incorrectly probed slots
        0--2.  This test uses a scalar rehash to place the target behind a
        collision in the correct contiguous probe tile, then verifies that a
        read-only CG lookup still finds it.
        """
        old_capacity = 640_000
        rehash_capacity = 1_280_000
        target = 7_387_261_146_061_804
        filler = 244_033
        # These keys occupy old-map slots 0, 1 and 2. They force the old CG
        # insertion of target past its malformed first tile, while scalar
        # rehash moves them away from slots 0--2 in the new map.
        prefill = (460_431, 332_399, 287_184)

        target_hash = _xxhash32_int64(target)
        target_bucket = target_hash % rehash_capacity
        self.assertEqual(target_hash, _U32_MASK)
        self.assertEqual(_xxhash32_int64(filler) % rehash_capacity, target_bucket)
        self.assertEqual(
            tuple(_xxhash32_int64(key) % old_capacity for key in prefill),
            (0, 1, 2),
        )

        ht = HashTable(
            embedding_shape=[1],
            block_size=self.block_size,
            dtype=self.dtype,
            device=torch.device(self.DEVICE),
            name="gpu_ht_cg_probe_regression",
        )

        def insert(keys):
            ids = torch.tensor(keys, dtype=torch.int64, device=self.DEVICE)
            embeddings = torch.zeros(
                (len(keys), 1), dtype=self.dtype, device=self.DEVICE
            )
            ht._hashtable_impl.insert(ids, embeddings)

        # Insert one key at a time to control the old table's physical layout.
        bootstrap = (*prefill, filler, target)
        for key in bootstrap:
            insert([key])

        # cuco grows this 640k-slot map at 60% load. Keep the malformed
        # overflow probe slots and the target's rehashed tile clear so the
        # expected scalar/CG placement is deterministic.
        before_rehash_count = int(old_capacity * 0.60) - 1
        background_count = before_rehash_count - len(bootstrap)
        background = []
        candidate = 1_000_000

        def safe_background_bucket(bucket):
            return (
                64 <= bucket < rehash_capacity - 64
                and abs(bucket - target_bucket) >= 64
            )

        while len(background) < background_count:
            bucket = _xxhash32_int64(candidate) % rehash_capacity
            if safe_background_bucket(bucket):
                background.append(candidate)
            candidate += 1
        insert(background)
        self.assertEqual(ht.id_info(), (before_rehash_count, old_capacity))

        # One more insertion invokes cuco's scalar rehash. It places the
        # target in lane 1 of its correct tile; the next read-only lookup is
        # the production find_and_mask CG path under test.
        while not safe_background_bucket(
            _xxhash32_int64(candidate) % rehash_capacity
        ):
            candidate += 1
        insert([candidate])
        self.assertEqual(ht.id_info(), (before_rehash_count + 1, rehash_capacity))

        target_ids = torch.tensor([target], dtype=torch.int64, device=self.DEVICE)
        index, _ = ht._hashtable_impl.embedding_lookup(target_ids, True)
        self.assertEqual(int(index.item()), len(bootstrap) - 1)


if __name__ == "__main__":
    unittest.main()
