import json
import os
import shutil
import tempfile
import unittest

import torch
from safetensors import safe_open

from recis.framework.checkpoint_manager import Saver as CheckpointSaver, SaverOptions
from recis.framework.filesystem import get_file_system
from recis.nn.modules.hashtable import HashTable, gen_slice
from recis.optim.named_optimizer import wrapped_named_optimizer
from recis.serialize import Saver
from recis.serialize.loader import Loader


def get_index(name: str):
    return int(name.split("_")[1])


def merge_ids_and_emb(state_dict: dict):
    ids_list = []
    emb_list = []
    for key, value in state_dict.items():
        if "id" in key:
            if len(ids_list) < get_index(key) + 1:
                ids_list.extend([None] * ((get_index(key) + 1) - len(ids_list)))
            ids_list[get_index(key)] = value
        if "embedding" in key:
            if len(emb_list) < get_index(key) + 1:
                emb_list.extend([None] * ((get_index(key) + 1) - len(emb_list)))
            emb_list[get_index(key)] = value
    return torch.concat(ids_list), torch.concat(emb_list)


class DenseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(10, 10)
        self.dense2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.dense1(x) + self.dense2(x)


class TestCase(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self._path = {}
        self._tensor_map_json = "tensorkey.json"
        self._rank_json = "torch_rank_weights_embs_table_multi_shard.json"
        self._index_file = "index"
        self._parallels = [1, 2, 4, 8]
        self._hash_tables = {}

    def _create_tensor(self, parallel, clear=False):
        tensor_for_save = {
            "one": torch.randn([2, 2]),
            "two": torch.randn([3, 3]),
            "three": torch.randn([100, 1]),
            "four": torch.randn([1, 10]),
            "five": torch.randn([1, 1]),
            "six": torch.randn([1, 1]),
        }

        block_size = 1 << 20
        ids_one = torch.arange(block_size)
        emb_one = torch.tile(ids_one.reshape([-1, 1]), [1, 8]).type(torch.float32)
        emb_one = torch.randn(emb_one.size())
        ids_two = torch.arange(block_size) + block_size
        emb_two = torch.tile(ids_two.reshape([-1, 1]), [1, 8]).type(torch.float32)
        emb_two = torch.randn(emb_two.size())
        ids_three = torch.arange(block_size // 2) + 2 * block_size
        emb_three = torch.tile(ids_three.reshape([-1, 1]), [1, 8]).type(torch.float32)
        emb_three = torch.randn(emb_three.size())
        ids = torch.concat([ids_one, ids_two, ids_three])
        emb = torch.concat([emb_one, emb_two, emb_three])

        hash_table = self._hash_tables[parallel]
        if clear:
            print("=" * 50, "clearing")
            hash_table.clear()

        hash_table.insert(ids, emb)

        return {
            "dense": tensor_for_save,
            "sparse": hash_table.state_dict(),
            "ids": ids,
            "emb": emb,
        }

    def _check_json(self, num_dense: int, num_sparse: int, parallel: int):
        with open(os.path.join(self._path[parallel], self._tensor_map_json)) as f:
            tensor_name_data = json.load(f)
        with open(os.path.join(self._path[parallel], self._rank_json)) as f:
            rank_data = json.load(f)

        if len(tensor_name_data) != num_sparse:
            print(self._tensor_map_json, f" keys not equal to {num_sparse}")
            return False

        if len(rank_data) != num_sparse // 2 + num_dense:
            # id is not saved in the meta file
            print(self._rank_json, f" keys not equal to {num_sparse // 2 + num_dense}")
            return False

        for k in tensor_name_data.keys():
            if ".embedding" not in k:
                continue
            if k not in rank_data.keys():
                return False

        for key, val in rank_data.items():
            if "embedding" in key:
                for subkey in [
                    "dimension",
                    "dtype",
                    "name",
                    "dense",
                    "dimension",
                    "dtype",
                    "hashmap_key",
                    "hashmap_key_dtype",
                    "hashmap_value",
                    "is_hashmap",
                ]:
                    if subkey not in val:
                        return False
                if not val["is_hashmap"]:
                    return False
                if "id" not in val["hashmap_key"]:
                    return False
            else:
                if val["dense"]:
                    for subkey in ["name", "dense", "dimension", "is_hashmap", "dtype"]:
                        if subkey not in val:
                            return False
                    if val["dimension"] != 0:
                        return False
                    if val["is_hashmap"]:
                        return False

        with open(os.path.join(self._path[parallel], self._index_file)) as f:
            data = json.load(f)
            index_data, file_data = data["block_index"], data["file_index"]
            if len(index_data) != len(rank_data) + num_sparse // 2:
                return False

            for val in index_data.values():
                if val < 0 or val >= len(file_data):
                    return False

        return True

    def _create_dir(self, parallel):
        tmp_file = tempfile.mkdtemp()
        self._path[parallel] = tmp_file

    def save_value_ok(self, parallel):
        self._create_dir(parallel=parallel)
        data = self._create_tensor(parallel)
        tensor_for_save, hashtable_for_save, ids, emb = (
            data["dense"],
            data["sparse"],
            data["ids"],
            data["emb"],
        )

        saver = Saver(
            0,
            1,
            parallel,
            self._path[parallel],
            tensors=tensor_for_save,
            hashtables=hashtable_for_save,
        )
        print("=" * 50, "saving")
        saver.save()

        num_dense = len(tensor_for_save)
        num_sparse = len(hashtable_for_save) * 2

        self.assertTrue(self._check_json(num_dense, num_sparse, parallel))

        with open(os.path.join(self._path[parallel], self._index_file)) as f:
            ckpt_files = json.load(f)["file_index"].keys()

        loaded_ids_embs = {}
        for ckpt_file in ckpt_files:
            with safe_open(os.path.join(self._path[parallel], ckpt_file), "pt") as f:
                keys = list(f.keys())
                for origin_key in ["one", "two", "three", "four", "five", "six"]:
                    for key in keys:
                        if origin_key in key:
                            self.assertTrue(
                                torch.allclose(
                                    tensor_for_save[origin_key], f.get_tensor(key)
                                )
                            )
                        else:
                            loaded_ids_embs[key] = f.get_tensor(key)
        load_ids, load_embs = merge_ids_and_emb(loaded_ids_embs)
        index = torch.argsort(load_ids)
        load_ids = load_ids[index]
        load_embs = load_embs[index]
        self.assertTrue(torch.allclose(ids, load_ids))
        self.assertTrue(torch.allclose(emb, load_embs))

    def _read_safetensor_file(self, path):
        with open(os.path.join(path, self._index_file)) as f:
            ckpt_files = json.load(f)["file_index"].keys()
        data = {}
        for ckpt_file in ckpt_files:
            with safe_open(os.path.join(path, ckpt_file), "pt") as f:
                keys = list(f.keys())
                for key in keys:
                    data[key] = {}
                    data[key]["tensor"] = f.get_tensor(key)
        return data

    def load_value_ok(self, parallel):
        data = self._create_tensor(parallel, clear=True)
        print("=" * 50, "loading")
        tensor_for_save, hashtable_for_save, _, _ = (
            data["dense"],
            data["sparse"],
            data["ids"],
            data["emb"],
        )
        loader = Loader(
            self._path[parallel],
            hashtables=hashtable_for_save,
            tensors=tensor_for_save,
            parallel=parallel,
        )
        loader.load()

        tmp_path = tempfile.mkdtemp()
        saver = Saver(
            0,
            1,
            parallel,
            tmp_path,
            tensors=tensor_for_save,
            hashtables=hashtable_for_save,
        )
        saver.save()

        data1 = self._read_safetensor_file(self._path[parallel])
        data2 = self._read_safetensor_file(tmp_path)

        for key, val in data1.items():
            self.assertTrue(key in data2)
            self.assertTrue(torch.allclose(val["tensor"], data2[key]["tensor"]))

    def _ckpt_read(self, parallel):
        data_cmp = self._read_safetensor_file(self._path[parallel])

        CheckpointReaderImpl = torch.classes.recis.CheckpointReader
        reader = CheckpointReaderImpl(self._path[parallel])
        reader.init()
        names = reader.list_tensor_names()

        data = {}
        for name in names:
            data[name] = {}
            tensor = reader.read_tensor(name)
            data[name]["tensor"] = tensor
            data[name]["type"] = reader.tensor_dtype(name).dtype
            data[name]["shape"] = reader.tensor_shape(name)

        for key, val in data.items():
            new_key = key
            if "@" in key:
                new_key = key.replace("@", ".")

            com_key = ""
            for _com_key in data_cmp.keys():
                if new_key in _com_key:
                    com_key = _com_key

            self.assertTrue(len(com_key) > 0)
            self.assertTrue(torch.allclose(val["tensor"], data_cmp[com_key]["tensor"]))

    def _run_dense_model(self, model):
        optimizer = wrapped_named_optimizer(torch.optim.Adam)(
            model.named_parameters(), lr=0.001
        )
        for _ in range(10):
            x = torch.randn(10, 10, device="cuda")
            y = model(x)
            loss = torch.sum(y)
            loss.backward()
            optimizer.step()

    def test_save_load_without_sparse(self):
        model = DenseModel()
        model = model.to("cuda")
        self._run_dense_model(model)
        tmp_path = tempfile.mkdtemp()

        saver_option = SaverOptions(
            model,
            None,
            tmp_path,
            None,
            20,
            1,
            None,
        )
        saver = CheckpointSaver(saver_option)
        saver.save(ckpt_id="ckpt_1")

        ckpt_1 = os.path.join(tmp_path, "ckpt_1")

        # test ckpt reader
        CheckpointReaderImpl = torch.classes.recis.CheckpointReader
        reader = CheckpointReaderImpl(ckpt_1)
        reader.init()
        names = reader.list_tensor_names()
        self.assertEqual(len(names), 0)

        # test loader
        saver = CheckpointSaver(saver_option)
        saver.restore()

        fs = get_file_system(ckpt_1)
        pt_file = os.path.join(ckpt_1, "model.pt")
        with fs.open(pt_file, "rb") as f:
            data1 = torch.load(f=f)

        for key, val in data1.items():
            self.assertTrue(key in model.state_dict())
            self.assertTrue(torch.allclose(val, model.state_dict()[key]))

        # load and run again
        saver.output_dir = tmp_path
        self._run_dense_model(model)

        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)

    def test_save_ok(self):
        def _create_hashtable(parallel, shard_idx=0, shard_num=1):
            return HashTable(
                [8],
                1024,
                dtype=torch.float32,
                name=f"hashtable{parallel}",
                slice=gen_slice(shard_idx, shard_num),
            )

        for para in [1, 2, 4, 8]:
            self._hash_tables[para] = _create_hashtable(parallel=para)
            self.save_value_ok(para)
            self.load_value_ok(para)


if __name__ == "__main__":
    # run
    unittest.main(verbosity=0)
