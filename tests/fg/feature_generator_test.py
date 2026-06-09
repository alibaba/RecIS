import json
import os
import unittest
from dataclasses import asdict

from recis.fg.feature_generator import FG
from recis.fg.fg_parser import FGParser
from recis.fg.mc_parser import MCParser
from recis.fg.shape_manager import ShapeManager


class TestFG(unittest.TestCase):
    def setUp(self):
        mc_file_name = os.path.join(os.path.split(__file__)[0], "./mc.json")
        self.mc_parser = MCParser(mc_config_path=mc_file_name, lower_case=True)
        with open(mc_file_name) as f:
            mc_conf = json.load(f)
        mc_conf["raw_longpay_seq_item_id_d"] = ["longpay_seq_item_id_d_raw"]
        self.mc_parser_mask = MCParser(mc_config=mc_conf, lower_case=True)
        fg_file_name = os.path.join(os.path.split(__file__)[0], "./fg.json")
        self.fg_parser_no_io_hash = FGParser(
            fg_file_name, self.mc_parser_mask, lower_case=True
        )
        self.shape_manager_no_io_hash = ShapeManager(self.fg_parser_no_io_hash)

        self.fg_parser_io_hash = FGParser(
            fg_file_name,
            self.mc_parser,
            lower_case=True,
            hash_in_io=True,
        )
        self.shape_manager_io_hash = ShapeManager(self.fg_parser_io_hash)

        self.fg_parser_already_hash = FGParser(
            fg_file_name,
            self.mc_parser,
            lower_case=True,
            hash_in_io=False,
            already_hashed=True,
        )
        self.shape_manager_already_hash = ShapeManager(self.fg_parser_already_hash)

    def test_feature_blocks(self):
        # same as mc parser
        fg = FG(self.fg_parser_no_io_hash, self.shape_manager_no_io_hash)
        self.assertTrue(fg.feature_blocks == self.mc_parser_mask.feature_blocks)

    def test_block_seq_len(self):
        # like fg parser
        fg = FG(self.fg_parser_no_io_hash, self.shape_manager_no_io_hash)
        self.assertTrue(fg.get_block_seq_len("raw_longpay_seq_item_id_d") == 200)
        self.assertTrue(fg.get_block_seq_len("longpay_seq") == 200)
        self.assertTrue(fg.get_block_seq_len("longpay_seq__context") == 200)

    def test_get_shape(self):
        # same as shape manager
        fg = FG(self.fg_parser_no_io_hash, self.shape_manager_no_io_hash)
        self.assertTrue(fg.get_shape("s_nid_pv30_c2c") == [-1, 8])
        self.assertTrue(fg.get_shape("longpay_seq") == [-1, 200, 72])
        self.assertTrue(fg.get_shape("longpay_seq_length") == [-1, 1])

    def test_label_ids(self):
        fg = FG(self.fg_parser_no_io_hash, self.shape_manager_no_io_hash)
        fg.add_label("click")
        self.assertTrue(fg.labels == ["click"])
        fg.add_id("id")
        self.assertTrue(fg.sample_ids == ["id"])

    def test_get_emb_confs(self):
        expect_emb_conf_file = os.path.join(
            os.path.split(__file__)[0], "./fg_conf/emb_conf.json"
        )
        with open(expect_emb_conf_file) as f:
            expect_emb_conf = json.load(f)
        fg = FG(self.fg_parser_no_io_hash, self.shape_manager_no_io_hash)
        emb_confs = fg.get_emb_confs()
        self.assertTrue(len(emb_confs.values()) == len(expect_emb_conf.values()))
        for emb_name, emb_conf in emb_confs.items():
            self.assertTrue(emb_name in expect_emb_conf)
            emb_conf = asdict(emb_conf)
            emb_conf.pop("initializer")
            emb_conf["device"] = str(emb_conf["device"].type)
            emb_conf["dtype"] = str(emb_conf["dtype"])
            self.assertTrue(emb_conf == expect_emb_conf[emb_name])
            if not emb_conf == expect_emb_conf[emb_name]:
                print(emb_conf)
                print(expect_emb_conf[emb_name])
                raise RuntimeError("!!!")

    def test_get_feature_confs_no_io_hash(self):
        expect_fea_conf_file = os.path.join(
            os.path.split(__file__)[0], "./fg_conf/feature_conf_no_io_hash.json"
        )
        with open(expect_fea_conf_file) as f:
            expect_fea_conf = json.load(f)
        fg = FG(self.fg_parser_no_io_hash, self.shape_manager_no_io_hash)
        fea_confs = fg.get_feature_confs()
        self.assertTrue(len(fea_confs) == len(expect_fea_conf.values()))
        for fea_conf in fea_confs:
            self.assertTrue(fea_conf.name in expect_fea_conf)
            self.assertTrue(len(fea_conf.ops) == len(expect_fea_conf[fea_conf.name]))
            for i, op in enumerate(fea_conf.ops):
                op_conf = op._get_config()
                # selectfield dtype not used
                if i == 0:
                    op_conf.pop("dtype")
                if "boundary" in op_conf or "multi_muls" in op_conf:
                    # float convert to tensor has little diff
                    continue
                self.assertTrue(op_conf == expect_fea_conf[fea_conf.name][i])

    def test_get_feature_confs_io_hash(self):
        expect_fea_conf_file = os.path.join(
            os.path.split(__file__)[0], "./fg_conf/feature_conf_io_hash.json"
        )
        with open(expect_fea_conf_file) as f:
            expect_fea_conf = json.load(f)
        fg = FG(self.fg_parser_io_hash, self.shape_manager_io_hash)
        fea_confs = fg.get_feature_confs()
        self.assertTrue(len(fea_confs) == len(expect_fea_conf.values()))
        for fea_conf in fea_confs:
            self.assertTrue(fea_conf.name in expect_fea_conf)
            self.assertTrue(len(fea_conf.ops) == len(expect_fea_conf[fea_conf.name]))
            for i, op in enumerate(fea_conf.ops):
                op_conf = op._get_config()
                # selectfield dtype not used
                if i == 0:
                    op_conf.pop("dtype")
                if "boundary" in op_conf or "multi_muls" in op_conf:
                    # float convert to tensor has little diff
                    continue
                self.assertTrue(op_conf == expect_fea_conf[fea_conf.name][i])

    def test_get_feature_confs_already_hash(self):
        expect_fea_conf_file = os.path.join(
            os.path.split(__file__)[0], "./fg_conf/feature_conf_already_hash.json"
        )
        with open(expect_fea_conf_file) as f:
            expect_fea_conf = json.load(f)
        fg = FG(self.fg_parser_already_hash, self.shape_manager_already_hash)
        fea_confs = fg.get_feature_confs()
        self.assertTrue(len(fea_confs) == len(expect_fea_conf.values()))
        for fea_conf in fea_confs:
            self.assertTrue(fea_conf.name in expect_fea_conf)
            self.assertTrue(len(fea_conf.ops) == len(expect_fea_conf[fea_conf.name]))
            for i, op in enumerate(fea_conf.ops):
                op_conf = op._get_config()
                # selectfield dtype not used
                if i == 0:
                    op_conf.pop("dtype")
                if "boundary" in op_conf or "multi_muls" in op_conf:
                    # float convert to tensor has little diff
                    continue
                self.assertTrue(op_conf == expect_fea_conf[fea_conf.name][i])


if __name__ == "__main__":
    unittest.main()
