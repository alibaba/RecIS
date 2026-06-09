import json
import os
import unittest
from dataclasses import asdict

from recis.fg.fg_parser import FGParser
from recis.fg.mc_parser import MCParser


class TestFGParser(unittest.TestCase):
    def setUp(self):
        mc_file_name = os.path.join(os.path.split(__file__)[0], "./mc.json")
        self.mc_parser = MCParser(mc_config_path=mc_file_name, lower_case=True)
        with open(mc_file_name) as f:
            mc_conf = json.load(f)
        mc_conf["raw_longpay_seq_item_id_d"] = ["longpay_seq_item_id_d_raw"]
        self.mc_parser_mask = MCParser(mc_config=mc_conf, lower_case=True)
        self.fg_file_name = os.path.join(os.path.split(__file__)[0], "./fg.json")
        with open(os.path.join(os.path.split(__file__)[0], "./fg_conf.json")) as f:
            self.parsed_fg_conf = json.load(f)
        with open(
            os.path.join(
                os.path.split(__file__)[0],
                "./fg_parser_conf/fg_io_conf_no_io_hash.json",
            ),
        ) as f:
            self.fg_io_conf_no_io_hash = json.load(f)
        with open(
            os.path.join(
                os.path.split(__file__)[0], "./fg_parser_conf/fg_io_conf_io_hash.json"
            ),
        ) as f:
            self.fg_io_conf_io_hash = json.load(f)
        with open(
            os.path.join(
                os.path.split(__file__)[0],
                "./fg_parser_conf/fg_io_conf_already_hash.json",
            ),
        ) as f:
            self.fg_io_conf_already_hash = json.load(f)
        with open(
            os.path.join(
                os.path.split(__file__)[0],
                "./fg_parser_conf/fg_emb_conf_no_io_hash.json",
            ),
        ) as f:
            self.fg_emb_conf_no_io_hash = json.load(f)
        with open(
            os.path.join(
                os.path.split(__file__)[0], "./fg_parser_conf/fg_emb_conf_io_hash.json"
            ),
        ) as f:
            self.fg_emb_conf_io_hash = json.load(f)
        with open(
            os.path.join(
                os.path.split(__file__)[0],
                "./fg_parser_conf/fg_emb_conf_already_hash.json",
            ),
        ) as f:
            self.fg_emb_conf_already_hash = json.load(f)

    def test_parsed_conf(self):
        fg_parser = FGParser(self.fg_file_name, self.mc_parser_mask, lower_case=True)
        self.assertTrue(len(fg_parser.parsed_conf_) == len(self.parsed_fg_conf.keys()))
        for fea_conf in fg_parser.parsed_conf_:
            fea_name = fea_conf.name
            fea_conf = asdict(fea_conf)
            self.assertTrue(fea_name in self.parsed_fg_conf)
            self.assertTrue(fea_conf == self.parsed_fg_conf[fea_name])

    def test_get_seq_len(self):
        fg_parser = FGParser(self.fg_file_name, self.mc_parser_mask, lower_case=True)
        self.assertTrue(fg_parser.get_seq_len("s_nid_pv30_c2c") == 0)
        self.assertTrue(fg_parser.get_seq_len("s_nid_ipv30_c2c") == 0)
        self.assertTrue(fg_parser.get_seq_len("usersex_d") == 0)
        self.assertTrue(fg_parser.get_seq_len("longpay_seq_item_id_d_raw") == 200)
        self.assertTrue(fg_parser.get_seq_len("longpay_seq_item_id_d") == 200)
        self.assertTrue(fg_parser.get_seq_len("longpay_seq_pricerank") == 200)
        self.assertTrue(fg_parser.get_seq_len("longpay_seq__context_lp_cnt") == 200)
        self.assertTrue(fg_parser.get_seq_len("longpay_seq__context_lp_time") == 200)
        self.assertTrue(fg_parser.get_seq_len("longpay_seq_length") == 0)
        self.assertTrue(fg_parser.get_seq_len("multimodal_correl_score_v2") == 0)
        self.assertTrue(fg_parser.get_seq_len("item_candidate_seq_item_id") == 2000)
        self.assertTrue(fg_parser.get_seq_len("item_candidate_seq_item_id_raw") == 2000)

    def test_io_conf_no_io_hash(self):
        fg_parser = FGParser(
            self.fg_file_name,
            self.mc_parser_mask,
            lower_case=True,
            hash_in_io=False,
        )
        self.assertTrue(
            len(fg_parser.io_configs.values()) == len(self.fg_io_conf_no_io_hash.keys())
        )
        for io_conf in fg_parser.io_configs.values():
            fea_name = io_conf.name
            io_conf = asdict(io_conf)
            self.assertTrue(fea_name in self.fg_io_conf_no_io_hash)
            self.assertTrue(io_conf == self.fg_io_conf_no_io_hash[fea_name])

    def test_io_conf_io_hash(self):
        fg_parser = FGParser(
            self.fg_file_name,
            self.mc_parser,
            lower_case=True,
            hash_in_io=True,
        )
        self.assertTrue(
            len(fg_parser.io_configs.values()) == len(self.fg_io_conf_io_hash.keys())
        )
        for io_conf in fg_parser.io_configs.values():
            fea_name = io_conf.name
            io_conf = asdict(io_conf)
            self.assertTrue(fea_name in self.fg_io_conf_io_hash)
            self.assertTrue(io_conf == self.fg_io_conf_io_hash[fea_name])

    def test_io_conf_already_hash(self):
        fg_parser = FGParser(
            self.fg_file_name,
            self.mc_parser,
            lower_case=True,
            hash_in_io=False,
            already_hashed=True,
        )
        self.assertTrue(
            len(fg_parser.io_configs.values())
            == len(self.fg_io_conf_already_hash.keys())
        )
        for io_conf in fg_parser.io_configs.values():
            fea_name = io_conf.name
            io_conf = asdict(io_conf)
            self.assertTrue(fea_name in self.fg_io_conf_already_hash)
            self.assertTrue(io_conf == self.fg_io_conf_already_hash[fea_name])

    def test_emb_conf_no_io_hash(self):
        fg_parser = FGParser(
            self.fg_file_name,
            self.mc_parser_mask,
            lower_case=True,
            hash_in_io=False,
            already_hashed=False,
        )
        self.assertTrue(
            len(fg_parser.emb_configs.values())
            == len(self.fg_emb_conf_no_io_hash.keys())
        )
        for emb_conf in fg_parser.emb_configs.values():
            fea_name = emb_conf.out_name
            emb_conf = asdict(emb_conf)
            emb_conf["id_transform_type"] = int(emb_conf["id_transform_type"])
            emb_conf["emb_transform_type"] = int(emb_conf["emb_transform_type"])
            emb_conf["dtype"] = str(emb_conf["dtype"])
            self.assertTrue(fea_name in self.fg_emb_conf_no_io_hash)
            self.assertTrue(emb_conf == self.fg_emb_conf_no_io_hash[fea_name])

    def test_emb_conf_io_hash(self):
        fg_parser = FGParser(
            self.fg_file_name,
            self.mc_parser,
            lower_case=True,
            hash_in_io=True,
            already_hashed=False,
        )
        self.assertTrue(
            len(fg_parser.emb_configs.values()) == len(self.fg_emb_conf_io_hash.keys())
        )
        for emb_conf in fg_parser.emb_configs.values():
            fea_name = emb_conf.out_name
            emb_conf = asdict(emb_conf)
            emb_conf["id_transform_type"] = int(emb_conf["id_transform_type"])
            emb_conf["emb_transform_type"] = int(emb_conf["emb_transform_type"])
            emb_conf["dtype"] = str(emb_conf["dtype"])
            self.assertTrue(fea_name in self.fg_emb_conf_io_hash)
            self.assertTrue(emb_conf == self.fg_emb_conf_io_hash[fea_name])

    def test_emb_conf_already_hash(self):
        fg_parser = FGParser(
            self.fg_file_name,
            self.mc_parser,
            lower_case=True,
            hash_in_io=False,
            already_hashed=True,
        )
        self.assertTrue(
            len(fg_parser.emb_configs.values())
            == len(self.fg_emb_conf_already_hash.keys())
        )
        for emb_conf in fg_parser.emb_configs.values():
            fea_name = emb_conf.out_name
            emb_conf = asdict(emb_conf)
            emb_conf["id_transform_type"] = int(emb_conf["id_transform_type"])
            emb_conf["emb_transform_type"] = int(emb_conf["emb_transform_type"])
            emb_conf["dtype"] = str(emb_conf["dtype"])
            self.assertTrue(fea_name in self.fg_emb_conf_already_hash)
            self.assertTrue(emb_conf == self.fg_emb_conf_already_hash[fea_name])


if __name__ == "__main__":
    unittest.main()
