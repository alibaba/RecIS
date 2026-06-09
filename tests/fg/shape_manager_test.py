import json
import os
import unittest

from recis.fg.fg_parser import FGParser
from recis.fg.mc_parser import MCParser
from recis.fg.shape_manager import ShapeManager


class TestShapeManeger(unittest.TestCase):
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
        self.fg_parser_io_hash = FGParser(
            fg_file_name,
            self.mc_parser,
            lower_case=True,
            devel_mode=True,
            hash_in_io=True,
        )
        self.fg_parser_already_hash = FGParser(
            fg_file_name,
            self.mc_parser,
            lower_case=True,
            devel_mode=True,
            hash_in_io=False,
            already_hashed=True,
        )
        self.shape_manager_no_io_hash = ShapeManager(self.fg_parser_no_io_hash)
        self.shape_manager_io_hash = ShapeManager(self.fg_parser_io_hash)
        self.shape_manager_already_hash = ShapeManager(self.fg_parser_already_hash)

    def test_get_feature_shape(self):
        for sm in [
            self.shape_manager_no_io_hash,
            self.shape_manager_io_hash,
            self.shape_manager_already_hash,
        ]:
            self.assertTrue(sm.get_feature_shape("s_nid_pv30_c2c") == [-1, 8])
            self.assertTrue(sm.get_feature_shape("s_nid_ipv30_c2c") == [-1, 8])
            self.assertTrue(sm.get_feature_shape("usersex_d") == [-1, 8])
            self.assertTrue(
                sm.get_feature_shape("longpay_seq_item_id_d") == [-1, 200, 64]
            )
            self.assertTrue(
                sm.get_feature_shape("longpay_seq_pricerank") == [-1, 200, 8]
            )
            self.assertTrue(
                sm.get_feature_shape("longpay_seq__context_lp_cnt") == [-1, 200, 4]
            )
            self.assertTrue(
                sm.get_feature_shape("longpay_seq__context_lp_time") == [-1, 200, 4]
            )
            self.assertTrue(sm.get_feature_shape("longpay_seq_length") == [-1, 1])
            self.assertTrue(
                sm.get_feature_shape("multimodal_correl_score_v2") == [-1, 50]
            )
            self.assertTrue(
                sm.get_feature_shape("item_candidate_seq_item_id") == [-1, 2000, 64]
            )
            self.assertTrue(
                sm.get_feature_shape("item_candidate_seq_item_id_raw") == [-1, 2000, 1]
            )
        self.assertTrue(
            self.shape_manager_no_io_hash.get_feature_shape("longpay_seq_item_id_d_raw")
            == [-1, 200, 1]
        )

    def test_get_block_shape(self):
        for sm in [
            self.shape_manager_no_io_hash,
            self.shape_manager_io_hash,
            self.shape_manager_already_hash,
        ]:
            self.assertTrue(sm.get_block_shape("item_columns") == [-1, 16])
            self.assertTrue(sm.get_block_shape("user_columns") == [-1, 40])
            self.assertTrue(sm.get_block_shape("attention_user") == [-1, 8])
            self.assertTrue(sm.get_block_shape("longpay_seq") == [-1, 200, 72])
            self.assertTrue(sm.get_block_shape("longpay_seq__context") == [-1, 200, 8])
            self.assertTrue(sm.get_block_shape("longpay_seq_length") == [-1, 1])
            self.assertTrue(sm.get_block_shape("multimodal_block") == [-1, 50])
            self.assertTrue(
                sm.get_block_shape("raw_item_candidate_seq_item_id") == [-1, 2000, 1]
            )
            self.assertTrue(sm.get_block_shape("item_candidate_seq") == [-1, 2000, 64])
        self.assertTrue(
            self.shape_manager_no_io_hash.get_block_shape("raw_longpay_seq_item_id_d")
            == [-1, 200, 1]
        )


if __name__ == "__main__":
    unittest.main()
