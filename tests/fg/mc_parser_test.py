import os
import unittest

from recis.fg.mc_parser import MCParser


class TestMCParser(unittest.TestCase):
    def setUp(self):
        self.mc_file_name = os.path.join(os.path.split(__file__)[0], "./mc.json")
        self.candidate_seq = {
            "longpay_seq": "longpay_seq",
            "longpay_seq__context": "longpay_seq",
            "item_candidate_seq": "item_candidate_seq",
        }

    def test_lower_case(self):
        MC_parser = MCParser(mc_config_path=self.mc_file_name, lower_case=False)
        self.assertTrue("Multimodal_block" in MC_parser.mc_conf)
        self.assertTrue(
            MC_parser.mc_conf["longpay_seq_length"][0] == "Longpay_seq_length"
        )
        mc_parser = MCParser(mc_config_path=self.mc_file_name, lower_case=True)
        self.assertTrue("multimodal_block" in mc_parser.mc_conf)
        self.assertTrue(
            mc_parser.mc_conf["longpay_seq_length"][0] == "longpay_seq_length"
        )

    def test_uses_columns(self):
        mc_parser = MCParser(
            mc_config_path=self.mc_file_name,
            uses_columns=["item_columns", "longpay_seq"],
            lower_case=True,
        )
        mc_parser.init_blocks(self.candidate_seq)
        self.assertTrue(mc_parser.has_fea("s_nid_pv30_c2c"))
        self.assertFalse(mc_parser.has_fea("usersex_d"))
        self.assertTrue(mc_parser.has_fea("longpay_seq"))
        self.assertTrue(mc_parser.has_fea("longpay_seq_item_id_d"))
        self.assertFalse(mc_parser.has_fea("longpay_seq__context"))
        self.assertFalse(mc_parser.has_fea("longpay_seq__context_lp_cnt"))

    def test_has_feature(self):
        mc_parser = MCParser(mc_config_path=self.mc_file_name, lower_case=True)
        mc_parser.init_blocks(self.candidate_seq)
        self.assertTrue(mc_parser.has_fea("s_nid_pv30_c2c"))
        self.assertTrue(mc_parser.has_fea("s_nid_ipv30_c2c"))
        self.assertTrue(mc_parser.has_fea("usersex_d"))
        self.assertFalse(mc_parser.has_fea("queryseg_norm_d"))
        self.assertFalse(mc_parser.has_fea("uid"))
        self.assertTrue(mc_parser.has_fea("longpay_seq"))
        self.assertTrue(mc_parser.has_fea("longpay_seq_item_id_d"))
        self.assertTrue(mc_parser.has_fea("longpay_seq_pricerank"))
        self.assertTrue(mc_parser.has_fea("longpay_seq__context"))
        self.assertTrue(mc_parser.has_fea("longpay_seq__context_lp_cnt"))
        self.assertTrue(mc_parser.has_fea("longpay_seq__context_lp_time"))
        self.assertFalse(mc_parser.has_fea("longpay_seq_lp_cnt"))
        self.assertFalse(mc_parser.has_fea("longpay_seq_lp_time"))
        self.assertTrue(mc_parser.has_fea("longpay_seq_length"))
        self.assertTrue(mc_parser.has_fea("multimodal_correl_score_v2"))
        self.assertTrue(mc_parser.has_fea("item_candidate_seq_item_id_raw"))
        self.assertTrue(mc_parser.has_fea("item_candidate_seq_item_id"))

        self.assertTrue(mc_parser.has_seq_fea("longpay_seq__context", "lp_cnt"))
        self.assertFalse(mc_parser.has_seq_fea("longpay_seq", "lp_cnt"))
        self.assertTrue(mc_parser.has_seq_fea("longpay_seq__context", "lp_time"))
        self.assertFalse(mc_parser.has_seq_fea("longpay_seq", "lp_time"))
        self.assertTrue(mc_parser.has_seq_fea("longpay_seq", "item_id_d"))
        self.assertTrue(mc_parser.has_seq_fea("longpay_seq", "pricerank"))
        self.assertTrue(mc_parser.has_seq_fea("item_candidate_seq", "item_id"))
        self.assertTrue(mc_parser.has_seq_fea("item_candidate_seq", "item_id_raw"))


if __name__ == "__main__":
    unittest.main()
