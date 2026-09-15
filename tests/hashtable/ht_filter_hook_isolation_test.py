"""Regression coverage for test isolation without allocating GPU tensors."""

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

from recis.nn.modules.hashtable import HashtableRegister
from recis.nn.modules.hashtable_hook_impl import HashtableHookFactory


spec = importlib.util.spec_from_file_location(
    "filter_hook_test_case", Path(__file__).with_name("ht_filter_hook_test.py")
)
filter_case_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filter_case_module)


class FilterHookIsolationTest(unittest.TestCase):
    def test_setup_preserves_implementations_and_restores_instances_on_failure(self):
        registry = HashtableRegister()
        factory = HashtableHookFactory()
        filters, admits = factory.get_filters(), factory.get_admits()
        implementations = dict(factory._registed_filter)
        admission_implementations = dict(factory._registed_admit)
        case = filter_case_module.HashTableFilterHookTest("testFilterHook")

        def probe():
            self.assertIsNot(HashtableRegister(), registry)
            HashtableRegister().register("ht3", "probe", object())
            self.assertIs(HashtableHookFactory(), factory)
            self.assertEqual(factory.get_filters(), {})
            self.assertEqual(factory.get_admits(), {})
            self.assertEqual(factory._registed_filter, implementations)
            self.assertEqual(factory._registed_admit, admission_implementations)
            self.assertIn("GlobalStepFilter", factory._registed_filter)
            self.assertIn("ReadOnly", factory._registed_admit)
            raise RuntimeError("probe setup failure")

        try:
            with mock.patch.object(case, "setIds", side_effect=probe):
                with self.assertRaisesRegex(RuntimeError, "probe setup failure"):
                    case.setUp()
        finally:
            case.doCleanups()
        self.assertIs(HashtableRegister(), registry)
        self.assertIs(factory.get_filters(), filters)
        self.assertIs(factory.get_admits(), admits)
