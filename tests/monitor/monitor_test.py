#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ruff: noqa: F401,UP009
import time
import unittest


class TestMonitor(unittest.TestCase):
    def test_monitor(self):
        import torch

        # Method 1: use global static factory
        from recis.monitor.monitor import ClientBase, GetFactory, PointType

        # Method 2: use a standalone factory
        # from recis.monitor.monitor import Factory, Client
        # factory = Factory()

        cli = GetFactory().get_client("test_client_1529")  # type: ClientBase

        cli.report(
            "demo_metric_1",
            0.9876,
            {"appid": "recis-abcdef", "epoch": "0"},
            PointType.kGauge,
        )
        time.sleep(1)
        cli.report(
            "demo_metric_2",
            0.3456,
            {"appid": "recis-abcdef", "epoch": "0"},
            PointType.kGauge,
        )
        time.sleep(1)
        cli.report(
            "demo_metric_1",
            0.6789,
            {"appid": "recis-abczxc", "epoch": "0"},
            PointType.kGauge,
        )
        time.sleep(1)
        # Same metric name + same tag, will be merged as one series line in report sdk
        cli.report(
            "demo_metric_1",
            0.1234,
            {"appid": "recis-abcdef", "epoch": "0"},
            PointType.kGauge,
        )
        time.sleep(1)

        # This is invalid: point_type could not change after client create this series
        assert not cli.report(
            "demo_metric_1",
            0.4321,
            {"appid": "recis-abcdef", "epoch": "0"},
            PointType.kSummary,
        )
        # This is invalid: all tag key and value should be string, according to torch type binding
        # NOTE: 这个case是有效的, 但因检测规则添加不方便故不进入ci测试流程
        # 预期: Cast error details: Unable to cast Python instance of type  to C++ type
        #       Expected a value of type 'Dict[str, str]' for argument '_3' but instead found type 'dict[str, Optional[str, int]]'
        # try:
        #     cli.report(
        #       "demo_metric_1",
        #       0.4321,
        #       {"appid": "recis-abcdef", "epoch": 0},
        #       PointType.kSummary,
        #     )
        # except Exception as e:
        #     pass

        print("test_monitor done")


if __name__ == "__main__":
    unittest.main()
