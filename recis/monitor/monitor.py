# ruff: noqa: F401,G001,UP032,UP009,UP015,RUF013
import os

import torch

import recis  # load recis shared lib, but not call directly


class PointType:
    kGauge = 100
    kGaugeSticky = 110
    kCounter = 200
    kSummary = 300
    # kHistogram = 400
    # kHistogramFull = 410


class ClientBase:
    # `ClientBase` is not a really workable class, just an type-hint wrapper of LIB torch.class_<recis.MonitorClient>
    def __init__(self, client_name: str):
        r"""
        Args:
            client_name (str):   the client name. composed of letters, numbers and underscores.
                            all report name will share the same client_name prefix.
        """

    def report(
        self, name: str, value: float, tag_dict: dict[str, str], point_type: PointType
    ):
        r"""report a metric instant.
        Args:
            name (str):     the metric name. composed of letters, numbers and underscores.
            value (float):  the metric value.
            tag_dict (dict[str, str]):  the metric tags. same tag_dict under the same name, will be merged as one series line in agent
            point_type (PointType):     the metric type. Note it could not change after client create this series
        """

    def reset_metric(self, name: str, tag_dict: dict[str, str]):
        r"""clear value of a metric name + tag_dict.
        Args:
            name (str): the metric name.
            tag_dict (dict[str, str]):  the metric tags.
        Note that this method will clear content by different PointType:
            - Gauge&Summary: only for this interval;
            - Counter: accumulatiion to 0 from start time;
        Any unexisted or un registed name + tag_dict will be ignored
        """

    def _take_snapshot(self) -> str:
        pass


class FactoryBase:
    def __init__(self, write_path: str = ""):
        self._fact: dict[str, ClientBase] = {}

    def get_client(self, client_name: str) -> ClientBase:
        self._fact[client_name] = self._fact.get(client_name, ClientBase(client_name))
        return self._fact[client_name]


class Factory(FactoryBase):
    r"""
    Factory is a factory definition. It's able to create、manage and dump snapshot of Client instances
    For performance consideration, it's better to create one Factory instance per process.
    Example::
        >>> factory = Factory() # static instance in main process
        >>> client1 = factory.get_client("client_1")
        # do some report...
        >>> client2 = factory.get_client("client_2")
        # do some report...
        # do more client and client.report...
        # ... After some seconds ...
        # factory automatically dumps client1 and client2's snapshot every interval
        >>> del factory # release factory after need. once factory released, all client report will be dropped
    """

    def __init__(self, write_path: str = ""):
        self._fact = torch.ops.recis.make_MonitorFactory(write_path)
        self._launch_collector_daemon()

    @staticmethod
    def _launch_collector_daemon():
        import subprocess

        collector_dir = os.path.dirname(os.path.abspath(__file__))
        collector_path = os.path.join(collector_dir, "collector.py")
        _ = subprocess.Popen(
            f"python {collector_path}",
            shell=True,
            stdout=subprocess.DEVNULL,  # drop stdout
            stderr=subprocess.DEVNULL,  # drop stderr
            stdin=subprocess.DEVNULL,  # drop stdin
            start_new_session=True,  # detach from self process
        )
        # logger.debug(f"launch collector daemon pid: {proc.pid}")

    def get_client(self, client_name: str) -> ClientBase:
        return self._fact.get_client(client_name)


_factory: Factory = None  #


def GetFactory() -> FactoryBase:
    global _factory
    if _factory is None:
        if os.environ.get("RECIS_MONITOR_ON", "1") == "1":
            _factory = Factory()
        else:
            print("[INFO] RECIS_MONITOR_ON is not 1, use empty monitor")
            _factory = FactoryBase()
    return _factory
