from unittest.mock import MagicMock, patch

from column_io.dataset.file_sharding import (
    OdpsTableSharding,
    OdpsComboTableSharding,
    LakeBatchSharding,
    LakeStreamSharding,
    OrcFileSharding,
)

def mock_get_table_size(table_name):
    table_path_map = {
        "odps://alimama_rank/tables/preranking_daily/ds=20251201": 1234,
        "odps://alimama_rank/tables/preranking_daily/ds=20251202": 4567,
        "odps://alimama_rank/tables/preranking_daily/ds=20251203": 7890,

        "odps://alimama_rank/tables/preranking_hourly/ds=20251201-23": 1234,
        "odps://alimama_rank/tables/preranking_hourly/ds=20251202-23": 4567,
        "odps://alimama_rank/tables/preranking_hourly/ds=20251203-23": 7890,
    }
    return table_path_map.get(table_name, 0)


def _parse_range(partition):
    path, query = partition.split("?")
    params = dict(item.split("=") for item in query.split("&"))
    return path, int(params["start"]), int(params["end"])


def _range_size(partition):
    _, start, end = _parse_range(partition)
    return end - start


@patch("column_io.dataset.file_sharding.OdpsTableDataset")
def test_OdpsTableDataset_add_path_get_row_count(mock_OdpsTableDataset):
    #### mock table size according to following function ####
    mock_OdpsTableDataset.get_table_size.side_effect = mock_get_table_size
    #########################################################
    
    # test bad table
    sharding = OdpsTableSharding()
    sharding.add_path("odps://project_unexist/tables/table_unexist/ds=20114514")
    assert sharding.get_row_count() == 0
    del sharding

    # test normal table
    sharding = OdpsTableSharding()
    sharding.add_path("odps://alimama_rank/tables/preranking_daily/ds=20251201")
    sharding.add_path("odps://alimama_rank/tables/preranking_daily/ds=20251202")
    assert sharding.get_row_count() == 1234 + 4567

@patch("column_io.dataset.file_sharding.OdpsTableDataset")
def test_OdpsTableDataset_partition(mock_OdpsTableDataset):
    mock_OdpsTableDataset.get_table_size.side_effect = mock_get_table_size
    
    sharding = OdpsTableSharding()
    sharding.add_path("odps://alimama_rank/tables/preranking_daily/ds=20251201")
    sharding.add_path("odps://alimama_rank/tables/preranking_daily/ds=20251202")
    sharding.add_path("odps://alimama_rank/tables/preranking_daily/ds=20251203")
    worker_num = 23
    slice_per_worker = 11
    partition_list = sharding.partition(
        worker_idx=7,
        worker_num=worker_num,
        slice_per_worker=slice_per_worker,
    )

    # Example output: [
    # 'odps://alimama_rank/tables/preranking_daily/ds=20251201?start=100&end=101'
    # 'odps://alimama_rank/tables/preranking_daily/ds=20251203?start=7890&end=7890', ]
    # Please note that the calculated values ​​of len(partition_list) and worker_row_count may change 
    #   due to partitioning logic; they do not need to be guaranteed to be fixed globally
    assert isinstance(partition_list, list)
    assert partition_list
    for partition in partition_list:
        assert isinstance(partition, str)
        assert partition.startswith("odps://alimama_rank/tables/preranking_daily/ds=")
        _, start, end = _parse_range(partition)
        assert 0 <= start < end

    all_partitions = []
    for worker_idx in range(worker_num):
        all_partitions.extend(
            sharding.partition(
                worker_idx=worker_idx,
                worker_num=worker_num,
                slice_per_worker=slice_per_worker,
            )
        )
    assert sum(_range_size(partition) for partition in all_partitions) == sharding.get_row_count()

@patch("column_io.dataset.file_sharding.OdpsComboDataset")
def test_OdpsComboTableDataset_partition(mock_OdpsComboDataset):
    mock_OdpsComboDataset.get_table_size.side_effect = mock_get_table_size

    sharding = OdpsComboTableSharding()
    sharding.add_path(
        table_group = [
            "odps://alimama_rank/tables/preranking_daily/ds=20251201",
            "odps://alimama_rank/tables/preranking_hourly/ds=20251201-23",
        ],
        begin_group = [0, 0],
        end_group = [1234, 1234]
    )
    sharding.add_path(
        table_group = [
            "odps://alimama_rank/tables/preranking_daily/ds=20251202",
            "odps://alimama_rank/tables/preranking_hourly/ds=20251202-23",
        ],
        begin_group = [0, 0],
        end_group = [4567, 4567]
    )
    sharding.add_path(
        table_group = [
            "odps://alimama_rank/tables/preranking_daily/ds=20251203",
            "odps://alimama_rank/tables/preranking_hourly/ds=20251203-23",
        ],
        begin_group = [0, 0],
        end_group = [7890, 7890]
    )

    # Please note that the calculated values ​​of len(partition_list) and worker_row_count may change 
    #   due to partitioning logic; they do not need to be guaranteed to be fixed globally

    worker_num = 23
    slice_per_worker = 11
    partition_list: list[tuple[str]] = sharding.partition(
        worker_idx=7,
        worker_num=worker_num,
        slice_per_worker=slice_per_worker,
    )
    assert isinstance(partition_list, list)
    assert partition_list

    for partition_tuple in partition_list:
        assert isinstance(partition_tuple, tuple)
        assert len(partition_tuple) == 2
        partition_base, partition_inc = partition_tuple[0], partition_tuple[1]
        assert partition_base.startswith("odps://alimama_rank/tables/preranking_daily/ds=")
        assert partition_inc.startswith("odps://alimama_rank/tables/preranking_hourly/ds=")
        assert partition_base.split("?")[1] == partition_inc.split("?")[1]
        _, start, end = _parse_range(partition_base)
        assert 0 <= start < end

    all_partition_tuples = []
    for worker_idx in range(worker_num):
        all_partition_tuples.extend(
            sharding.partition(
                worker_idx=worker_idx,
                worker_num=worker_num,
                slice_per_worker=slice_per_worker,
            )
        )
    assert sum(_range_size(partition_tuple[0]) for partition_tuple in all_partition_tuples) == sharding.get_row_count()


def test_OdpsComboTableDataset_recomputes_auto_slice_size_per_group():
    sharding = OdpsComboTableSharding()
    sharding.add_path(
        table_group=["g1_base", "g1_inc"],
        begin_group=[0, 0],
        end_group=[10000, 10000],
    )
    sharding.add_path(
        table_group=["g2_base", "g2_inc"],
        begin_group=[0, 0],
        end_group=[100, 100],
    )

    partitions_by_worker = [
        sharding.partition(worker_idx=worker_idx, worker_num=2, slice_per_worker=1)
        for worker_idx in range(2)
    ]

    g2_ranges_by_worker = []
    for worker_partitions in partitions_by_worker:
        g2_ranges = [
            _parse_range(partition_tuple[0])[1:]
            for partition_tuple in worker_partitions
            if partition_tuple[0].startswith("g2_base?")
        ]
        g2_ranges_by_worker.append(g2_ranges)

    assert all(g2_ranges for g2_ranges in g2_ranges_by_worker)
    assert sorted(range_ for ranges in g2_ranges_by_worker for range_ in ranges) == [
        (0, 50),
        (50, 100),
    ]
