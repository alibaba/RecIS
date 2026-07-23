from unittest.mock import MagicMock
import sys
_mock_py_interface = MagicMock()
_mock_interface = MagicMock()
sys.modules.setdefault("column_io.lib.py_interface", _mock_py_interface)
sys.modules.setdefault("column_io.lib.interface", _mock_interface)
from column_io.dataset import dataset as dataset_io
_mock_py_interface = MagicMock()
_mock_interface = MagicMock()

def test_pack_dataset_fake():
    try: 
        dataset_io.Dataset.pack()
    except Exception as e:
        print(e)


from column_io.dataset import dataset as dataset_io
from column_io.dataset.config import LakeConfig
from column_io.dataset.file_sharding import LakeStreamSharding

def test_lake_dataset():
    lake_config = LakeConfig(
        storageName="na61-7",
        projectName="alimama_ecpm_rank_odl",
        tableName="lma100k_ecpm_allscene_creative2_prod",
        columnFamilyName="default_column_family",
        partitionSpec="current/data")
    
    sharding=LakeStreamSharding()
    sharding.add_path(lake_config.get_v1_path(),1764777600000000,1764781200000000)
    batch_size=64
    select_column=["sample_id_0"]
    try:
        dataset_io.Dataset.from_lake_source(sharding.partition(0,1)[0], False, batch_size, select_column, [], [], [], [], [])
    except Exception as e:
        print(e)
        
    sharding=LakeStreamSharding()
    lake_config.add_columnfamily("inc5")
    sharding.add_path(lake_config.get_v1_path(),1771948800000000,1771952399999999)
    select_column=["sample_id_0","dummy_feature"]
    try:
        dataset_io.Dataset.from_lake_source(sharding.partition(0,1)[0], False, batch_size, select_column, [], [], [], [], [])
    except Exception as e:
        print(e)
