#cat lake_test.py 
#!/usr/bin/python
#****************************************************************#
# ScriptName: lake_test.py
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2025-12-04 14:20
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2025-12-04 14:20
# Function: 
#***************************************************************#
#






import unittest
from column_io.dataset import dataset as dataset_io
from column_io.dataset.config import LakeConfig
from column_io.dataset.file_sharding import LakeStreamSharding
from column_io.dataset.config import LakeBatchConfig
from column_io.dataset.file_sharding import LakeBatchSharding
class LakeIOTest(unittest.TestCase):
  def test_stream_read(self):
    lake_config = LakeConfig(
                storageName="na61-7",
                projectName="alimama_ecpm_rank_odl",
                tableName="lma100k_ecpm_allscene_creative2_prod",
                columnFamilyName="default_column_family",
                partitionSpec="current/data",
            )
    sharding=LakeStreamSharding()
    sharding.add_path(lake_config.get_v1_path(),1764777600000000,1764781200000000)

    batch_size=64

    select_column=["sample_id_0"]

    dataset = dataset_io.Dataset.from_lake_source(sharding.partition(0,1)[0], False, batch_size, select_column, [], [], [], [], [])
    iterator = iter(dataset)
    for i in range(0,100):
        print(next(iterator))

  def test_multi_cf_stream_read(self):
    lake_config = LakeConfig(
                storageName="ea119-7",
                projectName="recommend_gul_rank",
                tableName="gul_cnxh_cvr_live_sample",
                columnFamilyName="default_column_family",
                partitionSpec="current/data",
            )
    lake_config.add_columnfamily("inc5")

    sharding=LakeStreamSharding()
    sharding.add_path(lake_config.get_v1_path(),1771948800000000,1771952399999999)

    batch_size=64

    select_column=["pv_id","item_id","id","dummy_feature"]

    dataset = dataset_io.Dataset.from_lake_source(sharding.partition(0,1)[0], False, batch_size, select_column, [], [], [], [], [])
    iterator = iter(dataset)
    for i in range(0,100):
        print(next(iterator))

  def test_batch_read(self):
    lake_config = LakeBatchConfig(
                storageName="oss",
                projectName="idlefish_llm4recommend",
                tableName="idle_home_autoregression_ctr_all_feature",
                columnFamilyName="default",
                partitionSpec="20250720/data",
            )
    sharding=LakeBatchSharding()
    sharding.add_path(lake_config.get_v1_path())

    batch_size=64
    select_column=["col1"]

    dataset = dataset_io.Dataset.from_lake_batch_source(sharding.partition(0,1)[0], False, batch_size, select_column, [], [], [], [], [])
    iterator = iter(dataset)
    print(next(iterator))

if __name__ == "__main__":
  unittest.main()



