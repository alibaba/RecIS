from column_io.dataset.odps_env_setup import refresh_odps_io_config
from column_io.dataset.odps_env_setup import init_odps_open_storage_session
from column_io.dataset import dataset as dataset_io
from column_io.lib import interface
from column_io.dataset.open_storage_utils import extract_local_read_session
from column_io.dataset.open_storage_utils import local_register_read_session
import os
import numpy as np
import json

def are_nested_lists_equal(list1, list2):
    if len(list1) != len(list2):
        return False

    for sublist1, sublist2 in zip(list1, list2):
        if len(sublist1) != len(sublist2):
            return False

        for array1, array2 in zip(sublist1, sublist2):
            if not np.array_equal(array1, array2):
                return False

    return True

class TestOpenStorage():
    def __init__(self):
        with open('test.conf', 'r') as file:
            conf = json.load(file)

        self.default_project = conf.get('default_project', '')
        self.project_name = conf.get('project', '')
        self.table = conf.get('table', '')
        self.access_id = conf.get('access_id', '')
        self.access_key = conf.get('access_key', '')
        self.end_point = conf.get('endpoint', '')
        self.tunnel_endpoint = conf.get('tunnel_endpoint', '')
        self.partition = conf.get('partition', '')
        self.odps_path = conf.get('odps_path', '')
        self.ds = conf.get('ds', '')
        
        os.environ["access_key"] = self.access_key
        os.environ["access_id"] = self.access_id
        os.environ["project_name"] = self.project_name
        os.environ["end_point"] = self.end_point
        os.environ["tunnel_end_point"] = self.tunnel_endpoint
        
        self.sep = ","
        self.connect_timeout = 300
        self.rw_timeout = 300
        self.mode = "row"

    def test_data_accuracy(self):
        refresh_odps_io_config(self.project_name, self.access_id, self.access_key, self.end_point, table_name=self.odps_path)

        select_column = [
          "label", #dense double
          "1227_30", #struct
          "154_21"
        ]
        dense_columns = ["label"]
        dense_defaults = [[0, 0]]
        batch_size = 1024
        dataset1 = dataset_io.Dataset.from_odps_source([self.odps_path], True, batch_size, select_column, [], dense_columns, dense_defaults)
        iterator1 = iter(dataset1)

        init_odps_open_storage_session([self.odps_path], required_data_columns=select_column)
        dataset2 = dataset_io.Dataset.from_open_storage_source([self.odps_path], True, batch_size, select_column, [], dense_columns, dense_defaults)
        iterator2 = iter(dataset2)
        
        for idx in range(5):
            element1 = next(iterator1, None)
            element2 = next(iterator2, None)
        
            if element1 is not None and element2 is not None:
                flag = True
                for i in range(2):
                    for key in element1[i].keys():
                        if not are_nested_lists_equal(element1[i][key],element2[i][key]):
                            flag = False
                if flag:
                    print("same")
                else:
                    print("diff")
                    break
            else:
                print("over")
                break
    
    def test_open_storage_select_column(self):
        session_column = [
          "key", 
          "_indicator_1", 
          "group_key",
          "sample_id_0",
          "nickname",
          "label_0",
          "203_0",
          "204_0"
        ]

        select_column = [
          "key", 
          "_indicator_1", 
          "group_key"
        ]
        dense_columns = []
        dense_defaults = []
        batch_size = 1024

        init_odps_open_storage_session([self.odps_path], required_data_columns=session_column)
        dataset = dataset_io.Dataset.from_open_storage_source([self.odps_path], True, batch_size, select_column, [], [], [], dense_columns, dense_defaults, False)
        iterator = iter(dataset)
        
        for _ in range(5):
            element = next(iterator, None)
            print(element)
    
    def test_GetOdpsOpenStorageTableFeatures(self):
        features = interface.GetOdpsOpenStorageTableFeatures(self.odps_path.encode('utf-8'), False)
        print("features", features)
        decode_features = json.loads(features)
        print(decode_features)
        
    def test_register_and_refresh_odps_open_storage_session(self):
        session_id = "202408291708168b141b2100005d6c01target"
        required_data_columns = []
        for i in range(2):
            print("Running {}th time.".format(i))
            local_register_read_session(session_id, self.access_id, self.access_key,
                                        self.tunnel_endpoint, self.end_point,
                                        self.project_name, self.table, self.ds,
                                        required_data_columns,
                                        self.sep, self.mode, self.default_project,
                                        self.connect_timeout, self.rw_timeout)

            session_id, expiration_time, record_count = extract_local_read_session(
                                                          self.access_id, self.access_key,
                                                          self.project_name, self.table, self.ds)
            print("session_id: [{}]".format(session_id))
            print("expiration_time: [{}]".format(expiration_time))
            print("record_count: [{}]".format(record_count))

            st = interface.RefreshReadSessionBatch()
            print("The refresh result is {}. (0:success,1:failed)".format(st)) 

    def test_InitOdpsOpenStorageSessions(self):
        st = interface.InitOdpsOpenStorageSessions(self.access_id, self.access_key, self.tunnel_endpoint, self.end_point, self.project_name, self.table, self.ds,
                                                   '', self.sep, self.mode, self.default_project, self.connect_timeout, self.rw_timeout)
        print("The InitOdpsOpenStorageSessions result is {}. (0:success,1:failed)".format(st))

    def test_GetOdpsOpenStorageTableSize(self):
        st = interface.InitOdpsOpenStorageSessions(self.access_id, self.access_key, self.tunnel_endpoint, self.end_point, self.project_name, self.table, self.ds,
                                                   '', self.sep, self.mode, self.default_project, self.connect_timeout, self.rw_timeout)
        print("The InitOdpsOpenStorageSessions result is {}. (0:success,1:failed)".format(st))
        table_size = interface.GetOdpsOpenStorageTableSize(self.odps_path)
        print("The table size is:{}".format(table_size))

if __name__ == "__main__":
    test = TestOpenStorage()
    # test.test_data_accuracy()
    # test.test_GetOdpsOpenStorageTableFeatures()
    # test.test_register_and_refresh_odps_open_storage_session()
    # test.test_InitOdpsOpenStorageSessions()
    # test.test_GetOdpsOpenStorageTableSize()
    test.test_open_storage_select_column()
