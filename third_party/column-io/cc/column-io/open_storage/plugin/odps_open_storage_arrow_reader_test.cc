#include <iostream>

#include "plugin/odps_open_storage_session.h"
#include "plugin/odps_open_storage_arrow_reader.h"
#include "common-util/logging.h"

#include "nlohmann/json.hpp"
#include "arrow/ipc/api.h"
#include "arrow/api.h"
#include "arrow/type_fwd.h"
#include "arrow/io/memory.h"


/**
Compile command:

1. Compile paiio first

export CPATH=$CPATH:/apsara/alicpp/built/gcc-4.9.2/arrow-0.16.0/include;
export LIBRARY_PATH=$LIBRARY_PATH:/apsara/alicpp/built/gcc-4.9.2/arrow-0.16.0/lib64
source /venvs/default/bin/activate
make -j 64 TF_NEED_PANGU_TEMP=0 TF_NEED_ODPS_COLUMN=1


2. Compile test case

g++  -Wall -O3 -g -std=c++11 -fPIC -D_GLIBCXX_USE_CXX11_ABI=1 \
  -DTF_ENABLE_ODPS_COLUMN \
  -I/Paiio/third_party/odps_sdk -I/Paiio/built/odps_sdk/odps_sdk_cpp \
  -I/Paiio/built/odps_sdk/third_party/include \
  -I/Paiio/built/odps_sdk/odps_sdk_cpp/storage_api/thirdparty \
  -I/Paiio/third_party \
  -lpthread \
  -L/Paiio/built/pip/paiio/lib/odps_tunnel_third_party/lib64/  -larrow \
  /Paiio/built/third_party/common-util/common_util.o \
  /Paiio/built/third_party/common-util/logging.o \
  /Paiio/built/third_party/odps_sdk/plugin/odps_open_storage_session.o \
  /Paiio/built/third_party/odps_sdk/plugin/odps_open_storage_arrow_reader.o \
  /Paiio/third_party/odps_sdk/plugin/odps_open_storage_arrow_reader_test.cc -o built/third_party/odps_sdk/plugin/odps_open_storage_arrow_reader_test

**/

using namespace apsara::odps::tunnel::algo::tf;

class TestCases {
 public:
  TestCases() {
    LOG(INFO) << "TestCases constructor.";
  }

  Status test_InitOdpsOpenStorageSessions(OdpsOpenStorageSession* session) {
    LOG(INFO) << "InitOdpsOpenStorageSessions begin.";
    Status st = Status();
    {
      OdpsOpenStorageSession::InitOdpsOpenStorageSessions(access_id, access_key,
                                                          tunnel_endpoint, odps_endpoint,
                                                          projects, tables,
                                                          partition_specs, physical_partitions,
                                                          required_data_columns, sep);
      st = OdpsOpenStorageSession::GetOdpsOpenStorageSession(
                                     project, table, partition_spec, &session);
      if (!st.Ok()) {
        const std::string& error_msg = "open storage arrow reader impl init reader, initializing session error !";
        LOG(ERROR) << error_msg;
        st.Assign(Status::kInternal, error_msg);
        return st;
      }
      if (!session->IsInitialized()) {
        const std::string& error_msg =  "open storage arrow reader impl init reader, session is not initialized !";
        LOG(ERROR) << error_msg;
        st.Assign(Status::kInternal, error_msg);
        return st;
      }

      LOG(INFO) << "record_count_: " << std::to_string(session->record_count_);
      LOG(INFO) << "expiration_time_: " << std::to_string(session->expiration_time_);
      LOG(INFO) << "session_id_: " << session->session_id_;
    }
    LOG(INFO) << "InitOdpsOpenStorageSessions end.";
    return st;
  }

  OdpsOpenStorageSession test_GetReadSession() {
    std::string session_def_str;
    nlohmann::json session_def;

    std::string session_id;
    session_id = "20240508002634ae8b580b0000001602";
    session_id = "202405080004267c8b580b0000001202";
    const std::string& default_project = "";
    int connect_timeout = 300;
    int rw_timeout = 300;
    OdpsOpenStorageSession::GetReadSession(&session_def_str, &session_def,
                                           access_id, access_key, tunnel_endpoint,
                                           session_id, project, table, default_project,
                                           connect_timeout, rw_timeout);
    LOG(INFO) << "test_GetReadSession session_def_str: " << session_def_str;
    LOG(INFO) << "test_GetReadSession session_def: " << session_def.dump(4);

    OdpsOpenStorageSession session;

    apsara::odps::sdk::storage_api::AliyunAccount aliyun_account(access_id, access_key);
    apsara::odps::sdk::storage_api::Configuration configuration;
    configuration.SetDefaultProject(default_project);
    configuration.SetSocketConnectTimeout(connect_timeout);
    configuration.SetSocketTimeout(rw_timeout);
    configuration.SetAccount(aliyun_account);
    configuration.SetTunnelEndpoint(tunnel_endpoint);

    session.project_ = project;
    session.table_ = table;
    session.session_id_ = session_id;
    session.configuration_ = configuration;

    LOG(INFO) << "finish get read session.";
    return session;
  }

  void test_CreateReader() {
    //OdpsOpenStorageSession session = test_GetReadSession();   // 此种方式目前无法得到一个初始化完备的session
    OdpsOpenStorageSession* session = nullptr;
    Status st = test_InitOdpsOpenStorageSessions(session);

    //const std::string& path_str = "";  // 考虑增加这么一个 path_str = "" 的case测试
    const std::string& path_str = "odps://alimama_algo_s4_dev/tables/p4p_kgb_rank_darwin_v9_fix_channel_sample_farmhash/ds=20240429/tag=test?start=0&end=1000000";
    const int max_batch_rows = 10;
    std::shared_ptr<OdpsOpenStorageArrowReader> reader;
    LOG(INFO) << "before CreateReader";
    st = OdpsOpenStorageArrowReader::CreateReader(path_str, max_batch_rows, "TestReader", reader);
    LOG(INFO) << "CreateReader status: " << st.GetCode();
    
    std::shared_ptr<arrow::RecordBatch> batch;
    st = reader->ReadBatch(batch);
    auto num = batch->num_rows();
    LOG(INFO) << "ReadBatch status: " << st.GetCode() << " num_rows: " << num;
    int column_num = std::min(2, batch->num_columns());
    for (int64_t i = 0; i < num; i++) {
      for (int j = 0; j < column_num; j++) {
        auto vals = batch->column_data(j)->GetValues<int64_t>(1);
        printf("\t%ld\t", vals[i]);
      }
      printf("\n");
    }

    st = reader->Seek(10000);
    LOG(INFO) << "Seek status: " << st.GetCode();

    size_t position = reader->Tell();
    LOG(INFO) << "Tell position: " << position;

    st = reader->Close();
    LOG(INFO) << "Close status: " << st.GetCode();
  }

  const std::string access_id = "your_access_id";
  const std::string access_key = "your_access_key";
  const std::string tunnel_endpoint = "11.228.56.211:80";
  const std::string odps_endpoint = "";
  const std::string projects = "alimama_algo_s4_dev";
  const std::string project =  "alimama_algo_s4_dev";
  const std::string tables = "p4p_kgb_rank_darwin_v9_fix_channel_sample_farmhash";
  const std::string table =  "p4p_kgb_rank_darwin_v9_fix_channel_sample_farmhash";
  const std::string partition_specs = "ds=20240429/tag=test";
  const std::string partition_spec =  "ds=20240429/tag=test";
  const std::string physical_partitions = "";
  const std::string required_data_columns = "label";
  const std::string sep = ",";
};

int main() {
  auto test_cases = TestCases();
  test_cases.test_CreateReader();
}

