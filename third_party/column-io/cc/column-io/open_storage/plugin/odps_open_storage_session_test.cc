#include <iostream>

#include "plugin/odps_open_storage_session.h"
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
  -I/Paiio/built/odps_sdk/odps_sdk_cpp/storage_api/thirdparty/cpp-httplib/ \
  -I/Paiio/third_party \
  -lpthread \
  -L/Paiio/built/pip/paiio/lib/odps_tunnel_third_party/lib64/  -larrow \
  /Paiio/built/third_party/common-util/common_util.o \
  /Paiio/built/third_party/common-util/logging.o \
  /Paiio/built/third_party/odps_sdk/plugin/odps_open_storage_session.o \
  /Paiio/third_party/odps_sdk/plugin/odps_open_storage_session_test.cc -o built/third_party/odps_sdk/plugin/odps_open_storage_session_test

**/

using namespace apsara::odps::tunnel::algo::tf;

class TestCases {
 public:
  TestCases() {
    LOG(INFO) << "TestCases constructor.";
  }

  Status test_InitOdpsOpenStorageSessions(OdpsOpenStorageSession** session) {
    LOG(INFO) << "InitOdpsOpenStorageSessions begin.";
    Status st;
    {
      OdpsOpenStorageSession::InitOdpsOpenStorageSessions(access_id_, access_key_,
                                                          tunnel_endpoint_, odps_endpoint,
                                                          projects_, tables_,
                                                          partition_specs_, hysical_partitions_,
                                                          required_data_columns_, sep_);
      st = OdpsOpenStorageSession::GetOdpsOpenStorageSession(
                                     project_, table_, partition_spec_, session);
      if (!st.Ok()) {
        LOG(ERROR) << "open storage arrow reader impl init reader, initializing session error !";
        st.Assign(Status::kInternal, "initializing session error !");
        return st;
      }
      if (session == nullptr || *session == nullptr) {
        LOG(ERROR) << "open storage says get session but got null session !";
        st.Assign(Status::kInternal, "session is not initialized !");
        return st;
      }
      if (!(*session)->IsInitialized()) {
        LOG(ERROR) << "open storage arrow reader impl init reader, session is not initialized !";
        st.Assign(Status::kInternal, "session is not initialized !");
        return st;
      }

      LOG(INFO) << "record_count_: " << std::to_string((*session)->record_count_);
      LOG(INFO) << "expiration_time_: " << std::to_string((*session)->expiration_time_);
      LOG(INFO) << "session_id_: " << (*session)->session_id_;
    }
    LOG(INFO) << "InitOdpsOpenStorageSessions end.";
    return st;
  }

  void test_ExtractSessionDef() {
    std::string session_json = R"(
      {
        "DataSchema": {
          "DataColumns": [{
                "Comment": "",
                "Name": "sample_id",
                "Nullable": true,
                "Type": "string"},
             {
                "Comment": "",
                "Name": "label",
                "Nullable": true,
                "Type": "array<double>"},
             {
                "Comment": "",
                "Name": "client_type",
                "Nullable": true,
                "Type": "array<bigint>"},
             {
                "Comment": "",
                "Name": "pay_seq_is_p4p",
                "Nullable": true,
                "Type": "array<array<double>>"}
          ],
          "PartitionColumns": [{
                "Comment": "",
                "Name": "ds",
                "Nullable": true,
                "Type": "string"},
             {
                "Comment": "",
                "Name": "tag",
                "Nullable": true,
                "Type": "string"}]
        },
        "ExpirationTime": 1713465184134,
        "Message": "",
        "RecordCount": 1568854778,
        "SessionId": "202404180233044c8b580b00001ea902",
        "SessionStatus": "NORMAL",
        "SessionType": "BATCH_READ",
        "SplitsCount": -1,
        "SupportedDataFormat": [{
            "Type": "ARROW",
            "Version": "V5"}]
      }
    )";
    nlohmann::json session_def = nlohmann::json::parse(session_json);
    OdpsOpenStorageSession open_storage_session;

    nlohmann::json* ptr = &session_def;
    auto flag = ptr->contains("RecordCount");
    auto record_count = (*ptr)["RecordCount"].get<long>();
    LOG(INFO) << "flag: " << flag;
    LOG(INFO) << "record_count: " << record_count;

    auto st = open_storage_session.ExtractSessionDef(&session_def);
    LOG(INFO) << "record_count_: " << std::to_string(open_storage_session.record_count_);
    LOG(INFO) << "expiration_time_: " << std::to_string(open_storage_session.expiration_time_);
    LOG(INFO) << "session_id_: " << open_storage_session.session_id_;
    LOG(INFO) << "Status ok: " << st.Ok();
    LOG(INFO) << "last JSON: " << session_def.dump();
  }

  void test_GetReadSession() {
    std::string session_def_str;
    nlohmann::json session_def;

    std::string session_id;
    session_id = "202405080004267c8b580b0000001202";
    OdpsOpenStorageSession::GetReadSession(&session_def_str, &session_def,
                                           access_id_, access_key_, tunnel_endpoint_,
                                           session_id, project_, table_, default_project_,
                                           connect_timeout_, rw_timeout_);
    LOG(INFO) << "test_GetReadSession session_def_str: " << session_def_str;
    LOG(INFO) << "test_GetReadSession session_def: " << session_def.dump(4);

    OdpsOpenStorageSession session;

    apsara::odps::sdk::storage_api::AliyunAccount aliyun_account(access_id_, access_key_);
    apsara::odps::sdk::storage_api::Configuration configuration;
    configuration.SetDefaultProject(default_project_);
    configuration.SetSocketConnectTimeout(connect_timeout_);
    configuration.SetSocketTimeout(rw_timeout_);
    configuration.SetAccount(aliyun_account);
    configuration.SetTunnelEndpoint(tunnel_endpoint_);

    session.project_ = project_;
    session.table_ = table_;
    session.session_id_ = session_id;
    session.configuration_ = configuration;

    //auto table_size = session.GetTableSize();
    //LOG(INFO) << "table_size: " << std::to_string(table_size);
    int table_size = 160925259;
    int start = 0;
    int end = table_size/10;
    int max_batch_rows = 3;
    int cache_size = 1;
    auto reader = session.CreateOpenStorageReader(
                             start, end, max_batch_rows,
                             0, cache_size);

    std::shared_ptr<arrow::RecordBatch> batch;
    long total_line = 0;
    int read_loop_num = 0;
    while (reader->Read(batch)) {
      if (read_loop_num >= 5) {
        LOG(INFO) << "Loop enough time, quit loop.";
        break;
      }
      ++read_loop_num;
      auto num = batch->num_rows();
      LOG(INFO) << "batch num rows: " << std::to_string(num);
      total_line += num;
      int column_num = std::min(2, batch->num_columns());
      for (int64_t i = 0; i < num; i++) {
        for (int j = 0; j < column_num; j++) {
          auto vals = batch->column_data(j)->GetValues<int64_t>(1);
          printf("\t%ld\t", vals[i]);
        }
        printf("\n");
      }
    }
    //if(reader->GetStatus() != Status::OK){
    if(reader->GetStatus() != 0){
      printf("read rows error: %s\n", reader->GetErrorMessage().c_str());
      //return;
    } else {
      printf("total line: %ld\n", total_line);
    }

    std::this_thread::sleep_for(std::chrono::seconds(10));
    /**
[2024:05:08 01:43:23.135][4222]built/odps_sdk/odps_sdk_cpp/storage_api/include/storage_api_arrow.hpp:245: Fail to push record batch to the blocking queue
[2024:05:08 01:43:23.135][4222]built/odps_sdk/odps_sdk_cpp/storage_api/include/storage_api_arrow.hpp:455: Fail to consume arrow stream, buffer len is 10214: IOError, Fail to push record batch
[2024:05:08 01:43:23.135][4222]built/odps_sdk/odps_sdk_cpp/storage_api/include/storage_api.hpp:975: An error occurred when retriving data from server, res error: [Canceled], strerror(errno): [Success], the leading [1024] bytes are:
[/api/storage/v1/projects/alimama_algo_s4_dev/schemas/default/tables/p4p_kgb_rank_darwin_v9_fix_channel_sample_farmhash/data?curr_project=&max_batch_raw_size=16777216&max_batch_rows=1024&row_count=16092525&row_index=0&session_id=202405080004267c8b580b0000001202&split_index=13457112]  []
    [DEBUG] compression: UNCOMPRESSED, total: 26447, cost 2.874296 seconds, read speed: 0.008775 MB/s
    **/
    LOG(INFO) << "test_GetReadSession END";
  }

  void test_CreateOpenStorageReader(OdpsOpenStorageSession* session = nullptr) {
    LOG(INFO) << "test_CreateOpenStorageReader BEGIN";
    {
      bool create_new_session = false;
      if (session == nullptr) {
        LOG(ERROR) << "NO input session AT ALL, going to create new one";
        create_new_session = true;
      } else {
        if (!session->IsInitialized()) {
          LOG(ERROR) << "Input session is not initialized, going to create new one";
          create_new_session = true;
        }
      }

      if (create_new_session) {
        Status st = test_InitOdpsOpenStorageSessions(&session);
        if (st.Ok()) {
          LOG(INFO) << "Init open storage session successfully";
        } else {
          LOG(INFO) << "Init open storage session failed";
          return;
        }
        if (session == nullptr) {
          LOG(ERROR) << "session init successfully but session is nullptr !";
          return;
        }
        if (!session->IsInitialized()) {
          LOG(ERROR) << "session says init ok but not IsInitialized !";
          return;
        }
      }
      auto table_size = session->GetTableSize();
      LOG(INFO) << "table_size: " << std::to_string(table_size);

      int start = 0;
      int end = table_size/10;
      int max_batch_rows = 1024;
      int cache_size = 1;
      auto reader = session->CreateOpenStorageReader(
                               start, end, max_batch_rows,
                               0, cache_size);
      std::shared_ptr<arrow::RecordBatch> batch;
      reader->Read(batch);
      auto num = batch->num_rows();
      LOG(INFO) << "batch num rows: " << std::to_string(num);
    }
    LOG(INFO) << "test_CreateOpenStorageReader END";
  }

  void test_RegisterOdpsOpenStorageSession() {
    LOG(INFO) << "test_RegisterOdpsOpenStorageSession BEGIN";
	
    const bool register_light = false;
    const std::string& session_id = "202407012325195889580b0000000502target";

    Status st = OdpsOpenStorageSession::RegisterOdpsOpenStorageSession(
                                          access_id_, access_key_,
                                          tunnel_endpoint_, odps_endpoint,
                                          project_, table_, partition_spec_,
                                          required_data_columns_, sep_,
                                          mode_, default_project_,
                                          connect_timeout_, rw_timeout_,
                                          register_light, session_id);
    if (st.Ok()) {
      LOG(INFO) << "Registered well";
    } else {
      LOG(ERROR) << "Fail to register";
    }
    LOG(INFO) << "test_RegisterOdpsOpenStorageSession END";
  }

  void test_RegisterOdpsOpenStorageSessionLight(const std::string& session_id = "fake_session",
                                                const long expiration_time = 0,
                                                const long record_count = 0 ,
                                                const std::string& session_def_str = "fake_session_def_str") {
    LOG(INFO) << "test_RegisterOdpsOpenStorageSessionLight BEGIN";

    const bool register_light = true;
    Status st = OdpsOpenStorageSession::RegisterOdpsOpenStorageSession(
                                          access_id_, access_key_,
                                          tunnel_endpoint_, odps_endpoint,
                                          project_, table_, partition_spec_,
                                          required_data_columns_, sep_,
                                          mode_, default_project_,
                                          connect_timeout_, rw_timeout_,
                                          register_light, session_id,
                                          expiration_time, record_count,
                                          session_def_str);
    if (st.Ok()) {
      LOG(INFO) << "Registered well";
    } else {
      LOG(ERROR) << "Fail to register";
    }
    LOG(INFO) << "test_RegisterOdpsOpenStorageSessionLight END";
  }

  void test_RegisterOdpsOpenStorageSessionLight_n_CreateOpenStorageReader() {
	  LOG(INFO) << "test_egisterOdpsOpenStorageSessionLight_n_CreateOpenStorageReader BEGIN";

    const std::string& session_id = "20250219002143f6101b210014e47a01target";
    const long expiration_time = 1739982103163;
    const long record_count = 4155333;
    test_RegisterOdpsOpenStorageSessionLight(session_id, expiration_time, record_count);

    OdpsOpenStorageSession* session = nullptr;
    Status st = OdpsOpenStorageSession::GetOdpsOpenStorageSession(project_, table_, partition_spec_, &session);
    if (!st.Ok()) {
      LOG(ERROR) << "open storage arrow reader impl init reader, initializing session error !";
      return;
    }
    if (session == nullptr) {
      LOG(ERROR) << "open storage says get session but got null session !";
      return;
    }
    if (!session->IsInitialized()) {
      LOG(ERROR) << "open storage arrow reader impl init reader, session is not initialized !";
      return;
    }

    test_CreateOpenStorageReader(session);

	  LOG(INFO) << "test_egisterOdpsOpenStorageSessionLight_n_CreateOpenStorageReader END";
  }

  void test_RefreshReadSession() {
    LOG(INFO) << "test_RefreshReadSession BEGIN";
    Status st = Status();

    //long long ago
    //const std::string& session_id = "202407012325195889580b0000000502target";

    //2024/07/23 10:00-11:00
    const std::string& session_id = "202407231018179c8b580b0000000202target";
	

    const std::string& session_id_3 = "202407311429089c8b580b00001cbb01target";
    const std::string partition_spec_3 = "ds=20240717/tag=train";
    const std::string access_id_3 = "your_access_id_3";
    const std::string access_key_3 = "your_access_key_3";
    const std::string project_3 =  "alimama_algo_s4_dev";
    const std::string table_3 = "p4p_kgb_rank_darwin_v9_fix_channel_sample_farmhash";
    
    st = OdpsOpenStorageSession::RefreshReadSession(access_id_3, access_key_3, tunnel_endpoint_, session_id_3,
                                                    project_3, table_3, default_project_, connect_timeout_, rw_timeout_);
	
    if (st.Ok()) {
      LOG(INFO) << "RefreshReadSession well";
    } else if (st.NoNeedRefresh()) {
      LOG(INFO) << "Registration is less than 12 hours old, no need to refresh.";
    } else {
      LOG(ERROR) << "fail to RefreshReadSession";
    }
    LOG(INFO) << "test_RefreshReadSession END";
  }

  void test_RefreshReadSessionBatch() {
    LOG(INFO) << "test_RefreshReadSessionBatch BEGIN";
    Status st = Status();
    LOG(INFO) << "Need to register first!";

    const std::string& session_id_1 = "20240726143446ae8b580b000021bc01target";

    st = OdpsOpenStorageSession::RegisterOdpsOpenStorageSession(
                                          access_id_, access_key_,
                                          tunnel_endpoint_, odps_endpoint,
                                          project_, table_, partition_spec_,
                                          required_data_columns_, sep_,
                                          mode_, default_project_,
                                          connect_timeout_, rw_timeout_,
                                          false, session_id_1);
    if (!st.Ok()) {
      LOG(ERROR) << "fail to register";
      return;
	  }
	
    const std::string& session_id_2 = "20240726143334ae8b580b000021bb01target";
    const std::string partition_spec_2 = "ds=20230828"; 
    st = OdpsOpenStorageSession::RegisterOdpsOpenStorageSession(
                                          access_id_, access_key_,
                                          tunnel_endpoint_, odps_endpoint,
                                          project_, table_, partition_spec_2,
                                          required_data_columns_, sep_,
                                          mode_, default_project_,
                                          connect_timeout_, rw_timeout_,
                                          false, session_id_2);
    if (!st.Ok()) {
      LOG(ERROR) << "fail to register";
      return;
    }

    const std::string& session_id_3 = "202407261029369c8b580b00001ca301target";
    const std::string partition_spec_3 = "ds=20240717/tag=train";
    const std::string access_id_3 = "your_access_id_3";
    const std::string access_key_3 = "your_access_key_3";
    const std::string project_3 =  "alimama_algo_s4_dev";
    const std::string table_3 = "p4p_kgb_rank_darwin_v9_fix_channel_sample_farmhash";
    st = OdpsOpenStorageSession::RegisterOdpsOpenStorageSession(
                                          access_id_3, access_key_3,
                                          tunnel_endpoint_, odps_endpoint,
                                          project_3, table_3, partition_spec_3,
                                          required_data_columns_, sep_,
                                          mode_, default_project_,
                                          connect_timeout_, rw_timeout_,
                                          false, session_id_3);
    if (!st.Ok()) {
      LOG(ERROR) << "fail to register";
      return;
    }

	const std::string& session_id_4 = "202407261025179c8b580b00001ca001target";
    const std::string partition_spec_4 = "ds=2024071523";
    const std::string access_id_4 = "your_access_id_4";
    const std::string access_key_4 = "your_access_key_4";
    const std::string project_4 =  "palgo_fpage";
    const std::string table_4 = "newgul_track_sample_nebula_hh_columnize_std_neg_sample";
    st = OdpsOpenStorageSession::RegisterOdpsOpenStorageSession(
                                          access_id_4, access_key_4,
                                          tunnel_endpoint_, odps_endpoint,
                                          project_4, table_4, partition_spec_4,
                                          required_data_columns_, sep_,
                                          mode_, default_project_,
                                          connect_timeout_, rw_timeout_,
                                          false, session_id_4);
    if (!st.Ok()) {
      LOG(ERROR) << "fail to register";
      return;
    }

	LOG(INFO) << "Finish register!";

	st = OdpsOpenStorageSession::RefreshReadSessionBatch();

    if (st.Ok()) {
      LOG(INFO) << "Refresh successful";
    } else {
      LOG(ERROR) << "fail to Refresh";
    }
    LOG(INFO) << "test_RefreshReadSessionBatch END";
  }

  const std::string access_id_ = "your_access_id";
  const std::string access_key_ = "your_access_key";
  const std::string tunnel_endpoint_ = "ipc:///home/admin/halo/socket/halo.sock";
  const std::string odps_endpoint = "";
  const std::string projects_ = "nebula_ai_dev";
  const std::string project_ =  "nebula_ai_dev";
  const std::string tables_ = "nmd_daily_sample_allpid_final_nebula";
  const std::string table_ = "nmd_daily_sample_allpid_final_nebula";
  const std::string partition_specs_ = "ds=20230827";
  const std::string partition_spec_ = "ds=20230827";
  const std::string physical_partitions_ = "";

  const std::string required_data_columns_ = "";
  const std::string sep_ = ",";
  const std::string mode_ = "row";
  const std::string default_project_ = "";
  int connect_timeout_ = 300;
  int rw_timeout_ = 300;
};


int main() {
  auto test_cases = TestCases();
  OdpsOpenStorageSession* session = nullptr;
  //test_cases.test_InitOdpsOpenStorageSessions(&session);
  //test_cases.test_ExtractSessionDef();
  //test_cases.test_GetReadSession();
  //test_cases.test_CreateOpenStorageReader();
  //test_cases.test_RegisterOdpsOpenStorageSession();
  //test_cases.test_RegisterOdpsOpenStorageSessionLight();
  test_cases.test_RegisterOdpsOpenStorageSessionLight_n_CreateOpenStorageReader();
  //test_cases.test_RefreshReadSession();
  //test_cases.test_RefreshReadSessionBatch();
}

