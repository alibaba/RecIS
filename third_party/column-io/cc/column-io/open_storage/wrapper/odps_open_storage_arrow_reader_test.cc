#include <iostream>

#include "wrapper/odps_open_storage_arrow_reader.h"
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
  -I/Paiio/third_party \
  -I/Paiio/built/odps_sdk/odps_sdk_cpp/include \
  -I/Paiio/built/odps_sdk/odps_sdk_cpp/storage_api/thirdparty \
  -lpthread \
  -L/Paiio/built/pip/paiio/lib/odps_tunnel_third_party/lib64/  -larrow \
  /Paiio/built/third_party/common-util/common_util.o \
  /Paiio/built/third_party/common-util/logging.o \
  /Paiio/built/third_party/odps_sdk/wrapper/odps_open_storage_arrow_reader.o \
  /Paiio/built/third_party/odps_sdk/wrapper/odps_open_storage_reader_proxy.o \
  /Paiio/built/third_party/odps_sdk/wrapper/dl_wrapper_open_storage.o \
  -ldl \
  /Paiio/third_party/odps_sdk/wrapper/odps_open_storage_arrow_reader_test.cc -o built/third_party/odps_sdk/plugin/odps_open_storage_arrow_reader_test

**/

using namespace apsara::odps::tunnel::algo::tf;

class TestCases {
 public:
  TestCases() {
    LOG(INFO) << "TestCases constructor.";
  }

  Status test_InitOdpsOpenStorageSessions() {
    LOG(INFO) << "InitOdpsOpenStorageSessions begin.";
    Status st;
    OdpsOpenStorageArrowReader::InitOdpsOpenStorageSessions(access_id_, access_key_,
                                                            tunnel_endpoint_, odps_endpoint_,
                                                            projects_, tables_,
                                                            partition_specs_, physical_partitions_,
                                                            required_data_columns_, sep_);
    LOG(INFO) << "InitOdpsOpenStorageSessions end.";
    return st;
  }

  void test_RegisterOdpsOpenStorageSession() {
    LOG(INFO) << "test_RegisterOdpsOpenStorageSession BEGIN";

    const std::string& session_id = "202407012325195889580b0000000502target";
    const bool register_light = false;

    Status st = OdpsOpenStorageArrowReader::RegisterOdpsOpenStorageSession(
                                              access_id_, access_key_,
                                              tunnel_endpoint_, odps_endpoint_,
                                              project_, table_,
                                              partition_spec_, physical_partitions_,
                                              required_data_columns_, sep_,
                                              mode_, default_project_,
                                              connect_timeout_, rw_timeout_,
                                              register_light, session_id);
    if (st.Ok()) {
      LOG(INFO) << "registered well";
    } else {
      LOG(ERROR) << "fail to register";
    }
    LOG(INFO) << "test_RegisterOdpsOpenStorageSession END";
  }

  void test_RegisterOdpsOpenStorageSessionLight() {
    LOG(INFO) << "test_RegisterOdpsOpenStorageSessionLight BEGIN";

    const std::string& session_id = "202502231814268b141b210026ea8b01target";
    const long expiration_time = 1740392066548;
    const long record_count = 4155333;
    const bool register_light = true;
    const std::string& session_def_str = "fake_session_def_str";

    Status st = OdpsOpenStorageArrowReader::RegisterOdpsOpenStorageSession(
                                              access_id_, access_key_,
                                              tunnel_endpoint_, odps_endpoint_,
                                              project_, table_,
                                              partition_spec_, physical_partitions_,
                                              required_data_columns_, sep_,
                                              mode_, default_project_,
                                              connect_timeout_, rw_timeout_,
                                              register_light, session_id,
                                              expiration_time, record_count,
                                              session_def_str);
    if (st.Ok()) {
      LOG(INFO) << "Light registered well";
    } else {
      LOG(ERROR) << "Light fail to register";
    }
    LOG(INFO) << "test_RegisterOdpsOpenStorageSessionLight END";
  }

  void test_RegisterOdpsOpenStorageSessionLight_n_CreateReader() {
    LOG(INFO) << "test_RegisterOdpsOpenStorageSessionLight_n_CreateReader BEGIN";

    test_RegisterOdpsOpenStorageSessionLight();

    // e.g. odps://project_name/tables/table_name/ds=ds_name?start=0&end=30
    const int start = 0;
    const int end = 300;
    const int max_batch_rows = 100;
    std::string path_str = "odps://" + project_ + "/tables/" + table_ + "/" + partition_spec_ +
                           "?start=" + std::to_string(start) + "&end=" + std::to_string(end);
    const std::string& reader_name = "test_case_reader";
    std::shared_ptr<OdpsOpenStorageArrowReader> reader;
    LOG(INFO) << "test path str: " << path_str;
    Status st = OdpsOpenStorageArrowReader::CreateReader(path_str, max_batch_rows,
                                                         reader_name, reader);
    if (st.Ok()) {
      LOG(INFO) << "CreateReader ok.";
    } else {
      LOG(ERROR) << "Fail to CreateReader";
      return;
    }
    std::shared_ptr<arrow::RecordBatch> batch;
    reader->ReadBatch(batch);
    auto num = batch->num_rows();
    LOG(INFO) << "batch num rows: " << std::to_string(num);

    LOG(INFO) << "test_RegisterOdpsOpenStorageSessionLight_n_CreateReader END";
  }

  const std::string access_id_ = "your_access_id_";
  const std::string access_key_ = "your_access_key_";
  const std::string tunnel_endpoint_ = "ipc:///home/admin/halo/socket/halo.sock";
  const std::string odps_endpoint_ = "";
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
  //test_cases.test_InitOdpsOpenStorageSessions();
  //test_cases.test_RegisterOdpsOpenStorageSession();
  //test_cases.test_RegisterOdpsOpenStorageSessionLight();
  test_cases.test_RegisterOdpsOpenStorageSessionLight_n_CreateReader();
  return 0;
}

