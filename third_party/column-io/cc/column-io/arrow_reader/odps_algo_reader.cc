#include "column-io/arrow_reader/odps_algo_reader.h"

#if (_GLIBCXX_USE_CXX11_ABI != 1)
#include "column-io/odps/wrapper/odps_table_file_system.h"
#endif

namespace column {
namespace BatchReader {

OdpsAlgoReaderImpl::~OdpsAlgoReaderImpl() = default;

column::Status OdpsAlgoReaderImpl::Create(const std::string &path, int32_t batch_size, const std::vector<std::string> &input_columns, AbstractReaderPtr *out_reader) {
#if (_GLIBCXX_USE_CXX11_ABI == 1)
  return Status::Unimplemented("Cannot Create OdpsAlgoReader in cxx11abi=1 mode");
#else
  using namespace column::odps::wrapper;

  column::odps::AlgoReader *raw_reader = nullptr;

  auto algo_st = OdpsTableFileSystem::Instance()->CreateFileReader(
      path, &raw_reader, batch_size, input_columns);

  if (!algo_st.ok()) {
    return Status::InvalidArgument("Create odps file reader failed: ", path);
  }

  auto odps_reader_ptr = dynamic_cast<OdpsTableReader *>(raw_reader);
  if (odps_reader_ptr == nullptr) {
    delete raw_reader;
    return Status::InvalidArgument("Cast odps file reader failed: ", path);
  }

  std::unique_ptr<OdpsTableReader> reader_uptr(odps_reader_ptr);
  out_reader->reset(new OdpsAlgoReaderImpl(std::move(reader_uptr)));

  return Status::OK();
#endif
}


#if (_GLIBCXX_USE_CXX11_ABI == 1)
OdpsAlgoReaderImpl::OdpsAlgoReaderImpl() = default;
column::Status OdpsAlgoReaderImpl::ReadBatch(std::shared_ptr<arrow::RecordBatch> *data) {
  return Status::Unimplemented("OdpsAlgoReader.ReadBatch not supported in cxx11abi=1");
}
column::Status OdpsAlgoReaderImpl::Seek(int64_t offset) {
  return Status::Unimplemented("OdpsAlgoReader.Seek not supported in cxx11abi=1");
}
column::Status OdpsAlgoReaderImpl::GetTableSize(size_t *size) const {
  return Status::Unimplemented("OdpsAlgoReader.GetTableSize not supported in cxx11abi=1");
}
int64_t OdpsAlgoReaderImpl::Tell() const {
  return -1;
}

#else // _GLIBCXX_USE_CXX11_ABI == 0
OdpsAlgoReaderImpl::OdpsAlgoReaderImpl(std::unique_ptr<column::odps::wrapper::OdpsTableReader> reader): reader_(std::move(reader)) {}
column::Status OdpsAlgoReaderImpl::ReadBatch(std::shared_ptr<arrow::RecordBatch> *data) {
  return reader_->ReadBatch(data);
}
column::Status OdpsAlgoReaderImpl::Seek(int64_t offset) {
  return reader_->Seek(offset);
}
column::Status OdpsAlgoReaderImpl::GetTableSize(size_t *size) const {
  return reader_->CountRecords(size);
}
int64_t OdpsAlgoReaderImpl::Tell() const {
  return reader_->GetReadBytes();
}
#endif

} // namespace BatchReader
} // namespace column