#include "column-io/arrow_reader/odps_openstorage_reader.h"

#include "arrow/record_batch.h"
#include "arrow/type.h"
#include "column-io/open_storage/common-util/status.h"

namespace column {
namespace BatchReader {

namespace {
column::Status StatusConvert(apsara::odps::algo::commonio::Status source) {
  ErrorCode code;
  switch (source.GetCode()) {
  case apsara::odps::algo::commonio::Status::kOk:
    code = ErrorCode::OK;
    break;
  case apsara::odps::algo::commonio::Status::kCancelled:
    code = ErrorCode::CANCELLED;
    break;
  case apsara::odps::algo::commonio::Status::kUnknown:
    code = ErrorCode::UNKNOWN;
    break;
  case apsara::odps::algo::commonio::Status::kInvalidArgument:
    code = ErrorCode::INVALID_ARGUMENT;
    break;
  case apsara::odps::algo::commonio::Status::kDeadlineExceeded:
    code = ErrorCode::DEADLINE_EXCEEDED;
    break;
  case apsara::odps::algo::commonio::Status::kNotFound:
    code = ErrorCode::NOT_FOUND;
    break;
  case apsara::odps::algo::commonio::Status::kAlreadyExists:
    code = ErrorCode::ALREADY_EXISTS;
    break;
  case apsara::odps::algo::commonio::Status::kPermissionDenied:
    code = ErrorCode::PERMISSION_DENIED;
    break;
  case apsara::odps::algo::commonio::Status::kResourceExhausted:
    code = ErrorCode::RESOURCE_EXHAUSTED;
    break;
  case apsara::odps::algo::commonio::Status::kFailedPrecondition:
    code = ErrorCode::FAILED_PRECONDITION;
    break;
  case apsara::odps::algo::commonio::Status::kAborted:
    code = ErrorCode::ABORTED;
    break;
  case apsara::odps::algo::commonio::Status::kOutOfRange:
    code = ErrorCode::OUT_OF_RANGE;
    break;
  case apsara::odps::algo::commonio::Status::kUnimplemented:
    code = ErrorCode::UNIMPLEMENTED;
    break;
  case apsara::odps::algo::commonio::Status::kInternal:
    code = ErrorCode::INTERNAL;
    break;
  case apsara::odps::algo::commonio::Status::kUnavailable:
    code = ErrorCode::UNAVAILABLE;
    break;
  case apsara::odps::algo::commonio::Status::kDataLoss:
    code = ErrorCode::DATA_LOSS;
    break;
  case apsara::odps::algo::commonio::Status::kUnauthenticated:
    code = ErrorCode::UNAUTHENTICATED;
    break;
  default:
    code = ErrorCode::UNKNOWN;
    break;
  }
  std::string_view view(source.GetMsg());
  column::Status target(code, view);
  return target;
}
} // namespace

OdpsOpenstorageReaderImpl::OdpsOpenstorageReaderImpl(
    std::shared_ptr<apsara::odps::tunnel::algo::tf::OdpsOpenStorageArrowReader>
        reader,
    std::string path)
    : reader_(std::move(reader)), path_(std::move(path)) {}

OdpsOpenstorageReaderImpl::~OdpsOpenstorageReaderImpl() = default;

column::Status OdpsOpenstorageReaderImpl::Create(
    const std::string &path, int32_t batch_size,
    const std::vector<std::string> &input_columns,
    AbstractReaderPtr *out_reader) {
  using RawReader = apsara::odps::tunnel::algo::tf::OdpsOpenStorageArrowReader;
  std::shared_ptr<RawReader> raw_reader;
  auto algo_st = RawReader::CreateReader(path, batch_size, "ComboSample",
                                         raw_reader, input_columns);
  if (!algo_st.Ok()) {
    return StatusConvert(algo_st);
  }
  out_reader->reset(new OdpsOpenstorageReaderImpl(std::move(raw_reader), path));
  return Status::OK();
}

column::Status OdpsOpenstorageReaderImpl::ReadBatch(std::shared_ptr<arrow::RecordBatch> *data) {
  return StatusConvert(reader_->ReadBatch(*data));
}

column::Status OdpsOpenstorageReaderImpl::Seek(int64_t offset) {
  return StatusConvert(reader_->Seek(offset));
}

column::Status OdpsOpenstorageReaderImpl::GetTableSize(size_t *size) const {
  using RawReader = apsara::odps::tunnel::algo::tf::OdpsOpenStorageArrowReader;
  auto algo_st = RawReader::GetTableSize(path_, *size);
  return StatusConvert(algo_st);
}

int64_t OdpsOpenstorageReaderImpl::Tell() const { return reader_->Tell(); }

} // namespace BatchReader
} // namespace column
