#include "column-io/arrow_reader/abstract_reader.h"

#include <memory>
#include <stdexcept>

#include "arrow/api.h"
#include "arrow/record_batch.h"
#include "gtest/gtest.h"

namespace column {
namespace BatchReader {
namespace {


// AbstractReader 基类接口行为
TEST(AbstractReaderTest, DefaultMethodsThrow) {
  // The base class default implementations of Seek / GetTableSize / Tell
  // should throw std::runtime_error.
  class MinimalReader : public AbstractReader {
   public:
    column::Status ReadBatch(
        std::shared_ptr<arrow::RecordBatch> *) override {
      return column::Status::OK();
    }
  };

  MinimalReader reader;
  EXPECT_THROW(reader.Seek(0), std::runtime_error);
  EXPECT_THROW({ size_t unused; reader.GetTableSize(&unused); },
               std::runtime_error);
  EXPECT_THROW(reader.Tell(), std::runtime_error);
  EXPECT_THROW(reader.TellTimeStamp(), std::runtime_error);

  std::shared_ptr<arrow::RecordBatch> rb;
  EXPECT_NO_THROW(reader.ReadBatch(&rb));
}


// AbstractReader 不允许拷贝
TEST(AbstractReaderTest, NonCopyable) {
  EXPECT_FALSE(std::is_copy_constructible<AbstractReader>::value);
  EXPECT_FALSE(std::is_copy_assignable<AbstractReader>::value);
}


// AbstractReader 继承行为
class FakeReader : public AbstractReader {
 public:
  explicit FakeReader(std::vector<std::shared_ptr<arrow::RecordBatch>> batches)
      : batches_(std::move(batches)), cursor_(0) {}

  column::Status ReadBatch(
      std::shared_ptr<arrow::RecordBatch> *data) override {
    if (cursor_ >= static_cast<int64_t>(batches_.size())) {
      return column::Status::OutOfRange("No more batches");
    }
    *data = batches_[cursor_++];
    return column::Status::OK();
  }

  column::Status Seek(int64_t offset) override {
    if (offset < 0 ||
        offset > static_cast<int64_t>(batches_.size())) {
      return column::Status::InvalidArgument("Invalid seek offset");
    }
    cursor_ = offset;
    return column::Status::OK();
  }

  column::Status GetTableSize(size_t *size) const override {
    *size = batches_.size();
    return column::Status::OK();
  }

  int64_t Tell() const override { return cursor_; }

 private:
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
  int64_t cursor_;
};
// make a simple RecordBatch with one int32 column.
std::shared_ptr<arrow::RecordBatch> MakeSimpleBatch(
    const std::string &column_name, const std::vector<int32_t> &values) {
  arrow::Int32Builder builder;
  EXPECT_TRUE(builder.AppendValues(values).ok());
  std::shared_ptr<arrow::Array> array;
  EXPECT_TRUE(builder.Finish(&array).ok());

  auto schema = arrow::schema({arrow::field(column_name, arrow::int32())});
  return arrow::RecordBatch::Make(schema, static_cast<int64_t>(values.size()), {array});
}
// 测试 ReadBatch 返回值
TEST(FakeReaderTest, ReadBatchSequentially) {
  auto batch1 = MakeSimpleBatch("col", {1, 2, 3});
  auto batch2 = MakeSimpleBatch("col", {4, 5});

  FakeReader reader({batch1, batch2});

  std::shared_ptr<arrow::RecordBatch> result;
  ASSERT_TRUE(reader.ReadBatch(&result).ok());
  EXPECT_EQ(result->num_rows(), 3);

  ASSERT_TRUE(reader.ReadBatch(&result).ok());
  EXPECT_EQ(result->num_rows(), 2);

  // Third read should be out-of-range.
  auto status = reader.ReadBatch(&result);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), ErrorCode::OUT_OF_RANGE);
}
// 测试 Tell 返回 cursor
TEST(FakeReaderTest, TellReflectsCursor) {
  auto batch = MakeSimpleBatch("x", {10, 20});
  FakeReader reader({batch, batch, batch});

  EXPECT_EQ(reader.Tell(), 0);

  std::shared_ptr<arrow::RecordBatch> unused;
  reader.ReadBatch(&unused);
  EXPECT_EQ(reader.Tell(), 1);

  reader.ReadBatch(&unused);
  EXPECT_EQ(reader.Tell(), 2);
}
// 测试 Seek 返回值
TEST(FakeReaderTest, SeekAndRead) {
  auto batch0 = MakeSimpleBatch("v", {100});
  auto batch1 = MakeSimpleBatch("v", {200});
  FakeReader reader({batch0, batch1});

  // Read first batch, then seek back to 0.
  std::shared_ptr<arrow::RecordBatch> result;
  reader.ReadBatch(&result);
  EXPECT_EQ(reader.Tell(), 1);

  ASSERT_TRUE(reader.Seek(0).ok());
  EXPECT_EQ(reader.Tell(), 0);

  ASSERT_TRUE(reader.ReadBatch(&result).ok());
  EXPECT_EQ(result->num_rows(), 1);
}
TEST(FakeReaderTest, SeekInvalidOffset) {
  FakeReader reader({});
  auto status = reader.Seek(-1);
  EXPECT_FALSE(status.ok());

  status = reader.Seek(1);
  EXPECT_FALSE(status.ok());
}
// 测试 GetTableSize 返回值
TEST(FakeReaderTest, GetTableSize) {
  auto batch = MakeSimpleBatch("a", {1});
  FakeReader reader({batch, batch, batch});

  size_t table_size = 0;
  ASSERT_TRUE(reader.GetTableSize(&table_size).ok());
  EXPECT_EQ(table_size, 3u);
}
// 测试空 Reader
TEST(FakeReaderTest, EmptyReader) {
  FakeReader reader({});

  size_t table_size = 0;
  ASSERT_TRUE(reader.GetTableSize(&table_size).ok());
  EXPECT_EQ(table_size, 0u);

  std::shared_ptr<arrow::RecordBatch> result;
  auto status = reader.ReadBatch(&result);
  EXPECT_FALSE(status.ok());
}

TEST(FakeReaderTest, AbstractReaderPtrOwnership) {
  auto batch = MakeSimpleBatch("col", {42});
  AbstractReaderPtr ptr(new FakeReader({batch}));

  std::shared_ptr<arrow::RecordBatch> result;
  ASSERT_TRUE(ptr->ReadBatch(&result).ok());
  EXPECT_EQ(result->num_rows(), 1);
  EXPECT_EQ(ptr->Tell(), 1);
}

}  // namespace
}  // namespace BatchReader
}  // namespace column
