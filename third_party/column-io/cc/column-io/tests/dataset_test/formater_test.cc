#include "column-io/dataset/formater.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include "arrow/api.h"
#include "arrow/record_batch.h"
#include "gtest/gtest.h"

#include "column-io/framework/status.h"
#include "column-io/framework/tensor.h"
#include "column-io/tests/dataset_test/arrow_help.h"

namespace column {
namespace dataset {
namespace {

// ===========================================================================
// GetColumnDataFormater 工厂方法
// ===========================================================================

TEST(ColumnDataFormaterFactoryTest, CreateFlatListFormater) {
  auto formater = ColumnDataFormater::GetColumnDataFormater(
      /*is_compressed=*/false, /*is_large_list=*/false);
  ASSERT_NE(formater, nullptr);
}

TEST(ColumnDataFormaterFactoryTest, CreateCompressedListFormater) {
  auto formater = ColumnDataFormater::GetColumnDataFormater(
      /*is_compressed=*/true, /*is_large_list=*/false);
  ASSERT_NE(formater, nullptr);
}

TEST(ColumnDataFormaterFactoryTest, CreateWithNullFlag) {
  auto formater = ColumnDataFormater::GetColumnDataFormater(
      /*is_compressed=*/false, /*is_large_list=*/false, /*with_null=*/true);
  ASSERT_NE(formater, nullptr);
}


// ===========================================================================
// FlatColumnDataFormater — InitSchema
// ===========================================================================

class FlatFormaterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    formater_ = ColumnDataFormater::GetColumnDataFormater(
        /*is_compressed=*/false, /*is_large_list=*/false);
  }

  // Convenience: initialise schema with a single list<int32> column, no hash,
  // no dense defaults, no column filter.
  Status InitSimpleSchema(const std::string &column_name) {
    auto schema = arrow::schema(
        {arrow::field(column_name, arrow::list(arrow::int32()))});
    return formater_->InitSchema(
        schema,
        /*hash_features=*/{},
        /*hash_types=*/{},
        /*hash_buckets=*/{},
        /*dense_features=*/{},
        /*dense_defaults=*/{},
        /*selected_columns=*/{});
  }

  std::unique_ptr<ColumnDataFormater> formater_;
};

TEST_F(FlatFormaterTest, InitSchemaSucceeds) {
  auto status = InitSimpleSchema("feature_a");
  EXPECT_TRUE(status.ok()) << status.error_message();
}

TEST_F(FlatFormaterTest, GetOutputSchemaAfterInit) {
  ASSERT_TRUE(InitSimpleSchema("feature_a").ok());
  std::vector<std::map<std::string, std::string>> output_schema;
  auto status = formater_->GetOutputSchema(&output_schema);
  EXPECT_TRUE(status.ok());
  EXPECT_FALSE(output_schema.empty());
}

TEST_F(FlatFormaterTest, GetOutputSchemaBeforeInitFails) {
  std::vector<std::map<std::string, std::string>> output_schema;
  auto status = formater_->GetOutputSchema(&output_schema);
  EXPECT_FALSE(status.ok());
}

// ===========================================================================
// FlatColumnDataFormater — FormatSample
// ===========================================================================

TEST_F(FlatFormaterTest, FormatSampleBasic) {
  ASSERT_TRUE(InitSimpleSchema("col").ok());

  auto batch = MakeListBatch<int32_t>("col", {{1, 2}, {3, 4, 5}});
  std::vector<std::shared_ptr<arrow::RecordBatch>> formated;
  auto status = formater_->FormatSample(batch, &formated);
  EXPECT_TRUE(status.ok()) << status.error_message();
  EXPECT_FALSE(formated.empty());
}

// ===========================================================================
// FlatColumnDataFormater — Convert
// ===========================================================================

TEST_F(FlatFormaterTest, ConvertBasic) {
  ASSERT_TRUE(InitSimpleSchema("col").ok());

  auto batch = MakeListBatch<int32_t>("col", {{10, 20}, {30}});
  std::vector<std::shared_ptr<arrow::RecordBatch>> formated;
  ASSERT_TRUE(formater_->FormatSample(batch, &formated).ok());

  std::vector<std::map<std::string, std::vector<std::vector<Tensor>>>> output;
  auto status = formater_->Convert(formated, &output);
  EXPECT_TRUE(status.ok()) << status.error_message();
  EXPECT_FALSE(output.empty());
}

// ===========================================================================
// FlatColumnDataFormater — FlatConvert
// ===========================================================================

TEST_F(FlatFormaterTest, FlatConvertToTensorVector) {
  ASSERT_TRUE(InitSimpleSchema("col").ok());

  auto batch = MakeListBatch<int32_t>("col", {{1}, {2}});
  std::vector<std::shared_ptr<arrow::RecordBatch>> formated;
  ASSERT_TRUE(formater_->FormatSample(batch, &formated).ok());

  std::vector<Tensor> flat_output;
  auto status = formater_->FlatConvert(formated, &flat_output);
  EXPECT_TRUE(status.ok()) << status.error_message();
  EXPECT_FALSE(flat_output.empty());
}

// ===========================================================================
// FlatColumnDataFormater — GetInputColumns
// ===========================================================================

TEST_F(FlatFormaterTest, GetInputColumnsReturnsInitedColumns) {
  ASSERT_TRUE(InitSimpleSchema("my_feature").ok());

  std::vector<std::string> input_columns;
  formater_->GetInputColumns(&input_columns);
  ASSERT_EQ(input_columns.size(), 1u);
  EXPECT_EQ(input_columns[0], "my_feature");
}

// ===========================================================================
// FlatColumnDataFormater — DebugString / LogDebugString
// ===========================================================================

TEST_F(FlatFormaterTest, DebugStringNonNull) {
  auto batch = MakeListBatch<int32_t>("x", {{1}});
  std::string debug = formater_->DebugString(batch);
  EXPECT_FALSE(debug.empty());
}

TEST_F(FlatFormaterTest, DebugStringNullBatch) {
  std::string debug = formater_->DebugString(nullptr);
  EXPECT_NE(debug.find("empty"), std::string::npos);
}

// ===========================================================================
// CompressedColumnDataFormater — scaffold
// ===========================================================================

class CompressedFormaterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    formater_ = ColumnDataFormater::GetColumnDataFormater(
        /*is_compressed=*/true, /*is_large_list=*/false);
  }

  Status InitSimpleSchema(const std::vector<std::string> &column_name_list) {
    // auto schema = arrow::schema({arrow::field(column_name, arrow::list(arrow::int32()))});
    auto schema = arrow::schema(
        {arrow::field(column_name_list[0], arrow::list(arrow::int32()))});
    return formater_->InitSchema(
        schema,
        /*hash_features=*/{},
        /*hash_types=*/{},
        /*hash_buckets=*/{},
        /*dense_features=*/{},
        /*dense_defaults=*/{},
        /*selected_columns=*/{});
  }

  std::unique_ptr<ColumnDataFormater> formater_;
};

TEST_F(CompressedFormaterTest, CreateSucceeds) {
  ASSERT_NE(formater_, nullptr);
}

TEST_F(CompressedFormaterTest, FormatSampleOK) {
  auto batch = MakeRecordBatch<int64_t>(std::vector<std::pair<std::string, std::vector<std::vector<int64_t>>>>(
    {
      {"sample_id_0",   { {1008601, 1008602} }},
      {"_indicator_1",  { {0, 0},            }},
      {"feature_vw_0",  { {3001001, 3001002} }},
      {"feature_pg_1",  { {4012301},         }},
    }
  ));
  ASSERT_NE(batch, nullptr);

  // auto st = formater_->InitSchema(
  //         data->schema(), dataset()->hash_features_, dataset()->hash_types_, dataset()->hash_buckets_,
  //         dataset()->dense_features_, dataset()->dense_defaults_, selected_columns);
  column::Status st = formater_->InitSchema(
    batch->schema(),
    /*hash_features=*/{},
    /*hash_types=*/{},
    /*hash_buckets=*/{},
    /*dense_features=*/{},
    /*dense_defaults=*/{},
    /*selected_columns=*/{}
  );
  EXPECT_TRUE(st.ok()) << st.error_message();

  batch = MakeRecordBatch<int64_t>(std::vector<std::pair<std::string, std::vector<std::vector<int64_t>>>>(
    {
      {"sample_id_0",   { {1008601, 1008602}, {1008603}, {1008604, 1008605, 1008606} }},
      {"_indicator_1",  { {0, 0},             {0},       {0, 1, 1} }                  },
      {"feature_vw_0",  { {3001001, 3001002}, {3001003}, {3001004, 3001005, 3001006} }},
      {"feature_pg_1",  { {4012301},          {4012302}, {4012303, 4012304}}          },
    }
  ));

  std::vector<std::shared_ptr<arrow::RecordBatch>> formated_data;
  st = formater_->FormatSample(batch, &formated_data);
  EXPECT_TRUE(st.ok()) << st.error_message();
  EXPECT_EQ(formated_data.size(), size_t(2)) << " 压缩表格式下, formated_data 应该是 2 个group";

  std::vector<Tensor> out_tensors;
  st = formater_->FlatConvert(formated_data, &out_tensors);
  EXPECT_TRUE(st.ok()) << st.error_message();
  // out_tensors content:
  // out_tensors[0]: [,3001001,3001002,3001003,3001004,3001005,3001006]
  // out_tensors[1]: [,1008601,1008602,1008603,1008604,1008605,1008606]
  // out_tensors[2]: [,0,0,1,2,3,3]
  // out_tensors[3]: [,4012301,4012302,4012303,4012304]
  EXPECT_EQ(out_tensors.size(), size_t(4)) << " 压缩表格式下, out_tensors 应该是 4 个";

  // 注意 out_tensors[0~3] 受InitSchema阶段遍历顺序影响, 不永远固定, 此处仅为当前架构、表和arrow迭代器行为的顺序. 以后可能调整变化
  EXPECT_EQ(out_tensors[0].Shape()[0], 6) << " 压缩表格式下, out_tensors[0] sample_id_0 应该是 6 个";
  EXPECT_EQ(out_tensors[1].Shape()[0], 6) << " 压缩表格式下, out_tensors[1] feature_vw_0 应该是 6 个";
  EXPECT_EQ(out_tensors[2].Shape()[0], 6) << " 压缩表格式下, out_tensors[2] _indicator_1 应该是 6 个";
  EXPECT_EQ(out_tensors[3].Shape()[0], 4) << " 压缩表格式下, out_tensors[3] feature_pg_1 应该是 4 个";

}


}  // namespace
}  // namespace dataset
}  // namespace column
