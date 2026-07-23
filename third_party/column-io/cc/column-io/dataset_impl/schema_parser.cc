#include "column-io/dataset_impl/schema_parser.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
namespace column {
namespace dataset {
std::string SchemaParser::ExtractElementType(const std::shared_ptr<arrow::DataType>& type) {
  switch (type->id()) {
    case arrow::Type::LIST: {
      auto list_type = std::static_pointer_cast<arrow::ListType>(type);
      return ExtractElementType(list_type->value_type());
    }
    case arrow::Type::LARGE_LIST: {
      auto list_type = std::static_pointer_cast<arrow::LargeListType>(type);
      return ExtractElementType(list_type->value_type());
    }
    case arrow::Type::STRUCT: {
      auto struct_type = std::static_pointer_cast<arrow::StructType>(type);

      if (struct_type->num_fields() == 2 &&
          struct_type->field(0)->name() == "k" &&
          struct_type->field(1)->name() == "v") {
        return "{k:" + ExtractElementType(struct_type->field(0)->type()) +
               ", v:" + ExtractElementType(struct_type->field(1)->type()) + "}";
      }

      std::string s = "struct(";
      for (int i = 0; i < struct_type->num_fields(); i++) {
        if (i > 0) s += ",";
        s += struct_type->field(i)->name() + ":" +
             ExtractElementType(struct_type->field(i)->type());
      }
      s += ")";
      return s;
    }
    default:
      return type->ToString();
  }
}

std::string SchemaParser::BuildSchemaJson(const std::shared_ptr<arrow::Schema>& schema, bool is_compressed) {
  rapidjson::Document doc;
  doc.SetObject();
  auto& allocator = doc.GetAllocator();

  for (const auto& field : schema->fields()) {
    std::string type_str = ExtractElementType(field->type());
    std::string field_name;
    if (is_compressed) {
      size_t pos = field->name().find_last_of("_");
      field_name = field->name().substr(0, pos);
    } else {
      field_name = field->name();
    }
    rapidjson::Value key(field_name.c_str(), allocator);
    rapidjson::Value val(type_str.c_str(), allocator);
    doc.AddMember(key, val, allocator);
  }

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);

  return buffer.GetString();
}




std::tuple<
    std::vector<std::string>,
    std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>,
	std::string
		>
SchemaParser::ParseSchema(
    const std::vector<std::string> &paths,
    bool is_compressed,
    const std::unordered_set<std::string> &selected_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<Tensor> &dense_defaults) {

  std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>
      output_schema;
  std::vector<std::string> input_columns;
  std::string row_mode = std::getenv("ODPS_DATASET_ROW_MODE") ? std::getenv("ODPS_DATASET_ROW_MODE") : "0";
  bool row_mode_with_null = (row_mode == "1") ;
  std::string json_str;
  for (size_t i = 0; i < paths.size(); ++i) {
    std::shared_ptr<arrow::RecordBatch> data;
    auto st = rb_reader_(paths[i], selected_columns, dense_columns,
                         is_compressed, &data);
    CHECK(st.ok()) << st.error_message();
    std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>
        schema;
    std::vector<std::string> one_input_columns;
    st = ParseSchemaCommon(data, selected_columns, hash_features, hash_types, hash_buckets,
                           dense_columns, dense_defaults, is_compressed, false, &schema,
                           &one_input_columns, row_mode_with_null);
    CHECK(st.ok()) << st.error_message();
    if (!output_schema.empty()) {
      if (output_schema != schema) {
        CHECK(false) << "schema from path: " << paths[i]
                     << ", not the same as before";
      }
      if (input_columns != one_input_columns) {
        CHECK(false) << "input column from path: " << paths[i]
                     << ", not the same as before";
      }
    } else {
      output_schema = std::move(schema);
      input_columns = std::move(one_input_columns);
	  auto schema = data->schema();
	  json_str = BuildSchemaJson(schema, is_compressed);

      auto GetEnvInt = [](const char* name, int defval) -> int {
          const char* v = std::getenv(name);
          if (!v || !*v) return defval;
          try {
              return std::stoi(std::string(v));
          } catch (...) {
              return defval;
          }
      };

      // 用法
      int use_xrec = GetEnvInt("USE_XREC", 0);
      int is_recis = GetEnvInt("IS_RECIS", 0);
      if (use_xrec == 1 || is_recis == 1) {
          // LOG(INFO) << "json_str is [" << json_str << "]"; //TODO: use a internal log library. not absl log
          auto time_chrono = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
          char time_buf[64];
          std::strftime(time_buf, sizeof(time_buf), "%Y%m%d %H%M%S", std::localtime(&time_chrono));
          printf("[%s] [WARN] [%s:%d] json_str is: %s \n", time_buf, __FILE__, __LINE__, json_str.c_str());
      }
    }
  }
  return std::make_tuple(input_columns, output_schema, json_str);
}


std::pair<
    std::vector<std::string>,
    std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>>
SchemaParser::ParseSchemaByRows(
    const std::vector<std::string> &paths,
    bool is_compressed,
    const std::vector<std::string> &selected_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_columns,
    const std::vector<Tensor> &dense_defaults) {
    
    std::vector<std::string> input_columns;
    std::vector<std::map<std::string, std::vector<std::vector<std::string>>>> output_schema;
    
    //std::string row_mode = std::getenv("ODPS_DATASET_ROW_MODE") ? std::getenv("ODPS_DATASET_ROW_MODE") : "0";
    //bool row_mode_with_null = (row_mode == "1") ;
    bool row_mode_with_null = true;
    std::shared_ptr<arrow::RecordBatch> data;
    const std::unordered_set<std::string> selected_columns_set(selected_columns.begin(), selected_columns.end());
    for (size_t i = 0; i < paths.size(); ++i) {
        data.reset();
        auto st = rb_reader_(paths[i], selected_columns_set, dense_columns, is_compressed, &data);
        CHECK(st.ok()) << st.error_message();
        std::vector<std::string> batch_columns;
        std::vector<std::map<std::string, std::vector<std::vector<std::string>>>> batch_schema;
        st = ParseSchemaCommon(data, selected_columns_set, hash_features, hash_types, hash_buckets,
                               dense_columns, dense_defaults, is_compressed, false, &batch_schema,
                               &batch_columns, row_mode_with_null);
        CHECK(st.ok()) << st.error_message();
        if (!output_schema.empty()) {
        if (output_schema != batch_schema) {
            CHECK(false) << "batch_schema from path: " << paths[i] << ", not the same as before";
        }
        if (input_columns != batch_columns) {
            CHECK(false) << "input_column from path: " << paths[i] << ", not the same as before";
        }
        } else {
        output_schema = std::move(batch_schema);
        input_columns = std::move(batch_columns);
        }
    }
    // Conver input_columns to original order
    // 目标: 以指定的selected_columns(P1)或数据源data->schema()中的列顺序(P2)对input_columns(P3)进行重排列
    std::unordered_map<std::string, size_t> orderMap;
    size_t order = 0;
    if( selected_columns.empty() ){ // sort input_columns by data->schema()->fields()
        for (const auto& field : data->schema()->fields())  orderMap[field->name()] = order++;
    }else{ // sort input_columns by selected_columns
        for (const auto& col_name : selected_columns)   orderMap[col_name] = order++;
    }
    data.reset();
    std::sort(input_columns.begin(), input_columns.end(), 
            [&orderMap](const std::string& a, const std::string& b) {
                return orderMap.at(a) < orderMap.at(b);
            }
    );
    return std::make_pair(input_columns, output_schema);
}


Status SchemaParser::ParseSchemaCommon(
    std::shared_ptr<arrow::RecordBatch> &data,
    const std::unordered_set<std::string> &selected_columns,
    const std::vector<std::string> &hash_features,
    const std::vector<std::string> &hash_types,
    const std::vector<int64_t> &hash_buckets,
    const std::vector<std::string> &dense_features,
    const std::vector<Tensor> &dense_defaults,
    bool is_compressed,
    bool is_large_list,
    std::vector<std::map<std::string, std::vector<std::vector<std::string>>>>
        *output_schema,
    std::vector<std::string> *input_columns,
    bool with_null) {
  // parse schema
  auto formater =
      ColumnDataFormater::GetColumnDataFormater(is_compressed, is_large_list, with_null);
  auto st = formater->InitSchema(data->schema(), hash_features, hash_types, hash_buckets,
                                 dense_features, dense_defaults, selected_columns);
  if (!st.ok()) {
    return Status::InvalidArgument("fail to init formater, error info: ",
                                   st.error_message());
  }
  std::vector<std::shared_ptr<arrow::RecordBatch>> formated_data;
  st = formater->FormatSample(data, &formated_data);
  if (!st.ok()) {
    return Status::Internal("fail to format sample, error info: ",
                            st.error_message());
  }
  std::vector<std::map<std::string, std::vector<std::vector<Tensor>>>>
      out_tensors;
  st = formater->Convert(formated_data, &out_tensors);
  if (!st.ok()) {
    return Status::Internal("fail to convert sample, error info: ",
                            st.error_message());
  }
  // assemble tensor infos
  std::stringstream tensor_info;
  for (size_t i = 0; i < out_tensors.size(); ++i) {
    auto &map = out_tensors[i];
    output_schema->resize(i + 1);
    auto &out_map = (*output_schema)[i];
    for (auto &item : map) {
      auto &feature_tensors = out_map[item.first];
      for (size_t j = 0; j < item.second.size(); ++j) {
        feature_tensors.resize(j + 1);
        auto &one_vec = feature_tensors[j];
        for (auto &tensor : item.second[j]) {
          one_vec.emplace_back("Placeholder");
        }
      }
    }
  }
  formater->GetInputColumns(input_columns);
  return Status::OK();
}

/* Make: 创建一个解析列式结构的结构解析器
 * @fn: 读取一批列式结构的纯读函数
 * @return: 输出列式结构的解析器SchemaParser
*/ 
std::unique_ptr<SchemaParser> SchemaParser::Make(RecordBatchReaderFn fn) {
  return std::unique_ptr<SchemaParser>(new SchemaParser(fn));
}
/* MakeByRows: 创建一个解析输式结构的结构解析器
 * @fn: 读取一批列式结构的纯读函数 *注意* RecordBatch始终为列结构, 因此fn不存在天然的行结构输出
 * @return: 输出列式结构的解析器SchemaParser
*/ 
std::unique_ptr<SchemaParser> SchemaParser::MakeByRows(RecordBatchReaderFn fn) {
  return std::unique_ptr<SchemaParser>(new SchemaParser(fn));
}
} // namespace dataset
} // namespace column
