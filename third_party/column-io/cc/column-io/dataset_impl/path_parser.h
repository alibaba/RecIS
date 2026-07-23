#ifndef COLUMN_IO_CC_COLUMN_IO_DATASET_IMPL_H_PATH_PARSER_H_
#define COLUMN_IO_CC_COLUMN_IO_DATASET_IMPL_H_PATH_PARSER_H_

#include <string>

namespace column {
namespace dataset {

// 解析 odps URL，例如:
//   "odps://prj/tables/table_name/ds=20251231?start=1&end=10"
void ParseOdpsUrl(const std::string& filepath,
                  std::string& project, std::string& table);

// 兼容 tunnel 风格的表名（"proj.table" 或 "cluster.proj.table"），
void ParseTunnelTableFormat(const std::string& filepath,
                            std::string& project, std::string& table);

}  // namespace dataset
}  // namespace column

#endif  // COLUMN_IO_CC_COLUMN_IO_DATASET_IMPL_H_PATH_PARSER_H_
