#include "column-io/dataset_impl/path_parser.h"

#include <string_view>
#include <vector>

#include "absl/log/log.h"

namespace column {
namespace dataset {

void ParseOdpsUrl(const std::string& filepath,
                  std::string& project, std::string& table) {
  size_t scheme_end = filepath.find("://");
  size_t tables_pos = filepath.find("/tables/");
  if (scheme_end == std::string::npos || tables_pos == std::string::npos) {
    return;
  }
  project = filepath.substr(scheme_end + 3, tables_pos - scheme_end - 3);
  size_t table_start = tables_pos + 8;  // strlen("/tables/")
  size_t table_end = filepath.find_first_of("/?", table_start);
  table = (table_end == std::string::npos)
              ? filepath.substr(table_start)
              : filepath.substr(table_start, table_end - table_start);
}

void ParseTunnelTableFormat(const std::string& filepath,
                            std::string& project, std::string& table) {
  auto strip = [](std::string_view sv) -> std::string_view {
    constexpr std::string_view kWs = " \t\n\r\v\f";
    size_t b = sv.find_first_not_of(kWs);
    if (b == std::string_view::npos) return {};
    return sv.substr(b, sv.find_last_not_of(kWs) - b + 1);
  };

  std::vector<std::string> parts;
  std::string_view table_view(table);
  for (size_t start = 0; start <= table_view.size();) {
    size_t end = table_view.find('.', start);
    if (end == std::string_view::npos) end = table_view.size();
    parts.emplace_back(strip(table_view.substr(start, end - start)));
    start = end + 1;
  }

  if (parts.size() == 1) {
    table = parts[0];
  } else if (parts.size() == 2) {
    if (!parts[0].empty()) {
      project = parts[0];
    }
    if (!parts[1].empty()) {
      table = parts[1];
    }
  } else if (parts.size() == 3) {
    if (!parts[0].empty()) {
      project = parts[0];
    }
    if (!parts[2].empty()) {
      table = parts[2];
    }
  } else {
    // 监控辅助路径，不抛异常以免拖挂数据流；保留 ParseOdpsUrl 已经填好的 project/table。
    LOG(WARNING) << "Unrecognized tunnel table format, keep project=" << project
                 << " table=" << table << ", filepath=" << filepath;
  }
}

}  // namespace dataset
}  // namespace column
