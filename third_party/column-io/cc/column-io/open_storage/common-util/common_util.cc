#include "common_util.h"
#include <sstream>

namespace xdl
{
namespace paiio
{
namespace third_party
{
namespace common_util
{

std::vector<std::string> StrSplit(const std::string &text, const std::string &sepStr, bool ignoreEmpty) {
  std::vector<std::string> vec;
  std::string str(text);
  std::string sep(sepStr);
  size_t n = 0, old = 0;

  while (n != std::string::npos) {
    n = str.find(sep, n);

    if (n != std::string::npos) {
      if (!ignoreEmpty || n != old)
        vec.push_back(str.substr(old, n - old));

      n += sep.length();
      old = n;
    }
  }

  if (!ignoreEmpty || old < str.length()) {
    vec.push_back(str.substr(old, str.length() - old));
  }

  return std::move(vec);
}

std::string strip(const std::string& str) {
    const char* ws = " \t\n\r\f\v";
    size_t start = str.find_first_not_of(ws);
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(ws);
    return str.substr(start, end - start + 1);
}


std::vector<std::string> FilterEmptyStr(const std::vector<std::string>& origin_str_vec) {
  std::vector<std::string> ret;
  for (const auto & str: origin_str_vec) {
    if (str.size() != 0) {
      ret.push_back(str);
    }
  }
  return std::move(ret);
}

std::string JoinStr(const std::string& str_vec, const std::string& sep) {
  std::stringstream buf;
  for (int i = 0; i < str_vec.size(); i++) {
    buf << str_vec[i];
    if (i != (str_vec.size()-1)) {
      buf << sep;
    }
  }
  return buf.str();
}

}
}
}
}

