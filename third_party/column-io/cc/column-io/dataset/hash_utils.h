#include "farmhash.h"

#include <iostream>
#include <vector>

namespace column {
namespace dataset {

int64_t StringToHash(const char *data, size_t n, const std::string hash_type, int64_t hash_bucket);

uint64_t MurMurHash64(const char* data, size_t n, uint64_t seed);

inline uint64_t MurMurHash64(const char* data, size_t n) {
  return MurMurHash64(data, n, 0);
}

inline uint64_t MurMurHash64(const std::string& str) {
  return MurMurHash64(str.data(), str.size());
}

inline uint64_t FarmHash64(const std::string& str) {
  return ::util::Fingerprint64(str.data(), str.size());
}

inline uint64_t ByteAs64(char c);

inline uint64_t DecodeFixed64(const char* ptr);

}  // namespace utils
}  // namespace column

