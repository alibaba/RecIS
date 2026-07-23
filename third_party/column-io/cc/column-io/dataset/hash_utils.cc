#include "hash_utils.h"

#include <stdlib.h>
#include <string.h>

#include <iostream>

namespace column {
namespace dataset {

static const int64_t kint64max = ((int64_t)0x7FFFFFFFFFFFFFFFll);

int64_t StringToHash(const char *data, size_t n, const std::string hash_type, int64_t hash_bucket)
{
  // init hash func
  uint64_t (*hash_func)(const char *, size_t);
  if (hash_type == "murmur") {
    hash_func = MurMurHash64;
  } else {
    hash_func = ::util::Fingerprint64;
  }
  uint64_t hash_value = hash_func(data, n);
  int64_t out_value;
  if (hash_type == "ev_farm" && hash_bucket > 0) {
    out_value = (hash_value & kint64max) % hash_bucket;
  } else if (hash_type == "ev_farm" && hash_bucket == 0) {
    out_value = hash_value & kint64max;
  } else if (hash_bucket > 0) {
    out_value = hash_value % hash_bucket;
  } else {
    out_value = hash_value;
  }

  return out_value;
}

uint64_t MurMurHash64(const char* data, size_t n, uint64_t seed) {
  const uint64_t m = 0xc6a4a7935bd1e995;
  const int r = 47;

  uint64_t h = seed ^ (n * m);

  while (n >= 8) {
    uint64_t k = DecodeFixed64(data);
    data += 8;
    n -= 8;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  switch (n) {
    case 7:
      h ^= ByteAs64(data[6]) << 48;
    case 6:
      h ^= ByteAs64(data[5]) << 40;
    case 5:
      h ^= ByteAs64(data[4]) << 32;
    case 4:
      h ^= ByteAs64(data[3]) << 24;
    case 3:
      h ^= ByteAs64(data[2]) << 16;
    case 2:
      h ^= ByteAs64(data[1]) << 8;
    case 1:
      h ^= ByteAs64(data[0]);
      h *= m;
  }

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}

uint64_t ByteAs64(char c) { return static_cast<uint64_t>(c) & 0xff; }

uint64_t DecodeFixed64(const char* ptr) {
  uint64_t result;
  memcpy(&result, ptr, sizeof(result));
  return result;
}

}  // namespace utils
}  // namespace column

