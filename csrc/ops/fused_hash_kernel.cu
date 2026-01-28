// This file contains code snippets from multiple open-source projects.
// Each section is attributed below with its original license and source.
//
// ------------------------------------------------------------------------
// 1. MurMurHash
//    Copyright 2015 The TensorFlow Authors
//    Licensed under the Apache License, Version 2.0
//    Source: https://github.com/tensorflow/tensorflow
//    Modifications: adapted to cuda
//
// 2. FarmHash
//    Copyright (c) 2014 Google, Inc
//    Licensed under MIT license
//    Source: https://github.com/google/farmhash
//    Modifications: adapted to cuda

#include <ATen/cuda/CUDAContext.h>

#include "cuda/cuda_param.cuh"
#include "cuda/element_wise_kernel.cuh"
#include "cuda/utils.cuh"
#include "cuda_runtime.h"
#include "ops/fused_hash.h"

namespace recis {
namespace functional {

namespace murmurhash {

// MurMurHash implementation from https://github.com/tensorflow/tensorflow.
// Modifications: adapted to cuda.
// ------------------------------------------------------------------------
// Copyright 2015 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

__device__ uint64_t DecodeFixed64(const char* ptr) {
  uint64_t result;
  memcpy(&result, ptr, sizeof(result));
  return result;
}

__device__ uint64_t ByteAs64(char c) { return static_cast<uint64_t>(c) & 0xff; }

__device__ uint64_t MurMurHash64(const char* data, size_t n, uint64_t seed) {
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

template <typename scalar_t>
struct MurmurhashFactory {
  __device__ int64_t operator()(scalar_t* offsets, int8_t* inputs,
                                int64_t index) {
    scalar_t offset = offsets[index];
    scalar_t len = offsets[index + 1] - offset;
    char* s = reinterpret_cast<char*>(inputs + offset);
    return MurMurHash64(s, len, 0);
  }
};

}  // namespace murmurhash

namespace farmhash {
// Farmhash implementation from https://github.com/google/farmhash
// Modifications: adapted to cuda.
// ------------------------------------------------------------------------
// Copyright (c) 2014 Google, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

template <typename T1, typename T2>
struct pair {
  T1 first;
  T2 second;
  __device__ pair() {}
  __device__ pair(const T1& first, const T2& second)
      : first(first), second(second) {}
};

template <typename T1, typename T2>
__device__ pair<T1, T2> make_pair(const T1& a, const T2& b) {
  return pair<T1, T2>(a, b);
}

template <typename T>
__device__ void swap(T& a, T& b) {
  T tmp = a;
  a = b;
  b = tmp;
}

typedef pair<uint64_t, uint64_t> uint128_t;
__device__ inline uint64_t Uint128Low64(const uint128_t x) { return x.first; }
__device__ inline uint64_t Uint128High64(const uint128_t x) { return x.second; }
__device__ inline uint128_t Uint128(uint64_t lo, uint64_t hi) {
  return uint128_t(lo, hi);
}

inline __device__ uint64_t Fetch64(const char* p) {
  uint64_t result;
  memcpy(&result, p, sizeof(result));
  return result;
}

inline __device__ uint32_t Fetch32(const char* p) {
  uint32_t result;
  memcpy(&result, p, sizeof(result));
  return result;
}

// inline __device__ uint32_t Bswap32(uint32_t val) { return bswap_32(val); }
// inline __device__ uint64_t Bswap64(uint64_t val) { return bswap_64(val); }

// FARMHASH PORTABILITY LAYER: bitwise rot

inline __device__ uint32_t BasicRotate32(uint32_t val, int shift) {
  // Avoid shifting by 32: doing so yields an undefined result.
  return shift == 0 ? val : ((val >> shift) | (val << (32 - shift)));
}

inline __device__ uint64_t BasicRotate64(uint64_t val, int shift) {
  // Avoid shifting by 64: doing so yields an undefined result.
  return shift == 0 ? val : ((val >> shift) | (val << (64 - shift)));
}

inline __device__ uint32_t Rotate32(uint32_t val, int shift) {
  return BasicRotate32(val, shift);
}
inline __device__ uint64_t Rotate64(uint64_t val, int shift) {
  return BasicRotate64(val, shift);
}

// Some primes between 2^63 and 2^64 for various uses.
static const uint64_t k0 = 0xc3a5c85c97cb3127ULL;
static const uint64_t k1 = 0xb492b66fbe98f273ULL;
static const uint64_t k2 = 0x9ae16a3b2f90404fULL;
#undef Fetch
#define Fetch Fetch64

#undef Rotate
#define Rotate Rotate64

inline __device__ uint64_t ShiftMix(uint64_t val) { return val ^ (val >> 47); }

// Hash 128 input bits down to 64 bits of output.
// This is intended to be a reasonably good hash function.
// May change from time to time, may differ on different platforms, may differ
// depending on NDEBUG.
__device__ inline uint64_t Hash128to64(uint128_t x) {
  // Murmur-inspired hashing.
  const uint64_t kMul = 0x9ddfea08eb382d69ULL;
  uint64_t a = (Uint128Low64(x) ^ Uint128High64(x)) * kMul;
  a ^= (a >> 47);
  uint64_t b = (Uint128High64(x) ^ a) * kMul;
  b ^= (b >> 47);
  b *= kMul;
  return b;
}

inline __device__ uint64_t HashLen16(uint64_t u, uint64_t v) {
  return Hash128to64(Uint128(u, v));
}

inline __device__ uint64_t HashLen16(uint64_t u, uint64_t v, uint64_t mul) {
  // Murmur-inspired hashing.
  uint64_t a = (u ^ v) * mul;
  a ^= (a >> 47);
  uint64_t b = (v ^ a) * mul;
  b ^= (b >> 47);
  b *= mul;
  return b;
}

inline __device__ uint64_t HashLen0to16(const char* s, size_t len) {
  if (len >= 8) {
    uint64_t mul = k2 + len * 2;
    uint64_t a = Fetch(s) + k2;
    uint64_t b = Fetch(s + len - 8);
    uint64_t c = Rotate(b, 37) * mul + a;
    uint64_t d = (Rotate(a, 25) + b) * mul;
    return HashLen16(c, d, mul);
  }
  if (len >= 4) {
    uint64_t mul = k2 + len * 2;
    uint64_t a = Fetch32(s);
    return HashLen16(len + (a << 3), Fetch32(s + len - 4), mul);
  }
  if (len > 0) {
    uint8_t a = s[0];
    uint8_t b = s[len >> 1];
    uint8_t c = s[len - 1];
    uint32_t y = static_cast<uint32_t>(a) + (static_cast<uint32_t>(b) << 8);
    uint32_t z = len + (static_cast<uint32_t>(c) << 2);
    return ShiftMix(y * k2 ^ z * k0) * k2;
  }
  return k2;
}

// This probably works well for 16-byte strings as well, but it may be overkill
// in that case.
inline __device__ uint64_t HashLen17to32(const char* s, size_t len) {
  uint64_t mul = k2 + len * 2;
  uint64_t a = Fetch(s) * k1;
  uint64_t b = Fetch(s + 8);
  uint64_t c = Fetch(s + len - 8) * mul;
  uint64_t d = Fetch(s + len - 16) * k2;
  return HashLen16(Rotate(a + b, 43) + Rotate(c, 30) + d,
                   a + Rotate(b + k2, 18) + c, mul);
}

// Return a 16-byte hash for 48 bytes.  Quick and dirty.
// Callers do best to use "random-looking" values for a and b.
inline __device__ pair<uint64_t, uint64_t> WeakHashLen32WithSeeds(
    uint64_t w, uint64_t x, uint64_t y, uint64_t z, uint64_t a, uint64_t b) {
  a += w;
  b = Rotate(b + a + z, 21);
  uint64_t c = a;
  a += x;
  a += y;
  b += Rotate(a, 44);
  return make_pair(a + z, b + c);
}

// Return a 16-byte hash for s[0] ... s[31], a, and b.  Quick and dirty.
inline __device__ pair<uint64_t, uint64_t> WeakHashLen32WithSeeds(const char* s,
                                                                  uint64_t a,
                                                                  uint64_t b) {
  return WeakHashLen32WithSeeds(Fetch(s), Fetch(s + 8), Fetch(s + 16),
                                Fetch(s + 24), a, b);
}

// Return an 8-byte hash for 33 to 64 bytes.
inline __device__ uint64_t HashLen33to64(const char* s, size_t len) {
  uint64_t mul = k2 + len * 2;
  uint64_t a = Fetch(s) * k2;
  uint64_t b = Fetch(s + 8);
  uint64_t c = Fetch(s + len - 8) * mul;
  uint64_t d = Fetch(s + len - 16) * k2;
  uint64_t y = Rotate(a + b, 43) + Rotate(c, 30) + d;
  uint64_t z = HashLen16(y, a + Rotate(b + k2, 18) + c, mul);
  uint64_t e = Fetch(s + 16) * mul;
  uint64_t f = Fetch(s + 24);
  uint64_t g = (y + Fetch(s + len - 32)) * mul;
  uint64_t h = (z + Fetch(s + len - 24)) * mul;
  return HashLen16(Rotate(e + f, 43) + Rotate(g, 30) + h,
                   e + Rotate(f + a, 18) + g, mul);
}

__device__ uint64_t Hash64(const char* s, size_t len) {
  const uint64_t seed = 81;
  if (len <= 32) {
    if (len <= 16) {
      return HashLen0to16(s, len);
    } else {
      return HashLen17to32(s, len);
    }
  } else if (len <= 64) {
    return HashLen33to64(s, len);
  }

  // For strings over 64 bytes we loop.  Internal state consists of
  // 56 bytes: v, w, x, y, and z.
  uint64_t x = seed;
  uint64_t y = seed * k1 + 113;
  uint64_t z = ShiftMix(y * k2 + 113) * k2;
  pair<uint64_t, uint64_t> v(0, 0);
  pair<uint64_t, uint64_t> w(0, 0);
  x = x * k2 + Fetch(s);

  // Set end so that after the loop we have 1 to 64 bytes left to process.
  const char* end = s + ((len - 1) / 64) * 64;
  const char* last64 = end + ((len - 1) & 63) - 63;
  do {
    x = Rotate(x + y + v.first + Fetch(s + 8), 37) * k1;
    y = Rotate(y + v.second + Fetch(s + 48), 42) * k1;
    x ^= w.second;
    y += v.first + Fetch(s + 40);
    z = Rotate(z + w.first, 33) * k1;
    v = WeakHashLen32WithSeeds(s, v.second * k1, x + w.first);
    w = WeakHashLen32WithSeeds(s + 32, z + w.second, y + Fetch(s + 16));
    swap(z, x);
    s += 64;
  } while (s != end);
  uint64_t mul = k1 + ((z & 0xff) << 1);
  // Make s point to the last 64 bytes of input.
  s = last64;
  w.first += ((len - 1) & 63);
  v.first += w.first;
  w.first += v.first;
  x = Rotate(x + y + v.first + Fetch(s + 8), 37) * mul;
  y = Rotate(y + v.second + Fetch(s + 48), 42) * mul;
  x ^= w.second * 9;
  y += v.first * 9 + Fetch(s + 40);
  z = Rotate(z + w.first, 33) * mul;
  v = WeakHashLen32WithSeeds(s, v.second * mul, x + w.first);
  w = WeakHashLen32WithSeeds(s + 32, z + w.second, y + Fetch(s + 16));
  swap(z, x);
  return HashLen16(HashLen16(v.first, w.first, mul) + ShiftMix(y) * k0 + z,
                   HashLen16(v.second, w.second, mul) + x, mul);
}

__device__ uint64_t Fingerprint64(const char* s, size_t len) {
  return Hash64(s, len);
}

template <typename scalar_t>
struct FarmhashFactory {
  __device__ int64_t operator()(scalar_t* offsets, int8_t* inputs,
                                int64_t index) {
    scalar_t offset = offsets[index];
    scalar_t len = offsets[index + 1] - offset;
    char* s = reinterpret_cast<char*>(inputs + offset);
    return Fingerprint64(s, len);
  }
};

}  // namespace farmhash

namespace djb2hash {
__device__ uint64_t djb2_dev(const char* s, size_t len) {
  uint64_t hash = 5381;
  for (uint idx = 0; idx < len; ++idx) {
    int c = static_cast<int>(s[idx]);
    hash = ((hash << 5) + hash) + c;
  }
  return hash;
}  // end djb2_dev

template <typename scalar_t>
struct Djb2Factory {
  __device__ uint64_t operator()(scalar_t* offsets, int8_t* inputs,
                                 int64_t index) {
    scalar_t offset = offsets[index];
    scalar_t len = offsets[index + 1] - offset;
    char* s = reinterpret_cast<char*>(inputs + offset);
    return djb2_dev(s, len);
  }
};  // end struct

}  // namespace djb2hash

namespace sdbmhash {
__device__ uint64_t sdbm_dev(const char* s, size_t len) {
  uint64_t hash = 0;
  for (uint idx = 0; idx < len; ++idx) {
    int c = static_cast<int>(s[idx]);
    hash = c + (hash << 6) + (hash << 16) - hash;
  }
  return hash;
}  // end sdbm_dev

template <typename scalar_t>
struct SdbmFactory {
  __device__ uint64_t operator()(scalar_t* offsets, int8_t* inputs,
                                 int64_t index) {
    scalar_t offset = offsets[index];
    scalar_t len = offsets[index + 1] - offset;
    char* s = reinterpret_cast<char*>(inputs + offset);
    return sdbm_dev(s, len);
  }  // end struct
};  // end struct
}  // namespace sdbmhash

using namespace recis::cuda;
template <typename Factory, typename scalar_t>
__global__ void fused_hash_kernel(int8_t** inputs_ptrs,
                                  scalar_t** input_offsets_ptrs,
                                  int64_t** outputs_ptrs, int64_t* sizes,
                                  int64_t N, Factory factory) {
  int64_t vec_id = blockIdx.y;
  int64_t size_local = sizes[vec_id];
  int64_t threads_num = blockDim.x * gridDim.x;
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int64_t index = tid; index < size_local; index += threads_num) {
    outputs_ptrs[vec_id][index] =
        factory(input_offsets_ptrs[vec_id], inputs_ptrs[vec_id], index);
  }
}

template <typename Factory, typename scalar_t>
void fused_hash_launcher(int8_t** d_inputs_ptrs,
                         scalar_t** d_input_offsets_ptrs,
                         int64_t** d_outputs_ptrs, int64_t* sizes, int64_t N,
                         Factory factory, cudaStream_t stream) {
  int64_t sm_count = get_sm_count();
  int64_t max_size = 0;
  for (int64_t i = 0; i < N; ++i) {
    max_size = std::max(max_size, sizes[i]);
  }
  int64_t block_num =
      min(sm_count * 8, (max_size + KBLOCK_SIZE - 1) / KBLOCK_SIZE);
  dim3 grid(block_num, N);
  dim3 block(KBLOCK_SIZE);
  int64_t* d_sizes = cuda_malloc_and_copy<int64_t>(sizes, N, stream);
  fused_hash_kernel<Factory, scalar_t><<<grid, block, 0, stream>>>(
      d_inputs_ptrs, d_input_offsets_ptrs, d_outputs_ptrs, d_sizes, N, factory);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  delete_cuda_ptr(d_sizes);
}

void fused_hash_cuda(const std::vector<torch::Tensor>& inputs,
                     const std::vector<torch::Tensor>& input_offsets,
                     const std::vector<torch::Tensor>& outputs,
                     const std::string& hash_type) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t N = inputs.size();
  if (N == 0) return;
  AT_DISPATCH_INTEGRAL_TYPES(
      input_offsets[0].scalar_type(), "fused_hash_cuda", ([&] {
        std::vector<int64_t> sizes(N);
        CudaVecParam<int8_t*> inputs_ptrs(N, stream);
        CudaVecParam<scalar_t*> input_offsets_ptrs(N, stream);
        CudaVecParam<int64_t*> outputs_ptrs(N, stream);
        for (int64_t i = 0; i < N; ++i) {
          sizes[i] = outputs[i].numel();
          inputs_ptrs[i] = inputs[i].data_ptr<int8_t>();
          input_offsets_ptrs[i] = input_offsets[i].data_ptr<scalar_t>();
          outputs_ptrs[i] = outputs[i].data_ptr<int64_t>();
        }
        int8_t** d_inputs_ptrs = inputs_ptrs.data();
        scalar_t** d_input_offsets_ptrs = input_offsets_ptrs.data();
        int64_t** d_outputs_ptrs = outputs_ptrs.data();

        if (hash_type == "farm") {
          fused_hash_launcher(d_inputs_ptrs, d_input_offsets_ptrs,
                              d_outputs_ptrs, sizes.data(), N,
                              farmhash::FarmhashFactory<scalar_t>(), stream);
        } else if (hash_type == "murmur") {
          fused_hash_launcher(
              d_inputs_ptrs, d_input_offsets_ptrs, d_outputs_ptrs, sizes.data(),
              N, murmurhash::MurmurhashFactory<scalar_t>(), stream);
        } else if (hash_type == "djb2") {
          fused_hash_launcher(d_inputs_ptrs, d_input_offsets_ptrs,
                              d_outputs_ptrs, sizes.data(), N,
                              djb2hash::Djb2Factory<scalar_t>(), stream);
        } else if (hash_type == "sdbm") {
          fused_hash_launcher(d_inputs_ptrs, d_input_offsets_ptrs,
                              d_outputs_ptrs, sizes.data(), N,
                              sdbmhash::SdbmFactory<scalar_t>(), stream);
        } else {
          throw std::runtime_error(std::string("Unsupported hash type: ") +
                                   hash_type);
        }
      }));
}
}  // namespace functional
}  // namespace recis
