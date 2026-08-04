#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "fused_pack.h"

__global__ void fused_valid_ids_kernel(const int64_t* __restrict__ ids,
                                       int64_t* __restrict__ output,
                                       const int64_t* __restrict__ sorted_ptrs,
                                       const int* __restrict__ sorted_sizes,
                                       int64_t default_value, int world_size,
                                       int Q) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= Q) return;

  int64_t id = ids[q];
  bool found = false;
  for (int r = 0; r < world_size && !found; r++) {
    const int64_t* sorted = reinterpret_cast<const int64_t*>(sorted_ptrs[r]);
    int n = sorted_sizes[r];
    if (n == 0) continue;

    int lo = 0, hi = n - 1;
    while (lo <= hi) {
      int mid = lo + (hi - lo) / 2;
      int64_t val = sorted[mid];
      if (val == id) {
        found = true;
        break;
      }
      if (val < id)
        lo = mid + 1;
      else
        hi = mid - 1;
    }
  }

  output[q] = found ? id : default_value;
}

torch::Tensor fused_valid_sample_ids(torch::Tensor ids,
                                     std::vector<torch::Tensor> sorted_ids_list,
                                     int64_t default_value) {
  int Q = ids.size(0);
  int world_size = sorted_ids_list.size();
  auto output = ids.clone();
  if (Q == 0 || world_size == 0) return output;

  auto ptrs_cpu = torch::empty({world_size}, torch::dtype(torch::kInt64));
  auto sizes_cpu = torch::empty({world_size}, torch::dtype(torch::kInt32));
  for (int r = 0; r < world_size; r++) {
    ptrs_cpu[r] =
        reinterpret_cast<int64_t>(sorted_ids_list[r].data_ptr<int64_t>());
    sizes_cpu[r] = static_cast<int>(sorted_ids_list[r].numel());
  }

  auto ptrs = ptrs_cpu.to(ids.device());
  auto sizes = sizes_cpu.to(ids.device());
  // 跟随调用方的 current stream（torch.cuda.stream 上下文），与内部 ATen
  // 操作同流
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_valid_ids_kernel<<<(Q + 255) / 256, 256, 0, stream>>>(
      ids.data_ptr<int64_t>(), output.data_ptr<int64_t>(),
      ptrs.data_ptr<int64_t>(), sizes.data_ptr<int>(), default_value,
      world_size, Q);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("HAS_CUDA_KERNEL") = true;
  // 所有入口释放 GIL：负采样在 prefetch 子线程执行，持 GIL 阻塞会卡住主线程的
  // Python 层 autograd backward（2026-07-29 xdl-71dfd7e1cde2
  // 死锁链的放大因子）。
  m.def(
      "fused_valid_sample_ids", &fused_valid_sample_ids,
      "Fused valid sample IDs: replace not-found with default in single kernel",
      py::call_guard<py::gil_scoped_release>());
  m.def("fused_route_ids", &fused_route_ids,
        "Fused route IDs: route + replace not-found in single kernel",
        py::call_guard<py::gil_scoped_release>());
  m.def("unified_batched_gather", &unified_batched_gather,
        "Unified batched gather for all entries",
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_serialize", &fused_serialize,
        "Fused serialize: gathered entries into flat byte buffers",
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_deserialize_fixed", &fused_deserialize_fixed,
        "Fused deserialize Phase 1: parse lengths and scatter them to query "
        "order",
        py::call_guard<py::gil_scoped_release>());
  m.def(
      "fused_deserialize_values", &fused_deserialize_values,
      "Fused deserialize Phase 2: parse values and scatter them to query order",
      py::call_guard<py::gil_scoped_release>());
}
