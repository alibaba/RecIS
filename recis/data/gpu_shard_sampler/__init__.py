"""GPU 分片负样本采样器。

将负样本表分片到多张 GPU 上，特征常驻本地 GPU HBM，
查找时通过 fused CUDA kernel 路由 + NCCL all_to_all 跨 rank 通信。

接口与 recis.data.local_rpc_data_sampler.LocalRpcDataSampler 一致，
可作为 drop-in 替换。

编译产物为 recis/lib/gpu_shard_kernels.so，随 recis 主包一起构建：
    from recis.lib import gpu_shard_kernels
"""

try:
    from recis.lib import gpu_shard_kernels  # noqa: F401

    _HAS_CUDA_KERNEL = True
except ImportError:
    _HAS_CUDA_KERNEL = False

from recis.data.gpu_shard_sampler.sampler import GpuShardSampler
from recis.data.gpu_shard_sampler.shard import GpuShard, build_cpu_shard


__all__ = ["GpuShard", "build_cpu_shard", "GpuShardSampler"]
