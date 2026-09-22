import os

import torch

from recis.utils.logger import Logger


logger = Logger(__name__)
_SPARSE_GRAD_GROUP_REDUCE_STATS_STEP = 0


def _sparse_grad_group_reduce_pair_grad(
    flat_group_index,
    grad_outputs,
    num_unique,
    group_size,
    group_reduce_by,
):
    active_flat, inverse = torch.unique(
        flat_group_index, return_inverse=True, sorted=False
    )
    pair_grad = torch.zeros(
        [active_flat.numel()] + list(grad_outputs.shape)[1:],
        dtype=grad_outputs.dtype,
        device=grad_outputs.device,
    )
    pair_grad.index_add_(0, inverse, grad_outputs)

    if group_reduce_by == "id":
        counts = torch.zeros(
            [active_flat.numel()],
            dtype=grad_outputs.dtype,
            device=grad_outputs.device,
        )
        ones = torch.ones(
            [inverse.numel()],
            dtype=grad_outputs.dtype,
            device=grad_outputs.device,
        )
        counts.index_add_(0, inverse, ones)
        counts = counts.clamp_min_(1).view(
            [active_flat.numel()] + [1] * (grad_outputs.dim() - 1)
        )
        pair_grad = pair_grad / counts
    elif group_reduce_by == "worker":
        pair_grad = pair_grad / group_size

    pair_id = active_flat % num_unique
    return pair_id, pair_grad


def _sparse_grad_group_reduce_dense(
    index,
    source_group,
    grad_outputs,
    num_unique,
    num_groups,
    group_size,
    group_reduce_by,
):
    flat_group_index = source_group * num_unique + index
    grouped_shape = [num_groups * num_unique] + list(grad_outputs.shape)[1:]
    group_grad = torch.zeros(
        grouped_shape, dtype=grad_outputs.dtype, device=grad_outputs.device
    )
    group_grad.index_add_(0, flat_group_index, grad_outputs)

    if group_reduce_by == "id":
        counts = torch.zeros(
            [num_groups * num_unique],
            dtype=grad_outputs.dtype,
            device=grad_outputs.device,
        )
        ones = torch.ones(
            [flat_group_index.numel()],
            dtype=grad_outputs.dtype,
            device=grad_outputs.device,
        )
        counts.index_add_(0, flat_group_index, ones)
        counts = counts.clamp_min_(1).view(
            [num_groups * num_unique] + [1] * (grad_outputs.dim() - 1)
        )
        group_grad = group_grad / counts
    elif group_reduce_by == "worker":
        group_grad = group_grad / group_size

    group_grad = group_grad.view(
        [num_groups, num_unique] + list(grad_outputs.shape)[1:]
    )
    grad_sum = group_grad.sum(dim=0)
    grad_sq_sum = (group_grad * group_grad).sum(dim=0)
    return grad_sum, grad_sq_sum


def _sparse_grad_group_reduce_compact(
    index,
    source_group,
    grad_outputs,
    num_unique,
    group_size,
    group_reduce_by,
):
    flat_group_index = source_group * num_unique + index
    pair_id, pair_grad = _sparse_grad_group_reduce_pair_grad(
        flat_group_index,
        grad_outputs,
        num_unique,
        group_size,
        group_reduce_by,
    )
    grad_shape = [num_unique] + list(grad_outputs.shape)[1:]
    grad_sum = torch.zeros(
        grad_shape, dtype=grad_outputs.dtype, device=grad_outputs.device
    )
    grad_sq_sum = torch.zeros(
        grad_shape, dtype=grad_outputs.dtype, device=grad_outputs.device
    )
    grad_sum.index_add_(0, pair_id, pair_grad)
    grad_sq_sum.index_add_(0, pair_id, pair_grad * pair_grad)
    return grad_sum, grad_sq_sum


def _sparse_grad_group_reduce_chunk_compact(
    index,
    source_group,
    grad_outputs,
    num_unique,
    num_groups,
    group_size,
    group_reduce_by,
    chunk_groups,
):
    grad_shape = [num_unique] + list(grad_outputs.shape)[1:]
    grad_sum = torch.zeros(
        grad_shape, dtype=grad_outputs.dtype, device=grad_outputs.device
    )
    grad_sq_sum = torch.zeros(
        grad_shape, dtype=grad_outputs.dtype, device=grad_outputs.device
    )

    for group_begin in range(0, num_groups, chunk_groups):
        group_end = min(group_begin + chunk_groups, num_groups)
        in_chunk = (source_group >= group_begin) & (source_group < group_end)
        if not torch.any(in_chunk):
            continue

        chunk_index = index[in_chunk]
        chunk_group = source_group[in_chunk] - group_begin
        chunk_grad_outputs = grad_outputs[in_chunk]
        chunk_flat_group_index = chunk_group * num_unique + chunk_index
        pair_id, pair_grad = _sparse_grad_group_reduce_pair_grad(
            chunk_flat_group_index,
            chunk_grad_outputs,
            num_unique,
            group_size,
            group_reduce_by,
        )
        grad_sum.index_add_(0, pair_id, pair_grad)
        grad_sq_sum.index_add_(0, pair_id, pair_grad * pair_grad)

    return grad_sum, grad_sq_sum


def _sparse_grad_group_reduce(
    index,
    source_group,
    grad_outputs,
    num_unique,
    num_groups,
    group_size,
    group_reduce_by,
    group_reduce_impl,
    chunk_groups,
):
    global _SPARSE_GRAD_GROUP_REDUCE_STATS_STEP
    stats_interval = int(
        os.environ.get("RECIS_SPARSE_GRAD_GROUP_REDUCE_STATS_INTERVAL", "0")
    )
    if stats_interval > 0:
        _SPARSE_GRAD_GROUP_REDUCE_STATS_STEP += 1
    if (
        stats_interval > 0
        and _SPARSE_GRAD_GROUP_REDUCE_STATS_STEP % stats_interval == 0
    ):
        flat_group_index = source_group * num_unique + index
        active_pairs = int(torch.unique(flat_group_index).numel())
        dense_pairs = int(num_groups * num_unique)
        logger.info(
            "SPARSE_GRAD_GROUP_REDUCE_STATS "
            f"rank={os.environ.get('RANK', '0')} "
            f"local_step={_SPARSE_GRAD_GROUP_REDUCE_STATS_STEP} "
            f"impl={group_reduce_impl} "
            f"reduce_by={group_reduce_by} "
            f"group_size={group_size} "
            f"num_groups={num_groups} "
            f"chunk_groups={chunk_groups} "
            f"request_rows={flat_group_index.numel()} "
            f"num_unique={num_unique} "
            f"dense_pairs={dense_pairs} "
            f"active_pairs={active_pairs} "
            f"active_ratio={active_pairs / max(dense_pairs, 1):.6f}"
        )

    if group_reduce_impl == "dense":
        return _sparse_grad_group_reduce_dense(
            index,
            source_group,
            grad_outputs,
            num_unique,
            num_groups,
            group_size,
            group_reduce_by,
        )
    if group_reduce_impl == "compact":
        return _sparse_grad_group_reduce_compact(
            index,
            source_group,
            grad_outputs,
            num_unique,
            group_size,
            group_reduce_by,
        )
    if group_reduce_impl == "chunk_compact":
        return _sparse_grad_group_reduce_chunk_compact(
            index,
            source_group,
            grad_outputs,
            num_unique,
            num_groups,
            group_size,
            group_reduce_by,
            chunk_groups,
        )
    raise RuntimeError(
        f"Unsupported sparse gradient group reduce impl: {group_reduce_impl}"
    )


class GradGroupSumFunction(torch.autograd.Function):
    """Aggregate sparse gradients by source group for SparseAdagradSum.

    The incoming gradients are aligned with the pre-unique request sequence.
    ``index`` maps each request row to the unique embedding row, while
    ``source_group`` maps each request row to its sparse gradient source group.
    """

    @staticmethod
    def forward(
        ctx,
        embedding,
        index,
        source_group,
        hashtable,
        embedding_index,
        group_size,
        num_groups,
        group_reduce_by,
        group_reduce_impl,
        group_reduce_chunk_groups,
    ):
        ctx.save_for_backward(index, source_group, embedding_index)
        ctx.hashtable = hashtable
        ctx.group_size = group_size
        ctx.num_groups = num_groups
        ctx.group_reduce_by = group_reduce_by
        ctx.group_reduce_impl = group_reduce_impl
        ctx.group_reduce_chunk_groups = group_reduce_chunk_groups
        return torch.ops.recis.gather(index, embedding)

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.cuda()
        index, source_group, embedding_index = ctx.saved_tensors
        hashtable = ctx.hashtable

        if index.numel() == 0:
            grad_sum = torch.zeros(
                [0] + list(grad_outputs.shape)[1:], device=grad_outputs.device
            )
            grad_sq_sum = torch.zeros(
                [0] + list(grad_outputs.shape)[1:], device=grad_outputs.device
            )
            empty_index = torch.zeros([0], dtype=torch.long, device=grad_outputs.device)
            hashtable._hashtable_impl.accept_grad(empty_index, grad_sum)
            hashtable._hashtable_impl.accept_grad_sq(empty_index, grad_sq_sum)
            grad_sum._grad_already_handled = True
            return grad_sum, None, None, None, None, None, None, None, None, None

        index = index.to(device=grad_outputs.device, dtype=torch.long)
        source_group = source_group.to(device=grad_outputs.device, dtype=torch.long)
        source_group = source_group.view(-1)
        index = index.view(-1)

        if source_group.numel() != index.numel():
            raise RuntimeError(
                "source_group must align with the original request sequence: "
                f"source_group={source_group.numel()}, index={index.numel()}"
            )

        num_unique = int(embedding_index.numel())
        num_groups = int(ctx.num_groups)
        group_size = int(ctx.group_size)
        group_reduce_by = ctx.group_reduce_by
        group_reduce_impl = ctx.group_reduce_impl
        group_reduce_chunk_groups = ctx.group_reduce_chunk_groups
        if group_reduce_chunk_groups is None:
            group_reduce_chunk_groups = 1
        group_reduce_chunk_groups = int(group_reduce_chunk_groups)

        if source_group.numel() > 0:
            max_group = int(source_group.max().item())
            if max_group >= num_groups:
                raise RuntimeError(
                    f"source_group contains group {max_group}, "
                    f"but num_groups={num_groups}"
                )

        grad_sum, grad_sq_sum = _sparse_grad_group_reduce(
            index,
            source_group,
            grad_outputs,
            num_unique,
            num_groups,
            group_size,
            group_reduce_by,
            group_reduce_impl,
            group_reduce_chunk_groups,
        )

        index_cuda = embedding_index.to(device=grad_outputs.device)
        hashtable._hashtable_impl.accept_grad(index_cuda, grad_sum)
        hashtable._hashtable_impl.accept_grad_sq(index_cuda, grad_sq_sum)

        grad_sum._grad_already_handled = True
        return grad_sum, None, None, None, None, None, None, None, None, None
