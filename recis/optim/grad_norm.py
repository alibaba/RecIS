import torch
from torch.nn.utils import clip_grad_norm_


def _collect_sparse_grads(sparse_hashtables):
    """Collect sparse gradients from HashTable modules.

    Args:
        sparse_hashtables: Iterable of HashTable modules.

    Returns:
        List of (hashtable, sparse_coo_grad) pairs with non-empty gradients.
    """
    results = []
    for hashtable in sparse_hashtables:
        sparse_grad = hashtable.grad()
        if sparse_grad is not None and sparse_grad._nnz() > 0:
            results.append((hashtable, sparse_grad.coalesce()))
    return results


def clip_grad_by_global_norm_(
    parameters,
    max_norm,
    error_if_nonfinite=False,
    sparse_hashtables=None,
):
    """Clip gradients by global L2 norm, supporting both dense and sparse HashTable gradients.

    For dense parameters, delegates to ``torch.nn.utils.clip_grad_norm_``.
    When ``sparse_hashtables`` is provided, sparse gradients from HashTable modules
    are included in the global norm computation and clipped with the same scale factor.

    Args:
        parameters: Iterable of dense parameters (with ``.grad``).
        max_norm: Maximum allowed global norm.
        error_if_nonfinite: If True, raise error when global norm is nan/inf.
        sparse_hashtables: Optional iterable of HashTable modules whose
            sparse gradients should also be clipped.

    Returns:
        Global gradient norm (0-D tensor).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)

    # No sparse hashtables: just use torch's optimized implementation directly
    if not sparse_hashtables:
        return clip_grad_norm_(
            parameters, max_norm, error_if_nonfinite=error_if_nonfinite
        )

    # Collect dense grads
    dense_grads = [p.grad for p in parameters if p.grad is not None]

    # Collect sparse grads from HashTable modules
    sparse_grad_pairs = _collect_sparse_grads(sparse_hashtables)
    sparse_grad_values = [sg.values() for _, sg in sparse_grad_pairs]

    all_grads = dense_grads + sparse_grad_values
    if len(all_grads) == 0:
        return torch.tensor(0.0, dtype=torch.float32)

    device = all_grads[0].device

    # Compute global L2 norm across all grads
    total_norm_sq = torch.zeros((), dtype=torch.float32, device=device)
    for grad in all_grads:
        total_norm_sq += (grad.detach().to(torch.float32) ** 2).sum()
    total_norm = torch.sqrt(total_norm_sq)

    if error_if_nonfinite and (torch.isnan(total_norm) or torch.isinf(total_norm)):
        raise RuntimeError(
            f"The total norm of gradients is non-finite: {total_norm.item()}"
        )

    max_norm_t = torch.tensor(float(max_norm), device=device, dtype=torch.float32)
    clip_coef = max_norm_t / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    # Clip dense gradients in-place
    for grad in dense_grads:
        grad.mul_(clip_coef_clamped.to(dtype=grad.dtype))

    # Clip sparse gradients: clear original, write back scaled values
    for hashtable, sparse_grad in sparse_grad_pairs:
        indices = sparse_grad.indices()
        values = sparse_grad.values()
        scaled_values = values * clip_coef_clamped.to(dtype=values.dtype)
        hashtable.clear_grad()
        hashtable.accept_grad(indices.squeeze(0), scaled_values)

    return total_norm
