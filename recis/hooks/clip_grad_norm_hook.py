from recis.hooks.hook import Hook
from recis.optim.grad_norm import clip_grad_by_global_norm_


class ClipGradNormHook(Hook):
    """Hook that clips gradients by global norm after each backward pass.

    Supports both dense parameter gradients and sparse HashTable gradients.

    Args:
        dense_params: Iterable of dense parameters whose gradients should be clipped.
        sparse_params: Iterable of HashTable modules whose sparse gradients should be clipped.
        max_norm: Maximum allowed global norm of gradients.
        error_if_nonfinite: If True, raise error when global norm is nan/inf.
    """

    def __init__(self, dense_params, sparse_params, max_norm, error_if_nonfinite=False):
        self.max_norm = max_norm
        self.error_if_nonfinite = error_if_nonfinite
        self.dense_params = dense_params
        self.sparse_params = sparse_params

    def after_backward(self, *args, **kwargs):
        clip_grad_by_global_norm_(
            self.dense_params,
            self.max_norm,
            error_if_nonfinite=self.error_if_nonfinite,
            sparse_hashtables=self.sparse_params,
        )
