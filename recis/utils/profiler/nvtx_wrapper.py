"""NVTX wrapping for ncu profiling integration.

Wraps torch.ops.recis ops, torch.unique, and HashTable.embedding_lookup
with NVTX range markers, controlled by _ProfileState.enabled.

Key design:
  - Each NVTX wrapper uses a *mutable* target reference (a single-element
    list).  The wrapper always delegates to ``target[0]``, which can be
    swapped at runtime to compose with other tracking layers.
  - For torch.ops.recis ops, ``__getattr__`` on ``_OpNamespace`` handles
    lazy wrapping; ``_rewrap_cached_ops`` handles ops cached before this
    module was imported.
  - ``torch.unique`` is wrapped directly via ``setattr`` at import time.
    ``HashTable.forward`` is wrapped to capture the C++ embedding_lookup
    call (torch.unique and gather have their own nested NVTX ranges).
  - Coexistence with ``MemoryAccessTracker`` is achieved by monkey-patching
    ``MemoryAccessTracker.patch_all`` / ``unpatch_all`` at the bottom of
    this module.  No changes to ``memory_access.py`` are needed.
"""

import functools

import torch


class _ProfileState:
    enabled = False


# ── Storage ──────────────────────────────────────────────────────────
_wrapped_ops = {}  # key → NVTX wrapper callable
_orig_ops = {}  # key → original unwrapped callable
_targets = {}  # key → [current_callable]  (mutable reference)


# ── Core wrapping ────────────────────────────────────────────────────


def _wrap_op(orig, name, target):
    """Create an NVTX wrapper that calls ``target[0]`` at invocation time.

    Args:
        orig: The original unwrapped callable (kept for reference).
        name: Human-readable op name used in the NVTX range label.
        target: A single-element list ``[callable]``.  The wrapper always
            delegates to ``target[0]``, which can be swapped at runtime
            (e.g. by MemoryAccessTracker) to compose tracking layers.
    """

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        if _ProfileState.enabled:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"op:{name}")
        try:
            return target[0](*args, **kwargs)
        finally:
            if _ProfileState.enabled:
                torch.cuda.nvtx.range_pop()

    return wrapper


# ── _OpNamespace patching (for torch.ops.recis.* ops) ────────────────
#
# PyTorch's _OpNamespace.__getattr__ caches resolved ops in __dict__.
# Subsequent lookups find them in __dict__ and never call __getattr__.
# Since __dict__ contains the UNWRAPPED originals, we MUST override
# __getattribute__ to always return the NVTX-wrapped version from
# _wrapped_ops for recis ops.
#
# This works correctly with the MemoryAccessTracker monkey-patch below:
# patch_all sets _targets[key][0] = memory_wrapper, and __getattribute__
# returns the same NVTX wrapper (from _wrapped_ops) whose target now
# points to memory_wrapper.  No setattr conflict.

_orig_ns_getattr = torch._ops._OpNamespace.__getattr__
_orig_ns_getattribute = torch._ops._OpNamespace.__getattribute__


def _patched_ns_getattribute(self, name):
    """Intercept ALL attribute access on _OpNamespace instances.

    For recis ops, return the NVTX-wrapped version from _wrapped_ops.
    For everything else (including 'name', dunder attrs, etc.),
    delegate to the default __getattribute__.
    """
    if name.startswith("_"):
        return _orig_ns_getattribute(self, name)
    try:
        ns = _orig_ns_getattribute(self, "name")
        if ns == "recis":
            key = (ns, name)
            if key in _wrapped_ops:
                return _wrapped_ops[key]
    except Exception:
        pass
    return _orig_ns_getattribute(self, name)


def _patched_ns_getattr(self, op_name):
    """Handle first-time op resolution (op not yet in __dict__).

    Resolves the op via the original __getattr__, wraps it with NVTX,
    stores in _wrapped_ops, and returns the wrapped version.
    """
    op = _orig_ns_getattr(self, op_name)
    if self.name == "recis":
        key = (self.name, op_name)
        if key not in _wrapped_ops:
            _orig_ops[key] = op
            target = [op]
            _targets[key] = target
            _wrapped_ops[key] = _wrap_op(op, op_name, target)
        return _wrapped_ops[key]
    return op


# Apply both patches.  __getattribute__ is essential: without it,
# ops cached in __dict__ (unwrapped) would bypass __getattr__ entirely.
torch._ops._OpNamespace.__getattribute__ = _patched_ns_getattribute
torch._ops._OpNamespace.__getattr__ = _patched_ns_getattr


def _rewrap_cached_ops():
    """Wrap already-cached recis ops that were resolved before patching.

    Called at module import time.  Iterates the _OpNamespace __dict__
    for any ops cached by earlier code (e.g. _wrap_torch_ops) and
    registers NVTX-wrapped versions in _wrapped_ops.
    """
    try:
        recis_ns = torch.ops.recis
    except Exception:
        return

    ns_dict = recis_ns.__dict__
    for key_name, val in list(ns_dict.items()):
        if key_name.startswith("_") or not callable(val):
            continue
        wkey = ("recis", key_name)
        if wkey not in _wrapped_ops:
            _orig_ops[wkey] = val
            target = [val]
            _targets[wkey] = target
            _wrapped_ops[wkey] = _wrap_op(val, key_name, target)


_rewrap_cached_ops()


# ── Direct NVTX wrapping for non-recis ops ───────────────────────────

# torch.unique
try:
    _orig_unique = torch.unique
    _orig_ops["torch.unique"] = _orig_unique
    _unique_target = [_orig_unique]
    _targets["torch.unique"] = _unique_target
    _wrapped_unique = _wrap_op(_orig_unique, "torch.unique", _unique_target)
    _wrapped_ops["torch.unique"] = _wrapped_unique
    torch.unique = _wrapped_unique
except Exception:
    pass

# HashTable.embedding_lookup
# HashTable has no _embedding_lookup_internal method — the C++
# embedding_lookup is called from within HashTable.forward (both
# training and eval paths).  We wrap HashTable.forward with an NVTX
# range.  Since torch.unique and gather have their own NVTX ranges
# (nested inside), ncu attributes their kernels to the innermost
# range, leaving only the C++ embedding_lookup kernels under
# op:HashTable.embedding_lookup.
try:
    from recis.nn.modules.hashtable import HashTable as _HT

    _orig_ht_forward = _HT.forward
    _orig_ops["HashTable.embedding_lookup"] = _orig_ht_forward
    _ht_target = [_orig_ht_forward]
    _targets["HashTable.embedding_lookup"] = _ht_target
    _wrapped_ht_forward = _wrap_op(
        _orig_ht_forward, "HashTable.embedding_lookup", _ht_target
    )
    _wrapped_ops["HashTable.embedding_lookup"] = _wrapped_ht_forward
    _HT.forward = _wrapped_ht_forward
except Exception:
    pass


# ── Coexistence with MemoryAccessTracker (monkey-patch) ──────────────
#
# MemoryAccessTracker (in memory_access.py) dynamically patches/unpatches
# ops via setattr.  For recis ops, this writes to _OpNamespace.__dict__,
# but normal attribute lookup still finds the NVTX wrapper there first.
# For non-recis ops (torch.unique, HashTable), the NVTX wrapper captured
# the original in its closure, so setattr on the module/class doesn't
# affect the NVTX wrapper's call target.
#
# Solution: monkey-patch MemoryAccessTracker.patch_all / unpatch_all to
# redirect NVTX wrapper targets after the original patch/unpatch runs.
#
# For recis ops:
#   - _targets[key][0] is set to the memory_wrapper.
#   - The NVTX wrapper calls memory_wrapper, which calls its captured
#     original_func (= the NVTX wrapper itself, obtained via getattr at
#     setup time).  When _ProfileState.enabled is False, the inner NVTX
#     wrapper is a noop (just calls target[0] = original).
#   - When _ProfileState.enabled is True, the inner NVTX wrapper adds a
#     negligible sync overhead (one extra cuda synchronize).
#
# For non-recis ops (torch.unique, HashTable):
#   - A *composed* wrapper is created: NVTX range_push → memory_wrapper
#     → original_func → range_pop.  This avoids nested NVTX overhead
#     (the inner NVTX wrapper is bypassed entirely).


def _compose_nvtx_with_memory(name, memory_wrapper):
    """Create a composed wrapper: NVTX range → memory tracking → original.

    Used for non-recis ops (torch.unique, HashTable) to avoid the nested
    NVTX wrapper overhead.
    """

    def composed(*args, **kwargs):
        if _ProfileState.enabled:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"op:{name}")
        try:
            return memory_wrapper(*args, **kwargs)
        finally:
            if _ProfileState.enabled:
                torch.cuda.nvtx.range_pop()

    return composed


# Recis op names registered by _wrap_torch_ops in memory_access_setup.py
_recis_op_names = frozenset(
    [
        "bucketize_op",
        "uint64_mod",
        "fused_uint64_mod",
        "fused_bucketized",
        "fused_multi_hash",
        "fused_hash",
        "ids_encode",
        "ids_partition",
        "merge_offsets",
        "gen_segment_indices_by_offset",
        "fused_int64_to_string_int8",
        "fused_ragged_cutoff_2D",
        "fused_ragged_cutoff_3D",
        "ragged_tile",
        "segment_sum",
        "segment_mean",
        "segment_reduce_forward",
        "gather",
    ]
)


def _is_recis_op(op_name):
    return op_name in _recis_op_names


try:
    from recis.utils.profiler.memory_access import MemoryAccessTracker as _MAT

    _orig_patch_all = _MAT.patch_all.__func__
    _orig_unpatch_all = _MAT.unpatch_all.__func__

    @classmethod
    def _nvtx_patch_all(cls):
        """Enhanced patch_all: compose NVTX + memory tracking."""
        if cls._is_patched:
            return

        # Run original patch_all (sets setattr, _is_patched = True)
        _orig_patch_all(cls)

        # Post-process: redirect NVTX targets / compose wrappers
        for op_name in cls._patch_info:
            memory_wrapper = cls._wrapped_funcs.get(op_name)
            if memory_wrapper is None:
                continue

            if _is_recis_op(op_name):
                # Recis ops: redirect NVTX wrapper's mutable target.
                # The NVTX wrapper (in __dict__) calls target[0] = memory_wrapper.
                # memory_wrapper calls its captured original_func (= NVTX wrapper),
                # which is a noop when _ProfileState.enabled is False.
                key = ("recis", op_name)
                if key in _targets:
                    _targets[key][0] = memory_wrapper
            elif op_name in _targets:
                # Non-recis ops (torch.unique, HashTable): create composed wrapper.
                # This avoids nested NVTX overhead.
                composed = _compose_nvtx_with_memory(op_name, memory_wrapper)
                _targets[op_name][0] = composed

    @classmethod
    def _nvtx_unpatch_all(cls):
        """Enhanced unpatch_all: restore NVTX targets to originals."""
        if not cls._is_patched:
            return

        # Run original unpatch_all (restores setattr, _is_patched = False)
        _orig_unpatch_all(cls)

        # Post-process: restore NVTX targets to original unwrapped ops
        for op_name in cls._patch_info:
            if _is_recis_op(op_name):
                key = ("recis", op_name)
            else:
                key = op_name
            if key in _targets and key in _orig_ops:
                _targets[key][0] = _orig_ops[key]

    # Apply monkey-patches
    _MAT.patch_all = _nvtx_patch_all
    _MAT.unpatch_all = _nvtx_unpatch_all

except Exception:
    # MemoryAccessTracker not available or import failed.
    # NVTX wrapping still works standalone.
    pass
