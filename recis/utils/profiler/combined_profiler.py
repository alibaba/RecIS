"""Profile the dense and sparse towers together, in one window.

Running them as two separate hooks meant two profiling windows over different
steps, so their numbers could not be added or compared -- different batches,
different allocator state. Sharing one torch.profiler instance fixes that and also
makes a step-composition view possible: how much of the step is dense, how much is
sparse, and how much is neither (optimizer, data movement, framework overhead).

The two towers keep their own reports, because their cost has a different nature:
dense is FLOPs against a 123 TFLOPS ceiling, sparse is lookups, memory traffic and
all2all. A single merged table would have to drop half the columns of each.

    prof = CombinedProfiler(dense=model.dense_model, sparse=model.sparse_model)
    prof.start_profile()
    for _ in range(3):
        train_step()
        prof.step()
    prof.stop_profile()
    prof.print_report()
    prof.end_profile()
"""

import sys

import torch
from torch.profiler import ProfilerActivity, profile

from recis.utils.profiler.module_profiler import ModuleProfiler
from recis.utils.profiler.sparse_profiler import SparseProfiler


# recis class names that identify the sparse side. RecISModel owns the whole
# sparse pipeline; when a model does not use it, the two engines are the sparse
# side on their own.
_SPARSE_ROOT = "RecISModel"
_SPARSE_ENGINES = ("EmbeddingEngine", "FeatureEngine")


def split_towers(model):
    """Guess (dense, sparse) from a single model, so callers need not split it.

    Order of preference:
      1. explicit dense_model / sparse_model attributes, the convention in the
         training scripts
      2. a RecISModel anywhere in the tree -- it is by definition the sparse side
      3. otherwise EmbeddingEngine / FeatureEngine, which are the sparse side when
         RecISModel is not used
      4. nothing matched -> treat everything as dense

    dense stays the whole model even when sparse sits inside it: the two
    profilers use different record_function markers and each stops walking at the
    other's marker, so a nested sparse tower is not charged to dense.
    """
    d = getattr(model, "dense_model", None)
    sp = getattr(model, "sparse_model", None)
    if isinstance(d, torch.nn.Module) or isinstance(sp, torch.nn.Module):
        return (
            d if isinstance(d, torch.nn.Module) else None,
            sp if isinstance(sp, torch.nn.Module) else None,
        )

    for _, m in model.named_modules():
        if type(m).__name__ == _SPARSE_ROOT:
            return model, m

    engines = [
        m for _, m in model.named_modules() if type(m).__name__ in _SPARSE_ENGINES
    ]
    if engines:
        # several engines can appear; the shallowest common owner is not
        # necessarily a module, so the first one is used and the rest are covered
        # because SparseProfiler instruments the pipeline stages it finds
        return model, engines[0] if len(engines) == 1 else engines
    return model, None


class CombinedProfiler:
    """Own one profiler, drive a ModuleProfiler and a SparseProfiler off it.

    Args:
        model: the whole model. When given without dense/sparse, the two towers
            are detected automatically (see split_towers), so the caller does not
            have to know how the model is organised.
        dense: the dense tower, or None. Overrides detection when given.
        sparse: the sparse tower, or None. Overrides detection when given.
        max_depth: dense instrumentation depth. Keep it small; every wrapped call
            adds a CUDA event pair.
        include: dense name prefixes to instrument beyond max_depth.
        max_sparse_modules: cap on instrumented features (the tower has ~794).
        top_n: rows per ranking table.
    """

    def __init__(
        self,
        model=None,
        dense=None,
        sparse=None,
        max_depth=2,
        include=None,
        max_sparse_modules=64,
        top_n=15,
    ):
        if model is not None and dense is None and sparse is None:
            dense, sparse = split_towers(model)
            if isinstance(sparse, list):
                # more than one engine: profile the first, the stage-level
                # instrumentation inside SparseProfiler covers the others
                sparse = sparse[0]
        if dense is None and sparse is None:
            raise ValueError("need a model, or at least one of dense / sparse")
        self.dense = (
            ModuleProfiler(
                dense,
                max_depth=max_depth,
                include=include,
                with_flops=True,
                record_shapes=True,
            )
            if dense is not None
            else None
        )
        self.sparse = (
            SparseProfiler(sparse, max_modules=max_sparse_modules, top_n=top_n)
            if sparse is not None
            else None
        )
        self.top_n = top_n
        self._prof = None
        self._steps = 0
        self._started = False

    @property
    def model(self):
        return self.dense.model if self.dense is not None else self.sparse.model

    def start_profile(self):
        if self._started:
            raise RuntimeError("already started")
        # with_flops for the dense side, record_shapes for the sparse side's
        # communication tables; both towers read from this one instance
        self._prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True,
            record_shapes=True,
        )
        self._prof.__enter__()
        if self.dense is not None:
            self.dense.start_profile(shared_prof=self._prof)
        if self.sparse is not None:
            self.sparse.start_profile(shared_prof=self._prof)
        self._started = True
        self._steps = 0
        return self

    def step(self):
        self._steps += 1
        if self.dense is not None:
            self.dense.step()
        if self.sparse is not None:
            self.sparse.step()

    def stop_profile(self, steps=None):
        if not self._started:
            raise RuntimeError("not started")
        torch.cuda.synchronize()
        self._prof.__exit__(None, None, None)
        if steps is not None:
            self._steps = steps
        self._steps = max(self._steps, 1)
        # the sub-profilers no longer own the instance, so these only extract
        if self.dense is not None:
            self.dense.stop_profile(steps=self._steps)
        if self.sparse is not None:
            self.sparse.stop_profile(steps=self._steps)
        return self

    def end_profile(self):
        if self.dense is not None:
            self.dense.end_profile()
        if self.sparse is not None:
            self.sparse.end_profile()
        self._started = False
        return self

    def __enter__(self):
        return self.start_profile()

    def __exit__(self, *exc):
        self.stop_profile()
        return False

    # ---- reporting ----

    def backward_tower_split(self):
        """Split backward kernel time by tower using sequence_nr.

        An earlier attempt marked the gradient boundary on the sparse tower's
        output tensors and split by timestamp. On the real model that left 66.6% of
        backward unassigned: with 123 boundary tensors the autograd queue
        interleaves the two subgraphs over a long interval, so "dense finishes
        before sparse starts" simply does not hold.

        sequence_nr does hold. Every forward op carries one, the autograd node that
        consumes it carries the same one, so backward maps to whichever tower ran
        the forward op. Measured 2% unattributed on a controlled case.
        """
        if self._prof is None:
            return None
        seq_owner = {}
        if self.dense is not None:
            for k in self.dense.forward_seq_owner():
                seq_owner[k] = "dense"
        if self.sparse is not None:
            # sparse wins on conflict: its ops sit inside the dense root, so a
            # shared number belongs to the inner tower
            for k in self.sparse.forward_seq_owner():
                seq_owner[k] = "sparse"
        if not seq_owner:
            return {"seqs": 0}

        n = max(self._steps, 1)
        out = {"seqs": len(seq_owner), "dense": 0.0, "sparse": 0.0, "unattributed": 0.0}
        for ev in self._prof.events():
            nm = getattr(ev, "name", "") or ""
            if not nm.startswith("autograd::engine::evaluate_function"):
                continue
            stack, ks = [ev], 0.0
            while stack:
                cur = stack.pop()
                for k in getattr(cur, "kernels", None) or []:
                    ks += k.duration
                stack.extend(cur.cpu_children or [])
            if ks <= 0:
                continue
            ks = ks / 1e3 / n
            seq = getattr(ev, "sequence_nr", -1)
            owner = seq_owner.get(seq) if seq is not None and seq >= 0 else None
            out[owner or "unattributed"] += ks
        return out

    def print_report(self, file=None, depth=None, top_kernels=12):
        out = file or sys.stdout
        dense_stats = self.dense.get_module_stats() if self.dense else {}
        sparse_mods = self.sparse.get_module_stats() if self.sparse else {}
        d_root = dense_stats.get("<model>", {})
        s_root = sparse_mods.get("<sparse>", {})
        d_wall = d_root.get("wall_ms", 0.0)
        s_wall = s_root.get("wall_ms", 0.0)

        # whole-process device time, for the "neither tower" remainder
        proc_dev = 0.0
        for ev in self._prof.key_averages():
            t = (
                getattr(ev, "self_device_time_total", 0)
                or getattr(ev, "self_cuda_time_total", 0)
                or 0
            )
            if t > 0:
                proc_dev += t / 1e3 / self._steps

        def part(n, title):
            print(f"\n\n{'#' * 112}", file=out)
            print(f"#  PART {n}: {title}", file=out)
            print(f"{'#' * 112}", file=out)

        print(f"\n{'#' * 112}", file=out)
        print(
            f"#  COMBINED PROFILE   steps={self._steps}   "
            f"training={self.model.training}",
            file=out,
        )
        print(f"{'#' * 112}", file=out)
        print(
            "#  Both towers are measured in the SAME profiling window, so "
            "their numbers are directly comparable.",
            file=out,
        )
        print("#  PART 1 OVERVIEW  -- whole process, both towers together", file=out)
        print(
            "#  PART 2 DENSE     -- compute oriented: FLOPs, gemm share, TFLOP/s",
            file=out,
        )
        print(
            "#  PART 3 SPARSE    -- traffic oriented: memory class, all2all, "
            "host overhead",
            file=out,
        )
        print(f"{'#' * 112}", file=out)

        part(1, "OVERVIEW")

        print(f"\n{'*' * 112}", file=out)
        print("* STEP COMPOSITION (forward)", file=out)
        print(f"{'*' * 112}", file=out)
        print(f"{'tower':<30}{'wall_ms':>12}{'share of towers':>18}", file=out)
        print("-" * 112, file=out)
        tot = d_wall + s_wall
        for label, v in (("dense", d_wall), ("sparse", s_wall)):
            print(
                f"{label:<30}{v:12.3f}{(v / tot * 100) if tot else 0:17.1f}%", file=out
            )
        print("-" * 112, file=out)
        print(f"{'sum of towers':<30}{tot:12.3f}", file=out)
        print(f"{'whole-process kernel time':<30}{proc_dev:12.3f}", file=out)
        print(
            "Wall time is per tower and can overlap the rest of the step; the "
            "process figure counts every kernel, including optimizer, data "
            "movement and anything outside these two towers.",
            file=out,
        )

        bts = self.backward_tower_split()
        if bts and bts.get("seqs"):
            print(f"\n{'*' * 112}", file=out)
            print("* BACKWARD BY TOWER", file=out)
            print(f"{'*' * 112}", file=out)
            tot = bts["dense"] + bts["sparse"] + bts["unattributed"]
            print(f"{'tower':<30}{'kernel_ms':>12}{'share':>10}", file=out)
            print("-" * 112, file=out)
            for k in ("dense", "sparse", "unattributed"):
                print(
                    f"{k:<30}{bts[k]:12.3f}{(bts[k] / tot * 100) if tot else 0:9.1f}%",
                    file=out,
                )
            print("-" * 112, file=out)
            print(
                f"Attributed through sequence_nr: each autograd node carries the "
                f"number of the forward op it belongs to "
                f"({bts['seqs']} forward numbers mapped). No op-name matching.",
                file=out,
            )
            print(
                "unattributed = autograd nodes whose number has no forward "
                "match, e.g. custom autograd Functions that do not record one.",
                file=out,
            )
        elif bts is not None:
            print(f"\n{'*' * 112}", file=out)
            print("* BACKWARD BY TOWER", file=out)
            print(f"{'*' * 112}", file=out)
            print(
                "no forward sequence numbers were captured, so backward "
                "cannot be split by tower in this run",
                file=out,
            )

        # process-wide kernel totals and rankings: they span both towers, so they
        # belong here rather than inside either tower's section
        if self.dense is not None:
            self.dense.print_process_overview(top_kernels=top_kernels, file=out)

        if self.dense is not None:
            part(2, "DENSE TOWER (compute oriented)")
            self.dense.print_model_profile(
                depth=depth, top_kernels=top_kernels, file=out
            )

        if self.sparse is not None:
            part(3, "SPARSE TOWER (traffic and communication oriented)")
            self.sparse.print_report(file=out)
