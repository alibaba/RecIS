"""Per-module time, kernel and FLOPs attribution, with a deepspeed-like API.

Reports three times per module, which together identify the bottleneck class:

    wall    time the module occupies in the step (CUDA events around the call)
    kernel  time of the GPU kernels attributed to it
    gemm    time of the compute kernels among those (gemm / bmm / matmul / conv)

    wall - kernel  = launch gap, the device idling while the host submits work
    kernel - gemm  = memory-bound work (elementwise, copies, reductions)
    gemm / kernel  = how much of the device time is actually arithmetic

So a module with wall 400, kernel 200, gemm 100 is idle half the time and only
half of its device work is arithmetic -- two independent optimization targets.

Why the implementation looks the way it does (all four verified by probing this
torch build, after three earlier attribution attempts produced wrong numbers):

  - a record_function range appears TWICE in the event list: the real one has
    cpu_children and self_device_time_total == 0, the duplicate has no children
    and reports exactly twice the correct device total. Summing them or taking the
    max are both wrong, so device_time_total is never read here.
  - kernels are reached instead through FunctionEvent.kernels, and attributed by
    walking cpu_parent up to the nearest enclosing module range. That chain was
    verified: aten::addmm -> aten::linear -> ##module.
  - wall time comes from CUDA events, the one method whose numbers matched
    independent per-module benchmarks to within 1%.
  - there is NO per-module backward table. full_backward_hook does not fire for a
    module returning a dict (verified on this torch build) and reports only part of
    the interval for multi-output modules, so such a column would mislead rather
    than merely be incomplete. Backward is reported as aggregates plus a
    hook-independent autograd-node breakdown.

Usage:

    prof = ModuleProfiler(model, max_depth=2)
    prof.start_profile()
    for _ in range(3):
        model(x).sum().backward()
        prof.step()
    prof.stop_profile()
    prof.print_model_profile()
    prof.end_profile()          # restores the model

or via profile_steps(model, step_fn, steps=3, warmup=3).
"""

import sys
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile, record_function


# Must not be a prefix of, nor prefixed by, the sparse profiler's tag. Both
# towers can share one torch.profiler instance, and a bare "##" made the dense
# side claim the sparse anchors ("##sparse##..." also starts with "##"), which
# charged 458 ms of embedding_engine and feature_engine kernels to the dense
# model, recis::functional::segment_sum among them.
_TAG = "##dense##"
# the sparse profiler's marker. A sparse tower is usually NESTED inside the model
# passed here, so walking up from one of its kernels would reach this model's own
# marker and charge the kernel to it. Stopping at a foreign marker keeps the two
# towers separate even when one contains the other.
_FOREIGN_TAG = "##sparse##"

# PPU kernels are named gemm_ktype0_aiu1_..., cuda ones cutlass/sgemm/etc.
_COMPUTE_HINTS = (
    "gemm",
    "bmm",
    "matmul",
    "addmm",
    "cutlass",
    "conv",
    "dot_",
    "mma",
    "wgrad",
    "dgrad",
)
_BWD_HINTS = ("backward", "autograd::")


def _is_compute(kernel_name):
    n = kernel_name.lower()
    return any(h in n for h in _COMPUTE_HINTS)


def _fmt_n(v):
    if v >= 1e9:
        return f"{v / 1e9:7.2f}G"
    if v >= 1e6:
        return f"{v / 1e6:7.2f}M"
    if v >= 1e3:
        return f"{v / 1e3:7.2f}K"
    return f"{v:8.0f}"


class ModuleProfiler:
    """Attribute device time, kernels and FLOPs to modules without editing them.

    Args:
        model: module to profile.
        max_depth: instrument modules at most this deep (0 = the model itself).
            Keep it small: every instrumented call adds a CUDA event pair, and a
            wide model (this one has 173 FC blocks) skews shallow numbers if all
            leaves are wrapped.
        include: name prefixes to instrument beyond max_depth, e.g.
            ["main_net.transformer"], for drilling into one subtree.
        with_flops: have torch.profiler count FLOPs (matmul family; einsum lowers
            to bmm so it is included, unlike in deepspeed's MACs column).
    """

    def __init__(
        self, model, max_depth=3, include=None, with_flops=True, record_shapes=False
    ):
        self.model = model
        self.max_depth = max_depth
        self.include = tuple(include or ())
        self.with_flops = with_flops
        self.record_shapes = record_shapes
        self._orig = {}
        self._pairs = []
        self._prof = None
        self._owns_prof = True
        self._steps = 0
        self._started = False

    # ---- selection ----

    def _selected(self):
        out = []
        for name, mod in self.model.named_modules():
            depth = 0 if name == "" else name.count(".") + 1
            keep = depth <= self.max_depth
            if not keep and self.include:
                keep = any(name.startswith(p) for p in self.include)
            if keep:
                out.append((name or "<model>", mod))
        return out

    # ---- lifecycle ----

    def start_profile(self, shared_prof=None):
        """Instrument the model. shared_prof lets a caller profile several towers
        in one window, so their numbers come from the same steps and can be
        compared or added; the owner is then responsible for exiting it."""
        if self._started:
            raise RuntimeError("already started")
        self._pairs = []
        for name, mod in self._selected():
            self._orig[name] = (mod, mod.forward)
            mod.forward = self._wrap(mod.forward, name)
        if shared_prof is not None:
            self._prof = shared_prof
            self._owns_prof = False
        else:
            self._prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_flops=self.with_flops,
                record_shapes=self.record_shapes,
            )
            self._prof.__enter__()
            self._owns_prof = True
        self._started = True
        self._steps = 0
        return self

    def _wrap(self, fn, label):
        """CUDA events give wall time; record_function is the attribution anchor."""
        pairs = self._pairs
        tag = _TAG + label

        def wrapped(*a, **kw):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            with record_function(tag):
                r = fn(*a, **kw)
            e.record()
            pairs.append((label, s, e))
            return r

        return wrapped

    def step(self):
        self._steps += 1

    def stop_profile(self, steps=None):
        if not self._started:
            raise RuntimeError("not started")
        torch.cuda.synchronize()  # required before elapsed_time()
        if self._owns_prof:
            self._prof.__exit__(None, None, None)
        if steps is not None:
            self._steps = steps
        self._steps = max(self._steps, 1)
        return self

    def end_profile(self):
        for (mod, fn) in self._orig.values():
            mod.forward = fn
        self._orig.clear()
        self._started = False
        return self

    def __enter__(self):
        return self.start_profile()

    def __exit__(self, *exc):
        self.stop_profile()
        return False

    # ---- attribution ----

    def _walk_up(self, ev, limit=40):
        """(nearest module owner, is_backward) by walking cpu_parent.

        owner is None when the op belongs to a nested foreign tower, so its
        kernels are left to that tower's own profiler.
        """
        owner, is_bwd = None, False
        cur, n = ev, 0
        while cur is not None and n < limit:
            nm = getattr(cur, "name", "") or ""
            if owner is None and nm.startswith(_FOREIGN_TAG):
                return None, is_bwd
            if owner is None and nm.startswith(_TAG):
                owner = nm[len(_TAG) :]
            low = nm.lower()
            if any(h in low for h in _BWD_HINTS):
                is_bwd = True
            cur = getattr(cur, "cpu_parent", None)
            n += 1
        return owner, is_bwd

    def _attribute(self):
        """Split kernels and flops into forward (per module) and backward."""
        n = self._steps
        self_k = defaultdict(lambda: {"kernel": 0.0, "gemm": 0.0})
        self_flops = defaultdict(float)
        bwd_flops = 0.0
        fwd_kernels = defaultdict(float)
        model_fwd_kernels = defaultdict(float)
        per_module_kernels = defaultdict(lambda: defaultdict(float))
        bwd_kernels = defaultdict(float)
        bwd_tot = {"kernel": 0.0, "gemm": 0.0}
        fwd_tot = {"kernel": 0.0, "gemm": 0.0}
        bwd_nodes = defaultdict(float)

        for ev in self._prof.events():
            owner, is_bwd = None, None
            ks = getattr(ev, "kernels", None) or []
            fl = getattr(ev, "flops", 0) or 0
            if not ks and not fl:
                continue
            owner, is_bwd = self._walk_up(ev)

            if fl:
                if is_bwd:
                    bwd_flops += fl / n
                elif owner:
                    self_flops[owner] += fl / n

            for k in ks:
                dur = k.duration / 1e3 / n  # us -> ms per step
                comp = _is_compute(k.name)
                if is_bwd:
                    bwd_tot["kernel"] += dur
                    bwd_kernels[k.name] += dur
                    if comp:
                        bwd_tot["gemm"] += dur
                    node = getattr(ev, "name", "?")
                    bwd_nodes[node] += dur
                else:
                    fwd_tot["kernel"] += dur
                    fwd_kernels[k.name] += dur
                    if comp:
                        fwd_tot["gemm"] += dur
                    if owner:
                        # owner != None means the launching op sits inside this
                        # model's tree, so the kernel belongs to it. Without this
                        # split the ranking mixed in sparse kernels such as
                        # recis::functional::segment_sum, which run in the same
                        # step but belong to the other tower.
                        model_fwd_kernels[k.name] += dur
                        per_module_kernels[owner][k.name] += dur
                        self_k[owner]["kernel"] += dur
                        if comp:
                            self_k[owner]["gemm"] += dur
        return {
            "self_kernel": self_k,
            "self_flops": self_flops,
            "fwd_kernels": fwd_kernels,
            "bwd_kernels": bwd_kernels,
            "model_fwd_kernels": model_fwd_kernels,
            "per_module_kernels": per_module_kernels,
            "fwd_total": fwd_tot,
            "bwd_total": bwd_tot,
            "bwd_nodes": bwd_nodes,
            "bwd_flops": bwd_flops,
        }

    def get_module_stats(self):
        """Per module: wall/kernel/gemm/flops, forward; plus backward wall.

        kernel, gemm and flops are rolled up from descendants, so a parent
        includes its children the same way wall does.
        """
        n = self._steps
        att = self._attribute()
        names = set()
        wall = defaultdict(float)
        calls = defaultdict(float)
        for label, s, e in self._pairs:
            wall[label] += s.elapsed_time(e) / n
            calls[label] += 1.0 / n
            names.add(label)
        names |= set(att["self_kernel"]) | set(att["self_flops"])

        def is_desc(child, parent):
            if parent == "<model>":
                return child != "<model>"
            return child.startswith(parent + ".")

        stats = {}
        params = {
            nm or "<model>": sum(p.numel() for p in m.parameters())
            for nm, m in self.model.named_modules()
        }
        for nm in names:
            k = att["self_kernel"].get(nm, {"kernel": 0.0, "gemm": 0.0})
            kernel, gemm = k["kernel"], k["gemm"]
            flops = att["self_flops"].get(nm, 0.0)
            for other in names:
                if other != nm and is_desc(other, nm):
                    o = att["self_kernel"].get(other, {"kernel": 0.0, "gemm": 0.0})
                    kernel += o["kernel"]
                    gemm += o["gemm"]
                    flops += att["self_flops"].get(other, 0.0)
            stats[nm] = {
                "wall_ms": wall.get(nm, 0.0),
                "kernel_ms": kernel,
                "gemm_ms": gemm,
                "flops": flops,
                "calls": calls.get(nm, 0.0),
                "params": params.get(nm, 0),
            }
        self._att = att
        return stats

    def get_kernel_stats(self, phase="total"):
        """[(kernel, ms/step)] sorted.

        phase in {'model_forward', 'forward', 'backward', 'total'}.
        'model_forward' is restricted to kernels launched from inside this model;
        the others are process-wide and therefore also cover the sparse tower.
        """
        att = getattr(self, "_att", None) or self._attribute()
        if phase == "model_forward":
            agg = dict(att.get("model_fwd_kernels") or {})
        elif phase == "forward":
            agg = dict(att["fwd_kernels"])
        elif phase == "backward":
            agg = dict(att["bwd_kernels"])
        else:
            agg = defaultdict(float)
            for d in (att["fwd_kernels"], att["bwd_kernels"]):
                for k, v in d.items():
                    agg[k] += v
        return sorted(agg.items(), key=lambda kv: -kv[1])

    def get_backward_nodes(self):
        att = getattr(self, "_att", None) or self._attribute()
        return sorted(att["bwd_nodes"].items(), key=lambda kv: -kv[1])

    # ---- reporting ----

    def forward_seq_owner(self):
        """{sequence_nr: module} for this model's forward ops.

        torch stamps each forward op with a sequence number and the autograd node
        that later consumes it carries the same number. Mapping the number to the
        module that ran the forward op therefore attributes backward work to
        modules -- which full_backward_hook could not do.
        """
        out = {}
        for ev in self._prof.events():
            seq = getattr(ev, "sequence_nr", -1)
            if seq is None or seq < 0:
                continue
            owner, is_bwd = self._walk_up(ev)
            if owner and not is_bwd:
                out.setdefault(seq, owner)
        return out

    def backward_by_module(self, seq_owner=None):
        """({module: stats}, unattributed_stats) for backward, via sequence_nr.

        stats carry the same measurable fields as the forward table: kernel time,
        gemm time and FLOPs. wall time and the launch gap are absent by
        construction -- those come from CUDA events wrapped around a module's
        forward, and backward has no equivalent call to wrap.

        Only the autograd nodes are iterated: the leaf ops inside them report
        sequence_nr -1, so the number is read from the enclosing node and the whole
        subtree charged to it. Values are then rolled up into parents, matching how
        the forward table nests.
        """
        if self._prof is None:
            return {}, {"kernel_ms": 0.0, "gemm_ms": 0.0, "flops": 0.0}
        n = max(self._steps, 1)
        seq_owner = self.forward_seq_owner() if seq_owner is None else seq_owner

        def blank():
            return {"kernel_ms": 0.0, "gemm_ms": 0.0, "flops": 0.0}

        own = defaultdict(blank)
        unattributed = blank()
        for ev in self._prof.events():
            nm = getattr(ev, "name", "") or ""
            if not nm.startswith("autograd::engine::evaluate_function"):
                continue
            kms = gms = fl = 0.0
            stack = [ev]
            while stack:
                cur = stack.pop()
                for k in getattr(cur, "kernels", None) or []:
                    kms += k.duration
                    if _is_compute(k.name):
                        gms += k.duration
                fl += getattr(cur, "flops", 0) or 0
                stack.extend(cur.cpu_children or [])
            if kms <= 0 and fl <= 0:
                continue
            seq = getattr(ev, "sequence_nr", -1)
            owner = seq_owner.get(seq) if seq is not None and seq >= 0 else None
            tgt = own[owner] if owner else unattributed
            tgt["kernel_ms"] += kms / 1e3 / n
            tgt["gemm_ms"] += gms / 1e3 / n
            tgt["flops"] += fl / n

        # roll descendants into parents, so a row means the same thing as in the
        # forward table (a parent includes its children)
        def is_desc(child, parent):
            if parent == "<model>":
                return child != "<model>"
            return child.startswith(parent + ".")

        # ensure intermediate parents exist as rows even when they have no direct
        # backward kernel time (e.g. seq_net: only its children have backward
        # kernels, so without this the parent row was missing and its children
        # appeared at the wrong nesting level, visually under main_net)
        for nm in list(own.keys()):
            if nm == "<model>":
                continue
            parts = nm.split(".")
            for i in range(1, len(parts)):
                ancestor = ".".join(parts[:i])
                own[ancestor]  # defaultdict(blank) creates it
        own["<model>"]  # ensure root exists

        names = set(own)
        out = {}
        for nm in names:
            acc = dict(own[nm])
            for other in names:
                if other != nm and is_desc(other, nm):
                    for key in acc:
                        acc[key] += own[other][key]
            out[nm] = acc
        return out, unattributed

    def print_process_overview(self, top_kernels=12, file=None):
        out = file or sys.stdout
        """Process-wide view: every kernel in the step, both towers included.

        Kept separate from print_model_profile because these numbers are not
        attributable to this model -- they also cover the sparse tower, the
        optimizer and data movement. Mixing the two scopes in one section was
        what made a launch gap come out negative once.
        """
        att = getattr(self, "_att", None)
        if att is None:
            self.get_module_stats()
            att = self._att
        ft, bt = att["fwd_total"], att["bwd_total"]
        bfl = att["bwd_flops"]
        ffl = sum(att["self_flops"].values())
        bk, bg = bt["kernel"], bt["gemm"]

        def hdr(title):
            print(f"\n{'*' * 112}", file=out)
            print(f"* {title}", file=out)
            print(f"{'*' * 112}", file=out)

        hdr("WHOLE-PROCESS KERNEL TOTALS (both towers, optimizer, data movement)")
        print(
            f"{'phase':<20}{'kernel':>10}{'gemm':>10}{'gemm%':>8}"
            f"{'GFLOPs':>10}{'TF/s@gemm':>11}{'TF/s@kernel':>12}",
            file=out,
        )
        print("-" * 112, file=out)
        rows = (
            ("forward", ft["kernel"], ft["gemm"], ffl),
            ("backward", bk, bg, bfl),
            ("fwd + bwd", ft["kernel"] + bk, ft["gemm"] + bg, ffl + bfl),
        )
        for label, k, g, fl in rows:
            print(
                f"{label:<20}{k:10.3f}{g:10.3f}"
                f"{(g / k * 100) if k else 0:7.1f}%{fl / 1e9:10.2f}"
                f"{(fl / (g / 1e3) / 1e12) if g else 0:11.1f}"
                f"{(fl / (k / 1e3) / 1e12) if k else 0:12.1f}",
                file=out,
            )
        print("-" * 112, file=out)
        print(
            "GFLOPs covers the matmul family only, so custom/Triton kernels "
            "are missing from it; gemm time is matched by kernel name and "
            "therefore also misses them",
            file=out,
        )

        nodes = self.get_backward_nodes()
        if nodes:
            hdr("BACKWARD BY AUTOGRAD NODE (no per-module backward is available)")
            print(
                "full_backward_hook does not fire for modules returning a dict "
                "and reports partial intervals for multi-output ones, so this "
                "hook-independent view is what can be measured honestly.",
                file=out,
            )
            print(f"\n{'node':<60}{'ms/step':>10}{'%bwd':>8}", file=out)
            tot_b = max(bk, 1e-9)
            for k, v in nodes[:top_kernels]:
                print(f"  {k[:58]:<58}{v:10.3f}{v / tot_b * 100:7.1f}%", file=out)

        titles = {
            "forward": "TOP KERNELS (WHOLE PROCESS, FORWARD)",
            "backward": "TOP KERNELS (WHOLE PROCESS, BACKWARD)",
            "total": "TOP KERNELS (WHOLE PROCESS, TOTAL)",
        }
        for phase in ("forward", "backward", "total"):
            hdr(titles[phase])
            ks = self.get_kernel_stats(phase)
            if not ks:
                print("  none", file=out)
                continue
            tot = sum(v for _, v in ks) or 1.0
            print(f"{'kernel':<62}{'ms/step':>10}{'%':>8}{'compute':>9}", file=out)
            for k, v in ks[:top_kernels]:
                print(
                    f"  {k[:60]:<60}{v:10.3f}{v / tot * 100:7.1f}%"
                    f"{'yes' if _is_compute(k) else '-':>9}",
                    file=out,
                )

    def print_model_profile(self, depth=None, top_kernels=12, file=None):
        out = file or sys.stdout
        stats = self.get_module_stats()
        att = self._att
        if not stats:
            print("no modules captured", file=out)
            return
        depth = self.max_depth if depth is None else depth
        root = stats.get("<model>", {})
        total_wall = root.get("wall_ms") or max(
            (s["wall_ms"] for s in stats.values()), default=1.0
        )

        def hdr(title):
            print(f"\n{'*' * 100}", file=out)
            print(f"* {title}", file=out)
            print(f"{'*' * 100}", file=out)

        print(f"\n{'=' * 100}", file=out)
        print(
            f"= ModuleProfiler   steps={self._steps}   "
            f"training={self.model.training}   depth<={depth}",
            file=out,
        )
        print(f"{'=' * 100}", file=out)
        print(
            "= wall   : time the module occupies (CUDA events, includes "
            "submodules and idle gaps)",
            file=out,
        )
        print(
            "= kernel : GPU kernel time attributed to it (includes submodules)",
            file=out,
        )
        print("= gemm   : compute kernels among those (gemm/bmm/matmul/conv)", file=out)
        print("= gap%   : 1 - kernel/wall, device idle waiting on the host", file=out)
        print("= gemm%  : gemm/kernel, share of device time doing arithmetic", file=out)
        print(
            "= SCOPE  : the module table and 'THIS MODEL' rankings count only "
            "kernels launched from inside this model. Rows marked WHOLE "
            "PROCESS also include the sparse tower and data movement.",
            file=out,
        )
        if not self.model.training:
            print(
                "= NOTE: model is in eval mode; if it picks autocast dtype "
                "from self.training this is not the training cost",
                file=out,
            )
        print(f"{'=' * 100}", file=out)

        hdr("MODULE BREAKDOWN (FORWARD)")
        print(
            f"{'module':<34}{'wall':>9}{'kernel':>9}{'gemm':>9}"
            f"{'%wall':>7}{'gap%':>7}{'gemm%':>7}{'GFLOPs':>9}"
            f"{'TF/s@gemm':>10}{'TF/s@wall':>10}{'params':>9}",
            file=out,
        )
        print("-" * 112, file=out)

        def sort_key(item):
            return (0,) if item[0] == "<model>" else (1, item[0])

        for nm, s in sorted(stats.items(), key=sort_key):
            d = 0 if nm == "<model>" else nm.count(".") + 1
            if d > depth:
                continue
            label = ("  " * d + (nm.split(".")[-1] if d else nm))[:34]
            w, k, g = s["wall_ms"], s["kernel_ms"], s["gemm_ms"]
            gap = (1 - k / w) * 100 if w > 0 else 0.0
            gpct = (g / k * 100) if k > 0 else 0.0
            fl = s["flops"]
            tf_g = fl / (g / 1e3) / 1e12 if g > 0 else 0.0
            tf_w = fl / (w / 1e3) / 1e12 if w > 0 else 0.0
            print(
                f"{label:<34}{w:9.3f}{k:9.3f}{g:9.3f}"
                f"{w / total_wall * 100:6.1f}%{gap:6.1f}%{gpct:6.1f}%"
                f"{fl / 1e9:9.2f}{tf_g:10.1f}{tf_w:10.1f}"
                f"{_fmt_n(s['params'])}",
                file=out,
            )

        print("-" * 112, file=out)
        print(
            "Process-wide totals and the process-wide kernel rankings live in "
            "the OVERVIEW section, since they cover both towers.",
            file=out,
        )

        bwd_mod, bwd_un = self.backward_by_module()
        if bwd_mod:
            hdr("MODULE BREAKDOWN (BACKWARD, via sequence_nr)")
            print(
                "Each autograd node carries the sequence number of the forward "
                "op it belongs to, so backward maps back to modules. This works "
                "where full_backward_hook fails on dict-returning modules.",
                file=out,
            )
            print(
                "Same columns as the forward table except wall and gap%: those "
                "need CUDA events around a module call, and backward has no "
                "call to wrap. Rows nest, a parent includes its children.",
                file=out,
            )
            root_b = bwd_mod.get("<model>", {}).get("kernel_ms", 0.0)
            tot_b = root_b or max(
                (v["kernel_ms"] for v in bwd_mod.values()), default=1.0
            )
            print(
                f"\n{'module':<34}{'kernel':>9}{'gemm':>9}{'%bwd':>7}"
                f"{'gemm%':>7}{'GFLOPs':>9}{'TF/s@gemm':>10}"
                f"{'TF/s@kernel':>12}{'params':>9}",
                file=out,
            )
            print("-" * 112, file=out)
            for nm, v in sorted(bwd_mod.items(), key=sort_key):
                d = 0 if nm == "<model>" else nm.count(".") + 1
                if d > depth:
                    continue
                label = ("  " * d + (nm.split(".")[-1] if d else nm))[:34]
                k, g, fl = v["kernel_ms"], v["gemm_ms"], v["flops"]
                print(
                    f"{label:<34}{k:9.3f}{g:9.3f}"
                    f"{(k / tot_b * 100) if tot_b else 0:6.1f}%"
                    f"{(g / k * 100) if k else 0:6.1f}%{fl / 1e9:9.2f}"
                    f"{(fl / (g / 1e3) / 1e12) if g else 0:10.1f}"
                    f"{(fl / (k / 1e3) / 1e12) if k else 0:12.1f}"
                    f"{_fmt_n(stats.get(nm, {}).get('params', 0))}",
                    file=out,
                )
            print("-" * 112, file=out)
            ub = bwd_un["kernel_ms"]
            print(
                f"{'not this model (other tower / no seq)':<34}{ub:9.3f}"
                f"{bwd_un['gemm_ms']:9.3f}"
                f"{(ub / tot_b * 100) if tot_b else 0:6.1f}%",
                file=out,
            )

        hdr("TOP KERNELS (THIS MODEL ONLY, FORWARD)")
        ks = self.get_kernel_stats("model_forward")
        if not ks:
            print("  none", file=out)
        else:
            tot = sum(v for _, v in ks) or 1.0
            print(f"{'kernel':<62}{'ms/step':>10}{'%':>8}{'compute':>9}", file=out)
            for k, v in ks[:top_kernels]:
                print(
                    f"  {k[:60]:<60}{v:10.3f}{v / tot * 100:7.1f}%"
                    f"{'yes' if _is_compute(k) else '-':>9}",
                    file=out,
                )

        hdr("KERNELS BY MODULE (which module launched what)")
        print(
            "Answers 'why is this kernel attributed to me': each kernel is "
            "charged to the nearest enclosing module of the op that launched "
            "it. A kernel from another library still belongs here if this "
            "model called it.",
            file=out,
        )
        pmk = att.get("per_module_kernels") or {}
        ranked = sorted(
            ((nm, s["kernel_ms"]) for nm, s in stats.items() if nm != "<model>"),
            key=lambda kv: -kv[1],
        )
        for nm, kms in ranked[:6]:
            ks = sorted((pmk.get(nm) or {}).items(), key=lambda kv: -kv[1])
            if not ks:
                continue
            print(
                f"\n{nm}  (self kernels {sum(v for _, v in ks):.3f} ms of "
                f"{kms:.3f} ms including children)",
                file=out,
            )
            for k, v in ks[:5]:
                print(f"    {k[:70]:<70}{v:10.3f}", file=out)

        hdr("FLOPS")
        tf = sum(s["flops"] for nm, s in stats.items() if nm == "<model>")
        if tf <= 0:
            tf = max((s["flops"] for s in stats.values()), default=0.0)
        print(
            f"forward FLOPs per step: {tf / 1e9:.2f} G   "
            f"(matmul family incl. einsum; custom/Triton kernels excluded)",
            file=out,
        )

        hdr("SUMMARY")
        # use the model's own rolled-up numbers; the process-wide totals include
        # kernels from outside this model and would make the gap come out negative
        w = root.get("wall_ms", 0.0)
        mk = root.get("kernel_ms", 0.0)
        mg = root.get("gemm_ms", 0.0)
        print("THIS MODEL, FORWARD", file=out)
        print(f"  wall {w:9.3f} ms   kernel {mk:9.3f} ms   gemm {mg:9.3f} ms", file=out)
        if w > 0:
            print(
                f"  launch gap        {(w - mk):9.3f} ms "
                f"({(1 - mk / w) * 100:5.1f}% of wall)   "
                f"-> device idle waiting on the host",
                file=out,
            )
        if mk > 0:
            print(
                f"  memory-bound      {(mk - mg):9.3f} ms "
                f"({(1 - mg / mk) * 100:5.1f}% of kernel) "
                f"-> elementwise, copies, reductions",
                file=out,
            )
            print(
                f"  compute-bound     {mg:9.3f} ms "
                f"({mg / mk * 100:5.1f}% of kernel) -> arithmetic",
                file=out,
            )
        print(f"{'=' * 100}\n", file=out)


def profile_steps(
    model, step_fn, steps=3, warmup=3, max_depth=3, include=None, **report_kw
):
    """Warm up, profile `steps` iterations, print the report, restore the model.

    Warmup runs outside the profiled region, which matters here: Triton JIT and
    allocator growth otherwise land inside the measurement.
    """
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()

    prof = ModuleProfiler(model, max_depth=max_depth, include=include)
    prof.start_profile()
    try:
        for _ in range(steps):
            step_fn()
            prof.step()
    finally:
        prof.stop_profile()
    try:
        prof.print_model_profile(**report_kw)
    finally:
        prof.end_profile()
    return prof
