"""Sparse-side profiler: memory-traffic and communication oriented.

The dense report keys on FLOPs; that is meaningless for the sparse tower, whose
cost is lookups, gathers, segment reductions and all2all. So this reports, per
module and in aggregate:

    cpu_ms      host time (self), i.e. the CPU overhead of the op
    kernel_ms   device time of all its kernels, all2all included
    mem_ms      device time of the memory-class ops (gather/scatter/segment/copy)
    comm_ms     device time of all2all kernels
    sync_ms     host time spent in cudaDeviceSynchronize / cudaStreamSynchronize

Communication volume is derived from the chrome trace, because the Python
FunctionEvent API does not expose what is needed (verified: input_types is None).
The trace's args for c10d::alltoall_base_ do carry it:

    Concrete Inputs   ['', '', '', '[77353]', '[77353]', '-1']
    Input type        ['long int', 'long int', '', 'ScalarList', 'ScalarList', ...]
    Input Dims        [[77353], [77353], [], [], [], []]

so dtype comes from `Input type` (here long int = 8 bytes) and the split sizes from
`Concrete Inputs`. Note the kernel name is NOT a dtype source: PCCL/NCCL SendRecv
is point-to-point and moves raw bytes, so its name always says int8_t and using
that would under-count by the real dtype's width (8x here).

volume = sizeof(dtype) * (sum(in_splits) + sum(out_splits)) / 2, i.e. the one-way
volume when the exchange is balanced. Bandwidth uses the *kernel* device time, not
the op's host duration, so it reflects the link rather than link-plus-waiting.

Calls with empty split sizes and 1-element tensors are metadata exchanges (ranks
swapping lengths before the real transfer); they are counted separately so they do
not dilute the payload statistics.

Memory traffic comes from recis's own memory-access tracker, which this profiler
switches on for its window: the torch profiler exposes no traffic counter, and
input dims alone cannot describe a gather's footprint. Those bytes are formula
values over tensor shapes, so they are ideal traffic rather than measured DRAM
traffic, and only ops with a registered formula are covered.
"""

import json
import os
import sys
import tempfile
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile, record_function


# host-side C++ type names as they appear in the trace -> bytes
_DTYPE_BYTES = {
    "long int": 8,
    "long": 8,
    "int64": 8,
    "int64_t": 8,
    "int": 4,
    "int32": 4,
    "int32_t": 4,
    "unsigned int": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
    "c10::BFloat16": 2,
    "at::BFloat16": 2,
    "bfloat16": 2,
    "c10::Half": 2,
    "at::Half": 2,
    "half": 2,
    "float16": 2,
    "short": 2,
    "int16": 2,
    "int16_t": 2,
    "signed char": 1,
    "unsigned char": 1,
    "char": 1,
    "int8": 1,
    "bool": 1,
}

_COMM_HINTS = (
    "nccl",
    "pccl",
    "alltoall",
    "all_to_all",
    "all2all",
    "sendrecv",
    "allgather",
    "all_gather",
    "reduce_scatter",
    "allreduce",
    "broadcast",
)
# memory-class ops seen on this model (probed, not guessed)
_MEM_HINTS = (
    "gather",
    "scatter",
    "index",
    "segment",
    "embedding",
    "copy_",
    "memcpy",
    "memset",
    "cat",
    "unique",
    "sort",
    "hash",
    "lookup",
    "cutoff",
    "offsets",
    "fill_value",
    "partition",
    "ragged",
)
_SYNC_HINTS = (
    "cudadevicesynchronize",
    "cudastreamsynchronize",
    "cudaeventsynchronize",
    "cudamemcpyasync",
)


def _bytes_of(type_name):
    if not type_name:
        return None
    t = type_name.strip()
    if t in _DTYPE_BYTES:
        return _DTYPE_BYTES[t]
    for k, v in _DTYPE_BYTES.items():
        if k in t:
            return v
    return None


def _parse_scalar_list(s):
    """'[77353]' or '[1, 2, 3]' or '[]' -> list of ints."""
    if not s:
        return []
    s = str(s).strip()
    if not (s.startswith("[") and s.endswith("]")):
        return []
    inner = s[1:-1].strip()
    if not inner:
        return []
    out = []
    for part in inner.split(","):
        part = part.strip()
        try:
            out.append(int(part))
        except ValueError:
            pass
    return out


def _is_comm(name):
    n = (name or "").lower()
    return any(h in n for h in _COMM_HINTS)


def _is_mem(name):
    n = (name or "").lower()
    return any(h in n for h in _MEM_HINTS)


def _is_sync(name):
    n = (name or "").lower()
    return any(h in n for h in _SYNC_HINTS)


def _fmt_bytes(n):
    if n is None:
        return "n/a"
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}B"


class SparseProfiler:
    """Per-module CPU/kernel/memory/communication profile for the sparse tower.

    Args:
        model: the sparse module (e.g. model.sparse_model).
        module_prefix: where the real units live. On this model features sit at
            feature_engine._features.<name>, 794 of them, so instrumenting the
            whole tree (3261 modules) is pointless; only this level is wrapped.
        max_modules: cap on instrumented modules, to bound the CUDA-event cost.
        top_n: rows per ranking table.
    """

    def __init__(
        self,
        model,
        module_prefix="feature_engine._features",
        max_modules=64,
        top_n=15,
        per_feature=False,
        track_memory_access=True,
    ):
        self.model = model
        self.module_prefix = module_prefix
        self.per_feature = per_feature
        self.track_memory_access = track_memory_access
        self._traffic = None
        self._traffic_owner = False
        self.max_modules = max_modules
        self.top_n = top_n
        self._orig = {}
        self._pairs = []
        self._prof = None
        self._owns_prof = True
        self._steps = 0
        self._started = False
        self._parsed = None

    # ---- selection ----

    def _selected(self):
        """Root, the three pipeline stages, then optionally features.

        RecISModel.forward runs feature_engine -> embedding_engine ->
        block_builder. Wrapping only the feature level (as a first version did)
        missed the stage where all2all happens and left the module table with a
        single row, because features live under feature_engine and nothing else
        was instrumented.
        """
        out = [("<sparse>", self.model)]
        for stage in ("feature_engine", "embedding_engine", "block_builder"):
            mod = getattr(self.model, stage, None)
            if isinstance(mod, torch.nn.Module):
                out.append((stage, mod))
        if self.per_feature:
            for name, mod in self.model.named_modules():
                if type(mod).__name__ != "Feature":
                    continue
                out.append(("feature:" + name.split(".")[-1], mod))
                if len(out) > self.max_modules:
                    break
        return out

    # ---- lifecycle ----

    def start_profile(self, shared_prof=None):
        """See ModuleProfiler.start_profile: shared_prof profiles both towers in
        one window. record_shapes must be on for the communication tables, so a
        shared profiler has to enable it."""
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
                record_shapes=True,
            )
            self._prof.__enter__()
            self._owns_prof = True
        self._start_traffic()
        self._started = True
        self._steps = 0
        return self

    def _start_traffic(self):
        """Turn on recis memory-access tracking for this window.

        The tracker keeps a single active context, so if something else already
        started one this leaves it alone and reports nothing rather than fighting
        over it.
        """
        self._traffic = None
        self._traffic_owner = False
        if not self.track_memory_access:
            return
        try:
            from recis.utils.profiler.memory_access import MemoryAccessTracker
            from recis.utils.profiler.memory_access_setup import (
                setup_memory_access_tracking,
            )

            if MemoryAccessTracker.is_enabled():
                return
            setup_memory_access_tracking()
            MemoryAccessTracker.start_context("sparse_profiler")
            self._traffic_owner = True
        except Exception:
            self._traffic_owner = False

    def _stop_traffic(self):
        if not self._traffic_owner:
            return
        try:
            from recis.utils.profiler.memory_access import MemoryAccessTracker

            ctx = MemoryAccessTracker.end_context()
            self._traffic = ctx.summary() if ctx is not None else None
        except Exception:
            self._traffic = None
        finally:
            self._traffic_owner = False

    def _wrap(self, fn, label):
        """CUDA events for wall time; record_function marks the sparse subtree.

        The anchor matters: without it the op tables also counted dense ops that
        ran in the same step (aten::bmm from the transformer showed up at 161 ms),
        making every share meaningless because the denominator was the whole
        process rather than this tower.
        """
        pairs = self._pairs
        tag = "##sparse##" + label

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
        self._stop_traffic()
        torch.cuda.synchronize()
        if self._owns_prof:
            self._prof.__exit__(None, None, None)
        if steps is not None:
            self._steps = steps
        self._steps = max(self._steps, 1)
        self._parsed = self._parse_trace()
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

    # ---- trace parsing (the only way to reach dtype and split sizes) ----

    def _parse_trace(self):
        rank = os.environ.get("RANK", "0")
        path = os.path.join(
            tempfile.gettempdir(), f"sparse_prof_{os.getpid()}_rank{rank}.json"
        )
        try:
            self._prof.export_chrome_trace(path)
            with open(path) as f:
                data = json.load(f)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

        events = data.get("traceEvents", data if isinstance(data, list) else [])
        kern_by_ext = defaultdict(list)
        corr_kernels = defaultdict(list)
        kernels = []
        comm_ops = []
        for te in events:
            cat = te.get("cat")
            name = str(te.get("name", ""))
            args = te.get("args") or {}
            ext = args.get("External id")
            dur = te.get("dur") or 0.0
            if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
                kernels.append((name, dur, te.get("ts") or 0.0, ext))
                if ext is not None:
                    kern_by_ext[ext].append((name, dur))
                corr = args.get("correlation")
                if corr is not None:
                    corr_kernels[corr].append((name, dur))
            elif "alltoall_base" in name:
                comm_ops.append(te)
        comm_ops.sort(key=lambda t: t.get("ts") or 0.0)
        return {
            "events": events,
            "kernels": kernels,
            "kern_by_ext": kern_by_ext,
            "corr_kernels": corr_kernels,
            "comm_ops": comm_ops,
        }

    def _comm_records(self):
        """[(volume_bytes, dtype, dtype_bytes, kernel_ms, op_ms, is_meta)]"""
        p = self._parsed or {}
        if "comm_ops" not in p:
            return []
        kern_by_ext = p["kern_by_ext"]
        corr_kernels = p.get("corr_kernels") or {}
        # comm kernels in launch order, for the fallback pairing below
        pending_comm = sorted(
            [
                (name, dur, ts)
                for name, dur, ts, _e in (p.get("kernels") or [])
                if _is_comm(name)
            ],
            key=lambda t: t[2],
        )
        pending_comm = [(nm, d) for nm, d, _ in pending_comm]
        recs = []
        for te in p["comm_ops"]:
            args = te.get("args") or {}
            ci = args.get("Concrete Inputs") or []
            it = args.get("Input type") or []
            dims = args.get("Input Dims") or []
            out_splits = _parse_scalar_list(ci[2] if len(ci) > 2 else "")
            in_splits = _parse_scalar_list(ci[3] if len(ci) > 3 else "")
            # some builds order them (out, in); fall back to whichever is present
            if not out_splits and len(ci) > 3:
                out_splits = _parse_scalar_list(ci[3])
            if not in_splits and len(ci) > 4:
                in_splits = _parse_scalar_list(ci[4])
            dt = it[1] if len(it) > 1 else (it[0] if it else "")
            dtb = _bytes_of(dt)
            elems = (sum(in_splits) + sum(out_splits)) / 2.0
            if not elems and dims:
                d0 = dims[1] if len(dims) > 1 and dims[1] else (dims[0] if dims else [])
                elems = float(d0[0]) if d0 else 0.0
            vol = elems * dtb if dtb else None
            # metadata exchange: no real splits and a single-element payload
            is_meta = (not in_splits and not out_splits) or elems <= 1

            ext = args.get("External id")
            dur = te.get("dur") or 0.0
            kms = sum(d for _, d in kern_by_ext.get(ext, []))
            if kms == 0.0 and corr_kernels:
                # the launching op and its kernel share a correlation id
                corr = args.get("correlation")
                kms = sum(d for _, d in corr_kernels.get(corr, []))
            if kms == 0.0:
                # all_to_all_single is async: it returns a handle and the caller
                # waits later, so the kernel carries neither the op's external id
                # nor its correlation, and it can start after the op has returned.
                # Both id-based paths therefore miss (17 of 33 calls once). Pair
                # in launch order instead, which holds on a single comm stream.
                if pending_comm:
                    kn, kd = pending_comm.pop(0)
                    kms = kd
            recs.append(
                {
                    # one record = one call, so no division by steps here; the
                    # summary divides its totals instead
                    "volume": vol,
                    "dtype": dt or "?",
                    "dtype_bytes": dtb,
                    "kernel_ms": kms / 1e3,
                    "op_ms": dur / 1e3,
                    "is_meta": is_meta,
                    "in_elems": sum(in_splits),
                    "out_elems": sum(out_splits),
                }
            )
        return recs

    # ---- module stats ----

    def get_module_stats(self):
        n = self._steps
        wall, calls = defaultdict(float), defaultdict(float)
        for label, s, e in self._pairs:
            wall[label] += s.elapsed_time(e) / n
            calls[label] += 1.0 / n
        return {k: {"wall_ms": v, "calls": calls[k]} for k, v in wall.items()}

    def get_op_stats(self, scoped=True):
        """Aggregate per op: host time, device time, count.

        scoped=True keeps only ops inside the sparse subtree, found by walking
        cpu_parent up to the record_function anchor. Without this the numbers are
        process-wide and include the dense tower.
        """
        n = self._steps
        agg = defaultdict(lambda: {"cpu": 0.0, "dev": 0.0, "count": 0})
        for ev in self._prof.events():
            name = getattr(ev, "name", "") or ""
            if name.startswith("##sparse##"):
                continue
            if scoped and not self._in_sparse(ev):
                continue
            cpu = getattr(ev, "self_cpu_time_total", 0) or 0
            dev = (
                getattr(ev, "self_device_time_total", 0)
                or getattr(ev, "self_cuda_time_total", 0)
                or 0
            )
            if cpu <= 0 and dev <= 0:
                continue
            a = agg[name]
            a["cpu"] += cpu / 1e3 / n
            a["dev"] += dev / 1e3 / n
            a["count"] += 1.0 / n
        return agg

    def forward_seq_owner(self):
        """{sequence_nr: stage} for this tower's forward ops, for backward lookup."""
        out = {}
        for ev in self._prof.events():
            seq = getattr(ev, "sequence_nr", -1)
            if seq is None or seq < 0:
                continue
            if self._in_sparse(ev):
                out.setdefault(seq, "sparse")
        return out

    def _comm_kernel_ms(self):
        """all2all device time, matched by kernel name.

        Communication kernels are launched asynchronously by all_to_all_single, so
        they have no sparse op as CPU ancestor; the scoped filter therefore drops
        them and the report showed 0.000 ms while the communication tables clearly
        had traffic. Matching on the kernel name avoids the ancestor chain
        entirely.
        """
        n = self._steps
        tot = 0.0
        p = self._parsed or {}
        for name, dur, _ts, _ext in p.get("kernels") or []:
            if _is_comm(name):
                tot += dur / 1e3 / n
        return tot

    @staticmethod
    def _in_sparse(ev, limit=40):
        cur, i = ev, 0
        while cur is not None and i < limit:
            if (getattr(cur, "name", "") or "").startswith("##sparse##"):
                return True
            cur = getattr(cur, "cpu_parent", None)
            i += 1
        return False

    # ---- report ----

    def print_report(self, file=None):
        out = file or sys.stdout
        n = self._steps
        ops = self.get_op_stats(scoped=True)
        all_ops = self.get_op_stats(scoped=False)
        mods = self.get_module_stats()
        recs = self._comm_records()

        tot_cpu = sum(v["cpu"] for v in ops.values())
        tot_dev = sum(v["dev"] for v in ops.values())
        mem_dev = sum(v["dev"] for k, v in ops.items() if _is_mem(k))
        comm_dev = self._comm_kernel_ms()
        sync_cpu = sum(v["cpu"] for k, v in ops.items() if _is_sync(k))

        def hdr(t):
            print(f"\n{'*' * 104}", file=out)
            print(f"* {t}", file=out)
            print(f"{'*' * 104}", file=out)

        print(f"\n{'=' * 104}", file=out)
        print(
            f"= SparseProfiler   steps={n}   training={self.model.training}", file=out
        )
        print(f"{'=' * 104}", file=out)
        print(
            "= cpu    : host time (self). Large values here are CPU overhead, "
            "not device work",
            file=out,
        )
        print("= kernel : device time of the op's kernels (all2all included)", file=out)
        print(
            "= mem    : device time of memory-class ops "
            "(gather/scatter/segment/copy/hash/...)",
            file=out,
        )
        print("= comm   : device time of all2all kernels", file=out)
        print(
            "= sync   : host time in cuda*Synchronize -- the host blocking on "
            "the device",
            file=out,
        )
        print(
            "= NOTE   : byte-level traffic comes from recis's memory-access "
            "tracker and covers only the ops that have a registered "
            "formula; see the MEMORY ACCESS section",
            file=out,
        )
        print(f"{'=' * 104}", file=out)

        hdr("OVERALL (FORWARD)")
        print(f"{'metric':<34}{'ms/step':>12}{'share':>10}", file=out)
        print("-" * 104, file=out)
        print(f"{'host (cpu, self total)':<34}{tot_cpu:12.3f}{'':>10}", file=out)
        print(
            f"{'  of which cuda*Synchronize':<34}{sync_cpu:12.3f}"
            f"{(sync_cpu / tot_cpu * 100) if tot_cpu else 0:9.1f}%",
            file=out,
        )
        print(f"{'device (kernel total)':<34}{tot_dev:12.3f}{'':>10}", file=out)
        print(
            f"{'  of which memory-class':<34}{mem_dev:12.3f}"
            f"{(mem_dev / tot_dev * 100) if tot_dev else 0:9.1f}%",
            file=out,
        )
        print(
            f"{'  of which all2all':<34}{comm_dev:12.3f}"
            f"{(comm_dev / tot_dev * 100) if tot_dev else 0:9.1f}%"
            f"   (by kernel name; async launches have no CPU ancestor)",
            file=out,
        )
        print("-" * 104, file=out)
        print(
            f"verdict: {'CPU-bound' if tot_cpu > tot_dev else 'device-bound'}"
            f"  (host {tot_cpu:.1f} ms vs device {tot_dev:.1f} ms)",
            file=out,
        )
        aw = sum(v["cpu"] for v in all_ops.values())
        ad = sum(v["dev"] for v in all_ops.values())
        print(
            f"scope: this tower only. whole process for comparison: "
            f"host {aw:.1f} ms, device {ad:.1f} ms",
            file=out,
        )

        hdr("MODULE BREAKDOWN (FORWARD, wall time from CUDA events)")
        if mods:
            print(f"{'module':<46}{'wall_ms':>12}{'calls':>8}", file=out)
            print("-" * 104, file=out)
            for k, v in sorted(mods.items(), key=lambda kv: -kv[1]["wall_ms"])[
                : self.top_n
            ]:
                print(f"{k[:44]:<46}{v['wall_ms']:12.3f}{v['calls']:8.0f}", file=out)
            print(
                f"(showing top {self.top_n} of {len(mods)} instrumented; "
                f"the tower has ~794 features, capped at "
                f"{self.max_modules})",
                file=out,
            )
        else:
            print("no modules instrumented", file=out)

        hdr("TOP OPS BY HOST TIME (CPU overhead)")
        print(f"{'op':<58}{'cpu_ms':>10}{'dev_ms':>10}{'n':>8}", file=out)
        print("-" * 104, file=out)
        for k, v in sorted(ops.items(), key=lambda kv: -kv[1]["cpu"])[: self.top_n]:
            print(
                f"{k[:56]:<58}{v['cpu']:10.3f}{v['dev']:10.3f}{v['count']:8.0f}",
                file=out,
            )

        hdr("TOP OPS BY DEVICE TIME")
        print(
            "Every op of this tower, not only the memory-class ones: that class "
            "was decided by matching op names, which is a guess. time% is the "
            "share of this tower's total device time.",
            file=out,
        )
        print(f"\n{'op':<58}{'dev_ms':>10}{'time%':>8}{'calls':>8}", file=out)
        print("-" * 104, file=out)
        top_ops = [(k, v) for k, v in ops.items() if v["dev"] > 0]
        for k, v in sorted(top_ops, key=lambda kv: -kv[1]["dev"])[: self.top_n]:
            print(
                f"{k[:56]:<58}{v['dev']:10.3f}"
                f"{(v['dev'] / tot_dev * 100) if tot_dev else 0:7.1f}%"
                f"{v['count']:8.0f}",
                file=out,
            )

        self._write_traffic(out)

        # ---- communication ----
        payload = [r for r in recs if not r["is_meta"]]
        meta = [r for r in recs if r["is_meta"]]

        hdr("ALL2ALL SUMMARY")
        print(
            f"{'class':<22}{'calls':>8}{'volume/step':>12}"
            f"{'kernel_ms':>12}{'GB/s':>10}",
            file=out,
        )
        print("-" * 104, file=out)
        n_steps = max(self._steps, 1)
        for label, group in (("payload", payload), ("metadata", meta)):
            vols = [r["volume"] for r in group if r["volume"]]
            tv = (sum(vols) / n_steps) if vols else None
            tk = sum(r["kernel_ms"] for r in group) / n_steps
            # bandwidth only over calls whose kernel time is known, otherwise the
            # missing denominators inflate it (this produced 14.6 TB/s once)
            timed = [r for r in group if r["volume"] and r["kernel_ms"] > 0]
            tv_t = sum(r["volume"] for r in timed) / n_steps
            tk_t = sum(r["kernel_ms"] for r in timed) / n_steps
            bw = (tv_t / (tk_t / 1e3) / 1e9) if tk_t > 0 else None
            print(
                f"{label:<22}{len(group):8d}{_fmt_bytes(tv):>12}"
                f"{tk:12.3f}{(f'{bw:.2f}' if bw else 'n/a'):>10}"
                f"   ({len(timed)}/{len(group)} calls timed)",
                file=out,
            )
        print("-" * 104, file=out)
        print(
            "volume = sizeof(dtype) * (sum(in_splits) + sum(out_splits)) / 2, "
            "i.e. one-way volume when balanced; dtype comes from the op's "
            "Input type, never from the kernel name (PCCL SendRecv always "
            "says int8_t regardless of payload)",
            file=out,
        )
        print(
            "GB/s uses kernel device time, so it is link bandwidth, not link-plus-wait",
            file=out,
        )
        ck = self._comm_kernel_ms()
        acc = sum(r["kernel_ms"] for r in recs) / max(self._steps, 1)
        print(
            f"all2all kernel time: {ck:.3f} ms/step total by kernel name, "
            f"{acc:.3f} ms/step attributed to individual calls"
            f"{' -- the gap is unattributed' if abs(ck - acc) > 1e-3 else ''}",
            file=out,
        )

        hdr("ALL2ALL BY VOLUME (TOP)")
        self._print_comm_table(payload, key=lambda r: -(r["volume"] or 0), file=out)

        hdr("ALL2ALL BY BANDWIDTH (WORST FIRST)")

        def bw_of(r):
            if r["volume"] and r["kernel_ms"] > 0:
                return r["volume"] / (r["kernel_ms"] / 1e3) / 1e9
            return float("inf")

        self._print_comm_table(payload, key=bw_of, file=out)

        if meta:
            hdr("ALL2ALL METADATA EXCHANGES")
            ns = max(self._steps, 1)
            print(
                f"count {len(meta)} over {ns} steps, kernel "
                f"{sum(r['kernel_ms'] for r in meta) / ns:.3f} ms/step, host "
                f"{sum(r['op_ms'] for r in meta) / ns:.3f} ms/step",
                file=out,
            )
            print(
                "These are ranks swapping lengths before the real transfer. "
                "Their count is itself an optimization signal: many small "
                "exchanges may be batchable.",
                file=out,
            )

        if self._parsed and "error" in self._parsed:
            print(
                f"\nWARNING: trace parsing failed "
                f"({self._parsed['error']}); communication tables are empty",
                file=out,
            )
        print(f"\n{'=' * 104}\n", file=out)

    def _write_traffic(self, out):
        """Memory traffic and bandwidth for the ops recis tracks.

        Only ops with a registered byte formula appear. That is deliberate: joining
        this against the kernel-level table would need a name mapping (one op
        launches several kernels, and the CUPTI names differ from the op names),
        and a bad mapping would be worse than a narrower table.
        """

        def hdr(t):
            print(f"\n{'*' * 104}", file=out)
            print(f"* {t}", file=out)
            print(f"{'*' * 104}", file=out)

        hdr("MEMORY ACCESS (recis-tracked ops only)")
        tr = self._traffic
        if not tr or not tr.get("ops"):
            print(
                "no data. recis memory-access tracking did not run for this "
                "window -- either it was disabled (track_memory_access=False) "
                "or another component already held the tracker's context.",
                file=out,
            )
            return

        n = max(self._steps, 1)
        ops = tr["ops"]
        total_bytes = tr.get("total_bytes", 0) / n
        total_time = tr.get("total_time_ms", 0.0) / n
        avg_bw = (total_bytes / (total_time / 1e3) / 1e9) if total_time else 0.0

        print(
            "bytes are computed from per-op formulas over tensor shapes: the "
            "ideal traffic, NOT measured DRAM traffic, so cache hits are not "
            "reflected and the figure is an upper bound on what the memory "
            "system actually moved.",
            file=out,
        )
        print(
            "time is the op's own device time, from CUDA events around the "
            "call, so it includes gaps between that op's kernels. It is a "
            "different quantity from the kernel time in the table above.",
            file=out,
        )
        print(
            "coverage: only ops with a registered formula. aten ops such as "
            "copy_ and cat are not tracked, so the total below is a lower "
            "bound on this tower's real traffic.",
            file=out,
        )

        print(
            f"\n{'total tracked traffic':<30}{_fmt_bytes(total_bytes):>14}  per step",
            file=out,
        )
        print(f"{'total tracked op time':<30}{total_time:14.3f}  ms per step", file=out)
        print(f"{'average bandwidth':<30}{avg_bw:14.2f}  GB/s", file=out)

        rows = []
        for name, st in ops.items():
            b = st.get("bytes", 0) / n
            t = st.get("time_ms", 0.0) / n
            bw = (b / (t / 1e3) / 1e9) if t > 0 else 0.0
            rows.append((name, b, t, bw, st.get("count", 0) / n))

        def table(title, key, reverse):
            print(f"\n{title}", file=out)
            print(
                f"{'op':<40}{'traffic':>12}{'time_ms':>10}{'GB/s':>10}"
                f"{'calls':>8}{'%traffic':>10}",
                file=out,
            )
            print("-" * 104, file=out)
            for name, b, t, bw, c in sorted(rows, key=key, reverse=reverse)[
                : self.top_n
            ]:
                print(
                    f"{name[:38]:<40}{_fmt_bytes(b):>12}{t:10.3f}{bw:10.2f}"
                    f"{c:8.0f}"
                    f"{(b / total_bytes * 100) if total_bytes else 0:9.1f}%",
                    file=out,
                )

        table("top ops by traffic:", lambda r: r[1], True)
        table("top ops by bandwidth, best first:", lambda r: r[3], True)
        table(
            "top ops by bandwidth, worst first (optimization candidates):",
            lambda r: r[3],
            False,
        )

    def _print_comm_table(self, recs, key, file):
        print(
            f"{'#':<4}{'volume':>12}{'dtype':>16}{'in_elems':>12}"
            f"{'out_elems':>12}{'kernel_ms':>11}{'GB/s':>10}"
            f"   (per call, not per step)",
            file=file,
        )
        print("-" * 104, file=file)
        if not recs:
            print("  none (world_size=1 skips real payload exchange)", file=file)
            return
        for i, r in enumerate(sorted(recs, key=key)[: self.top_n], 1):
            bw = (
                r["volume"] / (r["kernel_ms"] / 1e3) / 1e9
                if r["volume"] and r["kernel_ms"] > 0
                else None
            )
            print(
                f"{i:<4}{_fmt_bytes(r['volume']):>12}{r['dtype'][:14]:>16}"
                f"{r['in_elems']:12d}{r['out_elems']:12d}"
                f"{r['kernel_ms']:11.3f}"
                f"{(f'{bw:.2f}' if bw else 'n/a'):>10}",
                file=file,
            )
