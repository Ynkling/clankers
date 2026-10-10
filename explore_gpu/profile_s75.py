#!/usr/bin/env python
"""
explore_gpu/profile_s75.py — the runtime estimate for S75 at S = 4 on a 6 GB and an 8 GB GPU, from a CPU profile of one
training update. EXPLORATORY tooling; an estimate, not a measurement of any GPU (explore_gpu/checks.py K9 measures it on
the user's GPU).

    python -m explore_gpu.profile_s75          -> explore_gpu/runtime_estimate.md and explore_gpu/profile_s75.json

WHAT IS MEASURED (CPU, 1 thread, the batched path with R = 1 and the register's "scan" form, i.e. what runs on CUDA), for
each S75 arm (ORACLE, SINGLE, WIDE_SPLIT, REG3) at the S75 configuration (ICMC S = 4, V = 4, NB = 16, 6..14, M = 176,
L = 768; B = 32; N = 256, D = 32, 3 layers, k = 4):
  - per ATen operation of one update (forward, backward, StackedAdam step), from torch.profiler with record_shapes and
    with_flops: its FLOPs (matmuls: 2 m n k; elementwise and reductions: counted as memory traffic only) and the bytes it
    reads and writes (input shapes recorded by the profiler, outputs taken as the largest input for element-wise /
    reduction ops and m x n for matmuls; 4 bytes per element);
  - the number of kernel-launching operations (ATen ops that are not views / metadata);
  - peak memory of one update of one run: the parameters, gradients and Adam state plus the peak of the tensors the
    forward, backward and Adam step allocate, counted storage by storage as they are created and freed (LiveBytes: what
    the CUDA caching allocator must hold for the same tensors, before block rounding); the same for one evaluation
    forward of 32 sequences; and the bytes autograd saves for backward (saved_tensors_hooks, distinct storages);
  - the CPU time of one update (median of 3), and of one evaluation's CPU part (explore_gpu.run.eval_cpu_part).
THE GPU MODEL (stated assumptions; low / high bracket): per op, time = max(FLOPs / (peak FLOP/s x e_gemm), bytes / (memory
bandwidth x e_bw)), plus one launch per kernel-launching op (8 us low, 15 us high; PyTorch eager on Linux). GTX 1660 SUPER
(6 GB): 5.03 TFLOP/s FP32 peak (1408 cores x 1.785 GHz x 2), 336 GB/s. RTX 3070 Laptop (8 GB): 15.97 TFLOP/s at the
highest boost (5120 x 1.56 GHz x 2; 80-125 W variants run 1.29-1.62 GHz), 448 GB/s (14 Gbps x 256 bit; 384 GB/s on 12 Gbps
parts). e_gemm 0.70 (Turing) / 0.50 (Ampere, whose doubled FP32 rate SGEMM does not reach) in the low bracket, 0.45 / 0.30
in the high; e_bw 0.75 low, 0.55 high. Memory: the update's peak plus 0.6 GB CUDA context and 15% allocator slack per
GPU, evaluation in chunks of 32 sequences (explore_gpu.run.EVAL_CHUNK) adding at most one chunk's forward.
"""

import json
import math
import os
import statistics
import sys
import time
import weakref

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

import explore_i_task as it
import explore_i_common as ic
from explore_gpu import run as gr
from explore_gpu.batched import BatchedBDH, StackedAdam

CFG = dict(S=4, V=4, NB=16, lmin=6, lmax=14, M=176)
ARMS = ("ORACLE", "SINGLE", "WIDE_SPLIT", "REG3")
RUNS = dict(ORACLE=2, SINGLE=10, WIDE_SPLIT=10, REG3=10)
ITERS, EVALS = 24000, 20
GPUS = {
    "GTX 1660 SUPER (6 GB)": dict(mem=6.0, flops=5.03e12, bw=336e9, e_gemm=(0.70, 0.45)),
    "RTX 3070 Laptop (8 GB)": dict(mem=8.0, flops=15.97e12, bw=448e9, e_gemm=(0.50, 0.30)),
}
E_BW = (0.75, 0.55)
LAUNCH = (8e-6, 15e-6)
CONTEXT_GB, SLACK = 0.6, 1.15
VIEW_OPS = {"aten::view", "aten::reshape", "aten::_reshape_alias", "aten::expand", "aten::as_strided", "aten::select",
            "aten::slice", "aten::unsqueeze", "aten::squeeze", "aten::transpose", "aten::t", "aten::permute",
            "aten::detach", "aten::alias", "aten::unbind", "aten::split", "aten::chunk", "aten::lift_fresh",
            "aten::empty", "aten::empty_like", "aten::empty_strided", "aten::resize_", "aten::_unsafe_view",
            "aten::numpy_T", "aten::mT", "aten::contiguous", "aten::result_type", "aten::item", "aten::_local_scalar_dense",
            "aten::is_nonzero", "aten::size", "aten::stride", "aten::is_same_size", "aten::set_", "aten::new_empty",
            "aten::new_empty_strided", "aten::view_as", "aten::reshape_as", "aten::expand_as", "aten::squeeze_",
            "aten::unsqueeze_", "aten::to", "aten::_to_copy"}
MATMULS = {"aten::mm", "aten::bmm", "aten::addmm", "aten::baddbmm", "aten::matmul", "aten::linalg_solve_triangular"}


def numel(shape):
    return int(math.prod(shape)) if shape else 1


def one_update_parts(bm, opt, tok, mask):
    def step():
        lb, _ = bm(tok[..., :-1])
        ce = F.cross_entropy(lb.reshape(-1, lb.shape[-1]), tok[..., 1:].reshape(-1), reduction="none").view(mask.shape)
        mf = mask.float()
        loss = ((ce * mf).sum((1, 2)) / mf.sum((1, 2))).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return step


def profile_arm(arm, task):
    spec = gr.arm_spec(arm, task.S)
    torch.manual_seed(830)
    m = gr.make_model(spec, task, 830)
    bm = BatchedBDH([m], "cpu", register="scan")
    opt = StackedAdam([dict(params=[p for p in bm.ps if p.requires_grad], lr=1e-3)], 1)
    tok = task.make_batch(gr.BATCH, torch.Generator().manual_seed(1))[0].unsqueeze(0)
    mask = gr.icmc_loss_mask(task, tok[0]).unsqueeze(0)
    step = one_update_parts(bm, opt, tok, mask)
    step()                                                   # warm-up (Adam state allocated)
    times = []
    for _ in range(3):
        t0 = time.time()
        step()
        times.append(time.time() - t0)
    # saved activations
    saved = {}

    def pack(t):
        try:
            key = t.untyped_storage().data_ptr()
            saved[key] = max(saved.get(key, 0), t.untyped_storage().nbytes())
        except Exception:
            pass
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        lb, _ = bm(tok[..., :-1])
    del lb
    act_gb = sum(saved.values()) / 2 ** 30
    # profile
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
        step()
    ops, flops, bytes_, launches = [], 0.0, 0.0, 0
    for e in prof.events():
        if not e.name.startswith("aten::") or e.name in VIEW_OPS:
            continue
        if e.cpu_parent is not None and e.cpu_parent.name.startswith("aten::") and e.cpu_parent.name not in VIEW_OPS:
            continue                                         # count top-level ATen ops only (their children are internal)
        shapes = [s for s in (e.input_shapes or []) if isinstance(s, (list, tuple)) and s and all(isinstance(x, int) for x in s)]
        fl = float(e.flops or 0)
        ins = sum(numel(s) for s in shapes) * 4
        if e.name in MATMULS and len(shapes) >= 2 and len(shapes[0]) >= 2 and len(shapes[1]) >= 2:
            a, b = shapes[0], shapes[1]
            outn = numel(a[:-1]) * b[-1]
            if fl == 0:
                fl = 2.0 * numel(a) * b[-1]
        else:
            outn = max((numel(s) for s in shapes), default=0)
        by = ins + outn * 4
        ops.append(dict(name=e.name, flops=fl, bytes=by))
        flops += fl
        bytes_ += by
        launches += 1
    return dict(arm=arm, cpu_secs=statistics.median(times), flops=flops, bytes=bytes_, launches=launches,
                act_gb=act_gb, ops=ops)


class LiveBytes(TorchDispatchMode):
    """Bytes held by live tensor storages created inside the mode: every ATen op's outputs (forward, autograd's backward
    on this thread, the optimiser), each storage counted once and released when it is freed. This is what the CUDA
    caching allocator must hold for the same program (before its block rounding and fragmentation: SLACK)."""

    def __init__(self):
        super().__init__()
        self.cur, self.peak, self.seen = 0, 0, {}

    def _add(self, x):
        st = x.untyped_storage()
        k = id(st)
        if k in self.seen:
            return
        n = st.nbytes()
        self.seen[k] = n
        self.cur += n
        self.peak = max(self.peak, self.cur)

        def free(k=k, n=n):
            if self.seen.pop(k, None) is not None:
                self.cur -= n
        weakref.finalize(st, free)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for x in tree_flatten(out)[0]:
            if isinstance(x, torch.Tensor) and x.device.type != "meta":
                self._add(x)
        return out


def _storages_gb(ts):
    seen = {}
    for t in ts:
        if t is not None:
            st = t.untyped_storage()
            seen[id(st)] = st.nbytes()
    return sum(seen.values()) / 2 ** 30


def mem_of(arm, task):
    """Peak memory of one training update of one run: the parameters, their gradients and the Adam state (resident across
    updates, after a warm-up update) plus the peak of the tensors that the forward, the backward and the Adam step
    allocate (LiveBytes), in GB."""
    torch.manual_seed(830)
    m = gr.make_model(gr.arm_spec(arm, task.S), task, 830)
    bm = BatchedBDH([m], "cpu", register="scan")
    opt = StackedAdam([dict(params=[p for p in bm.ps if p.requires_grad], lr=1e-3)], 1)
    tok = task.make_batch(gr.BATCH, torch.Generator().manual_seed(1))[0].unsqueeze(0)
    mask = gr.icmc_loss_mask(task, tok[0]).unsqueeze(0)
    step = one_update_parts(bm, opt, tok, mask)
    step()                                                   # Adam state allocated
    st = [v for s in opt.state.values() for v in s.values() if isinstance(v, torch.Tensor)]
    resident = _storages_gb(list(bm.ps) + [p.grad for p in bm.ps] + st + [tok, mask])
    with LiveBytes() as lb:
        step()
    return resident + lb.peak / 2 ** 30


def eval_mem_of(arm, task, n=32):
    """Peak memory of one evaluation forward of n sequences (no gradients; explore_gpu.run.EVAL_CHUNK = 32), in GB."""
    torch.manual_seed(830)
    m = gr.make_model(gr.arm_spec(arm, task.S), task, 830)
    x = task.make_batch(n, torch.Generator().manual_seed(2))[0][:, :-1]
    m.eval()
    with torch.no_grad(), LiveBytes() as lb:
        m(x)
    return lb.peak / 2 ** 30


def eval_cpu_side(task):
    """CPU seconds of one evaluation's statistics (explore_gpu.run.eval_cpu_part: loss, SET accuracy, routing on 256
    sequences with the float64 margin, the probe's four logistic regressions on 256 sequences), given the device part's
    tensors, for the WIDE_SPLIT (one gate layer) and REG3 (three) models."""
    out = {}
    for arm in ("WIDE_SPLIT", "REG3"):
        torch.manual_seed(830)
        m = gr.make_model(gr.arm_spec(arm, task.S), task, 830)
        ev, pdata = ic.eval_set(task), ic.probe_set(task)
        lg, gates, feats = gr.eval_device_part(m, task, ev, pdata, True)
        t0 = time.time()
        gr.eval_cpu_part(task, ev, pdata, lg, gates, feats)
        out[arm] = time.time() - t0
    return out


def gpu_time(res, g, which):
    e_g = g["e_gemm"][which]
    e_b = E_BW[which]
    t = 0.0
    for o in res["ops"]:
        t += max(o["flops"] / (g["flops"] * e_g), o["bytes"] / (g["bw"] * e_b))
    return t + res["launches"] * LAUNCH[which]


def main():
    torch.set_num_threads(1)
    task = it.ICMCTask(**CFG)
    res = {a: profile_arm(a, task) for a in ARMS}
    for a in ARMS:
        res[a]["peak_gb"] = mem_of(a, task)
        res[a]["eval_gb"] = eval_mem_of(a, task)
    ev = eval_cpu_side(task)
    out = dict(config=task.label, arms={a: {k: v for k, v in r.items() if k != "ops"} for a, r in res.items()}, eval=ev)
    L = ["# S75 at S = 4 on a GPU: runtime estimate (from a CPU profile; EXPLORATORY)", "",
         f"Configuration: {task.label}; B = 32; N = 256, D = 32, 3 layers, k = 4; 24,000 updates, 20 evaluations; "
         "runs: ORACLE 2, SINGLE 10, WIDE_SPLIT 10, REG3 10 (32 runs). Method and assumptions: explore_gpu/profile_s75.py's "
         "docstring. `python -m explore_gpu.checks --device cuda` (K9) measures the real numbers on the GPU.", "",
         "## One training update, per run (CPU profile)", "",
         "| arm | GFLOP | GB moved | kernel launches | peak memory GB (update) | eval chunk GB | saved activations GB | "
         "CPU s/update |", "|---|---|---|---|---|---|---|---|"]
    for a, r in res.items():
        L.append(f"| {a} | {r['flops'] / 1e9:.1f} | {r['bytes'] / 1e9:.2f} | {r['launches']} | {r['peak_gb']:.2f} | "
                 f"{r['eval_gb']:.2f} | {r['act_gb']:.2f} | {r['cpu_secs']:.2f} |")
    L += ["", "## Estimated GPU time per update and memory", "",
          "Memory per run: the update's peak (parameters, gradients, Adam state and the update's live tensors) x 1.15 "
          "for the allocator's rounding and fragmentation; one evaluation chunk of 32 "
          "sequences beside it; 0.6 GB CUDA context.", "",
          "| GPU | arm | s/update (low - high) | runs that fit at once (R) | 24k updates, one run (h, low - high) |",
          "|---|---|---|---|---|"]
    tot = {}
    for gname, g in GPUS.items():
        hours = [0.0, 0.0]
        for a, r in res.items():
            tl, th = gpu_time(r, g, 0), gpu_time(r, g, 1)
            per_run = r["peak_gb"] * SLACK
            eval_chunk = r["eval_gb"] * SLACK               # one 32-sequence evaluation forward (no gradients)
            R = max(0, int((g["mem"] - CONTEXT_GB - eval_chunk) // per_run))
            L.append(f"| {gname} | {a} | {tl:.3f} - {th:.3f} | {R} | {tl * ITERS / 3600:.1f} - {th * ITERS / 3600:.1f} |")
            for i, t in enumerate((tl, th)):
                hours[i] += RUNS[a] * ITERS * t / 3600
            out.setdefault("gpu", {}).setdefault(gname, {})[a] = dict(low=tl, high=th, R=R, per_run_gb=per_run)
        tot[gname] = hours
    cpu_eval = max(ev.values())
    t_min = min(gpu_time(res[a], g, 0) for a in ARMS for g in GPUS.values())
    L += ["", "## Evaluation", "",
          f"The CPU part of one evaluation (explore_gpu.run.eval_cpu_part: routing on 256 sequences with the float64 margin, "
          f"the probe's four logistic regressions, SET accuracy) took {ev['WIDE_SPLIT']:.0f} s (WIDE_SPLIT) and "
          f"{ev['REG3']:.0f} s (REG3) of one CPU core here (this container's Xeon at 2.1 GHz, shared with other jobs). On the "
          f"GPU path it runs in --stats-workers CPU processes (default 3) while training continues. A run is evaluated every "
          f"1200 updates, i.e. every {1200 * t_min:.0f} s or more of GPU time per run (R stacked runs: R evaluations every R "
          f"times as long), so the workers keep up while {cpu_eval:.0f} s / workers < {1200 * t_min:.0f} s: three workers "
          f"do; on a slower CPU, add one. The evaluation forwards (1024 + 256 + 256 sequences, forward only, ~1/3 of an "
          f"update per 32 sequences) cost ~16 updates' worth of GPU time per evaluation: 20 evaluations add ~1.3% to a run."]
    L += ["", "## S75 at S = 4, all 32 runs", ""]
    for gname, (lo, hi) in tot.items():
        L.append(f"- **{gname}** alone: {lo * 1.013:.0f} - {hi * 1.013:.0f} GPU hours (training, plus 1.3% for the "
                 f"evaluation forwards; the evaluation statistics run on the CPU alongside).")
    split = {}
    for gname, grp in (("RTX 3070 Laptop (8 GB)", ("WIDE_SPLIT", "REG3")), ("GTX 1660 SUPER (6 GB)", ("ORACLE", "SINGLE"))):
        v = out["gpu"][gname]
        split[gname] = [sum(RUNS[a] * ITERS * v[a][k] / 3600 for a in grp) * 1.013 for k in ("low", "high")]
        L.append(f"- Both, by arms (README (b)): the 8 GB laptop `--arms WIDE_SPLIT,REG3`, the 6 GB server "
                 f"`--arms ORACLE,SINGLE`: {split[gname][0]:.1f} - {split[gname][1]:.1f} h on the {gname}"
                 if gname.startswith("RTX") else
                 f"  and {split[gname][0]:.1f} - {split[gname][1]:.1f} h on the {gname}, at the same time.")
    out["two_machines"] = split
    rmin = {gname: min(v[a]["R"] for a in ARMS) for gname, v in out["gpu"].items()}
    L += ["", f"Stacking R runs costs about the same GPU time per run (L = 768 already fills the GPU); it hides Python and "
          f"launch overhead (~{min(r['launches'] for r in res.values())}-{max(r['launches'] for r in res.values())} "
          f"launches per update, a few ms of {1000 * t_min:.0f}+ ms). explore_gpu/s75_s4.py's default --runs-per-batch is "
          f"half the smallest R that fits above, rounded down, for margin (a display on the GPU, fragmentation beyond 15%, "
          f"cuBLAS workspaces): " + ", ".join(f"{max(1, r // 2)} on the {g} (R fits {r})" for g, r in rmin.items()) + "."]
    L += ["", "Caveats: the bracket is wide because consumer-GPU SGEMM efficiency on these shapes (batched 767 x 767 x 256 "
          "GEMMs) and the eager-mode launch overhead are not known here; the peak-memory column counts live tensor bytes "
          "(the CUDA caching allocator adds block rounding and fragmentation, taken as 15%); K9 on the GPU replaces all of this "
          "with measurements in a minute."]
    out["hours"] = tot
    txt = "\n".join(L) + "\n"
    print(txt)
    with open(os.path.join(HERE, "explore_gpu", "runtime_estimate.md"), "w") as f:
        f.write(txt)
    with open(os.path.join(HERE, "explore_gpu", "profile_s75.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
