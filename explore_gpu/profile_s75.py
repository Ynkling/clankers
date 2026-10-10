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
  - peak memory: the resident-memory high-water mark added by one update (forward, backward, Adam) in a fresh process
    (ru_maxrss; the CUDA caching allocator's peak for the same tensors is close to it), the same for one evaluation
    forward of 32 sequences, and the bytes autograd saves for backward (saved_tensors_hooks, distinct storages);
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

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

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
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True, profile_memory=True) as prof:
        step()
    ops, flops, bytes_, launches = [], 0.0, 0.0, 0
    mem_events = []
    for e in prof.events():
        if e.name == "[memory]":
            mem_events.append((e.time_range.start, e.cpu_memory_usage))
            continue
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
    mem_events.sort()
    cur, peak = 0, 0
    for _, d in mem_events:
        cur += d
        peak = max(peak, cur)
    return dict(arm=arm, cpu_secs=statistics.median(times), flops=flops, bytes=bytes_, launches=launches,
                peak_gb=peak / 2 ** 30, act_gb=act_gb, ops=ops)


MEM_PROG = r"""
import sys, resource, json, torch
sys.path.insert(0, %r)
torch.set_num_threads(1)
import torch.nn.functional as F
import explore_i_task as it
from explore_gpu import run as gr
from explore_gpu.batched import BatchedBDH, StackedAdam
task = it.ICMCTask(**%r)
spec = gr.arm_spec(%r, task.S)
torch.manual_seed(830)
m = gr.make_model(spec, task, 830)
bm = BatchedBDH([m], "cpu", register="scan")
opt = StackedAdam([dict(params=[p for p in bm.ps if p.requires_grad], lr=1e-3)], 1)
tok = task.make_batch(gr.BATCH, torch.Generator().manual_seed(1))[0].unsqueeze(0)
mask = gr.icmc_loss_mask(task, tok[0]).unsqueeze(0)
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
lb, _ = bm(tok[..., :-1])
ce = F.cross_entropy(lb.reshape(-1, lb.shape[-1]), tok[..., 1:].reshape(-1), reduction="none").view(mask.shape)
mf = mask.float()
loss = ((ce * mf).sum((1, 2)) / mf.sum((1, 2))).sum()
loss.backward()
opt.step()
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
with torch.no_grad():
    del lb, ce, loss
    import gc; gc.collect()
    e0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps(dict(step_gb=(peak - base) / 2 ** 20)))
"""


def mem_of(arm):
    """Peak resident memory added by one training update (forward, backward, Adam), in a fresh process (ru_maxrss)."""
    import subprocess
    out = subprocess.run([sys.executable, "-c", MEM_PROG % (HERE, CFG, arm)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])["step_gb"]


def eval_mem_of(arm, n=32):
    """Peak resident memory of one evaluation forward of n sequences (no gradients), in a fresh process."""
    import subprocess
    prog = MEM_PROG.split("base = ")[0] + (
        "base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "x = task.make_batch(%d, torch.Generator().manual_seed(2))[0][:, :-1]\n"
        "with torch.no_grad():\n    m.eval(); m(x)\n"
        "print(json.dumps(dict(step_gb=(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - base) / 2 ** 20)))\n" % n)
    out = subprocess.run([sys.executable, "-c", prog % (HERE, CFG, arm)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])["step_gb"]


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
        res[a]["peak_gb"] = mem_of(a)
        res[a]["eval_gb"] = eval_mem_of(a)
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
          "Memory per run: the update's peak resident memory (fresh process) x 1.15; one evaluation chunk of 32 "
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
    L += ["", "## Evaluation", "",
          f"The CPU part of one evaluation (explore_gpu.run.eval_cpu_part: routing on 256 sequences with the float64 margin, "
          f"the probe's four logistic regressions, SET accuracy) took {ev['WIDE_SPLIT']:.0f} s (WIDE_SPLIT) and "
          f"{ev['REG3']:.0f} s (REG3) here, 1 thread. On the GPU path it runs in --stats-workers CPU processes (default 3) "
          f"while training continues: one evaluation per run every 1200 updates, i.e. every "
          f"{1200 * min(gpu_time(res['SINGLE'], g, 0) for g in GPUS.values()):.0f} s or more per run, so three workers keep up "
          f"(the probe needs {cpu_eval:.0f} s / 3 workers). The evaluation forwards (1024 + 256 + 256 sequences, forward only, "
          f"~1/3 of an update per 32 sequences) cost ~16 updates' worth of GPU time per evaluation: 20 evaluations add "
          f"~1.3% to a run."]
    L += ["", "## S75 at S = 4, all 32 runs", ""]
    for gname, (lo, hi) in tot.items():
        L.append(f"- **{gname}**: {lo:.0f} - {hi:.0f} GPU hours of training (+1.3% for the evaluation forwards; the "
                 f"evaluation statistics run on the CPU alongside). Stacking R runs costs about the same GPU time per run "
                 f"(L = 768 already fills the GPU); it hides Python and launch overhead.")
    L += ["", "Caveats: the bracket is wide because consumer-GPU SGEMM efficiency on these shapes (batched 767 x 767 x 256 "
          "GEMMs) and the eager-mode launch overhead are not known here; the peak-memory column is the CPU allocator's "
          "peak, which the CUDA caching allocator matches to within its block rounding; K9 on the GPU replaces all of this "
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
