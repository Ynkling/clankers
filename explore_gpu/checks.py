#!/usr/bin/env python
"""
explore_gpu/checks.py — the GPU package's CHECKs. EXPLORATORY tooling, not a result.

    python -m explore_gpu.checks --device cpu      (run here; output committed: explore_gpu/checks_cpu_output.txt / .json)
    python -m explore_gpu.checks --device cuda     (the user's CUDA CHECK; writes explore_gpu/checks_cuda_<gpu>.txt / .json)

WHY TWO KINDS OF CHECK (measured here, CHECK K0): training is chaotic. Arm A seed 160 (X's recorded run) with every initial
parameter multiplied by (1 + u), |u| <= 1.2e-7 (one float32 ulp), differs from the unperturbed run by 2.9e-3 in its logits
after 100 updates and by 1.5-10 (O(1)) from update 200 on. Any difference in rounding between two paths therefore grows to
O(1) within ~200 updates: two free-running paths agree to 1e-5 for 2400 updates only if they are bit for bit identical.
  - FREE-RUNNING (as specified): on CPU the batched path is bit for bit the sequential path for every Hebbian model (the
    stacked operations have the sequential path's own shapes: CHECKed K2, K2b), so the specified check holds exactly
    (asserted: max |dlogits| <= 1e-5; observed 0). On CUDA the batched and sequential paths use different GEMM batchings,
    so they are not bit for bit equal and free-running agreement cannot be a tolerance: it is printed, not asserted.
  - LOCK-STEP (the CUDA CHECK): at EVERY one of 2400 updates the batched path's forward and backward are computed from the
    sequential path's own parameters on the same batch (the sequential path trains; the batched model is re-synchronised
    to it before each update), so the comparison follows the real training trajectory without the chaos. Tolerances,
    stated: same device (batched vs sequential): max |dlogits| <= 1e-5 and gradient max |d| / max |g| <= 1e-4 per
    parameter; CUDA vs CPU (sequential, the CPU trajectory as reference): max |dlogits| <= 1e-4 and gradients <= 1e-3
    relative (cuBLAS and MKL sum in different orders; float32, TF32 off).

CHECKS
  K0  (printed) the chaos calibration above, recomputed (arm A 160, two perturbation sizes, every 100 updates).          cpu
  K1  record reproduction: test_router_reliability.run_one(arm A, 160, 2400, test_short_conv's task, conv_stats and
      conv_grad_norms: test_short_conv.run_job's call) through explore_gpu.device.device_scope equals
      X's recorded A|160 (results/X/short_conv_results.json, Xeon @ 2.10GHz, torch 2.14.0) bit for bit: curve, gradient
      norms at 1, 10, 100, every statistic. Asserted when this CPU is the record's CPU, else printed.                 cpu
      On CUDA (printed, not asserted): the same run's curve beside the record's.                                       cuda
  K2  FREE-RUNNING batched vs sequential, arm A, seeds 160-163 stacked (R = 4): logits on the 2048-query evaluation
      batch at every 100 updates for 2400 updates, every run, max |d| <= 1e-5 (cpu: asserted; cuda: printed).
  K2b the same for B_conv, LOCAL3 on RandHeaderTask-lite, and on ICMC: LOCAL3 + SLOW + split, REG3 (register loop),
      the perfect gate, one channel (600 updates, R = 3): cpu asserted (bit for bit), cuda printed.
  K3  LOCK-STEP batched vs sequential, arm A, seeds 160-163, 2400 updates, every update: logits <= 1e-5, gradients
      <= 1e-4 relative per parameter. Asserted on every device.
  K3b LOCK-STEP batched vs sequential on ICMC (S = 4, L = 768, the S75 configuration) for each S75 arm (ORACLE, SINGLE,
      WIDE_SPLIT, REG3 with the register "scan" on CUDA / "loop" on CPU, WIN3_SPLIT, WIDE_SLOW), 20 updates (cpu) / 300
      (cuda), and for the delta memory (LOCAL3 + delta on RandHeaderTask-lite; one delta channel with learned beta and
      GDN decay), 300 updates. Asserted on every device, same tolerances.
  K4  CUDA vs CPU LOCK-STEP (cuda only): arm A 160, 2400 updates; and each S75 arm at L = 768 for 20 updates: the CUDA
      sequential path's forward/backward from the CPU trajectory's parameters. Asserted, tolerances above.
  K5  the ICMC trainer: explore_gpu.run.run_sequential_icmc equals explore_i_common.run_one bit for bit on CPU (curve,
      split rows, beta) for LOCAL3 + SLOW + split and REG3; the register's "scan" equals its "loop" (forward and gradients
      <= 1e-5 relative); the evaluation split into its device part and its CPU part equals explore_i_common.evaluate bit
      for bit (WIDE, REG3, one channel); the batched trainer with the CPU part in worker processes gives the very curves
      of the in-line evaluation.                                                                                    cpu
  K6  StackedAdam: with equal step counts bit for bit torch.optim.Adam (single-tensor path) over 50 steps; after zeroing
      one run's state (test_window_gate.zero_state semantics) bit for bit a torch Adam whose state was zeroed; inactive
      runs unchanged.                                                                                     cpu (cuda: 1e-6)
  K7  data: the vectorised ICMC loss mask equals ICMCTask.loss_mask (4096 sequences, S = 1, 2, 4); the prefetch process
      yields the very batches of the direct generators.                                                              cpu
  K8  fake-CUDA device audit (torch FakeTensorMode, no GPU needed): the forward of every model class wrapped by
      to_device("cuda"), the batched forward, and the split's noise path run on fake cuda tensors with no CPU tensor
      mixed in (forward only: fake tensors cannot run backward).                                                    cpu
  K9  (cuda only, printed) the S75 S = 4 benchmark: per-update time and peak memory of each S75 arm at L = 768 with
      R = 1 and R = 2 stacked runs, and the projected wall time of S75 at S = 4.
"""

import argparse
import copy
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F

import explore_i_task as it
import explore_i_common as ic
import explore_b16_gates as g16
import explore_delta_mem as dm
import explore_f_tasks as ft
import test_short_conv as tsc
import test_router_reliability as trr
import test_binding_onset as tbo
from test_multilayer_binding import build
from test_channel_binding import ARCH, eval_batch

from explore_gpu import device as dv
from explore_gpu import run as gr
from explore_gpu.batched import BatchedBDH, StackedAdam

REC_FILE = os.path.join(HERE, "results", "X", "short_conv_results.json")
ARM_A = tsc.ARM["A"]
SEEDS4 = (160, 161, 162, 163)
TOL_SAME = dict(logits=1e-5, grad=1e-4)
TOL_DEV = dict(logits=1e-4, grad=1e-3)
FREE_TOL = 1e-5
S75_CFG = dict(S=4, V=4, NB=16, lmin=6, lmax=14, M=176)
RHL = ft.RandHeaderTask(P=4, nb=2, lmin=1, lmax=3)
SMALL = it.ICMCTask(S=2, V=4, NB=3, lmin=6, lmax=14, M=30)


# ── K0 ────────────────────────────────────────────────────────────────────────
def k0_chaos(iters=2400, every=100):
    data = eval_batch(tsc.TASK)[:, :-1]

    def capture(store):
        def check(m, step, d):
            was = m.training
            m.eval()
            with torch.no_grad():
                store[step] = m(data)[0].clone()
            m.train(was)
        return check
    base = {}
    trr.run_one(ARM_A, 160, iters, eval_every=every, run_kw=dict(early_stop=False), check=capture(base))
    out = {}
    for eps in (1.2e-7, 1e-6):
        pert = {}

        def builder(a, seed, eps=eps):
            mk = trr.make_fn(a, seed)

            def make():
                m = mk()
                g = torch.Generator().manual_seed(1)
                with torch.no_grad():
                    for p in m.parameters():
                        p.mul_(1 + eps * (2 * torch.rand(p.shape, generator=g) - 1))
                return m
            return make
        trr.run_one(ARM_A, 160, iters, eval_every=every, run_kw=dict(early_stop=False), check=capture(pert),
                    builder=builder)
        out[str(eps)] = {s: float((pert[s] - base[s]).abs().max()) for s in base}
    return dict(ok=True, printed=True, divergence=out)


# ── K1 ────────────────────────────────────────────────────────────────────────
def k1_record(device):
    rec = json.load(open(REC_FILE))
    old = rec["runs"]["A|160"]
    # exactly as the record was made: test_short_conv.run_job's run_one call (its task, conv_stats, conv_grad_norms)
    new = gr.run_sequential_bind(ARM_A, 160, 2400, device=device, task=tsc.TASK, stats_fn=tsc.conv_stats,
                                 grad_fn=tsc.conv_grad_norms)
    curve_same = new["curve"] == old["curve"][:2]
    grad_same = {k: list(v) for k, v in new["grad"].items()} == {k: list(v) for k, v in old["grad"].items()}
    so = [x for x in old["stats"] if x["step"] != "end" and x["step"] <= 2400]
    sn = [x for x in new["stats"] if x["step"] != "end" and x["step"] <= 2400]
    stats_same = len(so) == len(sn) and all(all(a[k] == b[k] for k in set(a) & set(b)) for a, b in zip(sn, so))
    here = dv.cpu_model()
    asserted = torch.device(device).type == "cpu" and here == rec["meta"]["cpu"]
    ok = curve_same and grad_same and stats_same
    return dict(ok=ok if asserted else True, asserted=asserted, identical=ok, curve=new["curve"], recorded=old["curve"][:2],
                grad_same=grad_same, stats_same=stats_same, record_cpu=rec["meta"]["cpu"], here=here,
                device=new["device"])


# ── K2: free running ──────────────────────────────────────────────────────────
def free_bind(task, builder, seeds, iters, device, every=100, stats_fn=None):
    """Batched vs sequential (each run) logits on the evaluation batch every `every` updates."""
    stats_fn = stats_fn or trr.gate_stats_k
    data = eval_batch(task)[:, :-1]
    seq = {}
    for s in seeds:
        store = seq.setdefault(s, {})

        def check(m, step, d, store=store):
            was = m.training
            m.eval()
            with torch.no_grad():
                store[step] = m(data)[0].clone()
            m.train(was)
        with dv.device_scope(device):
            trr.run_one(dict(ARM_A, key="free"), s, iters, eval_every=every, run_kw=dict(early_stop=False), check=check,
                        stats_fn=stats_fn, task=task, builder=lambda a, s_: builder(s_))
    diffs = {}

    def ex(bm, views, step, active):
        with torch.no_grad():
            diffs[step] = max(float((views[r](data)[0] - seq[s][step]).abs().max()) for r, s in enumerate(seeds))
    gr.run_batched_bind(task, builder, seeds, iters, device=device, eval_every=every, early_stop=False,
                        on_eval_extra=ex, stats_fn=stats_fn)
    return diffs


def free_icmc(task, spec, seeds, iters, device, every=100, register="loop"):
    ev = ic.eval_set(task)
    x = ev["tokens"][:256, :-1]
    seq = {}
    for s in seeds:
        store = seq.setdefault(s, {})

        def ex_seq(m, step, store=store):
            with torch.no_grad():
                store[step] = ic.logits_chunked(m, x)
        gr.run_sequential_icmc(task, spec, s, iters, device=device, split_thr=0.99, with_probe=False, eval_every=every,
                               on_eval_extra=ex_seq)
    diffs = {}

    def ex_b(bm, views, step):
        with torch.no_grad():
            diffs[step] = max(float((ic.logits_chunked(views[r], x) - seq[s][step]).abs().max())
                              for r, s in enumerate(seeds))
    gr.run_batched_icmc(task, spec, seeds, iters, device=device, split_thr=0.99, with_probe=False, eval_every=every,
                        on_eval_extra=ex_b, register=register)
    return diffs


def k2_free(device, iters=2400):
    diffs = free_bind(tsc.TASK, lambda s: trr.make_fn(ARM_A, s), SEEDS4, iters, device)
    mx = max(diffs.values())
    cpu = torch.device(device).type == "cpu"
    return dict(ok=(mx <= FREE_TOL) if cpu else True, asserted=cpu, max=mx, by_step=diffs)


def k2b_free(device, iters=600):
    out = {}
    cases = [
        ("B_conv", lambda: free_bind(tsc.TASK, lambda s: trr.make_fn(tsc.ARM["B_conv"], s), (160, 161, 162), iters, device,
                                     stats_fn=tsc.conv_stats)),
        ("LOCAL3 RandHeaderTask-lite", lambda: free_bind(
            RHL, lambda s: (lambda: g16.to_local(build(RHL, ARM_A, ARCH, s), s)), (160, 161, 162), iters, device)),
        ("ICMC LOCAL3+SLOW+split", lambda: free_icmc(SMALL, dict(gate="win3", k=2, recipe="slow", split="split"),
                                                     (7, 8, 9), iters, device)),
        ("ICMC REG3 (loop)", lambda: free_icmc(SMALL, dict(gate="reg3", k=2, recipe="slow"), (7, 8, 9), iters, device)),
        ("ICMC perfect", lambda: free_icmc(SMALL, dict(gate="perfect", k=2, recipe="adam"), (7, 8, 9), iters, device)),
        ("ICMC single", lambda: free_icmc(SMALL, dict(gate="none", k=1, recipe="adam"), (7, 8, 9), iters, device)),
    ]
    cpu = torch.device(device).type == "cpu"
    for name, f in cases:
        d = f()
        out[name] = dict(max=max(d.values()), by_step=d)
    ok = all(v["max"] <= FREE_TOL for v in out.values()) if cpu else True
    return dict(ok=ok, asserted=cpu, cases=out)


# ── lock-step ─────────────────────────────────────────────────────────────────
def _sync(bm, models):
    with torch.no_grad():
        for n in bm.names:
            bm.P(n).copy_(torch.stack([dict(m.named_parameters())[n].detach().to(bm.device) for m in models]))


def _grad_cmp(bm, models):
    worst = 0.0
    for n in bm.names:
        for r, m in enumerate(models):
            p = dict(m.named_parameters())[n]
            if p.grad is None:
                continue
            gb = bm.P(n).grad
            ref = p.grad.detach().to(gb.device)
            worst = max(worst, float((gb[r] - ref).abs().max() / (ref.abs().max() + 1e-30)))
    return worst


def lockstep(task, make, seeds, iters, dev_seq, dev_b, loss="select", register="auto", slow_gate=None):
    """The sequential models train (Adam 1e-3, or SLOW when slow_gate gives the gate names); before every update the
    batched model is set to their parameters and computes logits and gradients on the same batch. Returns the worst
    logits difference and the worst relative gradient difference over all updates."""
    models = []
    for s in seeds:
        torch.manual_seed(s)
        models.append(make(s))
    cpu_models = [copy.deepcopy(m) for m in models]
    bm = BatchedBDH(cpu_models, dev_b, register=register)
    models = [dv.to_device(m, dev_seq, cpu_io=False) for m in models]
    opts = []
    for m in models:
        if slow_gate is None:
            opts.append(torch.optim.Adam(m.parameters(), lr=1e-3))
        else:
            named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
            g = [p for n, p in named if n in slow_gate]
            rest = [p for n, p in named if n not in slow_gate]
            opts.append(torch.optim.Adam([dict(params=g, lr=1e-3), dict(params=rest, lr=1e-4)], lr=1e-3))
    rngs = [torch.Generator().manual_seed(s + 10_000) for s in seeds]
    wl, wg = 0.0, 0.0
    for step in range(1, iters + 1):
        toks = [task.make_batch(gr.BATCH, g)[0] for g in rngs]
        _sync(bm, models)
        tb = torch.stack(toks).to(dev_b)
        lb, _ = bm(tb[..., :-1])
        lseq = []
        for r, m in enumerate(models):
            t = toks[r].to(dev_seq)
            lo = m(t[:, :-1])[0]
            lseq.append(lo)
            if loss == "select":
                ql, qt, _ = task.select(lo, t[:, 1:], None)
                L = F.cross_entropy(ql, qt)
            else:
                mask = gr.icmc_loss_mask(task, toks[r]).to(dev_seq)
                L = F.cross_entropy(lo[mask], t[:, 1:][mask])
            opts[r].zero_grad()
            L.backward()
        R_, B, T, V = lb.shape
        if loss == "select":
            ql, qt, _ = task.select(lb.reshape(R_ * B, T, V), tb[..., 1:].reshape(R_ * B, T), None)
            lr_ = F.cross_entropy(ql, qt, reduction="none").view(R_, -1).mean(1)
        else:
            mask = torch.stack([gr.icmc_loss_mask(task, t) for t in toks]).to(dev_b)
            ce = F.cross_entropy(lb.reshape(-1, V), tb[..., 1:].reshape(-1), reduction="none").view(R_, B, T)
            mf = mask.to(ce.dtype)
            lr_ = (ce * mf).sum((1, 2)) / mf.sum((1, 2))
        for p in bm.ps:
            p.grad = None
        lr_.sum().backward()
        wl = max(wl, max(float((lb[r].detach() - lseq[r].detach().to(lb.device)).abs().max()) for r in range(R_)))
        wg = max(wg, _grad_cmp(bm, models))
        for o in opts:
            o.step()
    return wl, wg


def k3_lockstep(device, iters=2400):
    dev = dv.setup(device)
    wl, wg = lockstep(tsc.TASK, lambda s: trr.make_fn(ARM_A, s)(), SEEDS4, iters, dev, dev)
    ok = wl <= TOL_SAME["logits"] and wg <= TOL_SAME["grad"]
    return dict(ok=ok, logits=wl, grad=wg, tol=TOL_SAME, updates=iters)


def k3b_lockstep(device, iters_icmc=None, iters_delta=300):
    dev = dv.setup(device)
    cuda = dev.type == "cuda"
    iters_icmc = iters_icmc or (300 if cuda else 20)
    task = it.ICMCTask(**S75_CFG)
    out = {}
    for arm in ("ORACLE", "SINGLE", "WIDE_SPLIT", "REG3", "WIN3_SPLIT", "WIDE_SLOW"):
        spec = gr.arm_spec(arm, task.S)
        slow = ic.gate_names(gr.make_model(spec, task, 1)) if spec["recipe"] == "slow" else None
        wl, wg = lockstep(task, lambda s, spec=spec: gr.make_model(spec, task, s), (830, 831), iters_icmc, dev, dev,
                          loss="icmc", slow_gate=slow)
        out[arm] = dict(logits=wl, grad=wg, updates=iters_icmc)
    for name, mk in (("LOCAL3 + delta (beta 1)", lambda s: dm.to_delta(g16.to_local(build(RHL, ARM_A, ARCH, s), s), beta=1.0)),
                     ("one delta channel, learned beta, GDN decay",
                      lambda s: dm.to_delta(build(RHL, tsc.ARM["B"], ARCH, s), beta="learned", decay="gdn", seed=s))):
        wl, wg = lockstep(RHL, mk, (160, 161, 162), iters_delta, dev, dev)
        out[name] = dict(logits=wl, grad=wg, updates=iters_delta)
    ok = all(v["logits"] <= TOL_SAME["logits"] and v["grad"] <= TOL_SAME["grad"] for v in out.values())
    return dict(ok=ok, tol=TOL_SAME, cases=out)


def k4_device_vs_cpu(device, iters=2400):
    dev = dv.setup(device)
    if dev.type != "cuda":
        return dict(ok=True, skipped="cuda only")
    out = {}
    wl, wg = lockstep(tsc.TASK, lambda s: trr.make_fn(ARM_A, s)(), (160,), iters, torch.device("cpu"), dev)
    out["arm A 160 (2400 updates)"] = dict(logits=wl, grad=wg)
    task = it.ICMCTask(**S75_CFG)
    for arm in ("ORACLE", "SINGLE", "WIDE_SPLIT", "REG3"):
        spec = gr.arm_spec(arm, task.S)
        slow = ic.gate_names(gr.make_model(spec, task, 1)) if spec["recipe"] == "slow" else None
        wl, wg = lockstep(task, lambda s, spec=spec: gr.make_model(spec, task, s), (830,), 20, torch.device("cpu"), dev,
                          loss="icmc", slow_gate=slow)
        out[f"S75 {arm} (20 updates)"] = dict(logits=wl, grad=wg)
    ok = all(v["logits"] <= TOL_DEV["logits"] and v["grad"] <= TOL_DEV["grad"] for v in out.values())
    return dict(ok=ok, tol=TOL_DEV, cases=out)


# ── K5 ────────────────────────────────────────────────────────────────────────
def k5_icmc(device):
    if torch.device(device).type != "cpu":
        return dict(ok=True, skipped="cpu only")
    out = {}
    strip = lambda c: [{k: v for k, v in e.items() if k != "secs"} for e in c]
    for name, spec in (("LOCAL3+SLOW+split", dict(gate="win3", k=2, recipe="slow", split="split")),
                       ("REG3", dict(gate="reg3", k=2, recipe="slow"))):
        a = ic.run_one(SMALL, spec, 7, 2400, split_thr=0.99)
        b = gr.run_sequential_icmc(SMALL, spec, 7, 2400, split_thr=0.99)
        out[name] = (strip(a["curve"]) == strip(b["curve"]) and a.get("checks") == b.get("checks")
                     and a.get("beta_src") == b.get("beta_src") and a["ok"] and b["ok"])
    # register scan vs loop
    task = it.ICMCTask(**S75_CFG)
    ms = []
    for s in (830, 831):
        torch.manual_seed(s)
        ms.append(gr.make_model(gr.arm_spec("REG3", 4), task, s))
    x = torch.stack([task.make_batch(4, torch.Generator().manual_seed(s))[0][:, :-1] for s in (1, 2)])
    worst = {}
    for mode in ("loop", "scan"):
        bm = BatchedBDH(ms, "cpu", register=mode)
        lg, _ = bm(x)
        lg.square().mean((1, 2, 3)).sum().backward()
        worst[mode] = (lg.detach(), {n: bm.P(n).grad.clone() for n in bm.names if bm.P(n).grad is not None})
    dl = float((worst["loop"][0] - worst["scan"][0]).abs().max())
    dg = max(float((worst["loop"][1][n] - worst["scan"][1][n]).abs().max() / (worst["loop"][1][n].abs().max() + 1e-30))
             for n in worst["loop"][1])
    out["register scan = loop"] = dl <= 1e-5 and dg <= 1e-5
    out["scan_vs_loop"] = dict(logits=dl, grad_rel=dg)
    # the evaluation split in two (device part + CPU part) equals explore_i_common.evaluate bit for bit
    ev, pdata = ic.eval_set(SMALL), ic.probe_set(SMALL)
    same = True
    for spec in (dict(gate="wide", k=2), dict(gate="reg3", k=2), dict(gate="none", k=1)):
        torch.manual_seed(5)
        m = gr.make_model(spec, SMALL, 5)
        a = ic.evaluate(m, SMALL, ev, pdata, True)
        b = gr.eval_cpu_part(SMALL, ev, pdata, *gr.eval_device_part(m, SMALL, ev, pdata, True))
        same &= json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    out["evaluation split = explore_i_common.evaluate"] = same
    # the batched trainer with the statistics in worker processes = in line
    spec = dict(gate="win3", k=2, recipe="slow")
    r0 = gr.run_batched_icmc(SMALL, spec, (7, 8), 1200, stats_workers=0, eval_every=600)
    r2 = gr.run_batched_icmc(SMALL, spec, (7, 8), 1200, stats_workers=2, eval_every=600)
    strip = lambda c: [{k: v for k, v in e.items() if k not in ("secs", "train_secs")} for e in c]
    out["statistics in workers = in line"] = all(strip(x["curve"]) == strip(y["curve"]) for x, y in zip(r0, r2))
    return dict(ok=all(v for k, v in out.items() if isinstance(v, bool)), **out)


# ── K6 ────────────────────────────────────────────────────────────────────────
def k6_adam(device):
    dev = dv.setup(device)
    exact = dev.type == "cpu"
    R, shapes = 3, [(5, 7), (11,), (4, 3, 2)]
    g = torch.Generator().manual_seed(0)
    init = [torch.randn((R,) + s, generator=g) for s in shapes]
    stacked = [torch.nn.Parameter(t.clone().to(dev)) for t in init]
    per = [[torch.nn.Parameter(t[r].clone().to(dev)) for t in init] for r in range(R)]
    opt = StackedAdam([dict(params=stacked, lr=1e-3)], R)
    topt = [torch.optim.Adam(ps, lr=1e-3) for ps in per]
    worst = 0.0
    for step in range(1, 51):
        grads = [torch.randn((R,) + s, generator=g) for s in shapes]
        for p, gg in zip(stacked, grads):
            p.grad = gg.clone().to(dev)
        for r in range(R):
            for p, gg in zip(per[r], grads):
                p.grad = gg[r].clone().to(dev)
        if step == 20:                                   # zero run 1's state of the first parameter (a split)
            opt.zero_run_state(stacked[0], 1)
            st = topt[1].state[per[1][0]]
            for v in st.values():
                if torch.is_tensor(v):
                    v.zero_()
        active = None if step < 30 or step > 35 else [True, False, True]
        opt.step(active)
        for r in range(R):
            if active is None or active[r]:
                topt[r].step()
        for i, p in enumerate(stacked):
            for r in range(R):
                d = float((p.detach()[r] - per[r][i].detach()).abs().max())
                worst = max(worst, d)
    ok = worst == 0.0 if exact else worst <= 1e-6
    return dict(ok=ok, worst=worst, exact=exact)


# ── K7 ────────────────────────────────────────────────────────────────────────
def k7_data(device):
    if torch.device(device).type != "cpu":
        return dict(ok=True, skipped="cpu only")
    out = {}
    for S in (1, 2, 4):
        t = it.ICMCTask(S=S, V=4, NB=16 // S if S < 4 else 4, lmin=6, lmax=14, M=176 // S if S > 1 else 176)
        tok = t.make_batch(4096 if S < 4 else 1024, torch.Generator().manual_seed(S))[0]
        out[f"mask S={S}"] = torch.equal(gr.icmc_loss_mask(t, tok), t.loss_mask(tok))
    b1 = gr.Batches(SMALL, (7, 8), icmc=True, prefetch=True)
    b2 = gr.Batches(SMALL, (7, 8), icmc=True, prefetch=False)
    same = True
    try:
        for _ in range(20):
            x1, m1 = b1.next()
            x2, m2 = b2.next()
            same &= torch.equal(x1, x2) and torch.equal(m1, m2)
    finally:
        b1.close()
    out["prefetch = direct (20 steps, R = 2)"] = same
    return dict(ok=all(out.values()), **out)


# ── K8 ────────────────────────────────────────────────────────────────────────
def k8_fake_cuda(device):
    if torch.device(device).type != "cpu":
        return dict(ok=True, skipped="cpu only (on CUDA the real checks above run)")
    from torch._subclasses.fake_tensor import FakeTensorMode
    task = it.ICMCTask(**dict(S75_CFG, NB=4, M=44))
    out = {}
    cases = {
        "MultiBDH recurrent (arm A)": (tsc.TASK, lambda s: trr.make_fn(ARM_A, s)()),
        "MultiBDH perfect + conv": (task, lambda s: gr.make_model(gr.arm_spec("ORACLE", 4), task, s)),
        "MultiBDH none + conv": (task, lambda s: gr.make_model(gr.arm_spec("SINGLE", 4), task, s)),
        "LocalBDH WIDE": (task, lambda s: gr.make_model(gr.arm_spec("WIDE_SPLIT", 4), task, s)),
        "Reg3Conv": (task, lambda s: gr.make_model(gr.arm_spec("REG3", 4), task, s)),
        "LocalBDH + delta": (RHL, lambda s: dm.to_delta(g16.to_local(build(RHL, ARM_A, ARCH, s), s), beta=1.0)),
        "delta learned beta + GDN": (RHL, lambda s: dm.to_delta(build(RHL, tsc.ARM["B"], ARCH, s), beta="learned",
                                                                decay="gdn", seed=s)),
    }
    for name, (tk, mk) in cases.items():
        res = {}
        try:
            torch.manual_seed(1)
            m = mk(1)
            m2 = copy.deepcopy(m)
            x = tk.make_batch(2, torch.Generator().manual_seed(1))[0][:, :-1]
            with FakeTensorMode(allow_non_fake_inputs=True) as mode:
                mf = mode.from_tensor
                for p in list(m.parameters()) + list(m.buffers()):
                    p.data = mf(p.data).to("cuda") if False else p.data
                fm = dv.to_device(m, "cuda", cpu_io=False)
                logits = fm(x)[0]
                res["sequential"] = str(logits.device)
                bm = BatchedBDH([m2], "cuda", register="scan")
                lb, _ = bm(x.unsqueeze(0).to("cuda"))
                res["batched"] = str(lb.device)
            res["ok"] = res["sequential"].startswith("cuda") and res["batched"].startswith("cuda")
        except Exception as e:
            res = dict(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
        out[name] = res
    # the split's noise path on a fake-cuda W_g
    try:
        with FakeTensorMode(allow_non_fake_inputs=True):
            m = gr.make_model(gr.arm_spec("WIDE_SPLIT", 4), task, 1)
            m = dv.to_device(m, "cuda")
            op = dv._split_wrapper(ic.split_op)

            class O:
                state = {}
            op(m, O(), 0, 1, 123)
        out["split noise on cuda W_g"] = dict(ok=True)
    except Exception as e:
        out["split noise on cuda W_g"] = dict(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
    return dict(ok=all(v["ok"] for v in out.values()), cases=out)


# ── K9 ────────────────────────────────────────────────────────────────────────
def k9_bench(device, steps=20):
    dev = dv.setup(device)
    if dev.type != "cuda":
        return dict(ok=True, skipped="cuda only")
    task = it.ICMCTask(**S75_CFG)
    out = {}
    for arm in ("ORACLE", "SINGLE", "WIDE_SPLIT", "REG3"):
        for R in (1, 2):
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
                spec = gr.arm_spec(arm, task.S)
                ms = []
                for s in range(R):
                    torch.manual_seed(830 + s)
                    ms.append(gr.make_model(spec, task, 830 + s))
                bm = BatchedBDH(ms, dev)
                opt = StackedAdam([dict(params=[p for p in bm.ps if p.requires_grad], lr=1e-3)], R)
                data = gr.Batches(task, [830 + s for s in range(R)], icmc=True, prefetch=True)
                try:
                    for i in range(steps + 3):
                        if i == 3:
                            torch.cuda.synchronize()
                            t0 = time.time()
                        tok, mask = data.next()
                        tok, mask = tok.to(dev), mask.to(dev)
                        lb, _ = bm(tok[..., :-1])
                        ce = F.cross_entropy(lb.reshape(-1, lb.shape[-1]), tok[..., 1:].reshape(-1),
                                             reduction="none").view(mask.shape)
                        mf = mask.float()
                        ((ce * mf).sum((1, 2)) / mf.sum((1, 2))).sum().backward()
                        opt.step()
                        opt.zero_grad()
                    torch.cuda.synchronize()
                    dt = (time.time() - t0) / steps
                finally:
                    data.close()
                out[f"{arm} R={R}"] = dict(secs_per_update=dt, secs_per_run_update=dt / R,
                                           peak_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30)
            except torch.cuda.OutOfMemoryError:
                out[f"{arm} R={R}"] = dict(oom=True)
            finally:
                del_ = [n for n in ("bm", "opt") if n in locals()]
    # projection: 2 ORACLE + 10 SINGLE + 10 WIDE_SPLIT + 10 REG3 runs x 24000 updates; evaluation not included
    proj = {}
    for R in (1, 2):
        tot = 0.0
        okR = True
        for arm, n in (("ORACLE", 2), ("SINGLE", 10), ("WIDE_SPLIT", 10), ("REG3", 10)):
            v = out.get(f"{arm} R={R}", {})
            if "secs_per_run_update" not in v:
                okR = False
                break
            tot += n * 24000 * v["secs_per_run_update"]
        proj[f"R={R}"] = (tot / 3600) if okR else None
    return dict(ok=True, printed=True, bench=out, s75_train_hours=proj)


# ── driver ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default=None, help="comma-separated CHECK names (K0..K9)")
    ap.add_argument("--quick", action="store_true", help="shorter runs (smoke test; not the CHECK)")
    a = ap.parse_args()
    if torch.device(a.device).type == "cpu":
        torch.set_num_threads(1)
    dev = dv.setup(a.device)
    info = dv.device_info(dev)
    tag = "cpu" if dev.type == "cpu" else "cuda_" + "".join(c if c.isalnum() else "_" for c in info["gpu"])
    out_txt = os.path.join(HERE, "explore_gpu", f"checks_{tag}_output.txt")
    out_json = os.path.join(HERE, "explore_gpu", f"checks_{tag}_output.json")
    q = a.quick
    plan = [
        ("K0", "chaos calibration (printed)", lambda: k0_chaos(600 if q else 2400) if dev.type == "cpu" else dict(ok=True, skipped="cpu only")),
        ("K1", "record reproduction, arm A 160", lambda: k1_record(dev)),
        ("K2", "free-running batched vs sequential, arm A 160-163, 2400 updates", lambda: k2_free(dev, 600 if q else 2400)),
        ("K2b", "free-running, other classes, 600 updates", lambda: k2b_free(dev, 200 if q else 600)),
        ("K3", "lock-step batched vs sequential, arm A 160-163, 2400 updates", lambda: k3_lockstep(dev, 300 if q else 2400)),
        ("K3b", "lock-step, S75 arms at L = 768 and the delta memory", lambda: k3b_lockstep(dev, 5 if q else None, 50 if q else 300)),
        ("K4", "CUDA vs CPU lock-step", lambda: k4_device_vs_cpu(dev, 300 if q else 2400)),
        ("K5", "ICMC trainer = explore_i_common.run_one; register scan = loop", lambda: k5_icmc(dev)),
        ("K6", "StackedAdam = torch Adam", lambda: k6_adam(dev)),
        ("K7", "loss mask, prefetch", lambda: k7_data(dev)),
        ("K8", "fake-CUDA device audit", lambda: k8_fake_cuda(dev)),
        ("K9", "S75 S=4 benchmark", lambda: k9_bench(dev)),
    ]
    only = None if a.only is None else set(a.only.split(","))
    results = dict(meta=dict(info, started=time.strftime("%Y-%m-%d %H:%M:%S"), quick=q,
                             tolerances=dict(free=FREE_TOL, same_device=TOL_SAME, cuda_vs_cpu=TOL_DEV)), checks={})
    lines = [f"explore_gpu CHECKs on {info}", ""]
    print(lines[0], flush=True)
    for key, name, f in plan:
        if only and key not in only:
            continue
        t0 = time.time()
        try:
            r = f()
        except Exception as e:
            r = dict(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
        r["secs"] = round(time.time() - t0, 1)
        results["checks"][key] = r
        status = "SKIP" if r.get("skipped") else ("info" if r.get("printed") else ("ok  " if r["ok"] else "FAIL"))
        if r.get("asserted") is False and status == "ok  ":
            status = "info"
        line = f"{status} {key} {name} ({r['secs']:.0f} s): " + json.dumps(
            {k: v for k, v in r.items() if k not in ("traceback", "by_step", "secs")}, default=str)[:900]
        print(line, flush=True)
        lines.append(line)
        with open(out_json, "w") as fh:
            json.dump(results, fh, indent=1, default=str)
    failed = [k for k, r in results["checks"].items() if not r["ok"]]
    lines.append("")
    lines.append("ALL ASSERTED CHECKS PASSED" if not failed else f"FAILED: {failed}")
    print(lines[-1], flush=True)
    with open(out_txt, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
