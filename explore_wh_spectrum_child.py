#!/usr/bin/env python
"""
explore_wh_spectrum_child.py — EXPLORATORY, not a result. S42's code that runs INSIDE a child process on the
main line's module tree at explore_main9c.MAIN_SHA (9c5939e). Parent: explore_wh_spectrum.

Configurations (reruns of recorded runs to 1600 updates, on their own run paths and recipes; S39's):
  D8_HINGE_M  S33's D8_HINGE (Muon lr 0.005, S25's groups, main's hinge; run_sc with D8: S=8, P=4, k=16)
  HINGE_D8_A  X's HINGE_D8 (test_slow_start.make_recipe(True, TAU), Adam; run_sc with D8)
MEASUREMENTS every 50 updates from 0 to 1600 (update 0: the built model; later: right after that update's
optimizer step, from a step post-hook, as S39; eval mode, no gradient, no random draw), on S39's 512-sequence
probe (seed 39000, so the values at S39's updates repeat S39's), with explore_gate_cap_child.measure:
sigma_max(W_h) and the spectral radius rho(W_h) (largest |eigenvalue|; both in float64); the median norms of h,
W_h h_(t-1) and W_in v_t at key positions; S38's decoder of the stream from h at key positions (and at the
stream-token positions, printed).
"""

import json

import torch

import explore_main9c as mt
import explore_muon_scale_child as s33c
import explore_gate_memory_child as s39c
import explore_gate_cap_child as s41c
import test_slow_start as tss

PROBE_N, PROBE_SEED = s39c.PROBE_N, s39c.PROBE_SEED
EVERY = 50
ITERS = 1600
AT = tuple(range(0, ITERS + 1, EVERY))
CFG = "b"
CONF = {"D8_HINGE_M": dict(opt="muon"), "HINGE_D8_A": dict(opt="adam")}
KEEP = ("sigma", "rho", "hn_key", "b_key", "a_key", "h_key", "h_stream", "sat_key", "gate_recompute_diff")


def measure(m, task, data):
    d = s41c.measure(m, task, data)
    return {k: d[k] for k in KEEP}


def run(p):
    """p: conf, seed, iters, mlr (Muon)."""
    import torch.optim.optimizer as topt
    c = CONF[p["conf"]]
    holder, out, box = {}, {}, dict(n=0)
    task = s33c.cfg_task(CFG)
    a = s33c.cfg_arm(CFG, "HINGE")
    data = task.make_batch(PROBE_N, torch.Generator().manual_seed(PROBE_SEED))[0]
    rc = s33c.make_rc("HINGE", p["mlr"], holder)[0] if c["opt"] == "muon" else tss.make_recipe(True, tss.TAU)[0]

    def rc2(kw):
        kw = rc(kw)
        b0 = kw["builder"]

        def builder(aa, seed):
            mk = b0(aa, seed)

            def make():
                m = mk()
                box["m"] = m
                box["builds"] = box.get("builds", 0) + 1
                out["0"] = measure(m, task, data)
                return m
            return make
        kw["builder"] = builder
        return kw

    def post(opt, args, kwargs):
        if c["opt"] == "muon" and not isinstance(opt, s33c.s25.SliceMuon):
            return
        box["n"] += 1
        if box["n"] in AT and "m" in box:
            out[str(box["n"])] = measure(box["m"], task, data)
    hk = topt.register_optimizer_step_post_hook(post)
    try:
        if c["opt"] == "muon":
            with s33c.s25.muon_in_onset_run(holder):
                rec = s33c.path(CFG, a, p["seed"], p["iters"], rc2)
        else:
            rec = s33c.path(CFG, a, p["seed"], p["iters"], rc2)
    finally:
        hk.remove()
    if rec.get("ok"):
        rec.update(conf=p["conf"], opt=c["opt"], spec=out, updates_counted=box["n"], builds=box.get("builds"),
                   muon_lr=p.get("mlr") if c["opt"] == "muon" else None, main_sha=mt.MAIN_SHA)
    return json.loads(json.dumps(rec, default=float))


def numpy_check(p):
    import numpy as np
    g = torch.Generator().manual_seed(7)
    W = torch.randn(32, 32, generator=g) * 0.3
    Wn = W.double().numpy()
    s_np = float(np.linalg.svd(Wn, compute_uv=False)[0])
    r_np = float(np.abs(np.linalg.eigvals(Wn)).max())
    s_t, r_t = s41c.sigma_max(W), s41c.rho(W)
    ds, dr = abs(s_t - s_np), abs(r_t - r_np)
    return [[f"sigma_max and rho (explore_gate_cap_child's, float64) agree with numpy on a random 32x32 matrix (sigma "
             f"{s_t:.8f} vs {s_np:.8f}, difference {ds:.1e}; rho {r_t:.8f} vs {r_np:.8f}, difference {dr:.1e}; tolerance 1e-6)",
             ds <= 1e-6 and dr <= 1e-6]]


def timing(p):
    """Per configuration: the median interval between optimizer steps over p["steps"] steps, the time of one
    evaluation (held-out set and the run path's statistics) and of one measurement."""
    import time
    import torch.optim.optimizer as topt
    import test_binding_onset as tbo
    import test_stream_curriculum as tscur
    import test_stream_recipe as tsr
    from test_channel_binding import eval_batch
    from test_multilayer_binding import probe_batch
    out = {}
    for conf in p["which"]:
        c = CONF[conf]
        stamps = []

        def post(opt, args, kwargs):
            if c["opt"] == "adam" or isinstance(opt, s33c.s25.SliceMuon):
                stamps.append(time.perf_counter())
        h = topt.register_optimizer_step_post_hook(post)
        try:
            run(dict(conf=conf, seed=0, iters=p["steps"], mlr=p["mlr"]))
        finally:
            h.remove()
        d = sorted(y - x for x, y in zip(stamps, stamps[1:]))
        task = s33c.cfg_task(CFG)
        a = s33c.cfg_arm(CFG, "HINGE")
        m = tsr.builder_for(a)(a, 0)()
        t0 = time.time()
        tbo.evaluate(m, task, eval_batch(task))
        tscur.sc_stats((tscur.T1, tscur.T2))(m, task, probe_batch(task, 0), 1200)
        te = time.time() - t0
        data = task.make_batch(PROBE_N, torch.Generator().manual_seed(PROBE_SEED))[0]
        t0 = time.time()
        measure(m, task, data)
        out[conf] = [d[len(d) // 2], te, time.time() - t0]
    return out
