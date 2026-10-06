#!/usr/bin/env python
"""
explore_b16_main.py — EXPLORATORY, not a result. Batch 16's recipe helpers for the main-line run paths. Runs INSIDE child
processes on the main line's module tree at explore_main9c.MAIN_SHA (9c5939e), as S33-S42's; imports S33/S35/S36/S37's
child modules read-only. Used by S44-S47's children.

CONFIGURATIONS (the recorded runs' own paths):
  "k16"  test_stream_recipe.run_attempt with ARM["A4k16"]: S=4, P=4, k=16, conv (X's HINGE4k16, S35's MUON_HINGE16)
  "a"    test_stream_recipe.run_attempt with ARM["A4k4"]: S=4, P=4, k=4, conv (S33's A_HINGE)
  "b"    test_stream_curriculum.run_sc with ARM["D8"]: S=8, P=4, k=16, conv, all 8 streams from step 1 (X's HINGE_D8,
         S33's D8_HINGE, S40's D8_PREV_A)
RECIPE (make_rc): test_slow_start.make_recipe(False, tau) — main's hinge (tau = TAU 0.2) or none, and its extra
statistics — plus this file's groups and lr switch, composed as S33's s33c.make_rc and S40's make_rc:
  adam_groups  test_slow_start.slow_groups' composition over the TRAINABLE parameters: the gate group (the names given,
               in that order) at lr throughout; optional groups held at a fixed lr throughout; every other trainable
               parameter at warm_lr for updates 1-2400 and lr after (switch after the evaluation at 2400, after the run
               path's own check). With the recurrent gate's names and the default lrs it is test_slow_start's HINGE
               recipe (CHECK: reproduces X's HINGE_D8|260 through 1200).
  muon_groups  S25's groups (s33c.make_groups) over the trainable parameters: the gate's 2-D weights on Muon at the Muon
               lr throughout; the gate's convolution-named weights (LOCAL3's window) on Adam at 1e-3 throughout; the
               other 2-D / 3-D weights on Muon at 0.1 x the Muon lr for updates 1-2400; the embedding, the stack's
               convolution, biases and 1-D parameters on Adam at 1e-4 for 1-2400; full lrs after. With the recurrent
               gate's names it is s33c.make_groups (CHECK).
  convert      model conversions applied to the run path's built model before the hinge is attached (LOCAL3, the
               reservoir, the decay).
  probe        a callback (model, step) after each evaluation's statistics and the lr switch (the decoder, measurements).
"""

import time

import torch
import torch.optim.optimizer as topt

import explore_main9c as mt
import explore_muon_scale_child as s33c
import explore_muon_k16_child as s35c
import explore_muon_recipe as s25
import explore_b16_gates as g16
import test_binding_onset as tbo
import test_scale_axes as tsa
import test_slow_start as tss
import test_stream_curriculum as tscur
import test_stream_recipe as tsr
from test_multilayer_binding import probe_batch
from test_channel_binding import eval_batch

LR, LR_WARM, WARM, TAU = tss.LR, tss.LR_WARM, tss.WARM, tss.TAU      # 1e-3, 1e-4, 2400, 0.2
SLOW = s33c.SLOW                                                     # 0.1
GATE_REC = tuple(s25.GATE)                                           # W_in, W_h, W_g
GATE_LOCAL = ("W_in", "W_g", "gate_conv.conv_w")
GATE_RES = ("W_in", "W_g")
MAIN_SHA = mt.MAIN_SHA


def task_of(cfg):
    return s35c.TASK if cfg == "k16" else s33c.cfg_task(cfg)


def arm_of(cfg, kind="HINGE"):
    if cfg == "k16":
        return s35c.arm(kind)
    return s33c.cfg_arm(cfg, kind)


def path(cfg, a, seed, iters, rc, eval_every=None):
    if cfg == "k16":
        return s35c.path(a, seed, iters, rc, eval_every=eval_every)
    return s33c.path(cfg, a, seed, iters, rc, eval_every=eval_every)


def stats_fn(cfg):
    return tscur.sc_stats((tscur.T1, tscur.T2)) if cfg == "b" else tsa.scale_stats


def adam_groups(holder, gate, lr=LR, warm_lr=LR_WARM, fixed=()):
    def param_groups(model):
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        nm = dict(named)
        fx = [n for ns, _ in fixed for n in ns]
        g = [n for n in gate if n in nm]
        rest = [n for n, _ in named if n not in g and n not in fx]
        groups = [dict(params=[nm[n] for n in g], lr=lr, lr_after=lr, names=g, tag="gate", kind="adam")]
        for ns, l in fixed:
            groups.append(dict(params=[nm[n] for n in ns], lr=l, lr_after=l, names=list(ns), tag="fixed", kind="adam"))
        groups.append(dict(params=[nm[n] for n in rest], lr=warm_lr, lr_after=lr, names=rest, tag="rest", kind="adam"))
        holder["groups"] = groups
        return groups
    return param_groups


def muon_groups(holder, mlr, gate, slow=True):
    def param_groups(model):
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        nm = dict(named)
        gm = [n for n, p in named if n in gate and not s33c.is_adam(n, p)]
        ga = [n for n, p in named if n in gate and s33c.is_adam(n, p)]
        adam = [n for n, p in named if n not in gate and s33c.is_adam(n, p)]
        rest = [n for n, p in named if n not in gate and n not in adam]
        sc = SLOW if slow else 1.0
        spec = [("gate", gm, "muon", mlr, mlr), ("gate", ga, "adam", LR, LR), ("rest", rest, "muon", sc * mlr, mlr),
                ("rest", adam, "adam", sc * LR, LR)]
        holder["groups"] = [dict(params=[nm[n] for n in ns], lr=l0, kind=k, tag=t, lr_after=l1, names=ns)
                            for t, ns, k, l0, l1 in spec if ns]
        return holder["groups"]
    return param_groups


def wrap_builder(b0, convert, on_build=None):
    def builder(a, seed):
        mk = b0(a, seed)

        def make():
            m = mk()
            for f in convert:
                m = f(m, seed)
            if on_build is not None:
                on_build(m)
            return m
        return make
    return builder


def make_rc(opt, holder, gate, tau=None, mlr=None, lr=LR, warm_lr=LR_WARM, fixed=(), convert=(), probe=None,
            on_build=None, keep=None, groups=True):
    """opt 'adam' / 'muon'; tau: the hinge (None = none); groups=False: no groups and no switch (onset_run's single Adam at
    the run path's lr; the perfect gate)."""
    base, state = tss.make_recipe(False, tau, keep=keep)

    def rc(kw):
        kw = dict(kw)
        if convert or on_build is not None:
            kw["builder"] = wrap_builder(kw["builder"], convert, on_build)
        kw = base(kw)
        if groups:
            pg = (muon_groups(holder, mlr, gate) if opt == "muon"
                  else adam_groups(holder, gate, lr=lr, warm_lr=warm_lr, fixed=fixed))
            kw["run_kw"] = dict(kw.get("run_kw") or {}, param_groups=pg)
        c0 = kw.get("check")

        def check(m, step, d):
            if c0 is not None:
                c0(m, step, d)
            if groups and step >= WARM:
                for g in holder["groups"]:
                    g["lr"] = g["lr_after"]
            if probe is not None:
                probe(m, step)
        kw["check"] = check
        return kw
    return rc, state


def with_probe(rc, probe, on_build=None):
    """Adds a probe (and an update-0 measurement) to an existing recipe (main's own HINGE recipe, S33's, S35's, S40's)."""
    def rc2(kw):
        kw = rc(kw)
        if on_build is not None:
            kw["builder"] = wrap_builder(kw["builder"], (), on_build)
        c0 = kw.get("check")

        def check(m, step, d):
            if c0 is not None:
                c0(m, step, d)
            probe(m, step)
        kw["check"] = check
        return kw
    return rc2


def decoder_probe(task, out, at):
    data = g16.probe_data(task)

    def probe(m, step):
        if step in at:
            out[str(step)] = g16.decode(m, task, data)

    def at0(m):
        if 0 in at:
            out["0"] = g16.decode(m, task, data)
    return probe, at0


def wh_probe(out, at):
    def probe(m, step):
        if step in at and hasattr(m, "W_h"):
            out[str(step)] = dict(rho=g16.rho(m.W_h), sigma=g16.sigma_max(m.W_h))
    return probe


def chain(*fs):
    fs = [f for f in fs if f is not None]

    def f(*a):
        for g in fs:
            g(*a)
    return f


def finish(rec, **extra):
    if rec.get("ok"):
        rec.update(maps=s33c.maps_of(rec), main_sha=MAIN_SHA, **extra)
        try:
            rec["outcome"] = tsr.outcome(rec)
        except Exception as e:                                   # the perfect gate has no learned-gate classes
            rec["outcome"] = f"n/a ({type(e).__name__})"
    return s33c.norm(rec)


def group_names(holder):
    return [dict(tag=g.get("tag"), kind=g.get("kind"), names=g.get("names"), lr_end=g["lr"]) for g in holder.get("groups", [])]


# ── CHECK helpers ────────────────────────────────────────────────────────────
def stub_lrs(cfg, a, rc_of, steps=WARM + 1, muon=False):
    """Each group's lr at every update through the real run path, the forward and evaluation stubbed (no early stop):
    rc_of(holder) -> recipe. Returns (holder, rec, lr log, group parameter ids, kept model)."""
    holder = {"log_lr": True}
    log, ids = [], {}

    def pre(opt, args, kwargs):
        if muon:
            return
        if not ids:
            ids["groups"] = [[id(q) for q in g["params"]] for g in opt.param_groups]
        log.append([g["lr"] for g in opt.param_groups])
    saved = (tbo.evaluate, tbo.logits_of)
    tbo.evaluate = lambda model, task, data: (0.5, 1.0)
    tbo.logits_of = lambda model, x: (torch.zeros(x.shape[0], x.shape[1], model.lm_head.shape[1])
                                      + model.lm_head.sum() * 0.0)
    h = topt.register_optimizer_step_pre_hook(pre)
    keep = {"at": steps}
    try:
        rc = rc_of(holder, keep)
        if muon:
            with s25.muon_in_onset_run(holder):
                rec = path(cfg, a, 3, steps, rc)
            log = holder["opts"][0].lr_log
            ids["groups"] = [[id(q) for q in g["params"]] for g in holder["groups"]]
        else:
            rec = path(cfg, a, 3, steps, rc)
    finally:
        tbo.evaluate, tbo.logits_of = saved
        h.remove()
    return holder, rec, log, ids.get("groups", []), keep


def cover(groups_ids, model):
    flat = [i for g in groups_ids for i in g]
    return len(flat) == len(set(flat)) and set(flat) == {id(q) for q in model.parameters() if q.requires_grad}


def same_upto(r, ref, t):
    """Curve and statistics of a run equal a record's through step t."""
    want = [c for c in ref["curve"] if c[0] <= t]
    sn = [s for s in s33c.norm(r["stats"]) if s["step"] != "end" and s["step"] <= t]
    so = [s for s in ref["stats"] if s["step"] != "end" and s["step"] <= t]
    return r["curve"] == want and sn == so, want


def step_timing(run_fn, adam_only):
    """The median interval between optimizer steps of one call of run_fn (Adam steps, or SliceMuon steps under Muon)."""
    stamps = []

    def post(opt, args, kwargs):
        if adam_only or isinstance(opt, s25.SliceMuon):
            stamps.append(time.perf_counter())
    h = topt.register_optimizer_step_post_hook(post)
    try:
        run_fn()
    finally:
        h.remove()
    d = sorted(y - x for x, y in zip(stamps, stamps[1:]))
    return d[len(d) // 2]


def eval_timing(cfg, m):
    task = task_of(cfg)
    t0 = time.time()
    tbo.evaluate(m, task, eval_batch(task))
    stats_fn(cfg)(m, task, probe_batch(task, 0), 1200)
    return time.time() - t0


def decode_timing(cfg, m):
    task = task_of(cfg)
    data = g16.probe_data(task)
    t0 = time.time()
    g16.decode(m, task, data)
    return time.time() - t0
