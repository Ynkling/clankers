#!/usr/bin/env python
"""
explore_muon_k16_child.py — EXPLORATORY, not a result. S35's code that runs INSIDE a child process on
the main line's module tree at explore_main9c.MAIN_SHA (9c5939e), as S33's (explore_muon_scale_child,
imported read-only for its recipe and optimizer groups). The parent side is explore_muon_k16.

Run path: test_stream_recipe.run_attempt with ARM["A4k16"] (S=4, P=4, k=16, conv; its restart check
recorded, never applied), the path X's HINGE4k16 ran on (test_slow_start.part_run, part "B"). The perfect
gate: ARM["ceiling4k16"] on the same path. MUON_HINGE16 = S33's MUON_HINGE recipe (s33c.make_rc
"HINGE": test_slow_start's hinge hook, pooled over channels, S25's optimizer groups and slow schedule)
plus a firing log: a forward hook registered after the hinge's records the update (the hinge state's
batch count) of every training batch on which the hinge's count rose.
"""

import json
import statistics
import time

import torch

import explore_main9c as mt
import explore_muon_scale_child as s33c
import explore_muon_recipe as s25
import test_binding_onset as tbo
import test_scale_axes as tsa
import test_slow_start as tss
import test_stream_recipe as tsr
from test_multilayer_binding import probe_batch
from test_channel_binding import eval_batch

WARM = tss.WARM
LR = tss.LR
TASK = tsr.TASKS["P4S4"]


def arm(kind):
    return dict(tsr.ARM["ceiling4k16"]) if kind == "CEIL" else dict(tsr.ARM["A4k16"])


def path(a, seed, iters, rc, eval_every=None):
    sched = dict(tsr.REAL, total=iters)
    if eval_every:
        sched["eval_every"] = eval_every
    return tsr.run_attempt(a, seed, sched, recipe=rc)


def with_fire_log(rc, state, log):
    def rc2(kw):
        kw = rc(kw)
        b0 = kw["builder"]

        def builder(a, seed):
            mk = b0(a, seed)

            def make():
                m = mk()
                last = [0]

                def hook(mod, args, out):
                    if mod.training and torch.is_grad_enabled() and state.get("fired", 0) > last[0]:
                        log.append(state["n"])
                        last[0] = state["fired"]
                m.register_forward_hook(hook)
                return m
            return make
        kw["builder"] = builder
        return kw
    return rc2


def make_rc(kind, mlr, holder, log, keep=None, tau=None):
    rc, state = s33c.make_rc(kind, mlr, holder, keep=keep, tau=tau)
    if kind == "HINGE":
        rc = with_fire_log(rc, state, log)
    return rc, state


def run(p):
    holder, log = {}, []
    a = arm(p["kind"])
    rc, _ = make_rc(p["kind"], p["mlr"], holder, log)
    with s25.muon_in_onset_run(holder):
        rec = path(a, p["seed"], p["iters"], rc)
    opts = holder.get("opts", [])
    if rec.get("ok"):
        u = opts[0].upd if opts else []
        rec.update(muon_lr=p["mlr"], kind=p["kind"], cfg="k16", n_opts_built=len(opts),
                   opt_groups=[dict(tag=g["tag"], kind=g["kind"], names=g["names"], lr_end=g["lr"]) for g in holder["groups"]],
                   gate_upd=dict(n=len(u), median=statistics.median(u), max=max(u)) if u else None,
                   maps=s33c.maps_of(rec), main_sha=mt.MAIN_SHA, fired_at=list(log) if p["kind"] == "HINGE" else None)
        try:
            rec["outcome"] = tsr.outcome(rec)
        except Exception as e:
            rec["outcome"] = f"n/a ({type(e).__name__})"
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs (called by explore_muon_k16.check in the parent) ──────────────────
def checks(p):
    mlr = p["mlr"]
    rows = []
    t = 1200
    # (1) the path: X's HINGE4k16|240 under main's own Adam path (test_slow_start's HINGE recipe)
    xs = mt.recorded("slow_start", mt.RESULTS_SHA)["HINGE4k16|240"]
    rc, _ = s33c.make_rc("HINGE", None, {}, adam_only=True)
    r = path(arm("HINGE"), 240, t, rc)
    st_new = {s["step"]: s for s in s33c.norm(r["stats"]) if s["step"] == t}
    st_ref = {s["step"]: s for s in xs["stats"] if s["step"] == t}
    want = [c for c in xs["curve"] if c[0] <= t]
    rows.append((f"path: test_slow_start's HINGE recipe (Adam) on run_attempt with A4k16 reproduces X's HINGE4k16|240 through "
                 f"{t} (curve {r['curve']} vs {want}; statistics at {t} {st_new == st_ref})",
                 r["curve"] == want and st_new == st_ref))
    # (2) the groups cover every parameter exactly once; Adam holds the embedding and the convolution
    for kind in ("HINGE", "CEIL"):
        a = arm(kind)
        m = tsr.builder_for(a)(a, 240)()
        gs = s33c.make_groups({}, mlr, kind != "CEIL")(m)
        adam = [n for g in gs if g["kind"] == "adam" for n in g["names"]]
        ok = s33c._cover(gs, m) and "embed.weight" in adam and any("conv" in n for n in adam)
        rows.append((f"{kind}: the groups cover every parameter exactly once {[(g['tag'], g['kind'], g['names']) for g in gs]}",
                     ok))
    # (3) each group's lr at updates 1, 2400, 2401 (MUON_HINGE16; forward and evaluation stubbed)
    holder = {"log_lr": True}
    saved = (tbo.evaluate, tbo.logits_of)
    tbo.evaluate = lambda model, task, data: (0.5, 1.0)
    tbo.logits_of = lambda model, x: (torch.zeros(x.shape[0], x.shape[1], model.lm_head.shape[1])
                                      + model.lm_head.sum() * 0.0)
    try:
        rc, _ = make_rc("HINGE", mlr, holder, [])
        with s25.muon_in_onset_run(holder):
            rec = path(arm("HINGE"), 3, WARM + 1, rc)
    finally:
        tbo.evaluate, tbo.logits_of = saved
    lg = holder["opts"][0].lr_log
    tags = [(g["tag"], g["kind"]) for g in holder["groups"]]
    w1, w2 = [mlr, s33c.SLOW * mlr, s33c.SLOW * LR], [mlr, mlr, LR]
    ok = (len(lg) == WARM + 1 and lg[0] == w1 and lg[WARM - 1] == w1 and lg[WARM] == w2
          and tags == [("gate", "muon"), ("rest", "muon"), ("rest", "adam")] and rec.get("ok"))
    rows.append((f"MUON_HINGE16: lrs per group {tags} at update 1 {lg[0]}, {WARM} {lg[WARM - 1]}, {WARM + 1} {lg[WARM]} "
                 f"({len(lg)} updates)", ok))
    # (4) the firing log: with TAU 0 the hinge fires on every training batch, and the log holds every update
    holder, log = {}, []
    rc, st = make_rc("HINGE", mlr, holder, log, tau=0.0)
    with s25.muon_in_onset_run(holder):
        r = path(arm("HINGE"), 240, 60, rc, eval_every=20)
    rows.append((f"the firing log (TAU 0, 60 steps): {len(log)} entries, updates {log[:3]}...{log[-2:]}; the hinge counted "
                 f"{st.get('fired')} of {st.get('n')}", log == list(range(1, 61)) and st.get("fired") == 60))
    # (5) the perfect gate zeroes every cross-stream score at every layer (test_stream_recipe.cross_scores)
    maxc, live, _ = tsr.cross_scores(arm("CEIL"), TASK, 11)
    rows.append((f"the perfect gate (ceiling4k16) zeroes every cross-stream score at every layer (max |cross| per layer "
                 f"{maxc}; max |same-stream| {[round(v, 4) for v in live]})",
                 len(maxc) == 3 and max(maxc) == 0.0 and min(live) > 0.0))
    return [[n, bool(v)] for n, v in rows]


# ── timing, constants, recorded summaries ────────────────────────────────────
def timing(p):
    """As s33c.timing: the median interval between Muon steps over p["steps"] steps, and the evaluation time."""
    import torch.optim.optimizer as topt
    out = {}
    for kind in p["which"]:
        a = arm(kind)
        stamps = []

        def post(opt, args, kwargs):
            if isinstance(opt, s25.SliceMuon):
                stamps.append(time.perf_counter())
        h = topt.register_optimizer_step_post_hook(post)
        try:
            holder = {}
            rc, _ = make_rc(kind, p["mlr"], holder, [])
            with s25.muon_in_onset_run(holder):
                path(a, 0, p["steps"], rc, eval_every=10 ** 9)
        finally:
            h.remove()
        d = sorted(y - x for x, y in zip(stamps, stamps[1:]))
        m = tsr.builder_for(a)(a, 0)()
        data, probe = eval_batch(TASK), probe_batch(TASK, 0)
        t0 = time.time()
        tbo.evaluate(m, TASK, data)
        tsa.scale_stats(m, TASK, probe, 1200)
        out[kind] = [d[len(d) // 2], time.time() - t0, len(d) + 1]
    return out


def constants(p):
    return dict(tss_LR=tss.LR, tsr_LR=tsr.LR, tss_WARM=tss.WARM, tss_TAU=tss.TAU, tss_LAMBDA=tss.LAMBDA,
                tsr_MAX_ITERS=tsr.MAX_ITERS, tsr_REAL=tsr.REAL, EVAL_EVERY=tbo.EVAL_EVERY, main_sha=mt.MAIN_SHA,
                params={k: [[n, list(q.shape)] for n, q in tsr.builder_for(arm(k))(arm(k), 0)().named_parameters()]
                        for k in ("HINGE", "CEIL")})


def recorded_summary(p):
    """Per-seed summaries of X's recorded HINGE4k16 (slow_start) and A4k16 (stream_recipe)."""
    out = {}
    for key, store, sha in (("HINGE4k16", "slow_start", mt.RESULTS_SHA), ("A4k16", "stream_recipe", mt.MAIN_SHA)):
        runs = mt.recorded(store, sha)
        for s in p["seeds"]:
            r = runs.get(f"{key}|{s}")
            if r is None:
                continue
            try:
                oc = tsr.outcome(r)
            except Exception as e:
                oc = f"n/a ({type(e).__name__})"
            out[f"{key}|{s}"] = dict(bound=r.get("transition") is not None, transition=r.get("transition"),
                                     acc=r.get("acc"), outcome=oc, maps=s33c.maps_of(r), end_map=r["end"].get("ch_map"),
                                     stopped_at=r.get("stopped_at"), hs_hinge=r["end"].get("hs_hinge"),
                                     hs_windows=r["end"].get("hs_windows"))
    return out
