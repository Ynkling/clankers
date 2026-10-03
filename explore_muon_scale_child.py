#!/usr/bin/env python
"""
explore_muon_scale_child.py — EXPLORATORY, not a result. S33's code that runs INSIDE a child process
on the main line's module tree at explore_main9c.MAIN_SHA (9c5939e), never in the batch driver's own
process (see explore_main9c). The parent side is explore_muon_scale.

Run paths (the recorded runs' own, with test_slow_start's recipe knob):
  config a  test_stream_recipe.run_attempt(ARM["A4k4"], seed, REAL with total) — S=4, P=4, k=4, conv
            (the perfect gate: test_scale_axes.ARM["ceiling4k4"] on the same path);
  config b  test_stream_curriculum.run_sc(ARM["D8"], seed, REAL with total) — S=8, P=4, k=16, conv, all
            8 streams from step 1 (the perfect gate: test_stream_recipe.ARM["ceiling8k16"], cur False).
The recipe for every arm starts from test_slow_start.make_recipe(slow=False, tau) — the main line's
hinge (tau = TAU 0.2, pooled over channels, through its forward hook and gradient-injection node) for
MUON_HINGE, none for MUON_SLOW and the perfect gate, and its extra statistics in every arm — and adds
S25's optimizer (explore_muon_recipe.MuonAdam in onset_run, via muon_in_onset_run) with these groups:
gate W_in, W_h, W_g on Muon at the Muon lr throughout; the other 2-D / 3-D weights on Muon; the
embedding, the convolution, biases and 1-D parameters on Adam at 1e-3 (test_slow_start.LR); in the
slow phase (MUON_SLOW, MUON_HINGE) every non-gate group at 0.1 x its lr for updates 1-2400, full after
the evaluation at 2400 (the run path's own check runs first). The perfect gate runs without the slow
phase (S25's lr selection).
"""

import json
import statistics
import time

import torch

import explore_main9c as mt
import test_binding_onset as tbo
import test_scale_axes as tsa
import test_slow_start as tss
import test_stream_curriculum as tscur
import test_stream_recipe as tsr
import explore_muon_recipe as s25
from test_multilayer_binding import probe_batch
from test_channel_binding import eval_batch

WARM = tss.WARM                      # 2400
LR = tss.LR                          # 1e-3
SLOW = 0.1
N_LAYER = 3


def cfg_arm(cfg, kind):
    if cfg == "a":
        return dict(tsa.ARM["ceiling4k4"]) if kind == "CEIL" else dict(tsr.ARM["A4k4"])
    return dict(tsr.ARM["ceiling8k16"], key="ceil8_D8", cur=False) if kind == "CEIL" else dict(tscur.ARM["D8"])


def cfg_task(cfg):
    return tsr.TASKS["P4S4"] if cfg == "a" else tsr.TASK48


def path(cfg, a, seed, iters, rc, eval_every=None):
    if cfg == "a":
        sched = dict(tsr.REAL, total=iters)
        if eval_every:
            sched["eval_every"] = eval_every
        return tsr.run_attempt(a, seed, sched, recipe=rc)
    sched = dict(tscur.REAL, total=iters)
    if eval_every:
        sched["eval_every"] = eval_every
    return tscur.run_sc(a, seed, sched, recipe=rc)


def is_adam(n, p):
    return n == "embed.weight" or "conv" in n or p.ndim == 1 or n.endswith("bias")


def make_groups(holder, mlr, slow):
    def param_groups(model):
        named = list(model.named_parameters())
        nm = dict(named)
        gate = [n for n, _ in named if n in s25.GATE]
        adam = [n for n, p in named if n not in s25.GATE and is_adam(n, p)]
        rest = [n for n, p in named if n not in s25.GATE and n not in adam]
        sc = SLOW if slow else 1.0
        spec = [("gate", gate, "muon", mlr, mlr), ("rest", rest, "muon", sc * mlr, mlr),
                ("rest", adam, "adam", sc * LR, LR)]
        holder["groups"] = [dict(params=[nm[n] for n in ns], lr=l0, kind=k, tag=t, lr_after=l1, names=ns)
                            for t, ns, k, l0, l1 in spec if ns]
        return holder["groups"]
    return param_groups


def make_rc(kind, mlr, holder, keep=None, tau=None, adam_only=False):
    """kind: SLOW, HINGE or CEIL. adam_only: test_slow_start's own HINGE recipe (Adam, its slow groups)."""
    if adam_only:
        return tss.make_recipe(True, tss.TAU if tau is None else tau, keep=keep)
    t = (tss.TAU if tau is None else tau) if kind == "HINGE" else None
    base, state = tss.make_recipe(False, t, keep=keep)
    slow = kind in ("SLOW", "HINGE")

    def rc(kw):
        kw = base(kw)
        kw["run_kw"] = dict(kw.get("run_kw") or {}, param_groups=make_groups(holder, mlr, slow))
        if slow:
            c0 = kw.get("check")

            def check(m, step, data):
                if c0 is not None:
                    c0(m, step, data)
                if step >= WARM:
                    for g in holder["groups"]:
                        g["lr"] = g["lr_after"]
            kw["check"] = check
        return kw
    return rc, state


def maps_of(rec):
    return {str(s["step"]): s.get("ch_map") for s in rec.get("stats", []) if s.get("ch_map") is not None}


def run(p):
    holder = {}
    a = cfg_arm(p["cfg"], p["kind"])
    rc, _ = make_rc(p["kind"], p["mlr"], holder)
    with s25.muon_in_onset_run(holder):
        rec = path(p["cfg"], a, p["seed"], p["iters"], rc)
    opts = holder.get("opts", [])
    if rec.get("ok"):
        u = opts[0].upd if opts else []
        rec.update(muon_lr=p["mlr"], kind=p["kind"], cfg=p["cfg"], n_opts_built=len(opts),
                   opt_groups=[dict(tag=g["tag"], kind=g["kind"], names=g["names"], lr_end=g["lr"]) for g in holder["groups"]],
                   gate_upd=dict(n=len(u), median=statistics.median(u), max=max(u)) if u else None,
                   maps=maps_of(rec), main_sha=mt.MAIN_SHA)
        try:
            rec["outcome"] = tsr.outcome(rec)
        except Exception as e:                                   # the perfect gate has no learned-gate classes
            rec["outcome"] = f"n/a ({type(e).__name__})"
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs (called by explore_muon_scale.check in the parent) ─────────────────
def _cover(groups, model):
    ids = [id(p) for g in groups for p in g["params"]]
    return len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters()}


def _stub_lrs(cfg, kind, mlr):
    holder = {"log_lr": True}
    saved = (tbo.evaluate, tbo.logits_of)
    tbo.evaluate = lambda model, task, data: (0.5, 1.0)
    tbo.logits_of = lambda model, x: (torch.zeros(x.shape[0], x.shape[1], model.lm_head.shape[1])
                                      + model.lm_head.sum() * 0.0)
    try:
        rc, _ = make_rc(kind, mlr, holder)
        with s25.muon_in_onset_run(holder):
            rec = path(cfg, cfg_arm(cfg, kind), 3, WARM + 1, rc)
    finally:
        tbo.evaluate, tbo.logits_of = saved
    return holder, rec


def norm(x):
    return json.loads(json.dumps(x, default=float))


def checks(p):
    mlr = p["mlr"]
    rows = []
    # (1) groups cover every parameter exactly once; Adam holds the embedding and the convolution
    for cfg in ("a", "b"):
        for kind in ("SLOW", "HINGE", "CEIL"):
            a = cfg_arm(cfg, kind)
            m = tsr.builder_for(a)(a, 240)()
            h = {}
            gs = make_groups(h, mlr, kind != "CEIL")(m)
            adam = [n for g in gs if g["kind"] == "adam" for n in g["names"]]
            ok = _cover(gs, m) and "embed.weight" in adam and any("conv" in n for n in adam)
            rows.append((f"config {cfg} {kind}: the groups cover every parameter exactly once "
                         f"{[(g['tag'], g['kind'], g['names']) for g in gs]}", ok))
    # (2) each group's lr at updates 1, 2400, 2401 through the real run path (forward and evaluation stubbed)
    for cfg in ("a", "b"):
        for kind in ("SLOW", "HINGE"):
            h, rec = _stub_lrs(cfg, kind, mlr)
            lg = h["opts"][0].lr_log
            tags = [(g["tag"], g["kind"]) for g in h["groups"]]
            w1 = [mlr, SLOW * mlr, SLOW * LR]
            w2 = [mlr, mlr, LR]
            ok = (len(lg) == WARM + 1 and lg[0] == w1 and lg[WARM - 1] == w1 and lg[WARM] == w2
                  and tags == [("gate", "muon"), ("rest", "muon"), ("rest", "adam")] and rec.get("ok"))
            rows.append((f"config {cfg} {kind}: lrs per group {tags} at update 1 {lg[0]}, {WARM} {lg[WARM - 1]}, "
                         f"{WARM + 1} {lg[WARM]} ({len(lg)} updates)", ok))
    # (3) the perfect gate zeroes every cross-stream score at every layer (test_stream_recipe.cross_scores:
    #     random convolution weights; same-stream scores non-zero)
    for cfg in ("a", "b"):
        maxc, live, _ = tsr.cross_scores(cfg_arm(cfg, "CEIL"), cfg_task(cfg), 11)
        rows.append((f"config {cfg}: the perfect gate zeroes every cross-stream score at every layer (max |cross| per "
                     f"layer {maxc}; max |same-stream| {[round(v, 4) for v in live]})",
                     len(maxc) == N_LAYER and max(maxc) == 0.0 and min(live) > 0.0))
    # (4) the child loads main's run paths: X's records reproduce through 1200 steps
    t = 1200
    xs = mt.recorded("slow_start", mt.RESULTS_SHA)["HINGE_D8|260"]
    rc, _ = make_rc("HINGE", None, {}, adam_only=True)
    r = path("b", cfg_arm("b", "SLOW"), 260, t, rc)
    st_new = {s["step"]: s for s in norm(r["stats"]) if s["step"] == t}
    st_ref = {s["step"]: s for s in xs["stats"] if s["step"] == t}
    rows.append((f"config b: test_slow_start's HINGE recipe (Adam) on run_sc reproduces X's HINGE_D8|260 through {t} "
                 f"(curve {r['curve']} vs {[c for c in xs['curve'] if c[0] <= t]}; statistics at {t} "
                 f"{st_new == st_ref})", r["curve"] == [c for c in xs["curve"] if c[0] <= t] and st_new == st_ref))
    xa = mt.recorded("stream_recipe", mt.MAIN_SHA)["A4k4|240"]
    r = path("a", cfg_arm("a", "SLOW"), 240, t, None)
    rows.append((f"config a: run_attempt (recipe None) reproduces X's A4k4|240 through {t} (curve {r['curve']} vs "
                 f"{[c for c in xa['curve'] if c[0] <= t]})", r["curve"] == [c for c in xa["curve"] if c[0] <= t]))
    # (5) the hinge's wiring: with TAU at 1.0, MUON_HINGE equals MUON_SLOW bit for bit (config a, 1200 steps)
    k1, k2 = {"at": t}, {"at": t}
    h1, h2 = {}, {}
    rc1, st1 = make_rc("HINGE", mlr, h1, keep=k1, tau=1.0)
    rc2, _ = make_rc("SLOW", mlr, h2, keep=k2)
    with s25.muon_in_onset_run(h1):
        r1 = path("a", cfg_arm("a", "HINGE"), 240, t, rc1)
    with s25.muon_in_onset_run(h2):
        r2 = path("a", cfg_arm("a", "SLOW"), 240, t, rc2)
    same = r1["curve"] == r2["curve"] and all(torch.equal(k1["snap"][n], k2["snap"][n]) for n in k2["snap"])
    rows.append((f"config a: with the hinge's TAU at 1.0, MUON_HINGE equals MUON_SLOW bit for bit (seed 240, {t} steps, "
                 f"Muon lr {mlr:g}; curves {r1['curve']} vs {r2['curve']}; weights equal; hinge fired {st1.get('fired')})",
                 same and st1.get("fired") == 0))
    return [[n, bool(v)] for n, v in rows]


# ── timing and recorded summaries ────────────────────────────────────────────
def timing(p):
    """Seconds per training step under MuonAdam with the arm's recipe: the median interval between consecutive
    Muon steps over one run of p["steps"][cfg] steps (no evaluation inside; robust to the start-up of other
    processes), and seconds per evaluation (held-out set and the run path's statistics), per (cfg, kind)."""
    import torch.optim.optimizer as topt
    out = {}
    for cfg, kind in p["which"]:
        a = cfg_arm(cfg, kind)
        stamps = []

        def post(opt, args, kwargs):
            if isinstance(opt, s25.SliceMuon):
                stamps.append(time.perf_counter())
        h = topt.register_optimizer_step_post_hook(post)
        try:
            holder = {}
            rc, _ = make_rc(kind, p["mlr"], holder)
            with s25.muon_in_onset_run(holder):
                path(cfg, a, 0, p["steps"][cfg], rc, eval_every=10 ** 9)
        finally:
            h.remove()
        d = sorted(y - x for x, y in zip(stamps, stamps[1:]))
        task = cfg_task(cfg)
        m = tsr.builder_for(a)(a, 0)()
        data, probe = eval_batch(task), probe_batch(task, 0)
        t0 = time.time()
        tbo.evaluate(m, task, data)
        sf = tsa.scale_stats if cfg == "a" else tscur.sc_stats((tscur.T1, tscur.T2))
        sf(m, task, probe, 1200)
        out[f"{cfg}|{kind}"] = [d[len(d) // 2], time.time() - t0, len(d) + 1]
    return out


def constants(p):
    """The main line's recipe constants, read in the child from its own modules."""
    return dict(tss_LR=tss.LR, tsr_LR=tsr.LR, tscur_LR=tscur.LR, tss_WARM=tss.WARM, tss_TAU=tss.TAU,
                tss_LAMBDA=tss.LAMBDA, tsr_MAX_ITERS=tsr.MAX_ITERS, tscur_TOTAL=tscur.TOTAL,
                tsr_REAL=tsr.REAL, tscur_REAL=tscur.REAL, EVAL_EVERY=tbo.EVAL_EVERY, main_sha=mt.MAIN_SHA,
                params={cfg + "|" + kind: [[n, list(q.shape)] for n, q in
                                           tsr.builder_for(cfg_arm(cfg, kind))(cfg_arm(cfg, kind), 0)().named_parameters()]
                        for cfg in ("a", "b") for kind in ("SLOW", "CEIL")})


def recorded_summary(p):
    """Per-seed summaries of X's recorded runs: A4k4 (stream_recipe) and HINGE_D8 (slow_start)."""
    out = {}
    for key, store, sha in (("A4k4", "stream_recipe", mt.MAIN_SHA), ("HINGE_D8", "slow_start", mt.RESULTS_SHA)):
        runs = mt.recorded(store, sha)
        for s in p["seeds"][key]:
            r = runs.get(f"{key}|{s}")
            if r is None:
                continue
            try:
                oc = tsr.outcome(r)
            except Exception as e:
                oc = f"n/a ({type(e).__name__})"
            out[f"{key}|{s}"] = dict(bound=r.get("transition") is not None, transition=r.get("transition"),
                                     acc=r.get("acc"), outcome=oc, maps=maps_of(r), end_map=r["end"].get("ch_map"),
                                     stopped_at=r.get("stopped_at"))
    return out
