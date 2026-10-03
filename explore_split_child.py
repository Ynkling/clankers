#!/usr/bin/env python
"""
explore_split_child.py — EXPLORATORY, not a result. S36's code that runs INSIDE a child process on the
main line's module tree at explore_main9c.MAIN_SHA (9c5939e), as S33's and S35's. The parent side is
explore_split_plateau.

Arms (each = its paired arm's recipe on its paired run path + the trigger below):
  SPLIT4k4_M  S33's A_HINGE (explore_muon_scale_child.make_rc "HINGE", config a: run_attempt with A4k4;
              S25's optimizer groups at Muon lr 0.005, main's hinge), 28800 steps.
  SPLIT_D8_M  S33's D8_HINGE (config b: run_sc with D8, all 8 streams from step 1), 43200 steps.
  SPLIT_D8_A  main's HINGE_D8 (test_slow_start.make_recipe(True, TAU) on run_sc with D8: Adam, the slow
              groups), 43200 steps. onset_run's torch.optim.Adam is built through a capturing factory
              (the same constructor and arguments) so the trigger can reach W_g's Adam state.
THE TRIGGER (label-free): the probe = 64 sequences of the arm's task from torch.Generator seeded 12345;
at updates 2400 (reference only), 4800, 7200, ... every 2400 through total - 2400 (26400 for (a), 40800
for (b)), after that evaluation's statistics and the run path's own check: probe accuracy
(test_binding_onset.evaluate) and the read gate's mean mass per channel over all probe positions (eval
mode, no gradient); from 4800 on, split once at that check if the accuracy is < 0.95 and it rose by
< 0.02 since the previous check.
THE SPLIT (S24's operation, explore_merge_split.apply_op): c* = the busiest channel, c0 = the idlest
(those masses); w = W_g[c*]; W_g[c0] = w + n2, W_g[c*] = w + n1 with n1, n2 ~ N(0, (0.1 std(w))^2) drawn
in that order from a generator seeded 12_000_000 + 1000 x seed + the run's split count (the model has no
gate bias); then W_g's optimizer state is zeroed in place (Adam: exp_avg, exp_avg_sq and step; Muon: the
momentum buffer), as a fresh state.
"""

import contextlib
import copy
import json
import time

import torch
import torch.nn.functional as F

import explore_main9c as mt
import explore_muon_scale_child as s33c
import explore_muon_recipe as s25
import test_binding_onset as tbo
import test_scale_axes as tsa
import test_slow_start as tss
import test_stream_recipe as tsr
from test_multilayer_binding import probe_batch
from test_channel_binding import eval_batch
from test_instrument_v2 import TAU_END

PROBE_N, PROBE_SEED = 64, 12345
ACC_THR, RISE = 0.95, 0.02
EVERY, FIRST, REF_AT = 2400, 4800, 2400
NOISE_REL, NOISE_BASE = 0.1, 12_000_000
ARM = {"SPLIT4k4_M": dict(cfg="a", opt="muon", total=28800),
       "SPLIT_D8_M": dict(cfg="b", opt="muon", total=43200),
       "SPLIT_D8_A": dict(cfg="b", opt="adam", total=43200)}


@contextlib.contextmanager
def adam_capture(holder):
    """onset_run's torch.optim.Adam(param_groups(model), lr=lr), built by the same constructor and kept."""
    def factory(params, lr=1e-3, **kw):
        opt = s25._ADAM(params, lr=lr, **kw)
        holder.setdefault("opts", []).append(opt)
        return opt
    torch.optim.Adam = factory
    try:
        yield
    finally:
        torch.optim.Adam = s25._ADAM


def probe_data(task):
    return task.make_batch(PROBE_N, torch.Generator().manual_seed(PROBE_SEED))[0]


@torch.no_grad()
def masses(m, data):
    was = m.training
    m.eval()
    try:
        gr = m(data[:, :-1], TAU_END)[2]
    finally:
        m.train(was)
    return gr.mean((0, 1))


def noise_for(w, seed):
    sd = NOISE_REL * w.std()
    g = torch.Generator().manual_seed(seed)
    n1 = torch.randn(w.shape, generator=g, dtype=w.dtype) * sd                   # row c*
    n2 = torch.randn(w.shape, generator=g, dtype=w.dtype) * sd                   # row c0
    return sd, n1, n2


def split_op(m, o, cs, c0, seed):
    """S24's SPLIT on W_g, then W_g's optimizer state (in o) zeroed in place."""
    assert not any(n.startswith("b_g") or n == "W_g_bias" for n, _ in m.named_parameters())
    W = m.W_g
    with torch.no_grad():
        w = W[cs].clone()
        before = float((W[cs] - W[c0]).norm())
        sd, n1, n2 = noise_for(w, seed)
        W[c0] = w + n2
        W[cs] = w + n1
    st = o.state.get(W)
    zeroed = []
    if st:
        for k, v in st.items():
            if torch.is_tensor(v):
                v.zero_()
                zeroed.append(k)
    return dict(noise_sd=float(sd), noise_seed=seed, row_dist_before=before,
                row_dist_after=float((W[cs] - W[c0]).detach().norm()), state_zeroed=zeroed)


def make_trigger(task, seed, info, opt_of, last, thr=ACC_THR, rise=RISE):
    data = probe_data(task)
    rprobe = probe_batch(task, seed)
    st = dict(prev=None, n=0)

    def trig(m, step):
        if step % EVERY or step < REF_AT or step > last:
            return
        acc, _ = tbo.evaluate(m, task, data)
        ms = masses(m, data)
        row = dict(step=step, acc=acc, masses=[round(float(v), 5) for v in ms], fired=False,
                   prev=st["prev"], c_star=int(ms.argmax()), c0=int(ms.argmin()))
        if step >= FIRST and st["prev"] is not None and acc < thr and acc - st["prev"] < rise:
            cs, c0 = int(ms.argmax()), int(ms.argmin())
            before = tsa.routing_k(m, task, rprobe)["ch_map"]
            res = split_op(m, opt_of(), cs, c0, NOISE_BASE + 1000 * seed + st["n"])
            st["n"] += 1
            row.update(fired=True, map_before=before, map_after=tsa.routing_k(m, task, rprobe)["ch_map"],
                       masses_after=[round(float(v), 5) for v in masses(m, data)], **res)
        st["prev"] = acc
        info["checks"].append(row)
    return trig


def with_trigger(rc, trig):
    def rc2(kw):
        kw = rc(kw)
        c0 = kw.get("check")

        def check(m, step, data):
            if c0 is not None:
                c0(m, step, data)
            trig(m, step)
        kw["check"] = check
        return kw
    return rc2


def task_of(cfg):
    return s33c.cfg_task(cfg)


def run(p):
    """p: arm, seed, iters, mlr (Muon arms), thr / rise (CHECKs only)."""
    arm = ARM[p["arm"]]
    cfg, holder, info = arm["cfg"], {}, dict(checks=[])
    a = s33c.cfg_arm(cfg, "HINGE")
    if arm["opt"] == "muon":
        rc, hst = s33c.make_rc("HINGE", p["mlr"], holder)
        ctx, opt_of = s25.muon_in_onset_run(holder), (lambda: holder["opts"][0].muon)
    else:
        rc, hst = tss.make_recipe(True, tss.TAU)
        ctx, opt_of = adam_capture(holder), (lambda: holder["opts"][0])
    trig = make_trigger(task_of(cfg), p["seed"], info, opt_of, arm["total"] - EVERY,
                        thr=p.get("thr", ACC_THR), rise=p.get("rise", RISE))
    with ctx:
        rec = s33c.path(cfg, a, p["seed"], p["iters"], with_trigger(rc, trig))
    if rec.get("ok"):
        c28 = [c for c in rec["curve"] if c[0] <= 28800]
        t28 = tbo.transition(c28)
        rec.update(arm=p["arm"], cfg=cfg, opt=arm["opt"], muon_lr=p.get("mlr") if arm["opt"] == "muon" else None,
                   checks=info["checks"], splits=sum(1 for c in info["checks"] if c["fired"]),
                   transition_28800=t28, bound_28800=t28 is not None, maps=s33c.maps_of(rec), main_sha=mt.MAIN_SHA,
                   n_opts_built=len(holder.get("opts", [])))
        try:
            rec["outcome"] = tsr.outcome(rec)
        except Exception as e:
            rec["outcome"] = f"n/a ({type(e).__name__})"
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _snap_state(o):
    return {id(p): {k: (v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in st.items()}
            for p, st in o.state.items()}


def synthetic(p):
    """The operation on a synthetic gate, under a populated Muon (MuonAdam) and Adam state: the rows as
    specified, every other parameter and every other optimizer state unchanged, W_g's state zero."""
    rows = []
    for kind in ("muon", "adam"):
        cfg = "b"
        a = s33c.cfg_arm(cfg, "HINGE")
        task = task_of(cfg)
        m = tsr.builder_for(a)(a, 260)()
        if kind == "muon":
            opt = s25.MuonAdam(s33c.make_groups({}, 0.005, False)(m), 1e-3)
            o_w, opts = opt.muon, opt.opts
        else:
            opt = s25._ADAM(m.parameters(), lr=1e-3)
            o_w, opts = opt, [opt]
        g = torch.Generator().manual_seed(7)
        for _ in range(3):                                          # populate the optimizer state
            x = task.make_batch(8, g)[0]
            loss = F.cross_entropy(tbo.logits_of(m, x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():                                       # the synthetic gate: rows of known scale
            m.W_g.copy_(torch.randn(m.W_g.shape, generator=g) * torch.linspace(0.5, 3.0, m.W_g.shape[0])[:, None])
        data = probe_data(task)
        ms = masses(m, data)
        cs, c0 = int(ms.argmax()), int(ms.argmin())
        W0 = m.W_g.detach().clone()
        P0 = {n: q.detach().clone() for n, q in m.named_parameters()}
        S0 = [_snap_state(o) for o in opts]
        seed = NOISE_BASE + 1000 * 260
        res = split_op(m, o_w, cs, c0, seed)
        _, n1, n2 = noise_for(W0[cs], seed)
        W = m.W_g.detach()
        rows_ok = (torch.equal(W[c0], W0[cs] + n2) and torch.equal(W[cs], W0[cs] + n1)
                   and all(torch.equal(W[r], W0[r]) for r in range(W.shape[0]) if r not in (cs, c0)))
        others_ok = all(torch.equal(q, P0[n]) for n, q in m.named_parameters() if n != "W_g")
        st_ok = True
        for o, s0 in zip(opts, S0):
            for q, st in o.state.items():
                if q is m.W_g:
                    continue
                for k, v in st.items():
                    st_ok &= (torch.equal(v, s0[id(q)][k]) if torch.is_tensor(v) else v == s0[id(q)][k])
        wst = o_w.state.get(m.W_g) or {}
        zero_ok = bool(wst) and all(bool((v == 0).all()) for v in wst.values() if torch.is_tensor(v))
        sd_ok = res["noise_sd"] == float(NOISE_REL * W0[cs].std())          # the same float32 computation
        rows.append((f"{kind}: on a synthetic gate (masses {[round(float(v), 3) for v in ms]}; c* {cs}, c0 {c0}) the operation "
                     f"sets W_g[c0] = W_g[c*] + n2 and W_g[c*] = W_g[c*] + n1 and leaves the other {W.shape[0] - 2} rows "
                     f"unchanged (rows {rows_ok}); noise sd {res['noise_sd']:.6g} = 0.1 x std(W_g[c*]) in float32 ({sd_ok})",
                     rows_ok and sd_ok))
        rows.append((f"{kind}: every other parameter and every other optimizer state unchanged", others_ok and st_ok))
        rows.append((f"{kind}: after the split W_g's {'Muon momentum' if kind == 'muon' else 'Adam state'} is zero "
                     f"({sorted(wst)} zeroed: {res['state_zeroed']})", zero_ok))
    return [[n, bool(v)] for n, v in rows]


# ── timing ───────────────────────────────────────────────────────────────────
def timing(p):
    """The median interval between optimizer steps over p["steps"] steps of the arm (trigger on, no check due
    in the window), and the evaluation time; run one child per arm at once for the pool's load."""
    import torch.optim.optimizer as topt
    out = {}
    for arm_name in p["which"]:
        arm = ARM[arm_name]
        stamps = []

        def post(opt, args, kwargs):
            if arm["opt"] == "adam" or isinstance(opt, s25.SliceMuon):
                stamps.append(time.perf_counter())
        h = topt.register_optimizer_step_post_hook(post)
        try:
            run(dict(arm=arm_name, seed=0, iters=p["steps"][arm["cfg"]], mlr=p["mlr"]))
        finally:
            h.remove()
        d = sorted(y - x for x, y in zip(stamps, stamps[1:]))
        cfg = arm["cfg"]
        a = s33c.cfg_arm(cfg, "HINGE")
        task = task_of(cfg)
        m = tsr.builder_for(a)(a, 0)()
        data, probe = eval_batch(task), probe_batch(task, 0)
        t0 = time.time()
        tbo.evaluate(m, task, data)
        sf = tsa.scale_stats if cfg == "a" else s33c.tscur.sc_stats((s33c.tscur.T1, s33c.tscur.T2))
        sf(m, task, probe, 1200)
        out[arm_name] = [d[len(d) // 2], time.time() - t0]
    return out
