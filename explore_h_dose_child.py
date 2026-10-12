#!/usr/bin/env python
"""
explore_h_dose_child.py — EXPLORATORY, not a result. Session H's S87c code that runs INSIDE a child process on THIS branch's
modules (no main-line hook), as S85's. Parent: explore_h_dose (readings in its docstring).

THE PATH: S85's run_b (explore_h_two_b_child: S52's run at k = 2 = test_window_gate's Part A, WIN3_SLOW, with S52's dense
measurements) with the key term's weight and window as parameters. The term is S52's attach_key_hinge (explore_keysplits_
child, unchanged) called with lam and window instead of its constants LAM = 1.0 and WINDOW = 2400; tau = 0.2 throughout.
  KEYHINGE   lam 1.0, window 2400 (the drafted constants; = S85's KEYHINGE).
  LAM03      lam 0.3, window 2400.
  LAM3       lam 3.0, window 2400.
  WIN1200    lam 1.0, window 1200.
  ORACLE     ceiling_conv (S85's).
"""

import json

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

import explore_common as ec
import explore_common2 as c2
import explore_window_recipe_child as s43c
import explore_keysplits_child as s52c
import explore_h_two_b_child as h85c
import explore_h_stability_child as h60c
import explore_b16_gates as g16
import test_binding_onset as tbo
import test_short_conv as tsc
from test_router_reliability import run_one
from test_multilayer_binding import probe_batch

ARM = {"KEYHINGE": dict(lam=1.0, window=2400), "LAM03": dict(lam=0.3, window=2400), "LAM3": dict(lam=3.0, window=2400),
       "WIN1200": dict(lam=1.0, window=1200), "ORACLE": dict(oracle=True)}


def run_d(p):
    """S85's run_b with the hinge's lam and window as parameters (slowgate off)."""
    arm = ARM[p["arm"]]
    lam, window = p.get("lam", arm["lam"]), p.get("window", arm["window"])
    seed, iters = p["seed"], p["iters"]
    task = ec.TASK
    holder, out, dense, box, hs = {}, {}, {}, {}, {}
    data = g16.probe_data(task)
    probe = probe_batch(task, seed)
    want = set(s52c.at_steps(iters)) if p.get("dense", True) else set()
    b0 = s43c.builder(True)

    def builder(aa, s):
        mk = b0(aa, s)

        def make():
            m = mk()
            s52c.attach_key_hinge(m, task.S, task.P, s52c.TAU, lam, window, hs)
            box["m"] = m
            if want:
                dense["0"] = s52c.cells(m, task, probe)
            return m
        return make
    n = [0]

    def post(opt, args, kwargs):
        n[0] += 1
        if n[0] in want:
            dense[str(n[0])] = s52c.cells(box["m"], task, probe)
    c0 = s43c.make_check(True, holder, out, data)
    h = register_optimizer_step_post_hook(post)
    try:
        kw = dict(check=c0, task=task, grad_fn=tsc.conv_grad_norms, lr=s43c.LR, builder=builder, stats_fn=c2.stats_b2,
                  run_kw=dict(param_groups=h85c.groups(holder, False)))
        rec = run_one(s43c.A, seed, iters, p.get("eval_every", tbo.EVAL_EVERY), **kw)
    finally:
        h.remove()
    if rec.get("ok"):
        rec.update(arm=p["arm"], decode=out, dense=dense, n_updates=n[0], group_names=holder.get("names"),
                   hinge=dict(window=window, lam=lam, tau=s52c.TAU, fired=hs.get("fired"), n=hs.get("n"),
                              windows=hs.get("windows"), weights=hs.get("weights"), eta_at=hs.get("eta_at")))
        rec["fail"] = None if rec["transition"] is not None else ec.fail_class(rec)
        rec["tag"] = ec.tag(rec)
    return json.loads(json.dumps(rec, default=float))


def run(p):
    if ARM[p["arm"]].get("oracle"):
        return h60c.run_oracle(p, "ceiling_conv")
    return run_d(p)


def checks(p):
    """The term's weight through the real path at S52's logged updates (1, 2399, 2400, 2401, 2402; 2402 updates run): lam at
    update 1; lam at 2399-2400 when the window is 2400, 0 when it is 1200; 0 at 2401-2402."""
    rows = []
    for a in ("KEYHINGE", "LAM03", "LAM3", "WIN1200"):
        arm = ARM[a]
        r = run_d(dict(arm=a, seed=2630, iters=2402, dense=False))
        w = (r.get("hinge") or {}).get("weights") or {}
        on = arm["lam"] if arm["window"] >= 2400 else 0.0
        want = {"1": arm["lam"], "2399": on, "2400": on, "2401": 0.0, "2402": 0.0}
        ok = all(w.get(k) == v for k, v in want.items()) and r["hinge"]["window"] == arm["window"]
        rows.append((f"{a}: the term's weight at the logged updates {w} (want lam {arm['lam']} through {arm['window']}, 0 after; "
                     f"S52 logs 1, 2399-2402)", ok))
    return [[n, bool(v)] for n, v in rows]
