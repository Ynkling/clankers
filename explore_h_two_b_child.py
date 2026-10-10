#!/usr/bin/env python
"""
explore_h_two_b_child.py — EXPLORATORY, not a result. Session H's S85 code that runs INSIDE a child process on THIS branch's
modules (no main-line hook), as S43's and S52's. Parent: explore_h_two_b (readings in its docstring). Imports read-only:
explore_keysplits_child (S52, identical to claude/outside-ideas' file), explore_window_recipe_child (S43),
explore_h_stability_child (ceiling_conv).

THE PATH: test_window_gate's Part A = S43's LOCAL3_SLOW (S=2, P=4, k=2, grouped, no conv, 24000 updates, LOCAL3, SLOW), run
through run_b below = explore_keysplits_child.run line for line at k = 2 (S52's run: S43's run_one call, the decoder
after the evaluations, and S52's dense measurements of the gate after updates 0, 25, ..., 1200, 1300, ..., 2400, then every
1200 — routing_k's pooled eta^2 at KEY / VAL positions, the margin, the map; eval mode in a step post-hook, training
untouched), with two switches:
  hinge     S52's key term (attach_key_hinge: 1.0 x relu(eta2_key - 0.2) on the read gate at key positions, updates 1-2400).
  slowgate  the gate group (W_in, W_g, window) at 1e-4 for updates 1-1200, then 1e-3 (switched right after the evaluation
            at 1200); every other trainable parameter as SLOW (1e-4 for updates 1-2400, 1e-3 after).
  REF                 neither (= WIN3_SLOW; CHECK: equals S43's LOCAL3_SLOW record).
  SLOWGATE            slowgate.
  KEYHINGE            hinge (= S52's W2_KEYHINGE; CHECK: equals its record).
  SLOWGATE_KEYHINGE   both.
  ORACLE              test_short_conv's ceiling_conv (the main line's CEIL_A).
"""

import json

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

import explore_common as ec
import explore_common2 as c2
import explore_window_recipe_child as s43c
import explore_keysplits_child as s52c
import explore_h_stability_child as h60c
import explore_b16_gates as g16
import test_binding_onset as tbo
import test_short_conv as tsc
from test_router_reliability import run_one
from test_multilayer_binding import probe_batch

GATE_SWITCH = 1200
ARM = {"REF": dict(), "SLOWGATE": dict(slowgate=True), "KEYHINGE": dict(hinge=True),
       "SLOWGATE_KEYHINGE": dict(hinge=True, slowgate=True), "ORACLE": dict(oracle=True)}


def groups(holder, slowgate):
    """explore_window_recipe_child.groups with the gate group's starting lr LR_WARM when slowgate."""
    def param_groups(model):
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        nm = dict(named)
        gate = [n for n in s43c.GATE_L if n in nm]
        rest = [n for n, _ in named if n not in gate]
        holder["names"] = [gate, rest]
        holder["groups"] = [dict(params=[nm[n] for n in gate], lr=s43c.LR_WARM if slowgate else s43c.LR),
                            dict(params=[nm[n] for n in rest], lr=s43c.LR_WARM)]
        return holder["groups"]
    return param_groups


def run_b(p):
    """p: arm, seed, iters; CHECK knobs: dense, keep, log_lr."""
    arm = ARM[p["arm"]]
    seed, iters = p["seed"], p["iters"]
    task = ec.TASK
    holder, out, dense, box, hs = {}, {}, {}, {}, {}
    data = g16.probe_data(task)
    probe = probe_batch(task, seed)
    want = set(s52c.at_steps(iters)) if p.get("dense", True) else set()
    b0 = s43c.builder(True)
    lrlog = {}

    def builder(aa, s):
        mk = b0(aa, s)

        def make():
            m = mk()
            if arm.get("hinge"):
                s52c.attach_key_hinge(m, task.S, task.P, s52c.TAU, s52c.LAM, s52c.WINDOW, hs)
            box["m"] = m
            if want:
                dense["0"] = s52c.cells(m, task, probe)
            return m
        return make
    n = [0]

    def post(opt, args, kwargs):
        n[0] += 1
        if p.get("log_lr") and n[0] in (1, 1199, 1200, 1201, 1202, 2400, 2401, 2402):
            lrlog[str(n[0])] = [g["lr"] for g in opt.param_groups]
        if n[0] in want:
            dense[str(n[0])] = s52c.cells(box["m"], task, probe)
    c0 = s43c.make_check(True, holder, out, data)

    def check(m, step, d):
        c0(m, step, d)
        if arm.get("slowgate") and step >= GATE_SWITCH:
            holder["groups"][0]["lr"] = s43c.LR
    h = register_optimizer_step_post_hook(post)
    try:
        kw = dict(check=check, task=task, grad_fn=tsc.conv_grad_norms, lr=s43c.LR, builder=builder, stats_fn=c2.stats_b2,
                  run_kw=dict(param_groups=groups(holder, arm.get("slowgate", False))))
        if p.get("keep") is not None:
            kw["keep"] = p["keep"]
        rec = run_one(s43c.A, seed, iters, p.get("eval_every", tbo.EVAL_EVERY), **kw)
    finally:
        h.remove()
    if rec.get("ok"):
        rec.update(arm=p["arm"], decode=out, dense=dense, n_updates=n[0], group_names=holder.get("names"), lr_log=lrlog,
                   hinge=(dict(window=s52c.WINDOW, lam=s52c.LAM, tau=s52c.TAU, fired=hs.get("fired"), n=hs.get("n"),
                               windows=hs.get("windows"), weights=hs.get("weights"), eta_at=hs.get("eta_at"))
                          if arm.get("hinge") else None))
        rec["fail"] = None if rec["transition"] is not None else ec.fail_class(rec)
        rec["tag"] = ec.tag(rec)
    return json.loads(json.dumps(rec, default=float))


def run(p):
    if ARM[p["arm"]].get("oracle"):
        return h60c.run_oracle(p, "ceiling_conv")
    return run_b(p)


def checks(p):
    """SLOWGATE's groups and lrs through the real path (2402 updates, evaluation included)."""
    r = run_b(dict(arm="SLOWGATE", seed=2310, iters=2402, dense=False, log_lr=True))
    lg = r["lr_log"]
    lr, w = s43c.LR, s43c.LR_WARM
    want = {"1": [w, w], "1199": [w, w], "1200": [w, w], "1201": [lr, w], "2400": [lr, w], "2401": [lr, lr]}
    ok = all(lg.get(k) == v for k, v in want.items()) and r["group_names"][0] == list(s43c.GATE_L)
    rows = [(f"SLOWGATE: groups {r['group_names'][0]} + {len(r['group_names'][1])} others; lrs [gate, rest] at updates "
             f"{ {k: lg.get(k) for k in want} } (want gate 1e-4 through 1200, 1e-3 from 1201; rest 1e-4 through 2400)", ok)]
    r0 = run_b(dict(arm="REF", seed=2310, iters=2402, dense=False, log_lr=True))
    ok = r0["lr_log"].get("1") == [lr, w] and r0["lr_log"].get("1201") == [lr, w] and r0["lr_log"].get("2401") == [lr, lr]
    rows.append((f"REF's lrs at updates 1, 1201, 2401: {r0['lr_log'].get('1')}, {r0['lr_log'].get('1201')}, {r0['lr_log'].get('2401')}", ok))
    return [[n, bool(v)] for n, v in rows]
