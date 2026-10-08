#!/usr/bin/env python
"""
explore_h_copy2x2_child.py — EXPLORATORY, not a result. Session H's S61 code that runs INSIDE a child process on the main
line's module tree at explore_main9c.MAIN_SHA (9c5939e). Parent: explore_h_copy2x2 (readings in its docstring). Imports
S58/S58b's children (explore_h_collapse_child, explore_h_eight_child), S36/S48's trigger code read-only; edits none.

CONFIGURATION: S58b's: run_sc with ARM["D8"] (S=8, P=4, k=16, conv, all 8 streams from step 1), LOCAL3 gate, SLOW (Adam),
no hinge, evaluation every 1200, 43200 updates. Every trigger arm uses the same plateau trigger
(explore_h_collapse_child.make_trigger_op = S48's make_trigger line for line with the operation as a parameter: checks every
2400 from 4800 through 40800 on S36's 64-sequence probe; fires if probe acc < 0.95 and it rose < 0.02 since the previous
check; <= 3 operations, >= 4800 apart; KEYMASS target c* (largest key-position read mass) -> c0 (smallest)).
OPERATIONS (the split's two actions):
  SPLIT         S36's split_op: W_g[c0] = w + n2, W_g[c*] = w + n1 (w = W_g[c*], n ~ N(0, (0.1 std(w))^2) from the
                generator 12_000_000 + 1000 seed + count), then W_g's Adam state zeroed (exp_avg, exp_avg_sq, step).
  RESET         W_g's Adam state zeroed only (explore_h_collapse_child.reset_op).
  COPY          the copy with noise, exactly split_op's row lines (same noise, same generator), W_g's Adam state untouched.
  COPY_NONOISE  W_g[c0] = W_g[c*] exactly (no noise, c*'s row unchanged), W_g's Adam state untouched.
  NONE          no trigger (= LOCAL3_SLOW_D8_A to 43200).
  ORACLE        the perfect gate ceil8_D8 + conv (S58b's).
"""

import contextlib

import torch

import explore_split_child as s36c
import explore_b16_gates as g16
import explore_b16_main as b16
import explore_h_collapse_child as hcc
import explore_h_eight_child as h8c
import explore_window_d8_child as s48c

CFG = "b"
TOTAL = 43200
OPS = {}


def copy_op(m, o, cs, c0, seed):
    """split_op's row lines (S36), without the optimizer-state reset."""
    W = m.W_g
    with torch.no_grad():
        w = W[cs].clone()
        before = float((W[cs] - W[c0]).norm())
        sd, n1, n2 = s36c.noise_for(w, seed)
        W[c0] = w + n2
        W[cs] = w + n1
    return dict(noise_sd=float(sd), noise_seed=seed, row_dist_before=before,
                row_dist_after=float((W[cs] - W[c0]).detach().norm()), state_zeroed=[])


def copy_nonoise_op(m, o, cs, c0, seed):
    W = m.W_g
    with torch.no_grad():
        before = float((W[cs] - W[c0]).norm())
        W[c0] = W[cs].clone()
    return dict(row_dist_before=before, row_dist_after=float((W[cs] - W[c0]).detach().norm()), state_zeroed=[])


OPS.update(SPLIT=s36c.split_op, RESET=hcc.reset_op, COPY=copy_op, COPY_NONOISE=copy_nonoise_op)
ARM = {"SPLIT": dict(op="SPLIT"), "RESET": dict(op="RESET"), "COPY": dict(op="COPY"), "COPY_NONOISE": dict(op="COPY_NONOISE"),
       "NONE": dict(), "ORACLE": dict(oracle=True)}


def run(p):
    arm = ARM[p["arm"]]
    if arm.get("oracle"):
        return h8c.run(dict(p, arm="ORACLE"))
    task = b16.task_of(CFG)
    holder, info = {}, dict(checks=[])
    rc, _ = b16.make_rc("adam", holder, b16.GATE_LOCAL, tau=None, convert=(g16.to_local,), probe=None, keep=p.get("keep"))
    if arm.get("op"):
        ctx, opt_of = s36c.adam_capture(holder), (lambda: holder["opts"][0])
        trig = hcc.make_trigger_op(task, p["seed"], info, opt_of, p.get("total", TOTAL) - hcc.EVERY, OPS[arm["op"]],
                                   thr=p.get("thr", hcc.ACC_THR))
        rc = s36c.with_trigger(rc, trig)
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        rec = b16.path(CFG, b16.arm_of(CFG), p["seed"], p["iters"], rc, eval_every=p.get("eval_every"))
    return b16.finish(rec, arm=p["arm"], cfg=CFG, opt="adam", kind="local", groups=b16.group_names(holder),
                      checks=info["checks"], splits=sum(1 for c in info["checks"] if c["fired"]), op=arm.get("op"))


def checks(p):
    """Each operation on a LOCAL3 model after one Adam step: rows and Adam state as specified."""
    rows = []
    a = b16.arm_of(CFG)
    task = b16.task_of(CFG)
    seed = 480
    res = {}
    for name, op in OPS.items():
        m = s48c._model(a, seed)
        opt = torch.optim.Adam(m.parameters(), lr=b16.LR)
        s48c._one_step(m, task, opt)
        st0 = {k: v.clone() for k, v in opt.state[m.W_g].items() if torch.is_tensor(v)}
        W0 = m.W_g.detach().clone()
        out = op(m, opt, 3, 7, 12_000_000 + 1000 * seed)
        st1 = {k: v.clone() for k, v in opt.state[m.W_g].items() if torch.is_tensor(v)}
        res[name] = (W0, m.W_g.detach().clone(), st0, st1, out)
    W0, Ws, s0, ss, _ = res["SPLIT"]
    _, Wc, _, sc, _ = res["COPY"]
    _, Wn, _, sn, _ = res["COPY_NONOISE"]
    _, Wr, _, sr, _ = res["RESET"]
    others = [r for r in range(W0.shape[0]) if r not in (3, 7)]
    rows.append(("COPY's rows equal SPLIT's bit for bit (same noise); COPY leaves W_g's Adam state untouched; SPLIT zeroes it",
                 torch.equal(Wc, Ws) and all(torch.equal(sc[k], s0[k]) for k in s0)
                 and all(float(ss[k].abs().max()) == 0 for k in ss)))
    rows.append(("COPY_NONOISE: W_g[c0] = W_g[c*] exactly, c*'s and the other rows unchanged, Adam state untouched",
                 torch.equal(Wn[7], W0[3]) and torch.equal(Wn[3], W0[3]) and all(torch.equal(Wn[r], W0[r]) for r in others)
                 and all(torch.equal(sn[k], s0[k]) for k in s0)))
    rows.append(("RESET: W_g unchanged, Adam state zeroed", torch.equal(Wr, W0) and all(float(sr[k].abs().max()) == 0 for k in sr)))
    rows.append(("SPLIT / COPY change only rows c*, c0", all(torch.equal(Ws[r], W0[r]) and torch.equal(Wc[r], W0[r]) for r in others)
                 and not torch.equal(Ws[7], W0[7])))
    return [[n, bool(v)] for n, v in rows]
