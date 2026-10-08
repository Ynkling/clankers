#!/usr/bin/env python
"""
explore_h_hardness_child.py — EXPLORATORY, not a result. Session H's S63 code that runs INSIDE a child process on the main
line's module tree at explore_main9c.MAIN_SHA (9c5939e). Parent: explore_h_hardness (readings in its docstring).

CONFIGURATION: S58b's: run_sc with ARM["D8"] (S=8, P=4, k=16, conv), LOCAL3 gate, SLOW (Adam), no hinge, 43200 updates,
evaluation every 1200, and S48's KEYMASS split on every learned arm (explore_window_d8_child.make_trigger: checks every 2400
from 4800 through 40800, <= 3 splits, >= 4800 apart).
  REF     LOCAL3 + SLOW + KEYMASS (= S58b's SPLIT recipe).
  ANNEAL  REF with the gate at temperature tau(n): g = softmax(z / tau(n)) for the read and the write gate, n = the number of
          optimizer updates completed (a step post-hook counts them; a training forward of update u sees tau(u - 1), an
          evaluation after update s sees tau(s)); tau = 1 for n <= 2400, 1 - 0.75 (n - 2400) / 21600 for 2400 < n < 24000,
          0.25 for n >= 24000 (linear from 1 to 0.25 over updates 2400-24000).
  HARD_W  REF with, for n >= 2400, the write gate one-hot(argmax_c z) (no gradient) and the read gate softmax(z) (the only
          gradient path into the gate); before that LOCAL3's gate. Also in evaluations (the model is defined that way).
  ORACLE  perfect gate ceil8_D8 + conv (S58b's).
"""

import contextlib

import torch
import torch.nn.functional as F
from torch.optim.optimizer import register_optimizer_step_post_hook

import explore_split_child as s36c
import explore_b16_gates as g16
import explore_b16_main as b16
import explore_window_d8_child as s48c
import explore_h_eight_child as h8c

CFG = "b"
TOTAL = 43200
T0, T1, TAU_END, HARD_FROM = 2400, 24000, 0.25, 2400
ARM = {"REF": dict(), "ANNEAL": dict(kind="anneal"), "HARD_W": dict(kind="hard"), "ORACLE": dict(oracle=True)}


def tau_of(n, end=TAU_END):
    if n <= T0:
        return 1.0
    if n >= T1:
        return float(end)
    return 1.0 - (1.0 - end) * (n - T0) / (T1 - T0)


class HardLocalBDH(g16.LocalBDH):
    def local_gates(self, v):
        u = self.gate_conv(v)
        h = torch.tanh(u @ self.W_in.T)
        z = h @ self.W_g.T
        n = self.n_upd
        if self.kind == "anneal":
            t = tau_of(n, self.tau_end)
            g = F.softmax(z / t, dim=-1) if t != 1.0 else F.softmax(z, dim=-1)
            return g, g
        g = F.softmax(z, dim=-1)
        if self.kind == "hard" and n >= self.hard_from:
            return g, F.one_hot(z.detach().argmax(-1), z.shape[-1]).to(z.dtype)
        return g, g


def to_hard(kind, tau_end=TAU_END, hard_from=HARD_FROM):
    def conv(m, seed):
        assert type(m) is g16.LocalBDH, type(m)
        m.__class__ = HardLocalBDH
        m.kind, m.tau_end, m.hard_from, m.n_upd = kind, tau_end, hard_from, 0
        return m
    return conv


def run(p):
    arm = ARM[p["arm"]]
    if arm.get("oracle"):
        return h8c.run(dict(p, arm="ORACLE"))
    task = b16.task_of(CFG)
    holder, info, box = {}, dict(checks=[]), {}
    convert = [g16.to_local]
    if arm.get("kind"):
        convert.append(to_hard(arm["kind"], p.get("tau_end", TAU_END), p.get("hard_from", HARD_FROM)))

    def on_build(m):
        box["m"] = m
    rc, _ = b16.make_rc("adam", holder, b16.GATE_LOCAL, tau=None, convert=tuple(convert), probe=None, keep=p.get("keep"),
                        on_build=on_build)
    opt_of = lambda: holder["opts"][0]                                                  # noqa: E731
    rc = s36c.with_trigger(rc, s48c.make_trigger(task, p["seed"], info, opt_of, p.get("total", TOTAL) - s48c.EVERY,
                                                 max_splits=3, thr=p.get("thr", s48c.ACC_THR)))

    def count(o, a, k):
        m = box.get("m")
        if m is not None and hasattr(m, "n_upd"):
            m.n_upd += 1
    h = register_optimizer_step_post_hook(count)
    try:
        with s36c.adam_capture(holder):
            rec = b16.path(CFG, b16.arm_of(CFG), p["seed"], p["iters"], rc, eval_every=p.get("eval_every"))
    finally:
        h.remove()
    m = box.get("m")
    return b16.finish(rec, arm=p["arm"], cfg=CFG, opt="adam", kind="local", groups=b16.group_names(holder),
                      checks=info["checks"], splits=sum(1 for c in info["checks"] if c["fired"]), hard=arm.get("kind"),
                      n_upd=getattr(m, "n_upd", None), tau_final=tau_of(m.n_upd, m.tau_end) if arm.get("kind") == "anneal" else None)


def checks(p):
    rows = []
    vals = {n: tau_of(n) for n in (0, 2400, 2401, 13200, 24000, 30000)}
    rows.append((f"tau schedule {vals}", vals[0] == 1 and vals[2400] == 1 and abs(vals[13200] - 0.625) < 1e-12
                 and vals[24000] == 0.25 and vals[30000] == 0.25 and 0.25 < vals[2401] < 1))
    a = b16.arm_of(CFG)
    task = b16.task_of(CFG)
    x = task.make_batch(4, torch.Generator().manual_seed(5))[0][:, :-1]
    m0 = s48c._model(a, 490)
    v = m0.embed(x)
    g0, _ = m0.local_gates(v)
    mh = to_hard("hard")(s48c._model(a, 490), 490)
    mh.n_upd = 2399
    r1, w1 = mh.local_gates(mh.embed(x))
    mh.n_upd = 2400
    rs = torch.get_rng_state()
    r2, w2 = mh.local_gates(mh.embed(x))
    onehot = bool(((w2 == 0) | (w2 == 1)).all()) and bool((w2.sum(-1) == 1).all())
    rows.append((f"HARD_W: before 2400 LOCAL3's gates ({torch.equal(r1, g0) and torch.equal(w1, g0)}); from 2400 the write gate is "
                 f"one-hot of argmax ({onehot}, argmax agrees {torch.equal(w2.argmax(-1), g0.argmax(-1))}) with no gradient "
                 f"({not w2.requires_grad}); the read gate is LOCAL3's soft gate with gradient ({torch.equal(r2, g0) and r2.requires_grad}); "
                 f"global RNG untouched ({torch.equal(rs, torch.get_rng_state())})",
                 torch.equal(r1, g0) and torch.equal(w1, g0) and onehot and not w2.requires_grad and torch.equal(r2, g0)
                 and r2.requires_grad and torch.equal(w2.argmax(-1), g0.argmax(-1)) and torch.equal(rs, torch.get_rng_state())))
    ma = to_hard("anneal")(s48c._model(a, 490), 490)
    ma.n_upd = 13200
    ra, wa = ma.local_gates(ma.embed(x))
    z = torch.tanh(ma.gate_conv(ma.embed(x)) @ ma.W_in.T) @ ma.W_g.T
    ma.n_upd = 2400
    rb, _ = ma.local_gates(ma.embed(x))
    rows.append((f"ANNEAL: at n = 13200 both gates = softmax(z / 0.625) ({torch.allclose(ra, F.softmax(z / 0.625, -1)) and torch.equal(ra, wa)}); "
                 f"at n = 2400 LOCAL3's gate bitwise ({torch.equal(rb, g0)})",
                 torch.allclose(ra, F.softmax(z / 0.625, -1)) and torch.equal(ra, wa) and torch.equal(rb, g0)))
    return [[n, bool(v_)] for n, v_ in rows]
