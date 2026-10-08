#!/usr/bin/env python
"""
explore_h_stability_child.py — EXPLORATORY, not a result. Session H's S60 code that runs INSIDE a child process on THIS
branch's modules (no main-line hook), as S43's. Parent: explore_h_stability (readings and definitions in its docstring).
Imports explore_window_recipe_child (S43), explore_local_gate (batch 1), explore_b16_gates read-only; edits none.

CONFIGURATION: S43's LOCAL3_SLOW: test_short_conv's TASK (S=2, P=4, k=2, grouped, no convolution), 24000 updates,
evaluation every 1200, early stop after 3 evaluations >= 0.95, batch 1's LOCAL3 model (explore_local_gate.make_model,
W_h frozen), S5's SLOW schedule (gate W_in, W_g, window 1e-3 throughout; every other trainable parameter 1e-4 for updates
1-2400, 1e-3 after), stats explore_common2.stats_b2, no hinge. REF is this, unchanged (CHECK: REF|160 equals S43's
LOCAL3_SLOW|160 record).
  REF         S43's LOCAL3_SLOW, the gate softmax at temperature 1.
  TEMP_FLOOR  one learnable scalar tau (nn.Parameter, init 1.0, registered after every other parameter, in the gate group at
              1e-3 throughout); gate = softmax(z / max(tau, 0.5)) with z = W_g h (read = write). With tau frozen at 1 (not
              trainable) the model is REF bit for bit (CHECK).
  STABLEMAX   gate = s(z) / sum_c s(z_c), s(x) = x + 1 for x >= 0 and 1 / (1 - x) for x < 0 (Prieto et al. 2501.04697),
              in place of softmax on the gate logits (read = write). With softmax in place of s's normalisation the
              plumbing is REF bit for bit (CHECK).
  EMA_EVAL    training exactly REF's; an exponential moving average of every parameter (decay 0.999, initialised at the
              initial weights, updated after every optimizer step by a step post-hook, no bias correction) is swapped in
              for every evaluation (held-out accuracy, hence the early stop and the transition, and every statistic,
              hence the outcome) and swapped out after (bit-exact restore). The raw weights' held-out curve is also
              recorded at every evaluation (raw_curve). With decay 0 (EMA = the current weights) it is REF bit for bit
              (CHECK); REF's curve equals EMA_EVAL's raw curve while both run (by construction; printed).
  ORACLE_CONV test_short_conv.ARM["ceiling_conv"] (the perfect gate + conv "layer", this layout's recorded validity arm;
              batch 18's S52 found the plain perfect gate does not bind this layout), its recipe (one Adam 1e-3).
  ORACLE_PLAIN test_multilayer_binding's plain perfect gate (no conv), printed only.
MEASURED (every gated arm): at every evaluation, on the run's probe (test_multilayer_binding.probe_batch(task, seed)), the
gate's saturation: the fraction of positions (all positions of the probe's inputs) whose max_c g_t[c] > 0.99, and
"saturated" = that fraction > 0.9; TEMP_FLOOR: tau; EMA_EVAL: the raw weights' held-out accuracy.
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import register_optimizer_step_post_hook

import explore_common as ec
import explore_common2 as c2
import explore_local_gate as s1
import explore_window_recipe_child as s43c
import explore_b16_gates as g16
import test_binding_onset as tbo
import test_short_conv as tsc
from test_multilayer_binding import probe_batch, arms_for, MultiBDH, build
from test_channel_binding import ARCH
from test_router_reliability import run_one

LR, LR_WARM, WARM, ITERS = s43c.LR, s43c.LR_WARM, s43c.WARM, s43c.ITERS
TAU_FLOOR, EMA_DECAY = 0.5, 0.999
SAT_P, SAT_FRAC = 0.99, 0.9
GATE_T = ("W_in", "W_g", "gate_conv.conv_w", "gate_tau")
ARM = {"REF": dict(), "TEMP_FLOOR": dict(temp=True), "STABLEMAX": dict(stablemax=True), "EMA_EVAL": dict(ema=EMA_DECAY),
       "ORACLE_CONV": dict(oracle="ceiling_conv"), "ORACLE_PLAIN": dict(oracle="ceiling")}


def stablemax(z):
    s = torch.where(z >= 0, z + 1, 1 / (1 - torch.clamp(z, max=0.0)))
    return s / s.sum(-1, keepdim=True)


class TempLocalBDH(s1.LocalGateBDH):
    """LOCAL3 with gate = softmax(z / max(tau, floor)) (kind 'temp') or stablemax(z) (kind 'stablemax'); kind 'softmax'
    is LOCAL3's own computation."""

    def local_gates(self, v):
        u = self.gate_conv(v)
        h = torch.tanh(u @ self.W_in.T)
        z = h @ self.W_g.T
        if self.gate_fn == "temp":
            g = F.softmax(z / torch.clamp(self.gate_tau, min=TAU_FLOOR), dim=-1)
        elif self.gate_fn == "stablemax":
            g = stablemax(z)
        else:
            g = F.softmax(z, dim=-1)
        return g, g


def convert(m, kind, learn_tau=True):
    m.__class__ = TempLocalBDH
    m.gate_fn = kind
    if kind == "temp":
        m.gate_tau = nn.Parameter(torch.ones(()), requires_grad=learn_tau)
    return m


def builder(arm, p):
    def b(a, seed):
        def make():
            m = g16.freeze_wh(s1.make_model(a, seed))
            if arm.get("temp"):
                m = convert(m, "temp", learn_tau=p.get("learn_tau", True))
            elif arm.get("stablemax"):
                m = convert(m, p.get("gate_fn", "stablemax"))
            return m
        return make
    return b


def groups(holder):
    def param_groups(model):
        named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
        nm = dict(named)
        gate = [n for n in GATE_T if n in nm]
        rest = [n for n, _ in named if n not in gate]
        holder["names"] = [gate, rest]
        holder["groups"] = [dict(params=[nm[n] for n in gate], lr=LR), dict(params=[nm[n] for n in rest], lr=LR_WARM)]
        return holder["groups"]
    return param_groups


@torch.no_grad()
def saturation(m, task, probe):
    was = m.training
    m.eval()
    try:
        gr = m(probe[0], None)[2]
    finally:
        m.train(was)
    frac = float((gr.max(-1).values > SAT_P).float().mean())
    return dict(sat_frac=frac, saturated=frac > SAT_FRAC)


class EMA:
    def __init__(self, decay):
        self.decay, self.m, self.shadow, self.n = decay, None, None, 0

    def attach(self, m):
        self.m = m
        self.shadow = {n: q.detach().clone() for n, q in m.named_parameters()}

    @torch.no_grad()
    def step(self, *a):
        if self.m is None:
            return
        d = self.decay
        for n, q in self.m.named_parameters():
            self.shadow[n].mul_(d).add_(q.detach(), alpha=1 - d)
        self.n += 1

    def swap_in(self):
        self.saved = {n: q.detach().clone() for n, q in self.m.named_parameters()}
        with torch.no_grad():
            for n, q in self.m.named_parameters():
                q.copy_(self.shadow[n])

    def swap_out(self):
        with torch.no_grad():
            for n, q in self.m.named_parameters():
                q.copy_(self.saved[n])
        self.saved = None


def run_oracle(p, kind):
    task = ec.TASK
    a = tsc.ARM["ceiling_conv"] if kind == "ceiling_conv" else next(x for x in arms_for(task) if x["key"] == "ceiling")
    rec = run_one(a, p["seed"], p["iters"], tbo.EVAL_EVERY, task=task, stats_fn=tsc.conv_stats, grad_fn=tsc.conv_grad_norms,
                  lr=LR)
    if rec.get("ok"):
        rec.update(arm=p["arm"], tag=ec.tag(rec))
    return json.loads(json.dumps(rec, default=float))


def run(p):
    """p: arm, seed, iters; CHECK knobs: learn_tau (TEMP_FLOOR), gate_fn (STABLEMAX), decay (EMA_EVAL), eval_every, probe."""
    arm = ARM[p["arm"]]
    if arm.get("oracle"):
        return run_oracle(p, arm["oracle"])
    task = ec.TASK
    holder, meas, raw = {}, {}, []
    probe = probe_batch(task, p["seed"])
    ema = EMA(p.get("decay", arm["ema"])) if arm.get("ema") is not None else None
    b0 = builder(arm, p)

    def b(a, seed):
        mk = b0(a, seed)

        def make():
            m = mk()
            holder["model"] = m
            if ema is not None:
                ema.attach(m)
            return m
        return make
    c0 = s43c.make_check(True, holder, None, None)

    def check(m, step, d):
        c0(m, step, d)
        if p.get("probe", True):
            x = dict(saturation(m, task, probe))
            if hasattr(m, "gate_tau"):
                x["tau"] = float(m.gate_tau.detach())
            meas[str(step)] = x
    stats_fn = c2.stats_b2
    hooks, saved = [], None
    if ema is not None:
        hooks.append(register_optimizer_step_post_hook(ema.step))
        saved = tbo.evaluate
        ev0 = tbo.evaluate

        def evaluate(model, task_, data):
            raw.append(list(ev0(model, task_, data)))
            ema.swap_in()
            try:
                return ev0(model, task_, data)
            finally:
                ema.swap_out()
        tbo.evaluate = evaluate

        def stats_fn(model, task_, pr, step):
            ema.swap_in()
            try:
                return c2.stats_b2(model, task_, pr, step)
            finally:
                ema.swap_out()
    kw = dict(check=check, task=task, grad_fn=tsc.conv_grad_norms, lr=LR, builder=b, stats_fn=stats_fn,
              run_kw=dict(param_groups=groups(holder)))
    try:
        rec = run_one(s43c.A, p["seed"], p["iters"], p.get("eval_every", tbo.EVAL_EVERY), **kw)
    finally:
        for h in hooks:
            h.remove()
        if saved is not None:
            tbo.evaluate = saved
    if rec.get("ok"):
        rec.update(arm=p["arm"], meas=meas, group_names=holder.get("names"), tag=ec.tag(rec),
                   fail=None if rec["transition"] is not None else ec.fail_class(rec),
                   raw_curve=[[c[0]] + r for c, r in zip(rec["curve"], raw)] if ema is not None else None,
                   ema_steps=ema.n if ema is not None else None, ema_decay=ema.decay if ema is not None else None)
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs (unit; the record equalities run as runs in the parent) ──────────
def checks(p):
    rows = []
    task = ec.TASK
    seed = 470
    x = task.make_batch(8, torch.Generator().manual_seed(5))[0][:, :-1]
    z = torch.tensor([[-3.0, -0.5, 0.0, 0.5, 2.0]])
    s = torch.tensor([[1 / 4, 1 / 1.5, 1.0, 1.5, 3.0]])
    sm = stablemax(z)
    rows.append((f"stablemax on {z.tolist()[0]} = s/sum s with s = {s.tolist()[0]} ({[round(v, 4) for v in sm.tolist()[0]]}); "
                 f"sums to 1; monotone", torch.allclose(sm, s / s.sum()) and abs(float(sm.sum()) - 1) < 1e-6
                 and bool((sm[0, 1:] > sm[0, :-1]).all())))
    zz = torch.randn(4, 7, 2, requires_grad=True)
    stablemax(zz).sum().backward()
    rows.append(("stablemax is finite with a finite gradient on N(0,1) logits", bool(torch.isfinite(zz.grad).all())))
    a = s43c.A
    m0 = g16.freeze_wh(s1.make_model(a, seed))
    m0.eval()
    for kind, kw in (("temp", dict(learn_tau=False)), ("temp", dict(learn_tau=True)), ("softmax", {})):
        m1 = convert(g16.freeze_wh(s1.make_model(a, seed)), kind, **kw)
        m1.eval()
        with torch.no_grad():
            eq = torch.equal(m1(x)[0], m0(x)[0]) and torch.equal(m1(x)[2], m0(x)[2])
        rows.append((f"gate '{kind}' {kw}: logits and gate bitwise LOCAL3's at tau = 1 / softmax ({eq})", eq))
    m1 = convert(g16.freeze_wh(s1.make_model(a, seed)), "temp")
    with torch.no_grad():
        m1.gate_tau.fill_(0.2)
        g02 = m1(x)[2]
        m1.gate_tau.fill_(0.5)
        g05 = m1(x)[2]
        m1.gate_tau.fill_(2.0)
        g2 = m1(x)[2]
    hh = torch.tanh(m1.gate_conv(m1.embed(x)) @ m1.W_in.T) @ m1.W_g.T
    rows.append((f"TEMP_FLOOR: tau 0.2 gives the gate at tau 0.5 bitwise (the floor: {torch.equal(g02, g05)}); tau 2 gives "
                 f"softmax(z / 2) ({torch.allclose(g2, F.softmax(hh / 2, -1))})",
                 torch.equal(g02, g05) and torch.allclose(g2, F.softmax(hh / 2, -1))))
    pg = groups({})(m1)
    names = [[n for n, q in m1.named_parameters() if any(q is r for r in g["params"])] for g in pg]
    rows.append((f"TEMP_FLOOR: gate_tau is in the gate group ({names[0]}), registered after every other parameter "
                 f"({list(dict(m1.named_parameters()))[-1]}); W_h frozen", "gate_tau" in names[0]
                 and list(dict(m1.named_parameters()))[-1] == "gate_tau" and not m1.W_h.requires_grad))
    # EMA swap is bit-exact and the shadow follows the weights
    m2 = g16.freeze_wh(s1.make_model(a, seed))
    e = EMA(0.5)
    e.attach(m2)
    before = {n: q.detach().clone() for n, q in m2.named_parameters()}
    with torch.no_grad():
        for q in m2.parameters():
            q.add_(1.0)
    e.step()
    e.swap_in()
    shadow_ok = all(torch.allclose(q.detach(), before[n] + 0.5) for n, q in m2.named_parameters())
    e.swap_out()
    restore_ok = all(torch.equal(q.detach(), before[n] + 1.0) for n, q in m2.named_parameters())
    rows.append((f"EMA: after one step at decay 0.5 the shadow is the average ({shadow_ok}); swap_out restores the weights "
                 f"bit for bit ({restore_ok})", shadow_ok and restore_ok))
    pr = probe_batch(task, seed)
    sat = saturation(m0, task, pr)
    rows.append((f"saturation at init: fraction of probe positions with max gate > 0.99 = {sat['sat_frac']:.3f} (near-uniform "
                 f"gate at init: expected 0)", sat["sat_frac"] == 0.0 and not sat["saturated"]))
    return [[n, bool(v)] for n, v in rows]
