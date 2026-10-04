#!/usr/bin/env python
"""
explore_gate_prev_child.py — EXPLORATORY, not a result. S40's code that runs INSIDE a child process on the
main line's module tree at explore_main9c.MAIN_SHA (9c5939e). Parent: explore_gate_prev.

GATE_PREV: the recurrent gate also reads the previous token's embedding. Instrument.gates' recurrence
h_t = tanh(W_in v_t + W_h h_{t-1}) becomes h_t = tanh(W_in v_t + W_prev v_{t-1} + W_h h_{t-1}) with v_{-1} = 0
(v: the gate's input, the embedding). The model is the run path's own (test_stream_recipe.builder_for, the same
RNG draws), converted after construction (PrevBDH: MultiBDH.forward line for line with the gate computed as
above; every other path unchanged) and given W_prev ~ N(0, 0.1^2) (W_in's init scale, MultiBDH's
torch.randn(h_gate, D) * 0.1) from its own generator seeded 40_000_000 + seed; no other parameter's initial
value changes. W_prev is in the gate group: Muon at the Muon lr throughout (D8_PREV_M), Adam at 1e-3 throughout
(D8_PREV_A, test_slow_start's gate group).
Arms (S=8, P=4, k=16, conv, all 8 streams from step 1: test_stream_curriculum.run_sc with ARM["D8"], 28800):
  D8_PREV_M  S33's D8_HINGE (S25's groups at Muon lr 0.005; main's hinge) + GATE_PREV.
  D8_PREV_A  X's HINGE_D8 (test_slow_start's HINGE recipe: Adam, the slow groups, the hinge) + GATE_PREV; the
             slow groups rebuilt with W_prev in the gate group (test_slow_start.slow_groups' composition and lr
             switch otherwise).
DECODABILITY (S38's decoder, explore_gate_state_child.ncm) of h (recomputed with the W_prev term) at key and
stream-token positions, on S38's 512-sequence probe (seed 38000), at 1200, 2400, 4800 and 9600.
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import explore_main9c as mt
import explore_muon_scale_child as s33c
import explore_gate_state_child as s38c
import explore_muon_recipe as s25
import test_binding_onset as tbo
import test_slow_start as tss
import test_stream_curriculum as tscur
import test_stream_recipe as tsr
from bdh import BDH
from test_multilayer_binding import MultiBDH
from test_instrument_v2 import Instrument, TAU_END, gumbel

PREV_BASE = 40_000_000
PREV_SCALE = 0.1
GATE = tuple(s25.GATE)                       # W_in, W_h, W_g
GATE_P = GATE + ("W_prev",)
AT = (1200, 2400, 4800, 9600)
PROBE_N, PROBE_SEED = s38c.PROBE_N, s38c.PROBE_SEED
CFG = "b"
ARM = {"D8_PREV_M": dict(opt="muon"), "D8_PREV_A": dict(opt="adam")}


def prev_gates(self, tokens, v, tau=None):
    """Instrument.gates for gate_kind 'recurrent' with the W_prev term; any other kind: Instrument.gates."""
    if self.gate_kind != "recurrent":
        return Instrument.gates(self, tokens, v, tau)
    B, T, Dv = v.shape
    h = torch.zeros(B, self.h_gate, dtype=v.dtype)
    vp = torch.zeros(B, Dv, dtype=v.dtype)
    logits, hs = [], []
    for t in range(T):
        h = torch.tanh(v[:, t] @ self.W_in.T + vp @ self.W_prev.T + h @ self.W_h.T)
        hs.append(h)
        logits.append(h @ self.W_g.T)
        vp = v[:, t]
    lg = torch.stack(logits, dim=1)
    if self.gate_to_readout:
        self._h_seq = torch.stack(hs, dim=1)
    gr = F.softmax(lg, dim=-1)
    gw = gumbel(lg, tau if tau is not None else TAU_END) if self.mode == 'asym' else gr
    return gr, gw


class PrevBDH(MultiBDH):
    def forward(self, tokens, tau=None, track_sat=False):
        """MultiBDH.forward line for line, the gate from prev_gates."""
        v = self.embed(tokens)
        vg = F.layer_norm(v, (v.shape[-1],)) if self.gate_ln else v
        gr, gw = prev_gates(self, tokens, vg, tau)
        if self.gate_noise > 0 and self.training:
            eps = torch.randn(gr.shape, generator=self.noise_gen, dtype=gr.dtype)
            gr = F.softmax(torch.log(gr) + self.gate_noise * eps, dim=-1)
            gw = gr
        self.attn.G = (None if self.gate_kind == "none"
                       else torch.einsum("btk,bsk->bts", gr, gw).unsqueeze(1))
        if self.conv is None:
            logits, _ = BDH.forward(self, tokens)
        else:
            logits = self.forward_conv(tokens)
        if self.gate_to_readout:
            h = self._h_seq.detach() if self.readout_sg else self._h_seq
            logits = logits + (h @ self.W_ro.T) @ self.lm_head
        return logits, None, gr, gw


def to_prev(m, seed, zero=False):
    assert type(m) is MultiBDH and m.gate_kind == "recurrent" and not m.gate_ln, (type(m), m.gate_kind)
    m.__class__ = PrevBDH
    hg, Dv = m.W_in.shape
    if zero:
        W = torch.zeros(hg, Dv)
    else:
        W = torch.randn(hg, Dv, generator=torch.Generator().manual_seed(PREV_BASE + seed)) * PREV_SCALE
    m.W_prev = nn.Parameter(W, requires_grad=not zero)
    return m


def prev_builder(b0, zero=False):
    def builder(a, seed):
        mk = b0(a, seed)

        def make():
            return to_prev(mk(), seed, zero)
        return make
    return builder


def muon_groups(holder, mlr):
    """s33c.make_groups (slow) with the gate group extended by W_prev (when it trains)."""
    def param_groups(model):
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        nm = dict(named)
        gate = [n for n, _ in named if n in GATE_P]
        adam = [n for n, p in named if n not in GATE_P and s33c.is_adam(n, p)]
        rest = [n for n, p in named if n not in GATE_P and n not in adam]
        spec = [("gate", gate, "muon", mlr, mlr), ("rest", rest, "muon", s33c.SLOW * mlr, mlr),
                ("rest", adam, "adam", s33c.SLOW * s33c.LR, s33c.LR)]
        holder["groups"] = [dict(params=[nm[n] for n in ns], lr=l0, kind=k, tag=t, lr_after=l1, names=ns)
                            for t, ns, k, l0, l1 in spec if ns]
        return holder["groups"]
    return param_groups


def adam_groups(holder):
    """test_slow_start.slow_groups with W_prev (when it trains) in the gate group."""
    def param_groups(model):
        named = dict(model.named_parameters())
        gate = [n for n in GATE_P if n in named and named[n].requires_grad]
        rest = [n for n, p in model.named_parameters() if n not in gate and p.requires_grad]
        holder["names"] = [gate, rest]
        holder["groups"] = [dict(params=[named[n] for n in gate], lr=tss.LR),
                            dict(params=[named[n] for n in rest], lr=tss.LR_WARM)]
        return holder["groups"]
    return param_groups


def make_rc(opt, holder, mlr=None, zero=False, probe_out=None, keep=None):
    base, state = tss.make_recipe(False, tss.TAU, keep=keep)
    data = (s33c.cfg_task(CFG).make_batch(PROBE_N, torch.Generator().manual_seed(PROBE_SEED))[0]
            if probe_out is not None else None)

    def rc(kw):
        kw = dict(kw)
        kw["builder"] = prev_builder(kw["builder"], zero)
        kw = base(kw)
        kw["run_kw"] = dict(kw.get("run_kw") or {},
                            param_groups=muon_groups(holder, mlr) if opt == "muon" else adam_groups(holder))
        c0 = kw.get("check")

        def check(m, step, d):
            if c0 is not None:
                c0(m, step, d)
            if step >= tss.WARM:
                if opt == "muon":
                    for g in holder["groups"]:
                        g["lr"] = g["lr_after"]
                else:
                    holder["groups"][1]["lr"] = tss.LR
            if probe_out is not None and step in AT:
                probe_out[str(step)] = decode(m, s33c.cfg_task(CFG), data)
        kw["check"] = check
        return kw
    return rc, state


@torch.no_grad()
def gate_states(m, x):
    was = m.training
    m.eval()
    try:
        v = m.embed(x)
        h = torch.zeros(v.shape[0], m.h_gate, dtype=v.dtype)
        vp = torch.zeros(v.shape[0], v.shape[-1], dtype=v.dtype)
        hs, us = [], []
        for t in range(v.shape[1]):
            u = v[:, t] @ m.W_in.T + vp @ m.W_prev.T
            h = torch.tanh(u + h @ m.W_h.T)
            hs.append(h)
            us.append(u)
            vp = v[:, t]
        H, U = torch.stack(hs, 1), torch.stack(us, 1)
        gr = m(x, TAU_END)[2]
    finally:
        m.train(was)
    return H, U, float((torch.softmax(H @ m.W_g.T, -1) - gr).abs().max())


def decode(m, task, data):
    x = data[:, :-1]
    S, P = task.S, task.P
    j = torch.arange(S * P)
    H, U, diff = gate_states(m, x)
    lab = x[:, 3 * j]
    half = x.shape[0] // 2

    def acc(F_):
        return s38c.ncm(F_[:half].reshape(-1, F_.shape[-1]), lab[:half].reshape(-1),
                        F_[half:].reshape(-1, F_.shape[-1]), lab[half:].reshape(-1), S)
    return dict(h_key=acc(H[:, 3 * j + 1]), h_stream=acc(H[:, 3 * j]), u_key=acc(U[:, 3 * j + 1]),
                gate_recompute_diff=diff, chance=1.0 / S)


def run(p):
    """p: arm, seed, iters, mlr (Muon), zero (CHECKs: W_prev fixed at 0 and excluded)."""
    arm = ARM[p["arm"]]
    holder, out = {}, {}
    zero = bool(p.get("zero"))
    a = s33c.cfg_arm(CFG, "HINGE")
    rc, st = make_rc(arm["opt"], holder, mlr=p.get("mlr"), zero=zero, probe_out=None if zero else out)
    if arm["opt"] == "muon":
        with s25.muon_in_onset_run(holder):
            rec = s33c.path(CFG, a, p["seed"], p["iters"], rc)
    else:
        rec = s33c.path(CFG, a, p["seed"], p["iters"], rc)
    if rec.get("ok"):
        rec.update(arm=p["arm"], opt=arm["opt"], muon_lr=p.get("mlr") if arm["opt"] == "muon" else None, zero=zero,
                   decode=out, maps=s33c.maps_of(rec), main_sha=mt.MAIN_SHA,
                   opt_groups=[dict(names=n) for n in holder.get("names", [])] if arm["opt"] == "adam" else
                   [dict(tag=g["tag"], kind=g["kind"], names=g["names"]) for g in holder.get("groups", [])])
        try:
            rec["outcome"] = tsr.outcome(rec)
        except Exception as e:
            rec["outcome"] = f"n/a ({type(e).__name__})"
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def checks(p):
    rows = []
    a = s33c.cfg_arm(CFG, "HINGE")
    task = s33c.cfg_task(CFG)
    b = tsr.builder_for(a)
    m0 = b(a, 260)()
    m1 = prev_builder(b)(a, 260)()
    p0 = dict(m0.named_parameters())
    shared = [n for n, _ in m1.named_parameters() if n != "W_prev"]
    eq = sorted(shared) == sorted(p0) and all(torch.equal(dict(m1.named_parameters())[n], p0[n]) for n in shared)
    rows.append((f"at update 0 every parameter shared with the paired run is bitwise equal ({len(shared)} parameters; W_prev "
                 f"{tuple(m1.W_prev.shape)}, std {float(m1.W_prev.detach().std()):.4f} vs W_in's {float(m1.W_in.detach().std()):.4f}; class "
                 f"{type(m1).__name__})", eq))
    data = task.make_batch(PROBE_N, torch.Generator().manual_seed(PROBE_SEED))[0]
    d1 = decode(m1, task, data)
    x = data[:, :-1]
    j = torch.arange(task.S * task.P)
    with torch.no_grad():
        U0 = m0.embed(x) @ m0.W_in.T
    lab = x[:, 3 * j]
    half = x.shape[0] // 2
    F0 = U0[:, 3 * j + 1]
    a0 = s38c.ncm(F0[:half].reshape(-1, F0.shape[-1]), lab[:half].reshape(-1), F0[half:].reshape(-1, F0.shape[-1]),
                  lab[half:].reshape(-1), task.S)
    rows.append((f"the gate's input at key positions (W_in v_t + W_prev v_(t-1)) decodes the stream at update 0 at "
                 f"{d1['u_key']:.3f} with GATE_PREV (>= 0.9) and W_in v_t alone at {a0:.3f} without it (chance "
                 f"{1 / task.S:.3f}, within 0.05)", d1["u_key"] >= 0.9 and abs(a0 - 1 / task.S) <= 0.05))
    for opt in ("muon", "adam"):
        m = prev_builder(b)(a, 260)()
        h = {}
        gs = muon_groups(h, 0.005)(m) if opt == "muon" else adam_groups(h)(m)
        ids = [id(q) for g in gs for q in g["params"]]
        cover = len(ids) == len(set(ids)) and set(ids) == {id(q) for q in m.parameters()}
        gate0 = gs[0]["params"]
        in_gate = any(q is m.W_prev for q in gate0)
        names = [g.get("names") for g in gs] if opt == "muon" else h["names"]
        rows.append((f"{opt}: the optimizer groups cover every parameter once, W_prev in the gate group ({names})",
                     cover and in_gate))
    m = prev_builder(b, zero=True)(a, 260)()
    gs = muon_groups({}, 0.005)(m)
    s33g = s33c.make_groups({}, 0.005, True)(b(a, 260)())
    rows.append(("with W_prev fixed at 0 and excluded, the Muon groups are S33's (names "
                 f"{[g['names'] for g in gs] == [g['names'] for g in s33g]})",
                 [g["names"] for g in gs] == [g["names"] for g in s33g] and not m.W_prev.requires_grad))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    import time
    import torch.optim.optimizer as topt
    from test_channel_binding import eval_batch
    from test_multilayer_binding import probe_batch
    out = {}
    for arm_name in p["which"]:
        arm = ARM[arm_name]
        stamps = []

        def post(opt, args, kwargs):
            if arm["opt"] == "adam" or isinstance(opt, s25.SliceMuon):
                stamps.append(time.perf_counter())
        h = topt.register_optimizer_step_post_hook(post)
        try:
            run(dict(arm=arm_name, seed=0, iters=p["steps"], mlr=p["mlr"]))
        finally:
            h.remove()
        d = sorted(y - x for x, y in zip(stamps, stamps[1:]))
        a = s33c.cfg_arm(CFG, "HINGE")
        task = s33c.cfg_task(CFG)
        m = prev_builder(tsr.builder_for(a))(a, 0)()
        t0 = time.time()
        tbo.evaluate(m, task, eval_batch(task))
        tscur.sc_stats((tscur.T1, tscur.T2))(m, task, probe_batch(task, 0), 1200)
        te = time.time() - t0
        data = task.make_batch(PROBE_N, torch.Generator().manual_seed(PROBE_SEED))[0]
        t0 = time.time()
        decode(m, task, data)
        out[arm_name] = [d[len(d) // 2], te, time.time() - t0]
    return out
