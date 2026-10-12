#!/usr/bin/env python
"""
explore_f_horizon_child.py — EXPLORATORY, not a result. S88b's run code (parent: explore_f_horizon, whose docstring holds
the readings). Runs INSIDE a child process on this branch's own modules (no main-line hook), 1 torch thread.

TASKS: 1x = S70's RandHeaderTask-lite, explore_f_tasks.RandHeaderTask(P=4, nb=2, lmin=1, lmax=3) (S = 2, n_vals = 16,
  n_q = 1; L = 23). 2x = RandHeader2x: the same draws, re-laid out so that every key sits twice as far from its block's CTX
  (the cue): one FILL token after each CTX and two between consecutive pairs of a block, so a block of n pairs
  [CTX k1 v1 k2 v2 ...] becomes [CTX F k1 v1 F F k2 v2 ...] (length 4n) and the j-th key's distance to its cue is
  2(2j − 1) instead of 2j − 1; pairs stay adjacent; L = 35; the vocabulary gains FILL.
MODEL: one channel (test_short_conv's arm B, k = 1, built on the task's vocabulary as S59's SINGLE arms), converted to the
  delta memory with β = 1, L2 keys, tied write (explore_delta_mem.to_delta, the parallel form), decay
    FIXED    α = 0.95;
    ROUTED   S59's routed decay (Raven's), written relative to the fixed decay (RelRouted): log α_t = ln(0.95)·f_t·g_t,
             f_t = exp(δ)·softplus(w·v_t) / softplus(0), with w = 0 (32-d) and δ = 0 at init; the same function as S59's
             −softplus(w·v_t)·exp(Δ)·g_t with Δ = Δ0 + δ (exp(Δ0)·ln 2 = −ln 0.95), so the same family and the same
             Adam dynamics. The decay matrix is computed as decay^(t−s) · exp(Λ_t − Λ_s), Λ = cumsum of
             ln(0.95)·(f·g − 1), which is exactly 0 at init: frozen, ROUTED is FIXED bit for bit (CHECK).
  The perfect gate (k = 2, test_channel_binding's 'ceiling'), the same delta memory with the fixed decay: validity.
RECIPE: S59's single-channel recipe — Adam 1e-3 on every parameter; evaluation every 1200 on 2048 held-out sequences;
  early stop after 3 evaluations >= 0.95 — for 24000 updates (S70's budget).
DIAGNOSTICS: at every evaluation, the held-out accuracy by the queried key's distance to its cue (1, 3, 5 at 1x; 2, 6, 10
  at 2x); at the end, the routed decay's α at CTX / KEY / VAL / FILL per layer, its parameters, and the write-order
  accuracy.
"""

import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import register_optimizer_step_post_hook

import explore_common as ec
import explore_common2 as c2
import explore_delta_mem as dm
import explore_f_tasks as ft
import test_binding_onset as tbo
import test_short_conv as tsc
from test_instrument_v2 import decay_mask
from test_multilayer_binding import build, probe_batch
from test_channel_binding import ARCH, arms as cb_arms
from test_router_reliability import run_one

HERE = os.path.dirname(os.path.abspath(__file__))
LR = 1e-3
ITERS = 24000


# ── The 2x task ──────────────────────────────────────────────────────────────
class RandHeader2x(ft.RandHeaderTask):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.body1 = self.body_len                                           # the 1x body's length
        self.fill = self.vocab
        self.vocab += 1
        self.key += "x2"
        self.label += ", cue-to-key distance x2 (FILL tokens)"
        self.body_len = 4 * self.S * self.P
        self.L = self.body_len + 3 * self.n_q
        self.T = self.L - 1
        self.qpos = [self.body_len + 3 * j + 1 for j in range(self.n_q)]

    def relayout(self, body):
        S, P, Fl = self.S, self.P, self.fill
        out = []
        for row in body.tolist():
            seq, first = [], False
            for t in row:
                if t < S:
                    seq += [t, Fl]
                    first = True
                elif t < S + P:
                    if not first:
                        seq += [Fl, Fl]
                    seq.append(t)
                    first = False
                else:
                    seq.append(t)
            out.append(seq)
        return torch.tensor(out, dtype=torch.long)

    def make_batch(self, B, gen):
        x, qs = super().make_batch(B, gen)
        return torch.cat([self.relayout(x[:, :self.body1]), x[:, self.body1:]], 1), qs

    def role_masks(self, x):
        S, P, nv = self.S, self.P, self.n_vals
        return {'ctx': x < S, 'key': (x >= S) & (x < S + P), 'val': (x >= S + P) & (x < S + P + nv), 'fill': x == self.fill}


T1 = ft.RandHeaderTask(P=4, nb=2, lmin=1, lmax=3)
T2 = RandHeader2x(P=4, nb=2, lmin=1, lmax=3)
TASKS = {"1X": T1, "2X": T2}


# ── The routed decay, relative to the fixed decay ────────────────────────────
class RelRouted(nn.Module):
    kind = "routed_rel"

    def __init__(self, decay, d=dm.D):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d))
        self.delta = nn.Parameter(torch.zeros(()))
        self.lnd = math.log(decay)

    def f(self, v):
        return torch.exp(self.delta) * F.softplus(v @ self.w) / F.softplus(torch.zeros((), dtype=v.dtype))

    def ell(self, v, gw):
        """log α_t − ln(decay), (B, T, k)."""
        return self.lnd * (self.f(v).unsqueeze(-1) * gw - 1.0)

    def log_alpha(self, v, gw):
        return self.lnd * self.f(v).unsqueeze(-1) * gw


def rel_parallel(kh, v, gr, b, w, ell, decay, skip_solve=False):
    """explore_delta_mem.delta_parallel with the decay matrix decay^(t−s) · exp(Λ_t − Λ_s), Λ = cumsum ell."""
    B, T, _ = kh.shape
    KK = kh @ kh.mT
    lam = ell.cumsum(1).transpose(1, 2)                                 # (B, k, T)
    diff = lam.unsqueeze(-1) - lam.unsqueeze(-2)
    low = torch.ones(T, T, dtype=torch.bool).tril(-1)
    A = decay_mask(T, decay, kh.dtype) * torch.exp(diff.masked_fill(~low, float("-inf")))
    M = KK.unsqueeze(1) * A
    rhs = w.transpose(1, 2).unsqueeze(-1) * v.unsqueeze(1)
    if skip_solve:
        U = rhs
    else:
        L = b.transpose(1, 2).unsqueeze(-1) * M
        U = torch.linalg.solve_triangular(L, rhs, upper=False, unitriangular=True)
    O = M @ U
    return (gr.transpose(1, 2).unsqueeze(-1) * O).sum(1), U


class HorizonMemory(dm.DeltaMemory):
    def forward(self, Q, K, V):
        if getattr(self, "rel", None) is None:
            return super().forward(Q, K, V)
        assert K is Q and Q.shape[1] == 1, "one head, Q = K"
        gr, gw = self._gates_now()
        q, v = Q[:, 0], V[:, 0]
        kh = self.keys(q)
        b, w = self.coefs(v, gw)
        o = rel_parallel(kh, v, gr, b, w, self.rel.ell(v, gw), self.decay, skip_solve=self.erase_off())[0]
        return o.unsqueeze(1)


def to_horizon(m, seed, decay="FIXED", freeze=False):
    m = dm.to_delta(m, beta=1.0, normalize=True, decay=None, seed=seed)
    if decay == "ROUTED":
        m.attn.__class__ = HorizonMemory
        m.attn.rel = RelRouted(m.attn.decay)
        if freeze:
            for q in m.attn.rel.parameters():
                q.requires_grad_(False)
        m.delta_cfg = dict(m.delta_cfg, decay="routed (relative)", frozen=bool(freeze))
    return m


# ── Arms ─────────────────────────────────────────────────────────────────────
def parse(arm_name):
    """FIXED_1X, ROUTED_2X, O_FIXED_1X, ..."""
    parts = arm_name.split("_")
    return dict(perfect=parts[0] == "O", decay=parts[-2], length=parts[-1])


def arm_def(arm_name):
    sp = parse(arm_name)
    task = TASKS[sp["length"]]
    return {x["key"]: x for x in cb_arms(task)}["ceiling"] if sp["perfect"] else tsc.ARM["B"]


def builder(arm_name, freeze=False):
    sp = parse(arm_name)
    task = TASKS[sp["length"]]

    def b(a, seed):
        def make():
            return to_horizon(build(task, a, ARCH, seed), seed, sp["decay"], freeze)
        return make
    return b


@torch.no_grad()
def acc_by_distance(m, task, data):
    """Held-out accuracy by the queried key's distance to its block's CTX: {distance: [n, acc]}."""
    was = m.training
    m.eval()
    try:
        inp, tgt = data[:, :-1], data[:, 1:]
        lg = tbo.logits_of(m, inp)
    finally:
        m.train(was)
    S, qp = task.S, task.qpos[0]
    ok = (lg[:, qp].argmax(-1) == tgt[:, qp])
    qs, qk = inp[:, qp - 1], inp[:, qp]
    body = inp[:, :qp - 1]
    lab = task.stream_labels(inp)[:, :qp - 1]
    idx = torch.arange(body.shape[1]).expand_as(body)
    hit = (body == qk[:, None]) & (lab == qs[:, None])
    assert bool((hit.sum(1) == 1).all())
    kpos = torch.where(hit, idx, torch.full_like(idx, -1)).max(1).values
    cue = torch.where((body < S) & (idx <= kpos[:, None]), idx, torch.full_like(idx, -1)).max(1).values
    dist = kpos - cue
    return {str(int(d)): [int((dist == d).sum()), float(ok[dist == d].float().mean())] for d in dist.unique()}


def stats_fn(arm_name):
    sp = parse(arm_name)
    return ft.rand_header_stats(c2.ROUTE_AT) if sp["perfect"] else tsc.conv_stats


@torch.no_grad()
def decay_by_role(m, task, seed):
    """Mean α per layer at CTX / KEY / VAL / FILL on the probe batch (the routed decay)."""
    pinp, plab, roles = probe_batch(task, seed)
    out = []

    def hook(mod, args, kwargs, o):
        V = kwargs["V"][:, 0]
        gr, gw = mod._gates
        al = torch.exp(mod.rel.log_alpha(V, gw))[..., 0]
        out.append({r: float(al[mask].mean()) for r, mask in roles.items() if bool(mask.any())})
    was = m.training
    m.eval()
    h = m.attn.register_forward_hook(hook, with_kwargs=True)
    try:
        m(pinp)
    finally:
        h.remove()
        m.train(was)
    return out


def run(p):
    arm_name, seed, iters = p["arm"], p["seed"], p["iters"]
    sp = parse(arm_name)
    task = TASKS[sp["length"]]
    keep = {"at": -1}
    by_d = {}

    def check(m, step, data):
        by_d[str(step)] = acc_by_distance(m, task, data)
    rec = run_one(arm_def(arm_name), seed, iters, p.get("eval_every", tbo.EVAL_EVERY), keep=keep, task=task,
                  stats_fn=stats_fn(arm_name), grad_fn=tsc.conv_grad_norms, lr=LR,
                  builder=builder(arm_name, p.get("freeze", False)), check=check if p.get("diag", True) else None)
    if rec.get("ok"):
        m = keep["model"]
        rec.update(arm=arm_name, length=sp["length"], decay=sp["decay"], perfect=sp["perfect"], k=arm_def(arm_name)["n_ch"],
                   delta_cfg=getattr(m, "delta_cfg", None), acc_by_distance=by_d or None, bound=rec["transition"] is not None)
        if p.get("diag", True):
            rec["write_order"] = ft.write_order_acc(m, task, tbo.eval_batch(task))
            if getattr(m.attn, "rel", None) is not None:
                rec["decay_by_role"] = decay_by_role(m, task, seed)
                rec["decay_params"] = dict(w_norm=float(m.attn.rel.w.norm()), delta=float(m.attn.rel.delta),
                                           Delta=float(m.attn.rel.delta) + dm.ROUTED_DELTA0)
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
S70_REC = ("F/rh_lite", "S70_DELTA", 330, "ff908bdbc0b2")
REC_CPU = "Intel(R) Xeon(R) Processor @ 2.10GHz"


def checks(p):
    rows = []
    cpu = ec.cpu_model()
    # (1) one S70 reproduction (the 1x task and S70's run path on this CPU)
    import explore_f_followup_child as fu
    store, rarm, sd, sha = S70_REC
    key = f"{rarm}|{sd}|{sha}|{REC_CPU}"
    ref = ec.load_store(store)["runs"][key]
    r = fu.run(dict(arm=rarm, seed=sd, iters=2400))
    want = [c for c in ref["curve"] if c[0] <= 2400]
    sn = [s for s in r["stats"] if s["step"] != "end" and s["step"] <= 2400]
    so = [s for s in ref["stats"] if s["step"] != "end" and s["step"] <= 2400]
    rows.append((f"S70 reproduction on this CPU ({cpu}; the record's: {REC_CPU}): {key} through 2400 bit for bit with S70's "
                 f"own run path: curve {r['curve']} vs {want}; statistics equal {sn == so}", r["curve"] == want and sn == so))
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    x1, q1 = T1.make_batch(64, g1)
    xs, _ = fu.RHL.make_batch(64, torch.Generator().manual_seed(5))
    rows.append((f"the 1x task is S70's RandHeaderTask-lite (same batches as explore_f_followup_child.RHL; L {T1.L}, vocab "
                 f"{T1.vocab})", torch.equal(x1, xs) and T1.L == 23))
    # (2) the 2x layout
    x2, q2 = T2.make_batch(64, g2)
    S, P = T1.S, T1.P

    def dists(task, x):
        body = x[:, :task.qpos[0] - 1]
        out = []
        for row in body.tolist():
            cue, d = None, []
            for i, t in enumerate(row):
                if t < S:
                    cue = i
                elif t < S + P:
                    d.append((t, row[i + 1], i - cue))
            out.append(d)
        return out
    d1, d2 = dists(T1, x1), dists(T2, x2)
    same_pairs = all([(k, v) for k, v, _ in a] == [(k, v) for k, v, _ in b] for a, b in zip(d1, d2))
    doubled = all(db == 2 * da for a, b in zip(d1, d2) for (_, _, da), (_, _, db) in zip(a, b))
    qsame = torch.equal(x1[:, T1.body_len:], x2[:, T2.body_len:]) and torch.equal(q1, q2)
    fills = int((x2 == T2.fill).sum(1)[0])
    rows.append((f"the 2x layout: L {T2.L} (body {T2.body_len}), vocab {T2.vocab} (FILL {T2.fill}); the same pairs in the same "
                 f"order ({same_pairs}); every key's distance to its cue doubled ({doubled}; 1x {sorted({d for a in d1 for _, _, d in a})}"
                 f" -> 2x {sorted({d for a in d2 for _, _, d in a})}); the same queries ({qsame}); {fills} FILL per sequence",
                 same_pairs and doubled and qsame and x2.shape[1] == T2.L and fills == S * T1.nb + 2 * (S * P - S * T1.nb)))
    lab2 = T2.stream_labels(x2[:, :-1])
    kp, _ = ft.body_positions(T2, x2[:, :-1])
    rows.append((f"2x: the stream label at every key is its block's CTX; role masks separate FILL from values",
                 bool((lab2.gather(1, kp) >= 0).all()) and not bool((T2.role_masks(x2)["val"] & (x2 == T2.fill)).any())))
    # (3) the routed decay: S59's function; the relative decay matrix; frozen it is the fixed decay bit for bit
    gD = torch.Generator().manual_seed(7)
    v = torch.randn(3, 11, dm.D, generator=gD, dtype=torch.float64)
    rr = RelRouted(0.95).double()
    s59 = dm.RoutedDecay().double()
    with torch.no_grad():
        rr.w.copy_(torch.randn(dm.D, generator=gD, dtype=torch.float64) * 0.3)
        rr.delta.fill_(0.4)
        s59.w.copy_(rr.w)
        s59.delta.fill_(dm.ROUTED_DELTA0 + 0.4)
    gw = torch.rand(3, 11, 2, generator=gD, dtype=torch.float64)
    e1 = float((rr.log_alpha(v, gw) - s59(v, gw)).abs().max())
    rows.append((f"RelRouted's log α equals S59's RoutedDecay (Δ = Δ0 + δ) to {e1:.1e} in float64 (random w, δ, v, gates)", e1 < 1e-12))
    kh = F.normalize(torch.randn(3, 11, 24, generator=gD, dtype=torch.float64), dim=-1)
    vv = torch.randn(3, 11, 8, generator=gD, dtype=torch.float64)
    gr = torch.rand(3, 11, 2, generator=gD, dtype=torch.float64)
    ell = rr.ell(v, gw)
    o1 = rel_parallel(kh, vv, gr, gw, gw, ell, 0.95)[0]
    o2 = dm.delta_parallel(kh, vv, gr, gw, gw, rr.log_alpha(v, gw), 0.95)[0]
    e2 = float((o1 - o2).abs().max())
    rows.append((f"rel_parallel equals explore_delta_mem.delta_parallel with log α = ln 0.95 + ell to {e2:.1e} in float64", e2 < 1e-10))
    for L in ("1X", "2X"):
        rf = run(dict(arm=f"FIXED_{L}", seed=2420, iters=2400, diag=False))
        rz = run(dict(arm=f"ROUTED_{L}", seed=2420, iters=2400, diag=False, freeze=True))
        sn = [s for s in rz["stats"] if s["step"] != "end"]
        so = [s for s in rf["stats"] if s["step"] != "end"]
        rows.append((f"{L}: the forget gate with its parameters frozen (ROUTED_{L}, w and δ frozen at 0) equals FIXED_{L} bit for "
                     f"bit through 2400 on seed 2420: curve {rz['curve']} vs {rf['curve']}; statistics {sn == so}",
                     rz["curve"] == rf["curve"] and sn == so))
    # (4) the routed decay trains; the models
    for L in ("1X", "2X"):
        m = builder(f"ROUTED_{L}")(tsc.ARM["B"], 2420)()
        mf = builder(f"FIXED_{L}")(tsc.ARM["B"], 2420)()
        task = TASKS[L]
        pf, pr = dict(mf.named_parameters()), dict(m.named_parameters())
        extra = sorted(set(pr) - set(pf))
        same = all(torch.equal(pr[n], pf[n]) for n in pf)
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        x = task.make_batch(8, torch.Generator().manual_seed(11))[0]
        ql, qt, _ = task.select(tbo.logits_of(m, x[:, :-1]), x[:, 1:], None)
        F.cross_entropy(ql, qt).backward()
        gn = float(m.attn.rel.w.grad.norm()), float(m.attn.rel.delta.grad.abs())
        opt.step()
        rows.append((f"{L}: ROUTED's model = FIXED's (k {m.n_ch}, {type(m.attn).__name__}, beta {m.attn.beta_fixed}, L2 "
                     f"{m.attn.normalize}) + {extra}, every other parameter equal ({same}); the decay trains (|grad w| {gn[0]:.2e}, "
                     f"|grad δ| {gn[1]:.2e})", extra == ["attn.rel.delta", "attn.rel.w"] and same and gn[0] > 0 and gn[1] > 0
                     and m.n_ch == 1 and m.attn.beta_fixed == 1.0 and m.attn.normalize))
        mo = builder(f"O_FIXED_{L}")(arm_def(f"O_FIXED_{L}"), 2420)()
        mo.eval()
        with torch.no_grad():
            g = mo(task.make_batch(4, torch.Generator().manual_seed(3))[0][:, :-1])[2]
        rows.append((f"{L}: the perfect gate (k {mo.n_ch}) on the delta memory is one-hot", bool(((g == 0) | (g == 1)).all())
                     and mo.n_ch == 2))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    stamps = []
    h = register_optimizer_step_post_hook(lambda o, a, k: stamps.append(time.perf_counter()))
    try:
        run(dict(arm=p["arm"], seed=0, iters=p["steps"], eval_every=10 ** 9, diag=False))
    finally:
        h.remove()
    dd = sorted(y - x for x, y in zip(stamps, stamps[1:]))
    task = TASKS[parse(p["arm"])["length"]]
    m = builder(p["arm"])(arm_def(p["arm"]), 0)()
    t0 = time.time()
    tbo.evaluate(m, task, tbo.eval_batch(task))
    acc_by_distance(m, task, tbo.eval_batch(task))
    return [dd[len(dd) // 2], 2 * (time.time() - t0)]
