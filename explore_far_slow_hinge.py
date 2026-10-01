#!/usr/bin/env python
"""
explore_far_slow_hinge.py — EXPLORATORY, not a result. Screen S17 (batch 6): S13's SLOW_HINGE on
the header layout, with no nudge (label-free).

BACKGROUND (post hoc)
- Header layout (explore_far_cue.HeaderTask: each stream's block [CTX_s, K, V, K, V, ...], the
  stream token once per block, blocks in random stream order). Label-free gates found the routing
  on no seed: far_A 0/10 (batch 2), far_A_slow (SLOW_MEM's schedule) 0/10 with 8 POSITION
  failures (batch 3). The 5% labelled nudge did: far_nudge 5/10 (batch 4), far_nudge_slow 10/10
  (batch 5).
- On the grouped layout S13's hinge (SLOW_MEM + 1.0 * [relu(eta2_index - 0.2) + relu(eta2_half -
  0.2)] of the read gate at key positions) turned SLOW_MEM's POSITION failures into discoveries
  (34/40 vs 24/40).

THE ARM (FAR_SLOW_HINGE). Arm A on the header layout (build(HEADER, ARM_A) parameters), trained
with explore_slow_mem's param groups and lr switch (gate W_in/W_h/W_g 1e-3 = test_channel_binding.
SUB_LR throughout; every other parameter 1e-4 for updates 1-2400, 1e-3 after; MAX_ITERS 24000;
far_A_slow's run exactly), plus on each training batch
    LAMBDA * [relu(eta2_index - TAU) + relu(eta2_half - TAU)],   LAMBDA = 1.0, TAU = 0.2,
added through explore_aux_gate's gradient-injection node, as in S13. The hinge's key positions are
the header layout's KEY positions (explore_far_cue.positions: pair i of block b at b*(1+2P)+1+2i);
eta2_index groups by the pair's index in the sequence (0..S*P-1, in order of appearance), eta2_half
by its first or second half (j < P or j >= P, i.e. the first or second block). The formula is
explore_slow_pos.pos_penalty's (the read gate's channel-0 probability; between-group over total sum
of squares, total + 1e-12), at these positions; it equals explore_far_cue.header_routing_stats'
eta_key_by_index / eta_key_by_half. No stream label is used. Statistics: explore_far_cue.
header_stats plus the hinge's firing counts. Seeds 160-169, paired with batch 3's far_A_slow and
batch 2's far_A (same initial parameters and batches).

A NOTE ON THE HEADER LAYOUT (arithmetic, not a result; printed with the CHECKs). Here the half IS
the block, so a perfectly routed gate (channel-0 probability 1 on stream 0's block, 0 on stream
1's) scores eta2_index = eta2_half = 4 (m - 1/2)^2 on a batch, m = the share of sequences whose
first block is stream 0's. On a probe batch (256 sequences) that is 0.004 on average; on a
training batch (32 sequences) it exceeds TAU = 0.2 when |32 m - 16| >= 8: probability
2 P(Bin(32, 1/2) >= 24) = 0.0070 per batch. So the hinge also fires, now and then, on a routed
gate. On the grouped layout the perfect gate's eta2_half is exactly 0 (each half holds both
streams of its keys).

RULE (S8's, fixed before any run), outcome DISCOVERED: promising if >= 4/10; not if <= 1/10;
otherwise inconclusive. Also printed: failure classes, ROUTED* at 1200 and at the end, stalls
(routed at the end or stream-partial, not bound), hinge firing counts, and McNemar against
far_A_slow and far_A.

CHECKS: with TAU = 1.0 the run equals far_A_slow bit for bit (curves and weights, 3600 steps; also
batch 3's record); the penalty equals header_routing_stats' eta_key_by_index / _by_half on a probe
batch (at init and on a gate far from uniform), and on the grouped layout's positions it equals
explore_slow_pos.pos_penalty bit for bit; on 20 probe batches of the header layout the perfect gate
scores below 0.1 on both terms and a gate set by block (first / second) scores above 0.5 on
eta2_half; the hinge's gradient reaches only the gate and the embedding and is exactly zero below
TAU.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F

import explore_common as ec
import explore_common2 as c2
import explore_slow_mem as s5
import explore_slow_pos as s12
import explore_far_cue as s6
import explore_far_gates as s8
import test_short_conv as tsc
from explore_aux_gate import _Inject
from test_multilayer_binding import MultiBDH, probe_batch
from test_channel_binding import ARCH
from test_instrument_v2 import perfect_gate_general
from test_router_confirm import mcnemar_exact
from test_router_layout import fail_class
from test_binding_onset import EVAL_EVERY, BATCH

NAME = "far_slow_hinge"
IDEA = "S13's SLOW_HINGE on the header layout, no nudge (label-free)"
SOURCE = ("batch 5's S13 (the hinge turned SLOW_MEM's POSITION failures into discoveries); batch 3's "
          "far_A_slow (0/10, 8 POSITION)")
CHANGE = ("far_A_slow + 1.0 * [relu(eta2_index - 0.2) + relu(eta2_half - 0.2)] of the read gate at the "
          "header layout's KEY positions, on each training batch")
PAIRING = ("seeds 160-169, paired with batch 3's far_A_slow and batch 2's far_A (same initial "
           "parameters and batches)")
SEEDS = tuple(range(160, 170))
LAMBDA, TAU = 1.0, 0.2
EPS = s12.EPS
HEADER = s6.HEADER
KPOS = s6.positions(HEADER)[0]
B3_STORE, B2_STORE = "far_gates", "far_cue"
PROMISING_N, NOT_N = 4, 1
PROBE_BATCHES = 20

ARMS = {
    "FAR_SLOW_HINGE": dict(s8.ARMS["far_A_slow"], key="FAR_SLOW_HINGE", seeds=SEEDS, prio=1,
                           label="far_A_slow + hinge on position eta^2 (TAU 0.2), header layout",
                           sched=s5.ARMS["SLOW_MEM"]["sched"] + f"; + {LAMBDA} * [relu(eta2_index - "
                                 f"{TAU}) + relu(eta2_half - {TAU})] at the header KEY positions on "
                                 "each training batch"),
}


def pos_penalty_at(gr, kpos, P):
    """explore_slow_pos.pos_penalty's formula at the key positions kpos (pairs in order)."""
    n = len(kpos)
    j = torch.arange(n)
    p = gr[:, kpos, 0]                                                  # (B, n)
    B = p.shape[0]
    mu = p.mean()
    tot = ((p - mu) ** 2).sum() + EPS
    between_idx = B * ((p.mean(0) - mu) ** 2).sum()
    h = j >= P
    between_half = B * (int((~h).sum()) * (p[:, ~h].mean() - mu) ** 2
                        + int(h.sum()) * (p[:, h].mean() - mu) ** 2)
    return between_idx / tot, between_half / tot


class HdrHingeBDH(MultiBDH):
    def __init__(self, *args, pen_lambda=None, tau=TAU, kpos=KPOS, pen_P=4, **kw):
        super().__init__(*args, **kw)
        self.pen_lambda, self.tau, self.kpos, self.pen_P = pen_lambda, tau, kpos, pen_P
        self.pen_last, self.n_batches, self.n_active = None, 0, 0

    def forward(self, tokens, tau=None, track_sat=False):
        logits, sat, gr, gw = super().forward(tokens, tau, track_sat)
        if self.training and self.pen_lambda is not None and torch.is_grad_enabled():
            e_i, e_h = pos_penalty_at(gr, self.kpos, self.pen_P)
            h_i, h_h = F.relu(e_i - self.tau), F.relu(e_h - self.tau)
            self.n_batches += 1
            self.n_active += int(bool(h_i > 0) or bool(h_h > 0))
            self.pen_last = (float(e_i.detach()), float(e_h.detach()))
            logits = _Inject.apply(logits, self.pen_lambda * (h_i + h_h))
        return logits, sat, gr, gw


def make_model(a, seed, tau=TAU, lam=LAMBDA):
    torch.manual_seed(seed)
    return HdrHingeBDH(HEADER.vocab, ARCH["n_layer"], a["gate"], a["n_ch"], ARCH["positional"],
                       gate_to_readout=a["g2r"], h_gate=a["h_gate"], mult=a["mult"],
                       ctx_tokens=HEADER.ctx_tokens, **a.get("model_kw", {}), pen_lambda=lam,
                       tau=tau, kpos=KPOS, pen_P=HEADER.P)


def stats_fn(model, task, probe, step):
    out = s6.header_stats(model, task, probe, step)
    out.update(pos_pen_last_train=model.pen_last, hinge_counts=[model.n_active, model.n_batches])
    return out


def run_path(a, seed, iters, tau=TAU, keep=None):
    holder = {}
    return ec.run_one(a, seed, iters, check=s5.make_check(holder), keep=keep, task=HEADER,
                      stats_fn=stats_fn, grad_fn=tsc.conv_grad_norms, lr=a["lr"],
                      builder=lambda aa, s: (lambda: make_model(aa, s, tau)),
                      run_kw=dict(param_groups=s5.make_groups(holder)))


def run_job(arm, seed):
    a = ARMS[arm]
    return run_path(a, seed, a["iters"])


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], HEADER, lambda: make_model(a, 0), a["lr"], stats_fn)]


def frac(r, step):
    d = c2.stat_at(r, step)
    if not d or not d.get("hinge_counts") or not d["hinge_counts"][1]:
        return None
    a, n = d["hinge_counts"]
    return a / n


def perfect_fire_prob(B=BATCH):
    """P(4 (m - 1/2)^2 > TAU) for m = Bin(B, 1/2) / B: the perfect gate's hinge on a training batch."""
    return sum(math.comb(B, k) for k in range(B + 1) if 4 * (k / B - 0.5) ** 2 > TAU) / 2 ** B


def check():
    ok = True
    seed, it = 160, 3 * EVAL_EVERY
    a = ARMS["FAR_SLOW_HINGE"]
    # (1) TAU = 1.0 is far_A_slow bit for bit
    k1, k8 = {"at": it}, {"at": it}
    r1 = run_path(a, seed, it, tau=1.0, keep=k1)
    r8 = s8.run_path("far_A_slow", seed, it, keep=k8)
    rec8 = [c for c in ec.load_store(B3_STORE)["runs"][f"far_A_slow|{seed}"]["curve"] if c[0] <= it]
    same1 = (r1["curve"] == r8["curve"] == rec8
             and all(torch.equal(k1["snap"][n], k8["snap"][n]) for n in k8["snap"]))
    f1 = frac(r1, "end")
    # (2) the penalty is header_routing_stats' eta^2; on the grouped positions it is S12's
    probe = probe_batch(HEADER, seed)
    diffs = []
    for scale in (1.0, 30.0):
        mm = make_model(a, seed)
        with torch.no_grad():
            mm.W_in.mul_(scale)
        mm.eval()
        with torch.no_grad():
            g = mm(probe[0], None)[2]
            e_i, e_h = pos_penalty_at(g, KPOS, HEADER.P)
        hr = s6.header_routing_stats(mm, HEADER, probe)
        diffs += [abs(e_i.item() - hr["eta_key_by_index"]), abs(e_h.item() - hr["eta_key_by_half"])]
    match = max(diffs) < 1e-5
    gg = torch.rand(64, ec.TASK.T, 2, generator=torch.Generator().manual_seed(3))
    n = ec.TASK.S * ec.TASK.P
    grouped_same = all(torch.equal(u, v) for u, v in zip(
        pos_penalty_at(gg, 3 * torch.arange(n) + 1, ec.TASK.P), s12.pos_penalty(gg, ec.TASK.S, ec.TASK.P)))
    # (3) 20 probe batches: the perfect gate below 0.1 on both terms; a block gate above 0.5 on half
    pf, bl = [], []
    for s in range(PROBE_BATCHES):
        x = probe_batch(HEADER, s)[0]
        pg = perfect_gate_general(x, HEADER.ctx_tokens)
        pf.append([v.item() for v in pos_penalty_at(pg, KPOS, HEADER.P)])
        blk = torch.zeros_like(pg)
        blk[:, :HEADER.blk, 0] = 1.0
        blk[:, HEADER.blk:, 1] = 1.0
        bl.append([v.item() for v in pos_penalty_at(blk, KPOS, HEADER.P)])
    pf_max = max(max(v) for v in pf)
    bl_min = min(v[1] for v in bl)
    # (4) the hinge's gradient: only the gate and the embedding; exactly zero below TAU
    xb = probe_batch(HEADER, seed)[0][:BATCH]
    m = make_model(a, seed, tau=0.0)
    m.train()
    gr = MultiBDH.forward(m, xb)[2]
    f_i, f_h = pos_penalty_at(gr, KPOS, HEADER.P)
    (F.relu(f_i) + F.relu(f_h)).backward()
    got = {nm for nm, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    reach = got == {"W_in", "W_h", "W_g", "embed.weight"}
    m2 = make_model(a, seed)
    m2.train()
    gr2 = MultiBDH.forward(m2, xb)[2]
    q_i, q_h = pos_penalty_at(gr2, KPOS, HEADER.P)
    below = float(q_i.detach()) < TAU and float(q_h.detach()) < TAU
    (F.relu(q_i - TAU) + F.relu(q_h - TAU)).backward()
    zero = all(p.grad is None or bool((p.grad == 0).all()) for p in m2.parameters())
    # (note) the perfect gate on training batches
    gen = torch.Generator().manual_seed(17)
    fires = 0
    for _ in range(2000):
        x = HEADER.make_batch(BATCH, gen)[0][:, :-1]
        e = pos_penalty_at(perfect_gate_general(x, HEADER.ctx_tokens), KPOS, HEADER.P)
        fires += int(max(v.item() for v in e) > TAU)
    for nm, v in ((f"TAU = 1.0 equals far_A_slow bit for bit through {it} steps (curves = a fresh "
                   f"far_A_slow run = batch 3's record; weights equal; hinge fraction {f1})",
                   same1 and f1 == 0.0),
                  (f"the penalty equals header_routing_stats' eta_key_by_index / _by_half on a probe "
                   f"batch, at init and on a gate far from uniform (max diff {max(diffs):.1e} < 1e-5); at "
                   f"the grouped layout's positions it equals explore_slow_pos.pos_penalty bit for bit",
                   match and grouped_same),
                  (f"key positions {KPOS.tolist()} (pairs in order of appearance; half = block)",
                   KPOS.tolist() == [1, 3, 5, 7, 10, 12, 14, 16]),
                  (f"on {PROBE_BATCHES} probe batches of the header layout the perfect gate scores below "
                   f"0.1 on both terms (max {pf_max:.4f}) and a gate set by block scores above 0.5 on "
                   f"eta2_half (min {bl_min:.4f})", pf_max < 0.1 and bl_min > 0.5),
                  (f"the hinge's gradient (TAU = 0) reaches only {sorted(got)}", reach),
                  (f"with both terms below TAU = {TAU} (eta2 index {float(q_i.detach()):.3f}, half "
                   f"{float(q_h.detach()):.3f}) the hinge's gradient is exactly zero", below and zero)):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    print(f"  NOTE {NAME} (arithmetic, not a check): on a training batch of {BATCH} the perfect gate's "
          f"eta2_index = eta2_half = 4 (m - 1/2)^2 (m = share of sequences whose first block is stream "
          f"0's); it exceeds TAU = {TAU} with probability {perfect_fire_prob():.4f} per batch (exact "
          f"binomial; {fires}/2000 = {fires / 2000:.4f} on sampled batches). The hinge also fires, "
          f"now and then, on a routed gate.", flush=True)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def stall(r):
    return bool(r and r.get("ok") and not ec.bound(r)
                and (c2.routed_at(r, "end") or fail_class(r) == "STREAM-PARTIAL"))


def rule(n):
    return "promising" if n >= PROMISING_N else ("not" if n <= NOT_N else "inconclusive")


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    b3 = ec.load_store(B3_STORE)["runs"]
    b2 = ec.load_store(B2_STORE)["runs"]
    print(f"  per seed (ROUTED* at 1200/2400/3600/end; b3 = batch 3's far_A_slow, b2 = batch 2's far_A; "
          f"hinge = fraction of training batches with the hinge on, updates 1-1200 / 1-2400 / whole run, "
          f"and the count):")
    print(f"    {'seed':>4} {'acc':>6} {'trans':>6} {'VALcos':>7} {'margin':>6} {'eta key s/k/h/i':>20} "
          f"{'ROUTED*':>8}  {'outcome':<15} {'b3 far_A_slow':<15} {'b2 far_A':<15} {'hinge':>23}")
    runs = {}
    fm = lambda v: "  -- " if v is None else f"{v:5.3f}"
    for s in SEEDS:
        r = store["runs"].get(f"FAR_SLOW_HINGE|{s}")
        tags = f"{ec.tag(b3.get(f'far_A_slow|{s}')):<15} {ec.tag(b2.get(f'far_A|{s}')):<15}"
        if r is None or not r.get("ok"):
            print(f"    {s:>4} {ec.tag(r):<70} {tags}")
            continue
        runs[s] = r
        m = r["end"].get("margin")
        hc = r["end"].get("hinge_counts", [0, 0])
        print(f"    {s:>4} {r['acc']:6.3f} {str(r['transition']):>6} {r['val_cos']:7.3f} "
              f"{'--' if m is None else f'{m:6.2f}':>6} {ec.eta_str(r):>20} {c2.rflags(r):>8}  "
              f"{ec.tag(r):<15} {tags} {fm(frac(r, EVAL_EVERY))}/{fm(frac(r, 2 * EVAL_EVERY))}/"
              f"{fm(frac(r, 'end'))} {hc[0]:>5}/{hc[1]}")

    def pair(other, key, fn, what, label):
        b = sum(1 for s, r in runs.items() if fn(r) and not fn(other.get(f"{key}|{s}")))
        c = sum(1 for s, r in runs.items() if fn(other.get(f"{key}|{s}")) and not fn(r))
        nn = sum(bool(fn(r)) for r in runs.values())
        no = sum(bool(fn(other.get(f"{key}|{s}"))) for s in runs)
        print(f"    {what:<13} FAR_SLOW_HINGE {nn:>2}/{len(runs)}   {label} {no:>2}/{len(runs)}   "
              f"FAR_SLOW_HINGE only {b}, {label} only {c}, McNemar two-sided p = {mcnemar_exact(b, c):.3g}")
        return nn

    out = {}
    for other, key, label in ((b3, "far_A_slow", "far_A_slow"), (b2, "far_A", "far_A")):
        print(f"  paired with {label}:")
        n = pair(other, key, ec.discovered, "DISCOVERED", label)
        pair(other, key, ec.bound, "BOUND", label)
        pair(other, key, lambda r: c2.routed_at(r, EVAL_EVERY), f"ROUTED*@{EVAL_EVERY}", label)
        pair(other, key, lambda r: c2.routed_at(r, "end"), "ROUTED*@end", label)
        pair(other, key, stall, "STALLS", label)
        out[key] = n

    def fails(get):
        f = {}
        for s in runs:
            r = get(s)
            if r and r.get("ok") and not ec.bound(r):
                f[fail_class(r)] = f.get(fail_class(r), 0) + 1
        return f

    print(f"    stalls (routed at the end or stream-partial, not bound): FAR_SLOW_HINGE "
          f"{[s for s, r in runs.items() if stall(r)]}   far_A_slow "
          f"{[s for s in runs if stall(b3.get(f'far_A_slow|{s}'))]}   far_A "
          f"{[s for s in runs if stall(b2.get(f'far_A|{s}'))]}")
    print(f"    failures FAR_SLOW_HINGE {fails(lambda s: runs[s])}   far_A_slow "
          f"{fails(lambda s: b3.get(f'far_A_slow|{s}'))}   far_A {fails(lambda s: b2.get(f'far_A|{s}'))}")
    tot = [runs[s]["end"].get("hinge_counts", [0, 0]) for s in runs]
    print(f"    hinge firings: {sum(1 for a, _ in tot if a > 0)}/{len(runs)} runs fired at least once; "
          f"total {sum(a for a, _ in tot)} of {sum(n for _, n in tot)} training batches")
    nd = out["far_A_slow"]
    return dict(n=len(runs), discovered=nd, rule=rule(nd))
