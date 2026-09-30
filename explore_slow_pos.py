#!/usr/bin/env python
"""
explore_slow_pos.py — EXPLORATORY, not a result. Screen S12 (batch 4): S5's SLOW_MEM plus a
penalty on the gate's position information.

BACKGROUND. SLOW_MEM (a slow memory for the first 2400 updates) found the stream routing on 24/40
seeds against arm A's 12/40 (batches 2-3, pooled; S7 promising on its own 20 seeds). Its failures
are mostly POSITION splits (11 of 16 pooled): the gate routes by where it is in the sequence.

THE ARM (SLOW_POS). explore_slow_mem's SLOW_MEM (its param groups and lr switch, unchanged: the
gate W_in/W_h/W_g at lr 1e-3 = test_channel_binding.SUB_LR throughout; everything else 1e-4 for
updates 1-2400, 1e-3 after; MAX_ITERS 24000), plus, on each training batch, the penalty
    lambda * (eta^2 by triple index + eta^2 by sequence half),   lambda = 1.0,
of the read gate's channel-0 probability at the body's key positions (3j+1), computed
differentiably in the training forward (between-group over total sum of squares, total + 1e-12)
and added to the loss through a gradient-injection node on the logits (explore_aux_gate._Inject),
so onset_run is unchanged. The penalty's gradient reaches the gate (W_in, W_h, W_g) and the
embedding it reads. Evaluation forwards are SLOW_MEM's. No stream label is used.
Seeds 160-179, paired with X's recorded arm A AND with batch 2's SLOW_MEM records on the same seeds
(explore_out/slow_mem_results.json): same initial parameters and batches.

RULE: batch 1's vs X's arm A (DISCOVERED; promising if b - c >= 4 and McNemar p < 0.10; not if
b - c <= 0; otherwise inconclusive). Also reported: vs SLOW_MEM (McNemar), and failure classes.

CHECKS
- with lambda = 0 the run equals SLOW_MEM bit for bit (curve and weights, 3600 steps, past the lr
  switch; and batch 2's recorded SLOW_MEM curve on that seed);
- the penalty's gradient reaches only the gate parameters and the embedding;
- the eta^2 penalty matches the diagnostic statistic (routing_stats' eta_key_by_index and
  eta_key_by_half) on a probe batch, at init and on a gate far from uniform.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch

import explore_common as ec
import explore_common2 as c2
import explore_slow_mem as s5
import test_short_conv as tsc
from explore_aux_gate import _Inject
from test_multilayer_binding import MultiBDH, build, probe_batch
from test_channel_binding import ARCH
from test_router_layout import routing_stats, fail_class
from test_router_confirm import mcnemar_exact

NAME = "slow_pos"
IDEA = "SLOW_MEM + a penalty on the read gate's position information (eta^2 by index and half)"
SOURCE = ("batch 2-3's SLOW_MEM (failures mostly POSITION splits); invariance penalties on a "
          "representation's nuisance variable (e.g. HSIC / decorrelation penalties)")
CHANGE = ("SLOW_MEM + 1.0 * (eta^2 by triple index + eta^2 by half) of the read gate at key "
          "positions, on each training batch")
PAIRING = ("seeds 160-179, paired with X's recorded arm A and with batch 2's SLOW_MEM on the same "
           "seeds (same initial parameters and batches)")
SEEDS = tuple(range(160, 180))
LAMBDA = 1.0
EPS = 1e-12
S5_STORE = "slow_mem"

ARMS = {
    "SLOW_POS": dict(s5.ARMS["SLOW_MEM"], key="SLOW_POS", seeds=SEEDS, prio=1,
                     label="SLOW_MEM + 1.0 * position penalty on the read gate",
                     sched=s5.ARMS["SLOW_MEM"]["sched"] + f"; + {LAMBDA} * eta^2(index + half) "
                           "penalty on each training batch"),
}


def pos_penalty(gr, S, P):
    """eta^2 by triple index and by half of gr[..., 0] at key positions 3j+1, differentiable."""
    n = S * P
    j = torch.arange(n)
    p = gr[:, 3 * j + 1, 0]                                             # (B, n)
    B = p.shape[0]
    mu = p.mean()
    tot = ((p - mu) ** 2).sum() + EPS
    between_idx = B * ((p.mean(0) - mu) ** 2).sum()
    h = j >= P
    between_half = B * (int((~h).sum()) * (p[:, ~h].mean() - mu) ** 2
                        + int(h.sum()) * (p[:, h].mean() - mu) ** 2)
    return between_idx / tot, between_half / tot


class SlowPosBDH(MultiBDH):
    def __init__(self, *args, pen_lambda=None, pen_SP=(2, 4), **kw):
        super().__init__(*args, **kw)
        self.pen_lambda, self.pen_SP, self.pen_last = pen_lambda, pen_SP, None

    def forward(self, tokens, tau=None, track_sat=False):
        logits, sat, gr, gw = super().forward(tokens, tau, track_sat)
        if self.training and self.pen_lambda is not None and torch.is_grad_enabled():
            e_i, e_h = pos_penalty(gr, *self.pen_SP)
            self.pen_last = (float(e_i.detach()), float(e_h.detach()))
            logits = _Inject.apply(logits, self.pen_lambda * (e_i + e_h))
        return logits, sat, gr, gw


def make_model(a, seed, lam=LAMBDA):
    task = ec.TASK
    torch.manual_seed(seed)
    return SlowPosBDH(task.vocab, ARCH["n_layer"], a["gate"], a["n_ch"], ARCH["positional"],
                      gate_to_readout=a["g2r"], h_gate=a["h_gate"], mult=a["mult"],
                      ctx_tokens=task.ctx_tokens, **a.get("model_kw", {}), pen_lambda=lam,
                      pen_SP=(task.S, task.P))


def stats_fn(model, task, probe, step):
    out = c2.stats_b2(model, task, probe, step)
    out["pos_pen_last_train"] = model.pen_last
    return out


def run_path(a, seed, iters, lam=LAMBDA, keep=None):
    holder = {}
    return ec.run_one(a, seed, iters, check=s5.make_check(holder), keep=keep, task=ec.TASK,
                      stats_fn=stats_fn, grad_fn=tsc.conv_grad_norms, lr=a["lr"],
                      builder=lambda aa, s: (lambda: make_model(aa, s, lam)),
                      run_kw=dict(param_groups=s5.make_groups(holder)))


def run_job(arm, seed):
    a = ARMS[arm]
    return run_path(a, seed, a["iters"])


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], ec.TASK, lambda: make_model(a, 0), a["lr"], stats_fn)]


def check():
    ok = True
    seed, it = 160, 3 * 1200
    # (1) lambda = 0 is SLOW_MEM bit for bit
    k0, k5 = {"at": it}, {"at": it}
    r0 = run_path(ARMS["SLOW_POS"], seed, it, lam=0.0, keep=k0)
    r5 = s5.run_path(s5.ARMS["SLOW_MEM"], seed, it, keep=k5)
    rec = ec.load_store(S5_STORE)["runs"][f"SLOW_MEM|{seed}"]
    want = [c for c in rec["curve"] if c[0] <= it]
    same0 = (r0["curve"] == r5["curve"] == want
             and all(torch.equal(k0["snap"][n], k5["snap"][n]) for n in k5["snap"]))
    # (2) the penalty's gradient reaches only the gate and the embedding
    m = make_model(ARMS["SLOW_POS"], seed)
    m.train()
    x = probe_batch(ec.TASK, seed)[0][:64]
    gr = MultiBDH.forward(m, x)[2]
    e_i, e_h = pos_penalty(gr, ec.TASK.S, ec.TASK.P)
    m.zero_grad()
    (e_i + e_h).backward()
    got = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    reach = got == {"W_in", "W_h", "W_g", "embed.weight"}
    # ... and it is injected into the training loss's backward (lambda 1 vs 0 differ on W_g only
    # through the penalty)
    m.zero_grad()
    m(x)[0][:, -1].sum().backward()
    g1 = m.W_g.grad.clone()
    m0 = make_model(ARMS["SLOW_POS"], seed, lam=0.0)
    m0.train(); m0.zero_grad()
    m0(x)[0][:, -1].sum().backward()
    injected = not torch.equal(g1, m0.W_g.grad) and torch.equal(m.W_g.detach(), m0.W_g.detach())
    # (3) the penalty matches the diagnostic statistic on a probe batch
    probe = probe_batch(ec.TASK, seed)
    diffs = []
    for scale in (1.0, 30.0):
        mm = make_model(ARMS["SLOW_POS"], seed)
        with torch.no_grad():
            mm.W_in.mul_(scale)
        mm.eval()
        with torch.no_grad():
            g = mm(probe[0], None)[2]
            a_i, a_h = pos_penalty(g, ec.TASK.S, ec.TASK.P)
        rs = routing_stats(mm, ec.TASK, probe)
        diffs += [abs(a_i.item() - rs["eta_key_by_index"]), abs(a_h.item() - rs["eta_key_by_half"])]
    match = max(diffs) < 1e-5
    for nm, v in ((f"lambda = 0 equals SLOW_MEM bit for bit through {it} steps (curves equal to a fresh "
                   f"SLOW_MEM run and to batch 2's record; weights equal)", same0),
                  (f"the penalty's gradient reaches only {sorted(got)}", reach),
                  ("the penalty enters the training loss's backward (W_g's gradient with lambda 1 "
                   "differs from lambda 0 on the same weights)", injected),
                  (f"the penalty matches routing_stats' eta_key_by_index / _by_half on the probe, at init "
                   f"and on a gate far from uniform (max diff {max(diffs):.1e} < 1e-5)", match)):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    c2.per_seed_table2(me, store)
    s = ARMS["SLOW_POS"]["seeds"]
    s5r = ec.load_store(S5_STORE)["runs"]
    runs = {x: r for x in s if (r := store["runs"].get(f"SLOW_POS|{x}")) and r.get("ok")}
    print("  per seed vs batch 2's SLOW_MEM (outcome; position penalty eta^2 index/half on the last "
          "training batch before 1200 and before the end):")
    for x, r in runs.items():
        pen = [st.get("pos_pen_last_train") for st in r["stats"] if st["step"] in (1200,)]
        pe = r["end"].get("pos_pen_last_train") if "pos_pen_last_train" in r["end"] else None
        fmt = lambda v: "--" if not v else f"{v[0]:.2f}/{v[1]:.2f}"
        print(f"    {x:>4} SLOW_POS {ec.tag(r):<15} SLOW_MEM {ec.tag(s5r.get(f'SLOW_MEM|{x}')):<15} "
              f"penalty@1200 {fmt(pen[0] if pen else None)}  end {fmt(pe)}")
    print("  paired with X's arm A (same seeds, initial parameters and batches):")
    d = ec.paired_vs_recorded(store, "SLOW_POS", s, ec.discovered, label="DISCOVERED")
    ec.paired_vs_recorded(store, "SLOW_POS", s, ec.bound, label="BOUND")
    for step in (1200, "end"):
        c2.paired_routed(store, "SLOW_POS", s, step, {})
    done = list(runs)
    nd5 = sum(ec.discovered(s5r.get(f"SLOW_MEM|{x}")) for x in done)
    b = sum(1 for x in done if ec.discovered(runs[x]) and not ec.discovered(s5r.get(f"SLOW_MEM|{x}")))
    c = sum(1 for x in done if ec.discovered(s5r.get(f"SLOW_MEM|{x}")) and not ec.discovered(runs[x]))
    rb = sum(1 for x in done if c2.routed_at(runs[x], 1200) and not c2.routed_at(s5r.get(f"SLOW_MEM|{x}"), 1200))
    rc = sum(1 for x in done if c2.routed_at(s5r.get(f"SLOW_MEM|{x}"), 1200) and not c2.routed_at(runs[x], 1200))
    print(f"  paired with batch 2's SLOW_MEM: DISCOVERED SLOW_POS {d['new']}/{len(done)} vs SLOW_MEM "
          f"{nd5}/{len(done)}, SLOW_POS only {b}, SLOW_MEM only {c}, McNemar p = {mcnemar_exact(b, c):.3g};"
          f"  ROUTED*@1200 SLOW_POS only {rb}, SLOW_MEM only {rc}, p = {mcnemar_exact(rb, rc):.3g}")
    f5 = {}
    for x in done:
        r5 = s5r.get(f"SLOW_MEM|{x}")
        if r5 and not ec.bound(r5):
            f5[fail_class(r5)] = f5.get(fail_class(r5), 0) + 1
    print(f"  failure classes: SLOW_POS {ec.fail_counts(store, 'SLOW_POS', done)}   SLOW_MEM {f5}   "
          f"X arm A {ec.recorded_fail_counts('A', done)}")
    d["vs_slow_mem"] = dict(b=b, c=c, p=mcnemar_exact(b, c), slow_mem=nd5)
    return d
