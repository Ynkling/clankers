#!/usr/bin/env python
"""
explore_far_express.py — EXPLORATORY, not a result. Screen S10 (batch 4): can any gate REPRESENT
the header-layout routing, and does the task loss finish it from a head start?

BACKGROUND (batches 2-3, header layout = explore_far_cue.HeaderTask: each stream's block is
[CTX_s, K, V, K, V, ...], the stream token once per block). The perfect gate binds 3/3 at step
1200. Learned gates found the routing on no seed: far_A 0/10, far_L3 0/10 (2 bound by key split),
far_EMA 0/10, far_SEL 0/10 (all key splits), far_A_slow 0/10. The EMA and SEL gates also failed on
the grouped layout (S9: 0/10 vs arm A's 4/10), so it is open whether the gates cannot express the
distant-cue routing or cannot discover it.

(a) EXPRESSIBILITY, no training runs. Each gate is fitted to the perfect header-layout routing
with CHECK 45's recipe (test_router_reliability.gate_fit_k, test_multilayer_binding.gate_fit):
64 sequences from a generator seeded 11, the model built with seed 3, 400 Adam steps at lr 0.01
on the gate's parameters and the embedding, MSE between the read gate and the perfect gate;
mse and argmax match are measured on the same 64 sequences. Gates:
  A        arm A's recurrent gate                (W_in, W_h, W_g, embed)
  EMA      explore_gates' multi-scale EMA gate   (W_in, W_g, theta, embed)
  SEL      explore_gates' selective gate          (W_in, W_g, W_z, b_z, embed)
  LOCAL3   explore_local_gate's width-3 gate      (W_in, W_g, gate_conv, embed) — a control,
           expected to fail by construction (the stream token is out of its window for pairs 2-4)
A gate FITS if argmax match >= 0.99 (test_multilayer_binding.FIT_PASS). Diagnostic, not part of
the readings: the same four fits on the grouped layout.

(b) far_nudge: arm A on the header layout with test_router_curriculum's A_nudge (test_router_
discovery.nudge: 300 Adam steps at lr 0.01 on W_in, W_h, W_g toward 0.475 + 0.05 * the perfect
gate, batches of 64 from a generator seeded seed + 55_555, inside make_model; then normal
training). lr test_channel_binding.SUB_LR = 1e-3, MAX_ITERS 24000, explore_far_cue.header_stats.
Seeds 160-169, paired with batch 2's far_A (same initial parameters before the nudge, same
batches).

READINGS (fixed before any run)
  a gate fits and far_nudge DISCOVERED >= 7/10  -> "the distant cue is a discovery problem"
  a gate fits and far_nudge DISCOVERED <= 3/10  -> "the loss cannot finish the distant-cue routing
                                                    even from a head start"
  no gate fits                                  -> "the gates cannot express the distant-cue routing"
  otherwise (a gate fits, far_nudge 4-6/10)     -> neither reading applies
Printed alongside: whether the nudged gate itself (arm A's) fits, and each far_nudge run's lean
after the pre-fit (sep at step 0).

CHECKS: the header routing target equals the perfect gate's (perfect_gate_general = the one-hot
of the block's stream = far_ceil's gate); fit('A') on the grouped layout reproduces CHECK 45's
gate_fit_k exactly (mse and argmax) at k=2; explore_far_cue's header-layout CHECKs.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F

import explore_common as ec
import explore_common2 as c2
import explore_gates as eg
import explore_local_gate as lg
import explore_far_cue as s6
import test_router_curriculum as trc
import test_short_conv as tsc
from test_router_discovery import model_fn
from test_router_reliability import gate_fit_k
from test_multilayer_binding import build, FIT_ITERS, FIT_PASS
from test_instrument_v2 import Instrument, perfect_gate_general
from test_channel_binding import ARCH
from test_router_confirm import mcnemar_exact
from test_router_layout import fail_class

NAME = "far_express"
IDEA = "expressibility of the header-layout routing by each gate; arm A from a 5% labelled lean"
SOURCE = "CHECK 45's gate fit; test_router_curriculum's A_nudge (positive control on the grouped layout)"
CHANGE = "(a) supervised gate fits, no training runs; (b) far_nudge: A_nudge on the header layout"
PAIRING = ("far_nudge seeds 160-169, paired with batch 2's far_A on the same seeds (explore_out/"
           "far_cue_results.json)")
SEEDS = tuple(range(160, 170))
B2_STORE = "far_cue"
FIT_SEED, FIT_DATA_SEED, FIT_B, FIT_LR = 3, 11, 64, 1e-2
KINDS = ("A", "EMA", "SEL", "LOCAL3")
FITS_FILE = os.path.join(ec.OUT_DIR, "far_express_fits.json")
DISCOVERY_N, HEADSTART_N = 7, 3

ARMS = {
    "far_nudge": dict(trc.ARM["A_nudge"], key="far_nudge", lr=ec.SUB_LR, iters=ec.MAX_ITERS,
                      seeds=SEEDS, prio=1, label="arm A + the 5% labelled nudge, header layout"),
}


# ── (a) the fits ─────────────────────────────────────────────────────────────
def fit_model(task, kind):
    """(model, gate parameters, read-gate function) for a gate kind, built with seed FIT_SEED."""
    if kind == "A":
        m = build(task, ec.ARM_A, ARCH, FIT_SEED)
        return m, [m.W_in, m.W_h, m.W_g], lambda x: Instrument.gates(m, x, m.embed(x))[0]
    if kind in ("EMA", "SEL"):
        m = eg.make_model(task, ec.ARM_A, FIT_SEED, kind.lower())
        extra = [m.theta] if kind == "EMA" else [m.W_z, m.b_z]
        return m, [m.W_in, m.W_g] + extra, lambda x: m.variant_gates(m.embed(x))[0]
    assert task.vocab == ec.TASK.vocab and task.ctx_tokens == ec.TASK.ctx_tokens
    m = lg.make_model(ec.ARM_A, FIT_SEED)
    return m, [m.W_in, m.W_g, m.gate_conv.conv_w], lambda x: m.local_gates(m.embed(x))[0]


def fit_target(task):
    g = torch.Generator().manual_seed(FIT_DATA_SEED)
    tok, _ = task.make_batch(FIT_B, g)
    x = tok[:, :-1]
    return x, F.pad(perfect_gate_general(x, task.ctx_tokens), (0, 2 - task.S))


def fit(task, kind):
    """gate_fit_k's recipe, with the gate's own parameters."""
    x, tgt = fit_target(task)
    m, params, gate = fit_model(task, kind)
    opt = torch.optim.Adam(params + [m.embed.weight], lr=FIT_LR)
    for _ in range(FIT_ITERS):
        loss = F.mse_loss(gate(x), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        gr = gate(x)
        return F.mse_loss(gr, tgt).item(), (gr.argmax(-1) == tgt.argmax(-1)).float().mean().item()


def all_fits():
    out = {}
    for lay, task in (("header", s6.HEADER), ("grouped", ec.TASK)):
        for k in KINDS:
            mse, am = fit(task, k)
            out[f"{lay}|{k}"] = dict(mse=mse, argmax=am, fits=am >= FIT_PASS)
    return out


# ── (b) far_nudge ────────────────────────────────────────────────────────────
def builder(a, seed):
    return lambda: model_fn(s6.HEADER, a, seed)()


def run_job(arm, seed):
    a = ARMS[arm]
    return ec.run_one(a, seed, a["iters"], task=s6.HEADER, stats_fn=s6.header_stats,
                      grad_fn=tsc.conv_grad_norms, lr=a["lr"], builder=builder)


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], s6.HEADER, builder(a, 0), a["lr"], s6.header_stats)]


# ── CHECKs ───────────────────────────────────────────────────────────────────
def check():
    ok = True
    x, tgt = fit_target(s6.HEADER)
    lab = s6.HEADER.stream_labels(x)
    onehot = F.one_hot(lab, s6.HEADER.S).float()
    ceil = build(s6.HEADER, s6.ARMS["far_ceil"], ARCH, FIT_SEED)
    with torch.no_grad():
        ceil_g = ceil(x)[2]
    same_t = bool((lab >= 0).all()) and torch.equal(tgt, onehot) and torch.equal(tgt, ceil_g)
    mine = fit(ec.TASK, "A")
    ref = gate_fit_k(ec.TASK, ec.ARM_A)
    same_fit = mine == ref
    for nm, v in (("the header routing target equals the perfect gate's (one-hot of the block's "
                   "stream = far_ceil's gate) on the fit batch", same_t),
                  (f"fit('A') on the grouped layout reproduces CHECK 45's gate_fit_k at k=2 exactly "
                   f"(mse {mine[0]:.3e} vs {ref[0]:.3e}, argmax {mine[1]:.4f} vs {ref[1]:.4f})", same_fit)):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    print("  (the header layout: explore_far_cue's CHECKs)")
    ok &= s6.check()
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def report(store, save=True):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    fits = all_fits()
    if save:
        with open(FITS_FILE, "w") as f:
            json.dump(dict(banner=ec.BANNER, recipe=f"gate_fit_k: {FIT_B} sequences (seed "
                           f"{FIT_DATA_SEED}), model seed {FIT_SEED}, {FIT_ITERS} Adam steps at lr "
                           f"{FIT_LR} on the gate parameters and embed", fits=fits), f, indent=1)
    print(f"  (a) EXPRESSIBILITY: CHECK 45's recipe ({FIT_ITERS} Adam steps at lr {FIT_LR} on the "
          f"gate's parameters and the embedding, {FIT_B} sequences), FITS if argmax >= {FIT_PASS}:")
    for lay in ("header", "grouped"):
        tagl = "" if lay == "header" else "   (diagnostic)"
        print(f"    {lay} layout{tagl}:")
        for k in KINDS:
            f = fits[f"{lay}|{k}"]
            print(f"      {k:<7} mse {f['mse']:.3e}   argmax match {f['argmax']:.4f}   "
                  f"{'FITS' if f['fits'] else 'does not fit'}")
    fit_any = [k for k in KINDS if fits[f"header|{k}"]["fits"]]
    b2 = ec.load_store(B2_STORE)["runs"]
    print(f"  (b) far_nudge, per seed (ROUTED* at 1200/2400/3600/end; sep0 = the lean after the "
          f"pre-fit; b2 = batch 2's far_A):")
    print(f"    {'seed':>4} {'sep0':>6} {'acc':>6} {'trans':>6} {'VALcos':>7} {'margin':>6} "
          f"{'eta key s/k/h/i':>20} {'ROUTED*':>8}  {'outcome':<15} {'b2 far_A':<15}")
    runs = {}
    for s in ARMS["far_nudge"]["seeds"]:
        r = store["runs"].get(f"far_nudge|{s}")
        b = ec.tag(b2.get(f"far_A|{s}"))
        if r is None or not r.get("ok"):
            print(f"    {s:>4} {ec.tag(r):<70} {b}")
            continue
        runs[s] = r
        m = r["end"].get("margin")
        sep0 = r["stats"][0].get("sep", float("nan"))
        print(f"    {s:>4} {sep0:6.3f} {r['acc']:6.3f} {str(r['transition']):>6} {r['val_cos']:7.3f} "
              f"{'--' if m is None else f'{m:6.2f}':>6} {ec.eta_str(r):>20} {c2.rflags(r):>8}  "
              f"{ec.tag(r):<15} {b}")
    nd = sum(ec.discovered(r) for r in runs.values())
    fails = {}
    for r in runs.values():
        if not ec.bound(r):
            fails[fail_class(r)] = fails.get(fail_class(r), 0) + 1
    bb = sum(1 for s, r in runs.items() if ec.discovered(r) and not ec.discovered(b2.get(f"far_A|{s}")))
    cc = sum(1 for s, r in runs.items() if ec.discovered(b2.get(f"far_A|{s}")) and not ec.discovered(r))
    print(f"    far_nudge DISCOVERED {nd}/{len(runs)}   bound {sum(ec.bound(r) for r in runs.values())}"
          f"/{len(runs)}   ROUTED*@1200 {sum(c2.routed_at(r, 1200) for r in runs.values())}/{len(runs)}"
          f"   ROUTED*@end {sum(c2.routed_at(r, 'end') for r in runs.values())}/{len(runs)}   failures "
          f"{fails}   vs b2 far_A: far_nudge only {bb}, far_A only {cc}, McNemar p = "
          f"{mcnemar_exact(bb, cc):.3g}")
    if not fit_any:
        reading = "the gates cannot express the distant-cue routing"
    elif nd >= DISCOVERY_N:
        reading = "the distant cue is a discovery problem"
    elif nd <= HEADSTART_N:
        reading = "the loss cannot finish the distant-cue routing even from a head start"
    else:
        reading = f"neither reading applies (a gate fits; far_nudge {nd}/{len(runs)})"
    print(f"  gates that fit the header routing: {fit_any or 'none'}; the nudged gate (arm A's) "
          f"{'fits' if 'A' in fit_any else 'does NOT fit'}")
    print(f"  READING S10: {reading}")
    return dict(n=len(runs), discovered=nd, fit_any=fit_any, reading=reading, fits=fits)
