#!/usr/bin/env python
"""
test_short_conv.py — two questions, fixed before any result:
  - Can ONE memory channel bind the two-stream task once every position can see the few
    tokens before it through a short causal convolution? Mamba, DeltaNet, GLA and their
    relatives all have such a convolution; bdh.py does not.
  - Does the learned gate still add anything on top of the convolution?

Run it directly:

    python test_short_conv.py                      # CHECKs, projection, all arms
    python test_short_conv.py --also results/L/short_conv_results.json

BACKGROUND
- Single channel: no single-channel arm has ever bound this task (0/100 distinct runs
  before test_router_layout, and 0/30 in each machine's run of it).
- Learned gate: arm A (k=2 recurrent gate) finds the stream routing on 154/240 = 64%
  [58, 70] of grouped-layout runs pooled over both machines. The restart check (held-out
  accuracy >= 0.6 at step 2400) picked winners with precision 138/139 across three layouts
  and both machines.
- The mechanism a convolution would supply: in every triple the context token directly
  precedes its key, so a causal convolution of width >= 3 lets the value position carry its
  (context, key) pair. The recurrent gate supplies the same context by a different route.
- Scratch check (not pre-registered). It ran in a container that reproduces X step for step,
  on seeds 0-3, grouped layout, 24000 steps, convolution width 4, identity init:
  - Convolution at every layer: bound 2/4 (seed 3 at step 1200, seed 2 at 14400; seeds 0
    and 1 stayed at 0.50). With random init, seed 0 bound at 15600.
  - Convolution on the input embeddings only: bound 1/3 (seed 0 at 15600).
  - Late binders sat at 0.50 (a key bound to both candidate values) and then jumped to 1.00
    within one evaluation.
  - Blocked and shuffled layouts, seed 0: both plateaued at 0.50 and did not bind. Plain B
    in those layouts stays at ~0.22-0.29, so the convolution still lifts them.
  - Plain binding with a single stream (S=1, P=8, seed 0): bound at step 1200 with the
    per-layer convolution. Without it, test_channel_binding's Gate 0 on X bound this config
    at steps 37200 and 28800 on two seeds, and the third seed never bound.
  - So a single channel with a convolution CAN bind. How often, and whether the gate adds
    anything, is what this test measures.
- Seed-level outcomes differ between machines; the verdict is per machine.

THE CONVOLUTION (a backward-compatible knob on test_multilayer_binding.MultiBDH)
- conv=None (the default) creates nothing and leaves forward() calling bdh.BDH.forward
  unmodified (CHECK 53).
- conv in {"in", "layer"}, conv_k = 4: a depthwise, strictly causal convolution
  (test_multilayer_binding.CausalConv) with weight conv_w of shape (D, K),
  out_t = sum_{j=0}^{K-1} conv_w[:, j] * x_{t-j}, zero-padded, so position t sees only
  t-K+1..t (CHECK 55).
- Identity init: conv_w[:, 0] = 1, every other lag 0, drawing nothing from any RNG. A conv
  arm therefore starts bit-identical to its base arm on the same seed (CHECK 54; paired).
- "in": applied once, to the embeddings, before the first LayerNorm.
- "layer": one shared convolution (bdh.py shares all weights across layers) applied to x at
  every layer before the encoder; its output feeds Q = K and V; the residual stream is not
  convolved.
- With conv set, MultiBDH.forward_conv is a line-for-line copy of bdh.BDH.forward's layer
  loop with the convolution inserted. The gate still reads the raw, unconvolved embeddings.
- test_router_reliability.run_one gained a grad_fn knob (gradient norms by group), inert at
  its default (CHECK 53).

ARMS AND SEEDS (P=4, S=2, grouped layout, test_router_confirm's substrate and recipe:
n_layer=3, decay, N=256, lr 1e-3, BATCH 32, evaluation every 1200 steps on 2048 queries,
early stop after 3 evaluations >= 0.95; MAX_ITERS = 24000; fresh seeds disjoint from 0-159)
  B_conv        single channel + conv "layer"          seeds 160-199 (40)
  A_conv        arm A + conv "layer"                    seeds 160-199 (40)
  A             arm A, no conv (this seed set's rate)   seeds 160-199 (40)
  B_conv_in     single channel + conv "in"              seeds 160-179 (20), descriptive only
  B             plain single channel (control)          seeds 160-179 (20)
  ceiling_conv  perfect gate + conv "layer"             seeds 160-164 (5)
A, A_conv and B_conv share seeds, so they share initial parameters and batches.

VALIDITY AND PRE-REGISTERED CLAIMS (each printed with its numbers)
Outcome for every arm: BOUND (transition not None). For arms with a gate, DISCOVERED (bound
AND final VAL cos < 0.5) is printed alongside, because with a convolution a gated model can
bind without routing.
  VALID: ceiling_conv binds 5/5, AND the conv changes nothing at init (CHECK 54). Otherwise
    every claim is UNTESTED.
  K1 CONV LETS ONE CHANNEL BIND: B_conv bound vs B bound, Fisher one-sided (B_conv higher).
     SHOWN if p < 0.05, else NOT SHOWN.
  K2 GATE VS CONV: A bound vs B_conv bound, Fisher two-sided, with an exact McNemar test on
     the pairs printed alongside; which is higher is printed. DIFFERENT if the Fisher p <
     0.05, else NO DIFFERENCE SHOWN.
  K3 GATE ADDS TO CONV: A_conv bound vs B_conv bound, Fisher one-sided (A_conv higher).
     SHOWN if p < 0.05, else NOT SHOWN.
  Band for each 40-seed arm: RELIABLE >= 36/40, MAJORITY 20-35, MINORITY 1-19, NEVER 0.
  The pre-registered reading, printed verbatim after the claims:
  - K1 SHOWN and K3 NOT SHOWN: on this task the standard convolution gives one channel what
    the gate gave. Channels must earn their place on capacity, so next is P-scaling with the
    convolution in every arm.
  - K1 SHOWN and K3 SHOWN: the gate adds to the convolution; carry both into P-scaling.
  - K1 NOT SHOWN: the convolution does not substitute for routing here; P-scaling compares A
    with B_conv.
Fisher and McNemar are test_router_confirm's (two-sided Fisher test_readout_path's).

DIAGNOSTICS, not part of the verdict
- Every A_conv run at the end: VAL cos, the query routing margin, eta^2 by stream, key, half
  and index (test_router_layout's statistics). Bound runs are classed WITH routing (margin
  >= 0.9) or WITHOUT; failures by test_router_layout's classes. Arm A alongside, for
  reference.
- The convolution weights at the end: mean |w| per lag, every conv run, bound vs not.
- Per-seed outcomes of A / A_conv / B_conv side by side (B and B_conv_in where run); exact
  McNemar for A_conv vs B_conv and A vs A_conv.
- Plateau length (steps at held-out accuracy 0.45-0.55 before the transition) per bound run,
  and the transition-step distribution per arm.
- The restart check per arm (accuracy >= 0.6 at 2400): pass, of those bound, and binders the
  check would have thrown away.
- Gradient norms at steps 1, 10 and 100 (gate, conv, rest); --also pooled counts.

CHECKS (printed before any training): test_router_layout's verification (which runs every
earlier test's), then
  53 conv=None is inert (A, B, E_mem vs 37f9939's modules; forward still calls
     bdh.BDH.forward; run_one's grad_fn knob inert);
  54 identity init reproduces the base arm (logits, parameters, global RNG state);
  55 the convolution is what it claims (explicit loop; model-level causality; "layer" is one
     parameter called n_layer times, "in" once);
  56 ceiling_conv applies the perfect gate at every layer (CHECK 6's test with the conv on);
  57 a worker's run is bit-identical to the same run in the main process.

LOGISTICS. Parallel single-thread workers; wall clock projected before training (no drop
rule). Results persist atomically to short_conv_results.json (gitignored; copied into
results/X/ after the run).

(Results are recorded at the bottom of this docstring after the run.)
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F

import bdh
from test_binding_onset import onset_run, fmt_step, logits_of, EVAL_EVERY
from test_multilayer_binding import CausalConv, D
from test_channel_binding import eval_batch, arms as cb_arms
from test_router_confirm import fisher_greater, mcnemar_exact
import test_router_curriculum as trc
from test_router_curriculum import (
    TASK, ARCH, SUB_LR, GATE_PARAMS, make_fn, get, count, load_store, init_worker, makespan,
    cpu_model, same_weights, save_results, time_arm, DISC_COS, COLLAPSE, TAU_END,
)
from test_router_discovery import load_legacy_pair, load_at
from test_readout_path import GRAD_STEPS, fisher_two_sided, grad_norms
from test_router_reliability import run_one, strip_all, curve_acc, med, curve_lines, T_CHECK, A_CHECK
import test_router_layout as trl
from test_router_layout import layout_stats, fail_class, ETA_GROUPS

# ── Settings (fixed before any result) ───────────────────────────────────────
MAX_ITERS = 24000
CONV_K = 4
ALPHA = 0.05
RELIABLE_K, MAJORITY_K = 36, 20            # bands, of 40 (MINORITY 1-19, NEVER 0)
ROUTED_MARGIN = 0.9                        # a bound gated run "routes" if margin >= 0.9
PLATEAU = (0.45, 0.55)
LEGACY_SHA = "37f9939"                     # head before this test's changes
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "short_conv_results.json"
EARLIER_SEEDS = tuple(range(160))          # every seed used by earlier router tests (0-159)

LAYER = dict(conv="layer", conv_k=CONV_K)
IN = dict(conv="in", conv_k=CONV_K)
_BASE = {a["key"]: a for a in cb_arms(TASK)}
A = trc.ARM["A"]
B = _BASE["B"]
ARMS = [
    dict(B, key="B_conv", model_kw=LAYER, label="B_conv        single channel + conv 'layer'"),
    dict(A, key="A_conv", model_kw=LAYER, label="A_conv        arm A + conv 'layer'"),
    dict(A, key="A", label="A             arm A, no conv"),
    dict(B, key="B_conv_in", model_kw=IN, label="B_conv_in     single channel + conv 'in'"),
    dict(B, key="B", label="B             plain single channel"),
    dict(_BASE["ceiling"], key="ceiling_conv", model_kw=LAYER,
         label="ceiling_conv  perfect gate + conv 'layer'"),
]
ARM = {a["key"]: a for a in ARMS}
KEYS = tuple(ARM)
GATED = tuple(a["key"] for a in ARMS if a["gate"] != "none")
CONV_KEYS = tuple(a["key"] for a in ARMS if a.get("model_kw", {}).get("conv"))
SEEDS = dict(B_conv=tuple(range(160, 200)), A_conv=tuple(range(160, 200)),
             A=tuple(range(160, 200)), B_conv_in=tuple(range(160, 180)),
             B=tuple(range(160, 180)), ceiling_conv=tuple(range(160, 165)))
FORTY = ("B_conv", "A_conv", "A")
READING = {
    "K1+ K3-": "on this task the standard convolution gives one channel what the gate gave. "
               "Channels must earn their place on capacity, so next is P-scaling with the "
               "convolution in every arm.",
    "K1+ K3+": "the gate adds to the convolution; carry both into P-scaling.",
    "K1-": "the convolution does not substitute for routing here; P-scaling compares A with "
           "B_conv.",
}


# ── Statistics and gradient groups ───────────────────────────────────────────
def conv_stats(model, task, probe, step):
    """run_one's stats_fn: test_router_layout.layout_stats, plus the convolution's mean
    |weight| per lag when the model has one."""
    out = layout_stats(model, task, probe, step)
    if getattr(model, "conv", None) is not None:
        w = model.short_conv.conv_w.detach()
        out["conv_absmean"] = [w[:, j].abs().mean().item() for j in range(w.shape[1])]
    return out


def conv_grad_norms(m):
    """run_one's grad_fn: gradient norms (gate = W_in, W_h, W_g; rest; conv = conv_w)."""
    gate = rest = conv = 0.0
    for n, p in m.named_parameters():
        if p.grad is None or n == "W_ro":
            continue
        v = float(p.grad.detach().pow(2).sum())
        if n in GATE_PARAMS:
            gate += v
        elif n.startswith("short_conv."):
            conv += v
        else:
            rest += v
    return math.sqrt(gate), math.sqrt(rest), math.sqrt(conv)


# ── Worker-side ──────────────────────────────────────────────────────────────
def run_job(sp):
    return run_one(ARM[sp["arm"]], sp["seed"], sp["iters"], task=TASK, stats_fn=conv_stats,
                   grad_fn=conv_grad_norms)


def spec(key, seed, iters):
    return dict(arm=key, seed=seed, iters=iters)


# ── Verification ─────────────────────────────────────────────────────────────
def verify(pool):
    print("test_router_layout.py's verification (which runs test_router_reliability's, and so on")
    print("down to test_multilayer_binding's):")
    trl.verify(pool)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True
    res = {}
    t = TASK
    data = eval_batch(t)
    x = t.make_batch(16, torch.Generator().manual_seed(53))[0][:, :-1]

    print(f"CHECK 53 conv=None is inert: 60-step runs (evaluations every 20, seed 3) vs {LEGACY_SHA}'s "
          f"modules:")
    leg_ml, leg_on = load_legacy_pair(LEGACY_SHA)
    print(f"         legacy MultiBDH takes conv: {'conv' in leg_ml.MultiBDH.__init__.__code__.co_varnames}")
    good = True
    for key, a in (("A", A), ("B", B), ("E_mem", _BASE["E_mem"])):
        m0, c0, _ = leg_on.onset_run(t, lambda: leg_ml.build(t, a, ARCH, 3), 3, max_iters=60,
                                     eval_every=20, data=data, early_stop=False, lr=SUB_LR)
        m1, c1, _ = onset_run(t, make_fn(a, 3), 3, max_iters=60, eval_every=20, data=data,
                              early_stop=False, lr=SUB_LR)
        g = same_weights(m0, m1) and c0 == c1 and not hasattr(m1, "short_conv")
        good &= g
        print(f"         {key:<6} weights equal {same_weights(m0, m1)}   curves equal {c0 == c1}   "
              f"accs {[round(v, 4) for _, v, _ in c1]}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    calls = {"n": 0}
    orig = bdh.BDH.forward

    def spy(self, *args, **kw):
        calls["n"] += 1
        return orig(self, *args, **kw)
    bdh.BDH.forward = spy
    try:
        with torch.no_grad():
            make_fn(A, 3)()(x, TAU_END)
            n_none, calls["n"] = calls["n"], 0
            make_fn(dict(A, model_kw=LAYER), 3)()(x, TAU_END)
            n_conv = calls["n"]
    finally:
        bdh.BDH.forward = orig
    g = n_none == 1 and n_conv == 0
    good &= g
    print(f"         bdh.BDH.forward calls per forward: conv=None {n_none}, conv='layer' {n_conv} "
          f"(its own copy of the loop)  -> {'OK' if g else 'WRONG'}")
    leg_trr = load_at(LEGACY_SHA, "test_router_reliability.py", "test_router_reliability_legacy")
    r_leg = leg_trr.run_one(A, 3, 60, eval_every=20)
    r_new = run_one(A, 3, 60, eval_every=20)
    r_exp = run_one(A, 3, 60, eval_every=20, grad_fn=grad_norms)
    g = strip_all(r_leg) == strip_all(r_new) == strip_all(r_exp) and r_new["ok"]
    good &= g
    print(f"         test_router_reliability.run_one's grad_fn knob at its default and explicit vs "
          f"{LEGACY_SHA}'s run_one: records equal {g}")
    ok &= good
    res[53] = good
    print()

    print("CHECK 54 identity init reproduces the base arm (seed 5, 16 training sequences):")
    good = True
    for bk, base in (("A", A), ("B", B)):
        for kw in (IN, LAYER):
            m0 = make_fn(base, 5)()
            r0 = torch.get_rng_state()
            m1 = make_fn(dict(base, model_kw=kw), 5)()
            r1 = torch.get_rng_state()
            p0, p1 = dict(m0.named_parameters()), dict(m1.named_parameters())
            same_p = all(torch.equal(p0[n], p1[n]) for n in p0) and set(p1) - set(p0) == {
                "short_conv.conv_w"}
            w = m1.short_conv.conv_w
            ident = bool((w[:, 0] == 1).all() and (w[:, 1:] == 0).all()) and tuple(w.shape) == (D, CONV_K)
            same_l = True
            for train in (True, False):
                m0.train(train), m1.train(train)
                with torch.no_grad():
                    o0, o1 = m0(x, TAU_END), m1(x, TAU_END)
                same_l &= torch.equal(o0[0], o1[0]) and torch.equal(o0[2], o1[2])
            same_rng = torch.equal(r0, r1)
            g = same_p and ident and same_l and same_rng
            good &= g
            print(f"         {bk} + conv {kw['conv']!r:<7} logits (and gate) bitwise equal, train and eval: "
                  f"{same_l}   base parameters equal: {same_p}   conv_w {tuple(w.shape)} identity: {ident}   "
                  f"global RNG state after build equal: {same_rng}  -> {'IDENTICAL AT INIT' if g else 'DIFFERS'}")
    ok &= good
    res[54] = good
    print()

    print("CHECK 55 the convolution is what it claims:")
    g_ = torch.Generator().manual_seed(55)
    conv = CausalConv(D, CONV_K)
    with torch.no_grad():
        conv.conv_w.copy_(torch.randn(D, CONV_K, generator=g_))
        xr = torch.randn(3, 1, 12, D, generator=g_)
        out = conv(xr)
        ref = torch.zeros_like(xr)
        for tt in range(xr.shape[-2]):
            for j in range(CONV_K):
                if tt - j >= 0:
                    ref[..., tt, :] += conv.conv_w[:, j] * xr[..., tt - j, :]
        dmax = (out - ref).abs().max().item()
        t0 = 7

        def bump(pos):
            x2 = xr.clone()
            x2[..., pos, :] += 1.0
            return (conv(x2)[..., t0, :] - out[..., t0, :]).abs().max().item()
        seen = [bump(p) > 0 for p in range(t0 - CONV_K, t0 + 2)]
    want = [False] + [True] * CONV_K + [False]
    good = dmax < 1e-6 and seen == want
    print(f"         explicit loop out_t = sum_j w_j * x_(t-j), random input and weights: max diff "
          f"{dmax:.1e}   position {t0} sees positions {t0 - CONV_K}..{t0 + 1}: {seen} (want {want})")
    tq = 10
    for bk, base in (("A", A), ("B", B)):
        for kw in (IN, LAYER):
            m = make_fn(dict(base, model_kw=kw), 7)()
            with torch.no_grad():
                m.short_conv.conv_w.copy_(torch.randn(D, CONV_K, generator=g_))
            x2 = x.clone()
            x2[:, tq + 1:] = (x2[:, tq + 1:] + 1) % t.vocab
            line = []
            for train in (True, False):
                m.train(train)
                with torch.no_grad():
                    d = (logits_of(m, x) - logits_of(m, x2)).abs()
                before, after = d[:, :tq + 1].max().item(), d[:, tq + 1:].max().item()
                g = before == 0.0 and after > 0
                good &= g
                line.append(f"{'train' if train else 'eval'} <= t {before:.1e}, > t {after:.1e}")
            print(f"         {bk} + conv {kw['conv']!r:<7} random conv weights, every token after "
                  f"position {tq} changed: logit change {'; '.join(line)}")
    for kw, want_calls in ((IN, 1), (LAYER, ARCH["n_layer"])):
        m = make_fn(dict(A, model_kw=kw), 7)()
        rec = []
        h = m.short_conv.register_forward_hook(lambda mod, inp, o: rec.append(id(mod.conv_w)))
        try:
            with torch.no_grad():
                m(x, TAU_END)
        finally:
            h.remove()
        n_params = sum(1 for n, _ in m.named_parameters() if n.startswith("short_conv."))
        g = len(rec) == want_calls and len(set(rec)) == 1 and n_params == 1
        good &= g
        print(f"         conv {kw['conv']!r:<7} calls per forward {len(rec)} (want {want_calls}), "
              f"distinct parameters used {len(set(rec))}, conv parameters in the model {n_params}")
    print(f"         -> {'AS CLAIMED' if good else 'WRONG'}")
    ok &= good
    res[55] = good
    print()

    print("CHECK 56 ceiling_conv applies the perfect gate at every layer (CHECK 6's test, conv on "
          "with random weights):")
    m = make_fn(ARM["ceiling_conv"], 6)().eval()
    with torch.no_grad():
        m.short_conv.conv_w.copy_(torch.randn(D, CONV_K, generator=g_))
    m.attn.record = []
    with torch.no_grad():
        m(x, TAU_END)
    lab = t.stream_labels(x)
    cross = (lab.unsqueeze(2) != lab.unsqueeze(1)) & (lab.unsqueeze(2) >= 0) & (lab.unsqueeze(1) >= 0)
    maxc = [s[:, 0][cross].abs().max().item() for s in m.attn.record]
    live = [s[:, 0][~cross].abs().max().item() for s in m.attn.record]
    m.attn.record = None
    good = len(maxc) == ARCH["n_layer"] and max(maxc) == 0.0 and min(live) > 0
    print(f"         layers seen: {len(maxc)} (n_layer={ARCH['n_layer']})")
    for i, (c, l) in enumerate(zip(maxc, live)):
        print(f"         layer {i}: max |cross-stream score| = {c:.2e}   max |same-stream score| = {l:.2e}")
    print(f"         -> {'APPLIED AT EVERY LAYER' if good else 'FAILED'}")
    ok &= good
    res[56] = good
    print()

    print("CHECK 57 a worker's run is bit-identical to the same run here:")
    good = True
    for key in ("A_conv", "B_conv"):
        sp = spec(key, 1, EVAL_EVERY)
        rw, rh = pool.submit(run_job, sp).result(), run_job(sp)
        g = strip_all(rw) == strip_all(rh) and rh["ok"]
        good &= g
        print(f"         {key:<7} seed 1, {EVAL_EVERY} steps: equal {strip_all(rw) == strip_all(rh)}   "
              f"acc {rh['acc']:.4f}   conv mean |w| per lag {[round(v, 3) for v in rh['end']['conv_absmean']]}"
              f"  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    res[57] = good
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"
    return res


# ── Reporting ────────────────────────────────────────────────────────────────
def bound(store, key, s):
    r = get(store, key, s)
    return bool(r and r.get("ok") and r.get("transition") is not None)


def disc(store, key, s):
    r = get(store, key, s)
    return bool(r and r.get("ok") and r.get("discovered"))


def ok_runs(store, key):
    return [(s, r) for s in SEEDS[key] if (r := get(store, key, s)) and r.get("ok")]


def band(c):
    return ("RELIABLE" if c >= RELIABLE_K else "MAJORITY" if c >= MAJORITY_K
            else "MINORITY" if c >= 1 else "NEVER")


def plateau(r):
    """Steps spent at held-out accuracy 0.45-0.55 before the transition."""
    tr = r["transition"]
    return sum(EVAL_EVERY for st, acc, _ in r["curve"]
               if st < tr and PLATEAU[0] <= acc <= PLATEAU[1])


def routed(r):
    return r["end"].get("margin", 0.0) >= ROUTED_MARGIN


def wstr(e):
    return "/".join(f"{v:.2f}" for v in e["conv_absmean"]) if "conv_absmean" in e else "--"


def report(store, wall, path, also):
    print("#" * 100)
    print("PER-SEED RAW RESULTS — every value, before any aggregate")
    print("#" * 100)
    print(f"  {'arm':<13} {'seed':>4} {'acc':>7} {'transition':>10} {'VAL cos':>8} {'margin':>7}  "
          f"eta^2 key pos s/k/h/i   conv mean|w| lag 0/1/2/3   flags")
    for k in GATED:
        for s in SEEDS[k]:
            r = get(store, k, s)
            if r is None:
                print(f"  {k:<13} {s:>4}  NOT RUN")
                continue
            if not r.get("ok"):
                print(f"  {k:<13} {s:>4}  FAILED — {r['error']}")
                continue
            e = r["end"]
            etas = "/".join(f"{e['eta_key_by_' + g]:.2f}" for g in ETA_GROUPS)
            fl = ((["BOUND"] + (["DISCOVERED"] if r["discovered"] else [])
                   + ["routed" if routed(r) else "NOT routed"]) if r["transition"] is not None
                  else [fail_class(r)])
            print(f"  {k:<13} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} {e['val_cos']:>8.4f} "
                  f"{e['margin']:>7.4f}  {etas:<21}   {wstr(e):<25}   {' '.join(fl)}")
        print()
    print(f"  {'arm':<13} {'seed':>4} {'acc':>7} {'transition':>10} {'stopped':>7}   conv mean|w| lag "
          f"0/1/2/3   flags")
    for k in KEYS:
        if k in GATED:
            continue
        for s in SEEDS[k]:
            r = get(store, k, s)
            if r is None:
                print(f"  {k:<13} {s:>4}  NOT RUN")
            elif not r.get("ok"):
                print(f"  {k:<13} {s:>4}  FAILED — {r['error']}")
            else:
                print(f"  {k:<13} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} "
                      f"{r['stopped_at']:>7}   {wstr(r['end']):<25}   {'BOUND' if r['transition'] else ''}")
        print()

    print("=" * 100)
    print("COUNTS (BOUND = transition not None; DISCOVERED = bound AND final VAL cos < 0.5, gated arms)")
    print("=" * 100)
    c, d = {}, {}
    for a in ARMS:
        k, ns = a["key"], len(SEEDS[a["key"]])
        c[k] = sum(bound(store, k, s) for s in SEEDS[k])
        d[k] = sum(disc(store, k, s) for s in SEEDS[k])
        rs = [get(store, k, s) for s in SEEDS[k]]
        failed = sum(1 for r in rs if r is not None and not r.get("ok"))
        print(f"  {a['label']:<44} bound {c[k]:>2}/{ns}"
              + (f"   discovered {d[k]:>2}/{ns}" if k in GATED else " " * 18)
              + f"   completed {sum(1 for r in rs if r and r.get('ok'))}/{ns}"
              + (f"   [{failed} FAILED]" if failed else ""))
    print()

    print("#" * 100)
    print("VALIDITY")
    print("#" * 100)
    init_ok = bool(store.get("meta", {}).get("checks", {}).get("54"))
    nc = len(SEEDS["ceiling_conv"])
    valid = c["ceiling_conv"] == nc and init_ok
    print(f"  ceiling_conv binds {c['ceiling_conv']}/{nc}   the conv changes nothing at init (CHECK 54): "
          f"{init_ok}")
    print(f"     *** {'VALID' if valid else 'UNTESTED'} ***")
    print()

    print("#" * 100)
    print("PRE-REGISTERED CLAIMS (outcome: BOUND)")
    print("#" * 100)
    n40, n20 = len(SEEDS["B_conv"]), len(SEEDS["B"])
    v = dict(bound=c, discovered=d, valid=valid)
    p1 = fisher_greater(c["B_conv"], n40, c["B"], n20)
    k1 = ("SHOWN" if p1 < ALPHA else "NOT SHOWN") if valid else "UNTESTED"
    print(f"  K1 CONV LETS ONE CHANNEL BIND (Fisher one-sided): B_conv {c['B_conv']}/{n40} vs B "
          f"{c['B']}/{n20}; p = {p1:.3g}")
    print(f"     *** K1: {k1} ***")
    seeds = SEEDS["A"]
    p2 = fisher_two_sided(c["A"], n40, c["B_conv"], n40)
    bb = sum(bound(store, "A", s) and not bound(store, "B_conv", s) for s in seeds)
    cc = sum(bound(store, "B_conv", s) and not bound(store, "A", s) for s in seeds)
    pm = mcnemar_exact(bb, cc)
    k2 = ("DIFFERENT" if p2 < ALPHA else "NO DIFFERENCE SHOWN") if valid else "UNTESTED"
    higher = "A" if c["A"] > c["B_conv"] else "B_conv" if c["B_conv"] > c["A"] else "neither (equal)"
    print(f"  K2 GATE VS CONV (Fisher two-sided): A {c['A']}/{n40} vs B_conv {c['B_conv']}/{n40}; "
          f"p = {p2:.3g}; higher: {higher}")
    print(f"     exact McNemar on the pairs (seeds {seeds[0]}-{seeds[-1]}): A only {bb}, B_conv only {cc}; "
          f"p = {pm:.3g}")
    print(f"     *** K2: {k2} ***")
    p3 = fisher_greater(c["A_conv"], n40, c["B_conv"], n40)
    k3 = ("SHOWN" if p3 < ALPHA else "NOT SHOWN") if valid else "UNTESTED"
    print(f"  K3 GATE ADDS TO CONV (Fisher one-sided): A_conv {c['A_conv']}/{n40} vs B_conv "
          f"{c['B_conv']}/{n40}; p = {p3:.3g}")
    print(f"     *** K3: {k3} ***")
    print(f"  BANDS (RELIABLE >= {RELIABLE_K}/40, MAJORITY {MAJORITY_K}-{RELIABLE_K - 1}, MINORITY 1-"
          f"{MAJORITY_K - 1}, NEVER 0), on BOUND:")
    for k in FORTY:
        print(f"     *** {k:<7} {c[k]}/{n40}: {band(c[k])} ***")
        v[f"band_{k}"] = band(c[k])
    v.update(K1=k1, K1_p=p1, K2=k2, K2_p=p2, K2_mcnemar_p=pm, K2_higher=higher, K3=k3, K3_p=p3)
    print()
    print("  THE PRE-REGISTERED READING:")
    if not valid:
        rk = None
        print("     (UNTESTED: no reading)")
    else:
        rk = "K1-" if k1 == "NOT SHOWN" else ("K1+ K3+" if k3 == "SHOWN" else "K1+ K3-")
        cond = {"K1-": "K1 NOT SHOWN", "K1+ K3+": "K1 SHOWN and K3 SHOWN",
                "K1+ K3-": "K1 SHOWN and K3 NOT SHOWN"}[rk]
        print(f"     {cond}: {READING[rk]}")
    v["reading"] = rk
    store["verdict"] = v
    save_results(path, store)
    print()

    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    print(f"  gated runs at the end: bound WITH routing (margin >= {ROUTED_MARGIN}) or WITHOUT; failures "
          f"by test_router_layout's classes:")
    for k in ("A_conv", "A"):
        rs = ok_runs(store, k)
        w_ = [s for s, r in rs if r["transition"] is not None and routed(r)]
        wo = [s for s, r in rs if r["transition"] is not None and not routed(r)]
        classes = {"STREAM-PARTIAL": [], "KEY": [], "POSITION": [], "OTHER": []}
        for s, r in rs:
            if r["transition"] is None:
                classes[fail_class(r)].append(s)
        print(f"    {k:<7} bound with routing {len(w_)}, WITHOUT routing {len(wo)}"
              + (f" ({' '.join(f's{s}' for s in wo)})" if wo else "")
              + "   failures: " + "   ".join(f"{m} {len(x)}" for m, x in classes.items()))
        for m, xs in classes.items():
            if xs:
                print(f"    {'':<7}   {m:<14} " + " ".join(f"s{s}" for s in xs))
        for lab, sub in (("bound", [r for _, r in rs if r["transition"] is not None]),
                         ("not bound", [r for _, r in rs if r["transition"] is None])):
            if sub:
                print(f"    {'':<7} {lab:<9} ({len(sub):>2}) VAL cos {med([r['end']['val_cos'] for r in sub])}   "
                      f"margin {med([r['end']['margin'] for r in sub])}")
                print(f"    {'':<7} {'':<9}      eta^2 at key positions: " + "  ".join(
                    f"{g} {med([r['end']['eta_key_by_' + g] for r in sub])}" for g in ETA_GROUPS))
    print()
    print("  the convolution's mean |w| per lag at the end, median over runs (lag 0 = the position itself):")
    for k in CONV_KEYS:
        rs = ok_runs(store, k)
        for lab, sub in (("bound", [r for _, r in rs if r["transition"] is not None]),
                         ("not bound", [r for _, r in rs if r["transition"] is None])):
            if sub:
                print(f"    {k:<13} {lab:<9} ({len(sub):>2}) " + "   ".join(
                    f"lag {j}: {med([r['end']['conv_absmean'][j] for r in sub])}" for j in range(CONV_K)))
    print()
    print("  per-seed outcome (B = bound, D = bound and discovered, . = neither, - = not run):")
    cols = ("A", "A_conv", "B_conv", "B_conv_in", "B")

    def mark(k, s):
        if s not in SEEDS[k]:
            return "-"
        return "D" if disc(store, k, s) else "B" if bound(store, k, s) else "."
    print("    seed  " + "  ".join(f"{k:>9}" for k in cols))
    for s in seeds:
        print(f"    {s:>4}  " + "  ".join(f"{mark(k, s):>9}" for k in cols))
    for k1_, k2_ in (("A_conv", "B_conv"), ("A", "A_conv"), ("A", "B_conv")):
        b_ = sum(bound(store, k1_, s) and not bound(store, k2_, s) for s in seeds)
        c_ = sum(bound(store, k2_, s) and not bound(store, k1_, s) for s in seeds)
        print(f"    McNemar exact two-sided (bound), {k1_} vs {k2_}: {k1_} only {b_}, {k2_} only {c_}; "
              f"p = {mcnemar_exact(b_, c_):.3g}")
    print()
    print(f"  transition steps and plateau length (steps at held-out accuracy {PLATEAU[0]}-{PLATEAU[1]} "
          f"before the transition), per bound run:")
    for k in KEYS:
        rs = [(s, r) for s, r in ok_runs(store, k) if r["transition"] is not None]
        if not rs:
            print(f"    {k:<13} none bound")
            continue
        trs = [r["transition"] for _, r in rs]
        pls = [plateau(r) for _, r in rs]
        print(f"    {k:<13} transition median {statistics.median(trs):.0f} [{min(trs)}, {max(trs)}]   "
              f"plateau median {statistics.median(pls):.0f} [{min(pls)}, {max(pls)}]")
        print(f"    {'':<13} " + "  ".join(f"s{s}:{r['transition']}/{plateau(r)}" for s, r in rs))
    print()
    print(f"  the restart check per arm (held-out accuracy >= {A_CHECK} at step {T_CHECK}):")
    for k in KEYS:
        rs = ok_runs(store, k)
        passed = [(s, r) for s, r in rs if (a_ := curve_acc(r, T_CHECK)) is not None and a_ >= A_CHECK]
        pb = sum(1 for _, r in passed if r["transition"] is not None)
        thrown = [s for s, r in rs if r["transition"] is not None and (curve_acc(r, T_CHECK) or 0.0) < A_CHECK]
        print(f"    {k:<13} pass {len(passed)}/{len(rs)}; of those bound {pb}/{len(passed)}; binders the "
              f"check would throw away: {len(thrown)}" + (f" ({' '.join(f's{s}' for s in thrown)})" if thrown else ""))
    print()
    print("  gradient norms (gate = W_in, W_h, W_g; conv = conv_w; rest = all other parameters), median "
          "[min, max] over seeds:")
    for step in GRAD_STEPS:
        for k in KEYS:
            g = [r["grad"][str(step)] for _, r in ok_runs(store, k) if str(step) in r.get("grad", {})]
            if not g:
                continue
            parts = []
            for i, name in ((0, "gate"), (2, "conv"), (1, "rest")):
                xs = [x[i] for x in g]
                parts.append(f"{name} {statistics.median(xs):.3e} [{min(xs):.3e}, {max(xs):.3e}]")
            print(f"    step {step:>3} {k:<13} " + "   ".join(parts))
    print(f"  collapsed runs (final accuracy < {COLLAPSE}): "
          + ", ".join(f"{k} {count(store, k, SEEDS[k], 'collapsed')}" for k in KEYS))
    print()
    if also:
        print(f"  --also: pooled with {also}:")
        try:
            with open(also) as f:
                other = json.load(f)
            print(f"    other machine: {other.get('meta', {}).get('cpu', 'unknown')}, git "
                  f"{other.get('meta', {}).get('git')}")
            for k in KEYS:
                runs = [r for kk, r in other.get("runs", {}).items() if kk.split("|")[0] == k]
                b2 = sum(1 for r in runs if r.get("ok") and r.get("transition") is not None)
                d2 = sum(1 for r in runs if r.get("ok") and r.get("discovered"))
                ns = len(SEEDS[k])
                print(f"    {k:<13} bound here {c[k]}/{ns}   other {b2}/{len(runs)}   pooled "
                      f"{c[k] + b2}/{ns + len(runs)}"
                      + (f"      discovered here {d[k]}/{ns}   other {d2}/{len(runs)}" if k in GATED else ""))
        except Exception as e:
            print(f"    could not read {also}: {e}")
        print()

    print("=" * 100)
    print(f"CURVES — accuracy x100 at each evaluation every {EVAL_EVERY} steps, plus VAL cos x100 for "
          f"gated arms")
    print("=" * 100)
    for k in KEYS:
        for s, r in ok_runs(store, k):
            tag = f"{k:<13} s{s}"
            if k in GATED:
                curve_lines(tag, r)
            else:
                accs = "".join(f" {round(acc * 100):>3}" for _, acc, _ in r["curve"])
                print(f"  {tag} acc {accs}   -> "
                      f"{'bound @' + str(r['transition']) if r['transition'] else 'not bound'}")
    print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: "
          f"{path}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Does a short causal convolution let one channel bind?")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--also", default=None,
                    help="a second machine's short_conv_results.json, for pooled counts")
    args = ap.parse_args()
    torch.set_num_threads(1)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    all_seeds = set().union(*SEEDS.values())

    print("=" * 100)
    print("Short convolution: can one channel bind with a causal convolution, and does the gate add?")
    print(f"  test_router_confirm's config (S=2 P=4 n_vals=16 n_q=1, n_layer=3 decay N=256, lr "
          f"{SUB_LR}), grouped layout, MAX_ITERS={MAX_ITERS}; conv width {CONV_K}, identity init")
    for a in ARMS:
        s = SEEDS[a["key"]]
        print(f"  {a['label']:<44} seeds {s[0]}-{s[-1]} ({len(s)})")
    print(f"  all seeds disjoint from earlier seeds 0-159: {not all_seeds & set(EARLIER_SEEDS)}")
    print(f"  torch {torch.__version__}   CPU {cpu_model()}   {args.workers} worker processes x 1 "
          f"thread   git {head}")
    print("=" * 100)
    print()

    os.environ.setdefault("PYTHONWARNINGS", "ignore:Failed to initialize NumPy")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                             initializer=init_worker) as pool:
        checks = verify(pool)
        store = load_store(args.results)
        store["meta"] = dict(torch=torch.__version__, cpu=cpu_model(), workers=args.workers,
                             git=head, max_iters=MAX_ITERS, conv_k=CONV_K,
                             seeds={k: list(v) for k, v in SEEDS.items()},
                             checks={str(k): v for k, v in checks.items()},
                             started=time.strftime("%Y-%m-%d %H:%M:%S"),
                             note="seed-level outcomes differ between machines")
        save_results(args.results, store)
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — time_per_step (200 steps) inside the pool, evaluation charged once per "
              f"{EVAL_EVERY}; worst case = every run to {MAX_ITERS}")
        print("=" * 100)
        costs = dict(zip(KEYS, pool.map(time_arm, [ARM[k] for k in KEYS])))
        for k in KEYS:
            print(f"  {k:<13} {costs[k][0] * 1000:6.1f} ms/step   x {len(SEEDS[k])} seeds")
        allspecs = [spec(k, s, MAX_ITERS) for k in KEYS for s in SEEDS[k]]
        durs = [costs[sp["arm"]][0] * sp["iters"] for sp in allspecs]
        print(f"  {len(durs)} runs: serial {sum(durs) / 3600:.2f} h, projected wall clock on "
              f"{args.workers} workers {makespan(durs, args.workers) / 3600:.2f} h (no drop rule)")
        store["meta"]["projected_wall_h"] = makespan(durs, args.workers) / 3600
        print()

        print("=" * 100)
        print(f"RUNS — {len(allspecs)} runs")
        print("=" * 100)
        todo = []
        for sp in allspecs:
            if get(store, sp["arm"], sp["seed"]) is not None and not args.force:
                print(f"   {sp['arm']:<13} seed {sp['seed']}  cached")
            else:
                todo.append(sp)
        todo.sort(key=lambda sp: -costs[sp["arm"]][0])
        futs = {pool.submit(run_job, sp): sp for sp in todo}
        for f in as_completed(futs):
            sp = futs[f]
            try:
                rec = f.result()
            except Exception as e:
                rec = dict(ok=False, seed=sp["seed"], error=f"worker {type(e).__name__}: {e}")
            store["runs"][f"{sp['arm']}|{sp['seed']}"] = rec
            save_results(args.results, store)
            el = (time.time() - t0) / 60
            if rec.get("ok"):
                e, tr = rec["end"], rec["transition"]
                extra = (f"VALcos={e['val_cos']:.3f} margin={e['margin']:.3f} " if "margin" in e else "")
                print(f"   {sp['arm']:<13} seed {sp['seed']}  "
                      f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} "
                      f"@{rec['stopped_at']:<5}  {extra}conv|w|={wstr(e)}"
                      f"{'  DISCOVERED' if rec['discovered'] else ''}  {rec['secs']:.0f}s  "
                      f"(elapsed {el:.1f}m)")
            else:
                print(f"   {sp['arm']:<13} seed {sp['seed']}  *** FAILED: {rec['error']}   "
                      f"(elapsed {el:.1f}m)")
        print()

    report(store, time.time() - t0, args.results, args.also)


if __name__ == "__main__":
    main()
