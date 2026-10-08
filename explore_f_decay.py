#!/usr/bin/env python
"""
explore_f_decay.py — EXPLORATORY, not a result. Session F, screen S59: the memory horizon on delta-rule channels. Does a
learned, token-conditioned forget gate — Raven's routed decay or Gated DeltaNet's per-channel decay — widen the gap
between the perfect partition and a single channel, compared with the fixed decay 0.95? Background:
docs/reading/distant_cues_2026-10.md §4.6; docs/reading/2412.06464.md (the official code's α parameterisation).

CONFIGURATION (explore_f_decay_child; this branch's modules, 1 torch thread). The header layout at P = 8, S = 2
(explore_far_cue.HeaderTask(8, S=2, n_vals=16, n_q=1); L = 37), no convolution; S53's recipe: Adam 1e-3 on every
parameter, evaluation every 1200 on 2048 held-out sequences, early stop after 3 evaluations >= 0.95, 48000 updates.
Every arm is on DELTA channels: explore_delta_mem.to_delta(beta=1.0, normalize=True) (β = 1, L2 keys, tied write),
with the decay:
  FIXED    α = 0.95.
  ROUTED   α^c_t = exp(a_t g^c_w[t]),  a_t = −softplus(w·v_t)·exp(Δ),  w = 0 and Δ = ln(−ln 0.95 / ln 2) ≈ −2.6037 at
           init (g = 1 gives 0.95 exactly; g = 0 gives 1: an unwritten channel does not decay).
  GDN      α^c_t = exp(−A_c softplus(w_a^c·v_t + b_Δ^c)),  A_c ~ U(0, 16) (A_log learnable), b_Δ^c = softplus⁻¹(dt_c),
           dt_c log-uniform in [0.001, 0.1] (floor 1e-4), w_a = 0; drawn from a generator seeded seed + 59000; no weight
           decay on A_log, b_Δ (no parameter has any).
v_t is the layer's write vector V; the decay modules' parameters train at the same lr as everything else.
ARMS (seeds 350-359, paired: the same batches; the same initial parameters except the decay module's), oracle first:
  ORACLE_{FIXED,ROUTED,GDN}   the perfect gate, k = 2 (test_channel_binding's 'ceiling' arm).
  SINGLE_{FIXED,ROUTED,GDN}   one channel, k = 1 (test_short_conv's arm B).
  D_DELTA_{ROUTED,GDN}        (conditional, below) S54(d)'s D_DELTA — LOCAL3 + SLOW on RandHeaderTask, learned β — with
                              that decay, seeds 330-339, 24000 updates; descriptive against S54(d)'s D_DELTA on the same
                              seeds.

VALIDITY (fixed before any run): a decay's arms are VALID if its ORACLE binds on >= 2 of seeds 350-352 within 48000;
the reading for ROUTED or GDN needs that decay and FIXED both VALID, else it is UNTESTED for that decay.

READING (fixed before any run): per seed and decay d, the oracle gap gap_d = (final held-out accuracy of ORACLE_d) −
(final held-out accuracy of SINGLE_d) (rec["acc"], the last evaluation). "the horizon knob matters" if for ROUTED or for
GDN, gap_d − gap_FIXED >= 0.15 on >= 7 of the 10 seeds. Otherwise "it does not". If it matters, the winning decay is the
one with more such seeds (a tie: the larger median of gap_d − gap_FIXED), and D_DELTA_<winner> runs on seeds 330-339
(descriptive: DISCOVERED, BOUND, ROUTED*@end and failure classes beside S54(d)'s D_DELTA on the same seeds). If it does
not matter, the rerun is not made.
DIAGNOSTICS: per arm the BOUND count (Wilson, band), transitions, final accuracy; the learned decay at the end (mean α per
layer at CTX / KEY / VAL; for the oracle on the token's own stream's channel and on the other channel); the decay
parameters (ROUTED: w norm, Δ; GDN: A, dt); SINGLE's write-order accuracy (does it keep the last-written stream?).
CHECKS (child.checks + explore_f_delta_checks every segment): every arm's model as stated (k, decay kind, β, L2 keys, no
conv); the oracle arms share every initial parameter except the decay's; at init ROUTED differs from FIXED only through
the unwritten channel's decay (logits differ); D_DELTA_* put the memory's and the decay's parameters in the rest group
with lrs [1e-3, 1e-4] at update 1 and [1e-3, 1e-3] at 2401.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common16 as c16
import explore_f_common as fc

NAME = fc.name("decay")
CHILD = "explore_f_decay_child"
MAIN = False
LR = 1e-3
ITERS, D_ITERS = 48000, 24000
SEEDS, D_SEEDS = tuple(range(350, 360)), tuple(range(330, 340))
VALID_SEEDS = (350, 351, 352)
GAP, N_SEEDS = 0.15, 7
DECAYS = ("FIXED", "ROUTED", "GDN")
ARMS = {}
for _k, _p in (("ORACLE", 2), ("SINGLE", 1)):
    for _d in DECAYS:
        ARMS[f"{_k}_{_d}"] = dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, prio=_p,
                                  label=f"{_k.lower()} on DELTA (beta 1, L2 keys), decay {_d}, header P=8",
                                  sched="Adam 1e-3 throughout")
for _d in ("ROUTED", "GDN"):
    ARMS[f"D_DELTA_{_d}"] = dict(opt="adam", lr=LR, iters=D_ITERS, seeds=D_SEEDS, prio=0, conditional=True,
                                 label=f"S54(d)'s D_DELTA + decay {_d} (run only if the reading names it)",
                                 sched="gate 1e-3 throughout; rest 1e-4 for 1-2400, 1e-3 after")


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ARMS[arm]["iters"]))


def timing(arm, steps):
    return child("timing", dict(arm=arm, steps=steps))


def check():
    ok = True
    for nm, v in child("checks", {}):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def line(arm, r):
    if not r.get("ok"):
        return f"FAILED {r.get('error')}"
    return f"acc {r['acc']:.3f}  trans {str(r['transition']):>5}  {r.get('tag') or ('BOUND' if r['transition'] is not None else 'unbound')}"


def report():
    import explore_f_reports as fr
    return fr.report_s59()


if __name__ == "__main__":
    import explore_f_decay as me
    fc.drive("s59", [me])
