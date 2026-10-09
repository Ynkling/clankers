#!/usr/bin/env python
"""
explore_f_hebb_decay.py — EXPLORATORY, not a result. Session F follow-up, screen S69: the forget gate on a single HEBBIAN
channel. S59 found that a learned decay (routed or GDN) lets a single DELTA channel bind the header layout at P = 8
(9/10, 8/10 against 3/10 fixed), by forgetting at CTX tokens. Is that the forget gate alone, or the forget gate with the
delta rule?

CONFIGURATION (explore_f_followup_child): explore_far_cue.HeaderTask(8, S=2, n_vals=16, n_q=1) (L = 37), no conv; one
channel (test_short_conv's arm B, k = 1); Adam 1e-3 on every parameter throughout; 48000 updates (S59's); evaluation every
1200, early stop after 3 >= 0.95. The memory is the HEBBIAN one with a decay module: explore_delta_mem.to_delta(beta=0.0,
write=1.0, normalize=False, decay=...) — no erase, write strength 1, raw keys, so the score is Σ_s A(t,s)(x_t·x_s) v_s with
A(t,s) = Π_{r=s+1..t} α_r (β plays no part: the Hebbian memory has no erase).
ARMS (seeds 350-359 — S59's, so the same batches as S59's SINGLE arms; same initial parameters as each other except the
decay's):
  S69_FIXED   α = 0.95 (CHECK: equals the unconverted Hebbian model's logits exactly).
  S69_ROUTED  α_t = exp(a_t g_t), a_t = −softplus(w·v_t) e^Δ, w = 0, Δ = ln(−ln 0.95 / ln 2) at init (k = 1: g = 1).
  S69_GDN     α_t = exp(−A softplus(w_a·v_t + b_Δ)), A ~ U(0, 16), b_Δ = softplus⁻¹(dt), dt log-uniform [0.001, 0.1],
              w_a = 0; generator seed + 59000 (S59's draw for the seed).
  S69_ORACLE  the perfect gate, k = 2, the same Hebbian memory with α = 0.95; seeds 350-351 (validity).
OUTCOME: BOUND. Wilson 95% and bands with every count.
VALIDITY (fixed before any run): S69_ORACLE binds on >= 1 of its 2 seeds within 48000, else UNTESTED.
READINGS (fixed before any run):
  "the forget gate needs the delta rule"  if ROUTED <= FIXED + 1 and GDN <= FIXED + 1 (BOUND counts);
  "the forget gate alone suffices"        if ROUTED >= 8/10 or GDN >= 8/10;
  otherwise "neither". Printed beside S59's delta counts (FIXED 3/10, ROUTED 9/10, GDN 8/10) on the same seeds.
DIAGNOSTICS: the learned α at CTX / KEY / VAL per layer at the end (median over runs), as in S59; the decay parameters;
transitions; final accuracy; write-order accuracy.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore_common16 as c16
import explore_f_common as fc

NAME = fc.name("hebb_decay")
CHILD = "explore_f_followup_child"
MAIN = False
LR, ITERS = 1e-3, 48000
SEEDS = tuple(range(350, 360))
S59_DELTA = dict(FIXED=3, ROUTED=9, GDN=8)
ARMS = {f"S69_{d}": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label=f"one Hebbian channel, decay {d}, header P=8",
                         sched="Adam 1e-3 throughout") for d in ("FIXED", "ROUTED", "GDN")}
ARMS["S69_ORACLE"] = dict(opt="adam", lr=LR, iters=ITERS, seeds=(350, 351), label="perfect gate, Hebbian, decay 0.95",
                          sched="Adam 1e-3 throughout")


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ARMS[arm]["iters"]))


def timing(arm, steps):
    return child("timing", dict(arm=arm, steps=steps))


def line(arm, r):
    return (f"acc {r['acc']:.3f}  trans {str(r['transition']):>5}  {'BOUND' if r['transition'] is not None else 'unbound'}"
            if r.get("ok") else f"FAILED {r.get('error')}")


def report():
    import explore_f_reports2 as fr
    return fr.report_s69()
