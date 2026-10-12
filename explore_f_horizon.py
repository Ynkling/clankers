#!/usr/bin/env python
"""
explore_f_horizon.py — EXPLORATORY, not a result. Session F, screen S88b: the horizon. S59's learned forget gate raised one
delta channel's binding on the P = 8 header task, but S59's reading on the horizon (the oracle gap) read "it does not".
Here the horizon is screened directly: one delta channel on RandHeaderTask-lite with the cue-to-key distance at 1x (S70's
configuration) and at 2x, the fixed decay against the learned forget gate.

CPU: this container is an Intel Xeon @ 2.80GHz; Session F's earlier screens ran on a Xeon @ 2.10GHz. Every run of
this screen is on this CPU, so pairs are bit for bit within one CPU. The 2.10GHz records reproduce bit for bit here
(checked on 12 October: S70_DELTA|330 and LOCAL3_SLOW|160 through 2400), so the reproduction CHECKs are asserted.

SEEDS: 2420-2429 at each length (the same seed numbers at 1x and at 2x); perfect gates on 2420-2421 at each length. The
block 2400-2459 appears in no record on any branch (see explore_f_write).

TASKS (explore_f_horizon_child): 1x = S70's RandHeaderTask-lite (explore_f_tasks.RandHeaderTask(P=4, nb=2, lmin=1, lmax=3):
  S = 2, 4 keys per stream in 2 blocks of 1-3 pairs, the 4 blocks in random order, one CTX per block, n_vals 16, one query;
  L = 23). 2x = the same draws with every key twice as far from its block's CTX: one FILL token after each CTX and two
  between consecutive pairs of a block ([CTX k1 v1 k2 v2] -> [CTX F k1 v1 F F k2 v2]); the j-th key's distance to its cue
  is 2(2j − 1) instead of 2j − 1 (1, 3, 5 -> 2, 6, 10); pairs stay adjacent; the same bindings, block order and query;
  L = 35; the vocabulary gains FILL (so the 2x models have one more embedding row).
MODEL AND RECIPE: one channel (test_short_conv's arm B, k = 1, built on the task's vocabulary as S59's SINGLE arms) with the
  delta memory, β = 1, L2 keys, tied write; S59's single-channel recipe (Adam 1e-3 on every parameter, evaluation every
  1200 on 2048 held-out sequences, early stop after 3 evaluations >= 0.95) for 24000 updates (S70's budget).
ARMS (paired across arms within a length: the same initial parameters except the decay's, the same batches):
  FIXED_1X, FIXED_2X      the decay fixed at 0.95.
  ROUTED_1X, ROUTED_2X    the learned forget gate: S59's routed decay (Raven's), log α_t = −softplus(w·v_t)·exp(Δ)·g_t,
                          w = 0 and α = 0.95 at init (with k = 1, g = 1). It is S59's stronger learned decay (9/10 against
                          GDN's 8/10 there), and the one that starts at the fixed decay. Gated DeltaNet's starts at random
                          rates, so a frozen GDN gate could not equal the fixed decay as the CHECK requires. It is implemented
                          relative to the fixed decay (the same function and Adam dynamics as S59's; see the child), so that
                          frozen it is FIXED bit for bit.
  O_FIXED_1X, O_FIXED_2X  the perfect gate (k = 2), the same delta memory with the fixed decay, Adam 1e-3: VALIDITY.
OUTCOME: BOUND (a transition: the first evaluation >= 0.95 that holds to the end). Wilson 95% intervals and bands with
  every count; exact one-sided McNemar both ways within each length.
VALIDITY (fixed before any run): the perfect gate binds on both of its seeds at 1x; if not, the screen is UNTESTED. At 2x
  likewise; if not, the readings that use 2x are UNTESTED.
READINGS (fixed before any run). Within a length, b = seeds where ROUTED alone binds, c = seeds where FIXED alone binds.
  Across lengths, for an arm X, drop_X = (seeds X binds at 1x and not at 2x) − (seeds X binds at 2x and not at 1x), on the
  same seed numbers.
    "untested"                          if at 1x FIXED_1X or ROUTED_1X binds on fewer than 8 of 10;
  otherwise:
    "the forget gate extends the horizon"  if at 2x ROUTED beats FIXED by >= 4 vs 0 (b >= 4, c = 0) (and at 1x both bind
                                           on >= 8/10);
    "the horizon is the decay's"           if drop_FIXED >= 4 and |drop_ROUTED − drop_FIXED| <= 1;
    "no reading applies"                   if neither holds (both are printed if both hold).
PRINTED, not readings: the held-out accuracy by the queried key's distance to its cue at every evaluation (1, 3, 5 / 2, 6,
  10); the routed decay's α at CTX / KEY / VAL / FILL per layer at the end and its parameters (w's norm, Δ); the write-order
  accuracy (did the channel keep the last-written stream?); transitions.
CHECKS (asserted before any run): explore_f_delta_checks (26) at the start of every segment; then (child):
  - one S70 reproduction: S70_DELTA|330 (explore_out/F/rh_lite_results.json, recorded on the 2.10GHz Xeon) through 2400
    bit for bit with S70's
    own run path; the 1x task gives S70's batches.
  - the 2x layout: the same pairs in the same order, every key's distance to its cue exactly doubled, the same queries,
    L 35, the FILL count, stream labels at keys, FILL not counted as a value.
  - the forget gate with its parameters frozen (ROUTED with w and δ frozen at their initial values) equals FIXED bit for bit
    through 2400 (curve and statistics) at 1x and at 2x on seed 2420.
  - the routed decay as implemented equals S59's RoutedDecay (Δ = Δ0 + δ) to 1e-12, and its decay matrix equals
    explore_delta_mem.delta_parallel's with the same log α to 1e-10 (float64, random parameters).
  - ROUTED's model is FIXED's plus w, δ (k = 1, β 1, L2 keys); the decay gets a gradient; the perfect gate is one-hot.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore_common16 as c16
import explore_f_common as fc

NAME = fc.name("horizon")
CHILD = "explore_f_horizon_child"
MAIN = False
LR, ITERS = 1e-3, 24000
SEEDS, OSEEDS = tuple(range(2420, 2430)), (2420, 2421)
LENGTHS = ("1X", "2X")
ARMS = {}
for _L in LENGTHS:
    for _d in ("FIXED", "ROUTED"):
        ARMS[f"{_d}_{_L}"] = dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, sched="Adam 1e-3 throughout",
                                  label=f"one delta channel, decay {_d.lower()}, cue-to-key distance {_L.lower()}")
    ARMS[f"O_FIXED_{_L}"] = dict(opt="adam", lr=LR, iters=ITERS, seeds=OSEEDS, sched="Adam 1e-3 throughout",
                                 label=f"perfect gate (k 2), delta, fixed decay, {_L.lower()}")


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
    bd = r.get("acc_by_distance") or {}
    last = bd[max(bd, key=int)] if bd else {}
    ds = " by distance " + " ".join(f"{d}:{v[1]:.2f}" for d, v in sorted(last.items(), key=lambda kv: int(kv[0]))) if last else ""
    return f"acc {r['acc']:.3f}  trans {str(r['transition']):>5}{ds}"


def report():
    import explore_f_reports5 as fr
    return fr.report_s88b()


if __name__ == "__main__":
    import explore_f_horizon as me
    fc.drive("s88b", [me], sys.argv[1:])
