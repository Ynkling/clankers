#!/usr/bin/env python
"""
explore_f_premise.py — EXPLORATORY, not a result. Session F follow-up, screen S68: the premise test. S53 found that a
single delta channel binds the two-stream header layout by learning stream-tagged keys. Does that scale with the number
of streams, or is the partition's advantage in the number of streams?

CONFIGURATION (explore_f_followup_child; this branch's modules; 1 torch thread): the grouped layout, P = 4, n_vals = 16,
n_q = 1, at S = 4 (test_channel_binding.task_for(4, 4), L = 51) and S = 8 (task_for(4, 8), L = 99) — the main line's TASK44 /
TASK48, the tasks of S54(b) / (c); conv 'layer' K = 4; Adam 1e-3 on every parameter throughout; 43200 updates; evaluation
every 1200 on 2048 held-out sequences, early stop after 3 evaluations >= 0.95.
ARMS
  S68_SINGLE_S4, S68_SINGLE_S8   seeds 300-309: one channel (test_short_conv's B_conv: k = 1, gate 'none', conv) on the
                 delta memory with β = 1 FIXED, L2 keys, tied write, decay 0.95 (explore_f_delta_fast).
  S68_ORACLE_S4, S68_ORACLE_S8   seeds 300-301 (validity): the perfect gate, k = S, conv, the same delta memory and recipe.
REFERENCES (recorded, counts only, other seeds): S54(b) B_HEBB (window gate + SLOW + KEYMASS, Hebbian, k = 16, S = 4,
seeds 310-329) BOUND ROUTED 19/20 (BOUND 20/20); S54(c) C_HEBB (S = 8, seeds 310-319) BOUND ROUTED 9/10; the recorded
oracles B_/C_ORACLE_HEBB 2/2 each.
OUTCOME: BOUND (a transition held to the end; k = 1 has no routing). Wilson 95% and bands with every count.
VALIDITY (fixed before any run): S = s is VALID if S68_ORACLE_S{s} binds on >= 1 of its 2 seeds within 43200; a reading
needs both S = 4 and S = 8 VALID, else UNTESTED.
READINGS (fixed before any run):
  "context-tagged keys scale"                          if S68_SINGLE_S8 BOUND >= 7/10;
  "the partition's advantage is in the number of streams"  if S68_SINGLE_S8 <= 3/10 while S68_SINGLE_S4 >= 7/10;
  otherwise "neither".
DIAGNOSTICS: transitions; final accuracy; write-order accuracy (last-written vs other pairs); β (fixed 1, printed).
CHECKS (child.checks, and explore_f_delta_checks every start): the models as stated; the oracle's gate one-hot on each
position's stream; L as stated.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore_common16 as c16
import explore_f_common as fc

NAME = fc.name("premise")
CHILD = "explore_f_followup_child"
MAIN = False
LR, ITERS = 1e-3, 43200
SEEDS, OSEEDS = tuple(range(300, 310)), (300, 301)
REC = dict(S4=("B_HEBB", 19, 20, 20), S8=("C_HEBB", 9, 10, 9))      # recorded BOUND ROUTED, n, BOUND
ARMS = {f"S68_{k}_S{s}": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS if k == "SINGLE" else OSEEDS,
                              label=f"{'one channel' if k == 'SINGLE' else 'perfect gate'} on delta (beta 1), S={s}, grouped, conv",
                              sched="Adam 1e-3 throughout")
        for s in (4, 8) for k in ("SINGLE", "ORACLE")}


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
    return (f"acc {r['acc']:.3f}  trans {str(r['transition']):>5}  {'BOUND' if r['transition'] is not None else 'unbound'}"
            + (f"  {r.get('tag')}" if r.get('tag') else "")) if r.get("ok") else f"FAILED {r.get('error')}"


def report():
    import explore_f_reports2 as fr
    return fr.report_s68()


if __name__ == "__main__":
    import explore_f_premise as a, explore_f_hebb_decay as b, explore_f_rh_lite as c
    fc.drive("followup", [a, b, c])
