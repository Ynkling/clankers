#!/usr/bin/env python
"""
explore_f_delta_controls.py — EXPLORATORY, not a result. Session F, screen S53: delta-rule channels, controls.

QUESTION. With the Hebbian channel write replaced by the delta rule (explore_delta_mem; β = 1 and L2 keys: one value
per key direction, the second binding of a key erases the first), does a single channel lose the partial credit it gets
by superposition — so that it fails on both layouts while the perfect (oracle) partition still binds — and does the
delta memory raise the oracle's ceiling at P = 8, where the Hebbian oracle did not bind within 24000 (S28)?
Background: docs/reading/distant_cues_2026-10.md §4.1 (prediction: single-channel BOUND on the header layout falls
from 3/10 to 0/10 at β ≈ 1; the oracle at P = 8 reaches budget where the Hebbian oracle did not).

CONFIGURATION (this branch's modules, child processes, 1 torch thread; test_router_reliability.run_one, i.e.
test_binding_onset.onset_run): Adam lr 1e-3 (test_channel_binding.SUB_LR) on every parameter throughout (no slow
schedule), batch 32, evaluation every 1200 on eval_batch(task) (2048 held-out sequences), early stop after 3
evaluations >= 0.95; model MultiBDH n_layer 3, positional 'decay' (0.95), N = 256 (mult 8), no convolution.
Tasks: P = 4, S = 2, n_vals = 16, n_q = 1 in two layouts — grouped (test_short_conv.TASK = task_for(4, 2)) and header
(explore_far_cue.HEADER); and the header layout at P = 8 (HeaderTask(8, S=2, n_vals=16, n_q=1)).

ARMS (seeds 300-309 at P = 4 on each layout, 24000 updates; the P = 8 pair on seeds 300-302, 48000 updates):
  SINGLE_HEBB_{G,H}     k = 1 (test_short_conv's arm B: gate 'none', no gate parameters), the current Hebbian memory
                        (no conversion: S23's B_FAR model on the header layout).
  SINGLE_DELTA_{G,H}    k = 1, to_delta(beta=1.0, normalize=True): β = 1 fixed, L2 keys, tied write, decay 0.95.
  SINGLE_DELTA_L_{G,H}  k = 1, to_delta(beta="learned"): β_t = σ(w_b·v_t + b_b), 0.5 at init, L2 keys.
  ORACLE_HEBB_{G,H}     the perfect gate, k = S = 2 (test_channel_binding's 'ceiling' arm), Hebbian.
  ORACLE_DELTA_{G,H}    the perfect gate, k = 2, to_delta(beta=1.0, normalize=True).
  ORACLE_HEBB_P8, ORACLE_DELTA_P8   the same two on the header layout at P = 8, seeds 300-302, 48000 updates.
PAIRING: within this batch, same seed = same batches; the single arms share their initial parameters (β fixed adds
none; learned β adds w_b = 0, b_b = 0), and so do the oracle arms. McNemar one-sided on paired seeds.

OUTCOME: BOUND = a transition in the held-out curve (test_binding_onset.transition: the first evaluation >= 0.95 that
holds to the end). k = 1 cannot be DISCOVERED. Oracle runs: BOUND. Printed with every count: Wilson 95% interval and
band (RELIABLE >= 9/10, MAJORITY >= 5/10, MINORITY >= 1, NEVER 0); transitions; final accuracy.

VALIDITY (fixed before any run): a layout (P = 4) is VALID FOR THE DELTA READING if ORACLE_DELTA binds on >= 2 of seeds
300-302 there (if the delta memory cannot bind with perfect routing, a single delta channel's failure says nothing).
ORACLE_HEBB's validity on each layout (>= 2 of 300-302) is printed alongside.

READINGS (fixed before any run):
  R1 "delta makes the single channel fail" if SINGLE_DELTA BOUND <= 1/10 on both layouts AND SINGLE_HEBB BOUND >= 3/10
     on at least one layout, both layouts valid for the delta reading. If a layout is not valid: R1 is UNTESTED.
     Otherwise: neither.
  R2 "delta raises the oracle" if, at P = 8 within 48000 updates, ORACLE_DELTA binds on >= 2 of the 3 seeds on which
     ORACLE_HEBB does not bind (seeds 300-302, paired). Otherwise: neither.
  Descriptive, fixed now: SINGLE_DELTA's final held-out accuracy by write order (explore_f_tasks.write_order_acc on the
  2048 held-out sequences): over queries whose (stream, key) was the LAST of its key's two writes, and over the others;
  for the others, the fraction answered with the last-written stream's value; and by query stream. "keeps exactly the
  last-written stream" if acc_last >= 0.9, acc_notlast <= 0.1 and (last-written value among the others) >= 0.8, on a
  layout's median run; "does not" otherwise. Printed for every arm.
DIAGNOSTICS: SINGLE_DELTA_L's β at CTX / KEY / VAL positions per layer at the end (mean over the probe); the oracle
arms' routing statistics (margin, η² at KEY and VAL positions) as a check that the perfect gate routes.

CHECKS (explore_f_delta_checks, asserted, every segment; and check() here): the arms' models are what the docstring
says (k, gate kind, memory type, β mode, no convolution); SINGLE_DELTA and SINGLE_HEBB share every initial parameter;
ORACLE_DELTA and ORACLE_HEBB likewise; the P = 8 header task is HeaderTask(8) with L = 37, qpos [35].
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common16 as c16
import explore_f_common as fc

NAME = fc.name("delta_controls")
CHILD = "explore_f_delta_controls_child"
MAIN = False
LR = 1e-3
ITERS, ITERS_P8 = 24000, 48000
SEEDS, SEEDS_P8 = tuple(range(300, 310)), (300, 301, 302)
VALID_SEEDS = (300, 301, 302)
LAYOUTS = {"G": "grouped", "H": "header"}
KINDS = ("SINGLE_HEBB", "SINGLE_DELTA", "SINGLE_DELTA_L", "ORACLE_HEBB", "ORACLE_DELTA")
LABEL = dict(SINGLE_HEBB="k = 1, Hebbian (arm B)", SINGLE_DELTA="k = 1, delta beta 1, L2 keys",
             SINGLE_DELTA_L="k = 1, delta learned beta, L2 keys", ORACLE_HEBB="perfect gate k = 2, Hebbian",
             ORACLE_DELTA="perfect gate k = 2, delta beta 1, L2 keys")
ARMS = {}
for _k in KINDS:
    for _l, _ln in LAYOUTS.items():
        ARMS[f"{_k}_{_l}"] = dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, kind=_k, layout=_ln,
                                  label=f"{LABEL[_k]}, {_ln} P=4", sched=f"Adam {LR:g} throughout")
for _k in ("ORACLE_HEBB", "ORACLE_DELTA"):
    ARMS[f"{_k}_P8"] = dict(opt="adam", lr=LR, iters=ITERS_P8, seeds=SEEDS_P8, kind=_k, layout="header8",
                            label=f"{LABEL[_k]}, header P=8", sched=f"Adam {LR:g} throughout")


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    a = ARMS[arm]
    return child("run", dict(kind=a["kind"], layout=a["layout"], seed=seed, iters=a["iters"]))


def timing(arm, steps):
    a = ARMS[arm]
    return child("timing", dict(kind=a["kind"], layout=a["layout"], steps=steps))


def check():
    ok = True
    for nm, v in child("checks", {}):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def line(arm, r):
    if not r.get("ok"):
        return f"FAILED {r.get('error')}"
    wo = r.get("write_order") or {}
    return (f"acc {r['acc']:.3f}  trans {str(r['transition']):>5}  {'BOUND' if r['transition'] is not None else 'unbound'}"
            f"  last/notlast {fc.c16.fmt2(wo.get('acc_last'))}/{fc.c16.fmt2(wo.get('acc_notlast'))}")


def report():
    import explore_f_reports as fr
    return fr.report_s53()


if __name__ == "__main__":
    import explore_f_delta_controls as me                    # by its module name (the pool imports it by name)
    fc.drive("s53", [me])
