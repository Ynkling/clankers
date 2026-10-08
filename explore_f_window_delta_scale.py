#!/usr/bin/env python
"""
explore_f_window_delta_scale.py — EXPLORATORY, not a result. Session F, screen S54 (b) and (c): the window-gate recipe
with the KEYMASS split (batch 17's W_SPLIT: LOCAL3 + SLOW + S37's targeted plateau split, <= 3 splits) on delta-rule
channels against the same recipe on the Hebbian memory, run here and paired, at four and eight streams. (a) and (d) are
in explore_f_window_delta; one batch runs both (driver: python explore_f_window_delta.py).

CONFIGURATION (explore_f_window_delta_scale_child; main-line modules at 9c5939e in child processes, as S44 / S48;
1 torch thread). Adam: gate W_in, W_g, window at 1e-3 throughout; every other trainable parameter (the delta memory's
attn.w_b, attn.b_b included) 1e-4 for updates 1-2400, 1e-3 after; no hinge; the KEYMASS trigger (checks every 2400 from
4800, 2400 the reference; fire if the 64-sequence probe accuracy is < 0.95 and rose < 0.02 since the previous check;
<= 3 splits, >= 4800 apart; last check at the run's length - 2400). DELTA = explore_delta_mem.to_delta(beta="learned")
after the window conversion (L2 keys, tied write, decay 0.95). The perfect gate (validity): one Adam at 1e-3.
ARMS
 (b) S=4, P=4, k=16, conv (run_attempt with A4k16), 28800 updates:
     B_DELTA, B_HEBB   seeds 310-329, paired (same initial parameters except w_b = 0, b_b = 0; same batches).
     B_ORACLE_DELTA, B_ORACLE_HEBB   perfect gate (ceiling4k16), seeds 310-311.
 (c) S=8, P=4, k=16, conv, all 8 streams from step 1 (run_sc with D8), 43200 updates:
     C_DELTA, C_HEBB   seeds 310-319, paired.
     C_ORACLE_DELTA, C_ORACLE_HEBB   perfect gate (ceil8_D8), seeds 310-311.
OUTCOME: BOUND ROUTED (test_stream_recipe.outcome: bound, and routed — every stream on its own channel(s) one-to-one at
VAL positions with per-stream accuracy, the main line's definition); other outcomes as that function prints them
(BOUND NOT routed, MERGED (n share), STREAM-PARTIAL one-to-one, non-stream KEY / POSITION / OTHER). (c) also the
transition (first evaluation >= 0.95 held to the end) of every BOUND ROUTED run. Wilson 95% and bands with every count.

VALIDITY (fixed before any run): per configuration and memory, VALID if the perfect gate binds on at least 1 of its 2
seeds within the configuration's budget; a reading needs both memories VALID, else UNTESTED.

READINGS (fixed before any run), per configuration, paired over its seeds, b = DELTA only, c = HEBB only (BOUND ROUTED):
   "delta not worse"  if c − b <= 2  (DELTA >= HEBB − 2; a bound, no p-value);
   "delta better"     if b >= 4 and c = 0;
   otherwise "delta worse" (c − b > 2). One-sided exact McNemar printed both ways.
DIAGNOSTICS: splits fired and their targets; the stream -> channel maps at the end; S38's decoder of the gate state at
KEY and VAL positions at 1200 / 2400 / 4800 / 9600 / end; learned β at CTX / KEY / VAL per layer at the end.
CHECKS (child.checks + explore_f_delta_checks every segment): (c) C_HEBB on seed 260 reproduces batch 17's W_SPLIT|260
(2.10GHz) through 2400 bit for bit (curve, statistics, decoder), and C_DELTA with the memory's path disabled (impl
"hebb") reproduces it too; (b) the same for B_HEBB / B_DELTA against batch 16's LOCAL3_SLOW16_A|242 (2.10GHz; no trigger
there, and the trigger's reference check at 2400 changes no training); the DELTA models' groups (gate W_in, W_g, window;
the memory's parameters in the rest group; every trainable parameter once); the perfect gate on the delta memory keeps
its one-hot routing.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common16 as c16
import explore_f_common as fc

NAME = fc.name("window_delta_scale")
CHILD = "explore_f_window_delta_scale_child"
MAIN = True
LR = 1e-3
B_ITERS, C_ITERS = 28800, 43200
B_SEEDS, C_SEEDS = tuple(range(310, 330)), tuple(range(310, 320))
NOT_WORSE_D, BETTER_B = 2, 4
_sl = ("gate W_in/W_g/window 1e-3 throughout; rest (incl. attn.w_b, attn.b_b) 1e-4 for 1-2400, 1e-3 after; KEYMASS "
       "trigger <= 3 splits")
ARMS = {
    "B_DELTA": dict(opt="adam", lr=LR, iters=B_ITERS, seeds=B_SEEDS, label="(b) S=4 k=16 W_SPLIT recipe on DELTA", sched=_sl),
    "B_HEBB": dict(opt="adam", lr=LR, iters=B_ITERS, seeds=B_SEEDS, label="(b) S=4 k=16 W_SPLIT recipe on HEBB", sched=_sl),
    "C_DELTA": dict(opt="adam", lr=LR, iters=C_ITERS, seeds=C_SEEDS, label="(c) S=8 k=16 W_SPLIT recipe on DELTA", sched=_sl),
    "C_HEBB": dict(opt="adam", lr=LR, iters=C_ITERS, seeds=C_SEEDS, label="(c) S=8 k=16 W_SPLIT recipe on HEBB", sched=_sl),
    "B_ORACLE_DELTA": dict(opt="adam", lr=LR, iters=B_ITERS, seeds=(310, 311), label="(b) perfect gate, DELTA",
                           sched="Adam 1e-3 throughout"),
    "B_ORACLE_HEBB": dict(opt="adam", lr=LR, iters=B_ITERS, seeds=(310, 311), label="(b) perfect gate, HEBB",
                          sched="Adam 1e-3 throughout"),
    "C_ORACLE_DELTA": dict(opt="adam", lr=LR, iters=C_ITERS, seeds=(310, 311), label="(c) perfect gate, DELTA",
                           sched="Adam 1e-3 throughout"),
    "C_ORACLE_HEBB": dict(opt="adam", lr=LR, iters=C_ITERS, seeds=(310, 311), label="(c) perfect gate, HEBB",
                          sched="Adam 1e-3 throughout"),
}


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
    return (f"acc {r['acc']:.3f}  trans {str(r['transition']):>5}  {r.get('outcome')}  splits {r.get('splits')}  map "
            f"{c16.chmap_str((r.get('maps') or {}).get('end'))}")


def report():
    import explore_f_reports as fr
    return fr.report_s54bc()
