#!/usr/bin/env python
"""
explore_f_configs_b.py — EXPLORATORY, not a result. Session F, screen S80 configuration (b): four streams (S = 4, P = 4,
k = 16, conv, 28800; test_window_gate's Part B path, WIN3_SLOW16 + H's exact-copy split) on DELTA* and on the Hebbian
memory, paired, seeds 2010-2019; the delta perfect gate on 2010-2011 (validity) and the Hebbian one (printed). Run code:
explore_f_configs_b_child (main-line modules at 9c5939e). THE READINGS, VALIDITY AND CHECKS ARE IN explore_f_configs'S
DOCSTRING (fixed before any run); explore_f_configs drives both configurations in one batch.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore_common16 as c16
import explore_f_common as fc

NAME = fc.name("configs_b")
CHILD = "explore_f_configs_b_child"
MAIN = True
LR, ITERS = 1e-3, 28800
SEEDS, OSEEDS = tuple(range(2010, 2020)), (2010, 2011)
REF = ("WIN3_SLOW16", 38, 40)
_sl = "gate 1e-3 throughout; rest 1e-4 for 1-2400, 1e-3 after; KEYMASS trigger, exact row copy"
ARMS = {
    "HEBB_B": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="(b) WIN3_SLOW16 + copy split, Hebbian", sched=_sl),
    "DELTA_B": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="(b) WIN3_SLOW16 + copy split, DELTA*", sched=_sl),
    "O_DELTA_B": dict(opt="adam", lr=LR, iters=ITERS, seeds=OSEEDS, label="(b) perfect gate, DELTA*", sched="Adam 1e-3"),
    "O_HEBB_B": dict(opt="adam", lr=LR, iters=ITERS, seeds=OSEEDS, label="(b) perfect gate, Hebbian", sched="Adam 1e-3"),
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
            f"{c16.chmap_str(r['end'].get('ch_map'))}")


def report():
    return None                      # explore_f_configs' report covers both configurations
