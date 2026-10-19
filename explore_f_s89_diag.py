#!/usr/bin/env python
"""
explore_f_s89_diag.py — EXPLORATORY, not a result. POST HOC, written after S89-F's pilots read "untested" on both pool
tasks (the Hebbian perfect gate unbound on 2800-2801 at 24000 and at 48000 with SLOW; the delta perfect gate bound at
2400-3600). It changes no reading of S89-F. It asks only which of two differences from S70's oracle stops the Hebbian
perfect gate, so that the user can decide the round: the SLOW recipe the pilots used (S70's own oracle used one Adam at
1e-3), or the key pool.
ARMS (seeds 2800-2801, 24000; child explore_f_s89_diag_child, the S89 models unchanged):
  D_HEBB_LITE_SLOW      the Hebbian perfect gate on lite, SLOW (rest 1e-4 to 2400).
  D_HEBB_POOL_ADAM      the Hebbian perfect gate on -pool, one Adam at 1e-3.
  D_HEBB_ALLPOOL_ADAM   the Hebbian perfect gate on -allpool, one Adam at 1e-3.
Descriptive: transitions or final accuracy per seed. No reading.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore_common16 as c16
import explore_f_common as fc

NAME = fc.name("s89_diag")
CHILD = "explore_f_s89_diag_child"
MAIN = False
ARMS = {a: dict(opt="adam", lr=1e-3, iters=24000, seeds=(2800, 2801), label=a, sched=s)
        for a, s in (("D_HEBB_LITE_SLOW", "SLOW"), ("D_HEBB_POOL_ADAM", "Adam 1e-3"), ("D_HEBB_ALLPOOL_ADAM", "Adam 1e-3"))}


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ARMS[arm]["iters"]))


def timing(arm, steps):
    return child("timing", dict(arm=arm, steps=steps))


def check():
    return all(v for _, v in child("checks", {}))


def line(arm, r):
    if not r.get("ok"):
        return f"FAILED {r.get('error')}"
    return f"acc {r['acc']:.3f}  trans {str(r['transition']):>5}"


def report():
    return None


if __name__ == "__main__":
    import explore_f_s89_diag as me
    fc.drive("s89diag", [me], sys.argv[1:])
