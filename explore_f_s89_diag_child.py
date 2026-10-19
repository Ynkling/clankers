#!/usr/bin/env python
"""
explore_f_s89_diag_child.py — EXPLORATORY, not a result. Run code of Session F's POST-HOC diagnostic after S89-F's pilots
read "untested" on both pool tasks (parent: explore_f_s89_diag). Runs INSIDE a child process on this branch's own modules,
1 torch thread. It reuses explore_f_s89_tasks_child's models unchanged; only the recipe differs by arm:
  D_HEBB_LITE_SLOW     the Hebbian perfect gate on RandHeaderTask-lite with the pilots' SLOW recipe (does SLOW alone stop
                       the Hebbian perfect gate on lite?).
  D_HEBB_POOL_ADAM, D_HEBB_ALLPOOL_ADAM   the Hebbian perfect gate on the pool tasks with S70's own oracle recipe: one
                       Adam at 1e-3 on every parameter (does the pool, or SLOW, stop it?).
24000 updates, evaluation every 1200, early stop after 3 evaluations >= 0.95.
"""

import json

import explore_f_s89_tasks_child as s89
import test_binding_onset as tbo
from test_router_reliability import run_one

ARMS = {"D_HEBB_LITE_SLOW": ("LITE", True), "D_HEBB_POOL_ADAM": ("POOL", False), "D_HEBB_ALLPOOL_ADAM": ("ALLPOOL", False)}


def run(p):
    task_name, slow = ARMS[p["arm"]]
    sp = dict(kind="ORACLE", task=task_name, memory="HEBB")
    holder, keep = {}, {"at": -1}
    kw = s89.run_kw(sp, holder, p["seed"])
    if not slow:
        kw.pop("run_kw", None)
        kw.pop("check", None)
    rec = run_one(s89.arm_def(sp), p["seed"], p["iters"], p.get("eval_every", tbo.EVAL_EVERY), keep=keep, **kw)
    if rec.get("ok"):
        rec.update(arm=p["arm"], task=task_name, recipe="SLOW" if slow else "Adam 1e-3", group_names=holder.get("names"))
    return json.loads(json.dumps(rec, default=float))


def checks(p):
    return [["the diagnostic reuses explore_f_s89_tasks_child's models (its CHECKs passed at the S89 stages)", True]]


def timing(p):
    return s89.timing(dict(p, arm="O_HEBB_POOL"))
