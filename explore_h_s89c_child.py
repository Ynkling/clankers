#!/usr/bin/env python
"""
explore_h_s89c_child.py — EXPLORATORY, not a result. Session H's S89c code that runs INSIDE a child process on the main
line's module tree at explore_main9c.MAIN_SHA (9c5939e), as S89's. Parent: explore_h_s89c (readings in its docstring).

THE PATH: S89's (explore_h_s89_child, imported unchanged: PoolTask at K = 32, make_trigger_s89, pool_rc, overlap; LOCAL3 +
SLOW, Adam, Hebbian memory, k = 16, conv, 43200 updates; the exact copy), with the trigger's constants as parameters:
  BASE        S89's BASE (target key; checks every 2400 from a 2400 reference; first eligible check 4800; cap 3; gap 4800;
              last check 40800; probe 64, rise 0.02, threshold 0.95).
  BASE_LATE   the first eligible check at 12000; every other constant BASE's.
  BASE_CAP6   the cap at 6; every other constant BASE's.
"""

import torch

import explore_split_child as s36c
import explore_b16_gates as g16
import explore_b16_main as b16
import explore_h_s89_child as h89c
import explore_h_ops as ops
import explore_h_fail_class_v2 as v2
import explore_window_d8_child as s48c
import test_stream_recipe as tsr

CFG = h89c.CFG
TOTAL = h89c.TOTAL
ARM = {"BASE": dict(), "BASE_LATE": dict(first=12000), "BASE_CAP6": dict(cap=6)}


def cfg_of(arm, thr=None):
    cfg = dict(h89c.BASE, target="key")
    cfg.update(ARM[arm])
    if thr is not None:
        cfg["thr"] = thr
    return cfg


def run(p):
    """p: arm, seed, iters; CHECK knobs: thr, total."""
    K = p.get("K", h89c.K_POOL)
    task = h89c.PoolTask(K)
    a = b16.arm_of(CFG)
    cfg = cfg_of(p["arm"], p.get("thr"))
    holder, info = {}, dict(checks=[])
    rc, _ = b16.make_rc("adam", holder, b16.GATE_LOCAL, tau=None, convert=(g16.to_local,), probe=None, keep=p.get("keep"))
    rc = s36c.with_trigger(rc, h89c.make_trigger_s89(task, p["seed"], info, lambda: holder["opts"][0],
                                                     p.get("total", TOTAL) - 2400, ops.copy_exact, **cfg))
    with s36c.adam_capture(holder):
        rec = b16.path(CFG, a, p["seed"], p["iters"], h89c.pool_rc(task, a, rc), eval_every=p.get("eval_every"))
    rec = b16.finish(rec, arm=p["arm"], cfg=CFG, opt="adam", kind="local", groups=b16.group_names(holder), trig_cfg=cfg,
                     checks=info["checks"], splits=sum(1 for c in info["checks"] if c["fired"]), pool=K)
    if rec.get("ok"):
        rec["outcome_v2"] = v2.outcome_v2(rec, tsr.routed)
    return rec


def checks(p):
    """The trigger's mechanics on S48's harness (pool 4 = the main line's task; threshold above 1, rise 1: every check from
    the first eligible one is eligible): BASE fires at 4800, 9600, 14400 and is cap-blocked after; BASE_LATE first fires at
    12000 (then 16800, 21600); BASE_CAP6 fires six times, 4800 apart (4800-28800), and is cap-blocked from 31200; at 2400
    and 4800 BASE_LATE's rows equal BASE's before BASE's first operation, and BASE_LATE does not fire at 4800."""
    rows = []
    t4 = h89c.PoolTask(4)
    a = b16.arm_of(CFG)
    seed = 2924
    steps = list(range(1200, 40801, 1200))
    got = {}
    for arm in ARM:
        cfg = dict(cfg_of(arm), thr=1.01, rise=1.0)
        r, _, _, _ = s48c._trigger_rows(a, t4, seed, lambda info, oo, cfg=cfg: h89c.make_trigger_s89(
            t4, seed, info, oo, 40800, ops.copy_exact, **cfg), steps)
        got[arm] = r
    fired = {arm: [x["step"] for x in r if x["fired"]] for arm, r in got.items()}
    capped = {arm: sorted({x["step"] for x in r if x.get("blocked_by") == "cap"}) for arm, r in got.items()}
    want = {"BASE": [4800, 9600, 14400], "BASE_LATE": [12000, 16800, 21600], "BASE_CAP6": [4800, 9600, 14400, 19200, 24000, 28800]}
    ok = fired == want and capped["BASE"][:1] == [16800] and capped["BASE_CAP6"][:1] == [31200]
    rows.append((f"the trigger's mechanics (fired {fired}; first cap-blocked check BASE {capped['BASE'][:1]}, BASE_CAP6 "
                 f"{capped['BASE_CAP6'][:1]})", ok))
    keys = ("step", "acc", "map_before", "key_masses", "all_masses", "key_cs", "key_c0", "key_ok")
    pre = [x for x in got["BASE_LATE"] if x["step"] <= 4800]
    same = all({k: x.get(k) for k in keys} == {k: y.get(k) for k in keys}
               for x, y in zip(pre, [y for y in got["BASE"] if y["step"] <= 4800])) and len(pre) == 2
    rows.append(("BASE_LATE's rows at 2400 and 4800 equal BASE's before BASE's first operation (accuracy, map, masses, pairs, "
                 "labels); BASE_LATE does not fire at 4800", same and not pre[-1]["fired"]))
    return [[n, bool(v)] for n, v in rows]
