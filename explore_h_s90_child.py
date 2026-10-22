#!/usr/bin/env python
"""
explore_h_s90_child.py — EXPLORATORY, not a result. Session H's S90 code that runs INSIDE a child process on the main
line's module tree at explore_main9c.MAIN_SHA (9c5939e), as S89's and S89b's. Parent: explore_h_s90 (readings in its
docstring).

THE PATH: S89's (explore_h_s89_child: PoolTask with the pool size K a parameter, make_trigger_s89, pool_rc; LOCAL3 + SLOW,
Adam, k = 16, conv; S84's trigger constants; the exact copy) and S89b's delta memory (explore_h_s89b_child.with_delta:
test_delta_recipe's to_delta_fast, beta 1, L2 keys, served from the main line at a581027), imported unchanged. The budget
(iters) and the trigger's last check (total - 2400) follow the payload.
  ORACLE_HEBB    the perfect gate ceil8_D8 + conv, one Adam 1e-3 (S89's ORACLE).
  ORACLE_DELTA   the same converted to delta (S89b's ORACLE_DELTA).
  BASE           Hebbian, LOCAL3 + SLOW + the key-position trigger with the exact copy (S89's BASE).
  DELTA_SPLIT    delta, the same recipe and trigger (S89b's DELTA_SPLIT; test_delta_recipe's DELTA_SPLIT on the pool task).
  DELTA_NOSPLIT  delta, no trigger (test_delta_recipe's DELTA_SLOW on the pool task).
"""

import ast
import inspect
import textwrap

import explore_split_child as s36c
import explore_b16_gates as g16
import explore_b16_main as b16
import explore_muon_scale_child as s33c
import explore_h_s89_child as h89c
import explore_h_s89b_child as h89b
import explore_main9c as mt
import explore_h_ops as ops
import explore_h_fail_class_v2 as v2
import test_stream_recipe as tsr

CFG = h89c.CFG
ARM = {"ORACLE_HEBB": dict(oracle=True, delta=False), "ORACLE_DELTA": dict(oracle=True, delta=True),
       "BASE": dict(delta=False), "DELTA_SPLIT": dict(delta=True), "DELTA_NOSPLIT": dict(delta=True, nosplit=True)}


def run(p):
    """p: arm, seed, iters, K; CHECK knobs: thr, total (default iters)."""
    arm = ARM[p["arm"]]
    K = p["K"]
    total = p.get("total", p["iters"])
    task = h89c.PoolTask(K)
    extra = dict(pool=K, budget=total, memory="delta" if arm["delta"] else "hebb")
    if arm["delta"]:
        extra["delta_cfg"] = dict(beta=h89b.BETA, write="tied", normalize=True, decay="fixed", impl="fast",
                                  served_at=h89b.DELTA_SHA[:7])
    if arm.get("oracle"):
        a = s33c.cfg_arm(CFG, "CEIL")
        rc = h89b.with_delta(None) if arm["delta"] else None
        rec = b16.path(CFG, a, p["seed"], p["iters"], h89c.pool_rc(task, a, rc), eval_every=p.get("eval_every"))
        return b16.finish(rec, arm=p["arm"], cfg=CFG, opt="adam", kind="oracle", **extra)
    a = b16.arm_of(CFG)
    cfg = dict(h89c.BASE, target="key")
    if p.get("thr") is not None:
        cfg["thr"] = p["thr"]
    holder, info = {}, dict(checks=[])
    rc, _ = b16.make_rc("adam", holder, b16.GATE_LOCAL, tau=None, convert=(g16.to_local,), probe=None, keep=p.get("keep"))
    if arm["delta"]:
        rc = h89b.with_delta(rc)
    trig = not arm.get("nosplit")
    if trig:
        rc = s36c.with_trigger(rc, h89c.make_trigger_s89(task, p["seed"], info, lambda: holder["opts"][0], total - 2400,
                                                         ops.copy_exact, **cfg))
    with s36c.adam_capture(holder):
        rec = b16.path(CFG, a, p["seed"], p["iters"], h89c.pool_rc(task, a, rc), eval_every=p.get("eval_every"))
    rec = b16.finish(rec, arm=p["arm"], cfg=CFG, opt="adam", kind="local", groups=b16.group_names(holder),
                     trig_cfg=dict(cfg, last=total - 2400) if trig else None, checks=info["checks"],
                     splits=sum(1 for c in info["checks"] if c["fired"]), **extra)
    if rec.get("ok"):
        rec["outcome_v2"] = v2.outcome_v2(rec, tsr.routed)
    return rec


def _fn(src, name):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(name)


def checks(p):
    """The delta recipe's code against test_delta_recipe's by syntax tree; the pool task's structure at 8 and 16."""
    rows = []
    tdr = mt.main_blob("test_delta_recipe.py", h89b.DELTA_SHA)
    mine = _fn(textwrap.dedent(inspect.getsource(h89b.with_delta)), "builder")
    theirs = _fn(tdr, "with_delta")
    theirs_b = next(n for n in ast.walk(theirs) if isinstance(n, ast.FunctionDef) and n.name == "builder")
    same_builder = ast.dump(mine) == ast.dump(theirs_b)
    same_copy = [ast.dump(s) for s in _fn(textwrap.dedent(inspect.getsource(ops.copy_exact)), "copy_exact").body] == \
        [ast.dump(s) for s in _fn(tdr, "copy_op").body]
    beta = next(n.value.value for n in ast.parse(tdr).body if isinstance(n, ast.Assign) and
                any(getattr(t, "id", None) == "BETA" for t in n.targets))
    rows.append((f"the delta recipe's code is test_delta_recipe's (at {h89b.DELTA_SHA[:7]}) by syntax tree: the model converter "
                 f"(with_delta's builder: to_delta_fast(mk(), seed=seed, beta=BETA)) {same_builder}; the exact copy (copy_exact "
                 f"vs copy_op, bodies) {same_copy}; BETA {beta} = {h89b.BETA}", same_builder and same_copy and beta == h89b.BETA))
    for K in (8, 16):
        d = h89c.describe(dict(K=K, n=4096))
        ok = (d["keys_distinct"] and d["key_groups"] and d["streams_per_group"] and d["values_distinct"] and d["answered"]
              and d["ids_in_vocab"] and d["vocab"] == 8 + K + 16)
        rows.append((f"PoolTask({K}): structure as at 32 (4 distinct keys per sequence, groups of 8, values distinct, the query "
                     f"answered, ids in [0, {d['vocab']})); key frequency per type {d['freq']['key_min']:.4f}-{d['freq']['key_max']:.4f}",
                     ok))
    return [[n, bool(v)] for n, v in rows]
