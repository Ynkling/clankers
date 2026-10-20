#!/usr/bin/env python
"""
explore_h_s89b_child.py — EXPLORATORY, not a result. Session H's S89b code that runs INSIDE a child process on the main
line's module tree at explore_main9c.MAIN_SHA (9c5939e), as S89's. Parent: explore_h_s89b (readings in its docstring).

THE DELTA MEMORY: test_delta_recipe's (the main line), i.e. explore_delta_mem.py (F's 9a6c2bf, byte for byte) with beta 1
fixed, L2-normalised keys, tied write, decay 0.95, applied after the LOCAL3 conversion through
explore_f_delta_fast.to_delta_fast(m, seed=seed, beta=1.0). Both modules are served from the main line at DELTA_SHA
(a581027, the commit test_delta_recipe's X run used; the same blobs at the main line's head ed9ca2b) into this child, as
explore_main9c serves the test modules: no file is added to this branch, and their blob hashes enter the code SHA.
THE PATH: S89's (explore_h_s89_child, imported unchanged: PoolTask at K = 32, make_trigger_s89, pool_rc, overlap; LOCAL3
+ SLOW, Adam, k = 16, conv, 43200 updates; S84's trigger constants; the exact copy), with the model converted to delta:
  ORACLE_DELTA   perfect gate ceil8_D8 + conv (S89's ORACLE), converted to delta (test_split_copy.b_recipe(True)'s form).
  DELTA_SPLIT    LOCAL3 + SLOW, delta, the key-position trigger (KEYMASS at 3j+1) with the exact copy.
  DELTA_NOSPLIT  the same without a trigger (test_delta_recipe's DELTA_SLOW).
  DELTA_ALLMASS  the all-position trigger (S87a / S89's ALLMASS) with the exact copy.
"""

import hashlib
import os
import sys
import types

import torch

import explore_main9c as mt
import explore_split_child as s36c
import explore_b16_gates as g16
import explore_b16_main as b16
import explore_muon_scale_child as s33c
import explore_h_s89_child as h89c
import explore_h_ops as ops
import explore_h_fail_class_v2 as v2
import test_stream_recipe as tsr
import test_stream_curriculum as tscur

DELTA_SHA = "a58102733776d4ab1fdb46018177355a12a131e9"   # a581027
BETA = 1.0
CFG = h89c.CFG
TOTAL = h89c.TOTAL
ARM = {"ORACLE_DELTA": dict(oracle=True), "DELTA_SPLIT": dict(target="key"), "DELTA_NOSPLIT": dict(nosplit=True),
       "DELTA_ALLMASS": dict(target="all")}


def _serve(name):
    """The main line's module `name` at DELTA_SHA, registered in sys.modules (its __main_sha__ makes code_sha hash the
    served blob)."""
    if name in sys.modules:
        return sys.modules[name]
    src = mt.main_blob(name + ".py", DELTA_SHA)
    mod = types.ModuleType(name)
    mod.__file__ = os.path.join(mt.HERE, name + ".py")
    mod.__main_sha__ = DELTA_SHA
    sys.modules[name] = mod
    exec(compile(src, f"<{DELTA_SHA[:7]}:{name}.py>", "exec"), mod.__dict__)
    return mod


assert not os.path.exists(os.path.join(mt.HERE, "explore_delta_mem.py")), "a branch copy would shadow the served module"
dm = _serve("explore_delta_mem")
ff = _serve("explore_f_delta_fast")


def with_delta(rc):
    """The recipe's model converted to the delta memory after every other conversion (test_delta_recipe.with_delta)."""
    def rc2(kw):
        kw = kw if rc is None else rc(kw)
        b0 = kw["builder"]

        def builder(a, seed):
            mk = b0(a, seed)
            return lambda: ff.to_delta_fast(mk(), seed=seed, beta=BETA)
        return dict(kw, builder=builder)
    return rc2


def run(p):
    """p: arm, seed, iters; schedule (the trigger's first / cap, from S89c's reading; empty = S89's); K (pool size, default
    32); CHECK knobs: thr, total."""
    arm = ARM[p["arm"]]
    K = p.get("K", h89c.K_POOL)
    task = h89c.PoolTask(K)
    dcfg = dict(beta=BETA, write="tied", normalize=True, decay="fixed", impl="fast", served_at=DELTA_SHA[:7])
    if arm.get("oracle"):
        a = s33c.cfg_arm(CFG, "CEIL")
        rec = b16.path(CFG, a, p["seed"], p["iters"], h89c.pool_rc(task, a, with_delta(None)), eval_every=p.get("eval_every"))
        return b16.finish(rec, arm=p["arm"], cfg=CFG, opt="adam", kind="oracle", pool=K, memory="delta", delta_cfg=dcfg)
    a = b16.arm_of(CFG)
    cfg = dict(h89c.BASE, target=arm.get("target", "key"))
    cfg.update({k: v for k, v in (p.get("schedule") or {}).items() if k in ("first", "cap")})   # S89c's, if it read so
    if p.get("thr") is not None:
        cfg["thr"] = p["thr"]
    holder, info = {}, dict(checks=[])
    rc, _ = b16.make_rc("adam", holder, b16.GATE_LOCAL, tau=None, convert=(g16.to_local,), probe=None, keep=p.get("keep"))
    rc = with_delta(rc)
    trig = not arm.get("nosplit")
    if trig:
        rc = s36c.with_trigger(rc, h89c.make_trigger_s89(task, p["seed"], info, lambda: holder["opts"][0],
                                                         p.get("total", TOTAL) - 2400, ops.copy_exact, **cfg))
    with s36c.adam_capture(holder):
        rec = b16.path(CFG, a, p["seed"], p["iters"], h89c.pool_rc(task, a, rc), eval_every=p.get("eval_every"))
    rec = b16.finish(rec, arm=p["arm"], cfg=CFG, opt="adam", kind="local", groups=b16.group_names(holder),
                     trig_cfg=cfg if trig else None, checks=info["checks"],
                     splits=sum(1 for c in info["checks"] if c["fired"]), pool=K, memory="delta", delta_cfg=dcfg)
    if rec.get("ok"):
        rec["outcome_v2"] = v2.outcome_v2(rec, tsr.routed)
    return rec


def checks(p):
    rows = []
    blobs = {n: hashlib.sha1(mt.main_blob(n + ".py", DELTA_SHA).encode()).hexdigest() for n in ("explore_delta_mem",
                                                                                                 "explore_f_delta_fast")}
    f_blob = hashlib.sha1(mt._git("show", "9a6c2bf:explore_delta_mem.py").encode()).hexdigest()
    head = {n: hashlib.sha1(mt._git("show", f"ed9ca2b:{n}.py").encode()).hexdigest() for n in blobs}
    ok = blobs["explore_delta_mem"] == f_blob and blobs == head and sys.modules["explore_delta_mem"].__main_sha__ == DELTA_SHA
    short = {k: v[:12] for k, v in blobs.items()}
    rows.append((f"the delta modules are served from the main line at {DELTA_SHA[:7]} (blob sha1 {short}); explore_delta_mem equals F's 9a6c2bf byte for byte "
                 f"({f_blob[:12]}); both equal the main line's head ed9ca2b", ok))
    fr = ff.fast_checks()
    rows.append((f"explore_f_delta_fast.fast_checks() (F's unit checks of the fused form): {len(fr)} rows, all pass", all(bool(r[1]) for r in fr)))
    # the conversion adds no parameter and swaps the memory class
    t = h89c.PoolTask(h89c.K_POOL)
    a = b16.arm_of(CFG)
    mk = h89c.pool_rc(t, a, None)(dict(builder=None))["builder"](a, 2959)
    m0 = g16.to_local(mk(), 2959)
    names0 = [n for n, _ in m0.named_parameters()]
    m1 = ff.to_delta_fast(g16.to_local(mk(), 2959), seed=2959, beta=BETA)
    names1 = [n for n, _ in m1.named_parameters()]
    same = names0 == names1 and all(torch.equal(p0, p1) for (_, p0), (_, p1) in zip(m0.named_parameters(), m1.named_parameters()))
    rows.append((f"to_delta_fast on the pool-32 LOCAL3 model adds no parameter and changes none ({len(names1)} parameters); the "
                 f"memory becomes {type(m1.attn).__name__} (impl {m1.attn.impl}, beta {m1.delta_cfg.get('beta')})",
                 same and type(m1.attn).__name__ == "FastDeltaMemory"))
    ceil_c = dict(tscur.ARM["SC8_ceil"], key="CEIL_C", cur=False)
    mine = s33c.cfg_arm(CFG, "CEIL")
    diff = sorted(k for k in set(ceil_c) | set(mine) if ceil_c.get(k) != mine.get(k))
    rows.append((f"the perfect gate's arm (ceil8_D8) against test_window_gate's CEIL_C (SC8_ceil, curriculum off), which "
                 f"test_delta_recipe's CEIL_DELTA uses: they differ only in {diff}", set(diff) <= {"key", "label"}))
    return [[n, bool(v)] for n, v in rows]
