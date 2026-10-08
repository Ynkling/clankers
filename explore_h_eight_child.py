#!/usr/bin/env python
"""
explore_h_eight_child.py — EXPLORATORY, not a result. Session H's S58b code that runs INSIDE a child process on the main
line's module tree at explore_main9c.MAIN_SHA (9c5939e). Parent: explore_h_eight (readings in its docstring). S58's
devices (explore_h_collapse_child: the Gumbel gate, the Switch hook, the reset trigger, the KEY-position map; imported
read-only) on batch 16/17's eight-stream path.

CONFIGURATION: S44(b) / S48's: test_stream_curriculum.run_sc with ARM["D8"] (S=8, P=4, k=16, conv, all 8 streams from step
1), LOCAL3's gate, SLOW (Adam), no hinge, evaluation every 1200, 43200 updates. NONE here = batch 16's LOCAL3_SLOW_D8_A
recipe (CHECK only: no NONE arm runs in S58b). Arms as S58's (explore_h_collapse_child.ARM) on this configuration; SPLIT's
trigger checks every 2400 from 4800 through 40800 (= 43200 - 2400), <= 3 splits, >= 4800 apart (S48's W_SPLIT rule with the
longer horizon). ORACLE: the perfect gate ceil8_D8 (S33's config b CEIL: ceiling8k16, all 8 streams from step 1) + conv
with its own recipe (one Adam 1e-3).
"""

import contextlib

import torch

import explore_main9c as mt
import explore_split_child as s36c
import explore_b16_gates as g16
import explore_b16_main as b16
import explore_window_d8_child as s48c
import explore_muon_scale_child as s33c
import explore_h_collapse_child as hcc
import test_stream_recipe as tsr
from test_multilayer_binding import probe_batch

CFG = "b"
TOTAL = 43200
EVERY = hcc.EVERY
ARM = hcc.ARM


def build_rc(arm_name, p, holder, out, info, log):
    arm = ARM[arm_name]
    task = b16.task_of(CFG)
    rprobe = probe_batch(task, p["seed"])

    def probe(m, step):
        if p.get("probe", True):
            out[str(step)] = dict(hcc.key_routing(m, task, rprobe),
                                  **({"switch": dict(log["last"])} if log.get("last") else {}),
                                  **({"gumbel_calls": m.gumbel_calls} if hasattr(m, "gumbel_calls") else {}))
    convert = [g16.to_local]
    if arm.get("gumbel"):
        convert.append(hcc.to_gumbel(arm["gumbel"], p.get("gumbel_scale", 1.0)))
    if arm.get("two_node"):
        convert.append(hcc.to_two_node)
    if arm.get("switch") is not None:
        convert.append(hcc.attach_switch(p.get("alpha", arm["switch"]), log))
    rc, _ = b16.make_rc("adam", holder, b16.GATE_LOCAL, tau=None, convert=tuple(convert), probe=probe, keep=p.get("keep"))
    if arm.get("trigger"):
        ctx, opt_of = s36c.adam_capture(holder), (lambda: holder["opts"][0])
        last = p.get("total", TOTAL) - EVERY
        thr = p.get("thr", hcc.ACC_THR)
        if arm["trigger"] == "split":
            trig = s48c.make_trigger(task, p["seed"], info, opt_of, last, max_splits=hcc.MAX_SPLITS, thr=thr)
        else:
            trig = hcc.make_trigger_op(task, p["seed"], info, opt_of, last, hcc.reset_op, thr=thr)
        rc = s36c.with_trigger(rc, trig)
    else:
        ctx = contextlib.nullcontext()
    return rc, ctx


def run(p):
    arm = ARM[p["arm"]]
    if arm.get("oracle"):
        rec = b16.path(CFG, s33c.cfg_arm(CFG, "CEIL"), p["seed"], p["iters"], None, eval_every=p.get("eval_every"))
        return b16.finish(rec, arm=p["arm"], cfg=CFG, opt="adam", kind="oracle")
    holder, out, info, log = {}, {}, dict(checks=[]), {}
    rc, ctx = build_rc(p["arm"], p, holder, out, info, log)
    with ctx:
        rec = b16.path(CFG, b16.arm_of(CFG), p["seed"], p["iters"], rc, eval_every=p.get("eval_every"))
    if out:
        last = max(out, key=int)
        out["end"] = dict(out[last], step=int(last))
    return b16.finish(rec, arm=p["arm"], cfg=CFG, opt="adam", kind="local", keyroute=out, groups=b16.group_names(holder),
                      checks=info["checks"], splits=sum(1 for c in info["checks"] if c["fired"]),
                      switch_n=log.get("n"), gumbel=arm.get("gumbel"), switch_alpha=arm.get("switch"),
                      max_splits=hcc.MAX_SPLITS if arm.get("trigger") else None, trigger=arm.get("trigger"))


def constants(p):
    t = b16.task_of(CFG)
    return dict(main_sha=mt.MAIN_SHA, task=[t.S, t.P], TOTAL=TOTAL, MAX_SPLITS=hcc.MAX_SPLITS)
