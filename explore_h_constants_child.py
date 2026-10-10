#!/usr/bin/env python
"""
explore_h_constants_child.py — EXPLORATORY, not a result. Session H's S79 code that runs INSIDE a child process on the main
line's module tree at explore_main9c.MAIN_SHA (9c5939e). Parent: explore_h_constants (readings in its docstring).

CONFIGURATION: test_window_gate's Part C / D path = S58b / S61's: run_sc with ARM["D8"] (S=8, P=4, k=16, conv, all 8
streams from step 1), LOCAL3 gate, SLOW (Adam), no hinge, 43200 updates, evaluation every 1200.
THE TRIGGER (make_trigger_cfg): explore_h_collapse_child.make_trigger_op (S48's make_trigger line for line, the operation as
a parameter) with its constants as parameters: every (check interval), ref_at (the first, reference-only check), first
(the first check that may fire), gap, cap, the probe's size (the first probe_n of S36's 64-sequence probe,
explore_split_child.probe_data(task)[:probe_n]), rise and thr; the last check at 43200 - 2400 = 40800 in every arm. The operation: the
exact copy W_g[c0] := W_g[c*] (explore_h_ops.copy_exact; no noise, no Adam reset). Labelled map, targets, rows: S48's.
  BASE     every 2400, ref_at 2400, first 4800, gap 4800, cap 3, probe 64, rise 0.02, thr 0.95.
  INT1200  every 1200, ref_at 1200, first 2400, gap 2400.
  CAP1     cap 1.
  CAP6     cap 6, gap 2400.
  PROBE16  probe 16 sequences.
  THR05    rise 0.05.
  ORACLE   perfect gate ceil8_D8 + conv (S58b's).
Every record also carries outcome_v2 (fail_class_v2's outcome, explore_h_fail_class_v2) beside test_stream_recipe.outcome.
"""

import torch

import explore_split_child as s36c
import explore_split_target_child as s37c
import explore_b16_gates as g16
import explore_b16_main as b16
import explore_h_collapse_child as hcc
import explore_h_eight_child as h8c
import explore_h_ops as ops
import explore_h_fail_class_v2 as v2
import explore_window_d8_child as s48c
import test_binding_onset as tbo
import test_scale_axes as tsa
import test_stream_recipe as tsr
from test_multilayer_binding import probe_batch

CFG = "b"
TOTAL = 43200
BASE = dict(every=2400, ref_at=2400, first=4800, gap=4800, cap=3, probe_n=64, rise=0.02, thr=0.95)
ARM = {"BASE": dict(), "INT1200": dict(every=1200, ref_at=1200, first=2400, gap=2400), "CAP1": dict(cap=1),
       "CAP6": dict(cap=6, gap=2400), "PROBE16": dict(probe_n=16), "THR05": dict(rise=0.05), "ORACLE": dict(oracle=True)}


def make_trigger_cfg(task, seed, info, opt_of, last, op, every, ref_at, first, gap, cap, probe_n, rise, thr):
    data = s36c.probe_data(task)[:probe_n]
    rprobe = probe_batch(task, seed)
    st = dict(prev=None, n=0, last=None)

    def trig(m, step):
        if step % every or step < ref_at or step > last:
            return
        acc, _ = tbo.evaluate(m, task, data)
        (kc, k0, km), (ac, a0, am) = s37c.select(s37c.read_gate(m, data), task.S, task.P)
        cmap = tsa.routing_k(m, task, rprobe)["ch_map"]
        row = dict(step=step, acc=acc, prev=st["prev"], map_before=cmap,
                   key_masses=[round(float(v), 5) for v in km], all_masses=[round(float(v), 5) for v in am],
                   key_cs=kc, key_c0=k0, all_cs=ac, all_c0=a0,
                   key_ok=s37c.targeting(cmap, kc, k0), all_ok=s37c.targeting(cmap, ac, a0), fired=False, blocked=False)
        eligible = step >= first and st["prev"] is not None and acc < thr and acc - st["prev"] < rise
        capped = st["n"] >= cap or (st["last"] is not None and step - st["last"] < gap)
        row["eligible"] = eligible
        if eligible and capped:
            row["blocked"] = True
            row["blocked_by"] = "cap" if st["n"] >= cap else "gap"
        elif eligible:
            res = op(m, opt_of(), kc, k0, s36c.NOISE_BASE + 1000 * seed + st["n"])
            st["n"] += 1
            st["last"] = step
            row.update(fired=True, map_after=tsa.routing_k(m, task, rprobe)["ch_map"],
                       key_masses_after=[round(float(v), 5) for v in s37c.select(s37c.read_gate(m, data), task.S, task.P)[0][2]],
                       **res)
        st["prev"] = acc
        info["checks"].append(row)
    return trig


def run(p):
    arm = ARM[p["arm"]]
    if arm.get("oracle"):
        return h8c.run(dict(p, arm="ORACLE"))
    cfg = dict(BASE, **{k: v for k, v in arm.items()})
    if p.get("thr") is not None:
        cfg["thr"] = p["thr"]
    task = b16.task_of(CFG)
    holder, info = {}, dict(checks=[])
    rc, _ = b16.make_rc("adam", holder, b16.GATE_LOCAL, tau=None, convert=(g16.to_local,), probe=None, keep=p.get("keep"))
    rc = s36c.with_trigger(rc, make_trigger_cfg(task, p["seed"], info, lambda: holder["opts"][0], p.get("total", TOTAL) - 2400,
                                                ops.copy_exact, **cfg))
    with s36c.adam_capture(holder):
        rec = b16.path(CFG, b16.arm_of(CFG), p["seed"], p["iters"], rc, eval_every=p.get("eval_every"))
    rec = b16.finish(rec, arm=p["arm"], cfg=CFG, opt="adam", kind="local", groups=b16.group_names(holder), trig_cfg=cfg,
                     checks=info["checks"], splits=sum(1 for c in info["checks"] if c["fired"]))
    if rec.get("ok"):
        rec["outcome_v2"] = v2.outcome_v2(rec, tsr.routed)
    return rec


def checks(p):
    """make_trigger_cfg at BASE's constants equals explore_h_collapse_child.make_trigger_op row for row (the exact copy as
    the operation; S48's _trigger_rows harness, threshold above 1); each variant's firing pattern on the same harness."""
    rows = []
    a = b16.arm_of(CFG)
    task = b16.task_of(CFG)
    seed = 1320
    keys = ("step", "acc", "prev", "map_before", "key_masses", "all_masses", "key_cs", "key_c0", "all_cs", "all_c0", "key_ok",
            "all_ok", "fired", "blocked", "blocked_by", "eligible", "map_after", "key_masses_after", "row_dist_before",
            "row_dist_after", "state_zeroed")
    steps = list(range(1200, 40801, 1200))
    base = dict(BASE, thr=1.01, rise=1.0)
    mine, m1, _, _ = s48c._trigger_rows(a, task, seed, lambda info, oo: make_trigger_cfg(
        task, seed, info, oo, 40800, ops.copy_exact, **base), steps)
    theirs, m2, _, _ = s48c._trigger_rows(a, task, seed, lambda info, oo: hcc.make_trigger_op(
        task, seed, info, oo, 40800, ops.copy_exact, thr=1.01, rise=1.0), steps)
    eq = (len(mine) == len(theirs) and all({k: r.get(k) for k in keys} == {k: q.get(k) for k in keys} for r, q in zip(mine, theirs))
          and torch.equal(m1.W_g.detach(), m2.W_g.detach()))
    rows.append((f"make_trigger_cfg at BASE equals make_trigger_op row for row (fired at {[r['step'] for r in mine if r['fired']]})", eq))
    want = {"BASE": [4800, 9600, 14400], "INT1200": [2400, 4800, 7200], "CAP1": [4800], "CAP6": [4800, 7200, 9600, 12000, 14400, 16800],
            "PROBE16": [4800, 9600, 14400], "THR05": [4800, 9600, 14400]}
    for name, w in want.items():
        cfg = dict(BASE)
        cfg.update(ARM[name])
        cfg.update(thr=1.01, rise=1.0)
        r, _, _, _ = s48c._trigger_rows(a, task, seed, lambda info, oo: make_trigger_cfg(
            task, seed, info, oo, 40800, ops.copy_exact, **cfg), steps)
        fired = [x["step"] for x in r if x["fired"]]
        rows.append((f"{name}: with every check eligible it fires at {fired} (expected {w}); checks at {[x['step'] for x in r][:4]}...",
                     fired == w))
    full = s36c.probe_data(task)
    rows.append((f"the trigger's probe: probe_n 64 is S36's probe itself, 16 its first 16 rows (shapes {tuple(full.shape)}, "
                 f"{tuple(full[:16].shape)})", full.shape[0] == 64 and torch.equal(full[:64], full)))
    return [[n, bool(v)] for n, v in rows]
