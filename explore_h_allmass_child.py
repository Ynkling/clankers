#!/usr/bin/env python
"""
explore_h_allmass_child.py — EXPLORATORY, not a result. Session H's S86 code that runs INSIDE a child process on the main
line's module tree at explore_main9c.MAIN_SHA (9c5939e). Parent: explore_h_allmass (readings in its docstring).

CONFIGURATION: S79 / S84's (run_sc with ARM["D8"]: S=8, P=4, k=16, conv; LOCAL3 + SLOW, Adam, Hebbian memory, no hinge;
43200 updates; exact-copy split; last check 40800).
THE TRIGGER (make_trigger_target): explore_h_guard_child.make_trigger_guard line for line (guard off) with the target rule
as a parameter. Both pairs are computed at every check, as before, by explore_split_target_child.select on the read gate
over the 64-sequence probe (the same function S37-S84 and test_window_gate use): the key-position pair (c*, c0 = the
largest / smallest mean read-gate mass at the probe's KEY positions 3j+1) and the all-position pair (the same over every
position of the probe's inputs). target "key" copies onto the key-position pair (KEYMASS), "all" onto the all-position
pair. Both labels are recorded at every check (key_ok, all_ok: c* holds >= 2 streams and c0 none on the labelled map before
the operation).
  BASE     S79's BASE: target key, the probe as source.
  ALLMASS  target all, the probe as source.
  EVAL     S84's EVAL: target key, the 2048-query held-out evaluation as the rule's source.
  ORACLE   perfect gate ceil8_D8 + conv.
"""

import torch

import explore_split_child as s36c
import explore_split_target_child as s37c
import explore_b16_gates as g16
import explore_b16_main as b16
import explore_h_guard_child as h84c
import explore_h_eight_child as h8c
import explore_h_ops as ops
import explore_h_fail_class_v2 as v2
import explore_window_d8_child as s48c
import test_binding_onset as tbo
import test_scale_axes as tsa
import test_stream_recipe as tsr
from test_channel_binding import eval_batch
from test_multilayer_binding import probe_batch

CFG = "b"
TOTAL = 43200
BASE = dict(h84c.BASE, target="key")
ARM = {"BASE": dict(), "ALLMASS": dict(target="all"), "EVAL": dict(source="eval"), "ORACLE": dict(oracle=True)}


def make_trigger_target(task, seed, info, opt_of, last, op, every, ref_at, first, gap, cap, probe_n, rise, thr, guard, source,
                        target):
    data = s36c.probe_data(task)[:probe_n]
    edata = eval_batch(task) if source == "eval" else None
    rprobe = probe_batch(task, seed)
    st = dict(prev=None, n=0, last=None)

    def trig(m, step):
        if step % every or step < ref_at or step > last:
            return
        acc, _ = tbo.evaluate(m, task, edata if source == "eval" else data)
        (kc, k0, km), (ac, a0, am) = s37c.select(s37c.read_gate(m, data), task.S, task.P)
        cmap = tsa.routing_k(m, task, rprobe)["ch_map"]
        row = dict(step=step, acc=acc, source=source, target=target, prev=st["prev"], map_before=cmap,
                   key_masses=[round(float(v), 5) for v in km], all_masses=[round(float(v), 5) for v in am],
                   key_cs=kc, key_c0=k0, all_cs=ac, all_c0=a0,
                   key_ok=s37c.targeting(cmap, kc, k0), all_ok=s37c.targeting(cmap, ac, a0), fired=False, blocked=False)
        eligible = step >= first and st["prev"] is not None and acc < thr and acc - st["prev"] < rise
        guarded = guard is not None and acc < guard
        capped = st["n"] >= cap or (st["last"] is not None and step - st["last"] < gap)
        row["eligible"] = eligible
        if eligible and guarded:
            row["blocked"] = True
            row["blocked_by"] = "guard"
        elif eligible and capped:
            row["blocked"] = True
            row["blocked_by"] = "cap" if st["n"] >= cap else "gap"
        elif eligible:
            cs, c0 = (kc, k0) if target == "key" else (ac, a0)
            res = op(m, opt_of(), cs, c0, s36c.NOISE_BASE + 1000 * seed + st["n"])
            st["n"] += 1
            st["last"] = step
            row.update(fired=True, op_cs=cs, op_c0=c0, op_ok=s37c.targeting(cmap, cs, c0),
                       map_after=tsa.routing_k(m, task, rprobe)["ch_map"],
                       key_masses_after=[round(float(v), 5) for v in s37c.select(s37c.read_gate(m, data), task.S, task.P)[0][2]],
                       **res)
        st["prev"] = acc
        info["checks"].append(row)
    return trig


def run(p):
    """p: arm, seed, iters; CHECK knobs: thr, total."""
    arm = ARM[p["arm"]]
    if arm.get("oracle"):
        return h8c.run(dict(p, arm="ORACLE"))
    cfg = dict(BASE)
    cfg.update(arm)
    if p.get("thr") is not None:
        cfg["thr"] = p["thr"]
    task = b16.task_of(CFG)
    holder, info = {}, dict(checks=[])
    rc, _ = b16.make_rc("adam", holder, b16.GATE_LOCAL, tau=None, convert=(g16.to_local,), probe=None, keep=p.get("keep"))
    rc = s36c.with_trigger(rc, make_trigger_target(task, p["seed"], info, lambda: holder["opts"][0],
                                                   p.get("total", TOTAL) - 2400, ops.copy_exact, **cfg))
    with s36c.adam_capture(holder):
        rec = b16.path(CFG, b16.arm_of(CFG), p["seed"], p["iters"], rc, eval_every=p.get("eval_every"))
    rec = b16.finish(rec, arm=p["arm"], cfg=CFG, opt="adam", kind="local", groups=b16.group_names(holder), trig_cfg=cfg,
                     checks=info["checks"], splits=sum(1 for c in info["checks"] if c["fired"]))
    if rec.get("ok"):
        rec["outcome_v2"] = v2.outcome_v2(rec, tsr.routed)
    return rec


def checks(p):
    """On S48's harness (threshold above 1): target key equals S84's make_trigger_guard row for row; target all copies onto
    the all-position pair."""
    rows = []
    a = b16.arm_of(CFG)
    task = b16.task_of(CFG)
    seed = 2330
    keys = ("step", "acc", "prev", "map_before", "key_masses", "all_masses", "key_cs", "key_c0", "all_cs", "all_c0", "key_ok",
            "all_ok", "fired", "blocked", "blocked_by", "eligible", "map_after", "key_masses_after", "row_dist_before",
            "row_dist_after")
    steps = list(range(1200, 40801, 1200))
    c0 = {k: v for k, v in BASE.items() if k != "target"}
    mine, m1, _, _ = s48c._trigger_rows(a, task, seed, lambda info, oo: make_trigger_target(
        task, seed, info, oo, 40800, ops.copy_exact, **dict(c0, thr=1.01, rise=1.0, target="key")), steps)
    theirs, m2, _, _ = s48c._trigger_rows(a, task, seed, lambda info, oo: h84c.make_trigger_guard(
        task, seed, info, oo, 40800, ops.copy_exact, **dict(c0, thr=1.01, rise=1.0)), steps)
    eq = len(mine) == len(theirs) and all({k: r.get(k) for k in keys} == {k: q.get(k) for k in keys} for r, q in zip(mine, theirs)) \
        and torch.equal(m1.W_g.detach(), m2.W_g.detach())
    rows.append((f"target key equals S84's make_trigger_guard (guard off, probe) row for row (fired at "
                 f"{[r['step'] for r in mine if r['fired']]})", eq))
    al, m3, _, _ = s48c._trigger_rows(a, task, seed, lambda info, oo: make_trigger_target(
        task, seed, info, oo, 40800, ops.copy_exact, **dict(c0, thr=1.01, rise=1.0, target="all")), steps)
    f = [r for r in al if r["fired"]]
    ok = all(r["op_cs"] == r["all_cs"] and r["op_c0"] == r["all_c0"] for r in f) and [r["step"] for r in f] == [4800, 9600, 14400]
    rows.append((f"target all copies onto the all-position pair at every firing ({[(r['step'], r['all_cs'], r['all_c0'], r['key_cs'], r['key_c0']) for r in f]}: "
                 f"(update, all c*, all c0, key c*, key c0))", ok))
    return [[n, bool(v)] for n, v in rows]
