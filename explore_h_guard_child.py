#!/usr/bin/env python
"""
explore_h_guard_child.py — EXPLORATORY, not a result. Session H's S84 code that runs INSIDE a child process on the main
line's module tree at explore_main9c.MAIN_SHA (9c5939e). Parent: explore_h_guard (readings in its docstring).

CONFIGURATION: S79's: test_window_gate's Part C / D path (run_sc with ARM["D8"]: S=8, P=4, k=16, conv), LOCAL3 gate, SLOW
(Adam), no hinge, Hebbian memory, 43200 updates, evaluation every 1200; the split = the exact copy W_g[c0] := W_g[c*]
onto the KEYMASS target (explore_h_ops.copy_exact; no noise, no Adam reset); last check 40800.
THE TRIGGER (make_trigger_guard): explore_h_constants_child.make_trigger_cfg line for line with two additions:
  source "probe" | "eval"   the accuracy the rule reads: "probe" = S36's 64-sequence probe (as before); "eval" = the run's
                            held-out evaluation (test_channel_binding.eval_batch: 2048 queries, the batch the curve is
                            computed on; recomputed at the check, the same value as the curve's). The KEYMASS target and
                            the labelled map stay on the 64-sequence probe and the run's routing probe.
  guard g (None = off)      a check whose accuracy (from the source) is below g cannot fire (logged blocked_by "guard");
                            the previous-check accuracy is updated as before.
  BASE                S79's BASE: every 2400, reference 2400, first 4800, gap 4800, cap 3, probe 64, rise 0.02, thr 0.95.
  GUARD               BASE + guard 0.5.
  GUARD_INT1200       guard 0.5, checks every 1200, reference 1200, first 2400, gap 2400 (S79's INT1200 + the guard).
  EVAL                BASE reading the held-out evaluation.
  EVAL_GUARD_INT1200  GUARD_INT1200 reading the held-out evaluation.
  ORACLE              perfect gate ceil8_D8 + conv (S58b's).
"""

import torch

import explore_split_child as s36c
import explore_split_target_child as s37c
import explore_b16_gates as g16
import explore_b16_main as b16
import explore_h_constants_child as h79c
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
BASE = dict(h79c.BASE, guard=None, source="probe")
INT = dict(every=1200, ref_at=1200, first=2400, gap=2400)
ARM = {"BASE": dict(), "GUARD": dict(guard=0.5), "GUARD_INT1200": dict(INT, guard=0.5), "EVAL": dict(source="eval"),
       "EVAL_GUARD_INT1200": dict(INT, guard=0.5, source="eval"), "ORACLE": dict(oracle=True)}


def make_trigger_guard(task, seed, info, opt_of, last, op, every, ref_at, first, gap, cap, probe_n, rise, thr, guard, source):
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
        row = dict(step=step, acc=acc, source=source, prev=st["prev"], map_before=cmap,
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
    """p: arm, seed, iters; CHECK knobs: disable (guard off and source probe), thr, total."""
    arm = ARM[p["arm"]]
    if arm.get("oracle"):
        return h8c.run(dict(p, arm="ORACLE"))
    cfg = dict(BASE)
    cfg.update(arm)
    if p.get("disable"):
        cfg.update(guard=None, source="probe")
    if p.get("thr") is not None:
        cfg["thr"] = p["thr"]
    task = b16.task_of(CFG)
    holder, info = {}, dict(checks=[])
    rc, _ = b16.make_rc("adam", holder, b16.GATE_LOCAL, tau=None, convert=(g16.to_local,), probe=None, keep=p.get("keep"))
    rc = s36c.with_trigger(rc, make_trigger_guard(task, p["seed"], info, lambda: holder["opts"][0], p.get("total", TOTAL) - 2400,
                                                  ops.copy_exact, **cfg))
    with s36c.adam_capture(holder):
        rec = b16.path(CFG, b16.arm_of(CFG), p["seed"], p["iters"], rc, eval_every=p.get("eval_every"))
    rec = b16.finish(rec, arm=p["arm"], cfg=CFG, opt="adam", kind="local", groups=b16.group_names(holder), trig_cfg=cfg,
                     checks=info["checks"], splits=sum(1 for c in info["checks"] if c["fired"]))
    if rec.get("ok"):
        rec["outcome_v2"] = v2.outcome_v2(rec, tsr.routed)
    return rec


def checks(p):
    rows = []
    a = b16.arm_of(CFG)
    task = b16.task_of(CFG)
    seed = 2300
    keys = ("step", "acc", "prev", "map_before", "key_masses", "all_masses", "key_cs", "key_c0", "all_cs", "all_c0", "key_ok",
            "all_ok", "fired", "blocked", "blocked_by", "eligible", "map_after", "key_masses_after", "row_dist_before",
            "row_dist_after", "state_zeroed")
    steps = list(range(1200, 40801, 1200))
    for name in ("BASE", "GUARD_INT1200"):
        cfg = dict(BASE)
        cfg.update(ARM[name])
        c0 = {k: v for k, v in cfg.items() if k not in ("guard", "source")}
        mine, m1, _, _ = s48c._trigger_rows(a, task, seed, lambda info, oo: make_trigger_guard(
            task, seed, info, oo, 40800, ops.copy_exact, **dict(c0, thr=1.01, rise=1.0, guard=None, source="probe")), steps)
        theirs, m2, _, _ = s48c._trigger_rows(a, task, seed, lambda info, oo: h79c.make_trigger_cfg(
            task, seed, info, oo, 40800, ops.copy_exact, **dict(c0, thr=1.01, rise=1.0)), steps)
        eq = len(mine) == len(theirs) and all({k: r.get(k) for k in keys} == {k: q.get(k) for k in keys} for r, q in zip(mine, theirs)) \
            and torch.equal(m1.W_g.detach(), m2.W_g.detach())
        rows.append((f"{name}'s schedule with the guard off and the probe as source equals S79's make_trigger_cfg row for row "
                     f"(fired at {[r['step'] for r in mine if r['fired']]})", eq))
        g, _, _, _ = s48c._trigger_rows(a, task, seed, lambda info, oo: make_trigger_guard(
            task, seed, info, oo, 40800, ops.copy_exact, **dict(c0, thr=1.01, rise=1.0, guard=0.5, source="probe")), steps)
        accs = sorted({round(r["acc"], 3) for r in g})
        rows.append((f"{name} with the guard at 0.5 on a model at chance (probe accuracy {accs[:3]}...): no firing, eligible checks "
                     f"blocked by the guard ({sum(1 for r in g if r.get('blocked_by') == 'guard')})",
                     not any(r["fired"] for r in g) and all(r.get("blocked_by") == "guard" for r in g if r["eligible"])
                     and max(r["acc"] for r in g) < 0.5))
    e, m3, _, _ = s48c._trigger_rows(a, task, seed, lambda info, oo: make_trigger_guard(
        task, seed, info, oo, 40800, ops.copy_exact, **dict(BASE, thr=1.01, rise=1.0, source="eval")), [2400])
    ref = tbo.evaluate(m3, task, eval_batch(task))[0]
    rows.append((f"source eval: the rule reads the 2048-query held-out evaluation ({e[0]['acc']} = evaluate(eval_batch) {ref}; "
                 f"eval_batch has {eval_batch(task).shape[0]} sequences)", e[0]["acc"] == ref and eval_batch(task).shape[0] == 2048))
    return [[n, bool(v)] for n, v in rows]
