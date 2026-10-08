#!/usr/bin/env python
"""
explore_split_k4_child.py — EXPLORATORY, not a result. S51's code that runs INSIDE a child process on the main line's
module tree at explore_main9c.MAIN_SHA (9c5939e). Parent: explore_split_k4.

CONFIGURATION "a" (explore_b16_main): test_stream_recipe.run_attempt with ARM["A4k4"] — S=4, P=4, k=4, conv, 28800 updates;
the path of S33's A_HINGE and S37's HINGE4k4_A, so a seed fixes the initial parameters and batches.
  W4k4        LOCAL3 (explore_b16_gates.to_local) + SLOW (explore_b16_main.make_rc "adam", GATE_LOCAL: gate W_in, W_g,
              window at 1e-3 throughout; every other trainable parameter 1e-4 for updates 1-2400, 1e-3 after), no hinge.
  W4k4_SPLIT  W4k4 + the KEYMASS trigger (explore_window_d8_child.make_trigger at S37's cap: at most 2 splits, >= 4800
              apart; checks every 2400 from 4800, 2400 = reference, through 26400); onset_run's Adam built through S36's
              capturing factory.
MEASURED: S38's decoder after every evaluation (chance 1/4); routing_k's map at every evaluation (the path's statistics);
the trigger's rows.
"""

import contextlib

import torch

import explore_main9c as mt
import explore_split_child as s36c
import explore_split_target_child as s37c
import explore_b16_gates as g16
import explore_b16_main as b16
import explore_window_d8_child as s48c
import test_stream_recipe as tsr

CFG = "a"
EVERY = s48c.EVERY
MAX_SPLITS = s37c.MAX_SPLITS                    # 2 (S37's)
ARM = {"W4k4": dict(opt="adam"), "W4k4_SPLIT": dict(opt="adam", trigger=True)}


def build_rc(arm_name, p, holder, out, info):
    arm = ARM[arm_name]
    task = b16.task_of(CFG)
    probe = s48c.every_probe(task, out) if p.get("probe", True) else None
    rc, _ = b16.make_rc("adam", holder, b16.GATE_LOCAL, tau=None, convert=(g16.to_local,), probe=probe, keep=p.get("keep"))
    if arm.get("trigger"):
        ctx, opt_of = s36c.adam_capture(holder), (lambda: holder["opts"][0])
        rc = s36c.with_trigger(rc, s48c.make_trigger(task, p["seed"], info, opt_of, p.get("total", p["iters"]) - EVERY,
                                                     max_splits=MAX_SPLITS, thr=p.get("thr", s48c.ACC_THR)))
    else:
        ctx = contextlib.nullcontext()
    return rc, ctx


def run(p):
    """p: arm, seed, iters, eval_every / probe (timing), thr / total (CHECK)."""
    arm = ARM[p["arm"]]
    holder, out, info = {}, {}, dict(checks=[])
    rc, ctx = build_rc(p["arm"], p, holder, out, info)
    with ctx:
        rec = b16.path(CFG, b16.arm_of(CFG), p["seed"], p["iters"], rc, eval_every=p.get("eval_every"))
    if out:
        last = max(out, key=int)
        out["end"] = dict(out[last], step=int(last))
    return b16.finish(rec, arm=p["arm"], cfg=CFG, opt="adam", kind="local", decode=out, groups=b16.group_names(holder),
                      checks=info["checks"], splits=sum(1 for c in info["checks"] if c["fired"]),
                      max_splits=MAX_SPLITS if arm.get("trigger") else None)


def run_s37(p):
    """S37's HINGE4k4_A on its own run code (explore_split_target_child.run), for the reproduction CHECK."""
    return s37c.run(dict(arm="HINGE4k4_A", seed=p["seed"], iters=p["iters"]))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def checks(p):
    rows = []
    a = b16.arm_of(CFG)
    task = b16.task_of(CFG)
    seed = 240
    m0 = tsr.builder_for(a)(a, seed)()
    m1 = g16.to_local(tsr.builder_for(a)(a, seed)(), seed)
    p0, p1 = dict(m0.named_parameters()), dict(m1.named_parameters())
    same = all(torch.equal(p1[n], p0[n]) for n in p0)
    extra = [n for n in p1 if n not in p0]
    rows.append((f"config a (S={task.S}, P={task.P}, k={m1.n_ch}): LOCAL3's conversion keeps every parameter of the run path's model "
                 f"bitwise ({len(p0)}), adds only {extra} = batch 1's window for the seed "
                 f"({torch.equal(m1.gate_conv.conv_w.detach(), g16.window_init(seed))}); W_h frozen ({not m1.W_h.requires_grad})",
                 same and extra == ["gate_conv.conv_w"] and torch.equal(m1.gate_conv.conv_w.detach(), g16.window_init(seed))
                 and not m1.W_h.requires_grad and task.S == 4 and m1.n_ch == 4))
    W = b16.WARM
    h, rec, lg, ids, keep = b16.stub_lrs(CFG, a, lambda hd, kp: build_rc("W4k4_SPLIT", dict(
        keep=kp, probe=False, seed=3, iters=W + 1, total=28800), hd, {}, dict(checks=[]))[0])
    mm = keep["model"]
    names = [g["names"] for g in h["groups"]]
    ok = (b16.cover(ids, mm) and names[0] == list(b16.GATE_LOCAL) and len(names) == 2 and len(lg) == W + 1
          and lg[0] == [b16.LR, b16.LR_WARM] and lg[W - 1] == [b16.LR, b16.LR_WARM] and lg[W] == [b16.LR, b16.LR] and rec.get("ok"))
    rows.append((f"config a W4k4 / W4k4_SPLIT: groups {[(g['tag'], len(g['names'])) for g in h['groups']]} cover every trainable "
                 f"parameter once, gate {names[0]}; lrs at update 1 {lg[0]}, {W} {lg[W - 1]}, {W + 1} {lg[W]}", ok))
    steps = list(range(EVERY, 26400 + 1, EVERY))
    keys = ("step", "acc", "prev", "map_before", "key_masses", "key_cs", "key_c0", "key_ok", "fired", "blocked", "eligible",
            "map_after", "row_dist_after", "state_zeroed")
    mine, mm, _, _ = s48c._trigger_rows(a, task, seed, lambda info, oo: s48c.make_trigger(task, seed, info, oo, 26400,
                                                                                          max_splits=2, thr=1.01, rise=1.0), steps)
    theirs, ms, _, _ = s48c._trigger_rows(a, task, seed, lambda info, oo: s37c.make_trigger(task, seed, info, oo, 26400,
                                                                                            thr=1.01, rise=1.0), steps)
    eq = (len(mine) == len(theirs) and all({k: r.get(k) for k in keys} == {k: q.get(k) for k in keys} for r, q in zip(mine, theirs))
          and torch.equal(mm.W_g.detach(), ms.W_g.detach()))
    rows.append((f"config a: the trigger at cap 2 equals S37's make_trigger row for row on LOCAL3 (fired at "
                 f"{[r['step'] for r in mine if r['fired']]} vs {[r['step'] for r in theirs if r['fired']]})", eq))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    out = {}
    for arm_name in p["which"]:
        st = b16.step_timing(lambda: run(dict(arm=arm_name, seed=0, iters=p["steps"], eval_every=10 ** 9, probe=False)), True)
        a = b16.arm_of(CFG)
        m = g16.to_local(tsr.builder_for(a)(a, 0)(), 0)
        out[arm_name] = [st, b16.eval_timing(CFG, m), b16.decode_timing(CFG, m)]
    return out
