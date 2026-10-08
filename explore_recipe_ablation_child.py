#!/usr/bin/env python
"""
explore_recipe_ablation_child.py — EXPLORATORY, not a result. S50's code that runs INSIDE a child process on the main line's
module tree at explore_main9c.MAIN_SHA (9c5939e). Parent: explore_recipe_ablation. Built from batch 17's S48 child
(explore_window_d8_child: the width-w window conversion, the KEYMASS trigger with the cap as a parameter, the decoder after
every evaluation), imported read-only, and explore_b16_main's recipe composition.

CONFIGURATION: S44(b) / S48's: test_stream_curriculum.run_sc with ARM["D8"] (S=8, P=4, k=16, conv, all 8 streams from
step 1), LOCAL3's gate, evaluation every 1200, no hinge, 28800 updates.
  W_RESET         LOCAL3_SLOW_D8_A + W_g's Adam state reset (explore_b18_gates.zero_state: S36's split_op's reset lines,
                  no row copy) after the evaluation at each update listed in p["reset_at"] (the parent passes the updates
                  where batch 17's W_SPLIT fired on that seed), at the trigger's place in the check chain (after the lr
                  switch and the decoder). onset_run's Adam is built through S36's capturing factory (the same constructor).
  W_SPLIT_NOSLOW  W_SPLIT with one rate: every trainable parameter at 1e-3 throughout (S48's W_NOSLOW + the trigger).
  W_SPLIT_W4      W_SPLIT with the width-4 window (S48's W_WIDTH4 + the trigger).
  W_SPLIT_M       S48's W_SPLIT_M: LOCAL3_SLOW_D8_M (S25's groups, Muon lr 0.005) + the trigger (W_g's momentum zeroed).
  W_M_REF         LOCAL3_SLOW_D8_M fresh on this CPU (W_SPLIT_M's pair when batch 16's records do not reproduce here).
THE TRIGGER (S48's): checks every 2400 from 4800 (2400 = reference) through iters - 2400 on S36's 64-sequence probe; fire
if the probe accuracy is < 0.95 and it rose < 0.02 since the previous check; at most 3 splits, >= 4800 apart; KEYMASS
target; S24's split; W_g's optimizer state zeroed.
"""

import contextlib

import torch

import explore_main9c as mt
import explore_split_child as s36c
import explore_muon_recipe as s25
import explore_b16_gates as g16
import explore_b16_main as b16
import explore_window_d8_child as s48c
import explore_b18_gates as g18
import test_binding_onset as tbo
import test_stream_recipe as tsr

CFG = "b"
EVERY = s48c.EVERY
MAX_SPLITS = 3
ARM = {
    "W_RESET": dict(opt="adam", reset=True),
    "W_SPLIT_NOSLOW": dict(opt="adam", trigger=True, noslow=True),
    "W_SPLIT_W4": dict(opt="adam", trigger=True, width=4),
    "W_SPLIT_M": dict(opt="muon", trigger=True),
    "W_M_REF": dict(opt="muon"),
}


def make_reset(steps, info, opt_of):
    steps = sorted(int(s) for s in steps)

    def reset(m, step):
        if step not in steps:
            return
        o = opt_of()
        W = m.W_g
        before = {k: float(v.abs().max()) for k, v in (o.state.get(W) or {}).items() if torch.is_tensor(v)}
        w0 = W.detach().clone()
        zeroed = g18.zero_state(o, W)
        after = {k: float(v.abs().max()) for k, v in (o.state.get(W) or {}).items() if torch.is_tensor(v)}
        info["resets"].append(dict(step=step, state_before_max=before, state_after_max=after, state_zeroed=zeroed,
                                   w_g_unchanged=bool(torch.equal(w0, W.detach()))))
    return reset


def build_rc(arm_name, p, holder, out, info):
    arm = ARM[arm_name]
    task = b16.task_of(CFG)
    probe = s48c.every_probe(task, out) if p.get("probe", True) else None
    rc, _ = b16.make_rc(arm["opt"], holder, b16.GATE_LOCAL, tau=None, mlr=p.get("mlr"),
                        warm_lr=b16.LR if arm.get("noslow") else b16.LR_WARM, convert=s48c.convert_of(arm), probe=probe,
                        keep=p.get("keep"))
    if arm.get("trigger"):
        if arm["opt"] == "muon":
            ctx, opt_of = s25.muon_in_onset_run(holder), (lambda: holder["opts"][0].muon)
        else:
            ctx, opt_of = s36c.adam_capture(holder), (lambda: holder["opts"][0])
        rc = s36c.with_trigger(rc, s48c.make_trigger(task, p["seed"], info, opt_of, p.get("total", p["iters"]) - EVERY,
                                                     max_splits=MAX_SPLITS, thr=p.get("thr", s48c.ACC_THR)))
    elif arm.get("reset"):
        ctx, opt_of = s36c.adam_capture(holder), (lambda: holder["opts"][0])
        rc = s36c.with_trigger(rc, make_reset(p["reset_at"], info, opt_of))
    elif arm["opt"] == "muon":
        ctx = s25.muon_in_onset_run(holder)
    else:
        ctx = contextlib.nullcontext()
    return rc, ctx


def run(p):
    """p: arm, seed, iters, mlr (Muon arms), reset_at (W_RESET), eval_every / probe (timing), thr / total (CHECK)."""
    arm = ARM[p["arm"]]
    holder, out, info = {}, {}, dict(checks=[], resets=[])
    rc, ctx = build_rc(p["arm"], p, holder, out, info)
    with ctx:
        rec = b16.path(CFG, b16.arm_of(CFG), p["seed"], p["iters"], rc, eval_every=p.get("eval_every"))
    if out:
        last = max(out, key=int)
        out["end"] = dict(out[last], step=int(last))
    return b16.finish(rec, arm=p["arm"], cfg=CFG, opt=arm["opt"], kind="local", width=arm.get("width", g16.WIDTH),
                      noslow=bool(arm.get("noslow")), muon_lr=p.get("mlr") if arm["opt"] == "muon" else None, decode=out,
                      groups=b16.group_names(holder), checks=info["checks"], resets=info["resets"],
                      reset_at=p.get("reset_at"), splits=sum(1 for c in info["checks"] if c["fired"]),
                      max_splits=MAX_SPLITS if arm.get("trigger") else None)


# ── CHECKs ───────────────────────────────────────────────────────────────────
def checks(p):
    rows = []
    a = b16.arm_of(CFG)
    task = b16.task_of(CFG)
    seed = 260
    # (1) the reset: S36's split_op's state lines without the copy
    def state_after(op):
        m = s48c._model(a, seed)
        opt = torch.optim.Adam(m.parameters(), lr=b16.LR)
        s48c._one_step(m, task, opt)
        w0 = m.W_g.detach().clone()
        before = sorted(k for k, v in opt.state[m.W_g].items() if torch.is_tensor(v))
        op(m, opt)
        st = opt.state[m.W_g]
        return before, {k: float(v.abs().max()) for k, v in st.items() if torch.is_tensor(v)}, torch.equal(w0, m.W_g.detach())
    b1, z1, same1 = state_after(lambda m, o: g18.zero_state(o, m.W_g))
    b2, z2, same2 = state_after(lambda m, o: s36c.split_op(m, o, 1, 2, 7))
    rows.append((f"the reset zeroes the same optimizer-state tensors of W_g as S36's split_op ({sorted(z1)} vs {sorted(z2)}, all zero "
                 f"after: {all(v == 0 for v in z1.values())}) and leaves W_g unchanged ({same1}; split_op changes it: {not same2})",
                 sorted(z1) == sorted(z2) == b1 == b2 and all(v == 0 for v in z1.values())
                 and all(v == 0 for v in z2.values()) and same1 and not same2))
    # (2) the width-4 window's causality
    x = task.make_batch(4, torch.Generator().manual_seed(5))[0][:, :-1]
    t = 30
    m = s48c._model(a, seed, width=4)
    m.eval()
    with torch.no_grad():
        g0 = m(x)[2]
        xa = x.clone(); xa[:, t - 4] = (xa[:, t - 4] + 1) % task.vocab
        xb = x.clone(); xb[:, t - 3] = (xb[:, t - 3] + 1) % task.vocab
        blind = torch.equal(m(xa)[2][:, t], g0[:, t])
        sees = not torch.equal(m(xb)[2][:, t], g0[:, t])
    rows.append((f"W_SPLIT_W4: the width-4 gate at t = {t} cannot see token t-4 ({blind}) and can see t-3 ({sees}); window "
                 f"shape {tuple(m.gate_conv.conv_w.shape)}", blind and sees and m.gate_conv.conv_w.shape[1] == 4))
    # (3) groups and lrs through the real run path, 2401 updates
    W = b16.WARM
    for arm_name, want1, want2 in (("W_SPLIT_NOSLOW", [b16.LR, b16.LR], [b16.LR, b16.LR]),
                                   ("W_SPLIT_W4", [b16.LR, b16.LR_WARM], [b16.LR, b16.LR]),
                                   ("W_RESET", [b16.LR, b16.LR_WARM], [b16.LR, b16.LR])):
        h, rec, lg, ids, keep = b16.stub_lrs(CFG, a, lambda hd, kp: build_rc(arm_name, dict(
            keep=kp, probe=False, seed=3, iters=W + 1, reset_at=[], total=28800), hd, {}, dict(checks=[], resets=[]))[0])
        mm = keep["model"]
        names = [g["names"] for g in h["groups"]]
        ok = (b16.cover(ids, mm) and names[0] == list(b16.GATE_LOCAL) and len(names) == 2 and len(lg) == W + 1
              and lg[0] == want1 and lg[W - 1] == want1 and lg[W] == want2 and rec.get("ok") and not mm.W_h.requires_grad
              and mm.gate_conv.conv_w.shape[1] == ARM[arm_name].get("width", 3))
        rows.append((f"{arm_name}: groups {[(g['tag'], len(g['names'])) for g in h['groups']]} cover every trainable parameter once, "
                     f"gate {names[0]}; lrs at update 1 {lg[0]}, {W} {lg[W - 1]}, {W + 1} {lg[W]}; window width "
                     f"{mm.gate_conv.conv_w.shape[1]}", ok))
    # (4) the trigger under Muon (if W_SPLIT_M runs)
    if p.get("muon"):
        steps = [2400, 4800]
        rm, mmu, omu, sbm = s48c._trigger_rows(a, task, seed, lambda info, oo: s48c.make_trigger(
            task, seed, info, oo, 26400, thr=1.01, rise=1.0), steps, muon=True, mlr=p["mlr"])
        st = omu.state[mmu.W_g]
        zeroed = all(float(v.abs().max()) == 0.0 for v in st.values() if torch.is_tensor(v)) and \
            any(float(v.abs().max()) > 0 for v in sbm.values())
        rows.append((f"W_SPLIT_M: W_g on Muon; the trigger fired at {[r['step'] for r in rm if r['fired']]} and zeroed W_g's Muon "
                     f"state {sorted(sbm)} ({zeroed})", [r["step"] for r in rm if r["fired"]] == [4800] and zeroed))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    out = {}
    for arm_name in p["which"]:
        arm = ARM[arm_name]
        st = b16.step_timing(lambda: run(dict(arm=arm_name, seed=0, iters=p["steps"], mlr=p["mlr"], eval_every=10 ** 9,
                                                probe=False, reset_at=[])), arm["opt"] == "adam")
        a = b16.arm_of(CFG)
        m = s48c._model(a, 0, width=arm.get("width", 3))
        out[arm_name] = [st, b16.eval_timing(CFG, m), b16.decode_timing(CFG, m)]
    return out
