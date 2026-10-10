#!/usr/bin/env python
"""
explore_f_eight_child.py — EXPLORATORY, not a result. S76's run code (parent: explore_f_eight). Runs INSIDE a child process
on the main line's module tree at explore_main9c.MAIN_SHA (9c5939e), as S54(c) and Session H's S61; 1 torch thread.

PATH: test_window_gate's Part C/D path = test_stream_curriculum.run_sc with ARM["D8"] (S = 8, P = 4, k = 16, conv, all 8
streams from step 1), via explore_b16_main.path("b", ...), 43200 updates, evaluation every 1200, early stop after 3 >= 0.95.
PHASE 1, the oracle: the perfect gate (ceil8_D8: k = 16, stream s -> channel s, conv), one Adam at 1e-3 on every parameter
  (the perfect gate's own recipe, as S54's C_ORACLE arms):
    O_HEBB        the Hebbian memory (reference).
    O_B1_L2       delta, β = 1 fixed, L2-normalised keys (S54's delta memory with β fixed).
    O_B025_L2     delta, β = 0.25 fixed, L2 keys.
    O_B1_RAW      delta, β = 1 fixed, raw keys (the Hebbian model's keys x_sparse, unnormalised).
    O_B025_RAW    delta, β = 0.25 fixed, raw keys.
  Delta memories: explore_f_delta_fast.to_delta_fast(beta, normalize), tied write, decay 0.95.
PHASE 2, the recipe: LOCAL3 + SLOW (Adam: gate W_in, W_g, window 1e-3 throughout; the rest 1e-4 for updates 1-2400, 1e-3 after;
  no hinge; no decoder probe) + the KEYMASS plateau trigger with Session H's simplified split, copied from claude/explore-H
  at 7e9197f (explore_h_collapse_child.make_trigger_op, explore_h_copy2x2_child.copy_nonoise_op): checks every 2400 from
  4800 (2400 the reference) through 40800 on S36's 64-sequence probe; fire if the probe accuracy is < 0.95 and rose < 0.02
  since the previous check; <= 3 operations, >= 4800 apart; the operation: W_g[c0] = W_g[c*] exactly (c* the largest
  key-position read mass, c0 the smallest), no noise, c*'s row unchanged, W_g's Adam state untouched.
    HEBB_SPLIT    on the Hebbian memory (H's S61 COPY_NONOISE exactly).
    DELTA_SPLIT   on DELTA* (the Phase-1 choice in explore_out/F/eight_delta_star.json).
DIVERGENCE GUARD: after each evaluation, a non-finite held-out loss or accuracy raises "DIVERGED at <step>" (the run is
  recorded ok = False with that error and counted as unbound, diverged); without it a diverged run would train 43200 steps.
"""

import contextlib
import json
import math
import os
import subprocess

import torch

import explore_main9c as mt
import explore_muon_scale_child as s33c
import explore_split_child as s36c
import explore_split_target_child as s37c
import explore_b16_gates as g16
import explore_b16_main as b16
import explore_delta_mem as dm
import explore_f_delta_fast as ff
import test_binding_onset as tbo
import test_scale_axes as tsa
import test_stream_recipe as tsr
from test_multilayer_binding import probe_batch

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = "b"
TOTAL = 43200
H_SHA = "7e9197f23b20688652d689fd147871648a37db0f"
EVERY, FIRST, REF_AT = s37c.EVERY, s37c.FIRST, s37c.REF_AT          # 2400, 4800, 2400
ACC_THR, RISE, GAP = s37c.ACC_THR, s37c.RISE, s37c.GAP              # 0.95, 0.02, 4800
MAX_SPLITS = 3
VARIANTS = {"O_B1_L2": (1.0, True), "O_B025_L2": (0.25, True), "O_B1_RAW": (1.0, False), "O_B025_RAW": (0.25, False)}
STAR = os.path.join(HERE, "explore_out", "F", "eight_delta_star.json")


# ── Session H's simplified split (copied from claude/explore-H at 7e9197f, line for line) ─────────────────────────────
def make_trigger_op(task, seed, info, opt_of, last, op, max_splits=MAX_SPLITS, gap=GAP, thr=ACC_THR, rise=RISE):
    """explore_window_d8_child.make_trigger line for line, except that the operation applied at a firing is op(m, o, c*,
    c0, noise_seed) -> dict (split: S36's split_op; reset: W_g's optimizer state zeroed only)."""
    data = s36c.probe_data(task)
    rprobe = probe_batch(task, seed)
    st = dict(prev=None, n=0, last=None)

    def trig(m, step):
        if step % EVERY or step < REF_AT or step > last:
            return
        acc, _ = tbo.evaluate(m, task, data)
        (kc, k0, km), (ac, a0, am) = s37c.select(s37c.read_gate(m, data), task.S, task.P)
        cmap = tsa.routing_k(m, task, rprobe)["ch_map"]
        row = dict(step=step, acc=acc, prev=st["prev"], map_before=cmap,
                   key_masses=[round(float(v), 5) for v in km], all_masses=[round(float(v), 5) for v in am],
                   key_cs=kc, key_c0=k0, all_cs=ac, all_c0=a0,
                   key_ok=s37c.targeting(cmap, kc, k0), all_ok=s37c.targeting(cmap, ac, a0), fired=False, blocked=False)
        eligible = step >= FIRST and st["prev"] is not None and acc < thr and acc - st["prev"] < rise
        capped = st["n"] >= max_splits or (st["last"] is not None and step - st["last"] < gap)
        row["eligible"] = eligible
        if eligible and capped:
            row["blocked"] = True
            row["blocked_by"] = "cap" if st["n"] >= max_splits else "gap"
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


def copy_nonoise_op(m, o, cs, c0, seed):
    W = m.W_g
    with torch.no_grad():
        before = float((W[cs] - W[c0]).norm())
        W[c0] = W[cs].clone()
    return dict(row_dist_before=before, row_dist_after=float((W[cs] - W[c0]).detach().norm()), state_zeroed=[])


# ── Arms ─────────────────────────────────────────────────────────────────────
def delta_star():
    return json.load(open(STAR)) if os.path.exists(STAR) else None


def mem_conv(arm, impl="fast"):
    """The memory conversion of an arm: () for Hebbian; the delta variant otherwise."""
    if arm in ("O_HEBB", "HEBB_SPLIT"):
        return ()
    if arm == "DELTA_SPLIT":
        d = delta_star()
        assert d is not None, "DELTA* not chosen yet (explore_out/F/eight_delta_star.json)"
        beta, norm = d["beta"], d["normalize"]
    else:
        beta, norm = VARIANTS[arm]
    if impl == "hebb":
        return (dm.converter(beta=beta, normalize=norm, impl="hebb"),)
    return (ff.converter(beta=beta, normalize=norm),)


def build_rc(arm, p, holder, info):
    task = b16.task_of(CFG)
    conv = mem_conv(arm, p.get("impl", "fast"))
    if arm.startswith("O_"):
        rc, _ = b16.make_rc("adam", holder, (), tau=None, convert=conv, keep=p.get("keep"), groups=False)
        return rc, contextlib.nullcontext()
    rc, _ = b16.make_rc("adam", holder, b16.GATE_LOCAL, tau=None, convert=(g16.to_local,) + conv, probe=None, keep=p.get("keep"))
    ctx, opt_of = s36c.adam_capture(holder), (lambda: holder["opts"][0])
    trig = make_trigger_op(task, p["seed"], info, opt_of, p.get("total", TOTAL) - EVERY, copy_nonoise_op, thr=p.get("thr", ACC_THR))
    return s36c.with_trigger(rc, trig), ctx


class Diverged(Exception):
    pass


def run(p):
    arm = p["arm"]
    holder, info = {}, dict(checks=[])
    rc, ctx = build_rc(arm, p, holder, info)
    orig_eval = tbo.evaluate

    def evaluate(model, task, data):                     # the divergence guard (held-out evaluations only)
        acc, loss = orig_eval(model, task, data)
        if not (math.isfinite(acc) and math.isfinite(loss)):
            raise Diverged(f"DIVERGED (non-finite held-out loss {loss})")
        return acc, loss
    tbo.evaluate = evaluate
    try:
        with ctx:
            a = s33c.cfg_arm(CFG, "CEIL") if arm.startswith("O_") else b16.arm_of(CFG)
            rec = b16.path(CFG, a, p["seed"], p["iters"], rc, eval_every=p.get("eval_every"))
    finally:
        tbo.evaluate = orig_eval
    return b16.finish(rec, arm=arm, cfg=CFG, opt="adam", kind="perfect" if arm.startswith("O_") else "local",
                      memory="hebb" if not mem_conv(arm) else "delta", delta=None if not mem_conv(arm) else
                      (delta_star() if arm == "DELTA_SPLIT" else dict(zip(("beta", "normalize"), VARIANTS[arm]))),
                      groups=b16.group_names(holder), checks=info["checks"],
                      splits=sum(1 for c in info["checks"] if c["fired"]))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def h_record(seed=487):
    raw = subprocess.run(["git", "-C", HERE, "show", f"{H_SHA}:explore_out/H/copy2x2_results.json"], capture_output=True,
                         text=True, check=True).stdout
    return json.loads(raw)["runs"][f"COPY_NONOISE|{seed}|0b17627cb1b8|Intel(R) Xeon(R) Processor @ 2.10GHz"]


def checks(p):
    import explore_common as ec
    rows = []
    cpu = ec.cpu_model()
    # (1) the simplified split's mechanics equal H's COPY_NONOISE on a recorded run through its first split (4800) + 2400
    ref = h_record(487)
    t = 7200
    if cpu == "Intel(R) Xeon(R) Processor @ 2.10GHz":
        todo = [("HEBB_SPLIT", "fast")] + ([("DELTA_SPLIT", "hebb")] if delta_star() is not None else [])
        for a_name, impl in todo:
            r = run(dict(arm=a_name, seed=487, iters=t, total=TOTAL, impl=impl))
            same, want = b16.same_upto(r, ref, t)
            rc = [c for c in r["checks"] if c["step"] <= t]
            hc = [c for c in ref["checks"] if c["step"] <= t]
            keys = ("step", "acc", "prev", "map_before", "key_cs", "key_c0", "fired", "map_after", "row_dist_before",
                    "row_dist_after")
            trig = len(rc) == len(hc) and all({k: x.get(k) for k in keys} == {k: y.get(k) for k in keys} for x, y in zip(rc, hc))
            rows.append((f"{a_name} (memory {impl}) on seed 487 reproduces H's COPY_NONOISE|487 (claude/explore-H {H_SHA[:7]}, "
                         f"2.10GHz) through {t} bit for bit: curve {r['curve']} vs {want}; statistics {same}; trigger rows "
                         f"(first split at {[c['step'] for c in rc if c['fired']]}) equal {trig}", same and trig))
    else:
        rows.append((f"H's record reproduction NOT APPLICABLE on {cpu}", True))
    # (2) the copy op: W_g[c0] = W_g[c*] exactly, other rows and the Adam state untouched (H's check)
    import explore_window_d8_child as s48c
    a = b16.arm_of(CFG)
    task = b16.task_of(CFG)
    m = s48c._model(a, 480)
    opt = torch.optim.Adam(m.parameters(), lr=b16.LR)
    s48c._one_step(m, task, opt)
    st0 = {k: v.clone() for k, v in opt.state[m.W_g].items() if torch.is_tensor(v)}
    W0 = m.W_g.detach().clone()
    copy_nonoise_op(m, opt, 3, 7, 0)
    W1 = m.W_g.detach()
    others = [r for r in range(W0.shape[0]) if r not in (3, 7)]
    rows.append(("copy_nonoise_op: W_g[c0] = W_g[c*] exactly, c*'s and the other rows unchanged, Adam state untouched",
                 torch.equal(W1[7], W0[3]) and torch.equal(W1[3], W0[3]) and all(torch.equal(W1[r], W0[r]) for r in others)
                 and all(torch.equal(opt.state[m.W_g][k], st0[k]) for k in st0)))
    # (3) the oracle variants' memories as stated; the perfect gate one-hot; β = 0 / raw keys equals the Hebbian model
    a_c = s33c.cfg_arm(CFG, "CEIL")
    x = task.make_batch(4, torch.Generator().manual_seed(3))[0][:, :-1]
    m0 = tsr.builder_for(a_c)(a_c, 1300)()
    m0.eval()
    with torch.no_grad():
        l0, g0 = m0(x)[0], m0(x)[2]
    ok = True
    desc = []
    for arm, (beta, norm) in VARIANTS.items():
        mm = mem_conv(arm)[0](tsr.builder_for(a_c)(a_c, 1300)(), 1300)
        ok &= isinstance(mm.attn, ff.FastDeltaMemory) and mm.attn.beta_fixed == beta and mm.attn.normalize == norm
        desc.append(f"{arm} beta {mm.attn.beta_fixed} L2 {mm.attn.normalize}")
    mz = dm.to_delta(tsr.builder_for(a_c)(a_c, 1300)(), beta=0.0, write=1.0, normalize=False)
    mz.eval()
    with torch.no_grad():
        lz, gz = mz(x)[0], mz(x)[2]
    dz = float((lz - l0).abs().max())
    rows.append((f"oracle variants {desc}; the perfect gate one-hot ({bool(((g0 == 0) | (g0 == 1)).all())}); beta 0 / write 1 / raw "
                 f"keys equals the Hebbian oracle model's logits to {dz:.1e} <= 1e-5", ok and dz <= 1e-5
                 and bool(((g0 == 0) | (g0 == 1)).all()) and torch.equal(gz, g0)))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    st = b16.step_timing(lambda: run(dict(arm=p["arm"], seed=0, iters=p["steps"], eval_every=10 ** 9)), True)
    return [st, 25.0 if p["arm"] != "O_HEBB" and p["arm"] != "HEBB_SPLIT" else 7.0]
