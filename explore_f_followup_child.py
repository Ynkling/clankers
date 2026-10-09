#!/usr/bin/env python
"""
explore_f_followup_child.py — EXPLORATORY, not a result. Run code of Session F's follow-up screens S68, S69 and S70
(parents explore_f_premise, explore_f_hebb_decay, explore_f_rh_lite). Runs INSIDE child processes on this branch's own
modules (no main-line hook), 1 torch thread. Training: test_router_reliability.run_one (onset_run: batch 32, evaluation
every 1200 on 2048 held-out sequences, early stop after 3 evaluations >= 0.95).

S68 (premise): test_channel_binding.task_for(4, S) (grouped, P=4, n_vals=16, n_q=1; S=4 L=51, S=8 L=99 — the main line's
  TASKS P4S4 / TASK48), conv 'layer' K=4.
    S68_SINGLE_S{4,8}   test_short_conv's B_conv (k=1, gate 'none', conv 'layer', K=4) converted to a delta memory with
                        β = 1 fixed, L2 keys, tied write, decay 0.95 (explore_f_delta_fast.to_delta_fast); Adam 1e-3 on
                        every parameter; 43200 updates.
    S68_ORACLE_S{4,8}   the perfect gate with k = S (test_short_conv's ceiling_conv with n_ch = S), the same delta memory
                        and recipe.
S69 (decay on Hebbian): explore_far_cue.HeaderTask(8, S=2, n_vals=16, n_q=1), no conv; Adam 1e-3; 48000 updates (S59's).
    S69_{FIXED,ROUTED,GDN}   test_short_conv's arm B (k=1), the HEBBIAN memory with that decay: explore_delta_mem.to_delta
                        (beta=0.0, write=1.0, normalize=False, decay=None/"routed"/"gdn") — the erase off, the write at 1,
                        raw keys: the current Hebbian score Σ_s A(t,s)(x_t·x_s) v_s with A(t,s) = Π α (CHECK: FIXED equals
                        the unconverted Hebbian model to 1e-5). Routed and GDN as S59 (same init and generator).
    S69_ORACLE          the perfect gate (k = 2), the same Hebbian memory with the fixed decay (validity).
S70 (RandHeaderTask-lite): explore_f_tasks.RandHeaderTask(P=4, nb=2, lmin=1, lmax=3): 4 keys per stream in 2 blocks per
  stream, the first block's length uniform on {1, 2, 3} and the remainder (3, 2, 1) in the second, the 4 blocks in random
  order, one CTX per block; L = 23. k = 2, no conv.
    S70_ORACLE_{HEBB,DELTA}[_2X]  the perfect gate (k = 2), Hebbian or delta (β = 1, L2 keys); Adam 1e-3; 24000 (48000
                        for _2X).
    S70_{HEBB,DELTA}    LOCAL3 + SLOW (S43's recipe: gate W_in, W_g, window 1e-3 throughout; the rest 1e-4 for updates
                        1-2400, 1e-3 after; no hinge) on the Hebbian memory or the delta memory with β = 1 fixed, L2 keys;
                        the budget fixed by the oracle phase (24000 or 48000).
After each run: the write-order accuracy; for delta / decay arms the decay and β at CTX / KEY / VAL per layer.
"""

import json
import time

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

import explore_common as ec
import explore_common2 as c2
import explore_delta_mem as dm
import explore_f_delta_fast as ff
import explore_far_cue as s6
import explore_f_tasks as ft
import explore_f_decay_child as dcc
import explore_f_window_delta_child as wdc
import explore_window_recipe_child as s43c
import test_binding_onset as tbo
import test_short_conv as tsc
from test_multilayer_binding import build, probe_batch
from test_channel_binding import ARCH, arms as cb_arms, task_for
from test_router_reliability import run_one

LR = 1e-3
T44, T48 = task_for(4, 4), task_for(4, 8)
H8 = s6.HeaderTask(8, S=2, n_vals=16, n_q=1)
RHL = ft.RandHeaderTask(P=4, nb=2, lmin=1, lmax=3)
STATS_RHL = ft.rand_header_stats(c2.ROUTE_AT)
DECAYS = {"FIXED": None, "ROUTED": "routed", "GDN": "gdn"}


def spec(arm):
    """task, model arm, builder conversion, recipe kind, stats."""
    if arm.startswith("S68_"):
        task = T44 if arm.endswith("S4") else T48
        if "SINGLE" in arm:
            a = tsc.ARM["B_conv"]
        else:
            a = dict(tsc.ARM["ceiling_conv"], key=f"ceiling_conv_k{task.S}", n_ch=task.S)
        return dict(task=task, a=a, conv=lambda m, seed: ff.to_delta_fast(m, seed=seed, beta=1.0), slow=False,
                    stats=tsc.conv_stats, local=False)
    if arm.startswith("S69_"):
        if arm == "S69_ORACLE":
            a, d = {x["key"]: x for x in cb_arms(H8)}["ceiling"], None
        else:
            a, d = tsc.ARM["B"], DECAYS[arm.split("_")[1]]
        return dict(task=H8, a=a, conv=lambda m, seed, d=d: dm.to_delta(m, seed=seed, beta=0.0, write=1.0, normalize=False,
                                                                          decay=d),
                    slow=False, stats=s6.header_stats if arm == "S69_ORACLE" else tsc.conv_stats, local=False)
    if arm.startswith("S70_"):
        delta = "DELTA" in arm
        conv = (lambda m, seed: ff.to_delta_fast(m, seed=seed, beta=1.0)) if delta else None
        if "ORACLE" in arm:
            return dict(task=RHL, a={x["key"]: x for x in cb_arms(RHL)}["ceiling"], conv=conv, slow=False,
                        stats=STATS_RHL, local=False)
        return dict(task=RHL, a=s43c.A, conv=conv, slow=True, stats=STATS_RHL, local=True)
    raise KeyError(arm)


def builder(sp):
    def b(a, seed):
        def make():
            m = wdc.local_model(sp["task"], a, seed) if sp["local"] else build(sp["task"], a, ARCH, seed)
            return sp["conv"](m, seed) if sp["conv"] is not None else m
        return make
    return b


def run_kw(sp, holder):
    kw = dict(task=sp["task"], grad_fn=tsc.conv_grad_norms, lr=LR, builder=builder(sp), stats_fn=sp["stats"])
    if sp["slow"]:
        kw.update(run_kw=dict(param_groups=s43c.groups(holder)), check=wdc.switch_check(holder))
    return kw


def run(p):
    arm, seed, iters = p["arm"], p["seed"], p["iters"]
    sp = spec(arm)
    holder = {}
    keep = {"at": -1}
    rec = run_one(sp["a"], seed, iters, p.get("eval_every", tbo.EVAL_EVERY), keep=keep, **run_kw(sp, holder))
    if rec.get("ok"):
        m = keep["model"]
        task = sp["task"]
        rec.update(arm=arm, k=sp["a"]["n_ch"], delta_cfg=getattr(m, "delta_cfg", None), group_names=holder.get("names"),
                   write_order=ft.write_order_acc(m, task, tbo.eval_batch(task)))
        if isinstance(m.attn, dm.DeltaMemory):
            rec["decay_by_role"] = dcc.decay_by_role(m, task, seed)
            rec["decay_params"] = dcc.decay_params(m)
            rec["decay_init"] = getattr(m.attn.decay_mod, "init", None)
        if sp["a"]["n_ch"] >= 2 and sp["a"]["gate"] != "perfect":
            rec["fail"] = None if rec["transition"] is not None else ec.fail_class(rec)
            rec["tag"] = ec.tag(rec)
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def checks(p):
    import copy
    rows = []
    # S68: models
    for S, task in ((4, T44), (8, T48)):
        ms = builder(spec(f"S68_SINGLE_S{S}"))(spec(f"S68_SINGLE_S{S}")["a"], 300)()
        mo = builder(spec(f"S68_ORACLE_S{S}"))(spec(f"S68_ORACLE_S{S}")["a"], 300)()
        ok = (ms.n_ch == 1 and ms.gate_kind == "none" and ms.conv == "layer" and mo.n_ch == S and mo.gate_kind == "perfect"
              and mo.conv == "layer" and isinstance(ms.attn, ff.FastDeltaMemory) and ms.attn.beta_fixed == 1.0
              and ms.attn.normalize and isinstance(mo.attn, ff.FastDeltaMemory) and task.L == 3 * (4 * S + 1))
        x = task.make_batch(4, torch.Generator().manual_seed(3))[0][:, :-1]
        mo.eval()
        with torch.no_grad():
            g = mo(x)[2]
        lab = task.stream_labels(x)
        onehot = bool((g.argmax(-1) == lab).all()) and bool(((g == 0) | (g == 1)).all())
        rows.append((f"S68 S={S}: SINGLE k=1 conv layer, delta beta 1 L2 keys; ORACLE k={S} conv, perfect gate one-hot on each "
                     f"position's stream ({onehot}); L = {task.L}", ok and onehot))
    # S69: FIXED equals the unconverted Hebbian single channel to 1e-5; the decay modules as S59's
    m0 = build(H8, tsc.ARM["B"], ARCH, 350)
    mf = builder(spec("S69_FIXED"))(tsc.ARM["B"], 350)()
    mr = builder(spec("S69_ROUTED"))(tsc.ARM["B"], 350)()
    mg = builder(spec("S69_GDN"))(tsc.ARM["B"], 350)()
    x = H8.make_batch(8, torch.Generator().manual_seed(4))[0][:, :-1]
    for m in (m0, mf, mr, mg):
        m.eval()
    with torch.no_grad():
        l0, lf, lr_ = m0(x)[0], mf(x)[0], mr(x)[0]
    d = float((l0 - lf).abs().max())
    dr = float((l0 - lr_).abs().max())
    s59g = dm.to_delta(build(H8, tsc.ARM["B"], ARCH, 350), beta=1.0, decay="gdn", seed=350).attn.decay_mod
    rows.append((f"S69: FIXED (beta 0, write 1, raw keys) equals the unconverted Hebbian single channel (logits max |diff| "
                 f"{d:.1e} <= 1e-5); ROUTED at init equals it too (k = 1, g = 1: alpha 0.95; {dr:.1e} <= 1e-5); the memory "
                 f"skips the erase ({mf.attn.erase_off()}); GDN's init equals S59's for the seed",
                 d <= 1e-5 and dr <= 1e-5 and mf.attn.erase_off() and mr.attn.erase_off()
                 and torch.equal(mg.attn.decay_mod.A_log, s59g.A_log) and torch.equal(mg.attn.decay_mod.dt_bias, s59g.dt_bias)))
    # S70: the task
    for nm, v in ft.check_rand_header(RHL):
        rows.append((f"S70 RandHeaderTask-lite: {nm}", v))
    rows.append((f"S70 RandHeaderTask-lite compositions {RHL.comps} (first block 1, 2 or 3; the remainder in the second)",
                 RHL.comps == [(1, 3), (2, 2), (3, 1)] and RHL.L == 23))
    sh, sd = builder(spec("S70_HEBB"))(s43c.A, 330)(), builder(spec("S70_DELTA"))(s43c.A, 330)()
    ph = dict(sh.named_parameters())
    same = all(torch.equal(q, ph[n]) for n, q in sd.named_parameters()) and set(ph) == set(dict(sd.named_parameters()))
    rows.append((f"S70: HEBB and DELTA share every initial parameter ({same}; beta 1 adds none); DELTA memory "
                 f"{type(sd.attn).__name__} beta {sd.attn.beta_fixed}, L2 keys {sd.attn.normalize}; window gate on vocab "
                 f"{RHL.vocab}", same and sd.attn.beta_fixed == 1.0 and sd.attn.normalize and sh.local))
    mo = build(RHL, {x["key"]: x for x in cb_arms(RHL)}["ceiling"], ARCH, 330)
    rs = ft.rand_header_routing_stats(mo, RHL, probe_batch(RHL, 330))
    rows.append((f"S70: the perfect gate routes perfectly (margin {rs['margin']:.3f}, eta2 by stream KEY / VAL "
                 f"{rs['eta_key_by_stream']:.3f} / {rs['eta_val_by_stream']:.3f})",
                 abs(rs["margin"] - 1) < 1e-6 and abs(rs["eta_val_by_stream"] - 1) < 1e-6))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    stamps = []
    h = register_optimizer_step_post_hook(lambda o, a, k: stamps.append(time.perf_counter()))
    try:
        run(dict(arm=p["arm"], seed=0, iters=p["steps"], eval_every=10 ** 9))
    finally:
        h.remove()
    dd = sorted(y - x for x, y in zip(stamps, stamps[1:]))
    sp = spec(p["arm"])
    m = builder(sp)(sp["a"], 0)()
    t0 = time.time()
    tbo.evaluate(m, sp["task"], tbo.eval_batch(sp["task"]))
    sp["stats"](m, sp["task"], probe_batch(sp["task"], 0), 2400)
    return [dd[len(dd) // 2], time.time() - t0]
