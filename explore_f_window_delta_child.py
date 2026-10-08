#!/usr/bin/env python
"""
explore_f_window_delta_child.py — EXPLORATORY, not a result. S54 (a) and (d)'s run code (parent: explore_f_window_delta).
Runs INSIDE a child process on this branch's own modules (no main-line hook; batch 16's S43 path), 1 torch thread.

LOCAL3 + SLOW is S43's LOCAL3_SLOW (explore_window_recipe_child): batch 1's LocalGateBDH (arm A's parameters + the
width-3 window from a generator seeded seed + 31337), W_h frozen and unused; one Adam, two groups — gate W_in, W_g,
gate_conv.conv_w at 1e-3 throughout; every other trainable parameter (the memory's new parameters attn.w_b, attn.b_b
included) at 1e-4 for updates 1-2400 and 1e-3 after (switched after the evaluation at 2400); no hinge.
  A_DELTA        test_short_conv.TASK (grouped S=2, P=4, k=2, no conv): S43's own run path (s43c.builder, groups,
                 make_check with S38's decoder at 1200/2400/4800/end, stats explore_common2.stats_b2) with the built
                 model converted by explore_delta_mem.to_delta(beta="learned") (L2 keys, tied write, decay 0.95).
  D_DELTA, D_HEBB  explore_f_tasks.RandHeaderTask() (S=2, P=8, NB=2, blocks 2..6; vocab 26): the same model rule built
                 on this task's vocabulary (explore_local_gate.LocalGateBDH with test_channel_binding's ARCH and arm A,
                 the same seeding as make_model), the same groups and switch, stats explore_f_tasks.rand_header_stats at
                 1200, 2400, 3600 and the end (no decoder: S38's decoder assumes the grouped positions); D_DELTA converted
                 as A_DELTA.
  A_ORACLE_{DELTA,HEBB}, D_ORACLE_{DELTA,HEBB}  the perfect gate (test_channel_binding's 'ceiling' arm, k = 2) on the
                 same task, one Adam at 1e-3 on every parameter (the perfect gate's own recipe, as batch 16's validity arms),
                 delta as above or Hebbian.
"""

import json
import time

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook

import explore_common as ec
import explore_common2 as c2
import explore_delta_mem as dm
import explore_local_gate as s1
import explore_b16_gates as g16
import explore_window_recipe_child as s43c
import explore_f_tasks as ft
import test_binding_onset as tbo
import test_short_conv as tsc
from test_multilayer_binding import build, probe_batch
from test_channel_binding import ARCH, arms as cb_arms
from test_router_reliability import run_one

LR, LR_WARM, WARM = s43c.LR, s43c.LR_WARM, s43c.WARM
RH = ft.RandHeaderTask()
DELTA_KW = dict(beta="learned", normalize=True)
ARM = {
    "A_DELTA": dict(task="grouped", gate="local", delta=True),
    "A_ORACLE_DELTA": dict(task="grouped", gate="perfect", delta=True),
    "A_ORACLE_HEBB": dict(task="grouped", gate="perfect", delta=False),
    "D_DELTA": dict(task="rh", gate="local", delta=True),
    "D_HEBB": dict(task="rh", gate="local", delta=False),
    "D_ORACLE_DELTA": dict(task="rh", gate="perfect", delta=True),
    "D_ORACLE_HEBB": dict(task="rh", gate="perfect", delta=False),
}
TASKS = {"grouped": ec.TASK, "rh": RH}
STATS_RH = ft.rand_header_stats(c2.ROUTE_AT)


def local_model(task, a, seed):
    """explore_local_gate.make_model on any task's vocabulary (the same seeding and constructor), W_h frozen."""
    torch.manual_seed(seed)
    m = s1.LocalGateBDH(task.vocab, ARCH["n_layer"], a["gate"], a["n_ch"], ARCH["positional"], gate_to_readout=a["g2r"],
                        h_gate=a["h_gate"], mult=a["mult"], ctx_tokens=task.ctx_tokens, **a.get("model_kw", {}),
                        local=True, width=s1.WIDTH, gconv_seed=seed + s1.GCONV_OFFSET)
    return g16.freeze_wh(m)


def builder(arm_name):
    arm = ARM[arm_name]
    task = TASKS[arm["task"]]

    def b(a, seed):
        def make():
            if arm["gate"] == "perfect":
                m = build(task, a, ARCH, seed)
            elif arm["task"] == "grouped":
                m = s43c.builder(True)(a, seed)()                  # S43's LOCAL3_SLOW model, unchanged
            else:
                m = local_model(task, a, seed)
            return dm.to_delta(m, seed=seed, **DELTA_KW) if arm["delta"] else m
        return make
    return b


def arm_def(arm_name):
    arm = ARM[arm_name]
    if arm["gate"] == "perfect":
        return {a["key"]: a for a in cb_arms(TASKS[arm["task"]])}["ceiling"]
    return s43c.A


def switch_check(holder):
    def check(m, step, d):
        if step >= WARM:
            holder["groups"][1]["lr"] = LR
    return check


def run_kw(arm_name, holder, out):
    arm = ARM[arm_name]
    task = TASKS[arm["task"]]
    kw = dict(task=task, grad_fn=tsc.conv_grad_norms, lr=LR, builder=builder(arm_name))
    if arm["gate"] == "perfect":
        kw["stats_fn"] = tsc.conv_stats if arm["task"] == "grouped" else STATS_RH
        return kw
    kw["run_kw"] = dict(param_groups=s43c.groups(holder))
    if arm["task"] == "grouped":
        kw.update(stats_fn=c2.stats_b2, check=s43c.make_check(True, holder, out, g16.probe_data(ec.TASK)))
    else:
        kw.update(stats_fn=STATS_RH, check=switch_check(holder))
    return kw


@torch.no_grad()
def beta_by_role(m, task, probe):
    pinp, _, roles = probe
    out = []

    def hook(mod, args, kwargs, o):
        b = mod.beta_of(kwargs["V"][:, 0])
        out.append({r: float(b[mask].mean()) for r, mask in roles.items()})
    was = m.training
    m.eval()
    h = m.attn.register_forward_hook(hook, with_kwargs=True)
    try:
        m(pinp)
    finally:
        h.remove()
        m.train(was)
    return out


def run(p):
    arm_name, seed, iters = p["arm"], p["seed"], p["iters"]
    arm = ARM[arm_name]
    task = TASKS[arm["task"]]
    holder, out = {}, {}
    keep = {"at": -1}
    kw = run_kw(arm_name, holder, out)
    rec = run_one(arm_def(arm_name), seed, iters, p.get("eval_every", tbo.EVAL_EVERY), keep=keep, **kw)
    if rec.get("ok"):
        m = keep["model"]
        rec.update(arm=arm_name, task=arm["task"], gate=arm["gate"], memory="delta" if arm["delta"] else "hebb",
                   delta_cfg=getattr(m, "delta_cfg", None), decode=out or None, group_names=holder.get("names"),
                   write_order=ft.write_order_acc(m, task, tbo.eval_batch(task)))
        if arm["gate"] != "perfect":
            rec["fail"] = None if rec["transition"] is not None else ec.fail_class(rec)
            rec["tag"] = ec.tag(rec)
        if arm["delta"]:
            rec["beta_by_role"] = beta_by_role(m, task, probe_batch(task, seed))
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _lrs(arm_name, seed, iters):
    """Each group's lr at every update through the real run path (loss and evaluation stubbed)."""
    lrs, ids = [], {}

    def hook(opt, args, kwargs):
        if not ids:
            ids["groups"] = [[id(q) for q in g["params"]] for g in opt.param_groups]
        lrs.append([g["lr"] for g in opt.param_groups])
    saved = (tbo.evaluate, tbo.logits_of)
    h = register_optimizer_step_pre_hook(hook)
    tbo.evaluate = lambda model, task, data: (0.5, 1.0)
    tbo.logits_of = lambda model, x: (torch.zeros(x.shape[0], x.shape[1], model.lm_head.shape[1]) + model.lm_head.sum() * 0.0)
    keep = {"at": iters}
    holder = {}
    try:
        kw = run_kw(arm_name, holder, {})
        if ARM[arm_name]["task"] == "grouped":
            kw["check"] = s43c.make_check(True, holder, None, None)
        rec = run_one(arm_def(arm_name), seed, iters, keep=keep, **kw)
    finally:
        tbo.evaluate, tbo.logits_of = saved
        h.remove()
    return lrs, ids, keep, holder, rec


def checks(p):
    rows = []
    for nm, v in ft.check_rand_header():
        rows.append((f"RandHeaderTask: {nm}", v))
    # models
    a = s43c.A
    m_ref = s43c.builder(True)(a, 310)()
    md = builder("A_DELTA")(a, 310)()
    pr = dict(m_ref.named_parameters())
    same = all(torch.equal(q, pr[n]) for n, q in md.named_parameters() if n in pr)
    new = [n for n, _ in md.named_parameters() if n not in pr]
    rows.append((f"A_DELTA = S43's LOCAL3_SLOW model + to_delta(learned beta): shares every parameter ({same}), adds {new}; "
                 f"W_h frozen ({not md.W_h.requires_grad}); memory {type(md.attn).__name__} beta {md.attn.beta_mode}, "
                 f"L2 keys {md.attn.normalize}, decay {md.attn.decay}",
                 same and new == ["attn.w_b", "attn.b_b"] and not md.W_h.requires_grad and md.attn.beta_mode == "learned"
                 and md.attn.normalize and md.attn.decay == 0.95))
    dh, dd = builder("D_HEBB")(a, 330)(), builder("D_DELTA")(a, 330)()
    ph = dict(dh.named_parameters())
    same_d = all(torch.equal(q, ph[n]) for n, q in dd.named_parameters() if n in ph)
    # the window gate on RH equals S43's model on the grouped vocabulary in every parameter except the embedding / head
    shapes = {n: tuple(q.shape) for n, q in dh.named_parameters()}
    rows.append((f"D_HEBB / D_DELTA: LOCAL3 on RandHeaderTask's vocabulary ({RH.vocab}; embed {shapes['embed.weight']}, "
                 f"lm_head {shapes['lm_head']}), share every initial parameter ({same_d}); gate group names present",
                 same_d and shapes["embed.weight"] == (RH.vocab, 32) and dh.local and not dh.W_h.requires_grad))
    x = RH.make_batch(4, torch.Generator().manual_seed(5))[0][:, :-1]
    dh.eval()
    with torch.no_grad():
        t = 20
        g0 = dh(x)[2]
        x3 = x.clone(); x3[:, t - 3] = (x3[:, t - 3] + 1) % RH.vocab
        x2 = x.clone(); x2[:, t - 2] = (x2[:, t - 2] + 1) % RH.vocab
        blind = torch.equal(dh(x3)[2][:, t], g0[:, t])
        sees = not torch.equal(dh(x2)[2][:, t], g0[:, t])
    rows.append((f"the window gate on RandHeaderTask ignores token t-3 ({blind}) and sees t-2 ({sees})", blind and sees))
    # groups and lrs through the real path, 2401 updates
    for arm_name in ("A_DELTA", "D_DELTA", "D_HEBB"):
        lrs, ids, keep, holder, rec = _lrs(arm_name, 310, WARM + 1)
        m = keep["model"]
        named = dict(m.named_parameters())
        train = {id(q) for q in m.parameters() if q.requires_grad}
        flat = [i for g in ids.get("groups", []) for i in g]
        cover = len(flat) == len(set(flat)) and set(flat) == train
        gate_ok = set(ids["groups"][0]) == {id(named[n]) for n in s43c.GATE_L}
        mem_rest = all(id(named[n]) in set(ids["groups"][1]) for n in dm.delta_params(m))
        n = len(lrs)
        lr_ok = n == WARM + 1 and lrs[0] == [LR, LR_WARM] and lrs[WARM - 1] == [LR, LR_WARM] and lrs[WARM] == [LR, LR]
        rows.append((f"{arm_name}: two groups cover every trainable parameter once ({cover}); gate group = W_in, W_g, window "
                     f"({gate_ok}); the memory's parameters {dm.delta_params(m)} in the rest group ({mem_rest}); lrs at 1 "
                     f"{lrs[0] if n else '?'}, {WARM} {lrs[WARM - 1] if n >= WARM else '?'}, {WARM + 1} "
                     f"{lrs[WARM] if n > WARM else '?'}", cover and gate_ok and mem_rest and lr_ok and rec.get("ok")))
    # the perfect gate on RH routes perfectly (routing statistics)
    ceil = arm_def("D_ORACLE_HEBB")
    mo = build(RH, ceil, ARCH, 330)
    rs = ft.rand_header_routing_stats(mo, RH, probe_batch(RH, 330))
    rows.append((f"the perfect gate on RandHeaderTask: margin {rs['margin']:.4f}, eta2 by stream at KEY / VAL "
                 f"{rs['eta_key_by_stream']:.4f} / {rs['eta_val_by_stream']:.4f}",
                 abs(rs["margin"] - 1) < 1e-6 and abs(rs["eta_key_by_stream"] - 1) < 1e-6 and abs(rs["eta_val_by_stream"] - 1) < 1e-6))
    return [[n, bool(v)] for n, v in rows]


def timing(p):
    stamps = []
    h = register_optimizer_step_post_hook(lambda o, a, k: stamps.append(time.perf_counter()))
    try:
        run(dict(arm=p["arm"], seed=0, iters=p["steps"], eval_every=10 ** 9))
    finally:
        h.remove()
    dd = sorted(y - x for x, y in zip(stamps, stamps[1:]))
    arm = ARM[p["arm"]]
    task = TASKS[arm["task"]]
    m = builder(p["arm"])(arm_def(p["arm"]), 0)()
    holder, out = {}, {}
    kw = run_kw(p["arm"], holder, out)
    t0 = time.time()
    tbo.evaluate(m, task, tbo.eval_batch(task))
    kw["stats_fn"](m, task, probe_batch(task, 0), 2400)
    if arm["task"] == "grouped" and arm["gate"] == "local":
        g16.decode(m, task, g16.probe_data(task))
    return [dd[len(dd) // 2], time.time() - t0]
