#!/usr/bin/env python
"""
explore_window_recipe_child.py — EXPLORATORY, not a result. S43's run code. It runs INSIDE a child process on this
branch's own modules (no main-line hook: batches 1-5 ran on them, and batch 1's LOCAL3 records are this code path's).
Parent: explore_window_recipe.

Configuration: test_short_conv's TASK (S=2, P=4, k=2, grouped, no convolution), 24000 steps, evaluation every 1200,
early stop after 3 evaluations >= 0.95, Adam, lr = test_channel_binding.SUB_LR (1e-3) read and passed explicitly.
  LOCAL3       batch 1's S1 run, unchanged: explore_local_gate.make_model (arm A's parameters + the width-3 window)
               through test_router_reliability.run_one with explore_common.run_recipe's arguments (stats
               test_short_conv.conv_stats), plus the decoder probe in run_one's per-evaluation callback.
  LOCAL3_SLOW  the same model with W_h frozen (it is unused by the window gate) + S5's SLOW schedule (explore_slow_mem:
               one Adam, two groups, moments carried over): gate group W_in, W_g, gate_conv.conv_w at 1e-3
               throughout; every other trainable parameter at 1e-4 for updates 1-2400 and 1e-3 after (the switch in
               run_one's per-evaluation callback after the evaluation at 2400, as S5); stats explore_common2.stats_b2
               (S5's and S13's). No hinge.
Decoder probe (explore_b16_gates.decode: S38's decoder at CTX / KEY / VAL positions, h and u, S38's 512-sequence
probe) after the evaluations at 1200, 2400 and 4800 and at the run's last evaluation ("end"); eval mode, no gradient,
its own generator: training is untouched (CHECK: LOCAL3|160 with the probe reproduces batch 1's record).
"""

import json
import time

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook

import explore_common as ec
import explore_common2 as c2
import explore_local_gate as s1
import explore_b16_gates as g16
import test_binding_onset as tbo
import test_short_conv as tsc
from test_multilayer_binding import build, probe_batch
from test_channel_binding import ARCH
from test_router_reliability import run_one

LR = ec.SUB_LR                       # 1e-3
LR_WARM = LR / 10                    # 1e-4
WARM = 2400
ITERS = ec.MAX_ITERS                 # 24000
GATE_L = ("W_in", "W_g", "gate_conv.conv_w")
AT = (1200, 2400, 4800)
ARM = {"LOCAL3": dict(slow=False), "LOCAL3_SLOW": dict(slow=True)}
A = s1.ARMS["LOCAL3"]                 # dict(arm A, lr SUB_LR, iters MAX_ITERS)


def builder(slow):
    def b(a, seed):
        def make():
            m = s1.make_model(a, seed)                       # batch 1's LocalGateBDH
            return g16.freeze_wh(m) if slow else m
        return make
    return b


def groups(holder):
    def param_groups(model):
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        nm = dict(named)
        gate = [n for n in GATE_L if n in nm]
        rest = [n for n, _ in named if n not in gate]
        holder["names"] = [gate, rest]
        holder["groups"] = [dict(params=[nm[n] for n in gate], lr=LR), dict(params=[nm[n] for n in rest], lr=LR_WARM)]
        return holder["groups"]
    return param_groups


def make_check(slow, holder, out, data):
    def check(m, step, d):
        if slow and step >= WARM:
            holder["groups"][1]["lr"] = LR
        if out is not None:
            dd = g16.decode(m, ec.TASK, data)
            if step in AT:
                out[str(step)] = dd
            out["end"] = dict(dd, step=step)
    return check


def run(p):
    """p: arm, seed, iters, eval_every (timing only), probe (default True)."""
    slow = ARM[p["arm"]]["slow"]
    holder = {}
    out = {} if p.get("probe", True) else None
    data = g16.probe_data(ec.TASK)
    kw = dict(check=make_check(slow, holder, out, data), task=ec.TASK, grad_fn=tsc.conv_grad_norms, lr=LR,
              builder=builder(slow))
    if slow:
        kw.update(stats_fn=c2.stats_b2, run_kw=dict(param_groups=groups(holder)))
    else:
        kw.update(stats_fn=tsc.conv_stats)                   # explore_common.run_recipe's (batch 1's)
    if p.get("keep") is not None:
        kw["keep"] = p["keep"]
    rec = run_one(A, p["seed"], p["iters"], p.get("eval_every", tbo.EVAL_EVERY), **kw)
    if rec.get("ok"):
        rec.update(arm=p["arm"], decode=out, group_names=holder.get("names"), fail=None if rec["transition"] is not None
                   else ec.fail_class(rec), tag=ec.tag(rec))
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _stub_lrs(seed, iters):
    lrs, ids = [], {}

    def hook(opt, args, kwargs):
        if not ids:
            ids["groups"] = [[id(q) for q in g["params"]] for g in opt.param_groups]
        lrs.append([g["lr"] for g in opt.param_groups])
    saved = (tbo.evaluate, tbo.logits_of)
    h = register_optimizer_step_pre_hook(hook)
    tbo.evaluate = lambda model, task, data: (0.5, 1.0)
    tbo.logits_of = lambda model, x: (torch.zeros(x.shape[0], x.shape[1], model.lm_head.shape[1])
                                      + model.lm_head.sum() * 0.0)
    keep = {"at": iters}
    try:
        p = dict(arm="LOCAL3_SLOW", seed=seed, iters=iters, probe=False, keep=keep)
        holder = {}
        kw = dict(check=make_check(True, holder, None, None), task=ec.TASK, grad_fn=tsc.conv_grad_norms, lr=LR,
                  builder=builder(True), stats_fn=c2.stats_b2, run_kw=dict(param_groups=groups(holder)), keep=keep)
        rec = run_one(A, p["seed"], p["iters"], **kw)
    finally:
        tbo.evaluate, tbo.logits_of = saved
        h.remove()
    return lrs, ids, keep, holder, rec


def checks(p):
    rows = []
    # batch 1's own checks (explore_local_gate.check: arm A's parameters, the window's causality)
    rows.append(("batch 1's LOCAL3 checks (explore_local_gate.check: local=False equals arm A, parameters and logits; "
                 "local=True shares arm A's parameters; the gate at t ignores token t-3 and sees t-2)", s1.check()))
    seed = 160
    task = ec.TASK
    # the window at [1, 0, 0] and W_h = 0: the gate equals arm A's with W_h = 0
    ref = build(task, ec.ARM_A, ARCH, seed)
    loc = s1.make_model(A, seed)
    with torch.no_grad():
        ref.W_h.zero_()
        w = torch.zeros_like(loc.gate_conv.conv_w)
        w[:, 0] = 1.0
        loc.gate_conv.conv_w.copy_(w)
    x = task.make_batch(8, torch.Generator().manual_seed(5))[0][:, :-1]
    ref.eval(), loc.eval()
    with torch.no_grad():
        g0, g1 = ref(x)[2], loc(x)[2]
    d = float((g0 - g1).abs().max())
    rows.append((f"with the window weights at [1, 0, 0] (lag 0) and W_h = 0 the LOCAL3 gate equals arm A's gate with W_h = 0 "
                 f"(max |diff| {d:.2e} <= 1e-6; bit-identical {torch.equal(g0, g1)})", d <= 1e-6))
    # the window is explore_b16_gates' (S44's conversion draws the same window)
    rows.append(("explore_b16_gates.window_init(seed) equals batch 1's window for seeds 160 and 199",
                 all(torch.equal(g16.window_init(s), s1.make_model(A, s).gate_conv.conv_w.detach()) for s in (160, 199))))
    # LOCAL3_SLOW's groups and lrs through the real run path (loss and evaluation stubbed), 2401 updates
    lrs, ids, keep, holder, rec = _stub_lrs(seed, WARM + 1)
    m = keep["model"]
    named = dict(m.named_parameters())
    train = {id(q) for q in m.parameters() if q.requires_grad}
    gid = ids.get("groups", [[], []])
    flat = [i for g in gid for i in g]
    cover = len(flat) == len(set(flat)) and set(flat) == train
    gate_ok = set(gid[0]) == {id(named[n]) for n in GATE_L} and len(gid[0]) == 3
    wh = not named["W_h"].requires_grad and id(named["W_h"]) not in set(flat)
    m0 = s1.make_model(A, seed)
    wh_same = torch.equal(m0.W_h, named["W_h"].detach())
    rows.append((f"LOCAL3_SLOW: two groups covering every trainable parameter once ({len(flat)} tensors; gate group "
                 f"{holder['names'][0]}); W_h frozen, in no group, unchanged after {len(lrs)} updates", cover and gate_ok and wh
                 and wh_same and len(holder["groups"]) == 2))
    n = len(lrs)
    lr_ok = (n == WARM + 1 and lrs[0] == [LR, LR_WARM] and lrs[WARM - 1] == [LR, LR_WARM] and lrs[WARM] == [LR, LR])
    rows.append((f"LOCAL3_SLOW lrs [gate, rest] at update 1 {lrs[0] if n else '?'}, {WARM} {lrs[WARM - 1] if n >= WARM else '?'}, "
                 f"{WARM + 1} {lrs[WARM] if n > WARM else '?'} ({n} updates, ok={rec.get('ok')})", lr_ok and rec.get("ok")))
    return [[nm, bool(v)] for nm, v in rows]


def repro(p):
    """LOCAL3|seed with the decoder probe, through p["iters"] steps (the parent compares with batch 1's record)."""
    return run(dict(arm="LOCAL3", seed=p["seed"], iters=p["iters"]))


def timing(p):
    """Per arm: the median interval between optimizer steps over p["steps"] steps (no evaluation), one evaluation with
    its statistics, and one decoder measurement."""
    out = {}
    for arm in p["which"]:
        stamps = []
        h = register_optimizer_step_post_hook(lambda o, a, k: stamps.append(time.perf_counter()))
        try:
            run(dict(arm=arm, seed=0, iters=p["steps"], eval_every=10 ** 9, probe=False))
        finally:
            h.remove()
        dd = sorted(y - x for x, y in zip(stamps, stamps[1:]))
        m = builder(ARM[arm]["slow"])(A, 0)()
        task = ec.TASK
        data, pr = tbo.eval_batch(task), probe_batch(task, 0)
        t0 = time.time()
        tbo.evaluate(m, task, data)
        (c2.stats_b2 if ARM[arm]["slow"] else tsc.conv_stats)(m, task, pr, 2400)
        te = time.time() - t0
        t0 = time.time()
        g16.decode(m, task, g16.probe_data(task))
        out[arm] = [dd[len(dd) // 2], te, time.time() - t0]
    return out
