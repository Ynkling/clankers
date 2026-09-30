#!/usr/bin/env python
"""
explore_slow_mem.py — EXPLORATORY, not a result. Screen S5 (batch 2): a slow memory at first.
Arm A with dense codes; the memory (every non-gate parameter) learns at a tenth of the lr for the
first 2400 updates while the gate learns at the full lr.

BACKGROUND (post hoc; see explore_kwta_warm's docstring for the numbers). Two readings of
KWTA16's early routing:
  (a) sparse codes: near-orthogonal key codes leave the stream split as the only useful one;
  (b) a race: a memory that learns slowly at first cannot exploit a non-stream split before the
      gate finds the stream split.
S5 separates them: (a) predicts that slow_mem does not raise routing at 1200; (b) predicts that
it does. The reading printed (fixed before any run) uses ROUTED* at 1200 paired with X's arm A,
b = SLOW_MEM-only seeds, c = A-only seeds, exact two-sided McNemar p, the screen rule's bands:
  (b) fits   b - c >= 4 and p < 0.10
  (a) fits   b - c <= 0
  neither    otherwise

THE CHANGE. Arm A (build(), dense codes, the recurrent gate), trained by explore_common's recipe
(MAX_ITERS 24000, evaluation every 1200, early stop after 3 evaluations >= 0.95) with ONE Adam
optimizer and two parameter groups; moments carry over:
  - gate group: W_in, W_h, W_g, at lr 1e-3 (test_channel_binding.SUB_LR) throughout;
  - every other parameter, the shared embedding included: lr 1e-4 (SUB_LR / 10) for updates
    1-2400, then 1e-3 from update 2401. run_one's per-evaluation callback sets the group's lr
    right after the evaluation at step 2400 (and after every later evaluation; idempotent).
Seeds 160-179, paired with X's recorded arm A.

CHECKS (explore_batch2 runs them before anything else): through run_job's own code path for the
full 24000 steps, with an optimizer step pre-hook recording each group's lr at every update (the
per-step loss stubbed to a cheap zero function and every held-out accuracy to 0.5, so there is no
early stop and the schedule runs in seconds; the statistics still run on the real model):
  - exactly two groups; the gate group holds exactly W_in, W_h, W_g; the other every other
    parameter (the embedding included);
  - the gate group is at 1e-3 at every update;
  - the other group is at 1e-4 for updates 1-2400 and 1e-3 for updates 2401-24000.
  And the model's parameters equal arm A's (build()) for the same seed.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

import explore_common as ec
import explore_common2 as c2
import test_binding_onset as tbo
import test_short_conv as tsc
from test_multilayer_binding import build
from test_channel_binding import ARCH
from test_router_discovery import GATE_PARAMS

NAME = "slow_mem"
IDEA = "slow memory first: non-gate parameters at lr/10 for 2400 updates, the gate at full lr"
SOURCE = ("the race reading of batch 1's KWTA16 (two-timescale training: fast router, slow "
          "experts); separates it from the sparse-code reading")
CHANGE = ("arm A, one Adam, two groups: gate (W_in, W_h, W_g) 1e-3 throughout; the rest 1e-4 for "
          "updates 1-2400, 1e-3 after")
PAIRING = ("seeds 160-179, paired with X's recorded arm A (test_short_conv, lr 1e-3; same initial "
           "parameters and batches); X's ROUTED* at 2400 from explore_ref_a2400")
GATE = ("W_in", "W_h", "W_g")
assert tuple(GATE) == tuple(GATE_PARAMS)
LR_GATE = ec.SUB_LR                  # 1e-3
LR_REST_WARM = ec.SUB_LR / 10        # 1e-4
WARM_UPDATES = 2400

ARMS = {
    "SLOW_MEM": dict(ec.ARM_A, key="SLOW_MEM", lr=ec.SUB_LR, iters=ec.MAX_ITERS, seeds=ec.SEEDS,
                     label="arm A; memory at lr/10 for the first 2400 updates",
                     lr_gate=LR_GATE, lr_rest_warm=LR_REST_WARM, warm_updates=WARM_UPDATES,
                     sched=f"gate W_in/W_h/W_g {LR_GATE:g} throughout; every other parameter "
                           f"{LR_REST_WARM:g} for updates 1-{WARM_UPDATES}, {ec.SUB_LR:g} after",
                     prio=2),
}


def builder(a, seed):
    return lambda: build(ec.TASK, a, ARCH, seed)


def make_groups(holder):
    def param_groups(model):
        named = dict(model.named_parameters())
        gate = [named[n] for n in GATE]
        rest = [p for n, p in model.named_parameters() if n not in GATE]
        holder["names"] = ([n for n in GATE], [n for n, _ in model.named_parameters()
                                                 if n not in GATE])
        holder["groups"] = [dict(params=gate, lr=LR_GATE), dict(params=rest, lr=LR_REST_WARM)]
        return holder["groups"]
    return param_groups


def make_check(holder):
    def check(m, step, data):
        if step >= WARM_UPDATES:
            holder["groups"][1]["lr"] = ec.SUB_LR
    return check


def run_path(a, seed, iters, keep=None, holder=None):
    holder = {} if holder is None else holder
    return ec.run_one(a, seed, iters, check=make_check(holder), keep=keep, task=ec.TASK,
                      stats_fn=c2.stats_b2, grad_fn=tsc.conv_grad_norms, lr=a["lr"],
                      builder=builder, run_kw=dict(param_groups=make_groups(holder)))


def run_job(arm, seed):
    a = ARMS[arm]
    return run_path(a, seed, a["iters"])


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], ec.TASK, builder(a, 0), a["lr"], c2.stats_b2)]


def check():
    ok = True
    seed = 160
    a = ARMS["SLOW_MEM"]
    ref = build(ec.TASK, ec.ARM_A, ARCH, seed)
    m0 = builder(a, seed)()
    pr = dict(ref.named_parameters())
    same = all(torch.equal(p, pr[n]) for n, p in m0.named_parameters())
    lrs, groups = [], {}

    def hook(opt, args, kwargs):
        if not groups:
            groups["ids"] = [[id(p) for p in g["params"]] for g in opt.param_groups]
        lrs.append([g["lr"] for g in opt.param_groups])

    saved = (tbo.evaluate, tbo.logits_of)
    h = register_optimizer_step_pre_hook(hook)
    tbo.evaluate = lambda model, task, data: (0.5, 1.0)
    tbo.logits_of = lambda model, x: (torch.zeros(x.shape[0], x.shape[1], model.lm_head.shape[1])
                                      + model.lm_head.sum() * 0.0)
    keep, holder = {"at": 1}, {}
    try:
        r = run_path(a, seed, ec.MAX_ITERS, keep=keep, holder=holder)
    finally:
        tbo.evaluate, tbo.logits_of = saved
        h.remove()
    m = keep["model"]
    named = dict(m.named_parameters())
    two = len(groups.get("ids", [])) == 2
    gate_ok = two and set(groups["ids"][0]) == {id(named[n]) for n in GATE} \
        and len(groups["ids"][0]) == 3
    rest_ok = two and set(groups["ids"][1]) == {id(p) for n, p in named.items() if n not in GATE} \
        and id(named["embed.weight"]) in set(groups["ids"][1])
    n = len(lrs)
    g_ok = n == ec.MAX_ITERS and all(x[0] == LR_GATE for x in lrs)
    r_ok = n == ec.MAX_ITERS and all(x[1] == (LR_REST_WARM if i < WARM_UPDATES else ec.SUB_LR)
                                     for i, x in enumerate(lrs))
    for nm, v in (("parameters equal arm A's (build) for the same seed", same),
                  (f"run completed all {ec.MAX_ITERS} updates under the stubs ({n} recorded, "
                   f"ok={r.get('ok')})", n == ec.MAX_ITERS and r.get("ok")),
                  (f"gate group holds exactly W_in, W_h, W_g ({holder.get('names', [[]])[0]})",
                   gate_ok),
                  (f"other group holds every other parameter, embed.weight included "
                   f"({len(holder.get('names', [[], []])[1])} tensors)", rest_ok),
                  (f"gate group lr {LR_GATE:g} at every update", g_ok),
                  (f"other group lr {LR_REST_WARM:g} for updates 1-{WARM_UPDATES}, {ec.SUB_LR:g} "
                   f"for {WARM_UPDATES + 1}-{ec.MAX_ITERS} (update 2400: {lrs[2399] if n > 2399 else '?'}, "
                   f"2401: {lrs[2400] if n > 2400 else '?'}, 24000: {lrs[-1] if lrs else '?'})", r_ok)):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def report(store, refs):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    c2.per_seed_table2(me, store, refs=refs)
    s = ARMS["SLOW_MEM"]["seeds"]
    print("  paired with X's arm A (same seeds, initial parameters and batches):")
    d = ec.paired_vs_recorded(store, "SLOW_MEM", s, ec.discovered, label="DISCOVERED")
    ec.paired_vs_recorded(store, "SLOW_MEM", s, ec.bound, label="BOUND")
    rt = {}
    for step in (1200, 2400, "end"):
        rt[step] = c2.paired_routed(store, "SLOW_MEM", s, step, refs)
    print(f"    failures SLOW_MEM {ec.fail_counts(store, 'SLOW_MEM', s)}   "
          f"X arm A {ec.recorded_fail_counts('A', s)}")
    b = c2.band(rt[1200])
    reading = {"up": "(b) a race", "not up": "(a) sparse codes", "middle": "neither"}[b]
    print(f"  READING (ROUTED* at 1200 vs X's arm A: {rt[1200]['b']} vs {rt[1200]['c']}, "
          f"p = {rt[1200]['p']:.3g}): the S5 pattern fits {reading}")
    d["reading"] = reading
    return d
