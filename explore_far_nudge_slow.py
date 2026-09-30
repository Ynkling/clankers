#!/usr/bin/env python
"""
explore_far_nudge_slow.py — EXPLORATORY, not a result. Screen S15 (batch 5): batch 4's far_nudge
with the SLOW_MEM schedule.

BACKGROUND. On the header layout (explore_far_cue.HeaderTask) arm A's gate can express the routing
(batch 4, S10(a): argmax 1.000), and the 5% labelled nudge let it route on 7/10 seeds by the end,
but only 5/10 bound: routed runs stalled at 0.52-0.55 (far_nudge 164, 165; 161 stream-partial).
SLOW_MEM (a slow memory for the first 2400 updates) raised arm A's discovery on the grouped layout
(24/40 vs 12/40, batches 2-3).

THE ARM (FAR_NUDGE_SLOW). far_nudge exactly (arm A on the header layout; the nudge pre-fit,
test_router_discovery.nudge, unchanged, inside make_model), trained with explore_slow_mem's
param groups and lr switch: the gate (W_in, W_h, W_g) at 1e-3 = test_channel_binding.SUB_LR
throughout; every other parameter, the embedding included, at 1e-4 for updates 1-2400 and 1e-3
after. MAX_ITERS 24000; statistics explore_far_cue.header_stats. Seeds 160-169, paired with batch
4's far_nudge (the same weights after the pre-fit, the same batches).

RULE (fixed before any run), outcome DISCOVERED: promising if >= 8/10; not if <= 5/10; otherwise
inconclusive. Also printed: ROUTED* at 1200 and at the end, and the stalls (routed at the end or
stream-partial, not bound), each paired with far_nudge.

CHECKS: the weights after the nudge pre-fit equal far_nudge's bit for bit on every seed (sep0 =
0.050, equal to batch 4's recorded sep0); an optimizer step pre-hook over the full 24000-step run
path (stubbed as in S5's check) confirms the run's two groups' lrs: gate 1e-3 at every update,
the rest 1e-4 for updates 1-2400 and 1e-3 after, the gate group exactly W_in, W_h, W_g.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

import explore_common as ec
import explore_common2 as c2
import explore_slow_mem as s5
import explore_far_cue as s6
import explore_far_express as s10
import test_binding_onset as tbo
import test_short_conv as tsc
from test_multilayer_binding import probe_batch
from test_router_confirm import mcnemar_exact
from test_router_layout import fail_class

NAME = "far_nudge_slow"
IDEA = "the 5% labelled nudge on the header layout, trained with the SLOW_MEM schedule"
SOURCE = "batch 4's far_nudge (routed 7/10, bound 5/10) and batches 2-3's SLOW_MEM"
CHANGE = ("far_nudge + SLOW_MEM's groups: gate 1e-3 throughout; everything else 1e-4 for updates "
          "1-2400, 1e-3 after; nudge pre-fit unchanged")
PAIRING = "seeds 160-169, paired with batch 4's far_nudge (same weights after the pre-fit, same batches)"
SEEDS = tuple(range(160, 170))
B4_STORE = "far_express"
PROMISING_N, NOT_N = 8, 5

ARMS = {
    "FAR_NUDGE_SLOW": dict(s10.ARMS["far_nudge"], key="FAR_NUDGE_SLOW", seeds=SEEDS, prio=1,
                           lr=ec.SUB_LR, lr_gate=s5.LR_GATE, lr_rest_warm=s5.LR_REST_WARM,
                           warm_updates=s5.WARM_UPDATES,
                           label="far_nudge + the SLOW_MEM schedule, header layout",
                           sched="nudge pre-fit unchanged; then " + s5.ARMS["SLOW_MEM"]["sched"]),
}


def run_path(a, seed, iters, keep=None, holder=None):
    holder = {} if holder is None else holder
    return ec.run_one(a, seed, iters, check=s5.make_check(holder), keep=keep, task=s6.HEADER,
                      stats_fn=s6.header_stats, grad_fn=tsc.conv_grad_norms, lr=a["lr"],
                      builder=s10.builder, run_kw=dict(param_groups=s5.make_groups(holder)))


def run_job(arm, seed):
    a = ARMS[arm]
    return run_path(a, seed, a["iters"])


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], s6.HEADER, s10.builder(a, 0), a["lr"], s6.header_stats)]


def check():
    ok = True
    a = ARMS["FAR_NUDGE_SLOW"]
    b4 = ec.load_store(B4_STORE)["runs"]
    same_all, seps = True, []
    for s in SEEDS:
        m1 = s10.builder(s10.ARMS["far_nudge"], s)()
        m2 = s10.builder(a, s)()
        p1 = dict(m1.named_parameters())
        same_all &= all(torch.equal(p, p1[n]) for n, p in m2.named_parameters())
        sep = s6.header_stats(m2, s6.HEADER, probe_batch(s6.HEADER, s), 0)["sep"]   # as recorded
        seps.append(sep)
        same_all &= sep == b4[f"far_nudge|{s}"]["stats"][0]["sep"]
    # the lr schedule through the run path (stubbed as S5's check); the run's optimizer is the one
    # with two groups (the nudge pre-fit's own Adam has one)
    lrs, groups = [], {}

    def hook(opt, args, kwargs):
        if len(opt.param_groups) != 2:
            return
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
        r = run_path(a, 160, ec.MAX_ITERS, keep=keep, holder=holder)
    finally:
        tbo.evaluate, tbo.logits_of = saved
        h.remove()
    m = keep["model"]
    named = dict(m.named_parameters())
    gate_ok = bool(groups) and set(groups["ids"][0]) == {id(named[n]) for n in s5.GATE}
    rest_ok = bool(groups) and set(groups["ids"][1]) == {id(p) for n, p in named.items() if n not in s5.GATE}
    n = len(lrs)
    g_ok = n == ec.MAX_ITERS and all(x[0] == s5.LR_GATE for x in lrs)
    r_ok = n == ec.MAX_ITERS and all(x[1] == (s5.LR_REST_WARM if i < s5.WARM_UPDATES else ec.SUB_LR)
                                     for i, x in enumerate(lrs))
    for nm, v in ((f"weights after the nudge pre-fit equal far_nudge's bit for bit on seeds 160-169, and "
                   f"sep0 equals batch 4's recorded sep0 (sep0 {min(seps):.3f}-{max(seps):.3f})", same_all),
                  (f"the run completed all {ec.MAX_ITERS} updates under the stubs ({n} recorded on the "
                   f"two-group optimizer, ok={r.get('ok')})", n == ec.MAX_ITERS and r.get("ok")),
                  ("gate group holds exactly W_in, W_h, W_g; the other group every other parameter",
                   gate_ok and rest_ok),
                  (f"gate group lr {s5.LR_GATE:g} at every update", g_ok),
                  (f"other group lr {s5.LR_REST_WARM:g} for updates 1-{s5.WARM_UPDATES}, {ec.SUB_LR:g} "
                   f"after (update 2400: {lrs[2399] if n > 2399 else '?'}, 2401: "
                   f"{lrs[2400] if n > 2400 else '?'})", r_ok)):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def stall(r):
    return bool(r and r.get("ok") and not ec.bound(r)
                and (c2.routed_at(r, "end") or fail_class(r) == "STREAM-PARTIAL"))


def rule(n):
    return "promising" if n >= PROMISING_N else ("not" if n <= NOT_N else "inconclusive")


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    b4 = ec.load_store(B4_STORE)["runs"]
    print(f"  per seed (ROUTED* at 1200/2400/3600/end; b4 = batch 4's far_nudge):")
    print(f"    {'seed':>4} {'sep0':>6} {'acc':>6} {'trans':>6} {'VALcos':>7} {'margin':>6} "
          f"{'eta key s/k/h/i':>20} {'ROUTED*':>8}  {'outcome':<15} {'b4 far_nudge':<15} {'b4 acc':>6}")
    runs = {}
    for s in SEEDS:
        r = store["runs"].get(f"FAR_NUDGE_SLOW|{s}")
        o = b4.get(f"far_nudge|{s}")
        if r is None or not r.get("ok"):
            print(f"    {s:>4} {ec.tag(r)}")
            continue
        runs[s] = r
        m = r["end"].get("margin")
        print(f"    {s:>4} {r['stats'][0].get('sep', float('nan')):6.3f} {r['acc']:6.3f} "
              f"{str(r['transition']):>6} {r['val_cos']:7.3f} {'--' if m is None else f'{m:6.2f}':>6} "
              f"{ec.eta_str(r):>20} {c2.rflags(r):>8}  {ec.tag(r):<15} {ec.tag(o):<15} "
              f"{o['acc'] if o else float('nan'):6.3f}")

    def pair(fn, what):
        b = sum(1 for s, r in runs.items() if fn(r) and not fn(b4.get(f"far_nudge|{s}")))
        c = sum(1 for s, r in runs.items() if fn(b4.get(f"far_nudge|{s}")) and not fn(r))
        nn = sum(bool(fn(r)) for r in runs.values())
        no = sum(bool(fn(b4.get(f"far_nudge|{s}"))) for s in runs)
        print(f"    {what:<13} FAR_NUDGE_SLOW {nn:>2}/{len(runs)}   far_nudge {no:>2}/{len(runs)}   "
              f"FAR_NUDGE_SLOW only {b}, far_nudge only {c}, McNemar two-sided p = {mcnemar_exact(b, c):.3g}")
        return nn

    print("  paired with batch 4's far_nudge:")
    nd = pair(ec.discovered, "DISCOVERED")
    pair(ec.bound, "BOUND")
    pair(lambda r: c2.routed_at(r, 1200), "ROUTED*@1200")
    pair(lambda r: c2.routed_at(r, "end"), "ROUTED*@end")
    pair(stall, "STALLS")
    fails = {}
    for r in runs.values():
        if not ec.bound(r):
            fails[fail_class(r)] = fails.get(fail_class(r), 0) + 1
    print(f"    stalls (routed at the end or stream-partial, not bound): FAR_NUDGE_SLOW "
          f"{[s for s, r in runs.items() if stall(r)]}   far_nudge "
          f"{[s for s in runs if stall(b4.get(f'far_nudge|{s}'))]}")
    print(f"    failures FAR_NUDGE_SLOW {fails}")
    return dict(n=len(runs), discovered=nd, rule=rule(nd))
