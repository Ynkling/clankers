#!/usr/bin/env python
"""
explore_far_gates.py — EXPLORATORY, not a result. Screen S8 (batch 3): can a gate find the stream
routing when the stream token is a block header (S6's HeaderTask), out of view for most pairs?

BACKGROUND (batch 2, S6 far_cue, header layout). far_ceil (perfect gate) bound 3/3 at step 1200;
far_A (arm A's recurrent gate) bound 0/10 (2 routed, not bound); far_L3 (LOCAL3) bound 2/10, both
by a key split, 0 routed. With the stream token out of view, neither gate found the stream
routing.

ARMS (header layout, explore_far_cue.HEADER; seeds 160-169, paired with batch 2's far_A on the
same seeds: same initial parameters for everything they share, the same batches)
  far_EMA     explore_gates' fading-memory gate: u_t = W_in v_t; c_t = a c_{t-1} + (1-a) u_t per
              unit, a = sigmoid(theta); theta learnable, initialized so the 32 units' half-lives
              are log-spaced 1..32 tokens (a = 2^(-1/h)), the assignment to units a permutation
              from its own generator; g_t = softmax(W_g tanh(c_t)). W_h unused.
  far_SEL     explore_gates' selective recurrence (minGRU-style, Feng et al. 2024):
              z_t = sigmoid(W_z v_t + b_z), b_z = -1 at init; h_t = (1-z_t) h_{t-1} + z_t W_in v_t;
              g_t = softmax(W_g tanh(h_t)). W_z from its own generator. W_h unused.
  far_A_slow  arm A with S5's schedule (explore_slow_mem's param groups and lr switch, unchanged):
              gate W_in/W_h/W_g at 1e-3 throughout; everything else 1e-4 for updates 1-2400, then
              1e-3.
Recipe otherwise explore_common's: lr test_channel_binding.SUB_LR = 1e-3 (passed explicitly),
MAX_ITERS 24000, evaluation every 1200 on eval_batch(HEADER), early stop after 3 >= 0.95;
statistics explore_far_cue.header_stats (ROUTED* at 1200, 2400, 3600 and the end).

RULE per arm (fixed before any run): DISCOVERED >= 4/10 promising (batch 2's far_A: 0/10);
<= 1/10 not; otherwise inconclusive. Printed alongside: bound, ROUTED* at 1200 and the end,
failure classes, and the per-seed pairing with batch 2's far_A (McNemar).

CHECKS: explore_gates.check_gate on HEADER for both gates (closed form = explicit per-step loop,
causal in train and eval, shared parameters bitwise arm A's, EMA's initial half-lives); far_A_slow's
lr schedule by an optimizer step pre-hook over its full 24000-step run path (as S5's check);
explore_far_cue.check() (the header layout and the perfect gate).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

import explore_common as ec
import explore_common2 as c2
import explore_gates as eg
import explore_slow_mem as s5
import explore_far_cue as s6
import test_binding_onset as tbo
import test_short_conv as tsc
from test_multilayer_binding import build
from test_channel_binding import ARCH
from test_router_confirm import mcnemar_exact
from test_router_layout import fail_class

NAME = "far_gates"
IDEA = "gates with memory beyond a window on the header layout: multi-scale EMA, selective (minGRU)"
SOURCE = ("multi-timescale fading memory (e.g. RetNet / S4-style decays); minGRU (Feng et al. 2024); "
          "S5's slow-memory schedule")
CHANGE = "far_EMA, far_SEL: the gate's recurrence (explore_gates); far_A_slow: S5's schedule"
PAIRING = ("seeds 160-169, paired with batch 2's far_A on the same seeds (explore_out/"
           "far_cue_results.json; same shared initial parameters, same batches)")
SEEDS = tuple(range(160, 170))
B2_STORE = "far_cue"
PROMISING_N, NOT_N = 4, 1

_A = dict(ec.ARM_A, lr=ec.SUB_LR, iters=ec.MAX_ITERS, seeds=SEEDS, prio=1)
ARMS = {
    "far_EMA": dict(_A, key="far_EMA", label="multi-scale EMA gate (half-lives 1..32), header layout"),
    "far_SEL": dict(_A, key="far_SEL", label="selective (minGRU-style) gate, header layout"),
    "far_A_slow": dict(_A, key="far_A_slow", label="arm A + S5's schedule, header layout",
                       lr_gate=s5.LR_GATE, lr_rest_warm=s5.LR_REST_WARM,
                       warm_updates=s5.WARM_UPDATES, sched=s5.ARMS["SLOW_MEM"]["sched"]),
}
KIND = {"far_EMA": "ema", "far_SEL": "sel"}


def builder_for(arm):
    if arm in KIND:
        return eg.builder(s6.HEADER, KIND[arm])
    return s6.builder_for("far_A")


def run_path(arm, seed, iters, keep=None, holder=None):
    a = ARMS[arm]
    kw = dict(keep=keep, task=s6.HEADER, stats_fn=s6.header_stats, grad_fn=tsc.conv_grad_norms,
              lr=a["lr"], builder=builder_for(arm))
    if arm == "far_A_slow":
        holder = {} if holder is None else holder
        return ec.run_one(a, seed, iters, check=s5.make_check(holder),
                          run_kw=dict(param_groups=s5.make_groups(holder)), **kw)
    return ec.run_one(a, seed, iters, **kw)


def run_job(arm, seed):
    return run_path(arm, seed, ARMS[arm]["iters"])


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], s6.HEADER, builder_for(arm)(a, 0), a["lr"], s6.header_stats)]


def schedule_check(seed=160):
    """explore_slow_mem.check's stubbed pre-hook check, through far_A_slow's run path."""
    ok = True
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
        r = run_path("far_A_slow", seed, ec.MAX_ITERS, keep=keep, holder=holder)
    finally:
        tbo.evaluate, tbo.logits_of = saved
        h.remove()
    m = keep["model"]
    named = dict(m.named_parameters())
    ref = build(s6.HEADER, ec.ARM_A, ARCH, seed)
    same = all(torch.equal(p, dict(ref.named_parameters())[n]) for n, p in
               builder_for("far_A_slow")(ARMS["far_A_slow"], seed)().named_parameters())
    two = len(groups.get("ids", [])) == 2
    gate_ok = two and set(groups["ids"][0]) == {id(named[n]) for n in s5.GATE}
    rest_ok = two and set(groups["ids"][1]) == {id(p) for n, p in named.items() if n not in s5.GATE}
    n = len(lrs)
    g_ok = n == ec.MAX_ITERS and all(x[0] == s5.LR_GATE for x in lrs)
    r_ok = n == ec.MAX_ITERS and all(
        x[1] == (s5.LR_REST_WARM if i < s5.WARM_UPDATES else ec.SUB_LR) for i, x in enumerate(lrs))
    for nm, v in (("far_A_slow: parameters equal arm A's (build on HEADER) for the same seed", same),
                  (f"far_A_slow: run completed all {ec.MAX_ITERS} updates under the stubs ({n} "
                   f"recorded, ok={r.get('ok')})", n == ec.MAX_ITERS and r.get("ok")),
                  ("far_A_slow: gate group holds exactly W_in, W_h, W_g", gate_ok),
                  ("far_A_slow: other group holds every other parameter", rest_ok),
                  (f"far_A_slow: gate group lr {s5.LR_GATE:g} at every update", g_ok),
                  (f"far_A_slow: other group lr {s5.LR_REST_WARM:g} for updates 1-{s5.WARM_UPDATES}, "
                   f"{ec.SUB_LR:g} after (update 2400: {lrs[2399] if n > 2399 else '?'}, 2401: "
                   f"{lrs[2400] if n > 2400 else '?'}, 24000: {lrs[-1] if lrs else '?'})", r_ok)):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def check():
    ok = True
    ok &= eg.check_gate(s6.HEADER, "ema", f"{NAME}/far_EMA")
    ok &= eg.check_gate(s6.HEADER, "sel", f"{NAME}/far_SEL")
    ok &= schedule_check()
    print("  (the header layout: explore_far_cue's CHECKs)")
    ok &= s6.check()
    return ok


def rule(n):
    return "promising" if n >= PROMISING_N else ("not" if n <= NOT_N else "inconclusive")


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    b2 = ec.load_store(B2_STORE)["runs"]
    print(f"  per seed (ROUTED* = margin >= {c2.ROUTE_MARGIN} and eta_key_by_stream > {c2.ROUTE_ETA}, "
          f"at 1200/2400/3600/end; b2 columns are batch 2's far_A and far_L3 on the same seeds):")
    for k, a in ARMS.items():
        print(f"   arm {k} (lr {a['lr']:g}; {c2.sched(a)})")
        print(f"    {'seed':>4} {'acc':>6} {'trans':>6} {'VALcos':>7} {'margin':>6} "
              f"{'eta key s/k/h/i':>20} {'ROUTED*':>8}  {'outcome':<15} {'b2 far_A':<15} {'b2 far_L3':<15}")
        for s in a["seeds"]:
            r = store["runs"].get(f"{k}|{s}")
            tags = f"{ec.tag(b2.get(f'far_A|{s}')):<15} {ec.tag(b2.get(f'far_L3|{s}')):<15}"
            if r is None or not r.get("ok"):
                print(f"    {s:>4} {ec.tag(r):<60} {tags}")
                continue
            m = r["end"].get("margin")
            print(f"    {s:>4} {r['acc']:6.3f} {str(r['transition']):>6} {r['val_cos']:7.3f} "
                  f"{'--' if m is None else f'{m:6.2f}':>6} {ec.eta_str(r):>20} {c2.rflags(r):>8}  "
                  f"{ec.tag(r):<15} {tags}")
    out = {}
    bA = [b2.get(f"far_A|{s}") for s in SEEDS]
    print(f"  batch 2's far_A on seeds 160-169: DISCOVERED {sum(ec.discovered(r) for r in bA)}/10, "
          f"bound {sum(ec.bound(r) for r in bA)}/10")
    for k in ARMS:
        runs = {s: store["runs"].get(f"{k}|{s}") for s in SEEDS}
        runs = {s: r for s, r in runs.items() if r and r.get("ok")}
        nd = sum(ec.discovered(r) for r in runs.values())
        fails = {}
        for r in runs.values():
            if not ec.bound(r):
                fails[fail_class(r)] = fails.get(fail_class(r), 0) + 1
        b = sum(1 for s, r in runs.items() if ec.discovered(r) and not ec.discovered(b2.get(f"far_A|{s}")))
        c = sum(1 for s, r in runs.items() if ec.discovered(b2.get(f"far_A|{s}")) and not ec.discovered(r))
        print(f"    {k:<10} DISCOVERED {nd}/{len(runs)}   bound {sum(ec.bound(r) for r in runs.values())}"
              f"/{len(runs)}   ROUTED*@1200 {sum(c2.routed_at(r, 1200) for r in runs.values())}"
              f"/{len(runs)}   ROUTED*@end {sum(c2.routed_at(r, 'end') for r in runs.values())}"
              f"/{len(runs)}   failures {fails}   vs b2 far_A (DISCOVERED): {k} only {b}, far_A only "
              f"{c}, McNemar p = {mcnemar_exact(b, c):.3g}")
        out[k] = dict(n=len(runs), discovered=nd, rule=rule(nd))
    return out
