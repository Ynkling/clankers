#!/usr/bin/env python
"""
explore_far_single.py — EXPLORATORY, not a result. Screen S23 (batch 8): the missing control — a
single channel (no partition) on the header layout.

BACKGROUND (post hoc, from batch 7): S19's FAR_TOK bound 9/10 header-layout runs, but in 3 of them
(161, 162, 165) every eta^2 is near 0 and VAL cos is 0.74-1.00, i.e. the bound model barely uses the
partition. No single-channel run exists on the header layout; on the grouped layout X's arm B (plain
single channel) bound 0/20.

THE ARMS (header layout, explore_far_cue.HEADER; model test_multilayer_binding.build with
test_short_conv's arm B: gate 'none', n_ch 1 (no gate parameters), mult 8, no conv; X's recipe
otherwise: Adam, BATCH 32, MAX_ITERS 24000, lr = test_channel_binding.SUB_LR = 1e-3 passed explicitly;
statistics test_short_conv.conv_stats (role cosines only at k = 1))
  B_FAR       lr 1e-3 throughout. Seeds 160-169.
  B_FAR_SLOW  the slow schedule: every parameter at 1e-4 for updates 1-2400, 1e-3 after (there is no
              gate group), set through onset_run's lr_at. Seeds 160-169.
Outcome BOUND (a transition in the held-out curve); per seed the final accuracy and the transition.
Printed alongside (same seeds, same batches): S19's FAR_TOK, batch 3's far_A_slow, batch 2's far_A.

RULES (fixed before any run)
- "header binding needs the partition" if B_FAR and B_FAR_SLOW each bind <= 1/10; "a single channel
  binds the header layout" if either binds >= 3/10; otherwise neither.

CHECKS: B_FAR's model is k = 1 with no gate parameters (no W_in, W_h, W_g; gate 'none'); with the slow
schedule's low lr set to 1e-3, B_FAR_SLOW equals B_FAR bit for bit (curves and weights, 3600 steps);
B_FAR_SLOW's optimizer sees lr 1e-4 on every group for updates 1-2400 and 1e-3 from update 2401 (read
from the optimizer at each step), and its curve differs from B_FAR's.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.optim.optimizer as topt

import explore_common as ec
import explore_common2 as c2
import explore_far_cue as s6
import explore_slow_mem as s5
import test_short_conv as tsc
from test_multilayer_binding import build
from test_channel_binding import ARCH
from test_router_confirm import mcnemar_exact
from test_binding_onset import EVAL_EVERY

NAME = "far_single"
IDEA = "the missing control: a single channel (no gate, no partition) on the header layout"
SOURCE = "batch 7's S19 (FAR_TOK bound 9/10 on the header layout, 3 of them with every eta^2 near 0)"
CHANGE = "test_short_conv's arm B (k = 1, gate 'none') on the header layout; B_FAR_SLOW with the slow schedule"
PAIRING = ("seeds 160-169, printed beside S19's FAR_TOK, batch 3's far_A_slow and batch 2's far_A (same "
           "batches; the initial parameters differ: k = 1 has no gate)")
HEADER = s6.HEADER
SEEDS = tuple(range(160, 170))
LR_LOW, WARM = 1e-4, s5.WARM_UPDATES                     # 1e-4 for updates 1-2400
NEEDS, BINDS = 1, 3
S19_STORE, B3_STORE, B2_STORE = "token_hinge", "far_gates", "far_cue"

_B = dict(tsc.ARM["B"], lr=ec.SUB_LR, iters=ec.MAX_ITERS, seeds=SEEDS, prio=1, layout="header")
ARMS = {
    "B_FAR": dict(_B, key="B_FAR", label="plain single channel (arm B), header layout"),
    "B_FAR_SLOW": dict(_B, key="B_FAR_SLOW", label="arm B with the slow schedule, header layout",
                       lr_low=LR_LOW, sched=f"every parameter at {LR_LOW:g} for updates 1-{WARM}, "
                                            f"{ec.SUB_LR:g} after (onset_run's lr_at)"),
}


def make_lr_at(a, low=None):
    lo = a["lr_low"] if low is None else low

    def lr_at(step):
        return lo if step <= WARM else a["lr"]
    return lr_at


def builder(a, seed):
    return lambda: build(HEADER, a, ARCH, seed)


def run_path(a, seed, iters, keep=None, low=None):
    run_kw = dict(lr_at=make_lr_at(a, low)) if "lr_low" in a else None
    return ec.run_one(a, seed, iters, keep=keep, task=HEADER, stats_fn=tsc.conv_stats,
                      grad_fn=tsc.conv_grad_norms, lr=a["lr"], builder=builder, run_kw=run_kw)


def run_job(arm, seed):
    a = ARMS[arm]
    return run_path(a, seed, a["iters"])


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], HEADER, builder(a, 0), a["lr"], tsc.conv_stats)]


def check():
    ok = True
    rows = []
    it = 3 * EVAL_EVERY
    m = builder(ARMS["B_FAR"], 160)()
    names = sorted(n for n, _ in m.named_parameters())
    gate_free = not any(n in ("W_in", "W_h", "W_g") or n.startswith("W_g") for n in names)
    rows.append((f"B_FAR's model is k = {m.n_ch} with gate '{ARMS['B_FAR']['gate']}' and no gate parameters "
                 f"(parameters: {names})", m.n_ch == 1 and ARMS["B_FAR"]["gate"] == "none" and gate_free))
    k1, k2 = {"at": it}, {"at": it}
    r1 = run_path(ARMS["B_FAR"], 160, it, keep=k1)
    r2 = run_path(ARMS["B_FAR_SLOW"], 160, it, keep=k2, low=ec.SUB_LR)
    same = r1["curve"] == r2["curve"] and all(torch.equal(k1["snap"][n], k2["snap"][n]) for n in k1["snap"])
    rows.append((f"with the slow schedule's low lr set to {ec.SUB_LR:g}, B_FAR_SLOW equals B_FAR bit for bit "
                 f"(curves and weights, seed 160, {it} steps)", same))
    seen = {}

    def hook(opt, args, kwargs):
        seen[len(seen) + 1] = sorted({g["lr"] for g in opt.param_groups})
    handle = topt.register_optimizer_step_pre_hook(hook)
    try:
        r3 = run_path(ARMS["B_FAR_SLOW"], 160, it)
    finally:
        handle.remove()
    sched_ok = (len(seen) == it and all(seen[s] == [LR_LOW] for s in range(1, WARM + 1))
                and all(seen[s] == [ec.SUB_LR] for s in range(WARM + 1, it + 1)))
    rows.append((f"B_FAR_SLOW's optimizer: lr {seen.get(1)} at update 1, {seen.get(WARM)} at {WARM}, "
                 f"{seen.get(WARM + 1)} at {WARM + 1}, {seen.get(it)} at {it} ({len(seen)} steps seen); its curve "
                 f"differs from B_FAR's ({r3['curve'] != r1['curve']})", sched_ok and r3["curve"] != r1["curve"]))
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def rule(nb, ns):
    if nb <= NEEDS and ns <= NEEDS:
        return "header binding needs the partition"
    if nb >= BINDS or ns >= BINDS:
        return "a single channel binds the header layout"
    return "neither"


def runs_of(store_name, key):
    return {int(k.split("|")[1]): r for k, r in ec.load_store(store_name)["runs"].items()
            if k.startswith(key + "|") and r.get("ok")}


def acc_at(r, step):
    for c in r["curve"]:
        if c[0] == step:
            return c[1]
    return None


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    got = {arm: {int(k.split("|")[1]): r for k, r in store["runs"].items()
                 if k.startswith(arm + "|") and r.get("ok")} for arm in ARMS}
    refs = (("S19 FAR_TOK", runs_of(S19_STORE, "FAR_TOK")), ("b3 far_A_slow", runs_of(B3_STORE, "far_A_slow")),
            ("b2 far_A", runs_of(B2_STORE, "far_A")))
    steps = (2400, 4800, 12000)
    print("  per seed (acc at 2400/4800/12000 and at the last evaluation; transition = the bound step):")
    for arm in ARMS:
        print(f"   arm {arm} (lr {ARMS[arm]['lr']:g}; {c2.sched(ARMS[arm])})")
        print(f"    {'seed':>4} {'acc@2400/4800/12000':>21} {'acc end':>8} {'stopped':>8} {'trans':>6}  "
              f"{'outcome':<8} " + " ".join(f"{lab:<15}" for lab, _ in refs))
        for s in SEEDS:
            r = got[arm].get(s)
            tags = " ".join(f"{ec.tag(d.get(s)):<15}" for _, d in refs)
            if r is None:
                print(f"    {s:>4} {'not run':<50} {tags}")
                continue
            ac = "/".join("--" if acc_at(r, t) is None else f"{acc_at(r, t):.3f}" for t in steps)
            print(f"    {s:>4} {ac:>21} {r['acc']:8.3f} {r['stopped_at']:>8} {str(r['transition']):>6}  "
                  f"{'BOUND' if ec.bound(r) else 'unbound':<8} {tags}")
    nb = sum(ec.bound(r) for r in got["B_FAR"].values())
    ns = sum(ec.bound(r) for r in got["B_FAR_SLOW"].values())
    for arm, n in (("B_FAR", nb), ("B_FAR_SLOW", ns)):
        rs = got[arm]
        fin = sorted(r["acc"] for r in rs.values())
        print(f"  {arm}: BOUND {n}/{len(rs)}; final acc min {fin[0] if fin else float('nan'):.3f}, median "
              f"{fin[len(fin) // 2] if fin else float('nan'):.3f}, max {fin[-1] if fin else float('nan'):.3f}")
    b = sum(1 for s, r in got["B_FAR_SLOW"].items() if ec.bound(r) and not ec.bound(got["B_FAR"].get(s)))
    c = sum(1 for s, r in got["B_FAR_SLOW"].items() if ec.bound(got["B_FAR"].get(s)) and not ec.bound(r))
    print(f"  B_FAR_SLOW vs B_FAR (paired, BOUND): B_FAR_SLOW only {b}, B_FAR only {c}, McNemar p = "
          f"{mcnemar_exact(b, c):.3g}")
    for lab, d in refs:
        print(f"  {lab}: BOUND {sum(ec.bound(d.get(s)) for s in SEEDS)}/{sum(1 for s in SEEDS if s in d)} on these seeds")
    out = dict(b_far=dict(n=len(got["B_FAR"]), bound=nb), b_far_slow=dict(n=len(got["B_FAR_SLOW"]), bound=ns),
               rule=rule(nb, ns))
    print(f"  RULE (B_FAR {nb}/{len(got['B_FAR'])}, B_FAR_SLOW {ns}/{len(got['B_FAR_SLOW'])} BOUND; 'needs the "
          f"partition' if each <= {NEEDS}/10, 'a single channel binds' if either >= {BINDS}/10): {out['rule']}")
    return out
