#!/usr/bin/env python
"""
explore_kwta_warm.py — EXPLORATORY, not a result. Screen S4 (batch 2): k-WTA as a warm-up only.
kWTA (K_TOP = 16 of 256) for the first 2400 updates and the evaluation at step 2400; dense codes
(K_TOP = N, i.e. arm A's forward) after that.

BACKGROUND (post hoc, from batch 1 and X's recorded arm A in test_short_conv)
- ROUTED* (margin >= 0.9 and eta_key_by_stream > 0.9) at step 1200 -> at the end: arm A 6 -> 7,
  LOCAL3 16 -> 16, KWTA16 15 -> 14, AUX1 0 -> 1. The split is settled by 1200.
- KWTA16 left the gate unchanged, yet 15/20 runs were ROUTED* at 1200 (vs arm A: 12 vs 3,
  p = 0.035). It bound only 2/20 because kWTA slows the memory for the whole run: 12 runs routed
  and did not bind; the kWTA ceiling bound 2/3, at steps 14400 and 21600.

QUESTIONS. Routing at 1200 matches batch 1's KWTA16 by construction (the run is bit-identical to
it through the evaluation at step 2400). Does the routing survive the switch to dense codes (a
mid-run change broke routing in test_load_curriculum), and do the runs then bind?

THE CHANGE. explore_kwta's KWTA16 model (arm A + kWTA 16/256 on x_sparse at every layer), trained
by explore_common's recipe (lr test_channel_binding.SUB_LR = 1e-3, passed explicitly; MAX_ITERS
24000). run_one's per-evaluation callback sets model.k_top = None right after the evaluation (and
its statistics) at step 2400, so updates 1-2400 and the evaluations at 1200 and 2400 use kWTA, and
every update and evaluation from step 2401 on uses arm A's dense forward (MultiBDH.forward).
Seeds 160-179, paired with X's recorded arm A.

CHECKS (explore_batch2 runs them before anything else):
  - through the evaluation at step 2400 the run is bit-identical to batch 1's KWTA16 on seed 160:
    curve points at 1200 and 2400 equal batch 1's recorded ones and a fresh batch-1-path run's,
    and the weights after update 2400 equal the fresh batch-1-path run's;
  - k_top is 16 at the evaluations at 1200 and 2400 and None after the switch;
  - after the switch, on the same weights, the forward (logits and gates) equals arm A's.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import json

import torch

import explore_common as ec
import explore_common2 as c2
import explore_kwta as kw
import test_short_conv as tsc
from test_multilayer_binding import build
from test_channel_binding import ARCH

NAME = "kwta_warm"
IDEA = "k-WTA (16/256) only as a warm-up: through the evaluation at step 2400, dense after"
SOURCE = ("batch 1's KWTA16 (FlyHash, Dasgupta et al. 2017; k-WTA, Ahmad & Hawkins 2019), "
          "which routed early but bound slowly")
CHANGE = ("explore_kwta's KWTA16 model; model.k_top = None right after the evaluation at step "
          "2400 (arm A's dense forward from update 2401)")
PAIRING = ("seeds 160-179, paired with X's recorded arm A (test_short_conv, lr 1e-3; same initial "
           "parameters and batches); X's ROUTED* at 2400 from explore_ref_a2400")
SWITCH = 2400
K_TOP = kw.K_TOP

ARMS = {
    "KWTA_WARM": dict(ec.ARM_A, key="KWTA_WARM", lr=ec.SUB_LR, iters=ec.MAX_ITERS, seeds=ec.SEEDS,
                      label="arm A + kWTA 16/256 through the evaluation at 2400, dense after",
                      sched=f"lr {ec.SUB_LR:g} throughout; kWTA {K_TOP}/256 for updates 1-{SWITCH} "
                            f"and the evaluations at 1200 and {SWITCH}; dense after",
                      prio=2),
}


def switch(m, step, data):
    """run_one's check: called after each evaluation's statistics."""
    if step == SWITCH:
        m.k_top = None


def run_path(a, seed, iters, keep=None):
    return ec.run_one(a, seed, iters, check=switch, keep=keep, task=ec.TASK,
                      stats_fn=c2.stats_b2, grad_fn=tsc.conv_grad_norms, lr=a["lr"],
                      builder=kw.builder)


def run_job(arm, seed):
    a = ARMS[arm]
    return run_path(a, seed, a["iters"])


def segments(arm):
    a = ARMS[arm]
    return [(min(SWITCH, a["iters"]), ec.TASK, kw.builder(a, 0), a["lr"], c2.stats_b2),
            (max(0, a["iters"] - SWITCH), ec.TASK, lambda: build(ec.TASK, ec.ARM_A, ARCH, 0),
             a["lr"], c2.stats_b2)]


def check():
    ok = True
    seed = 160
    k1, k4 = {"at": SWITCH}, {"at": SWITCH}
    # batch 1's KWTA16 path (explore_common.run_recipe's arguments) with a weight snapshot
    r1 = ec.run_one(kw.ARMS["KWTA16"], seed, SWITCH, keep=k1, task=ec.TASK,
                    stats_fn=tsc.conv_stats, grad_fn=tsc.conv_grad_norms, lr=ec.SUB_LR,
                    builder=kw.builder)
    r4 = run_path(ARMS["KWTA_WARM"], seed, SWITCH, keep=k4)
    with open(ec.store_path("kwta")) as f:
        b1 = json.load(f)["runs"][f"KWTA16|{seed}"]
    want = [c for c in b1["curve"] if c[0] <= SWITCH]
    same_rec = r4["curve"] == want
    same_run = r4["curve"] == r1["curve"]
    same_w = (k1["snap"].keys() == k4["snap"].keys()
              and all(torch.equal(k1["snap"][n], k4["snap"][n]) for n in k1["snap"]))
    ktops = [st.get("k_top", "missing") for st in r4["stats"]] + [r4["end"].get("k_top", "missing")]
    m = k4["model"]
    ktop_ok = ktops == [K_TOP, K_TOP, K_TOP, None] and m.k_top is None
    ref = build(ec.TASK, ec.ARM_A, ARCH, seed)
    ref.load_state_dict(m.state_dict())
    g = torch.Generator().manual_seed(5)
    x, _ = ec.TASK.make_batch(8, g)
    x = x[:, :-1]
    ref.eval(); m.eval()
    with torch.no_grad():
        o_ref, o_m = ref(x), m(x)
        fwd = torch.equal(o_ref[0], o_m[0]) and torch.equal(o_ref[2], o_m[2])
    for nm, v in ((f"curve through {SWITCH} equals batch 1's recorded KWTA16 seed {seed} "
                   f"({r4['curve']})", same_rec),
                  (f"curve through {SWITCH} equals a fresh batch-1-path KWTA16 run", same_run),
                  (f"weights after update {SWITCH} equal the batch-1-path run's", same_w),
                  (f"k_top 16 at steps 0/1200/{SWITCH} and None after the switch ({ktops})",
                   ktop_ok),
                  ("after the switch, same weights: logits and gates equal arm A's", fwd)):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def report(store, refs):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    c2.per_seed_table2(me, store, refs=refs)
    s = ARMS["KWTA_WARM"]["seeds"]
    print("  paired with X's arm A (same seeds, initial parameters and batches):")
    d = ec.paired_vs_recorded(store, "KWTA_WARM", s, ec.discovered, label="DISCOVERED")
    ec.paired_vs_recorded(store, "KWTA_WARM", s, ec.bound, label="BOUND")
    for step in (1200, 2400, "end"):
        c2.paired_routed(store, "KWTA_WARM", s, step, refs)
    print(f"    failures KWTA_WARM {ec.fail_counts(store, 'KWTA_WARM', s)}   "
          f"X arm A {ec.recorded_fail_counts('A', s)}")
    # diagnostic (fixed in the code before any run): does routing survive the switch?
    runs = [store["runs"].get(f"KWTA_WARM|{x}") for x in s]
    runs = [r for r in runs if r and r.get("ok")]
    r24 = [r for r in runs if c2.routed_at(r, SWITCH)]
    r36 = [r for r in r24 if c2.routed_at(r, 3600)]
    rend = [r for r in r24 if c2.routed_at(r, "end")]
    lost = [r["seed"] for r in r24 if not c2.routed_at(r, 3600)]
    print(f"    diagnostic, the switch: ROUTED* at {SWITCH} (kWTA) {len(r24)}/{len(runs)}; of those, "
          f"ROUTED* at 3600 (first dense evaluation) {len(r36)}, at the end {len(rend)}, bound "
          f"{sum(ec.bound(r) for r in r24)}; lost by 3600: seeds {lost}")
    acc = [(r["seed"], dict((c[0], c[1]) for c in r["curve"])) for r in runs]
    print("    diagnostic, held-out accuracy at 2400 -> 3600: " +
          " ".join(f"{sd}:{a.get(2400, float('nan')):.2f}->{a.get(3600, float('nan')):.2f}"
                   for sd, a in acc))
    return d
