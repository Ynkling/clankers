#!/usr/bin/env python
"""
explore_reservoir_gate.py — EXPLORATORY, not a result. Screen S45 (batch 16): a frozen-reservoir gate at eight streams.

BACKGROUND: batch 15's S42 (the stream leaves the gate state when rho(W_h) passes 1, in 7 of 10 runs) and S41 (a cap on
sigma_max(W_h) shrank W_h 2.2x at update 1); the audit of 6 October 2026: W_h ~ N(0, 0.1^2) gives rho ~ 0.6 and
sigma_max ~ 1.08 at init; an echo-state reservoir keeps the recurrence contractive by construction.

ARMS (explore_reservoir_gate_child; S=8, P=4, k=16, conv, all 8 streams from step 1, 28800 steps, seeds 260-269):
  RES_D8_A  arm A's gate with W_h frozen at its initial value rescaled to spectral radius 0.5; W_in, W_g trained;
            + SLOW + main's HINGE (Adam 1e-3; the slow groups without W_h); paired with X's HINGE_D8 (0/10).
  RES_D8_M  the same with S25's groups at Muon lr 0.005; paired with S44's D8_HINGE_M_REF (S33's D8_HINGE, fresh on
            this CPU).
READING (as S44(b); BOUND ROUTED at 28800; fixed before any run): "breaks the eight-stream stall" if either arm has
>= 3/10; "does not" if both have 0/10; otherwise neither (an arm cut by the runtime rule drops out).
DIAGNOSTICS: as S44(b) (decodability at CTX/KEY/VAL at 1200/2400/4800/9600, channel maps, merges, transitions) plus
rho(W_h) and sigma_max(W_h) at those steps (should be constant).
CHECKS: W_h receives no update over 50 steps (Adam and Muon); rho(W_h) = 0.5 +- 1e-6 at init and after 50 steps; every
other parameter's initial value equals the run path's model's; the groups cover every trainable parameter once, W_h in
none.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common16 as c16
import explore_b16_report as rp
import explore_window_scale as s44

NAME = "reservoir_gate"
CHILD = "explore_reservoir_gate_child"
MAIN = True
IDEA = "a frozen reservoir gate: W_h fixed at its init rescaled to spectral radius 0.5 (echo-state), W_in and W_g trained"
SOURCE = "the audit of 6 October 2026, section 4 (LRU / contractive recurrences); batch 15's S41 and S42"
CHANGE = "arm A's gate with W_h <- W_h0 x 0.5 / rho(W_h0), frozen; + SLOW + main's HINGE (Adam) or S25's groups (Muon)"
PAIRING = ("seeds 260-269: Adam with X's HINGE_D8, Muon with S44's D8_HINGE_M_REF (fresh on this CPU); same initial "
           "parameters (W_h rescaled) and batches")
LR, MUON_LR, ITERS = 1e-3, 0.005, 28800
SB = tuple(range(260, 270))
BREAK_N, NOT_N = 3, 0
_ad = (f"gate W_in/W_g {LR:g} throughout; every other trainable parameter {LR / 10:g} for updates 1-2400, {LR:g} after; "
       f"W_h frozen at rho 0.5; + main's HINGE hinge")
_mu = (f"Muon lr {MUON_LR:g}: gate W_in/W_g throughout, other Muon weights x0.1 for updates 1-2400; Adam on embedding and "
       f"conv {LR / 10:g} for 1-2400, {LR:g} after; W_h frozen at rho 0.5; + main's HINGE hinge")
ARMS = {
    "RES_D8_A": dict(key="RES_D8_A", opt="adam", lr=LR, muon_lr=None, iters=ITERS, seeds=SB,
                     label="reservoir gate (W_h frozen, rho 0.5) + SLOW + HINGE (Adam)", sched=_ad),
    "RES_D8_M": dict(key="RES_D8_M", opt="muon", lr=LR, muon_lr=MUON_LR, iters=ITERS, seeds=SB,
                     label="reservoir gate (W_h frozen, rho 0.5) + S25's groups + HINGE (Muon)", sched=_mu),
}
CUT = set()


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    a = ARMS[arm]
    return child("run", dict(arm=arm, seed=seed, iters=a["iters"], mlr=a["muon_lr"]))


def timing(arms):
    return child("timing", dict(which=list(arms), steps=300, mlr=MUON_LR))


def n_meas(arm):
    return 4


def n_extra(arm):
    return 0


def check():
    ok = True
    for nm, v in child("checks", dict(mlr=MUON_LR)):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def report():
    me = sys.modules[__name__]
    import explore_common2 as c2
    c2.print_screen_header2(me)
    if CUT:
        print(f"  cut by the runtime rule: {sorted(CUT)} (no runs, no reading)")
    got = {arm: c16.runs_of(me, arm) for arm in ARMS}
    pair = {"RES_D8_A": ("X HINGE_D8", s44.x_runs("HINGE_D8", SB)),
            "RES_D8_M": ("S44 D8_HINGE_M_REF", c16.runs_of(s44, "D8_HINGE_M_REF") if "D8_HINGE_M_REF" in s44.ARMS else {})}
    out = {}
    for arm, a in ARMS.items():
        rs = got[arm]
        plab, pr = pair[arm]
        rp.arm_table(arm, rs, a["seeds"], [(plab, pr)], a["label"])
        d = rp.compare(rs, pr, a["seeds"], arm, plab.replace(" ", "_"))
        print(f"    failure classes at the end: {arm} {rp.classes(rs)}; {plab} {rp.classes({s: pr[s] for s in a['seeds'] if s in pr})}")
        print(f"    transitions: {rp.transitions(rs)}")
        rp.maps_block(rs)
        meds = rp.decode_block(rs, chance=1 / 8)
        print("    rho(W_h) / sigma_max(W_h) at init (before / after the rescaling) and at 1200, 2400, 4800, 9600:")
        for s, r in sorted(rs.items()):
            ri, wh = r.get("res_init") or {}, r.get("wh") or {}
            print(f"      {s}: init rho {ri.get('rho0', float('nan')):.3f} -> {ri.get('rho', float('nan')):.6f}, sigma "
                  f"{ri.get('sigma0', float('nan')):.3f} -> {ri.get('sigma', float('nan')):.4f}; "
                  + ", ".join(f"{t}: {wh[t]['rho']:.6f} / {wh[t]['sigma']:.4f}" for t in ("1200", "2400", "4800", "9600") if t in wh))
        nb, n = sum(rp.br(r) for r in rs.values()), len(rs)
        out[arm] = dict(d, bound_routed=nb, n=n, complete=n == len(a["seeds"]), classes=rp.classes(rs), meds=meds)
    if any(not o["complete"] for o in out.values()) or not out:
        rd = "INCOMPLETE (" + ", ".join(f"{arm} {o['n']}/{len(ARMS[arm]['seeds'])}" for arm, o in out.items()) + "), no reading"
    else:
        nbs = [o["bound_routed"] for o in out.values()]
        rd = (("breaks the eight-stream stall" if max(nbs) >= BREAK_N else "does not" if max(nbs) == NOT_N else "neither")
              + " (" + "; ".join(f"{arm} {o['bound_routed']}/{o['n']}" for arm, o in out.items())
              + f"; breaks: either >= {BREAK_N}/10; does not: both 0/10)")
    print(f"  READING S45: {rd}")
    return dict(reading=rd, arms=out)
