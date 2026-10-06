#!/usr/bin/env python
"""
explore_write_decoders.py — EXPLORATORY, not a result. Screen S47 (batch 16), descriptive: S38's decoder at the write
positions.

BACKGROUND: the audit of 6 October 2026, section 3.2 — the model reads the answer at the query's KEY position (CTX one
back), but the stream-labelled write happens at VAL positions (CTX two back), and the routing criterion (the
stream-to-channel map) is measured at VAL positions; S38-S41 decoded the gate state at KEY and CTX positions only, so
Revision 7's "the gate needs only one token of memory" (Section 11) rests on the read positions alone. For GATE_PREV the
key-position decodability of its input is 1.0 by construction (u_key = W_in e_KEY + W_prev e_CTX).

CONFIGURATIONS (explore_write_decoders_child; reruns to 4800 on their own paths and recipes):
  HINGE_D8_A  X's HINGE_D8 260-264 (Adam)            — must reproduce X's records through 4800
  D8_HINGE_M  S33's D8_HINGE 260-264 (Muon)          — fresh on this CPU (records: 2.10GHz)
  A_HINGE_M   S33's A_HINGE 240-244 (Muon; S=4, k=4) — fresh on this CPU
  HINGE16_M   S35's MUON_HINGE16 240-244 (Muon; k=16) — fresh on this CPU
  D8_PREV_A   S40's D8_PREV_A 260-264 (Adam)         — must reproduce S40's records through 4800
PRINTED (no reading): per run, the stream decodability of the gate state h at CTX (3j), KEY (3j+1) and VAL (3j+2) positions
side by side at 0, 1200, 2400 and 4800 (and of GATE_PREV's input u); whether the rerun reproduces its record through 4800
(curve and statistics); medians per configuration.
CHECKS: the decoder passes S38's synthetic checks (separable 1.0; shuffled labels within 0.05 of chance) and equals S38's
ncm; the fit and score halves are disjoint; the positions hold the stream, key and value tokens; the read gate recomputed
from h equals the model's for the recurrent, GATE_PREV, LOCAL3 and reservoir gates.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_b16_report as rp

NAME = "write_decoders"
CHILD = "explore_write_decoders_child"
MAIN = True
IDEA = "decode the stream from the gate state at the write (VAL) positions, beside CTX and KEY"
SOURCE = "the audit of 6 October 2026, section 3.2 (one token of memory at the read, two at the write)"
CHANGE = "none to the runs (reruns to 4800); S38's decoder at CTX, KEY and VAL at 0, 1200, 2400, 4800"
PAIRING = "each rerun against its record (Adam: reproduce through 4800; Muon: fresh on this CPU)"
ITERS = 4800
MUON_LR = 0.005
D5, A5 = tuple(range(260, 265)), tuple(range(240, 245))
STEPS = ("0", "1200", "2400", "4800")
REC = {"HINGE_D8_A": ("X", "HINGE_D8"), "D8_HINGE_M": ("muon_scale", "D8_HINGE"), "A_HINGE_M": ("muon_scale", "A_HINGE"),
       "HINGE16_M": ("muon_k16", "MUON_HINGE16"), "D8_PREV_A": ("gate_prev", "D8_PREV_A")}
_h = "; + main's HINGE hinge"
_ad = "gate W_in/W_h/W_g 0.001 throughout; every other parameter 0.0001 for updates 1-2400, 0.001 after" + _h
_mu = ("Muon lr 0.005: gate W_in/W_h/W_g throughout, other Muon weights x0.1 for updates 1-2400; Adam on embedding and conv "
       "0.0001 for 1-2400, 0.001 after" + _h)
ARMS = {
    "HINGE_D8_A": dict(key="HINGE_D8_A", opt="adam", lr=1e-3, muon_lr=None, iters=ITERS, seeds=D5, chance=1 / 8,
                       label="X's HINGE_D8 (Adam), rerun to 4800", sched=_ad),
    "D8_HINGE_M": dict(key="D8_HINGE_M", opt="muon", lr=1e-3, muon_lr=MUON_LR, iters=ITERS, seeds=D5, chance=1 / 8,
                       label="S33's D8_HINGE (Muon), fresh on this CPU, to 4800", sched=_mu),
    "A_HINGE_M": dict(key="A_HINGE_M", opt="muon", lr=1e-3, muon_lr=MUON_LR, iters=ITERS, seeds=A5, chance=1 / 4,
                      label="S33's A_HINGE (Muon; S=4, k=4), fresh on this CPU, to 4800", sched=_mu),
    "HINGE16_M": dict(key="HINGE16_M", opt="muon", lr=1e-3, muon_lr=MUON_LR, iters=ITERS, seeds=A5, chance=1 / 4,
                      label="S35's MUON_HINGE16 (Muon; S=4, k=16), fresh on this CPU, to 4800", sched=_mu),
    "D8_PREV_A": dict(key="D8_PREV_A", opt="adam", lr=1e-3, muon_lr=None, iters=ITERS, seeds=D5, chance=1 / 8,
                      label="S40's D8_PREV_A (GATE_PREV, Adam), rerun to 4800", sched="W_prev in the gate group; " + _ad),
}
CUT = set()


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    a = ARMS[arm]
    return child("run", dict(conf=arm, seed=seed, iters=a["iters"], mlr=a["muon_lr"]))


def timing(arms):
    return child("timing", dict(which=list(arms), steps=300, mlr=MUON_LR))


def n_meas(arm):
    return 4


def n_extra(arm):
    return 0


def check():
    ok = True
    for nm, v in child("checks", {}):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def records(arm, seeds):
    src, key = REC[arm]
    if src == "X":
        got = child("recorded", dict(keys=[f"{key}|{s}" for s in seeds], t=ITERS))
        return {int(k.split("|")[1]): v for k, v in got.items()}
    st = ec.load_store(src)["runs"]
    return {s: st[f"{key}|{s}"] for s in seeds if f"{key}|{s}" in st}


def reproduces(r, ref):
    if ref is None:
        return None
    want = [c for c in ref["curve"] if c[0] <= ITERS]
    sn = [s for s in r["stats"] if s["step"] != "end" and s["step"] <= ITERS]
    so = [s for s in ref["stats"] if s["step"] != "end" and s["step"] <= ITERS]
    return r["curve"] == want and sn == so


def report():
    me = sys.modules[__name__]
    import explore_common2 as c2
    import explore_window_scale as s44
    c2.print_screen_header2(me)
    out = {}
    fresh_ref = {"D8_HINGE_M": "D8_HINGE_M_REF", "HINGE16_M": "MUON_HINGE16_REF"}
    for arm, a in ARMS.items():
        rs = c16.runs_of(me, arm)
        recs = records(arm, a["seeds"])
        src, key = REC[arm]
        same = {s: reproduces(r, recs.get(s)) for s, r in rs.items()}
        extra = ""
        if arm in fresh_ref and fresh_ref[arm] in s44.ARMS:
            refs = c16.runs_of(s44, fresh_ref[arm], seeds=a["seeds"])
            eq = {s: reproduces(r, refs.get(s)) for s, r in rs.items() if s in refs}
            extra = f"; equal to S44's {fresh_ref[arm]} (same CPU) through {ITERS}: {eq}"
        feats = ("h_ctx", "h_key", "h_val") + (("u_ctx", "u_key", "u_val") if arm == "D8_PREV_A" else ())
        print(f"  CONFIG {arm} ({a['label']}): reproduces {'X' if src == 'X' else src}'s {key} record through {ITERS}: "
              + ", ".join(f"{s} {same[s]}" for s in sorted(same)) + (" (Muon: a record from the 2.10GHz CPU; fresh here)"
                                                                     if a["opt"] == "muon" else "") + extra)
        meds = rp.decode_block(rs, steps=STEPS, chance=a["chance"], feats=feats)
        out[arm] = dict(n=len(rs), complete=len(rs) == len(a["seeds"]), reproduces=same, meds=meds)
    print("  SIDE BY SIDE (medians over the runs; h = the gate state; chance 1/S): CTX (3j) | KEY (3j+1) | VAL (3j+2)")
    for arm, o in out.items():
        print(f"    {arm:<11} " + "  ".join(
            f"{t:>4}: {rp.f2(o['meds'].get((t, 'h_ctx')), 4)} | {rp.f2(o['meds'].get((t, 'h_key')), 4)} | "
            f"{rp.f2(o['meds'].get((t, 'h_val')), 4)}" for t in STEPS))
    return out
