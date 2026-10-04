#!/usr/bin/env python
"""
explore_gate_state.py — EXPLORATORY, not a result. Screen S38 (batch 13), descriptive: does the gate state
carry the stream? Reruns to 9600 updates, five seeds per configuration, with labelled decoders.

BACKGROUND: see explore_split_target (S36: at S=8 the split's target channel was empty and the copy held >= 2
streams; no run bound by 43200).

CONFIGURATIONS (reruns of recorded runs on their own run paths at 9c5939e, child process;
explore_gate_state_child):
  D8_HINGE_M  S33's D8_HINGE (Muon, S=8, k=16), seeds 260-264
  HINGE_D8_A  X's HINGE_D8 (Adam, S=8, k=16), seeds 260-264
  A_HINGE_M   S33's A_HINGE (Muon, S=4, k=4), seeds 240-244
  HINGE16_M   S35's MUON_HINGE16 (Muon, S=4, k=16), seeds 240-244
At 1200, 2400, 4800 and 9600, on a 512-sequence probe (its own generator, seed 38000), labelled:
  - stream decodability from the gate state h at key positions: a nearest-class-mean decoder fit on half the
    probe, accuracy on the other half (chance 1/S);
  - the same from the gate's input (the embedding) at those positions, as a reference;
  - the same from h at each block's first key and at its last key (a block: the S adjacent triples sharing a
    key in the grouped layout).
A run that stops early (bound, early stop) has no later measurements.
READINGS (S=8 configurations, at 4800; the median over the five seeds of h at key positions): "the
eight-stream gate state does not carry the stream" if < 0.5; "it does" if >= 0.9; otherwise "partial". The
S=4 configurations are printed alongside.
CHECKS: each rerun reproduces its recorded curve through 9600 (checked per run; a configuration with a run
that does not is reported, its reading withheld), and, before any run, one rerun per configuration through
2400 reproduces its record and the recomputed read gate equals the model's (max difference < 1e-5); the
decoder scores 1.0 on a synthetic separable set and within 0.05 of chance with shuffled labels.
"""

import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common2 as c2
import explore_main9c as mt
from test_channel_binding import SUB_LR

NAME = "gate_state"
IDEA = "descriptive: is the stream decodable from the gate state h at key positions, at S=8 vs S=4?"
SOURCE = "batch 12's S36 (splits at S=8 copied channels onto empty ones and never bound); S33, S35, X's HINGE_D8"
CHANGE = "none to the runs (reruns to 9600); labelled nearest-class-mean decoders on h and on the gate input"
PAIRING = "each rerun against its recorded run (the curve must reproduce through 9600)"
CHILD = "explore_gate_state_child"
LR = SUB_LR
MUON_LR = 0.005
ITERS = 9600
AT = (1200, 2400, 4800, 9600)
NOT_MAX, DOES_MIN = 0.5, 0.9
TIME_STEPS = 200
REC = {"D8_HINGE_M": ("muon_scale", "D8_HINGE"), "HINGE_D8_A": (None, "HINGE_D8"),
       "A_HINGE_M": ("muon_scale", "A_HINGE"), "HINGE16_M": ("muon_k16", "MUON_HINGE16")}
_d8, _k4 = tuple(range(260, 265)), tuple(range(240, 245))
ARMS = {
    "D8_HINGE_M": dict(key="D8_HINGE_M", seeds=_d8, S=8, lr=LR, iters=ITERS, muon_lr=MUON_LR, prio=1,
                       label="rerun of S33's D8_HINGE (Muon, S=8, k=16) to 9600", sched="S33's D8_HINGE recipe"),
    "HINGE_D8_A": dict(key="HINGE_D8_A", seeds=_d8, S=8, lr=LR, iters=ITERS, muon_lr=None, prio=1,
                       label="rerun of X's HINGE_D8 (Adam, S=8, k=16) to 9600", sched="test_slow_start's HINGE recipe"),
    "A_HINGE_M": dict(key="A_HINGE_M", seeds=_k4, S=4, lr=LR, iters=ITERS, muon_lr=MUON_LR, prio=1,
                      label="rerun of S33's A_HINGE (Muon, S=4, k=4) to 9600", sched="S33's A_HINGE recipe"),
    "HINGE16_M": dict(key="HINGE16_M", seeds=_k4, S=4, lr=LR, iters=ITERS, muon_lr=MUON_LR, prio=1,
                      label="rerun of S35's MUON_HINGE16 (Muon, S=4, k=16) to 9600", sched="S35's MUON_HINGE16 recipe"),
}


def child(func, payload, timeout=None):
    return mt.run_child(CHILD, func, payload, timeout=timeout)


def recorded(conf, seed):
    store, key = REC[conf]
    if store is None:
        return mt.recorded("slow_start", mt.RESULTS_SHA).get(f"{key}|{seed}")
    return ec.load_store(store)["runs"].get(f"{key}|{seed}")


def run_job(arm, seed):
    a = ARMS[arm]
    r = child("run", dict(conf=arm, seed=seed, iters=a["iters"], mlr=a["muon_lr"]))
    if r.get("ok"):
        ref = recorded(arm, seed)
        want = [c for c in ref["curve"] if c[0] <= a["iters"]]
        r["reproduces"] = r["curve"] == want
        r["recorded_curve_upto"] = want[-1][0] if want else None
    return r


def segments(arm):
    raise NotImplementedError("S38 is projected by its own child timing (explore_batch13)")


def timing(confs):
    return child("timing", dict(which=list(confs), steps=TIME_STEPS, mlr=MUON_LR))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _rerun(conf):
    seed = ARMS[conf]["seeds"][0]
    r = child("rerun_check", dict(conf=conf, seed=seed, mlr=ARMS[conf]["muon_lr"]))
    ref = recorded(conf, seed)
    want = [c for c in ref["curve"] if c[0] <= 2400]
    d = r["decode"]
    diff = max(v["gate_recompute_diff"] for v in d.values()) if d else None
    return (f"{conf}|{seed} rerun reproduces its record through 2400 ({r['curve']} vs {want}); probes at {sorted(d)}; the "
            f"recomputed read gate equals the model's (max difference {diff:.2e})",
            r["curve"] == want and sorted(d) == ["1200", "2400"] and diff is not None and diff < 1e-5)


def check():
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=5) as ex:
        fs = [ex.submit(_rerun, c) for c in ARMS]
        fd = ex.submit(child, "decoder_check", {})
        rows = [f.result() for f in fs] + [tuple(x) for x in fd.result()]
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    print(f"  CHECK {NAME}: children at {mt.MAIN_SHA[:7]} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def fmt(x):
    return "  -- " if x is None else f"{x:5.2f}"


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    out = {}
    for conf, a in ARMS.items():
        rs = {int(k.split("|")[1]): r for k, r in store["runs"].items() if k.startswith(conf + "|") and r.get("ok")}
        S = a["S"]
        nrep = sum(1 for r in rs.values() if r.get("reproduces"))
        print(f"  CONFIG {conf} ({a['label']}); chance {1 / S:.3f}; reruns reproduce their records through 9600: {nrep}/{len(rs)}")
        print(f"    {'seed':>4} {'outcome at 9600':<24} {'stop':>5} | " + " | ".join(f"{t:>5}: h_key v_key h_1st h_last" for t in AT))
        for s, r in sorted(rs.items()):
            cells = []
            for t in AT:
                d = (r.get("decode") or {}).get(str(t))
                cells.append(f"{'':>5}  {fmt(d and d['h_key'])} {fmt(d and d['v_key'])} {fmt(d and d['h_first'])} "
                             f"{fmt(d and d['h_last'])}")
            print(f"    {s:>4} {r['outcome']:<24} {r['stopped_at']:>5} | " + " | ".join(cells)
                  + ("" if r.get("reproduces") else "   (DOES NOT reproduce its record)"))
        meds = {}
        for t in AT:
            vals = {f: [r["decode"][str(t)][f] for r in rs.values() if str(t) in (r.get("decode") or {})]
                    for f in ("h_key", "v_key", "h_first", "h_last")}
            meds[t] = {f: (statistics.median(v) if v else None) for f, v in vals.items()}
            n_t = len(vals["h_key"])
            print(f"    median at {t:>5} ({n_t} runs): h at key positions {fmt(meds[t]['h_key'])}, gate input "
                  f"{fmt(meds[t]['v_key'])}, h at the block's first key {fmt(meds[t]['h_first'])}, at its last key "
                  f"{fmt(meds[t]['h_last'])}")
        m48 = meds[4800]["h_key"]
        if S == 8:
            if nrep < len(rs):
                rd = "withheld (a rerun does not reproduce its record)"
            elif m48 is None:
                rd = "no measurement at 4800"
            else:
                rd = ("the eight-stream gate state does not carry the stream" if m48 < NOT_MAX else
                      "it does" if m48 >= DOES_MIN else "partial") + f" (median h_key at 4800 {m48:.3f}; < {NOT_MAX} not, >= {DOES_MIN} does)"
            print(f"  READING S38 {conf}: {rd}")
        else:
            rd = f"printed alongside (median h_key at 4800 {fmt(m48).strip()})"
            print(f"  S38 {conf} (S=4, printed alongside): median h_key at 4800 {fmt(m48).strip()}")
        out[conf] = dict(reading=rd, median_4800=m48, reproduce=nrep, n=len(rs), complete=len(rs) == len(a["seeds"]),
                         medians={str(t): meds[t] for t in AT})
    return out
