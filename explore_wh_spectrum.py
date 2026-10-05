#!/usr/bin/env python
"""
explore_wh_spectrum.py — EXPLORATORY, not a result. Screen S42 (batch 15), descriptive: does the gate's recurrent
gain cross 1 (spectral radius rho(W_h) > 1) just before |W_h h_(t-1)| at key positions takes off?

BACKGROUND (docstring): see explore_gate_cap (S39: under Muon |W_h h|@key 0.19 at 400 -> 1.9 at 600; under Adam the
jump falls between 1200 and 1600).

CONFIGURATIONS (reruns to 1600 updates of recorded runs on their own run paths at 9c5939e, child process;
explore_wh_spectrum_child):
  D8_HINGE_M  S33's D8_HINGE (Muon, S=8, k=16), seeds 260-264
  HINGE_D8_A  X's HINGE_D8 (Adam, S=8, k=16), seeds 260-264
MACHINE (after the first dry run, by the user's decision; see explore_gate_cap): on this CPU (Xeon @ 2.80GHz) the
Adam runs reproduce their records, the Muon runs do not. HINGE_D8_A is checked against X's records through 1600 (the
curve and statistics there). D8_HINGE_M runs fresh, labelled as not the recorded runs: each run is checked instead
against this CPU's S33 D8_HINGE rerun with the same seed (S41's REF_HINGE_M) through 1200 (the curve and statistics
there; S41's runs evaluate at 1200, not 1600), and the 2.10GHz record comparison is printed.
MEASUREMENTS every 50 updates from 0 to 1600, on S39's probe (seed 39000): sigma_max(W_h) and rho(W_h) (largest
|eigenvalue|); the median |h|, |W_h h_(t-1)| and |W_in v_t| at key positions; S38's decoder of the stream from h at
key positions.
PER RUN: the first measured update with rho(W_h) > 1 and the first with |W_h h|@key > 1 (median over the probe's key
positions), at 50-update resolution.
LABEL per configuration (fixed before any run): "gain crossing" if in >= 4/5 runs rho first exceeds 1 within 150
updates before |W_h h|@key first exceeds 1 (both cross by 1600 and 0 <= (first |W_h h|@key > 1) - (first rho > 1)
<= 150, at 50-update resolution); otherwise "not".
CHECKS: before any run, one rerun per configuration through 1600 (with every measurement) reproduces its reference
(HINGE_D8_A: X's record through 1600; D8_HINGE_M: this CPU's REF_HINGE_M|260 through 1200, the driver's Muon repro
run); every S42 run is checked the same way (a configuration with a run that does not is reported, its label
withheld); rho and sigma_max agree with numpy on a random matrix to 1e-6.
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

NAME = "wh_spectrum"
IDEA = "descriptive: does rho(W_h) cross 1 just before |W_h h|@key takes off (the stream leaving the gate state)?"
SOURCE = "batch 14's S39 (the stream leaves h at S=8 as |W_h h_(t-1)| outgrows |W_in v_t|)"
CHANGE = "none to the runs (reruns to 1600); sigma_max(W_h), rho(W_h), the gate's term norms and h's decodability every 50 updates"
PAIRING = "each rerun against its recorded run (the curve and statistics must reproduce through 1600)"
CHILD = "explore_wh_spectrum_child"
LR = SUB_LR
MUON_LR = 0.005
ITERS = 1600
EVERY = 50
AT = tuple(range(0, ITERS + 1, EVERY))
WINDOW, MIN_RUNS = 150, 4
TIME_STEPS = 200
REC = {"D8_HINGE_M": ("muon_scale", "D8_HINGE"), "HINGE_D8_A": (None, "HINGE_D8")}
S39_STORE = "gate_memory"
MUON = ("D8_HINGE_M",)                       # checked against this CPU's REF_HINGE_M (S41), through REF_AT
REF_AT = 1200
MUON_REF = None                              # the driver's REF_HINGE_M|260 run through 1200 (this CPU), when set
_d8 = tuple(range(260, 265))
ARMS = {
    "D8_HINGE_M": dict(key="D8_HINGE_M", seeds=_d8, S=8, lr=LR, iters=ITERS, muon_lr=MUON_LR, prio=1,
                       label="rerun of S33's D8_HINGE (Muon, S=8, k=16) to 1600",
                       sched=("Muon lr 0.005: gate W_in/W_h/W_g throughout, other Muon weights x0.1 for updates 1-2400; Adam "
                              "on embedding and conv 0.0001 for 1-2400; + main's HINGE hinge")),
    "HINGE_D8_A": dict(key="HINGE_D8_A", seeds=_d8, S=8, lr=LR, iters=ITERS, muon_lr=None, prio=1,
                       label="rerun of X's HINGE_D8 (Adam, S=8, k=16) to 1600",
                       sched="gate W_in/W_h/W_g 0.001 throughout; every other parameter 0.0001 for updates 1-2400; + main's HINGE hinge"),
}


def child(func, payload, timeout=None):
    return mt.run_child(CHILD, func, payload, timeout=timeout)


def recorded(conf, seed):
    store, key = REC[conf]
    if store is None:
        return mt.recorded("slow_start", mt.RESULTS_SHA).get(f"{key}|{seed}")
    return ec.load_store(store)["runs"].get(f"{key}|{seed}")


def _stats_upto(stats, t):
    return {s["step"]: s for s in stats if s["step"] != "end" and s["step"] <= t}


def reproduces(r, conf, seed, upto):
    ref = recorded(conf, seed)
    want = [c for c in ref["curve"] if c[0] <= upto]
    sn, sr = _stats_upto(r["stats"], upto), _stats_upto(ref["stats"], upto)
    same = sn.keys() == sr.keys() and all(sn[k] == sr[k] for k in sr)
    return r["curve"] == want and same, want, sorted(sr)


def same_as(r, ref, t):
    want = [c for c in ref["curve"] if c[0] <= t]
    got = [c for c in r["curve"] if c[0] <= t]
    sn, sr = _stats_upto(r["stats"], t), _stats_upto(ref["stats"], t)
    return got == want and sn.keys() == sr.keys() and all(sn[k] == sr[k] for k in sr) and len(sr) >= 2, want, sorted(sr)


def run_job(arm, seed):
    a = ARMS[arm]
    r = child("run", dict(conf=arm, seed=seed, iters=a["iters"], mlr=a["muon_lr"]))
    if r.get("ok"):
        ok, want, _ = reproduces(r, arm, seed, a["iters"])
        r["recorded_curve_upto"] = want[-1][0] if want else None
        if arm in MUON:
            r["reproduces_record"] = ok              # the 2.10GHz record (printed); checked against REF_HINGE_M in the report
            r["reproduces"] = None
        else:
            r["reproduces"] = ok
    return r


def muon_check(rs):
    """D8_HINGE_M runs against S41's REF_HINGE_M runs (this CPU) through REF_AT; None where the reference is missing."""
    import explore_gate_cap as s41
    refs = ec.load_store(s41.NAME)["runs"]
    out = {}
    for s, r in rs.items():
        ref = refs.get(f"REF_HINGE_M|{s}")
        out[s] = None if not (ref and ref.get("ok")) else same_as(r, ref, REF_AT)[0]
    return out


def segments(arm):
    raise NotImplementedError("S42 is projected by its own child timing (explore_batch15)")


def timing(confs):
    return child("timing", dict(which=list(confs), steps=TIME_STEPS, mlr=MUON_LR))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _rerun(conf):
    seed = ARMS[conf]["seeds"][0]
    r = child("run", dict(conf=conf, seed=seed, iters=ITERS, mlr=ARMS[conf]["muon_lr"]))
    if conf in MUON:
        assert MUON_REF is not None, "the driver's Muon repro run sets MUON_REF before the CHECKs"
        ok, want, steps = same_as(r, MUON_REF, REF_AT)
        what, upto = "this CPU's REF_HINGE_M|260 (the driver's Muon repro run)", REF_AT
    else:
        ok, want, steps = reproduces(r, conf, seed, ITERS)
        what, upto = "its record", ITERS
    sp = r["spec"]
    diff = max(v["gate_recompute_diff"] for v in sp.values())
    return (f"{conf}|{seed} rerun with every measurement reproduces {what} through {upto} (curve {r['curve']} vs {want}; "
            f"statistics at {steps}); measured at {len(sp)} updates ({min(map(int, sp))}-{max(map(int, sp))}, every {EVERY}; "
            f"{r['updates_counted']} updates counted, {r['builds']} model built); the recomputed read gate equals the model's "
            f"(max difference {diff:.2e})",
            ok and sorted(map(int, sp)) == list(AT) and r["updates_counted"] == ITERS and r["builds"] == 1 and diff < 1e-5)


def check():
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as ex:
        fs = [ex.submit(_rerun, c) for c in ARMS]
        fn = ex.submit(child, "numpy_check", {})
        rows = [f.result() for f in fs] + [tuple(x) for x in fn.result()]
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    print(f"  CHECK {NAME}: children at {mt.MAIN_SHA[:7]} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def first_above(sp, key, thr=1.0):
    for t in AT:
        d = sp.get(str(t))
        if d is not None and d[key] > thr:
            return t
    return None


def crossing(sp):
    fr, fb = first_above(sp, "rho"), first_above(sp, "b_key")
    met = fr is not None and fb is not None and 0 <= fb - fr <= WINDOW
    return fr, fb, (None if fr is None or fb is None else fb - fr), met


def line(vals, w=5, d=2):
    return " ".join(f"{'--':>{w}}" if v is None else f"{v:{w}.{d}f}" for v in vals)


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    s39 = ec.load_store(S39_STORE)["runs"]
    s39key = {"D8_HINGE_M": "D8_HINGE_M", "HINGE_D8_A": "HINGE_D8_A"}
    out = {}
    for conf, a in ARMS.items():
        rs = {int(k.split("|")[1]): r for k, r in store["runs"].items() if k.startswith(conf + "|") and r.get("ok")}
        if conf in MUON:
            mc = muon_check(rs)
            for sd, r in rs.items():
                r["reproduces"] = mc[sd]
            nrec = sum(1 for r in rs.values() if r.get("reproduces_record"))
            nrep = sum(1 for r in rs.values() if r.get("reproduces"))
            print(f"  CONFIG {conf} ({a['label']}); fresh runs on this CPU: they reproduce this CPU's REF_HINGE_M (S41) through "
                  f"{REF_AT}: {nrep}/{len(rs)} ({sum(1 for v in mc.values() if v is None)} without a reference yet); they "
                  f"reproduce the 2.10GHz records through {ITERS}: {nrec}/{len(rs)} (printed)")
        else:
            nrep = sum(1 for r in rs.values() if r.get("reproduces"))
            print(f"  CONFIG {conf} ({a['label']}); reruns reproduce their records through {ITERS}: {nrep}/{len(rs)}")
        print(f"    per run: the first update with rho(W_h) > 1, with |W_h h|@key > 1, the difference (window 0-{WINDOW}):")
        n_met = 0
        cross = {}
        for s, r in sorted(rs.items()):
            fr, fb, dd, met = crossing(r["spec"])
            n_met += met
            cross[s] = dict(rho=fr, whh=fb, diff=dd, met=met)
            agree = []
            r39 = s39.get(f"{s39key[conf]}|{s}")
            if r39:
                shared = [t for t in r39["mem"] if t in r["spec"]]
                agree = [t for t in shared if r39["mem"][t]["dec_key"] == r["spec"][t]["h_key"]
                         and r39["mem"][t]["wh_key"] == r["spec"][t]["b_key"] and r39["mem"][t]["win_key"] == r["spec"][t]["a_key"]]
                agr = f"; S39's measurements at its {len(shared)} shared updates repeated exactly at {len(agree)}"
            else:
                agr = ""
            print(f"    {s:>4}: rho > 1 first at {str(fr):>5}, |W_h h|@key > 1 first at {str(fb):>5}, difference {str(dd):>5}: "
                  f"{'within' if met else 'NOT within'}" + agr + ("" if r.get("reproduces") else
                                                                 "   (DOES NOT reproduce its reference)"))
        print(f"    per run, every {EVERY} updates from 0 to {ITERS} (columns: updates {AT[0]}, {AT[1]}, ..., {AT[-1]}):")
        for key, lab in (("rho", "rho(W_h)"), ("sigma", "sig_max(W_h)"), ("b_key", "|Wh h|@key"), ("a_key", "|Win v|@key"),
                         ("hn_key", "|h|@key"), ("h_key", "dec h@key")):
            print(f"      {lab}:")
            for s, r in sorted(rs.items()):
                print(f"        {s}: " + line([(r['spec'].get(str(t)) or {}).get(key) for t in AT]))
        print(f"    medians over the runs:")
        cols = (("sigma", "sig_max"), ("rho", "rho"), ("hn_key", "|h|@key"), ("b_key", "|Wh h|@key"), ("a_key", "|Win v|@key"),
                ("h_key", "dec h@key"), ("h_stream", "dec h@str"), ("sat_key", "sat@key"))
        print(f"    {'update':>6} {'n':>2} | " + " ".join(f"{lab:>11}" for _, lab in cols))
        meds = {}
        for t in AT:
            ds = [r["spec"][str(t)] for r in rs.values() if str(t) in r["spec"]]
            meds[t] = {k: (statistics.median(d[k] for d in ds) if ds else None) for k, _ in cols}
            print(f"    {t:>6} {len(ds):>2} | " + " ".join(f"{'--':>11}" if meds[t][k] is None else f"{meds[t][k]:11.3f}"
                                                          for k, _ in cols))
        if nrep < len(rs):
            lb = "withheld (a rerun does not reproduce its reference)"
        else:
            lb = ("gain crossing" if n_met >= MIN_RUNS else "not") + \
                 f" (rho first exceeds 1 within {WINDOW} updates before |W_h h|@key first exceeds 1 in {n_met}/{len(rs)} runs; " \
                 f">= {MIN_RUNS} = gain crossing)"
        print(f"  LABEL S42 {conf}: {lb}")
        out[conf] = dict(label=lb, n_met=n_met, n=len(rs), reproduce=nrep, complete=len(rs) == len(a["seeds"]), cross=cross,
                         medians={str(t): meds[t] for t in AT})
    return out
