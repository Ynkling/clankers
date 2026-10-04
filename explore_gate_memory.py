#!/usr/bin/env python
"""
explore_gate_memory.py — EXPLORATORY, not a result. Screen S39 (batch 14), descriptive: does the gate state
form a memory of the stream token in the first 2400 updates, and does it keep it?

BACKGROUND (docstring)
- Grouped layout: every key is immediately preceded by its stream's token (triples: stream, key, value), so the
  gate needs one token of memory at key positions.
- S38: stream decodability from the gate state h at key positions, median at 4800: D8_HINGE (Muon) 0.13,
  HINGE_D8 (Adam) 0.14 (chance 0.125); A_HINGE (Muon, S=4, k=4) 0.84; MUON_HINGE16 1.00 (chance 0.25).
  HINGE_D8 262: 0.81 at 1200, 0.13 at 2400.
- S37: KEYMASS aimed 10/11 splits correctly (Muon). Routed: SPLITK_M 6/10 vs A_HINGE 3/10; SPLITK_A 6/10 vs
  HINGE4k4_A 5/10. SPLITK_A 248 bound with all four streams on one channel, so at S=4, k=4 the outcome to use
  is BOUND ROUTED.

CONFIGURATIONS (reruns to 2400 of recorded runs on their own run paths at 9c5939e, child process;
explore_gate_memory_child), each checked bit-identical to its record through 2400:
  D8_HINGE_M  S33's D8_HINGE (Muon, S=8, k=16), seeds 260-264
  D8_SLOW_M   S33's D8_SLOW (Muon, S=8, k=16, no hinge), seeds 260-264
  HINGE_D8_A  X's HINGE_D8 (Adam, S=8, k=16), seeds 260-264
  A_HINGE_M   S33's A_HINGE (Muon, S=4, k=4), seeds 240-244, a reference
MEASUREMENTS at updates 0, 50, 100, 200, 300, 400, 600, 800, 1200, 1600, 2400 on a 512-sequence probe (its own
generator, seed 39000), labelled: S38's decoder (nearest class mean, fit on half the probe, scored on the other
half) of the stream from h at stream-token and at key positions; the median norms of W_in v_t and W_h h_{t-1}
(before the tanh) at stream-token and key positions; the fraction of h units with |h| > 0.95 (key positions;
the whole body); the read gate's entropy at key positions; the distinct channels holding the streams.
LABELS per configuration (fixed before any run), from the median over its seeds of the decodability at key
positions: "formed, then lost" if >= 0.5 at some evaluation up to 1200 and <= 0.25 at 2400; "never formed" if
< 0.25 at every evaluation; otherwise "partial". The hinge vs no-hinge curves (D8_HINGE_M vs D8_SLOW_M) are
printed side by side.
CHECKS: before any run, one rerun per configuration through 1200 (with every measurement) reproduces its record
and the recomputed read gate equals the model's (max difference < 1e-5); every S39 run is checked against its
record through 2400 (a configuration with a run that does not is reported, its label withheld); the decoder
scores 1.0 on a synthetic separable set and within 0.05 of chance with shuffled labels.
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

NAME = "gate_memory"
IDEA = "descriptive: does the gate state form a one-token memory of the stream in updates 0-2400, and keep it?"
SOURCE = "batch 13's S38 (S=8 h decodability at chance at 4800; HINGE_D8 262 at 0.81 at 1200, 0.13 at 2400); S33; X"
CHANGE = ("none to the runs (reruns to 2400); labelled decoders of h, the pre-tanh terms' norms, saturation, read-gate "
          "entropy and distinct channels at 11 updates")
PAIRING = "each rerun against its recorded run (the curve must reproduce through 2400)"
CHILD = "explore_gate_memory_child"
LR = SUB_LR
MUON_LR = 0.005
ITERS = 2400
AT = (0, 50, 100, 200, 300, 400, 600, 800, 1200, 1600, 2400)
FORMED_MIN, FORMED_BY, LOST_MAX, NEVER_MAX = 0.5, 1200, 0.25, 0.25
CHECK_AT = 1200
TIME_STEPS = 200
REC = {"D8_HINGE_M": ("muon_scale", "D8_HINGE"), "D8_SLOW_M": ("muon_scale", "D8_SLOW"),
       "HINGE_D8_A": (None, "HINGE_D8"), "A_HINGE_M": ("muon_scale", "A_HINGE")}
_d8, _k4 = tuple(range(260, 265)), tuple(range(240, 245))
_mu = ("Muon lr 0.005: gate W_in/W_h/W_g throughout, other Muon weights x0.1 for updates 1-2400; Adam on embedding and "
       "conv 0.0001 for 1-2400")
ARMS = {
    "D8_HINGE_M": dict(key="D8_HINGE_M", seeds=_d8, S=8, lr=LR, iters=ITERS, muon_lr=MUON_LR, prio=1,
                       label="rerun of S33's D8_HINGE (Muon, S=8, k=16) to 2400", sched=_mu + "; + main's HINGE hinge"),
    "D8_SLOW_M": dict(key="D8_SLOW_M", seeds=_d8, S=8, lr=LR, iters=ITERS, muon_lr=MUON_LR, prio=1,
                      label="rerun of S33's D8_SLOW (Muon, S=8, k=16, no hinge) to 2400", sched=_mu),
    "HINGE_D8_A": dict(key="HINGE_D8_A", seeds=_d8, S=8, lr=LR, iters=ITERS, muon_lr=None, prio=1,
                       label="rerun of X's HINGE_D8 (Adam, S=8, k=16) to 2400",
                       sched="gate W_in/W_h/W_g 0.001 throughout; every other parameter 0.0001 for updates 1-2400; + "
                             "main's HINGE hinge"),
    "A_HINGE_M": dict(key="A_HINGE_M", seeds=_k4, S=4, lr=LR, iters=ITERS, muon_lr=MUON_LR, prio=1,
                      label="rerun of S33's A_HINGE (Muon, S=4, k=4) to 2400, a reference", sched=_mu + "; + main's HINGE hinge"),
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
    raise NotImplementedError("S39 is projected by its own child timing (explore_batch14)")


def timing(confs):
    return child("timing", dict(which=list(confs), steps=TIME_STEPS, mlr=MUON_LR))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _rerun(conf):
    seed = ARMS[conf]["seeds"][0]
    r = child("run", dict(conf=conf, seed=seed, iters=CHECK_AT, mlr=ARMS[conf]["muon_lr"]))
    ref = recorded(conf, seed)
    want = [c for c in ref["curve"] if c[0] <= CHECK_AT]
    mem = r["mem"]
    want_at = [str(t) for t in AT if t <= CHECK_AT]
    diff = max(v["gate_recompute_diff"] for v in mem.values()) if mem else None
    return (f"{conf}|{seed} rerun with every measurement reproduces its record through {CHECK_AT} ({r['curve']} vs {want}); "
            f"measured at {sorted(mem, key=int)} ({r['updates_counted']} updates counted, {r.get('builds')} model built); the "
            f"recomputed read gate equals the model's (max difference {diff:.2e})",
            r["curve"] == want and sorted(mem, key=int) == want_at and r["updates_counted"] == CHECK_AT
            and r.get("builds") == 1 and diff is not None and diff < 1e-5)


def check():
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=5) as ex:
        fs = [ex.submit(_rerun, c) for c in ARMS]
        fd = ex.submit(mt.run_child, "explore_gate_state_child", "decoder_check", {})
        rows = [f.result() for f in fs] + [tuple(x) for x in fd.result()]
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    print(f"  CHECK {NAME}: children at {mt.MAIN_SHA[:7]} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
FIELDS = (("dec_stream", "dec h@stream"), ("dec_key", "dec h@key"), ("win_stream", "|Win v|@str"), ("wh_stream", "|Wh h|@str"),
          ("win_key", "|Win v|@key"), ("wh_key", "|Wh h|@key"), ("sat_key", "sat@key"), ("sat_body", "sat body"),
          ("entropy_key", "H(gr)@key"), ("distinct", "distinct"))


def fmt(x, w=6):
    return f"{'--':>{w}}" if x is None else f"{x:{w}.3f}" if isinstance(x, float) else f"{x:>{w}}"


def medians(rs):
    out = {}
    for t in AT:
        ms = [r["mem"][str(t)] for r in rs.values() if str(t) in (r.get("mem") or {})]
        out[t] = {f: (statistics.median(m[f] for m in ms) if ms else None) for f, _ in FIELDS}
        out[t]["n"] = len(ms)
    return out


def label(md):
    dk = {t: md[t]["dec_key"] for t in AT if md[t]["dec_key"] is not None}
    if ITERS not in dk:
        return "no measurement at 2400"
    early = [v for t, v in dk.items() if t <= FORMED_BY]
    if early and max(early) >= FORMED_MIN and dk[ITERS] <= LOST_MAX:
        rd = "formed, then lost"
    elif all(v < NEVER_MAX for v in dk.values()):
        rd = "never formed"
    else:
        rd = "partial"
    return rd + (f" (median dec h@key: max up to {FORMED_BY} {max(early):.3f}, at {ITERS} {dk[ITERS]:.3f}, max overall "
                 f"{max(dk.values()):.3f}; formed = >= {FORMED_MIN} by {FORMED_BY}, lost = <= {LOST_MAX} at {ITERS}, never = "
                 f"< {NEVER_MAX} throughout)")


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    out = {}
    meds = {}
    for conf, a in ARMS.items():
        rs = {int(k.split("|")[1]): r for k, r in store["runs"].items() if k.startswith(conf + "|") and r.get("ok")}
        S = a["S"]
        nrep = sum(1 for r in rs.values() if r.get("reproduces"))
        print(f"  CONFIG {conf} ({a['label']}); chance {1 / S:.3f}; reruns reproduce their records through {ITERS}: "
              f"{nrep}/{len(rs)}")
        print(f"    per seed, decodability of h at key positions (stream-token positions) by update:")
        print(f"    {'seed':>4} | " + " ".join(f"{t:>11}" for t in AT))
        for s, r in sorted(rs.items()):
            mem = r.get("mem") or {}
            print(f"    {s:>4} | " + " ".join(
                (f"{mem[str(t)]['dec_key']:.2f} ({mem[str(t)]['dec_stream']:.2f})" if str(t) in mem else f"{'--':>11}")
                for t in AT) + ("" if r.get("reproduces") else "   (DOES NOT reproduce its record)"))
        print(f"    per seed, distinct channels holding the streams (value-position map) by update:")
        for s, r in sorted(rs.items()):
            mem = r.get("mem") or {}
            print(f"    {s:>4} | " + " ".join(f"{(mem[str(t)]['distinct'] if str(t) in mem else '--'):>4}" for t in AT)
                  + f"   map at {ITERS}: {(mem.get(str(ITERS)) or {}).get('ch_map')}")
        md = medians(rs)
        meds[conf] = md
        print(f"    medians over the seeds (norms: median over positions and sequences; sat = fraction of h units with |h| > "
              f"0.95; H(gr) in nats, log k = {'2.773' if S == 8 else '1.386'}):")
        print(f"    {'update':>6} {'n':>2} | " + " ".join(f"{lab:>11}" for _, lab in FIELDS))
        for t in AT:
            print(f"    {t:>6} {md[t]['n']:>2} | " + " ".join(fmt(md[t][f], 11) for f, _ in FIELDS))
        if nrep < len(rs):
            rd = "withheld (a rerun does not reproduce its record)"
        else:
            rd = label(md)
        print(f"  LABEL S39 {conf}: {rd}")
        out[conf] = dict(label=rd, reproduce=nrep, n=len(rs), complete=len(rs) == len(a["seeds"]),
                         dec_key={str(t): md[t]["dec_key"] for t in AT}, dec_stream={str(t): md[t]["dec_stream"] for t in AT})
    print(f"  HINGE vs NO HINGE (S=8, Muon; medians over seeds 260-264): D8_HINGE_M | D8_SLOW_M")
    cols = ("dec_key", "dec_stream", "win_key", "wh_key", "sat_key", "entropy_key", "distinct")
    print(f"    {'update':>6} | " + " ".join(f"{dict(FIELDS)[c]:>11}" for c in cols) + " | "
          + " ".join(f"{dict(FIELDS)[c]:>11}" for c in cols))
    for t in AT:
        h, s = meds["D8_HINGE_M"][t], meds["D8_SLOW_M"][t]
        print(f"    {t:>6} | " + " ".join(fmt(h[c], 11) for c in cols) + " | " + " ".join(fmt(s[c], 11) for c in cols))
    return out
