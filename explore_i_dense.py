#!/usr/bin/env python
"""
explore_i_dense.py — EXPLORATORY, not a result. Session I, screen S75: does the channel partition emerge under a dense
next-token loss on interleaved sources? Task explore_i_task.ICMCTask at the pilot's configuration; harness, models,
recipes and metrics explore_i_common (its docstring defines every model, recipe and metric); CHECKs explore_i_checks.
Everything below is fixed and committed before any S75 run.

CONSTANTS (from S74; explore_out/I/s74_pilot_log.md and report_1.md)
  __CONSTANTS__

ARMS (S = 2, k = S = 2; seeds __SEEDS_TXT__; __ITERS__ updates; evaluation every 1200 on 1024 held-out sequences)
  ORACLE      perfect gate, ADAM (validity; 810-811)
  SINGLE      one channel, ADAM
  WIN3_SPLIT  LOCAL3 + SLOW + SPLIT
  WIDE_SPLIT  WIDE (width LMAX + 2) + SLOW + SPLIT
  REG3        Session G's register gate at all three layers + SLOW, no nudge, no split
  WIDE_SLOW   WIDE + SLOW, no split
  REG3_NUDGE  REG3 + the 5% labelled nudge
  SPLIT_THR (the trigger's SET-accuracy threshold; the rise stays 0.02): __SPLIT_THR__
  Every arm shares the seed's batches; the learned arms share the stack's initial parameters (built from the same arm A
  model); SINGLE and ORACLE draw the stack from the same seed.
OUTCOMES (at the end, from the final evaluation)
  ROUTED := at every gate layer (one for the window gates; three for REG3), eta^2 by source of the gate at word positions
            >= 0.9 and a one-to-one source -> channel map at word positions.
  GAP    := (SET(arm) - SET(SINGLE, same seed)) / (SET(ORACLE) - SET(SINGLE, same seed)), SET(ORACLE) = the mean final SET
            accuracy of the two ORACLE runs; the fraction of the interference gap closed. Also per position bin.
  VALID  := both ORACLE runs end with SET >= 0.9 (the pilot's usability level); otherwise every reading is UNTESTED.
STATISTICS: Wilson 95% intervals and bands with every count; exact one-sided McNemar on paired seeds ("A beats B" :=
  p < 0.05, which over at most ten pairs needs at least 5 vs 0).
READINGS (__READING_SCALE__)
  R1 "the dense loss supplies the cue"                 WIDE_SPLIT ROUTED >= __HI__ and median GAP >= 0.8
  R2 "the partition emerges with the three-token window" WIN3_SPLIT ROUTED >= __HI__
  R3 "the dense loss does not supply the cue"          every label-free learned-gate arm (WIN3_SPLIT, WIDE_SPLIT, REG3,
                                                       WIDE_SLOW) ROUTED <= __LO__ (then reported: eta^2 by source at
                                                       block positions 1-2 against 5+, and the probe: does the stack carry
                                                       the source here where the sparse-loss stacks did not?)
  R4 "the register is discovered under a dense loss"    REG3 beats WIN3_SPLIT on ROUTED
  R5 "the split matters here"                          WIDE_SPLIT beats WIDE_SLOW on ROUTED (shared seeds)
  Descriptive: GAP by position within block for every arm; REG3_NUDGE (labelled) beside the label-free arms.
S = 4: __S4__

Run:  python explore_i_dense.py [--workers 4] [--report-only]
"""

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_i_task as it
import explore_i_common as ic

NAME = "s75_dense"
CONFIG = None                      # filled from S74 before the commit
ITERS = None
SPLIT_THR = None
SEEDS_ALL = None                   # per arm
HI, LO = None, None
LEARNED = ("WIN3_SPLIT", "WIDE_SPLIT", "REG3", "WIDE_SLOW")
SPECS = {
    "ORACLE": dict(gate="perfect", recipe="adam"),
    "SINGLE": dict(gate="none", k=1, recipe="adam"),
    "WIN3_SPLIT": dict(gate="win3", recipe="slow", split=True),
    "WIDE_SPLIT": dict(gate="wide", recipe="slow", split=True),
    "REG3": dict(gate="reg3", recipe="slow"),
    "WIDE_SLOW": dict(gate="wide", recipe="slow"),
    "REG3_NUDGE": dict(gate="reg3", recipe="slow", nudge=True),
}
COST = {"ORACLE": 1.0, "SINGLE": 1.0, "WIN3_SPLIT": 1.05, "WIDE_SPLIT": 1.05, "REG3": 1.4, "WIDE_SLOW": 1.05,
        "REG3_NUDGE": 1.45}


def task_for(S):
    return it.ICMCTask(**dict(CONFIG, S=S, NB=CONFIG["NB"] * 2 // S, M=CONFIG["M"] * 2 // S))


def spec_for(arm, S):
    s = dict(SPECS[arm])
    s.setdefault("k", S)
    return s


def job(p):
    task = task_for(p["S"])
    spec = spec_for(p["base_arm"], p["S"])
    rec = ic.run_one(task, spec, p["seed"], p["iters"], split_thr=SPLIT_THR, tag=f"{p['arm']}|{p['seed']}")
    rec.update(task=task.label, spec=spec)
    return rec


# ── outcomes ──────────────────────────────────────────────────────────────────
def routed(r):
    rt = r["end"].get("routing") or {}
    return bool(rt) and all(v["eta"] >= 0.9 and v["one_to_one"] for v in rt.values())


def gap(r, single, oracle_set, field="set"):
    s = single["end"][field]
    o = oracle_set[field]
    if o is None or s is None or r["end"][field] is None or o - s == 0:
        return None
    return (r["end"][field] - s) / (o - s)


def count_line(arm, R, outcome):
    rr = R.get(arm, {})
    n = len(rr)
    k = sum(outcome(r) for r in rr.values())
    lo, hi = ic.wilson(k, n)
    return k, n, f"{k}/{n} [{lo:.2f}, {hi:.2f}] {ic.band(k, n)}"


def paired(ra, rb, f):
    seeds = sorted(set(ra) & set(rb))
    b = sum(f(ra[s]) and not f(rb[s]) for s in seeds)
    c = sum(f(rb[s]) and not f(ra[s]) for s in seeds)
    return b, c, ic.mcnemar_greater(b, c), len(seeds)


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def fmt(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


def report(S=2):
    st = ic.load(NAME)
    arms = list(SPECS) if S == 2 else ["ORACLE", "SINGLE"] + list(S4_ARMS)
    R = {a: ic.runs(NAME, f"S{S}_{a}", st) for a in arms}
    L = [f"S75 (S = {S}): {task_for(S).label}; {ITERS} updates; code SHA {st['meta'].get('sha')}; CPU {st['meta'].get('cpu')}",
         ""]
    orc = list(R["ORACLE"].values())
    valid = len(orc) == 2 and all(r["end"]["set"] >= 0.9 for r in orc)
    oset = {f: (statistics.fmean(r["end"][f] for r in orc) if orc else None)
            for f in ["set"] + [f"set_{b}" for b in it.BIN_NAMES]}
    L.append(f"VALIDITY: ORACLE final SET {[round(r['end']['set'], 3) for r in orc]} -> "
             f"{'VALID' if valid else 'UNTESTED' if len(orc) == 2 else 'incomplete'}")
    L += ["", "| arm | ROUTED (Wilson; band) | median GAP | GAP by bin 1 / 2 / 3-4 / 5+ | median SET | splits (on target) | "
              "median eta (layers) | margin |", "|---|---|---|---|---|---|---|---|"]
    sing = R["SINGLE"]
    gaps = {}
    for a in arms:
        rr = R[a]
        if not rr:
            continue
        _, _, cl = count_line(a, R, routed)
        g = {s: gap(r, sing[s], oset) for s, r in rr.items() if s in sing}
        gaps[a] = g
        gb = [med([gap(r, sing[s], oset, f"set_{b}") for s, r in rr.items() if s in sing]) for b in it.BIN_NAMES]
        sp = [r.get("splits") for r in rr.values() if "splits" in r]
        on = sum(c["on_target"] for r in rr.values() for c in r.get("checks", []) if c["fired"])
        etas = [med([r["end"]["routing"][Ly]["eta"] for r in rr.values() if r["end"].get("routing")])
                for Ly in sorted((next(iter(rr.values()))["end"].get("routing") or {}))]
        mg = med([statistics.fmean(v["margin"] for v in r["end"]["routing"].values()) for r in rr.values()
                  if r["end"].get("routing")])
        L.append(f"| {a} | {cl} | {fmt(med(g.values()))} | " + " / ".join(fmt(x) for x in gb) +
                 f" | {fmt(med([r['end']['set'] for r in rr.values()]), 3)} | "
                 f"{(str(sum(sp)) + ' (' + str(on) + ')') if sp else '-'} | {' / '.join(fmt(x) for x in etas) or '-'} | {fmt(mg)} |")
    rd = {}
    if S == 2:
        def n_r(a):
            return sum(routed(r) for r in R[a].values())
        k1 = n_r("WIDE_SPLIT")
        rd["R1 the dense loss supplies the cue"] = (k1 >= HI and (med(gaps.get("WIDE_SPLIT", {}).values()) or 0) >= 0.8,
                                                     f"WIDE_SPLIT ROUTED {k1}/{len(R['WIDE_SPLIT'])}, median GAP "
                                                     f"{fmt(med(gaps.get('WIDE_SPLIT', {}).values()))}")
        k2 = n_r("WIN3_SPLIT")
        rd["R2 the partition emerges with the three-token window"] = (k2 >= HI, f"WIN3_SPLIT ROUTED {k2}/{len(R['WIN3_SPLIT'])}")
        ks = {a: n_r(a) for a in LEARNED}
        rd["R3 the dense loss does not supply the cue"] = (all(v <= LO for v in ks.values()), f"ROUTED {ks}")
        b, c, p, n = paired(R["REG3"], R["WIN3_SPLIT"], routed)
        rd["R4 the register is discovered under a dense loss"] = (p < 0.05, f"REG3 vs WIN3_SPLIT {b} vs {c}, p {p:.3g}, n {n}")
        b, c, p, n = paired(R["WIDE_SPLIT"], R["WIDE_SLOW"], routed)
        rd["R5 the split matters here"] = (p < 0.05, f"WIDE_SPLIT vs WIDE_SLOW {b} vs {c}, p {p:.3g}, n {n}")
        best = sorted(LEARNED, key=lambda a: (-ks[a], -(med(gaps.get(a, {}).values()) or -9), LEARNED.index(a)))[:2]
        rd["S = 4 arms (two label-free learned arms with the most ROUTED at S = 2; ties by median GAP)"] = (None, str(best))
    else:
        k = max(sum(routed(r) for r in R[a].values()) for a in S4_ARMS)
        rd["carries to four sources (best arm ROUTED >= 8/10)"] = (k >= 8, f"best {k}")
    L += ["", "READINGS (pre-fixed):"]
    for name, (v, why) in rd.items():
        tag = "-" if v is None else ("APPLIES" if v and valid else "UNTESTED" if not valid else "does not apply")
        L.append(f"  {name}: {tag} ({why})")
    L += ["", "DIAGNOSTICS (per arm, medians at the end): eta^2 by source at word index 1-2 / 5+ (layer 1; REG3 all layers), "
              "probe accuracy (residual entering L1 / L2 / L3 / final -> source), REG3's beta^R at SRC / word"]
    for a in arms:
        rr = R[a]
        if not rr:
            continue
        e12 = med([r["end"]["routing"]["L1"]["eta_pos12"] for r in rr.values() if r["end"].get("routing")])
        e5 = med([r["end"]["routing"]["L1"]["eta_pos5p"] for r in rr.values() if r["end"].get("routing")])
        pr = {k: med([r["end"]["probe"][k]["acc"] for r in rr.values() if r["end"].get("probe")])
              for k in ("L1", "L2", "L3", "final")}
        bt = (fmt(med([r.get("beta_src") for r in rr.values()])), fmt(med([r.get("beta_word") for r in rr.values()])))
        L.append(f"  {a:<11} eta 1-2 {fmt(e12)} / 5+ {fmt(e5)}; probe " + " / ".join(fmt(v) for v in pr.values()) +
                 (f"; beta^R {bt[0]} / {bt[1]}" if a.startswith("REG3") else "") +
                 f"; minutes {fmt(med([r['secs'] / 60 for r in rr.values()]), 0)}")
    L += ["", "PER RUN: arm seed ROUTED SET GAP maps eta margin splits"]
    for a in arms:
        for s, r in sorted(R[a].items()):
            rt = r["end"].get("routing") or {}
            L.append(f"  {a:<11} {s} {'R' if routed(r) else '.'} {r['end']['set']:.3f} "
                     f"{fmt(gap(r, sing[s], oset) if s in sing else None)} "
                     + " ".join(f"{Ly}:{v['ch_map']}/{v['eta']:.2f}/{v['margin']:.2f}" for Ly, v in rt.items())
                     + (f" splits {r.get('splits')} at {[c['step'] for c in r.get('checks', []) if c['fired']]}"
                        if "splits" in r else ""))
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(ic.OUT_DIR, f"s75_S{S}_summary.md"), "w") as f:
        f.write(txt + "\n")


def jobs_for(S):
    arms = list(SPECS) if S == 2 else ["ORACLE", "SINGLE"] + list(S4_ARMS)
    out = []
    for a in arms:
        for s in SEEDS_ALL[S][a]:
            out.append(dict(arm=f"S{S}_{a}", base_arm=a, S=S, seed=s, iters=ITERS, est=COST[a] + (5 if a == "ORACLE" else 0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(ic.OUT_DIR, exist_ok=True)
    if not a.report_only:
        import torch
        import explore_i_checks as ck
        torch.set_num_threads(1)
        t0 = time.time()
        print(f"[S75] CHECKs (S = {a.S})", flush=True)
        res = ck.checks([task_for(a.S)])
        ck.assert_checks(res, log=lambda s: print(s, flush=True))
        st = ic.load(NAME)
        st["meta"].setdefault("checks", {})[f"S{a.S}"] = dict(when=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                               secs=time.time() - t0,
                                                               results=json.loads(json.dumps(res, default=str)))
        ic.save(NAME, st)
        ic.run_jobs(NAME, "explore_i_dense", jobs_for(a.S), workers=a.workers, log=lambda s: print(s, flush=True))
    report(a.S)


if __name__ == "__main__":
    main()
