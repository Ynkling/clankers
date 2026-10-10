#!/usr/bin/env python
"""
explore_i_pilot.py — EXPLORATORY, not a result. Session I, screen S74: make the ICMC task usable (constants only, no
readings). Task: explore_i_task.ICMCTask; harness, models and metrics: explore_i_common; CHECKs: explore_i_checks.

ARMS (seeds 800-803; 12000 updates; evaluation every 1200 on 1024 held-out sequences; the ADAM recipe)
  ORACLE  the perfect gate, k = S
  SINGLE  one channel, no gate (k = 1)
  NOMIX   S = 1 source per sequence at the same L (k = 1): the floor without interference. Its configuration keeps V,
          LMIN..LMAX and the total block count: NB_nomix = S·NB, M_nomix = S·M.
USABLE at a configuration (the user's rule, fixed): ORACLE's SET accuracy at the end >= 0.9 and SINGLE's <= ORACLE's - 0.2
  (paired seed) on >= 3/4 seeds. Adjust V, LMIN..LMAX, NB, L, B (never the model) until it is; every change is logged in
  explore_out/I/s74_pilot_log.md with its reason; the wall time per run is recorded (rec["secs"]).
STOP: if nothing is usable within a day of compute, report and stop.

CONFIGURATIONS (each appended here, with its reason, before it runs; see the log for the reasoning in full)
  P0  the starting point L = 192, NB = 3, LMIN..LMAX = 6..14, V = 16, B = 32 is infeasible: at a fixed length every
      source has M = (L - S·NB)/S words in NB blocks of 6..14, so 18 <= M <= 42 and L <= 90 at S = 2 (L <= 180 at
      S = 4). Not run.
  P1  the exact Bayes bound on SET accuracy (explore_i_task.ideal_set_acc "own": tables are uniform and independent, so
      given the observed successors O of w in its source, every 3-set containing O is equally likely and no predictor
      beats 1/C(V-|O|, 3-|O|) per position) is below 0.9 for V >= 5 at every L <= 288 tried and for V = 4 at S = 2,
      L <= 288 (0.90) and at S = 4, L <= 288 (0.82); at S = 4 it needs L ~ 768 (two sources' chains at S = 2's
      L = 384), about 2.5 s per update on this CPU, ~10 h per 12000-update run: not affordable in the pilot's day,
      so S = 4 is not run (the bound, not a run, excludes every affordable S = 4 configuration). Not run.
  C1  S = 2, V = 4, LMIN..LMAX = 6..14, NB = 16, M = 176 (L = 384), B = 32: the shortest L at which the bound (0.93)
      leaves any margin above 0.9; the faded ideal learner (memory 0.95 per token, evidence >= 0.05) predicts ORACLE
      ~0.85 and SINGLE ~0.67, and on a 3600-update smoke test at L = 96 the models matched the ideal learners within
      0.01, so C1 asks whether the model's memory reaches past the faded learner's horizon. NOMIX: NB = 32, M = 352.
      RESULT: NOT USABLE (ORACLE >= 0.9 on 4/4; SINGLE <= ORACLE - 0.2 on 2/4); NOMIX stopped after the decision.
  C2  S = 2, V = 4, LMIN..LMAX = 3..7, NB = 40, M = 200 (L = 480), B = 32: shorter blocks raise the mixed memory's
      interference (faded mixed learner 0.60 against C1's 0.67) and the longer L keeps the bound at 0.936. NOMIX:
      NB = 80, M = 400.

Run:  python explore_i_pilot.py --config C1 [--workers 4] [--report-only]
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

NAME = "s74_pilot"
SEEDS = (800, 801, 802, 803)
ITERS = 12000
CONFIGS = {
    "C1": dict(S=2, V=4, NB=16, lmin=6, lmax=14, M=176),
    "C2": dict(S=2, V=4, NB=40, lmin=3, lmax=7, M=200),
}
SPECS = {
    "ORACLE": lambda S: dict(gate="perfect", k=S, recipe="adam"),
    "SINGLE": lambda S: dict(gate="none", k=1, recipe="adam"),
    "NOMIX": lambda S: dict(gate="none", k=1, recipe="adam"),
}


def task_for(cfg, arm):
    c = dict(CONFIGS[cfg])
    if arm == "NOMIX":
        c.update(S=1, NB=c["S"] * c["NB"], M=c["S"] * c["M"])
    return it.ICMCTask(**c)


def job(p):
    task = task_for(p["cfg"], p["base_arm"])
    spec = SPECS[p["base_arm"]](task.S if p["base_arm"] == "ORACLE" else 1)
    rec = ic.run_one(task, spec, p["seed"], p["iters"], tag=f"{p['arm']}|{p['seed']}")
    rec.update(config=p["cfg"], task=task.label, spec=spec)
    return rec


def report(cfg):
    st = ic.load(NAME)
    R = {a: ic.runs(NAME, f"{cfg}_{a}", st) for a in SPECS}
    lines = [f"S74 pilot, configuration {cfg}: {task_for(cfg, 'ORACLE').label}; NOMIX {task_for(cfg, 'NOMIX').label}",
             f"code SHA {st['meta'].get('sha')}, CPU {st['meta'].get('cpu')}", ""]
    lines.append("| arm | seed | SET end | SET 1 / 2 / 3-4 / 5+ | loss end | probe L1/L2/L3/final | minutes |")
    lines.append("|---|---|---|---|---|---|---|")
    for a, rr in R.items():
        for s, r in sorted(rr.items()):
            e = r["end"]
            pr = e.get("probe") or {}
            lines.append(f"| {a} | {s} | {e['set']:.3f} | " + " / ".join(f"{e['set_' + b]:.3f}" for b in it.BIN_NAMES)
                         + f" | {e['loss']:.3f} | " + " / ".join(f"{pr[k]['acc']:.2f}" for k in ("L1", "L2", "L3", "final")
                                                                  if k in pr) + f" | {r['secs'] / 60:.0f} |")
    ok = []
    for s in SEEDS:
        if s in R["ORACLE"] and s in R["SINGLE"]:
            o, g = R["ORACLE"][s]["end"]["set"], R["SINGLE"][s]["end"]["set"]
            ok.append(o >= 0.9 and g <= o - 0.2)
    n_ok = sum(ok)
    usable = len(ok) == len(SEEDS) and n_ok >= 3
    lines.append("")
    lines.append(f"USABLE (ORACLE >= 0.9 and SINGLE <= ORACLE - 0.2 on >= 3/4 seeds): {n_ok}/{len(ok)} seeds -> "
                 f"{'USABLE' if usable else 'NOT USABLE' if len(ok) == len(SEEDS) else 'incomplete'}")
    for a, rr in R.items():
        if rr:
            v = [r["end"]["set"] for r in rr.values()]
            t = [r["secs"] / 60 for r in rr.values()]
            lines.append(f"  {a}: SET median {statistics.median(v):.3f} (range {min(v):.3f}-{max(v):.3f}); "
                         f"wall time per run median {statistics.median(t):.0f} min")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(ic.OUT_DIR, f"s74_{cfg}_summary.md"), "w") as f:
        f.write(txt + "\n")
    return usable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="C1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--skip-checks", action="store_true")
    a = ap.parse_args()
    os.makedirs(ic.OUT_DIR, exist_ok=True)
    if not a.report_only:
        import explore_i_checks as ck
        import torch
        torch.set_num_threads(1)
        if not a.skip_checks:
            t0 = time.time()
            print(f"[S74] CHECKs ({a.config})", flush=True)
            res = ck.checks([task_for(a.config, "ORACLE"), task_for(a.config, "NOMIX")])
            ck.assert_checks(res, log=lambda s: print(s, flush=True))
            st = ic.load(NAME)
            st["meta"].setdefault("checks", {})[a.config] = dict(when=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                                 secs=time.time() - t0,
                                                                 results=json.loads(json.dumps(res, default=str)))
            ic.save(NAME, st)
        order = ("ORACLE", "SINGLE", "NOMIX")
        jobs = [dict(arm=f"{a.config}_{arm}", base_arm=arm, cfg=a.config, seed=s, iters=ITERS,
                     est=len(order) - order.index(arm)) for arm in order for s in SEEDS]
        ic.run_jobs(NAME, "explore_i_pilot", jobs, workers=a.workers, log=lambda s: print(s, flush=True))
    report(a.config)


if __name__ == "__main__":
    main()
