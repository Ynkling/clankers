#!/usr/bin/env python
"""
explore_h_run.py — EXPLORATORY, not a result. Session H's driver. Same harness and rules as batches 16-18
(explore_common16, imported read-only): 1 torch thread per run; runs cached under arm|seed|code SHA (12 hex)|CPU model in
explore_out/H/<screen>_results.json; a run with ok = False is stored with its error and retried (up to 3 attempts); no arm
is cut. Every invocation first reproduces X's recorded arm A (seed 160, 2400 updates) bit for bit (a failure stops it).
The screen CHECKs (each screen's check(): unit checks and the record equalities with the new code paths disabled) run
whenever the screen's code SHA differs from the one stored at its last full CHECK pass (explore_out/H/checks.json); a
failed CHECK stops the invocation before any screen run. Only Adam runs are used in session H, so pairing within this
container's CPU is bit for bit; records from other CPUs are compared by counts only.

Run:  python explore_h_run.py SCREEN [SCREEN ...] [--workers 4] [--report-only]
      SCREEN: collapse (S58) | eight (S58b) | stability (S60) | copy2x2 (S61) | query (S62) | hardness (S63) | two (S78) | constants (S79)
"""

import argparse
import importlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_main9c as mt

MODS = {"collapse": "explore_h_collapse", "eight": "explore_h_eight", "stability": "explore_h_stability",
        "copy2x2": "explore_h_copy2x2", "query": "explore_h_query_gate", "hardness": "explore_h_hardness",
        "two": "explore_h_two", "constants": "explore_h_constants"}
OUT = os.path.join(ec.OUT_DIR, "H")
CHECKS = os.path.join(OUT, "checks.json")


def load_checks():
    if os.path.exists(CHECKS):
        with open(CHECKS) as f:
            return json.load(f)
    return {}


def line(j, rec):
    if not rec.get("ok"):
        return ""
    s = f"acc {rec.get('acc'):.3f}  trans {str(rec.get('transition')):>5}  outcome {rec.get('outcome')}"
    e = rec.get("end") or {}
    if e.get("ch_map") is not None:
        s += f"; map {e.get('ch_map')}"
    kr = ((rec.get("keyroute") or {}).get("end") or {}).get("ch_map_key")
    if kr is not None:
        s += f" key {kr}"
    f = [c["step"] for c in rec.get("checks") or [] if c.get("fired")]
    if f:
        s += f"; {rec.get('trigger')} fired at {f}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("screens", nargs="+", choices=sorted(MODS))
    ap.add_argument("--workers", type=int, default=ec.WORKERS)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    screens = [importlib.import_module(MODS[s]) for s in args.screens]
    for m in list(screens):
        if hasattr(m, "configure") and not m.configure():
            screens.remove(m)
    if not screens:
        sys.exit(0)
    cpu = ec.cpu_model()
    pv = ec.print_banner("session H: " + ", ".join(m.NAME for m in screens))
    c16.STATE["batch_cpu"] = cpu
    c16.STATE["other_cpus"] = []
    shas = {}
    for m in screens:
        sha, rows = c16.code_sha(m.CHILD, m.MAIN)
        shas[m.NAME] = sha
        print(f"  code SHA {m.NAME}: {sha[:12]} over {len(rows)} modules ({sum(1 for r in rows if r[2] != 'branch')} served from "
              f"main at {mt.MAIN_SHA[:7]}): " + " ".join(r[0] for r in rows), flush=True)
    c16.STATE["shas"] = shas
    print(f"  CPU {cpu}; git {ec.git_head()}; torch {pv.get('torch') if isinstance(pv, dict) else ''}", flush=True)
    if not args.report_only:
        if not ec.repro_check():
            print("  ! this container does not reproduce X's arm A: stopping.")
            sys.exit(1)
        rec = load_checks()
        for m in screens:
            done = rec.get(m.NAME)
            if done and done.get("sha") == shas[m.NAME] and done.get("cpu") == cpu:
                print(f"  CHECKs of {m.NAME} skipped: passed at {done['time']} with this code SHA on this CPU (log above)", flush=True)
                continue
            t0 = time.time()
            if not m.check():
                print(f"  ! CHECK failed in {m.NAME}; stopping before any screen run.")
                sys.exit(1)
            rec[m.NAME] = dict(sha=shas[m.NAME], cpu=cpu, git=ec.git_head(),
                               time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), minutes=(time.time() - t0) / 60)
            with open(CHECKS, "w") as f:
                json.dump(rec, f, indent=1)
            print(f"  all CHECKs of {m.NAME} passed ({(time.time() - t0) / 60:.1f} min)", flush=True)
        for m in screens:
            st = ec.load_store(m.NAME)
            st["meta"].setdefault("provenance", pv)
            st["meta"]["arms"] = {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in a.items()}
                                  for k, a in m.ARMS.items()}
            st["meta"].setdefault("code_shas", [])
            if shas[m.NAME] not in st["meta"]["code_shas"]:
                st["meta"]["code_shas"].append(shas[m.NAME])
            ec.save_store(m.NAME, st)
        jobs, skipped = c16.queue(screens, cpu)
        if skipped:
            print(f"  not run: {skipped}")
        print(f"  {len(jobs)} runs queued on {args.workers} workers (1 torch thread each)", flush=True)
        t0 = time.time()
        c16.run_pool(jobs, args.workers, lambda j: sys.modules[j["screen"]].ARMS[j["arm"]]["iters"], None, line)
        print(f"  wall clock {(time.time() - t0) / 60:.1f} min", flush=True)
    for m in screens:
        res, got = m.report()
        if hasattr(m, "per_seed_log"):
            m.per_seed_log(got)
        print(f"  {m.NAME}: readings {res['readings']}; complete {res['complete']}; valid {res['valid']}", flush=True)


if __name__ == "__main__":
    main()
