#!/usr/bin/env python
"""
explore_f_common.py — EXPLORATORY, not a result. Session F's harness. It imports explore_common, explore_common16 and
explore_main9c unchanged and uses explore_common16's machinery as batches 16-18 did:
  - CACHE KEY "arm|seed|code SHA (12 hex)|CPU model"; a stored run is reused only when its code SHA is the screen's
    current one AND it ran on this container's CPU (no other CPU is accepted: STATE["other_cpus"] stays empty, so
    pairing bit for bit happens only within this CPU; recorded runs from other CPUs are compared by counts only).
  - CODE SHA: SHA-1 over the sources of every repository module the screen's child module imports (explore_common16.
    code_sha, in a fresh child of the same kind as the runs).
  - RETRY: a run with ok = False is stored with its error and re-queued, up to 3 attempts.
  - POOL: 4 workers, 1 torch thread each, every run in a child process; jobs longest-first by projected worst case;
    with a segment budget a worker only starts a run whose projected worst case fits in the time left.
  - CHECKs: explore_f_delta_checks.py runs (asserted) at the start of every segment of every batch.
Outputs: explore_out/F/<screen>_results.json (runs), explore_out/F/<batch>_runtime.json (timings), logs and reports.

Statistics (fixed for every Session F screen): Wilson 95% intervals (z = 1.96) with every count; bands on n runs:
RELIABLE >= ceil(0.9 n), MAJORITY >= ceil(0.5 n), MINORITY >= 1, NEVER 0; exact one-sided McNemar on paired seeds
(P(Bin(b + c, 1/2) >= b), b = pairs where the first arm alone succeeded: test_conv_lr.mcnemar_greater); one-sided
Fisher (test_router_confirm.fisher_greater) between unpaired groups.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16

F_DIR = os.path.join(ec.OUT_DIR, "F")
WORKERS = 4
EVAL_EVERY = 1200


# ── Statistics ───────────────────────────────────────────────────────────────
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def band(k, n):
    if n == 0:
        return "--"
    if k >= math.ceil(0.9 * n):
        return "RELIABLE"
    if k >= math.ceil(0.5 * n):
        return "MAJORITY"
    return "MINORITY" if k >= 1 else "NEVER"


def mcnemar_greater(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n


def fisher_greater(a, n1, b, n2):
    from test_router_confirm import fisher_greater as fg
    return fg(a, n1, b, n2)


def count_str(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} [{lo:.2f}, {hi:.2f}] {band(k, n)}"


def paired(new, old, seeds):
    """new, old: {seed: bool}. Counts over seeds in both; b = new only, c = old only; one-sided p (new > old)."""
    ss = [s for s in seeds if s in new and s in old]
    b = sum(1 for s in ss if new[s] and not old[s])
    c = sum(1 for s in ss if old[s] and not new[s])
    return dict(n=len(ss), new=sum(bool(new[s]) for s in ss), old=sum(bool(old[s]) for s in ss), b=b, c=c,
                p_new_gt_old=mcnemar_greater(b, c), p_old_gt_new=mcnemar_greater(c, b))


def med(xs):
    return c16.med(xs)


# ── Harness ──────────────────────────────────────────────────────────────────
def name(screen_name):
    return f"F/{screen_name}"


def setup(screens):
    os.makedirs(F_DIR, exist_ok=True)
    cpu = ec.cpu_model()
    c16.STATE["batch_cpu"] = cpu
    c16.STATE["other_cpus"] = []
    for m in screens:
        sha, rows = c16.code_sha(m.CHILD, main=m.MAIN)
        c16.STATE["shas"][m.NAME] = sha
        print(f"  code SHA {m.NAME}: {sha[:12]} over {len(rows)} modules "
              f"({', '.join(r[0] + ('@main' if r[2] != 'branch' else '') for r in rows)})", flush=True)
    return cpu


def run_checks(tag):
    t = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, "explore_f_delta_checks.py"), "--tag", tag], cwd=HERE,
                       env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"), capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        if "CHECK" in ln or "ALL CHECKS" in ln:
            print("  " + ln.strip(), flush=True)
    assert r.returncode == 0, f"explore_f_delta_checks failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    print(f"  (explore_f_delta_checks: {time.time() - t:.0f} s, all passed)", flush=True)


def runtime_path(batch):
    return os.path.join(F_DIR, f"{batch}_runtime.json")


def timing(batch, screens, steps=150):
    """Per (screen, arm): [seconds per optimizer step, seconds per evaluation], measured with the arms' children run 4 at
    a time (pool load); reused from the batch's runtime file when present and the code SHAs are unchanged."""
    p = runtime_path(batch)
    old = json.load(open(p)) if os.path.exists(p) else {}
    want = {m.NAME: c16.STATE["shas"][m.NAME] for m in screens}
    if old.get("shas") == want and old.get("cpu") == c16.STATE["batch_cpu"] and all(
            f"{m.NAME}:{a}" in old.get("t", {}) for m in screens for a in m.ARMS):
        print(f"  timings reused from {os.path.relpath(p, HERE)}", flush=True)
        return old["t"]
    from concurrent.futures import ThreadPoolExecutor
    todo = [(m, a) for m in screens for a in m.ARMS]
    t0 = time.time()

    def one(ma):
        m, a = ma
        return f"{m.NAME}:{a}", m.timing(a, steps)
    with ThreadPoolExecutor(WORKERS) as ex:
        t = dict(ex.map(one, todo))
    json.dump(dict(shas=want, cpu=c16.STATE["batch_cpu"], t=t, measured=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(p, "w"), indent=1)
    print(f"  timings measured ({time.time() - t0:.0f} s): " +
          ", ".join(f"{k} {v[0] * 1000:.1f} ms/step" for k, v in t.items()), flush=True)
    return t


def dur_fn(t, screens):
    mods = {m.__name__: m for m in screens}

    def dur(j):
        m = mods[j["screen"]]
        a = m.ARMS[j["arm"]]
        step, ev = t[f"{m.NAME}:{j['arm']}"]
        return a["iters"] * step + (a["iters"] // EVAL_EVERY) * ev + 20
    return dur


def projection(jobs, dur, workers=WORKERS):
    tot = sum(dur(j) for j in jobs)
    loads = [0.0] * workers
    for d in sorted((dur(j) for j in jobs), reverse=True):
        i = loads.index(min(loads))
        loads[i] += d
    return tot, max(loads) if jobs else 0.0


def tagline(screens):
    mods = {m.__name__: m for m in screens}

    def line(j, rec):
        return mods[j["screen"]].line(j["arm"], rec)
    return line


def drive(batch, screens, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-min", type=float, default=None, help="segment budget (minutes); None: no budget")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--no-checks", action="store_true", help="(report-only passes)")
    ap.add_argument("--only", default=None, help="comma-separated arm names")
    args = ap.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    pv = ec.provenance()
    print("=" * 110)
    print(f"{ec.BANNER.upper()} — Session F {batch} ({', '.join(m.NAME for m in screens)})")
    print(f"  cpu {pv['cpu']} | torch {pv['torch']} | git {pv['git']} | python {pv['python']} | torch threads "
          f"{torch.get_num_threads()} | started {pv['started']}")
    print("=" * 110, flush=True)
    cpu = setup(screens)
    if not args.report_only:
        if not args.no_checks:
            run_checks(f"{batch}_{time.strftime('%Y%m%d_%H%M%S')}")
        for m in screens:
            if hasattr(m, "check"):
                assert m.check(), f"{m.NAME}: a screen CHECK failed"
        jobs, skipped = c16.queue(screens, cpu)
        if args.only:
            keep = set(args.only.split(","))
            jobs = [j for j in jobs if j["arm"] in keep]
        for s in skipped:
            print(f"  skipped {s}")
        t = timing(batch, screens)
        dur = dur_fn(t, screens)
        tot, mk = projection(jobs, dur)
        print(f"  {len(jobs)} runs queued; projected worst case {tot / 3600:.2f} h of runs, makespan {mk / 3600:.2f} h on "
              f"{WORKERS} workers" + (f"; segment budget {args.budget_min:.0f} min" if args.budget_min else ""), flush=True)
        if jobs:
            n = c16.run_pool(jobs, WORKERS, dur, budget_s=None if args.budget_min is None else 60 * args.budget_min,
                             line=tagline(screens))
            print(f"  segment saved {n} runs", flush=True)
        left, _ = c16.queue(screens, cpu)
        print(f"  runs left: {len(left)}", flush=True)
        if left:
            return False
    for m in screens:
        m.report()
    return True


def load(screen):
    return ec.load_store(screen.NAME)


def runs(screen, arm, seeds=None):
    return c16.runs_of(screen, arm, seeds)
