#!/usr/bin/env python
"""
explore_batch12.py — EXPLORATORY, not a result. Batch 12 of the outside-ideas screens, with batches 1-11's
harness, rules and labels and their resume machinery. Everything below was fixed before any run.

  S36 explore_split_plateau  a label-free plateau trigger (every 2400 updates from 4800: probe accuracy
                             < 0.95 and a rise < 0.02 since the previous check) applying S24's SPLIT (W_g's
                             busiest row copied to the idlest, noise on both, W_g's optimizer state reset):
                             (a) S33's A_HINGE at S=4, k=4 (Muon), 28800 steps, seeds 240-249; (b) S33's
                             D8_HINGE (Muon) and main's HINGE_D8 (Adam) at S=8, k=16, 43200 steps, seeds
                             260-269; main's run paths at 9c5939e in a child process

BACKGROUND (docstring): see explore_split_plateau.

RECIPE: the Adam groups at test_slow_start.LR = test_stream_recipe.LR = test_stream_curriculum.LR = 1e-3
(read in the child; the branch's SUB_LR = 1e-3), passed explicitly; Muon lr 0.005 (S33's validity choice,
read from muon_scale_valid_results.json); the slow phase's non-gate groups at a tenth for updates 1-2400;
evaluation every 1200; the run paths' own early stops; 1 torch thread per run.

READINGS: in explore_split_plateau. An incomplete arm gets no reading.

CHECKS (before any run, after the repro check; any failure stops the batch)
- repro: X's arm A, seed 160, 2400 steps, bit-identical to X's record (in every segment).
- S36: with the accuracy threshold at 0 each arm equals its paired run bit for bit through 7200; on a
  synthetic gate the operation sets the rows as specified and leaves every other parameter and optimizer
  state unchanged; after a split W_g's Muon momentum (Adam state) is zero; the forced trigger splits at
  4800 and the run continues; the served modules.

SEQUENCE: repro check; CHECKs; the projection with pool-load timing (batch 10's method: one timing child
per arm at once, SPLIT_D8_M twice, so 4 processes; the median interval between optimizer steps); RUNTIME
RULE: if the projected worst-case makespan is over 9 h, SPLIT_D8_A is cut to 260-264 (stored); then one
pool, the runs submitted in waves of one per worker: (b) runs (43200 steps, ~77 min), then (a) runs (28800
steps, ~27 min), alternately, so that a 2-hour segment ends a wave of (a) runs rather than losing a wave of
(b) runs in flight (submission order only; every run is the same whatever its order).
SEGMENTS: batches 4-11's machinery (--resume against explore_out/batch12_checks.json, --report-only,
--stopped-early, --dry); a watcher outside this file pushes every saved run.

Run:  python explore_batch12.py [--workers 4] [--dry] [--resume] [--report-only]
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common2 as c2
import explore_split_plateau as s36
from test_binding_recipe import makespan
from test_binding_onset import EVAL_EVERY

SCREENS = (s36,)
DRY_ITERS = 3600
CHECKS_RECORD = os.path.join(ec.OUT_DIR, "batch12_checks.json")


def git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True, check=True).stdout


def code_changes(rev):
    files = git("diff", "--name-only", rev).split() + git("ls-files", "--others", "--exclude-standard").split()
    return sorted(f for f in set(files) if not f.startswith("explore_out/") and f != "explore_batch12.py")


def base(m):
    return m.NAME[4:] if m.NAME.startswith("dry_") else m.NAME


def make_dry():
    c2.DRY.update(on=True, iters=DRY_ITERS)
    for m in SCREENS:
        m.NAME = "dry_" + m.NAME
        m.ARMS = {k: dict(a, iters=min(a["iters"], DRY_ITERS), seeds=tuple(a["seeds"][:1])) for k, a in m.ARMS.items()}


def completeness(m, arm=None):
    st = ec.load_store(m.NAME)
    want = [(k, s) for k, a in m.ARMS.items() for s in a["seeds"] if arm in (None, k)]
    have = [1 for k, s in want if (r := st["runs"].get(f"{k}|{s}")) and r.get("ok")]
    return len(have), len(want)


def run_checks():
    t0 = time.time()
    for s in SCREENS:
        if not s.check():
            print(f"  ! CHECK failed in {s.NAME}; stopping.")
            sys.exit(1)
    print(f"  all CHECKs passed ({(time.time() - t0) / 60:.1f} min)", flush=True)


def projection(workers, dry):
    from concurrent.futures import ThreadPoolExecutor
    jobs = ["SPLIT4k4_M", "SPLIT_D8_M", "SPLIT_D8_A", "SPLIT_D8_M"]
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:          # each job is a child process
        res = list(ex.map(s36.timing, jobs))
    t = {}
    for arm, v in zip(jobs, res):
        t[arm] = max(t.get(arm, (0, 0)), tuple(v))                   # the slower of a duplicate

    def show(tag):
        d, rows = [], []
        for k, a in s36.ARMS.items():
            ts, te = t[k]
            per = a["iters"] * ts + (a["iters"] // EVAL_EVERY) * te
            d += [per] * len(a["seeds"])
            rows.append((k, len(a["seeds"]), per, a["iters"], ts, te))
        print(f"  PROJECTION{tag} (worst case: every run to its iters; pool-load timing: the median interval between "
              f"optimizer steps, one child per arm at once (SPLIT_D8_M twice), 1 thread each, this machine):")
        for k, n, per, it, ts, te in rows:
            print(f"    {s36.NAME:<14} {k:<11} {n:>2} runs x {per / 60:5.1f} min  ({it} steps x {ts * 1e3:.1f} ms + "
                  f"{it // EVAL_EVERY} evals x {te:.1f} s)")
        ms = makespan(d, workers)
        print(f"    total {sum(d) / 3600:.2f} h of runs; makespan on {workers} workers {ms / 3600:.2f} h (worst case)",
              flush=True)
        return ms
    ms = show("")
    cut = s36.decide_cut(ms / 3600, workers, dry=dry)
    if cut["cut"]:
        show(" AFTER THE CUT")
    return ms


def run_jobs_waves(workers, pv):
    """explore_common2.run_jobs2 (the same stores, worker, records and log lines) with the submission order in
    waves: `workers` (b) runs, then `workers` (a) runs, alternately."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    st = ec.load_store(s36.NAME)
    st["meta"].setdefault("provenance", pv or ec.provenance())
    st["meta"]["dry"] = c2.DRY["on"]
    st["meta"]["arms"] = {k: {kk: vv for kk, vv in a.items() if kk != "seeds"} | {"seeds": list(a["seeds"]),
                                                                                  "sched": c2.sched(a)}
                          for k, a in s36.ARMS.items()}
    ec.save_store(s36.NAME, st)
    todo = [dict(screen=s36.__name__, name=s36.NAME, arm=k, seed=seed, prio=a.get("prio", 1))
            for k, a in s36.ARMS.items() for seed in a["seeds"] if f"{k}|{seed}" not in st["runs"]]
    long_ = sorted((j for j in todo if s36.ARMS[j["arm"]]["cfg"] == "b"), key=lambda j: (j["seed"], j["arm"]))
    short = sorted((j for j in todo if s36.ARMS[j["arm"]]["cfg"] == "a"), key=lambda j: j["seed"])
    jobs = []
    while long_ or short:
        jobs += long_[:workers] + short[:workers]
        long_, short = long_[workers:], short[workers:]
    print(f"  {len(jobs)} runs queued on {workers} workers (1 torch thread each), submitted in waves: "
          + " ".join(f"{j['arm']}|{j['seed']}" for j in jobs), flush=True)
    if not jobs:
        return
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"), initializer=ec._worker_init) as ex:
        futs = [ex.submit(c2._worker2, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            job, rec = f.result()
            st["runs"][f"{job['arm']}|{job['seed']}"] = rec
            ec.save_store(job["name"], st)
            print(f"  [{n:>3}/{len(jobs)}] {(time.time() - t0) / 60:6.1f} min  {job['name']:<14} {job['arm']:<10} seed "
                  f"{job['seed']}  lr {rec['lr']:g} ({rec['sched']})\n          "
                  f"{ec.line(rec) if rec.get('ok') else 'FAILED ' + str(rec.get('error'))}  splits {rec.get('splits')}",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=ec.WORKERS)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--stopped-early", default=None)
    args = ap.parse_args()
    if args.dry:
        make_dry()
    dl = c2.dry_label()
    if args.resume:
        print("#" * 20 + f" RESUME {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC — the "
              "previous segment was stopped by the session's background time limit; saved runs are "
              f"reused, missing runs re-run ({ec.BANNER}) " + "#" * 20)
    pv = ec.print_banner("batch 12: split_plateau (S36)" + dl)
    c = s36.child("constants", {}, module="explore_muon_scale_child")
    assert c["tss_LR"] == c["tsr_LR"] == c["tscur_LR"] == s36.LR, c
    ch = s36.s33_choice()
    assert ch.get("a") == ch.get("b") == s36.MUON_LR, ch
    print(f"  recipe read in the child from main at {c['main_sha'][:7]}: test_slow_start.LR {c['tss_LR']:g}, "
          f"test_stream_recipe.LR {c['tsr_LR']:g}, test_stream_curriculum.LR {c['tscur_LR']:g} (Adam groups; the branch's "
          f"SUB_LR {ec.SUB_LR:g}); WARM {c['tss_WARM']}, TAU {c['tss_TAU']}, LAMBDA {c['tss_LAMBDA']}; Muon lr "
          f"{s36.MUON_LR:g} (S33's validity choice {ch}); schedules {c['tsr_REAL']} (a, total 28800), {c['tscur_REAL']} "
          f"(b, total set to 43200); 1 torch thread per run")
    for k, a in s36.ARMS.items():
        print(f"  lr read for {s36.NAME} {k}: {a['lr']:g} (Adam groups)" + (f"; Muon lr {a['muon_lr']:g}" if a["muon_lr"] else ""))
    print(f"  S36 readings: (a) 'the split fixes four-stream merges under Muon' if SPLIT4k4_M binds >= {s36.FIX_N}/10, 'it does "
          f"not' if <= {s36.NOT_N}/10, otherwise neither; (b) per arm 'it breaks the eight-stream stall' if >= {s36.BREAK_N}/10 "
          f"bind by 43200, otherwise 'it does not' (BOUND at 28800 printed, the paired outcome)")
    print(f"  queue order: the (b) arms' 43200-step runs first (arm priority), then (a)")
    if not args.report_only:
        if not ec.repro_check():
            print("  ! this container does not reproduce X: the recorded pairing is invalid; stopping.")
            sys.exit(1)
        rec = None
        if args.resume and os.path.exists(CHECKS_RECORD):
            with open(CHECKS_RECORD) as f:
                rec = json.load(f)
        changed = code_changes(rec["git"]) if rec else None
        if rec and not rec["code_changed"] and not changed:
            print(f"  CHECKs skipped (repro check above still run): the screen code is unchanged — no "
                  f"file outside explore_out/ and explore_batch12.py differs from {rec['git']}, where "
                  f"the full CHECK suite passed ({rec['time']}; its output is earlier in this log). "
                  f"git diff --stat {rec['git']}:")
            for ln in git("diff", "--stat", rec["git"]).rstrip().splitlines():
                print(f"    {ln}")
        else:
            if args.resume:
                print(f"  running every CHECK (record: {rec}; code changes since: {changed})")
            run_checks()
            if not args.dry and not args.resume:
                head = git("rev-parse", "--short", "HEAD").strip()
                with open(CHECKS_RECORD, "w") as f:
                    json.dump(dict(git=head, code_changed=code_changes("HEAD"),
                                   time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                                   banner=ec.BANNER), f, indent=1)
                print(f"  recorded the full CHECK pass at {head} in {CHECKS_RECORD}")
        s36.apply_cut()
        for s in SCREENS:
            c2.print_screen_header2(s)
        projection(args.workers, args.dry)
        t0 = time.time()
        run_jobs_waves(args.workers, pv)
        print(f"  batch wall clock {(time.time() - t0) / 60:.1f} min (this segment)")
    else:
        print("  REPORT ONLY: from the saved runs (no repro check, CHECKs or runs in this call)")
        s36.apply_cut()
    print("#" * 100)
    print(f"REPORTS — {ec.BANNER}{dl}")
    print("#" * 100)
    tagp = "DRY RUN " if args.dry else ""
    d = s36.report(ec.load_store(s36.NAME))
    v = {}
    for arm in s36.ARMS:
        n, N = completeness(s36, arm)
        if n == N:
            v[arm] = d[arm]["reading"]
        elif args.stopped_early and arm == args.stopped_early:
            v[arm] = f"stopped early, {n}/{N} seeds, no reading"
        else:
            v[arm] = f"INCOMPLETE, {n}/{N} runs, no reading"
        print(f"  {tagp}READING {s36.NAME} {arm}: {v[arm]}  [{ec.BANNER}]")
    print()
    print("#" * 100)
    print(f"SUMMARY — {ec.BANNER}{dl}")
    for arm in s36.ARMS:
        dd = d[arm]
        print(f"  S36 {arm}: {v[arm]} (BOUND {dd['bound']}/{dd['n']}, at 28800 {dd['bound28']}/{dd['n']} vs {s36.PAIR[arm]} "
              f"{dd['d28']['old']}/{dd['d28']['n']}; splits {dd['splits']}; HARM {dd['harm']} runs)")
    print("#" * 100)


if __name__ == "__main__":
    main()
