#!/usr/bin/env python
"""
explore_batch10.py — EXPLORATORY, not a result. Batch 10 of the outside-ideas screens (the revision:
order S32, S33, S31, then S29-S30 if time remains), with batches 1-9's harness, rules and labels and
their resume machinery. Everything below was fixed before any run.

  S32 explore_muon_slow    S25's MUON_HINGE without the hinge (MUON_SLOW), seeds 160-199; MUON_HINGE on
                           180-199 (160-179: S25's runs, re-run to their recorded stop for the logs);
                           extra routing evaluations at updates 300, 600, 900; screens' configuration
                           (S=2, P=4, k=2, grouped, no conv, 24000 steps), Muon lr 0.005
  S33 explore_muon_scale   MUON_SLOW and MUON_HINGE (main's HINGE hinge) in the main line's
                           configurations on main's run paths at 9c5939e (child process): (a) S=4 P=4
                           k=4, seeds 240-249 vs X's A4k4; (b) S=8 P=4 k=16, 8 streams from step 1,
                           seeds 260-269 vs X's HINGE_D8; Muon lr from the perfect gate (0.005, then
                           0.0025, then 0.01); 28800 steps
  S31, S29, S30            queued after S33; their specs were not received in this session (asked for).

BACKGROUND (docstring): see explore_muon_slow and explore_muon_scale.

RECIPE: the Adam groups at lr test_channel_binding.SUB_LR = 1e-3 (S32; S33 reads test_slow_start.LR,
test_stream_recipe.LR, test_stream_curriculum.LR in the child, all 1e-3), passed explicitly; the Muon
groups at the Muon lr; the slow phase's non-gate groups at a tenth for updates 1-2400. MAX_ITERS
test_short_conv.MAX_ITERS = 24000 (S32), test_stream_recipe.MAX_ITERS = test_stream_curriculum.TOTAL =
28800 (S33); evaluation every 1200; the run paths' own early stops; 1 torch thread per run.

READINGS: in explore_muon_slow (S32) and explore_muon_scale (S33). An incomplete screen gets no reading.

CHECKS (before any run, after the repro check; any failure stops the batch)
- repro: X's arm A, seed 160, 2400 steps, bit-identical to X's record (in every segment).
- S32: with the extra evaluations, MUON_HINGE|160 equals S25's recorded run bit for bit through 2400;
  the evaluations are logged at 300/600/900; one firing-log entry per counted firing; with the hinge's
  TAU at 1.0, MUON_HINGE equals MUON_SLOW bit for bit (1200 steps).
- S33: groups cover every parameter exactly once; each group's lr at updates 1, 2400, 2401; the
  perfect gate zeroes cross-stream scores at every layer; the child's run paths reproduce X's records
  through 1200; TAU 1.0: MUON_HINGE = MUON_SLOW (1200 steps); the served modules are exactly the test
  modules that differ between this branch and 9c5939e.

SEQUENCE: repro check; CHECKs; S33's validity (its Muon lr must be known before its runs are queued;
stored, a resume reuses it); the projection (S32 from S25's measured MuonAdam time per step on this
machine; S33 from the child's timing; S33's runtime rule); then one pool, S32's runs queued first.
SEGMENTS: batches 4-9's machinery (--resume against explore_out/batch10_checks.json, --report-only,
--stopped-early, --dry); a watcher outside this file pushes every saved run.

Run:  python explore_batch10.py [--workers 4] [--dry] [--resume] [--report-only]
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common2 as c2
import explore_muon_recipe as s25
import explore_muon_slow as s32
import explore_muon_scale as s33

SCREENS = (s32, s33)
ORDER = (s32, s33)
DRY_ITERS = 3600
CHECKS_RECORD = os.path.join(ec.OUT_DIR, "batch10_checks.json")


def git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True, check=True).stdout


def code_changes(rev):
    files = git("diff", "--name-only", rev).split() + git("ls-files", "--others", "--exclude-standard").split()
    return sorted(f for f in set(files) if not f.startswith("explore_out/") and f != "explore_batch10.py")


def base(m):
    return m.NAME[4:] if m.NAME.startswith("dry_") else m.NAME


def make_dry():
    c2.DRY.update(on=True, iters=DRY_ITERS)
    for m in SCREENS:
        m.NAME = "dry_" + m.NAME
        m.ARMS = {k: dict(a, iters=min(a["iters"], DRY_ITERS), seeds=tuple(a["seeds"][:1])) for k, a in m.ARMS.items()}


def apply_order():
    """S32's runs before S33's (the batch order); within S33, its own priorities (longest first)."""
    n = len(ORDER)
    for i, m in enumerate(ORDER):
        for a in m.ARMS.values():
            a["prio"] = 10 * (n - i) + a.get("prio", 1)
    print(f"  order: {', '.join(base(m) for m in ORDER)} — run_jobs2 queues runs by arm priority (S32 first; "
          f"within S33, (b) before (a), the longer runs first)")


def completeness(m, arm=None):
    st = ec.load_store(m.NAME)
    want = [(k, s) for k, a in m.ARMS.items() for s in a["seeds"] if arm in (None, k)]
    have = [1 for k, s in want if (r := st["runs"].get(f"{k}|{s}")) and r.get("ok")]
    return len(have), len(want)


def run_checks():
    t0 = time.time()
    for s in ORDER:
        if not s.check():
            print(f"  ! CHECK failed in {s.NAME}; stopping.")
            sys.exit(1)
    print(f"  all CHECKs passed ({(time.time() - t0) / 60:.1f} min)", flush=True)


def s32_projection(workers):
    """S32 worst case from S25's recorded MuonAdam runs on this machine (same model, optimizer, task):
    median seconds per step (secs_wall / stopped_at, evaluations included) x each run's iters."""
    st = ec.load_store(s25.NAME)["runs"]
    per = {arm: statistics.median(r["secs_wall"] / r["stopped_at"] for k, r in st.items()
                                  if k.startswith(arm + "|") and r.get("ok"))
           for arm in ("MUON_A", "MUON_HINGE")}
    durs, total = [], 0.0
    print(f"  S32 PROJECTION (worst case: every run to its iters; S25's recorded runs here: {per['MUON_A'] * 1e3:.1f} "
          f"ms/step MUON_A, {per['MUON_HINGE'] * 1e3:.1f} ms/step MUON_HINGE, evaluations included):")
    for k, a in s32.ARMS.items():
        ps = per["MUON_HINGE"] if a["hinge"] else per["MUON_A"]
        ds = [ps * s32.iters_for(k, s) if not c2.DRY["on"] else ps * a["iters"] for s in a["seeds"]]
        durs += ds
        total += sum(ds)
        print(f"    {s32.NAME:<14} {k:<15} {len(a['seeds']):>2} runs, {sum(ds) / 3600:5.2f} h "
              f"(max {max(ds) / 60 if ds else 0:4.1f} min per run)")
    ms = c2.makespan(durs, workers)
    print(f"    S32 total {total / 3600:.2f} h of runs; makespan on {workers} workers {ms / 3600:.2f} h", flush=True)
    return durs


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
    pv = ec.print_banner("batch 10: muon_slow (S32), muon_scale (S33)" + dl)
    print(f"  recipe: S32 lr = test_channel_binding.SUB_LR = {ec.SUB_LR:g} (Adam on the embedding; Muon lr "
          f"{s32.MUON_LR:g}; slow phase x{s25.SLOW_SCALE} for updates 1-{s25.WARM}), MAX_ITERS = test_short_conv."
          f"MAX_ITERS = {ec.MAX_ITERS}, task P={ec.TASK.P} S={ec.TASK.S} grouped; 1 torch thread per run")
    s33.print_constants()
    for m in SCREENS:
        for k, a in m.ARMS.items():
            print(f"  lr read for {m.NAME} {k}: {a['lr']:g} (Adam groups)" + (
                f"; Muon lr {a['muon_lr']:g}" if a.get("muon_lr") else "; Muon lr from the validity runs"))
    print(f"  S32 readings: 'the hinge is unnecessary under Muon' if MUON_SLOW DISCOVERED >= {s32.UNNEEDED_MIN}/40 and "
          f"b - c <= 1; 'the hinge matters under Muon' if b - c >= {s32.MATTERS_D} and p < {s32.MATTERS_P}; otherwise "
          f"neither (b = MUON_HINGE-only, c = MUON_SLOW-only, exact McNemar two-sided)")
    print(f"  S33 readings: (a) 'promising at four streams' if the better Muon arm binds >= {s33.PROMISING_A}/10; (b) "
          f"'breaks the coarse split' if an arm binds >= {s33.BREAK_N} runs or its median number of distinct channels "
          f"at the end is >= {s33.BREAK_MED}; UNTESTED configurations get no reading")
    print(f"  ROUTED* = margin >= {c2.ROUTE_MARGIN} and eta_key_by_stream > {c2.ROUTE_ETA}")
    apply_order()
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
                  f"file outside explore_out/ and explore_batch10.py differs from {rec['git']}, where "
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
        s33.select_lr(args.workers, dry=args.dry)
        for s in SCREENS:
            c2.print_screen_header2(s)
        d32 = s32_projection(args.workers)
        ms33 = s33.projection(args.workers, dry=args.dry)
        print(f"  BATCH PROJECTION: S32 then S33 on {args.workers} workers, worst case about "
              f"{(sum(d32) / args.workers + ms33) / 3600:.1f} h (S32's runs / workers + S33's makespan)", flush=True)
        t0 = time.time()
        c2.run_jobs2(SCREENS, workers=args.workers, pv=pv)
        print(f"  batch wall clock {(time.time() - t0) / 60:.1f} min (this segment)")
    else:
        print("  REPORT ONLY: from the saved runs (no repro check, CHECKs or runs in this call)")
        s33.select_lr(args.workers, dry=args.dry, run=False)
    print("#" * 100)
    print(f"REPORTS — {ec.BANNER}{dl}")
    print("#" * 100)
    tagp = "DRY RUN " if args.dry else ""

    def status(m, v):
        n, N = completeness(m)
        if n == N:
            return v, True
        if args.stopped_early and base(m) == args.stopped_early:
            return f"stopped early, {n}/{N} seeds, no reading", False
        return f"INCOMPLETE, {n}/{N} runs, no reading", False

    d32 = s32.report(ec.load_store(s32.NAME))
    v32, _ = status(s32, d32["rule"])
    sm = d32["slow_mem"]
    print(f"  {tagp}READING {s32.NAME}: {v32} (MUON_HINGE DISCOVERED {d32['new']}/{d32['n']} vs MUON_SLOW "
          f"{d32['old']}/{d32['n']}, {d32['b']} vs {d32['c']}, p = {d32['p']:.3g}; MUON_SLOW vs SLOW_MEM {sm['new']}/"
          f"{sm['n']} vs {sm['old']}/{sm['n']})  [{ec.BANNER}]\n")
    d33 = s33.report(ec.load_store(s33.NAME))
    v33 = {}
    for c in s33.CFG:
        arms = [k for k, a in s33.ARMS.items() if a["cfg"] == c]
        n = sum(completeness(s33, k)[0] for k in arms)
        N = sum(completeness(s33, k)[1] for k in arms)
        if d33["untested"].get(c):
            v33[c] = d33[c]["reading"]
        elif n == N:
            v33[c] = d33[c]["reading"]
        else:
            v33[c] = f"INCOMPLETE, {n}/{N} runs, no reading"
        print(f"  {tagp}READING {s33.NAME} ({c}): {v33[c]}  [{ec.BANNER}]")
    print()
    print("#" * 100)
    print(f"SUMMARY — {ec.BANNER}{dl}")
    print(f"  S32 muon_slow: {v32} (MUON_SLOW {d32['old']}/{d32['n']} vs MUON_HINGE {d32['new']}/{d32['n']} "
          f"DISCOVERED; MUON_SLOW vs SLOW_MEM {sm['new']}/{sm['n']} vs {sm['old']}/{sm['n']})")
    for c in s33.CFG:
        print(f"  S33 muon_scale ({c}, Muon lr {d33['mlr'].get(c)}): {v33[c]}")
    print("#" * 100)


if __name__ == "__main__":
    main()
