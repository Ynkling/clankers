#!/usr/bin/env python
"""
explore_batch11.py — EXPLORATORY, not a result. Batch 11 of the outside-ideas screens, with batches
1-10's harness, rules and labels and their resume machinery. Everything below was fixed before any run.
S31 is dropped (S33(a) answered it: merges held under Muon from 9600 to the end in 12 of 13 runs); S29
and S30 are deferred.

  S34 explore_hinge_window  the hinge only on updates 1-2400 (weight 0 after): MUON_H2400 (S25's
                            MUON_HINGE) and ADAM_H2400 (S13's SLOW_HINGE), seeds 160-199; screens'
                            configuration (S=2, P=4, k=2, grouped, no conv, 24000 steps); firing
                            diagnostics (hinge vs task gradient on the gate; Muon momentum cosine)
  S35 explore_muon_k16      S33's MUON_HINGE at S=4, P=4, k=16, conv (A4k16's path on main at 9c5939e,
                            child process), seeds 240-259 vs X's HINGE4k16 and A4k16; Muon lr from the
                            perfect gate (0.005, then 0.0025, then 0.01); 28800 steps

BACKGROUND (docstring): see explore_hinge_window and explore_muon_k16.

RECIPE: the Adam groups at lr test_channel_binding.SUB_LR = 1e-3 (S34; S35 reads test_slow_start.LR and
test_stream_recipe.LR in the child, both 1e-3), passed explicitly; the Muon groups at the Muon lr; the slow
phase's non-gate groups at a tenth for updates 1-2400. MAX_ITERS test_short_conv.MAX_ITERS = 24000 (S34),
test_stream_recipe.MAX_ITERS = 28800 (S35); evaluation every 1200; the run paths' own early stops; 1 torch
thread per run.

READINGS: in explore_hinge_window (S34, per optimizer) and explore_muon_k16 (S35). An incomplete screen
gets no reading.

CHECKS (before any run, after the repro check; any failure stops the batch)
- repro: X's arm A, seed 160, 2400 steps, bit-identical to X's record (in every segment).
- S34: window end 24000 = S25's MUON_HINGE|160 and S13's SLOW_HINGE|160 bit for bit through 2400; window
  end 0 = MUON_SLOW|160 and SLOW_MEM|160 through 1200; the weight is 1.0 at update 2400 and 0 at 2401;
  one diagnostic row per firing.
- S35: X's HINGE4k16|240 under main's Adam path through 1200; groups cover each parameter once; each
  group's lr at updates 1, 2400, 2401; the firing log; the perfect gate's cross-stream scores; the served
  modules.

SEQUENCE: repro check; CHECKs; S35's validity (stored; a resume reuses it); the projection with pool-load
timing (batch 10's method: one timing process per arm at once, the median interval between optimizer
steps; S34's two arms and S35's arm twice, so 4 processes); RUNTIME RULE: if the batch's projected
worst-case makespan is over 9 h, S35 is cut to 240-249 (S34 never cut; stored); then one pool, S35's runs
queued first (the longest runs first), then S34's.
SEGMENTS: batches 4-10's machinery (--resume against explore_out/batch11_checks.json, --report-only,
--stopped-early, --dry); a watcher outside this file pushes every saved run.

Run:  python explore_batch11.py [--workers 4] [--dry] [--resume] [--report-only]
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
import explore_hinge_window as s34
import explore_muon_k16 as s35
from test_binding_recipe import makespan
from test_binding_onset import EVAL_EVERY

SCREENS = (s34, s35)
QUEUE = (s35, s34)                               # longest runs first
CHECK_ORDER = (s34, s35)
DRY_ITERS = 3600
CHECKS_RECORD = os.path.join(ec.OUT_DIR, "batch11_checks.json")


def git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True, check=True).stdout


def code_changes(rev):
    files = git("diff", "--name-only", rev).split() + git("ls-files", "--others", "--exclude-standard").split()
    return sorted(f for f in set(files) if not f.startswith("explore_out/") and f != "explore_batch11.py")


def base(m):
    return m.NAME[4:] if m.NAME.startswith("dry_") else m.NAME


def make_dry():
    c2.DRY.update(on=True, iters=DRY_ITERS)
    for m in SCREENS:
        m.NAME = "dry_" + m.NAME
        m.ARMS = {k: dict(a, iters=min(a["iters"], DRY_ITERS), seeds=tuple(a["seeds"][:1])) for k, a in m.ARMS.items()}


def apply_order():
    n = len(QUEUE)
    for i, m in enumerate(QUEUE):
        for a in m.ARMS.values():
            a["prio"] = 10 * (n - i) + a.get("prio", 1)
    print(f"  queue order: {', '.join(base(m) for m in QUEUE)} — run_jobs2 queues runs by arm priority (S35's 28800-step "
          f"runs first, then S34's)")


def completeness(m, arm=None):
    st = ec.load_store(m.NAME)
    want = [(k, s) for k, a in m.ARMS.items() for s in a["seeds"] if arm in (None, k)]
    have = [1 for k, s in want if (r := st["runs"].get(f"{k}|{s}")) and r.get("ok")]
    return len(have), len(want)


def run_checks():
    t0 = time.time()
    for s in CHECK_ORDER:
        if not s.check():
            print(f"  ! CHECK failed in {s.NAME}; stopping.")
            sys.exit(1)
    print(f"  all CHECKs passed ({(time.time() - t0) / 60:.1f} min)", flush=True)


def _time_job(job):
    import torch
    torch.set_num_threads(ec.THREADS)
    if job[0] == "s34":
        return job, s34.time_step(job[1])
    return job, s35.timing(job[1])


def projection(workers, dry):
    """Pool-load timing: S34's two arms and S35's arm (twice) timed at once, 4 processes."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    jobs = [("s34", "MUON_H2400"), ("s34", "ADAM_H2400"), ("s35", s35.STATE["mlr"]), ("s35", s35.STATE["mlr"])]
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=mp.get_context("fork"),
                             initializer=ec._worker_init) as ex:
        res = list(ex.map(_time_job, jobs))
    t34 = {j[1]: v for j, v in res if j[0] == "s34"}
    t35 = [v for j, v in res if j[0] == "s35"]
    ts35 = sorted(v[0] for v in t35)[-1]                    # the slower of the two
    te35 = sorted(v[1] for v in t35)[-1]

    def durs():
        out, rows = [], []
        for k, a in s34.ARMS.items():
            ts, te = t34[k]
            per = a["iters"] * ts + (a["iters"] // EVAL_EVERY) * te
            out += [per] * len(a["seeds"])
            rows.append((s34.NAME, k, len(a["seeds"]), per, a["iters"], ts, te))
        for k, a in s35.ARMS.items():
            per = a["iters"] * ts35 + (a["iters"] // EVAL_EVERY) * te35
            out += [per] * len(a["seeds"])
            rows.append((s35.NAME, k, len(a["seeds"]), per, a["iters"], ts35, te35))
        return out, rows

    def show(tag):
        d, rows = durs()
        print(f"  PROJECTION{tag} (worst case: every run to its iters; pool-load timing: the median interval between "
              f"optimizer steps, one process per arm at once (S34 x2, S35 x2), 1 thread each, this machine):")
        for nm, k, n, per, it, ts, te in rows:
            print(f"    {nm:<14} {k:<12} {n:>2} runs x {per / 60:5.1f} min  ({it} steps x {ts * 1e3:.1f} ms + "
                  f"{it // EVAL_EVERY} evals x {te:.1f} s)")
        ms = makespan(d, workers)
        print(f"    total {sum(d) / 3600:.2f} h of runs; batch makespan on {workers} workers {ms / 3600:.2f} h (worst case)",
              flush=True)
        return ms
    ms = show("")
    cut = s35.decide_cut(ms / 3600, workers, dry=dry)
    if cut["cut"]:
        show(" AFTER THE CUT")
    return ms


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
    pv = ec.print_banner("batch 11: hinge_window (S34), muon_k16 (S35)" + dl)
    print(f"  recipe: S34 lr = test_channel_binding.SUB_LR = {ec.SUB_LR:g} (MUON_H2400: Adam on the embedding, Muon lr "
          f"{s34.MUON_LR:g}, slow phase x0.1 for updates 1-2400; ADAM_H2400: gate {ec.SUB_LR:g} throughout, the rest "
          f"{ec.SUB_LR / 10:g} for updates 1-2400), MAX_ITERS = test_short_conv.MAX_ITERS = {ec.MAX_ITERS}, task P="
          f"{ec.TASK.P} S={ec.TASK.S} grouped; hinge LAMBDA {s34.s13.LAMBDA}, TAU {s34.s13.TAU}, window updates 1-"
          f"{s34.WINDOW}; 1 torch thread per run")
    s35.print_constants()
    for m in SCREENS:
        for k, a in m.ARMS.items():
            print(f"  lr read for {m.NAME} {k}: {a['lr']:g} (Adam groups)" + (
                f"; Muon lr {a['muon_lr']:g}" if a.get("muon_lr") else
                ("; Muon lr from the validity runs" if m is s35 else "")))
    print(f"  S34 readings (per optimizer; DISCOVERED; b = full hinge only, c = window only): 'an early window suffices' "
          f"if b - c <= {s34.SUFFICES_D}; 'the late hinge matters' if b - c >= {s34.MATTERS_D} and p < {s34.MATTERS_P}; "
          f"otherwise neither. Description: 'directional kick' if the median ratio >= {s34.KICK_RATIO:g} and the "
          f"Muon momentum's cosine stays >= {s34.COS_THR} for a median of >= {s34.KICK_SPAN} updates")
    print(f"  S35 readings (BOUND; b = HINGE4k16 only, c = MUON_HINGE16 only): 'the Muon recipe carries to four streams "
          f"at k=16' if MUON_HINGE16 >= {s35.CARRIES_N}/20 (>= {s35.CARRIES_N_CUT}/10 if cut) and b - c <= "
          f"{s35.CARRIES_D}; 'worse under Muon' if b - c >= {s35.WORSE_D} and p < {s35.WORSE_P}; otherwise neither; "
          f"UNTESTED if no Muon lr binds the perfect gate on both seeds")
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
                  f"file outside explore_out/ and explore_batch11.py differs from {rec['git']}, where "
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
        s35.select_lr(args.workers, dry=args.dry)
        for s in SCREENS:
            c2.print_screen_header2(s)
        projection(args.workers, args.dry)
        t0 = time.time()
        c2.run_jobs2(SCREENS, workers=args.workers, pv=pv)
        print(f"  batch wall clock {(time.time() - t0) / 60:.1f} min (this segment)")
    else:
        print("  REPORT ONLY: from the saved runs (no repro check, CHECKs or runs in this call)")
        s35.select_lr(args.workers, dry=args.dry, run=False)
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

    d34 = s34.report(ec.load_store(s34.NAME))
    v34 = {}
    for opt, arm in (("MUON", "MUON_H2400"), ("ADAM", "ADAM_H2400")):
        n, N = completeness(s34, arm)
        v34[opt] = d34[opt]["rule"] if n == N else f"INCOMPLETE, {n}/{N} runs, no reading"
        dd = d34[opt]
        print(f"  {tagp}READING {s34.NAME} ({opt}): {v34[opt]} (full only {dd['b']}, window only {dd['c']}, p = "
              f"{dd['p']:.3g}; window {dd['old']}/{dd['n']} vs full {dd['new']}/{dd['n']} DISCOVERED; window vs no hinge "
              f"{dd['vs_base']['new']}/{dd['vs_base']['n']} vs {dd['vs_base']['old']}/{dd['vs_base']['n']})  [{ec.BANNER}]")
    full34 = all(completeness(s34, a)[0] == completeness(s34, a)[1] for a in s34.ARMS)
    k34 = d34["kick"] if full34 else "none (S34 incomplete)"
    print(f"  {tagp}DESCRIPTION {s34.NAME}: {k34}  [{ec.BANNER}]\n")
    d35 = s35.report(ec.load_store(s35.NAME))
    v35 = d35["rule"] if d35["untested"] else status(s35, d35["rule"])[0]
    print(f"  {tagp}READING {s35.NAME}: {v35} (Muon lr {d35['mlr']}; MUON_HINGE16 BOUND {d35['new']}/{d35['n']} vs X "
          f"HINGE4k16 {d35['old']}/{d35['n']}, {d35['b']} vs {d35['c']}, p = {d35['p']:.3g}; vs X A4k16 "
          f"{d35['vs_a']['old']}/{d35['vs_a']['n']})  [{ec.BANNER}]\n")
    print("#" * 100)
    print(f"SUMMARY — {ec.BANNER}{dl}")
    for opt in ("MUON", "ADAM"):
        dd = d34[opt]
        print(f"  S34 hinge_window ({opt}): {v34[opt]} (window {dd['old']}/{dd['n']} vs full hinge {dd['new']}/{dd['n']} "
              f"DISCOVERED, {dd['b']} vs {dd['c']}, p = {dd['p']:.3g}; median hinge/task gradient ratio at firings "
              f"{dd['ratio_median']})")
    print(f"  S34 description: {k34} (median updates until the Muon momentum's cosine is below {s34.COS_THR}: "
          f"{d34['cos_below_median']}, {d34['censored']} of {d34['n_tracks']} censored)")
    print(f"  S35 muon_k16 (Muon lr {d35['mlr']}{', cut to 240-249' if d35['cut'] else ''}): {v35} (MUON_HINGE16 "
          f"{d35['new']}/{d35['n']} vs X HINGE4k16 {d35['old']}/{d35['n']} BOUND)")
    print("#" * 100)


if __name__ == "__main__":
    main()
