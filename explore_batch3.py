#!/usr/bin/env python
"""
explore_batch3.py — EXPLORATORY, not a result. Batch 3 of the outside-ideas screens, with batch 2's
harness, rules, labels and resume machinery. Everything below was fixed before any run.

  S7 explore_slow_mem_ext  S5's SLOW_MEM, unchanged, on seeds 180-199                  (SLOW_MEM)
  S8 explore_far_gates     header layout: multi-scale EMA gate, selective gate, arm A +
                           S5's schedule; seeds 160-169                                 (far_EMA, far_SEL, far_A_slow)
  S9 explore_near_check    the two new gates on the grouped task; seeds 160-169         (EMA_near, SEL_near)
  The gates: explore_gates.

BACKGROUND (batch 2; post hoc)
- S5 slow_mem: DISCOVERED 12/20 vs X's arm A 7/20 (6 vs 1, p = 0.125), inconclusive.
  ROUTED* at 1200: 12 vs 6 (7 vs 1, p = 0.070); all 12 routed runs bound. Failures: 7 POSITION,
  1 OTHER, no KEY (arm A: KEY 4).
- S4 kwta_warm: 7/20 vs 7/20. 15 routed at 2400; 7 of them never bound (0.29-0.54).
- S6 far_cue (header layout): far_ceil 3/3 at 1200; far_A bound 0/10 (2 routed, not bound);
  far_L3 bound 2/10, both by key split, 0 routed. With the stream token out of view, neither
  gate found the stream routing.

RECIPE (every arm): P=4, S=2, n_vals=16, n_q=1; MultiBDH n_layer=3, decay, N=256; Adam, BATCH 32;
lr test_channel_binding.SUB_LR = 1e-3, passed explicitly (the slow-memory arms' warm-up group at
SUB_LR/10); MAX_ITERS test_short_conv.MAX_ITERS = 24000; evaluation every 1200; early stop after 3
evaluations >= 0.95; 1 torch thread per run. ROUTED* = margin >= 0.9 and eta_key_by_stream > 0.9.

RULES
- S7 (the 20 new seeds): batch 1's, unchanged (explore_batch1.verdict): DISCOVERED vs X's arm A,
  promising if b - c >= 4 and McNemar p < 0.10; not if b - c <= 0; otherwise inconclusive.
  Also printed: the 40-seed pool 160-199 (DISCOVERED, ROUTED* at 1200, failure classes).
- S8, per arm: promising if DISCOVERED >= 4/10 (batch 2's far_A: 0/10); not if <= 1/10; otherwise
  inconclusive.
- S9, descriptive, per arm: "keeps the near case" if DISCOVERED >= X's arm A's (4/10) - 1.
- An incomplete screen gets no verdict or reading ("INCOMPLETE", or "stopped early, n/N seeds, no
  verdict" when named in --stopped-early).

CHECKS (before any run; any failure stops the batch)
- repro: X's arm A, seed 160, 2400 steps, bit-identical to X's record.
- S7: explore_slow_mem.check() (S7 runs that path unchanged).
- S8: explore_gates.check_gate on the header task for both gates (closed form = explicit per-step
  loop, also on inflated weights; causal in train and eval; shared parameters bitwise arm A's;
  EMA's initial half-lives the log-spaced set); far_A_slow's lr schedule by a step pre-hook over
  the full 24000-step run path; explore_far_cue.check() (the header layout).
- S9: explore_gates.check_gate on the grouped task for both gates.

SEGMENTS (batch 2's machinery)
  --order a,b,c     queue order by screen NAME (in-memory arm priorities; screen files unchanged).
  --resume          prints a RESUME line; always runs the repro check. The first full CHECK pass
                    records its git head in explore_out/batch3_checks.json; a resume skips the rest
                    of the CHECK suite only if that record shows no code change at the time and no
                    file outside explore_out/ and explore_batch3.py differs from that head now
                    (printing the git diff --stat); otherwise every CHECK runs.
  --report-only     the reports from the saved runs; --stopped-early NAME labels an incomplete screen.
  --dry             1 seed per arm, runs capped at 3600 steps, dry_<screen> stores, labelled DRY RUN.
The runs are pushed as they are saved by a watcher outside this file (see batch3.log's notes).

OUTPUT: the projection first; per-run progress lines (seed, lr schedule); per-seed tables; reports;
a SUMMARY. Records: explore_out/<screen>_results.json.

Run:  python explore_batch3.py [--workers 4] [--dry] [--order ...] [--resume] [--report-only]
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
import explore_slow_mem_ext as s7
import explore_far_gates as s8
import explore_near_check as s9
from explore_batch1 import verdict, PROMISING_D, PROMISING_P

SCREENS = (s7, s8, s9)
DRY_ITERS = 3600
CHECKS_RECORD = os.path.join(ec.OUT_DIR, "batch3_checks.json")
DEFAULT_ORDER = "slow_mem_ext,far_gates,near_check"


def git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True,
                          check=True).stdout


def code_changes(rev):
    files = git("diff", "--name-only", rev).split() + \
        git("ls-files", "--others", "--exclude-standard").split()
    return sorted(f for f in set(files)
                  if not f.startswith("explore_out/") and f != "explore_batch3.py")


def base(m):
    return m.NAME[4:] if m.NAME.startswith("dry_") else m.NAME


def make_dry():
    c2.DRY.update(on=True, iters=DRY_ITERS)
    for m in SCREENS:
        m.NAME = "dry_" + m.NAME
        m.ARMS = {k: dict(a, iters=min(a["iters"], DRY_ITERS), seeds=tuple(a["seeds"][:1]))
                  for k, a in m.ARMS.items()}


def apply_order(order):
    mods = {base(m): m for m in SCREENS}
    names = order.split(",")
    assert sorted(names) == sorted(mods), f"--order must name each of {sorted(mods)}"
    for i, n in enumerate(names):
        for a in mods[n].ARMS.values():
            a["prio"] = len(names) - i
    print(f"  order (--order): {', '.join(names)} — run_jobs2 queues runs by arm priority, set in "
          f"memory to {len(names)}..1 in this order (the screen files are unchanged)")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=ec.WORKERS)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--order", default=None)
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
    pv = ec.print_banner("batch 3: slow_mem_ext (S7), far_gates (S8), near_check (S9)" + dl)
    print(f"  recipe: lr = test_channel_binding.SUB_LR = {ec.SUB_LR:g} (passed explicitly), "
          f"MAX_ITERS = test_short_conv.MAX_ITERS = {ec.MAX_ITERS}, task P={ec.TASK.P} S={ec.TASK.S} "
          f"n_vals={ec.TASK.n_vals} n_q={ec.TASK.n_q}, 1 torch thread per run")
    print(f"  S7 rule (20 new seeds): promising if b - c >= {PROMISING_D} and McNemar p < "
          f"{PROMISING_P}; not if b - c <= 0; else inconclusive (DISCOVERED vs X's arm A)")
    print(f"  S8 rule per arm: promising if DISCOVERED >= {s8.PROMISING_N}/10; not if <= "
          f"{s8.NOT_N}/10; else inconclusive (batch 2's far_A: 0/10)")
    print("  S9 per arm: 'keeps the near case' if DISCOVERED >= X's arm A (4/10) - 1")
    print(f"  ROUTED* = margin >= {c2.ROUTE_MARGIN} and eta_key_by_stream > {c2.ROUTE_ETA}")
    if args.order:
        apply_order(args.order)
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
                  f"file outside explore_out/ and explore_batch3.py differs from {rec['git']}, where "
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
        for s in SCREENS:
            c2.print_screen_header2(s)
        c2.projection(SCREENS, args.workers)
        t0 = time.time()
        c2.run_jobs2(SCREENS, workers=args.workers, pv=pv)
        print(f"  batch wall clock {(time.time() - t0) / 60:.1f} min (this segment)")
    else:
        print("  REPORT ONLY: from the saved runs (no repro check, CHECKs or runs in this call)")
    print("#" * 100)
    print(f"REPORTS — {ec.BANNER}{dl}")
    print("#" * 100)
    tagp = "DRY RUN " if args.dry else ""

    def status(m, v, arm=None):
        n, N = completeness(m, arm)
        if n == N:
            return v, True
        if args.stopped_early and base(m) == args.stopped_early:
            return f"stopped early, {n}/{N} seeds, no verdict", False
        return f"INCOMPLETE, {n}/{N} runs, no verdict", False

    d7 = s7.report(ec.load_store(s7.NAME))
    v7, _ = status(s7, verdict(d7))
    print(f"  {tagp}VERDICT {s7.NAME} (20 new seeds): {v7}  [{ec.BANNER}]\n")
    d8 = s8.report(ec.load_store(s8.NAME))
    v8 = {}
    for k, d in d8.items():
        v8[k], _ = status(s8, d["rule"], k)
        print(f"  {tagp}VERDICT {s8.NAME} {k}: {v8[k]} (DISCOVERED {d['discovered']}/{d['n']})  "
              f"[{ec.BANNER}]")
    print()
    d9 = s9.report(ec.load_store(s9.NAME))
    v9 = {}
    for k, d in d9.items():
        v9[k], _ = status(s9, d["reading"], k)
        print(f"  {tagp}S9 {s9.NAME} {k}: {v9[k]} (DISCOVERED {d['new']}/{d['n']})  (descriptive)  "
              f"[{ec.BANNER}]")
    print()
    print("#" * 100)
    print(f"SUMMARY — {ec.BANNER}{dl}")
    pd, pr = d7["pool"]["discovered"], d7["pool"]["routed1200"]
    print(f"  S7 slow_mem_ext: {v7} (DISCOVERED {d7['new']}/{d7['n']} vs X A {d7['old']}/{d7['n']}, "
          f"{d7['b']} vs {d7['c']}, p = {d7['p']:.3g})")
    print(f"     pool 160-199: DISCOVERED {pd['new']}/{pd['n']} vs X A {pd['old']}/{pd['n']} "
          f"({pd['b']} vs {pd['c']}, p = {pd['p']:.3g}); ROUTED*@1200 {pr['b']} vs {pr['c']} "
          f"(p = {pr['p']:.3g})")
    for k, d in d8.items():
        print(f"  S8 {k:<10} {v8[k]} (DISCOVERED {d['discovered']}/{d['n']}; batch 2's far_A 0/10)")
    for k, d in d9.items():
        print(f"  S9 {k:<10} {v9[k]} (DISCOVERED {d['new']}/{d['n']} vs X A {d['old']}/{d['n']})")
    print("#" * 100)


if __name__ == "__main__":
    main()
