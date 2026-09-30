#!/usr/bin/env python
"""
explore_batch2.py — EXPLORATORY, not a result. Batch 2 of the outside-ideas screens: three screens
that follow up batch 1, run in one worker pool, with batch 1's harness, rules and labels.

  S4 explore_kwta_warm  kWTA 16/256 through the evaluation at step 2400, dense after   (KWTA_WARM)
  S5 explore_slow_mem   memory at lr 1e-4 for 2400 updates, the gate at 1e-3            (SLOW_MEM)
  S6 explore_far_cue    a header layout: the stream token once per block                (far_ceil, far_A, far_L3)
  reference: explore_ref_a2400, X's arm A re-run through step 2400                       (A_2400)

Everything below was fixed before any run.

BACKGROUND (post hoc, from batch 1 and X's recorded arm A in test_short_conv; checked against the
records before this file was written)
- ROUTED* = margin >= 0.9 and eta_key_by_stream > 0.9. ROUTED* at step 1200 -> at the end:
  arm A 6 -> 7, LOCAL3 16 -> 16, KWTA16 15 -> 14, AUX1 0 -> 1. The split is settled by 1200.
- KWTA16 left the gate unchanged, yet 15/20 runs were ROUTED* at 1200 (vs arm A: 12 vs 3,
  p = 0.035). It bound only 2/20 because kWTA slows the memory for the whole run: 12 runs routed
  and did not bind; the ceiling bound 2/3, at steps 14400 and 21600.
- LOCAL3's gate sees the stream token at every key, value and query (at t-1 or t-2). This
  follows from how the task is built.
- Two readings of KWTA16's early routing:
  (a) sparse codes: near-orthogonal key codes leave the stream split as the only useful one;
  (b) a race: a memory that learns slowly at first cannot exploit a non-stream split before the
      gate finds the stream split.
  S5 separates them: (a) predicts that slow_mem does not raise routing at 1200; (b) predicts
  that it does.

RECIPE (every arm): P=4, S=2, n_vals=16, n_q=1; MultiBDH n_layer=3, decay, N=256; Adam, BATCH 32;
lr test_channel_binding.SUB_LR = 1e-3, passed explicitly (S5's warm-up group at SUB_LR/10);
MAX_ITERS test_short_conv.MAX_ITERS = 24000; evaluation every 1200; early stop after 3
evaluations >= 0.95; 1 torch thread per run. Seeds and arms: each screen's docstring.

SCREEN RULE (S4 and S5; unchanged from batch 1, explore_batch1.verdict)
  Outcome DISCOVERED, vs X's arm A, b = candidate-only seeds, c = A-only seeds, exact two-sided
  McNemar p: promising if b - c >= 4 and p < 0.10; not if b - c <= 0; otherwise inconclusive.
  Also printed for S4 and S5, paired with X's arm A, with McNemar: ROUTED* at 1200, at 2400 and
  at the end. X's record has routing statistics only at 1200 and the end, so X's arm A at 2400 is
  read from explore_ref_a2400 (arm A re-run here through 2400, used only on seeds whose re-run
  reproduces X's record exactly).
S5 READING (ROUTED* at 1200 vs X's arm A, the rule's bands): (b) if b - c >= 4 and p < 0.10;
  (a) if b - c <= 0; neither otherwise.
S6 (descriptive): per arm bound, ROUTED* at 1200, failure types; far_L3 minus far_A in bound
  runs. far_ceil below 2/3 -> S6 UNTESTED. far_L3 >= far_A + 3: "LOCAL3 survives a distant cue";
  far_L3 <= far_A: "LOCAL3's gain needs the cue in view"; otherwise neither reading applies.

CHECKS (before anything runs; any failure stops the batch)
- repro: X's arm A, seed 160, 2400 steps, bit-identical to X's record.
- S4: bit-identical to batch 1's KWTA16 through the evaluation at step 2400 (curve at 1200 and
  2400, weights after update 2400); after the switch, on the same weights, the forward equals
  arm A's.
- S5: an optimizer step pre-hook over run_job's full 24000-step path records each group's lr:
  gate group 1e-3 at every update; the other 1e-4 for updates 1-2400 and 1e-3 after; the gate
  group holds exactly W_in, W_h, W_g.
- S6: see explore_far_cue (the layout, stream labels, and the perfect gate's zero cross-stream
  scores at every layer).

OUTPUT. The projection is printed before the runs; per-run progress lines carry seed and lr
schedule; per-seed tables (with X's A and A_conv columns for S4 and S5), then the reports and
verdicts. Records: explore_out/<screen>_results.json. --dry writes dry_<screen>_results.json:
1 seed per arm, runs capped at 3600 steps (so both switches at 2400 run), labelled DRY RUN.

SEGMENTS (added after the dry run, before any batch-2 result was read; rules and arms unchanged)
The session's background time limit stops long runs, so the batch runs in segments; every saved
run is reused and only missing runs re-run (runs are deterministic per seed).
  --order a,b,c,d   queue order by screen NAME. run_jobs2 queues runs by arm priority; the driver
                    sets the priorities in memory (len..1 in this order). Screen files unchanged.
  --resume          prints a RESUME line; always runs the repro check. If no file outside
                    explore_out/ and explore_batch2.py differs from CHECKS_SHA (the head whose full
                    CHECK suite is in the first segment's log), the rest of the CHECK suite is
                    skipped and the git diff --stat is printed; otherwise every CHECK runs.
  --report-only     prints the reports from the saved runs (no repro check, CHECKs or runs).
  --stopped-early N a screen left incomplete on purpose is reported "stopped early, n/N seeds, no
                    verdict". Any incomplete screen gets no verdict (and no S5/S6 reading).

Run:  python explore_batch2.py [--workers 4] [--dry] [--order ...] [--resume] [--report-only]
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common2 as c2
import explore_kwta_warm as s4
import explore_slow_mem as s5
import explore_far_cue as s6
import explore_ref_a2400 as ref
from explore_batch1 import verdict, PROMISING_D, PROMISING_P

SCREENS = (s4, s5, s6)
DRY_ITERS = 3600
CHECKS_SHA = "6f529b5"          # the head whose full CHECK suite ran in segment 1


def make_dry():
    c2.DRY.update(on=True, iters=DRY_ITERS)
    for m in SCREENS + (ref,):
        m.NAME = "dry_" + m.NAME
        m.ARMS = {k: dict(a, iters=min(a["iters"], DRY_ITERS), seeds=tuple(a["seeds"][:1]))
                  for k, a in m.ARMS.items()}


def git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True,
                          check=True).stdout


def code_changes():
    """Files outside explore_out/ and explore_batch2.py that differ from CHECKS_SHA (working
    tree, tracked) or are untracked (not ignored)."""
    files = git("diff", "--name-only", CHECKS_SHA).split() + \
        git("ls-files", "--others", "--exclude-standard").split()
    return sorted(f for f in set(files)
                  if not f.startswith("explore_out/") and f != "explore_batch2.py")


def base(m):
    return m.NAME[4:] if m.NAME.startswith("dry_") else m.NAME


def apply_order(order):
    mods = {base(m): m for m in SCREENS + (ref,)}
    names = order.split(",")
    assert sorted(names) == sorted(mods), f"--order must name each of {sorted(mods)}"
    for i, n in enumerate(names):
        for a in mods[n].ARMS.values():
            a["prio"] = len(names) - i
    print(f"  order (--order): {', '.join(names)} — run_jobs2 queues runs by arm priority, set in "
          f"memory to {len(names)}..1 in this order (the screen files are unchanged)")


def completeness(m):
    st = ec.load_store(m.NAME)
    want = [(k, s) for k, a in m.ARMS.items() for s in a["seeds"]]
    have = [1 for k, s in want if (r := st["runs"].get(f"{k}|{s}")) and r.get("ok")]
    return len(have), len(want)


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
    pv = ec.print_banner("batch 2: kwta_warm (S4), slow_mem (S5), far_cue (S6), ref_a2400" + dl)
    print(f"  recipe: lr = test_channel_binding.SUB_LR = {ec.SUB_LR:g} (passed explicitly), "
          f"MAX_ITERS = test_short_conv.MAX_ITERS = {ec.MAX_ITERS}, task P={ec.TASK.P} S={ec.TASK.S} "
          f"n_vals={ec.TASK.n_vals} n_q={ec.TASK.n_q}, 1 torch thread per run")
    print(f"  screen rule (S4, S5): promising if b - c >= {PROMISING_D} and McNemar p < "
          f"{PROMISING_P}; not if b - c <= 0; else inconclusive (DISCOVERED vs X's arm A)")
    print(f"  ROUTED* = margin >= {c2.ROUTE_MARGIN} and eta_key_by_stream > {c2.ROUTE_ETA}")
    print("  S5 reading: (b) a race if ROUTED*@1200 b - c >= 4 and p < 0.10; (a) sparse codes if "
          "b - c <= 0; neither otherwise")
    print("  S6: UNTESTED if far_ceil < 2/3; 'LOCAL3 survives a distant cue' if far_L3 >= far_A + 3; "
          "'LOCAL3's gain needs the cue in view' if far_L3 <= far_A")
    if args.order:
        apply_order(args.order)
    if not args.report_only:
        if not ec.repro_check():
            print("  ! this container does not reproduce X: the recorded pairing is invalid; stopping.")
            sys.exit(1)
        changed = code_changes() if args.resume else None
        if args.resume and not changed:
            print(f"  CHECKs skipped (repro check above still run): the screen code is unchanged — no "
                  f"file outside explore_out/ and explore_batch2.py differs from {CHECKS_SHA}; the "
                  f"first segment's full CHECK output is in this log. git diff --stat {CHECKS_SHA}:")
            for ln in git("diff", "--stat", CHECKS_SHA).rstrip().splitlines():
                print(f"    {ln}")
        else:
            if args.resume:
                print(f"  code differs from {CHECKS_SHA} outside explore_out/ and explore_batch2.py "
                      f"({changed}): running every CHECK")
            t0 = time.time()
            for s in SCREENS:
                if not s.check():
                    print(f"  ! CHECK failed in {s.NAME}; stopping.")
                    sys.exit(1)
            print(f"  all CHECKs passed ({(time.time() - t0) / 60:.1f} min)", flush=True)
        for s in SCREENS + (ref,):
            c2.print_screen_header2(s)
        c2.projection(SCREENS + (ref,), args.workers)
        t0 = time.time()
        c2.run_jobs2(SCREENS + (ref,), workers=args.workers, pv=pv)
        print(f"  batch wall clock {(time.time() - t0) / 60:.1f} min (this segment)")
    else:
        print("  REPORT ONLY: from the saved runs (no repro check, CHECKs or runs in this call)")
    print("#" * 100)
    print(f"REPORTS — {ec.BANNER}{dl}")
    print("#" * 100)
    refs = ref.report(ec.load_store(ref.NAME))
    print()
    tagp = "DRY RUN " if args.dry else ""

    def status(m, v):
        n, N = completeness(m)
        if n == N:
            return v, True
        if args.stopped_early and base(m) == args.stopped_early:
            return f"stopped early, {n}/{N} seeds, no verdict", False
        return f"INCOMPLETE, {n}/{N} runs, no verdict", False

    d4 = s4.report(ec.load_store(s4.NAME), refs)
    v4, _ = status(s4, verdict(d4))
    print(f"  {tagp}VERDICT {s4.NAME}: {v4}  [{ec.BANNER}]\n")
    d5 = s5.report(ec.load_store(s5.NAME), refs)
    v5, full5 = status(s5, verdict(d5))
    if not full5:
        d5["reading"] = "none (S5 incomplete; the READING line above is not applied)"
    print(f"  {tagp}VERDICT {s5.NAME}: {v5}  [{ec.BANNER}]\n")
    d6 = s6.report(ec.load_store(s6.NAME))
    v6, full6 = status(s6, d6["reading"])
    print(f"  {tagp}S6 {s6.NAME}: {v6}  (descriptive)  [{ec.BANNER}]\n")
    print("#" * 100)
    print(f"SUMMARY — {ec.BANNER}{dl}")
    print(f"  S4 kwta_warm: {v4} (DISCOVERED {d4['new']}/{d4['n']} vs X A {d4['old']}/{d4['n']}, "
          f"{d4['b']} vs {d4['c']}, p = {d4['p']:.3g})")
    print(f"  S5 slow_mem:  {v5} (DISCOVERED {d5['new']}/{d5['n']} vs X A {d5['old']}/{d5['n']}, "
          f"{d5['b']} vs {d5['c']}, p = {d5['p']:.3g}); reading: {d5['reading']}")
    print(f"  S6 far_cue:   {v6}")
    print("#" * 100)


if __name__ == "__main__":
    main()
