#!/usr/bin/env python
"""
explore_batch5.py — EXPLORATORY, not a result. Batch 5 of the outside-ideas screens, with batches
1-4's harness, rules and labels and batches 2-4's resume machinery. Everything below was fixed
before any run.

  S13 explore_slow_hinge      SLOW_MEM + 1.0 * [relu(eta2_index - 0.2) + relu(eta2_half - 0.2)] on
                              the read gate at key positions; seeds 160-199         (SLOW_HINGE)
  S14 explore_stall_probe     diagnostic: stalled routed runs re-run to their recorded end, then
                              SWAP / CONTINUE / GATE PROFILE; header far_nudge 161, 164, 165 (160 a
                              control), far_A 165, 166; blocked X's A_blocked 120, 130, 136
                                                                  (hdr_far_nudge, hdr_far_A, blk_A)
  S15 explore_far_nudge_slow  batch 4's far_nudge with the SLOW_MEM schedule; seeds 160-169
                                                                                     (FAR_NUDGE_SLOW)

BACKGROUND (post hoc, from batch 4): see each screen's docstring (S13: S12's penalty at its
sampling floor, uncommitted gates, the key-split attractor after resets; S14 and S15: routed-but-
unbound runs on the header and blocked layouts; the gates can express the header routing).

RECIPE (every arm): lr test_channel_binding.SUB_LR = 1e-3, passed explicitly (the SLOW_MEM-schedule
arms' warm-up group at SUB_LR/10); MAX_ITERS test_short_conv.MAX_ITERS = 24000 (S14: each run to its
recorded end, then 4800 continued updates); evaluation every 1200; early stop after 3 evaluations
>= 0.95; 1 torch thread per run. Tasks: S13 the grouped BindTask (P=4, S=2, n_vals=16, n_q=1); S15
the header layout; S14 each run's own layout. ROUTED* = margin >= 0.9 and eta_key_by_stream > 0.9.

RULES
- S13 vs SLOW_MEM over the 40 seeds, outcome DISCOVERED: promising if b - c >= 4 and McNemar p < 0.10;
  not if b - c <= 0; otherwise inconclusive (explore_batch1.verdict). Also printed vs X's arm A (40)
  and vs SLOW_POS (160-179) with ROUTED* at 1200/2400/end. Reading on 160-179: "noise floor" if
  ROUTED*@1200 >= 8/20 and at most 2 failures are uncommitted gates; "not the noise floor" if
  ROUTED*@1200 <= 3/20; otherwise neither.
- S14: diagnostic, no verdict; per-run readings SWAP-FIXES / MEMORY-STUCK / SLOW-ONLY / NEITHER
  (UNTESTED if a rerun does not reproduce its recorded curve).
- S15 vs far_nudge, outcome DISCOVERED: promising if >= 8/10; not if <= 5/10; otherwise inconclusive.
- An incomplete screen gets no verdict or reading.

CHECKS (before any run; any failure stops the batch)
- repro: X's arm A, seed 160, 2400 steps, bit-identical to X's record (in every segment).
- S13: TAU = 1.0 equals SLOW_MEM and TAU = 0 equals SLOW_POS bit for bit (3600 steps each); the
  hinge's gradient reaches only the gate and the embedding and is exactly zero below TAU.
- S14: the perfect gate zeroes the cross-stream scores at every layer on both layouts; the
  continuation code continues a run exactly; both continuations see identical batches; SWAP's
  learned-gate path reproduces the recorded accuracy (and, in every job: the rerun reproduces its
  recorded curve at every evaluation).
- S15: the weights after the nudge pre-fit equal far_nudge's on every seed; the two groups' lrs by an
  optimizer step pre-hook, as in S5.

SEGMENTS: batch 4's machinery (--order, --resume against explore_out/batch5_checks.json,
--report-only, --stopped-early, --dry); a watcher outside this file pushes every saved run.

Run:  python explore_batch5.py [--workers 4] [--dry] [--order ...] [--resume] [--report-only]
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
import explore_slow_hinge as s13
import explore_stall_probe as s14
import explore_far_nudge_slow as s15
from explore_batch1 import verdict, PROMISING_D, PROMISING_P

SCREENS = (s13, s14, s15)
DRY_ITERS = 3600
CHECKS_RECORD = os.path.join(ec.OUT_DIR, "batch5_checks.json")
DEFAULT_ORDER = "stall_probe,slow_hinge,far_nudge_slow"


def git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True,
                          check=True).stdout


def code_changes(rev):
    files = git("diff", "--name-only", rev).split() + \
        git("ls-files", "--others", "--exclude-standard").split()
    return sorted(f for f in set(files)
                  if not f.startswith("explore_out/") and f != "explore_batch5.py")


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
    pv = ec.print_banner("batch 5: slow_hinge (S13), stall_probe (S14), far_nudge_slow (S15)" + dl)
    print(f"  recipe: lr = test_channel_binding.SUB_LR = {ec.SUB_LR:g} (passed explicitly), "
          f"MAX_ITERS = test_short_conv.MAX_ITERS = {ec.MAX_ITERS}, task P={ec.TASK.P} S={ec.TASK.S} "
          f"n_vals={ec.TASK.n_vals} n_q={ec.TASK.n_q}, 1 torch thread per run")
    print(f"  S13 rule (vs SLOW_MEM, 40 seeds): promising if b - c >= {PROMISING_D} and McNemar p < "
          f"{PROMISING_P}; not if b - c <= 0; else inconclusive (DISCOVERED)")
    print("  S13 reading (160-179): 'noise floor' if ROUTED*@1200 >= 8/20 and <= 2 uncommitted failures; "
          "'not the noise floor' if ROUTED*@1200 <= 3/20; else neither")
    print("  S14: diagnostic; SWAP-FIXES / MEMORY-STUCK / SLOW-ONLY / NEITHER per run")
    print(f"  S15 rule (vs far_nudge): promising if DISCOVERED >= {s15.PROMISING_N}/10; not if <= "
          f"{s15.NOT_N}/10; else inconclusive")
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
                  f"file outside explore_out/ and explore_batch5.py differs from {rec['git']}, where "
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

    d13 = s13.report(ec.load_store(s13.NAME))
    v13, full13 = status(s13, verdict(d13))
    rd13 = d13["reading"] if full13 else "none (S13 incomplete)"
    print(f"  {tagp}VERDICT {s13.NAME} (vs SLOW_MEM, 40 seeds): {v13}; reading: {rd13}  [{ec.BANNER}]\n")
    d14 = s14.report(ec.load_store(s14.NAME))
    n14, N14 = completeness(s14)
    print(f"  {tagp}S14 {s14.NAME}: diagnostic, {n14}/{N14} probes"
          + ("" if n14 == N14 else " (INCOMPLETE)") + f"  [{ec.BANNER}]\n")
    d15 = s15.report(ec.load_store(s15.NAME))
    v15, _ = status(s15, d15["rule"])
    print(f"  {tagp}VERDICT {s15.NAME}: {v15} (DISCOVERED {d15['discovered']}/{d15['n']})  [{ec.BANNER}]\n")
    print("#" * 100)
    print(f"SUMMARY — {ec.BANNER}{dl}")
    xa, sp = d13["x_a"], d13["slow_pos"]
    print(f"  S13 slow_hinge: {v13} vs SLOW_MEM (DISCOVERED {d13['new']}/{d13['n']} vs {d13['old']}/{d13['n']}, "
          f"{d13['b']} vs {d13['c']}, p = {d13['p']:.3g}); reading: {rd13}")
    print(f"      vs X A {xa['new']}/{xa['n']} vs {xa['old']}/{xa['n']} ({xa['b']} vs {xa['c']}, p = {xa['p']:.3g}); "
          f"vs SLOW_POS (160-179) {sp['new']}/{sp['n']} vs {sp['old']}/{sp['n']} ({sp['b']} vs {sp['c']}, p = {sp['p']:.3g})")
    for lay, items in d14["readings"].items():
        print(f"  S14 {lay:<8} " + ", ".join(f"{a} {s}: {rd}" for a, s, rd in items))
    print(f"  S15 far_nudge_slow: {v15} (DISCOVERED {d15['discovered']}/{d15['n']}; batch 4's far_nudge 5/10)")
    print("#" * 100)


if __name__ == "__main__":
    main()
