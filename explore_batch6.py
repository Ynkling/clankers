#!/usr/bin/env python
"""
explore_batch6.py — EXPLORATORY, not a result. Batch 6 of the outside-ideas screens, with batches
1-5's harness, rules and labels and their resume machinery. Everything below was fixed before any
run.

  S16 explore_kick_control    SLOW_MEM + a random kick of the hinge gradient's size wherever S13's
                              hinge would fire (at most N_s per seed); the 20 seeds where S13's
                              hinge fired                                            (KICK)
  S17 explore_far_slow_hinge  S13's SLOW_HINGE on the header layout, no nudge; seeds 160-169
                                                                                     (FAR_SLOW_HINGE)
  S18 explore_stuck_memory    diagnostic: S14's 8 stalled runs re-run to their end state; (a)
                              breakdown by block order, key position, other stream's value, with
                              the learned and the perfect gate; (b) 24000 perfect-gate updates
                                                                  (hdr_far_nudge, hdr_far_A, blk_A)

BACKGROUND (post hoc, from batch 5): see each screen's docstring. S13 SLOW_HINGE 34/40 vs SLOW_MEM
24/40 (10 vs 0); the hinge never fired on 20 seeds; where it fired SLOW_MEM 4/20 -> SLOW_HINGE 14/20,
mostly from a few firings early. S14: the stalled runs' learned gates match the perfect gate; the
perfect gate does not free them in 4800 updates (far_nudge 161 is a gate failure). S15: SLOW_MEM's
schedule bound the far_nudge seeds 10/10. Batch 3's far_A_slow: 0/10, 8 POSITION.

RECIPE (every arm): lr test_channel_binding.SUB_LR = 1e-3, passed explicitly (the SLOW_MEM-schedule
arms' warm-up group at SUB_LR/10); MAX_ITERS test_short_conv.MAX_ITERS = 24000 (S18: each run to its
recorded end, then 24000 perfect-gate updates); evaluation every 1200; early stop after 3
evaluations >= 0.95; 1 torch thread per run. Tasks: S16 the grouped BindTask (P=4, S=2, n_vals=16,
n_q=1); S17 the header layout; S18 each run's own layout. ROUTED* = margin >= 0.9 and
eta_key_by_stream > 0.9.

RULES
- S16, outcome DISCOVERED on its 20 seeds: "the direction matters" if KICK <= 7/20; "a kick of that
  size suffices" if KICK >= 12/20; otherwise neither. Also printed: failure classes, ROUTED* at
  1200, kicks used per seed (and McNemar against SLOW_HINGE and SLOW_MEM).
- S17 (S8's rule): promising if DISCOVERED >= 4/10; not if <= 1/10; otherwise inconclusive. Also
  printed: failure classes, ROUTED* at 1200 and at the end, stalls, hinge firing counts.
- S18: diagnostic, no verdict; per run "recency" / "primacy" (or neither) and "slow" (with the
  update) / "basin" (UNTESTED if a rerun does not reproduce its recorded curve).
- An incomplete screen gets no verdict or reading.

CHECKS (before any run; any failure stops the batch)
- repro: X's arm A, seed 160, 2400 steps, bit-identical to X's record (in every segment).
- S16: N_s = 0 equals SLOW_MEM bit for bit (3600 steps); with the random tensor replaced by the
  hinge's own gradient the path reproduces S13's SLOW_HINGE on seed 162 (bit for bit, or else the
  largest curve difference through 3600 steps, and the batch continues only if the outcome
  matches); each logged kick's norm equals the hinge gradient's it replaced.
- S17: TAU = 1.0 equals far_A_slow bit for bit (3600 steps); on 20 probe batches of the header
  layout the perfect gate scores below 0.1 on both terms and a gate set by block above 0.5 on
  eta2_half.
- S18: every rerun reproduces its recorded curve (each job); the breakdown's overall accuracy
  equals the evaluation's.

SEGMENTS: batch 4-5's machinery (--order, --resume against explore_out/batch6_checks.json,
--report-only, --stopped-early, --dry); a watcher outside this file pushes every saved run.

Run:  python explore_batch6.py [--workers 4] [--dry] [--order ...] [--resume] [--report-only]
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
import explore_kick_control as s16
import explore_far_slow_hinge as s17
import explore_stuck_memory as s18

SCREENS = (s16, s17, s18)
DRY_ITERS = 3600
CHECKS_RECORD = os.path.join(ec.OUT_DIR, "batch6_checks.json")
DEFAULT_ORDER = "stuck_memory,kick_control,far_slow_hinge"


def git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True,
                          check=True).stdout


def code_changes(rev):
    files = git("diff", "--name-only", rev).split() + \
        git("ls-files", "--others", "--exclude-standard").split()
    return sorted(f for f in set(files)
                  if not f.startswith("explore_out/") and f != "explore_batch6.py")


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
    pv = ec.print_banner("batch 6: kick_control (S16), far_slow_hinge (S17), stuck_memory (S18)" + dl)
    print(f"  recipe: lr = test_channel_binding.SUB_LR = {ec.SUB_LR:g} (passed explicitly), "
          f"MAX_ITERS = test_short_conv.MAX_ITERS = {ec.MAX_ITERS}, task P={ec.TASK.P} S={ec.TASK.S} "
          f"n_vals={ec.TASK.n_vals} n_q={ec.TASK.n_q}, 1 torch thread per run")
    print(f"  S16 rule (DISCOVERED on its 20 seeds): 'the direction matters' if KICK <= "
          f"{s16.DIRECTION_MAX}/20; 'a kick of that size suffices' if KICK >= {s16.SUFFICES_MIN}/20; "
          f"else neither")
    print(f"  S17 rule (S8's): promising if DISCOVERED >= {s17.PROMISING_N}/10; not if <= {s17.NOT_N}/10; "
          f"else inconclusive")
    print(f"  S18: diagnostic; per run 'recency' (learned gate: last-block >= {s18.REC_HI}, first-block "
          f"<= {s18.REC_LO}) / 'primacy' (the reverse) / neither, and 'slow' (perfect-gate continuation "
          f">= {s18.BIND}) / 'basin'")
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
                  f"file outside explore_out/ and explore_batch6.py differs from {rec['git']}, where "
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

    d16 = s16.report(ec.load_store(s16.NAME))
    v16, _ = status(s16, d16["rule"])
    print(f"  {tagp}READING {s16.NAME} (KICK DISCOVERED {d16['discovered']}/{d16['n']}): {v16}  "
          f"[{ec.BANNER}]\n")
    d17 = s17.report(ec.load_store(s17.NAME))
    v17, _ = status(s17, d17["rule"])
    print(f"  {tagp}VERDICT {s17.NAME}: {v17} (DISCOVERED {d17['discovered']}/{d17['n']})  [{ec.BANNER}]\n")
    d18 = s18.report(ec.load_store(s18.NAME))
    n18, N18 = completeness(s18)
    print(f"  {tagp}S18 {s18.NAME}: diagnostic, {n18}/{N18} probes"
          + ("" if n18 == N18 else " (INCOMPLETE)") + f"  [{ec.BANNER}]\n")
    print("#" * 100)
    print(f"SUMMARY — {ec.BANNER}{dl}")
    sh, sm = d16["slow_hinge"], d16["slow_mem"]
    print(f"  S16 kick_control: {v16} (KICK DISCOVERED {d16['discovered']}/{d16['n']}; SLOW_HINGE "
          f"{sh['old']}/{sh['n']}, KICK only {sh['b']} vs SLOW_HINGE only {sh['c']}, p = {sh['p']:.3g}; "
          f"SLOW_MEM {sm['old']}/{sm['n']}, KICK only {sm['b']} vs SLOW_MEM only {sm['c']}, p = {sm['p']:.3g})")
    print(f"  S17 far_slow_hinge: {v17} (DISCOVERED {d17['discovered']}/{d17['n']}; batch 3's far_A_slow 0/10, "
          f"batch 2's far_A 0/10)")
    for lay, items in d18["readings"].items():
        print(f"  S18 {lay:<8} " + ", ".join(f"{a} {s}: {rd}" for a, s, rd in items))
    if d18.get("s14_fixed"):
        print("  S18 extra (S14 corrected): " + ", ".join(f"{a} {s}: {o} -> {n}"
                                                         for a, s, o, n in d18["s14_fixed"]))
    print("#" * 100)


if __name__ == "__main__":
    main()
