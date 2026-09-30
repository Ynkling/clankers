#!/usr/bin/env python
"""
explore_batch4.py — EXPLORATORY, not a result. Batch 4 of the outside-ideas screens, with batches
2-3's harness, rules, labels and resume machinery. Everything below was fixed before any run.

  S10 explore_far_express  (a) can each gate fit the header-layout routing (CHECK 45's recipe;
                           no training runs); (b) far_nudge: arm A + the 5% labelled nudge on
                           the header layout, seeds 160-169                          (far_nudge)
  S11 explore_gate_reset   arm A + a label-free detector of position/key gate splits at 1200
                           that re-draws only the gate (max 3 resets), seeds 160-179  (RESET)
  S12 explore_slow_pos     SLOW_MEM + 1.0 * eta^2(index + half) penalty on the read gate at
                           key positions, seeds 160-179                               (SLOW_POS)

RECIPE (every arm): lr test_channel_binding.SUB_LR = 1e-3, passed explicitly (SLOW_POS's warm-up
group at SUB_LR/10, as SLOW_MEM); MAX_ITERS test_short_conv.MAX_ITERS = 24000; evaluation every
1200; early stop after 3 evaluations >= 0.95; 1 torch thread per run. ROUTED* = margin >= 0.9 and
eta_key_by_stream > 0.9. Tasks: S11, S12 the grouped BindTask (P=4, S=2, n_vals=16, n_q=1); S10 the
header layout (explore_far_cue.HeaderTask).

RULES
- S10 readings: a gate fits (argmax >= 0.99) and far_nudge DISCOVERED >= 7/10 -> "the distant cue
  is a discovery problem"; a gate fits but far_nudge <= 3/10 -> "the loss cannot finish the
  distant-cue routing even from a head start"; no gate fits -> "the gates cannot express the
  distant-cue routing"; otherwise neither reading applies.
- S11 and S12: batch 1's rule, unchanged (explore_batch1.verdict): DISCOVERED vs X's arm A,
  promising if b - c >= 4 and McNemar p < 0.10; not if b - c <= 0; otherwise inconclusive.
  S11 also: resets used, outcome by number of resets. S12 also: vs batch 2's SLOW_MEM (McNemar)
  and failure classes.
- An incomplete screen gets no verdict or reading.

CHECKS (before any run; any failure stops the batch)
- repro: X's arm A, seed 160, 2400 steps, bit-identical to X's record.
- S10: the header routing target equals the perfect gate's; the fit recipe reproduces CHECK 45 at
  k=2 on the grouped layout; explore_far_cue's header-layout CHECKs.
- S11: the detector reads no stream label; a reset changes only W_in, W_h, W_g and their Adam
  state; with the threshold at 2.0 the run equals arm A bit for bit.
- S12: lambda = 0 equals SLOW_MEM bit for bit; the penalty's gradient reaches only the gate
  parameters and the embedding; the penalty matches the diagnostic eta^2 on a probe batch.

SEGMENTS: batch 3's machinery (--order, --resume against explore_out/batch4_checks.json,
--report-only, --stopped-early, --dry); a watcher outside this file pushes every saved run.

Run:  python explore_batch4.py [--workers 4] [--dry] [--order ...] [--resume] [--report-only]
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
import explore_far_express as s10
import explore_gate_reset as s11
import explore_slow_pos as s12
from explore_batch1 import verdict, PROMISING_D, PROMISING_P

SCREENS = (s10, s11, s12)
DRY_ITERS = 3600
CHECKS_RECORD = os.path.join(ec.OUT_DIR, "batch4_checks.json")
DEFAULT_ORDER = "far_express,gate_reset,slow_pos"


def git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True,
                          check=True).stdout


def code_changes(rev):
    files = git("diff", "--name-only", rev).split() + \
        git("ls-files", "--others", "--exclude-standard").split()
    return sorted(f for f in set(files)
                  if not f.startswith("explore_out/") and f != "explore_batch4.py")


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
    pv = ec.print_banner("batch 4: far_express (S10), gate_reset (S11), slow_pos (S12)" + dl)
    print(f"  recipe: lr = test_channel_binding.SUB_LR = {ec.SUB_LR:g} (passed explicitly), "
          f"MAX_ITERS = test_short_conv.MAX_ITERS = {ec.MAX_ITERS}, task P={ec.TASK.P} S={ec.TASK.S} "
          f"n_vals={ec.TASK.n_vals} n_q={ec.TASK.n_q}, 1 torch thread per run")
    print(f"  S10 readings: a gate fits (argmax >= 0.99) and far_nudge >= {s10.DISCOVERY_N}/10 -> discovery "
          f"problem; a gate fits and far_nudge <= {s10.HEADSTART_N}/10 -> the loss cannot finish it; no gate "
          f"fits -> cannot express")
    print(f"  S11, S12 rule: promising if b - c >= {PROMISING_D} and McNemar p < {PROMISING_P}; not if "
          f"b - c <= 0; else inconclusive (DISCOVERED vs X's arm A)")
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
                  f"file outside explore_out/ and explore_batch4.py differs from {rec['git']}, where "
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

    d10 = s10.report(ec.load_store(s10.NAME), save=not args.dry)
    v10, _ = status(s10, d10["reading"])
    print(f"  {tagp}S10 {s10.NAME}: {v10}  [{ec.BANNER}]\n")
    d11 = s11.report(ec.load_store(s11.NAME))
    v11, _ = status(s11, verdict(d11))
    print(f"  {tagp}VERDICT {s11.NAME}: {v11}  [{ec.BANNER}]\n")
    d12 = s12.report(ec.load_store(s12.NAME))
    v12, _ = status(s12, verdict(d12))
    print(f"  {tagp}VERDICT {s12.NAME}: {v12}  [{ec.BANNER}]\n")
    print("#" * 100)
    print(f"SUMMARY — {ec.BANNER}{dl}")
    fa = ", ".join(f"{k} {d10['fits'][f'header|{k}']['argmax']:.3f}" for k in s10.KINDS)
    print(f"  S10 far_express: {v10}")
    print(f"      header-layout fits (argmax): {fa}; far_nudge DISCOVERED {d10['discovered']}/{d10['n']}")
    print(f"  S11 gate_reset: {v11} (DISCOVERED {d11['new']}/{d11['n']} vs X A {d11['old']}/{d11['n']}, "
          f"{d11['b']} vs {d11['c']}, p = {d11['p']:.3g})")
    vs = d12["vs_slow_mem"]
    print(f"  S12 slow_pos:   {v12} (DISCOVERED {d12['new']}/{d12['n']} vs X A {d12['old']}/{d12['n']}, "
          f"{d12['b']} vs {d12['c']}, p = {d12['p']:.3g}); vs SLOW_MEM {vs['slow_mem']}/{d12['n']}: "
          f"{vs['b']} vs {vs['c']}, p = {vs['p']:.3g}")
    print("#" * 100)


if __name__ == "__main__":
    main()
