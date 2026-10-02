#!/usr/bin/env python
"""
explore_batch8.py — EXPLORATORY, not a result. Batch 8 of the outside-ideas screens, with batches
1-7's harness, rules and labels and their resume machinery. Everything below was fixed before any
run.

  S22 explore_far_tok_check  S19's FAR_TOK on fresh seeds 170-179 (FAR_TOK_NEW); the key-token term
                             only from update 2401 on the header (FAR_TOK_LATE, 160-179) and grouped
                             (NEAR_TOK_LATE, 160-179) layouts; every key-term firing logged
  S23 explore_far_single     the missing control: a single channel (arm B, k=1, no gate) on the
                             header layout, lr 1e-3 (B_FAR) and the slow schedule (B_FAR_SLOW), 160-169
  S24 explore_merge_split    S21's 8 merged S=4, k=4 runs: rerun to 9600, the gate state of the merged
                             and a non-merged pair, then 9600 updates three ways: CONTROL, SPLIT (copy
                             the busiest channel's W_g row to the idlest, noise on both, W_g's Adam
                             state reset), NOISE (the same noise without the copy, same reset)

BACKGROUND (post hoc, from batch 7): see each screen's docstring. S19 FAR_TOK DISCOVERED 6/10, BOUND
9/10, ROUTED* at the end 3/10; NEAR_TOK's key term fired at the start of every run and routing at 1200
fell to 0/20. S20: the trigger matters (KICK120 vs SLOW_HINGE 0 vs 7). S21: merged pairs have equal
read gates; 100x kicks split 0/8.

RECIPE (every arm): lr test_channel_binding.SUB_LR = 1e-3, passed explicitly (S22's and S23's slow
arms at 1e-4 for updates 1-2400 on every non-gate parameter; S22's gate group at 1e-3 throughout);
MAX_ITERS test_short_conv.MAX_ITERS = 24000 for S22 and S23 (S24: the recorded runs' own path to 9600,
then 9600 continued updates three times); evaluation every 1200; early stop after 3 evaluations >= 0.95
(S24's continuations run their 9600 updates); 1 torch thread per run. Tasks: S22 the header and
grouped layouts (P=4, S=2, n_vals=16, n_q=1); S23 header; S24 task_for(4, 4) (S=4, P=4), arm A with
k=4 and conv 'layer' width 4.

RULES
- S22 FAR_TOK_NEW (DISCOVERED and ROUTED* at the end both printed): "replicates" if DISCOVERED >= 4/10
  and ROUTED*@end >= 2/10; "does not" if DISCOVERED <= 1/10; otherwise neither. FAR_TOK_LATE vs FAR_TOK
  (20 seeds), ROUTED*@end: "the late start helps" if b - c >= 4 and McNemar p < 0.1; "it costs" if
  c - b >= 4; otherwise neither (DISCOVERED and BOUND printed alongside). NEAR_TOK_LATE: "routing
  returns" if ROUTED*@1200 >= 10/20 and DISCOVERED >= 16/20.
- S23: "header binding needs the partition" if B_FAR and B_FAR_SLOW each bind <= 1/10; "a single
  channel binds the header layout" if either binds >= 3/10; otherwise neither.
- S24: "splitting un-merges" if SPLIT splits >= 4/8 and NOISE <= 1/8; "noise suffices" if NOISE >= 4/8;
  "neither works" if SPLIT <= 1/8. Separately: "the gate state tells the merged streams apart" if
  ||h_a - h_b|| / mean ||h|| for the merged pair is >= 0.5 x that of the non-merged pair in >= 6/8 runs.
- An incomplete screen gets no verdict or reading.

CHECKS (before any run; any failure stops the batch)
- repro: X's arm A, seed 160, 2400 steps, bit-identical to X's record (in every segment).
- S22: FAR_TOK_NEW's code path equals S19's (seed 160 reproduces S19's FAR_TOK record to 3600);
  FAR_TOK_LATE equals FAR_SLOW_HINGE (and NEAR_TOK_LATE equals SLOW_HINGE) bit for bit through update
  2400 on seed 160; the firing logs agree with the hinge counts.
- S23: B_FAR's model is k=1 with no gate parameters; with the slow schedule's low lr set to 1e-3,
  B_FAR_SLOW equals B_FAR bit for bit (3600 steps); B_FAR_SLOW's optimizer sees 1e-4 to 2400, 1e-3 after.
- S24: the machinery on one run of each path; in each job the rerun reproduces to 9600 and CONTROL the
  record to 19200, SPLIT's / NOISE's W_g rows after the operation match the spec, and the three
  continuations see identical batches and start from separately stored optimizer states.

SEGMENTS: batches 4-7's machinery (--order, --resume against explore_out/batch8_checks.json,
--report-only, --stopped-early, --dry); a watcher outside this file pushes every saved run.

Run:  python explore_batch8.py [--workers 4] [--dry] [--order ...] [--resume] [--report-only]
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
import explore_far_tok_check as s22
import explore_far_single as s23
import explore_merge_split as s24

SCREENS = (s22, s23, s24)
DRY_ITERS = 3600
CHECKS_RECORD = os.path.join(ec.OUT_DIR, "batch8_checks.json")
DEFAULT_ORDER = "merge_split,far_tok_check,far_single"


def git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True,
                          check=True).stdout


def code_changes(rev):
    files = git("diff", "--name-only", rev).split() + \
        git("ls-files", "--others", "--exclude-standard").split()
    return sorted(f for f in set(files)
                  if not f.startswith("explore_out/") and f != "explore_batch8.py")


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
    pv = ec.print_banner("batch 8: far_tok_check (S22), far_single (S23), merge_split (S24)" + dl)
    print(f"  recipe: lr = test_channel_binding.SUB_LR = {ec.SUB_LR:g} (passed explicitly; slow arms at "
          f"{s23.LR_LOW:g} for updates 1-{s23.WARM} on every non-gate parameter), MAX_ITERS = "
          f"test_short_conv.MAX_ITERS = {ec.MAX_ITERS} (S24: rerun to {s24.T0}, then {s24.T_CONT} continued "
          f"updates three times), task P={ec.TASK.P} S={ec.TASK.S} n_vals={ec.TASK.n_vals} n_q={ec.TASK.n_q} "
          f"(S24: task_for(4, 4), k=4, conv), 1 torch thread per run")
    for a_name, m in (("FAR_TOK_NEW", s22), ("FAR_TOK_LATE", s22), ("NEAR_TOK_LATE", s22), ("B_FAR", s23),
                      ("B_FAR_SLOW", s23), ("SR_A4k4", s24), ("SA_A4k4", s24)):
        print(f"  lr read for {m.NAME} {a_name}: {m.ARMS[a_name]['lr']:g}")
    print("  S22 rules: FAR_TOK_NEW 'replicates' if DISCOVERED >= 4/10 and ROUTED*@end >= 2/10, 'does not' if "
          "DISCOVERED <= 1/10, else neither; FAR_TOK_LATE vs FAR_TOK (ROUTED*@end) 'the late start helps' if "
          "b - c >= 4 and McNemar p < 0.1, 'it costs' if c - b >= 4, else neither; NEAR_TOK_LATE 'routing "
          "returns' if ROUTED*@1200 >= 10/20 and DISCOVERED >= 16/20")
    print(f"  S23 rule: 'header binding needs the partition' if B_FAR and B_FAR_SLOW each bind <= {s23.NEEDS}/10; "
          f"'a single channel binds the header layout' if either binds >= {s23.BINDS}/10; else neither")
    print(f"  S24 rules: 'splitting un-merges' if SPLIT >= {s24.SPLIT_MIN}/8 and NOISE <= {s24.NOT_MAX}/8; 'noise "
          f"suffices' if NOISE >= {s24.SPLIT_MIN}/8; 'neither works' if SPLIT <= {s24.NOT_MAX}/8 (split: the merged "
          f"pair on different channels and accuracy >= {s24.BIND} at some evaluation up to {s24.T0 + s24.T_CONT}); "
          f"'the gate state tells the merged streams apart' if merged ratio >= {s24.GATE_REL} x non-merged in >= "
          f"{s24.GATE_RUNS}/8")
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
                  f"file outside explore_out/ and explore_batch8.py differs from {rec['git']}, where "
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

    d22 = s22.report(ec.load_store(s22.NAME))
    n22, l22, r22 = d22["new"], d22["late"], d22["near"]
    v22n, _ = status(s22, n22["rule"], "FAR_TOK_NEW")
    v22l, full_l = status(s22, l22["rule"], "FAR_TOK_LATE")
    if full_l and completeness(s22, "FAR_TOK_NEW")[0] < completeness(s22, "FAR_TOK_NEW")[1]:
        v22l = "INCOMPLETE (FAR_TOK_NEW, the pair for 170-179, incomplete), no reading"
    v22r, _ = status(s22, r22["rule"], "NEAR_TOK_LATE")
    print(f"  {tagp}READING {s22.NAME} FAR_TOK_NEW: {v22n} (DISCOVERED {n22['discovered']}/{n22['n']}, ROUTED*@end "
          f"{n22['routed']}/{n22['n']})  [{ec.BANNER}]")
    print(f"  {tagp}READING {s22.NAME} FAR_TOK_LATE vs FAR_TOK: {v22l} (ROUTED*@end {l22['new']}/{l22['n']} vs "
          f"{l22['old']}/{l22['n']}, {l22['b']} vs {l22['c']}, p = {l22['p']:.3g})  [{ec.BANNER}]")
    print(f"  {tagp}READING {s22.NAME} NEAR_TOK_LATE: {v22r} (ROUTED*@1200 {r22['routed1200']}/{r22['n']}, "
          f"DISCOVERED {r22['discovered']}/{r22['n']})  [{ec.BANNER}]\n")
    d23 = s23.report(ec.load_store(s23.NAME))
    v23, _ = status(s23, d23["rule"])
    print(f"  {tagp}READING {s23.NAME}: {v23} (BOUND B_FAR {d23['b_far']['bound']}/{d23['b_far']['n']}, B_FAR_SLOW "
          f"{d23['b_far_slow']['bound']}/{d23['b_far_slow']['n']})  [{ec.BANNER}]\n")
    d24 = s24.report(ec.load_store(s24.NAME))
    v24, _ = status(s24, "; ".join(d24["rule"]))
    v24g, _ = status(s24, d24["gate_rule"])
    print(f"  {tagp}READING {s24.NAME}: {v24} (split by {s24.T0 + s24.T_CONT}: SPLIT {d24['split']}/{d24['n']}, NOISE "
          f"{d24['noise']}/{d24['n']}, CONTROL {d24['control']}/{d24['n']})  [{ec.BANNER}]")
    print(f"  {tagp}READING {s24.NAME} gate state: {v24g} (merged ratio >= {s24.GATE_REL} x non-merged in "
          f"{d24['gate_runs']}/{d24['n']})  [{ec.BANNER}]\n")
    print("#" * 100)
    print(f"SUMMARY — {ec.BANNER}{dl}")
    print(f"  S22 far_tok_check FAR_TOK_NEW: {v22n} (DISCOVERED {n22['discovered']}/{n22['n']}, ROUTED*@end "
          f"{n22['routed']}/{n22['n']})")
    print(f"  S22 far_tok_check FAR_TOK_LATE vs FAR_TOK: {v22l} (ROUTED*@end {l22['new']}/{l22['n']} vs "
          f"{l22['old']}/{l22['n']}, {l22['b']} vs {l22['c']}, p = {l22['p']:.3g})")
    print(f"  S22 far_tok_check NEAR_TOK_LATE: {v22r} (ROUTED*@1200 {r22['routed1200']}/{r22['n']}, DISCOVERED "
          f"{r22['discovered']}/{r22['n']})")
    print(f"  S23 far_single: {v23} (BOUND B_FAR {d23['b_far']['bound']}/{d23['b_far']['n']}, B_FAR_SLOW "
          f"{d23['b_far_slow']['bound']}/{d23['b_far_slow']['n']})")
    print(f"  S24 merge_split: {v24} (SPLIT {d24['split']}/{d24['n']}, NOISE {d24['noise']}/{d24['n']}, CONTROL "
          f"{d24['control']}/{d24['n']}); gate state: {v24g} ({d24['gate_runs']}/{d24['n']})")
    print("#" * 100)


if __name__ == "__main__":
    main()
