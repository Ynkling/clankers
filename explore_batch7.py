#!/usr/bin/env python
"""
explore_batch7.py — EXPLORATORY, not a result. Batch 7 of the outside-ideas screens, with batches
1-6's harness, rules and labels and their resume machinery. Everything below was fixed before any
run.

  S19 explore_token_hinge   S13's SLOW_HINGE plus a third term on the key token (eta2_key):
                            FAR_TOK (header layout, seeds 160-169), NEAR_TOK (grouped, 160-179)
  S20 explore_kick_untimed  SLOW_MEM + one Gaussian kick at update 120, sized as S16's median first
                            kick, no trigger; seeds 160-199                       (KICK120)
  S21 explore_merge_kick    X's recorded S=4, k=4 A4k4 runs merged at 9600 that stayed merged (main
                            branch, loaded read-only): rerun to 9600, then 9600 updates twice,
                            CONTROL and KICKS (100x the gradient norm every 600)  (SR_A4k4, SA_A4k4)

BACKGROUND (post hoc, from batches 5-6): see each screen's docstring. S16: random kicks on the
hinge's batches 12/20 vs SLOW_HINGE 14/20, SLOW_MEM 4/20; direction mattered on 3 seeds; kicks routed
later. S11, S17: blocking one cheap split exposes the next; on the header layout the key token. S18:
the stalled runs are stuck in their non-gate parameters; S15: slow memory prevents it. Main line:
of 22 S=4, k=4 runs with streams sharing a channel at 9600, 10 split later and bound, 12 never did.

RECIPE (every arm): lr test_channel_binding.SUB_LR = 1e-3, passed explicitly (the SLOW_MEM-schedule
arms' warm-up group at SUB_LR/10); MAX_ITERS test_short_conv.MAX_ITERS = 24000 for S19 and S20 (S21:
the recorded runs' own path to 9600, then 9600 continued updates twice); evaluation every 1200;
early stop after 3 evaluations >= 0.95 (S21's continuations run their 9600 updates); 1 torch thread
per run. Tasks: S19 the header and grouped layouts (P=4, S=2, n_vals=16, n_q=1); S20 grouped; S21
task_for(4, 4) (S=4, P=4, n_vals=16, n_q=1), arm A with k=4 and conv 'layer' width 4.

RULES
- S19 FAR_TOK, DISCOVERED: promising if >= 4/10; not if <= 1/10; otherwise inconclusive. NEAR_TOK:
  "keeps the near case" if >= 16/20; "costs the near case" if <= 13/20; otherwise neither.
- S20 vs SLOW_MEM over 40 seeds, DISCOVERED: promising if b - c >= 4 and McNemar p < 0.10; not if
  b - c <= 0; otherwise inconclusive (explore_batch1.verdict). Reading: "the trigger is not needed"
  if >= 18/20 on the seeds where S13's hinge never fired and >= 10/20 where it fired; "an untimed
  kick costs the easy seeds" if <= 16/20 on the first group; otherwise neither.
- S21: "kicks split merges" if >= 4/8 split under KICKS; "they do not" if <= 1/8; otherwise neither.
- An incomplete screen gets no verdict or reading.

CHECKS (before any run; any failure stops the batch)
- repro: X's arm A, seed 160, 2400 steps, bit-identical to X's record (in every segment).
- S19: with the key term's TAU at 1.0, FAR_TOK equals S17's FAR_SLOW_HINGE and NEAR_TOK equals S13's
  SLOW_HINGE bit for bit (3600 steps); on 20 probe batches of each layout the perfect gate scores
  below 0.1 on eta2_key and a gate set by key token above 0.5.
- S20: with the kick's norm at 0 the run equals SLOW_MEM bit for bit (3600 steps); exactly one kick;
  logged norms equal the targets.
- S21: each rerun reproduces its recorded curve to 9600 and CONTROL to 19200 (each job); the
  continuations see identical batches and start from equal, separately stored optimizer states;
  kick norms equal 100x the batch gradient norm.

SEGMENTS: batches 4-6's machinery (--order, --resume against explore_out/batch7_checks.json,
--report-only, --stopped-early, --dry); a watcher outside this file pushes every saved run.

Run:  python explore_batch7.py [--workers 4] [--dry] [--order ...] [--resume] [--report-only]
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
import explore_token_hinge as s19
import explore_kick_untimed as s20
import explore_merge_kick as s21
from explore_batch1 import PROMISING_D, PROMISING_P

SCREENS = (s19, s20, s21)
DRY_ITERS = 3600
CHECKS_RECORD = os.path.join(ec.OUT_DIR, "batch7_checks.json")
DEFAULT_ORDER = "merge_kick,token_hinge,kick_untimed"


def git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True,
                          check=True).stdout


def code_changes(rev):
    files = git("diff", "--name-only", rev).split() + \
        git("ls-files", "--others", "--exclude-standard").split()
    return sorted(f for f in set(files)
                  if not f.startswith("explore_out/") and f != "explore_batch7.py")


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
    pv = ec.print_banner("batch 7: token_hinge (S19), kick_untimed (S20), merge_kick (S21)" + dl)
    print(f"  recipe: lr = test_channel_binding.SUB_LR = {ec.SUB_LR:g} (passed explicitly), "
          f"MAX_ITERS = test_short_conv.MAX_ITERS = {ec.MAX_ITERS} (S21: rerun to {s21.T0}, then {s21.T_CONT} "
          f"continued updates twice), task P={ec.TASK.P} S={ec.TASK.S} n_vals={ec.TASK.n_vals} n_q={ec.TASK.n_q} "
          f"(S21: task_for(4, 4), k=4, conv), 1 torch thread per run")
    print(f"  S19 rule: FAR_TOK promising if DISCOVERED >= {s19.FAR_PROMISING}/10, not if <= {s19.FAR_NOT}/10, else "
          f"inconclusive; NEAR_TOK 'keeps the near case' if >= {s19.NEAR_KEEPS}/20, 'costs the near case' if <= "
          f"{s19.NEAR_COSTS}/20, else neither")
    print(f"  S20 rule (vs SLOW_MEM, 40 seeds): promising if b - c >= {PROMISING_D} and McNemar p < {PROMISING_P}; "
          f"not if b - c <= 0; else inconclusive (DISCOVERED); reading: 'the trigger is not needed' if easy >= "
          f"{s20.TRIG_GOOD}/20 and hard >= {s20.TRIG_HARD}/20; 'an untimed kick costs the easy seeds' if easy <= "
          f"{s20.COST_EASY}/20; else neither")
    print(f"  S21 rule: 'kicks split merges' if >= {s21.SPLIT_MIN}/8 split under KICKS; 'they do not' if <= "
          f"{s21.NOT_MAX}/8; else neither (split: the merged pair on different channels and accuracy >= "
          f"{s21.BIND} at some evaluation up to {s21.T0 + s21.T_CONT})")
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
                  f"file outside explore_out/ and explore_batch7.py differs from {rec['git']}, where "
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

    d19 = s19.report(ec.load_store(s19.NAME))
    v19f, _ = status(s19, d19["far"]["rule"], "FAR_TOK")
    v19n, _ = status(s19, d19["near"]["rule"], "NEAR_TOK")
    print(f"  {tagp}VERDICT {s19.NAME} FAR_TOK: {v19f} (DISCOVERED {d19['far']['new']}/{d19['far']['n']})  [{ec.BANNER}]")
    print(f"  {tagp}READING {s19.NAME} NEAR_TOK: {v19n} (DISCOVERED {d19['near']['new']}/{d19['near']['n']})  "
          f"[{ec.BANNER}]\n")
    d20 = s20.report(ec.load_store(s20.NAME))
    v20, full20 = status(s20, d20["verdict"])
    rd20 = d20["reading"] if full20 else "none (S20 incomplete)"
    print(f"  {tagp}VERDICT {s20.NAME} (vs SLOW_MEM, 40 seeds): {v20}; reading: {rd20}  [{ec.BANNER}]\n")
    d21 = s21.report(ec.load_store(s21.NAME))
    v21, _ = status(s21, d21["rule"])
    print(f"  {tagp}READING {s21.NAME}: {v21} (split under KICKS {d21['split']}/{d21['n']}; CONTROL "
          f"{d21['split_control']}/{d21['n']})  [{ec.BANNER}]\n")
    print("#" * 100)
    print(f"SUMMARY — {ec.BANNER}{dl}")
    f, n = d19["far"], d19["near"]
    print(f"  S19 token_hinge FAR_TOK: {v19f} (DISCOVERED {f['new']}/{f['n']}; S17's FAR_SLOW_HINGE {f['old']}/{f['n']}, "
          f"FAR_TOK only {f['b']} vs {f['c']}, p = {f['p']:.3g})")
    print(f"  S19 token_hinge NEAR_TOK: {v19n} (DISCOVERED {n['new']}/{n['n']}; S13's SLOW_HINGE {n['old']}/{n['n']}, "
          f"NEAR_TOK only {n['b']} vs {n['c']}, p = {n['p']:.3g})")
    sh, kk = d20["slow_hinge"], d20["kick"]
    print(f"  S20 kick_untimed: {v20} vs SLOW_MEM (DISCOVERED {d20['new']}/{d20['n']} vs {d20['old']}/{d20['n']}, "
          f"{d20['b']} vs {d20['c']}, p = {d20['p']:.3g}); reading: {rd20} (easy {d20['n_easy']}/{d20['n_easy_run']}, "
          f"hard {d20['n_hard']}/{d20['n_hard_run']})")
    print(f"      vs SLOW_HINGE {sh['new']}/{sh['n']} vs {sh['old']}/{sh['n']} ({sh['b']} vs {sh['c']}, p = {sh['p']:.3g}); "
          f"vs S16's KICK on its seeds {kk['new']}/{kk['n']} vs {kk['old']}/{kk['n']} ({kk['b']} vs {kk['c']}, "
          f"p = {kk['p']:.3g})")
    print(f"  S21 merge_kick: {v21} (split under KICKS {d21['split']}/{d21['n']}; under CONTROL "
          f"{d21['split_control']}/{d21['n']})")
    print("#" * 100)


if __name__ == "__main__":
    main()
