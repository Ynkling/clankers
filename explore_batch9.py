#!/usr/bin/env python
"""
explore_batch9.py — EXPLORATORY, not a result. Batch 9 of the outside-ideas screens, with batches
1-8's harness, rules and labels and their resume machinery. Everything below was fixed before any
run (the batch 9 spec with its amendment: S27 balance_d8 replaced by S27 zloss_merge, S28 added).

  S25 explore_muon_recipe   arm A and S13's HINGE recipe under Muon (Newton-Schulz orthogonalized
                            momentum on every 2-D weight, Adam 1e-3 on the embedding); the Muon lr
                            chosen first on the perfect gate (0.005 / 0.01 / 0.02, seeds 160-161, 6000
                            steps); MUON_A, MUON_HINGE, seeds 160-179
  S26 explore_gate_langevin SLOW_MEM + N(0, sigma_t^2) on W_in, W_h, W_g after each step, sigma_t =
                            0.01 x init std, decaying linearly to 0 at update 2400; seeds 160-199
  S27 explore_zloss_merge   S24's 8 merged S=4, k=4 runs: rerun to 9600, then 9600 updates with + 1e-3 x
                            mean (logsumexp of the gate logits)^2 (ST-MoE's router z-loss)
  S28 explore_far_p8        the header layout at P=8: perfect gate (validity), one channel with the
                            slow schedule, S19's FAR_TOK; seeds 160-169

BACKGROUND (docstring; from the main line and batches 6-8)
- test_slow_start (main branch, both machines): slow memory + hinge (HINGE) without restarts: S=2
  P=4 39/40, P=8 37/40, S=4 k=16 35/40 pooled; S=8 0/20. On L, 4 of 10 HINGE_D8 runs split the 8
  streams into two channels of four and stalled near 0.2.
- Batches 6-7: a gradient far above a weight's running gradient gives Adam a fixed-size step
  (sign-like) and inflates the second moment for thousands of updates (W_g x1e6 at 4800). The
  hinge's effect partly runs through this. Muon has no per-weight second moment.
- Ideas from the literature: Langevin noise for saddle escape (Muon meets Tamed Langevin, arXiv
  2610.02158); balanced routing (Sinkhorn/BASE layers; usage-balance losses) against collapse.
- S24: SPLIT un-merged 8/8 by bringing the spare channel's logit level with the shared one; ST-MoE's
  router z-loss keeps logits from saturating, continuously. S23: one channel binds the header layout
  at P=4 (B_FAR 3/10, B_FAR_SLOW 7/10); one channel bound 0/24 on the grouped layout at P=8.

RECIPE (every arm): lr test_channel_binding.SUB_LR = 1e-3, passed explicitly (Adam groups; S25's
Muon groups at the selected Muon lr; the slow arms' warm-up groups at a tenth); MAX_ITERS
test_short_conv.MAX_ITERS = 24000 for S25, S26, S28 (S27: the recorded runs' own path to 9600, then
9600 continued updates); evaluation every 1200; early stop after 3 evaluations >= 0.95 (S27's
continuation runs its 9600 updates); 1 torch thread per run.

RULES
- S25: MUON_A vs X's arm A, MUON_HINGE vs SLOW_HINGE, DISCOVERED. "the recipe survives Muon" if
  MUON_HINGE >= 15/20; "it does not" if <= 10/20; otherwise neither. UNTESTED if no Muon lr binds.
- S26 vs SLOW_MEM (40 seeds), DISCOVERED: promising if b - c >= 4 and McNemar p < 0.1; not if b - c
  <= 0; otherwise inconclusive. Also printed: vs SLOW_HINGE, and ROUTED*@1200.
- S27: "z-loss un-merges" if >= 4/8 split by 19200; "it does not" if <= 1/8; otherwise neither.
- S28: "P=8 header needs the partition" if FAR8_CEIL >= 2/3 and B_FAR8_SLOW <= 2/10; otherwise "it
  does not". FAR_TOK8: DISCOVERED and ROUTED*@end printed; "promising" if ROUTED*@end >= 4/10 and the
  first reading holds.
- An incomplete screen gets no verdict or reading.

CHECKS (before any run; any failure stops the batch; run in the order S26, S28, S27, S25, so that
S25's singular-value range is the last CHECK)
- repro: X's arm A, seed 160, 2400 steps, bit-identical to X's record (in every segment).
- S25: groups cover every parameter once; SliceMuon = torch.optim.Muon on 2-D weights; the Muon/Adam
  swap reaches onset_run (all-Adam reproduces X's arm A); the slow schedule's lrs; TAU 1.0: MUON_HINGE
  = MUON with the slow schedule alone (1200 steps); Newton-Schulz singular values within [0.7, 1.3].
- S26: sigma 0 = SLOW_MEM bit for bit (3600 steps); the noise's std per tensor = sigma_t (2%) at
  updates 1, 1200, 2399; zero from 2400.
- S27: weight 0 = CONTROL bit for bit (1200 updates); the z-loss gradient reaches only the gate and
  the embedding.
- S28: the P=8 header task (each (stream, key) once, S distinct values per key, one CTX per block);
  the perfect gate zeroes cross-stream scores at every layer.

SEGMENTS: batches 4-8's machinery (--order, --resume against explore_out/batch9_checks.json,
--report-only, --stopped-early, --dry); a watcher outside this file pushes every saved run. S25's lr
selection runs once, after the CHECKs and before the projection, and is stored (a resume reuses it).

Run:  python explore_batch9.py [--workers 4] [--dry] [--order ...] [--resume] [--report-only]
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
import explore_muon_recipe as s25
import explore_gate_langevin as s26
import explore_zloss_merge as s27
import explore_far_p8 as s28

SCREENS = (s25, s26, s27, s28)
CHECK_ORDER = (s26, s28, s27, s25)
DRY_ITERS = 3600
CHECKS_RECORD = os.path.join(ec.OUT_DIR, "batch9_checks.json")
DEFAULT_ORDER = "zloss_merge,muon_recipe,far_p8,gate_langevin"


def git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True,
                          check=True).stdout


def code_changes(rev):
    files = git("diff", "--name-only", rev).split() + \
        git("ls-files", "--others", "--exclude-standard").split()
    return sorted(f for f in set(files)
                  if not f.startswith("explore_out/") and f != "explore_batch9.py")


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
    for s in CHECK_ORDER:
        if not s.check():
            print(f"  ! CHECK failed in {s.NAME}; stopping.")
            sys.exit(1)
    print(f"  all CHECKs passed ({(time.time() - t0) / 60:.1f} min)", flush=True)


def muon_note():
    t = s25.muon_overhead()
    ratio = t["muon"] / t["adam"]
    print(f"  PROJECTION NOTE: the projection above times S25's arms under Adam; measured here (50 steps of arm A, "
          f"1 thread) {t['muon']:.1f} ms/step under MUON vs {t['adam']:.1f} ms/step under Adam (x{ratio:.2f}); "
          f"S25's worst case scales by that factor", flush=True)


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
    pv = ec.print_banner("batch 9: muon_recipe (S25), gate_langevin (S26), zloss_merge (S27), far_p8 (S28)" + dl)
    print(f"  recipe: lr = test_channel_binding.SUB_LR = {ec.SUB_LR:g} (passed explicitly to every Adam group; S25's "
          f"Muon groups at the selected Muon lr; slow arms' warm-up groups at a tenth), MAX_ITERS = "
          f"test_short_conv.MAX_ITERS = {ec.MAX_ITERS} (S27: rerun to {s27.T0}, then {s27.T_CONT} continued updates), "
          f"tasks: S25/S26 P={ec.TASK.P} S={ec.TASK.S} grouped; S27 task_for(4, 4), k=4, conv; S28 header layout P=8 "
          f"(L {s28.HEADER8.L}); 1 torch thread per run")
    for m in SCREENS:
        for k, a in m.ARMS.items():
            print(f"  lr read for {m.NAME} {k}: {a['lr']:g}" + (" (Adam groups; the Muon lr is selected below)"
                                                                  if m is s25 else ""))
    print(f"  S25 rule: 'the recipe survives Muon' if MUON_HINGE DISCOVERED >= {s25.SURVIVES}/20, 'it does not' if <= "
          f"{s25.FAILS}/20, else neither; UNTESTED if no Muon lr binds the perfect gate on both seeds")
    print("  S26 rule (vs SLOW_MEM, 40 seeds, DISCOVERED): promising if b - c >= 4 and McNemar p < 0.1; not if b - c "
          "<= 0; else inconclusive")
    print(f"  S27 rule: 'z-loss un-merges' if >= {s27.SPLIT_MIN}/8 split by {s27.T0 + s27.T_CONT}, 'it does not' if <= "
          f"{s27.NOT_MAX}/8, else neither (split: the merged pair on different channels and accuracy >= {s27.BIND})")
    print(f"  S28 rules: 'P=8 header needs the partition' if FAR8_CEIL >= {s28.CEIL_MIN}/3 and B_FAR8_SLOW <= "
          f"{s28.BSLOW_MAX}/10, else 'it does not'; FAR_TOK8 'promising' if ROUTED*@end >= {s28.TOK_MIN}/10 and the "
          f"first reading holds")
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
                  f"file outside explore_out/ and explore_batch9.py differs from {rec['git']}, where "
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
        s25.select_lr(args.workers, dry=args.dry)
        for s in SCREENS:
            c2.print_screen_header2(s)
        c2.projection(SCREENS, args.workers)
        if s25.ARMS["MUON_A"]["seeds"]:
            muon_note()
        t0 = time.time()
        c2.run_jobs2(SCREENS, workers=args.workers, pv=pv)
        print(f"  batch wall clock {(time.time() - t0) / 60:.1f} min (this segment)")
    else:
        print("  REPORT ONLY: from the saved runs (no repro check, CHECKs or runs in this call)")
        s25.select_lr(args.workers, dry=args.dry, run=False)
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

    d25 = s25.report(ec.load_store(s25.NAME))
    if d25["untested"]:
        v25 = "UNTESTED (no Muon lr bound the perfect gate on both seeds)"
    else:
        v25, _ = status(s25, d25["rule"])
    h = d25.get("h") or {}
    a_ = d25.get("a") or {}
    print(f"  {tagp}READING {s25.NAME}: {v25}" + ("" if d25["untested"] else
          f" (Muon lr {d25['muon_lr']:g}; MUON_HINGE DISCOVERED {h['new']}/{h['n']} vs SLOW_HINGE {h['old']}/{h['n']}, "
          f"{h['b']} vs {h['c']}, p = {h['p']:.3g}; MUON_A {a_['new']}/{a_['n']} vs X arm A {a_['old']}/{a_['n']}, "
          f"{a_['b']} vs {a_['c']}, p = {a_['p']:.3g})") + f"  [{ec.BANNER}]\n")
    d26 = s26.report(ec.load_store(s26.NAME))
    v26, _ = status(s26, d26["verdict"])
    sh = d26["slow_hinge"]
    print(f"  {tagp}VERDICT {s26.NAME} (vs SLOW_MEM, 40 seeds): {v26} (DISCOVERED {d26['new']}/{d26['n']} vs "
          f"{d26['old']}/{d26['n']}, {d26['b']} vs {d26['c']}, p = {d26['p']:.3g}; vs SLOW_HINGE {sh['new']}/{sh['n']} vs "
          f"{sh['old']}/{sh['n']})  [{ec.BANNER}]\n")
    d27 = s27.report(ec.load_store(s27.NAME))
    v27, _ = status(s27, d27["rule"])
    print(f"  {tagp}READING {s27.NAME}: {v27} (split by the end of the continuation: ZLOSS {d27['split']}/{d27['n']}; "
          f"S24 on these runs CONTROL {d27['s24']['control']}, SPLIT {d27['s24']['split']}, NOISE {d27['s24']['noise']})"
          f"  [{ec.BANNER}]\n")
    d28 = s28.report(ec.load_store(s28.NAME))
    v28, full28 = status(s28, d28["reading"])
    v28t = d28["tok_reading"] if full28 else "none (S28 incomplete)"
    print(f"  {tagp}READING {s28.NAME}: {v28} (FAR8_CEIL {d28['ceil']}/{d28['n_ceil']}, B_FAR8_SLOW {d28['bslow']}/"
          f"{d28['n_bslow']}); FAR_TOK8: {v28t} (DISCOVERED {d28['tok_disc']}/{d28['n_tok']}, ROUTED*@end "
          f"{d28['tok_routed']}/{d28['n_tok']})  [{ec.BANNER}]\n")
    print("#" * 100)
    print(f"SUMMARY — {ec.BANNER}{dl}")
    print(f"  S25 muon_recipe: {v25}" + ("" if d25["untested"] else
          f" (Muon lr {d25['muon_lr']:g}; MUON_HINGE {h['new']}/{h['n']} vs SLOW_HINGE {h['old']}/{h['n']}; MUON_A "
          f"{a_['new']}/{a_['n']} vs X arm A {a_['old']}/{a_['n']})"))
    print(f"  S26 gate_langevin: {v26} (DISCOVERED {d26['new']}/{d26['n']} vs SLOW_MEM {d26['old']}/{d26['n']}, "
          f"{d26['b']} vs {d26['c']}, p = {d26['p']:.3g}; vs SLOW_HINGE {sh['new']}/{sh['n']} vs {sh['old']}/{sh['n']})")
    print(f"  S27 zloss_merge: {v27} (ZLOSS {d27['split']}/{d27['n']}; S24 SPLIT {d27['s24']['split']}, NOISE "
          f"{d27['s24']['noise']}, CONTROL {d27['s24']['control']})")
    print(f"  S28 far_p8: {v28} (FAR8_CEIL {d28['ceil']}/{d28['n_ceil']}, B_FAR8_SLOW {d28['bslow']}/{d28['n_bslow']}); "
          f"FAR_TOK8 {v28t} (DISCOVERED {d28['tok_disc']}/{d28['n_tok']}, ROUTED*@end {d28['tok_routed']}/{d28['n_tok']})")
    print("#" * 100)


if __name__ == "__main__":
    main()
