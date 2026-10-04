#!/usr/bin/env python
"""
explore_batch13.py — EXPLORATORY, not a result. Batch 13 of the outside-ideas screens, with batches 1-12's
harness, rules and labels and their resume machinery. Everything below was fixed before any run.

  S37 explore_split_target  S36's plateau trigger, capped (<= 2 splits per run, >= 4800 updates apart), with the
                            split targeted by KEYMASS (the largest / smallest mean read-gate mass at the hinge's
                            key positions); S=4, P=4, k=4, conv, 28800 steps, seeds 240-249: SPLITK_M (S33's
                            A_HINGE, Muon), HINGE4k4_A (main's A4k4 + test_slow_start's HINGE recipe, Adam, run
                            here), SPLITK_A (HINGE4k4_A + the targeted trigger)
  S38 explore_gate_state    descriptive: reruns to 9600 of S33's D8_HINGE, X's HINGE_D8 (S=8, k=16; 260-264),
                            S33's A_HINGE (S=4, k=4) and S35's MUON_HINGE16 (S=4, k=16; 240-244), with labelled
                            nearest-class-mean decoders of the stream from the gate state h and from the gate input

BACKGROUND (docstring): see explore_split_target and explore_gate_state.

RECIPE: the Adam groups at test_slow_start.LR = test_stream_recipe.LR = test_stream_curriculum.LR = 1e-3 (read
in the child; the branch's SUB_LR = 1e-3), passed explicitly; Muon lr 0.005 (S33's validity choice); the slow
phase's non-gate groups at a tenth for updates 1-2400; evaluation every 1200; the run paths' own early stops;
main's run paths at 9c5939e in child processes; 1 torch thread per run.

READINGS: in explore_split_target (per optimizer) and explore_gate_state (S=8 configurations). An incomplete arm
or configuration gets no reading.

CHECKS (before any run, after the repro check; any failure stops the batch)
- repro: X's arm A, seed 160, 2400 steps, bit-identical to X's record (in every segment).
- S37: with the trigger off, SPLITK_M = S33's A_HINGE|240 and SPLITK_A = a fresh HINGE4k4_A|240 bit for bit
  through 7200; on a synthetic gate whose key-position map is 2+1+1, KEYMASS picks the shared and the empty
  channel; the served modules.
- S38: one rerun per configuration through 2400 reproduces its record (and every S38 run is checked through
  9600 against its record); the recomputed read gate equals the model's; the decoder scores 1.0 on a synthetic
  separable set and within 0.05 of chance with shuffled labels.

SEQUENCE: repro check; CHECKs; the projection with pool-load timing (batch 10's method: one timing child per
arm or configuration at once, two rounds of 4; the median interval between optimizer steps); RUNTIME RULE: if the
projected worst-case makespan is over 9 h, HINGE4k4_A and SPLITK_A are cut to 240-245 (stored); then one pool,
S37's 28800-step runs queued first (arm priority), then S38's.
SEGMENTS: batches 4-12's machinery (--resume against explore_out/batch13_checks.json, --report-only,
--stopped-early, --dry); a watcher outside this file pushes every saved run.

Run:  python explore_batch13.py [--workers 4] [--dry] [--resume] [--report-only]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common2 as c2
import explore_main9c as mt
import explore_split_target as s37
import explore_gate_state as s38
from test_binding_recipe import makespan
from test_binding_onset import EVAL_EVERY

SCREENS = (s37, s38)
DRY_ITERS = 3600
CHECKS_RECORD = os.path.join(ec.OUT_DIR, "batch13_checks.json")
CUT_H = 9.0
CUT_ARMS = ("HINGE4k4_A", "SPLITK_A")


def git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True, check=True).stdout


def code_changes(rev):
    files = git("diff", "--name-only", rev).split() + git("ls-files", "--others", "--exclude-standard").split()
    return sorted(f for f in set(files) if not f.startswith("explore_out/") and f != "explore_batch13.py")


def base(m):
    return m.NAME[4:] if m.NAME.startswith("dry_") else m.NAME


def make_dry():
    c2.DRY.update(on=True, iters=DRY_ITERS)
    for m in SCREENS:
        m.NAME = "dry_" + m.NAME
        m.ARMS = {k: dict(a, iters=min(a["iters"], DRY_ITERS), seeds=tuple(a["seeds"][:1])) for k, a in m.ARMS.items()}


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


def rule_name():
    return s37.NAME + "_runtime"


def apply_cut(cut):
    if cut and cut.get("cut"):
        for k in CUT_ARMS:
            s37.ARMS[k]["seeds"] = tuple(s for s in s37.ARMS[k]["seeds"] if s in s37.CUT_SEEDS)


def projection(workers, dry):
    jobs1 = [(s37, "SPLITK_M"), (s37, "SPLITK_A"), (s38, "D8_HINGE_M"), (s38, "HINGE_D8_A")]
    jobs2 = [(s37, "HINGE4k4_A"), (s38, "A_HINGE_M"), (s38, "HINGE16_M"), (s38, "HINGE16_M")]
    t = {}
    for jobs in (jobs1, jobs2):
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:            # each job is a child process
            res = list(ex.map(lambda j: (j[1], j[0].timing([j[1]])[j[1]]), jobs))
        for k, v in res:
            t[k] = max(t.get(k, [0, 0, 0]), list(v) + [0] * (3 - len(v)))

    def show(tag):
        d, rows = [], []
        for m in SCREENS:
            for k, a in m.ARMS.items():
                ts, te, td = t[k]
                per = a["iters"] * ts + (a["iters"] // EVAL_EVERY) * te + (len(s38.AT) * td if m is s38 else 0)
                d += [per] * len(a["seeds"])
                rows.append((m.NAME, k, len(a["seeds"]), per, a["iters"], ts, te, td))
        print(f"  PROJECTION{tag} (worst case: every run to its iters; pool-load timing: the median interval between "
              f"optimizer steps, one child per arm at once, two rounds of 4, 1 thread each, this machine):")
        for nm, k, n, per, it, ts, te, td in rows:
            print(f"    {nm:<14} {k:<11} {n:>2} runs x {per / 60:5.1f} min  ({it} steps x {ts * 1e3:.1f} ms + "
                  f"{it // EVAL_EVERY} evals x {te:.1f} s" + (f" + {len(s38.AT)} probes x {td:.1f} s" if td else "") + ")")
        ms = makespan(d, workers)
        print(f"    total {sum(d) / 3600:.2f} h of runs; makespan on {workers} workers {ms / 3600:.2f} h (worst case)", flush=True)
        return ms
    ms = show("")
    st = ec.load_store(rule_name())
    cut = st["meta"].get("cut")
    if cut is None:
        do = ms / 3600 > CUT_H
        cut = dict(cut=do, makespan_h=ms / 3600, workers=workers, time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
        if not dry:
            st["meta"]["cut"] = cut
            st["meta"].setdefault("provenance", ec.provenance())
            ec.save_store(rule_name(), st)
        print(f"  RUNTIME RULE: projected makespan {ms / 3600:.2f} h {'>' if do else '<='} {CUT_H:g} h: "
              + (f"HINGE4k4_A and SPLITK_A cut to 240-245" if do else "no cut") + (" (dry run: not stored)" if dry else " (stored)"),
              flush=True)
    else:
        print(f"  RUNTIME RULE (stored {cut['time']}: makespan {cut['makespan_h']:.2f} h on {cut['workers']} workers): "
              + ("HINGE4k4_A and SPLITK_A cut to 240-245" if cut["cut"] else "no cut"), flush=True)
    apply_cut(cut)
    if cut["cut"]:
        show(" AFTER THE CUT")


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
    pv = ec.print_banner("batch 13: split_target (S37), gate_state (S38)" + dl)
    c = mt.run_child("explore_muon_scale_child", "constants", {})
    assert c["tss_LR"] == c["tsr_LR"] == c["tscur_LR"] == s37.LR == s38.LR, c
    ch = ec.load_store("muon_scale_valid")["meta"].get("choice") or {}
    assert ch.get("a") == ch.get("b") == s37.MUON_LR == s38.MUON_LR, ch
    print(f"  recipe read in the child from main at {c['main_sha'][:7]}: test_slow_start.LR {c['tss_LR']:g}, "
          f"test_stream_recipe.LR {c['tsr_LR']:g}, test_stream_curriculum.LR {c['tscur_LR']:g} (Adam groups; the branch's "
          f"SUB_LR {ec.SUB_LR:g}); WARM {c['tss_WARM']}, TAU {c['tss_TAU']}, LAMBDA {c['tss_LAMBDA']}; Muon lr {s37.MUON_LR:g} "
          f"(S33's validity choice {ch}); S37 28800 steps, S38 reruns to 9600; 1 torch thread per run")
    for m in SCREENS:
        for k, a in m.ARMS.items():
            print(f"  lr read for {m.NAME} {k}: {a['lr']:g} (Adam groups)" + (f"; Muon lr {a['muon_lr']:g}" if a["muon_lr"] else ""))
    print(f"  S37 readings (per optimizer): 'the targeted split fixes four-stream merges' if >= {s37.FIX_N}/10 bind, 'it does "
          f"not' if <= {s37.NOT_N}/10, otherwise neither")
    print(f"  S38 readings (S=8, at 4800, median over 5 seeds of the stream decodability from h at key positions): 'the "
          f"eight-stream gate state does not carry the stream' if < {s38.NOT_MAX}; 'it does' if >= {s38.DOES_MIN}; otherwise "
          f"'partial'")
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
                  f"file outside explore_out/ and explore_batch13.py differs from {rec['git']}, where "
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
        projection(args.workers, args.dry)
        t0 = time.time()
        c2.run_jobs2(SCREENS, workers=args.workers, pv=pv)
        print(f"  batch wall clock {(time.time() - t0) / 60:.1f} min (this segment)")
    else:
        print("  REPORT ONLY: from the saved runs (no repro check, CHECKs or runs in this call)")
        apply_cut(ec.load_store(rule_name())["meta"].get("cut"))
    print("#" * 100)
    print(f"REPORTS — {ec.BANNER}{dl}")
    print("#" * 100)
    tagp = "DRY RUN " if args.dry else ""
    d37 = s37.report(ec.load_store(s37.NAME))
    v37 = {}
    for arm, opt in (("SPLITK_M", "MUON"), ("SPLITK_A", "ADAM")):
        n, N = completeness(s37, arm)
        v37[opt] = d37[arm]["reading"] if n == N else f"INCOMPLETE, {n}/{N} runs, no reading"
        print(f"  {tagp}READING {s37.NAME} ({opt}): {v37[opt]}  [{ec.BANNER}]")
    print()
    d38 = s38.report(ec.load_store(s38.NAME))
    v38 = {}
    for conf in ("D8_HINGE_M", "HINGE_D8_A"):
        n, N = completeness(s38, conf)
        v38[conf] = d38[conf]["reading"] if n == N else f"INCOMPLETE, {n}/{N} runs, no reading"
        print(f"  {tagp}READING {s38.NAME} {conf}: {v38[conf]}  [{ec.BANNER}]")
    print()
    print("#" * 100)
    print(f"SUMMARY — {ec.BANNER}{dl}")
    for arm, opt in (("SPLITK_M", "MUON"), ("SPLITK_A", "ADAM")):
        dd = d37[arm]
        print(f"  S37 split_target ({opt}): {v37[opt]} (KEYMASS targeting on the fired splits {dd['fired_key']}/{dd['fired']}, "
              f"the all-position rule {dd['fired_all']}/{dd['fired']}; HARM {dd['harm']} runs)")
    print(f"  S37 HINGE4k4_A (Adam, k=4, no trigger): BOUND {d37['HINGE4k4_A']['bound']}/{d37['HINGE4k4_A']['n']}")
    for conf in s38.ARMS:
        print(f"  S38 gate_state {conf}: " + (v38[conf] if conf in v38 else d38[conf]["reading"]))
    print("#" * 100)


if __name__ == "__main__":
    main()
