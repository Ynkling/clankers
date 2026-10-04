#!/usr/bin/env python
"""
explore_batch14.py — EXPLORATORY, not a result. Batch 14 of the outside-ideas screens, with batches 1-13's
harness, rules and labels and their resume machinery. Everything below was fixed before any run.

  S39 explore_gate_memory  descriptive: reruns to 2400 of S33's D8_HINGE, S33's D8_SLOW (Muon), X's HINGE_D8 (Adam;
                           S=8, k=16; 260-264) and S33's A_HINGE (Muon, S=4, k=4; 240-244), measured at 11 updates:
                           the stream decodability of h at stream-token and key positions, the pre-tanh terms'
                           norms, saturation, read-gate entropy, distinct channels; labels "formed, then lost" /
                           "never formed" / "partial"
  S40 explore_gate_prev    GATE_PREV (the gate also reads the previous token's embedding) at S=8, P=4, k=16, conv,
                           all 8 streams from step 1, 28800 steps, seeds 260-269: D8_PREV_M (S33's D8_HINGE +
                           GATE_PREV, Muon), D8_PREV_A (X's HINGE_D8 + GATE_PREV, Adam)

BACKGROUND (docstring): see explore_gate_memory and explore_gate_prev.

RECIPE: the Adam groups at test_slow_start.LR = test_stream_recipe.LR = test_stream_curriculum.LR = 1e-3 (read
in the child; the branch's SUB_LR = 1e-3), passed explicitly; Muon lr 0.005 (S33's validity choice); the slow
phase's non-gate groups at a tenth for updates 1-2400; evaluation every 1200; the run paths' own early stops;
main's run paths at 9c5939e in child processes; 1 torch thread per run.

READINGS / LABELS: in explore_gate_prev (per arm, BOUND ROUTED at 28800) and explore_gate_memory (per
configuration). An incomplete arm or configuration gets no reading.

CHECKS (before any run, after the repro check; any failure stops the batch)
- repro: X's arm A, seed 160, 2400 steps, bit-identical to X's record (in every segment).
- S39: one rerun per configuration through 1200, with every measurement, reproduces its record (and every S39 run
  is checked through 2400 against its record); the recomputed read gate equals the model's; the decoder scores 1.0
  on a synthetic separable set and within 0.05 of chance with shuffled labels.
- S40: at update 0 every parameter shared with the paired run is bitwise equal; with W_prev fixed at 0 and
  excluded from the optimizer each arm equals its paired run bit for bit through 1200; the gate's input at key
  positions decodes the stream at >= 0.9 at update 0 with GATE_PREV and at chance without it; the optimizer groups
  cover every parameter once; the served modules.

SEQUENCE: repro check; CHECKs; the projection with pool-load timing (batch 10's method: one timing child per arm
or configuration at once, two rounds of 4; the median interval between optimizer steps); RUNTIME RULE: if the
projected worst-case makespan is over 9 h, D8_PREV_A is cut to 260-264 (stored); then one pool, submitted as: the
first `workers` S40 runs, then S39's runs, then the rest of S40's.
SEGMENTS: batches 4-13's machinery (--resume against explore_out/batch14_checks.json, --report-only,
--stopped-early, --dry); a watcher outside this file pushes every saved run.

Run:  python explore_batch14.py [--workers 4] [--dry] [--resume] [--report-only]
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
import explore_gate_memory as s39
import explore_gate_prev as s40
from test_binding_recipe import makespan
from test_binding_onset import EVAL_EVERY

SCREENS = (s40, s39)
DRY_ITERS = 2400
CHECKS_RECORD = os.path.join(ec.OUT_DIR, "batch14_checks.json")
CUT_H = 9.0
CUT_ARMS = ("D8_PREV_A",)


def git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True, check=True).stdout


def code_changes(rev):
    files = git("diff", "--name-only", rev).split() + git("ls-files", "--others", "--exclude-standard").split()
    return sorted(f for f in set(files) if not f.startswith("explore_out/") and f != "explore_batch14.py")


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
    diff = mt.differing()
    v = diff == sorted(mt.SERVED)
    print(f"  CHECK batch 14: the modules served from main are exactly the test modules that differ between this branch "
          f"and {mt.MAIN_SHA[:7]} ({diff}): {'ok' if v else 'FAIL'}", flush=True)
    if not v:
        print("  ! CHECK failed; stopping.")
        sys.exit(1)
    print(f"  all CHECKs passed ({(time.time() - t0) / 60:.1f} min)", flush=True)


def rule_name():
    return s40.NAME + "_runtime"


def apply_cut(cut):
    if cut and cut.get("cut"):
        for k in CUT_ARMS:
            s40.ARMS[k]["seeds"] = tuple(s for s in s40.ARMS[k]["seeds"] if s in s40.CUT_SEEDS)


def per_run(m, k, t):
    a = m.ARMS[k]
    if m is s40:
        ts, te, td = t[k][:3]
        return (a["iters"] * ts + (a["iters"] // EVAL_EVERY) * te + len(s40.AT) * td,
                f"{a['iters']} steps x {ts * 1e3:.1f} ms + {a['iters'] // EVAL_EVERY} evals x {te:.1f} s + {len(s40.AT)} probes x "
                f"{td:.1f} s")
    ts, te, tm = t[k][:3]
    return (a["iters"] * ts + (a["iters"] // EVAL_EVERY) * te + len(s39.AT) * tm,
            f"{a['iters']} steps x {ts * 1e3:.1f} ms + {a['iters'] // EVAL_EVERY} evals x {te:.1f} s + {len(s39.AT)} "
            f"measurements x {tm:.1f} s")


def projection(workers, dry):
    jobs1 = [(s40, "D8_PREV_M"), (s40, "D8_PREV_A"), (s39, "D8_HINGE_M"), (s39, "D8_SLOW_M")]
    jobs2 = [(s39, "HINGE_D8_A"), (s39, "A_HINGE_M"), (s40, "D8_PREV_M"), (s40, "D8_PREV_A")]
    t = {}
    for jobs in (jobs1, jobs2):
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:            # each job is a child process
            res = list(ex.map(lambda j: (j[1], j[0].timing([j[1]])[j[1]]), jobs))
        for k, v in res:
            t[k] = list(v) if k not in t or v[0] > t[k][0] else t[k]

    def show(tag):
        d, rows = [], []
        for m in SCREENS:
            for k, a in m.ARMS.items():
                per, txt = per_run(m, k, t)
                d += [per] * len(a["seeds"])
                rows.append((m.NAME, k, len(a["seeds"]), per, txt))
        print(f"  PROJECTION{tag} (worst case: every run to its iters; pool-load timing: the median interval between "
              f"optimizer steps, one child per arm at once, two rounds of 4 (the slower of an arm's two timings), 1 thread "
              f"each, this machine):")
        for nm, k, n, per, txt in rows:
            print(f"    {nm:<14} {k:<11} {n:>2} runs x {per / 60:5.1f} min  ({txt})")
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
              + ("D8_PREV_A cut to 260-264" if do else "no cut") + (" (dry run: not stored)" if dry else " (stored)"),
              flush=True)
    else:
        print(f"  RUNTIME RULE (stored {cut['time']}: makespan {cut['makespan_h']:.2f} h on {cut['workers']} workers): "
              + ("D8_PREV_A cut to 260-264" if cut["cut"] else "no cut"), flush=True)
    apply_cut(cut)
    if cut["cut"]:
        show(" AFTER THE CUT")


def run_waves(workers, pv):
    """explore_common2.run_jobs2 (the same stores, worker, records and log lines) with the submission order: the first
    `workers` S40 runs, then S39's, then the rest of S40's."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    stores = {}
    todo = {}
    for m in SCREENS:
        st = ec.load_store(m.NAME)
        st["meta"].setdefault("provenance", pv or ec.provenance())
        st["meta"]["dry"] = c2.DRY["on"]
        st["meta"]["arms"] = {k: {kk: vv for kk, vv in a.items() if kk != "seeds"} | {"seeds": list(a["seeds"]),
                                                                                      "sched": c2.sched(a)}
                              for k, a in m.ARMS.items()}
        ec.save_store(m.NAME, st)
        stores[m.NAME] = st
        todo[m] = sorted((dict(screen=m.__name__, name=m.NAME, arm=k, seed=seed, prio=a.get("prio", 1))
                          for k, a in m.ARMS.items() for seed in a["seeds"] if f"{k}|{seed}" not in st["runs"]),
                         key=lambda j: (j["seed"], j["arm"]))
    jobs = todo[s40][:workers] + todo[s39] + todo[s40][workers:]
    print(f"  {len(jobs)} runs queued on {workers} workers (1 torch thread each), submitted in this order: "
          + " ".join(f"{j['arm']}|{j['seed']}" for j in jobs), flush=True)
    if not jobs:
        return
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"), initializer=ec._worker_init) as ex:
        futs = [ex.submit(c2._worker2, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            job, rec = f.result()
            st = stores[job["name"]]
            st["runs"][f"{job['arm']}|{job['seed']}"] = rec
            ec.save_store(job["name"], st)
            extra = (f"outcome {rec.get('outcome')}" if job["screen"] == s40.__name__
                     else f"reproduces its record {rec.get('reproduces')}")
            print(f"  [{n:>3}/{len(jobs)}] {(time.time() - t0) / 60:6.1f} min  {job['name']:<14} {job['arm']:<10} seed "
                  f"{job['seed']}  lr {rec['lr']:g} ({rec['sched']})\n          "
                  f"{ec.line(rec) if rec.get('ok') else 'FAILED ' + str(rec.get('error'))}  {extra}", flush=True)


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
    pv = ec.print_banner("batch 14: gate_memory (S39), gate_prev (S40)" + dl)
    c = mt.run_child("explore_muon_scale_child", "constants", {})
    assert c["tss_LR"] == c["tsr_LR"] == c["tscur_LR"] == s39.LR == s40.LR, c
    ch = ec.load_store("muon_scale_valid")["meta"].get("choice") or {}
    assert ch.get("a") == ch.get("b") == s39.MUON_LR == s40.MUON_LR, ch
    print(f"  recipe read in the child from main at {c['main_sha'][:7]}: test_slow_start.LR {c['tss_LR']:g}, "
          f"test_stream_recipe.LR {c['tsr_LR']:g}, test_stream_curriculum.LR {c['tscur_LR']:g} (Adam groups; the branch's "
          f"SUB_LR {ec.SUB_LR:g}); WARM {c['tss_WARM']}, TAU {c['tss_TAU']}, LAMBDA {c['tss_LAMBDA']}; Muon lr {s40.MUON_LR:g} "
          f"(S33's validity choice {ch}); S40 28800 steps, S39 reruns to 2400; 1 torch thread per run")
    for m in SCREENS:
        for k, a in m.ARMS.items():
            print(f"  lr read for {m.NAME} {k}: {a['lr']:g} (Adam groups)" + (f"; Muon lr {a['muon_lr']:g}" if a["muon_lr"] else ""))
    print(f"  S40 readings (per arm, BOUND ROUTED at 28800): 'the look-back breaks the eight-stream stall' if >= {s40.BREAK_N}/10; "
          f"'it does not' if {s40.NOT_N}/10; otherwise neither")
    print(f"  S39 labels (per configuration, median decodability of h at key positions): 'formed, then lost' if >= "
          f"{s39.FORMED_MIN} at some evaluation up to {s39.FORMED_BY} and <= {s39.LOST_MAX} at {s39.ITERS}; 'never formed' if "
          f"< {s39.NEVER_MAX} at every evaluation; otherwise 'partial'")
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
                  f"file outside explore_out/ and explore_batch14.py differs from {rec['git']}, where "
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
        run_waves(args.workers, pv)
        print(f"  batch wall clock {(time.time() - t0) / 60:.1f} min (this segment)")
    else:
        print("  REPORT ONLY: from the saved runs (no repro check, CHECKs or runs in this call)")
        apply_cut(ec.load_store(rule_name())["meta"].get("cut"))
    print("#" * 100)
    print(f"REPORTS — {ec.BANNER}{dl}")
    print("#" * 100)
    tagp = "DRY RUN " if args.dry else ""
    d40 = s40.report(ec.load_store(s40.NAME))
    v40 = {}
    for arm in s40.ARMS:
        n, N = completeness(s40, arm)
        v40[arm] = d40[arm]["reading"] if n == N else f"INCOMPLETE, {n}/{N} runs, no reading"
        print(f"  {tagp}READING {s40.NAME} {arm}: {v40[arm]}  [{ec.BANNER}]")
    print()
    d39 = s39.report(ec.load_store(s39.NAME))
    v39 = {}
    for conf in s39.ARMS:
        n, N = completeness(s39, conf)
        v39[conf] = d39[conf]["label"] if n == N else f"INCOMPLETE, {n}/{N} runs, no label"
        print(f"  {tagp}LABEL {s39.NAME} {conf}: {v39[conf]}  [{ec.BANNER}]")
    print()
    print("#" * 100)
    print(f"SUMMARY — {ec.BANNER}{dl}")
    for arm in s40.ARMS:
        dd = d40[arm]
        md = dd["decode_medians"]
        print(f"  S40 gate_prev {arm}: {v40[arm]}; failure classes {dd['classes']}; median decodability of h at key positions "
              + ", ".join(f"{t} {('--' if k is None else f'{k:.2f}')}" for t, (k, _, _, _) in md.items()))
    for conf in s39.ARMS:
        dk = d39[conf]["dec_key"]
        print(f"  S39 gate_memory {conf}: {v39[conf]}; median decodability of h at key positions "
              + ", ".join(f"{t} {('--' if v is None else f'{v:.2f}')}" for t, v in dk.items()))
    print("#" * 100)


if __name__ == "__main__":
    main()
