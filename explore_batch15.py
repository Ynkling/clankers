#!/usr/bin/env python
"""
explore_batch15.py — EXPLORATORY, not a result. Batch 15 of the outside-ideas screens, with batches 1-14's
harness, rules and labels and their resume machinery. Everything below was fixed before any run.

  S41 explore_gate_cap     CAP (sigma_max(W_h) <= 0.5 after every optimizer step) and NOREC (W_h = 0, frozen: the
                           gate reads GATE_PREV's input only) at S=8, P=4, k=16, conv, all 8 streams from step 1,
                           28800 steps, seeds 260-269: CAP_M (S33's D8_HINGE + CAP, Muon), PREV_CAP_M (S40's
                           D8_PREV_M + CAP), PREV_NOREC_M (S40's D8_PREV_M with NOREC), PREV_CAP_A (S40's D8_PREV_A +
                           CAP, Adam)
  S42 explore_wh_spectrum  descriptive: reruns to 1600 of S33's D8_HINGE (Muon) and X's HINGE_D8 (Adam), seeds
                           260-264, measured every 50 updates: sigma_max(W_h), rho(W_h), the gate's term norms and h's
                           decodability; label "gain crossing" / "not"

BACKGROUND (docstring): see explore_gate_cap and explore_wh_spectrum.

MACHINE: Muon runs (bfloat16 Newton-Schulz) are CPU-specific, Adam runs are not. The first dry run
(explore_out/batch15_dry_check_failed.log) and segment 1 ran on a Xeon @ 2.80GHz, where the recorded Muon runs do not
reproduce; by the user's first decision the second dry run and segment 1 used fresh Muon references (REF arms) and
a per-segment fingerprint. Segment 2's container had the Xeon @ 2.10GHz of batches 1-14 again, where the records
reproduce (D8_HINGE|260 through 1200) and Adam runs equal the 2.80GHz ones (PREV_CAP_A|260); by the user's second
decision batch 15 returned to its pre-registered pairing with the recorded runs, on the 2.10GHz CPU: the REF arms
are dropped; segment 1's eight 2.80GHz Muon runs (CAP_M|260, PREV_CAP_M|260, PREV_NOREC_M|260, S42's D8_HINGE_M
260-264) are moved to explore_out/gate_cap_2p80_results.json and wh_spectrum_2p80_results.json (unused) and its six
Adam runs are kept; the runtime rule is applied again to this plan on this CPU (stored in gate_cap_runtime_2p10;
segment 1's 2.80GHz decision stays in gate_cap_runtime); and a MUON RECORD CHECK runs in every segment: S33's
D8_HINGE|260 (S41's REF_HINGE_M arm, the same recipe) through 1200 must reproduce its record (curve and statistics),
or the batch stops (another CPU).

RECIPE: the Adam groups at test_slow_start.LR = test_stream_recipe.LR = test_stream_curriculum.LR = 1e-3 (read
in the child; the branch's SUB_LR = 1e-3), passed explicitly; Muon lr 0.005 (S33's validity choice); the slow
phase's non-gate groups at a tenth for updates 1-2400; evaluation every 1200; the run paths' own early stops;
main's run paths at 9c5939e in child processes; 1 torch thread per run.

READINGS / LABELS: in explore_gate_cap (per arm, BOUND ROUTED at 28800) and explore_wh_spectrum (per
configuration). An incomplete arm or configuration gets no reading.

CHECKS (before any run, after the repro check; any failure stops the batch)
- repro: X's arm A, seed 160, 2400 steps, bit-identical to X's record (in every segment).
- Muon record check: S33's D8_HINGE|260 through 1200 bit-identical to its record (in every segment).
- S41: with the cap at infinity, CAP_M and PREV_CAP_M equal their paired recorded runs bit for bit through 1200, and
  PREV_CAP_A equals S40's recorded D8_PREV_A through 1200; over 50 steps of each CAP arm sigma_max(W_h) <= 0.5 + 1e-6
  after every step; NOREC: W_h exactly 0 at every step and no update, every other parameter's initial value equals
  D8_PREV_M's; the optimizer groups cover every trainable parameter once; S38's decoder check.
- S42: one rerun per configuration through 1600 reproduces its record (and every S42 run is checked through 1600);
  rho and sigma_max agree with numpy on a random matrix to 1e-6.
- The served modules.

SEQUENCE: repro check; CHECKs; the projection with pool-load timing (batch 10's method: one timing child per arm
or configuration at once, two rounds of 4; the median interval between optimizer steps); RUNTIME RULE: if the
projected worst-case makespan is over 9 h, PREV_CAP_A is cut to 260-264 (stored); then one pool, submitted as: the
first `workers` S41 runs, then S42's, then the rest of S41's.
SEGMENTS: batches 4-14's machinery (--resume against explore_out/batch15_checks.json, --report-only,
--stopped-early, --dry); a watcher outside this file pushes every saved run.

Run:  python explore_batch15.py [--workers 4] [--dry] [--resume] [--report-only]
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
import explore_gate_cap as s41
import explore_wh_spectrum as s42
from test_binding_recipe import makespan
from test_binding_onset import EVAL_EVERY

SCREENS = (s41, s42)
DRY_ITERS = 2400
CHECKS_RECORD = os.path.join(ec.OUT_DIR, "batch15_checks.json")
CUT_H = 9.0
CUT_ARMS = ("PREV_CAP_A",)


def git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True, check=True).stdout


def code_changes(rev):
    files = git("diff", "--name-only", rev).split() + git("ls-files", "--others", "--exclude-standard").split()
    return sorted(f for f in set(files) if not f.startswith("explore_out/") and f != "explore_batch15.py")


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
    print(f"  CHECK batch 15: the modules served from main are exactly the test modules that differ between this branch "
          f"and {mt.MAIN_SHA[:7]} ({diff}): {'ok' if v else 'FAIL'}", flush=True)
    if not v:
        print("  ! CHECK failed; stopping.")
        sys.exit(1)
    print(f"  all CHECKs passed ({(time.time() - t0) / 60:.1f} min)", flush=True)


def muon_record_check():
    """S33's D8_HINGE|260 (S41's REF_HINGE_M arm: the same recipe, no cap) through 1200 on this machine, against its
    record (curve and statistics). False on a mismatch (another CPU)."""
    t0 = time.time()
    r = s41.child("run", dict(arm="REF_HINGE_M", seed=260, iters=s41.EQ_AT, mlr=s41.MUON_LR))
    ref = ec.load_store("muon_scale")["runs"]["D8_HINGE|260"]
    ok, want, steps = s41.same_upto(r, ref, s41.EQ_AT)
    print(f"  muon_record_check: S33's D8_HINGE|260 (Muon) through {s41.EQ_AT} on {ec.cpu_model()}: "
          + ("BIT-IDENTICAL" if ok else "NOT bit-identical") + f" to its record (curve {r['curve']} vs {want}; statistics at "
          f"{steps}) ({time.time() - t0:.0f}s)", flush=True)
    return ok


def rule_name():
    return s41.NAME + "_runtime_2p10"


def apply_cut(cut):
    if cut and cut.get("cut"):
        for k in CUT_ARMS:
            s41.ARMS[k]["seeds"] = tuple(s for s in s41.ARMS[k]["seeds"] if s in s41.CUT_SEEDS)


def per_run(m, k, t):
    a = m.ARMS[k]
    ts, te, tm = t[k][:3]
    n_ev = a["iters"] // EVAL_EVERY
    n_m = len(s41.DIAG) if m is s41 else len([x for x in s42.AT if x <= a["iters"]])
    return (a["iters"] * ts + n_ev * te + n_m * tm,
            f"{a['iters']} steps x {ts * 1e3:.1f} ms + {n_ev} evals x {te:.1f} s + {n_m} measurements x {tm:.1f} s")


def projection(workers, dry):
    jobs1 = [(s41, "CAP_M"), (s41, "PREV_CAP_M"), (s41, "PREV_NOREC_M"), (s41, "PREV_CAP_A")]
    jobs2 = [(s42, "D8_HINGE_M"), (s42, "HINGE_D8_A"), (s41, "CAP_M"), (s41, "PREV_CAP_A")]
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
              f"optimizer steps, one child per arm or configuration at once, two rounds of 4 (the slower of an arm's two "
              f"timings), 1 thread each, this machine):")
        for nm, k, n, per, txt in rows:
            print(f"    {nm:<14} {k:<12} {n:>2} runs x {per / 60:5.1f} min  ({txt})")
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
              + ("PREV_CAP_A cut to 260-264" if do else "no cut") + (" (dry run: not stored)" if dry else " (stored)"),
              flush=True)
    else:
        print(f"  RUNTIME RULE (stored {cut['time']}: makespan {cut['makespan_h']:.2f} h on {cut['workers']} workers): "
              + ("PREV_CAP_A cut to 260-264" if cut["cut"] else "no cut"), flush=True)
    apply_cut(cut)
    if cut["cut"]:
        show(" AFTER THE CUT")


def run_waves(workers, pv):
    """explore_common2.run_jobs2 (the same stores, worker, records and log lines) with the submission order: the first
    `workers` S41 runs, then S42's, then the rest of S41's."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    stores, todo = {}, {}
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
    jobs = todo[s41][:workers] + todo[s42] + todo[s41][workers:]
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
            if job["screen"] == s41.__name__:
                s0 = rec.get("sigma0")
                extra = (f"outcome {rec.get('outcome')}; sigma_max(W_h) at init {'--' if s0 is None else f'{s0:.4f}'}, the cap "
                         f"first at update {rec.get('first_cap')}, on {rec.get('n_cap')} updates")
            elif "reproduces_record" in rec:
                extra = (f"reproduces the 2.10GHz record {rec['reproduces_record']} (a fresh run on this CPU; checked against "
                         f"REF_HINGE_M in the report)")
            else:
                extra = f"reproduces its record {rec.get('reproduces')}"
            print(f"  [{n:>3}/{len(jobs)}] {(time.time() - t0) / 60:6.1f} min  {job['name']:<14} {job['arm']:<12} seed "
                  f"{job['seed']}  lr {rec['lr']:g} ({rec['sched']})\n          "
                  f"{ec.line(rec) if rec.get('ok') else 'FAILED ' + str(rec.get('error'))}  {extra}", flush=True)


def fmt2(x):
    return "--" if x is None else f"{x:.2f}"


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
    pv = ec.print_banner("batch 15: gate_cap (S41), wh_spectrum (S42)" + dl)
    c = mt.run_child("explore_muon_scale_child", "constants", {})
    assert c["tss_LR"] == c["tsr_LR"] == c["tscur_LR"] == s41.LR == s42.LR, c
    ch = ec.load_store("muon_scale_valid")["meta"].get("choice") or {}
    assert ch.get("a") == ch.get("b") == s41.MUON_LR == s42.MUON_LR, ch
    print(f"  recipe read in the child from main at {c['main_sha'][:7]}: test_slow_start.LR {c['tss_LR']:g}, "
          f"test_stream_recipe.LR {c['tsr_LR']:g}, test_stream_curriculum.LR {c['tscur_LR']:g} (Adam groups; the branch's "
          f"SUB_LR {ec.SUB_LR:g}); WARM {c['tss_WARM']}, TAU {c['tss_TAU']}, LAMBDA {c['tss_LAMBDA']}; Muon lr {s41.MUON_LR:g} "
          f"(S33's validity choice {ch}); S41 28800 steps, the cap at sigma_max(W_h) 0.5; S42 reruns to 1600; 1 torch thread "
          f"per run")
    for m in SCREENS:
        for k, a in m.ARMS.items():
            print(f"  lr read for {m.NAME} {k}: {a['lr']:g} (Adam groups)" + (f"; Muon lr {a['muon_lr']:g}" if a["muon_lr"] else ""))
    print(f"  S41 readings (per arm, BOUND ROUTED at 28800): 'it breaks the eight-stream stall' if >= {s41.BREAK_N}/10; "
          f"'it does not' if {s41.NOT_N}/10; otherwise neither")
    print(f"  S42 labels (per configuration): 'gain crossing' if in >= {s42.MIN_RUNS}/5 runs rho(W_h) first exceeds 1 within "
          f"{s42.WINDOW} updates before |W_h h|@key first exceeds 1; otherwise 'not'")
    if not args.report_only:
        if not ec.repro_check():
            print("  ! this container does not reproduce X: the recorded pairing is invalid; stopping.")
            sys.exit(1)
        if not muon_record_check():
            print("  ! the Muon runs do not reproduce their records on this machine (another CPU): the pairing with the "
                  "recorded Muon runs would not hold; stopping.")
            sys.exit(1)
        rec = None
        if args.resume and os.path.exists(CHECKS_RECORD):
            with open(CHECKS_RECORD) as f:
                rec = json.load(f)
        changed = code_changes(rec["git"]) if rec else None
        if rec and not rec["code_changed"] and not changed:
            print(f"  CHECKs skipped (repro check above still run): the screen code is unchanged — no "
                  f"file outside explore_out/ and explore_batch15.py differs from {rec['git']}, where "
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
    d41 = s41.report(ec.load_store(s41.NAME))
    v41 = {}
    for arm in s41.ARMS:
        n, N = completeness(s41, arm)
        v41[arm] = d41[arm]["reading"] if n == N else f"INCOMPLETE, {n}/{N} runs, no reading"
        print(f"  {tagp}{'REFERENCE' if arm in s41.REF_ARMS else 'READING'} {s41.NAME} {arm}: {v41[arm]}  [{ec.BANNER}]")
    print()
    d42 = s42.report(ec.load_store(s42.NAME))
    v42 = {}
    for conf in s42.ARMS:
        n, N = completeness(s42, conf)
        v42[conf] = d42[conf]["label"] if n == N else f"INCOMPLETE, {n}/{N} runs, no label"
        print(f"  {tagp}LABEL {s42.NAME} {conf}: {v42[conf]}  [{ec.BANNER}]")
    print()
    print("#" * 100)
    print(f"SUMMARY — {ec.BANNER}{dl}")
    for arm in s41.ARMS:
        dd = d41[arm]
        md = dd["medians"]
        print(f"  S41 gate_cap {arm}: {v41[arm]}; failure classes {dd['classes']}; median decodability of h at key positions "
              + ", ".join(f"{t} {fmt2(md[t]['h_key'])}" for t in s41.DIAG))
    for conf in s42.ARMS:
        dd = d42[conf]
        print(f"  S42 wh_spectrum {conf}: {v42[conf]}; per run (first rho > 1, first |W_h h|@key > 1): "
              + ", ".join(f"{s} ({c['rho']}, {c['whh']})" for s, c in sorted(dd["cross"].items())))
    print("#" * 100)


if __name__ == "__main__":
    main()
