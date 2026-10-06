#!/usr/bin/env python
"""
explore_batch16.py — EXPLORATORY, not a result. Batch 16 of the outside-ideas screens: the specification in
docs/audit_2026-10.md, Appendix A (branch claude/bdh-growth-hebbian-inference-w90069, the audit of 6 October 2026), with
the user's two changes (S44(b) gains LOCAL3_SLOW_D8_A_SPLIT; batch 15's README paragraph on NOREC is rewritten). Same
harness, rules and labels as batches 1-15. Everything below was fixed before any run.

  S43 explore_window_recipe   the width-3 window gate + S5's slow memory, no hinge (S=2, P=4, k=2, no conv, 24000)
  S44 explore_window_scale    the window gate + slow memory at four (k=16) and eight streams (conv, 28800; BOUND ROUTED)
  S45 explore_reservoir_gate  a frozen reservoir gate (W_h at spectral radius 0.5) at eight streams
  S46 explore_stream_knobs    descriptive: eight-stream lr x0.3 / x3, W_h at lr/10, decay 0.98, the perfect gate at 0.98
  S47 explore_write_decoders  descriptive: S38's decoder at CTX, KEY and VAL positions on recorded configurations

BACKGROUND, ARMS, READINGS, DIAGNOSTICS, CHECKS: in each screen's docstring.

HARNESS (explore_common16): 1 torch thread per run; pairing only within a CPU: every segment reproduces X's recorded
arm A (seed 160, 2400 steps) bit for bit (the repro check; a failure stops the segment); Muon arms pair with fresh Muon
references run on this batch's CPU (the CPU of the full CHECK pass; Muon jobs never run on another CPU); runs are cached
under arm|seed|code SHA|CPU (the code SHA covers every repository module the run's child imports); runs with ok = False
are retried (up to 3 attempts).

RECIPE: Adam groups at test_slow_start.LR = 1e-3 (read in a child from main at 9c5939e; S43: the branch's SUB_LR 1e-3),
the slow phase's non-gate groups at a tenth for updates 1-2400, Muon lr 0.005 (S33's validity choice); evaluation every
1200; main's run paths at 9c5939e in child processes (S44-S47), the branch's own (S43).

RUNTIME: the projection with pool-load timing (one timing child per arm, four at once; the median interval between
optimizer steps, one evaluation with statistics, one decoder measurement; worst case: every run to its iters). If the
projected makespan on 4 workers is over 10 h, arms are cut one at a time in this order, re-projecting after each, until
it is not: S46's LR3E4, S45's RES_D8_M, S44's LOCAL3_SLOW_D8_A_SPLIT (the user: before any other S44 arm), S44's
LOCAL3_SLOW16_M. A reference arm runs only while an arm it pairs with runs. S43, S44(b)'s Adam arm and S47 are never cut.
The decision is stored at the first full segment and kept.

SEGMENTS: --resume (the full CHECK pass is recorded in explore_out/batch16_checks.json with each screen's code SHA and the
batch's CPU; later segments skip the CHECKs while every code SHA is unchanged; the repro check runs in every segment);
--budget (minutes left in this segment: a freed worker takes the longest job that fits); --report-only; --dry (one seed
per arm, runs capped at 2400 steps, separate dry_ stores; the projection is printed for the full plan, nothing stored).
A watcher outside this file pushes every saved run.

Run:  python explore_batch16.py [--workers 4] [--dry] [--resume] [--budget MIN] [--report-only]
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common2 as c2
import explore_common16 as c16
import explore_main9c as mt
import explore_window_recipe as s43
import explore_window_scale as s44
import explore_reservoir_gate as s45
import explore_stream_knobs as s46
import explore_write_decoders as s47
from test_binding_recipe import makespan
from test_binding_onset import EVAL_EVERY

SCREENS = (s43, s44, s45, s46, s47)
DRY_ITERS = 2400
CHECKS_RECORD = os.path.join(ec.OUT_DIR, "batch16_checks.json")
RUNTIME_STORE = "batch16_runtime"
CUT_H = 10.0
CUT_ORDER = (("stream_knobs", "LR3E4"), ("reservoir_gate", "RES_D8_M"), ("window_scale", "LOCAL3_SLOW_D8_A_SPLIT"),
             ("window_scale", "LOCAL3_SLOW16_M"))
FULL = {m.NAME: {k: dict(a) for k, a in m.ARMS.items()} for m in SCREENS}       # the full plan, before any dry change
BY_NAME = {m.NAME: m for m in SCREENS}


def make_dry():
    c2.DRY.update(on=True, iters=DRY_ITERS)
    for m in SCREENS:
        m.NAME = "dry_" + m.NAME
        m.ARMS = {k: dict(a, iters=min(a["iters"], DRY_ITERS), seeds=tuple(a["seeds"][:1])) for k, a in m.ARMS.items()}


# ── Projection and the runtime rule ─────────────────────────────────────────
def timings(workers):
    jobs = [(m, k) for m in SCREENS for k in m.ARMS]
    t = {}
    for i in range(0, len(jobs), workers):
        rnd = jobs[i:i + workers]
        with ThreadPoolExecutor(max_workers=len(rnd)) as ex:
            res = list(ex.map(lambda j: (j[0].NAME.replace("dry_", ""), j[1], j[0].timing([j[1]])[j[1]]), rnd))
        for nm, k, v in res:
            t[(nm, k)] = v
    return t


def per_run(name, arm, a, t):
    m = BY_NAME[name]
    ts, te, tm = t[(name, arm)][:3]
    n_ev = a["iters"] // EVAL_EVERY
    return a["iters"] * ts + (n_ev + getattr(m, "n_extra", lambda k: 0)(arm)) * te + m.n_meas(arm) * tm


def needed(name, arm, a, cut):
    nb = a.get("needed_by")
    return (name, arm) not in cut and (nb is None or any(x not in cut for x in nb))


def plan(arms_of, t, cut, workers, show=False, tag=""):
    durs, rows = [], []
    for name, arms in arms_of.items():
        for k, a in arms.items():
            if not needed(name, k, a, cut):
                continue
            d = per_run(name, k, a, t)
            durs += [d] * len(a["seeds"])
            rows.append((name, k, len(a["seeds"]), d, t[(name, k)]))
    ms = makespan(durs, workers)
    if show:
        print(f"  PROJECTION{tag} (worst case: every run to its iters; pool-load timing, 1 thread each, this machine):")
        for name, k, n, d, tt in rows:
            print(f"    {name:<15} {k:<24} {n:>2} runs x {d / 60:6.1f} min  (step {tt[0] * 1e3:.1f} ms, evaluation {tt[1]:.1f} s, "
                  f"decoder {tt[2]:.1f} s)")
        print(f"    total {sum(durs) / 3600:.2f} h of runs; makespan on {workers} workers {ms / 3600:.2f} h (worst case)", flush=True)
    return ms


def runtime_rule(workers, dry):
    st = ec.load_store(RUNTIME_STORE)
    stored = st["meta"].get("cut")
    if stored is not None and stored.get("timings") and not dry:
        t = {tuple(k.split("|")): v for k, v in stored["timings"].items()}
        print(f"  RUNTIME RULE (stored {stored['time']} on {stored['cpu']}: makespan {stored['makespan_h0']:.2f} h on "
              f"{stored['workers']} workers): cut {[tuple(c) for c in stored['cut']] or 'nothing'} -> {stored['makespan_h']:.2f} h "
              f"(the stored timings are reused)", flush=True)
        return t, stored
    t = timings(workers)
    full = {nm: arms for nm, arms in FULL.items()}
    if dry:
        plan({m.NAME.replace("dry_", ""): m.ARMS for m in SCREENS}, t, set(), workers, show=True, tag=" OF THIS DRY RUN")
    ms0 = plan(full, t, set(), workers, show=True, tag=" OF THE FULL BATCH")
    if stored is None:
        cut, steps, ms = [], [], ms0
        while ms / 3600 > CUT_H and len(cut) < len(CUT_ORDER):
            cut.append(CUT_ORDER[len(cut)])
            ms = plan(full, t, set(cut), workers)
            steps.append(dict(cut=list(cut[-1]), makespan_h=ms / 3600))
        dec = dict(cut=[list(c) for c in cut], makespan_h0=ms0 / 3600, steps=steps, makespan_h=ms / 3600, workers=workers,
                   cpu=ec.cpu_model(), time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                   timings={f"{a}|{b}": v for (a, b), v in t.items()})
        print(f"  RUNTIME RULE: projected makespan {ms0 / 3600:.2f} h " + (f"> {CUT_H:g} h: cut, re-projecting after each: "
              + "; ".join(f"{s['cut'][0]} {s['cut'][1]} -> {s['makespan_h']:.2f} h" for s in steps) if cut else f"<= {CUT_H:g} h: no cut")
              + (f"; still over {CUT_H:g} h after every listed cut: S43, S44(b)'s Adam arm and S47 are never cut, so the batch "
                 f"runs at {ms / 3600:.2f} h" if ms / 3600 > CUT_H else "")
              + (" (dry run: not stored)" if dry else " (stored)"), flush=True)
        if not dry:
            st["meta"]["cut"] = dec
            st["meta"].setdefault("provenance", ec.provenance())
            ec.save_store(RUNTIME_STORE, st)
        stored = dec
    else:
        print(f"  RUNTIME RULE (stored {stored['time']} on {stored['cpu']}: makespan {stored['makespan_h0']:.2f} h on "
              f"{stored['workers']} workers): cut {[tuple(c) for c in stored['cut']] or 'nothing'} -> {stored['makespan_h']:.2f} h",
              flush=True)
    return t, stored


def apply_cut(dec):
    cut = {tuple(c) for c in (dec or {}).get("cut", [])}
    for m in SCREENS:
        nm = m.NAME.replace("dry_", "")
        for k in list(m.ARMS):
            if not needed(nm, k, FULL[nm][k], cut):
                del m.ARMS[k]
                m.CUT.add(k)
    if cut:
        print(f"  arms cut: {sorted(f'{a}:{b}' for a, b in cut)}; references dropped with them: "
              + str(sorted(f"{m.NAME}:{k}" for m in SCREENS for k in m.CUT if (m.NAME.replace('dry_', ''), k) not in cut)), flush=True)


# ── Code SHAs, CHECKs ────────────────────────────────────────────────────────
def code_shas():
    out = {}
    with ThreadPoolExecutor(max_workers=len(SCREENS)) as ex:
        res = dict(zip([m.NAME for m in SCREENS], ex.map(lambda m: c16.code_sha(m.CHILD, m.MAIN), SCREENS)))
    for m in SCREENS:
        sha, rows = res[m.NAME]
        out[m.NAME] = sha
        print(f"  code SHA {m.NAME}: {sha[:12]} over {len(rows)} modules ({sum(1 for r in rows if r[2] != 'branch')} served from "
              f"main at {mt.MAIN_SHA[:7]}): " + " ".join(r[0] for r in rows), flush=True)
    return out


def run_checks():
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(SCREENS)) as ex:
        res = list(ex.map(lambda m: (m.NAME, m.check()), SCREENS))
    bad = [nm for nm, ok in res if not ok]
    diff = mt.differing()
    v = diff == sorted(mt.SERVED)
    print(f"  CHECK batch 16: the modules served from main are exactly the test modules that differ between this branch and "
          f"{mt.MAIN_SHA[:7]} ({diff}): {'ok' if v else 'FAIL'}", flush=True)
    if bad or not v:
        print(f"  ! CHECK failed in {bad or 'the served modules'}; stopping.")
        sys.exit(1)
    print(f"  all CHECKs passed ({(time.time() - t0) / 60:.1f} min)", flush=True)


def constants():
    c = mt.run_child("explore_muon_scale_child", "constants", {})
    assert c["tss_LR"] == c["tsr_LR"] == c["tscur_LR"] == 1e-3 and c["tss_WARM"] == 2400 and c["tss_TAU"] == 0.2, c
    assert ec.SUB_LR == s43.LR == 1e-3
    print(f"  recipe read in a child from main at {c['main_sha'][:7]}: test_slow_start.LR {c['tss_LR']:g}, test_stream_recipe.LR "
          f"{c['tsr_LR']:g}, test_stream_curriculum.LR {c['tscur_LR']:g} (Adam groups), WARM {c['tss_WARM']}, TAU {c['tss_TAU']}, "
          f"LAMBDA {c['tss_LAMBDA']}; S43: the branch's test_channel_binding.SUB_LR {ec.SUB_LR:g}, test_short_conv.MAX_ITERS "
          f"{ec.MAX_ITERS}; Muon lr {s44.MUON_LR:g}; 1 torch thread per run", flush=True)
    for m in SCREENS:
        for k, a in m.ARMS.items():
            print(f"  lr for {m.NAME} {k}: {a['lr']:g}" + (f"; Muon lr {a['muon_lr']:g}" if a.get("muon_lr") else "")
                  + f"; iters {a['iters']}; seeds {ec.fmt_seeds(a['seeds'])}")


def line(j, rec):
    if not rec.get("ok"):
        return ""
    if j["name"].endswith("window_recipe"):
        return f"{ec.line(rec)}"
    acc = rec.get("acc")
    s = f"acc {acc:.3f}  trans {str(rec.get('transition')):>5}  outcome {rec.get('outcome')}"
    d = rec.get("decode") or {}
    last = max((k for k in d if k.isdigit()), key=int, default=None)
    if last is not None:
        s += f"; decoder at {last}: h CTX/KEY/VAL {d[last]['h_ctx']:.2f}/{d[last]['h_key']:.2f}/{d[last]['h_val']:.2f}"
    if rec.get("splits") is not None and j["arm"].endswith("SPLIT"):
        s += f"; splits {rec['splits']}"
    return s


def dur_fn(t, cpu):
    """A job's expected duration for packing a segment: 1.05 x the longest wall time of the arm's completed runs on this
    CPU when there is one (the eight-stream runs go the distance), else the projection's worst case."""
    seen = {}
    for m in SCREENS:
        for r in ec.load_store(m.NAME)["runs"].values():
            if r.get("ok") and r.get("cpu") == cpu and r.get("secs_wall"):
                k = (m.NAME, r["arm"])
                seen[k] = max(seen.get(k, 0.0), r["secs_wall"])

    def dur(j):
        nm = j["name"].replace("dry_", "")
        w = seen.get((j["name"], j["arm"]))
        return 1.05 * w if w else per_run(nm, j["arm"], BY_NAME[nm].ARMS[j["arm"]], t)
    return dur


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=ec.WORKERS)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--budget", type=float, default=None, help="minutes left in this segment")
    args = ap.parse_args()
    t_start = time.time()
    if args.dry:
        make_dry()
    dl = c2.dry_label()
    if args.resume:
        print("#" * 20 + f" RESUME {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC — the previous segment was stopped by "
              f"the session's background time limit; saved runs are reused, missing runs re-run ({ec.BANNER}) " + "#" * 20)
    pv = ec.print_banner("batch 16: window_recipe (S43), window_scale (S44), reservoir_gate (S45), stream_knobs (S46), "
                         "write_decoders (S47)" + dl)
    cpu = ec.cpu_model()
    rec = None
    if os.path.exists(CHECKS_RECORD) and not args.dry:
        with open(CHECKS_RECORD) as f:
            rec = json.load(f)
    c16.STATE["batch_cpu"] = rec["cpu"] if rec else cpu
    shas = code_shas()
    c16.STATE["shas"] = shas
    c16.STATE["other_cpus"] = sorted(set(c16.other_cpus_seen(SCREENS)) | {cpu})
    same_cpu = cpu == c16.STATE["batch_cpu"]
    print(f"  this segment's CPU: {cpu}; the batch's CPU (Muon runs and their references): {c16.STATE['batch_cpu']}"
          + ("" if same_cpu else " — ANOTHER CPU: only Adam runs run in this segment (each accepted once this CPU passes the "
                                 "repro check)"), flush=True)
    if args.report_only:
        print("  REPORT ONLY: from the saved runs (no repro check, CHECKs or runs in this call)")
        apply_cut(ec.load_store(RUNTIME_STORE)["meta"].get("cut"))
        constants()
    else:
        if not ec.repro_check():
            print("  ! this container does not reproduce X: the recorded pairing is invalid; stopping.")
            sys.exit(1)
        t, dec = runtime_rule(args.workers, args.dry)
        if args.dry:
            print("  dry run: the runtime rule's decision is printed, not applied (every arm is exercised)", flush=True)
        else:
            apply_cut(dec)
        constants()
        if rec and rec.get("shas") == shas and not args.dry:
            print(f"  CHECKs skipped (repro check above still run): every screen's code SHA equals the one at the full CHECK pass "
                  f"({rec['time']}, git {rec['git']}, on {rec['cpu']}; its output is earlier in this log)")
        else:
            if rec:
                print(f"  running every CHECK (code SHAs at the full pass {rec.get('shas')} vs now {shas})")
            run_checks()
            if not args.dry:
                with open(CHECKS_RECORD, "w") as f:
                    json.dump(dict(git=ec.git_head(), shas=shas, cpu=c16.STATE["batch_cpu"], banner=ec.BANNER,
                                   time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())), f, indent=1)
                print(f"  recorded the full CHECK pass (code SHAs, the batch's CPU {c16.STATE['batch_cpu']}) in {CHECKS_RECORD}")
        for m in SCREENS:
            c2.print_screen_header2(m)
            st = ec.load_store(m.NAME)
            st["meta"].setdefault("provenance", pv)
            st["meta"]["dry"] = args.dry
            st["meta"]["arms"] = {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in a.items()}
                                  for k, a in m.ARMS.items()}
            st["meta"].setdefault("code_shas", [])
            if shas[m.NAME] not in st["meta"]["code_shas"]:
                st["meta"]["code_shas"].append(shas[m.NAME])
            st["meta"]["cut"] = sorted(m.CUT)
            ec.save_store(m.NAME, st)
        jobs, skipped = c16.queue(SCREENS, cpu, dry=args.dry)
        if skipped:
            print(f"  not run in this segment: {skipped}")
        budget = None if args.budget is None else args.budget * 60 - (time.time() - t_start)
        print(f"  {len(jobs)} runs queued on {args.workers} workers (1 torch thread each)"
              + (f"; segment budget {budget / 60:.0f} min left" if budget else ""), flush=True)
        t0 = time.time()
        c16.run_pool(jobs, args.workers, dur_fn(t, cpu), budget, line)
        print(f"  batch wall clock {(time.time() - t0) / 60:.1f} min (this segment)")
    print("#" * 100)
    print(f"REPORTS — {ec.BANNER}{dl}")
    print("#" * 100)
    tagp = "DRY RUN " if args.dry else ""
    r43 = s43.report()
    print(f"  {tagp}READING {s43.NAME}: {r43['reading']}  [{ec.BANNER}]\n")
    r44 = s44.report()
    print(f"  {tagp}READING {s44.NAME} (a): {r44['a']}  [{ec.BANNER}]")
    print(f"  {tagp}READING {s44.NAME} (b): {r44['b']}  [{ec.BANNER}]")
    if r44["split"] is not None:
        print(f"  {tagp}READING {s44.NAME} (b) LOCAL3_SLOW_D8_A_SPLIT: {r44['split']}  [{ec.BANNER}]")
    print()
    r45 = s45.report()
    print(f"  {tagp}READING {s45.NAME}: {r45['reading']}  [{ec.BANNER}]\n")
    r46 = s46.report()
    print()
    r47 = s47.report()
    print()
    print("#" * 100)
    print(f"SUMMARY — {ec.BANNER}{dl}")
    print(f"  S43 window_recipe: {r43['reading']}; vs SLOW_MEM: LOCAL3_SLOW only {r43['vs_sm']['b']}, SLOW_MEM only "
          f"{r43['vs_sm']['c']}, p = {r43['vs_sm']['p']:.3g}; LOCAL3_SLOW failure classes {r43['fails']}")
    print(f"  S44 window_scale (a): {r44['a']}")
    print(f"  S44 window_scale (b): {r44['b']}")
    if r44["split"] is not None:
        print(f"  S44 window_scale (b) SPLIT: {r44['split']}")
    print(f"  S45 reservoir_gate: {r45['reading']}")
    for arm, o in r46.items():
        print(f"  S46 stream_knobs {arm}: bound {o['bound']}/{o['n']}, BOUND ROUTED {o['bound_routed']}/{o['n']}, median final "
              f"accuracy {c16.fmt2(o['acc'])}; classes {o['classes']} (descriptive)")
    for arm, o in r47.items():
        m = o["meds"]
        print(f"  S47 write_decoders {arm}: median h decodability CTX/KEY/VAL at 4800 "
              f"{c16.fmt2(m.get(('4800', 'h_ctx')))}/{c16.fmt2(m.get(('4800', 'h_key')))}/{c16.fmt2(m.get(('4800', 'h_val')))}"
              f" (at 0: {c16.fmt2(m.get(('0', 'h_ctx')))}/{c16.fmt2(m.get(('0', 'h_key')))}/{c16.fmt2(m.get(('0', 'h_val')))});"
              f" reproduces its record: {sum(1 for v in o['reproduces'].values() if v)}/{o['n']} (descriptive)")
    print("#" * 100)


if __name__ == "__main__":
    main()
