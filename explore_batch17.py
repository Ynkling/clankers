#!/usr/bin/env python
"""
explore_batch17.py — EXPLORATORY, not a result. Batch 17 of the outside-ideas screens (the user's specification of
7 October 2026). Same harness, rules and labels as batches 1-16 (explore_common16, imported read-only). Everything below
was fixed before any run.

  S48 explore_window_d8     seven one-knob changes to the eight-stream window gate LOCAL3_SLOW_D8_A (3/10) and _M (3/10):
                            W_SPLIT, W_D98, W_SPLIT_D98, W_LONG, W_NOSLOW, W_WIDTH4 (Adam), W_SPLIT_M (Muon); S=8, P=4,
                            k=16, conv, seeds 260-269, BOUND ROUTED; "improves" if >= 6/10, "does not" if <= 3/10
  S49 explore_window_fail   descriptive: S43's nine unbound LOCAL3_SLOW two-stream runs, rerun with the gate measured
                            densely: what the STREAM-PARTIAL gates do at VAL positions; whether KEY splits form before or
                            after update 600

BACKGROUND, ARMS, READINGS, DIAGNOSTICS, CHECKS: in each screen's docstring.

HARNESS (explore_common16): 1 torch thread per run; every segment reproduces X's recorded arm A (seed 160, 2400 steps)
bit for bit (the repro check; a failure stops the segment); Adam arms pair with batch 16's LOCAL3_SLOW_D8_A records
(Adam runs are bit-identical across the CPUs the repro check has passed on; CHECK: W_SPLIT|260 at threshold 0 equals the
record through 6000); the Muon arm pairs only within a CPU (batch 16's LOCAL3_SLOW_D8_M records if this batch's CPU is
theirs, else W_M_REF, run here); Muon jobs never run on a CPU other than the batch's (the CPU of the full CHECK pass);
runs are cached under arm|seed|code SHA|CPU; runs with ok = False are retried (up to 3 attempts).

RECIPE: Adam groups at test_slow_start.LR = 1e-3 (read in a child from main at 9c5939e), the slow phase's non-gate group
at a tenth for updates 1-2400 (W_NOSLOW: 1e-3 throughout); Muon lr 0.005; evaluation every 1200; S49: the branch's
test_channel_binding.SUB_LR 1e-3 (S43's).

RUNTIME: the projection with pool-load timing (one timing child per arm, four at once; worst case: every run to its
iters). If the projected makespan on 4 workers is over 10 h (batch 16's budget), arms are cut one at a time in the user's
order, re-projecting after each, until it is not: W_WIDTH4, W_NOSLOW, W_SPLIT_M (with W_M_REF). Every other arm and S49
are never cut. The decision is stored at the first full segment and kept.

SEGMENTS: --resume (CHECKs skipped while every code SHA is unchanged; the repro check runs in every segment); --budget
(minutes left in this segment); --report-only; --dry (one seed per arm, runs capped at 2400 steps, separate dry_ stores;
the projection is printed for the full plan, nothing stored). A watcher outside this file pushes every saved run.

Run:  python explore_batch17.py [--workers 4] [--dry] [--resume] [--budget MIN] [--report-only]
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
import explore_window_d8 as s48
import explore_window_fail as s49
from test_binding_recipe import makespan
from test_binding_onset import EVAL_EVERY

SCREENS = (s48, s49)
DRY_ITERS = 2400
CHECKS_RECORD = os.path.join(ec.OUT_DIR, "batch17_checks.json")
RUNTIME_STORE = "batch17_runtime"
CUT_H = 10.0
CUT_ORDER = (("window_d8", "W_WIDTH4"), ("window_d8", "W_NOSLOW"), ("window_d8", "W_SPLIT_M"))
BY_NAME = {m.NAME: m for m in SCREENS}
FULL = {}                                                   # the full plan, before any dry change (set in main)


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
    tt = t[(name, arm)]
    ts, te, tm = tt[:3]
    n_ev = a["iters"] // EVAL_EVERY
    cells = getattr(m, "n_cells", lambda k: 0)(arm) * tt[3] if len(tt) > 3 else 0.0
    return a["iters"] * ts + (n_ev + m.n_extra(arm)) * te + m.n_meas(arm) * tm + cells


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
            print(f"    {name:<12} {k:<18} {n:>2} runs x {d / 60:6.1f} min  (step {tt[0] * 1e3:.1f} ms, evaluation {tt[1]:.1f} s, "
                  f"decoder {tt[2]:.1f} s" + (f", gate cells {tt[3]:.2f} s" if len(tt) > 3 else "") + ")")
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
    if dry:
        plan({m.NAME.replace("dry_", ""): m.ARMS for m in SCREENS}, t, set(), workers, show=True, tag=" OF THIS DRY RUN")
    ms0 = plan(FULL, t, set(), workers, show=True, tag=" OF THE FULL BATCH")
    cut, steps, ms = [], [], ms0
    while ms / 3600 > CUT_H and len(cut) < len(CUT_ORDER):
        cut.append(CUT_ORDER[len(cut)])
        ms = plan(FULL, t, set(cut), workers)
        steps.append(dict(cut=list(cut[-1]), makespan_h=ms / 3600))
    dec = dict(cut=[list(c) for c in cut], makespan_h0=ms0 / 3600, steps=steps, makespan_h=ms / 3600, workers=workers,
               cpu=ec.cpu_model(), time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
               timings={f"{a}|{b}": v for (a, b), v in t.items()})
    print(f"  RUNTIME RULE: projected makespan {ms0 / 3600:.2f} h " + (f"> {CUT_H:g} h: cut, re-projecting after each: "
          + "; ".join(f"{s['cut'][0]} {s['cut'][1]} -> {s['makespan_h']:.2f} h" for s in steps) if cut else f"<= {CUT_H:g} h: no cut")
          + (f"; still over {CUT_H:g} h after every listed cut: the other arms and S49 are never cut, so the batch runs at "
             f"{ms / 3600:.2f} h" if ms / 3600 > CUT_H else "")
          + (" (dry run: not stored)" if dry else " (stored)"), flush=True)
    if not dry:
        st["meta"]["cut"] = dec
        st["meta"].setdefault("provenance", ec.provenance())
        ec.save_store(RUNTIME_STORE, st)
    return t, dec


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
    print(f"  CHECK batch 17: the modules served from main are exactly the test modules that differ between this branch and "
          f"{mt.MAIN_SHA[:7]} ({diff}): {'ok' if v else 'FAIL'}", flush=True)
    if bad or not v:
        print(f"  ! CHECK failed in {bad or 'the served modules'}; stopping.")
        sys.exit(1)
    print(f"  all CHECKs passed ({(time.time() - t0) / 60:.1f} min)", flush=True)


def constants():
    c = mt.run_child("explore_muon_scale_child", "constants", {})
    assert c["tss_LR"] == c["tsr_LR"] == c["tscur_LR"] == 1e-3 and c["tss_WARM"] == 2400, c
    k = c16.run_child(s48.CHILD, "constants", {}, main=True)
    assert k["LR"] == 1e-3 and k["LR_WARM"] == 1e-4 and k["MAX_SPLITS"] == 3 and k["GAP"] == 4800 and k["DECAY"] == 0.98, k
    assert ec.SUB_LR == s49.LR == 1e-3
    print(f"  recipe read in children from main at {c['main_sha'][:7]}: test_slow_start.LR {c['tss_LR']:g}, test_stream_recipe.LR "
          f"{c['tsr_LR']:g}, test_stream_curriculum.LR {c['tscur_LR']:g} (Adam groups), WARM {c['tss_WARM']}; the warm lr "
          f"{k['LR_WARM']:g}; the trigger: every {k['EVERY']} from {k['FIRST']} (reference {k['REF_AT']}), acc < {k['ACC_THR']}, "
          f"rise < {k['RISE']}, <= {k['MAX_SPLITS']} splits (S37: {k['S37_MAX_SPLITS']}), gap {k['GAP']}; decay {k['DECAY']}; "
          f"S49: the branch's test_channel_binding.SUB_LR {ec.SUB_LR:g}, test_short_conv.MAX_ITERS {ec.MAX_ITERS}; Muon lr "
          f"{s48.MUON_LR:g}; 1 torch thread per run", flush=True)
    for m in SCREENS:
        for kk, a in m.ARMS.items():
            print(f"  lr for {m.NAME} {kk}: {a['lr']:g}" + (f"; Muon lr {a['muon_lr']:g}" if a.get("muon_lr") else "")
                  + f"; iters {a['iters']}; seeds {ec.fmt_seeds(a['seeds'])}")


def line(j, rec):
    if not rec.get("ok"):
        return ""
    if j["name"].endswith("window_fail"):
        return f"{ec.line(rec)}"
    acc = rec.get("acc")
    s = f"acc {acc:.3f}  trans {str(rec.get('transition')):>5}  outcome {rec.get('outcome')}"
    d = (rec.get("decode") or {}).get("end")
    if d:
        s += f"; decoder at {d.get('step')}: h CTX/KEY/VAL {d['h_ctx']:.2f}/{d['h_key']:.2f}/{d['h_val']:.2f}"
    if "SPLIT" in j["arm"]:
        s += f"; splits {rec.get('splits')} at {[c['step'] for c in rec.get('checks') or [] if c.get('fired')]}"
    return s


def dur_fn(t, cpu):
    """A job's expected duration for packing a segment: 1.05 x the longest wall time of the arm's completed runs on this
    CPU when there is one; else its projected worst case scaled by this CPU's median observed / projected ratio (x 1.10),
    or the projection itself before any run on this CPU."""
    seen = {}
    for m in SCREENS:
        for r in ec.load_store(m.NAME)["runs"].values():
            if r.get("ok") and r.get("cpu") == cpu and r.get("secs_wall"):
                k = (m.NAME, r["arm"])
                seen[k] = max(seen.get(k, 0.0), r["secs_wall"])
    ratios = sorted(w / per_run(nm.replace("dry_", ""), arm, BY_NAME[nm.replace("dry_", "")].ARMS[arm], t)
                    for (nm, arm), w in seen.items() if arm in BY_NAME[nm.replace("dry_", "")].ARMS)
    factor = ratios[len(ratios) // 2] if ratios else None

    def dur(j):
        nm = j["name"].replace("dry_", "")
        w = seen.get((j["name"], j["arm"]))
        if w:
            return 1.05 * w
        p = per_run(nm, j["arm"], BY_NAME[nm].ARMS[j["arm"]], t)
        return 1.10 * factor * p if factor else p
    if factor:
        print(f"  packing: arms without a run on this CPU are sized at 1.10 x {factor:.2f} x their projection (the median "
              f"observed / projected ratio over {len(ratios)} arms run on this CPU)", flush=True)
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
    if args.resume:
        print("#" * 20 + f" RESUME {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC — a new segment (the previous one "
              f"ended: background time limit or container restart); saved runs are reused, missing runs re-run ({ec.BANNER}) "
              + "#" * 20)
    cpu = ec.cpu_model()
    rec = None
    if os.path.exists(CHECKS_RECORD) and not args.dry:
        with open(CHECKS_RECORD) as f:
            rec = json.load(f)
    c16.STATE["batch_cpu"] = rec["cpu"] if rec else cpu
    ref_runs = s48.configure_ref(c16.STATE["batch_cpu"])
    FULL.update({m.NAME: {k: dict(a) for k, a in m.ARMS.items()} for m in SCREENS})
    if args.dry:
        make_dry()
    dl = c2.dry_label()
    pv = ec.print_banner("batch 17: window_d8 (S48), window_fail (S49)" + dl)
    print(f"  S48's Muon pair: " + (f"W_M_REF (LOCAL3_SLOW_D8_M fresh on {c16.STATE['batch_cpu']}; batch 16's records are on "
                                    f"{s48.b16_cpus('LOCAL3_SLOW_D8_M')})" if ref_runs else
                                    f"batch 16's LOCAL3_SLOW_D8_M records (on this batch's CPU {c16.STATE['batch_cpu']})"),
          flush=True)
    shas = code_shas()
    c16.STATE["shas"] = shas
    c16.STATE["other_cpus"] = sorted(set(c16.other_cpus_seen(SCREENS)) | {cpu})
    same_cpu = cpu == c16.STATE["batch_cpu"]
    print(f"  this segment's CPU: {cpu}; the batch's CPU (Muon runs and their pairs): {c16.STATE['batch_cpu']}"
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
    r48 = s48.report()
    for arm, o in r48.items():
        if o.get("reading"):
            print(f"  {tagp}READING {s48.NAME} {arm}: {o['reading']}  [{ec.BANNER}]")
    print()
    r49 = s49.report()
    print()
    print("#" * 100)
    print(f"SUMMARY — {ec.BANNER}{dl}")
    for arm, o in r48.items():
        print(f"  S48 window_d8 {arm}: " + (o["reading"] if o.get("reading") else f"reference, BOUND ROUTED {o['bound_routed']}/{o['n']}")
              + f"; bound seeds {o['bound_seeds']}; classes {o['classes']}")
    if s48.CUT:
        print(f"  S48 cut by the runtime rule: {sorted(s48.CUT)}")
    for k, v in r49["summary"].items():
        print(f"  S49 window_fail: {v} (descriptive)")
    print("#" * 100)


if __name__ == "__main__":
    main()
