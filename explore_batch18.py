#!/usr/bin/env python
"""
explore_batch18.py — EXPLORATORY, not a result. Batch 18 of the outside-ideas screens (the user's specification of
8 October 2026). Same harness, rules and labels as batches 1-17 (explore_common16, imported read-only). Everything below
was fixed before any run.

  S50 explore_recipe_ablation  the window recipe at eight streams, ablated: the reset-only control (W_RESET), no slow phase
                               (W_SPLIT_NOSLOW), width 4 (W_SPLIT_W4), Muon (W_SPLIT_M); against batch 17's W_SPLIT (9/10)
  S51 explore_split_k4         the recipe and its split at the old four-stream merge configuration (S=4, P=4, k=4)
  S52 explore_keysplits        the two-stream key splits: two spare channels (W2_K4), the hinge's key term early (W2_KEYHINGE)

BACKGROUND, DEFINITIONS, ARMS, READINGS, DIAGNOSTICS, CHECKS: in each screen's docstring.

HARNESS (explore_common16): 1 torch thread per run; every segment reproduces X's recorded arm A (seed 160, 2400 steps) bit
for bit (the repro check; a failure stops the segment); Adam runs pair with records from either Xeon (bit-identical through
the first divergence, as batch 17 showed); Muon runs pair only with records made on this batch's CPU (the CPU of the full
CHECK pass), never run on another CPU, and run first; runs are cached under arm|seed|code SHA|CPU; runs with ok = False
are retried (up to 3 attempts).

S50'S MUON REFERENCE (before the projection, in the first segment; stored and kept): batch 16's LOCAL3_SLOW_D8_M|260 rerun
here through 1200; if it reproduces its record (curve, statistics, decoder), batch 16's records are W_SPLIT_M's reference,
otherwise W_M_REF (LOCAL3_SLOW_D8_M fresh on this CPU) runs as its pair.

RUNTIME: the projection with pool-load timing (worst case: every run to its iters). "Fits" = a simulation of this pool's
own segment packing (Muon first, then longest first; a freed worker takes the first job that fits in 0.97 x the time left;
nothing starts that does not fit) on the projected durations needs at most 4 segments: the rest of the first (its budget
minus the time already used and a 30 min allowance for the CHECK pass) and three more (each its budget minus 3 min). If it
does not fit, cut in the user's order, re-simulating after each, until it does: S50's W_SPLIT_W4; S51's W4k4 (W4k4_SPLIT
kept); S50's W_SPLIT_NOSLOW; S52's W2_KEYHINGE to seeds 160-179. Never cut: W_RESET, W_SPLIT_M (+ its reference),
W4k4_SPLIT, W2_K4. If it still does not fit, the rest runs on. The decision is stored at the first full segment and kept.

SEGMENTS: --resume, --budget MIN, --report-only, --dry (one seed per arm, runs capped at 2400 steps, separate dry_ stores;
the runtime decision is printed for the full plan, nothing stored). A watcher outside this file pushes every saved run.

Run:  python explore_batch18.py [--workers 4] [--dry] [--resume] [--budget MIN] [--report-only]
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
import explore_recipe_ablation as s50
import explore_split_k4 as s51
import explore_keysplits as s52
from test_binding_recipe import makespan
from test_binding_onset import EVAL_EVERY

SCREENS = (s50, s51, s52)
DRY_ITERS = 2400
CHECKS_RECORD = os.path.join(ec.OUT_DIR, "batch18_checks.json")
RUNTIME_STORE = "batch18_runtime"
SEGMENTS_ALLOWED = 4
CHECK_ALLOW_MIN, LATER_SETUP_MIN, DEFAULT_BUDGET_MIN = 30, 3, 116
CUT_ORDER = (("window_recipe_ablation", "W_SPLIT_W4"), ("split_k4", "W4k4"), ("window_recipe_ablation", "W_SPLIT_NOSLOW"),
             ("two_stream_keysplits", "W2_KEYHINGE", tuple(range(160, 180))))
NEVER = (("window_recipe_ablation", "W_RESET"), ("window_recipe_ablation", "W_SPLIT_M"), ("window_recipe_ablation", "W_M_REF"),
         ("split_k4", "W4k4_SPLIT"), ("two_stream_keysplits", "W2_K4"))
BY_NAME = {m.NAME: m for m in SCREENS}
FULL = {}


def make_dry():
    c2.DRY.update(on=True, iters=DRY_ITERS)
    for m in SCREENS:
        m.NAME = "dry_" + m.NAME
        m.ARMS = {k: dict(a, iters=min(a["iters"], DRY_ITERS), seeds=tuple(a["seeds"][:1])) for k, a in m.ARMS.items()}


# ── S50's Muon reference ─────────────────────────────────────────────────────
def muon_ref(dry):
    st = ec.load_store(RUNTIME_STORE)
    dec = st["meta"].get("muon_ref")
    if dec is not None and not dry:
        print(f"  S50 MUON REFERENCE (stored {dec['time']} on {dec['cpu']}): batch 16's LOCAL3_SLOW_D8_M|260 "
              f"{'reproduces' if dec['reproduces'] else 'does not reproduce'} here through {s50.REF_CHECK_ITERS} -> "
              f"{'batch 16 records' if dec['reproduces'] else 'W_M_REF runs'}", flush=True)
        return dec
    t0 = time.time()
    dec = s50.muon_ref_decision()
    dec["time"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"  S50 MUON REFERENCE: batch 16's LOCAL3_SLOW_D8_M|260 (record on {dec['record_cpu']}) rerun here on {dec['cpu']} through "
          f"{s50.REF_CHECK_ITERS}: {'reproduces' if dec['reproduces'] else 'does not reproduce'} (here {dec['curve']} vs record "
          f"{dec['record_curve']}; {time.time() - t0:.0f}s) -> {'batch 16 records are the reference' if dec['reproduces'] else 'W_M_REF (LOCAL3_SLOW_D8_M fresh on this CPU) runs as the pair'}"
          + (" (dry run: not stored)" if dry else " (stored)"), flush=True)
    if not dry:
        st["meta"]["muon_ref"] = dec
        st["meta"].setdefault("provenance", ec.provenance())
        ec.save_store(RUNTIME_STORE, st)
    return dec


# ── Projection, packing simulation and the runtime rule ─────────────────────
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
    cells = getattr(m, "n_cells", lambda k: 0)(arm) * tt[3] if len(tt) > 3 else 0.0
    return a["iters"] * ts + (a["iters"] // EVAL_EVERY + m.n_extra(arm)) * te + m.n_meas(arm) * tm + cells


def cut_arms(cut):
    """{(screen, arm): None (cut) or seeds (cut to)}."""
    return {(c[0], c[1]): (c[2] if len(c) > 2 else None) for c in cut}


def needed(name, arm, a, cut):
    cm = cut_arms(cut)
    if (name, arm) in cm and cm[(name, arm)] is None:
        return False
    nb = a.get("needed_by")
    return nb is None or any(not ((x[0], x[1]) in cm and cm[(x[0], x[1])] is None) for x in nb)


def plan_jobs(arms_of, t, cut):
    cm = cut_arms(cut)
    jobs = []
    for name, arms in arms_of.items():
        for k, a in arms.items():
            if not needed(name, k, a, cut):
                continue
            seeds = cm.get((name, k)) or a["seeds"]
            d = per_run(name, k, a, t)
            jobs += [dict(name=name, arm=k, opt=a["opt"], dur=d) for _ in seeds]
    return jobs


def simulate(jobs, workers, first_left_s, seg_s, max_segments=30):
    """This pool's segment packing (explore_common16.run_pool's take rule) on the given durations: segments needed."""
    todo = sorted(jobs, key=lambda j: (j["opt"] != "muon", -j["dur"]))
    seg = 0
    while todo and seg < max_segments:
        left = first_left_s if seg == 0 else seg_s
        t, running, took = 0.0, [], 0

        def take(now):
            for i, j in enumerate(todo):
                if j["dur"] <= 0.97 * (left - now):
                    return todo.pop(i)
            return None
        while True:
            while len(running) < workers:
                j = take(t)
                if j is None:
                    break
                running.append(t + j["dur"])
                took += 1
            if not running:
                break
            running.sort()
            t = running.pop(0)
        seg += 1
        if took == 0 and seg > 1:
            return float("inf")
    return seg if not todo else float("inf")


def plan(t, cut, workers, budgets, show=False, tag=""):
    jobs = plan_jobs(FULL, t, cut)
    durs = [j["dur"] for j in jobs]
    segs = simulate(jobs, workers, *budgets)
    if show:
        cm = cut_arms(cut)
        print(f"  PROJECTION{tag} (worst case: every run to its iters; pool-load timing, 1 thread each, this machine):")
        for name, arms in FULL.items():
            for k, a in arms.items():
                if not needed(name, k, a, cut):
                    continue
                tt = t[(name, k)]
                n = len(cm.get((name, k)) or a["seeds"])
                print(f"    {name:<23} {k:<15} {n:>2} runs x {per_run(name, k, a, t) / 60:6.1f} min  (step {tt[0] * 1e3:.1f} ms, "
                      f"evaluation {tt[1]:.1f} s, decoder {tt[2]:.1f} s" + (f", gate cells {tt[3]:.2f} s" if len(tt) > 3 else "") + ")")
        print(f"    total {sum(durs) / 3600:.2f} h of runs; makespan on {workers} workers {makespan(durs, workers) / 3600:.2f} h; the "
              f"packing simulation needs {segs} segment(s) (first {budgets[0] / 60:.0f} min, then {budgets[1] / 60:.0f} min each)",
              flush=True)
    return segs, makespan(durs, workers) / 3600


def runtime_rule(workers, dry, budget_min, t_start):
    st = ec.load_store(RUNTIME_STORE)
    stored = st["meta"].get("cut")
    if stored is not None and stored.get("timings") and not dry:
        t = {tuple(k.split("|")): v for k, v in stored["timings"].items()}
        print(f"  RUNTIME RULE (stored {stored['time']} on {stored['cpu']}: {stored['segments0']} segment(s), makespan "
              f"{stored['makespan_h0']:.2f} h on {stored['workers']} workers): cut {[tuple(c[:2]) + ((f'to {len(c[2])} seeds',) if len(c) > 2 else ()) for c in stored['cut']] or 'nothing'} "
              f"-> {stored['segments']} segment(s), {stored['makespan_h']:.2f} h (the stored timings are reused)", flush=True)
        return t, stored
    t = timings(workers)
    used = time.time() - t_start
    budgets = ((budget_min - CHECK_ALLOW_MIN) * 60 - used, (budget_min - LATER_SETUP_MIN) * 60)
    if dry:
        jobs = plan_jobs({m.NAME.replace("dry_", ""): m.ARMS for m in SCREENS}, t, [])
        print(f"  PROJECTION OF THIS DRY RUN: {len(jobs)} runs, {sum(j['dur'] for j in jobs) / 3600:.2f} h of runs")
    segs0, ms0 = plan(t, [], workers, budgets, show=True, tag=" OF THE FULL BATCH")
    cut, steps, segs, ms = [], [], segs0, ms0
    while segs > SEGMENTS_ALLOWED and len(cut) < len(CUT_ORDER):
        cut.append(CUT_ORDER[len(cut)])
        segs, ms = plan(t, cut, workers, budgets)
        steps.append(dict(cut=[list(x) if isinstance(x, tuple) else x for x in cut[-1]], segments=segs, makespan_h=ms))
    dec = dict(cut=[[c[0], c[1]] + ([list(c[2])] if len(c) > 2 else []) for c in cut], segments0=segs0, makespan_h0=ms0,
               steps=steps, segments=segs, makespan_h=ms, workers=workers, budgets_min=[b / 60 for b in budgets],
               cpu=ec.cpu_model(), time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
               timings={f"{a}|{b}": v for (a, b), v in t.items()})
    def cdesc(s):
        c = s["cut"]
        return f"{c[0]} {c[1]}" + (f" to seeds {c[2][0]}-{c[2][-1]}" if len(c) > 2 else "")
    print(f"  RUNTIME RULE: the packing simulation needs {segs0} segment(s) (makespan {ms0:.2f} h) "
          + (f"> {SEGMENTS_ALLOWED}: cut, re-simulating after each: " + "; ".join(f"{cdesc(s)} -> {s['segments']} segment(s), "
                                                                                     f"{s['makespan_h']:.2f} h" for s in steps)
             if cut else f"<= {SEGMENTS_ALLOWED}: no cut")
          + (f"; still over {SEGMENTS_ALLOWED} after every listed cut: W_RESET, W_SPLIT_M (+ its reference), W4k4_SPLIT and W2_K4 "
             f"are never cut, so the batch runs on" if segs > SEGMENTS_ALLOWED else "")
          + (" (dry run: not stored)" if dry else " (stored)"), flush=True)
    if not dry:
        st["meta"]["cut"] = dec
        st["meta"].setdefault("provenance", ec.provenance())
        ec.save_store(RUNTIME_STORE, st)
    return t, dec


def apply_cut(dec):
    cut = [tuple(c[:2]) + ((tuple(c[2]),) if len(c) > 2 else ()) for c in (dec or {}).get("cut", [])]
    cm = cut_arms(cut)
    for m in SCREENS:
        nm = m.NAME.replace("dry_", "")
        for k in list(m.ARMS):
            if not needed(nm, k, FULL[nm][k], cut):
                del m.ARMS[k]
                m.CUT.add(k)
            elif cm.get((nm, k)):
                m.ARMS[k] = dict(m.ARMS[k], seeds=tuple(s for s in m.ARMS[k]["seeds"] if s in cm[(nm, k)]))
                m.CUT.add(f"{k} (seeds {cm[(nm, k)][0]}-{cm[(nm, k)][-1]} only)")
    if cut:
        print(f"  arms cut: {sorted(f'{c[0]}:{c[1]}' + (f' to {len(c[2])} seeds' if len(c) > 2 else '') for c in cut)}; references "
              f"dropped with them: " + str(sorted(f"{m.NAME}:{k}" for m in SCREENS for k in m.CUT
                                                 if isinstance(k, str) and "(" not in k and (m.NAME.replace('dry_', ''), k) not in cm)),
              flush=True)


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
    print(f"  CHECK batch 18: the modules served from main are exactly the test modules that differ between this branch and "
          f"{mt.MAIN_SHA[:7]} ({diff}): {'ok' if v else 'FAIL'}", flush=True)
    if bad or not v:
        print(f"  ! CHECK failed in {bad or 'the served modules'}; stopping.")
        sys.exit(1)
    print(f"  all CHECKs passed ({(time.time() - t0) / 60:.1f} min)", flush=True)


def constants():
    c = mt.run_child("explore_muon_scale_child", "constants", {})
    assert c["tss_LR"] == c["tsr_LR"] == c["tscur_LR"] == 1e-3 and c["tss_WARM"] == 2400, c
    k = c16.run_child("explore_window_d8_child", "constants", {}, main=True)
    assert k["LR"] == 1e-3 and k["LR_WARM"] == 1e-4 and k["GAP"] == 4800 and k["S37_MAX_SPLITS"] == 2, k
    assert ec.SUB_LR == s52.LR == 1e-3
    print(f"  recipe read in children from main at {c['main_sha'][:7]}: test_slow_start.LR {c['tss_LR']:g}, test_stream_recipe.LR "
          f"{c['tsr_LR']:g}, test_stream_curriculum.LR {c['tscur_LR']:g} (Adam groups), WARM {c['tss_WARM']}; the warm lr "
          f"{k['LR_WARM']:g}; the trigger: every {k['EVERY']} from {k['FIRST']} (reference {k['REF_AT']}), acc < {k['ACC_THR']}, "
          f"rise < {k['RISE']}, gap {k['GAP']}; <= 3 splits (S50), <= {k['S37_MAX_SPLITS']} (S51, S37's cap); S52: the branch's "
          f"test_channel_binding.SUB_LR {ec.SUB_LR:g}, test_short_conv.MAX_ITERS {ec.MAX_ITERS}, the key term 1.0 x relu(eta2_key "
          f"- 0.2) on updates 1-2400; Muon lr {s50.MUON_LR:g}; 1 torch thread per run", flush=True)
    for m in SCREENS:
        for kk, a in m.ARMS.items():
            print(f"  lr for {m.NAME} {kk}: {a['lr']:g}" + (f"; Muon lr {a['muon_lr']:g}" if a.get("muon_lr") else "")
                  + f"; iters {a['iters']}; seeds {ec.fmt_seeds(a['seeds'])}")


def line(j, rec):
    if not rec.get("ok"):
        return ""
    acc = rec.get("acc")
    s = f"acc {acc:.3f}  trans {str(rec.get('transition')):>5}  "
    if j["name"].endswith("two_stream_keysplits"):
        s += (f"outcome {rec.get('outcome')}" if rec.get("outcome") else f"{'DISCOVERED' if ec.discovered(rec) else 'not discovered'}"
              + (f" ({rec.get('fail')})" if rec.get("fail") else ""))
        if rec.get("hinge"):
            s += f"; key term fired on {rec['hinge']['fired']} updates"
        return s
    s += f"outcome {rec.get('outcome')}"
    d = (rec.get("decode") or {}).get("end")
    if d:
        s += f"; decoder at {d.get('step')}: h CTX/KEY/VAL {d['h_ctx']:.2f}/{d['h_key']:.2f}/{d['h_val']:.2f}"
    if "SPLIT" in j["arm"]:
        s += f"; splits {rec.get('splits')} at {[c['step'] for c in rec.get('checks') or [] if c.get('fired')]}"
    if rec.get("resets"):
        s += f"; resets at {[x['step'] for x in rec['resets']]}"
    return s


def dur_fn(t, cpu):
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
        print("#" * 20 + f" RESUME {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC — a new segment; saved runs are reused, "
              f"missing runs re-run ({ec.BANNER}) " + "#" * 20)
    cpu = ec.cpu_model()
    rec = None
    if os.path.exists(CHECKS_RECORD) and not args.dry:
        with open(CHECKS_RECORD) as f:
            rec = json.load(f)
    c16.STATE["batch_cpu"] = rec["cpu"] if rec else cpu
    if args.dry:
        make_dry()
    dl = c2.dry_label()
    pv = ec.print_banner("batch 18: window_recipe_ablation (S50), split_k4 (S51), two_stream_keysplits (S52)" + dl)
    shas = code_shas()
    c16.STATE["shas"] = shas
    c16.STATE["other_cpus"] = sorted(set(c16.other_cpus_seen(SCREENS)) | {cpu})
    same_cpu = cpu == c16.STATE["batch_cpu"]
    print(f"  this segment's CPU: {cpu}; the batch's CPU (Muon runs and their pairs): {c16.STATE['batch_cpu']}"
          + ("" if same_cpu else " — ANOTHER CPU: only Adam runs run in this segment (each accepted once this CPU passes the "
                                 "repro check)"), flush=True)
    if args.report_only:
        print("  REPORT ONLY: from the saved runs (no repro check, CHECKs or runs in this call)")
        st = ec.load_store(RUNTIME_STORE)
        s50.configure_ref(st["meta"].get("muon_ref"))
        FULL.update({m.NAME.replace("dry_", ""): {k: dict(a) for k, a in m.ARMS.items()} for m in SCREENS})
        apply_cut(st["meta"].get("cut"))
        constants()
    else:
        if not ec.repro_check():
            print("  ! this container does not reproduce X: the recorded pairing is invalid; stopping.")
            sys.exit(1)
        dec_m = muon_ref(args.dry) if (same_cpu or ec.load_store(RUNTIME_STORE)["meta"].get("muon_ref")) else None
        s50.configure_ref(dec_m)
        FULL.update({m.NAME.replace("dry_", ""): {k: dict(a) for k, a in (dict(m.ARMS) if not args.dry else m.ARMS).items()}
                     for m in SCREENS})
        if args.dry:                                          # the full plan's arms, not the dry run's capped ones
            FULL["window_recipe_ablation"] = {k: dict(a, iters=s50.ITERS, seeds=s50.SB) for k, a in s50.ARMS.items()}
            FULL["split_k4"] = {k: dict(a, iters=s51.ITERS, seeds=s51.SEEDS) for k, a in s51.ARMS.items()}
            FULL["two_stream_keysplits"] = {k: dict(a, iters=s52.ITERS, seeds=s52.SEEDS) for k, a in s52.ARMS.items()}
        t, dec = runtime_rule(args.workers, args.dry, args.budget or DEFAULT_BUDGET_MIN, t_start)
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
            st["meta"]["cut"] = sorted(str(x) for x in m.CUT)
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
    r50 = s50.report()
    for arm, o in r50.items():
        if o.get("reading"):
            print(f"  {tagp}READING {s50.NAME} {arm}: {o['reading']}  [{ec.BANNER}]")
    print()
    r51 = s51.report()
    if "W4k4_SPLIT" in r51:
        print(f"  {tagp}READING {s51.NAME}: {r51['W4k4_SPLIT']['reading']}  [{ec.BANNER}]")
    print()
    r52 = s52.report()
    for arm, o in r52.items():
        print(f"  {tagp}READING {s52.NAME} {arm}: {o['reading']}  [{ec.BANNER}]")
    print()
    print("#" * 100)
    print(f"SUMMARY — {ec.BANNER}{dl}")
    for arm, o in r50.items():
        print(f"  S50 window_recipe_ablation {arm}: " + (o["reading"] if o.get("reading") else
                                                       f"reference, BOUND ROUTED {o['bound_routed']}/{o['n']}")
              + f"; bound seeds {o['bound_seeds']}; classes {o['classes']}")
    for arm, o in r51.items():
        print(f"  S51 split_k4 {arm}: " + (o.get("reading") or f"BOUND ROUTED {o['bound_routed']}/{o['n']} (printed alongside)")
              + f"; bound seeds {o['bound_seeds']}; classes {o['classes']}")
    for arm, o in r52.items():
        print(f"  S52 two_stream_keysplits {arm}: {o['reading']}; unsuccessful classes {o['classes']}; key splits held to the end "
              f"{len(o['held'])}")
    for m in SCREENS:
        if m.CUT:
            print(f"  {m.NAME} cut by the runtime rule: {sorted(str(x) for x in m.CUT)}")
    print("#" * 100)


if __name__ == "__main__":
    main()
