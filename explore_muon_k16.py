#!/usr/bin/env python
"""
explore_muon_k16.py — EXPLORATORY, not a result. Screen S35 (batch 11): does S25's Muon recipe carry to
the main line's working four-stream configuration (S=4, P=4, k=16)?

BACKGROUND (docstring): see explore_hinge_window. S33 (batch 10): at S=4 k=4 Muon did not break the
merges (MUON_HINGE 4/10, MUON_SLOW 3/10, X's A4k4 4/10); at S=8 k=16 0/10. The main line's working
four-stream configuration is k=16: X's HINGE4k16 binds 17/20 (35/40 pooled), untested under Muon.

THE CONFIGURATION: the main line's (S=4, P=4, k=16, conv, MAX_ITERS 28800, test_slow_start.LR = 1e-3 for
the Adam groups), on main's run path at 9c5939e in a child process, as S33 (explore_main9c; code in
explore_muon_k16_child): test_stream_recipe.run_attempt with ARM["A4k16"], the path X's HINGE4k16 ran on.
  MUON_HINGE16  S33's MUON_HINGE: S25's optimizer groups (gate W_in, W_h, W_g on Muon at the Muon lr
                throughout; decoder, encoder, encoder_v, lm_head on Muon; embedding and convolution on
                Adam 1e-3; non-gate groups x0.1 for updates 1-2400) and the main line's HINGE hinge
                (test_slow_start.make_recipe's hook: 1.0 x [relu(eta2_index - 0.2) + relu(eta2_half -
                0.2)], pooled over channels). Seeds 240-259, paired with X's HINGE4k16 (slow_start at
                b8c6007) and A4k16 (stream_recipe at 9c5939e).
VALIDITY: the perfect gate (ceiling4k16 on the same path) under Muon at 0.005 (no slow phase), seeds
240-241, 28800 steps; if fewer than 2 bind, 0.0025, then 0.01; S25's rule (the smallest lr at which both
bind); UNTESTED if none binds.
READINGS (fixed before any run; BOUND; b = HINGE4k16-only seeds, c = MUON_HINGE16-only seeds; exact
McNemar, two-sided):
  - "the Muon recipe carries to four streams at k=16" if MUON_HINGE16 >= 16/20 and b - c <= 1;
  - "worse under Muon" if b - c >= 4 and p < 0.1;
  - otherwise neither.
  If the runtime rule cuts S35 to 240-249, the first threshold is >= 8/10 (the same fraction); b - c and
  p unchanged. Also printed: vs X's A4k16.
DIAGNOSTICS: failure classes (test_stream_recipe.outcome), merged runs (two streams on one channel at
the end), distinct channels holding the 4 streams and streams per channel at 4800, 9600 and the end, and
the hinge's firings with their updates.
RUNTIME (batch level, explore_batch11): the projection with pool-load timing; if over 9 h, S35 is cut
to 240-249 (S34 is never cut). Stored with the validity record, so a resumed segment uses the same seeds.
CHECKS (child process): X's HINGE4k16|240 reproduced under main's own Adam path through 1200 (path
check); the groups cover each parameter once; each group's lr at updates 1, 2400, 2401; the firing log
(TAU 0: one entry per update); the perfect gate zeroes every cross-stream score at every layer; the
served modules are exactly the test modules that differ between this branch and 9c5939e.
"""

import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common2 as c2
import explore_main9c as mt
import explore_muon_scale as s33
from test_channel_binding import SUB_LR
from test_router_confirm import mcnemar_exact
from test_binding_onset import EVAL_EVERY

NAME = "muon_k16"
IDEA = "S33's MUON_HINGE in the main line's working four-stream configuration: S=4, P=4, k=16"
SOURCE = ("batch 10's S33 (Muon did not break the merges at k=4; 0/10 at S=8); X's HINGE4k16 17/20 at k=16 under "
          "Adam")
CHANGE = ("main's run path at 9c5939e (child process), A4k16 with S25's optimizer groups and the main line's hinge "
          "(S33's MUON_HINGE)")
PAIRING = ("seeds 240-259 with X's recorded HINGE4k16 (slow_start at b8c6007) and A4k16 (stream_recipe); same "
           "initial parameters and batches")
CHILD = "explore_muon_k16_child"
LR = SUB_LR
ITERS = 28800
VALID_LRS = (0.005, 0.0025, 0.01)
VALID_SEEDS = (240, 241)
SEEDS = tuple(range(240, 260))
CUT_SEEDS = tuple(range(240, 250))
CUT_H = 9.0
CARRIES_N, CARRIES_N_CUT, CARRIES_D, WORSE_D, WORSE_P = 16, 8, 1, 4, 0.1
TIME_STEPS = 300
STEPS_DIAG = (4800, 9600, "end")
STATE = dict(const=None, mlr=None, untested=False, note="", cut=None, recorded=None)

ARMS = {
    "MUON_HINGE16": dict(key="MUON_HINGE16", kind="HINGE", seeds=SEEDS, lr=LR, iters=ITERS, prio=2, muon_lr=None,
                         label="S33's MUON_HINGE at S=4, P=4, k=16, conv (A4k16's path)",
                         sched=("Muon lr (validity): gate W_in/W_h/W_g throughout, other Muon weights x0.1 for updates "
                                "1-2400; Adam on embedding and conv 0.0001 for 1-2400, 0.001 after; + main's HINGE hinge "
                                "(1.0 x [relu(eta2_index - 0.2) + relu(eta2_half - 0.2)], pooled over channels)")),
}


def child(func, payload, timeout=None):
    return mt.run_child(CHILD, func, payload, timeout=timeout)


def constants():
    if STATE["const"] is None:
        c = child("constants", {})
        assert c["tss_LR"] == c["tsr_LR"] == LR, c
        assert c["tsr_MAX_ITERS"] == ITERS and c["EVAL_EVERY"] == EVAL_EVERY, c
        STATE["const"] = c
    return STATE["const"]


def print_constants():
    c = constants()
    print(f"  S35 recipe read in the child from main at {c['main_sha'][:7]}: test_slow_start.LR {c['tss_LR']:g}, "
          f"test_stream_recipe.LR {c['tsr_LR']:g} (Adam groups; the branch's SUB_LR {SUB_LR:g}); WARM {c['tss_WARM']}, "
          f"TAU {c['tss_TAU']}, LAMBDA {c['tss_LAMBDA']}; MAX_ITERS {c['tsr_MAX_ITERS']}; schedule {c['tsr_REAL']}",
          flush=True)
    for k, ps in c["params"].items():
        print(f"    parameters {k}: " + ", ".join(f"{n} {tuple(s)}" for n, s in ps))


def run_job(arm, seed):
    a = ARMS[arm]
    assert a["muon_lr"] is not None, arm
    return child("run", dict(kind=a["kind"], seed=seed, iters=a["iters"], mlr=a["muon_lr"]))


def segments(arm):
    raise NotImplementedError("S35 is projected by its own child timing (explore_batch11)")


# ── validity ─────────────────────────────────────────────────────────────────
def valid_name():
    return NAME + "_valid"


def _valid_job(job):
    t = time.time()
    rec = child("run", dict(kind="CEIL", seed=job["seed"], iters=job["iters"], mlr=job["lr"]))
    rec.update(secs_wall=time.time() - t, lr=LR, muon_lr=job["lr"])
    return job, rec


def select_lr(workers, dry=False, run=True):
    st = ec.load_store(valid_name())
    st["meta"].setdefault("provenance", ec.provenance())
    iters = min(ITERS, c2.DRY.get("iters", ITERS)) if dry else ITERS
    seeds = VALID_SEEDS[:1] if dry else VALID_SEEDS
    st["meta"].update(candidates=list(VALID_LRS), seeds=list(seeds), iters=iters, dry=dry)
    print(f"  S35 VALIDITY ({ec.BANNER}): the perfect gate (ceiling4k16) under MUON (no slow phase), Muon lr "
          f"{VALID_LRS[0]:g}, then {VALID_LRS[1]:g}, then {VALID_LRS[2]:g} while fewer than {len(seeds)} bind; seeds "
          f"{list(seeds)}; {iters} steps; Adam lr {LR:g} on the embedding and conv", flush=True)
    chosen = None
    t0 = time.time()
    for lr in VALID_LRS:
        jobs = [dict(lr=lr, seed=s, iters=iters) for s in seeds if f"{lr}|{s}" not in st["runs"]]
        if jobs and run:
            with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
                for job, rec in ex.map(_valid_job, jobs):
                    st["runs"][f"{job['lr']}|{job['seed']}"] = rec
                    ec.save_store(valid_name(), st)
                    print(f"  [valid {job['lr']:g} seed {job['seed']}] {(time.time() - t0) / 60:5.1f} min  (Adam lr {LR:g}, "
                          f"{iters} steps): {ec.line(rec)}", flush=True)
        rs = [st["runs"].get(f"{lr}|{s}") for s in seeds]
        print(f"    Muon lr {lr:<6g} " + "  ".join(
            f"seed {s}: {('BOUND at ' + str(r['transition'])) if ec.bound(r) else 'unbound'} (acc {r['acc']:.3f})"
            if r and r.get("ok") else f"seed {s}: {ec.tag(r)}" for s, r in zip(seeds, rs)), flush=True)
        if all(ec.bound(r) for r in rs):
            chosen = lr
            break
    if chosen is not None:
        STATE.update(mlr=chosen, untested=False, note=f"smallest lr at which both seeds bind (S25's rule): {chosen:g}")
    elif dry:
        STATE.update(mlr=VALID_LRS[0], untested=False,
                     note=f"DRY: no lr bound in the shortened runs; {VALID_LRS[0]:g} used to exercise the code")
    else:
        STATE.update(mlr=None, untested=True, note="no lr bound both seeds: S35 UNTESTED")
    a = ARMS["MUON_HINGE16"]
    a["muon_lr"] = STATE["mlr"]
    if STATE["untested"]:
        a["seeds"] = ()
    else:
        a["sched"] = a["sched"].replace("Muon lr (validity)", f"Muon lr {STATE['mlr']:g} (validity)")
    print(f"  S35 LR CHOICE: {STATE['note']}", flush=True)
    st["meta"]["choice"] = STATE["mlr"]
    ec.save_store(valid_name(), st)
    apply_cut(st)
    return STATE


def apply_cut(st=None):
    st = ec.load_store(valid_name()) if st is None else st
    cut = st["meta"].get("cut")
    STATE["cut"] = cut
    if cut and cut.get("cut"):
        a = ARMS["MUON_HINGE16"]
        a["seeds"] = tuple(s for s in a["seeds"] if s in CUT_SEEDS)
    return cut


def decide_cut(batch_h, workers, dry=False):
    """The batch-level runtime rule (explore_batch11 computes the batch's projected makespan)."""
    st = ec.load_store(valid_name())
    cut = st["meta"].get("cut")
    if cut is None:
        do = batch_h > CUT_H
        cut = dict(cut=do, makespan_h=batch_h, workers=workers,
                   time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
        if not dry:
            st["meta"]["cut"] = cut
            ec.save_store(valid_name(), st)
        if do:
            ARMS["MUON_HINGE16"]["seeds"] = tuple(s for s in ARMS["MUON_HINGE16"]["seeds"] if s in CUT_SEEDS)
        print(f"  RUNTIME RULE: projected batch makespan {batch_h:.2f} h {'>' if do else '<='} {CUT_H:g} h: "
              + (f"S35 cut to {CUT_SEEDS[0]}-{CUT_SEEDS[-1]} (S34 never cut)" if do else "no cut")
              + (" (dry run: not stored)" if dry else " (stored)"), flush=True)
    else:
        print(f"  RUNTIME RULE (stored {cut['time']}: batch makespan {cut['makespan_h']:.2f} h on {cut['workers']} workers): "
              + (f"S35 cut to {CUT_SEEDS[0]}-{CUT_SEEDS[-1]}" if cut["cut"] else "no cut"), flush=True)
    STATE["cut"] = cut
    return cut


def timing(mlr=None):
    """(seconds per step, seconds per evaluation) for MUON_HINGE16 from one child (call it concurrently with
    the other arms' timing for the pool's load)."""
    t = child("timing", dict(mlr=mlr or VALID_LRS[0], which=["HINGE"], steps=TIME_STEPS))
    return t["HINGE"][:2]


# ── CHECKs ───────────────────────────────────────────────────────────────────
def check():
    t0 = time.time()
    rows = child("checks", dict(mlr=VALID_LRS[0]))
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    diff = mt.differing()
    v = diff == sorted(mt.SERVED)
    print(f"  CHECK {NAME}: child at {mt.MAIN_SHA[:7]} ({(time.time() - t0) / 60:.1f} min); the modules served from main "
          f"are exactly the test modules that differ between this branch and {mt.MAIN_SHA[:7]} ({diff}): "
          f"{'ok' if v else 'FAIL'}", flush=True)
    return ok and v


# ── Report ───────────────────────────────────────────────────────────────────
def recorded():
    if STATE["recorded"] is None:
        STATE["recorded"] = child("recorded_summary", dict(seeds=list(SEEDS)))
    return STATE["recorded"]


def compare(new, old, seeds, lab_new, lab_old):
    both = [s for s in seeds if s in new and s in old]
    b = sum(1 for s in both if new[s] and not old[s])
    c = sum(1 for s in both if old[s] and not new[s])
    p = mcnemar_exact(b, c)
    print(f"    BOUND  {lab_new} {sum(new[s] for s in both)}/{len(both)}  {lab_old} {sum(old[s] for s in both)}/{len(both)}  "
          f"{lab_new} only {b}, {lab_old} only {c}, McNemar two-sided p = {p:.3g}")
    return dict(n=len(both), new=sum(new[s] for s in both), old=sum(old[s] for s in both), b=b, c=c, p=p)


def _xrec(x):
    return dict(end=dict(ch_map=x["end_map"]), maps=x["maps"], stopped_at=x.get("stopped_at"))


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    vs = ec.load_store(valid_name())
    print("    validity (ceiling4k16): " + "; ".join(
        f"Muon lr {k.split('|')[0]} seed {k.split('|')[1]} "
        f"{('BOUND at ' + str(r['transition'])) if ec.bound(r) else 'unbound'} (acc {r['acc']:.3f})"
        if r.get("ok") else f"{k} {ec.tag(r)}" for k, r in sorted(vs["runs"].items())) + f" -> {STATE['note'] or '--'}")
    a = ARMS["MUON_HINGE16"]
    seeds = a["seeds"]
    cut = bool((STATE.get("cut") or {}).get("cut"))
    rec = recorded()
    got = {int(k.split("|")[1]): r for k, r in store["runs"].items() if k.startswith("MUON_HINGE16|") and r.get("ok")}
    xh = {s: rec[f"HINGE4k16|{s}"] for s in SEEDS if f"HINGE4k16|{s}" in rec}
    xa = {s: rec[f"A4k16|{s}"] for s in SEEDS if f"A4k16|{s}" in rec}
    print(f"  per seed (ch = distinct channels holding the 4 streams at 4800/9600/end; X HINGE4k16's firings per "
          f"{EVAL_EVERY}-update window, its timing within windows not logged):")
    print(f"    {'seed':>4} | {'MUON_HINGE16':<26} {'trans':>5} {'acc':>5} {'ch':>7} {'per ch':>8} {'firings at updates':<30} | "
          f"{'X HINGE4k16':<26} {'trans':>5} {'ch':>7} {'fired':>5} | {'X A4k16':<26} {'trans':>5}")
    for s in SEEDS:
        r = got.get(s)
        if r is None:
            c1 = f"{'not run' if s not in seeds else 'missing':<26} {'':>5} {'':>5} {'':>7} {'':>8} {'':<30}"
        else:
            dd = "/".join(str(s33.distinct(s33.map_at(r, t))) if s33.map_at(r, t) else "--" for t in STEPS_DIAG)
            fa = r.get("fired_at") or []
            fs = (str(fa[:4])[:-1] + (f", ... ({len(fa)})]" if len(fa) > 4 else "]")) if fa else "none"
            c1 = (f"{r['outcome']:<26} {str(r['transition']):>5} {r['acc']:5.3f} {dd:>7} "
                  f"{s33.per_channel(s33.map_at(r, 'end')):>8} {fs:<30}")
        x = xh.get(s)
        c2_ = (f"{x['outcome']:<26} {str(x['transition']):>5} "
               f"{'/'.join(str(s33.distinct(s33.map_at(_xrec(x), t))) if s33.map_at(_xrec(x), t) else '--' for t in STEPS_DIAG):>7} "
               f"{(x['hs_hinge'] or ['--'])[0]:>5}" if x else f"{'--':<46}")
        y = xa.get(s)
        c3 = f"{y['outcome']:<26} {str(y['transition']):>5}" if y else "--"
        print(f"    {s:>4} | {c1} | {c2_} | {c3}")
    b16 = {s: ec.bound(r) for s, r in got.items()}
    print("  MUON_HINGE16 vs X HINGE4k16 (the rule):")
    d = compare(b16, {s: x["bound"] for s, x in xh.items()}, seeds, "MUON_HINGE16", "X HINGE4k16")
    print("  MUON_HINGE16 vs X A4k16 (printed):")
    da = compare(b16, {s: x["bound"] for s, x in xa.items()}, seeds, "MUON_HINGE16", "X A4k16")
    mg = [s for s, r in sorted(got.items()) if s33.merged_end(r, 4)]
    cls = {}
    for r in got.values():
        cls[r["outcome"]] = cls.get(r["outcome"], 0) + 1
    xcls = {}
    for s in seeds:
        if s in xh:
            xcls[xh[s]["outcome"]] = xcls.get(xh[s]["outcome"], 0) + 1
    print(f"    merged at the end (two streams on one channel): {len(mg)}/{len(got)} {mg}; outcomes {cls}; X HINGE4k16 on "
          f"these seeds {xcls}")
    for t in STEPS_DIAG:
        ds = [s33.distinct(s33.map_at(r, t)) for _, r in sorted(got.items()) if s33.map_at(r, t) is not None]
        xd = [s33.distinct(s33.map_at(_xrec(x), t)) for s_, x in sorted(xh.items()) if s_ in seeds
              and s33.map_at(_xrec(x), t) is not None]
        print(f"    distinct channels at {t}: {ds} (median {statistics.median(ds) if ds else '--'}); streams per channel "
              f"{[s33.per_channel(s33.map_at(r, t)) for _, r in sorted(got.items())]}; X HINGE4k16 {xd} (median "
              f"{statistics.median(xd) if xd else '--'})")
    fa = {s: r.get("fired_at") or [] for s, r in sorted(got.items())}
    allf = [u for v in fa.values() for u in v]
    edges = ((0, 300), (300, 600), (600, 1200), (1200, 2400), (2400, 4800), (4800, 10 ** 9))
    print(f"    MUON_HINGE16 firings: {len(allf)} in {sum(1 for v in fa.values() if v)}/{len(fa)} runs; by update "
          f"300/600/1200/2400/4800/after: {'/'.join(str(sum(1 for u in allf if lo < u <= hi)) for lo, hi in edges)}; "
          f"per seed {{seed: (count, first, last)}}: {{" + ", ".join(f"{s}: ({len(v)}, {v[0]}, {v[-1]})" if v else f"{s}: 0"
                                                                     for s, v in fa.items()) + "}")
    xf = {s: (x["hs_hinge"] or [None])[0] for s, x in sorted(xh.items()) if s in seeds}
    print(f"    X HINGE4k16 firings per seed (count): {xf}")
    need = CARRIES_N_CUT if cut else CARRIES_N
    nmax = len(CUT_SEEDS) if cut else len(SEEDS)
    if STATE["untested"]:
        rd = "UNTESTED (the perfect gate bound at no Muon lr on both seeds)"
    elif d["new"] >= need and d["b"] - d["c"] <= CARRIES_D:
        rd = "the Muon recipe carries to four streams at k=16"
    elif d["b"] - d["c"] >= WORSE_D and d["p"] < WORSE_P:
        rd = "worse under Muon"
    else:
        rd = "neither"
    print(f"  RULE S35 (MUON_HINGE16 BOUND {d['new']}/{d['n']}; HINGE4k16 only {d['b']}, MUON_HINGE16 only {d['c']}, p = "
          f"{d['p']:.3g}; 'carries' if >= {need}/{nmax} and b - c <= {CARRIES_D}, 'worse' if b - c >= {WORSE_D} and p < "
          f"{WORSE_P}): {rd}")
    return dict(d, rule=rd, vs_a=da, untested=STATE["untested"], mlr=STATE["mlr"], cut=cut, merged=len(mg),
                complete=len(got) == len(seeds))
