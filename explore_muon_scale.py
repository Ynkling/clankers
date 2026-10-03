#!/usr/bin/env python
"""
explore_muon_scale.py — EXPLORATORY, not a result. Screen S33 (batch 10): does S25's Muon recipe
carry to the main line's configurations (four streams at k=4, eight streams at k=16)?

BACKGROUND (docstring): see explore_muon_slow (S25's results; under Adam, the main line's HINGE binds
S=4 k=16 but not S=8: HINGE_D8 0/10 on X, 260-269, the 8 streams on a median of 2 channels at the end,
"the coarse split"; X's A4k4 (k=4, plain) binds 4/20 on 240-259, merged runs stalling).

THE CONFIGURATIONS (the main line's, conv on, MAX_ITERS 28800, test_slow_start.LR = 1e-3 for the Adam
groups), on the main line's own run paths at commit 9c5939e (explore_main9c: every run in a child
process with main's module tree; code in explore_muon_scale_child):
  (a) S=4, P=4, k=4: test_stream_recipe.run_attempt with ARM["A4k4"] (its restart check is recorded,
      never applied); seeds 240-249, paired with X's recorded A4k4 (results/X/stream_recipe_results.json).
  (b) S=8, P=4, k=16, all 8 streams from step 1: test_stream_curriculum.run_sc with ARM["D8"]; seeds
      260-269, paired with X's recorded HINGE_D8 (results/X/slow_start_results.json at b8c6007).
ARMS per configuration (S25's optimizer groups: gate W_in, W_h, W_g on Muon at the Muon lr throughout;
the other 2-D / 3-D weights (decoder, encoder, encoder_v, lm_head) on Muon; the embedding and the
convolution (and any bias or 1-D parameter; there are none) on Adam at 1e-3; every non-gate group at
0.1 x its lr for updates 1-2400, full after the evaluation at 2400):
  MUON_SLOW    the above.
  MUON_HINGE   + the main line's HINGE hinge (test_slow_start.make_recipe's hook: 1.0 x [relu(eta2_index -
               0.2) + relu(eta2_half - 0.2)], eta2 of the read gate at key positions pooled over channels).
VALIDITY, per configuration: the perfect gate (ceiling4k4; ceiling8k16 on run_sc, cur False) under Muon
at 0.005 (no slow phase), seeds 240-241 / 260-261, 28800 steps. If fewer than 2 bind, 0.0025, then
0.01; S25's rule picks the smallest lr at which both bind. If none binds, the configuration is UNTESTED
(its arms are not run).
OUTCOME: BOUND.
DIAGNOSTICS: merged runs (two streams on one channel at the end) and failure classes
(test_stream_recipe.outcome); the number of distinct channels holding the streams, and the streams per
channel, at 4800, 9600 and the end (from routing_k's ch_map; a run that stopped early keeps its last map).
READINGS (fixed before any run):
  (a) "promising at four streams" if the better Muon arm binds >= 8/10; otherwise "not promising at
      four streams".
  (b) "breaks the coarse split" if an arm binds >= 3 runs, or its median number of distinct channels at
      the end is >= 6 (HINGE_D8's median printed alongside); otherwise "does not break the coarse
      split". If (b) is cut to 260-264, the thresholds stay 3 runs and median 6 (not rescaled).
RUNTIME: the projection is printed before the runs (S33's runs alone, worst case: every run to 28800);
if it is over 10 h, (b) is cut to 260-264 per arm, never (a). The decision is stored with the validity
record, so a resumed segment uses the same seeds.
CHECKS (child process): the groups cover every parameter exactly once (both configurations, every
kind); each group's lr at updates 1, 2400, 2401 (MUON_SLOW and MUON_HINGE, both configurations;
forward and evaluation stubbed); the perfect gate zeroes every cross-stream score at every layer
(test_stream_recipe.cross_scores); the child's run paths reproduce X's records through 1200 (HINGE_D8|260
with test_slow_start's own HINGE recipe; A4k4|240 plain); with TAU 1.0, MUON_HINGE equals MUON_SLOW
bit for bit (configuration a, 1200 steps).
"""

import json
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
from test_channel_binding import SUB_LR
from test_router_confirm import mcnemar_exact
from test_binding_onset import EVAL_EVERY

NAME = "muon_scale"
IDEA = "S25's Muon recipe (MUON_SLOW, MUON_HINGE) in the main line's configurations: S=4 k=4, and S=8 k=16"
SOURCE = ("batch 9's S25 (MUON_HINGE 20/20 at S=2, stream-routed at 1200 on every seed); the main line's HINGE "
          "binds 0/10 at S=8 (the coarse split)")
CHANGE = ("main's run paths at 9c5939e (child process) with S25's optimizer groups: gate on Muon at the Muon lr; "
          "other weights on Muon; embedding and conv on Adam 1e-3; non-gate groups x0.1 for updates 1-2400")
PAIRING = ("(a) seeds 240-249 with X's recorded A4k4 (stream_recipe); (b) seeds 260-269 with X's recorded "
           "HINGE_D8 (slow_start at b8c6007); same initial parameters and batches")
CHILD = "explore_muon_scale_child"
LR = SUB_LR                                    # test_slow_start.LR = SUB_LR (asserted against the child's)
ITERS = 28800                                  # test_stream_recipe.MAX_ITERS = test_stream_curriculum.TOTAL
VALID_LRS = (0.005, 0.0025, 0.01)
VALID_SEEDS = {"a": (240, 241), "b": (260, 261)}
PROMISING_A, BREAK_N, BREAK_MED = 8, 3, 6
CUT_H = 10.0
CUT_SEEDS = tuple(range(260, 265))
TIME_STEPS = {"a": 400, "b": 200}            # about the same wall time per arm under load
STEPS_DIAG = (4800, 9600, "end")
CFG = {"a": dict(label="S=4, P=4, k=4, conv", S=4, x_key="A4k4", x_name="X A4k4"),
       "b": dict(label="S=8, P=4, k=16, conv, 8 streams from step 1", S=8, x_key="HINGE_D8", x_name="X HINGE_D8")}
STATE = dict(const=None, mlr={}, untested={}, notes={}, cut=None, recorded=None)


def _sched(kind):
    s = ("Muon lr (validity): gate W_in/W_h/W_g throughout, other Muon weights x0.1 for updates 1-2400; Adam on "
         "embedding and conv 0.0001 for 1-2400, 0.001 after")
    return s + ("; + main's HINGE hinge (1.0 x [relu(eta2_index - 0.2) + relu(eta2_half - 0.2)], pooled over channels)"
                if kind == "HINGE" else "")


ARMS = {}
for _c, _seeds, _p in (("a", tuple(range(240, 250)), 1), ("b", tuple(range(260, 270)), 2)):
    for _k in ("SLOW", "HINGE"):
        _key = ("A" if _c == "a" else "D8") + "_" + _k
        ARMS[_key] = dict(key=_key, cfg=_c, kind=_k, seeds=_seeds, lr=LR, iters=ITERS, prio=_p, muon_lr=None,
                          label=f"MUON_{_k} ({CFG[_c]['label']})", sched=_sched(_k))


def arm_of(cfg, kind):
    return ARMS[("A" if cfg == "a" else "D8") + "_" + kind]


def child(func, payload, timeout=None):
    return mt.run_child(CHILD, func, payload, timeout=timeout)


def constants():
    if STATE["const"] is None:
        c = child("constants", {})
        assert c["tss_LR"] == c["tsr_LR"] == c["tscur_LR"] == LR, c
        assert c["tsr_MAX_ITERS"] == c["tscur_TOTAL"] == ITERS, c
        assert c["EVAL_EVERY"] == EVAL_EVERY, c
        STATE["const"] = c
    return STATE["const"]


def print_constants():
    c = constants()
    print(f"  S33 recipe read in the child from main at {c['main_sha'][:7]}: test_slow_start.LR {c['tss_LR']:g}, "
          f"test_stream_recipe.LR {c['tsr_LR']:g}, test_stream_curriculum.LR {c['tscur_LR']:g} (Adam groups; the "
          f"branch's SUB_LR {SUB_LR:g}); WARM {c['tss_WARM']}, TAU {c['tss_TAU']}, LAMBDA {c['tss_LAMBDA']}; "
          f"MAX_ITERS {c['tsr_MAX_ITERS']} / TOTAL {c['tscur_TOTAL']}; schedules {c['tsr_REAL']} (a), "
          f"{c['tscur_REAL']} (b)", flush=True)
    for k, ps in c["params"].items():
        print(f"    parameters {k}: " + ", ".join(f"{n} {tuple(s)}" for n, s in ps))


# ── runs ─────────────────────────────────────────────────────────────────────
def run_job(arm, seed):
    a = ARMS[arm]
    assert a["muon_lr"] is not None, arm
    return child("run", dict(cfg=a["cfg"], kind=a["kind"], seed=seed, iters=a["iters"], mlr=a["muon_lr"]))


def segments(arm):
    raise NotImplementedError("S33 is projected by its own child timing (projection below)")


# ── validity ─────────────────────────────────────────────────────────────────
def valid_name():
    return NAME + "_valid"


def _valid_job(job):
    t = time.time()
    rec = child("run", dict(cfg=job["cfg"], kind="CEIL", seed=job["seed"], iters=job["iters"], mlr=job["lr"]))
    rec.update(secs_wall=time.time() - t, lr=LR, muon_lr=job["lr"])
    return job, rec


def select_lr(workers, dry=False, run=True):
    """Per configuration: the perfect gate at 0.005, then (if fewer than 2 bind) 0.0025, then 0.01; the
    smallest lr at which every seed binds. Stored; a resume reuses it. Sets the arms' muon_lr (or UNTESTED)."""
    st = ec.load_store(valid_name())
    st["meta"].setdefault("provenance", ec.provenance())
    iters = min(ITERS, c2.DRY.get("iters", ITERS)) if dry else ITERS
    seeds = {c: (v[:1] if dry else v) for c, v in VALID_SEEDS.items()}
    st["meta"].update(candidates=list(VALID_LRS), seeds=seeds, iters=iters, dry=dry)
    print(f"  S33 VALIDITY ({ec.BANNER}): the perfect gate under MUON (no slow phase), Muon lr {VALID_LRS[0]:g}, "
          f"then {VALID_LRS[1]:g}, then {VALID_LRS[2]:g} while fewer than {len(seeds['a'])} bind; seeds (a) "
          f"{list(seeds['a'])}, (b) {list(seeds['b'])}; {iters} steps; Adam lr {LR:g} on the embedding and conv",
          flush=True)
    chosen = {}
    t0 = time.time()
    for lr in VALID_LRS:
        todo = [c for c in CFG if c not in chosen]
        if not todo:
            break
        jobs = [dict(cfg=c, lr=lr, seed=s, iters=iters) for c in todo for s in seeds[c]
                if f"{c}|{lr}|{s}" not in st["runs"]]
        if jobs and run:
            with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
                for n, (job, rec) in enumerate(ex.map(_valid_job, jobs), 1):
                    st["runs"][f"{job['cfg']}|{job['lr']}|{job['seed']}"] = rec
                    ec.save_store(valid_name(), st)
                    print(f"  [valid {job['cfg']} {job['lr']:g} seed {job['seed']}] {(time.time() - t0) / 60:5.1f} min  "
                          f"(Adam lr {LR:g}, {iters} steps): {ec.line(rec)}", flush=True)
        for c in todo:
            rs = [st["runs"].get(f"{c}|{lr}|{s}") for s in seeds[c]]
            print(f"    config {c} Muon lr {lr:<6g} " + "  ".join(
                f"seed {s}: {('BOUND at ' + str(r['transition'])) if ec.bound(r) else 'unbound'} (acc {r['acc']:.3f})"
                if r and r.get("ok") else f"seed {s}: {ec.tag(r)}" for s, r in zip(seeds[c], rs)), flush=True)
            if all(ec.bound(r) for r in rs):
                chosen[c] = lr
    for c in CFG:
        if c in chosen:
            STATE["mlr"][c], STATE["untested"][c] = chosen[c], False
            STATE["notes"][c] = f"smallest lr at which both seeds bind (S25's rule): {chosen[c]:g}"
        elif dry:
            STATE["mlr"][c], STATE["untested"][c] = VALID_LRS[0], False
            STATE["notes"][c] = f"DRY: no lr bound in the shortened runs; {VALID_LRS[0]:g} used to exercise the code"
        else:
            STATE["mlr"][c], STATE["untested"][c] = None, True
            STATE["notes"][c] = "no lr bound both seeds: configuration UNTESTED"
        for a in ARMS.values():
            if a["cfg"] == c:
                a["muon_lr"] = STATE["mlr"][c]
                if STATE["untested"][c]:
                    a["seeds"] = ()
                else:
                    a["sched"] = a["sched"].replace("Muon lr (validity)", f"Muon lr {STATE['mlr'][c]:g} (validity)")
        print(f"  S33 LR CHOICE config {c}: {STATE['notes'][c]}", flush=True)
    st["meta"]["choice"] = dict(STATE["mlr"])
    ec.save_store(valid_name(), st)
    apply_cut(st)
    return STATE


def apply_cut(st=None):
    st = ec.load_store(valid_name()) if st is None else st
    cut = st["meta"].get("cut")
    STATE["cut"] = cut
    if cut and cut.get("cut"):
        for k in ("D8_SLOW", "D8_HINGE"):
            ARMS[k]["seeds"] = tuple(s for s in ARMS[k]["seeds"] if s in CUT_SEEDS)
    return cut


# ── projection ───────────────────────────────────────────────────────────────
def projection(workers, dry=False):
    """S33's runs alone, worst case (every run to its iters). Over CUT_H h: (b) cut to 260-264 (stored)."""
    st = ec.load_store(valid_name())
    which = [[a["cfg"], a["kind"]] for a in ARMS.values()]
    with ThreadPoolExecutor(max_workers=len(which)) as ex:          # one child per arm at once: the pool's load
        t = {k: v for r in ex.map(lambda w: child("timing", dict(mlr=VALID_LRS[0], which=[w], steps=TIME_STEPS)), which)
             for k, v in r.items()}
    durs, total = [], 0.0
    rows = []
    for k, a in ARMS.items():
        ts, te, _ = t[f"{a['cfg']}|{a['kind']}"]
        per = a["iters"] * ts + (a["iters"] // EVAL_EVERY) * te
        n = len(a["seeds"])
        durs += [per] * n
        total += per * n
        rows.append((k, n, per, ts, te))
    ms = c2.makespan(durs, workers)
    print(f"  S33 PROJECTION (worst case: every run to its iters; MuonAdam timed in the child: the median "
          f"interval between optimizer steps over {TIME_STEPS['a']} (a) / {TIME_STEPS['b']} (b) steps, one child per arm running at once ({len(which)} processes, the pool's load), "
          f"1 thread each, this machine):")
    for k, n, per, ts, te in rows:
        print(f"    {NAME:<14} {k:<10} {n:>2} runs x {per / 60:5.1f} min  ({ARMS[k]['iters']} steps x {ts * 1e3:.1f} ms "
              f"+ {ARMS[k]['iters'] // EVAL_EVERY} evals x {te:.1f} s)")
    print(f"    S33 total {total / 3600:.2f} h of runs; makespan on {workers} workers {ms / 3600:.2f} h (worst case)",
          flush=True)
    cut = st["meta"].get("cut")
    if cut is None:
        do = ms / 3600 > CUT_H
        cut = dict(cut=do, makespan_h=ms / 3600, workers=workers, time=time.strftime("%Y-%m-%d %H:%M:%S UTC",
                                                                                          time.gmtime()))
        if not dry:
            st["meta"]["cut"] = cut
            ec.save_store(valid_name(), st)
        if do:
            for k in ("D8_SLOW", "D8_HINGE"):
                ARMS[k]["seeds"] = tuple(s for s in ARMS[k]["seeds"] if s in CUT_SEEDS)
        print(f"  S33 RUNTIME RULE: projected makespan {ms / 3600:.2f} h {'>' if do else '<='} {CUT_H:g} h: "
              + (f"(b) cut to {CUT_SEEDS[0]}-{CUT_SEEDS[-1]} per arm ((a) never cut)" if do else "no cut")
              + (" (dry run: not stored)" if dry else " (stored)"), flush=True)
    else:
        print(f"  S33 RUNTIME RULE (stored {cut['time']}: makespan {cut['makespan_h']:.2f} h on {cut['workers']} "
              f"workers): " + (f"(b) cut to {CUT_SEEDS[0]}-{CUT_SEEDS[-1]} per arm" if cut["cut"] else "no cut"),
              flush=True)
    return ms


# ── CHECKs ───────────────────────────────────────────────────────────────────
def check():
    t0 = time.time()
    rows = child("checks", dict(mlr=VALID_LRS[0]))
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    print(f"  CHECK {NAME}: child at {mt.MAIN_SHA[:7]} ({(time.time() - t0) / 60:.1f} min); served from main: "
          f"{', '.join(mt.SERVED)}", flush=True)
    diff = mt.differing()
    v = diff == sorted(mt.SERVED)
    print(f"  CHECK {NAME}: the modules served from main are exactly the test modules that differ between this "
          f"branch and {mt.MAIN_SHA[:7]} ({diff}): {'ok' if v else 'FAIL'}", flush=True)
    return ok and v


# ── Report ───────────────────────────────────────────────────────────────────
def recorded():
    if STATE["recorded"] is None:
        STATE["recorded"] = child("recorded_summary", dict(seeds=dict(A4k4=list(range(240, 250)),
                                                                      HINGE_D8=list(range(260, 270)))))
    return STATE["recorded"]


def map_at(r, step):
    """routing_k's ch_map (stream -> channel) at step; a run that stopped before keeps its last map."""
    maps = r.get("maps") or {}
    if step == "end":
        return r["end"].get("ch_map")
    if str(step) in maps:
        return maps[str(step)]
    if r.get("stopped_at") is not None and r["stopped_at"] < step:
        return r["end"].get("ch_map")
    return None


def distinct(m):
    return None if m is None else len(set(m))


def per_channel(m):
    if m is None:
        return "--"
    cnt = {}
    for ch in m:
        cnt[ch] = cnt.get(ch, 0) + 1
    return "+".join(str(v) for v in sorted(cnt.values(), reverse=True))


def merged_end(r, S):
    m = r["end"].get("ch_map")
    return m is not None and len(set(m)) < S


def hinge_n(r):
    h = r["end"].get("hs_hinge")
    return "--" if h is None else f"{h[0]}"


def median_distinct(rs):
    xs = [distinct(map_at(r, "end")) for r in rs if map_at(r, "end") is not None]
    return (statistics.median(xs) if xs else None), xs


def compare(new, old, seeds, lab_new, lab_old):
    both = [s for s in seeds if s in new and s in old]
    b = sum(1 for s in both if new[s] and not old[s])
    c = sum(1 for s in both if old[s] and not new[s])
    p = mcnemar_exact(b, c)
    print(f"    BOUND  {lab_new} {sum(new[s] for s in both)}/{len(both)}  {lab_old} {sum(old[s] for s in both)}/"
          f"{len(both)}  {lab_new} only {b}, {lab_old} only {c}, McNemar two-sided p = {p:.3g} (printed)")
    return dict(n=len(both), new=sum(new[s] for s in both), b=b, c=c, p=p)


def validity_report():
    st = ec.load_store(valid_name())
    for c in CFG:
        rows = sorted((k, r) for k, r in st["runs"].items() if k.startswith(c + "|"))
        print(f"    validity config {c} ({CFG[c]['label']}): " + "; ".join(
            f"Muon lr {k.split('|')[1]} seed {k.split('|')[2]} {('BOUND at ' + str(r['transition'])) if ec.bound(r) else 'unbound'}"
            f" (acc {r['acc']:.3f})" if r.get("ok") else f"{k} {ec.tag(r)}" for k, r in rows)
            + f" -> {STATE['notes'].get(c, '--')}")


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    validity_report()
    rec = recorded()
    cut = STATE.get("cut") or {}
    out = dict(untested=dict(STATE["untested"]), mlr=dict(STATE["mlr"]), cut=bool(cut.get("cut")))
    for c in CFG:
        S = CFG[c]["S"]
        arms = [k for k, a in ARMS.items() if a["cfg"] == c]
        seeds = ARMS[arms[0]]["seeds"] if not STATE["untested"].get(c) else ()
        allseeds = tuple(range(240, 250)) if c == "a" else tuple(range(260, 270))
        print(f"  CONFIG ({c}) {CFG[c]['label']}; Muon lr {STATE['mlr'].get(c)}; seeds {ec.fmt_seeds(seeds) if seeds else 'none'}"
              + ("; UNTESTED" if STATE["untested"].get(c) else ""))
        got = {k: {int(kk.split("|")[1]): r for kk, r in store["runs"].items() if kk.startswith(k + "|") and r.get("ok")}
               for k in arms}
        xk = CFG[c]["x_key"]
        xr = {s: rec[f"{xk}|{s}"] for s in allseeds if f"{xk}|{s}" in rec}
        print(f"    {'seed':>4} | {CFG[c]['x_name']:<34} | " + " | ".join(
            f"{'MUON_' + ARMS[k]['kind']:<26} {'trans':>5} {'acc':>5} {'ch 4800/9600/end':>16} {'per ch end':>11}"
            + (" fired" if ARMS[k]["kind"] == "HINGE" else "") for k in arms))
        for s in allseeds:
            x = xr.get(s)
            xp = dict(end=dict(ch_map=x["end_map"]), maps=x["maps"], stopped_at=x.get("stopped_at")) if x else None
            xs = (f"{x['outcome']:<22} {str(x['transition']):>5} "
                  f"{'/'.join(str(distinct(map_at(xp, t))) if map_at(xp, t) else '--' for t in STEPS_DIAG):>7}"
                  if x else f"{'--':<34}")
            cells = []
            for k in arms:
                r = got[k].get(s)
                if r is None:
                    cells.append(f"{'not run' if s not in ARMS[k]['seeds'] else 'missing':<26} {'':>5} {'':>5} {'':>16} "
                                 f"{'':>11}" + ("      " if ARMS[k]["kind"] == "HINGE" else ""))
                    continue
                dd = "/".join(str(distinct(map_at(r, t))) if map_at(r, t) else "--" for t in STEPS_DIAG)
                cells.append(f"{r['outcome']:<26} {str(r['transition']):>5} {r['acc']:5.3f} {dd:>16} "
                             f"{per_channel(map_at(r, 'end')):>11}" + (f" {hinge_n(r):>5}" if ARMS[k]["kind"] == "HINGE" else ""))
            print(f"    {s:>4} | {xs:<34} | " + " | ".join(cells))
        xb = {s: x["bound"] for s, x in xr.items()}
        res = {}
        for k in arms:
            rs = got[k]
            b = {s: ec.bound(r) for s, r in rs.items()}
            print(f"  {k} (MUON_{ARMS[k]['kind']}) vs {CFG[c]['x_name']}:")
            d = compare(b, xb, seeds, k, CFG[c]["x_name"])
            med, xs = median_distinct(list(rs.values()))
            mg = [s for s, r in sorted(rs.items()) if merged_end(r, S)]
            cls = {}
            for r in rs.values():
                cls[r["outcome"]] = cls.get(r["outcome"], 0) + 1
            print(f"    merged at the end (two streams on one channel): {len(mg)}/{len(rs)} {mg}; outcomes {cls}")
            for t in STEPS_DIAG:
                ds = [distinct(map_at(r, t)) for r in rs.values() if map_at(r, t) is not None]
                print(f"    distinct channels holding the {S} streams at {t}: {ds} (median "
                      f"{statistics.median(ds) if ds else '--'}); streams per channel "
                      f"{[per_channel(map_at(r, t)) for _, r in sorted(rs.items())]}")
            if ARMS[k]["kind"] == "HINGE":
                fired = {s: r["end"].get("hs_hinge") for s, r in sorted(rs.items())}
                print(f"    hinge firings (training batches with a term above TAU, of all): {fired}")
            res[k] = dict(d, n_run=len(rs), bound=sum(b.values()), median_end=med, distinct_end=xs, merged=len(mg),
                          complete=len(rs) == len(ARMS[k]["seeds"]))
        xmed, xds = median_distinct([dict(end=dict(ch_map=x["end_map"]), maps=x["maps"], stopped_at=x.get("stopped_at"))
                                     for x in xr.values()])
        print(f"    {CFG[c]['x_name']}: BOUND {sum(xb.values())}/{len(xb)}; distinct channels at the end {xds} "
              f"(median {xmed}); outcomes {[x['outcome'] for _, x in sorted(xr.items())]}")
        if STATE["untested"].get(c):
            rd = "UNTESTED (the perfect gate bound neither lr on both seeds)"
        elif c == "a":
            best = max(res.values(), key=lambda v: v["bound"])
            kbest = [k for k, v in res.items() if v is best][0]
            rd = ("promising at four streams" if best["bound"] >= PROMISING_A else "not promising at four streams") + \
                 f" (better Muon arm {kbest} {best['bound']}/{best['n_run']}; >= {PROMISING_A}/10 needed)"
        else:
            hits = [k for k, v in res.items() if v["bound"] >= BREAK_N
                    or (v["median_end"] is not None and v["median_end"] >= BREAK_MED)]
            rd = ("breaks the coarse split" if hits else "does not break the coarse split") + " (" + "; ".join(
                f"{k} BOUND {v['bound']}/{v['n_run']}, median distinct channels at the end {v['median_end']}"
                for k, v in res.items()) + f"; X HINGE_D8 median {xmed}; >= {BREAK_N} bound or median >= {BREAK_MED})"
        print(f"  READING S33 ({c}): {rd}")
        out[c] = dict(arms=res, reading=rd, x_median=xmed, complete=all(v["complete"] for v in res.values()))
    return out
