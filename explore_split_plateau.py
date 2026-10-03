#!/usr/bin/env python
"""
explore_split_plateau.py — EXPLORATORY, not a result. Screen S36 (batch 12): a label-free plateau trigger
that applies S24's SPLIT (copy the busiest gate row to the idlest, plus noise; reset W_g's optimizer
state) to merged or stalled runs, at four streams under Muon and at eight streams under Muon and Adam.

BACKGROUND (docstring)
- S24: copying the busiest gate row to the idlest un-merged 8/8 merged runs (NOISE 1/8).
- S33(a) (Muon, S=4, k=4): 7/10 merged by 4800 and to the end, in both arms.
- S33(b) and HINGE_D8 (S=8, k=16): 0/10 and 0/20, with streams on 1-2 channels.
- A mass threshold of 1.5/k cannot trigger the split label-free at k=16: a correct 8-stream gate puts
  1/8 on each channel.

SPLIT: S24's operation (explore_split_child.split_op): c* = the busiest channel, c0 = the idlest (mean
read-gate mass over all probe positions); W_g[c0] <- W_g[c*]; noise N(0, (0.1 std(W_g[c*]))^2) on both
rows (the model has no gate bias); W_g's optimizer state reset (Adam: moments and step; Muon: the momentum
buffer), zeroed in place.
PROBE: 64 sequences of the arm's task from a generator seeded 12345, the same everywhere.
TRIGGER (label-free, from the task only): at updates 4800, 7200, ... (every 2400) through total - 2400
(26400 for (a), 40800 for (b)), split once if the probe accuracy is < 0.95 and it rose by < 0.02 since
the previous check (the probe accuracy is also read at 2400 as the first previous value; no split there).
The check runs after that evaluation's statistics and the run path's own check.

ARMS (main's run paths at 9c5939e in a child process, as S33; explore_split_child):
  (a) S=4, P=4, k=4, conv, 28800 steps, seeds 240-249.
      SPLIT4k4_M  S33's A_HINGE (S25's groups, Muon lr 0.005, main's hinge) + the trigger; paired with
                  A_HINGE (S33, 4/10).
  (b) S=8, P=4, k=16, conv, all 8 streams from step 1, 43200 steps, checks through 40800, seeds 260-269.
      SPLIT_D8_M  S33's D8_HINGE + the trigger; paired with D8_HINGE (S33, 0/10).
      SPLIT_D8_A  main's HINGE_D8 (test_slow_start.make_recipe(True, TAU): Adam, slow groups, the hinge)
                  + the trigger; paired with X's recorded HINGE_D8 (slow_start at b8c6007, 0/10).
The Adam groups at test_slow_start.LR = 1e-3 (read in the child); Muon lr 0.005 (S33's validity choice).
READINGS (fixed before any run; BOUND = test_binding_onset.transition on the curve):
  (a) "the split fixes four-stream merges under Muon" if >= 8/10 bind; "it does not" if <= 5/10;
      otherwise neither.
  (b) per arm: "it breaks the eight-stream stall" if >= 3/10 bind by 43200; otherwise "it does not".
      Printed: BOUND at 28800 (the transition on the curve through 28800: the paired outcome) and at 43200.
DIAGNOSTICS: per check: probe accuracy, channel masses, fired or not, c* and c0; splits per run; the
distinct channels holding the streams (test_scale_axes.routing_k's ch_map on the run's labelled probe)
before and right after each split and at the next check; HARM: runs whose held-out accuracy (the run's
curve) 2400 updates after a split is > 0.05 below the value at the split.
RUNTIME: projection with pool-load timing (batch 10's method); if over 9 h, SPLIT_D8_A is cut to 260-264
(stored with the CHECK-free state below, so a resume uses the same seeds).
CHECKS (child processes): with the accuracy threshold at 0 (never fires; the trigger's measurements still
run at 2400, 4800, 7200), each arm equals its paired run bit for bit through 7200 (curve and statistics);
on a synthetic gate under populated Muon and Adam states the operation sets the rows as specified and
leaves every other parameter and optimizer state unchanged; after it W_g's Muon momentum (Adam state) is
zero; with the thresholds forced (accuracy < 1.01, rise < 10), SPLIT4k4_M splits at 4800 and runs on.
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
import explore_muon_scale as s33
from test_channel_binding import SUB_LR
from test_router_confirm import mcnemar_exact
from test_binding_onset import EVAL_EVERY

NAME = "split_plateau"
IDEA = "a label-free plateau trigger that applies S24's SPLIT to merged or stalled multi-stream runs"
SOURCE = ("batch 8's S24 (SPLIT un-merged 8/8 merged runs); batch 10's S33 (Muon keeps merges at S=4 k=4; the "
          "coarse split at S=8)")
CHANGE = ("every 2400 updates from 4800: if probe accuracy < 0.95 and rose < 0.02, W_g[idlest] <- W_g[busiest] + "
          "noise (both rows, 0.1 std), W_g's optimizer state reset")
PAIRING = ("(a) 240-249 with S33's A_HINGE; (b) 260-269 with S33's D8_HINGE (Muon) and X's recorded HINGE_D8 "
           "(Adam); same initial parameters and batches until the first split")
CHILD = "explore_split_child"
LR = SUB_LR
MUON_LR = 0.005
FIX_N, NOT_N, BREAK_N = 8, 5, 3
HARM_DROP = 0.05
CUT_H, CUT_SEEDS = 9.0, tuple(range(260, 265))
EQ_AT = 7200
TIME_STEPS = {"a": 300, "b": 150}
STATE = dict(cut=None, xrec=None)
_trig = "; + the trigger (every 2400 from 4800: probe acc < 0.95 and rise < 0.02 -> S24's SPLIT, W_g state reset)"
_mu = ("Muon lr 0.005: gate W_in/W_h/W_g throughout, other Muon weights x0.1 for updates 1-2400; Adam on embedding "
       "and conv 0.0001 for 1-2400, 0.001 after; + main's HINGE hinge")
ARMS = {
    "SPLIT4k4_M": dict(key="SPLIT4k4_M", cfg="a", opt="muon", seeds=tuple(range(240, 250)), lr=LR, iters=28800,
                       muon_lr=MUON_LR, prio=1, label="S33's A_HINGE + the trigger (S=4, P=4, k=4, conv)",
                       sched=_mu + _trig),
    "SPLIT_D8_M": dict(key="SPLIT_D8_M", cfg="b", opt="muon", seeds=tuple(range(260, 270)), lr=LR, iters=43200,
                       muon_lr=MUON_LR, prio=2, label="S33's D8_HINGE + the trigger (S=8, P=4, k=16, conv)",
                       sched=_mu + _trig),
    "SPLIT_D8_A": dict(key="SPLIT_D8_A", cfg="b", opt="adam", seeds=tuple(range(260, 270)), lr=LR, iters=43200,
                       muon_lr=None, prio=2, label="main's HINGE_D8 + the trigger (S=8, P=4, k=16, conv; Adam)",
                       sched="gate W_in/W_h/W_g 0.001 throughout; every other parameter 0.0001 for updates 1-2400, "
                             "0.001 after; + main's HINGE hinge" + _trig),
}
PAIR = {"SPLIT4k4_M": "A_HINGE", "SPLIT_D8_M": "D8_HINGE", "SPLIT_D8_A": "X HINGE_D8"}


def child(func, payload, module=CHILD, timeout=None):
    return mt.run_child(module, func, payload, timeout=timeout)


def run_job(arm, seed):
    a = ARMS[arm]
    return child("run", dict(arm=arm, seed=seed, iters=a["iters"], mlr=a["muon_lr"]))


def segments(arm):
    raise NotImplementedError("S36 is projected by its own child timing (explore_batch12)")


def s33_choice():
    ch = ec.load_store("muon_scale_valid")["meta"].get("choice") or {}
    return ch


# ── paired records ───────────────────────────────────────────────────────────
def paired(arm):
    if arm in ("SPLIT4k4_M", "SPLIT_D8_M"):
        key = PAIR[arm]
        return {int(k.split("|")[1]): r for k, r in ec.load_store(s33.NAME)["runs"].items()
                if k.startswith(key + "|") and r.get("ok")}
    if STATE["xrec"] is None:
        STATE["xrec"] = mt.recorded("slow_start", mt.RESULTS_SHA)
    return {s: STATE["xrec"][f"HINGE_D8|{s}"] for s in range(260, 270)}


def x_outcomes():
    return child("recorded_summary", dict(seeds=dict(A4k4=[], HINGE_D8=list(range(260, 270)))),
                 module="explore_muon_scale_child")


# ── runtime rule ─────────────────────────────────────────────────────────────
def rule_name():
    return NAME + "_runtime"


def apply_cut():
    st = ec.load_store(rule_name())
    cut = st["meta"].get("cut")
    STATE["cut"] = cut
    if cut and cut.get("cut"):
        ARMS["SPLIT_D8_A"]["seeds"] = tuple(s for s in ARMS["SPLIT_D8_A"]["seeds"] if s in CUT_SEEDS)
    return cut


def decide_cut(h, workers, dry=False):
    st = ec.load_store(rule_name())
    cut = st["meta"].get("cut")
    if cut is None:
        do = h > CUT_H
        cut = dict(cut=do, makespan_h=h, workers=workers, time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
        if not dry:
            st["meta"]["cut"] = cut
            st["meta"].setdefault("provenance", ec.provenance())
            ec.save_store(rule_name(), st)
        print(f"  RUNTIME RULE: projected makespan {h:.2f} h {'>' if do else '<='} {CUT_H:g} h: "
              + (f"SPLIT_D8_A cut to {CUT_SEEDS[0]}-{CUT_SEEDS[-1]}" if do else "no cut")
              + (" (dry run: not stored)" if dry else " (stored)"), flush=True)
    else:
        print(f"  RUNTIME RULE (stored {cut['time']}: makespan {cut['makespan_h']:.2f} h on {cut['workers']} workers): "
              + (f"SPLIT_D8_A cut to {CUT_SEEDS[0]}-{CUT_SEEDS[-1]}" if cut["cut"] else "no cut"), flush=True)
    STATE["cut"] = cut
    if cut["cut"]:
        ARMS["SPLIT_D8_A"]["seeds"] = tuple(s for s in ARMS["SPLIT_D8_A"]["seeds"] if s in CUT_SEEDS)
    return cut


def timing(arm):
    return child("timing", dict(which=[arm], steps=TIME_STEPS, mlr=MUON_LR))[arm]


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _eq_job(arm):
    seed = 240 if arm == "SPLIT4k4_M" else 260
    r = child("run", dict(arm=arm, seed=seed, iters=EQ_AT, mlr=MUON_LR if ARMS[arm]["opt"] == "muon" else None, thr=0.0))
    ref = paired(arm)[seed]
    want = [c for c in ref["curve"] if c[0] <= EQ_AT]
    sn = {s["step"]: s for s in r["stats"] if s["step"] != "end" and s["step"] <= EQ_AT}
    sr = {s["step"]: s for s in ref["stats"] if s["step"] != "end" and s["step"] <= EQ_AT}
    same_st = sn.keys() == sr.keys() and all(sn[k] == sr[k] for k in sr)
    meas = [(c["step"], round(c["acc"], 3), c["fired"]) for c in r["checks"]]
    return (f"{arm} with the accuracy threshold at 0 equals {PAIR[arm]}|{seed} bit for bit through {EQ_AT} (curve "
            f"{r['curve'] == want}; statistics at {sorted(sr)} {same_st}; the trigger measured at {meas})",
            r["curve"] == want and same_st and all(not f for _, _, f in meas) and len(meas) == 3)


def _forced_job(_):
    r = child("run", dict(arm="SPLIT4k4_M", seed=241, iters=6000, mlr=MUON_LR, thr=1.01, rise=10.0))
    ch = r["checks"]
    fired = [c["step"] for c in ch if c["fired"]]
    f = next((c for c in ch if c["fired"]), {})
    return (f"with the thresholds forced (accuracy < 1.01, rise < 10) SPLIT4k4_M|241 splits at {fired} (c* {f.get('c_star')} "
            f"-> c0 {f.get('c0')}; channels {f.get('map_before')} -> {f.get('map_after')}; W_g state zeroed "
            f"{f.get('state_zeroed')}) and runs on to {r['stopped_at']} (curve {r['curve'][-1]})",
            fired == [4800] and r["stopped_at"] == 6000 and f.get("state_zeroed") == ["momentum_buffer"])


def _synth_job(_):
    return child("synthetic", {})


def check():
    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        eq = [ex.submit(_eq_job, a) for a in ARMS]
        fo = ex.submit(_forced_job, None)
        sy = ex.submit(_synth_job, None)
        rows += [f.result() for f in eq]
        rows += [tuple(x) for x in sy.result()]
        rows.append(fo.result())
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    diff = mt.differing()
    v = diff == sorted(mt.SERVED)
    print(f"  CHECK {NAME}: children at {mt.MAIN_SHA[:7]} ({(time.time() - t0) / 60:.1f} min); the modules served from main "
          f"are exactly the test modules that differ between this branch and {mt.MAIN_SHA[:7]} ({diff}): "
          f"{'ok' if v else 'FAIL'}", flush=True)
    return ok and v


# ── Report ───────────────────────────────────────────────────────────────────
def curve_acc(r, step):
    for s, a, _ in r["curve"]:
        if s == step:
            return a
    return None


def harm(r):
    """(split step, acc at it, acc 2400 later) for the splits followed by a drop > HARM_DROP."""
    out = []
    for c in r.get("checks") or []:
        if c["fired"]:
            a0, a1 = curve_acc(r, c["step"]), curve_acc(r, c["step"] + 2400)
            if a0 is not None and a1 is not None and a1 < a0 - HARM_DROP:
                out.append((c["step"], round(a0, 3), round(a1, 3)))
    return out


def fmt_masses(ms):
    return "[" + " ".join(f"{v:.2f}" for v in ms) + "]"


def compare(new, old, seeds, lab_new, lab_old, what="BOUND"):
    both = [s for s in seeds if s in new and s in old]
    b = sum(1 for s in both if new[s] and not old[s])
    c = sum(1 for s in both if old[s] and not new[s])
    p = mcnemar_exact(b, c)
    print(f"    {what:<12} {lab_new} {sum(new[s] for s in both)}/{len(both)}  {lab_old} {sum(old[s] for s in both)}/"
          f"{len(both)}  {lab_new} only {b}, {lab_old} only {c}, McNemar two-sided p = {p:.3g} (printed)")
    return dict(n=len(both), new=sum(new[s] for s in both), old=sum(old[s] for s in both), b=b, c=c, p=p)


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    cut = bool((STATE.get("cut") or {}).get("cut"))
    xo = x_outcomes()
    out = dict(cut=cut)
    for arm, a in ARMS.items():
        got = {int(k.split("|")[1]): r for k, r in store["runs"].items() if k.startswith(arm + "|") and r.get("ok")}
        pr = paired(arm)
        seeds = a["seeds"]
        S = 4 if a["cfg"] == "a" else 8
        print(f"  ARM {arm} ({a['label']}); seeds {ec.fmt_seeds(seeds) if seeds else 'none'}; paired with {PAIR[arm]}")
        print(f"    {'seed':>4} | {arm:<26} {'trans':>5} {'acc':>5} {'B28800':>6} {'splits':>6} {'ch end':>6} {'per ch':>10} | "
              f"{PAIR[arm]:<26} {'trans':>5} {'ch end':>6}")
        for s in seeds:
            r, q = got.get(s), pr.get(s)
            if r is None:
                c1 = f"{'missing':<26} {'':>5} {'':>5} {'':>6} {'':>6} {'':>6} {'':>10}"
            else:
                me_ = s33.map_at(r, "end")
                c1 = (f"{r['outcome']:<26} {str(r['transition']):>5} {r['acc']:5.3f} {'Y' if r['bound_28800'] else '-':>6} "
                      f"{r['splits']:>6} {str(s33.distinct(me_)):>6} {s33.per_channel(me_):>10}")
            if q is None:
                c2_ = "--"
            else:
                oc = q.get("outcome") or (xo.get(f"HINGE_D8|{s}") or {}).get("outcome", "--")
                c2_ = f"{oc:<26} {str(q['transition']):>5} {str(s33.distinct(q['end'].get('ch_map'))):>6}"
            print(f"    {s:>4} | {c1} | {c2_}")
        b43 = {s: ec.bound(r) for s, r in got.items()}
        b28 = {s: bool(r["bound_28800"]) for s, r in got.items()}
        bp = {s: q["transition"] is not None for s, q in pr.items()}
        print(f"  {arm} vs {PAIR[arm]} (paired):")
        d28 = compare(b28, bp, seeds, arm, PAIR[arm], "BOUND@28800")
        d = compare(b43, bp, seeds, arm, PAIR[arm], f"BOUND@{a['iters']}")
        print(f"  {arm} checks per run (step: probe acc, '*' = split c*->c0; masses at each check):")
        for s in seeds:
            r = got.get(s)
            if r is None:
                continue
            ch = r.get("checks") or []
            print(f"    {s}: " + "  ".join(f"{c['step']}:{c['acc']:.3f}" + (f"*{c['c_star']}->{c['c0']}" if c["fired"] else "")
                                         for c in ch))
            for c in ch:
                print(f"         {c['step']:>5} acc {c['acc']:.3f} {'FIRED' if c['fired'] else 'no   '} c* {c['c_star']:>2} c0 {c['c0']:>2} "
                      f"masses {fmt_masses(c['masses'])}")
        sp = {s: r["splits"] for s, r in sorted(got.items())}
        print(f"  {arm} splits per run: {sp} (total {sum(sp.values())}; runs with a split {sum(1 for v in sp.values() if v)}/"
              f"{len(sp)})")
        print(f"  {arm} distinct channels holding the {S} streams (labelled: routing_k's ch_map on the run's probe) before -> "
              f"right after -> at the next check, per split:")
        for s, r in sorted(got.items()):
            rows = []
            for c in r.get("checks") or []:
                if c["fired"]:
                    nxt = s33.map_at(r, c["step"] + 2400)
                    rows.append(f"{c['step']} {c['c_star']}->{c['c0']}: {s33.distinct(c['map_before'])} "
                                f"({s33.per_channel(c['map_before'])}) -> {s33.distinct(c['map_after'])} "
                                f"({s33.per_channel(c['map_after'])}) -> {s33.distinct(nxt) if nxt else '--'}"
                                + (f" ({s33.per_channel(nxt)})" if nxt else ""))
            if rows:
                print(f"    {s}: " + "; ".join(rows))
        hm = {s: harm(r) for s, r in sorted(got.items()) if harm(r)}
        print(f"  {arm} HARM (held-out accuracy 2400 updates after a split > {HARM_DROP} below the value at it): "
              f"{len(hm)}/{len(got)} runs {hm}")
        n_b = sum(b43.values())
        if arm == "SPLIT4k4_M":
            rd = ("the split fixes four-stream merges under Muon" if n_b >= FIX_N else
                  "it does not" if n_b <= NOT_N else "neither") + f" (BOUND {n_b}/{len(got)}; >= {FIX_N} fixes, <= {NOT_N} not)"
        else:
            rd = ("it breaks the eight-stream stall" if n_b >= BREAK_N else "it does not") + \
                 f" (BOUND by {a['iters']} {n_b}/{len(got)}, at 28800 {sum(b28.values())}/{len(got)}; >= {BREAK_N} needed)"
        print(f"  READING S36 {arm}: {rd}")
        out[arm] = dict(d, d28=d28, bound=n_b, bound28=sum(b28.values()), n=len(got), reading=rd, harm=len(hm),
                        splits=sum(sp.values()), complete=len(got) == len(seeds))
    return out
