#!/usr/bin/env python
"""
explore_gate_prev.py — EXPLORATORY, not a result. Screen S40 (batch 14): give the recurrent gate the previous
token's embedding (GATE_PREV), so the stream token that precedes every key reaches the gate at the key without
having to be kept in h; S=8, P=4, k=16, conv, all 8 streams from step 1, 28800 steps.

BACKGROUND (docstring): see explore_gate_memory (the grouped layout puts the stream token immediately before
its key; S38: at S=8 the stream is not decodable from h at key positions at 4800, 0.13 / 0.14 at chance 0.125).

GATE_PREV: the gate's pre-activation becomes W_in v_t + W_prev v_{t-1} + W_h h_{t-1} (v_{-1} = 0); W_prev from
its own generator (seed 40_000_000 + seed) at W_in's init scale (N(0, 0.1^2)); every other parameter's initial
value unchanged (the run path's own build); W_prev in the gate group (explore_gate_prev_child).
ARMS (main's run_sc with D8 at 9c5939e, child process):
  D8_PREV_M  S33's D8_HINGE (Muon lr 0.005, S25's groups, main's hinge) + GATE_PREV, seeds 260-269; paired with
             S33's D8_HINGE (0/10 bound).
  D8_PREV_A  X's HINGE_D8 (test_slow_start's HINGE recipe, Adam) + GATE_PREV, seeds 260-269; paired with X's
             HINGE_D8 (0/10 bound).
OUTCOME: BOUND ROUTED at 28800 (test_stream_recipe.outcome).
READINGS (per arm; fixed before any run): "the look-back breaks the eight-stream stall" if >= 3/10 BOUND ROUTED;
"it does not" if 0/10; otherwise neither. (Under the runtime cut D8_PREV_A has 5 seeds; the same counts apply,
labelled as of 5.)
DIAGNOSTICS: S38's decodability (nearest class mean, S38's 512-sequence probe, seed 38000) of h (with the W_prev
term) at key and at stream-token positions at 1200, 2400, 4800, 9600; distinct channels and streams per channel
at 4800, 9600 and the end; failure classes; the hinge's firings (cumulative training batches with a term > 0).
CHECKS (child processes): at update 0 every parameter shared with the paired run is bitwise equal; with W_prev
fixed at 0 and excluded from the optimizer each arm equals its paired run bit for bit through 1200 (curve and
statistics: D8_PREV_M vs S33's D8_HINGE|260, D8_PREV_A vs X's HINGE_D8|260); the gate's input at key positions
decodes the stream at >= 0.9 at update 0 with GATE_PREV and at chance (within 0.05) without it; the optimizer
groups cover every parameter once, W_prev in the gate group.
RUNTIME (batch level): if the projection is over 9 h, D8_PREV_A is cut to seeds 260-264.
"""

import os
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common2 as c2
import explore_main9c as mt
import explore_muon_scale as s33
from test_channel_binding import SUB_LR
from test_router_confirm import mcnemar_exact

NAME = "gate_prev"
IDEA = "GATE_PREV: the recurrent gate also reads the previous token's embedding (W_prev v_(t-1) in its pre-activation)"
SOURCE = ("batch 13's S38 (at S=8 the gate state does not carry the stream at key positions; the stream token is the "
          "token right before every key)")
CHANGE = ("pre-activation W_in v_t + W_prev v_(t-1) + W_h h_(t-1), v_(-1) = 0; W_prev ~ N(0, 0.1^2) from its own generator; "
          "W_prev in the gate group; nothing else")
PAIRING = ("seeds 260-269: D8_PREV_M with S33's D8_HINGE, D8_PREV_A with X's HINGE_D8; every shared parameter's initial "
           "value and the batch stream are the paired run's")
CHILD = "explore_gate_prev_child"
LR = SUB_LR
MUON_LR = 0.005
SEEDS = tuple(range(260, 270))
CUT_SEEDS = tuple(range(260, 265))
BREAK_N, NOT_N = 3, 0
EQ_AT = 1200
AT = (1200, 2400, 4800, 9600)
MAP_AT = (4800, 9600, "end")
TIME_STEPS = 200
_mu = ("Muon lr 0.005: gate W_in/W_h/W_g/W_prev throughout, other Muon weights x0.1 for updates 1-2400; Adam on embedding "
       "and conv 0.0001 for 1-2400, 0.001 after; + main's HINGE hinge")
_ad = ("gate W_in/W_h/W_g/W_prev 0.001 throughout; every other parameter 0.0001 for updates 1-2400, 0.001 after; + main's "
       "HINGE hinge")
ARMS = {
    "D8_PREV_M": dict(key="D8_PREV_M", opt="muon", seeds=SEEDS, lr=LR, iters=28800, muon_lr=MUON_LR, prio=2,
                      label="S33's D8_HINGE + GATE_PREV (Muon)", sched=_mu),
    "D8_PREV_A": dict(key="D8_PREV_A", opt="adam", seeds=SEEDS, lr=LR, iters=28800, muon_lr=None, prio=2,
                      label="X's HINGE_D8 + GATE_PREV (Adam)", sched=_ad),
}


def child(func, payload, timeout=None):
    return mt.run_child(CHILD, func, payload, timeout=timeout)


def run_job(arm, seed):
    a = ARMS[arm]
    return child("run", dict(arm=arm, seed=seed, iters=a["iters"], mlr=a["muon_lr"]))


def segments(arm):
    raise NotImplementedError("S40 is projected by its own child timing (explore_batch14)")


def timing(arms):
    return child("timing", dict(which=list(arms), steps=TIME_STEPS, mlr=MUON_LR))


def x_hinge_d8(seeds):
    """X's HINGE_D8 records summarised in a child on main's modules (outcome, maps)."""
    return mt.run_child("explore_muon_scale_child", "recorded_summary",
                        dict(seeds={"A4k4": [], "HINGE_D8": list(seeds)}))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _stats_upto(stats, t):
    return {s["step"]: s for s in stats if s["step"] != "end" and s["step"] <= t}


def _eq(arm):
    r = child("run", dict(arm=arm, seed=260, iters=EQ_AT, mlr=ARMS[arm]["muon_lr"], zero=True))
    if arm == "D8_PREV_M":
        ref, lab = ec.load_store(s33.NAME)["runs"]["D8_HINGE|260"], "S33's D8_HINGE|260"
    else:
        ref, lab = mt.recorded("slow_start", mt.RESULTS_SHA)["HINGE_D8|260"], "X's HINGE_D8|260"
    want = [c for c in ref["curve"] if c[0] <= EQ_AT]
    sn, sr = _stats_upto(r["stats"], EQ_AT), _stats_upto(ref["stats"], EQ_AT)
    same = sn.keys() == sr.keys() and all(sn[k] == sr[k] for k in sr)
    g = r.get("opt_groups")
    return (f"{arm} with W_prev fixed at 0 and excluded from the optimizer equals {lab} bit for bit through {EQ_AT} (curves "
            f"{r['curve']} vs {want}; statistics at {sorted(sr)} {same}; groups {g})",
            r["curve"] == want and same and len(sr) >= 2)


def check():
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as ex:
        fm, fa, fc = ex.submit(_eq, "D8_PREV_M"), ex.submit(_eq, "D8_PREV_A"), ex.submit(child, "checks", {})
        rows = [tuple(x) for x in fc.result()] + [fm.result(), fa.result()]
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    print(f"  CHECK {NAME}: children at {mt.MAIN_SHA[:7]} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def compare(new, old, seeds, lab_new, lab_old):
    both = [s for s in seeds if s in new and s in old]
    b = sum(1 for s in both if new[s] and not old[s])
    c = sum(1 for s in both if old[s] and not new[s])
    p = mcnemar_exact(b, c)
    print(f"    BOUND ROUTED  {lab_new} {sum(new[s] for s in both)}/{len(both)}  {lab_old} {sum(old[s] for s in both)}/{len(both)}  "
          f"{lab_new} only {b}, {lab_old} only {c}, McNemar two-sided p = {p:.3g} (printed)")
    return dict(n=len(both), new=sum(new[s] for s in both), old=sum(old[s] for s in both), b=b, c=c, p=p)


def fmt(x):
    return "  -- " if x is None else f"{x:5.2f}"


def hinge_at(r, step):
    for s in r.get("stats") or []:
        if s["step"] == step and s.get("hs_hinge") is not None:
            return s["hs_hinge"]
    if step == "end":
        return (r.get("end") or {}).get("hs_hinge")
    return None


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    got = {arm: {int(k.split("|")[1]): r for k, r in store["runs"].items() if k.startswith(arm + "|") and r.get("ok")}
           for arm in ARMS}
    d8h = {int(k.split("|")[1]): r for k, r in ec.load_store(s33.NAME)["runs"].items() if k.startswith("D8_HINGE|")}
    xs = x_hinge_d8(SEEDS)
    xh = {int(k.split("|")[1]): v for k, v in xs.items() if k.startswith("HINGE_D8|")}
    paired = {"D8_PREV_M": ({s: r["outcome"] for s, r in d8h.items()}, "S33 D8_HINGE"),
              "D8_PREV_A": ({s: v["outcome"] for s, v in xh.items()}, "X HINGE_D8")}
    out = {}
    for arm, a in ARMS.items():
        rs = got[arm]
        po, plab = paired[arm]
        print(f"  ARM {arm} ({a['label']}), seeds {ec.fmt_seeds(a['seeds'])}; paired with {plab}")
        print(f"    {'seed':>4} | {arm:<24} {'trans':>5} {'stop':>5} {'ch@4800':>7} {'ch@9600':>7} {'ch@end':>7} "
              f"{'hinge fired/batches at end':>27} | {plab:<24}")
        for s in a["seeds"]:
            r = rs.get(s)
            if r is None:
                print(f"    {s:>4} | {'not run':<24}")
                continue
            hf = hinge_at(r, "end")
            print(f"    {s:>4} | {r['outcome']:<24} {str(r['transition']):>5} {r['stopped_at']:>5} "
                  + " ".join(f"{s33.per_channel(s33.map_at(r, t)):>7}" for t in MAP_AT)
                  + f" {('--' if hf is None else f'{hf[0]}/{hf[1]}'):>27} | {po.get(s, '--'):<24}")
        br = {s: r["outcome"] == "BOUND ROUTED" for s, r in rs.items()}
        bo = {s: o == "BOUND ROUTED" for s, o in po.items()}
        d = compare(br, bo, a["seeds"], arm, plab.replace(" ", "_"))
        print(f"    failure classes at the end: {arm} {dict(Counter(r['outcome'] for r in rs.values()))}; {plab} "
              f"{dict(Counter(po[s] for s in a['seeds'] if s in po))}")
        print(f"    distinct channels holding the 8 streams (routing_k's map; streams per channel), per seed:")
        for t in MAP_AT:
            ms = {s: s33.map_at(r, t) for s, r in sorted(rs.items())}
            print(f"      at {str(t):>5}: " + "  ".join(f"{s}:{s33.distinct(m) if m else '--'} ({s33.per_channel(m)})"
                                                  for s, m in ms.items()))
        print(f"    the hinge's firings (cumulative training batches with a term > 0 / training batches) at 2400, 4800, 9600, end:")
        print("      " + "  ".join(f"{s}: " + ", ".join((lambda h: '--' if h is None else f'{h[0]}/{h[1]}')(hinge_at(r, t))
                                                      for t in (2400, 4800, 9600, "end")) for s, r in sorted(rs.items())))
        print(f"    S38's decodability of the stream from h (with the W_prev term), at key (stream-token) positions; chance 0.125:")
        print(f"    {'seed':>4} | " + " | ".join(f"{t:>5}: key  (str)" for t in AT))
        for s, r in sorted(rs.items()):
            dd = r.get("decode") or {}
            print(f"    {s:>4} | " + " | ".join(
                f"{'':>5}  {fmt((dd.get(str(t)) or {}).get('h_key'))} ({fmt((dd.get(str(t)) or {}).get('h_stream'))})" for t in AT))
        meds = {}
        for t in AT:
            vk = [r["decode"][str(t)]["h_key"] for r in rs.values() if str(t) in (r.get("decode") or {})]
            vs = [r["decode"][str(t)]["h_stream"] for r in rs.values() if str(t) in (r.get("decode") or {})]
            meds[str(t)] = (statistics.median(vk) if vk else None, statistics.median(vs) if vs else None, len(vk))
        print(f"    medians: " + "; ".join(f"{t}: key {fmt(k).strip()}, stream-token {fmt(v).strip()} ({n} runs)"
                                         for t, (k, v, n) in meds.items()))
        nb = sum(br.values())
        n = len(rs)
        rd = ("the look-back breaks the eight-stream stall" if nb >= BREAK_N else "it does not" if nb <= NOT_N else "neither") + \
             f" (BOUND ROUTED {nb}/{n}; >= {BREAK_N} breaks, {NOT_N} does not)"
        print(f"  READING S40 {arm}: {rd}")
        out[arm] = dict(d, bound_routed=nb, n=n, reading=rd, complete=n == len(a["seeds"]), decode_medians=meds,
                        bound=sum(1 for r in rs.values() if r["transition"] is not None),
                        classes=dict(Counter(r["outcome"] for r in rs.values())))
    return out
