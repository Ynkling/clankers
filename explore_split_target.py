#!/usr/bin/env python
"""
explore_split_target.py — EXPLORATORY, not a result. Screen S37 (batch 13): S36's split with the target
chosen at the key positions the hinge uses (KEYMASS), capped at 2 splits per run, at S=4, k=4 under Muon
and under Adam.

BACKGROUND (docstring)
- S36(a), from S36's per-check records:
  - Of 31 splits, 15 copied a channel holding >= 2 streams onto one holding none.
  - All-position read-gate masses were 0.17-0.38; an empty channel can have the largest (241 at 4800: map
    [2,3,0,0], masses [0.30,0.34,0.18,0.18], c* = 1 (empty), c0 = 2 (a stream's channel)).
  - First splits: 3 correctly targeted (240 and 247 rescued; 244 unchanged); 4 mistargeted (241, 243, 249
    collapsed a singleton channel; 245 copied empty onto empty).
- S36(b): the target was empty in 97/98 (Muon) and 97/97 (Adam) splits, and the copied channel held >= 2
  streams in 68/98 and 88/97; no run bound by 43200.
- S24: SPLIT un-merged 8/8 Adam runs at k=4 from 9600.

THE CONFIGURATION: S=4, P=4, k=4, conv, 28800 steps (test_stream_recipe.run_attempt with A4k4 on main's
modules at 9c5939e, child process; explore_split_target_child), seeds 240-249.
KEYMASS targeting: c* = the channel with the largest mean read-gate mass at the key positions the hinge uses
(3j+1), on the probe (S36's: 64 sequences, generator 12345); c0 = the smallest there. At every check the
all-position rule's choice (S36's) and the labelled map before any split (routing_k's ch_map on the run's
probe) are logged too.
TRIGGER: S36's plateau rule (every 2400 from 4800: probe accuracy < 0.95 and a rise < 0.02 since the
previous check), at most 2 splits per run, none within 4800 updates of a previous split. S24's operation
(Muon: W_g's momentum reset; Adam: moments and step).
ARMS:
  SPLITK_M    S33's A_HINGE (Muon lr 0.005) + trigger + KEYMASS; paired with A_HINGE (S33, 4/10) and S36's
              SPLIT4k4_M (5/10).
  HINGE4k4_A  main's A4k4 + test_slow_start's HINGE recipe (Adam), run here (no trigger).
  SPLITK_A    HINGE4k4_A + trigger + KEYMASS; paired with HINGE4k4_A.
READINGS (per optimizer; BOUND; fixed before any run): "the targeted split fixes four-stream merges" if >= 8/10
bind; "it does not" if <= 5/10; otherwise neither. Printed: the labelled targeting accuracy of each rule (a
split, or a check, is correctly targeted when c* holds >= 2 streams and c0 holds none in the labelled map),
over the fired splits and over every check whose labelled map has a shared and an empty channel.
CHECKS (child processes): with the trigger off (threshold 0; its measurements still run), SPLITK_M equals
S33's A_HINGE|240 and SPLITK_A equals a fresh HINGE4k4_A|240 bit for bit through 7200 (curve and
statistics); on a synthetic gate whose key-position map is 2+1+1, KEYMASS picks the shared channel and the
empty one.
RUNTIME (batch level): if the projection is over 9 h, HINGE4k4_A and SPLITK_A are cut to 240-245.
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

NAME = "split_target"
IDEA = "S36's split targeted by the read-gate mass at the hinge's key positions (KEYMASS), at most 2 per run"
SOURCE = ("batch 12's S36 (the all-position target was wrong in half the splits: an empty channel can carry the "
          "largest mass); S24 (SPLIT un-merged 8/8 Adam runs at k=4)")
CHANGE = ("S36's trigger, c* / c0 = the largest / smallest mean read-gate mass at key positions, <= 2 splits, >= 4800 "
          "updates apart; S24's operation")
PAIRING = ("seeds 240-249: SPLITK_M with S33's A_HINGE and S36's SPLIT4k4_M; SPLITK_A with HINGE4k4_A (run here); same "
           "initial parameters and batches until the first split")
CHILD = "explore_split_target_child"
LR = SUB_LR
MUON_LR = 0.005
SEEDS = tuple(range(240, 250))
CUT_SEEDS = tuple(range(240, 246))
FIX_N, NOT_N = 8, 5
EQ_AT = 7200
HARM_DROP = 0.05
TIME_STEPS = 300
_trig = ("; + the trigger (every 2400 from 4800: probe acc < 0.95 and rise < 0.02; <= 2 splits, >= 4800 apart) -> S24's "
         "SPLIT on KEYMASS's c* -> c0, W_g's state reset")
_mu = ("Muon lr 0.005: gate W_in/W_h/W_g throughout, other Muon weights x0.1 for updates 1-2400; Adam on embedding and "
       "conv 0.0001 for 1-2400, 0.001 after; + main's HINGE hinge")
_ad = "gate W_in/W_h/W_g 0.001 throughout; every other parameter 0.0001 for updates 1-2400, 0.001 after; + main's HINGE hinge"
ARMS = {
    "SPLITK_M": dict(key="SPLITK_M", opt="muon", seeds=SEEDS, lr=LR, iters=28800, muon_lr=MUON_LR, prio=2,
                     label="S33's A_HINGE + trigger + KEYMASS (Muon)", sched=_mu + _trig),
    "HINGE4k4_A": dict(key="HINGE4k4_A", opt="adam", seeds=SEEDS, lr=LR, iters=28800, muon_lr=None, prio=2,
                       label="main's A4k4 + test_slow_start's HINGE recipe (Adam), run here", sched=_ad),
    "SPLITK_A": dict(key="SPLITK_A", opt="adam", seeds=SEEDS, lr=LR, iters=28800, muon_lr=None, prio=2,
                     label="HINGE4k4_A + trigger + KEYMASS (Adam)", sched=_ad + _trig),
}


def child(func, payload, timeout=None):
    return mt.run_child(CHILD, func, payload, timeout=timeout)


def run_job(arm, seed):
    a = ARMS[arm]
    return child("run", dict(arm=arm, seed=seed, iters=a["iters"], mlr=a["muon_lr"]))


def segments(arm):
    raise NotImplementedError("S37 is projected by its own child timing (explore_batch13)")


def timing(arms):
    return child("timing", dict(which=list(arms), steps=TIME_STEPS, mlr=MUON_LR))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _stats_upto(stats, t):
    return {s["step"]: s for s in stats if s["step"] != "end" and s["step"] <= t}


def _eq_m(_):
    r = child("run", dict(arm="SPLITK_M", seed=240, iters=EQ_AT, mlr=MUON_LR, thr=0.0))
    ref = ec.load_store(s33.NAME)["runs"]["A_HINGE|240"]
    want = [c for c in ref["curve"] if c[0] <= EQ_AT]
    sn, sr = _stats_upto(r["stats"], EQ_AT), _stats_upto(ref["stats"], EQ_AT)
    same = sn.keys() == sr.keys() and all(sn[k] == sr[k] for k in sr)
    meas = [(c["step"], c["fired"]) for c in r["checks"]]
    return (f"SPLITK_M with the trigger off equals S33's A_HINGE|240 bit for bit through {EQ_AT} (curve {r['curve'] == want}; "
            f"statistics at {sorted(sr)} {same}; the trigger measured at {meas})",
            r["curve"] == want and same and len(meas) == 3 and not any(f for _, f in meas))


def _eq_a(_):
    with ThreadPoolExecutor(max_workers=2) as ex:
        fb = ex.submit(child, "run", dict(arm="HINGE4k4_A", seed=240, iters=EQ_AT))
        fs = ex.submit(child, "run", dict(arm="SPLITK_A", seed=240, iters=EQ_AT, thr=0.0))
        b, s = fb.result(), fs.result()
    sn, sb = _stats_upto(s["stats"], EQ_AT), _stats_upto(b["stats"], EQ_AT)
    same = sn.keys() == sb.keys() and all(sn[k] == sb[k] for k in sb)
    meas = [(c["step"], c["fired"]) for c in s["checks"]]
    return (f"SPLITK_A with the trigger off equals HINGE4k4_A|240 (a fresh run) bit for bit through {EQ_AT} (curves "
            f"{s['curve']} vs {b['curve']}; statistics at {sorted(sb)} {same}; the trigger measured at {meas})",
            s["curve"] == b["curve"] and same and len(meas) == 3 and not any(f for _, f in meas))


def check():
    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        fm, fa, fy = ex.submit(_eq_m, None), ex.submit(_eq_a, None), ex.submit(child, "synthetic", {})
        rows += [fm.result(), fa.result()] + [tuple(x) for x in fy.result()]
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    diff = mt.differing()
    v = diff == sorted(mt.SERVED)
    print(f"  CHECK {NAME}: children at {mt.MAIN_SHA[:7]} ({(time.time() - t0) / 60:.1f} min); the modules served from main "
          f"are exactly the test modules that differ between this branch and {mt.MAIN_SHA[:7]}: {'ok' if v else 'FAIL'}",
          flush=True)
    return ok and v


# ── Report ───────────────────────────────────────────────────────────────────
def compare(new, old, seeds, lab_new, lab_old):
    both = [s for s in seeds if s in new and s in old]
    b = sum(1 for s in both if new[s] and not old[s])
    c = sum(1 for s in both if old[s] and not new[s])
    p = mcnemar_exact(b, c)
    print(f"    BOUND  {lab_new} {sum(new[s] for s in both)}/{len(both)}  {lab_old} {sum(old[s] for s in both)}/{len(both)}  "
          f"{lab_new} only {b}, {lab_old} only {c}, McNemar two-sided p = {p:.3g} (printed)")
    return dict(n=len(both), new=sum(new[s] for s in both), old=sum(old[s] for s in both), b=b, c=c, p=p)


def curve_acc(r, step):
    for s, a, _ in r["curve"]:
        if s == step:
            return a
    return None


def harm(r):
    out = []
    for c in r.get("checks") or []:
        if c["fired"]:
            a0, a1 = curve_acc(r, c["step"]), curve_acc(r, c["step"] + 2400)
            if a0 is not None and a1 is not None and a1 < a0 - HARM_DROP:
                out.append((c["step"], round(a0, 3), round(a1, 3)))
    return out


def merged_check(c):
    m = c["map_before"]
    return any(m.count(x) >= 2 for x in set(m)) and any(m.count(ch) == 0 for ch in range(len(c["key_masses"])))


def fmt_m(ms):
    return "[" + " ".join(f"{v:.2f}" for v in ms) + "]"


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    got = {arm: {int(k.split("|")[1]): r for k, r in store["runs"].items() if k.startswith(arm + "|") and r.get("ok")}
           for arm in ARMS}
    ahinge = {int(k.split("|")[1]): r for k, r in ec.load_store(s33.NAME)["runs"].items() if k.startswith("A_HINGE|")}
    s36 = {int(k.split("|")[1]): r for k, r in ec.load_store("split_plateau")["runs"].items() if k.startswith("SPLIT4k4_M|")}
    xr = mt.recorded("stream_recipe", mt.MAIN_SHA)
    xa = {s: xr[f"A4k4|{s}"] for s in SEEDS if f"A4k4|{s}" in xr}
    print(f"    {'seed':>4} | {'SPLITK_M':<22} {'trans':>5} {'spl':>3} {'ch':>7} | {'A_HINGE':<18} | {'S36 SPLIT4k4_M':<18} || "
          f"{'SPLITK_A':<22} {'trans':>5} {'spl':>3} {'ch':>7} | {'HINGE4k4_A':<22} {'trans':>5} {'ch':>7} | {'X A4k4':<20}")

    def cell(r, w=22, splits=True):
        if r is None:
            return f"{'not run':<{w}} {'':>5}" + (f" {'':>3}" if splits else "") + f" {'':>7}"
        me_ = s33.map_at(r, "end")
        return (f"{r['outcome']:<{w}} {str(r['transition']):>5}" + (f" {r.get('splits', 0):>3}" if splits else "")
                + f" {s33.per_channel(me_):>7}")
    for s in SEEDS:
        x = xa.get(s)
        xo = "--" if x is None else ("BOUND" if x["transition"] is not None else "unbound")
        print(f"    {s:>4} | {cell(got['SPLITK_M'].get(s))} | {(ahinge.get(s) or {}).get('outcome', '--'):<18} | "
              f"{(s36.get(s) or {}).get('outcome', '--'):<18} || {cell(got['SPLITK_A'].get(s))} | "
              f"{cell(got['HINGE4k4_A'].get(s), splits=False)} | {xo:<20}")
    bnd = {arm: {s: ec.bound(r) for s, r in got[arm].items()} for arm in ARMS}
    out = {}
    print("  SPLITK_M vs S33's A_HINGE and vs S36's SPLIT4k4_M (paired):")
    dm = compare(bnd["SPLITK_M"], {s: r["transition"] is not None for s, r in ahinge.items()}, SEEDS, "SPLITK_M", "A_HINGE")
    compare(bnd["SPLITK_M"], {s: r["transition"] is not None for s, r in s36.items()}, SEEDS, "SPLITK_M", "SPLIT4k4_M")
    print("  SPLITK_A vs HINGE4k4_A (paired):")
    seeds_a = ARMS["SPLITK_A"]["seeds"]
    da = compare(bnd["SPLITK_A"], bnd["HINGE4k4_A"], seeds_a, "SPLITK_A", "HINGE4k4_A")
    print("  HINGE4k4_A vs X's A4k4 (printed):")
    compare(bnd["HINGE4k4_A"], {s: x["transition"] is not None for s, x in xa.items()}, ARMS["HINGE4k4_A"]["seeds"],
            "HINGE4k4_A", "X A4k4")
    for arm in ("SPLITK_M", "SPLITK_A"):
        rs = got[arm]
        print(f"  {arm} checks per run (step: probe acc; E eligible, F fired, B blocked by the cap; KEYMASS c*->c0 / all-position "
              f"c*->c0; labelled map before; +/- = copies a >= 2-stream channel onto an empty one):")
        fired_k = fired_a = n_f = 0
        chk_k = chk_a = n_c = 0
        for s, r in sorted(rs.items()):
            print(f"    {s}: splits {r.get('splits', 0)}")
            for c in r.get("checks") or []:
                tag = "F" if c["fired"] else ("B" if c.get("blocked") else ("E" if c.get("eligible") else "."))
                print(f"         {c['step']:>5} acc {c['acc']:.3f} {tag} key {c['key_cs']}->{c['key_c0']} "
                      f"{'+' if c['key_ok'] else '-'}  all {c['all_cs']}->{c['all_c0']} {'+' if c['all_ok'] else '-'}  map "
                      f"{c['map_before']} ({s33.per_channel(c['map_before'])})"
                      + (f" -> {c['map_after']} ({s33.per_channel(c['map_after'])})" if c["fired"] else "")
                      + f"  key masses {fmt_m(c['key_masses'])}  all {fmt_m(c['all_masses'])}")
                if c["fired"]:
                    n_f += 1
                    fired_k += c["key_ok"]
                    fired_a += c["all_ok"]
                if merged_check(c):
                    n_c += 1
                    chk_k += c["key_ok"]
                    chk_a += c["all_ok"]
        print(f"  {arm} labelled targeting accuracy (copies a >= 2-stream channel onto an empty one): over the {n_f} fired splits "
              f"KEYMASS {fired_k}/{n_f}, the all-position rule (at the same checks) {fired_a}/{n_f}; over the {n_c} checks whose "
              f"labelled map has a shared and an empty channel KEYMASS {chk_k}/{n_c}, all-position {chk_a}/{n_c}")
        for s, r in sorted(rs.items()):
            rows = []
            for c in r.get("checks") or []:
                if c["fired"]:
                    nxt = s33.map_at(r, c["step"] + 2400)
                    rows.append(f"{c['step']} {c['key_cs']}->{c['key_c0']}: {s33.per_channel(c['map_before'])} -> "
                                f"{s33.per_channel(c['map_after'])} -> {s33.per_channel(nxt) if nxt else '--'}")
            if rows:
                print(f"    {arm} {s} splits (channels before -> right after -> at the next check): " + "; ".join(rows))
        hm = {s: harm(r) for s, r in sorted(rs.items()) if harm(r)}
        print(f"  {arm} HARM (held-out accuracy 2400 updates after a split > {HARM_DROP} below the value at it): "
              f"{len(hm)}/{len(rs)} runs {hm}")
        out[arm] = dict(fired=n_f, fired_key=fired_k, fired_all=fired_a, checks=n_c, chk_key=chk_k, chk_all=chk_a, harm=len(hm),
                        splits=sum(r.get("splits", 0) for r in rs.values()))
    for arm, opt, d in (("SPLITK_M", "MUON", dm), ("SPLITK_A", "ADAM", da)):
        n_b = sum(bnd[arm].values())
        n = len(got[arm])
        rd = ("the targeted split fixes four-stream merges" if n_b >= FIX_N else "it does not" if n_b <= NOT_N else "neither") + \
             f" (BOUND {n_b}/{n}; >= {FIX_N} fixes, <= {NOT_N} not)"
        print(f"  READING S37 {opt} ({arm}): {rd}")
        out[arm].update(d, bound=n_b, n=n, reading=rd, complete=n == len(ARMS[arm]["seeds"]))
    out["HINGE4k4_A"] = dict(bound=sum(bnd["HINGE4k4_A"].values()), n=len(got["HINGE4k4_A"]),
                             complete=len(got["HINGE4k4_A"]) == len(ARMS["HINGE4k4_A"]["seeds"]))
    return out
