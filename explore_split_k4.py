#!/usr/bin/env python
"""
explore_split_k4.py — EXPLORATORY, not a result. Screen S51 (batch 18): the window recipe and its split at the old
four-stream merge configuration, k = 4 (four channels for four streams: no spare channel).

BACKGROUND: S37 (batch 13) ran S24's split with KEYMASS targeting on the recurrent gate at S=4, P=4, k=4, conv: main's
HINGE4k4_A (SLOW + the hinge, Adam) bound 7/10 and routed 5/10 on seeds 240-249 (two MERGED 2 share, one non-stream KEY);
SPLITK_A (+ the trigger) routed 6/10. Batch 16's LOCAL3_SLOW16_A routed 20/20 at S=4 with k=16; batch 17's W_SPLIT routed
9/10 at S=8, k=16. Does the window recipe with the split fix the four-stream merges when there is no spare channel?

ARMS (explore_split_k4_child; configuration "a": test_stream_recipe.run_attempt with ARM["A4k4"], S=4, P=4, k=4, conv,
28800 updates, seeds 240-249; Adam lr 1e-3 = test_slow_start.LR read in a child; 1 torch thread per run):
  W4k4         LOCAL3 + SLOW, no hinge.
  W4k4_SPLIT   W4k4 + SPLIT (S37's KEYMASS trigger as run in batch 17 with S37's cap: <= 2 splits, >= 4800 apart).
PAIRING: S37's HINGE4k4_A records (explore_out/split_target_results.json; 7/10 bound, 5/10 routed) on the same seeds =
same initial parameters (+ the window) and batches; W4k4_SPLIT also with W4k4 (equal until the first split).
READING (fixed before any run; BOUND ROUTED): "the split fixes four-stream merges at k=4" if W4k4_SPLIT >= 8/10; "it does
not" if <= 5/10; otherwise neither. Printed alongside: W4k4 against HINGE4k4_A (does the window gate alone merge less than
the recurrent gate at k=4?), and W4k4_SPLIT against W4k4. Exact McNemar two-sided. W4k4 may be cut by the runtime rule
(W4k4_SPLIT is never cut).
DIAGNOSTICS: the map at every check and the end; each split's labelled target; S38's decoder at CTX / KEY / VAL (chance
1/4) at 1200, 2400, 4800, 9600 and the end; transitions; W4k4_SPLIT's agreement with W4k4 through its first split.
CHECKS: LOCAL3's conversion on configuration a keeps every parameter and adds batch 1's window; the groups and lrs at
updates 1, 2400, 2401 through the real path; the trigger at cap 2 equals S37's make_trigger row for row on LOCAL3 at
configuration a; S37's HINGE4k4_A|240 reproduces its record through 1200 on this machine (curve and statistics); with the
threshold at 0, W4k4_SPLIT|240 equals W4k4|240 bit for bit through 3600.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_b16_report as rp
import explore_window_d8 as s48

NAME = "split_k4"
CHILD = "explore_split_k4_child"
MAIN = True
IDEA = "the window recipe and its split at the old merge configuration: S=4, P=4, k=4 (no spare channel)"
SOURCE = "S37 (batch 13): HINGE4k4_A 7/10 bound, 5/10 routed; batch 17's W_SPLIT 9/10 at S=8, k=16; the user's batch 18"
CHANGE = "LOCAL3 + SLOW on configuration a (k=4); W4k4_SPLIT: + the KEYMASS trigger (<= 2 splits)"
PAIRING = "seeds 240-249 with S37's HINGE4k4_A records (same initial parameters, + the window, and batches); W4k4_SPLIT with W4k4"
LR = 1e-3
ITERS = 28800
SEEDS = tuple(range(240, 250))
S37 = "split_target"
FIX_N, NOT_N = 8, 5
REPRO_SEED, REPRO_ITERS, INERT_ITERS = 240, 1200, 3600
DIAG = ("1200", "2400", "4800", "9600", "end")
_sl = (f"gate W_in/W_g/window {LR:g} throughout; every other trainable parameter {LR / 10:g} for updates 1-2400, {LR:g} "
       f"after; W_h frozen (unused); no hinge")
_trig = ("; + S37's KEYMASS trigger (every 2400 from 4800: probe acc < 0.95 and rise < 0.02; <= 2 splits, >= 4800 apart) -> "
         "S24's SPLIT of W_g's row c* onto c0, W_g's Adam state zeroed")
ARMS = {
    "W4k4": dict(key="W4k4", opt="adam", lr=LR, muon_lr=None, iters=ITERS, seeds=SEEDS,
                 label="LOCAL3 + SLOW, no hinge (S=4, P=4, k=4, conv)", sched=_sl),
    "W4k4_SPLIT": dict(key="W4k4_SPLIT", opt="adam", lr=LR, muon_lr=None, iters=ITERS, seeds=SEEDS,
                       label="W4k4 + the KEYMASS trigger (<= 2 splits)", sched=_sl + _trig),
}
CUT = set()


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ARMS[arm]["iters"]))


def timing(arms):
    return child("timing", dict(which=list(arms), steps=300))


def n_meas(arm):
    return ARMS[arm]["iters"] // 1200


def n_extra(arm):
    return ARMS[arm]["iters"] // 2400 - 1 if "SPLIT" in arm else 0


def s37_runs(seeds=SEEDS):
    st = ec.load_store(S37)["runs"]
    return {s: st[f"HINGE4k4_A|{s}"] for s in seeds if f"HINGE4k4_A|{s}" in st}


# ── CHECKs ───────────────────────────────────────────────────────────────────
def check():
    from concurrent.futures import ThreadPoolExecutor
    rows = [tuple(x) for x in child("checks", {})]
    jobs = [("S37", "run_s37", dict(seed=REPRO_SEED, iters=REPRO_ITERS)),
            ("SPLIT0", "run", dict(arm="W4k4_SPLIT", seed=REPRO_SEED, iters=INERT_ITERS, thr=0.0, total=ITERS)),
            ("PLAIN", "run", dict(arm="W4k4", seed=REPRO_SEED, iters=INERT_ITERS))]
    with ThreadPoolExecutor(max_workers=3) as ex:
        res = dict(zip([j[0] for j in jobs], ex.map(lambda j: child(j[1], j[2]), jobs)))
    rec = s37_runs((REPRO_SEED,)).get(REPRO_SEED)
    r = res["S37"]
    want = [c for c in rec["curve"] if c[0] <= REPRO_ITERS] if rec else None
    sn = [s for s in r["stats"] if s["step"] != "end" and s["step"] <= REPRO_ITERS]
    so = [s for s in (rec or {}).get("stats", []) if s["step"] != "end" and s["step"] <= REPRO_ITERS]
    v = rec is not None and r["curve"] == want and sn == so
    rows.append((f"S37's HINGE4k4_A|{REPRO_SEED} reproduces its record through {REPRO_ITERS} on this machine (curve {r['curve']} vs "
                 f"{want}; statistics at {[s['step'] for s in sn]} equal {sn == so})", v))
    a, b = res["SPLIT0"], res["PLAIN"]
    same = s48.eq_upto(a, b, INERT_ITERS) and a["curve"] == b["curve"]
    rows.append((f"with the threshold at 0, W4k4_SPLIT|{REPRO_SEED} equals W4k4|{REPRO_SEED} bit for bit through {INERT_ITERS} (curve "
                 f"{a['curve']}; statistics and decoder equal {same}; checks logged at {[c['step'] for c in a['checks']]})",
                 same and a["splits"] == 0 and len(a["checks"]) >= 1))
    ok = True
    for nm, vv in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if vv else 'FAIL'}", flush=True)
        ok &= bool(vv)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def reading(o):
    if not o["complete"]:
        return f"INCOMPLETE ({o['n']}/{len(SEEDS)}), no reading"
    nb = o["bound_routed"]
    v = "the split fixes four-stream merges at k=4" if nb >= FIX_N else "it does not" if nb <= NOT_N else "neither"
    return (f"{v} (W4k4_SPLIT BOUND ROUTED {nb}/{o['n']}; fixes: >= {FIX_N}/10; does not: <= {NOT_N}/10; vs S37's HINGE4k4_A: "
            f"arm only {o['b']}, HINGE4k4_A only {o['c']}, p = {o['p']:.3g})")


def report():
    me = sys.modules[__name__]
    import explore_common2 as c2
    c2.print_screen_header2(me)
    store = ec.load_store(NAME)
    got = {arm: c16.runs_of(me, arm, store=store) for arm in ARMS}
    h4 = s37_runs()
    print(f"  pair: S37's HINGE4k4_A ({len(h4)} records): bound {sum(1 for r in h4.values() if r.get('transition') is not None)}/{len(h4)}, "
          f"BOUND ROUTED {sum(rp.br(r) for r in h4.values())}/{len(h4)}; outcomes {rp.classes(h4)}")
    if CUT:
        print(f"  cut by the runtime rule: {sorted(CUT)} (no runs, no reading)")
    out = {}
    for arm, a in ARMS.items():
        rs = got[arm]
        print(f"  ARM {arm} ({a['label']}), seeds {ec.fmt_seeds(a['seeds'])}; {a['sched']}")
        s48.maps_table(arm, a, rs, "S37 HINGE4k4_A", h4)
        d = rp.compare(rs, h4, a["seeds"], arm, "HINGE4k4_A")
        print(f"    failure classes at the end: {arm} {rp.classes(rs)}; HINGE4k4_A {rp.classes(h4)}")
        print(f"    transitions: {arm} {rp.transitions(rs)}; HINGE4k4_A {rp.transitions(h4)}")
        if arm == "W4k4_SPLIT":
            s48.splits_block(rs)
            if "W4k4" in ARMS:
                rp.compare(rs, got["W4k4"], a["seeds"], arm, "W4k4")
                s48.agreement("W4k4_SPLIT", rs, got["W4k4"], "W4k4")
        meds = rp.decode_block(rs, steps=DIAG, chance=1 / 4)
        nb, n = sum(rp.br(r) for r in rs.values()), len(rs)
        out[arm] = dict(d, bound_routed=nb, n=n, complete=n == len(a["seeds"]), classes=rp.classes(rs), meds=meds,
                        bound_seeds=sorted(s for s, r in rs.items() if rp.br(r)))
        if arm == "W4k4_SPLIT":
            out[arm]["reading"] = reading(out[arm])
            print(f"  READING S51: {out[arm]['reading']}")
        else:
            print(f"  S51 {arm} (printed alongside, no reading): BOUND ROUTED {nb}/{n}; vs HINGE4k4_A: W4k4 only {d['b']}, "
                  f"HINGE4k4_A only {d['c']}, p = {d['p']:.3g}")
        print()
    return out
