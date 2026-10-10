#!/usr/bin/env python
"""
explore_h_two.py — EXPLORATORY, not a result. Session H, screen S78: the KEYMASS split at k = S = 2, where it was never
applied (no channel is idle), against the two-stream window gate's key splits. Branch claude/explore-H; outputs under
explore_out/H/. Child: explore_h_two_child (this branch's modules, S43's path).

SEEDS: the specification gave 1100-1119 (S78) and 1120-1129 (S79) from the block 1100-1159, "shifted by 100 if any appears
in explore_out/ or results/". 1100-1149 appear (test_router_reliability's trial seeds in results/X and results/L
router_reliability_results.json, on every branch); 1200-1259 appear too (restart-attempt seeds 1240-1259 in
stream_recipe_results.json, 1200-1219 in curriculum_confirm_results.json; explore-G logs). 1300-1359 appears nowhere in
explore_out/ or results/ on any branch (scanned before any run). So: S78 seeds 1300-1319, S79 seeds 1320-1329, perfect gate
on 1300-1301.

BACKGROUND: the main line's test_window_gate: WIN3_SLOW discovers 69/80 at S=2, P=4, k=2, no conv; its failures are key
splits. Batch 18's S52 W2_K4 (k = 4, two spare channels, same path) bound routed 36/40 on seeds 160-199 (outcome BOUND
ROUTED; 2 MERGED and 2 non-stream KEY, all four holding a key split from update 0-325). S61: at eight streams the split's whole effect is an exact row copy onto an idle channel; at k = 2 the "idle"
channel is the other busy one.

CONFIGURATION: test_window_gate's Part A path = S43's LOCAL3_SLOW (S=2, P=4, k=2, grouped, no conv, 24000 updates, LOCAL3,
SLOW, no hinge; test_window_gate's CHECK 135 equates its WIN3_SLOW with it). Seeds 1300-1319, all arms paired (the k = 2
arms share initial parameters, window and batches; K4's W_g has 4 rows, so the parameters drawn after it differ; same
window and batches). 1 torch thread per run.
ARMS (explore_h_two_child):
  REF           WIN3_SLOW (S43's LOCAL3_SLOW).
  SPLIT2        REF + the KEYMASS trigger at k = 2: checks every 2400 from 4800 (2400 reference) through 21600 on S36's
                64-sequence probe; fire if probe accuracy < 0.95 and it rose < 0.02 since the previous check; cap 3, gap
                4800; c* = the row with the larger key-position read mass, c0 = the other row; W_g[c0] := W_g[c*] exactly,
                no noise, no Adam reset.
  SPLIT2_NOISE  the same trigger with the main line's operation: W_g[c0] = w + n2, W_g[c*] = w + n1 (w = W_g[c*], noise
                0.1 std(w)), W_g's Adam state zeroed.
  K4            S52's W2_K4 (k = 4, two spare channels; explore_keysplits_child, copied from claude/outside-ideas
                unchanged), 24000 updates.
  ORACLE        test_short_conv's perfect gate ceiling_conv (the main line's CEIL_A) on seeds 1300-1301.
OUTCOMES: the k = 2 arms DISCOVERED (explore_common.discovered: BOUND and final VAL cos < 0.5, S43's and test_window_gate's
Part A outcome). K4: S52's outcome, BOUND ROUTED (test_stream_recipe.outcome, copied into explore_keysplits_child from main
at 9c5939e: bound, the VAL-position stream -> channel map one-to-one and every stream's held-out accuracy >= 0.9).
Failure classes (unbound runs) under v1 (test_router_layout.fail_class; K4: test_stream_recipe.outcome) AND v2
(fail_class_v2 at ba8901a, copied unchanged as explore_h_fail_class_v2; K4: outcome_v2), both reported.
VALIDITY (fixed now): VALID only if ORACLE binds both seeds within 24000; otherwise UNTESTED.
READINGS (fixed now; 20 paired seeds; "X only" = X successful and Y not; exact one-sided McNemar; Wilson and bands printed):
  R1 "the split carries to two streams"   if SPLIT2 beats REF on DISCOVERED with one-sided p < 0.05 (5 vs 0, 7 vs 1,
                                          9 vs 2 or better).
  R2 "noise is needed at k = 2"           if SPLIT2_NOISE only >= 4 and SPLIT2 only = 0 (SPLIT2_NOISE vs SPLIT2, DISCOVERED).
  R3 "spare channels are the better device"  if K4 only >= 4 and SPLIT2 only = 0 (K4 BOUND ROUTED vs SPLIT2 DISCOVERED).
  R4 "neither device"                     if none of SPLIT2, SPLIT2_NOISE, K4 beats REF by R1's criterion (one-sided
                                          p < 0.05 on its own outcome vs REF's DISCOVERED).
DIAGNOSTICS: for every split, eta^2 by key and by stream at KEY and at VAL positions (and the margin) at the check that
fired, right after the operation, and at the next check (did the copy break the key split, and what formed instead?); the
v1 / v2 disagreements with their eta^2 by key and margin; transitions.
CHECKS (before any run; a failure stops the invocation): explore_common.repro_check; the copied modules' git blob SHAs
(explore_h_fail_class_v2.py = ba8901a:fail_class_v2.py; explore_keysplits_child.py = claude/outside-ideas'); REF|160 equals
S43's recorded LOCAL3_SLOW|160 bit for bit through 2400; SPLIT2 at seed 1300 with the trigger's threshold at 0 equals REF
at 1300 bit for bit through 6000 (curve, every statistic, the decoder); the trigger's mechanics at k = 2 (fires at 4800,
9600, 14400; c0 is a busy row; SPLIT2 leaves the rows equal and the Adam state untouched; SPLIT2_NOISE zeroes it);
explore_h_ops' operations equal S36's split_op and S61's exact copy bit for bit (main-line child); S52's W2_K4|160 record
(claude/outside-ideas) reproduced bit for bit through 2400 by the copied code on this CPU; fail_class_v2 equals v1 on
every recorded two-stream run of this branch (S43, S52, S60) except exactly those with eta^2 by key >= 0.5 and margin >=
0.25 (the margin rule fired), listed. The same property is checked on S78's own runs in the report.
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_h_common as hc

NAME = "H/two"
CHILD = "explore_h_two_child"
MAIN = False
IDEA = "the exact-copy KEYMASS split at k = S = 2 against the window gate's key splits; vs spare channels (k = 4)"
SOURCE = "S61 (the split is an exact row copy); main's test_window_gate WIN3_SLOW 69/80; S52 W2_K4"
CHANGE = "WIN3_SLOW + the k=2 split (exact copy / noise + reset); W2_K4"
PAIRING = "seeds 1300-1319 (shifted from 1100-1119), all arms paired"
LR = 1e-3
ITERS = 24000
SEEDS = tuple(range(1300, 1320))
ORACLE_SEEDS = (1300, 1301)
OI = "origin/claude/outside-ideas"
FCV2 = "ba8901a"
ARMS = {"REF": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="WIN3_SLOW (S43's LOCAL3_SLOW)", sched="LOCAL3 + SLOW"),
        "SPLIT2": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="+ k=2 split, exact copy", sched="LOCAL3 + SLOW + copy"),
        "SPLIT2_NOISE": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="+ k=2 split, noise + reset",
                             sched="LOCAL3 + SLOW + split_op"),
        "K4": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="S52's W2_K4 (k=4)", sched="LOCAL3 + SLOW, k=4"),
        "ORACLE": dict(opt="adam", lr=LR, iters=ITERS, seeds=ORACLE_SEEDS, label="ceiling_conv", sched="one Adam 1e-3")}
CUT = set()


def child(func, payload, main=MAIN):
    return c16.run_child(CHILD if not main else "explore_h_classes_child", func, payload, main=main)


def run_job(arm, seed):
    return c16.run_child(CHILD, "run", dict(arm=arm, seed=seed, iters=ITERS), main=False)


def _git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True).stdout.strip()


def oi_store(name):
    import json
    return json.loads(_git("show", f"{OI}:explore_out/{name}_results.json"))["runs"]


def classify(runs):
    """{key: record} -> {key: v1/v2 classes}, in a main-line child."""
    payload = {}
    for k, r in runs.items():
        kind = r["_kind"]
        e = r["end"]
        if kind == "outcome":
            keep = ("ch_map", "one_to_one", "stream_acc", "margin", "etak_key_by_key", "etak_key_by_block", "etak_key_by_index",
                    "stream_gate")
            payload[k] = dict(kind=kind, transition=r["transition"], end={x: e[x] for x in keep if x in e})
        else:
            payload[k] = dict(kind=kind, end={x: e[x] for x in ("margin", "eta_key_by_key", "eta_key_by_half", "eta_key_by_index")})
    return c16.run_child("explore_h_classes_child", "classify", dict(runs=payload), main=True)


def v2_property(cls):
    """Disagreements equal exactly the runs with eta^2 by key >= 0.5 and margin >= 0.25."""
    dis = sorted(k for k, c in cls.items() if c["v1"] != c["v2"])
    pred = sorted(k for k, c in cls.items() if c["predicted_differ"])
    return dis == pred, dis, pred


def check():
    import json
    rows = []
    b1, b2 = _git("hash-object", "explore_h_fail_class_v2.py"), _git("rev-parse", f"{FCV2}:fail_class_v2.py")
    k1, k2 = _git("hash-object", "explore_keysplits_child.py"), _git("rev-parse", f"{OI}:explore_keysplits_child.py")
    rows.append((f"copied modules: explore_h_fail_class_v2.py {b1[:12]} = {FCV2}:fail_class_v2.py {b2[:12]}; explore_keysplits_child.py "
                 f"{k1[:12]} = claude/outside-ideas' {k2[:12]}", b1 == b2 and k1 == k2 and len(b1) == 40))
    s43 = {k: r for k, r in ec.load_store("window_recipe")["runs"].items() if r.get("ok")}
    rec160 = next(r for k, r in s43.items() if k.startswith("LOCAL3_SLOW|160|46867e792241|"))
    k4rec = next(r for k, r in oi_store("two_stream_keysplits").items() if k.startswith("W2_K4|160|") and r.get("ok"))
    jobs = [("REF160", dict(arm="REF", seed=160, iters=2400)), ("K4_160", dict(arm="K4", seed=160, iters=2400)),
            ("REF1300", dict(arm="REF", seed=1300, iters=6000)),
            ("SPLIT2_1300", dict(arm="SPLIT2", seed=1300, iters=6000, thr=0.0, total=ITERS))]
    with ThreadPoolExecutor(max_workers=4) as ex:
        fu = ex.submit(lambda: c16.run_child(CHILD, "checks", {}, main=False))
        fo = ex.submit(lambda: c16.run_child("explore_h_classes_child", "ops_check", {}, main=True))
        res = dict(ex.map(lambda j: (j[0], c16.run_child(CHILD, "run", j[1], main=False)), jobs))
        rows += [tuple(x) for x in fu.result()] + [tuple(x) for x in fo.result()]

    def upto(r, ref, t, decode=True):
        c1 = [c for c in r["curve"] if c[0] <= t]
        s1 = [s for s in r["stats"] if s["step"] != "end" and s["step"] <= t]
        s2 = [s for s in ref["stats"] if s["step"] != "end" and s["step"] <= t]
        d = True
        if decode:
            d1, d2 = r.get("decode") or {}, ref.get("decode") or {}
            d = all(d1[k] == d2[k] for k in d1 if k.isdigit() and int(k) <= t and k in d2)
        return bool(c1) and c1 == [c for c in ref["curve"] if c[0] <= t] and s1 == s2 and d, c1
    eq, c = upto(res["REF160"], rec160, 2400)
    rows.append((f"REF|160 equals S43's recorded LOCAL3_SLOW|160 ({rec160.get('cpu')}) bit for bit through 2400 (curve {c})", eq))
    eq, c = upto(res["K4_160"], k4rec, 2400, decode=False)
    gn = res["K4_160"].get("grad") == [g for g in (k4rec.get("grad") or []) if g[0] <= 2400] if k4rec.get("grad") else True
    rows.append((f"K4|160 (the copied W2_K4 code) equals S52's recorded W2_K4|160 (claude/outside-ideas, {k4rec.get('cpu')}) bit for "
                 f"bit through 2400 (curve {c}; statistics)", eq))
    r2, r0 = res["SPLIT2_1300"], res["REF1300"]
    eq, c = upto(r2, r0, 6000)
    eq = eq and r2["splits"] == 0 and len(r2["checks"]) >= 2 and r2["curve"] == r0["curve"]
    rows.append((f"SPLIT2|1300 with the threshold at 0 equals REF|1300 bit for bit through 6000 (curve {c}; statistics; decoder; "
                 f"trigger checks at {[x['step'] for x in r2['checks']]}, fired {r2['splits']})", eq))
    # v2 vs v1 on the recorded two-stream runs of this branch
    recs = {}
    for k, r in s43.items():
        if r.get("transition") is None:
            recs["S43:" + k] = dict(r, _kind="k2")
    for name in ("H/stability",):
        for k, r in ec.load_store(name)["runs"].items():
            if r.get("ok") and r.get("transition") is None and not k.startswith("ORACLE") and "eta_key_by_key" in r["end"]:
                recs[name + ":" + k] = dict(r, _kind="k2")
    for k, r in oi_store("two_stream_keysplits").items():
        if r.get("ok") and k.startswith("W2_K4|") and r.get("transition") is None:
            recs["S52:" + k] = dict(r, _kind="outcome")
        elif r.get("ok") and k.startswith("W2_KEYHINGE|") and r.get("transition") is None:
            recs["S52:" + k] = dict(r, _kind="k2")
    cls = classify(recs)
    ok, dis, pred = v2_property(cls)
    rows.append((f"fail_class_v2 vs v1 on {len(cls)} recorded unbound two-stream runs of this branch (S43, S60, S52): they differ on "
                 f"{len(dis)} runs, exactly those with eta^2 by key >= 0.5 and margin >= 0.25 ({len(pred)}): "
                 + "; ".join(f"{k.split('|')[0]}|{k.split('|')[1]} {cls[k]['v1']} -> {cls[k]['v2']} (key {cls[k]['eta_key']:.4f}, margin "
                             f"{cls[k]['margin']:.3f})" for k in dis), ok))
    ok_all = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok_all &= bool(v)
    return ok_all


# ── Report ───────────────────────────────────────────────────────────────────
def success(arm, r):
    if arm == "K4":
        return r.get("outcome") == "BOUND ROUTED"
    return ec.discovered(r)


def report():
    me = sys.modules[__name__]
    store = ec.load_store(NAME)
    got = {arm: c16.runs_of(me, arm, store=store) for arm in ARMS}
    res = dict(arms={}, cmp={}, readings={})
    orc = got["ORACLE"]
    n_or = sum(1 for r in orc.values() if r.get("transition") is not None)
    valid = len(orc) == len(ORACLE_SEEDS) and n_or == len(ORACLE_SEEDS)
    res["oracle"] = dict(bound=n_or, n=len(orc), transitions={s: r.get("transition") for s, r in orc.items()}, valid=valid)
    arms = ("REF", "SPLIT2", "SPLIT2_NOISE", "K4")
    recs = {}
    for a in arms:
        for s, r in got[a].items():
            if r.get("transition") is None:
                recs[f"{a}|{s}"] = dict(r, _kind="outcome" if a == "K4" else "k2")
    cls = classify(recs) if recs else {}
    res["v2_property"] = v2_property(cls) if cls else (True, [], [])
    res["classes"] = cls
    for a in arms:
        rs = got[a]
        n, k = len(rs), sum(success(a, r) for r in rs.values())
        tr = sorted(r["transition"] for r in rs.values() if success(a, r))
        c1 = dict(__import__("collections").Counter(cls[f"{a}|{s}"]["v1"] for s in rs if f"{a}|{s}" in cls))
        c2 = dict(__import__("collections").Counter(cls[f"{a}|{s}"]["v2"] for s in rs if f"{a}|{s}" in cls))
        bnd = sum(1 for r in rs.values() if r.get("transition") is not None and not success(a, r))
        fired = [sum(1 for c in r.get("checks") or [] if c.get("fired")) for r in rs.values()]
        res["arms"][a] = dict(n=n, k=k, band=hc.band(k, n), wilson=hc.wilson(k, n), bound_not=bnd, v1=c1, v2=c2,
                              trans_med=c16.med(tr), trans=tr, ops=sum(fired), ops_runs=sum(1 for f in fired if f),
                              complete=n == len(SEEDS), seeds_ok=sorted(s for s, r in rs.items() if success(a, r)))
    S = {a: {s: success(a, r) for s, r in got[a].items()} for a in arms}
    for x, y in (("SPLIT2", "REF"), ("SPLIT2_NOISE", "REF"), ("K4", "REF"), ("SPLIT2_NOISE", "SPLIT2"), ("K4", "SPLIT2")):
        res["cmp"][(x, y)] = hc.paired(S[x], S[y], SEEDS)
    cm = res["cmp"]
    complete = all(res["arms"][a]["complete"] for a in arms) and len(orc) == len(ORACLE_SEEDS)
    if not complete:
        rd = {k: "INCOMPLETE, no reading" for k in ("R1", "R2", "R3", "R4")}
    elif not valid:
        rd = {k: f"UNTESTED (ORACLE bound {n_or}/{len(ORACLE_SEEDS)})" for k in ("R1", "R2", "R3", "R4")}
    else:
        c = cm[("SPLIT2", "REF")]
        rd = {"R1": ("the split carries to two streams" if c["p"] < 0.05 else "does not apply")
              + f" (SPLIT2 vs REF {c['b']} vs {c['c']}, one-sided p = {c['p']:.3g})"}
        c = cm[("SPLIT2_NOISE", "SPLIT2")]
        rd["R2"] = (("noise is needed at k = 2" if (c["b"] >= 4 and c["c"] == 0) else "does not apply")
                    + f" (SPLIT2_NOISE vs SPLIT2 {c['b']} vs {c['c']})")
        c = cm[("K4", "SPLIT2")]
        rd["R3"] = (("spare channels are the better device" if (c["b"] >= 4 and c["c"] == 0) else "does not apply")
                    + f" (K4 vs SPLIT2 {c['b']} vs {c['c']})")
        beats = [a for a in ("SPLIT2", "SPLIT2_NOISE", "K4") if cm[(a, "REF")]["p"] < 0.05]
        rd["R4"] = ("neither device" if not beats else f"does not apply ({', '.join(beats)} beat REF)") + " (" + "; ".join(
            f"{a} vs REF {cm[(a, 'REF')]['b']} vs {cm[(a, 'REF')]['c']}, p = {cm[(a, 'REF')]['p']:.3g}"
            for a in ("SPLIT2", "SPLIT2_NOISE", "K4")) + ")"
    res["readings"], res["complete"], res["valid"] = rd, complete, valid
    return res, got


def split_diag(got):
    """Per firing: eta^2 at the firing check, right after the operation, and at the next check."""
    out = []
    for a in ("SPLIT2", "SPLIT2_NOISE"):
        for s, r in sorted(got[a].items()):
            cks = r.get("checks") or []
            for i, c in enumerate(cks):
                if c.get("fired"):
                    nxt = cks[i + 1] if i + 1 < len(cks) else None
                    out.append(dict(arm=a, seed=s, step=c["step"], before=c["diag"], after=c.get("diag_after"),
                                    next_step=nxt["step"] if nxt else None, next=nxt["diag"] if nxt else None,
                                    outcome=r.get("tag"), discovered=ec.discovered(r), cs=c["cs"], c0=c["c0"]))
    return out
