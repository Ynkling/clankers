#!/usr/bin/env python
"""
explore_h_stability.py — EXPLORATORY, not a result. Session H, screen S60: two stability devices on the gate softmax (a
learned temperature with a floor; stable-max) and an evaluation-side EMA, at two streams. Branch claude/explore-H; seeds
470-479 (session H's block); outputs under explore_out/H/. Child: explore_h_stability_child (this branch's modules, as
S43's).

BACKGROUND: docs/reading/distant_cues_2026-10.md 4.7 ("stability kit": EMA 0.999 of weights for evaluation; stable-max /
a temperature floor on the gate softmax — Prieto et al. 2501.04697: softmax saturation kills gradients and causes
plateaus); docs/reading/2510.04871.md 5.5 (TRM's EMA). The audit (2.1): BOUND is asymmetric — early stop needs three
evaluations >= 0.95, a run reaching its budget counts as bound on one final evaluation.

CONFIGURATION: S43's LOCAL3_SLOW (S=2, P=4, k=2, grouped, no conv, 24000 updates; LOCAL3 window gate; SLOW schedule:
gate group 1e-3 throughout, the rest 1e-4 for updates 1-2400, 1e-3 after; no hinge), seeds 470-479, all arms paired
(same initial parameters, window and batches). 1 torch thread per run.
ARMS (explore_h_stability_child):
  REF          S43's LOCAL3_SLOW unchanged (gate softmax at temperature 1).
  TEMP_FLOOR   a learned scalar tau (init 1, gate group, 1e-3 throughout); gate = softmax(z / max(tau, 0.5)) (the user's
               choice of 8 October 2026: learned tau, floor 0.5).
  STABLEMAX    gate = stable-max of the logits, s(x) = x + 1 (x >= 0), 1 / (1 - x) (x < 0), normalised.
  EMA_EVAL     training unchanged; every evaluation (held-out accuracy, early stop, transition, statistics, outcome) uses an
               EMA (0.999) of the weights; the raw weights' held-out accuracy is recorded alongside.
  ORACLE_CONV  validity: test_short_conv's ceiling_conv (perfect gate + conv layer; this layout's recorded validity arm),
               seeds 470-471. ORACLE_PLAIN: the plain perfect gate (no conv), seeds 470-471, printed only (batch 18 found it
               does not bind this layout).
OUTCOMES (as the harness defines them for this configuration, explore_common): BOUND = transition not None; DISCOVERED =
BOUND and VAL cos < 0.5; failure classes KEY / POSITION / STREAM-PARTIAL / OTHER (explore_common.fail_class); transition
time = the run's transition. Counts printed with bands and Wilson intervals.
VALIDITY (fixed now): VALID only if ORACLE_CONV binds both seeds within 24000; otherwise everything below is UNTESTED.
READINGS (descriptive; fixed now; per arm vs REF on the 10 paired seeds, with exact one-sided McNemar both ways):
  D1 DISCOVERED counts and transition times (median, range); a device "changes discovery" only if its discordant pairs
     with REF are >= 3 vs 0 either way (printed as "more" / "fewer"), else "no change at n = 10".
  D2 saturation: at each evaluation, "saturated" = the gate's max probability > 0.99 at > 90% of the probe's positions.
     Per arm: the fraction of evaluations saturated BEFORE the transition (evaluations at steps < transition, bound runs)
     and AT/AFTER it, and over the whole run for unbound runs; "saturates before binding" counts the bound runs with any
     saturated evaluation before their transition. A device "delays saturation" if its pooled before-transition saturated
     fraction is lower than REF's by >= 0.2 (absolute).
  D3 EMA_EVAL and the BOUND asymmetry: per arm, the runs that count as BOUND on a single final evaluation (transition =
     the last evaluation and that evaluation is the budget's, 24000) and the runs whose held-out accuracy fell below 0.95
     after first reaching it ("flips", evaluations after the first >= 0.95). "The asymmetry shrinks under EMA" if
     EMA_EVAL's single-final-evaluation bindings plus flips are fewer than REF's (both counted on the 10 seeds); with
     REF's typical 0-1 such runs this is expected to be uninformative at n = 10, which is then stated.
CHECKS (before any run; a failure stops the batch): explore_common.repro_check; the child's unit checks (stable-max's values
and gradient; the gate at tau = 1 / softmax bitwise LOCAL3's; the floor; tau in the gate group; EMA averaging and bit-exact
swap; zero saturation at init); and, with each new code path disabled, a run equal BIT FOR BIT to S43's recorded
LOCAL3_SLOW|160 (curve and every statistic) through 2400: REF; TEMP_FLOOR with tau frozen at 1; STABLEMAX with softmax as
the gate function; EMA_EVAL at decay 0 (EMA = the weights).
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_h_common as hc

NAME = "H/stability"
CHILD = "explore_h_stability_child"
MAIN = False
IDEA = "gate-softmax stability (learned temperature with floor 0.5; stable-max) and EMA evaluation, at two streams"
SOURCE = "docs/reading/distant_cues_2026-10.md 4.7; 2501.04697; 2510.04871 5.5; audit 2.1"
CHANGE = "S43's LOCAL3_SLOW + one of: learned tau floored at 0.5, stable-max gate, EMA-weight evaluation"
PAIRING = "seeds 470-479, all gated arms paired"
LR = 1e-3
ITERS = 24000
SEEDS = tuple(range(470, 480))
ORACLE_SEEDS = (470, 471)
S43 = "window_recipe"
REC_SEED = 160
CHECK_ITERS = 2400
ARMS = {
    "REF": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="S43's LOCAL3_SLOW (tau = 1)"),
    "TEMP_FLOOR": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="learned tau, softmax(z / max(tau, 0.5))"),
    "STABLEMAX": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="stable-max gate"),
    "EMA_EVAL": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="evaluation with EMA 0.999 weights"),
    "ORACLE_CONV": dict(opt="adam", lr=LR, iters=ITERS, seeds=ORACLE_SEEDS, label="ceiling_conv (validity)"),
    "ORACLE_PLAIN": dict(opt="adam", lr=LR, iters=ITERS, seeds=ORACLE_SEEDS, label="plain perfect gate (printed)"),
}
CUT = set()


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ARMS[arm]["iters"]))


def s43_record(arm, seed):
    st = ec.load_store(S43)["runs"]
    for k, r in st.items():
        parts = k.split("|")
        if parts[0] == arm and int(parts[1]) == seed and r.get("ok"):
            return r
    return None


def eq_upto(r, ref, t):
    c1 = [c for c in r["curve"] if c[0] <= t]
    c2 = [c for c in ref["curve"] if c[0] <= t]
    s1 = [s for s in r["stats"] if s["step"] != "end" and s["step"] <= t]
    s2 = [s for s in ref["stats"] if s["step"] != "end" and s["step"] <= t]
    return bool(c1) and c1 == c2 and s1 == s2 and len(c1) == t // 1200, c1


def check():
    ref = s43_record("LOCAL3_SLOW", REC_SEED)
    rows = []
    jobs = [("REF", dict(arm="REF", seed=REC_SEED, iters=CHECK_ITERS), "as is"),
            ("TEMP_FLOOR", dict(arm="TEMP_FLOOR", seed=REC_SEED, iters=CHECK_ITERS, learn_tau=False), "tau frozen at 1"),
            ("STABLEMAX", dict(arm="STABLEMAX", seed=REC_SEED, iters=CHECK_ITERS, gate_fn="softmax"), "softmax as gate function"),
            ("EMA_EVAL", dict(arm="EMA_EVAL", seed=REC_SEED, iters=CHECK_ITERS, decay=0.0), "EMA decay 0")]
    with ThreadPoolExecutor(max_workers=4) as ex:
        fu = ex.submit(child, "checks", {})
        res = list(ex.map(lambda j: (j, child("run", j[1])), jobs))
        rows += [tuple(x) for x in fu.result()]
    for (arm, p, how), r in res:
        eq, c = eq_upto(r, ref, CHECK_ITERS) if ref is not None else (False, None)
        extra = ""
        if arm == "EMA_EVAL":
            extra = f"; raw curve {[x[:2] for x in r.get('raw_curve') or []]}; EMA steps {r.get('ema_steps')}"
            eq = eq and r.get("ema_steps") == CHECK_ITERS and [x[:3] for x in r["raw_curve"]] == c
        rows.append((f"{arm} ({how}) equals S43's LOCAL3_SLOW|{REC_SEED} ({ref and ref.get('cpu')}) bit for bit through {CHECK_ITERS} "
                     f"(curve {c}; statistics){extra}", eq))
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def sat_split(r):
    """(saturated evaluations before the transition, evaluations before, saturated at/after, at/after) from r['meas']."""
    t = r.get("transition")
    m = {int(k): v for k, v in (r.get("meas") or {}).items()}
    if t is None:
        return None, None, sum(v["saturated"] for v in m.values()), len(m)
    pre = [v["saturated"] for s, v in m.items() if s < t]
    post = [v["saturated"] for s, v in m.items() if s >= t]
    return sum(pre), len(pre), sum(post), len(post)


def flips(r):
    c = r["curve"]
    first = next((i for i, x in enumerate(c) if x[1] >= 0.95), None)
    return 0 if first is None else sum(1 for x in c[first + 1:] if x[1] < 0.95)


def single_final(r):
    c = r["curve"]
    return bool(r.get("transition") is not None and r["transition"] == c[-1][0] == ITERS)


def report():
    me = sys.modules[__name__]
    store = ec.load_store(NAME)
    got = {arm: c16.runs_of(me, arm, store=store) for arm in ARMS}
    res = dict(arms={}, cmp={}, readings={})
    orc = got["ORACLE_CONV"]
    n_or = sum(1 for r in orc.values() if r.get("transition") is not None)
    valid = len(orc) == len(ORACLE_SEEDS) and n_or == len(ORACLE_SEEDS)
    res["oracle"] = dict(bound=n_or, n=len(orc), transitions={s: r.get("transition") for s, r in orc.items()}, valid=valid,
                         plain={s: r.get("transition") for s, r in got["ORACLE_PLAIN"].items()},
                         plain_acc={s: r.get("acc") for s, r in got["ORACLE_PLAIN"].items()})
    gated = ("REF", "TEMP_FLOOR", "STABLEMAX", "EMA_EVAL")
    for arm in gated:
        rs = got[arm]
        n = len(rs)
        disc = sum(ec.discovered(r) for r in rs.values())
        tr = sorted(r["transition"] for r in rs.values() if ec.discovered(r))
        pre_s = pre_n = post_s = post_n = un_s = un_n = 0
        sat_before_runs = 0
        for r in rs.values():
            a, b, c, d = sat_split(r)
            if a is None:
                un_s, un_n = un_s + c, un_n + d
            else:
                pre_s, pre_n, post_s, post_n = pre_s + a, pre_n + b, post_s + c, post_n + d
                sat_before_runs += a > 0
        taus = [((r.get("meas") or {}).get(str(r["curve"][-1][0])) or {}).get("tau") for r in rs.values()]
        res["arms"][arm] = dict(
            n=n, disc=disc, bound=sum(ec.bound(r) for r in rs.values()), band=hc.band(disc, n), wilson=hc.wilson(disc, n),
            fails=dict(__import__("collections").Counter(r.get("fail") for r in rs.values() if r.get("fail"))),
            not_disc_bound=sum(1 for r in rs.values() if ec.bound(r) and not ec.discovered(r)),
            trans_med=c16.med(tr), trans=tr, pre=(pre_s, pre_n), post=(post_s, post_n), unbound=(un_s, un_n),
            sat_before_runs=sat_before_runs, single_final=sum(single_final(r) for r in rs.values()),
            flips=sum(flips(r) > 0 for r in rs.values()), tau_end=[t for t in taus if t is not None],
            complete=n == len(ARMS[arm]["seeds"]), disc_seeds=sorted(s for s, r in rs.items() if ec.discovered(r)))
    D = {arm: {s: ec.discovered(r) for s, r in got[arm].items()} for arm in gated}
    for arm in gated[1:]:
        res["cmp"][arm] = hc.paired(D[arm], D["REF"], SEEDS)
    complete = all(res["arms"][a]["complete"] for a in gated) and len(orc) == len(ORACLE_SEEDS)
    rd = {}
    if not complete:
        rd = {"D": "INCOMPLETE, no reading"}
    elif not valid:
        rd = {"D": f"UNTESTED (ORACLE_CONV bound {n_or}/{len(ORACLE_SEEDS)})"}
    else:
        R = res["arms"]["REF"]
        ref_pre = R["pre"][0] / R["pre"][1] if R["pre"][1] else None
        for arm in gated[1:]:
            c = res["cmp"][arm]
            d1 = "more" if (c["b"] >= 3 and c["c"] == 0) else "fewer" if (c["c"] >= 3 and c["b"] == 0) else "no change at n = 10"
            A = res["arms"][arm]
            pre = A["pre"][0] / A["pre"][1] if A["pre"][1] else None
            d2 = ("delays saturation" if (pre is not None and ref_pre is not None and ref_pre - pre >= 0.2)
                  else "does not delay saturation") + f" (before-transition saturated fraction {pre if pre is None else round(pre, 2)} vs REF {ref_pre if ref_pre is None else round(ref_pre, 2)})"
            rd[arm] = f"D1 {d1} (vs REF {c['b']} vs {c['c']}); D2 {d2}"
        A = res["arms"]["EMA_EVAL"]
        ea, ra = A["single_final"] + A["flips"], R["single_final"] + R["flips"]
        rd["D3"] = (("the asymmetry shrinks under EMA" if ea < ra else "does not shrink")
                    + f" (single-final-evaluation bindings + runs with flips: EMA_EVAL {A['single_final']} + {A['flips']}, REF "
                      f"{R['single_final']} + {R['flips']})" + ("; uninformative: REF has none at n = 10" if ra == 0 else ""))
    res["readings"], res["complete"], res["valid"] = rd, complete, valid
    return res, got
