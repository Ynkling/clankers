#!/usr/bin/env python
"""
explore_h_collapse.py — EXPLORATORY, not a result. Session H, screen S58: cheaper, continuous alternatives to the
plateau-triggered KEYMASS split, and the split's missing reset-only control (audit section 2.3), on the existing Hebbian
code. Branch claude/explore-H (from claude/outside-ideas); seeds 420-459 (session H's block 420-479); outputs under
explore_out/H/. Child: explore_h_collapse_child (main line's modules at 9c5939e, read-only).

BACKGROUND: docs/reading/distant_cues_2026-10.md section 4.7 (on the main line); Raven 2607.25357 section 5.2 (Gumbel(0,1)
logit noise, training only, no annealing, its sole anti-collapse device); MoM 2502.13685 section 5.3 (Switch loss 1e-3,
Fig. 9: without it most layers collapse onto a fixed memory); Active Dendrites 2201.00042 section 5.2 (kWTA). Batch 16's
S44(a) LOCAL3_SLOW16_A bound routed 20/20 on seeds 240-259 at this configuration without any split; batch 17's S48 W_SPLIT
(the KEYMASS trigger, <= 3 splits) took the eight-stream window gate from 3/10 to 9/10. The split bundles a row copy and a
reset of W_g's Adam state; the reset alone was never controlled at k = 16 (audit 2.3). EXPECTATION STATED BEFORE ANY RUN:
since NONE's recipe bound 20/20 at this configuration in batch 16, the "> NONE by >= 4 vs 0" readings can only apply if
NONE fails on >= 4 of these 40 seeds; a ceiling makes them "does not apply" by construction, which is then a statement
about this configuration, not about the devices.

CONFIGURATION: S=4, P=4, k=16, conv (test_stream_recipe.run_attempt with ARM["A4k16"]), LOCAL3 gate (width-3 window,
explore_b16_gates.to_local), SLOW schedule (Adam: gate W_in, W_g, window 1e-3 throughout; every other trainable
parameter 1e-4 for updates 1-2400, 1e-3 after), no hinge, 28800 updates, evaluation every 1200, early stop after three
evaluations >= 0.95 past 4800 (the run path's). All learned arms on seeds 420-459, paired: same seed = same initial
parameters, window and training batches (the Gumbel noise has its own generator, seeded seed + 58000; the global RNG is
never drawn by the new code). 1 torch thread per run.

ARMS (explore_h_collapse_child):
  SPLIT      S48's KEYMASS plateau trigger (checks every 2400 from 4800 on S36's 64-sequence probe; fire if probe acc < 0.95
             and it rose < 0.02 since the previous check; <= 3 splits, >= 4800 apart; copy W_g's row c* (largest key-
             position read mass) onto c0 (smallest) with 0.1-std noise, then zero W_g's Adam state). The reference.
  RESET      the same trigger and rule; the operation = W_g's Adam state zeroed only (no row copy).
  GUMBEL_W   write gate softmax(z + n), n ~ Gumbel(0,1) i.i.d. per token and channel, training forwards only, no annealing;
             read gate softmax(z). No split.
  GUMBEL_RW  the same noise (one draw) on both gates. No split.
  SWITCH     L_aux = 1e-3 * k * sum_c f_c P_c (P_c = mean_t g_t[c], f_c = fraction of tokens with argmax g_t = c, over every
             position of the training batch) added to the loss. No split.
  NONE       neither split, noise nor loss (batch 16's LOCAL3_SLOW16_A on new seeds).
  ORACLE     the perfect gate (ceiling4k16) + conv, its own recipe (one Adam 1e-3), seeds 420-421. Validity.

OUTCOME: BOUND ROUTED (test_stream_recipe.outcome: transition not None — the first evaluation >= 0.95 that holds to the
end — and, at the end, routing_k's stream -> channel map at VAL positions one-to-one and every stream's held-out accuracy
>= 0.9). Failure classes from the same function: BOUND NOT routed, MERGED (n share), STREAM-PARTIAL one-to-one,
non-stream KEY / POSITION / OTHER. Transition time = the run's transition (update). Routing is also measured at KEY
positions (the read gate's stream -> channel map at 3j+1, every evaluation; one-to-one at the end) and printed beside the
VAL-position result; the query routing margin (routing_stats) printed per arm (median at the end).

VALIDITY (fixed now): the screen is VALID only if ORACLE binds on both seeds (420, 421) within 28800; otherwise every
reading below is UNTESTED (not a negative).

READINGS (fixed now; BOUND ROUTED on the 40 paired seeds; "X only" = seeds where X is BOUND ROUTED and the other arm is
not; every comparison printed with the exact one-sided McNemar p in both directions and both arms' Wilson intervals and
bands):
  R1 "noise replaces the split"  if (SPLIT only - GUMBEL_W only) <= 1  AND  (GUMBEL_W only >= 4 and NONE only = 0 in
                                 GUMBEL_W vs NONE). Otherwise "does not apply".
  R2 "the split is the reset"    if (SPLIT only - RESET only) <= 1. Otherwise "does not apply". Qualifier fixed now: if SPLIT
                                 and NONE differ on <= 1 discordant pair in SPLIT's favour (SPLIT only - NONE only <= 1), the
                                 split has no measurable effect here and R2 is printed as "uninformative (the split itself
                                 does nothing at this configuration)" next to its rule result.
  R3 "the balance loss helps"    if SWITCH only >= 4 and NONE only = 0 (SWITCH vs NONE). Otherwise "does not apply".
  GUMBEL_RW has no reading; its counts and comparisons with NONE and SPLIT are printed (descriptive).
DESCRIPTIVE (no reading): per arm, merges (MERGED class) and the other failure classes; transition times (median, range);
splits/resets fired; KEY-position one-to-one at the end; the plateau-vs-dead question: of the runs still unbound at 14400
(held-out accuracy < 0.95 at the 14400 evaluation), and of those "flat" there (also rose < 0.02 since 12000), how many bind
(transition not None) by 28800.
CHECKS (before any screen run; asserted, a failure stops the batch): explore_common.repro_check (X's arm A, seed 160, 2400
updates, bit for bit); the child's unit checks (the Gumbel sampler's moments; noise only on training forwards, on the
stated side, global RNG untouched; noise scale 0 gives LOCAL3's gates bitwise; the Switch loss's value and gradient by
hand; the Switch hook fires on training forwards only and leaves the logits unchanged; RESET's operation zeroes W_g's Adam
state and copies no row; RESET's trigger with the split as its operation equals S48's row for row; the KEY-position map of
the perfect gate is the identity); and, with each new code path disabled, a run equal BIT FOR BIT to batch 16's recorded
LOCAL3_SLOW16_A|240 (curve and every statistic): NONE through 2400; GUMBEL_RW at noise scale 0 through 2400; SWITCH at
alpha 0 through 2400; SPLIT and RESET with the trigger's threshold at 0 (never fires) through 6000. GUMBEL_W (CHECK
amended after the first CHECK pass, before any screen run; arms and readings unchanged): at noise scale 0 its forward is
LOCAL3's bit for bit (unit check), but its read and write gates are two autograd nodes, so the backward sums their
gradients at z in a different float order and the run does not equal the record bit for bit (it did not: 0.1860 vs
0.1831 at 1200). Its CHECK is therefore: GUMBEL_W at noise scale 0 equals BIT FOR BIT through 2400 the child's TWO_NODE
control (LOCAL3 with the two gates as two separate softmax(z) calls, no noise code), i.e. the noise path is inert.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_h_common as hc

NAME = "H/collapse"
CHILD = "explore_h_collapse_child"
MAIN = True
IDEA = "continuous alternatives to the KEYMASS split (Gumbel write/read noise, Switch loss) and its reset-only control"
SOURCE = "docs/reading/distant_cues_2026-10.md 4.7; 2607.25357 5.2; 2502.13685 5.3; audit 2.3"
CHANGE = "batch 16's LOCAL3_SLOW16_A (S=4, P=4, k=16, conv) + one of: split, reset, Gumbel W, Gumbel RW, Switch, nothing"
PAIRING = "seeds 420-459, all learned arms paired (same initial parameters, window and batches)"
LR = 1e-3
ITERS = 28800
SEEDS = tuple(range(420, 460))
ORACLE_SEEDS = (420, 421)
B16 = "window_scale"
B16_SHA = "01767493e840"
REC_SEED = 240
CHECK_SHORT, CHECK_TRIG = 2400, 6000
_sl = ("LOCAL3 + SLOW (gate W_in/W_g/window 1e-3 throughout; the rest 1e-4 for 1-2400, 1e-3 after); no hinge")
ARMS = {
    "SPLIT": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="+ S48's KEYMASS split (<= 3)", sched=_sl + "; + split"),
    "RESET": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="+ the trigger, W_g Adam reset only",
                  sched=_sl + "; + reset only"),
    "GUMBEL_W": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="+ Gumbel(0,1) on write-gate logits",
                     sched=_sl + "; + Gumbel write"),
    "GUMBEL_RW": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="+ Gumbel(0,1) on both gates",
                      sched=_sl + "; + Gumbel read+write"),
    "SWITCH": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="+ Switch loss 1e-3", sched=_sl + "; + Switch 1e-3"),
    "NONE": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="LOCAL3_SLOW16_A", sched=_sl),
    "ORACLE": dict(opt="adam", lr=LR, iters=ITERS, seeds=ORACLE_SEEDS, label="perfect gate ceiling4k16 + conv",
                   sched="one Adam 1e-3 for every parameter"),
}
CUT = set()


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ARMS[arm]["iters"]))


def timing(arms):
    return child("timing", dict(which=list(arms), steps=300))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def b16_record(arm, seed):
    st = ec.load_store(B16)["runs"]
    for k, r in st.items():
        a, s, h, cpu = k.split("|", 3)
        if a == arm and int(s) == seed and h == B16_SHA and r.get("ok"):
            return r
    return None


def eq_upto(r, ref, t):
    c1 = [c for c in r["curve"] if c[0] <= t]
    c2 = [c for c in ref["curve"] if c[0] <= t]
    s1 = [s for s in r["stats"] if s["step"] != "end" and s["step"] <= t]
    s2 = [s for s in ref["stats"] if s["step"] != "end" and s["step"] <= t]
    return bool(c1) and c1 == c2 and s1 == s2 and len(c1) == t // 1200, c1


def check():
    ref = b16_record("LOCAL3_SLOW16_A", REC_SEED)
    rows = [tuple(x) for x in hc.check_stats()]
    jobs = [("NONE", dict(arm="NONE", seed=REC_SEED, iters=CHECK_SHORT), CHECK_SHORT, "as is"),
            ("GUMBEL_W", dict(arm="GUMBEL_W", seed=REC_SEED, iters=CHECK_SHORT, gumbel_scale=0.0), CHECK_SHORT, "noise scale 0"),
            ("TWO_NODE", dict(arm="TWO_NODE", seed=REC_SEED, iters=CHECK_SHORT), CHECK_SHORT, "control"),
            ("GUMBEL_RW", dict(arm="GUMBEL_RW", seed=REC_SEED, iters=CHECK_SHORT, gumbel_scale=0.0), CHECK_SHORT, "noise scale 0"),
            ("SWITCH", dict(arm="SWITCH", seed=REC_SEED, iters=CHECK_SHORT, alpha=0.0), CHECK_SHORT, "alpha 0"),
            ("SPLIT", dict(arm="SPLIT", seed=REC_SEED, iters=CHECK_TRIG, thr=0.0, total=ITERS), CHECK_TRIG, "threshold 0"),
            ("RESET", dict(arm="RESET", seed=REC_SEED, iters=CHECK_TRIG, thr=0.0, total=ITERS), CHECK_TRIG, "threshold 0")]
    with ThreadPoolExecutor(max_workers=4) as ex:
        fu = ex.submit(child, "checks", {})
        res = list(ex.map(lambda j: (j, child("run", j[1])), jobs))
        rows += [tuple(x) for x in fu.result()]
    two = next(r for (j, r) in res if j[0] == "TWO_NODE")
    for (arm, p, t, how), r in res:
        if arm == "TWO_NODE":
            eq, c = eq_upto(r, ref, t)
            print(f"  (TWO_NODE control vs the record through {t}: {'equal' if eq else 'differs'} (float order of the backward); "
                  f"curve {c})", flush=True)
            continue
        eq, c = eq_upto(r, ref, t) if ref is not None else (False, None)
        extra = ""
        if arm in ("SPLIT", "RESET"):
            extra = f"; trigger checks at {[c_['step'] for c_ in r['checks']]}, fired {r['splits']}"
            eq = eq and r["splits"] == 0 and len(r["checks"]) >= 2
        if arm.startswith("GUMBEL"):
            gc = (r.get("keyroute") or {}).get("end", {}).get("gumbel_calls")
            extra = f"; noisy-path forwards {gc}"
            eq = eq and gc == t
        if arm == "GUMBEL_W":
            eq2, c2_ = eq_upto(r, two, t)
            rows.append((f"GUMBEL_W (noise scale 0) equals the TWO_NODE control bit for bit through {t} (curve {c2_}; statistics); "
                         f"noisy-path forwards {gc}", eq2 and gc == t))
            continue
        if arm == "SWITCH":
            extra = f"; Switch hook calls {r.get('switch_n')}"
            eq = eq and r.get("switch_n") == t
        rows.append((f"{arm} ({how}) equals batch 16's LOCAL3_SLOW16_A|{REC_SEED} (code {B16_SHA}, {ref and ref['cpu']}) bit for "
                     f"bit through {t} (curve {c}; statistics){extra}", eq))
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def kr_end(r):
    return ((r.get("keyroute") or {}).get("end") or {})


def margin_end(r):
    return (r.get("end") or {}).get("margin")


def report(out=print):
    me = sys.modules[__name__]
    store = ec.load_store(NAME)
    got = {arm: c16.runs_of(me, arm, store=store) for arm in ARMS}
    res = dict(arms={}, cmp={}, readings={})
    learned = [a for a in ARMS if a != "ORACLE"]
    orc = got["ORACLE"]
    n_or = sum(1 for r in orc.values() if r.get("transition") is not None)
    valid = len(orc) == len(ORACLE_SEEDS) and n_or == len(ORACLE_SEEDS)
    res["oracle"] = dict(bound=n_or, n=len(orc), transitions={s: r.get("transition") for s, r in orc.items()}, valid=valid)
    for arm in learned:
        rs = got[arm]
        n = len(rs)
        k = sum(hc.br(r) for r in rs.values())
        nb = sum(1 for r in rs.values() if r.get("transition") is not None)
        tr = sorted(r["transition"] for r in rs.values() if hc.br(r))
        unb14 = [s for s, r in rs.items() if hc.flat_at(r, 14400, 12000)[0]]
        flat14 = [s for s, r in rs.items() if hc.flat_at(r, 14400, 12000)[1]]
        key11 = sum(1 for r in rs.values() if hc.br(r) and kr_end(r).get("one_to_one_key"))
        fired = [sum(1 for c in r.get("checks") or [] if c.get("fired")) for r in rs.values()]
        res["arms"][arm] = dict(
            n=n, br=k, bound=nb, band=hc.band(k, n), wilson=hc.wilson(k, n), classes=hc.classes(rs),
            merged=sum(1 for r in rs.values() if hc.fclass(r) == "MERGED"),
            trans_med=c16.med(tr), trans_min=tr[0] if tr else None, trans_max=tr[-1] if tr else None, trans=tr,
            unbound14=len(unb14), unbound14_bind=sum(1 for s in unb14 if rs[s].get("transition") is not None),
            flat14=len(flat14), flat14_bind=sum(1 for s in flat14 if rs[s].get("transition") is not None),
            br_key_too=key11, margin_med=c16.med([margin_end(r) for r in rs.values()]),
            ops=sum(fired), ops_runs=sum(1 for f in fired if f), complete=n == len(ARMS[arm]["seeds"]),
            bound_seeds=sorted(s for s, r in rs.items() if hc.br(r)))
    B = {arm: {s: hc.br(r) for s, r in got[arm].items()} for arm in learned}
    for x, y in (("GUMBEL_W", "SPLIT"), ("GUMBEL_W", "NONE"), ("RESET", "SPLIT"), ("SWITCH", "NONE"), ("SPLIT", "NONE"),
                 ("RESET", "NONE"), ("GUMBEL_RW", "NONE"), ("GUMBEL_RW", "SPLIT"), ("SWITCH", "SPLIT")):
        res["cmp"][(x, y)] = hc.paired(B[x], B[y], SEEDS)
    cm = res["cmp"]
    complete = all(res["arms"][a]["complete"] for a in learned) and len(orc) == len(ORACLE_SEEDS)
    if not complete:
        rd = {k: "INCOMPLETE, no reading" for k in ("R1", "R2", "R3")}
    elif not valid:
        rd = {k: f"UNTESTED (ORACLE bound {n_or}/{len(ORACLE_SEEDS)})" for k in ("R1", "R2", "R3")}
    else:
        a, b = cm[("GUMBEL_W", "SPLIT")], cm[("GUMBEL_W", "NONE")]
        r1 = (a["c"] - a["b"] <= 1) and (b["b"] >= 4 and b["c"] == 0)
        rd = {"R1": ("noise replaces the split" if r1 else "does not apply")
              + f" (SPLIT only {a['c']} - GUMBEL_W only {a['b']} = {a['c'] - a['b']} [<= 1: {a['c'] - a['b'] <= 1}]; GUMBEL_W vs "
                f"NONE {b['b']} vs {b['c']} [>= 4 vs 0: {b['b'] >= 4 and b['c'] == 0}])"}
        a, sn = cm[("RESET", "SPLIT")], cm[("SPLIT", "NONE")]
        r2 = a["c"] - a["b"] <= 1
        rd["R2"] = (("the split is the reset" if r2 else "does not apply")
                    + f" (SPLIT only {a['c']} - RESET only {a['b']} = {a['c'] - a['b']} [<= 1])"
                    + (f"; uninformative (the split itself does nothing at this configuration: SPLIT vs NONE {sn['b']} vs {sn['c']})"
                       if sn["b"] - sn["c"] <= 1 else f"; SPLIT vs NONE {sn['b']} vs {sn['c']}"))
        a = cm[("SWITCH", "NONE")]
        rd["R3"] = (("the balance loss helps" if (a["b"] >= 4 and a["c"] == 0) else "does not apply")
                    + f" (SWITCH vs NONE {a['b']} vs {a['c']} [>= 4 vs 0])")
    res["readings"] = rd
    res["complete"], res["valid"] = complete, valid
    return res, got


def per_seed_log(got, out=print):
    learned = [a for a in ARMS if a != "ORACLE"]
    out("  per seed (outcome, transition, map at the end VAL / KEY; splits or resets fired):")
    out("    seed | " + " | ".join(f"{a:<30}" for a in learned))
    for s in SEEDS:
        cells = []
        for a in learned:
            r = got[a].get(s)
            if r is None:
                cells.append(f"{'not run':<30}")
                continue
            o = r["outcome"].replace("BOUND ROUTED", "BR").replace("BOUND NOT routed", "BNR").replace("non-stream ", "")
            f = sum(1 for c in r.get("checks") or [] if c.get("fired"))
            cells.append(f"{o[:12]:<12} {str(r['transition']):>5} {hc.per_channel(r['end'].get('ch_map')):>7}/"
                         f"{hc.per_channel(kr_end(r).get('ch_map_key')):<7}" + (f"x{f}" if f else "  "))
        out(f"    {s:>4} | " + " | ".join(cells))
