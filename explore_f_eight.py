#!/usr/bin/env python
"""
explore_f_eight.py — EXPLORATORY, not a result. Session F, screen S76: why the delta recipe stalls at eight streams
(S54(c): BOUND ROUTED 3/10 on delta channels vs 9/10 Hebbian; the delta oracle bound at 22800 / 28800 against 2400 / 4800),
and whether it can be made not to.

SEEDS: the assigned block 1000-1059 appears in recorded runs (results/X/router_reliability_results.json), and so do
1100-1159 and 1200-1259; shifted by 100 three times to 1300-1359, which appears nowhere in explore_out/ or results/ on any
branch. Phase 1 uses 1300-1303, Phase 2 1310-1319.

PATH (explore_f_eight_child; main-line modules at 9c5939e, as S54(c) and Session H's S61): test_window_gate's Part C/D path,
test_stream_curriculum.run_sc with ARM["D8"]: S = 8, P = 4, k = 16, conv, all 8 streams from step 1; 43200 updates;
evaluation every 1200 on 2048 held-out sequences; early stop after 3 evaluations >= 0.95; 1 torch thread.

PHASE 1, the oracle (seeds 1300-1303): the perfect gate (ceil8_D8), one Adam 1e-3 on every parameter.
  O_HEBB (Hebbian, the reference); O_B1_L2 (delta, β = 1, L2 keys); O_B025_L2 (β = 0.25, L2 keys); O_B1_RAW (β = 1, raw keys:
  the Hebbian model's x_sparse); O_B025_RAW (β = 0.25, raw keys). Tied write, decay 0.95. A run whose held-out loss becomes
  non-finite is stopped at that evaluation and counted as DIVERGED (unbound).
  Per seed s, a delta arm X MEETS THE MARK on s if X binds with transition <= 2 x O_HEBB's transition on s (if O_HEBB does
  not bind on s: if X binds). X "binds within 2x" if it meets the mark on >= 3 of the 4 seeds; "does not" if on <= 2.
  READINGS (fixed before any run):
    "the slowness is the overwrite"          if O_B025_L2 and O_B025_RAW bind within 2x and O_B1_L2 and O_B1_RAW do not;
    "the slowness is the key normalisation"  if O_B1_RAW and O_B025_RAW bind within 2x and O_B1_L2 and O_B025_L2 do not;
    "neither" otherwise; then the oracles' accuracy curves and their held-out accuracy at 12000 and 24000 are reported.
  DELTA* = the delta variant with the earliest median transition over its 4 seeds (unbound or diverged = infinity; the median
  of 4 is the mean of the middle two, infinite if either is); ties: more seeds bound, then the lower mean transition over
  bound seeds, then the order O_B1_L2, O_B025_L2, O_B1_RAW, O_B025_RAW. Written to explore_out/F/eight_delta_star.json and
  committed before Phase 2 runs.
PHASE 2, the recipe (seeds 1310-1319), paired: LOCAL3 + SLOW (gate W_in, W_g, window 1e-3 throughout; the rest 1e-4 for
  updates 1-2400, 1e-3 after; no hinge) + the KEYMASS trigger with Session H's simplified split (an exact copy of W_g's
  busiest row onto the idlest, no noise, no Adam reset; checks every 2400 from 4800, fire if the probe accuracy < 0.95 and
  rose < 0.02, <= 3 splits >= 4800 apart: test_window_gate's Part D trigger with H's operation).
    HEBB_SPLIT   Hebbian channels (H's S61 COPY_NONOISE: 9/10 on 480-489).
    DELTA_SPLIT  DELTA* channels.
  OUTCOME: BOUND ROUTED (test_stream_recipe.outcome) with its transition; MERGED and the other outcomes as that function
  prints them. Wilson 95% and bands with every count; one-sided exact McNemar both ways.
  VALIDITY: O_HEBB binds on >= 2 of its 4 seeds and DELTA*'s oracle arm binds on >= 2 of its 4; else the reading is UNTESTED.
  READINGS (fixed before any run), b = DELTA_SPLIT only, c = HEBB_SPLIT only (BOUND ROUTED):
    "the delta recipe keeps eight streams"   if c − b <= 1 (DELTA_SPLIT >= HEBB_SPLIT − 1 discordant pair);
    "the delta recipe loses eight streams"   if c >= 4 and b = 0;
    "inconclusive" otherwise.
  Reported: every split (update, c* -> c0, target labelled ok: c* holds >= 2 streams and c0 none), MERGED counts per arm.
CHECKS (child.checks + explore_f_delta_checks — β = 0 equals the Hebbian memory; the recorded LOCAL3_SLOW|160 reproduced
with the delta path disabled): HEBB_SPLIT on seed 487 reproduces H's COPY_NONOISE|487 (claude/explore-H 7e9197f, 2.10GHz)
through 7200 (its first split at 4800 + one check) bit for bit, curve, statistics and trigger rows; in Phase 2, DELTA_SPLIT
with its delta path disabled does too; copy_nonoise_op copies the row exactly and leaves the other rows and the Adam state
untouched; the oracle variants' memories as stated; β = 0 with raw keys equals the Hebbian oracle model.
"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore_common16 as c16
import explore_f_common as fc

NAME = fc.name("eight")
CHILD = "explore_f_eight_child"
MAIN = True
LR, ITERS = 1e-3, 43200
OSEEDS, SEEDS = (1300, 1301, 1302, 1303), tuple(range(1310, 1320))
VARIANTS = {"O_B1_L2": (1.0, True), "O_B025_L2": (0.25, True), "O_B1_RAW": (1.0, False), "O_B025_RAW": (0.25, False)}
STAR = os.path.join(fc.F_DIR, "eight_delta_star.json")


def _arms():
    arms = {"O_HEBB": dict(opt="adam", lr=LR, iters=ITERS, seeds=OSEEDS, label="perfect gate, Hebbian", sched="Adam 1e-3")}
    for k, (b, n) in VARIANTS.items():
        arms[k] = dict(opt="adam", lr=LR, iters=ITERS, seeds=OSEEDS, label=f"perfect gate, delta beta {b}, {'L2' if n else 'raw'} keys",
                       sched="Adam 1e-3")
    if os.path.exists(STAR):
        sl = "gate 1e-3 throughout; rest 1e-4 for 1-2400, 1e-3 after; KEYMASS trigger, exact row copy"
        arms["HEBB_SPLIT"] = dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="LOCAL3+SLOW+copy split, Hebbian", sched=sl)
        arms["DELTA_SPLIT"] = dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="LOCAL3+SLOW+copy split, DELTA*", sched=sl)
    return arms


ARMS = _arms()


def all_records(arm):
    """{seed: record} including diverged (ok False with a DIVERGED error) records on this SHA and CPU."""
    st = fc.load(sys.modules[__name__])
    sha = c16.STATE["shas"][NAME][:12]
    out = {}
    for s in ARMS[arm]["seeds"]:
        r = st["runs"].get(f"{arm}|{s}|{sha}|{c16.STATE['batch_cpu']}")
        if r is not None and (r.get("ok") or "DIVERGED" in str(r.get("error"))):
            out[s] = r
    return out


def trans(r):
    return r["transition"] if (r.get("ok") and r.get("transition") is not None) else math.inf


def choose_star():
    me = sys.modules[__name__]
    if NAME not in c16.STATE["shas"]:
        fc.setup([me])
    R = {a: all_records(a) for a in ["O_HEBB"] + list(VARIANTS)}
    assert all(len(R[a]) == 4 for a in R), f"phase 1 incomplete: {[(a, len(R[a])) for a in R]}"

    def key(a):
        ts = sorted(trans(R[a][s]) for s in OSEEDS)
        med = (ts[1] + ts[2]) / 2
        nb = sum(t < math.inf for t in ts)
        mean_b = sum(t for t in ts if t < math.inf) / nb if nb else math.inf
        return (med, -nb, mean_b, list(VARIANTS).index(a))
    best = min(VARIANTS, key=key)
    b, n = VARIANTS[best]
    star = dict(arm=best, beta=b, normalize=n, key=[str(x) for x in key(best)])
    json.dump(star, open(STAR, "w"), indent=1)
    print(f"  DELTA* = {star}", flush=True)
    return star


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ARMS[arm]["iters"]))


def timing(arm, steps):
    return child("timing", dict(arm=arm, steps=steps))


def check():
    ok = True
    for nm, v in child("checks", {}):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def line(arm, r):
    if not r.get("ok"):
        return f"FAILED {r.get('error')}"
    return (f"acc {r['acc']:.3f}  trans {str(r['transition']):>5}  {r.get('outcome')}  splits {r.get('splits')}  map "
            f"{c16.chmap_str(r['end'].get('ch_map'))}")


def report():
    import explore_f_reports3 as fr
    return fr.report_s76()


if __name__ == "__main__":
    import explore_f_eight as me
    if len(sys.argv) > 1 and sys.argv[1] == "--choose-star":
        choose_star()
    else:
        fc.drive("s76", [me], sys.argv[1:])
