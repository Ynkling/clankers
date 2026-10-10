#!/usr/bin/env python
"""
explore_f_trigger.py — EXPLORATORY, not a result. Session F, screen S81: the split's trigger on delta memory. S76's delta
recipe bound 9/10 at eight streams, but 6 of its 12 splits fired before the map had merged (target not labelled ok);
here two devices that might stop that, alone and together, against the rule as it stands.

SEEDS: 2020-2029 (the assigned block 2000-2059 appears in no record on any branch; see explore_f_configs).

PATH (explore_f_trigger_child; main-line modules at 9c5939e): S76's Phase 2 exactly — S = 8, P = 4, k = 16, conv, all 8
streams from step 1 (test_stream_curriculum.run_sc with D8), 43200 updates, evaluation every 1200 on 2048 held-out
sequences, early stop after 3 evaluations >= 0.95; LOCAL3 + SLOW (gate W_in, W_g, window at 1e-3 throughout; the rest at
1e-4 for updates 1-2400, 1e-3 after; no hinge); DELTA* channels (β = 1, L2 keys; explore_out/F/eight_delta_star.json);
H's exact-copy split (W_g[c0] = W_g[c*] exactly, c* / c0 the largest / smallest key-position read mass on the probe; no
noise; W_g's Adam state untouched). THE RULE AS IT STANDS (BASE): checks every 2400 from 2400 (2400 is the reference)
through 40800; at a check the rule reads its accuracy r and the previous check's r_prev; eligible if the update is >=
4800, r < 0.95 and r - r_prev < 0.02; at most 3 splits, >= 4800 updates apart (capped or gapped checks are logged as
blocked).
ARMS (paired: on a seed every arm starts from the same parameters and sees the same batches):
    BASE        the rule reads S36's 64-sequence probe (generator 12345): S76's DELTA_SPLIT.
    GUARD       BASE, and no split while the probe's accuracy is below 0.5 (a check is eligible only if r >= 0.5): H's
                flat-start reading (S78: a split spent while held-out accuracy is 0.09-0.50, before any stream has bound,
                is wasted).
    EVAL        BASE, but the rule reads the run's own held-out evaluation at the check's update (the 2048-sequence
                evaluation the path has just computed and recorded in its curve) instead of the 64-sequence probe: Session
                R's finding that a 64-query probe can read >= 0.95 while the model is at 0.88.
    GUARD_EVAL  both: the rule reads the evaluation, and no split while the evaluation is below 0.5.
    O_DELTA_C   the perfect gate (ceil8_D8) on DELTA*, one Adam at 1e-3, seeds 2020-2021: VALIDITY.
  The arm names and the guard's rule are Session H's S84 ones (BASE, GUARD, EVAL), so the two screens pair by name.
RECORDED at every check of every arm: both readings (the probe and the evaluation, each with its previous check's value)
  and the rule's eligibility read on each (with the arm's guard). A check is a PROBE-VS-EVALUATION DISAGREEMENT if the
  two eligibilities differ (one reading says "plateau below 0.95" and the other does not). Every firing: its update,
  c* -> c0, both readings, whether the check was a disagreement, and whether its target was LABELLED OK (c* holds >= 2
  streams and c0 none on the labelled stream -> channel map just before it; s37c.targeting).
OUTCOME: BOUND ROUTED (test_stream_recipe.outcome == "BOUND ROUTED": bound, one-to-one, every stream >= 0.9) with the
  transition; unbound runs' outcome under v1 and v2 (explore_f_fail_class_v2.outcome_v2). Wilson 95% intervals and bands
  with every count; exact one-sided McNemar both ways against BASE.
VALIDITY (fixed before any run): O_DELTA_C binds on both 2020 and 2021 within 43200; otherwise every reading is UNTESTED.
READINGS (fixed before any run). For an arm X against BASE on the 10 paired seeds (BOUND ROUTED): b = seeds where X alone
  succeeds, c = seeds where BASE alone succeeds.
  "the guard aims the split"            if GUARD's firings are labelled ok on >= 90% of them (GUARD fires at least once)
                                        AND c - b <= 1 for GUARD (GUARD >= BASE - 1 discordant pair). (S76's BASE: 6/12, 50%.)
  "the evaluation is the better probe"  if c - b <= 1 for EVAL AND none of EVAL's firings is at a probe-vs-evaluation
                                        disagreement (as defined above: every EVAL firing is one the probe would also have
                                        made at that check).
  "the guard costs binding"             if BASE beats GUARD by c >= 4 and b = 0;
  "the evaluation costs binding"        if BASE beats EVAL by c >= 4 and b = 0  (the two together: "either device costs
                                        binding"); the same rule on GUARD_EVAL is printed as "both devices cost binding".
  A device with none of its readings is reported as "no reading applies". GUARD_EVAL's labelled-ok rate, disagreement
  firings and pairing with BASE are printed beside the readings.
PRINTED, not readings: per arm, firings (total; runs; labelled ok), guarded checks, disagreement checks by direction (the
  probe eligible and the evaluation not, and the reverse), checks where the probe read >= 0.95 while the evaluation was
  below 0.95 (R's seed-914 mode), MERGED counts, median transitions.
CHECKS (asserted before any run): explore_f_delta_checks (26) at the start of every segment; then (child):
  - BASE on seed 1315 reproduces S76's DELTA_SPLIT|1315 (explore_out/F/eight_results.json, this CPU) bit for bit through
    7200 (its first split at 4800 + one check): curve, statistics, and the trigger rows.
  - GUARD and EVAL with their condition disabled (GUARD: guard 0 and no guard; EVAL: reading the probe; GUARD_EVAL: both)
    equal BASE bit for bit through the first split (+ one check): curve, statistics, trigger rows.
  - EVAL's reading at every check is the curve's held-out accuracy at that update exactly, and the captured evaluations
    equal the curve.
  - The trigger's mechanics on an untrained model (threshold 1.1, rise 1, no training between checks): BASE fires at 4800,
    9600, 14400; GUARD (probe < 0.5) never fires; GUARD_EVAL on held-out readings 0.3 then 0.6 from 7200 fires at 7200,
    12000, 16800.
  - The perfect gate on DELTA* at S = 8 is one-hot and equal to the Hebbian model's; DELTA* as committed.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore_common16 as c16
import explore_f_common as fc

NAME = fc.name("trigger")
CHILD = "explore_f_trigger_child"
MAIN = True
LR, ITERS = 1e-3, 43200
SEEDS, OSEEDS = tuple(range(2020, 2030)), (2020, 2021)
_sl = "gate 1e-3 throughout; rest 1e-4 for 1-2400, 1e-3 after; KEYMASS trigger, exact row copy"
ARMS = {
    "BASE": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="the rule on the 64-sequence probe (S76's DELTA_SPLIT)", sched=_sl),
    "GUARD": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="BASE + no split below 0.5", sched=_sl),
    "EVAL": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="the rule on the held-out evaluation", sched=_sl),
    "GUARD_EVAL": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="EVAL + no split below 0.5", sched=_sl),
    "O_DELTA_C": dict(opt="adam", lr=LR, iters=ITERS, seeds=OSEEDS, label="perfect gate, DELTA*", sched="Adam 1e-3"),
}


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ARMS[arm]["iters"]))


def timing(arm, steps):
    return child("timing", dict(arm=arm, steps=steps))


PARTS = ("base", "guard0", "guardnone", "evalprobe", "geprobe", "evalread", "mech")


def check():
    """The child's CHECK parts, 4 at a time (each its own child process, 1 thread)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(fc.WORKERS) as ex:
        res = list(ex.map(lambda part: child("checks", dict(part=part)), PARTS))
    ok = True
    for rows in res:
        for nm, v in rows:
            print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
            ok &= bool(v)
    return ok


def line(arm, r):
    if not r.get("ok"):
        return f"FAILED {r.get('error')}"
    fired = [c for c in r.get("checks", []) if c.get("fired")]
    return (f"acc {r['acc']:.3f}  trans {str(r['transition']):>5}  {r.get('outcome')}  splits "
            f"{[(c['step'], 'ok' if c.get('key_ok') else 'NOT', 'D' if c.get('disagree') else '') for c in fired]}  map "
            f"{c16.chmap_str(r['end'].get('ch_map'))}")


def report():
    import explore_f_reports4 as fr
    return fr.report_s81()


if __name__ == "__main__":
    import explore_f_trigger as me
    fc.drive("s81", [me], sys.argv[1:])
