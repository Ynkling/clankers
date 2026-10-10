#!/usr/bin/env python
"""
explore_f_configs.py — EXPLORATORY, not a result. Session F, screen S80: the delta recipe at the main line's other two
configurations (two streams, four streams), paired with the Hebbian memory. This file is configuration (a) and the whole
screen's readings; configuration (b) is explore_f_configs_b (main-line modules). One batch runs both.

DELTA* (explore_out/F/eight_delta_star.json, S76's Phase-1 choice, committed 1db6d28): the delta rule with β = 1 fixed,
L2-normalised keys, tied write, decay 0.95 (explore_f_delta_fast; the parallel form's numbers). It adds no parameter, so
DELTA and HEBB arms on a seed share every initial parameter and every training batch (paired bit for bit up to the
memory). The Hebbian memory is the main line's.

SEEDS: the assigned block 2000-2059 appears in no record key ("arm|seed|...") and no "seed" field in explore_out/ or
results/ on any branch (the string "2000" appears only as a step key of dense statistics). (a) 2000-2009, (b) 2010-2019;
S81 2020-2029.

(a) TWO STREAMS: S = 2, P = 4, k = 2, no conv (test_short_conv.TASK, grouped), 24000 updates; test_window_gate's Part A
  path for WIN3_SLOW, which is batch 16's S43 LOCAL3_SLOW path (the main line's CHECK 135 asserts the two equal through
  2400 at seed 160), run as S54(a) on this branch's modules (explore_window_recipe_child): the LOCAL3 window gate (W_h
  frozen and unused); one Adam, gate group (W_in, W_g, gate_conv.conv_w) at 1e-3 throughout, every other trainable
  parameter at 1e-4 for updates 1-2400 and 1e-3 after (switched after the evaluation at 2400); no hinge; NO SPLIT (k = S);
  S38's decoder at 1200 / 2400 / 4800 / end (S43's path; its own data, training untouched); evaluation every 1200 on
  2048 held-out sequences; early stop after 3 evaluations >= 0.95.
    HEBB_A     WIN3_SLOW on the Hebbian memory, seeds 2000-2009.
    DELTA_A    WIN3_SLOW on DELTA*, seeds 2000-2009 (paired with HEBB_A).
    O_DELTA_A  the perfect gate on DELTA* (test_channel_binding's 'ceiling', k = 2), one Adam at 1e-3 on every parameter
               (the perfect gate's own recipe), seeds 2000-2001: VALIDITY.
    O_HEBB_A   the same on the Hebbian memory, seeds 2000-2001 (printed, not used by a reading).
  OUTCOME: DISCOVERED = bound (a transition) and final VAL cos < 0.5 (explore_common.discovered).
  FAILURE CLASSES (diagnostic): every run that is not DISCOVERED; a bound one is listed as "BOUND, VAL cos >= 0.5";
    an unbound one is classed by fail_class_v2 (explore_f_fail_class_v2.fail_class_v2, the main line's ba8901a module,
    copied unchanged: η² by key >= 0.5 is KEY before the margin rule), with v1 (test_router_layout.fail_class) beside it;
    every run where the two differ is listed.
  PROBE (descriptive, no reading): S77's probe at the end of every (a) run and on the same seed's model at
    initialisation: Session G's S55 logistic regression (explore_f_probe_common.residuals and fit_score, unchanged) on
    the residual stream, from the embedding, entering layers 1-3, and the final layer, at the 8 body KEY positions (3j+1)
    and VAL positions (3j+2) of the grouped layout, to the triple's stream (its CTX); 2000 probe sequences (seed 55000),
    fit on 1600 and scored on the other 400 (split by sequence); chance 0.5. Printed: per arm the median at each layer
    and role, the runs with L3 >= 0.9 at both KEY and VAL, and DELTA_A - HEBB_A at L3 per seed. (Does the delta stack
    carry the stream at two streams, where the gate has a channel per stream?)
  Reference (other CPUs and seeds; counts only): WIN3_SLOW pooled X + L 69/80.

(b) FOUR STREAMS: S = 4, P = 4, k = 16, conv, 28800 updates; test_window_gate's Part B path (test_stream_recipe.run_attempt
  with A4k16, via explore_b16_main.path("k16"), on the main line's module tree at 9c5939e), WIN3_SLOW16 = LOCAL3 + SLOW
  as in (a) (batch 16's LOCAL3_SLOW16_A), PLUS H's exact-copy split exactly as S76's Phase 2: checks every 2400 from
  4800 (2400 the reference) through 26400 on S36's 64-sequence probe; fire if the probe accuracy < 0.95 and rose < 0.02
  since the previous check; <= 3 splits, >= 4800 apart; W_g[c0] = W_g[c*] exactly (c* the largest key-position read
  mass, c0 the smallest), no noise, W_g's Adam state untouched. The split is part of the recipe here because k = 16 > S
  (the spec's "no split" is for k = S); a run in which it never fires is WIN3_SLOW16's path bit for bit (CHECK).
    HEBB_B     on the Hebbian memory, seeds 2010-2019.
    DELTA_B    on DELTA*, seeds 2010-2019 (paired).
    O_DELTA_B  the perfect gate (test_stream_recipe's ceiling4k16) on DELTA*, one Adam 1e-3, seeds 2010-2011: VALIDITY.
    O_HEBB_B   the same on the Hebbian memory, seeds 2010-2011 (printed).
  OUTCOME: BOUND ROUTED = test_stream_recipe.outcome == "BOUND ROUTED" (bound, one-to-one, every stream >= 0.9), with
    the transition. Unbound runs: outcome under v1 (test_stream_recipe.outcome) and v2 (explore_f_fail_class_v2.outcome_v2).
    Every split with its update, c* -> c0 and whether its target was labelled ok (c* holds >= 2 streams and c0 none).
  Reference (other CPUs and seeds; counts only): WIN3_SLOW16 pooled X + L 38/40 (no split).

VALIDITY (fixed before any run), per configuration: the delta perfect gate (O_DELTA_A, O_DELTA_B) binds on both of its 2
  seeds within the configuration's budget; otherwise that configuration's reading is UNTESTED (not a negative).

READINGS (fixed before any run), per configuration, on its outcome (DISCOVERED in (a), BOUND ROUTED in (b)), paired over
  its 10 seeds; b = seeds where DELTA alone succeeds, c = seeds where HEBB alone succeeds:
    "the delta recipe keeps it"   if c - b <= 1   (DELTA >= HEBB - 1 discordant pair);
    "the delta recipe loses it"   if c >= 4 and b = 0;
    "inconclusive"                otherwise.
  Exact one-sided McNemar both ways printed; Wilson 95% intervals and bands with every count; the transitions per seed
  and their medians per arm.

CHECKS (asserted before any run): explore_f_delta_checks (26: β = 0 equals the Hebbian memory; the recorded
  LOCAL3_SLOW|160 reproduced with the delta path disabled; ...) at the start of every segment; then
  (a) child: HEBB_A, and DELTA_A with the delta path disabled (impl "hebb"), on seed 160 reproduce batch 16's
      LOCAL3_SLOW|160 (explore_out/window_recipe_results.json, Xeon @ 2.10GHz) bit for bit through 2400 (curve, gradient
      norms, statistics, decoder); DELTA_A's model is HEBB_A's with only the memory changed (same parameters, none added;
      β 1, L2 keys = DELTA*); the two groups and their lrs at updates 1, 2400, 2401; G's residual loop gives each
      memory's own logits bit for bit at k = 2; the probe's positions and labels on the grouped layout and its 1600 / 400
      split; G's synthetic probe check; the perfect gate on DELTA* is one-hot and equal to the Hebbian model's.
  (b) child: HEBB_B, and DELTA_B with the delta path disabled, on seed 242 reproduce batch 16's LOCAL3_SLOW16_A|242
      (explore_out/window_scale_results.json, 2.10GHz) bit for bit through 2400 (curve and statistics); HEBB_B with the
      threshold at 0 (it never fires) equals the path without a trigger bit for bit through 7200; copy_nonoise_op at
      k = 16 copies the row exactly and leaves the other rows and the Adam state untouched (H's check); the classifier
      copy equals ba8901a's file; v2 equals v1 on every stored Session F record except where η² by key >= 0.5 and the
      margin rule fired (margin >= 0.25) — those are listed; the perfect gate on DELTA* at k = 16 is one-hot.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore_common16 as c16
import explore_f_common as fc

NAME = fc.name("configs_a")
CHILD = "explore_f_configs_child"
MAIN = False
LR, ITERS = 1e-3, 24000
SEEDS, OSEEDS = tuple(range(2000, 2010)), (2000, 2001)
REF = ("WIN3_SLOW", 69, 80)
_slow = "gate W_in/W_g/window 1e-3 throughout; rest 1e-4 for 1-2400, 1e-3 after; no split"
ARMS = {
    "HEBB_A": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="(a) WIN3_SLOW, Hebbian", sched=_slow),
    "DELTA_A": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="(a) WIN3_SLOW, DELTA*", sched=_slow),
    "O_DELTA_A": dict(opt="adam", lr=LR, iters=ITERS, seeds=OSEEDS, label="(a) perfect gate, DELTA*", sched="Adam 1e-3"),
    "O_HEBB_A": dict(opt="adam", lr=LR, iters=ITERS, seeds=OSEEDS, label="(a) perfect gate, Hebbian", sched="Adam 1e-3"),
}


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
    e = r["end"]
    pr = r.get("probe") or {}
    l3 = "/".join(f"{pr[f'L3|{x}']['acc']:.2f}" for x in "KV") if "L3|K" in pr else "-"
    return (f"acc {r['acc']:.3f}  trans {str(r['transition']):>5}  VALcos {r['val_cos']:.3f}  margin "
            f"{c16.fmt2(e.get('margin'))}  {r.get('tag')}  probe L3 K/V {l3}")


def report():
    import explore_f_reports4 as fr
    return fr.report_s80()


if __name__ == "__main__":
    import explore_f_configs as me
    import explore_f_configs_b as me2
    fc.drive("s80", [me, me2], sys.argv[1:])
