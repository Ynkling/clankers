#!/usr/bin/env python
"""
explore_f_write.py — EXPLORATORY, not a result. Session F, screen S88a: the write strength. The delta recipe keeps 2, 4
and 8 streams with β fixed at 1 (S76, S80); Gated DeltaNet learns β per token. Does a learned write strength keep the
eight-stream partition?

CPU: this container is an Intel Xeon @ 2.80GHz; Session F's earlier screens ran on a Xeon @ 2.10GHz. Every run of
this screen is on this CPU, so pairs are bit for bit within one CPU. The 2.10GHz records reproduce bit for bit here
(checked on 12 October: S70_DELTA|330 and LOCAL3_SLOW|160 through 2400), so the reproduction CHECKs are asserted.

SEEDS: 2400-2409 (the assigned block 2400-2459 appears in no record key "arm|seed|", no "seed" field and no "seed N" log
line in explore_out/, results/ or records/ on any branch; the strings "2400"... elsewhere are step counts). Perfect gates
on 2400-2401.

PATH (explore_f_write_child; main-line modules at 9c5939e): S76's DELTA_SPLIT exactly — S = 8, P = 4, k = 16, conv, all 8
streams from step 1 (test_stream_curriculum.run_sc with D8), 43200 updates, evaluation every 1200 on 2048 held-out
sequences, early stop after 3 evaluations >= 0.95; LOCAL3 + SLOW (gate W_in, W_g, window at 1e-3 throughout; every other
trainable parameter at 1e-4 for updates 1-2400, 1e-3 after; no hinge); S76's KEYMASS trigger (checks every 2400 from
4800 on the 64-sequence probe; fire if accuracy < 0.95 and rose < 0.02; <= 3 splits >= 4800 apart) with H's exact-copy
split. The delta memory: L2 keys, tied write, decay 0.95.
ARMS (paired: on a seed every arm starts from the same parameters — the learned arms add w = 0, b = 0 — and sees the same
batches):
  DELTA_FIXED         β = 1 (S76's recipe, DELTA*).
  DELTA_LEARNED       β_t = 2·sigmoid(w·v_t + b) per token, w = 0 and b = 0 at init; w and b in the gate's group (1e-3
                      throughout).
  DELTA_LEARNED_SLOW  the same, w and b in the slow group (1e-4 for updates 1-2400, 1e-3 after).
  O_DELTA (validity) and O_DELTA_LEARNED (printed): the perfect gate (ceil8_D8), one Adam at 1e-3, β = 1 / β learned,
                      seeds 2400-2401.
  v_t is the layer's write vector V (its input; the token's hidden state, as Gated DeltaNet's x_t); w (32-d) and b are
  shared by the 3 layers, as every BDH weight is.
  THE PARAMETRISATION, a departure from the specification's "β_t = sigmoid(w·x_t + b), b initialised so that β starts at
  the fixed value": the fixed value is 1, which a sigmoid reaches only in saturation. In float32, sigmoid(b) = 1.0 exactly
  from b = 17, and there its gradient is exactly 0 (torch: sigmoid(17) = 1.0, gradient 0.0), so β could not learn. β = 2·sigmoid(·) with b = 0 starts at exactly 1.0 with a
  live gradient. Its range is (0, 2), the delta rule's stable range with unit keys (I − β k kᵀ has eigenvalue 1 − β); the
  values above 1 (overshoot erase) are outside Gated DeltaNet's (0, 1), so the fraction of positions with β > 1 is printed.
  With w and b frozen it is DELTA_FIXED bit for bit (CHECK).
OUTCOME: BOUND ROUTED (test_stream_recipe.outcome == "BOUND ROUTED": bound, one-to-one, every stream >= 0.9) with the
  transition. Unbound runs: the outcome under v1 and v2 (fail_class_v2). Wilson 95% intervals and bands with every count;
  exact one-sided McNemar both ways, each learned arm against DELTA_FIXED.
VALIDITY (fixed before any run): O_DELTA binds on both 2400 and 2401 within 43200; otherwise the readings are UNTESTED.
READINGS (fixed before any run), on BOUND ROUTED over the 10 paired seeds; for a learned arm X, b = seeds where X alone
  succeeds, c = seeds where DELTA_FIXED alone succeeds (so c − b = DELTA_FIXED's count − X's count):
    "a learned write strength keeps the partition"  if the better learned arm (more BOUND ROUTED) is within 1 pair of
                                                    DELTA_FIXED: c − b <= 1 for it;
    "a learned write strength costs it"             if DELTA_FIXED beats both learned arms by >= 4 vs 0 (c >= 4, b = 0 for
                                                    each);
    otherwise "no reading applies".
PRINTED (learned arms, every evaluation, on the run's probe batch, per layer): β's median over all positions (the
  trajectory), its medians at CTX / KEY / VAL, the per-stream medians at KEY positions, η² of β by stream at KEY and at VAL
  positions, the fraction with β > 1, min and max. "β DIVERGES BY STREAM" (descriptive, fixed now) in a run if at the end
  η² of β by stream at KEY positions is >= 0.5 (the harness's η² threshold) in any layer; the per-stream spread (max − min
  of the 8 medians) is printed beside it. Also every split with its target label and the median transitions.
CHECKS (asserted before any run): explore_f_delta_checks (26) at the start of every segment; then (child, in parallel):
  - DELTA_FIXED on seed 1315 reproduces S76's DELTA_SPLIT|1315 (explore_out/F/eight_results.json, recorded on the 2.10GHz
    Xeon) bit for bit
    through 7200 (its first split at 4800 + one check): curve, statistics and trigger rows.
  - DELTA_LEARNED and DELTA_LEARNED_SLOW with w and b frozen at their initial values equal DELTA_FIXED bit for bit through
    the same 7200 (so through 2400 as specified, and through the first split), with β = 1.0 exactly at every probe
    position and evaluation (the β probe runs, so it is shown not to disturb training).
  - The learned model is DELTA_FIXED's plus w, b (zeros), with equal logits at init; w and b in the gate group (LEARNED)
    or the rest group (LEARNED_SLOW) at the stated lrs; every trainable parameter in exactly one group; one Adam step
    moves w and b (a live gradient); explore_f_delta_fast's checks (β per token included); the perfect gate one-hot on both
    memories with equal logits at init; DELTA* as committed.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore_common16 as c16
import explore_f_common as fc

NAME = fc.name("write")
CHILD = "explore_f_write_child"
MAIN = True
LR, ITERS = 1e-3, 43200
SEEDS, OSEEDS = tuple(range(2400, 2410)), (2400, 2401)
ETA = 0.5
_sl = "gate 1e-3 throughout; rest 1e-4 for 1-2400, 1e-3 after; KEYMASS trigger, exact row copy"
ARMS = {
    "DELTA_FIXED": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="beta 1 (S76's DELTA_SPLIT)", sched=_sl),
    "DELTA_LEARNED": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="beta = 2 sigmoid(w.v + b), w, b in the gate group",
                          sched=_sl),
    "DELTA_LEARNED_SLOW": dict(opt="adam", lr=LR, iters=ITERS, seeds=SEEDS, label="the same, w, b in the slow group",
                               sched=_sl),
    "O_DELTA": dict(opt="adam", lr=LR, iters=ITERS, seeds=OSEEDS, label="perfect gate, beta 1", sched="Adam 1e-3"),
    "O_DELTA_LEARNED": dict(opt="adam", lr=LR, iters=ITERS, seeds=OSEEDS, label="perfect gate, beta learned", sched="Adam 1e-3"),
}
PARTS = ("s76", "frozen", "frozen_slow", "mech")


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ARMS[arm]["iters"]))


def timing(arm, steps):
    return child("timing", dict(arm=arm, steps=steps))


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
    bt = r.get("beta_traj") or {}
    last = bt[max(bt, key=int)] if bt else None
    bs = " beta med " + "/".join(f"{L['median']:.2f}" for L in last) if last else ""
    return (f"acc {r['acc']:.3f}  trans {str(r['transition']):>5}  {r.get('outcome')}  splits "
            f"{[(c['step'], 'ok' if c.get('key_ok') else 'NOT') for c in fired]}  map {c16.chmap_str(r['end'].get('ch_map'))}{bs}")


def report():
    import explore_f_reports5 as fr
    return fr.report_s88a()


if __name__ == "__main__":
    import explore_f_write as me
    fc.drive("s88a", [me], sys.argv[1:])
