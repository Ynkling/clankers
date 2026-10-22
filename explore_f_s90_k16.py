#!/usr/bin/env python
"""
explore_f_s90_k16.py — EXPLORATORY, not a result. Session F, after S90-F (22 October, evening): the type-level perfect gate at
k = 16, the ceiling G's S91b reads against (G runs S91b at k = 16 on -ctxpool and -allpool because of report 10's cap
table: a fixed type -> channel map separates every sequence at k = 16). Four runs; nothing else.

ARMS (child explore_f_s90_k16_child; seed 3060; SLOW as every pilot of this round; evaluation every 1200 on 2048 held-out
sequences, bound = the first evaluation >= 0.95 that holds to the end, early stop after 3):
  TL16_HEBBCONV_CTXPOOL   Hebbian + conv 'layer', -ctxpool, 24000 (the memory's valid budget there, S90-F)
  TL16_DELTA_CTXPOOL      delta (beta 1, L2 keys), -ctxpool, 24000 (S90-F)
  TL16_HEBBCONV_ALLPOOL   Hebbian + conv 'layer', -allpool, 48000 ("needs the longer budget", report 9 addendum)
  TL16_DELTA_ALLPOOL      delta, -allpool, 24000 (valid at 24000, S89-F)
  Each: k = 16 channels, the type-level perfect gate with one channel per CTX type (explore_f_tasks.type_level_gate,
  identity map), in place of the per-sequence index at k = 2.
STATED BEFORE THE RUNS (CHECKed, asserted): a sequence's contexts are distinct types, so this gate's same-channel matrix
  equals the per-sequence perfect gate's exactly; the perfect-gate model's parameters do not depend on k; at
  initialisation the logits equal the k = 2 pilot's bit for bit, on both tasks and both memories; and through 2400 on
  -ctxpool seed 3060 the k = 16 run gives the k = 2 pilot's curve bit for bit (both memories). So at k = 16 the
  type-level ceiling is the per-sequence ceiling: the expectation is that TL16_*_CTXPOOL|3060 repeats O_*_CTXPOOL|3060
  (transitions 2400 Hebbian + conv, 1200 delta). On -allpool no k = 2 pilot ran on 3060 (S89's ran on 2800-2801).
READING (fixed now): "the type-level gate binds at k = 16" if 4/4 bind. Otherwise no reading; the count is printed.
  Printed beside it: the per-sequence gate's 2/2 (-ctxpool: S90-F O_HEBBCONV / O_DELTA on 3060-3061; -allpool: S89-post
  O_HEBB_CONV_ALLPOOL_48K and S89-F O_DELTA_ALLPOOL on 2800-2801), and every transition (G needs them).
  A failed run is retried by the harness (3 attempts); still failed, it counts as not bound.
STATISTICS: margin and the within-sequence sep_key_inv / seq_separated_frac / val_cos_inv_k; the channel-0 and pooled
  fields are NaN at k = 16 (child docstring), so no DISCOVERED is computed.
CHECKS (asserted before the runs): explore_f_delta_checks (26); the child's: the gate, the same-channel matrix, the
  parameters and the initial logits against the k = 2 pilot (4 arms); groups and lrs (SLOW; the conv in the rest group);
  the tie: S90's path reproduces O_DELTA_CTXPOOL|3060 and O_HEBBCONV_CTXPOOL|3060 (2.10GHz records, read at d9783bb)
  through 2400 on this CPU, and the k = 16 path gives the same curves and k-invariant statistics.
CPU: whichever this container is (a Xeon @ 2.80GHz on the evening of 22 October; S90-F ran on the 2.10GHz one); the tie above runs on
  it before the runs.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore_common16 as c16
import explore_f_common as fc

NAME = fc.name("s90_k16")
CHILD = "explore_f_s90_k16_child"
MAIN = False
LR = 1e-3
SEEDS = (3060,)
SLOW = "SLOW (rest 1e-4 to 2400)"
BUDGET = {"TL16_HEBBCONV_CTXPOOL": 24000, "TL16_DELTA_CTXPOOL": 24000, "TL16_HEBBCONV_ALLPOOL": 48000,
          "TL16_DELTA_ALLPOOL": 24000}
ARMS = {a: dict(opt="adam", lr=LR, iters=b, seeds=SEEDS, sched=SLOW, label=f"type-level perfect gate, k = 16, {a.lower()}")
        for a, b in BUDGET.items()}
# the per-sequence gate (k = 2) beside it: (screen store, arm, seeds, code SHA, ref)
PER_SEQ = {"TL16_HEBBCONV_CTXPOOL": ("F/s90_ctx", "O_HEBBCONV_CTXPOOL", (3060, 3061), "809fde1f5168"),
           "TL16_DELTA_CTXPOOL": ("F/s90_ctx", "O_DELTA_CTXPOOL", (3060, 3061), "809fde1f5168"),
           "TL16_HEBBCONV_ALLPOOL": ("F/s89_post", "O_HEBB_CONV_ALLPOOL_48K", (2800, 2801), "267d08456c38"),
           "TL16_DELTA_ALLPOOL": ("F/s89_tasks", "O_DELTA_ALLPOOL", (2800, 2801), "ea9728c636c6")}


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
    e = r.get("end") or {}
    return (f"acc {r['acc']:.3f}  trans {str(r['transition']):>5}  margin {c16.fmt2(e.get('margin'))} "
            f"sep {c16.fmt2(e.get('seq_separated_frac'))}")


def reading(n_bound, n_failed, n):
    if n + n_failed < len(ARMS):
        return "INCOMPLETE, no reading"
    return "the type-level gate binds at k = 16" if n_bound == 4 else f"no reading (the type-level gate binds {n_bound}/4 at k = 16)"


def report():
    from explore_f_reports7 import report_s90_k16
    return report_s90_k16()


if __name__ == "__main__":
    import explore_f_s90_k16 as me
    fc.drive("s90k16", [me], sys.argv[1:])
