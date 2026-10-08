#!/usr/bin/env python
"""
explore_f_window_delta.py — EXPLORATORY, not a result. Session F, screen S54 (a) and (d): the current best two-stream
recipe (the window gate LOCAL3 + slow memory, no hinge: batch 16's S43 LOCAL3_SLOW) on delta-rule channels, and the
randomised header layout (RandHeaderTask), whose counts are the BASELINE Session G must beat. (b) and (c) are in
explore_f_window_delta_scale (main-line modules); one batch runs both.

CONFIGURATION (explore_f_window_delta_child; this branch's modules, 1 torch thread). LOCAL3 + SLOW = S43's LOCAL3_SLOW
exactly (LocalGateBDH, W_h frozen; Adam, gate group W_in, W_g, window at 1e-3 throughout; every other trainable parameter
— the memory's attn.w_b, attn.b_b included — at 1e-4 for updates 1-2400 and 1e-3 after; no hinge), evaluation every
1200 on 2048 held-out sequences, early stop after 3 evaluations >= 0.95. DELTA = explore_delta_mem.to_delta(beta=
"learned"): β_t = σ(w_b·v_t + b_b) (0.5 at init), L2 keys, tied write, decay 0.95. HEBB = the current memory. The
perfect gate (validity) uses its own recipe: one Adam at 1e-3 on every parameter.

ARMS
 (a) grouped S=2, P=4, k=2, no conv (test_short_conv.TASK), 24000 updates:
     A_DELTA          LOCAL3 + SLOW on DELTA, seeds 310-349.
     A_ORACLE_DELTA, A_ORACLE_HEBB   perfect gate, seeds 310-311 (validity).
     Reference (recorded, counts only, another seed block): batch 16's S43 LOCAL3_SLOW on the Hebbian memory, seeds
     160-199: DISCOVERED 31/40 (failures STREAM-PARTIAL 5, KEY 4).
 (d) RandHeaderTask (explore_f_tasks: S=2, P=8 keys per stream, NB=2 blocks per stream, block length uniform on {2..6},
     4 blocks in random order, one CTX per block; L = 39), k=2, no conv, 24000 updates:
     D_DELTA, D_HEBB  LOCAL3 + SLOW on DELTA and on HEBB, seeds 330-349 (paired: same initial parameters except the delta
                      memory's w_b = 0, b_b = 0, same batches).
     D_ORACLE_DELTA, D_ORACLE_HEBB   perfect gate, seeds 330-331 (validity).

OUTCOMES (as the harness defines them): BOUND = a transition in the held-out curve; DISCOVERED = BOUND and final VAL cos
< 0.5 (the read gates of the two streams at VAL positions differ); ROUTED*@end = margin >= 0.9 and η² of the read gate
at KEY positions by stream > 0.9; failure classes (unbound runs): test_router_layout.fail_class on the end statistics
(STREAM-PARTIAL margin >= 0.25; else KEY η²_key_by_key >= 0.5; else POSITION max(η²_key_by_half, by_index) >= 0.5;
else OTHER). On RandHeaderTask the routing statistics are explore_f_tasks.rand_header_routing_stats (positions read per
sequence; η² at KEY and at VAL positions; margin from the write gate at VAL positions). Wilson 95% intervals and bands
(RELIABLE >= ceil(0.9n), MAJORITY >= ceil(0.5n), MINORITY >= 1, NEVER 0) with every count.

VALIDITY (fixed before any run): a memory's arm is VALID if its perfect-gate arm binds on at least 1 of its 2 seeds within
24000 updates; otherwise its reading is UNTESTED.

READINGS (fixed before any run)
 (a) on DISCOVERED, A_DELTA (40 seeds) against the recorded Hebbian LOCAL3_SLOW 31/40 (unpaired, another CPU-independent
     seed block; counts only, one-sided Fisher printed): "delta not worse" if A_DELTA >= 29/40 (31 - 2: the bound);
     "delta better" if A_DELTA >= 35/40 (31 + 4); "delta worse" if A_DELTA <= 28/40. (Both "not worse" and "better"
     hold at >= 35.)
 (d) descriptive: per memory, BOUND, DISCOVERED and ROUTED*@end counts, failure classes (expected: KEY splits),
     transitions; paired D_DELTA vs D_HEBB (one-sided McNemar both ways, printed). The BASELINE for Session G is
     D_HEBB's and D_DELTA's DISCOVERED counts (with BOUND and ROUTED*@end alongside).
DIAGNOSTICS: learned β at CTX / KEY / VAL per layer at the end; (a) S38's decoder of the gate state at KEY and VAL
positions at 1200 / 2400 / 4800 / end; write-order accuracy (explore_f_tasks.write_order_acc).
CHECKS (child.checks + explore_f_delta_checks every segment): RandHeaderTask's own CHECKs; A_DELTA = S43's model +
to_delta with no other change; D_HEBB / D_DELTA share their initial parameters; the window gate on RandHeaderTask
ignores t-3 and sees t-2; the groups cover every trainable parameter once with the memory's parameters in the rest group
and the lrs [gate, rest] at updates 1, 2400, 2401 = [1e-3, 1e-4], [1e-3, 1e-4], [1e-3, 1e-3]; the perfect gate on
RandHeaderTask has margin 1 and η² by stream 1 at KEY and VAL.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common16 as c16
import explore_f_common as fc

NAME = fc.name("window_delta")
CHILD = "explore_f_window_delta_child"
MAIN = False
LR = 1e-3
ITERS = 24000
A_SEEDS, D_SEEDS = tuple(range(310, 350)), tuple(range(330, 350))
REC_LOCAL3_SLOW = 31                       # batch 16's S43 LOCAL3_SLOW DISCOVERED on 160-199 (Hebbian), of 40
NOT_WORSE, BETTER, WORSE_MAX = 29, 35, 28
_slow = "gate W_in/W_g/window 1e-3 throughout; rest (incl. attn.w_b, attn.b_b) 1e-4 for 1-2400, 1e-3 after"
ARMS = {
    "A_DELTA": dict(opt="adam", lr=LR, iters=ITERS, seeds=A_SEEDS, label="(a) LOCAL3 + SLOW on DELTA, grouped S=2", sched=_slow),
    "A_ORACLE_DELTA": dict(opt="adam", lr=LR, iters=ITERS, seeds=(310, 311), label="(a) perfect gate, DELTA",
                           sched="Adam 1e-3 throughout"),
    "A_ORACLE_HEBB": dict(opt="adam", lr=LR, iters=ITERS, seeds=(310, 311), label="(a) perfect gate, HEBB",
                          sched="Adam 1e-3 throughout"),
    "D_DELTA": dict(opt="adam", lr=LR, iters=ITERS, seeds=D_SEEDS, label="(d) LOCAL3 + SLOW on DELTA, RandHeaderTask", sched=_slow),
    "D_HEBB": dict(opt="adam", lr=LR, iters=ITERS, seeds=D_SEEDS, label="(d) LOCAL3 + SLOW on HEBB, RandHeaderTask", sched=_slow),
    "D_ORACLE_DELTA": dict(opt="adam", lr=LR, iters=ITERS, seeds=(330, 331), label="(d) perfect gate, DELTA",
                           sched="Adam 1e-3 throughout"),
    "D_ORACLE_HEBB": dict(opt="adam", lr=LR, iters=ITERS, seeds=(330, 331), label="(d) perfect gate, HEBB",
                          sched="Adam 1e-3 throughout"),
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
    return (f"acc {r['acc']:.3f}  trans {str(r['transition']):>5}  VALcos {r['val_cos']:.3f}  margin "
            f"{fc.c16.fmt2(e.get('margin'))}  {r.get('tag') or ('BOUND' if r['transition'] is not None else 'unbound')}")


def report():
    import explore_f_reports as fr
    return fr.report_s54ad()


if __name__ == "__main__":
    import explore_f_window_delta as me
    import explore_f_window_delta_scale as me2
    fc.drive("s54", [me, me2])
