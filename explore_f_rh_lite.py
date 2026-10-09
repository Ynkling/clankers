#!/usr/bin/env python
"""
explore_f_rh_lite.py — EXPLORATORY, not a result. Session F follow-up, screen S70: RandHeaderTask-lite, replacing S54(d)
(UNTESTED: neither perfect gate bound the P = 8 RandHeaderTask in 24000). Its Hebbian counts are the BASELINE for
Session G.

TASK: explore_f_tasks.RandHeaderTask(P=4, nb=2, lmin=1, lmax=3), S = 2, n_vals = 16, n_q = 1: each stream's 4 keys in 2
blocks; the first block's length is uniform on {1, 2, 3} and the remainder (3, 2, 1) is the second block (the
compositions (1,3), (2,2), (3,1), uniform); the 4 blocks in a uniformly random order; one CTX token at the start of each
block; every (stream, key) once, S distinct values per key; L = 2·(2 + 8) + 3 = 23, qpos 21. k = 2, no conv.
PHASE 1, ORACLE FIRST (seeds 330-332, Adam 1e-3, 24000): S70_ORACLE_HEBB (the perfect gate, Hebbian) and S70_ORACLE_DELTA
  (the perfect gate, delta β = 1, L2 keys). An oracle arm BINDS if >= 2 of its 3 seeds bind. If either does not bind,
  both are run once more at double the budget (S70_ORACLE_HEBB_2X, S70_ORACLE_DELTA_2X, 48000, the same seeds); if either
  still does not bind, S70 is UNTESTED and stops. The budget of phase 2 is the one at which both bound (24000 or 48000).
  The decision is written to explore_out/F/rh_lite_budget.json by decide() before phase 2 is queued.
PHASE 2 (seeds 330-349): S70_HEBB, S70_DELTA — LOCAL3 + SLOW (S43's recipe; gate W_in, W_g, window 1e-3 throughout; the
  rest 1e-4 for updates 1-2400, 1e-3 after; no hinge) on the Hebbian memory, and on the delta memory with β = 1 FIXED, L2
  keys, tied write, decay 0.95. Paired (same initial parameters and batches).
OUTCOMES: BOUND; DISCOVERED (bound and final VAL cos < 0.5); ROUTED*@end (margin >= 0.9 and η² by stream at KEY > 0.9);
failure classes (test_router_layout.fail_class: STREAM-PARTIAL / KEY / POSITION / OTHER) on
explore_f_tasks.rand_header_routing_stats (η² at KEY and VAL positions, positions read per sequence; margin from the
write gate at VAL positions). Wilson 95% and bands with every count; one-sided exact McNemar DELTA vs HEBB both ways.
READING (descriptive, fixed now): the BASELINE for Session G is S70_HEBB's DISCOVERED count (BOUND and ROUTED*@end
alongside) at the phase-2 budget; S70_DELTA printed beside it. No other reading.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore_common16 as c16
import explore_f_common as fc

NAME = fc.name("rh_lite")
CHILD = "explore_f_followup_child"
MAIN = False
LR = 1e-3
OSEEDS, SEEDS = (330, 331, 332), tuple(range(330, 350))
DECISION = os.path.join(fc.F_DIR, "rh_lite_budget.json")
_slow = "gate W_in/W_g/window 1e-3 throughout; rest 1e-4 for 1-2400, 1e-3 after"


def decision():
    return json.load(open(DECISION)) if os.path.exists(DECISION) else {}


def _arms():
    d = decision()
    arms = {f"S70_ORACLE_{m}": dict(opt="adam", lr=LR, iters=24000, seeds=OSEEDS, label=f"perfect gate, {m}",
                                    sched="Adam 1e-3 throughout") for m in ("HEBB", "DELTA")}
    if d.get("phase1") == "double" or d.get("budget") == 48000 or d.get("budget") == "UNTESTED":
        for m in ("HEBB", "DELTA"):
            arms[f"S70_ORACLE_{m}_2X"] = dict(opt="adam", lr=LR, iters=48000, seeds=OSEEDS, label=f"perfect gate, {m}, 2x",
                                              sched="Adam 1e-3 throughout")
    if isinstance(d.get("budget"), int):
        for m in ("HEBB", "DELTA"):
            arms[f"S70_{m}"] = dict(opt="adam", lr=LR, iters=d["budget"], seeds=SEEDS,
                                    label=f"LOCAL3 + SLOW on {m}{' (beta 1)' if m == 'DELTA' else ''}", sched=_slow)
    return arms


ARMS = _arms()


def decide():
    """Apply the phase-1 rule to the stored oracle runs and write the decision."""
    import explore_f_common as fc_
    fc_.setup([sys.modules[__name__]])
    b = lambda arm: sum(1 for r in fc_.runs(sys.modules[__name__], arm).values() if r["transition"] is not None)
    n = lambda arm: len(fc_.runs(sys.modules[__name__], arm))
    d = decision()
    if all(n(f"S70_ORACLE_{m}") == 3 for m in ("HEBB", "DELTA")) and "phase1" not in d:
        ok = all(b(f"S70_ORACLE_{m}") >= 2 for m in ("HEBB", "DELTA"))
        d = dict(phase1="bound" if ok else "double", oracle_24k={m: b(f"S70_ORACLE_{m}") for m in ("HEBB", "DELTA")})
        if ok:
            d["budget"] = 24000
    if d.get("phase1") == "double" and "budget" not in d and all(n(f"S70_ORACLE_{m}_2X") == 3 for m in ("HEBB", "DELTA")):
        ok = all(b(f"S70_ORACLE_{m}_2X") >= 2 for m in ("HEBB", "DELTA"))
        d["oracle_48k"] = {m: b(f"S70_ORACLE_{m}_2X") for m in ("HEBB", "DELTA")}
        d["budget"] = 48000 if ok else "UNTESTED"
    json.dump(d, open(DECISION, "w"), indent=1)
    print(f"  S70 decision: {d}", flush=True)
    return d


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ARMS[arm]["iters"]))


def timing(arm, steps):
    return child("timing", dict(arm=arm, steps=steps))


def line(arm, r):
    if not r.get("ok"):
        return f"FAILED {r.get('error')}"
    return f"acc {r['acc']:.3f}  trans {str(r['transition']):>5}  {r.get('tag') or ('BOUND' if r['transition'] is not None else 'unbound')}"


def report():
    import explore_f_reports2 as fr
    return fr.report_s70()


if __name__ == "__main__":
    decide()
