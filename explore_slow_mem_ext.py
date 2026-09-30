#!/usr/bin/env python
"""
explore_slow_mem_ext.py — EXPLORATORY, not a result. Screen S7 (batch 3): S5's SLOW_MEM arm,
unchanged, on 20 new seeds.

BACKGROUND (batch 2). S5 slow_mem, seeds 160-179: DISCOVERED 12/20 vs X's arm A 7/20 (6 vs 1,
p = 0.125), inconclusive. ROUTED* at 1200: 12 vs 6 (7 vs 1, p = 0.070); all 12 routed runs bound.
Failures: 7 POSITION, 1 OTHER, no KEY (arm A: KEY 4).

THE ARM. explore_slow_mem's SLOW_MEM exactly (its run_path, builder, param groups and lr switch;
that file is not changed): arm A with one Adam and two groups, the gate (W_in, W_h, W_g) at lr
1e-3 = test_channel_binding.SUB_LR throughout, every other parameter at 1e-4 for updates 1-2400
and 1e-3 after. MAX_ITERS 24000. Seeds 180-199, paired with X's recorded arm A (test_short_conv
ran seeds 160-199; on 180-199 X's arm A DISCOVERED 5/20, ROUTED* at 1200 5/20).

RULE (batch 1's, on the 20 new seeds): DISCOVERED vs X's arm A, b = SLOW_MEM-only, c = A-only,
exact two-sided McNemar p: promising if b - c >= 4 and p < 0.10; not if b - c <= 0; otherwise
inconclusive. Also printed: the 40-seed pool 160-199 (batch 2's S5 records + these), DISCOVERED
and ROUTED* at 1200 paired with X's arm A, and the failure classes.

CHECKS: explore_slow_mem.check() (the lr schedule by an optimizer step pre-hook over the full
24000-step run path; the gate group; arm A's parameters), since S7 runs that path unchanged.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common2 as c2
import explore_slow_mem as s5

NAME = "slow_mem_ext"
IDEA = "S5's slow-memory schedule, unchanged, on 20 new seeds"
SOURCE = "batch 2's S5 slow_mem (two-timescale training; the race reading)"
CHANGE = s5.CHANGE
PAIRING = ("seeds 180-199, paired with X's recorded arm A (test_short_conv, lr 1e-3; same initial "
           "parameters and batches); pool 160-199 with batch 2's S5 records")
SEEDS = tuple(range(180, 200))
POOL_SEEDS = tuple(range(160, 200))
S5_STORE = "slow_mem"

ARMS = {
    "SLOW_MEM": dict(s5.ARMS["SLOW_MEM"], seeds=SEEDS, prio=1,
                     label="S5's SLOW_MEM, unchanged; seeds 180-199"),
}


def run_job(arm, seed):
    a = ARMS[arm]
    return s5.run_path(a, seed, a["iters"])


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], ec.TASK, s5.builder(a, 0), a["lr"], c2.stats_b2)]


def check():
    print("  (S7 runs explore_slow_mem's run path unchanged; its CHECKs:)")
    return s5.check()


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    c2.per_seed_table2(me, store)
    s = ARMS["SLOW_MEM"]["seeds"]
    print("  paired with X's arm A on the new seeds (same seeds, initial parameters and batches):")
    d = ec.paired_vs_recorded(store, "SLOW_MEM", s, ec.discovered, label="DISCOVERED")
    ec.paired_vs_recorded(store, "SLOW_MEM", s, ec.bound, label="BOUND")
    for step in (1200, "end"):
        c2.paired_routed(store, "SLOW_MEM", s, step, {})
    print(f"    failures SLOW_MEM {ec.fail_counts(store, 'SLOW_MEM', s)}   "
          f"X arm A {ec.recorded_fail_counts('A', s)}")
    # the 40-seed pool: batch 2's S5 records (160-179) and these (180-199)
    s5_store = ec.load_store(S5_STORE)
    pool = {"runs": {**{k: v for k, v in s5_store["runs"].items() if k.startswith("SLOW_MEM|")},
                     **{k: v for k, v in store["runs"].items() if k.startswith("SLOW_MEM|")}}}
    n_s5 = sum(1 for k in s5_store["runs"] if k.startswith("SLOW_MEM|"))
    print(f"  POOL, seeds 160-199 (batch 2's S5: {n_s5} runs from {S5_STORE}_results.json; S7: "
          f"{sum(1 for k in store['runs'] if k.startswith('SLOW_MEM|'))} runs):")
    pd = ec.paired_vs_recorded(pool, "SLOW_MEM", POOL_SEEDS, ec.discovered, label="DISCOVERED")
    pr = c2.paired_routed(pool, "SLOW_MEM", POOL_SEEDS, 1200, {})
    print(f"    failures SLOW_MEM {ec.fail_counts(pool, 'SLOW_MEM', POOL_SEEDS)}   "
          f"X arm A {ec.recorded_fail_counts('A', POOL_SEEDS)}")
    d["pool"] = dict(discovered=pd, routed1200=pr)
    return d
