#!/usr/bin/env python
"""
explore_batch1.py — EXPLORATORY, not a result. Runs the first batch of outside-idea screens in one
worker pool and prints each screen's report and verdict.

  S1 explore_local_gate  Mamba-style gate from a width-3 causal conv, no recurrence   (LOCAL3)
  S2 explore_kwta        FlyHash / k-WTA: top 16 of 256 units of x_sparse            (KWTA16, KWTA16_ceil)
  S3 explore_aux_gate    UNREAL/CPC-style next-token aux loss on the gate's state    (AUX1)

Recipe, seeds and pairing: explore_common's docstring (test_short_conv's P=4 recipe, lr =
test_channel_binding.SUB_LR passed explicitly, seeds 160-179 paired with X's recorded arm A).

SCREEN RULE (fixed before any run). Outcome DISCOVERED, paired by seed with X's recorded arm A;
b = candidate-only seeds, c = A-only seeds, p = exact two-sided McNemar:
  promising     b - c >= 4 and p < 0.10
  not           b - c <= 0
  inconclusive  otherwise
S2 needs its validity arm: KWTA16_ceil must bind >= 2 of 3, else S2 is "inconclusive (invalid)".
A screen is never a result; a promising screen is handed back for a pre-registered test.

Run:  python explore_batch1.py [--workers 4] [--only S1,S2,S3] [--skip-repro]
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_local_gate as s1
import explore_kwta as s2
import explore_aux_gate as s3

SCREENS = {"S1": s1, "S2": s2, "S3": s3}
PROMISING_D, PROMISING_P = 4, 0.10


def verdict(d, valid=True):
    if not valid:
        return "inconclusive (invalid: the validity arm did not bind)"
    diff = d["b"] - d["c"]
    if diff >= PROMISING_D and d["p"] < PROMISING_P:
        return "promising"
    if diff <= 0:
        return "not"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=ec.WORKERS)
    ap.add_argument("--only", default="S1,S2,S3")
    ap.add_argument("--skip-repro", action="store_true")
    args = ap.parse_args()
    screens = [SCREENS[k] for k in args.only.split(",")]
    pv = ec.print_banner("batch 1: " + ", ".join(s.NAME for s in screens))
    print(f"  recipe: lr = test_channel_binding.SUB_LR = {ec.SUB_LR:g} (passed explicitly), "
          f"MAX_ITERS = test_short_conv.MAX_ITERS = {ec.MAX_ITERS}, task P={ec.TASK.P} "
          f"S={ec.TASK.S} n_vals={ec.TASK.n_vals} n_q={ec.TASK.n_q}, seeds {ec.fmt_seeds(ec.SEEDS)}")
    print(f"  screen rule: promising if b - c >= {PROMISING_D} and McNemar p < {PROMISING_P}; "
          f"not if b - c <= 0; else inconclusive (outcome DISCOVERED, vs X's recorded arm A)")
    repro = True if args.skip_repro else ec.repro_check()
    if not repro:
        print("  ! this container does not reproduce X: the recorded pairing is invalid; stopping.")
        sys.exit(1)
    for s in screens:
        assert s.check(), f"{s.NAME} CHECK failed"
    for s in screens:
        ec.print_screen_header(s)
    t0 = time.time()
    ec.run_jobs(screens, workers=args.workers, pv=pv)
    print(f"  batch wall clock {(time.time() - t0) / 60:.1f} min")
    print("#" * 100)
    print(f"REPORTS — {ec.BANNER}")
    print("#" * 100)
    for s in screens:
        store = ec.load_store(s.NAME)
        d = s.report(store)
        print(f"  VERDICT {s.NAME}: {verdict(d, d.get('valid', True))}  [{ec.BANNER}]")
        print()


if __name__ == "__main__":
    main()
