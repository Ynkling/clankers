#!/usr/bin/env python
"""
explore_h_onetoone_counts.py — EXPLORATORY, not a result. Session H, round 5 (2): over the main line's recorded split runs
at eight streams, the firings whose stream -> channel map before the fire was already one-to-one, and whether those runs
were BOUND ROUTED at the end. No runs.

RUNS: WIN3_SPLIT_D8 in results/X and results/L window_gate_results.json and in split_copy_results.json (split_copy's copy arm
is named WIN3_SPLIT_D8 there too: the copy + noise + Adam reset; its other arm with checks, WIN3_RESET_D8, copies nothing
and is excluded). Read with git show from claude/bdh-growth-hebbian-inference-w90069.
  - "the 50 runs": the main line at 0282955 (window_gate X 20, L 20; split_copy X 10), the set S86's count used;
  - "now": the main line's current head, which also holds L's split_copy file (10 more runs).
FIELD: checks[].map_before — routing_k's stream -> channel map (argmax of the mean read gate at VAL positions per stream) on
the run's routing probe, computed by the trigger at that check before the operation; the labels key_ok / all_ok use the
same map. One-to-one = 8 distinct channels for the 8 streams. BOUND ROUTED at the end = test_stream_recipe.outcome, read
from the record's stored outcome field when present, else recomputed here from its "end" statistics and transition with
the same rule (bound, the end map one-to-one, every stream's held-out accuracy >= 0.9).
FIRST: S86's count (key pair labelled at 107, all-position pair at 97 of 107 firings) is reproduced at 0282955.

Run:  python explore_h_onetoone_counts.py
"""

import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
PIN = "0282955"
FILES = (("window_gate X", "results/X/window_gate_results.json"), ("window_gate L", "results/L/window_gate_results.json"),
         ("split_copy X", "results/X/split_copy_results.json"), ("split_copy L", "results/L/split_copy_results.json"))
ARM = "WIN3_SPLIT_D8"
S = 8


def load(ref, path):
    r = subprocess.run(["git", "-C", HERE, "show", f"{ref}:{path}"], capture_output=True, text=True)
    return json.loads(r.stdout)["runs"] if r.returncode == 0 else None


def bound_routed(r):
    if r.get("outcome"):
        return r["outcome"] == "BOUND ROUTED"
    e = r["end"]
    return bool(r.get("transition") is not None and e.get("one_to_one") and min(e["stream_acc"]) >= 0.9)


def count(ref):
    rows = []
    for lab, path in FILES:
        runs = load(ref, path)
        if runs is None:
            continue
        runs = {k: r for k, r in runs.items() if k.split("|")[0] == ARM and r.get("checks")}
        fires = [(k, c) for k, r in runs.items() for c in r["checks"] if c.get("fired")]
        one = [(k, c) for k, c in fires if len(set(c["map_before"])) == S]
        rows.append(dict(file=lab, runs=len(runs), firings=len(fires), key_ok=sum(bool(c["key_ok"]) for _, c in fires),
                         all_ok=sum(bool(c["all_ok"]) for _, c in fires), one=len(one),
                         one_runs=sorted({k for k, _ in one}),
                         one_detail=[(k.split("|")[1], c["step"], c["map_before"], c.get("map_after"), bound_routed(runs[k]),
                                      runs[k].get("transition")) for k, c in one],
                         br=sum(bound_routed(r) for r in runs.values())))
    return rows


def show(rows, title):
    print(title)
    print(f"  {'file':<14} {'runs':>4} {'firings':>7} {'key ok':>6} {'all ok':>6} {'map 1:1 before':>14} {'runs BR at end':>14}")
    t = dict(runs=0, firings=0, key_ok=0, all_ok=0, one=0, br=0)
    for d in rows:
        print(f"  {d['file']:<14} {d['runs']:>4} {d['firings']:>7} {d['key_ok']:>6} {d['all_ok']:>6} {d['one']:>14} {d['br']:>14}")
        for k in t:
            t[k] += d[k]
    print(f"  {'total':<14} {t['runs']:>4} {t['firings']:>7} {t['key_ok']:>6} {t['all_ok']:>6} {t['one']:>14} {t['br']:>14}")
    for d in rows:
        for seed, step, mb, ma, br, tr in d["one_detail"]:
            print(f"    {d['file']} seed {seed}: fired at {step} with a one-to-one map before {mb} -> after {ma}; "
                  f"the run BOUND ROUTED at the end: {br} (transition {tr})")
    return t


def main():
    pin = subprocess.run(["git", "-C", HERE, "rev-parse", PIN], capture_output=True, text=True).stdout.strip()
    head = subprocess.run(["git", "-C", HERE, "rev-parse", MAIN], capture_output=True, text=True).stdout.strip()
    t0 = show(count(PIN), f"(a) the 50 runs: the main line at {pin[:7]} (S86's set)")
    ok = (t0["runs"], t0["firings"], t0["key_ok"], t0["all_ok"]) == (50, 107, 107, 97)
    print(f"  REPRODUCES S86's count (50 runs, 107 firings, key pair 107, all-position pair 97): {ok}")
    t1 = show(count(MAIN), f"(b) now: the main line at {head[:7]} (L's split_copy file added)")
    print(f"  field: checks[].map_before (routing_k's VAL-position map on the run's probe, before the operation)")
    print(f"  firings with a one-to-one map before: (a) {t0['one']} of {t0['firings']}; (b) {t1['one']} of {t1['firings']}")
    return ok


if __name__ == "__main__":
    main()
