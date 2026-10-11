#!/usr/bin/env python
"""
explore_h_allmass_counts.py — EXPLORATORY, not a result. Session H, S86's preliminary count (no runs): over the main line's
split runs (WIN3_SPLIT_D8 in results/X/window_gate_results.json, results/L/window_gate_results.json and
results/X/split_copy_results.json, read from claude/bdh-growth-hebbian-inference-w90069 with git show), at every firing:
whether the key-position pair (key_cs -> key_c0, the trigger's KEYMASS target) and the all-position pair (all_cs -> all_c0,
the mean read-gate mass over all probe positions, recorded beside it) were labelled targets (key_ok / all_ok: c* holds >= 2
streams and c0 none on the labelled map before the firing), and whether the two pairs coincide (same c* and same c0).

Run:  python explore_h_allmass_counts.py [--ref <branch or commit>]
"""

import argparse
import json
import subprocess
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
FILES = (("window_gate X", "results/X/window_gate_results.json"), ("window_gate L", "results/L/window_gate_results.json"),
         ("split_copy X", "results/X/split_copy_results.json"))
ARM = "WIN3_SPLIT_D8"


def load(ref, path):
    return json.loads(subprocess.run(["git", "-C", HERE, "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout)


def count(ref=MAIN):
    rows, tot = [], dict(runs=0, firings=0, key_ok=0, all_ok=0, same=0, both_ok=0)
    for lab, path in FILES:
        runs = {k: r for k, r in load(ref, path)["runs"].items() if k.split("|")[0] == ARM and r.get("checks")}
        f = [c for r in runs.values() for c in r["checks"] if c.get("fired")]
        d = dict(file=lab, runs=len(runs), firings=len(f), key_ok=sum(bool(c["key_ok"]) for c in f),
                 all_ok=sum(bool(c["all_ok"]) for c in f),
                 same=sum(c["key_cs"] == c["all_cs"] and c["key_c0"] == c["all_c0"] for c in f),
                 both_ok=sum(bool(c["key_ok"]) and bool(c["all_ok"]) for c in f))
        rows.append(d)
        for k in tot:
            tot[k] += d[k]
    return rows, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=MAIN)
    a = ap.parse_args()
    rows, tot = count(a.ref)
    sha = subprocess.run(["git", "-C", HERE, "rev-parse", a.ref], capture_output=True, text=True).stdout.strip()
    print(f"main line at {sha[:7]}; arm {ARM}; firings = checks with fired True")
    print(f"{'file':<15} {'runs':>4} {'firings':>7} {'key pair ok':>11} {'all pair ok':>11} {'pairs equal':>11} {'both ok':>7}")
    for d in rows + [dict(tot, file="total")]:
        print(f"{d['file']:<15} {d['runs']:>4} {d['firings']:>7} {d['key_ok']:>11} {d['all_ok']:>11} {d['same']:>11} {d['both_ok']:>7}")


if __name__ == "__main__":
    main()
