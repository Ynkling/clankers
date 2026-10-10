#!/usr/bin/env python
"""check_appendixB.py -- read-only helper for check_appendixB.md (Revision 9, Appendix B and tex:205/384).

Reads the main line's window_gate records and code at origin/claude/bdh-growth-hebbian-inference-w90069 (32c121b)
through `git show` (nothing is checked out, nothing is trained) and prints:
  1. which fields the WIN3_SPLIT_D8 / WIN3_RESET_D8 'checks' rows hold (probe reading, masses, map);
  2. per check: the probe's accuracy against the held-out evaluation at the same step (the curve);
     counts of checks where the probe read >= 0.95 while the evaluation read < 0.95;
  3. whether per-stream gate masses are stored at every evaluation (stats[*].stream_gate);
  4. the meta's start-time string (time zone or not) and commit;
  5. the split probe's per-stream query counts (BindTask.make_batch(64, Generator(12345)) on TASK48),
     if --code DIR points at an extracted copy of the main line's *.py (git archive), else skipped.

Usage: python check_appendixB.py [--code DIR]   (run from the repository root)
"""
import argparse
import json
import subprocess
import sys

BR = "origin/claude/bdh-growth-hebbian-inference-w90069"
ARMS = ("WIN3_SPLIT_D8", "WIN3_RESET_D8")


def load(machine):
    src = subprocess.run(["git", "show", f"{BR}:results/{machine}/window_gate_results.json"], capture_output=True,
                         text=True, check=True).stdout
    return json.loads(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default=None)
    args = ap.parse_args()
    tot = dict(checks=0, probe_hi_eval_lo=0, blocked_by_hi=0, runs_affected=set())
    for mach in "XL":
        d = load(mach)
        me = d["meta"]
        print(f"== {mach}: meta.git={me.get('git')} started={me.get('started')!r} starts={me.get('starts')} "
              f"cpu={me.get('cpu')!r}")
        for arm in ARMS:
            keys = sorted(k for k in d["runs"] if k.split("|")[0] == arm)
            n_chk = n_hi = n_blk = 0
            fields = set()
            sg_ok = True
            fired_zeroed = set()
            rows_out = []
            for k in keys:
                r = d["runs"][k]
                curve = {c[0]: c[1] for c in r["curve"]}
                stats_steps = {s["step"] for s in r["stats"] if "stream_gate" in s}
                n_split, last = 0, None
                for c in r["checks"]:
                    # would the gap/cap have blocked this check anyway? (state before this check)
                    capped = n_split >= 3 or (last is not None and c["step"] - last < 4800)
                    rise_ok = c["prev"] is not None and c["acc"] - c["prev"] < 0.02
                    c = dict(c, _capped=capped, _rise_ok=rise_ok)
                    if c.get("fired"):
                        n_split, last = n_split + 1, c["step"]
                    fields |= {f for f in c if not f.startswith("_")}
                    n_chk += 1
                    ev = curve.get(c["step"])
                    if c["step"] not in stats_steps:
                        sg_ok = False
                    if c.get("fired"):
                        fired_zeroed |= set(c.get("state_zeroed", []))
                    if ev is not None and c["acc"] >= 0.95 and ev < 0.95:
                        n_hi += 1
                        # a check from 4800 with probe >= 0.95 is never eligible: blocked by the probe
                        blocked = c["step"] >= 4800 and not c["eligible"]
                        n_blk += blocked
                        tot["runs_affected"].add(f"{mach}:{k}")
                        rows_out.append(f"     {k} step {c['step']}: probe {c['acc']:.4f}  eval {ev:.4f}  "
                                        f"eligible {c['eligible']}  fired {c['fired']}  gap/cap would block anyway "
                                        f"{c['_capped']}  rise < 0.02 {c['_rise_ok']}  map_before {c['map_before']}")
            print(f"  {arm}: {len(keys)} runs, {n_chk} check rows; row fields: {sorted(fields)}")
            print(f"    every check step has stats[*].stream_gate (per-stream x per-channel mean gate at value "
                  f"positions, 256-sequence probe): {sg_ok}")
            print(f"    Adam state keys zeroed at fired rows: {sorted(fired_zeroed)}")
            print(f"    checks with probe >= 0.95 and the evaluation at the same step < 0.95: {n_hi} "
                  f"(of which at a check >= 4800, i.e. not eligible because of the probe: {n_blk})")
            for line in rows_out:
                print(line)
            if arm == "WIN3_SPLIT_D8":
                tot["checks"] += n_chk
                tot["probe_hi_eval_lo"] += n_hi
                tot["blocked_by_hi"] += n_blk
    print(f"== WIN3_SPLIT_D8 pooled X+L: {tot['checks']} check rows; probe >= 0.95 while eval < 0.95: "
          f"{tot['probe_hi_eval_lo']} ({tot['blocked_by_hi']} at checks >= 4800); runs (both arms) "
          f"{sorted(tot['runs_affected'])}")
    if args.code:
        sys.path.insert(0, args.code)
        import torch
        import test_stream_curriculum as tscur
        task = tscur.TASK48
        x, qs = task.make_batch(64, torch.Generator().manual_seed(12345))
        print("split probe per-stream query counts (streams 0-7):",
              [int((qs.reshape(-1) == s).sum()) for s in range(task.S)], "n sequences", x.shape[0],
              "length", x.shape[1], "qpos", task.qpos)


if __name__ == "__main__":
    main()
