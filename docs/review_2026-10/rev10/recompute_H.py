#!/usr/bin/env python3 -I
import json, math
H = "branch claude/explore-H/explore_out/H/"
M = "the main-line clone/results/"

def ok(r): return bool(r and r.get("ok"))
def bound(r): return ok(r) and r.get("transition") is not None
def routed(r):
    e = r["end"]; return bool(e["one_to_one"] and min(e["stream_acc"]) >= 0.9)
def br(r): return bound(r) and routed(r)
def load(p):
    d = json.load(open(p)); runs = d.get("runs", d); by = {}
    for k, r in runs.items():
        parts = k.split("|"); a, s = parts[0], parts[1]; by.setdefault(a, {})[int(s)] = r
    return d, by

# S87a allmass2
d, by = load(H + "allmass2_results.json")
print("allmass2 arms", {a: sorted(v)[0:1] + [len(v)] for a, v in by.items()})
seeds = sorted(by["BASE"])
for a in ("BASE", "ALLMASS"):
    c = sum(br(by[a][s]) for s in seeds)
    fired = [ch for s in seeds for ch in by[a][s].get("checks", []) if ch.get("fired")]
    print(a, f"BR {c}/{len(seeds)}; firings {len(fired)}; key_ok {sum(1 for ch in fired if ch.get('key_ok'))}; all_ok {sum(1 for ch in fired if ch.get('all_ok'))}; own pair ok {sum(1 for ch in fired if ch.get('own_ok', ch.get('all_ok')))}; 1:1 before {sum(1 for ch in fired if ch.get('map_before') and len(set(ch['map_before'])) == len(ch['map_before']))}")
    if fired: print("  check keys", sorted(fired[0].keys()))
b = sum(br(by["ALLMASS"][s]) and not br(by["BASE"][s]) for s in seeds)
c = sum(br(by["BASE"][s]) and not br(by["ALLMASS"][s]) for s in seeds)
print("ALLMASS only", b, "BASE only", c)
print("ORACLE", [(s, r["transition"]) for s, r in sorted(by.get("ORACLE", {}).items())])
# S86 pooled: allmass_results.json
d2, by2 = load(H + "allmass_results.json")
print("S86 arms", {a: (sorted(v)[0], sorted(v)[-1], len(v)) for a, v in by2.items()})
s86 = sorted(by2["BASE"])
for a in ("BASE", "ALLMASS"):
    print(" S86", a, sum(br(by2[a][s]) for s in s86), "/", len(s86))
b2 = sum(br(by2["ALLMASS"][s]) and not br(by2["BASE"][s]) for s in s86)
c2 = sum(br(by2["BASE"][s]) and not br(by2["ALLMASS"][s]) for s in s86)
print(" S86 ALLMASS only", b2, "BASE only", c2, "-> pooled", b + b2, c + c2)

# S87c dose
d3, by3 = load(H + "dose_results.json")
print("dose arms", {a: len(v) for a, v in by3.items()})
for a, v in by3.items():
    print(" ", a, "discovered", sum(bool(r.get("discovered")) for r in v.values()), "/", len(v), "bound", sum(bound(r) for r in v.values()))

# the one-to-one guard count over the main line's 60 runs
tot_runs = tot_f = tot_one = 0
for f, arm, seeds_ in (("X/window_gate_results.json", "WIN3_SPLIT_D8", range(540, 560)), ("L/window_gate_results.json", "WIN3_SPLIT_D8", range(1540, 1560)),
                       ("X/split_copy_results.json", None, None), ("L/split_copy_results.json", None, None)):
    dd, bb = load(M + f)
    if arm is None:
        arms = ["WIN3_SPLIT_D8"]
        print(f, "arms", list(bb.keys()))
        arm = arms[0]; seeds_ = sorted(bb[arm])
    runs = [bb[arm][s] for s in seeds_]
    fired = [ch for r in runs for ch in r.get("checks", []) if ch.get("fired")]
    one = sum(1 for ch in fired if ch.get("map_before") and len(set(ch["map_before"])) == len(ch["map_before"]))
    keyok = sum(1 for ch in fired if ch.get("key_ok"))
    print(f, arm, "runs", len(runs), "firings", len(fired), "key_ok", keyok, "1:1 before", one, "BR", sum(br(r) for r in runs))
    tot_runs += len(runs); tot_f += len(fired); tot_one += one
print("TOTAL runs", tot_runs, "firings", tot_f, "one-to-one before", tot_one)
