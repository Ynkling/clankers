#!/usr/bin/env python3 -I
import json, math, sys
P = "branch claude/explore-I/explore_out/I/gpu/mainline/test_window_gate_NVIDIA_GeForce_GTX_1660_SUPER_results.json"
d = json.load(open(P))
meta = d.get("meta", {})
print("meta keys", list(meta.keys()))
for k in ("started", "torch", "device", "git", "code_sha", "cpu", "seeds", "workers"):
    if k in meta:
        print(" ", k, str(meta[k])[:300])
runs = d["runs"] if "runs" in d else d
by = {}
for k, r in runs.items():
    arm, seed = k.split("|")
    by.setdefault(arm, {})[int(seed)] = r
print("arms", {a: len(v) for a, v in by.items()}, "total", sum(len(v) for v in by.values()))
def ok(r): return bool(r and r.get("ok"))
def bound(r): return ok(r) and r.get("transition") is not None
def routed(r):
    e = r["end"]; return bool(e["one_to_one"] and min(e["stream_acc"]) >= 0.9)
def br(r): return bound(r) and routed(r)
def mc(b, c):
    n = b + c; return 1.0 if n == 0 else sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n
print("CEIL_C", [(s, r["transition"], ok(r)) for s, r in sorted(by["CEIL_C"].items())])
seeds = sorted(by["WIN3_SPLIT_D8"])
print("seeds", seeds[0], seeds[-1], len(seeds), "all ok", all(ok(r) for v in by.values() for r in v.values()))
cnt = {a: sum(br(by[a][s]) for s in seeds) for a in ("WIN3_SPLIT_D8", "WIN3_SLOW_D8", "HINGE_D8")}
print("BR counts", cnt)
for a, b_ in (("WIN3_SPLIT_D8", "HINGE_D8"), ("WIN3_SPLIT_D8", "WIN3_SLOW_D8"), ("WIN3_SLOW_D8", "HINGE_D8")):
    b = sum(br(by[a][s]) and not br(by[b_][s]) for s in seeds)
    c = sum(br(by[b_][s]) and not br(by[a][s]) for s in seeds)
    print(f" {a} vs {b_}: b={b} c={c} p={mc(b, c):.2e}")
shas = set(r.get("code_sha") or r.get("code") or r.get("git") for v in by.values() for r in v.values())
print("code shas in records", shas)
r0 = by["WIN3_SPLIT_D8"][seeds[0]]
print("record keys", list(r0.keys()))
print("device/torch fields:", {k: r0.get(k) for k in ("device", "torch", "threads", "cpu", "iters", "code_sha")})
print("secs sum h", sum(r["secs"] for v in by.values() for r in v.values()) / 3600)
# one-to-one before firing count for guard check (H's claim): count firings with map one-to-one before
fires = 0; one = 0
for a in ("WIN3_SPLIT_D8",):
    for s in seeds:
        for ch in by[a][s].get("checks", []):
            if ch.get("fired"):
                fires += 1
                mb = ch.get("map_before")
                if mb and len(set(mb)) == len(mb):
                    one += 1
print("GPU WIN3_SPLIT firings", fires, "one-to-one before", one)
