#!/usr/bin/env python3
"""check_onetoone_v5.py -- the referee's own recount of tex6:441 (c916996): "None of the 126 fired on a map that was
already one-to-one (session H's count)", and the firing counts beside it (126 firings in 60 runs; the all-position pair
labelled at 115; the two pairs coincide at 54).

    python3 docs/review_2026-10/rev9/check_onetoone_v5.py      (stdlib; reads git objects; writes nothing)

Records: WIN3_SPLIT_D8 in results/{X,L}/window_gate_results.json and results/{X,L}/split_copy_results.json on the main line
at 8ae660d (L's split_copy from ffb0771). A firing is a check with fired true; its map is checks[].map_before (the
trigger's stream -> channel map before the operation); one-to-one = 8 distinct channels. Independent of H's script: the
key-position and all-position pair labels are read from key_ok / all_ok, and "same" is the two chosen pairs (c*, c0) being equal.
"""
import json, subprocess
REF = "8ae660d"
FILES = ("results/X/window_gate_results.json", "results/L/window_gate_results.json",
         "results/X/split_copy_results.json", "results/L/split_copy_results.json")
tot = dict(runs=0, fired=0, one=0, key=0, all=0, same=0)
for p in FILES:
    runs = json.loads(subprocess.check_output(["git", "show", f"{REF}:{p}"]))["runs"]
    rs = {k: r for k, r in runs.items() if k.split("|")[0] == "WIN3_SPLIT_D8" and r.get("checks")}
    f = [c for r in rs.values() for c in r["checks"] if c.get("fired")]
    one = sum(len(set(c["map_before"])) == 8 for c in f)
    key = sum(bool(c.get("key_ok")) for c in f); al = sum(bool(c.get("all_ok")) for c in f)
    same = sum(1 for c in f if (c["key_cs"], c["key_c0"]) == (c["all_cs"], c["all_c0"]))
    print(f"{p}: runs {len(rs)}, firings {len(f)}, on a one-to-one map {one}, key_ok {key}, all_ok {al}, same pair {same}")
    for k, v in zip(("runs", "fired", "one", "key", "all", "same"), (len(rs), len(f), one, key, al, same)):
        tot[k] += v
print("total:", tot)
print("fields of a fired check:", sorted(next(c for r in rs.values() for c in r["checks"] if c.get("fired"))))

# Second check of c916996: the abstract's "(the delta recipe 19/20 against the Hebbian 16/20, and without a trigger 19/20
# against 7/20, 10 vs 1)". Discordant pairs (bound routed, rlib) on X's test_delta_recipe records at 8ae660d.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rlib
st = rlib.load("8ae660d", "results/X/delta_recipe_results.json")
R = {a: rlib.runs(st, a) for a in ("HEBB_SPLIT", "HEBB_SLOW", "DELTA_SPLIT", "DELTA_SLOW")}
for a, b in (("DELTA_SPLIT", "HEBB_SPLIT"), ("DELTA_SLOW", "HEBB_SLOW"), ("HEBB_SPLIT", "HEBB_SLOW"), ("DELTA_SPLIT", "DELTA_SLOW")):
    x, y, n = rlib.pairs(R[a], R[b], rlib.bound_routed)
    print(f"{a} vs {b}: {sum(map(rlib.bound_routed, R[a].values()))}/{n} vs {sum(map(rlib.bound_routed, R[b].values()))}/{n}; {x} vs {y}")
