#!/usr/bin/env python3
"""count_allpos.py: the second Revision 9 draft's limits count (tex2:436 at f67bc07):
"at the 107 firings of the 50 recorded split runs at eight streams (window_gate X and L, split_copy X) the all-position
pair was a labelled target ... at 97 (21 of 22, 46 of 47 and 30 of 38), the key-position pair at all 107; the two pairs
coincide at 46."

    python3 docs/review_2026-10/rev9/count_allpos.py      (stdlib + rlib; writes count_allpos_out.txt)

For every WIN3_SPLIT_D8 run (window_gate X 540-559, L 1540-1559; split_copy X 560-569) and every check that fired:
the stored flags all_ok / key_ok, and the same labels recomputed from map_before (a stream -> channel list):
a pair (c*, c0) is a labelled target if c* holds >= 2 streams and c0 holds none.
"""
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from rlib import load, runs  # noqa: E402

MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
out = []


def say(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s)


def labelled(mp, cs, c0):
    c = collections.Counter(mp)
    return c.get(cs, 0) >= 2 and c.get(c0, 0) == 0


SRC = [("window_gate X", MAIN, "results/X/window_gate_results.json"),
       ("window_gate L", MAIN, "results/L/window_gate_results.json"),
       ("split_copy X", "592c574", "results/X/split_copy_results.json")]
tot = collections.Counter()
for name, ref, path in SRC:
    R = runs(load(ref, path), "WIN3_SPLIT_D8")
    c = collections.Counter()
    keys_seen = set()
    for s, r in sorted(R.items()):
        c["runs"] += 1
        for ck in r.get("checks", []):
            if not ck.get("fired"):
                continue
            keys_seen |= set(ck)
            c["fired"] += 1
            mp = ck.get("map_before")
            ks, k0, as_, a0 = ck.get("key_cs"), ck.get("key_c0"), ck.get("all_cs"), ck.get("all_c0")
            c["key_ok stored"] += bool(ck.get("key_ok"))
            c["all_ok stored"] += bool(ck.get("all_ok"))
            if mp is not None and None not in (ks, k0, as_, a0):
                c["key_ok recomputed"] += labelled(mp, ks, k0)
                c["all_ok recomputed"] += labelled(mp, as_, a0)
                c["pairs coincide"] += (ks, k0) == (as_, a0)
                c["c* coincide"] += ks == as_
                c["stored != recomputed"] += (bool(ck.get("all_ok")) != labelled(mp, as_, a0)) + \
                    (bool(ck.get("key_ok")) != labelled(mp, ks, k0))
            else:
                c["fields missing"] += 1
    say(f"{name} ({ref}:{path}): " + ", ".join(f"{k} {v}" for k, v in c.items()))
    tot.update(c)
say("TOTAL: " + ", ".join(f"{k} {v}" for k, v in tot.items()))
say("draft (tex2:436): runs 50, firings 107, all-position labelled 97 (21/22 split_copy X, 46/47 window_gate X, "
    "30/38 window_gate L), key-position 107, pairs coincide 46")
with open(os.path.join(HERE, "count_allpos_out.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
