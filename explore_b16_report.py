#!/usr/bin/env python
"""
explore_b16_report.py — EXPLORATORY, not a result. Batch 16's shared report helpers (parent side, no runs): per-seed
outcome tables with the paired runs, BOUND ROUTED counts with exact McNemar, channel maps, decodability tables.
"""

from collections import Counter

import explore_common as ec
import explore_common16 as c16
import explore_muon_scale as s33

MAP_AT = (4800, 9600, "end")
DIAG = ("1200", "2400", "4800", "9600")


def br(r):
    return bool(r is not None and r.get("outcome") == "BOUND ROUTED")


def f2(x, w=5):
    return f"{'--':>{w}}" if x is None else f"{x:{w}.2f}"


def compare(new, old, seeds, lab_new, lab_old, what="BOUND ROUTED"):
    d = c16.mcnemar({s: br(r) for s, r in new.items()}, {s: br(r) for s, r in old.items()}, seeds)
    print(f"    {what}  {lab_new} {d['new']}/{d['n']}  {lab_old} {d['old']}/{d['n']}  {lab_new} only {d['b']}, {lab_old} only "
          f"{d['c']}, McNemar two-sided p = {d['p']:.3g}")
    return d


def dec(r, t, f):
    d = (r.get("decode") or {}).get(t) if r else None
    return None if d is None else d.get(f)


def arm_table(arm, runs, seeds, pairs, label):
    """pairs: [(label, {seed: record})] printed alongside (outcome)."""
    print(f"  ARM {arm} ({label}), seeds {ec.fmt_seeds(seeds)}")
    print(f"    {'seed':>4} | {arm:<24} {'trans':>5} {'stop':>5} {'acc':>5} {'ch@4800':>9} {'ch@9600':>9} {'ch@end':>9} | "
          + " | ".join(f"{lab:<22}" for lab, _ in pairs))
    for s in seeds:
        r = runs.get(s)
        if r is None:
            print(f"    {s:>4} | {'not run':<24}")
            continue
        print(f"    {s:>4} | {r['outcome']:<24} {str(r['transition']):>5} {r.get('stopped_at')!s:>5} {r['acc']:5.2f} "
              + " ".join(f"{s33.per_channel(s33.map_at(r, t)):>9}" for t in MAP_AT) + " | "
              + " | ".join(f"{(p.get(s) or {}).get('outcome', '--'):<22}" for _, p in pairs))


def maps_block(runs):
    print("    distinct channels holding the streams (routing_k's map; streams per channel), per seed:")
    for t in MAP_AT:
        print(f"      at {str(t):>5}: " + "  ".join(f"{s}:{s33.distinct(s33.map_at(r, t)) or '--'} ({s33.per_channel(s33.map_at(r, t))})"
                                                for s, r in sorted(runs.items())))


def decode_block(runs, steps=DIAG, chance=None, feats=("h_ctx", "h_key", "h_val")):
    labs = {"h_ctx": "CTX", "h_key": "KEY", "h_val": "VAL", "u_ctx": "uCTX", "u_key": "uKEY", "u_val": "uVAL"}
    print(f"    S38's decoder: stream decodability of the gate state h at " + " / ".join(labs[f] for f in feats)
          + " positions" + (f" (chance {chance:.3f})" if chance else "") + ", per seed:")
    print(f"    {'seed':>4} | " + " | ".join(f"{t:>5}: " + " ".join(f"{labs[f]:>5}" for f in feats) for t in steps))
    for s, r in sorted(runs.items()):
        print(f"    {s:>4} | " + " | ".join(f"{'':>5}  " + " ".join(f2(dec(r, t, f)) for f in feats) for t in steps))
    meds = {}
    print(f"    {'med':>4} | " + " | ".join(
        f"{'':>5}  " + " ".join(f2(meds.setdefault((t, f), c16.med([dec(r, t, f) for r in runs.values()]))) for f in feats)
        for t in steps))
    return meds


def classes(runs):
    return dict(Counter(r["outcome"] for r in runs.values()))


def transitions(runs):
    return sorted(r["transition"] for r in runs.values() if r.get("transition") is not None)
