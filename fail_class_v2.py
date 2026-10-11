#!/usr/bin/env python
"""
fail_class_v2.py — the failure classes with the key rule ahead of the margin rule.

    python fail_class_v2.py            # reclassify the recorded two-stream runs, print the table
    python fail_class_v2.py --write    # and write it to docs/fail_class_v2.md

WHY
- test_router_layout.fail_class (v1) consults the routing margin first: margin >= MARGIN_PARTIAL
  is STREAM-PARTIAL before eta^2 by key is looked at. A hard gate that separates the two streams
  on two of the four keys and puts each of the other two keys wholly on a different channel has
  eta^2 by key 0.5 and margin about 0.5, so v1 labels it STREAM-PARTIAL although half of its
  split is by key identity. Batch 18 (claude/outside-ideas, explore_out/README.md, W2_KEYHINGE
  163, 182, 186) put the "removes" reading on which side of the threshold that configuration
  falls.
- v2: eta^2 by key >= ETA_SPLIT is KEY before the margin rule is consulted; the remaining rules
  are v1's, in v1's order. The thresholds are v1's (imported, not copied).
- test_router_layout.py is untouched, so every recorded verdict reproduces under v1. Verdicts
  read under v1 stay as recorded; docs/fail_class_v2.md says which recorded counts v2 would
  change. New tests use v2.
- The rule is applied as written: ">= ETA_SPLIT" on the stored float. A gate that is not fully
  hard puts the two-of-four pattern a little below or above 0.5 (batch 18's 182 and 186 at
  0.49986 and 0.49998), and those below stay STREAM-PARTIAL. The table lists every record
  within 1e-3 of the threshold.
"""

import argparse
import collections
import glob
import json
import os

from test_router_layout import MARGIN_PARTIAL, ETA_SPLIT
from test_stream_channels import merged, shared_max

CLASSES = ("STREAM-PARTIAL", "KEY", "POSITION", "OTHER")
NEAR = 1e-3                                # listed as near the threshold (printed only)


def fail_class_v2(r):
    e = r["end"]
    if e["eta_key_by_key"] >= ETA_SPLIT:
        return "KEY"
    if e["margin"] >= MARGIN_PARTIAL:
        return "STREAM-PARTIAL"
    if max(e["eta_key_by_half"], e["eta_key_by_index"]) >= ETA_SPLIT:
        return "POSITION"
    return "OTHER"


def multichannel(e):
    """test_scale_axes.fail_class_k's field mapping (block for half)."""
    return {"end": dict(margin=e["margin"], eta_key_by_key=e["etak_key_by_key"],
                        eta_key_by_half=e["etak_key_by_block"], eta_key_by_index=e["etak_key_by_index"])}


def fail_class_k_v2(r):
    """test_scale_axes.fail_class_k under v2."""
    return fail_class_v2(multichannel(r["end"]))


def outcome_v2(r, routed):
    """test_stream_recipe.outcome under v2 (routed: test_stream_recipe.routed or an equivalent)."""
    if r["transition"] is not None:
        return "BOUND ROUTED" if routed(r) else "BOUND NOT routed"
    fc = fail_class_k_v2(r)
    if fc == "STREAM-PARTIAL":
        return f"MERGED ({shared_max(r)} share)" if merged(r) else "STREAM-PARTIAL one-to-one"
    return f"non-stream {fc}"


# ── Reclassification of the recorded runs ───────────────────────────────────
# Population and classifier as each test's report applied them: test_router_layout classifies
# every non-discovering run, the others every unbound run (transition None); test_scale_axes
# uses fail_class_k, the others fail_class; test_curriculum_confirm and test_load_curriculum
# also classify the gate at the switch when it is not routed there (routed_sw).
ROUTER_LAYOUT = "router_layout"
MULTICHANNEL_K2 = ("scale_axes",)
SWITCH = ("curriculum_confirm", "load_curriculum")


def load(path):
    d = json.load(open(path))
    return d.get("runs", d)


def test_of(path):
    return os.path.basename(path)[:-len("_results.json")]


def two_stream(r):
    """A two-stream gated record with the end statistics fail_class reads (k=2; on the main
    line every k=2 record has S=2)."""
    e = r.get("end")
    return (r.get("k") == 2 and isinstance(e, dict) and "margin" in e and "eta_key_by_key" in e
            and len(e.get("stream_acc", [0, 0])) == 2)


def classified(test, r):
    if test == ROUTER_LAYOUT:
        return not r["discovered"]
    return r["transition"] is None


def classify(test, r):
    if test in MULTICHANNEL_K2:
        from test_scale_axes import fail_class_k
        return fail_class_k(r), fail_class_k_v2(r), r["end"]["etak_key_by_key"]
    from test_router_layout import fail_class
    return fail_class(r), fail_class_v2(r), r["end"]["eta_key_by_key"]


def switch_rows(r):
    from test_load_curriculum import routed_sw, sw_margin
    from test_router_layout import fail_class
    if sw_margin(r) is None or routed_sw(r):
        return None
    sw = {"end": r["switch"]}
    return fail_class(sw), fail_class_v2(sw), r["switch"]["eta_key_by_key"]


def reclassify(files):
    """{(file, arm): {"n", "v1", "v2", "changed", "near"}} for the end, and for the switch."""
    end, sw = collections.OrderedDict(), collections.OrderedDict()
    skipped = []
    for path in files:
        test, runs = test_of(path), load(path)
        if not isinstance(runs, dict):
            continue
        for key, r in runs.items():
            if not isinstance(r, dict) or not r.get("ok", True):
                continue
            arm = r.get("arm", key.split("|")[0])
            if r.get("k") == 2 and not two_stream(r):
                skipped.append((path, arm))
                continue
            if not two_stream(r):
                continue
            for table, row in ((end, classify(test, r) if classified(test, r) else None),
                               (sw, switch_rows(r) if test in SWITCH else None)):
                if row is None:
                    continue
                t = table.setdefault((path, arm), dict(n=0, v1=collections.Counter(), v2=collections.Counter(),
                                                       changed=[], near=[]))
                a, b, eta = row
                t["n"] += 1
                t["v1"][a] += 1
                t["v2"][b] += 1
                if a != b:
                    t["changed"].append((r["seed"], a, b, eta))
                if abs(eta - ETA_SPLIT) < NEAR:
                    t["near"].append((r["seed"], a, b, eta))
    return end, sw, sorted(set(skipped))


def beyond_two(files):
    """Records with more than two streams whose recorded outcome (test_stream_recipe.outcome,
    or fail_class_k on unbound runs below k=16) would change under v2."""
    from test_scale_axes import fail_class_k
    out = []
    for path in files:
        runs = load(path)
        if not isinstance(runs, dict):
            continue
        for key, r in runs.items():
            if not isinstance(r, dict) or not r.get("ok", True) or r.get("transition") is not None:
                continue
            e = r.get("end")
            if not isinstance(e, dict) or "etak_key_by_key" not in e or len(e.get("stream_acc", [])) <= 2:
                continue
            a, b = fail_class_k(r), fail_class_k_v2(r)
            if a != b:
                out.append((path, r.get("arm", key.split("|")[0]), r["seed"], len(e["stream_acc"]), r["k"],
                            a, b, e["etak_key_by_key"], e["margin"]))
    return sorted(out)


def counts(c):
    return " / ".join(str(c.get(m, 0)) for m in CLASSES)


def table(rows, title):
    out = [f"### {title}", "",
           "Counts are STREAM-PARTIAL / KEY / POSITION / OTHER.", "",
           "| File | Arm | classified | v1 | v2 | would change |",
           "|---|---|---|---|---|---|"]
    for (path, arm), t in rows.items():
        ch = ", ".join(f"s{s} {a} -> {b} (eta^2 by key {eta:.7f})" for s, a, b, eta in t["changed"]) or "none"
        out.append(f"| {path[len('results/'):]} | {arm} | {t['n']} | {counts(t['v1'])} | {counts(t['v2'])} | {ch} |")
    return out


def report(files):
    end, sw, skipped = reclassify(files)
    lines = ["# fail_class v2: reclassification of the recorded two-stream runs", "",
             "Generated by `python fail_class_v2.py --write` (the module holds the rule and this table's code).", "",
             "## The rule", "",
             f"- v1 (`test_router_layout.fail_class`, unchanged): margin >= {MARGIN_PARTIAL} is STREAM-PARTIAL; "
             f"else eta^2 by key >= {ETA_SPLIT} is KEY; else eta^2 by half or index >= {ETA_SPLIT} is POSITION; else OTHER.",
             f"- v2 (`fail_class_v2.fail_class_v2`): eta^2 by key >= {ETA_SPLIT} is KEY first; then v1's rules in v1's "
             "order. `fail_class_k_v2` and `outcome_v2` are `test_scale_axes.fail_class_k` and "
             "`test_stream_recipe.outcome` with v2 in place of v1.",
             "- The two differ only where eta^2 by key >= 0.5 and margin >= 0.25 (v1 STREAM-PARTIAL, v2 KEY).",
             "- No recorded verdict changes: every verdict was read under v1 and stays as recorded. This table says "
             "which recorded counts v2 would change. New tests use v2.", "",
             "## Population", "",
             "Every main-line record (results/X and results/L) with two streams and a gate (k=2) and the end "
             "statistics fail_class reads, classified as its test's report classified it: test_router_layout every "
             "non-discovering run; every other test every unbound run (transition None); test_scale_axes through "
             "fail_class_k (its multichannel eta^2), the others through fail_class. test_curriculum_confirm and "
             "test_load_curriculum also classify the gate at the switch where it is not routed there; that is the "
             "second table. Bound runs that the tests did not classify are not counted.", ""]
    lines += table(end, "At the end") + [""]
    lines += table(sw, "At the switch (curriculum tests)") + [""]
    tot = collections.Counter()
    for t in list(end.values()) + list(sw.values()):
        tot["n"] += t["n"]
        tot["changed"] += len(t["changed"])
    lines += [f"In all: {tot['n']} classifications, {tot['changed']} would change "
              f"(all STREAM-PARTIAL -> KEY).", ""]
    lines += ["## Recorded counts v2 would change", "",
              "Verdicts: none. No test computes a claim from the failure classes; they appear in per-seed tables "
              "and diagnostics only (the one outcome comparison in a claim, test_window_gate's "
              "`tsr.outcome(r) == \"BOUND ROUTED\"`, is on bound runs, which v2 does not touch).", "",
              "Diagnostic counts written in docstrings (X):", ""] + RECORDED + ["",
              "L's changed runs (slow_start A0 s292, recipe_scope HONLY0 s286, window_gate WIN3_SLOW s1538) "
              "are in the table above; no docstring quotes L's failure-class counts for these arms.", ""]
    lines += ["## Near the threshold", "",
              f"Every classified record with eta^2 by key within {NEAR:g} of {ETA_SPLIT}, with its stored value. "
              "These are the two-of-four pattern on a nearly hard gate; the rule is applied to the stored value as "
              "written, so the ones a little below 0.5 stay STREAM-PARTIAL under v2.", "",
              "| File | Arm | seed | where | eta^2 by key | v1 | v2 |", "|---|---|---|---|---|---|---|"]
    for where, rows in (("end", end), ("switch", sw)):
        for (path, arm), t in rows.items():
            for s, a, b, eta in t["near"]:
                lines.append(f"| {path[len('results/'):]} | {arm} | {s} | {where} | {eta!r} | {a} | {b} |")
    lines += ["", "## Batch 18 (exploratory, not on the main line; quoted for the rule's motivation)", "",
              "claude/outside-ideas, explore_out/two_stream_keysplits_results.json, W2_KEYHINGE's three runs at "
              "the two-of-four pattern (stored values):", "",
              "| seed | eta^2 by key | margin | v1 | v2 |", "|---|---|---|---|---|"]
    from test_router_layout import fail_class
    for s, eta, m, half, index in BATCH18:
        e = {"end": dict(eta_key_by_key=eta, margin=m, eta_key_by_half=half, eta_key_by_index=index)}
        lines.append(f"| {s} | {eta!r} | {m:.4f} | {fail_class(e)} | {fail_class_v2(e)} |")
    lines += ["", "So the rule as written labels 163 KEY and leaves 182 and 186 STREAM-PARTIAL: their gates are "
              "nearly hard and their stored eta^2 by key is 1.4e-4 and 1.5e-5 below 0.5. The same holds on the main "
              "line: X WIN3_SLOW s522 sits 2e-11 below 0.5 (float rounding of the exact pattern) and L WIN3 s1518 "
              "1e-5 below, so both stay STREAM-PARTIAL while s517, s520, s531, s290 and s286 just above become KEY. "
              "Labelling every instance of the pattern KEY would need a tolerance, which v2 as specified does not "
              "have.", ""]
    bt = beyond_two(files)
    lines += ["## Beyond two streams (not reclassified; listed because new tests use v2)", "",
              "Unbound records with four or eight streams whose fail_class_k label would change under v2. "
              "For k=16 this changes test_stream_recipe.outcome from MERGED / STREAM-PARTIAL one-to-one to "
              "non-stream KEY. So v1 and v2 can disagree at eight streams too.", "",
              "| File | Arm | seed | S | k | v1 | v2 | eta^2 by key (multichannel) | margin |",
              "|---|---|---|---|---|---|---|---|---|"]
    for path, arm, s, S, k, a, b, eta, m in bt:
        lines.append(f"| {path[len('results/'):]} | {arm} | {s} | {S} | {k} | {a} | {b} | {eta:.4f} | {m:.4f} |")
    if not bt:
        lines.append("| none | | | | | | | | |")
    lines += ["", "## Not reclassifiable", ""]
    lines += [f"- {p[len('results/'):]} {a}: k=2 without the end statistics fail_class reads." for p, a in skipped]
    lines += ["- Files with k=None records (router_confirm, router_curriculum, router_discovery, readout_path; "
              "channel_binding where present) predate the failure classes; k=1 arms have no gate.", ""]
    return "\n".join(lines)


RECORDED = [
    "- test_slow_start.py, Part 0 failures: A0 \"9 POSITION, 2 KEY, 1 STREAM-PARTIAL\" would be 9 POSITION, "
    "3 KEY (s290). test_recipe_scope.py quotes the same A0 counts.",
    "- test_window_gate.py, Part A: \"WIN3 KEY 5 + STREAM-PARTIAL 2, WIN3_SLOW KEY 4 + STREAM-PARTIAL 3\" would "
    "be WIN3 KEY 7 (s517, s520), WIN3_SLOW KEY 5 + STREAM-PARTIAL 2 (s531; s522 stays: 0.4999999999784073).",
]

# W2_KEYHINGE 163, 182, 186 (claude/outside-ideas at eb7957b, explore_out/two_stream_keysplits_results.json):
# (seed, end eta_key_by_key, margin, eta_key_by_half, eta_key_by_index), copied from the stored records.
BATCH18 = ((163, 0.5000037867886511, 0.499654620885849, 0.00024397910601992257, 0.008482683629538237),
           (182, 0.4998599290721762, 0.5194861888885498, 0.002192222792202422, 0.007172654920327219),
           (186, 0.4999847869259766, 0.5100514888763428, 0.0007477328460012137, 0.0042112557682644595))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true")
    a = p.parse_args()
    files = sorted(glob.glob("results/X/*_results.json")) + sorted(glob.glob("results/L/*_results.json"))
    text = report(files)
    print(text)
    if a.write:
        os.makedirs("docs", exist_ok=True)
        with open("docs/fail_class_v2.md", "w") as f:
            f.write(text + "\n")
        print("\nwritten: docs/fail_class_v2.md")


if __name__ == "__main__":
    main()
