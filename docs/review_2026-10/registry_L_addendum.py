#!/usr/bin/env python3
"""registry_L_addendum.py: the registry's L rows (Phases III-IV) against the L results files that reached the
main line after claims_registry.md was written (9a53302, 9c53e16; merged into review-V at 59b6773).

    python3 docs/review_2026-10/registry_L_addendum.py      (stdlib; writes registry_L_addendum_out.txt)

For every L row of registry/main_*.tsv whose test now has results/L/<test>_results.json:
  the file's CPU and commit; the verdict and p the file stores for the claim (if it stores one);
  the registry's verdict as printed (SECONDARY / NOT RECORDED / primary); and whether they agree.
"""
import csv
import glob
import json
import math
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
out = []


def say(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s)


rows = []
for p in sorted(glob.glob(os.path.join(HERE, "registry", "main_*.tsv"))):
    with open(p) as f:
        rows += [r for r in csv.DictReader(f, delimiter="\t")]


def added_by(path):
    return subprocess.check_output(["git", "-C", ROOT, "log", "--diff-filter=A", "--format=%h %cs", "-1", "--", path]
                                   ).decode().strip()


def lookup(v, cid):
    keys = [cid.replace("-", "_"), cid.replace("-", "_").upper(), cid]
    sv = next((v[k] for k in keys if k in v and isinstance(v[k], str)), None)
    sp = next((v[k + "_p"] for k in keys if k + "_p" in v), None)
    return sv, sp


tests = sorted({r["test"] for r in rows if r["machine"] == "L" and r["phase"] in ("III", "IV")})
tally = {}
for t in tests:
    name = t.replace("test_", "").replace(".py", "")
    rel = f"results/L/{name}_results.json"
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        say(f"\n{t}: no {rel}")
        continue
    d = json.load(open(path))
    meta = d.get("meta", {})
    cpu = meta.get("cpu") or next((r.get("cpu") for r in d.values() if isinstance(r, dict) and r.get("cpu")), "?")
    v = d.get("verdict")
    say(f"\n{t}: {rel} added {added_by(rel)}; CPU {str(cpu)[:45]}; stored verdict: "
        f"{'none' if v is None else ('dict' if isinstance(v, dict) else v)}")
    for r in rows:
        if r["test"] != t or r["machine"] != "L":
            continue
        printed = r["verdict_printed"]
        status = ("SECONDARY" if printed.upper().startswith("SECONDARY") else
                  "NOT RECORDED" if printed.upper().startswith(("NOT RECORDED", "NOT PRINTED")) else "PRIMARY")
        sv, sp = lookup(v, r["claim_id"]) if isinstance(v, dict) else (None, None)
        if sv is None:
            agree = "no stored verdict for this claim"
        else:
            body = printed.upper().replace("SECONDARY:", "")
            agree = "AGREE" if (sv.upper() in body and not (sv.upper() == "SHOWN" and "NOT SHOWN" in body)) else \
                ("NEW (registry had none)" if status == "NOT RECORDED" else "DIFFERS")
        sp_s = "" if sp is None or (isinstance(sp, float) and math.isnan(sp)) else f" p={sp:.4g}"
        tally[(status, agree.split(" ")[0])] = tally.get((status, agree.split(" ")[0]), 0) + 1
        say(f"  {r['claim_id']:8s} [{r['kind']}] registry: {status:12s} {printed[:60]!r} | file: {sv}{sp_s} -> {agree}")

say("\n== tally (registry status, comparison)")
for k in sorted(tally):
    say(f"  {k[0]:12s} {k[1]:8s} {tally[k]}")

import re  # noqa: E402


SUP = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")
NUM = r"([0-9]*\.?[0-9]+(?:[eE]-?[0-9]+)?)"


def first_p(s):  # registry.py's parser
    s = re.sub(r"\s*[×x·]\s*10\^?([⁻⁰¹²³⁴⁵⁶⁷⁸⁹\-0-9]+)", lambda m: "e" + m.group(1).translate(SUP), s or "")
    s = s.replace("−", "-")
    m = re.search(r"p\s*[=<]\s*" + NUM, s) or re.match(r"\s*" + NUM, s)
    return float(m.group(1)) if m else None


say("\n== L's 53 hypothesis tests: p from the L results file where it stores one, else as printed")
T = [r for r in rows if r["machine"] == "L" and r["kind"] == "test"]
ps = []
src = {"file": 0, "printed": 0, "none": 0}
for r in T:
    name = r["test"].replace("test_", "").replace(".py", "")
    path = os.path.join(ROOT, f"results/L/{name}_results.json")
    sp = None
    if os.path.exists(path):
        v = json.load(open(path)).get("verdict")
        if isinstance(v, dict):
            sp = lookup(v, r["claim_id"])[1]
            if isinstance(sp, float) and math.isnan(sp):
                sp = None
    if sp is not None:
        src["file"] += 1
    else:
        sp = first_p(r["p_or_bound"])
        src["printed" if sp is not None else "none"] += 1
    ps.append((sp, r))
say(f"  tests {len(T)}; p from file {src['file']}, from the printed text only {src['printed']}, none {src['none']}")
say("  tests with no p anywhere: " + ", ".join(f"{r['test'][5:-3]}:{r['claim_id']}" for p, r in ps if p is None))
known = sorted([(p, r) for p, r in ps if p is not None], key=lambda t: t[0])
m_ = len(T)
holm = 0
for i, (p, r) in enumerate(known):
    if p <= 0.05 / (m_ - i):
        holm += 1
    else:
        break
bh_k = max([i + 1 for i, (p, r) in enumerate(known) if p <= 0.05 * (i + 1) / m_] or [0])
say(f"  with a p: {len(known)}; p < 0.05: {sum(1 for p, _ in known if p < 0.05)}; Holm FWER 0.05 keeps {holm}; "
    f"BH FDR 0.05 keeps {bh_k}")



def shown(r):
    v = r["verdict_printed"].upper().replace("SECONDARY:", "").strip()
    return v.startswith(("SHOWN", "SUPPORTED", "CONFIRMED", "GRADIENT", "TIMING HELPS", "DIFFERENT")) and "NOT SHOWN" not in v


kept = {id(r) for _, r in known[:bh_k]}
pos = [(p, r) for p, r in ps if shown(r)]
say(f"  positive verdicts as printed: {len(pos)}; not kept by BH: " +
    ", ".join(f"{r['test'][5:-3]}:{r['claim_id']} ({'-' if p is None else f'{p:.3g}'})" for p, r in pos if id(r) not in kept))
say("  p now taken from the file for rows the registry had without a printed p: " +
    ", ".join(f"{r['test'][5:-3]}:{r['claim_id']} ({p:.3g})" for p, r in ps
              if p is not None and first_p(r["p_or_bound"]) is None))

with open(os.path.join(HERE, "registry_L_addendum_out.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
