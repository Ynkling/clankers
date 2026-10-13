#!/usr/bin/env python3
"""registry.py — checks and counts for claims_registry.md (the referee's follow-up, 10 October 2026).

    python3 docs/review_2026-10/registry.py      -> registry_checks.txt (and prints it)

Inputs: registry/main_*.tsv (one row per pre-registered main-line claim and machine) and registry/exploratory_*.tsv,
extracted from the test docstrings and git by delegated read-only audits and checked here:
  1. every printed McNemar p is recomputed from the printed discordant pair, every printed Fisher p from the printed
     counts (one-sided unless the row says two);
  2. every X row (and L row where results/L holds the test's file) is compared with the verdict and p the results file
     itself stores (`verdict[...]`), where it stores one;
  3. for test_slow_start, test_recipe_scope and test_early_recipe, which store no verdict, the counts and discordant
     pairs are recomputed from the per-run records (DISCOVERED / BOUND / bound by 4800);
  4. claims are counted by kind and machine, and the error rates the count supports are computed from the printed p's.
"""
import csv
import glob
import math
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rlib import load, runs, bound, discovered, transition, pairs, mcnemar1, mcnemar2, fisher1  # noqa: E402

out = []


def say(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s)


def read(pattern):
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "registry", pattern))):
        with open(f) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                r["_file"] = os.path.basename(f)
                rows.append(r)
    return rows


MAIN = read("main_*.tsv")
EXPL_E = read("exploratory_E.tsv")
EXPL_FGH = read("exploratory_FGH.tsv")
NUM = r"([0-9]*\.?[0-9]+(?:e-?[0-9]+)?)"


SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def first_p(s):
    s = re.sub(r"\s*[×x·]\s*10\^?([⁻⁰¹²³⁴⁵⁶⁷⁸⁹\-0-9]+)", lambda m: "e" + m.group(1).translate(SUP), s)
    s = s.replace("−", "-")
    m = re.search(r"p\s*[=<]\s*" + NUM, s) or re.match(r"\s*" + NUM, s)
    return float(m.group(1)) if m else None


def close(a, b):
    if a is None or b is None:
        return None
    if a == b:
        return True
    # printed values are rounded to 1-3 significant figures
    return abs(a - b) <= max(0.06 * max(a, b), 0.0006) or (min(a, b) > 0 and abs(math.log10(a / b)) < 0.03)


# ── 1. printed p against the printed pairs / counts ─────────────────────────────────────────────────────────────
say("== 1. Printed p-values recomputed from the printed pairs or counts (main line, kind = test)")
n_chk = n_ok = 0
bad = []
for r in MAIN:
    if r["kind"] != "test":
        continue
    p = first_p(r["p_or_bound"])
    if p is None:
        continue
    tu = r["test_used"].lower()
    two = r["sided"].strip().lower().startswith("two")
    rec = None
    m = re.match(r"\s*(\d+)\s*vs\s*(\d+)", r["discordant"])
    if "mcnemar" in tu and m and not tu.startswith("fisher"):
        b, c = int(m.group(1)), int(m.group(2))
        rec = mcnemar2(b, c) if two else mcnemar1(b, c)
    elif tu.startswith("fisher"):
        fr = re.findall(r"(\d+)/(\d+)", r["counts"])
        if len(fr) >= 2:
            (a, n1), (c, n2) = [(int(x), int(y)) for x, y in fr[:2]]
            up, down = fisher1(a, n1, c, n2), fisher1(c, n2, a, n1)
            if two:
                rec = min(1.0, 2 * min(up, down))
            elif r["claim_id"] == "MG":      # "fewer merges": the second arm higher is the claimed direction
                rec = down
            else:
                rec = up
    if rec is None:
        continue
    n_chk += 1
    ok = close(p, rec)
    n_ok += bool(ok)
    if not ok:
        bad.append((r["test"], r["claim_id"], r["machine"], r["counts"], r["discordant"], r["p_or_bound"], f"{rec:.4g}"))
say(f"checked {n_chk}, agree {n_ok}")
for b_ in bad:
    say("  DIFFERS:", " | ".join(b_))

# ── 2. registry rows against the verdicts the results files store ─────────────────────────────────────────────────
say("\n== 2. Registry rows against the verdict and p stored in results/<machine>/<test>_results.json")
n2 = ok2 = 0
diff2 = []
for r in MAIN:
    if r["machine"] not in ("X", "L"):
        continue
    t = r["test"].replace("test_", "").replace(".py", "")
    path = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results", r["machine"], f"{t}_results.json")
    if not os.path.exists(path):
        continue
    v = load(None, f"results/{r['machine']}/{t}_results.json").get("verdict")
    if not isinstance(v, dict):
        continue
    cid = r["claim_id"].replace("-", "_")
    keys = [cid, cid.upper(), r["claim_id"]]
    sv = next((v[k] for k in keys if k in v and isinstance(v[k], str)), None)
    if sv is None:
        continue
    n2 += 1
    printed = r["verdict_printed"].upper()
    same = sv.upper() in printed and not (sv.upper() == "SHOWN" and "NOT SHOWN" in printed)
    sp = next((v[k + "_p"] for k in keys if k + "_p" in v), None)
    pp = first_p(r["p_or_bound"])
    same_p = True if sp is None or pp is None or (isinstance(sp, float) and math.isnan(sp)) else close(pp, sp)
    if same and same_p:
        ok2 += 1
    else:
        diff2.append((t, r["claim_id"], r["machine"], sv, r["verdict_printed"][:50], sp, r["p_or_bound"][:40]))
say(f"compared {n2}, agree {ok2}")
for d in diff2:
    say("  DIFFERS:", " | ".join(str(x) for x in d))

# ── 3. recount from the per-run records where no verdict is stored ─────────────────────────────────────────────────
say("\n== 3. Counts and discordant pairs recomputed from per-run records (tests that store no verdict)")
FILES = {m: sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(HERE)), "results", m, "*_results.json"))) for m in "XL"}


def arm_runs(m, arm, prefer):
    order = sorted(FILES[m], key=lambda f: (prefer not in f, f))
    for f in order:
        rs = runs(load(None, os.path.relpath(f, os.path.dirname(os.path.dirname(HERE)))), arm)
        if rs:
            return rs, os.path.basename(f)
    return None, None


def measure(text):
    t = text.upper()
    if "4800" in t:
        return lambda r: bound(r) and transition(r["curve"]) <= 4800, "bound by 4800"
    if "DISCOVERED" in t:
        return discovered, "discovered"
    if "8-STREAM SET" in t or "STAGE" in t:
        return None, "staged outcome (not recomputed)"
    return bound, "bound"


n3 = ok3 = 0
for r in MAIN:
    t = r["test"].replace("test_", "").replace(".py", "")
    if t not in ("slow_start", "recipe_scope", "early_recipe") or r["kind"] != "test" or r["machine"] not in "XL":
        continue
    arms = re.findall(r"([A-Z][A-Z0-9_]+)\s+(\d+)/(\d+)", r["counts"])
    f, what = measure(r["outcome_measure"])
    if len(arms) < 2 or f is None:
        say(f"  {t} {r['claim_id']} {r['machine']}: {what}")
        continue
    (a1, c1, n1), (a2, c2, n2_) = arms[:2]
    ra, fa = arm_runs(r["machine"], a1, t)
    rb, fb = arm_runs(r["machine"], a2, t)
    if not ra or not rb:
        say(f"  {t} {r['claim_id']} {r['machine']}: arm records not found ({a1}: {fa}, {a2}: {fb})")
        continue
    seeds = sorted(set(ra) & set(rb))
    ca, cb = sum(f(ra[s]) for s in seeds), sum(f(rb[s]) for s in seeds)
    x, y, _ = pairs(ra, rb, f, seeds)
    m = re.match(r"\s*(\d+)\s*vs\s*(\d+)", r["discordant"])
    good = (ca, cb) == (int(c1), int(c2)) and (not m or (x, y) == (int(m.group(1)), int(m.group(2))))
    n3 += 1
    ok3 += good
    say(f"  {t} {r['claim_id']} {r['machine']} ({what}; {a1} from {fa}, {a2} from {fb}, {len(seeds)} seeds): "
        f"printed {c1}/{n1} vs {c2}/{n2_}, {r['discordant']}; records {ca} vs {cb}, {x} vs {y} -> {'MATCH' if good else 'DIFFERS'}")
say(f"recomputed {n3}, match {ok3}")

# ── 4. counts ─────────────────────────────────────────────────────────────────────────────────────────────────────
say("\n== 4. Counts of main-line pre-registered claims (Phases III-VI)")
kinds = ["test", "bound", "band", "gate/validity", "reading"]
tab = Counter((r["machine"], r["kind"]) for r in MAIN)
for mach in ("X", "L", "pooled"):
    say(f"  {mach:6s} " + ", ".join(f"{k} {tab[(mach, k)]}" for k in kinds) + f"; total {sum(tab[(mach, k)] for k in kinds)}")
by_phase = Counter((r["phase"], r["machine"]) for r in MAIN if r["kind"] == "test")
say("  hypothesis tests by phase and machine: " + ", ".join(f"{p}/{m} {n}" for (p, m), n in sorted(by_phase.items())))
tests_n = Counter(r["test"] for r in MAIN)
say(f"  tests: {len(tests_n)}")


def is_primary(r):
    v = r["verdict_printed"]
    return not v.startswith(("SECONDARY", "NOT RECORDED", "NOT PRINTED"))


rec = Counter((r["machine"], is_primary(r)) for r in MAIN if r["kind"] == "test")
say(f"  hypothesis tests whose verdict is recorded in the test's own docstring or results file: X {rec[('X', True)]}, "
    f"L {rec[('L', True)]} (L's other {rec[('L', False)]} are second-hand or unrecorded)")


def shown(r):
    v = r["verdict_printed"].upper().replace("SECONDARY:", "").strip()
    return v.startswith(("SHOWN", "SUPPORTED", "CONFIRMED", "GRADIENT", "TIMING HELPS", "DIFFERENT")) and "NOT SHOWN" not in v


for mach in ("X", "L", "pooled"):
    T = [r for r in MAIN if r["machine"] == mach and r["kind"] == "test"]
    ps = [(first_p(r["p_or_bound"]), r) for r in T]
    known = sorted([(p, r) for p, r in ps if p is not None and not math.isnan(p)], key=lambda t: t[0])
    sh = [r for r in T if shown(r)]
    m_ = len(T)
    holm = []
    for i, (p, r) in enumerate(known):
        if p <= 0.05 / (m_ - i):
            holm.append((p, r))
        else:
            break
    bh_k = max([i + 1 for i, (p, r) in enumerate(known) if p <= 0.05 * (i + 1) / m_] or [0])
    bonf = [(p, r) for p, r in known if p <= 0.05 / m_]
    say(f"\n  {mach}: {m_} hypothesis tests, {len(known)} with a printed p; positive verdicts as printed: {len(sh)}")
    say(f"    expected false positives if every null were true: {0.05 * m_:.2f}")
    say(f"    survive Bonferroni at FWER 0.05 (p <= {0.05 / m_:.5f}): {len(bonf)}: " +
        ", ".join(f"{r['test'][5:-3]}:{r['claim_id']} ({p:.2g})" for p, r in bonf))
    say(f"    survive Holm at FWER 0.05: {len(holm)}")
    say(f"    survive Benjamini-Hochberg at FDR 0.05: {bh_k}: " +
        ", ".join(f"{r['test'][5:-3]}:{r['claim_id']} ({p:.2g})" for p, r in known[:bh_k]))
    lost = [r for r in sh if all(r is not rr for _, rr in known[:bh_k])]
    say(f"    printed positive but not kept by BH: " + ", ".join(f"{r['test'][5:-3]}:{r['claim_id']} ({r['p_or_bound'][:18]})" for r in lost))

say("\n== 5. Exploratory readings")
say(f"  session E: {len(EXPL_E)} table rows with a reading, {sum(1 for r in EXPL_E if first_p(r['p']) is not None)} with a p")
bad_e = []
for r in EXPL_E:
    m = re.match(r"\s*(\d+)\s*/\s*(\d+)", r["discordant"])
    p = first_p(r["p"])
    if m and p is not None:
        b, c = int(m.group(1)), int(m.group(2))
        rec_ = mcnemar2(b, c) if "two" in r["sided"].lower() else mcnemar1(b, c)
        if not close(p, rec_):
            bad_e.append((r["batch"], r["screen"], r["arm"], r["discordant"], r["p"], f"{rec_:.4g}"))
say(f"  E's printed p recomputed from its pairs: {len(bad_e)} differ " + str(bad_e[:5]))
fgh = Counter(r["branch"].split("-")[-1] for r in EXPL_FGH)
say(f"  sessions F, G, H: {len(EXPL_FGH)} readings ({dict(fgh)})")

with open(os.path.join(HERE, "registry_checks.txt"), "w") as fh:
    fh.write("\n".join(out) + "\n")


# ── 6. claims_registry.md ─────────────────────────────────────────────────────────────────────────────────────────
def esc(s, n=None):
    s = (s or "").replace("|", "\\|").replace("\n", " ").strip()
    return s if n is None or len(s) <= n else s[: n - 1] + "…"


def commit(c, d):
    c = (c or "").strip()
    return "UNKNOWN" if not c or c == "UNKNOWN" else f"`{c}` {(d or '')[:10]}"


SUMMARY = open(os.path.join(HERE, "registry", "summary.md")).read()
md = [SUMMARY, "", "## Table 1. Main-line pre-registered claims, Phases III–VI (one row per claim and machine)", "",
      "*Columns as requested. `kind`: test = a hypothesis test with a p-value; bound = a margin rule without one; band, "
      "gate/validity, reading as named in the docstrings. Verdicts beginning `SECONDARY:` are quoted from a later "
      "document (the next test's BACKGROUND or `docs/revision6.md` / `revision7.md`), not from the test's own record; "
      "`NOT RECORDED` / `NOT PRINTED` mean no verdict was ever written. Statements are abridged; the full text, the "
      "outcome measure's details and the docstring line are in `registry/main_*.tsv`.*", "",
      "| # | test | claim | mach. | kind | statement | outcome | test used | discordant | p or bound | verdict as printed | fixed | recorded |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
order = {"III": 0, "IV": 1, "V": 2, "VI": 3}
mo = {"X": 0, "L": 1, "pooled": 2}
srt = sorted(MAIN, key=lambda r: (order.get(r["phase"], 9), r["test"], {"gate/validity": 0, "test": 1, "bound": 2, "band": 3,
                                                                         "reading": 4}.get(r["kind"], 5), r["claim_id"], mo.get(r["machine"], 3)))
for i, r in enumerate(srt, 1):
    md.append(f"| {i} | {r['test'][5:-3]} | {esc(r['claim_id'])} | {r['machine']} | {r['kind']} | {esc(r['statement'], 140)} | "
              f"{esc(r['outcome_measure'], 50)} | {esc(r['test_used'] + (' (' + r['sided'] + ')' if r['sided'] not in ('NA', '') else ''), 40)} | "
              f"{esc(r['discordant'], 40)} | {esc(r['p_or_bound'], 40)} | {esc(r['verdict_printed'], 70)} | "
              f"{commit(r['fixed_commit'], r['fixed_date'])} | {commit(r['recorded_commit'], r['recorded_date'])} |")
md += ["", "## Table 2. EXPLORATORY readings (not claims; screens on side branches)", "",
       "*Session E (claude/outside-ideas), one row per verdict-table row that carries a reading, from "
       "`explore_out/README.md` at `5dfc2b5`. Every p is an exact two-sided McNemar on the discordant pair, recomputed here. "
       "Full rows in `registry/exploratory_E.tsv`.*", "",
       "| # | batch | screen | arm | outcome | counts | reference | discordant | p | reading or verdict | readings fixed | verdict recorded |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for i, r in enumerate(EXPL_E, 1):
    rc, vc = (r["readings_commit"] or "").split(" ", 1) + [""], (r["verdict_commit"] or "").split(" ", 1) + [""]
    md.append(f"| E{i} | {r['batch']} | {esc(r['screen'], 30)} | {esc(r['arm'], 30)} | {esc(r['outcome_measure'], 20)} | "
              f"{esc(r['counts'], 30)} | {esc(r['reference'], 30)} | {esc(r['discordant'])} | {esc(r['p'])} | "
              f"{esc(r['reading_or_verdict'], 70)} | {commit(rc[0], rc[1])} | {commit(vc[0], vc[1])} |")
md += ["", "*Sessions F, G and H (claude/explore-F, -G, -H), one row per pre-fixed reading in the screen's docstring. "
       "Full rows, with the rule's text, in `registry/exploratory_FGH.tsv`; readings stated in reports but never "
       "pre-registered are listed in `registry/notes_EXPL_FGH.md`.*", "",
       "| # | branch | screen | reading | rule (abridged) | counts | discordant | p | outcome as reported | readings fixed | reported |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
for i, r in enumerate(EXPL_FGH, 1):
    rc, vc = (r["readings_commit"] or "").split(" ", 1) + [""], (r["report_commit"] or "").split(" ", 1) + [""]
    md.append(f"| F{i} | {r['branch'].replace('claude/', '')} | {esc(r['screen'], 20)} | {esc(r['reading_id'] + ' ' + r['reading_text'], 60)} | "
              f"{esc(r['rule'], 110)} | {esc(r['counts'], 40)} | {esc(r['discordant'], 25)} | {esc(r['p'], 25)} | "
              f"{esc(r['outcome_as_reported'], 60)} | {commit(rc[0], rc[1])} | {commit(vc[0], vc[1])} |")
with open(os.path.join(HERE, "claims_registry.md"), "w") as fh:
    fh.write("\n".join(md) + "\n")
print(f"wrote claims_registry.md: {len(srt)} main-line rows, {len(EXPL_E)} + {len(EXPL_FGH)} exploratory rows")
