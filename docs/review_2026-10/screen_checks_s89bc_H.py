#!/usr/bin/env python3
"""screen_checks_s89bc_H.py: the referee's checks behind screen_review_s89bc_H.md (Session H's commits after 13fe610 on
claude/explore-H: 2a98deff (S89 fixes), c81398ca (S89c), bccb7cbd (S89b), 555d6816 / cf67855d (S89c CHECK log)).

    python3 docs/review_2026-10/screen_checks_s89bc_H.py   (stdlib + numpy; reads git only; trains nothing; imports no
                                                             torch; writes screen_checks_s89bc_H_out.txt beside itself)

0. State: H's commits after 13fe610 with times; the S89c store and log at the head; any S89b output on any head; the
   referee's commits these cite or that bear on the order.
1. 2a98deff: files changed; S89's records and the S89 child's code-SHA modules unchanged; the 'four runs' text; the
   commits cited in the REVISION note; explore_h_s89.report() at 2a98deff exec'd from source (store loaders stubbed) on
   S89's real store and on synthetic outcomes (every predicate pattern of the rules), so that 'moot' is seen to print
   no name.
2. S89c: report() at the head exec'd on every composition of 10 seeds into the 8 per-seed patterns of (BASE,
   BASE_LATE, BASE_CAP6); compared with H's explore_h_oc.rule_s89c (cut out by ast) and with the referee's reading of
   the docstring; the CHECK's record and its first commit; S89's curves at 9600 / 12000 (what the 12000 choice saw).
3. S89b: report() at the head exec'd on every composition of 10 seeds into the 8 per-seed patterns of (DELTA_SPLIT,
   DELTA_NOSPLIT, DELTA_ALLMASS) and four ORACLE_DELTA states; compared with rule_s89b and the referee's reading;
   s89c_schedule's choice rule on synthetic S89c comparisons; X's delta records at 8ae660d.
4. Seeds 2912-2959: every JSON / JSONL blob on every remote head and in all history; text mentions on the heads.
5. Operating characteristics, the referee's own exact enumeration (V's model), against H's docstring tables, plus the
   gate and naming probabilities at the rates S89 observed.
"""
import ast
import collections
import itertools
import json
import math
import os
import re
import subprocess
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
out = []


def say(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s, flush=True)


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a)


def show(ref, path):
    return git("show", f"{ref}:{path}").decode()


def exists(ref, path):
    return subprocess.run(["git", "-C", ROOT, "cat-file", "-e", f"{ref}:{path}"], capture_output=True).returncode == 0


def blob(ref, path):
    try:
        return git("rev-parse", f"{ref}:{path}").decode().strip()
    except subprocess.CalledProcessError:
        return None


HB = "origin/claude/explore-H"
HEAD = git("rev-parse", "--short=8", HB).decode().strip()
BASE13, FIX, S89C, S89B, LOGC, CF = "13fe6100", "2a98deff", "c81398ca", "bccb7cbd", "555d6816", "cf67855d"

# ══ 0. state ═════════════════════════════════════════════════════════════════
say("== 0. State")
say(f"  explore-H head {HEAD}; commits after {BASE13}:")
for ln in git("log", "--reverse", "--format=%h %cI %s", f"{BASE13}..{HB}").decode().splitlines():
    say("    " + ln[:150])
for c in ("ee5217b", "adecd3db", "3c772c3", "dcd549d"):
    say(f"  review-V {git('log', '-1', '--format=%h %cI %s', c).decode().strip()[:150]}")
say("  files changed per commit:")
for c in (FIX, S89C, S89B, LOGC, CF):
    say(f"    {c}: {git('show', '--format=', '--name-only', c).decode().split()}")
lc = show(HB, "explore_out/H/s89c.log").splitlines()
say(f"  s89c.log at the head: {len(lc)} lines; CHECK rows {sum('CHECK H/s89c' in x for x in lc)} "
    f"(ok {sum('CHECK H/s89c' in x and x.rstrip().endswith(': ok') for x in lc)}); last: {lc[-1].strip()[:120]}")
st = json.loads(show(HB, "explore_out/H/s89c_results.json"))
say(f"  s89c_results.json at the head: runs stored {len(st['runs'])}; code_shas {st['meta']['code_shas']}; "
    f"started {st['meta']['provenance']['started']} (git {st['meta']['provenance']['git']})")
ck = json.loads(show(HB, "explore_out/H/checks.json"))
say(f"  checks.json: H/s89c {ck.get('H/s89c')}; H/s89b present: {'H/s89b' in ck}")
heads = [b for b in git("for-each-ref", "refs/remotes/origin", "--format=%(refname:short)").decode().split()
         if not b.endswith("/HEAD")]
s89b_files = [(b, p) for b in heads for p in git("ls-tree", "-r", "--name-only", b).decode().split() if "s89b" in p.lower()]
say(f"  files named *s89b* on any head: {s89b_files}")

# ══ 1. 2a98deff ══════════════════════════════════════════════════════════════
say(f"\n== 1. {FIX}: S89's fixes")
for p in ("explore_out/H/s89_results.json", "explore_out/H/s89.log", "explore_out/H/s89_oracle.log",
          "explore_out/H/s89_runs.md"):
    say(f"  {p}: blob at {BASE13} {str(blob(BASE13, p))[:10]}, at {FIX} {str(blob(FIX, p))[:10]}, at head "
        f"{str(blob(HB, p))[:10]}")
say(f"  checks.json H/s89 entry equal at {BASE13} and head: "
    f"{json.loads(show(BASE13, 'explore_out/H/checks.json'))['H/s89'] == ck['H/s89']}")
sha_line = next(x for x in show(BASE13, "explore_out/H/s89_oracle.log").splitlines() if "code SHA H/s89" in x)
mods = sha_line.split("):", 1)[1].split()
on_branch = [m for m in mods if exists(BASE13, m + ".py")]
changed = [m for m in on_branch if blob(BASE13, m + ".py") != blob(HB, m + ".py")]
say(f"  S89's code SHA covers {len(mods)} modules; {len(on_branch)} are files on the branch; changed between {BASE13} "
    f"and the head: {changed or 'none'}; explore_h_s89 / explore_h_report in the list: "
    f"{'explore_h_s89' in mods} / {'explore_h_report' in mods}")
for p in ("explore_out/H/report_10.md", "explore_out/H/report_10_notes.md", "explore_out/README.md"):
    for ref in (BASE13, HB):
        hits = [f"{i}: {x.strip()[:150]}" for i, x in enumerate(show(ref, p).splitlines(), 1)
                if re.search(r"\bfour (bound|runs)\b|three of the four", x, re.I)]
        say(f"  'four' (runs) in {p} at {ref[:12]}: {hits or 'none'}")
src_fix = show(FIX, "explore_h_s89.py")
for nm, pat in (("fix1 s89:3-5", "rare key types"), ("fix1 CAVEAT", "CAVEAT (printed"), ("fix2", "Referee V's fix 2"),
                ("fix3", "Referee V's fix 3"), ("fix4", "OPERATING CHARACTERISTICS"), ("fix5", "fix 5"),
                ("fix6 MAINREF", 'MAINREF = "628eb7c"'), ("fix7", "50dc8e4"), ("fix8", "fix 8"), ("REVISION", "REVISION"),
                ("cites 3c772c3", "3c772c3")):
    say(f"  {nm}: lines {[i for i, x in enumerate(src_fix.splitlines(), 1) if pat in x]}")
say(f"  628eb7c has results/X/window_gate_results.json with WIN3_SPLIT_D8|540: "
    f"{'WIN3_SPLIT_D8|540' in json.loads(show('628eb7c', 'results/X/window_gate_results.json'))['runs']}")
f_tasks = show("52e2ea7", "explore_f_tasks.py")
CLS = re.compile(r"^class (\w+)", re.M)
say(f"  F's explore_f_tasks.py at 52e2ea7: classes {CLS.findall(f_tasks)}")
f_head = git("rev-parse", "--short=8", "origin/claude/explore-F").decode().strip()
say(f"  ... at F's head {f_head}: classes {CLS.findall(show(f_head, 'explore_f_tasks.py'))}")

# --- exec report() ---
SYN, STORE = {}, {"meta": {}, "runs": {}}


def _stub(name, **kw):
    m = types.ModuleType(name)
    m.__dict__.update(kw)
    sys.modules[name] = m
    return m


def _med(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = len(xs)
    return xs[k // 2] if k % 2 else 0.5 * (xs[k // 2 - 1] + xs[k // 2])


_stub("explore_common", load_store=lambda name: STORE, cpu_model=lambda: "stub", OUT_DIR="/nonexistent")
_stub("explore_common16", runs_of=lambda screen, arm, seeds=None, store=None: dict(SYN.get(arm, {})), med=_med, STATE={},
      code_sha=lambda *a: ("stub", []), key=lambda *a: "stub", run_child=lambda *a, **k: None)
_stub("explore_muon_scale")
_stub("explore_h_collapse")
_stub("explore_h_allmass2", NAME="H/allmass2")


def load(name, ref, path=None):
    m = types.ModuleType(name)
    m.__file__ = os.path.join(HERE, "_nonexistent_", name + ".py")
    sys.modules[name] = m
    exec(compile(show(ref, path or name + ".py"), f"{name}.py@{ref}", "exec"), m.__dict__)
    return m


hc = load("explore_h_common", HB)
s89 = load("explore_h_s89", FIX)
assert blob(FIX, "explore_h_s89.py") == blob(HB, "explore_h_s89.py")


def rec(ok, fired=None, acc=0.5):
    return dict(ok=True, outcome="BOUND ROUTED" if ok else "MERGED (2 share)", outcome_v2="BOUND ROUTED" if ok else
                "MERGED (2 share)", transition=12000 if ok else None, acc=acc, checks=fired or [])


# (a) S89's real store
real = json.loads(show(HB, "explore_out/H/s89_results.json"))
SYN.clear()
for k, r in real["runs"].items():
    SYN.setdefault(k.split("|")[0], {})[r["seed"]] = r
res, _ = s89.report()
say(f"  report() at {FIX} on S89's 42 stored records: readings {res['readings']}")
say(f"    trigger_lines {list(res['trigger_lines'].values())}")
say(f"    repeat_note {res.get('repeat_note')}")


# (b) synthetic outcomes over the 4 learned arms (16 patterns), one representative per predicate pattern
def compositions(n, k):
    bars = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(n + k - 1), k - 1)),
                       dtype=np.int16).reshape(-1, k - 1)
    ext = np.concatenate([np.full((bars.shape[0], 1), -1, np.int16), bars,
                          np.full((bars.shape[0], 1), n + k - 1, np.int16)], 1)
    return (np.diff(ext, axis=1) - 1).astype(np.int16)


LEARN = ("BASE", "ALLMASS", "REPEATMASS", "NOSPLIT")
comp = compositions(10, 16)
bit = {a: ((np.arange(16) >> i) & 1).astype(bool) for i, a in enumerate(LEARN)}


def onl(c, x, y):
    return c[:, bit[x] & ~bit[y]].sum(1)


nB = comp[:, bit["BASE"]].sum(1)
NBb, NBc = onl(comp, "NOSPLIT", "BASE"), onl(comp, "BASE", "NOSPLIT")
coarse = np.stack([nB < 7, NBc - NBb <= 1] + [z for t in ("ALLMASS", "REPEATMASS") for z in (
    onl(comp, "BASE", t) - onl(comp, t, "BASE") <= 1, onl(comp, t, "NOSPLIT") >= 4, onl(comp, "NOSPLIT", t) == 0,
    onl(comp, "BASE", t) >= 4, onl(comp, t, "BASE") == 0)], 1)
_, cfirst = np.unique(coarse, axis=0, return_index=True)
rng = np.random.default_rng(89)
pick = np.unique(np.concatenate([cfirst, rng.choice(comp.shape[0], 5000, replace=False)]))
FC = dict(fired=True, acc=0.3, map_before=[0, 0, 1, 2, 3, 4, 5, 6], op_ok=True, key_ok=True, all_ok=True, rep_ok=True,
          key_cs=1, key_c0=7, all_cs=1, all_c0=7, rep_cs=1, rep_c0=7, step=4800)
NAMES = ("survives", "needs key positions", "no reading applies")
bad, moot_named, n_moot, n_calls, named_once = [], 0, 0, 0, 0
for i in pick:
    pats = np.repeat(np.arange(16), comp[i])
    for withf in (False, True):
        SYN.clear()
        SYN["ORACLE"] = {2900: rec(True), 2901: rec(True)}
        for a in LEARN:
            SYN[a] = {s: rec(bool((p >> LEARN.index(a)) & 1), [dict(FC)] if (withf and a != "NOSPLIT") else None)
                      for s, p in zip(s89.SEEDS, pats)}
        if withf:
            for r in SYN["REPEATMASS"].values():
                r["overlap"] = dict(repeat=6005, positions=6272)
        r_, _ = s89.report()
        n_calls += 1
        rd = r_["readings"]
        if "split" in rd and rd["split"].startswith("the split is not needed"):
            n_moot += 1
            for t in ("ALLMASS", "REPEATMASS"):
                if any(nm in rd[t] for nm in NAMES):
                    moot_named += 1
                    bad.append(rd[t][:160])
        elif "split" in rd:
            for t in ("ALLMASS", "REPEATMASS"):
                k = sum(nm in rd[t] for nm in NAMES)
                named_once += k == 1
                if k != 1:
                    bad.append(rd[t][:160])
say(f"  report() at {FIX} on synthetic outcomes: {len(pick)} compositions of 10 seeds into 16 patterns (one per pattern "
    f"of the rules' predicates, {len(cfirst)}, plus 5000 random) x (no firings / one firing per triggered run): "
    f"{n_calls} calls; moot {n_moot}, of which a trigger reading carries a name: {moot_named}; non-moot trigger readings "
    f"with exactly one name: {named_once}; defects: {bad[:3] or 'none'}")
SYN.clear()
SYN["ORACLE"] = {2900: rec(True), 2901: rec(True)}
pats = [0b1111] * 6 + [0b1001, 0b0001, 0b0110, 0b0110]     # BASE 8, NOSPLIT 7: moot; ALLMASS/REPEATMASS 8
for a in LEARN:
    SYN[a] = {s: rec(bool((p >> LEARN.index(a)) & 1), [dict(FC)] if a != "NOSPLIT" else None) for s, p in zip(s89.SEEDS, pats)}
for r in SYN["REPEATMASS"].values():
    r["overlap"] = dict(repeat=6005, positions=6272)
r_, _ = s89.report()
say(f"  example moot print: {r_['readings']}")

# (c) report 10's new descriptive paragraph (2a98deff), recomputed from the store
def _acc(r, t):
    return next((c[1] for c in r["curve"] if c[0] == t), None)


R10 = {(k.split("|")[0], r["seed"]): r for k, r in real["runs"].items()}
orc_ = [R10[("ORACLE", s)] for s in (2900, 2901)]
say("  report 10 (2a98deff) 'What happened', recomputed:")
say(f"    ORACLE held-out at 1200 {[round(_acc(r, 1200), 3) for r in orc_]}, 4800 {[round(_acc(r, 4800), 3) for r in orc_]}, "
    f"8400 {[round(_acc(r, 8400), 3) for r in orc_]}, 9600 {[round(_acc(r, 9600), 3) for r in orc_]}; transitions "
    f"{[r['transition'] for r in orc_]}")
for arm in ("BASE", "ALLMASS"):
    rs_ = {s: R10[(arm, s)] for s in s89.SEEDS}
    fails = {s: r for s, r in rs_.items() if r["outcome"] != "BOUND ROUTED"}
    f05 = sorted(next((c[0] for c in r["curve"] if c[1] >= 0.5), -1) for r in rs_.values())
    below = sorted(sum(a < 0.9 for a in r["end"]["stream_acc"]) for r in fails.values())
    lastmax = sum(r["curve"][-1][1] >= max(c[1] for c in r["curve"]) for r in fails.values())
    say(f"    {arm}: failures {len(fails)}; final held-out {min(r['acc'] for r in fails.values()):.3f}-"
        f"{max(r['acc'] for r in fails.values()):.3f} (curve end {min(r['curve'][-1][1] for r in fails.values()):.3f}-"
        f"{max(r['curve'][-1][1] for r in fails.values()):.3f}); streams < 0.9 per failure {below}; classes "
        f"{dict(collections.Counter(r['outcome'] for r in fails.values()))}; last evaluation = curve max in {lastmax}; "
        f"first >= 0.5 at {f05}; bound {sorted((s, r['transition']) for s, r in rs_.items() if r['outcome'] == 'BOUND ROUTED')}")
r2910 = R10[("BASE", 2910)]
say(f"    BASE 2910 first >= 0.8 at {next((c[0] for c in r2910['curve'] if c[1] >= 0.8), None)}; ALLMASS 2910 "
    f"{R10[('ALLMASS', 2910)]['outcome']}, end {R10[('ALLMASS', 2910)]['acc']:.3f}, streams < 0.9 "
    f"{sum(a < 0.9 for a in R10[('ALLMASS', 2910)]['end']['stream_acc'])}/8")

# ══ 2. S89c ═════════════════════════════════════════════════════════════════
say(f"\n== 2. S89c (explore_h_s89c.py at {S89C}; identical at the head: "
    f"{blob(S89C, 'explore_h_s89c.py') == blob(HB, 'explore_h_s89c.py')}, child "
    f"{blob(S89C, 'explore_h_s89c_child.py') == blob(HB, 'explore_h_s89c_child.py')})")
s89c = load("explore_h_s89c", HB)
ocsrc = show(HB, "explore_h_oc.py")
ocm = types.ModuleType("explore_h_oc_ast")
tree = ast.parse(ocsrc)
tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.FunctionDef))]
exec(compile(tree, "explore_h_oc.py@head(ast: imports and defs only)", "exec"), ocm.__dict__)
A3 = ("BASE", "BASE_LATE", "BASE_CAP6")
comp8 = compositions(10, 8)
say(f"  {comp8.shape[0]} compositions of 10 seeds into the 8 per-seed patterns of {A3}")


def v_s89c(n, only):
    """The referee's reading of S89c's docstring (lines 21-24)."""
    if any(only[(v, 0)] >= 4 and only[(0, v)] == 0 for v in (1, 2)):
        return "the schedule was the limit"
    if all(n[v] < 4 for v in (1, 2)):
        return "the channel is the limit"
    return "no reading applies"


def stats3(c):
    pats = list(itertools.product((0, 1), repeat=3))
    cnt = dict(zip(pats, c))
    n = [sum(v for p, v in cnt.items() if p[i]) for i in range(3)]
    only = {(i, j): sum(v for p, v in cnt.items() if p[i] and not p[j]) for i in range(3) for j in range(3) if i != j}
    return cnt, n, only


PAT3 = list(itertools.product((0, 1), repeat=3))
tally, dis, chan_base = collections.Counter(), [], collections.Counter()
for c in comp8:
    cnt, n, only = stats3(c)
    seq = [p for p, k in zip(PAT3, c) for _ in range(k)]
    SYN.clear()
    for i, a in enumerate(A3):
        SYN[a] = {s: rec(bool(p[i])) for s, p in zip(s89c.SEEDS, seq)}
    r_, _ = s89c.report()
    txt = r_["readings"]["S89c"]
    names = [nm for nm in ("the schedule was the limit", "the channel is the limit", "no reading applies") if txt.startswith(nm)]
    hname = names[0] if len(names) == 1 else f"?{names}"
    tally[hname] += 1
    if hname != ocm.rule_s89c(cnt) or hname != v_s89c(n, only):
        dis.append((tuple(c), hname, ocm.rule_s89c(cnt), v_s89c(n, only)))
    if hname == "the channel is the limit":
        chan_base[n[0]] += 1
say(f"  report() names: {dict(tally)}; disagreements with rule_s89c or the referee's reading: {len(dis)} {dis[:2]}")
say(f"  'the channel is the limit' by BASE's own count (count: compositions): {dict(sorted(chan_base.items()))}; "
    f"with BASE >= 4: {sum(v for k, v in chan_base.items() if k >= 4)}, with BASE >= 7: "
    f"{sum(v for k, v in chan_base.items() if k >= 7)}")
SYN.clear()
SYN.update({a: {s: rec(False) for s in s89c.SEEDS[:9]} for a in A3})
r_, _ = s89c.report()
say(f"  9 of 10 seeds stored: {r_['readings']}; valid {r_['valid']} (hard-coded)")
say(f"  printed by explore_h_run.py at the end: readings, complete, valid only "
    f"({[i for i, x in enumerate(show(HB, 'explore_h_run.py').splitlines(), 1) if 'readings {res' in x]}); "
    f"explore_h_report.py has an s89c / s89b section: "
    f"{'s89c' in show(HB, 'explore_h_report.py') or 's89b' in show(HB, 'explore_h_report.py')}")
# the CHECK's records
first_has = None
for c in git("log", "--reverse", "--format=%h", HB, "--", "explore_out/H/s89_results.json").decode().split():
    d = json.loads(show(c, "explore_out/H/s89_results.json"))
    if any(k.startswith("BASE|2902|") for k in d["runs"]) and first_has is None:
        first_has = (c, git("log", "-1", "--format=%cI %s", c).decode().strip()[:90])
        kb = next(k for k in d["runs"] if k.startswith("BASE|2902|"))
        same = d["runs"][kb] == real["runs"][kb]
first_n = None
for c in git("log", "--reverse", "--format=%h", HB, "--", "explore_out/H/s89_results.json").decode().split():
    d = json.loads(show(c, "explore_out/H/s89_results.json"))
    if any(k.startswith("NOSPLIT|2902|") for k in d["runs"]):
        kn = next(k for k in d["runs"] if k.startswith("NOSPLIT|2902|"))
        first_n = (c, d["runs"][kn] == real["runs"][kn])
        break
say(f"  S89's BASE|2902 first committed at {first_has}; identical at the head: {same}; key {kb}")
say(f"  S89's NOSPLIT|2902 first committed at {first_n[0]}; identical at the head: {first_n[1]}")
say("  s89.store_run (what the CHECK reads):\n" + "\n".join("    " + x for x in
    ("def store_run" + src_fix.split("def store_run", 1)[1].split("\ndef ", 1)[0]).splitlines()))
say(f"  child checks() seed (mechanics on S48's harness, pool 4): "
    f"{re.findall(r'seed = ([0-9]+)', show(HB, 'explore_h_s89c_child.py'))}")


# S89's records: what the choice of 12000 saw
def acc_at(r, t):
    return next((c[1] for c in r["curve"] if c[0] == t), None)


def first_at(r, thr):
    return next((c[0] for c in r["curve"] if c[1] >= thr), None)


runs = {(k.split("|")[0], r["seed"]): r for k, r in real["runs"].items()}
for arm in ("BASE", "NOSPLIT"):
    rs = [runs[(arm, s)] for s in s89.SEEDS]
    say(f"  S89 {arm}: held-out acc at 9600 {[round(acc_at(r, 9600), 3) for r in rs]}; at 12000 "
        f"{[round(acc_at(r, 12000), 3) for r in rs]}; first >= 0.3 at {sorted(first_at(r, 0.3) or -1 for r in rs)}")
ns = [runs[("NOSPLIT", s)] for s in s89.SEEDS]
flat12 = sum(acc_at(r, 12000) - acc_at(r, 9600) < 0.02 for r in ns)
say(f"  S89 NOSPLIT (= BASE_LATE through 12000 by the CHECK's construction): held-out rise 9600 -> 12000 < 0.02 on "
    f"{flat12}/10 seeds (held-out curve, not the 64-sequence probe)")
fb = [c for s in s89.SEEDS for c in runs[("BASE", s)]["checks"] if c.get("fired")]
say(f"  S89 BASE firings: {len(fb)}; before 12000 {sum(c['step'] < 12000 for c in fb)}; at probe < 0.2 "
    f"{sum(c['acc'] < 0.2 for c in fb)}; probe at the 12000 check per seed "
    f"{[round(next((c['acc'] for c in runs[('BASE', s)]['checks'] if c['step'] == 12000), float('nan')), 3) for s in s89.SEEDS]}")
say(f"  report_10 text added in {FIX} (21:38:45, 3 min before {S89C}): "
    f"{[x.strip()[:120] for x in show(FIX, 'explore_out/H/report_10.md').splitlines() if 'first reaches 0.3' in x]}")

# ══ 3. S89b ═════════════════════════════════════════════════════════════════
say(f"\n== 3. S89b (explore_h_s89b.py at {S89B}; identical at the head: "
    f"{blob(S89B, 'explore_h_s89b.py') == blob(HB, 'explore_h_s89b.py')})")
cc_tree = ast.parse(show(HB, "explore_h_s89c_child.py"))
ARMC = next(eval(compile(ast.Expression(n.value), "ARM", "eval"), {"__builtins__": {}, "dict": dict}) for n in cc_tree.body if isinstance(n, ast.Assign) and
            getattr(n.targets[0], "id", "") == "ARM")
_stub("explore_h_s89c_child", ARM=ARMC)
s89b = load("explore_h_s89b", HB)
A3b = ("DELTA_SPLIT", "DELTA_NOSPLIT", "DELTA_ALLMASS")


def v_s89b(n, only):
    """The referee's reading of S89b's docstring (lines 36-47); indices 0 SPLIT, 1 NOSPLIT, 2 ALLMASS."""
    if n[0] < 7:
        return "the labelled recipe itself fails here"
    if only[(0, 1)] - only[(1, 0)] <= 1:
        return "the split is not needed here"
    if only[(0, 2)] - only[(2, 0)] <= 1 and only[(2, 1)] >= 4 and only[(1, 2)] == 0:
        return "survives"
    if only[(0, 2)] >= 4 and only[(2, 0)] == 0:
        return "needs key positions"
    return "no reading applies"


def h_name_b(rd):
    if "S89b" in rd:
        s = rd["S89b"]
        for nm in ("UNTESTED", "INCOMPLETE", "the labelled recipe itself fails here"):
            if s.startswith(nm):
                return nm
        return "?" + s
    sp, al = rd["split"], rd["DELTA_ALLMASS"]
    if sp.startswith("the split is not needed here"):
        if any(nm in al for nm in NAMES):
            return "?moot with a name"
        return "the split is not needed here"
    names = [nm for nm in NAMES if f"DELTA_ALLMASS: the trigger {nm}" in al or f"DELTA_ALLMASS: {nm}" in al]
    return {"survives": "survives", "needs key positions": "needs key positions",
            "no reading applies": "no reading applies"}[names[0]] if len(names) == 1 else f"?{names}"


tallyb, disb = collections.Counter(), []
ORC = {"2/2": {2912: rec(True), 2913: rec(True)}, "1/2": {2912: rec(True), 2913: rec(False)},
       "1 stored": {2912: rec(True)}, "0/2": {2912: rec(False), 2913: rec(False)}}
for oname, orc in ORC.items():
    tb = collections.Counter()
    for c in comp8:
        cnt, n, only = stats3(c)
        seq = [p for p, k in zip(PAT3, c) for _ in range(k)]
        SYN.clear()
        SYN["ORACLE_DELTA"] = dict(orc)
        for i, a in enumerate(A3b):
            SYN[a] = {s: rec(bool(p[i])) for s, p in zip(s89b.SEEDS, seq)}
        r_, _ = s89b.report()
        hn = h_name_b(r_["readings"])
        tb[hn] += 1
        if oname == "2/2":
            # rule_s89b's arm order is (SPLIT, ALLMASS, NOSPLIT)
            cnt2 = {(p[0], p[2], p[1]): v for p, v in cnt.items()}
            if hn != ocm.rule_s89b(cnt2) or hn != v_s89b(n, only):
                disb.append((tuple(c), hn, ocm.rule_s89b(cnt2), v_s89b(n, only)))
    say(f"  ORACLE_DELTA {oname}: report() names {dict(tb)}")
say(f"  disagreements (ORACLE 2/2) with rule_s89b or the referee's reading: {len(disb)} {disb[:2]}")


# s89c_schedule's choice
def sched_of(lb, lc_, cb, cc_):
    st_ = {"meta": {"code_shas": ["x"]}, "runs": {}}
    fake = types.SimpleNamespace(NAME="H/s89c", NEW=("BASE_LATE", "BASE_CAP6"), report=lambda: (dict(complete=True, cmp={
        "BASE_LATE vs BASE": dict(b=lb, c=lc_), "BASE_CAP6 vs BASE": dict(b=cb, c=cc_)}), None))
    sys.modules["explore_h_s89c"] = fake
    old = sys.modules["explore_common"].load_store
    sys.modules["explore_common"].load_store = lambda name: st_
    try:
        return s89b.s89c_schedule()
    finally:
        sys.modules["explore_common"].load_store = old


for case in ((4, 0, 4, 0), (4, 0, 5, 0), (6, 0, 4, 0), (4, 1, 5, 0), (3, 0, 3, 0)):
    say(f"  s89c_schedule with (LATE only, BASE only; CAP6 only, BASE only) = {case}: {sched_of(*case)}")
sys.modules["explore_h_s89c"] = s89c
# X's delta records
xd = json.loads(show("8ae660d", "results/X/delta_recipe_results.json"))
for k in ("DELTA_SPLIT|583", "DELTA_SLOW|583", "CEIL_DELTA|580"):
    r = xd["runs"][k]
    say(f"  8ae660d {k}: cpu {r['cpu']}; transition {r.get('transition')}; fired "
        f"{[c['step'] for c in r.get('checks') or [] if c.get('fired')]}; memory {r.get('memory')}; trigger_op {r.get('trigger_op')}")
say(f"  8ae660d: {git('log', '-1', '--format=%s', '8ae660d').decode().strip()}")
say(f"  S89b CHECK seed 2959 / ORACLE seeds {s89b.ORACLE_SEEDS} / learned {s89b.SEEDS[0]}-{s89b.SEEDS[-1]}")
srcb = show(HB, "explore_h_s89b.py")
say(f"  'Referee V' in explore_h_s89b.py at lines {[i for i, x in enumerate(srcb.splitlines(), 1) if 'Referee V' in x]}")

# ══ 4. seeds ═════════════════════════════════════════════════════════════════
say("\n== 4. Seeds 2912-2959 (S89b 2912-2923, S89c 2924-2933, CHECK 2959)")
BLOCK = set(range(2912, 2960))
KEYRE = re.compile(r"^[A-Za-z0-9_.\-]+\|(-?\d+)(\|.*)?$")


def walk(o, hit, parent=""):
    if isinstance(o, dict):
        for k, v in o.items():
            m = KEYRE.match(k) if isinstance(k, str) else None
            if m and int(m.group(1)) in BLOCK:
                hit.add(("key", k.split("|")[0], int(m.group(1))))
            if k == "seed" and isinstance(v, int) and not isinstance(v, bool) and v in BLOCK:
                hit.add(("seed", parent, v))
            if k == "seeds" and isinstance(v, list):
                for x in v:
                    if isinstance(x, int) and not isinstance(x, bool) and x in BLOCK:
                        hit.add(("seeds", parent, x))
            walk(v, hit, k)
    elif isinstance(o, list):
        for x in o:
            walk(x, hit, parent)


def parse(text):
    try:
        return [json.loads(text)]
    except ValueError:
        objs = []
        for ln in text.splitlines():
            try:
                objs.append(json.loads(ln))
            except ValueError:
                pass
        return objs


def scan(blobs):
    hits = {}
    p = subprocess.Popen(["git", "-C", ROOT, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    for sha in blobs:
        p.stdin.write((sha + "\n").encode())
        p.stdin.flush()
        size = int(p.stdout.readline().split()[2])
        data = p.stdout.read(size)
        p.stdout.read(1)
        h = set()
        for o in parse(data.decode("utf-8", "replace")):
            walk(o, h)
        if h:
            hits[sha] = h
    p.stdin.close()
    p.wait()
    return hits


tip = {}
for b in heads:
    for line in git("ls-tree", "-r", b).decode().splitlines():
        meta, path = line.split("\t", 1)
        _, typ, sha = meta.split()
        if typ == "blob" and path.endswith((".json", ".jsonl")):
            tip.setdefault(sha, []).append((b.split("/")[-1], path))
hits = scan(tip)
say(f"  {len(heads)} heads, {len(tip)} distinct JSON/JSONL blobs; blobs with 2912-2959: {len(hits)}")
for s, v in hits.items():
    say(f"    {tip[s][:2]}: {sorted(v)[:4]} ... ({len(v)} hits; seeds {sorted({x[2] for x in v})})")
hist = {}
for line in git("rev-list", "--all", "--objects").decode().splitlines():
    parts = line.split(" ", 1)
    if len(parts) == 2 and parts[1].endswith((".json", ".jsonl")):
        hist.setdefault(parts[0], []).append(parts[1])
hh = scan(hist)
say(f"  history: {len(hist)} distinct JSON/JSONL blobs; with 2912-2959: {len(hh)}")
for s, v in hh.items():
    say(f"    {hist[s][0]}: seeds {sorted({x[2] for x in v})}; kinds {sorted({x[0] for x in v})}")
pat = r"(^|[^0-9.e])(29(1[2-9]|[2-5][0-9]))([^0-9]|$)"
txt = collections.defaultdict(set)
for b in heads:
    try:
        r = git("grep", "-n", "-E", pat, b, "--", "*.py", "*.json", "*.md", "*.tex", "*.txt", "*.log").decode()
    except subprocess.CalledProcessError:
        r = ""
    for ln in r.splitlines():
        _, path, lno = ln.split(":", 3)[:3]
        txt[path].add(b.split("/")[-1])
say("  text mentions of 2912-2959 on the heads (path: branches):")
for p, bs in sorted(txt.items()):
    say(f"    {p}: {sorted(bs)[:4]}{' ...' if len(bs) > 4 else ''}")

# ══ 5. operating characteristics ═════════════════════════════════════════════
say("\n== 5. Operating characteristics (the referee's own enumeration; V's model: with probability rho one shared "
    "U ~ U(0,1) decides every arm, else independent)")


def probs(rates, rho):
    m = len(rates)
    pats_ = list(itertools.product((0, 1), repeat=m))
    out_ = []
    for p in pats_:
        ind = math.prod(r if b else 1 - r for r, b in zip(rates, p))
        lo = max([0.0] + [r for r, b in zip(rates, p) if not b])
        hi = min([1.0] + [r for r, b in zip(rates, p) if b])
        out_.append(rho * max(0.0, hi - lo) + (1 - rho) * ind)
    return pats_, out_


LF = [math.lgamma(i + 1) for i in range(11)]


def oc(rates, rule, rho):
    pats_, pr = probs(rates, rho)
    res_ = collections.Counter()
    for c in comp8:
        if any(k and q == 0 for k, q in zip(c, pr)):
            continue
        lp = LF[10] - sum(LF[k] for k in c) + sum(k * math.log(q) for k, q in zip(c, pr) if k)
        cnt = dict(zip(pats_, (int(x) for x in c)))
        n = [sum(v for p, v in cnt.items() if p[i]) for i in range(3)]
        only = {(i, j): sum(v for p, v in cnt.items() if p[i] and not p[j]) for i in range(3) for j in range(3) if i != j}
        res_[rule(n, only)] += math.exp(lp)
    return res_


def row(lab, rates, rule, names):
    rs_ = [oc(rates, rule, r) for r in (0.0, 0.5, 1.0)]
    say(f"  {lab}: " + "; ".join(f"{nm} " + "/".join(f"{x.get(nm, 0):.2f}" for x in rs_) for nm in names))


N3 = ("the schedule was the limit", "the channel is the limit", "no reading applies")
say(" S89c (BASE, BASE_LATE, BASE_CAP6); rho 0 / 0.5 / 1")
for b in (0.1, 0.2, 0.3):
    for v in (0.1, 0.2, 0.3, 0.5, 0.7):
        row(f"BASE {b}, both new {v}", (b, v, v), v_s89c, N3)
for v in (0.3, 0.5, 0.7):
    row(f"BASE 0.1, one new {v}, other 0.1", (0.1, v, 0.1), v_s89c, N3)
row("BASE 0.5, both new 0.5", (0.5, 0.5, 0.5), v_s89c, N3)
row("BASE 0.8, both new 0.8", (0.8, 0.8, 0.8), v_s89c, N3)
row("BASE 0.8, both new 0.3", (0.8, 0.3, 0.3), v_s89c, N3)


def binom_ge(k, n, p):
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))


say(" P(BASE rerun >= 4/10) at a true 0.1 / 0.2 / 0.3: " + " / ".join(f"{binom_ge(4, 10, p):.3f}" for p in (0.1, 0.2, 0.3)))
say(" S89 BASE 1/10: Wilson " + "%.2f-%.2f" % tuple(hc.wilson(1, 10)))
N5 = ("the labelled recipe itself fails here", "the split is not needed here", "survives", "needs key positions",
      "no reading applies")
say(" S89b gate: P(DELTA_SPLIT >= 7/10) at a true rate p: " + "; ".join(
    f"{p} {binom_ge(7, 10, p):.3f}" for p in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)))
say(" S89b (DELTA_SPLIT, DELTA_NOSPLIT, DELTA_ALLMASS); rho 0 / 0.5 / 1")
for s_, ns_, t_ in ((0.9, 0.9, 0.9), (0.9, 0.9, 0.5), (0.95, 0.95, 0.9), (0.9, 0.5, 0.9), (0.9, 0.5, 0.7), (0.9, 0.5, 0.5),
                    (0.7, 0.7, 0.7), (0.7, 0.3, 0.7), (0.5, 0.5, 0.5), (0.5, 0.2, 0.5), (0.3, 0.3, 0.3), (0.1, 0.1, 0.1)):
    row(f"SPLIT {s_}, NOSPLIT {ns_}, ALLMASS {t_}", (s_, ns_, t_), v_s89b, N5)

with open(os.path.join(HERE, "screen_checks_s89bc_H_out.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
