#!/usr/bin/env python
"""check_F_report10.py — Referee V: recompute every record-checkable number in Session F's report 10 (S90-F,
claude/explore-F d9783bb1) and its README section from git objects. Reads git objects only (and this clone's reflog for
one push time), trains nothing, writes nothing: it prints.
F's own functions are extracted by ast and exec'd on the records with in-memory stubs (no file is opened for writing):
  explore_f_s90_ctx.py at 73749e8a: decision, _arms, outcomes, decide, reading_task, reading_screen (+ constants);
  explore_f_common.py at 73749e8a: wilson, band, count_str, mcnemar_greater, paired;
  explore_f_tasks.py at 9017b823: balanced_map, _esym, _partitions, type_map_cap;
and rlib (transition, bound, wilson, band, mcnemar1).
Sections:
  A code SHA and blobs per commit; B records; C decide() stage by stage, readings, ARMS per stage; D pilots;
  E screen; F type-level gate; G post-hoc account; H write order; I phase 0; J cap table; K runtime and CHECK counts;
  L erratum; M seeds; N the hold timeline; O the lite references; P operating characteristics; Q text terms."""
import ast
import copy
import hashlib
import itertools as it
import json
import math
import os
import re
import statistics
import subprocess
import sys
import types

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rlib  # noqa: E402

ROOT = rlib.ROOT
TIP, PRE, MOD = "d9783bb1", "73749e8a", "9017b823"
SHA = "809fde1f5168"
CPU = "Intel(R) Xeon(R) Processor @ 2.10GHz"
RES = "explore_out/F/s90_ctx_results.json"
DEC = "explore_out/F/s90_decision.json"


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a, stderr=subprocess.DEVNULL)


def show(ref, path):
    return git("show", f"{ref}:{path}")


def store(ref, path):
    return json.loads(show(ref, path))


def H(t):
    print("\n" + "=" * 110 + f"\n{t}\n" + "=" * 110)


def extract(src, names):
    tree = ast.parse(src)
    out = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            out.append(ast.get_source_segment(src, n))
        elif isinstance(n, ast.Assign):
            ids = {x.id for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name)}
            if ids & names:
                out.append(ast.get_source_segment(src, n))
    return "\n\n".join(out)


R = store(TIP, RES)["runs"]
DFIN = store(TIP, DEC)


def rec(arm, seed):
    return R.get(f"{arm}|{seed}|{SHA}|{CPU}")


def arm_runs(arm, runs=None):
    runs = R if runs is None else runs
    return {int(k.split("|")[1]): r for k, r in runs.items() if k.split("|")[0] == arm}


# F's helpers
fcsrc = show(PRE, "explore_f_common.py").decode()
FC = {"math": math}
exec(extract(fcsrc, {"wilson", "band", "count_str", "mcnemar_greater", "paired"}), FC)
TASKS, MEMS = ("CTXPOOL", "DOCS3", "DOCS4"), ("HEBBCONV", "DELTA")
P0, P2 = (3060, 3061), tuple(range(3062, 3072))

# ── A ────────────────────────────────────────────────────────────────────────
H("A  Code SHA (27 modules, as the log lists) recomputed from blobs; code/rule blobs per commit 9017b823^..d9783bb1")
log = show(TIP, "explore_out/F/s90.log").decode()
m = re.search(r"code SHA F/s90_ctx: 809fde1f5168 over 27 modules \(([^)]*)\)", log)
MODS = [x.strip() for x in m.group(1).split(",")]
print("  modules:", len(MODS))


def code_sha(ref, mods):
    rows = []
    for mm in sorted(mods):
        try:
            b = show(ref, mm + ".py")
        except subprocess.CalledProcessError:
            return None
        rows.append([mm, hashlib.sha1(b.decode().encode()).hexdigest(), "branch"])
    return hashlib.sha1(json.dumps(rows).encode()).hexdigest()


commits = git("rev-list", "--reverse", f"{MOD}~1..{TIP}").decode().split()
shas = {c[:8]: code_sha(c, MODS) for c in commits}
print("  per commit:", {k: (v[:12] if v else None) for k, v in shas.items()})
print("  recorded code_sha:", {r["code_sha"] for r in R.values()})
print("  at 73749e8a:", shas[PRE[:8]], "MATCH" if shas[PRE[:8]] in {r["code_sha"] for r in R.values()} else "DISAGREE")
print("  constant from 73749e8a to tip:", len({v for k, v in shas.items() if commits.index(next(c for c in commits if c.startswith(k)))
                                               >= commits.index(next(c for c in commits if c.startswith(PRE)))}) == 1)
FILES = ["explore_f_s90_ctx.py", "explore_f_s90_ctx_child.py", "explore_f_reports7.py", "explore_f_tasks.py",
         "explore_f_common.py", "explore_common16.py", "explore_f_reports.py", "explore_f_reports3.py", "explore_f_reports4.py",
         "explore_f_s89_tasks_child.py", "explore_out/F/run_s90.sh"]
for f in FILES:
    ids = []
    for c in commits:
        try:
            ids.append(git("rev-parse", f"{c}:{f}").decode().strip()[:10])
        except subprocess.CalledProcessError:
            ids.append("-")
    after = [i for c, i in zip(commits, ids) if commits.index(c) >= commits.index(next(x for x in commits if x.startswith(PRE)))]
    print(f"  {f:34s} blobs from 73749e8a to tip: {sorted(set(after))}")
print("  explore_f_tasks.py full blob at 9017b823:", git("rev-parse", f"{MOD}:explore_f_tasks.py").decode().strip())
for ref in ("origin/claude/explore-G", "origin/claude/explore-H"):
    try:
        print(f"  {ref} explore_f_tasks.py blob:", git("rev-parse", f"{ref}:explore_f_tasks.py").decode().strip()[:10])
    except subprocess.CalledProcessError:
        print(f"  {ref}: no explore_f_tasks.py")

# ── B ────────────────────────────────────────────────────────────────────────
H("B  Records (96)")
print("  n", len(R), "ok", sum(r["ok"] for r in R.values()), "attempts", {r["attempts"] for r in R.values()},
      "collapsed", {r["collapsed"] for r in R.values()}, "cpu", {r["cpu"] for r in R.values()},
      "torch", {r["torch"] for r in R.values()})
print("  NaN in any curve:", any(any(x != x for row in r["curve"] for x in row) for r in R.values()))
print("  seeds:", sorted({r["seed"] for r in R.values()}))
print("  iters per arm:", sorted({(k.split("|")[0], r["iters"]) for k, r in R.items() if not k.startswith("NOMEM")})[:4], "...",
      {r["iters"] for r in R.values()})
print("  sched:", {(k.split("|")[0].split("_")[0], r["sched"]) for k, r in R.items()})
print("  group_names (SINGLE):", {str(r["group_names"]) for k, r in R.items() if k.startswith("SINGLE")})
print("  k (SINGLE):", {r["k"] for k, r in R.items() if k.startswith("SINGLE")}, "memory:",
      {(k.split("|")[0], r["memory"], str(r["conv"])) for k, r in R.items() if k.startswith("SINGLE")})
print("  record git fields:", sorted({r["git"] for r in R.values()}))

# ── C ────────────────────────────────────────────────────────────────────────
H("C  decide() / _arms() / outcomes() / reading_task() / reading_screen() from 73749e8a, stage by stage (run_s90.sh)")
s90src = show(PRE, "explore_f_s90_ctx.py").decode()
fsrc = extract(s90src, {"LR", "P0_SEEDS", "P1_SEEDS", "P2_SEEDS", "TASKS", "MEMS", "NOMEM_THR", "SCREEN_HI", "SCREEN_LO",
                        "SLOW", "decision", "_arms", "outcomes", "decide", "reading_task", "reading_screen"})
MEM = {}


class JsonShim:
    dumps = staticmethod(json.dumps)

    @staticmethod
    def load(f):
        return copy.deepcopy(MEM["d"])

    @staticmethod
    def dump(d, f, indent=None):
        MEM["d"] = copy.deepcopy(d)


class OsShim:
    class path:
        exists = staticmethod(lambda p: "d" in MEM)
        join = staticmethod(os.path.join)


def make_ns(runs_store):
    ns = {}

    def fruns(me, arm, seeds=None):
        return {int(k.split("|")[1]): r for k, r in runs_store.items()
                if k.split("|")[0] == arm and r.get("ok") and k.split("|")[2] == SHA}
    fc = types.SimpleNamespace(setup=lambda mods: None, runs=fruns, load=lambda me: {"runs": runs_store}, F_DIR="F")
    c16 = types.SimpleNamespace(MAX_ATTEMPTS=3, STATE={})
    ns.update(json=JsonShim, os=OsShim, open=lambda *a, **k: None, fc=fc, c16=c16, DECISION="F/s90_decision.json",
              sys=types.SimpleNamespace(modules={"__x__": None}), __name__="__x__", print=lambda *a, **k: None)
    exec(fsrc, ns)
    ns["accept_other_cpus"] = lambda: []
    return ns


STAGES = [("phase0", "a4d14c24"), ("phase1", "79f00415"), ("phase1b", "d1072a46"), ("phase2", "4c9612f6"), ("final", "b502f1c3")]
prev_keys = set()
for st, c in STAGES:
    runs_c = store(c, RES)["runs"]
    ns0 = make_ns(runs_c)
    arms_at_start = ns0["_arms"]()          # what the stage's import queued (decision before this stage's --decide)
    new = sorted({k.split("|")[0] for k in set(runs_c) - prev_keys})
    queued_new = sorted(a for a in arms_at_start if any(k.split("|")[0] == a for k in set(runs_c) - prev_keys))
    ns = make_ns(runs_c)
    ns["ARMS"] = ns["_arms"]()              # --decide re-imports
    out = ns["decide"]()
    filed = store(c, DEC)
    print(f"  {st:8s} @ {c}: records {len(runs_c)} (+{len(set(runs_c) - prev_keys)}); arms queued at the stage's import: "
          f"{len(arms_at_start)}; new arms with records this stage: {new}")
    print(f"           every new record's arm was queued at import: {set(new) <= set(arms_at_start)}; recomputed decision == "
          f"file at {c}: {out == filed}")
    prev_keys = set(runs_c)
print("  final == s90_decision.json at tip:", MEM["d"] == DFIN)
ns = make_ns(R)
ns["ARMS"] = ns["_arms"]()
print("  SINGLE arms at the final decision:", sorted(a for a in ns["ARMS"] if a.startswith("SINGLE")),
      "iters", {ns["ARMS"][a]["iters"] for a in ns["ARMS"] if a.startswith("SINGLE")})
print("  fix 12 (iters on record == ARMS iters):", all(r["iters"] == ns["ARMS"][k.split("|")[0]]["iters"] for k, r in R.items()))
for t in TASKS:
    for mm in MEMS:
        print(f"   reading_task({t}, {mm}) = {ns['reading_task'](DFIN, t, mm)!r}")
print("   reading_screen =", repr(ns["reading_screen"](DFIN)))
o, a = ns["outcomes"]("SINGLE_DELTA_CTXPOOL", P2)
print("   outcomes SINGLE_DELTA_CTXPOOL:", o)
# the decision at 79f00415 (phase 1's) already queues SINGLE: the arms the phase1b stage's import saw
MEM["d"] = store("79f00415", DEC)
ns1 = make_ns(store("79f00415", RES)["runs"])
print("  ARMS at the phase1b import (decision of 79f00415):", sorted(a for a in ns1["_arms"]() if a.startswith("SINGLE")))

# ── D ────────────────────────────────────────────────────────────────────────
H("D  Pilots (O_<M>_<T>, seeds 3060/3061): transitions (record and rlib), acc, end margin / seq_separated_frac; count_str")
for t in TASKS:
    for mm in MEMS:
        arm = f"O_{mm}_{t}"
        rs = [rec(arm, s) for s in P0]
        nb = sum(rlib.bound(r) for r in rs)
        print(f"  {arm:20s} " + "; ".join(
            f"{r['seed']} trans {r['transition']} rlib {rlib.transition(r['curve'])} acc {r['acc']:.4f} stop {r['stopped_at']} "
            f"margin {r['end'].get('margin'):.4f} sep {r['end'].get('seq_separated_frac'):.4f} "
            f"sep_key_inv {r['end'].get('sep_key_inv'):.4f}" for r in rs) + f" | F {FC['count_str'](nb, 2)} | rlib "
            f"[{rlib.wilson(nb, 2)[0]:.2f}, {rlib.wilson(nb, 2)[1]:.2f}] {rlib.band(nb, 2)}")
print("  48K arms present:", [k for k in R if "_48K" in k])

# ── E ────────────────────────────────────────────────────────────────────────
H("E  Screen (SINGLE_<M>_<T>, 3062-3071)")
SING = {}
for t in TASKS:
    for mm in ("DELTA", "HEBBCONV"):
        arm = f"SINGLE_{mm}_{t}"
        rs = arm_runs(arm)
        assert sorted(rs) == list(P2)
        b = {s: rlib.bound(r) for s, r in rs.items()}
        SING[(t, mm)] = b
        nb = sum(b.values())
        tr = sorted(r["transition"] for r in rs.values() if rlib.bound(r))
        accs = [rs[s]["acc"] for s in P2]
        ub = [rs[s]["acc"] for s in P2 if not b[s]]
        print(f"  {arm:24s} {FC['count_str'](nb, 10)} (rlib [{rlib.wilson(nb, 10)[0]:.2f}, {rlib.wilson(nb, 10)[1]:.2f}] "
              f"{rlib.band(nb, 10)}); rec-trans == rlib: {all(r['transition'] == rlib.transition(r['curve']) for r in rs.values())}"
              f"; median {statistics.median(tr) if tr else None} range {(min(tr), max(tr)) if tr else None}")
        print(f"     final acc: " + " ".join(f"{s}:{rs[s]['acc']:.4f}" for s in P2))
        print(f"     unbound range {min(ub):.4f}-{max(ub):.4f} mean {statistics.mean(ub):.4f}" if ub else "     unbound: none",
              f"; mean all {statistics.mean(accs):.4f}")
for t in TASKS:
    pr = FC["paired"](SING[(t, "DELTA")], SING[(t, "HEBBCONV")], P2)
    print(f"  paired {t}: {pr}; rlib mcnemar1 {rlib.mcnemar1(pr['b'], pr['c']):.6f}; 1/1024 = {1 / 1024:.6f}")
r = rec("SINGLE_DELTA_DOCS3", 3065)
print("  SINGLE_DELTA_DOCS3|3065 curve:", [(s, round(a, 4)) for s, a, *_ in r["curve"]])
print("     max", max(a for _, a, *_ in r["curve"]), "last", r["curve"][-1][:2], "stopped_at", r["stopped_at"])

# ── F ────────────────────────────────────────────────────────────────────────
H("F  Type-level gate (TL_<M>_<T>)")
for t in TASKS:
    for mm in MEMS:
        arm = f"TL_{mm}_{t}"
        rs = [rec(arm, s) for s in P0]
        print(f"  {arm:20s} " + "; ".join(
            f"{r['seed']} trans {r['transition']} rlib {rlib.transition(r['curve'])} acc {r['acc']:.4f} "
            f"sep end {r['end']['seq_separated_frac']:.4f} (x256 = {r['end']['seq_separated_frac'] * 256:.1f}) "
            f"margin {r['end']['margin']:.4f} sep_key_inv {r['end']['sep_key_inv']:.4f} val_cos_k {r['end']['val_cos_inv_k']:.4f}"
            f" sep over evals {sorted({round(x['seq_separated_frac'], 4) for x in r['stats'] if 'seq_separated_frac' in x})}"
            for r in rs))
        print(f"     mean sep {statistics.mean(r['end']['seq_separated_frac'] for r in rs):.4f}")
CAPS = {2: 64 / 120, 3: 150 / 560, 4: 256 / 1820}
for t, D in (("CTXPOOL", 2), ("DOCS3", 3), ("DOCS4", 4)):
    for s in P0:
        x = round(rec(f"TL_DELTA_{t}", s)["end"]["seq_separated_frac"] * 256)
        p = CAPS[D]
        lo = sum(math.comb(256, i) * p ** i * (1 - p) ** (256 - i) for i in range(0, x + 1))
        hi = sum(math.comb(256, i) * p ** i * (1 - p) ** (256 - i) for i in range(x, 257))
        print(f"  probe {t} {s}: {x}/256 = {x / 256:.4f} vs cap {p:.4f}: P(X <= x) {lo:.2e}, P(X >= x) {hi:.2e}; "
              f"z {(x - 256 * p) / math.sqrt(256 * p * (1 - p)):+.2f}")
    xs = sum(round(rec(f"TL_DELTA_{t}", s)["end"]["seq_separated_frac"] * 256) for s in P0)
    print(f"  {t} both seeds {xs}/512 = {xs / 512:.4f}; z {(xs - 512 * CAPS[D]) / math.sqrt(512 * CAPS[D] * (1 - CAPS[D])):+.2f}")

# ── G ────────────────────────────────────────────────────────────────────────
H("G  The post-hoc account: expected accuracy of the balanced type map at k = D (query document uniform)")
for D in (2, 3, 4):
    cmap = [t * D // 16 for t in range(16)]
    tot = {"hebb": 0.0, "delta_1/m": 0.0, "delta_2/m": 0.0, "delta_0": 0.0}
    n = 0
    mdist = {}
    for sub in it.combinations(range(16), D):
        for q in sub:
            mq = sum(1 for t in sub if cmap[t] == cmap[q])
            mdist[mq] = mdist.get(mq, 0) + 1
            tot["hebb"] += 1 / mq
            tot["delta_1/m"] += 1 if mq <= 2 else 1 / mq
            tot["delta_2/m"] += 1 if mq <= 2 else 2 / mq
            tot["delta_0"] += 1 if mq <= 2 else 0
            n += 1
    print(f"  D {D}: m distribution {dict(sorted((k, round(v / n, 4)) for k, v in mdist.items()))}; "
          + "; ".join(f"{k} {v / n:.4f}" for k, v in tot.items()))
for t in TASKS:
    print(f"  observed {t}: HEBBCONV " + " / ".join(f"{rec(f'TL_HEBBCONV_{t}', s)['acc']:.4f}" for s in P0)
          + "; DELTA " + " / ".join(f"{rec(f'TL_DELTA_{t}', s)['acc']:.4f}" for s in P0))

# ── H ────────────────────────────────────────────────────────────────────────
H("H  Write order (write_order_acc_idx fields): per-arm means over the 10 screen seeds; rep7 l.125 'or 0' check")
for t in TASKS:
    for mm in ("HEBBCONV", "DELTA"):
        arm = f"SINGLE_{mm}_{t}"
        ws = [rec(arm, s)["write_order"] for s in P2]
        nones = sum(1 for w in ws for f in ("acc_last", "acc_notlast", "lastval_notlast") if w.get(f) is None)
        mean = lambda f, ws=ws: statistics.mean(w[f] for w in ws)
        fails = [rec(arm, s)["write_order"] for s in P2 if not rlib.bound(rec(arm, s))]
        print(f"  {arm:24s} last {mean('acc_last'):.4f} notlast {mean('acc_notlast'):.4f} lastval_notlast "
              f"{mean('lastval_notlast'):.4f} acc {mean('acc'):.4f}; None fields {nones}; n_last/n "
              f"{statistics.mean(w['n_last'] / w['n'] for w in ws):.4f}")
        if fails and len(fails) < 10:
            print(f"     unbound only: acc {statistics.mean(w['acc'] for w in fails):.4f} lastval_notlast "
                  f"{statistics.mean(w['lastval_notlast'] for w in fails):.4f}")
        if mm == "DELTA" and t != "CTXPOOL":
            ex = [rec(arm, s)["write_order"] for s in P2 if s != 3065]
            print(f"     excluding 3065: acc {statistics.mean(w['acc'] for w in ex):.4f} notlast "
                  f"{statistics.mean(w['acc_notlast'] for w in ex):.4f} lastval_notlast {statistics.mean(w['lastval_notlast'] for w in ex):.4f}")
for arm in ("O_DELTA_CTXPOOL", "O_HEBBCONV_CTXPOOL"):
    print(f"  {arm}: " + "; ".join(f"{s} {rec(arm, s)['write_order']['acc_last']:.3f}/{rec(arm, s)['write_order']['acc_notlast']:.3f}" for s in P0))
print("  bound runs: |last - notlast| max:",
      max(abs(r["write_order"]["acc_last"] - r["write_order"]["acc_notlast"]) for r in R.values()
          if rlib.bound(r) and r.get("write_order")))

# ── I ────────────────────────────────────────────────────────────────────────
H("I  Phase 0 (held-out acc at the end) and the decision")
for t in TASKS:
    print(f"  {t}: NOMEM " + " / ".join(f"{rec(f'NOMEM_{t}', s)['acc']:.5f}" for s in P0) + "; NOMEM_SELF "
          + " / ".join(f"{rec(f'NOMEM_SELF_{t}', s)['acc']:.5f}" for s in P0) + f"; stopped {DFIN['phase0'][t]['stopped']}")
tab = show(TIP, "explore_out/F/s90_ctx_tables.md").decode()
for line in tab.splitlines():
    if re.match(r"\| (CTXPOOL|DOCS3|DOCS4) \| [0-9]", line):
        print("  tables.md:", line)

# ── J ────────────────────────────────────────────────────────────────────────
H("J  Cap of a fixed type -> channel map: F's type_map_cap (exec'd from 9017b823) and an independent count")
tsrc = show(MOD, "explore_f_tasks.py").decode()
TM = {}
exec(extract(tsrc, {"balanced_map", "_esym", "_partitions", "type_map_cap"}), TM)


def my_parts(n, k, mx=None):
    mx = n if mx is None else mx
    if n == 0:
        yield ()
        return
    if k == 0:
        return
    for a in range(min(n, mx), 0, -1):
        for r in my_parts(n - a, k - 1, a):
            yield (a,) + r


def count_distinct(sizes, D):
    """direct count of D-subsets of the 16 types whose channels are distinct, for a map with these channel sizes."""
    chan = [c for c, n in enumerate(sizes) for _ in range(n)]
    return sum(1 for sub in it.combinations(range(16), D) if len({chan[t] for t in sub}) == D)


rng = np.random.default_rng(90)
for D in (2, 3, 4):
    tot = math.comb(16, D)
    for k in sorted({2, D, 2 * D, 16}):
        cap, best, cnt, tt = TM["type_map_cap"](16, D, k)
        mine = max(count_distinct(p, D) for p in my_parts(16, k)) if k <= 8 else None
        bal = TM["balanced_map"](16, k)
        sizes = [bal.count(j) for j in range(k)]
        bal_cnt = count_distinct(sizes, D)
        sub = np.argsort(rng.random((200000, 16)), 1)[:, :D]
        ch = np.array(bal)[sub]
        sim = np.mean([len(set(r)) == D for r in ch[:50000]])
        print(f"  D {D} k {k:>2}: F {cnt}/{tt} = {cap:.4f} best {best}; independent max over partitions (direct count) "
              f"{mine}; balanced map {sizes} direct {bal_cnt}/{tot} = {bal_cnt / tot:.4f}; simulated (50000) {sim:.4f}")

# ── K ────────────────────────────────────────────────────────────────────────
H("K  Runtime, stage windows, CHECK counts per segment")
print("  sum secs_wall h:", round(sum(r["secs_wall"] for r in R.values()) / 3600, 3), "sum secs h:",
      round(sum(r["secs"] for r in R.values()) / 3600, 3))
for mt in re.finditer(r"=== S90 (\S+) (.*)", log):
    print("  log:", mt.group(0))
segs = re.split(r"=== S90 ", log)[1:]
for seg in segs:
    nm = seg.split()[0]
    ok = len(re.findall(r"CHECK F/s90_ctx: .*: ok$", seg, re.M))
    fail = len(re.findall(r"CHECK F/s90_ctx: .*: FAIL$", seg, re.M))
    dl = re.findall(r"ALL CHECKS PASSED \((\d+) ok", seg)
    q = re.findall(r"(\d+) runs queued", seg)
    print(f"   {nm:8s} S90 CHECK lines ok {ok} FAIL {fail}; delta checks {dl}; queued {q}")
for k_ in ("first", "last"):
    pass
for arm_prefix, label in (("NOMEM", "phase 0"), ("O_", "pilots"), ("TL_", "TL"), ("SINGLE", "screen")):
    rs = [r for k, r in R.items() if k.startswith(arm_prefix)]
    print(f"  {label}: {len(rs)} runs, secs_wall h {sum(r['secs_wall'] for r in rs) / 3600:.3f}")

# ── L ────────────────────────────────────────────────────────────────────────
H("L  Erratum (a0dc3c34): the old write_order_acc on -allpool; n_last fraction in S89 records under the old function")
for ref, path in (("52e2ea7", "explore_out/F/s89_tasks_results.json"), ("3e3b1bfe", "explore_out/F/s89_post_results.json")):
    st = store(ref, path)["runs"]
    for k, r0 in sorted(st.items()):
        w = r0.get("write_order")
        if not w or not r0.get("ok"):
            continue
        a = k.split("|")[0]
        if "POOL" in a:
            print(f"  {ref} {a:26s} {k.split('|')[1]}: n_last/n {w['n_last']}/{w['n']} = {w['n_last'] / w['n']:.3f}; "
                  f"last/notlast {w['acc_last'] if w['acc_last'] is None else round(w['acc_last'], 3)} / "
                  f"{w['acc_notlast'] if w['acc_notlast'] is None else round(w['acc_notlast'], 3)}; acc {r0['acc']:.3f}")
# numpy: P(query CTX type != its first-appearance index) on -allpool (2 of 16 types; index 0/1)
rng = np.random.default_rng(91)
pair = np.argsort(rng.random((200000, 16)), 1)[:, :2]       # types of stream 0 (first) and 1
qs = rng.integers(0, 2, 200000)
qtype = pair[np.arange(200000), qs]
print(f"  replica: P(query CTX type != index) = {np.mean(qtype != qs):.4f} (15/16 = {15 / 16:.4f}); of 512: "
      f"{512 * np.mean(qtype != qs):.1f}; F's CHECK 485/512 = {485 / 512:.4f}; binomial sd {math.sqrt(512 * 15 / 16 / 16):.1f}")
rep9a = show("a0dc3c34", "explore_out/F/report_9.md").decode()
print("  erratum present at tip report_9:", "Erratum (22 October)" in show(TIP, "explore_out/F/report_9.md").decode(),
      "; in report_10:", "Erratum" in show(TIP, "explore_out/F/report_10.md").decode())

# ── M ────────────────────────────────────────────────────────────────────────
H("M  Seeds 3060-3071 on every remote branch tip (json/jsonl/log)")
refs = git("for-each-ref", "--format=%(refname:short)", "refs/remotes").decode().split()
pat = r'\|30(6[0-9]|7[01])\||"seed": 30(6[0-9]|7[01])\b|seed 30(6[0-9]|7[01])\b'
for ref in refs:
    try:
        out = git("grep", "-c", "-E", pat, ref, "--", "*.json", "*.jsonl", "*.log").decode()
    except subprocess.CalledProcessError:
        out = ""
    hits = [l for l in out.splitlines() if l]
    if hits:
        print(f"  {ref}: " + "; ".join(h.split(":", 1)[1] for h in hits))
print("  remote refs scanned:", len(refs))

# ── N ────────────────────────────────────────────────────────────────────────
H("N  The hold: review push vs phase 2 start")
print("  ecbaf771 commit:", git("log", "-1", "--format=%ad", "--date=iso", "ecbaf771").decode().strip())
try:
    rl = git("reflog", "show", "--date=iso", "origin/claude/review-V").decode()
    print("  reflog (this clone's pushes):", [l for l in rl.splitlines() if l.startswith("ecbaf771") or l.startswith("db562c4e")])
except subprocess.CalledProcessError:
    print("  no reflog")
for c in commits:
    t = git("log", "-1", "--format=%h %ad %s", "--date=format:%H:%M:%S", c).decode().strip()
    fs = git("show", "--name-only", "--format=", c).decode().split()
    try:
        st = store(c, RES)["runs"]
        nrec = len(st)
        nsing = sum(1 for k in st if k.startswith("SINGLE"))
    except subprocess.CalledProcessError:
        nrec, nsing = 0, 0
    print(f"  {t[:70]:70s} records {nrec:3d} SINGLE {nsing:2d} files {[f.replace('explore_out/F/', '') for f in fs]}")
i = log.index("=== S90 phase1b")
seg = log[i:log.index("=== S90 phase2")]
print("  phase1b header:", seg.splitlines()[0])
first = re.search(r"start\s+0\.0 min\s+F/s90_ctx\s+(\S+)\s+seed (\d+)", seg)
print("  first run queued in phase1b:", first.group(1), first.group(2))
print("  checks file of phase1b: checks_s90_20261022_060742.json (CHECKs began 06:07:42); the queue's first start follows the "
      "CHECKs and timings")
cj = store(TIP, "explore_out/F/checks_s90_20261022_060742.json")
print("  phase1b checks json provenance:", {k: cj.get("provenance", {}).get(k) for k in ("git", "started", "time", "cpu")},
      "keys", list(cj)[:6])
rt = store("906216b8", "explore_out/F/s90_runtime.json")
print("  s90_runtime.json at 906216b8:", json.dumps(rt)[:300])

# ── O ────────────────────────────────────────────────────────────────────────
H("O  The lite references")
dc = store(TIP, "explore_out/F/delta_controls_results.json")["runs"]
for a in ("SINGLE_DELTA_H", "SINGLE_DELTA_G"):
    rs = [r for k, r in dc.items() if k.split("|")[0] == a]
    print(f"  S53 {a}: {sum(rlib.bound(r) for r in rs)}/{len(rs)} seeds {sorted(r['seed'] for r in rs)} sched {({r.get('sched') for r in rs})}")
hz = store(TIP, "explore_out/F/horizon_results.json")["runs"]
rs = [r for k, r in hz.items() if k.split("|")[0] == "FIXED_1X"]
print(f"  S88b FIXED_1X: {sum(rlib.bound(r) for r in rs)}/{len(rs)} seeds {sorted(r['seed'] for r in rs)} sched {({r.get('sched') for r in rs})}")
print(f"  Fisher one-sided 10/10 (S90) vs 8/10 (S53): {rlib.fisher1(10, 10, 8, 10):.3f}; vs 9/10 (S88b): {rlib.fisher1(10, 10, 9, 10):.3f}")

# ── P ────────────────────────────────────────────────────────────────────────
H("P  Operating characteristics of the screen line on 10 seeds")
for p in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
    ge7 = sum(math.comb(10, i) * p ** i * (1 - p) ** (10 - i) for i in range(7, 11))
    print(f"  p {p}: P(>= 7/10) {ge7:.3f}; P(10/10) {p ** 10:.4f}")
print("  Wilson 10/10:", FC["wilson"](10, 10), rlib.wilson(10, 10))

# ── Q ────────────────────────────────────────────────────────────────────────
H("Q  Report 10 / README text terms")
rep = show(TIP, "explore_out/F/report_10.md").decode()
readme = show(TIP, "explore_out/README.md").decode()
sec = readme[readme.index("### Session F, round 7"):]
for term in ("review", "V's", "go", "hold", "Adam", "S88b", "8/10", "operating", "seq_separated", "val_cos", "DISCOVERED",
             "dirty", "bytes", "cfe3fe27", "0.36", "0.79", "0.81", "equal the cap", "107"):
    print(f"  {term!r}: report {rep.count(term)}, README section {sec.count(term)}")
print("  report lines with 'hold':", [l.strip()[:100] for l in rep.splitlines() if "hold" in l])
