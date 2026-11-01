#!/usr/bin/env python3
"""check_H_s92_phase1.py — Referee V's check of session H's round-7 screen phase 1 (the prompt's S92; H's "S90") on
claude/explore-H at 2bc50f9b ("S90 phase 1 complete (P* = 8 ...); report writer batch 12; phase 2 running") and the
commits after it.

    python3 -I docs/review_2026-10/check_H_s92_phase1.py        (prints to stdout; no output file is written)

Run from the repository root after `git fetch`. Prints only. Every file is read with `git show`; nothing is trained, no
torch is imported and no session code is run as a program. H's explore_h_s90.py (cell_reading, phase1) is exec'd from
its source with the harness stubbed and served the stored records; the main line's transition() is cut out by ast.
"""
import ast
import json
import os
import subprocess
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCREEN = "23dccf24"          # the screen's readings
P1DONE = "2bc50f9b"          # phase 1 complete
H = "origin/claude/explore-H"
MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
MAIN9C = "9c5939e"
REVIEW = "980498e4"          # screen_review_s92_H.md
SHORT, LONG = 43200, 86400
POOLS, MEMS, P1 = (4, 8, 16, 32), ("HEBB", "DELTA"), (3020, 3021, 3022, 3023)


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a).decode()


def show(ref, path):
    try:
        return subprocess.check_output(("git", "-C", ROOT, "show", f"{ref}:{path}"), stderr=subprocess.DEVNULL).decode()
    except subprocess.CalledProcessError:
        return None


def say(*a):
    print(*a, flush=True)


# ══ 0. pins and order ═════════════════════════════════════════════════════════
say("== 0. Pins and order")
for r in (SCREEN, REVIEW, P1DONE, H, MAIN):
    say(f"   {r:<50} {git('log', '-1', '--format=%h %cI %s', r).strip()[:140]}")
revs = git("rev-list", "--reverse", f"{SCREEN}^..{H}").split()
say(f"   commits {SCREEN}..{H} (inclusive): {len(revs)}")
CODE = ("explore_h_s90.py", "explore_h_s90_child.py", "explore_h_run.py", "explore_h_report.py", "explore_h_common.py",
        "explore_common16.py", "explore_common.py", "README.md")
prev = {}
for c in revs:
    t, s = git("log", "-1", "--format=%cI|%s", c).strip().split("|", 1)
    blobs = {p: git("rev-parse", f"{c}:{p}").strip()[:10] if show(c, p) is not None else None for p in CODE}
    ch = [p for p in CODE if prev and blobs[p] != prev[p]]
    files = git("show", "--name-only", "--format=", c).split()
    other = [f for f in files if not f.startswith("explore_out/H/")]
    if "saved" not in s or ch or other:
        say(f"   {c[:8]} {t} {s[:90]}  | code changed: {ch or '-'}; non-store files: {other or '-'}")
    prev = blobs
say(f"   blobs at {H}: " + ", ".join(f"{p} {b}" for p, b in prev.items()))
d = git("diff", "--stat", SCREEN, H, "--", ".", ":!explore_out").strip()
say(f"   diff {SCREEN}..{H} outside explore_out: {d.splitlines()}")
saved = [(c, git("log", "-1", "--format=%cI|%s", c).strip()) for c in revs]
saved = [(c[:8], x.split("|")[0], x.split("|")[1].split(" saved")[0].split("H/s90 ")[-1]) for c, x in saved if " saved " in x]
say(f"   'saved' commits: {len(saved)}; first {saved[0]}; last phase-1 {[x for x in saved if '_P' in x[2]][-1]}; "
    f"phase 2 {[x for x in saved if '_P' not in x[2]]}")
say(f"   review {REVIEW} committed {git('log', '-1', '--format=%cI', REVIEW).strip()}; on origin/claude/review-V: "
    f"{subprocess.run(['git', '-C', ROOT, 'merge-base', '--is-ancestor', REVIEW, 'origin/claude/review-V']).returncode == 0}")

# ══ 1. the records ════════════════════════════════════════════════════════════
say("\n== 1. Phase-1 records at " + H)
st = json.loads(show(H, "explore_out/H/s90_results.json"))
runs = st["runs"]
say(f"   records: {len(runs)}; code SHAs {sorted({r['code_sha'][:12] for r in runs.values()})}; cpus "
    f"{sorted({r['cpu'] for r in runs.values()})}; meta code_shas {[s[:12] for s in st['meta']['code_shas']]}")
st1 = json.loads(show(P1DONE, "explore_out/H/s90_results.json"))
say(f"   records at {P1DONE}: {len(st1['runs'])}; the 32 phase-1 records identical at {P1DONE} and the head: "
    f"{all(st1['runs'].get(k) == r for k, r in runs.items() if '_P' in k.split('|')[0])}")

tbo = {}
for ref in (MAIN9C, MAIN):
    src = show(ref, "test_binding_onset.py")
    tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "transition"][0]
    bp = [n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "BIND_PASS"][0]
    tbo[ref] = ast.dump(fn) + ast.dump(bp)
    ns = {}
    exec(compile(ast.Module([bp, fn], []), f"transition@{ref}", "exec"), ns)
    tr = ns["transition"]
say(f"   main line's transition() + BIND_PASS equal at {MAIN9C} and {MAIN}: {tbo[MAIN9C] == tbo[MAIN]}")

rows = {}
bad = []
for k, r in sorted(runs.items()):
    arm, seed = k.split("|")[0], int(k.split("|")[1])
    if "_P" not in arm:
        continue
    mem, pool = arm.split("_P")[0], int(arm.split("_P")[1])
    cv = r["curve"]
    steps = [x[0] for x in cv]
    t_full = tr(cv)
    t_cut = tr([x for x in cv if x[0] <= SHORT])
    end = steps[-1]
    tail3 = len(cv) >= 3 and all(x[1] >= 0.95 for x in cv[-3:])
    ends_ok = (end == LONG) or (tail3 and end < LONG)
    a43 = next((x[1] for x in cv if x[0] == SHORT), None)
    rows[(mem, pool, seed)] = dict(t=r["transition"], t_full=t_full, t_cut=t_cut, end=end, a43=a43, acc=r["acc"],
                                   outcome=r["outcome"])
    chk = dict(budget=r["budget"] == LONG, iters=r["iters"] == LONG, pool=r["pool"] == pool,
               memory=r["memory"] == mem.lower().replace("hebb", "hebb"), kind=r["kind"] == "oracle",
               steps=steps == list(range(1200, end + 1, 1200)), stored_eq_recomputed=r["transition"] == t_full,
               full_eq=r.get("transition_full") == t_full, stopped=r["stopped_at"] == end, ends_ok=ends_ok, ok=r["ok"])
    if not all(chk.values()):
        bad.append((k, {c: v for c, v in chk.items() if not v}))
say(f"   32 phase-1 records: each has budget 86400, iters 86400, its cell's pool and memory, kind oracle, evaluations every "
    f"1200 from 1200 without gaps, stored transition = transition() on its curve = transition_full, stopped_at = the last "
    f"evaluation, and ends at 86400 or after 3 evaluations >= 0.95: {not bad} {bad or ''}")

say("\n   cell | seed transitions (H = stored/recomputed; cut = transition() on the curve cut at 43200) | by 43200 H | "
    "by 43200 cut | by 86400")
diff = []
for mem in MEMS:
    for pool in POOLS:
        rr = [rows[(mem, pool, s)] for s in P1]
        bh = [x["t"] is not None and x["t"] <= SHORT for x in rr]
        bc = [x["t_cut"] is not None for x in rr]
        bl = [x["t"] is not None for x in rr]
        say(f"   {mem}_P{pool:<3} H {[x['t'] for x in rr]}; cut {[x['t_cut'] for x in rr]}; "
            f"{sum(bh)}/4 | {sum(bc)}/4 | {sum(bl)}/4; end acc {[round(x['acc'], 3) for x in rr]}")
        for s, x, a, b in zip(P1, rr, bh, bc):
            if a != b:
                diff.append((mem, pool, s, x))
say(f"   cells where H's 'by 43200' and the cut-curve 'by 43200' differ: {len(diff)} {diff or ''}")
for mem, pool, s in [(m, p, s) for (m, p, s), x in rows.items() if x["t"] is None]:
    x = rows[(mem, pool, s)]
    cv = runs[[k for k in runs if k.startswith(f"{mem}_P{pool}|{s}|")][0]]["curve"]
    w = [round(c[1], 3) for c in cv if 36000 <= c[0] <= 45600]
    say(f"   unbound {mem}_P{pool}|{s}: acc at 43200 {x['a43']:.3f}; max acc to 43200 "
        f"{max(c[1] for c in cv if c[0] <= SHORT):.3f}, over the run {max(c[1] for c in cv):.3f} at "
        f"{max(cv, key=lambda c: c[1])[0]}; evaluations >= 0.95 {sum(c[1] >= 0.95 for c in cv)}; 36000-45600 {w}; "
        f"outcome {x['outcome']}")

# ══ 2. H's readings exec'd, and the cut-curve readings ═══════════════════════
say(f"\n== 2. Readings: H's cell_reading/phase1 exec'd from explore_h_s90.py at {P1DONE}, served the stored records")


def _stub(name, **kw):
    m = types.ModuleType(name)
    m.__dict__.update(kw)
    sys.modules[name] = m
    return m


_stub("explore_common", load_store=lambda name: st, cpu_model=lambda: "stub", OUT_DIR="/nonexistent")
_stub("explore_common16", runs_of=lambda *a, **k: {}, med=lambda xs: None, STATE={"shas": {}}, code_sha=lambda *a: ("x", []),
      key=lambda *a: "x", run_child=lambda *a, **k: None)
_stub("explore_muon_scale")
_stub("explore_h_collapse")
_stub("explore_h_s89", FIELDS=(), NAME="H/s89", MAINREF="628eb7c", CHILD="explore_h_s89_child")
_stub("explore_h_s89b", NAME="H/s89b", DELTA_REC="8ae660d")


def load(name, ref):
    m = types.ModuleType(name)
    m.__file__ = os.path.join(ROOT, "_nonexistent_", name + ".py")
    sys.modules[name] = m
    exec(compile(show(ref, name + ".py"), f"{name}.py@{ref}", "exec"), m.__dict__)
    return m


load("explore_h_common", P1DONE)
s90 = load("explore_h_s90", P1DONE)
say(f"   explore_h_s90.py identical at {SCREEN}, {P1DONE} and {H}: "
    f"{show(SCREEN, 'explore_h_s90.py') == show(P1DONE, 'explore_h_s90.py') == show(H, 'explore_h_s90.py')}")


def roc(c):
    return {int(k.split("|")[1]): r for k, r in runs.items() if k.split("|")[0] == c and r.get("ok")}


out, complete, pstar, budget = s90.phase1(roc)
say(f"   H's phase1(): complete {complete}; P* {pstar}; budgets {budget}")
for c, v in out.items():
    say(f"     {c:<9} {v['reading']}")


def cut_reading(mem, pool):
    rr = [rows[(mem, pool, s)] for s in P1]
    if all(x["t_cut"] is not None for x in rr):
        return "reliable"
    if all(x["t"] is not None for x in rr):
        return "needs the longer budget"
    return "the oracle is the limit"


cr = {f"{m}_P{p}": cut_reading(m, p) for m in MEMS for p in POOLS}
ok = [p for p in POOLS if all(cr[f"{m}_P{p}"] != "the oracle is the limit" for m in MEMS)]
p2 = max(ok) if ok else None
b2 = {m: SHORT if cr[f"{m}_P{p2}"] == "reliable" else LONG for m in MEMS} if p2 else None
say(f"   fix-2 definition (by 43200 = the curve cut at 43200): readings differ from H's in "
    f"{[c for c in cr if cr[c] != out[c]['reading']] or 'no cell'}; P* {p2}; budgets {b2}")
ref_fail = [m for m in MEMS if out[f"{m}_P4"]["reading"] == "the oracle is the limit"]
nested = [p for p in POOLS if pstar and p < pstar and p not in ok]
say(f"   fix 3 (the pool-4 reference): fails on {ref_fail or 'neither memory'}; fix 4 (pools below P* that fail): "
    f"{nested or 'none'}")
for m in MEMS:
    say(f"   {m}: cells k/4 by 86400 " + ", ".join(f"P{p} {sum(rows[(m, p, s)]['t'] is not None for s in P1)}/4" for p in POOLS)
        + "; by 43200 " + ", ".join(f"P{p} {sum(rows[(m, p, s)]['t_cut'] is not None for s in P1)}/4" for p in POOLS))
rd = [ln for ln in show(H, "explore_out/H/s90_phase1.log").splitlines() if "readings {" in ln]
say(f"   the log's last readings line: {rd[-1].strip()[:400]}")
say(f"   H's log line equals the exec'd phase1(): P* {('P* = ' + str(pstar)) in rd[-1]}; budgets "
    f"{str(budget) in rd[-1]}")

# ══ 3. phase 2 ════════════════════════════════════════════════════════════════
say("\n== 3. Phase 2")
m2 = st["meta"]["arms"]
say(f"   store meta arms (phase 2): " + "; ".join(f"{a}: iters {m2[a]['iters']}, {m2[a]['label']}"
                                                  for a in ("BASE", "DELTA_SPLIT", "DELTA_NOSPLIT") if a in m2))
for k, r in sorted(runs.items()):
    if "_P" in k.split("|")[0]:
        continue
    f = [(c["step"], round(c["acc"], 3), c.get("op_ok")) for c in r.get("checks") or [] if c.get("fired")]
    say(f"   {k.split('|')[0]}|{k.split('|')[1]}: pool {r['pool']}, memory {r['memory']}, budget {r['budget']}, iters "
        f"{r['iters']}, trig last {r.get('trig_cfg', {}).get('last')}; transition {r['transition']} (recomputed "
        f"{tr(r['curve'])}); stopped {r['stopped_at']}; {r['outcome']}; fired {f}; git {r['git']}; code {r['code_sha'][:12]}")
lg2 = show(H, "explore_out/H/s90_phase2.log")
say(f"   s90_phase2.log: {len(lg2.splitlines())} lines; 'CHECK' rows {lg2.count('CHECK H/s90')}; "
    f"{[ln.strip() for ln in lg2.splitlines() if 'skipped' in ln or 'queued' in ln or '=== start' in ln]}")
src = show(H, "explore_h_s90.py")
say(f"   explore_h_s90.py at the head: 'S92' {src.count('S92')}; 'S90' {src.count('S90')}; 'moot' {src.count('moot')}; "
    f"'NOSPLIT|583' / 'DELTA_SLOW' {src.count('DELTA_SLOW')}; 'ORACLE|2600' or 'CEIL_DELTA' "
    f"{src.count('2600') + src.count('CEIL_DELTA')}; 'budget mismatch' {src.count('budget mismatch')}; "
    f"'reference fails' {src.count('reference fails')}")
rep = show(H, "explore_h_report.py")
i = rep.find("def s90_section")
sec = rep[i:rep.find("\ndef ", i + 5)]
say(f"   explore_h_report.py s90_section: prints caveat {'caveat' in sec}; prints oc {'oc' in sec}; 'S92' {sec.count('S92')}; "
    f"'S90' {sec.count('S90')}")
run = show(H, "explore_h_run.py")
say(f"   explore_h_run.py identical at {SCREEN} and the head: {run == show(SCREEN, 'explore_h_run.py')}; "
    f"run:152 {run.splitlines()[151].strip()[:120]}")
say(f"   report_12.md committed on {H}: {show(H, 'explore_out/H/report_12.md') is not None}")

# ══ 4. how much P* rests on: one seed changed, H's phase1() exec'd ═══════════
say("\n== 4. Sensitivity of P* (H's phase1() on the stored records with one transition replaced)")
for cell, seed, t in (("DELTA_P32", 3022, 20000), ("DELTA_P32", 3022, 60000), ("DELTA_P16", 3021, 20000),
                      ("DELTA_P8", 3023, None), ("DELTA_P8", 3023, 60000), ("HEBB_P8", 3020, None)):
    def roc2(c, cell=cell, seed=seed, t=t):
        g = roc(c)
        if c == cell:
            g = dict(g)
            g[seed] = dict(g[seed], transition=t)
        return g
    o, cpl, ps, bu = s90.phase1(roc2)
    say(f"   {cell}|{seed} transition -> {t}: P* {ps}; budgets {bu}; {cell} reads {o[cell]['reading']}")
