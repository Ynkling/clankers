#!/usr/bin/env python
"""check_F_report9.py — Referee V: recompute every record-checkable number in Session F's report 9 (S89-F,
claude/explore-F 52e2ea7) from git objects. Reads git objects only, trains nothing, writes nothing: it prints.
F's own functions (decide, _arms, _bound, decision from explore_f_s89_tasks.py at 8ba99bb; the reading expression and
routed_end from explore_f_reports6.py; wilson/band/count_str from explore_f_common.py) are extracted by ast and exec'd
on the records with in-memory stubs (no file is opened for writing)."""
import ast
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import types

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rlib  # noqa: E402

ROOT = rlib.ROOT
TIP, PRE = "52e2ea7", "8ba99bb"
SHA_S89, CPU = "ea9728c636c6", "Intel(R) Xeon(R) Processor @ 2.80GHz"


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a, stderr=subprocess.DEVNULL)


def show(ref, path):
    return git("show", f"{ref}:{path}")


def store(ref, path):
    return json.loads(show(ref, path))


def H(t):
    print("\n" + "=" * 110 + f"\n{t}\n" + "=" * 110)


def extract(src, names):
    """Source of the named top-level functions / assignments of a module."""
    tree = ast.parse(src)
    out = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef,)) and n.name in names:
            out.append(ast.get_source_segment(src, n))
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets):
            out.append(ast.get_source_segment(src, n))
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Tuple) and any(getattr(e, "id", None) in names for e in t.elts)
                                               for t in n.targets):
            out.append(ast.get_source_segment(src, n))
    return "\n\n".join(out)


R = store(TIP, "explore_out/F/s89_tasks_results.json")["runs"]
D = store(TIP, "explore_out/F/s89_diag_results.json")["runs"]


def rec(store_runs, arm, seed, sha=SHA_S89):
    return store_runs.get(f"{arm}|{seed}|{sha}|{CPU}")


# ── A. Code SHAs ─────────────────────────────────────────────────────────────
H("A  Code SHAs recomputed from git objects (explore_common16.code_sha: sha1(json.dumps(sorted [name, sha1(src), 'branch'])))")
MODS26 = ("bdh bdh_recurrent explore_b16_gates explore_common explore_common2 explore_delta_mem explore_f_delta_fast "
          "explore_f_s89_tasks_child explore_f_tasks explore_f_window_delta_child explore_local_gate "
          "explore_window_recipe_child test_binding_capacity test_binding_onset test_binding_recipe test_channel_binding "
          "test_instrument_v2 test_interference test_multilayer_binding test_readout_path test_router_confirm "
          "test_router_curriculum test_router_discovery test_router_layout test_router_reliability test_short_conv").split()
MODS27 = sorted(MODS26 + ["explore_f_s89_diag_child"])


def code_sha(ref, mods):
    rows = []
    for m in sorted(mods):
        try:
            b = show(ref, m + ".py")
        except subprocess.CalledProcessError:
            return None
        rows.append([m, hashlib.sha1(b.decode().encode()).hexdigest(), "branch"])
    return hashlib.sha1(json.dumps(rows).encode()).hexdigest()


commits = git("rev-list", "--reverse", f"{PRE}~1..{TIP}").decode().split()
shas26 = {c[:7]: code_sha(c, MODS26) for c in commits}
print("26-module SHA (s89) per commit:", {k: (v[:12] if v else None) for k, v in shas26.items()})
recorded = {r["code_sha"] for r in R.values()}
print("recorded code_sha in s89_tasks_results.json:", recorded)
print("recomputed at 8ba99bb:", shas26[PRE][:40], "MATCH" if shas26[PRE] in recorded else "DISAGREE")
print("constant 8ba99bb..52e2ea7:", len({v for k, v in shas26.items() if k != commits[0][:7]}) == 1)
dsha = code_sha("4639d16", MODS27)
print("27-module SHA (diag) at 4639d16:", dsha, "recorded:", {r["code_sha"] for r in D.values()},
      "MATCH" if dsha in {r["code_sha"] for r in D.values()} else "DISAGREE")
print("27-module SHA (diag) at 52e2ea7:", code_sha(TIP, MODS27)[:12])
print("record 'git' fields s89:", sorted({r["git"] for r in R.values()}))
print("record 'git' fields diag:", sorted({r["git"] for r in D.values()}))
print("torch:", {r["torch"] for r in list(R.values()) + list(D.values())}, "cpu:", {r["cpu"] for r in list(R.values()) + list(D.values())})

# ── B. Files: blobs unchanged? ───────────────────────────────────────────────
H("B  Code and rule files per commit (blob ids) 8ba99bb..52e2ea7")
FILES = ["explore_f_s89_tasks.py", "explore_f_s89_tasks_child.py", "explore_f_tasks.py", "explore_f_reports6.py",
         "explore_f_common.py", "explore_common16.py", "explore_f_reports.py", "explore_f_s89_diag.py", "explore_f_s89_diag_child.py"]
for f in FILES:
    ids = []
    for c in commits:
        try:
            ids.append(git("rev-parse", f"{c}:{f}").decode().strip()[:10])
        except subprocess.CalledProcessError:
            ids.append("-")
    print(f"  {f:34s} distinct blobs over {len(commits)} commits: {sorted(set(ids))}  first present at "
          f"{next((c[:7] for c, i in zip(commits, ids) if i != '-'), None)}")
print("  explore_f_tasks.py at 50dc8e4 (full):", git("rev-parse", "50dc8e4").decode().strip(),
      "blob", git("rev-parse", "50dc8e4:explore_f_tasks.py").decode().strip()[:10])
a = show("a199d5e", "explore_f_tasks.py")
b = show("50dc8e4", "explore_f_tasks.py")
print(f"  a199d5e file {len(a)} bytes = {len(a.decode())} characters; 50dc8e4's first {len(a)} bytes equal it: {b[:len(a)] == a}; "
      f"first 13771 equal: {b[:13771] == a[:13771]}; 13771 == len(a199d5e): {len(a) == 13771}")
print("  50dc8e4 commit time:", git("log", "-1", "--format=%ad", "--date=iso", "50dc8e4").decode().strip())

# ── C. Phase 0 ───────────────────────────────────────────────────────────────
H("C  Phase 0 (held-out acc at the end; stop if > 0.6)")
for t in ("LITE", "POOL", "ALLPOOL"):
    row = []
    for v in ("NOMEM", "NOMEM_SELF"):
        rs = [rec(R, f"{v}_{t}", s) for s in (2800, 2801)]
        row.append(" / ".join(f"{r['acc']:.3f}" for r in rs) + f"  (raw {[r['acc'] for r in rs]}; iters {[r['stopped_at'] for r in rs]})")
    print(f"  {t:8s} NOMEM {row[0]}   NOMEM_SELF {row[1]}")
print("  max phase-0 acc on POOL/ALLPOOL:", max(rec(R, f"{v}_{t}", s)["acc"] for v in ("NOMEM", "NOMEM_SELF")
                                              for t in ("POOL", "ALLPOOL") for s in (2800, 2801)))

# ── D. decide() exec'd stage by stage ────────────────────────────────────────
H("D  decide() from 8ba99bb exec'd stage by stage on the stores at each stage's commit (run_s89.sh order)")
s89src = show(PRE, "explore_f_s89_tasks.py").decode()
fsrc = extract(s89src, {"LR", "P0_SEEDS", "P1_SEEDS", "P2_SEEDS", "TASKS", "NOMEM_THR", "decision", "_arms", "_bound", "decide"})
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

    def fruns(me, arm):
        a = ns["ARMS"][arm]
        out = {}
        for s in a["seeds"]:
            r = runs_store.get(f"{arm}|{s}|{SHA_S89}|{CPU}")
            if r is not None and r.get("ok"):
                out[s] = r
        return out
    fc = types.SimpleNamespace(setup=lambda mods: None, runs=fruns, F_DIR="F")
    ns.update(json=JsonShim, os=OsShim, open=lambda *a, **k: None, fc=fc, DECISION="F/s89_decision.json",
              sys=types.SimpleNamespace(modules={"__x__": None}), __name__="__x__", print=lambda *a, **k: None)
    exec(fsrc, ns)
    return ns


STAGES = [("phase0", "368d26c"), ("phase1", "e036951"), ("phase1b", "15b550d"), ("phase2", "dd09246")]
for st, c in STAGES:
    ns = make_ns(store(c, "explore_out/F/s89_tasks_results.json")["runs"])
    ns["ARMS"] = ns["_arms"]()          # the --decide process re-imports the module
    out = ns["decide"]()
    filed = store(c, "explore_out/F/s89_decision.json")
    print(f"  {st:8s} @ {c}: recomputed {json.dumps(out, sort_keys=True)[:400]}")
    print(f"           equals the decision file at {c}: {out == filed}")
fin = store(TIP, "explore_out/F/s89_decision.json")
print("  final recomputed == s89_decision.json at 52e2ea7:", MEM["d"] == fin)
ns = make_ns(R)
MEM_keep = copy.deepcopy(MEM["d"])
ns["ARMS"] = ns["_arms"]()
print("  ARMS at the final decision (what phase 2 would queue):", sorted(ns["ARMS"]))
# the reading expression of explore_f_reports6 l.72-74
r6 = show(PRE, "explore_f_reports6.py").decode()
tree = ast.parse(r6)
expr = None
for n in ast.walk(tree):
    if isinstance(n, ast.Assign) and isinstance(n.value, ast.IfExp) and isinstance(n.targets[0], ast.Subscript):
        expr = ast.get_source_segment(r6, n.value)
print("  rep6 reading expression:", " ".join(expr.split()))
for t in ("POOL", "ALLPOOL"):
    print(f"   {t}: budget {fin['budget'][t]!r} -> {eval('(' + expr + ')', {'b': fin['budget'][t]})}")

# ── E. Pilots ────────────────────────────────────────────────────────────────
H("E  Pilots: transitions (record and rlib.transition(curve)), final acc, Wilson/band (F's count_str and rlib)")
fcsrc = show(TIP, "explore_f_common.py").decode()
fns = {"math": math}
exec(extract(fcsrc, {"wilson", "band", "count_str"}), fns)
for t in ("POOL", "ALLPOOL"):
    for suf in ("", "_48K"):
        for m in ("HEBB", "DELTA"):
            arm = f"O_{m}_{t}{suf}"
            rs = [rec(R, arm, s) for s in (2800, 2801)]
            nb = sum(rlib.bound(r) for r in rs)
            lo, hi = rlib.wilson(nb, 2)
            print(f"  {arm:20s} " + "; ".join(
                f"{r['seed']} rec-trans {r['transition']} rlib {rlib.transition(r['curve'])} acc {r['acc']:.4f} ({r['acc']:.2f}) "
                f"stopped_at {r['stopped_at']} end margin {r['end']['margin']:.4f} eta_S {r['end']['eta_key_by_stream']:.4f} "
                f"attempts {r['attempts']} collapsed {r['collapsed']}" for r in rs)
                + f" | F {fns['count_str'](nb, 2)} | rlib [{lo:.2f}, {hi:.2f}] {rlib.band(nb, 2)}")
print("  delta pilots bound:", sum(rlib.bound(r) for k, r in R.items() if k.startswith("O_DELTA")), "of",
      sum(1 for k in R if k.startswith("O_DELTA")), "transitions", sorted({r['transition'] for k, r in R.items() if k.startswith('O_DELTA')}))
print("  all 28 ok:", sum(r["ok"] for r in R.values()), "/", len(R), "; attempts", {r["attempts"] for r in R.values()},
      "; collapsed", {r["collapsed"] for r in R.values()},
      "; any nan in curves", any(any(x != x for row in r["curve"] for x in row) for r in R.values()))

H("E2 Hebbian pilot curves (acc per evaluation)")
for t in ("POOL", "ALLPOOL"):
    for suf in ("", "_48K"):
        for s in (2800, 2801):
            r = rec(R, f"O_HEBB_{t}{suf}", s)
            cv = [(st, round(a, 3)) for st, a, *_ in r["curve"]]
            print(f"  O_HEBB_{t}{suf} {s}: " + " ".join(f"{st}:{a}" for st, a in cv))
for s in (2800, 2801):
    for t in ("POOL", "ALLPOOL"):
        r = rec(R, f"O_HEBB_{t}_48K", s)
        late = [a for st, a, *_ in r["curve"] if st >= 32400]
        print(f"  {t} {s} 48K: max acc {max(a for _, a, *_ in r['curve']):.4f} at "
              f"{max(r['curve'], key=lambda x: x[1])[0]}; range 32400-48000 {min(late):.3f}-{max(late):.3f}; "
              f"first 1200-6000 {[round(a, 3) for st, a, *_ in r['curve'] if st <= 6000]}")
for t in ("POOL", "ALLPOOL"):
    for s in (2800, 2801):
        r = rec(R, f"O_HEBB_{t}", s)
        print(f"  {t} {s} 24K: min {min(a for _, a, *_ in r['curve']):.3f} max {max(a for _, a, *_ in r['curve']):.3f}")

H("E3 Write order (acc_last vs acc_notlast) on every Hebbian pilot")
mx = []
for t in ("POOL", "ALLPOOL"):
    for suf in ("", "_48K"):
        for s in (2800, 2801):
            w = rec(R, f"O_HEBB_{t}{suf}", s)["write_order"]
            mx.append((abs(w["acc_last"] - w["acc_notlast"]), f"O_HEBB_{t}{suf}|{s}"))
            print(f"  O_HEBB_{t}{suf} {s}: last {w['acc_last']:.4f} not-last {w['acc_notlast']:.4f} |diff| "
                  f"{abs(w['acc_last'] - w['acc_notlast']):.4f}")
print("  max |diff|:", max(mx))

H("E4 48000 arms vs 24000 arms: curves and statistics identical over the common prefix?")
for t in ("POOL", "ALLPOOL"):
    for m in ("HEBB", "DELTA"):
        for s in (2800, 2801):
            a, b = rec(R, f"O_{m}_{t}", s), rec(R, f"O_{m}_{t}_48K", s)
            n = min(len(a["curve"]), len(b["curve"]))
            ns_ = min(len(a["stats"]), len(b["stats"]))
            print(f"  O_{m}_{t} {s}: curve prefix {n} evals (to {a['curve'][n - 1][0]}) identical {a['curve'][:n] == b['curve'][:n]}; "
                  f"stats prefix {ns_} identical {a['stats'][:ns_] == b['stats'][:ns_]}; grad identical {a['grad'] == b['grad']}; "
                  f"lens {len(a['curve'])}/{len(b['curve'])}")

# ── F. Diagnostic ────────────────────────────────────────────────────────────
H("F  Post-hoc diagnostic records")
for arm in ("D_HEBB_LITE_SLOW", "D_HEBB_POOL_ADAM", "D_HEBB_ALLPOOL_ADAM"):
    rs = [D[f"{arm}|{s}|d4af41b711b3|{CPU}"] for s in (2800, 2801)]
    nb = sum(rlib.bound(r) for r in rs)
    for r in rs:
        e = r.get("end") or {}
        accs = [a for _, a, *_ in r["curve"]]
        mstats = [x.get("margin") for x in r["stats"] if "margin" in x]
        estats = [x.get("eta_key_by_stream") for x in r["stats"] if "eta_key_by_stream" in x]
        print(f"  {arm} {r['seed']}: ok {r['ok']} rec-trans {r['transition']} rlib {rlib.transition(r['curve'])} acc {r['acc']:.4f} "
              f"curve min/max {min(accs):.3f}/{max(accs):.3f} stopped_at {r['stopped_at']} end margin {e.get('margin')} "
              f"eta_S {e.get('eta_key_by_stream')} margin over evals {min(mstats) if mstats else None}-{max(mstats) if mstats else None} "
              f"eta over evals {min(estats) if estats else None}-{max(estats) if estats else None} recipe {r.get('recipe')} "
              f"groups {r.get('group_names')} attempts {r['attempts']}")
    print(f"   -> F {fns['count_str'](nb, 2)}")
print("  diag secs_wall total h:", round(sum(r["secs_wall"] for r in D.values()) / 3600, 3),
      "secs total h:", round(sum(r["secs"] for r in D.values()) / 3600, 3))
s70 = store(TIP, "explore_out/F/rh_lite_results.json")["runs"]
print("  S70_ORACLE_HEBB (lite, Adam 1e-3):", {k.split("|")[1]: (r["transition"], r.get("sched")) for k, r in s70.items()
                                               if k.startswith("S70_ORACLE_HEBB|")})
print("  SLOW 24K Hebbian pilots reaching >= 0.45 at any evaluation:",
      [f"{t}|{s}" for t in ("POOL", "ALLPOOL") for s in (2800, 2801)
       if max(a for _, a, *_ in rec(R, f"O_HEBB_{t}", s)["curve"]) >= 0.45])

# ── G. Runtime ───────────────────────────────────────────────────────────────
H("G  Runtime")
print("  s89 secs_wall total h:", round(sum(r["secs_wall"] for r in R.values()) / 3600, 3),
      "(store_meta uses secs_wall); secs total h:", round(sum(r["secs"] for r in R.values()) / 3600, 3))
log = show(TIP, "explore_out/F/s89.log").decode()
for m in re.finditer(r"=== S89 (\S+) (.*)", log):
    print("  log:", m.group(0))
dlog = show(TIP, "explore_out/F/s89diag.log").decode()
for m in re.finditer(r"=== S89 diag (\S+) (.*)", dlog):
    print("  diag log:", m.group(0))
PASS_RE = r"ALL CHECKS PASSED [(]([0-9]+) ok"
print("  child CHECK lines per segment (ok / FAIL):")
for seg in re.split(r"=== S89 ", log)[1:]:
    print(f"   {seg.split()[0]:8s} ok {len(re.findall(r'CHECK F/s89_tasks: .*: ok', seg))} FAIL "
          f"{len(re.findall(r'CHECK F/s89_tasks: .*: FAIL', seg))}; delta checks: "
          f"{re.findall(PASS_RE, seg)}")
print("  diag delta checks:", re.findall(r"ALL CHECKS PASSED \((\d+) ok", dlog))

# ── H. Cuts ─────────────────────────────────────────────────────────────────
H("H  Frequency cells (s89_tasks_tables.md / log CHECK lines); derived numbers")
tab = show(TIP, "explore_out/F/s89_tasks_tables.md").decode()
for line in tab.splitlines():
    if line.startswith("| LITE") or line.startswith("| POOL") or line.startswith("| ALLPOOL"):
        print("  ", line)
print("  2 CTX mass on POOL: 0.1136+0.1137 =", 0.1136 + 0.1137, "; ALLPOOL ratio min VAL / max CTX = 0.0220/0.0151 =",
      round(0.0220 / 0.0151, 3))
for l in log.splitlines():
    if "frequencies (S87's sample)" in l or "no token type above 1/8" in l or "no key type above" in l:
        print("  log:", l.strip()[:220])
        if "ALLPOOL frequencies" in l:
            break

# ── I. Seeds ────────────────────────────────────────────────────────────────
H("I  Seeds 2800-2839 on every remote branch tip (json/jsonl/log)")
refs = git("for-each-ref", "--format=%(refname:short)", "refs/remotes").decode().split()
pat = r'\|28[0-3][0-9]\||"seed": 28[0-3][0-9]\b|seed 28[0-3][0-9]\b'
for ref in refs:
    try:
        out = git("grep", "-c", "-E", pat, ref, "--", "*.json", "*.jsonl", "*.log").decode()
    except subprocess.CalledProcessError:
        out = ""
    hits = [l for l in out.splitlines() if l]
    if hits:
        print(f"  {ref}: " + "; ".join(h.split(":", 1)[1] for h in hits))
print("  seeds in the S89 records:", sorted({r["seed"] for r in R.values()}), "diag:", sorted({r["seed"] for r in D.values()}))

# ── J. Order ────────────────────────────────────────────────────────────────
H("J  Commits 8ba99bb..52e2ea7 and the decision file at each")
for c in commits:
    t = git("log", "-1", "--format=%h %ad %s", "--date=format:%H:%M:%S", c).decode().strip()
    fs = git("show", "--name-only", "--format=", c).decode().split()
    try:
        dd = store(c, "explore_out/F/s89_decision.json")
        dk = {k: (v if k == "budget" else sorted(v)) for k, v in dd.items()}
    except subprocess.CalledProcessError:
        dk = None
    try:
        nrec = len(store(c, "explore_out/F/s89_tasks_results.json")["runs"])
    except subprocess.CalledProcessError:
        nrec = 0
    print(f"  {t[:95]}\n     files {[f.replace('explore_out/F/', '') for f in fs]}; records {nrec}; decision {dk}")
print("  referee review commit:", git("log", "-1", "--format=%h %ad", "--date=iso", "3ae815b").decode().strip())

# ── K. Report text: fixes mentioned? ────────────────────────────────────────
H("K  Report 9 / README text: does it mention the referee's fixes?")
rep = show(TIP, "explore_out/F/report_9.md").decode()
readme = show(TIP, "explore_out/README.md").decode()
sec = readme[readme.index("### Session F, round 6"):]
for term in ("0.533", "8/15", "64/120", "invariant", "within each sequence", "_inv", "Bayes", "0.42", "dirty", "p⁴", "p^4",
             "referee", "Referee", "screen_review", "post hoc", "post-hoc", "Post hoc", "Post-hoc", "untested", "fail_class_v2",
             "v2", "unpaired", "cause"):
    print(f"  {term!r}: report {rep.count(term)}, README section {sec.count(term)}")

# ── L. The CHECK lines' printed curves against the pinned records ────────────
H("L  CHECK lines' printed reference curves vs the records at pinned commits")
g = store("e46799a", "explore_out/G/s87_results.json")["runs"]["BASELINE|2700|f205db31dfed|Intel(R) Xeon(R) Processor @ 2.10GHz"]
s7 = store("61ba46e", "explore_out/F/rh_lite_results.json")["runs"]["S70_DELTA|330|ff908bdbc0b2|Intel(R) Xeon(R) Processor @ 2.10GHz"]
for seg in re.split(r"=== S89 ", log)[1:-1]:
    m1 = re.search(r"reproduces G's S87 .*?curve (\[\[.*?\]\]) vs (\[\[.*?\]\])", seg)
    m2 = re.search(r"S70 reproduction on this CPU.*?curve (\[\[.*?\]\]) vs (\[\[.*?\]\])", seg)
    print(f"  {seg.split()[0]:8s} G87: run == printed ref {json.loads(m1.group(1)) == json.loads(m1.group(2))}, printed ref == "
          f"record@e46799a[:2] {json.loads(m1.group(2)) == g['curve'][:2]}; S70_DELTA: run == ref "
          f"{json.loads(m2.group(1)) == json.loads(m2.group(2))}, ref == record@61ba46e[:2] {json.loads(m2.group(2)) == s7['curve'][:2]}")
print("  explore_f_tasks.py blob: a199d5e", git("rev-parse", "a199d5e:explore_f_tasks.py").decode()[:10],
      "e46799a (G's vendored copy)", git("rev-parse", "e46799a:explore_f_tasks.py").decode()[:10])
