#!/usr/bin/env python3
"""check_H_report11.py — Referee V's check of session H's report 11 (S89c, S89b) on claude/explore-H at 4360cafa.

    python3 -I docs/review_2026-10/check_H_report11.py

Run from the repository root after `git fetch`. Prints only; writes nothing. Every file is read with `git show`; no
session code is run as a program and nothing is trained. The main line's outcome (test_stream_recipe.outcome and what it
calls) is cut out by ast at the main head and at 9c5939e; rlib.outcome_k is the referee's own re-implementation. H's
report() / summarize() / s89c_schedule() and the report writer (explore_h_report.main, batch 11) are exec'd from their
source with the harness stubbed (stores served from git; file writes captured in memory). explore_h_oc's pure functions
are exec'd from source.
"""
import ast
import io
import json
import math
import os
import statistics
import subprocess
import sys
import types
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "docs", "review_2026-10"))
import rlib  # noqa: E402

HREF = "4360cafa"
H = "origin/claude/explore-H"
MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
MAIN9C = "9c5939e"
S89C_PRE = "c81398ca"
S89B_PRE = "bccb7cbd"
CPU = "Intel(R) Xeon(R) Processor @ 2.10GHz"
F_C, F_B, F_89 = "explore_out/H/s89c_results.json", "explore_out/H/s89b_results.json", "explore_out/H/s89_results.json"


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a).decode()


def show(ref, path):
    try:
        return subprocess.check_output(("git", "-C", ROOT, "show", f"{ref}:{path}"), stderr=subprocess.DEVNULL).decode()
    except subprocess.CalledProcessError:
        return None


def say(*a):
    print(*a, flush=True)


HEAD = git("rev-parse", HREF).strip()
say(f"== 0. Pins: {HREF} = {HEAD[:8]} ({git('log', '-1', '--format=%cI %s', HREF).strip()[:110]})")
say(f"   {H} = {git('rev-parse', '--short=8', H).strip()}; main head {git('rev-parse', '--short=8', MAIN).strip()}")


# ── main-line outcome, by ast ────────────────────────────────────────────────
def extract(ref, path, names):
    src = show(ref, path)
    tree = ast.parse(src)
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names:
            out[n.name] = ast.get_source_segment(src, n)
        elif isinstance(n, ast.Assign):
            tg = [t.id for t in n.targets if isinstance(t, ast.Name)]
            tg += [e.id for t in n.targets if isinstance(t, ast.Tuple) for e in t.elts if isinstance(e, ast.Name)]
            hit = [t for t in tg if t in names]
            if hit:
                out[hit[0]] = ast.get_source_segment(src, n)
    assert not [x for x in names if x not in out], (path, names, list(out))
    return [out[x] for x in names]


SOURCES = [("test_router_layout.py", ["MARGIN_PARTIAL", "ETA_SPLIT", "fail_class"]),
           ("test_scale_axes.py", ["fail_class_k"]),
           ("test_stream_channels.py", ["shared_max", "merged"]),
           ("test_stream_recipe.py", ["ROUTED_ACC", "routed", "outcome", "wilson"])]


def mainline_ns(ref):
    ns = {"math": math, "statistics": statistics}
    srcs = []
    for path, names in SOURCES:
        for s in extract(ref, path, names):
            exec(compile(s, f"{path}@{ref}", "exec"), ns)
            srcs.append(ast.dump(ast.parse(s)))
    return ns, srcs


NS, T1 = mainline_ns(MAIN)
NS9, T2 = mainline_ns(MAIN9C)
say(f"   main-line outcome syntax trees equal at the main head and {MAIN9C}: {T1 == T2}")

# ── stores ───────────────────────────────────────────────────────────────────
ST = {"H/s89c": json.loads(show(HREF, F_C)), "H/s89b": json.loads(show(HREF, F_B)), "H/s89": json.loads(show(HREF, F_89))}


def byarm(store):
    out = {}
    for k, r in store["runs"].items():
        a, s, sha, cpu = k.split("|")
        assert (a, int(s)) not in out, k
        out[(a, int(s))] = r
        assert r["ok"] and r["attempts"] == 1 and cpu == CPU and r["code_sha"].startswith(sha), k
    return out


C = byarm(ST["H/s89c"])
B = byarm(ST["H/s89b"])
say(f"\n== 1. S89c store at {HREF}: {len(C)} runs {dict(Counter(a for a, _ in C))}; code_shas {ST['H/s89c']['meta']['code_shas']}; "
    f"all ok, attempt 1, one CPU (asserted)")
say(f"   S89b store: {len(B)} runs {dict(Counter(a for a, _ in B))}; code_shas {ST['H/s89b']['meta']['code_shas']}")
diff = []
for (a, s), r in sorted(list(C.items()) + list(B.items())):
    o = (r["outcome"], NS["outcome"](r), NS9["outcome"](r), rlib.outcome_k(r))
    t = rlib.transition(r["curve"])
    if len(set(o)) > 1 or t != r["transition"]:
        diff.append((a, s, o, r["transition"], t))
say(f"   outcome stored / main head / {MAIN9C} / rlib, and stored vs curve transition: disagreements {diff or 'none'}")

ARMS_C = ("BASE", "BASE_LATE", "BASE_CAP6")
SEEDS_C = tuple(range(2924, 2934))
G = {a: {s: C[(a, s)] for s in SEEDS_C} for a in ARMS_C}
BR = {a: {s: NS["outcome"](G[a][s]) == "BOUND ROUTED" for s in SEEDS_C} for a in ARMS_C}
TAB = {}
for a in ARMS_C:
    rs = G[a]
    k = sum(BR[a].values())
    w = rlib.wilson(k, 10)
    f = [c for r in rs.values() for c in r["checks"] if c.get("fired")]
    v1 = Counter("MERGED" if NS["outcome"](r).startswith("MERGED") else NS["outcome"](r).split(" ")[0]
                 for s, r in rs.items() if not BR[a][s])
    v2 = Counter(r["outcome_v2"] for r in rs.values() if r["outcome_v2"] != "BOUND ROUTED")
    TAB[a] = dict(br=k, band=rlib.band(k, 10), wilson=f"{w[0]:.2f}-{w[1]:.2f}", seeds=[s for s in SEEDS_C if BR[a][s]],
                  v1=dict(v1), v2=dict(v2), trans=sorted(r["transition"] for s, r in rs.items() if BR[a][s]),
                  ops=len(f), op_ok=sum(bool(c["op_ok"]) for c in f), flat=sum(c["acc"] < 0.2 for c in f),
                  o2o=sum(len(set(c["map_before"])) == 8 for c in f),
                  capped=sum(1 for r in rs.values() for c in r["checks"] if c.get("blocked_by") == "cap"),
                  last=sorted(max([c["step"] for c in r["checks"] if c.get("fired")], default=0) for r in rs.values()),
                  acc_end=sorted(round(r["acc"], 2) for r in rs.values()),
                  v1v2=[s for s, r in rs.items() if r["outcome_v2"] != r["outcome"]],
                  trig=json.dumps({x: rs[2924]["trig_cfg"][x] for x in ("first", "cap")}),
                  trig_same=len({json.dumps(r["trig_cfg"], sort_keys=True) for r in rs.values()}) == 1)
    say(f"   {a}: {TAB[a]}")
say("   pairs (b = x only, c = y only; rlib.mcnemar1 one-sided)")
CMP = {}
for x, y in (("BASE_LATE", "BASE"), ("BASE_CAP6", "BASE"), ("BASE_LATE", "BASE_CAP6")):
    b = sum(BR[x][s] and not BR[y][s] for s in SEEDS_C)
    c = sum(BR[y][s] and not BR[x][s] for s in SEEDS_C)
    CMP[(x, y)] = (b, c)
    say(f"   {x} vs {y}: {TAB[x]['br']}/10 vs {TAB[y]['br']}/10; {x} only {b}, {y} only {c}; p(x>y) {rlib.mcnemar1(b, c):.3g}, "
        f"p(y>x) {rlib.mcnemar1(c, b):.3g}")

# readings: committed rule (explore_h_oc.rule_s89c by ast), docstring restated, referee's proposed fixed rule
ocns = {"math": math, "itertools": __import__("itertools")}
for s in extract(HREF, "explore_h_oc.py", ["pattern_probs", "compositions", "oc", "only", "total", "rule_s89c", "rule_s89b"]):
    exec(compile(s, "explore_h_oc.py@4360cafa", "exec"), ocns)
cnt = Counter((int(BR["BASE"][s]), int(BR["BASE_LATE"][s]), int(BR["BASE_CAP6"][s])) for s in SEEDS_C)
say(f"\n   per-seed patterns (BASE, BASE_LATE, BASE_CAP6): {dict(cnt)}")
say(f"   committed rule (explore_h_oc.rule_s89c, ast): {ocns['rule_s89c'](dict(cnt))!r}")
beat = [v for v in ("BASE_LATE", "BASE_CAP6") if CMP[(v, 'BASE')][0] >= 4 and CMP[(v, 'BASE')][1] == 0]
mine = ("the schedule was the limit" if beat else
        "the channel is the limit" if all(TAB[v]["br"] < 4 for v in ("BASE_LATE", "BASE_CAP6")) else "no reading applies")
say(f"   docstring c:21-24 restated: {mine!r}; 'channel' condition both new arms < 4/10: "
    f"{[(v, TAB[v]['br']) for v in ('BASE_LATE', 'BASE_CAP6')]}")
fixed = ("S89's failure does not replicate here" if TAB["BASE"]["br"] >= 4 else
         "the schedule was the limit" if beat else
         "neither schedule change reaches 4/10" if all(TAB[v]["br"] < 4 for v in ("BASE_LATE", "BASE_CAP6"))
         else "no reading applies")
say(f"   referee's proposed fixed rule (premise BASE >= 4/10 first; renamed second reading): {fixed!r}")
s89 = byarm(ST["H/s89"])
k89 = sum(NS["outcome"](r) == "BOUND ROUTED" for (a, s), r in s89.items() if a == "BASE")
say(f"   S89 BASE (seeds 2902-2911): {k89}/10; S89c BASE rerun {TAB['BASE']['br']}/10 (seeds {TAB['BASE']['seeds']}); "
    f"Wilson S89 {rlib.wilson(k89, 10)[0]:.2f}-{rlib.wilson(k89, 10)[1]:.2f}, S89c {TAB['BASE']['wilson']}; pooled "
    f"{k89 + TAB['BASE']['br']}/20 Wilson {rlib.wilson(k89 + TAB['BASE']['br'], 20)[0]:.2f}-{rlib.wilson(k89 + TAB['BASE']['br'], 20)[1]:.2f}; "
    f"Fisher one-sided P(rerun >= {TAB['BASE']['br']} | pooled) {rlib.fisher1(TAB['BASE']['br'], 10, k89, 10):.3f}")
say(f"   S89 ORACLE transitions (validity borrowed): {sorted((s, r['transition']) for (a, s), r in s89.items() if a == 'ORACLE')}")

# ── 3. H's report() and the report writer, exec'd ───────────────────────────
say(f"\n== 3. H's report() (S89c, S89b) and explore_h_report.main (batch 11) exec'd on the stores at {HREF}")


def stub(name, **kw):
    m = types.ModuleType(name)
    m.__dict__.update(kw)
    sys.modules[name] = m
    return m


def fresh_state():
    return dict(batch_cpu=None, shas={}, other_cpus=[])


ec = stub("explore_common", load_store=lambda name: ST[name], cpu_model=lambda: CPU, OUT_DIR="/nonexistent-out")
c16 = stub("explore_common16", STATE=fresh_state(), ec=ec)
for s in extract(HREF, "explore_common16.py", ["key", "cpus_for", "pick", "runs_of", "med"]):
    exec(compile(s, "explore_common16.py@H", "exec"), c16.__dict__)
stub("explore_muon_scale")
stub("explore_h_collapse")
stub("explore_h_allmass2", NAME="H/allmass2")
hc = stub("explore_h_common")
exec(compile(show(HREF, "explore_h_common.py"), "explore_h_common.py@H", "exec"), hc.__dict__)


def load_mod(name, ref, path):
    m = stub(name)
    m.__file__ = f"/nonexistent/{path}"
    exec(compile(show(ref, path), f"{path}@{ref}", "exec"), m.__dict__)
    return m


m89 = load_mod("explore_h_s89", HREF, "explore_h_s89.py")
mc = load_mod("explore_h_s89c", HREF, "explore_h_s89c.py")
mb = load_mod("explore_h_s89b", HREF, "explore_h_s89b.py")
c16.STATE.update(batch_cpu=CPU, shas={"H/s89c": ST["H/s89c"]["meta"]["code_shas"][-1],
                                      "H/s89b": ST["H/s89b"]["meta"]["code_shas"][-1]})
resc, gotc = mc.report()
say(f"   S89c report(): readings {resc['readings']}; complete {resc['complete']}")
for a in ARMS_C:
    x, t = resc["arms"][a], TAB[a]
    ag = (x["br"] == t["br"] and x["v1"] == t["v1"] and x["v2"] == t["v2"] and x["trans"] == t["trans"] and x["ops"] == t["ops"]
          and x["op_ok"] == t["op_ok"] and x["flat"] == t["flat"] and x["one_to_one_before"] == t["o2o"]
          and x["capped"] == t["capped"] and x["last_fire"] == t["last"] and x["acc_end"] == t["acc_end"])
    say(f"   {a}: report() summary == recomputation: {ag}")
for k, cc in resc["cmp"].items():
    x, y = k.split(" vs ")
    say(f"   cmp {k}: b {cc['b']} c {cc['c']} p {cc['p']:.4g} p_rev {cc['p_rev']:.4g}; mine {CMP[(x, y)]}: {(cc['b'], cc['c']) == CMP[(x, y)]}")
resb, gotb = mb.report()
say(f"   S89b report(): readings {resb['readings']}; oracle {resb['oracle']}; complete {resb['complete']}; valid {resb['valid']}; "
    f"learned runs {sum(len(gotb[a]) for a in mb.LEARNED)}")

CAPT = {}


class _Cap(io.StringIO):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def close(self):
        CAPT[os.path.basename(self.p)] = self.getvalue()
        super().close()

    def __exit__(self, *a):
        self.close()


NOTES = show(HREF, "explore_out/H/report_11_notes.md")


def fake_open(p, mode="r", *a, **k):
    if "w" in mode:
        return _Cap(p)
    if p.endswith("checks.json"):
        return io.StringIO(show(HREF, "explore_out/H/checks.json"))
    if p == "NOTES":
        return io.StringIO(NOTES)
    raise RuntimeError(p)


rep = stub("explore_h_report")
rep.__file__ = "/nonexistent/explore_h_report.py"
exec(compile(show(HREF, "explore_h_report.py"), "explore_h_report.py@H", "exec"), rep.__dict__)
rep.__dict__["open"] = fake_open
rep.__dict__["print"] = lambda *a, **k: None
c16.STATE.update(fresh_state())
argv = sys.argv
sys.argv = ["explore_h_report.py", "11", "--notes", "NOTES"]
rep.main()
sys.argv = argv
r11 = show(HREF, "explore_out/H/report_11.md")
g11 = CAPT.get("report_11.md")
say(f"   regenerated report_11.md (with report_11_notes.md as --notes) == committed: {g11 == r11}")
if g11 != r11:
    for i, (x, y) in enumerate(zip(g11.split("\n"), r11.split("\n"))):
        if x != y:
            say(f"     line {i + 1}: gen {x[:140]!r}\n               rep {y[:140]!r}")
say(f"   regenerated s89c_runs.md == committed: {CAPT.get('s89c_runs.md') == show(HREF, 'explore_out/H/s89c_runs.md')}")

# ── 4. descriptive claims (S89c) ─────────────────────────────────────────────
say("\n== 4. Descriptive claims, S89c")
p12 = sorted(round(c["acc"], 3) for r in G["BASE_LATE"].values() for c in r["checks"] if c["step"] == 12000)
say(f"   BASE_LATE probe at the 12000 check, 10 runs: {p12}")
fl = [(s, c["step"], round(c["acc"], 3)) for s, r in G["BASE_LATE"].items() for c in r["checks"] if c.get("fired") and c["acc"] < 0.2]
say(f"   BASE_LATE firings at probe < 0.2: {len(fl)} of {TAB['BASE_LATE']['ops']}: {fl}")
first_fire = {a: sorted(min([c['step'] for c in r['checks'] if c.get('fired')], default=0) for r in G[a].values()) for a in ARMS_C}
say(f"   first firing per run: {first_fire}")


def first_at(r, th):
    return next((x[0] for x in r["curve"] if x[1] >= th), None)


for a in ARMS_C:
    f3 = sorted(first_at(r, 0.3) for r in G[a].values())
    say(f"   {a}: held-out first >= 0.3 at {f3} (range {f3[0]}-{f3[-1]})")
for s in SEEDS_C:
    x, y = G["BASE"][s], G["BASE_CAP6"][s]
    same = all(x[k] == y[k] for k in ("curve", "stats", "end", "grad", "checks"))
    fx = [c["step"] for c in x["checks"] if c.get("fired")]
    fy = [c["step"] for c in y["checks"] if c.get("fired")]
    cd = next((a_[0] for a_, b_ in zip(x["curve"], y["curve"]) if a_ != b_), None)
    say(f"   seed {s}: BASE fired {fx}; CAP6 fired {fy}; records equal (curve, stats, end, grad, checks): {same}; first "
        f"differing curve step {cd}; CAP6 4th firing {fy[3] if len(fy) > 3 else None}; {NS['outcome'](x)} / {NS['outcome'](y)}")
ext = [(s, c["step"], bool(c["op_ok"])) for s, r in G["BASE_CAP6"].items()
       for c in [c for c in r["checks"] if c.get("fired")][3:]]
say(f"   CAP6 firings beyond the third: {len(ext)} (ok {sum(o for _, _, o in ext)}), steps {sorted(t for _, t, _ in ext)}; "
    f"runs with a 4th firing: {sorted({s for s, _, _ in ext})}")
nb = [s for s in SEEDS_C if not BR["BASE_CAP6"][s]]
say(f"   CAP6 runs not bound {nb}: outcomes {dict(Counter(NS['outcome'](G['BASE_CAP6'][s]) for s in nb))}; final acc "
    f"{min(G['BASE_CAP6'][s]['acc'] for s in nb):.3f}-{max(G['BASE_CAP6'][s]['acc'] for s in nb):.3f}; all CAP6 firings "
    f"{TAB['BASE_CAP6']['ops']} ok {TAB['BASE_CAP6']['op_ok']}; fired steps on the 8 seeds "
    f"{sorted(c['step'] for s in nb for c in G['BASE_CAP6'][s]['checks'] if c.get('fired'))}")
capc = [(s, c["step"]) for s, r in G["BASE_CAP6"].items() for c in r["checks"] if c.get("blocked_by") == "cap"]
say(f"   CAP6 cap-blocked checks: {capc}")
for a in ARMS_C:
    fails = [s for s in SEEDS_C if not BR[a][s]]
    lastmax = [s for s in fails if G[a][s]["curve"][-1][1] >= max(x[1] for x in G[a][s]["curve"])]
    strict = [s for s in fails if G[a][s]["curve"][-1][1] > max(x[1] for x in G[a][s]["curve"][:-1])]
    say(f"   {a}: failures {len(fails)}; last evaluation is the curve's maximum (>=) on {len(lastmax)} {lastmax}; strictly "
        f"above every earlier one on {len(strict)}")
wall = {n: sum(r["secs_wall"] for r in ST[n]["runs"].values()) / 3600 for n in ("H/s89c", "H/s89b")}
cpu = {n: sum(r["secs"] for r in ST[n]["runs"].values()) / 3600 for n in ("H/s89c", "H/s89b")}
say(f"   runtime: secs_wall sum {wall}; secs sum {cpu}")
say(f"   checks.json: H/s89c {json.loads(show(HREF, 'explore_out/H/checks.json')).get('H/s89c')}")
say(f"                H/s89b {json.loads(show(HREF, 'explore_out/H/checks.json')).get('H/s89b')}")
for name, f in (("s89c", "explore_out/H/s89c.log"), ("s89b_oracle", "explore_out/H/s89b_oracle.log")):
    lg = show(HREF, f)
    say(f"   {name}.log: CHECK rows {sum(1 for ln in lg.split(chr(10)) if ln.strip().startswith('CHECK '))}, "
        f"FAIL rows {sum(1 for ln in lg.split(chr(10)) if ln.strip().startswith('CHECK ') and ln.rstrip().endswith('FAIL'))}; "
        f"'resumed after' lines {lg.count('resumed after')}")
    for ln in lg.split("\n"):
        if "code SHA" in ln and "modules" in ln:
            mods = ln.split("):", 1)[1].split()
            say(f"     code-SHA module list: {len(mods)} modules; parent module in it: "
                f"{[m for m in mods if m in ('explore_h_s89c', 'explore_h_s89b', 'explore_h_s89', 'explore_h_oc', 'explore_h_run')]}")
            break
lg = show(HREF, "explore_out/H/s89c.log")
logged = Counter()
for ln in lg.split("\n"):
    if "H/s89c" in ln and " lr 0.001" in ln:
        mm = __import__("re").search(r"H/s89c\s+(\w+)\s+seed (\d+)", ln)
        logged[(mm.group(1), int(mm.group(2)))] += 1
missing = sorted(k for k in C if k not in logged)
say(f"   S89c runs with a completion line in s89c.log: {len(logged)} distinct ({sum(logged.values())} lines); runs without one "
    f"{len(missing)}: {missing}")
# each record at its first commit equals the head's
for f in (F_C, F_B):
    seen, changed = {}, []
    for rv in git("rev-list", "--reverse", f"{S89C_PRE}..{HEAD}", "--", f).split():
        for k, r in json.loads(show(rv, f))["runs"].items():
            if k not in seen:
                seen[k] = rv
                if r != ST["H/s89c" if f == F_C else "H/s89b"]["runs"].get(k):
                    changed.append(k)
    say(f"   {f}: keys seen {len(seen)}; first committed version differs from head: {changed or 'none'}; dropped "
        f"{sorted(set(seen) - set(ST['H/s89c' if f == F_C else 'H/s89b']['runs'])) or 'none'}")

# ── 5. S89b's records ───────────────────────────────────────────────────────
say("\n== 5. S89b ORACLE_DELTA")
for (a, s), r in sorted(B.items()):
    cv = {x[0]: round(x[1], 3) for x in r["curve"]}
    acc = [x[1] for x in r["curve"]]
    say(f"   {a} {s}: outcome {NS['outcome'](r)}; transition {r['transition']}; at 1200/9600/10800/12000/24000/end "
        f"{[cv.get(t) for t in (1200, 9600, 10800, 12000, 24000)]} / {r['acc']:.3f}; last eval is max {acc[-1] >= max(acc)}; "
        f"last three {[round(v, 3) for v in acc[-3:]]}; stream_acc {min(r['end']['stream_acc']):.3f}-{max(r['end']['stream_acc']):.3f}; "
        f"map {r['end']['ch_map']}; {r['secs_wall'] / 60:.1f} min")
    say(f"     curve to 14400: {[(x[0], round(x[1], 3)) for x in r['curve'] if x[0] <= 14400]}")
xd = json.loads(show("8ae660d", "results/X/delta_recipe_results.json"))["runs"]
say(f"   X's CEIL_DELTA at 8ae660d (pool 4): {sorted((k, r['transition']) for k, r in xd.items() if k.startswith('CEIL_DELTA'))}")
s87 = json.loads(show(HREF, "explore_out/H/allmass2_results.json"))["runs"]
say(f"   S87a ORACLE (Hebbian, pool 4): {sorted((k.split('|')[1], r['transition']) for k, r in s87.items() if k.startswith('ORACLE|'))}")
say(f"   S89b gate: committed validity = ORACLE_DELTA binds 2/2 (b:34-35); bound {sum(r['transition'] is not None for r in B.values())}/2 "
    f"-> UNTESTED; P(2/2) at a true oracle rate q: " + ", ".join(f"q={q}: {q * q:.2f}" for q in (0.5, 0.7, 0.9)))

# ── 6. the gate-lookup fix ──────────────────────────────────────────────────
say(f"\n== 6. s89c_schedule() before ({S89B_PRE}) and after ({HREF}) the fix, with the harness's STATE as configure() sees it "
    f"(batch_cpu None, shas {{}})")
out = {}
for ref in (S89B_PRE, HREF):
    c16.STATE.clear()
    c16.STATE.update(fresh_state())
    m = load_mod("explore_h_s89b", ref, "explore_h_s89b.py")
    out[ref] = m.s89c_schedule()
    say(f"   {ref}: s89c_schedule() -> {out[ref]}; STATE after {c16.STATE}")
src0, src1 = show(S89B_PRE, "explore_h_s89b.py"), show(HREF, "explore_h_s89b.py")
t0, t1 = ast.parse(src0), ast.parse(src1)
f0 = {n.name: ast.dump(n) for n in t0.body if isinstance(n, ast.FunctionDef)}
f1 = {n.name: ast.dump(n) for n in t1.body if isinstance(n, ast.FunctionDef)}
a0 = {ast.get_source_segment(src0, n) for n in t0.body if isinstance(n, ast.Assign)}
a1 = {ast.get_source_segment(src1, n) for n in t1.body if isinstance(n, ast.Assign)}
say(f"   functions changed: {[k for k in f1 if f0.get(k) != f1[k]]}; module-level assignments changed: {len(a0 ^ a1)}; "
    f"docstring changed: {ast.get_docstring(t0) != ast.get_docstring(t1)}")
say(f"   explore_h_s89b_child.py changed {S89B_PRE}..{HREF}: {show(S89B_PRE, 'explore_h_s89b_child.py') != show(HREF, 'explore_h_s89b_child.py')}")
say(f"   S89b store at bbb8df93 == at {HREF}: {show('bbb8df93', F_B) == show(HREF, F_B)}")
say(f"   s89b.log (added at {HREF}): {show(HREF, 'explore_out/H/s89b.log').strip()!r}")
say(f"   driver order: configure() before STATE['batch_cpu'] is set: "
    f"{show(HREF, 'explore_h_run.py').find('m.configure()') < show(HREF, 'explore_h_run.py').find('c16.STATE[\"batch_cpu\"] = cpu')}")

# ── 7. pre-registration order ───────────────────────────────────────────────
say(f"\n== 7. Commits {S89C_PRE}^..{HREF}: time, records per store, code files differing from their first version in the range")
CODE = ("explore_h_s89c.py", "explore_h_s89c_child.py", "explore_h_s89b.py", "explore_h_s89b_child.py", "explore_h_oc.py",
        "explore_h_run.py", "explore_h_report.py", "explore_h_s89.py", "explore_h_s89_child.py", "explore_h_common.py",
        "explore_common16.py", "explore_common.py")


def blob(rv, p):
    try:
        return subprocess.check_output(("git", "-C", ROOT, "rev-parse", f"{rv}:{p}"), stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        return None


first = {}
firstrec = {}
for rv in git("rev-list", "--reverse", f"{S89C_PRE}^..{HEAD}").split():
    t = git("log", "-1", "--format=%cI", rv).strip()
    nC = len(json.loads(show(rv, F_C) or '{"runs": {}}')["runs"])
    nB = len(json.loads(show(rv, F_B) or '{"runs": {}}')["runs"])
    for n, v in (("C", nC), ("B", nB)):
        if v and n not in firstrec:
            firstrec[n] = (rv[:8], t)
    ch = []
    for p in CODE:
        b_ = blob(rv, p)
        if p not in first and b_ is not None:
            first[p] = b_
        elif p in first and b_ != first[p]:
            ch.append(p)
    say(f"   {rv[:8]} {t[:19]} s89c {nC:2d} s89b {nB} changed: {ch or '-'}")
say(f"   first S89c record: {firstrec.get('C')}; first S89b record: {firstrec.get('B')}")
say(f"   referee review screen_review_s89bc_H.md: {git('log', '-1', '--format=%h %cI', '43cea898').strip()}")
hc_c = show(HREF, "explore_h_s89c.py") + show(HREF, "explore_h_oc.py")
say("   referee's S89c fix text at the head: " + ", ".join(f"{k} {k in hc_c}" for k in (
    "does not replicate", "neither schedule change", "POST-HOC", "read off", "7939dd4")))
hb = show(HREF, "explore_h_s89b.py")
say("   referee's S89b fix text at the head: " + ", ".join(f"{k} {k in hb}" for k in ("EXPECTED RATE", "LABELS:", "mixed schedules", "53578896")))
rd = show(HREF, "explore_out/README.md")
say(f"   README 'the four bound learned runs' still present: {'the four bound learned runs' in rd}; report_10 'seven of the nine': "
    f"{'seven of the nine' in show(HREF, 'explore_out/H/report_10.md')}; '3c772c3' in explore_h_s89.py: {'3c772c3' in show(HREF, 'explore_h_s89.py')}")

# ── 8. operating characteristics quoted in report 11 ────────────────────────
say("\n== 8. explore_h_oc (exec'd) at the rates quoted in report 11 line 23, BASE 0.1, rho 0 / 0.5 / 1")
for lab, rates in (("both 0.3", (0.1, 0.3, 0.3)), ("both 0.5", (0.1, 0.5, 0.5)), ("both 0.7", (0.1, 0.7, 0.7)),
                   ("one 0.5", (0.1, 0.5, 0.1))):
    rr = [ocns["oc"](rates, ocns["rule_s89c"], rho=x) for x in (0.0, 0.5, 1.0)]
    say(f"   {lab}: " + "; ".join(f"{nm[4:12]} " + "/".join(f"{d.get(nm, 0):.2f}" for d in rr)
                                  for nm in ("the schedule was the limit", "the channel is the limit")))
for b in (0.1, 0.2, 0.3):
    say(f"   P(BASE rerun <= 3/10 | true {b}) = {sum(math.comb(10, i) * b ** i * (1 - b) ** (10 - i) for i in range(4)):.3f}")
say(f"   P(BASE_LATE = BASE_CAP6 = BASE pattern as observed) -- descriptive only; CAP6 equals BASE wherever BASE fired < 4 times")

# ── 9. small follow-ups ─────────────────────────────────────────────────────
say("\n== 9. Follow-ups")
r2912 = B[("ORACLE_DELTA", 2912)]
say(f"   ORACLE_DELTA 2912 last four evaluations (raw): {[(x[0], x[1]) for x in r2912['curve'][-4:]]}")
for s in (2926, 2928):
    x, y = G["BASE"][s], G["BASE_CAP6"][s]
    say(f"   seed {s}: BASE {NS['outcome'](x)} acc {x['acc']:.3f} map {x['end']['ch_map']} min stream {min(x['end']['stream_acc']):.3f}; "
        f"CAP6 {NS['outcome'](y)} acc {y['acc']:.3f} map {y['end']['ch_map']} min stream {min(y['end']['stream_acc']):.3f}")
lg0 = show("568a12f0", "explore_out/H/s89b_oracle.log")
say(f"   s89b_oracle.log at 568a12f0 ({len(lg0.splitlines())} lines); git in its header lines: "
    f"{[ln.strip()[:90] for ln in lg0.splitlines() if 'git ' in ln][:3]}")
fails = [(a, s) for a in ARMS_C for s in SEEDS_C if not BR[a][s]]
say(f"   all S89c failures {len(fails)}; last evaluation the maximum (>=) on "
    f"{sum(G[a][s]['curve'][-1][1] >= max(x[1] for x in G[a][s]['curve']) for a, s in fails)}")
