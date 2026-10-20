#!/usr/bin/env python3
"""check_H_report10.py — Referee V's check of session H's report 10 (S89) on claude/explore-H, and the search for an
S89b screen.

    python3 -I docs/review_2026-10/check_H_report10.py > /dev/stdout

Run from the repository root after `git fetch`. Prints only; writes nothing. Every file is read with `git show`; no
session code is run as a program and nothing is trained. Pure functions are cut out by ast (the main line's
test_stream_recipe.outcome and what it calls) or exec'd from their source with the harness stubbed (H's report() and
summarize() from explore_h_s89.py, H's explore_h_common, explore_common16's runs_of / pick, and explore_h_report's
s89_section, whose file writes are captured in memory). docs/review_2026-10/rlib.py gives an independent re-implementation
of the outcome (transition from the curve, the map from stream_gate).
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

H = "origin/claude/explore-H"
MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
MAIN9C = "9c5939e"
PRE = "d4a9a7d"
STORE_F = "explore_out/H/s89_results.json"
S87A_F = "explore_out/H/allmass2_results.json"
S79_F = "explore_out/H/constants_results.json"
CPU = "Intel(R) Xeon(R) Processor @ 2.10GHz"


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a).decode()


def show(ref, path):
    try:
        return subprocess.check_output(("git", "-C", ROOT, "show", f"{ref}:{path}"), stderr=subprocess.DEVNULL).decode()
    except subprocess.CalledProcessError:
        return None


def say(*a):
    print(*a, flush=True)


HEAD = git("rev-parse", H).strip()
say(f"== 0. Pins: {H} = {HEAD[:8]} ({git('log', '-1', '--format=%cI %s', H).strip()})")
say(f"   main line {MAIN} = {git('rev-parse', '--short=8', MAIN).strip()}")

# ── the main line's outcome, cut out by ast ──────────────────────────────────
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


NS, SRC_HEAD = mainline_ns(MAIN)
NS9, SRC_9C = mainline_ns(MAIN9C)
say(f"   main-line outcome functions (fail_class, fail_class_k, shared_max, merged, ROUTED_ACC, routed, outcome, wilson): "
    f"syntax trees at {MAIN9C} (the child's tree) == at the main head: {SRC_HEAD == SRC_9C}")

# ── the store ────────────────────────────────────────────────────────────────
store = json.loads(show(H, STORE_F))
RUNS = store["runs"]
say(f"\n== 1. The store {STORE_F} at {HEAD[:8]}: {len(RUNS)} runs; meta code_shas {store['meta']['code_shas']}")
by = {}
for k, r in RUNS.items():
    a, s, sha, cpu = k.split("|")
    assert (a, int(s)) not in by, k
    by[(a, int(s))] = r
    assert r["ok"] and r["attempts"] == 1 and cpu == CPU and r["code_sha"].startswith(sha), k
arms = Counter(a for a, _ in by)
say(f"   per arm: {dict(arms)}; all ok, 1 attempt, one CPU ({CPU}), one code SHA: True (asserted)")
LEARNED = ("BASE", "ALLMASS", "REPEATMASS", "NOSPLIT")
TRIG = ("BASE", "ALLMASS", "REPEATMASS")
SEEDS = tuple(range(2902, 2912))
G = {a: {s: by[(a, s)] for s in SEEDS} for a in LEARNED}
G["ORACLE"] = {s: by[("ORACLE", s)] for s in (2900, 2901)}

# outcome three ways
say("\n   outcome per record: stored field / test_stream_recipe.outcome (main head; 9c5939e) / rlib.outcome_k (transition "
    "from the curve, map from stream_gate)")
diff = []
for (a, s), r in sorted(by.items()):
    o_st, o_m, o_9, o_r = r["outcome"], NS["outcome"](r), NS9["outcome"](r), rlib.outcome_k(r)
    t_r = rlib.transition(r["curve"])
    if not (o_st == o_m == o_9 == o_r) or t_r != r["transition"] or r.get("outcome_v2", o_st) != o_st:
        diff.append((a, s, o_st, o_m, o_9, o_r, r["transition"], t_r, r.get("outcome_v2")))
say(f"   records where any of stored / main / 9c5939e / rlib outcome, stored vs curve transition, or v1 vs v2 differ: "
    f"{diff if diff else 'none'}  (rlib ties: {rlib.TIES})")
BR = {a: {s: NS["outcome"](G[a][s]) == "BOUND ROUTED" for s in G[a]} for a in G}

# ── 2. per-arm counts, recomputed ─────────────────────────────────────────────
say("\n== 2. Per arm (recomputed from records with the main line's outcome; rlib for band / Wilson / McNemar)")
TAB = {}
for a in LEARNED:
    rs = G[a]
    k = sum(BR[a].values())
    w = NS["wilson"](k, 10)
    v1 = Counter()
    for r in rs.values():
        o = NS["outcome"](r)
        if o != "BOUND ROUTED":
            v1["MERGED" if o.startswith("MERGED") else o.split(" ", 1)[1] if o.startswith("non-stream") else o] += 1
    v2 = Counter(r["outcome_v2"] for r in rs.values() if r["outcome_v2"] != "BOUND ROUTED")
    tr = sorted(r["transition"] for s, r in rs.items() if BR[a][s])
    row = dict(br=k, band=rlib.band(k, 10), wilson=f"{w[0]:.2f}-{w[1]:.2f}", v1=dict(sorted(v1.items())),
               v2=dict(sorted(v2.items())), trans=tr, med=(statistics.median(tr) if tr else None))
    if a in TRIG:
        f = [c for r in rs.values() for c in r["checks"] if c.get("fired")]
        row.update(ops=len(f), op_ok=sum(bool(c["op_ok"]) for c in f), key_ok=sum(bool(c["key_ok"]) for c in f),
                   all_ok=sum(bool(c["all_ok"]) for c in f), rep_ok=sum(bool(c["rep_ok"]) for c in f),
                   flat=sum(c["acc"] < 0.2 for c in f),
                   same_key_all=sum(c["key_cs"] == c["all_cs"] and c["key_c0"] == c["all_c0"] for c in f),
                   same_key_rep=sum(c["key_cs"] == c["rep_cs"] and c["key_c0"] == c["rep_c0"] for c in f),
                   same_all_rep=sum(c["all_cs"] == c["rep_cs"] and c["all_c0"] == c["rep_c0"] for c in f),
                   o2o=[(s, c["step"], c["op_ok"]) for s, r in rs.items() for c in r["checks"]
                        if c.get("fired") and len(set(c["map_before"])) == 8])
    TAB[a] = row
    say(f"   {a}: {row}")
oc = {s: (r["transition"], NS["outcome"](r)) for s, r in G["ORACLE"].items()}
say(f"   ORACLE: {oc}")

say("\n   pairs (b = x only, c = y only; one-sided exact McNemar, rlib.mcnemar1)")
CMP = {}
for x, y in (("ALLMASS", "BASE"), ("REPEATMASS", "BASE"), ("NOSPLIT", "BASE"), ("ALLMASS", "NOSPLIT"),
             ("REPEATMASS", "NOSPLIT"), ("ALLMASS", "REPEATMASS")):
    b = sum(BR[x][s] and not BR[y][s] for s in SEEDS)
    c = sum(BR[y][s] and not BR[x][s] for s in SEEDS)
    CMP[(x, y)] = (b, c)
    say(f"   {x} vs {y}: {sum(BR[x].values())}/10 vs {sum(BR[y].values())}/10; b {b}, c {c}; p(x>y) "
        f"{rlib.mcnemar1(b, c):.3g}, p(y>x) {rlib.mcnemar1(c, b):.3g}")
say("   bound-routed seeds: " + "; ".join(f"{a} {[s for s in SEEDS if BR[a][s]]}" for a in LEARNED))


def prompt_read(nB, cmp):
    """The prompt's rules (prompt:1216-1220) as the referee's pre-run review read them."""
    if nB < 7:
        return {"S89": "FAILS"}
    nb, nc = cmp[("NOSPLIT", "BASE")]
    moot = nc - nb <= 1
    out = {"split": "NOT NEEDED" if moot else "NEEDED"}
    for t in ("ALLMASS", "REPEATMASS"):
        b, c = cmp[(t, "BASE")]
        tb, tc = cmp[(t, "NOSPLIT")]
        out[t] = ("moot " if moot else "") + ("SURVIVES" if c - b <= 1 and tb >= 4 and tc == 0 else
                                              "NEEDS KEYPOS" if c >= 4 and b == 0 else "NONE")
    return out


say(f"   the prompt's rules on the recomputed outcomes: {prompt_read(TAB['BASE']['br'], CMP)}")

# ── 3. H's own report() and the report writer, exec'd on the records ─────────
say("\n== 3. explore_h_s89.report() (source at the head) and explore_h_report.s89_section on the stored records")


def stub(name, **kw):
    m = types.ModuleType(name)
    m.__dict__.update(kw)
    sys.modules[name] = m
    return m


def c16_src(names):
    return extract(H, "explore_common16.py", names)


ec = stub("explore_common", load_store=lambda name: store, cpu_model=lambda: CPU, OUT_DIR="/nonexistent-out")
c16 = stub("explore_common16", STATE=dict(batch_cpu=None, shas={}, other_cpus=[]), ec=ec)
for s in c16_src(["key", "cpus_for", "pick", "runs_of", "med"]):
    exec(compile(s, "explore_common16.py@H", "exec"), c16.__dict__)
stub("explore_muon_scale")
stub("explore_h_collapse")
stub("explore_h_allmass2", NAME="H/allmass2")
hc = stub("explore_h_common")
exec(compile(show(H, "explore_h_common.py"), "explore_h_common.py@H", "exec"), hc.__dict__)
s89 = stub("explore_h_s89")
s89.__file__ = "/nonexistent/explore_h_s89.py"
exec(compile(show(H, "explore_h_s89.py"), "explore_h_s89.py@H", "exec"), s89.__dict__)
CAPT = {}


class _Cap(io.StringIO):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def close(self):
        CAPT[self.p] = self.getvalue()
        super().close()

    def __exit__(self, *a):
        self.close()


def fake_open(p, mode="r", *a, **k):
    if "w" in mode:
        return _Cap(p)
    if p.endswith("checks.json"):
        return io.StringIO(show(H, "explore_out/H/checks.json"))
    raise RuntimeError(p)


rep = stub("explore_h_report")
rep.__file__ = "/nonexistent/explore_h_report.py"
src_rep = show(H, "explore_h_report.py")
exec(compile(src_rep, "explore_h_report.py@H", "exec"), rep.__dict__)
rep.__dict__["open"] = fake_open
L = []
rep.s89_section(L)
nr, h, cpus, shas = rep.runtime(s89)
c = rep.checks_line(s89.NAME)
L.append(f"\n**Runtime:** {nr} runs, {h:.1f} CPU-hours on {', '.join(cpus)}; code SHA {', '.join(shas)}. **CHECKs:** "
         + (f"all passed ({c['time']}, git {c['git']}, {c['minutes']:.1f} min)" if c else "NOT RECORDED"))
gen = "\n".join(L).split("\n")
rep10 = show(H, "explore_out/H/report_10.md").split("\n")
body = rep10[2:2 + len(gen)]
say(f"   regenerated section: {len(gen)} lines; equal to report_10.md lines 3-{2 + len(gen)}: {gen == body}")
for i, (g, b) in enumerate(zip(gen, body)):
    if g != b:
        say(f"     line {i + 3}: generated {g!r}\n               report    {b!r}")
runs_md = show(H, "explore_out/H/s89_runs.md")
say(f"   regenerated s89_runs.md equal to the committed file: {list(CAPT.values()) == [runs_md]}")
res, got = s89.report()
say(f"   report() readings: {res['readings']}; complete {res['complete']}; valid {res['valid']}; oracle {res['oracle']}")
for a in LEARNED:
    x = res["arms"][a]
    mine = TAB[a]
    agree = (x["br"] == mine["br"] and x["band"] == mine["band"] and x["trans"] == mine["trans"]
             and x["v1"] == mine["v1"] and x["v2"] == mine["v2"])
    if a in TRIG:
        agree &= all(x[k] == mine[k] for k in ("ops", "op_ok", "key_ok", "all_ok", "rep_ok", "flat", "same_key_all",
                                               "same_key_rep", "same_all_rep")) and x["one_to_one_before"] == len(mine["o2o"])
    say(f"   {a}: report() summary agrees with the recomputation: {agree}; v1v2_differ {x['v1v2_differ']}")
for k, cc in res["cmp"].items():
    x, y = k.split(" vs ")
    say(f"   report() cmp {k}: b {cc['b']} c {cc['c']} p {cc['p']:.4g} p_rev {cc['p_rev']:.4g}; mine "
        f"{CMP[(x, y)]}: {(cc['b'], cc['c']) == CMP[(x, y)]}")
say(f"   one-to-one map before a firing: " + "; ".join(f"{a} {TAB[a]['o2o']}" for a in TRIG))

# ── 4. the descriptive claims ────────────────────────────────────────────────
say("\n== 4. Descriptive claims in report 10's notes and the README section")
s87 = json.loads(show(H, S87A_F))["runs"]
s87o = sorted((int(k.split('|')[1]), r["transition"], r["outcome"]) for k, r in s87.items() if k.startswith("ORACLE|"))
s87b = sorted(r["transition"] for k, r in s87.items() if k.startswith("BASE|") and NS["outcome"](r) == "BOUND ROUTED")
s87bn = sum(1 for k in s87 if k.startswith("BASE|"))
say(f"   S87a ORACLE: {s87o}; median {statistics.median([t for _, t, _ in s87o])}; S89 ORACLE median "
    f"{statistics.median([t for t, _ in oc.values()])}; ratio "
    f"{statistics.median([t for t, _ in oc.values()]) / statistics.median([t for _, t, _ in s87o]):.2f}")
say(f"   S87a BASE bound-routed transitions ({len(s87b)} of {s87bn} runs): min {min(s87b)} max {max(s87b)} median "
    f"{statistics.median(s87b)}")
bound = [(a, s, G[a][s]["transition"], 43200 - G[a][s]["transition"]) for a in LEARNED for s in SEEDS if BR[a][s]]
say(f"   bound-routed learned runs: {len(bound)} runs on seeds {sorted({s for _, s, _, _ in bound})}: {bound}")
first = Counter(a for a in TRIG for r in G[a].values() if [c["step"] for c in r["checks"] if c.get("fired")][:1] == [4800])
say(f"   triggered runs whose first firing is at 4800: {dict(first)} (of 10 each)")
fire_acc_4800 = sorted(round(c["acc"], 3) for a in TRIG for r in G[a].values() for c in r["checks"]
                       if c.get("fired") and c["step"] == 4800)
say(f"   probe accuracy at the 4800 firings: min {fire_acc_4800[0]} max {fire_acc_4800[-1]}")
spl = {a: Counter(r["splits"] for r in G[a].values()) for a in TRIG}
last = sorted(max(c["step"] for c in r["checks"] if c.get("fired")) for a in TRIG for r in G[a].values())
say(f"   splits per run: {spl}; step of the third firing over the 30 runs: min {last[0]} max {last[-1]}")
capb = {a: sum(1 for r in G[a].values() for c in r["checks"] if c.get("blocked") and c.get("blocked_by") == "cap")
        for a in TRIG}
othb = {a: Counter(c.get("blocked_by") for r in G[a].values() for c in r["checks"] if c.get("blocked")) for a in TRIG}
say(f"   checks blocked by the cap per arm: {capb}; all blocks by cause: {othb}")
for a in LEARNED:
    unb = sorted(round(r["acc"], 3) for s, r in G[a].items() if not BR[a][s])
    allf = sorted(round(r["acc"], 3) for r in G[a].values())
    a24 = sorted(round(hc.acc_at(r, 24000), 3) for r in G[a].values())
    sm = Counter(rlib.shared_max(r) for s, r in G[a].items() if not BR[a][s])
    say(f"   {a}: final acc of unbound runs {unb[0]}-{unb[-1]}; all runs {allf[0]}-{allf[-1]}; held-out at 24000 "
        f"{a24[0]}-{a24[-1]}; shared_max of unbound runs {dict(sorted(sm.items()))}")
trig24 = sorted(hc.acc_at(r, 24000) for a in TRIG for r in G[a].values())
trig_unb = sorted(r["acc"] for a in TRIG for s, r in G[a].items() if not BR[a][s])
say(f"   triggered arms together: held-out at 24000 {trig24[0]:.3f}-{trig24[-1]:.3f}; final acc of unbound "
    f"{trig_unb[0]:.3f}-{trig_unb[-1]:.3f}")
same, firstdiff = [], {}
for s in SEEDS:
    x, y = G["ALLMASS"][s], G["REPEATMASS"][s]
    eqc = x["curve"] == y["curve"] and x["stats"] == y["stats"] and x["end"] == y["end"] and x["grad"] == y["grad"]
    if eqc:
        same.append(s)
    else:
        fx = [c for c in x["checks"] if c.get("fired")]
        fy = [c for c in y["checks"] if c.get("fired")]
        firstdiff[s] = [(i + 1, cx["step"], (cx["op_cs"], cx["op_c0"]), (cy["op_cs"], cy["op_c0"]))
                        for i, (cx, cy) in enumerate(zip(fx, fy)) if (cx["op_cs"], cx["op_c0"]) != (cy["op_cs"], cy["op_c0"])][:1]
say(f"   ALLMASS and REPEATMASS records equal (curve, stats, end, grad) on seeds {same} ({len(same)} of 10); first "
    f"differing firing elsewhere (index, step, ALLMASS pair, REPEATMASS pair): {firstdiff}; outcomes there: "
    f"{ {s: (NS['outcome'](G['ALLMASS'][s]), NS['outcome'](G['REPEATMASS'][s])) for s in firstdiff} }")
ov = G["REPEATMASS"][2902]["overlap"]
ovs = {s: G["REPEATMASS"][s]["overlap"] == ov for s in SEEDS}
say(f"   REPEATMASS probe overlap (same on every seed: {all(ovs.values())}): repeat {ov['repeat']}/{ov['positions']} = "
    f"{ov['repeat'] / ov['positions']:.4f}; VAL {ov['repeat_val']}/{ov['all_val']} = {ov['repeat_val'] / ov['all_val']:.4f}; "
    f"key share of the set {ov['key_in_repeat'] / ov['repeat']:.4f}; docstring corpus figures 385149/401408 = "
    f"{385149 / 401408:.4f}, VAL 114813/131072 = {114813 / 131072:.4f}, key share 131072/385149 = {131072 / 385149:.4f}")
wall = sum(r["secs_wall"] for r in RUNS.values()) / 3600
cpu_h = sum(r["secs"] for r in RUNS.values()) / 3600
say(f"   runtime: {len(RUNS)} runs, secs_wall sum {wall:.2f} h (secs sum {cpu_h:.2f} h)")
s79 = json.loads(show(H, S79_F))["runs"]
for a in ("BASE", "CAP6", "CAP1"):
    rs = [r for k, r in s79.items() if k.split("|")[0] == a]
    say(f"   S79 {a}: bound routed {sum(NS['outcome'](r) == 'BOUND ROUTED' for r in rs)}/{len(rs)}")
# saved records unchanged after the restart: each record at its first commit equals the head's
firstseen, changed = {}, []
for rv in git("rev-list", "--reverse", f"{PRE}..{HEAD}", "--", STORE_F).split():
    for k, r in json.loads(show(rv, STORE_F))["runs"].items():
        if k not in firstseen:
            firstseen[k] = rv
            if r != RUNS.get(k):
                changed.append(k)
say(f"   records whose first committed version differs from the head's: {changed or 'none'} ({len(firstseen)} keys seen; "
    f"keys ever seen but not at the head: {sorted(set(firstseen) - set(RUNS)) or 'none'})")
for a in TRIG:
    lo = sum(G["NOSPLIT"][s]["acc"] < G[a][s]["acc"] for s in SEEDS)
    say(f"   final held-out accuracy, NOSPLIT below {a} on {lo} of 10 seeds; medians NOSPLIT "
        f"{statistics.median(r['acc'] for r in G['NOSPLIT'].values()):.3f}, {a} {statistics.median(r['acc'] for r in G[a].values()):.3f}")
cfgs = Counter((r["k"], r["iters"], r.get("pool"), json.dumps(r.get("trig_cfg"), sort_keys=True)) for a in TRIG for r in G[a].values())
say(f"   triggered runs' (k, iters, pool, trig_cfg): {dict(cfgs)}")
say(f"   checks.json H/s89: {json.loads(show(H, 'explore_out/H/checks.json')).get('H/s89')}")

# ── 5. pre-registration order ────────────────────────────────────────────────
say(f"\n== 5. Commits {PRE}..{HEAD[:8]} (committer time UTC) and the S89 code at each")
CODE = ("explore_h_s89.py", "explore_h_s89_child.py", "explore_h_common.py", "explore_common16.py", "explore_common.py",
        "explore_h_ops.py", "explore_split_target_child.py", "explore_h_report.py")
base = {p: git("rev-parse", f"{PRE}:{p}").strip() for p in CODE}
revs = git("rev-list", "--reverse", f"{PRE}^..{HEAD}").split()
first_learned = None
for rv in revs:
    t = git("log", "-1", "--format=%cI %s", rv).strip()
    files = git("diff-tree", "--no-commit-id", "--name-only", "-r", rv).split()
    changed = [p for p in CODE if git("rev-parse", f"{rv}:{p}").strip() != base[p]]
    st = show(rv, STORE_F)
    nlearn = sum(1 for k in json.loads(st)["runs"] if k.split("|")[0] in LEARNED) if st else 0
    if nlearn and first_learned is None:
        first_learned = rv
    say(f"   {rv[:8]} {t[:25]} learned records {nlearn:2d}; code differing from {PRE}: {changed or '-'}; files {files}")
say(f"   first commit holding a learned-arm record: {first_learned[:8]}")
say(f"   learned runs' record 'git' fields: {sorted({r['git'] for (a, s), r in by.items() if a in LEARNED})[:3]} ...; "
    f"code_sha {sorted({r['code_sha'][:12] for r in RUNS.values()})}")
rv = git("log", "-1", "--format=%h %cI %s", "ee5217bc").strip()
say(f"   the referee's screen review: {rv[:120]}")
s89h = show(H, "explore_h_s89.py")
probes = {1: "rare key types", 2: "not a second finding", 3: "with no other name", 4: "Operating characteristics",
          5: "one_to_one_before)", 6: 'MAINREF = "628eb7c"', 7: "50dc8e4", 8: "printed True on curve"}
say("   fix text present in explore_h_s89.py at the head: " + ", ".join(f"fix {k} {v in s89h}" for k, v in probes.items()))
say(f"   s89:283 at the head: {s89h.splitlines()[282].strip()!r}")
say(f"   s89:106 at the head: {s89h.splitlines()[105].strip()!r}")

# ── 6. S89b ──────────────────────────────────────────────────────────────────
say("\n== 6. S89b: every ref, every path and every commit message")
refs = git("for-each-ref", "--format=%(refname)").split()
hits = []
for r in refs:
    try:
        out = git("grep", "-l", "-I", "-i", "-E", r"s89b|s89 ?\(b\)|s89-b|s89_b", r)
        hits += [ln for ln in out.split("\n") if ln]
    except subprocess.CalledProcessError:
        pass
msgs = git("log", "--all", "--format=%h %s", "-i", "-E", r"--grep=s89b|s89 ?\(b\)").strip()
paths = sorted({p for p in git("log", "--all", "--name-only", "--format=").split()
                if "s89" in p.lower() and p.startswith("explore_h")})
say(f"   refs searched {len(refs)}; content hits {hits or 'none'}; commit messages {msgs or 'none'}")
say(f"   explore_h* paths with 's89' in any commit of any ref: {paths}")
say(f"   explore_h_*.py on the head added after {PRE}: "
    f"{git('diff', '--name-only', '--diff-filter=A', PRE, HEAD, '--', 'explore_h_*.py').split() or 'none'}")

# seeds left in the round-6 block, and the BASE >= 7/10 gate's pass probability
import re as _re
pat = _re.compile(r'"[A-Za-z0-9_]+\|(29[0-5][0-9])\|')
used = Counter()
for r in [x for x in git("for-each-ref", "--format=%(refname)", "refs/remotes").split() if not x.endswith("/HEAD")]:
    for path in git("ls-tree", "-r", "--name-only", r).split():
        if path.endswith((".json", ".jsonl")):
            txt = show(r, path) or ""
            for m in pat.finditer(txt):
                used[(r.split("/")[-1], path, int(m.group(1)))] += 1
blk = sorted({x[2] for x in used})
say(f"   record keys with a seed in 2900-2959 on the remote heads: seeds {blk[0]}-{blk[-1] if blk else ''} in "
    f"{sorted({(b, p) for b, p, _ in used})}; free after S89: 2912-2959 unless listed: {[x for x in blk if x >= 2912] or 'none'}")
say("   P(BASE >= 7/10) at a true BASE rate p: " + ", ".join(
    f"p={p_}: {sum(math.comb(10, i) * p_ ** i * (1 - p_) ** (10 - i) for i in range(7, 11)):.3f}" for p_ in (0.1, 0.3, 0.5, 0.7, 0.8, 0.9)))
