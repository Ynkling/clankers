#!/usr/bin/env python3
"""check_R_v4.py -- the referee's fourth-pass check of session R's Phase 4 as folded into the Revision 9
draft, fourth version (claude/rev9-draft at c66cc76; tex4:NNN = multichannel_hebbian_report_v9.tex there).

    python3 docs/review_2026-10/rev9/check_R_v4.py        (stdlib only; prints; writes nothing)

Records are read with `git show <ref>:<path>` from claude/replicate-R (1f8be04, interim 0364e28,
pre-registration 464be8d). R's outcome is recomputed with replication_check.py's r_transition /
r_bound_routed, taken from that file's source by ast (the module is not imported, so it does not run).
"""
import ast
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REV))
sys.path.insert(0, REV)
from rlib import load as rl_load, runs as rl_runs, bound_routed as ml_bound_routed, ch_map, wilson, mcnemar1  # noqa: E402

R_FINAL, R_INTERIM, R_PREREG, R_P3 = "1f8be04", "0364e28", "464be8d", "100dced"
TEX4, TEX3, AMEND = "c66cc76", "a5f67a2", "db22857"

# r_transition / r_bound_routed from replication_check.py, without running that script
_src = open(os.path.join(REV, "replication_check.py")).read()
_ns = {}
for node in ast.parse(_src).body:
    if isinstance(node, ast.FunctionDef) and node.name in ("r_transition", "r_bound_routed"):
        exec(compile(ast.Module([node], []), "replication_check.py", "exec"), _ns)
r_transition, r_bound_routed = _ns["r_transition"], _ns["r_bound_routed"]


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a).decode()


def jsonl(ref, path):
    return [json.loads(l) for l in git("show", f"{ref}:{path}").splitlines() if l.strip()]


def cdf(n, k, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def ctime(ref):
    return datetime.fromisoformat(git("log", "-1", "--format=%aI", ref).strip())


def H(t):
    print("\n== " + t)


tex4 = git("show", f"{TEX4}:multichannel_hebbian_report_v9.tex").split("\n")


def where(pat):
    return [i + 1 for i, l in enumerate(tex4) if re.search(pat, l)]


# ---------------------------------------------------------------------------------------------------
H("0. Which rule: phase4.py at 464be8d (pre-registration), 0364e28 (interim records), 1f8be04 (final)")
src = {ref: git("show", f"{ref}:phase4.py") for ref in (R_PREREG, R_INTERIM, R_FINAL)}


def rule_part(s):
    # docstring up to the RESULT block, plus all code after the docstring
    head = s.split("RESULT (")[0]
    code = s.split('"""', 2)[2]
    return head, code


for ref in (R_INTERIM, R_FINAL):
    h0, c0 = rule_part(src[R_PREREG])
    h1, c1 = rule_part(src[ref])
    print(f"  {R_PREREG} -> {ref}: design/readings text identical: {h0 == h1}; code identical: {c0 == c1}")
print("  pre-registered RESULT block at 464be8d:", repr(src[R_PREREG].split("RESULT (")[1].split("\n")[2]))
print("  rule (phase4.py docstring): 'consistent' = Wilson(split, 40 pooled) overlaps [0.835, 0.986] AND McNemar p < 0.05;"
      " 'a lower rate' = upper < 0.835; else 'neither'")
print("  phase4.jsonl at 464be8d exists:", "records/phase4.jsonl" in git("ls-tree", "-r", "--name-only", R_PREREG))
rep = git("show", f"{R_FINAL}:replicate.py")
print("  amended-prompt arms in replicate.py at 1f8be04: COPY_ONLY", "COPY_ONLY" in rep, "; RESET_ONLY", "RESET_ONLY" in rep)
amend_txt = git("show", f"{AMEND}:docs/reading/parallel_sessions_prompts.md").split("\n")
print("  amended prompt db22857:858-884, first line:", amend_txt[857][:110])
prompt_hits = [i + 1 for i, l in enumerate(amend_txt) if "27/32" in l or "914 and 915" in l or "COPY_ONLY" in l]
print("  amended prompt lines with 914/915, 27/32, COPY_ONLY:", prompt_hits)
r_tree = git("grep", "-n", "-i", "-E", "amended|27/32|COPY_ONLY|RESET_ONLY|914 and 915", R_FINAL, "--",
             "phase4.py", "replicate.py", "replicate_report.md", "analyze.py").strip() if subprocess.call(
    ("git", "-C", ROOT, "grep", "-q", "-i", "-E", "amended|27/32|COPY_ONLY|RESET_ONLY|914 and 915", R_FINAL, "--",
     "phase4.py", "replicate.py", "replicate_report.md", "analyze.py")) == 0 else ""
print("  R's branch mentions of the amended prompt (amended|27/32|COPY_ONLY|RESET_ONLY|914 and 915):", r_tree or "none")
for ref in (R_PREREG, "01bee1f", R_INTERIM, AMEND, R_FINAL, "9657edb", TEX3, TEX4):
    print(f"  {ref} {ctime(ref).isoformat()}  {git('log', '-1', '--format=%s', ref).strip()[:90]}")

# ---------------------------------------------------------------------------------------------------
H("1. Records: integrity and per-arm counts (R's outcome recomputed)")
p2 = jsonl(R_FINAL, "records/phase2.jsonl")
p3 = jsonl(R_FINAL, "records/phase3.jsonl")
p4 = jsonl(R_FINAL, "records/phase4.jsonl")
p4i = jsonl(R_INTERIM, "records/phase4.jsonl")
print(f"  lines: phase2 {len(p2)}, phase3 {len(p3)}, phase4 {len(p4)} (interim {len(p4i)})")
print("  interim records are the first", len(p4i), "final records unchanged:", p4[:len(p4i)] == p4i)
print("  phase2/phase3 unchanged since 100dced:",
      git("show", f"{R_P3}:records/phase2.jsonl") == git("show", f"{R_FINAL}:records/phase2.jsonl"),
      git("show", f"{R_P3}:records/phase3.jsonl") == git("show", f"{R_FINAL}:records/phase3.jsonl"))
log4 = git("show", f"{R_FINAL}:records/phase4.log").splitlines()
print(f"  phase4.log: {len(log4)} lines; 'jobs to run' headers: {[l for l in log4 if 'jobs to run' in l]}")
key = lambda r: (r["arm"], r["seed"])
for name, rs in (("phase2", p2), ("phase3", p3), ("phase4", p4)):
    dup = len(rs) - len({key(r) for r in rs})
    mism = [key(r) for r in rs if r_bound_routed(r) != r["bound_routed"] or r_transition(r["evals"]) != r["transition"]]
    print(f"  {name}: duplicate (arm, seed): {dup}; stored flags/transition disagreeing with the recount: {mism}")
by = {}
for ph, rs in (("P2", p2), ("P3", p3), ("P4", p4)):
    for r in rs:
        by.setdefault((ph, r["arm"]), {})[r["seed"]] = r
for (ph, arm), d in sorted(by.items()):
    c = sum(r_bound_routed(r) for r in d.values())
    lo, hi = wilson(c, len(d))
    print(f"  {ph} {arm} seeds {min(d)}-{max(d)}: bound routed {c}/{len(d)} Wilson [{lo:.3f}, {hi:.3f}]")
print("  Phase 4 arms present:", sorted({r['arm'] for r in p4}), "seeds", min(r['seed'] for r in p4), "-", max(r['seed'] for r in p4))
print("  any Phase 4 record on 914/915 or 940-951:", [key(r) for r in p4 if r["seed"] in (914, 915) or r["seed"] >= 940])

H("2. Validity (ORACLE_ADAM, 920-921)")
for s in (920, 921):
    r = by[("P4", "ORACLE_ADAM")][s]
    print(f"  ORACLE_ADAM {s}: bound {r_transition(r['evals']) is not None}, transition {r_transition(r['evals'])}, "
          f"bound routed {r_bound_routed(r)}, updates {r['updates']}")

H("3. The 40-seed pooled count (Phase 2 900-919 WIN3_SPLIT_D8 + Phase 4 920-939 WIN3_SPLIT_D8_ML) and the reading")
split = {**by[("P2", "WIN3_SPLIT_D8")], **by[("P4", "WIN3_SPLIT_D8_ML")]}
slow = {**by[("P2", "WIN3_SLOW_D8")], **by[("P4", "WIN3_SLOW_D8")]}
seeds = sorted(set(split) & set(slow))
xs = sum(r_bound_routed(split[s]) for s in seeds)
xb = sum(r_bound_routed(slow[s]) for s in seeds)
b = sum(r_bound_routed(split[s]) and not r_bound_routed(slow[s]) for s in seeds)
c = sum(r_bound_routed(slow[s]) and not r_bound_routed(split[s]) for s in seeds)
lo, hi = wilson(xs, len(seeds))
blo, bhi = wilson(xb, len(seeds))
p = mcnemar1(b, c)
print(f"  pairs {len(seeds)} ({seeds[0]}-{seeds[-1]}); split {xs}/{len(seeds)} Wilson [{lo:.4f}, {hi:.4f}]; "
      f"no split {xb}/{len(seeds)} [{blo:.4f}, {bhi:.4f}]; discordant {b} vs {c}; one-sided exact p = {p:.4g} (2^-23 = {2**-23:.4g})")
print(f"  point estimate {xs/len(seeds):.3f}; Wilson upper {hi:.5f} {'<' if hi < 0.95 else '>='} 0.95 -> the interval "
      f"{'EXCLUDES' if hi < 0.95 else 'includes'} 0.95; lower {lo:.4f}")
overlap = hi >= 0.835 and lo <= 0.986
reading = "consistent with the report" if overlap and p < 0.05 else ("a lower rate" if hi < 0.835 else "neither")
print(f"  overlap with [0.835, 0.986]: {overlap}; READING (phase4.py rule): {reading!r}")
print(f"  of the 35: from Phase 2 (R's own probe/key positions) {sum(r_bound_routed(split[s]) for s in range(900, 920))}/20, "
      f"from Phase 4 (main line's split) {sum(r_bound_routed(split[s]) for s in range(920, 940))}/20")

H("4. 920-939 alone (secondary, printed with no reading in phase4.py)")
ml, sl4 = by[("P4", "WIN3_SPLIT_D8_ML")], by[("P4", "WIN3_SLOW_D8")]
x4 = sum(r_bound_routed(r) for r in ml.values())
y4 = sum(r_bound_routed(r) for r in sl4.values())
b4 = sum(r_bound_routed(ml[s]) and not r_bound_routed(sl4[s]) for s in ml)
c4 = sum(r_bound_routed(sl4[s]) and not r_bound_routed(ml[s]) for s in ml)
print(f"  WIN3_SPLIT_D8_ML {x4}/20 Wilson [{wilson(x4, 20)[0]:.3f}, {wilson(x4, 20)[1]:.3f}]; WIN3_SLOW_D8 {y4}/20 "
      f"[{wilson(y4, 20)[0]:.3f}, {wilson(y4, 20)[1]:.3f}]; pairs {b4} vs {c4}, p = {mcnemar1(b4, c4):.4g}")
print(f"  exact one-sided against 0.95 on the 20 fresh seeds: P(Bin(20, 0.95) <= {x4}) = {cdf(20, x4, 0.95):.3f} (descriptive)")
fails = sorted(s for s, r in ml.items() if not r_bound_routed(r))
for s in fails:
    r = ml[s]
    m = r["map"]
    print(f"  ML failure {s}: splits {len(r['splits'])} at {[x['update'] for x in r['splits']]}; map {m}; "
          f"shared channels {sorted({ch for ch in m if m.count(ch) > 1})}; per-stream min {min(r['per_stream_acc']):.2f}")
allspl = [x for r in ml.values() for x in r["splits"]]
on_t = sum(1 for x in allspl if x["streams_on_c_star"] >= 2 and x["streams_on_c0"] == 0)
print(f"  Phase 4 splits {len(allspl)}, on target by labels {on_t}; latest split update {max(x['update'] for x in allspl)} "
      f"(main line's last check 40,800: D-5 {'inert' if max(x['update'] for x in allspl) <= 40800 else 'NOT inert'})")
sf = [r for r in list(by[("P2", "WIN3_SLOW_D8")].values()) + list(sl4.values()) if not r_bound_routed(r)]
print(f"  WIN3_SLOW_D8 failures over 40: {len(sf)}; of them with a shared channel: "
      f"{sum(1 for r in sf if len(set(r['map'])) < 8)}")


def med(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


print(f"  median transition, bound routed over the 40: split {med([r_transition(split[s]['evals']) for s in seeds if r_bound_routed(split[s])])}, "
      f"no split {med([r_transition(slow[s]['evals']) for s in seeds if r_bound_routed(slow[s])])}")

H("5. The main line's pooled count (X 540-559 + L 1540-1559, window_gate) and X's failures")
mlx = {}
for m in "XL":
    W = rl_load(None, f"results/{m}/window_gate_results.json")
    mlx[m] = (rl_runs(W, "WIN3_SPLIT_D8"), rl_runs(W, "WIN3_SLOW_D8"))
sx = sum(ml_bound_routed(r) for m in "XL" for r in mlx[m][0].values())
nx = sum(ml_bound_routed(r) for m in "XL" for r in mlx[m][1].values())
bx = sum(ml_bound_routed(mlx[m][0][s]) and not ml_bound_routed(mlx[m][1][s]) for m in "XL" for s in mlx[m][0])
cx = sum(ml_bound_routed(mlx[m][1][s]) and not ml_bound_routed(mlx[m][0][s]) for m in "XL" for s in mlx[m][0])
print(f"  main line: split {sx}/40 Wilson [{wilson(sx, 40)[0]:.3f}, {wilson(sx, 40)[1]:.3f}]; no split {nx}/40; {bx} vs {cx}")
for m in "XL":
    for s, r in sorted(mlx[m][0].items()):
        if not ml_bound_routed(r):
            mp = ch_map(r)
            print(f"  {m} {s}: fired splits {sum(1 for ck in r['checks'] if ck.get('fired'))}; map {mp}; shared {sorted({ch for ch in mp if mp.count(ch) > 1})}")

H("6. Probe-versus-evaluation blocks, as phase4.py's docstring defines them")


def blocks(rs):
    tot = dict(checks=0, blocked=0, eligible=0, converse=0)
    hit, hit_fail, nonmult = [], [], 0
    for r in rs:
        ev = dict((u, a) for u, a in r["evals"])
        spl = [x["update"] for x in r["splits"]]
        el = 0
        for u, pa in r["probes"]:
            if u < 4800:
                continue
            tot["checks"] += 1
            nonmult += u % 2400 != 0
            if u not in ev:
                continue
            before = [s for s in spl if s < u]
            free = len(before) < 3 and (not before or u - before[-1] >= 4800)
            if pa >= 0.95 and ev[u] < 0.95:
                tot["blocked"] += 1
                tot["eligible"] += free
                el += free
            if pa < 0.95 and ev[u] >= 0.95:
                tot["converse"] += 1
        if el:
            hit.append(r["seed"])
            if not r_bound_routed(r):
                hit_fail.append(r["seed"])
    return tot, hit, hit_fail, nonmult


rows = [("Phase 2 WIN3_SPLIT_D8 900-919 (per-seed probe)", list(by[("P2", "WIN3_SPLIT_D8")].values())),
        ("R1 SPLIT_R1_PROBE 900-909 (fixed probe)", list(by[("P3", "SPLIT_R1_PROBE")].values())),
        ("R2 SPLIT_R2_BODYKEYS 900-909 (per-seed probe, body keys)", list(by[("P3", "SPLIT_R2_BODYKEYS")].values())),
        ("Phase 4 WIN3_SPLIT_D8_ML 920-939 (fixed probe)", list(ml.values()))]
res = {}
for name, rs in rows:
    t, hit, hf, nm = blocks(rs)
    res[name[:2]] = t
    print(f"  {name}: {t['checks']} checks, blocked {t['blocked']} ({t['eligible']} otherwise eligible), runs {hit} "
          f"(not bound routed {hf}); converse {t['converse']}; checks not a multiple of 2400: {nm}")
fx = blocks(rows[1][1] + rows[3][1])[0]
ps = blocks(rows[0][1] + rows[2][1])[0]
print(f"  fixed probe pooled (R1 + Phase 4): {fx['blocked']} of {fx['checks']} checks, {fx['eligible']} eligible")
print(f"  per-seed probe pooled (Phase 2 + R2): {ps['blocked']} of {ps['checks']} checks; Phase 2 alone 13 of 192 as the draft quotes")
r914 = by[("P2", "WIN3_SPLIT_D8")][914]
ev914 = dict((u, a) for u, a in r914["evals"])
print("  seed 914 (Phase 2): splits", [x["update"] for x in r914["splits"]], "; blocked checks (u, probe, eval):",
      [(u, pa, round(ev914[u], 3)) for u, pa in r914["probes"] if u >= 4800 and u in ev914 and pa >= 0.95 and ev914[u] < 0.95])
for s in (906, 914):
    t, _, _, _ = blocks([by[("P2", "WIN3_SPLIT_D8")][s]])
    print(f"  Phase 2 seed {s}: blocked {t['blocked']}, eligible {t['eligible']}, bound routed {r_bound_routed(by[('P2', 'WIN3_SPLIT_D8')][s])}")
print("  914 rerun under the main line's protocol (amended part (a)): none in any record")

H("7. Calibration: what the pre-registered rule could detect; the amended rule's numbers")
xmin = min(x for x in range(41) if wilson(x, 40)[1] >= 0.835)
print(f"  'a lower rate' needs Wilson upper < 0.835: x <= {xmin - 1}/40; with Phase 2's 17/20 pooled, <= {xmin - 1 - 17} of 20 new")
for pr in (0.95, 0.85, 0.80, 0.75, 0.70, 0.65):
    print(f"  true {pr}: P('a lower rate' | 17/20 kept) = {cdf(20, xmin - 18, pr):.3f}; "
          f"P(upper >= 0.835, i.e. not 'lower') = {1 - cdf(20, xmin - 18, pr):.3f}")
print(f"  amended (b): 'below 0.95' if x <= 27/32: size {cdf(32, 27, 0.95):.3f}; power at 0.85 {cdf(32, 27, 0.85):.3f}, "
      f"at 0.80 {cdf(32, 27, 0.80):.3f}")
for alt in (0.85,):
    for n in range(20, 400):
        ks = [k for k in range(n + 1) if cdf(n, k, 0.95) <= 0.05]
        if ks and cdf(n, ks[-1], alt) >= 0.8:
            print(f"  one-sample, 0.85 vs fixed 0.95 (alpha 0.05 one-sided, power 0.8): n = {n}, x <= {ks[-1]}")
            break
z = lambda q: {0.975: 1.959964, 0.95: 1.644854, 0.8: 0.841621}[q]
p1, p2_ = 0.95, 0.85
pb = (p1 + p2_) / 2
for q in (0.975, 0.95):
    n2 = (z(q) * math.sqrt(2 * pb * (1 - pb)) + z(0.8) * math.sqrt(p1 * (1 - p1) + p2_ * (1 - p2_))) ** 2 / (p1 - p2_) ** 2
    print(f"  two-sample (two arms), 0.85 vs 0.95, alpha {'0.05 two-sided' if q == 0.975 else '0.05 one-sided'}, power 0.8: "
          f"about {math.ceil(n2)} per arm (normal approximation) -- R's 'on the order of 150 per arm'")

H("8. CPU and wall time in R's records")
for name, rs in (("phase2", p2), ("phase3", p3), ("phase4", p4)):
    ks = sorted({k for r in rs for k in r if re.search("cpu|host|machine|device|platform", k, re.I)})
    tv = sorted({r.get("torch") for r in rs})
    print(f"  {name}: CPU-like fields {ks or 'none'}; torch {tv}; sum wall_s {sum(r['wall_s'] for r in rs) / 3600:.1f} h "
          f"(runs {len(rs)})")
print("  R's prose: replicate.py:36 'PyTorch 2.14.1 (CPU), 37.4 h' (Phase 2); replicate_report.md:109 '4 workers x 1 thread'")

H("9. When Phase 4 started (simulated from wall_s; 4 workers, jobs in replicate.py's order, imap_unordered)")
jobs = [("ORACLE_ADAM", 920), ("ORACLE_ADAM", 921)] + [j for s in range(920, 940) for j in (("WIN3_SPLIT_D8_ML", s), ("WIN3_SLOW_D8", s))]
wall = {key(r): r["wall_s"] for r in p4}
free = [0.0] * 4
fin = []
for j in jobs:
    w = min(range(4), key=lambda i: free[i])
    free[w] += wall[j]
    fin.append((free[w], j))
fin.sort()
order_ok = [j for _, j in fin] == [key(r) for r in p4]
t21, t22, t42 = fin[20][0], fin[21][0], fin[41][0]
ti, tf = ctime(R_INTERIM), ctime(R_FINAL)
print(f"  simulated completion order equals the record order: {order_ok}")
print(f"  21st completion at +{t21 / 3600:.2f} h, 22nd at +{t22 / 3600:.2f} h, last at +{t42 / 3600:.2f} h")
print(f"  start no later than {(ti - timedelta(seconds=t21)).isoformat()} (interim commit) and "
      f"{(tf - timedelta(seconds=t42)).isoformat()} (final commit); if the interim commit preceded the 22nd record, "
      f"after {(ti - timedelta(seconds=t22)).isoformat()}")
print(f"  pre-registration {ctime(R_PREREG).isoformat()}; referee's replication review 01bee1f {ctime('01bee1f').isoformat()}")

H("10. UNSTATED.md labels (tex4:36, tex4:330)")
un = [l for l in git("show", f"{R_FINAL}:UNSTATED.md").splitlines() if l.startswith("| U-")]
lab = {}
for l in un:
    what = l.split("|")[2]
    m = re.search(r"\[([^\]]+)\]\s*$", what.strip())
    lab.setdefault(m.group(1) if m else "(none: process)", []).append(l.split("|")[1].strip())
for k, v in sorted(lab.items()):
    print(f"  [{k}]: {len(v)} {v if len(v) <= 4 else ''}")
print(f"  total {len(un)}")

H("11. Provenance row tex4:513")
print("  de8df0f:", git("log", "-1", "--format=%s", "de8df0f").strip())
print("  00990ea:", git("log", "-1", "--format=%s", "00990ea").strip())
print("  100dced:", git("log", "-1", "--format=%s", "100dced").strip())
print(f"  runs: Phase 2 {len(p2)}, Phase 3 {len(p3)}, Phase 4 {len(p4)}")

H("12. Where the draft says it (tex4 line numbers)")
for pat in (r"35/40", r"18/20\$ on twenty further", r"18/20\$ under the main line", r"includes \$0\.95\$", r"one time in nine",
            r"3 of 263", r"13 of 192", r"db22857", r"carries the first rule", r"never saw the code", r"R's records carry no CPU",
            r"detected only rates near", r"150 seeds", r"464be8d", r"was replicated from the text alone"):
    print(f"  {pat!r}: tex4 {where(pat)}")
