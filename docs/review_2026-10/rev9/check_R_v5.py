#!/usr/bin/env python3
"""check_R_v5.py -- the referee's fifth-pass check of session R's Phase 4 as folded into the Revision 9
draft, fifth version (claude/rev9-draft at c5e4508; tex5:NNN = multichannel_hebbian_report_v9.tex there),
adapted from check_R_v4.py (fourth version c66cc76, tex4:NNN). Sections 0-11 are v4's record recomputations,
unchanged; section 12 adds the numbers the fifth version's new wording rests on; section 13 locates the
wording; section 14 prints the rows, each anchored on a fragment of its tex5 line, and the counts.

    python3 docs/review_2026-10/rev9/check_R_v5.py        (stdlib only; prints; writes nothing)

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
TEX5, TEX4, TEX3, AMEND = "c5e4508", "c66cc76", "a5f67a2", "db22857"

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
tex5 = git("show", f"{TEX5}:multichannel_hebbian_report_v9.tex").split("\n")


def where(pat, t=None):
    return [i + 1 for i, l in enumerate(tex5 if t is None else t) if re.search(pat, l)]


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
for ref in (R_PREREG, "01bee1f", R_INTERIM, AMEND, R_FINAL, "9657edb", TEX3, TEX4, TEX5):
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

H("10. UNSTATED.md labels (tex5:36, tex5:332)")
un = [l for l in git("show", f"{R_FINAL}:UNSTATED.md").splitlines() if l.startswith("| U-")]
lab = {}
for l in un:
    what = l.split("|")[2]
    m = re.search(r"\[([^\]]+)\]\s*$", what.strip())
    lab.setdefault(m.group(1) if m else "(none: process)", []).append(l.split("|")[1].strip())
for k, v in sorted(lab.items()):
    print(f"  [{k}]: {len(v)} {v if len(v) <= 4 else ''}")
print(f"  total {len(un)}")

H("11. Provenance row tex5:517")
print("  de8df0f:", git("log", "-1", "--format=%s", "de8df0f").strip())
print("  00990ea:", git("log", "-1", "--format=%s", "00990ea").strip())
print("  100dced:", git("log", "-1", "--format=%s", "100dced").strip())
print(f"  runs: Phase 2 {len(p2)}, Phase 3 {len(p3)}, Phase 4 {len(p4)}")


# ---------------------------------------------------------------------------------------------------
H("12. Numbers the fifth version's new wording rests on")
print("  origin/claude/replicate-R head:", git("rev-parse", "--short", "origin/claude/replicate-R").strip(),
      "(nothing newer than 1f8be04)" if git("rev-parse", "--short", "origin/claude/replicate-R").strip() == R_FINAL else "(NEWER)")
w17 = wilson(17, 20)
w55 = wilson(55, 60)
print(f"  Wilson(35, 40) upper {hi:.5f}: rounds to {hi:.3f}; '0.945, just below 0.95': {round(hi, 3) == 0.945 and hi < 0.95}")
print(f"  Wilson(17, 20) = [{w17[0]:.4f}, {w17[1]:.4f}] (upper {'<' if w17[1] < 0.95 else '>='} 0.95); "
      f"Wilson(55, 60) = [{w55[0]:.3f}, {w55[1]:.3f}]")


def fisher2(a, n1, c, n2):
    K, N = a + c, n1 + n2
    pr = lambda x: math.comb(K, x) * math.comb(N - K, n1 - x) / math.comb(N, n1)
    p0 = pr(a)
    return sum(pr(x) for x in range(max(0, K - n2), min(K, n1) + 1) if pr(x) <= p0 * (1 + 1e-9))


print(f"  Fisher two-sided 17/20 vs 38/40: p = {fisher2(17, 20, 38, 40):.3f}")
print(f"  P(Bin(20, 0.90) >= 18) = {1 - cdf(20, 17, 0.90):.3f} (fails a true 0.90 recipe {cdf(20, 17, 0.90):.3f}, 'one time in three'); "
      f"P(Bin(20, 0.95) <= 17) = {cdf(20, 17, 0.95):.4f}")
pl70 = cdf(20, xmin - 18, 0.70)
print(f"  P('a lower rate' | 0.70) = {pl70:.3f} = 1 in {1 / pl70:.1f} ('one time in nine'); at 0.65 {cdf(20, xmin - 18, 0.65):.3f}, "
      f"0.60 {cdf(20, xmin - 18, 0.60):.3f}, 0.75 {cdf(20, xmin - 18, 0.75):.3f}")
ks52 = max(k for k in range(53) if cdf(52, k, 0.95) <= 0.05)
print(f"  one-sample n = 52 against 0.95: x <= {ks52}, size {cdf(52, ks52, 0.95):.3f}, power at 0.85 {cdf(52, ks52, 0.85):.3f}; "
      f"'46/52' at tex5 {where(r'46/52')}; section label sec:method at tex5 {where(r'label.sec:method')}")
t2 = blocks(rows[2][1])[0]
tP2 = blocks(rows[0][1])[0]
t914 = blocks([r914])[0]
print(f"  R2 (the second rerun) blocked {t2['blocked']} of {t2['checks']}; Phase 2 {tP2['blocked']} of {tP2['checks']}; "
      f"per-seed pooled {ps['blocked']} of {ps['checks']} = {tP2['checks']} + {t2['checks']}; 914: {t914['blocked']} blocked, {t914['eligible']} eligible")


def n_two(q):
    return math.ceil((z(q) * math.sqrt(2 * pb * (1 - pb)) + z(0.8) * math.sqrt(p1 * (1 - p1) + p2_ * (1 - p2_))) ** 2 / (p1 - p2_) ** 2)


fail_txt = "; ".join(f"{s_}: {len(ml[s_]['splits'])} splits, shared {sorted({ch for ch in ml[s_]['map'] if ml[s_]['map'].count(ch) > 1})}" for s_ in fails)
pr914 = [(u, pa) for u, pa in r914["probes"] if u >= 16800]
ev914s = [ev914[u] for u, _ in pr914 if u in ev914]
print(f"  914 from 16,800 on: probe {min(p for _, p in pr914):.3f}-{max(p for _, p in pr914):.3f} at {len(pr914)} checks; "
      f"evaluation at those updates {min(ev914s):.3f}-{max(ev914s):.3f}; final map {r914['map']}")
rr = git("show", f"{R_FINAL}:replicate_report.md").split("\n")
print("  replicate_report.md lines on the interval:",
      [(i + 1, l.strip()[:90]) for i, l in enumerate(rr) if re.search(r"includes 0\.95|contains the report's 38/40", l)])
for c_ in ("de8df0f", "00990ea", R_PREREG):
    print(f"  {c_} {ctime(c_).isoformat()} {git('log', '-1', '--format=%s', c_).strip()[:70]}")
print("  phase4.py:19-20 and :47:", [l.strip()[:100] for l in src[R_FINAL].split("\n")[18:20] + [src[R_FINAL].split("\n")[46]]])
amd = amend_txt[857:884]
print("  amended prompt part (a):", [l.strip()[:110] for l in amd if l.strip().startswith("(a)")])

# ---------------------------------------------------------------------------------------------------
H("13. Where the fifth version says it (tex5 line numbers; tex4 for comparison)")
for pat in (r"35/40", r"includes \$0\.95\$", r"reaches \$0\.945\$", r"one time in nine", r"3 of 263", r"13 of 192", r"13 of 282",
            r"never saw the code", r"had not seen the code", r"main line's protocol", r"split's probe and key positions",
            r"on R's inference", r"150 seeds", r"as two arms against each other", r"464be8d", r"db22857",
            r"from the text alone", r"R's records carry no CPU", r"carries the first rule", r"queue is empty"):
    print(f"  {pat!r}: tex5 {where(pat)}; tex4 {where(pat, tex4)}")

# ---------------------------------------------------------------------------------------------------
H("14. Rows (each anchored on a fragment of its tex5 line) and counts")
ROWS = [
    # (tex5 line, anchor fragment (regex on that line), item, draft says, recomputed, status, note)
    (36, r"from the report alone, by a session that had not seen the code when it ran its first twenty seeds, bound eight streams \$17/20\$.*\$18/20\$ on twenty further seeds after taking the split's probe and key positions from our code",
     "the abstract's subject 'from the report alone'",
     "'A clean-room replication from the report alone, by a session that had not seen the code when it ran its first twenty seeds, bound ... 17/20 ..., and 18/20 on twenty further seeds after taking the split's probe and key positions from our code (35/40 ...)'",
     "17/20 (Phase 2, de8df0f 13:55) predates the diff (00990ea 14:01); 18/20 and 35/40 are Phase 4, after it",
     "PARTIAL", "The fix of tex4:36 is in (the clause is exact), but the subject 'a clean-room replication from the report alone' still governs 18/20 and 35/40, which the same sentence says used our code. Minor: 'a clean-room replication, from the report alone for its first twenty seeds' would remove the tension. The referee's own suggested fix left this phrase."),
    (36, r"had not seen the code when it ran its first twenty seeds", "the abstract's 'never saw the code' (fourth pass §1 item 2)",
     "'by a session that had not seen the code when it ran its first twenty seeds'",
     f"Phase 2 records de8df0f ({ctime('de8df0f').strftime('%H:%M')}) before DIFF.md 00990ea ({ctime('00990ea').strftime('%H:%M')}); Phase 4 pre-registered at 464be8d ({ctime(R_PREREG).strftime('%H:%M')})",
     "MATCH", "The fourth pass's suggested wording, verbatim."),
    (36, r"\$17/20\$ against \$7/20\$ without the split \(10 vs 0\), and \$18/20\$.*\(\$35/40\$ against \$12/40\$ pooled, 23 vs 0\)",
     "abstract counts", "17/20 against 7/20 (10 vs 0); 18/20; 35/40 against 12/40 pooled, 23 vs 0",
     f"P2 split {sum(r_bound_routed(r) for r in by[('P2','WIN3_SPLIT_D8')].values())}/20, no split {sum(r_bound_routed(r) for r in by[('P2','WIN3_SLOW_D8')].values())}/20; P4 {x4}/20; pooled {xs}/40 vs {xb}/40, {b} vs {c}",
     "MATCH", ""),
    (36, r"44 details it logged \(34 of them guesses\)", "UNSTATED count", "44 details (34 guesses)",
     f"{len(un)} entries; [A] {len(lab.get('A', []))} + [B] {len(lab.get('B', []))} + [C] {len(lab.get('C', []))}", "MATCH", "See §10 for the label tally."),
    (25, r"\$17/20\$ in a replication from the text alone", "title", "'17/20 in a replication from the text alone'", "17/20, Phase 2", "MATCH", ""),
    (41, r"R's Phase~4 are complete", "Phase 4 complete", "'... and R's Phase 4 are complete, every session's queue is empty'",
     f"{len(p4)} of 42 jobs, one '42 jobs to run' header; origin/claude/replicate-R still at {R_FINAL}", "MATCH",
     "R's queue is empty (nothing newer on its branch). 'Every session's queue is empty' is false for E (batch 20 running at c5e4508): check_E19_v5 row 1."),
    (41, r"R's records carry no CPU model", "CPU", "'R's records carry no CPU model'", "no CPU, host or device field in phase2/3/4 (§8)", "MATCH", ""),
    (62, r"\$17/20\$ against \$7/20\$ \(10 vs 0\) from the text alone; \$18/20\$ with the split's probe and key positions as the main line specifies them; \$35/40\$ against \$12/40\$ pooled \(23 vs 0\), ``consistent with the report''",
     "index row ('the main line's protocol', fourth pass §2 / v4 row 4)", "'18/20 with the split's probe and key positions as the main line specifies them'",
     "WIN3_SPLIT_D8_ML = D-1 (fixed probe) + D-2 (32 body keys), phase4.py:5-9", "MATCH", "R's own wording ('as the main line specifies it')."),
    (207, r"with the fixed probe the mode occurred at 3 of 263 checks and never decided a run", "fixed-probe count",
     "3 of 263, never decided a run", f"R1 + Phase 4: {fx['blocked']} of {fx['checks']}, {fx['eligible']} eligible (905, bound routed)", "MATCH", ""),
    (207, r"with one probe drawn per seed at 13 of 282 and, on R's inference, it decided one run", "per-seed count and '914' (fourth pass §2, tex4:205)",
     "'13 of 282 and, on R's inference, it decided one run'", f"Phase 2 + R2: {ps['blocked']} of {ps['checks']}; 914: {t914['blocked']} blocked, {t914['eligible']} eligible, bound routed {r_bound_routed(r914)}",
     "MATCH", "Pooled the same way as the fixed-probe 263 (both include a 900-909 rerun); 'on R's inference' marks the attribution."),
    (207, r"probe read \$0\.953\$--\$1\.000\$ at every check from 16\{,\}800 on while the 2048-query evaluation sat at \$0\.871\$--\$0\.887\$",
     "914's readings (unchanged text)", "probe 0.953-1.000 from 16,800, evaluation 0.871-0.887",
     f"probe {min(p for _, p in pr914):.3f}-{max(p for _, p in pr914):.3f}, evaluation {min(ev914s):.3f}-{max(ev914s):.3f}", "MATCH", ""),
    (330, r"bound routed \$\\mathbf\{17/20\}\$ \(Wilson \$\[0\.640, 0\.948\]\$\); without it \$7/20\$.*\(10 vs 0, \$p = 0\.00098\$\)",
     "Phase 2 (unchanged text)", "17/20 [0.640, 0.948]; 7/20; 10 vs 0, p = 0.00098",
     f"17/20 [{w17[0]:.3f}, {w17[1]:.3f}]; 7/20; 10 vs 0, p = {mcnemar1(10, 0):.5f}", "MATCH", ""),
    (330, r"Fisher two-sided \$p = 0\.32\$, and pooled over X, L and R the rate is \$55/60\$ \$\[0\.819, 0\.964\]\$",
     "against the report (unchanged)", "Fisher two-sided p = 0.32; 55/60 [0.819, 0.964]",
     f"p = {fisher2(17, 20, 38, 40):.3f}; 55/60 [{w55[0]:.3f}, {w55[1]:.3f}]", "MATCH", ""),
    (330, r"it fails a true \$0\.90\$ recipe a third of the time", "Phase 2 bar", "fails a true 0.90 recipe a third of the time",
     f"P(Bin(20, 0.9) <= 17) = {cdf(20, 17, 0.90):.3f}", "MATCH", ""),
    (330, r"A fourth phase ran the split with its probe and key positions as the main line specifies them \(the fixed probe, the body's key positions; its other choices, among them the map on the held-out sequences, R's own\)",
     "Phase 4 design (v4 rows 4, 18)", "'the split with its probe and key positions as the main line specifies them (...; its other choices, among them the map on the held-out sequences, R's own)'",
     "D-1, D-2 adopted; D-4..D-10 R's own (D-10: map on 2048 held-out sequences)", "MATCH", ""),
    (330, r"with the single-rate oracle as validity \(\$2/2\$ at 2400\)", "validity", "2/2 at 2400",
     f"ORACLE_ADAM 920, 921 transitions {[r_transition(by[('P4','ORACLE_ADAM')][s]['evals']) for s in (920, 921)]}", "MATCH", ""),
    (330, r"under a rule pre-registered before any run \(\\texttt\{464be8d\}\)", "pre-registration", "'pre-registered before any run (464be8d)'",
     f"no phase4.jsonl at 464be8d; rule and code unchanged to 1f8be04; replayed start no later than {(tf - timedelta(seconds=t42)).strftime('%H:%M:%S')} UTC vs 464be8d {ctime(R_PREREG).strftime('%H:%M:%S')}",
     "NOT RECORD-CHECKABLE", "Unchanged from v4 row 10: the records carry no timestamps."),
    (330, r"the split arm bound routed \$\\mathbf\{35/40\}\$ \(Wilson \$\[0\.739, 0\.945\]\$\) against \$12/40\$ without the split, 23 vs 0 \(one-sided \$p = 1\.2\\times10\^\{-7\}\$\)",
     "pooled count", "35/40 [0.739, 0.945]; 12/40; 23 vs 0; p = 1.2e-7", f"{xs}/40 [{lo:.4f}, {hi:.4f}]; {xb}/40; {b} vs {c}; p = {p:.3g}", "MATCH", ""),
    (330, r"``consistent with the report'' applies \(the intervals overlap; the report's own pooled count is \$38/40\$ against \$15/40\$, also 23 vs 0\)",
     "reading", "'consistent with the report'; main line 38/40 vs 15/40, 23 vs 0", f"{reading!r}; main line {sx}/40 vs {nx}/40, {bx} vs {cx}", "MATCH", ""),
    (330, r"On the fresh seeds alone, with those two choices, the split arm bound \$18/20\$, the count Phase~2's bar had asked for, which \\texttt\{phase4\.py\} prints as secondary without a reading",
     "920-939 alone (fourth pass §2 '920-939 alone'; v4 rows 5, 21)", "'18/20 ..., which phase4.py prints as secondary without a reading'",
     f"{x4}/20 vs {y4}/20, {b4} vs {c4}; phase4.py:19-20 'printed beside it, as a secondary count', :125 prints it, no reading", "MATCH", ""),
    (330, r"both failures were two streams sharing a channel after three splits, as X's were", "failures", "two streams sharing a channel after three splits",
     f"{fail_txt}; X 555, 556 each 3 fired splits and a shared channel (§5)", "MATCH", ""),
    (330, r"the interval reaches \$0\.945\$, just below \$0\.95\$ \(R's report says it includes \$0\.95\$; the pre-registered reading, which tests overlap with the report's interval, is unaffected\)",
     "the interval vs 0.95 (fourth pass §1 item 1)", "'the interval reaches 0.945, just below 0.95 (R's report says it includes 0.95; ... unaffected)'",
     f"Wilson(35, 40) upper {hi:.4f} < 0.95; replicate_report.md:138 'includes 0.95'", "MATCH", "The fourth pass's fix, and R's own error now attributed to R."),
    (330, r"it excludes rates below \$0\.74\$", "lower bound", "excludes rates below 0.74", f"lower {lo:.4f}", "MATCH", ""),
    (330, r"separating \$0\.85\$ from \$0\.95\$ as two arms against each other would need about 150 seeds per arm \(against a fixed rate on fresh seeds, the 52 of Section~\\ref\{sec:method\} answer a different question\)",
     "150 vs 52 (fourth pass §2)", "'as two arms against each other ... about 150 seeds per arm (against a fixed rate ..., the 52 of Section sec:method ...)'",
     f"two-sample normal approx: {n_two(0.975)} per arm (two-sided), {n_two(0.95)} (one-sided); R says 'on the order of 150' (replicate_report.md:139); one-sample n = 52, x <= {ks52}, size {cdf(52, ks52, 0.95):.3f}, power {cdf(52, ks52, 0.85):.3f} (tex5:430)",
     "MATCH", "R's figure is labelled R's; the two questions are now told apart."),
    (330, r"detects a true rate of \$0\.70\$ about one time in nine", "the first rule's power", "0.70 about one time in nine", f"{pl70:.3f} = 1 in {1 / pl70:.1f}", "MATCH", ""),
    (330, r"32 fresh seeds, ``below \$0\.95\$'' at \$x \\leq 27\$, size \$0\.020\$, power \$0\.54\$ at \$0\.85\$",
     "the amended reading's numbers", "x <= 27/32, size 0.020, power 0.54", f"size {cdf(32, 27, 0.95):.3f}, power {cdf(32, 27, 0.85):.3f}", "MATCH", ""),
    (330, r"was written after the referee's review but existed in no commit until \\texttt\{db22857\}, after Phase~4 had started, and was not used",
     "the amended reading's history", "written after the review; no commit until db22857, after Phase 4 had started; not used",
     f"db22857 {ctime(AMEND).strftime('%m-%d %H:%M')} after 0364e28 {ctime(R_INTERIM).strftime('%m-%d %H:%M')} (21 records); no amended arm, seed or rule in R's records",
     "NOT RECORD-CHECKABLE", "Unchanged from v4 row 11: 'written after the referee's review' rests on chat; the rest is checkable and matches."),
    (330, r"with one probe drawn per seed \(Phase~2\) it occurred at 13 of 192 \(13 of 282 with R's second rerun, which had none\) and, on R's inference, decided one run in twenty \(914\); the rerun that would test that inference, 914 under the main line's protocol, was not run",
     "per-seed count, 'decided' (fourth pass §2, tex4:328)", "'13 of 192 (13 of 282 with R's second rerun, which had none) and, on R's inference, decided one run in twenty (914); the rerun that would test that inference, ... was not run'",
     f"Phase 2 {tP2['blocked']}/{tP2['checks']}, R2 {t2['blocked']}/{t2['checks']}, pooled {ps['blocked']}/{ps['checks']}; amended part (a) (914, 915) has no record",
     "MATCH", "'under the main line's protocol' here is amended part (a)'s own wording (db22857:866)."),
    (330, r"with the main line's fixed probe, a check at which the probe read \$\\geq 0\.95\$ while the evaluation did not occurred at 3 of 263 checks and never decided a run",
     "fixed-probe count", "3 of 263; never decided a run", f"{fx['blocked']} of {fx['checks']}, {fx['eligible']} eligible", "MATCH", ""),
    (330, r"R's advice: specify the probe as one fixed draw, say that 64 queries can over-read a merged model, and gate the trigger on more queries",
     "R's advice", "as stated", "replicate_report.md Phase 4, 'For the report'", "MATCH", ""),
    (332, r"left 34 to be guessed \(the initialisation scales, the parameter dtype, the evaluation's sampling and the trigger's details among them\): 20 on which it was silent, 12 on which it was ambiguous and 2 carried over .*, with 3 further entries of process",
     "UNSTATED labels and the parenthesis (fourth pass §1 item 9)", "4 outright, 3 in part, 34 guessed (parenthesis after it): 20, 12, 2; 3 process",
     f"stated 4, in part 3, [A] {len(lab.get('A', []))}, [B] {len(lab.get('B', []))}, [C] {len(lab.get('C', []))}, process 3 (total {len(un)})",
     "MATCH", "The parenthesis now follows the guesses; U-6, U-11, U-16, U-23 are guessed entries."),
    (349, r"the amended reading existed in no commit until \\texttt\{db22857\}; R's branch carries the first rule", "R's branch (unchanged)", "R's branch carries the first rule",
     "phase4.py byte-identical 464be8d..1f8be04 (§0)", "MATCH", ""),
    (356, r"\\textbf\{Stands\}, and was replicated, on our assessment, in session~R: \$17/20\$ from the text alone \(``partially replicated'' by R's bar\), then \$18/20\$ with the split's probe and key positions taken from our code, \$35/40\$ pooled \(``consistent with the report''",
     "'replicated from the text alone' (fourth pass §1 item 3; v4 rows 3-5)", "'replicated, on our assessment, in session R: 17/20 from the text alone (\"partially replicated\" by R's bar), then 18/20 with the split's probe and key positions taken from our code, 35/40 pooled (\"consistent with the report\"'",
     "Phase 2 reading 'partially replicated'; Phase 4 reading 'consistent with the report'; 18/20 used D-1, D-2", "MATCH",
     "'from the text alone' now covers 17/20 only; 'replicated' marked as the authors' assessment; R's two readings quoted."),
    (393, r"would have detected only rates near \$0\.70\$ or below, and \$0\.70\$ itself one time in nine",
     "what the first rule could detect (fourth pass §2, tex4:391)", "'only rates near 0.70 or below, and 0.70 itself one time in nine'",
     f"P('a lower rate') 0.113 at 0.70, {cdf(20, xmin - 18, 0.65):.3f} at 0.65, {cdf(20, xmin - 18, 0.60):.3f} at 0.60, 0.041 at 0.75", "MATCH", "The review's (01bee1f:148) 'or below' restored."),
    (393, r"existed in no commit until \\texttt\{db22857\}, after the referee's second pass had found no record of it, and R's branch carries the first rule",
     "the amended reading's commit (unchanged)", "as stated", "db22857's message says the same; phase4.py unchanged", "MATCH", ""),
    (430, r"\$x \\leq 46/52\$ against \$0\.95\$ has size \$0\.045\$ and power \$0\.81\$ at \$0\.85\$", "the 52 the new tex5:330 cites",
     "x <= 46/52, size 0.045, power 0.81", f"x <= {ks52}, size {cdf(52, ks52, 0.95):.3f}, power {cdf(52, ks52, 0.85):.3f}", "MATCH", ""),
    (472, r"in 17 of 20 in a replication from the text alone", "conclusion", "17 of 20 from the text alone", "17/20", "MATCH", ""),
    (477, r"a calibrated reading for any further replication phase", "next steps", "as stated", "the amended Phase 4 was not run", "MATCH", ""),
    (517, r"Phase 2 \(\\texttt\{de8df0f\}\), Phase 3 diff and reruns \(\\texttt\{00990ea\}, \\texttt\{100dced\}\); Phase 4, the 40-seed count \(rule \\texttt\{464be8d\}; report \\texttt\{1f8be04\}\) & \\texttt\{1f8be04\} & 42; 20; 42",
     "provenance", "commits; 42; 20; 42", f"runs {len(p2)}; {len(p3)}; {len(p4)}", "MATCH", ""),
]
order = {"DISAGREE": 0, "PARTIAL": 1, "NOT RECORD-CHECKABLE": 2, "MATCH": 3}
cnt = {k: 0 for k in order}
bad = 0
for i, (ln, frag, item, says, rec, st, note) in enumerate(sorted(ROWS, key=lambda r: (order[r[5]], r[0])), 1):
    ok = re.search(frag, tex5[ln - 1]) is not None
    bad += not ok
    cnt[st] += 1
    print(f"  {i:2d} | tex5:{ln} | anchor {'ok' if ok else 'MISSING'} | {st} | {item} | recomputed: {rec}" + (f" | {note}" if note else ""))
print(f"\n  COUNTS: {cnt['DISAGREE']} DISAGREE, {cnt['PARTIAL']} PARTIAL, {cnt['NOT RECORD-CHECKABLE']} NOT RECORD-CHECKABLE, "
      f"{cnt['MATCH']} MATCH ({len(ROWS)} rows); anchors missing: {bad}")
