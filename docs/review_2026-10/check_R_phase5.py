#!/usr/bin/env python3
"""check_R_phase5.py -- Referee V's check of session R's Phase 5 (claude/replicate-R): the protocol 550a736
(22:47:46 UTC, 12 October, "Phase 5 protocol: copy vs reset vs one delta channel (seeds 940-959)"), the interim
records 29ff87a and the report commit 31c39fc (replicate_report.md:166-223, phase5.py's RESULT block).

    python3 docs/review_2026-10/check_R_phase5.py        (stdlib only; prints; writes nothing)

Everything is read with `git show <ref>:<path>`; nothing of R's or the main line's is run. R's outcome is
recomputed with replication_check.py's r_transition / r_bound_routed (taken from that file's source by ast, so the
module does not run); the main line's (test_stream_recipe.outcome: bound = a transition held to the end; routed =
one-to-one map and every stream >= 0.9) with rlib.py's re-implementation, on the main line's records and, through
an adapter, on R's. Sections:
  0  R's commits from 550a736 to the head, with times and files
  1  pre-registration: what changed after the protocol; the records' order; the 4-worker replay of the run's start
  2  seeds 940-959 before 550a736: R's earlier records, the main line's split_copy seeds, every branch tip, history
  3  R's Phase 5 counts, intervals, discordant pairs, McNemar, the reading rule, the diagnostics
  4  the main line's test_split_copy recomputed (X 592c574, L ffb0771) and R's statements about it
  5  'from the spec and docstring alone': what git can show
  6  UNSTATED.md at the report commit
  7  the rows and the counts
"""
import ast
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import rlib  # noqa: E402

PROT, INTERIM, FINAL, P4_FINAL = "550a736", "29ff87a", "31c39fc", "1f8be04"
RBR = "origin/claude/replicate-R"
MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
ML_X, ML_L = "592c574", "ffb0771"
P3_READ = "404e3a0"          # the main-line commit R's DIFF.md says it read in Phase 3
SEEDS = range(940, 960)

_src = open(os.path.join(HERE, "replication_check.py")).read()
_ns = {}
for node in ast.parse(_src).body:
    if isinstance(node, ast.FunctionDef) and node.name in ("r_transition", "r_bound_routed"):
        exec(compile(ast.Module([node], []), "replication_check.py", "exec"), _ns)
r_transition, r_bound_routed = _ns["r_transition"], _ns["r_bound_routed"]


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a).decode()


def show(ref, path):
    return git("show", f"{ref}:{path}")


def exists(ref, path):
    return subprocess.run(("git", "-C", ROOT, "cat-file", "-e", f"{ref}:{path}"), capture_output=True).returncode == 0


def blob(ref, path):
    return git("rev-parse", f"{ref}:{path}").strip() if exists(ref, path) else None


def jsonl(ref, path):
    return [json.loads(l) for l in show(ref, path).splitlines() if l.strip()]


def ctime(ref):
    return datetime.fromisoformat(git("log", "-1", "--format=%cI", ref).strip())


def cdf(n, k, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def fisher_two(a, n1, c, n2):
    """Two-sided Fisher exact for a/n1 vs c/n2 (sum of tables no more probable than the observed)."""
    K, N = a + c, n1 + n2
    pr = lambda x: math.comb(n1, x) * math.comb(n2, K - x) / math.comb(N, K)
    p0 = pr(a)
    return sum(pr(x) for x in range(max(0, K - n2), min(n1, K) + 1) if pr(x) <= p0 * (1 + 1e-9))


def H(t):
    print("\n== " + t)


ROWS = []


def row(src, stmt, recomputed, status, note=""):
    ROWS.append((src, stmt, recomputed, status, note))


def same(a, b):
    return "MATCH" if a == b else "DISAGREE"


# ── 0 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
H("0. R's commits from 550a736 to the head (commit time = author time, UTC)")
head = git("rev-parse", "--short=7", RBR).strip()
print(f"  {RBR} head: {head}; root {git('rev-list', '--max-parents=0', RBR).strip()[:7]}; "
      f"merge-base with the main line: {subprocess.run(('git', '-C', ROOT, 'merge-base', RBR, MAIN), capture_output=True, text=True).stdout.strip() or 'none'}")
commits = git("log", "--reverse", "--format=%h|%cI|%aI|%s", f"{PROT}^..{RBR}").strip().splitlines()
for c in commits:
    h, ci, ai, s = c.split("|", 3)
    files = git("show", "--numstat", "--format=", h).strip().replace("\t", " ").splitlines()
    print(f"  {h} committed {ci} (authored {ai}) {s}")
    for f in files:
        print(f"      {f}")
t_prot, t_int, t_fin = ctime(PROT), ctime(INTERIM), ctime(FINAL)

# ── 1 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
H("1. Pre-registration")
p5_0, p5_f = show(PROT, "phase5.py"), show(FINAL, "phase5.py")
cut0, cutf = p5_0.index("RESULT ("), p5_f.index("RESULT (")
proto_same = p5_0[:cut0] == p5_f[:cutf]
end0, endf = p5_0.index('"""', 5), p5_f.index('"""', 5)
code_same = p5_0[end0:] == p5_f[endf:]
print(f"  phase5.py 550a736 -> 31c39fc: text before 'RESULT (' identical: {proto_same}; code after the docstring identical: "
      f"{code_same}; the change is the RESULT block only (550a736: '(not yet run)')")
print(f"  phase5.py at 29ff87a equals 550a736: {blob(INTERIM, 'phase5.py') == blob(PROT, 'phase5.py')}")
for p in ("replicate.py", "analyze.py", "phase4.py", "test_replicate.py", "UNSTATED.md", "DIFF.md", ".gitignore"):
    print(f"  {p:18s} identical 550a736 / 29ff87a / 31c39fc: "
          f"{blob(PROT, p) == blob(INTERIM, p) == blob(FINAL, p)}")
for p in ("records/phase2.jsonl", "records/phase3.jsonl", "records/phase4.jsonl"):
    print(f"  {p} identical 1f8be04 -> 31c39fc: {blob(P4_FINAL, p) == blob(FINAL, p)}")
print(f"  records/phase5* at 550a736: {[p for p in git('ls-tree', '-r', '--name-only', PROT).split() if 'phase5' in p and p.startswith('records')]}")
p5i, p5 = jsonl(INTERIM, "records/phase5.jsonl"), jsonl(FINAL, "records/phase5.jsonl")
prefix = show(FINAL, "records/phase5.jsonl").splitlines()[:len(p5i)] == show(INTERIM, "records/phase5.jsonl").splitlines()
print(f"  phase5.jsonl: {len(p5i)} records at 29ff87a, {len(p5)} at 31c39fc; the interim lines are the final file's first "
      f"{len(p5i)} byte for byte: {prefix}")
logi, logf = show(INTERIM, "records/phase5.log").splitlines(), show(FINAL, "records/phase5.log").splitlines()
print(f"  phase5.log headers ('N jobs to run'): {[l for l in logf if 'jobs to run' in l]}; the log's run order equals the "
      f"jsonl's: {[tuple(l.split()[:2]) for l in logf[1:]] == [(r['arm'], str(r['seed'])) for r in p5]}")
print(f"  timestamp-like fields in the records: {sorted({k for r in p5 for k in r if re.search('time|date|start|host|cpu', k, re.I)})}")
# the 4-worker replay (replicate.py --phase 5: Pool(4).imap_unordered over the job list in order)
jobs = [(o, s) for o in ("ORACLE_ADAM", "ORACLE_DELTA") for s in (940, 941)]
for s in SEEDS:
    jobs += [("DELTA1", s), ("RESET_D8_ML", s), ("WIN3_SPLIT_D8_ML", s)]
wall = {(r["arm"], r["seed"]): r["wall_s"] for r in p5}
free, fin = [0.0] * 4, []
for j in jobs:
    w = min(range(4), key=lambda i: free[i])
    free[w] += wall[j]
    fin.append((free[w], j))
fin.sort()
order_ok = [j for _, j in fin] == [(r["arm"], r["seed"]) for r in p5]
t19, t64 = fin[len(p5i) - 1][0], fin[-1][0]
start_int = t_int - timedelta(seconds=t19)
start_fin = t_fin - timedelta(seconds=t64)
print(f"  replay (wall_s, 4 workers, jobs in replicate.py's order, each to the first free worker): completion order "
      f"equals the records' order: {order_ok}")
print(f"  19th completion at +{t19 / 3600:.2f} h, 20th at +{fin[len(p5i)][0] / 3600:.2f} h, last (64th) at +{t64 / 3600:.2f} h; "
      f"sum of wall_s {sum(wall.values()) / 3600:.2f} h")
print(f"  protocol 550a736 {t_prot.isoformat()}; interim 29ff87a {t_int.isoformat()} (+{(t_int - t_prot).total_seconds() / 3600:.2f} h); "
      f"report 31c39fc {t_fin.isoformat()} (+{(t_fin - t_prot).total_seconds() / 3600:.2f} h)")
print(f"  latest start consistent with the interim commit holding 19 records: {start_int.isoformat()}; "
      f"with the final commit holding all 64: {start_fin.isoformat()}")
print(f"  => the run started no later than {min(start_int, start_fin).strftime('%H:%M:%S')} UTC on "
      f"{min(start_int, start_fin).strftime('%d %b')}; protocol commit at {t_prot.strftime('%H:%M:%S')} UTC on "
      f"{t_prot.strftime('%d %b')}; slack between the protocol commit and that latest start: "
      f"{(min(start_int, start_fin) - t_prot).total_seconds() / 60:.1f} min")
slack_min = (min(start_int, start_fin) - t_prot).total_seconds() / 60
t20 = fin[len(p5i)][0]
print(f"  if the interim commit was made before the 20th record existed and the runs had no overhead beyond wall_s: the run "
      f"started after {(t_int - timedelta(seconds=t20)).strftime('%H:%M:%S')} UTC (a bound only under that assumption)")
last_if_at_commit = t_prot + timedelta(seconds=t64)
print(f"  a start at the protocol commit puts the last record at {last_if_at_commit.strftime('%H:%M:%S')} UTC or later, "
      f"{(t_fin - last_if_at_commit).total_seconds() / 60:.1f} min before the report commit; the last record is "
      f"{p5[-1]['arm']} {p5[-1]['seed']}")
chains = Counter()
free2 = [0.0] * 4
for j in jobs:
    w = min(range(4), key=lambda i: free2[i])
    free2[w] += wall[j]
    chains[w] += 1
print(f"  jobs per worker in the replay: {sorted(chains.values())}; worker end times (h): {[round(f / 3600, 2) for f in sorted(free2)]}")

# ── 2 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
H("2. Seeds 940-959 before 550a736")
for p in ("records/phase2.jsonl", "records/phase3.jsonl", "records/phase4.jsonl"):
    ss = sorted({r["seed"] for r in jsonl(PROT, p)})
    print(f"  R {p} at 550a736: seeds {ss[0]}-{ss[-1]} ({len(ss)} distinct); any in 940-959: {[s for s in ss if s in SEEDS]}")
print(f"  R pilot/oracle_seed1.log present at 550a736: {exists(PROT, 'pilot/oracle_seed1.log')} (seed 1)")
for ref, path in ((ML_X, "results/X/split_copy_results.json"), (ML_L, "results/L/split_copy_results.json")):
    st = rlib.load(ref, path)
    ss = sorted({int(k.split("|")[1]) for k in st["runs"]})
    print(f"  main line {ref}:{path}: seeds {ss}; any in 940-959: {[s for s in ss if s in SEEDS]}")


def walk(o, hits, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(k, str):
                m = re.match(r"^[A-Za-z][\w\-.+]*\|(\d+)", k)
                if m and int(m.group(1)) in SEEDS:
                    hits.append(f"{path}/{k}")
            if k == "seed" and isinstance(v, int) and v in SEEDS:
                hits.append(f"{path}/seed={v}")
            walk(v, hits, f"{path}/{k}" if len(path) < 60 else path)
    elif isinstance(o, list):
        for i, v in enumerate(o[:100000]):
            walk(v, hits, path)


tips = git("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin").split()
seen, tip_hits, nfiles = {}, [], 0
for b in tips:
    for when in ("now", "at 550a736"):
        ref = b if when == "now" else git("rev-list", "-1", f"--before={t_prot.isoformat()}", b).strip()
        if not ref:
            continue
        for ln in git("ls-tree", "-r", ref).splitlines():
            meta, path = ln.split("\t", 1)
            sha = meta.split()[2]
            if not re.search(r"\.(json|jsonl)$", path):
                continue
            if sha not in seen:
                nfiles += 1
                txt = git("cat-file", "-p", sha)
                hits = []
                try:
                    objs = [json.loads(txt)] if path.endswith(".json") else [json.loads(l) for l in txt.splitlines() if l.strip()]
                    for o in objs:
                        walk(o, hits)
                except Exception as e:  # noqa: BLE001
                    hits = [f"UNPARSED {type(e).__name__}"]
                seen[sha] = hits
            if seen[sha]:
                tip_hits.append((b, when, path, len(seen[sha]), seen[sha][:2]))
print(f"  every remote branch tip ({len(tips)}), now and at 550a736's time: {nfiles} distinct json/jsonl blobs parsed")
for h in sorted(set((b, w, p, n, tuple(x)) for b, w, p, n, x in tip_hits)):
    print(f"    hit: {h[0]} ({h[1]}) {h[2]}: {h[3]} entries, e.g. {list(h[4])}")
pre_hits = [h for h in tip_hits if h[1] == "at 550a736"]
print(f"  hits in branch tips as of 550a736's time: {len(pre_hits)}")
PAT = r'(\|9[45][0-9]["|]|"seed": ?9[45][0-9][,}]|range\(9[45][0-9]|\b9[45][0-9] ?- ?9[45][0-9]\b)'
hist = git("log", "--all", f"--until={t_prot.isoformat()}", "--format=%h %cI %s", "-G", PAT).strip().splitlines()
print(f"  history (all refs, commits up to 550a736's time) whose diff adds or removes a line matching {PAT}: {len(hist)}")
for c in hist:
    h = c.split()[0]
    lines = [l for l in git("show", "--format=", "-U0", h).splitlines()
             if l[:1] in "+-" and not l.startswith(("+++", "---")) and re.search(PAT, l)]
    print(f"    {c[:110]}")
    for l in lines[:3]:
        print(f"        {l.strip()[:150]}")
amend = show("db22857", "docs/reading/parallel_sessions_prompts.md")
print(f"  the Phase 4 amendment (db22857) assigned R 'WIN3_SPLIT_D8 on 920-951': {'920-951' in amend}; R's Phase 4 records "
      f"(1f8be04) hold seeds {min(r['seed'] for r in jsonl(P4_FINAL, 'records/phase4.jsonl'))}-"
      f"{max(r['seed'] for r in jsonl(P4_FINAL, 'records/phase4.jsonl'))} only")
r5 = show("0754d61", "docs/reading/parallel_sessions_prompts.md")
print(f"  round-5 prompts (0754d61, {ctime('0754d61').isoformat()}): 'R 940-959' in the seed blocks: {'R 940-959' in r5}")

# ── 3 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
H("3. R's Phase 5 records recomputed (31c39fc:records/phase5.jsonl)")
REC = {(r["arm"], r["seed"]): r for r in p5}
assert len(REC) == len(p5) == 64
ARM = {"COPY": "WIN3_SPLIT_D8_ML", "RESET": "RESET_D8_ML", "DELTA1": "DELTA1"}
runs5 = {n: {s: REC[(a, s)] for s in SEEDS} for n, a in ARM.items()}


def r_succ(n, r):
    return r_transition(r["evals"]) is not None if n == "DELTA1" else r_bound_routed(r)


def ml_adapt(r):
    """R's record in the shape rlib's main-line outcome reads: curve = evals, the map as a one-hot stream gate."""
    k = max(r["k"], max(r["map"]) + 1)
    gate = [[1.0 if c == m else 0.0 for c in range(k)] for m in r["map"]]
    return dict(ok=True, curve=r["evals"], end=dict(stream_acc=r["per_stream_acc"], stream_gate=gate, ch_map=r["map"]))


def ml_succ(n, r):
    a = ml_adapt(r)
    return rlib.bound(a) if n == "DELTA1" else rlib.bound_routed(a)


for o in ("ORACLE_ADAM", "ORACLE_DELTA"):
    print(f"  {o}: " + "; ".join(f"{s} bound {r_transition(REC[(o, s)]['evals']) is not None} T {r_transition(REC[(o, s)]['evals'])}"
                                  for s in (940, 941)))
print(f"  ORACLE_ADAM 940 and 941 evaluations at 1200 and 2400: {REC[('ORACLE_ADAM', 940)]['evals'][:2]} / "
      f"{REC[('ORACLE_ADAM', 941)]['evals'][:2]} (identical at two steps on different seeds; from 3600 they differ)")
flag_mis = [(r["arm"], r["seed"]) for r in p5
            if r_transition(r["evals"]) != r["transition"] or bool(r["bound"]) != (r_transition(r["evals"]) is not None)
            or bool(r["bound_routed"]) != r_bound_routed(r)]
print(f"  stored bound / bound_routed / transition disagreeing with R's recomputed outcome: {flag_mis}")
COUNT, WIL = {}, {}
for n, rs in runs5.items():
    x = sum(r_succ(n, r) for r in rs.values())
    xm = sum(ml_succ(n, r) for r in rs.values())
    lo, hi = rlib.wilson(x, 20)
    COUNT[n], WIL[n] = x, (lo, hi)
    print(f"  {n:6s} ({ARM[n]}): R's outcome {x}/20, main-line outcome on the same records {xm}/20 "
          f"({'same' if x == xm else 'DIFFERENT'}), Wilson [{lo:.4f}, {hi:.4f}]; successes {sorted(s for s, r in rs.items() if r_succ(n, r))}")
DISC = {}
for a, b in (("COPY", "RESET"), ("COPY", "DELTA1")):
    bb = sum(r_succ(a, runs5[a][s]) and not r_succ(b, runs5[b][s]) for s in SEEDS)
    cc = sum(r_succ(b, runs5[b][s]) and not r_succ(a, runs5[a][s]) for s in SEEDS)
    DISC[(a, b)] = (bb, cc, rlib.mcnemar1(bb, cc))
    print(f"  discordant {a} vs {b}: {bb} vs {cc} over 20 pairs; exact one-sided McNemar p = {rlib.mcnemar1(bb, cc):.4g}")
# the rule
H("3b. The reading rule (phase5.py:29-35): 'consistent' iff x/20 lies in the main line's pooled Wilson 95% interval, ends included")
MLP = {"COPY": (18, 0.699, 0.972), "RESET": (6, 0.145, 0.519), "DELTA1": (0, 0.0, 0.161)}
RULE_OK = {}
for n, (mx, mlo, mhi) in MLP.items():
    lo_u, hi_u = rlib.wilson(mx, 20)
    rng = [x for x in range(21) if mlo <= x / 20 <= mhi]
    rng_u = [x for x in range(21) if lo_u <= x / 20 <= hi_u]
    lab = "consistent" if mlo <= COUNT[n] / 20 <= mhi else "outside the main line's interval"
    RULE_OK[n] = (rng[0], rng[-1], lab, rng == rng_u)
    print(f"  {n}: main line {mx}/20, printed interval [{mlo}, {mhi}], unrounded Wilson [{lo_u:.4f}, {hi_u:.4f}]; consistent iff "
          f"{rng[0]} <= x <= {rng[-1]} (unrounded gives the same set: {rng == rng_u}); R's x = {COUNT[n]} -> {lab}")
    for p in sorted({mx / 20, 0.3, 0.5, 0.7, 0.9, 0.95, 0.05, 0.15}):
        pc = cdf(20, rng[-1], p) - (cdf(20, rng[0] - 1, p) if rng[0] > 0 else 0)
        if abs(p - mx / 20) < 1e-9 or (n == "COPY" and p in (0.7, 0.95)) or (n == "RESET" and p in (0.15, 0.5)) or (n == "DELTA1" and p in (0.05, 0.15)):
            print(f"      P(reads 'consistent' | true rate {p:.2f}) = {pc:.3f}")
# diagnostics
H("3c. The report's diagnostics")
cp, rs_ = runs5["COPY"], runs5["RESET"]
both = sorted(s for s in SEEDS if r_succ("COPY", cp[s]) and r_succ("RESET", rs_[s]))
nofire = [s for s in both if not cp[s]["splits"] and not rs_[s]["splits"]]
fired = [s for s in both if rs_[s]["splits"]]
ident = {s: [k for k in cp[s] if k not in ("arm", "wall_s") and cp[s][k] != rs_[s][k]] for s in nofire}
print(f"  shared binds (COPY and RESET both bound routed): {len(both)} {both}; no trigger in either arm: {nofire}; "
      f"fields other than arm/wall_s differing there: {ident}; RESET fired and bound: {len(fired)} {fired}")
fails = {n: sorted(s for s in SEEDS if not r_succ(n, runs5[n][s])) for n in ("COPY", "RESET")}
shares = {n: {s: max(Counter(runs5[n][s]["map"]).values()) for s in fails[n]} for n in ("COPY", "RESET")}
print(f"  failures: COPY {fails['COPY']} max streams per channel {shares['COPY']}; RESET ({len(fails['RESET'])}) {fails['RESET']} "
      f"max share {shares['RESET']}; R's failure labels {sorted({runs5[n][s]['failure'] for n in ('COPY', 'RESET') for s in fails[n]})}")
print(f"  RESET failed on COPY's failing seeds: {[s for s in fails['COPY'] if s in fails['RESET']]}")
nsp = {n: sum(len(runs5[n][s]["splits"]) for s in SEEDS) for n in ("COPY", "RESET")}
okt = {n: sum(1 for s in SEEDS for x in runs5[n][s]["splits"] if x["streams_on_c_star"] >= 2 and x["streams_on_c0"] == 0)
       for n in ("COPY", "RESET")}
latest = max(x["update"] for n in ("COPY", "RESET") for s in SEEDS for x in runs5[n][s]["splits"])
print(f"  trigger firings: COPY {nsp['COPY']}, RESET {nsp['RESET']}; labelled ok (c* >= 2 streams, c0 none): "
      f"COPY {okt['COPY']}/{nsp['COPY']}, RESET {okt['RESET']}/{nsp['RESET']}; latest firing {latest}")


def blocks(r):
    ev = dict((u, a) for u, a in r["evals"])
    sp = [x["update"] for x in r["splits"]]
    out = dict(checks=0, blocked=0, eligible=0)
    for u, pa in r["probes"]:
        if u < 4800:
            continue
        out["checks"] += 1
        if u in ev and pa >= 0.95 and ev[u] < 0.95:
            out["blocked"] += 1
            before = [x for x in sp if x < u]
            out["eligible"] += int(len(before) < 3 and (not before or u - before[-1] >= 4800))
    return out


BLK = {}
for n in ("COPY", "RESET"):
    tot = Counter()
    where = []
    for s in SEEDS:
        b = blocks(runs5[n][s])
        tot.update(b)
        if b["blocked"]:
            where.append((s, b["blocked"], b["eligible"], r_succ(n, runs5[n][s])))
    BLK[n] = (tot["checks"], tot["blocked"], tot["eligible"], where)
    print(f"  probe >= 0.95 while the evaluation < 0.95, checks >= 4800: {n} {tot['blocked']} of {tot['checks']} "
          f"({tot['eligible']} otherwise eligible); (seed, blocked, eligible, bound routed): {where}")
d1 = runs5["DELTA1"]
fa = sorted(r["final_acc"] for r in d1.values())
print(f"  DELTA1 final accuracy {fa[0]:.4f}-{fa[-1]:.4f}, median {statistics.median(fa):.4f}")
spread = {s: max(r["per_stream_acc"]) - min(r["per_stream_acc"]) for s, r in d1.items()}
p942 = d1[942]["per_stream_acc"]
print(f"  DELTA1 942 per-stream {min(p942):.4f}-{max(p942):.4f}; per-run spread (max - min stream) median "
      f"{statistics.median(spread.values()):.3f}, range {min(spread.values()):.3f}-{max(spread.values()):.3f}; runs with spread >= 0.25: "
      f"{sorted(s for s, v in spread.items() if v >= 0.25)}")
big = {s: (round(min(d1[s]['per_stream_acc']), 2), round(max(d1[s]['per_stream_acc']), 2)) for s, v in spread.items() if v >= 0.25}
print(f"     those runs' stream ranges: {big}")
dw = sorted(r["wall_s"] / 60 for r in d1.values())
print(f"  DELTA1 wall {dw[0]:.1f}-{dw[-1]:.1f} min; all 64 runs {sum(wall.values()) / 3600:.2f} h")
print(f"  9 vs 0 against 12 vs 0 (discordant pairs out of 20): Fisher two-sided p = {fisher_two(9, 20, 12, 20):.3f}; "
      f"RESET 9/20 vs the main line's 6/20: Fisher two-sided p = {fisher_two(9, 20, 6, 20):.3f}")

# ── 4 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
H("4. The main line's test_split_copy recomputed (X 592c574, L ffb0771; rlib = test_stream_recipe's outcome)")
X, L = rlib.load(ML_X, "results/X/split_copy_results.json"), rlib.load(ML_L, "results/L/split_copy_results.json")
print(f"  X's file identical at 592c574 and ffb0771: {blob(ML_X, 'results/X/split_copy_results.json') == blob(ML_L, 'results/X/split_copy_results.json')}")
MARMS = ("WIN3_SPLIT_D8", "WIN3_RESET_D8", "WIN3_SLOW_D8", "SINGLE_DELTA_D8", "SINGLE_HEBB_D8", "CEIL_C", "CEIL_DELTA_D8")
MR = {(m, a): rlib.runs(st, a) for m, st in (("X", X), ("L", L)) for a in MARMS}


def ml_ok(a, r):
    return rlib.bound_routed(r) if a.startswith("WIN3") else rlib.bound(r)


MLC = {}
for a in MARMS:
    xs = {m: sum(ml_ok(a, r) for r in MR[(m, a)].values()) for m in "XL"}
    ns = {m: len(MR[(m, a)]) for m in "XL"}
    rec_t = all(rlib.transition(r["curve"]) == r["transition"] for m in "XL" for r in MR[(m, a)].values())
    x, n = xs["X"] + xs["L"], ns["X"] + ns["L"]
    lo, hi = rlib.wilson(x, n)
    MLC[a] = (x, n, lo, hi)
    extra = ""
    if a.startswith("CEIL"):
        extra = " transitions " + str({m: [rlib.transition(r["curve"]) for _, r in sorted(MR[(m, a)].items())] for m in "XL"})
    if a.startswith("SINGLE"):
        accs = [r["acc"] for m in "XL" for r in MR[(m, a)].values()]
        extra = f" final accuracy {min(accs):.4f}-{max(accs):.4f} (X {min(r['acc'] for r in MR[('X', a)].values()):.2f}-" \
                f"{max(r['acc'] for r in MR[('X', a)].values()):.2f}, L {min(r['acc'] for r in MR[('L', a)].values()):.2f}-" \
                f"{max(r['acc'] for r in MR[('L', a)].values()):.2f})"
    print(f"  {a:16s} X {xs['X']}/{ns['X']}, L {xs['L']}/{ns['L']}, pooled {x}/{n} Wilson [{lo:.4f}, {hi:.4f}]; "
          f"recorded transition = recomputed: {rec_t};{extra}")
MLD = {}
for a, b in (("WIN3_SPLIT_D8", "WIN3_RESET_D8"), ("WIN3_SPLIT_D8", "SINGLE_DELTA_D8"), ("WIN3_RESET_D8", "WIN3_SLOW_D8")):
    bb = cc = 0
    for m in "XL":
        ra, rb = MR[(m, a)], MR[(m, b)]
        for s in sorted(set(ra) & set(rb)):
            bb += ml_ok(a, ra[s]) and not ml_ok(b, rb[s])
            cc += ml_ok(b, rb[s]) and not ml_ok(a, ra[s])
    MLD[(a, b)] = (bb, cc, rlib.mcnemar1(bb, cc))
    print(f"  pooled discordant {a} vs {b}: {bb} vs {cc}, exact one-sided p = {rlib.mcnemar1(bb, cc):.3g}")
doc = show(ML_L, "test_split_copy.py")
print(f"  ffb0771 docstring: pooled bands line present: {'WIN3_SPLIT_D8 RELIABLE [0.699, 0.972]; WIN3_RESET_D8 MINORITY [0.145, 0.519]' in doc}; "
      f"'SINGLE_DELTA_D8 NEVER [0, 0.161]': {'SINGLE_DELTA_D8 NEVER [0, 0.161]' in doc}; "
      f"X 'final accuracy 0.08-0.76': {'(final accuracy 0.08-0.76)' in doc}; L '0.13-0.55': {'accuracy 0.13-0.55)' in doc}")

# ── 5 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
H("5. 'From the spec and docstring alone': what git can show")
for ref in (PROT, INTERIM, FINAL):
    tree = git("ls-tree", "-r", "--name-only", ref).split()
    ml = [p for p in tree if re.search(r"split_copy|test_|specs/|results/|explore_", p) and p != "test_replicate.py"]
    print(f"  R's tree at {ref}: {len(tree)} files; any main-line file (test_*.py other than test_replicate.py, specs/, results/, "
          f"explore_*): {ml}")
diff_md = show(FINAL, "DIFF.md")
print(f"  DIFF.md (Phase 3, 00990ea) says R read the main line at 404e3a0 incl. test_window_gate.py: "
      f"{'404e3a0' in diff_md and 'test_window_gate.py' in diff_md}; and test_stream_recipe (outcome, routed): "
      f"{'test_stream_recipe' in diff_md}")
t404 = git("ls-tree", "-r", "--name-only", P3_READ).split()
print(f"  main line at 404e3a0 ({ctime(P3_READ).isoformat()}) already holds: "
      f"{[p for p in ('test_split_copy.py', 'specs/test_split_copy.md', 'explore_delta_mem.py', 'explore_f_delta_fast.py') if p in t404]}")
twg = show(P3_READ, "test_window_gate.py").splitlines()
ro = [i + 1 for i, l in enumerate(twg) if "def reset_op" in l or 'kind="reset"' in l]
print(f"  test_window_gate.py at 404e3a0: WIN3_RESET_D8 and reset_op at lines {ro}: "
      f"{[twg[i - 1].strip()[:90] for i in ro]} / '{twg[ro[-1]].strip()[:90]}'")
p4doc = show(P4_FINAL, "phase4.py")
print(f"  phase4.py:6-9 (R's Phase 4 recipe = Phase 5's COPY) takes the main line's probe and key positions: "
      f"'the split as the main line specifies it' {'the split as the main line specifies it' in p4doc}, 'Generator(12345)' "
      f"{'Generator(12345)' in p4doc}, '32 body key positions' {'32 body key positions' in p4doc}; DIFF.md D-1/D-2 give them "
      f"as read from the code ('Theirs'): {'D-1' in diff_md and 'D-2' in diff_md}")

# ── 6 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
H("6. UNSTATED.md at 31c39fc")
un = show(FINAL, "UNSTATED.md").splitlines()
urows = [l for l in un if re.match(r"\|\s*U-\d+", l)]
ids = [int(re.match(r"\|\s*U-(\d+)", l).group(1)) for l in urows]
p5rows = [l for l in urows if 50 <= int(re.match(r"\|\s*U-(\d+)", l).group(1)) <= 70]
labs = Counter(re.search(r"\[([A-Z/]+)\]", l.split("|")[2]).group(1) if re.search(r"\[([A-Z/]+)\]", l.split("|")[2]) else "-"
               for l in p5rows)
print(f"  rows: {len(urows)} (ids {min(ids)}-{max(ids)}, distinct {len(set(ids))}); Phase 5 rows U-50..U-70: {len(p5rows)} "
      f"(contiguous: {sorted(i for i in ids if i >= 50) == list(range(50, 71))}); labels {dict(labs)}")
n_un_p4 = len([l for l in show(P4_FINAL, "UNSTATED.md").splitlines() if re.match(r"\|\s*U-\d+", l)])
print(f"  unchanged 550a736 -> 31c39fc: {blob(PROT, 'UNSTATED.md') == blob(FINAL, 'UNSTATED.md')}; rows at 1f8be04: "
      f"{n_un_p4}")
for key, uid in (("W_g's Adam state only", 50), ("L2-normalised", 52), ("exactly the Hebbian", 53), ("N = 256, D = 32", 58)):
    hit = [l for l in p5rows if l.startswith(f"| U-{uid} ")]
    print(f"  U-{uid} holds '{key}': {bool(hit) and key in hit[0]}")

# ── 7 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
H("7. Rows")
rr, ph = "replicate_report.md", "phase5.py"
x5 = (COUNT["COPY"], COUNT["RESET"], COUNT["DELTA1"])
row(f"{rr}:174, git log", "protocol committed (550a736) before the first record; not changed after",
    f"550a736 {t_prot.strftime('%m-%d %H:%M:%S')} has no records/phase5*; first records 29ff87a {t_int.strftime('%m-%d %H:%M:%S')}; "
    f"phase5.py above RESULT, replicate.py, analyze.py, phase4.py, UNSTATED.md identical 550a736 -> 31c39fc ({proto_same and code_same})",
    "MATCH" if proto_same and code_same else "DISAGREE")
row(f"{rr}:174, {ph}:11", "committed before the first run",
    f"replay of wall_s on 4 workers: start no later than {min(start_int, start_fin).strftime('%H:%M:%S')} UTC, "
    f"{slack_min:.1f} min after the protocol commit (the 64 runs fill {t64 / 3600:.2f} of the "
    f"{(t_fin - t_prot).total_seconds() / 3600:.2f} h between protocol and report commits)",
    "NOT RECORD-CHECKABLE",
    "Consistent with the claim but tight: the records carry no timestamps, and a start a few minutes before 22:47:46 cannot "
    "be excluded from git (Phase 4's replayed start was 70 s after 464be8d).")
row(f"{ph}:11, UNSTATED.md:93, :136 (U-70)", "fixed before any Phase 5 run; every entry written before any run; no pilot",
    "phase5.py's protocol, replicate.py, UNSTATED.md identical 550a736 -> 31c39fc; only the RESULT block was filled",
    "NOT RECORD-CHECKABLE", "What ran before the commit (U-70's unit tests, the 920 regression, a timing) leaves no record.")
row(f"{ph}:13, commit 550a736 title", "seeds 940-959, unused before",
    f"R's phase2-4 seeds 900-939; main-line split_copy 560-569/1560-1569; no json/jsonl record with a 940-959 seed on any "
    f"branch tip as of 550a736 ({len(pre_hits)} hits); history pickaxe: no record line",
    "MATCH" if not pre_hits else "DISAGREE", "940-951 were named for R in the Phase 4 amendment (db22857) but never run.")
p24 = all(blob(P4_FINAL, q) == blob(FINAL, q) for q in ("records/phase2.jsonl", "records/phase3.jsonl", "records/phase4.jsonl"))
row(f"{ph}:14", "Phases 2-4 records are not touched", f"phase2/3/4 .jsonl identical 1f8be04 -> 31c39fc: {p24}", "MATCH" if p24 else "DISAGREE")
row(f"{rr}:172-173", "from the spec and docstring at ffb0771 only; no code below the docstring, nothing of F's delta modules",
    "R's branch is an orphan (no merge-base) and never holds a main-line file; but R read the main line at 404e3a0 in Phase 3, "
    "whose tree held test_split_copy.py, the spec and both delta modules",
    "NOT RECORD-CHECKABLE", "What R opened is not in git. See section 5.")
row(f"{rr}:174", "the Phase 4 amended text states a different rule (<= 27/32 on fresh seeds)",
    f"db22857 'Phase 4, amended' (b): 'x <= 27/32' on 920-951: {'27/32' in amend}", "MATCH" if "27/32" in amend else "DISAGREE")
row(f"{rr}:175-179", "the guesses are U-50 to U-70; the four that matter most",
    f"U-50..U-70 = {len(p5rows)} rows (labels {dict(labs)}); RESET = U-50, raw queries/L2 keys = U-52, decay = U-53, width = U-58",
    "MATCH", "U-50 calls 'W_g's Adam state zeroed, W_g untouched' the spec's wording: it is the docstring's; the spec says "
    "'Adam state zeroed, W_g untouched' (spec :12) and 'W_g's Adam state zeroed' (:138). The main-line code R read in Phase 3 "
    "(test_window_gate.py:510-511 at 404e3a0) zeroes W_g's state only.")
row(f"{rr}:194, {ph}:49", "COPY 18/20 [0.699, 0.972]",
    f"{COUNT['COPY']}/20 [{WIL['COPY'][0]:.3f}, {WIL['COPY'][1]:.3f}]",
    same((COUNT["COPY"], round(WIL["COPY"][0], 3), round(WIL["COPY"][1], 3)), (18, 0.699, 0.972)))
row(f"{rr}:195, {ph}:50", "RESET 9/20 [0.258, 0.658]",
    f"{COUNT['RESET']}/20 [{WIL['RESET'][0]:.3f}, {WIL['RESET'][1]:.3f}]",
    same((COUNT["RESET"], round(WIL["RESET"][0], 3), round(WIL["RESET"][1], 3)), (9, 0.258, 0.658)))
row(f"{rr}:196, {ph}:51", "DELTA1 0/20 [0, 0.161]",
    f"{COUNT['DELTA1']}/20 [{WIL['DELTA1'][0]:.3f}, {WIL['DELTA1'][1]:.3f}]",
    same((COUNT["DELTA1"], round(WIL["DELTA1"][1], 3)), (0, 0.161)))
row(f"{rr}:194-196, {ph}:49-51, commit 31c39fc title", "each arm 'consistent'",
    "; ".join(f"{n} {COUNT[n]}/20 in {RULE_OK[n][0]}..{RULE_OK[n][1]} -> {RULE_OK[n][2]}" for n in ("COPY", "RESET", "DELTA1")),
    "MATCH" if all(RULE_OK[n][2] == "consistent" for n in RULE_OK) else "DISAGREE",
    "The rule's ranges (phase5.py:32-34) recomputed: the same with the unrounded intervals.")
row(f"{ph}:32-34", "consistent iff 14..19 / 3..10 / 0..3",
    "; ".join(f"{n} {RULE_OK[n][0]}..{RULE_OK[n][1]}" for n in ("COPY", "RESET", "DELTA1")),
    same(tuple((RULE_OK[n][0], RULE_OK[n][1]) for n in ("COPY", "RESET", "DELTA1")), ((14, 19), (3, 10), (0, 3))))
row(f"{rr}:195", "0.45 is inside [0.145, 0.519]", f"9/20 = 0.45, 0.145 <= 0.45 <= 0.519", "MATCH")
oa = [r_transition(REC[("ORACLE_ADAM", s)]["evals"]) for s in (940, 941)]
od = [r_transition(REC[("ORACLE_DELTA", s)]["evals"]) for s in (940, 941)]
row(f"{rr}:197, {ph}:48", "ORACLE_ADAM 2/2 (6000, 6000); ORACLE_DELTA 2/2 (1200, 1200)", f"{oa}; {od}",
    same((oa, od), ([6000, 6000], [1200, 1200])),
    "Not in the report: ORACLE_ADAM 940 and 941 have identical evaluations at 1200 (0.64893) and 2400 (0.79346).")
ceil_c = [rlib.transition(r["curve"]) for m in "XL" for _, r in sorted(MR[(m, "CEIL_C")].items())]
ceil_d = [rlib.transition(r["curve"]) for m in "XL" for _, r in sorted(MR[(m, "CEIL_DELTA_D8")].items())]
row(f"{rr}:197", "main line CEIL_C 4/4; CEIL_DELTA_D8 4/4 (all at 1200)", f"CEIL_C {ceil_c}; CEIL_DELTA_D8 {ceil_d}",
    "MATCH" if len(ceil_c) == 4 and None not in ceil_c and ceil_d == [1200] * 4 else "DISAGREE",
    "'(all at 1200)' holds for CEIL_DELTA_D8 only; CEIL_C bound at 4800, 3600 on each machine.")
for n, a in (("COPY", "WIN3_SPLIT_D8"), ("RESET", "WIN3_RESET_D8"), ("DELTA1", "SINGLE_DELTA_D8")):
    x, nn, lo, hi = MLC[a]
    want = {"COPY": (18, 0.699, 0.972), "RESET": (6, 0.145, 0.519), "DELTA1": (0, 0.0, 0.161)}[n]
    row(f"{rr}:194-196, {ph}:6-7, :32-34", f"main line pooled {a} {want[0]}/20 [{want[1]}, {want[2]}]",
        f"{x}/{nn} [{lo:.4f}, {hi:.4f}] from X 592c574 + L ffb0771",
        same((x, nn, round(lo, 3), round(hi, 3)), (want[0], 20, want[1], want[2])))
b, c, p = DISC[("COPY", "RESET")]
row(f"{rr}:200, {ph}:52", "COPY vs RESET 9 vs 0 (p = 0.0020)", f"{b} vs {c}, p = {p:.4f}", same((b, c, round(p, 4)), (9, 0, 0.002)))
b, c, p = DISC[("COPY", "DELTA1")]
row(f"{rr}:201, {ph}:52-53", "COPY vs DELTA1 18 vs 0 (p = 3.8e-6)", f"{b} vs {c}, p = {p:.3g}", same((b, c, f"{p:.2g}"), (18, 0, "3.8e-06")))
mb, mc, mp = MLD[("WIN3_SPLIT_D8", "WIN3_RESET_D8")]
row(f"{rr}:200, {ph}:6", "the main line had 12 vs 0", f"{mb} vs {mc} (p = {mp:.3g})", same((mb, mc), (12, 0)))
mb, mc, mp = MLD[("WIN3_SPLIT_D8", "SINGLE_DELTA_D8")]
row(f"{rr}:201, {ph}:7", "the main line had 18 vs 0", f"{mb} vs {mc} (p = {mp:.3g})", same((mb, mc), (18, 0)))
row(f"{rr}:204", "no seed went the other way", f"RESET only {DISC[('COPY', 'RESET')][1]}", same(DISC[("COPY", "RESET")][1], 0))
row(f"{rr}:205-207", "9 shared binds; 941, 943 no trigger, bit-identical; on the other seven the reset fired",
    f"{len(both)} {both}; no trigger {nofire}, differing fields {ident}; fired {len(fired)}",
    "MATCH" if (len(both), nofire, len(fired)) == (9, [941, 943], 7) and not any(ident.values()) else "DISAGREE",
    "'bit-identical' checked on every recorded field but wall_s (curve, probes, map, accuracies).")
row(f"{rr}:208", "RESET 0.45 against 0.30, inside their interval", f"9/20 = 0.45; main line 6/20 = 0.30", "MATCH")
row(f"{rr}:209", "every failure a merge; 11 RESET failures, COPY 945 and 947 (three share); RESET failed on both",
    f"COPY {shares['COPY']}; RESET {len(fails['RESET'])} failures, max share {sorted(set(shares['RESET'].values()))}; "
    f"RESET failed on {[s for s in fails['COPY'] if s in fails['RESET']]}",
    "MATCH" if (len(fails["RESET"]), shares["COPY"]) == (11, {945: 3, 947: 3}) and set(shares["RESET"].values()) <= {2, 3}
    and all(s in fails["RESET"] for s in fails["COPY"]) else "DISAGREE",
    "'Merge' is R's map-based label (U-35, DIFF.md D-14), not the main line's eta^2 fail_class (R records no eta^2).")
row(f"{rr}:210, {ph}:53", "37 copies and 46 resets; every target labelled ok",
    f"COPY {nsp['COPY']} ({okt['COPY']} ok), RESET {nsp['RESET']} ({okt['RESET']} ok)",
    same((nsp["COPY"], okt["COPY"], nsp["RESET"], okt["RESET"]), (37, 37, 46, 46)))
row(f"{rr}:211, {ph}:53-54", "blocks: 1 of 175 in COPY (958, bound); 2 of 246 in RESET (neither eligible); decided no run",
    f"COPY {BLK['COPY'][1]} of {BLK['COPY'][0]} {BLK['COPY'][3]}; RESET {BLK['RESET'][1]} of {BLK['RESET'][0]}, eligible {BLK['RESET'][2]} {BLK['RESET'][3]}",
    "MATCH" if (BLK["COPY"][:2], BLK["RESET"][:3]) == ((175, 1), (246, 2, 0)) and BLK["COPY"][3] and BLK["COPY"][3][0][0] == 958
    and BLK["COPY"][3][0][3] else "DISAGREE",
    "'Decided no run': the only eligible block is on a run that bound routed; the RESET blocks were not eligible.")
row(f"{rr}:213, {ph}:54-55", "DELTA1 never bound; final 0.09-0.78, median 0.29",
    f"{COUNT['DELTA1']}/20; {fa[0]:.3f}-{fa[-1]:.3f}, median {statistics.median(fa):.3f}",
    same((COUNT["DELTA1"], round(fa[0], 2), round(fa[-1], 2), round(statistics.median(fa), 2)), (0, 0.09, 0.78, 0.29)))
accs = [r["acc"] for m in "XL" for r in MR[(m, "SINGLE_DELTA_D8")].values()]
row(f"{rr}:213", "the main line reports 0.08-0.76", f"SINGLE_DELTA_D8 pooled {min(accs):.3f}-{max(accs):.3f} (X 0.08-0.76, L 0.13-0.55)",
    same((round(min(accs), 2), round(max(accs), 2)), (0.08, 0.76)), "The range is X's; L's lies inside it.")
row(f"{rr}:214", "per-stream accuracy roughly even within a run (942: 0.72-0.79)",
    f"942 {min(p942):.3f}-{max(p942):.3f}; spread median {statistics.median(spread.values()):.2f}, >= 0.25 on "
    f"{len(big)} of 20 runs {big}",
    "PARTIAL" if big else "MATCH",
    "942's range matches; 'roughly even' holds for most runs, not for these.")
row(f"{rr}:215", "the perfect gate on delta channels bound at 1200 on both seeds, so the failure is not in the delta write",
    f"ORACLE_DELTA {od}", "MATCH" if od == [1200, 1200] else "DISAGREE",
    "The count matches; 'not in the delta write' is R's inference (the oracle is k = 16 with a one-hot gate, not k = 1).")
row(f"{rr}:216, {ph}:55", "78.3 h; 4 workers x 1 thread; each DELTA1 about 2 h",
    f"sum wall_s {sum(wall.values()) / 3600:.2f} h; DELTA1 {dw[0]:.0f}-{dw[-1]:.0f} min",
    same(round(sum(wall.values()) / 3600, 1), 78.3), "Workers and threads are not in the records (the replay with 4 reproduces the order).")
row(f"{rr}:218-221", "from the written specification alone, an independent implementation reproduces the copy arm's count, "
    "the reset within the interval, DELTA1's 0/20",
    "counts 18/20, 9/20 (in 3..10), 0/20 match; but COPY is R's Phase 4 recipe (D-1 probe, D-2 key positions taken from the "
    "main line's code, phase4.py) and RESET shares that trigger; R read test_window_gate.py (WIN3_RESET_D8, reset_op) in Phase 3",
    "PARTIAL", "True of the numbers; 'from the written specification alone' holds at most for DELTA1. The report's own "
    "Arms table (:185) and the user's prompt say COPY is the Phase 4 recipe.")
row(f"{rr}:223", "advantage holds in direction (9 vs 0 against 12 vs 0); smaller because RESET bound more often; "
    "9 vs 0 and 12 vs 0 not distinguishable with 20 seeds",
    f"COPY 18 = 18; RESET 9 vs 6; Fisher two-sided on 9/20 vs 12/20 discordant p = {fisher_two(9, 20, 12, 20):.2f}, "
    f"on RESET 9/20 vs 6/20 p = {fisher_two(9, 20, 6, 20):.2f}", "MATCH", "'Not distinguishable' is a reading; the tests above support it.")
row(f"{ph}:6-7", "main line C1 18/20 vs 6/20 (12 vs 0); C3 18/20 vs 0/20 (18 vs 0)",
    f"{MLC['WIN3_SPLIT_D8'][0]}/20 vs {MLC['WIN3_RESET_D8'][0]}/20 ({MLD[('WIN3_SPLIT_D8', 'WIN3_RESET_D8')][0]} vs "
    f"{MLD[('WIN3_SPLIT_D8', 'WIN3_RESET_D8')][1]}); vs {MLC['SINGLE_DELTA_D8'][0]}/20 ({MLD[('WIN3_SPLIT_D8', 'SINGLE_DELTA_D8')][0]} vs "
    f"{MLD[('WIN3_SPLIT_D8', 'SINGLE_DELTA_D8')][1]})",
    same((MLC["WIN3_SPLIT_D8"][0], MLC["WIN3_RESET_D8"][0], MLC["SINGLE_DELTA_D8"][0]) + MLD[("WIN3_SPLIT_D8", "WIN3_RESET_D8")][:2]
         + MLD[("WIN3_SPLIT_D8", "SINGLE_DELTA_D8")][:2], (18, 6, 0, 12, 0, 18, 0)))
row(f"{ph}:15-18, {rr}:185", "COPY = my Phase 4 recipe unchanged",
    f"replicate.py changed 1f8be04 -> 550a736 (Split.apply refactored into _copy, op='copy' default; new arms); the copy "
    f"path's operations and order are the same in the diff",
    "NOT RECORD-CHECKABLE", "No Phase 5 record reruns a Phase 4 run; U-70 reports a 2400-update regression of 920 that "
    "left no record.")

st = Counter(r[3] for r in ROWS)
for i, (src, stmt, rec, s, note) in enumerate(ROWS, 1):
    print(f"  {i:2d} | {s:20s} | {src} | {stmt} | recomputed: {rec}" + (f" | {note}" if note else ""))
print("\n  counts: " + ", ".join(f"{k} {st.get(k, 0)}" for k in ("MATCH", "PARTIAL", "DISAGREE", "NOT RECORD-CHECKABLE"))
      + f"; total {len(ROWS)}")
