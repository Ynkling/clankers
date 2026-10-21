#!/usr/bin/env python3
"""
screen_checks_s89_G_v2.py -- Referee V, 20 October 2026. Pre-run checks of Session G's revised S89 (claude/explore-G at
003a370e and db14e9b9: explore_g_s89.py, explore_g_s89_child.py, the vendored explore_f_tasks.py) for
screen_review_s89_G_v2.md.

Reads git objects only (git show / cat-file / ls-tree / rev-list); trains nothing; imports nothing from the repository.
torch is not installed here, so G's torch code is not executed. What IS executed:
  - the parent's reading code (explore_g_s89.py at db14e9b9: readings, arm_name, clamps, commit, med_commit, _fake,
    inv_view, recorded's lookup rule) and explore_g_rhl_parent's bound / routed_star / routed_bound / paired / beats and
    test_conv_lr.mcnemar_greater, exec'd from the blobs (pure Python, no torch), on every outcome and on S87's records;
  - F's pool_outcome (explore_f_tasks.py at b0c6332), exec'd from the blob with DISC_COS = 0.5 (test_router_discovery);
  - numpy replicas of RandHeaderTask(P=4, nb=2, lmin=1, lmax=3).make_batch, F's pool / allpool relabelling, G's
    repeat_mask, S87's largest-ratio cut, explore_b18_gates.eta2_pool, F's eta2_rows / inv_from_gate and
    pool_routing_stats' margin (distributional replicas: same distributions, not torch-bit-identical draws).
Output: screen_checks_s89_G_v2_out.txt (sections §0-§9).
"""
import ast
import functools
import itertools
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
import types
from collections import Counter, defaultdict

import numpy as np

G1, G2 = "003a370e", "db14e9b9"
GTIP = "origin/claude/explore-G"
F_NEW, F_MOD, F_OLD = "b0c6332", "50dc8e4", "a199d5e"
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


def git(*args, binary=False):
    r = subprocess.run(["git", *args], capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode("utf-8", "replace")


def show(rev, path, binary=False):
    return git("show", f"{rev}:{path}", binary=binary)


def lines_of(src, needle):
    return [i for i, l in enumerate(src.splitlines(), 1) if needle in l]


# ══ §0 refs ═════════════════════════════════════════════════════════════════════════════════════════════════════════
say("§0 REFS (after git fetch)")
for ref in (GTIP, "origin/claude/explore-F", "origin/claude/explore-H"):
    say(f"  {ref}: {git('log', '-1', '--format=%h %ci %s', ref).strip()[:160]}")
say("  explore-G commits after db14e9b9: " + (git("log", "--format=%h %ci %s", f"{G2}..{GTIP}").strip() or "none"))
s89_files = [l for l in git("ls-tree", "-r", "--name-only", GTIP).splitlines() if "s89" in l]
say(f"  S89 files at the explore-G tip: {s89_files}")
if "explore_out/G/s89.log" in s89_files:
    log = show(GTIP, "explore_out/G/s89.log")
    say(f"  explore_out/G/s89.log: {len(log.splitlines())} lines; git line: "
        f"{[l.strip() for l in log.splitlines() if 'git ' in l][:1]}; mentions 'S89 validity': {'S89 validity' in log}; "
        f"'runs queued': {'runs queued' in log}; any learned arm key (pool./allpool.): "
        f"{bool(re.search(r'(pool|allpool)\\.(BASELINE|REG3)', log))}; last line: {log.splitlines()[-1].strip()[:140]}")
say(f"  s89_results.json / s89_validity.json / checks_s89_*.json at the tip: "
    f"{[f for f in s89_files if f.endswith('.json')] or 'none'}")

# ══ §1 the vendored module ══════════════════════════════════════════════════════════════════════════════════════════
say("\n§1 VENDORED explore_f_tasks.py")
blobs = {}
for rev in (G1, G2, GTIP, F_NEW, F_MOD, F_OLD, "origin/claude/explore-F"):
    b = show(rev, "explore_f_tasks.py", binary=True)
    blobs[rev] = b
    say(f"  {rev:26s} blob {git('rev-parse', f'{rev}:explore_f_tasks.py').strip()}  {len(b)} bytes")
say(f"  G db14e9b9 == F b0c6332: {blobs[G2] == blobs[F_NEW]}; == F's tip: {blobs[G2] == blobs['origin/claude/explore-F']}; "
    f"G 003a370e == F 50dc8e4: {blobs[G1] == blobs[F_MOD]}")
say(f"  b0c6332 starts with 50dc8e4's {len(blobs[F_MOD])} bytes: {blobs[F_NEW].startswith(blobs[F_MOD])}; with a199d5e's "
    f"{len(blobs[F_OLD])} bytes: {blobs[F_NEW].startswith(blobs[F_OLD])}")
ch = show(G2, "explore_g_s89_child.py")
consts = {m.group(1): m.group(2) for m in re.finditer(r'\("?(\w{7})"?, "([0-9a-f]{40})", (\d+)\)', ch)}
say(f"  child constants (child:90-92): F_BLOB {re.search(r'F_BLOB = .*?\"([0-9a-f]{40})\"', ch).group(1)[:12]}, PREFIXES "
    f"{[(m.group(1), m.group(2)[:12], m.group(3)) for m in re.finditer(r'\(\"(\w{7})\", \"([0-9a-f]{40})\", (\d+)\)', ch)]}")
for c, bl, ln in [(m.group(1), m.group(2), int(m.group(3))) for m in re.finditer(r'\("(\w{7})", "([0-9a-f]{40})", (\d+)\)', ch)]:
    say(f"    {c}: rev-parse {git('rev-parse', f'{c}:explore_f_tasks.py').strip()[:12]} (constant {bl[:12]}), "
        f"length {len(blobs[F_MOD] if c == F_MOD else blobs[F_OLD])} (constant {ln})")
say(f"  the hash CHECK (child:528-532) passes when F's commit is absent from the clone (hf None) on the constant alone: "
    f"{'hf is None or hf == F_BLOB' in ch}")

# ══ §2 the code paths (ast) ═════════════════════════════════════════════════════════════════════════════════════════
say("\n§2 CODE PATHS (ast of db14e9b9)")
par = show(G2, "explore_g_s89.py")
pt, ct = ast.parse(par), ast.parse(ch)


def defs(tree):
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out.setdefault(n.name, n)
    return out


pf, cf = defs(pt), defs(ct)
say(f"  parent: {len(par.splitlines())} lines; defs {sorted(pf)}")
for fn in ("arm_name", "clamps", "readings", "report", "check_readings", "inv_view", "recorded"):
    n = pf[fn]
    calls = sorted({ast.unparse(c.func) for c in ast.walk(n) if isinstance(c, ast.Call)
                    and ast.unparse(c.func).startswith(("hp.", "gc.", "rb", "readings", "inv_view", "arm_name", "clamps"))})
    say(f"    {fn} (s89:{n.lineno}-{n.end_lineno}) calls {calls}")
say(f"  rb = hp.routed_bound (s89:{lines_of(par, 'rb = hp.routed_bound')}); hp.not_worse_1 used: {'not_worse_1(' in par}")


def enclosing(tree, src, needle):
    res = []
    for ln in lines_of(src, needle):
        best = None
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.lineno <= ln <= n.end_lineno:
                if best is None or n.lineno > best.lineno:
                    best = n
        res.append(f"{ln}({best.name if best else 'module'})")
    return res


say("  lite-vocabulary functions in the child, by line(enclosing function): on the pool path?")
for needle in ("rc.stats_fn(", "ft.body_positions(", "dc.key_eta(", "gc.stream_acc(", "ft.write_order_acc(", "rc.acc_by_cell(",
               "rc.layer_stats(", "rc.probe_rh(", "rc.roles(", "perfect_gate_general(", "- task.S", "x < S", "ft.rand_header"):
    say(f"    {needle!r}: {enclosing(ct, ch, needle)}")
run_src = ast.get_source_segment(ch, cf["run"])
say(f"  run() (child:{cf['run'].lineno}-{cf['run'].end_lineno}) passes stats_fn89 to run_one: {'stats_fn=stats_fn89' in run_src}; "
    f"post-run: {[f for f in ('stream_acc89', 'probe_rh89', 'acc_by_cell89', 'write_order_acc89') if f in run_src]}")
say(f"  STATS_POOL: {[l.strip() for l in ch.splitlines() if l.startswith('STATS_POOL')]}")
# what the hinge path reads from the task
for name in ("_assert_tokens", "repeat_mask", "frequent", "hinge_mask", "term_value", "key_eta89", "key_pos_ids"):
    node = cf[name]
    attrs = sorted({ast.unparse(n) for n in ast.walk(node) if isinstance(n, ast.Attribute)
                    and ast.unparse(n).startswith(("task.", "t.", "self.t_"))})
    calls = sorted({ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)
                    and not ast.unparse(n.func).startswith(("torch.", "F.", "int", "bool", "len", "isinstance", "str"))})
    say(f"    {name} (child:{node.lineno}-{node.end_lineno}): task attributes {attrs}; calls {calls}")
fmod = blobs[F_NEW].decode()
fs = ast.get_source_segment(fmod, defs(ast.parse(fmod))["freq_sample"])
say(f"    ft.freq_sample (F b0c6332:{lines_of(fmod, 'def freq_sample')}): {fs.splitlines()[-1].strip()}")
fw = method = next(n for n in cf["Term89"].body if isinstance(n, ast.FunctionDef) and n.name == "forward")
say(f"    Term89.forward (child:{fw.lineno}-{fw.end_lineno}) calls term_value(self.t_kind, live, tokens, self.t_task): "
    f"{'term_value(self.t_kind, live, tokens, self.t_task)' in ast.get_source_segment(ch, fw)}; the tokens argument is the "
    f"forward's input; training gate: {[l.strip() for l in ast.get_source_segment(ch, fw).splitlines() if 'training =' in l]}")

# ══ §3 the readings: the parent's own code on every outcome ═════════════════════════════════════════════════════════
say("\n§3 READINGS: explore_g_s89.readings() exec'd from db14e9b9 on every count vector of the 8 per-seed patterns")
rhp = show(G2, "explore_g_rhl_parent.py")
tcl = show(G2, "test_conv_lr.py")
mc_src = ast.get_source_segment(tcl, defs(ast.parse(tcl))["mcnemar_greater"])
ns_gc = {"math": math}
exec(mc_src, ns_gc)
MC = functools.lru_cache(None)(ns_gc["mcnemar_greater"])
gc_ns = types.SimpleNamespace(mcnemar_greater=MC, cnt=lambda k, n: f"{k}/{n}")
hp_ns = {"gc": gc_ns, "ROUTE_MARGIN": 0.9, "ROUTE_ETA": 0.9, "DISC_COS": 0.5}
rhp_defs = defs(ast.parse(rhp))
for fn in ("bound", "routed_star", "routed_bound", "paired", "beats", "not_worse_1"):
    exec(ast.get_source_segment(rhp, rhp_defs[fn]), hp_ns)
say(f"  explore_g_rhl_parent constants at db14e9b9: {[l for l in rhp.splitlines() if l.startswith(('ROUTE_MARGIN', 'DISC_COS'))]}")
hp = types.SimpleNamespace(**{k: v for k, v in hp_ns.items() if callable(v)})
P_NS = {"hp": hp, "gc": gc_ns, "statistics": statistics, "rb": hp.routed_bound}
for node in pt.body:                                   # module-level constants the readings use
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("P4", "NAMES", "NEEDS_ALLPOOL", "ROLE")
                                            for t in node.targets):
        exec(ast.unparse(node), P_NS)
for fn in ("commit", "med_commit", "arm_name", "clamps", "readings", "_fake", "inv_view"):
    exec(ast.get_source_segment(par, pf[fn]), P_NS)
readings, _fake, inv_view = P_NS["readings"], P_NS["_fake"], P_NS["inv_view"]
NAMES, NEEDS_ALLPOOL, P4 = P_NS["NAMES"], P_NS["NEEDS_ALLPOOL"], P_NS["P4"]
say(f"  P4 = {P4} (= mcnemar(4, 0) {MC(4, 0)}); NAMES {NAMES}; allpool FQ 'needs' -> {NEEDS_ALLPOOL!r}")


def compositions(n, k):
    for c in itertools.combinations(range(n + k - 1), k - 1):
        prev, out = -1, []
        for x in c:
            out.append(x - prev - 1)
            prev = x
        out.append(n + k - 2 - prev)
        yield out


# pattern index i: bit 4 = KH, bit 2 = arm, bit 1 = BASELINE (G's own convention, s89:298)
GOOD = dict(transition=1200, end=dict(margin=1.0, eta_key_by_stream=1.0), stats=[dict(step=2400, spread=0.5)])
BAD = dict(transition=None, end=dict(margin=0.0, eta_key_by_stream=0.0), stats=[dict(step=2400, spread=0.5)])


def records_of(cnt):
    kh, ar, ba = [], [], []
    for i, c in enumerate(cnt):
        kh += [bool(i & 4)] * c
        ar += [bool(i & 2)] * c
        ba += [bool(i & 1)] * c
    mk = lambda v: {s: (GOOD if o else BAD) for s, o in enumerate(v)}
    return mk(kh), mk(ar), mk(ba)


def independent_names(cnt, n, task, role):
    """the docstring's rules (s89:64-82), written here independently."""
    kh_only = sum(cnt[i] for i in range(8) if i & 4 and not i & 2)
    arm_only = sum(cnt[i] for i in range(8) if i & 2 and not i & 4)
    nb = sum(cnt[i] for i in range(8) if i & 1)
    khb, bkh = sum(cnt[i] for i in range(8) if i & 4 and not i & 1), sum(cnt[i] for i in range(8) if i & 1 and not i & 4)
    ab, ba = sum(cnt[i] for i in range(8) if i & 2 and not i & 1), sum(cnt[i] for i in range(8) if i & 1 and not i & 2)
    if nb >= n - 4:
        return "does not need"
    if not MC(khb, bkh) < 0.05:
        return "fails here"
    disc, needs = MC(ab, ba) < 0.05, MC(kh_only, arm_only) <= P4
    nm = NAMES[role]
    if needs:
        return NEEDS_ALLPOOL if (role == "FQ" and task == "allpool") else nm[2]
    hits = [nm[0]] * (disc and kh_only - arm_only <= 1) + [nm[1]] * (disc and kh_only - arm_only >= 2 and not needs) \
        + [nm[3]] * (not disc and not needs)
    assert len(hits) == 1, (cnt, hits)
    return hits[0]


CLASSIFY = {}
for n in (10, 20):
    seeds = list(range(n))
    tally = Counter()
    bad = 0
    names_vec = []
    for cnt in compositions(n, 8):
        KH, AR, BA = records_of(cnt)
        rd = readings(dict(B=BA, KH=KH, FQ=AR, RP=AR), seeds, "pool")
        rda = readings(dict(B=BA, KH=KH, FQ=AR), seeds, "allpool")
        first = rd["first"]
        if first.startswith("the task does not need the hinge"):
            got = got_rp = got_a = "does not need"
        elif first == "the labelled hinge itself fails here":
            got = got_rp = got_a = "fails here"
        else:
            assert first == "the labelled hinge is discovered; per-arm names below", first
            pick = lambda s, cands: [c for c in cands if s.startswith(c.upper() + " (")]
            g1 = pick(rd["REG3_FREQHINGE_D"], NAMES["FQ"])
            g2 = pick(rd["REG3_REPEATHINGE_D"], NAMES["RP"])
            g3 = pick(rda["REG3_FREQHINGE_D"], list(NAMES["FQ"]) + [NEEDS_ALLPOOL])
            assert len(g1) == len(g2) == len(g3) == 1, (cnt, rd, rda)
            got, got_rp, got_a = g1[0], g2[0], g3[0]
        want = (independent_names(cnt, n, "pool", "FQ"), independent_names(cnt, n, "pool", "RP"),
                independent_names(cnt, n, "allpool", "FQ"))
        bad += (got, got_rp, got_a) != want
        tally[got] += 1
        names_vec.append(got)
    CLASSIFY[n] = names_vec
    say(f"  n = {n}: {len(names_vec)} count vectors; readings() vs the docstring's rules written independently: {bad} "
        f"disagreements (FQ on pool, RP on pool, FQ on allpool); every vector got exactly one name (asserted)")
    say(f"    FQ names: " + "; ".join(f"{k} {v}" for k, v in sorted(tally.items(), key=lambda x: -x[1])))
# edge cases through the parent's own code
s20 = list(range(20))
ok_rec = _fake([1] * 20)
e1 = readings(dict(B=_fake([0] * 20), KH=ok_rec), s20, "pool", valid=False)
e2 = readings(dict(B=_fake([0] * 20), KH={s: ok_rec[s] for s in range(19)}), s20, "pool")
e3 = readings(dict(B=_fake([0] * 20), KH=ok_rec, R={s: ok_rec[s] for s in range(19)}), s20, "pool")
say(f"  invalid -> {e1['first']!r}; a missing KH seed -> {e2['first'][:60]!r}; a missing REG3_D (descriptive) seed -> "
    f"{e3['first'][:60]!r}")
# clamps: the commitment conjunct
lowc = {s: dict(GOOD if s < 2 else BAD, stats=[dict(step=2400, spread=0.1)]) for s in range(20)}
cl, why = P_NS["clamps"](lowc, _fake([1] * 1 + [0] * 19))
say(f"  clamps on an arm 2/20 with commitment 0.1 vs BASELINE 1/20 at 0.5: {cl} ({why}) -- an accompaniment, printed with "
    f"any of the four names")

# ══ §4 the parent's CHECK on S87's records ══════════════════════════════════════════════════════════════════════════
say("\n§4 S87's RECORDS RE-LABELLED (explore_g_s89.check_readings' inputs, from explore_out/G/s87_results.json at db14e9b9)")
st87 = json.loads(show(G2, "explore_out/G/s87_results.json"))
sha = st87["meta"]["sha"][:12]


def recorded(arm, seeds):
    out = {}
    for s in seeds:
        hits = [v for k, v in st87["runs"].items() if k.startswith(f"{arm}|{s}|{sha}|") and v.get("ok")]
        assert len(hits) <= 1
        if hits:
            out[s] = hits[0]
    return out


S87_DELTA = dict(B=("BASELINE_D", tuple(range(2720, 2730))), R=("REG3_D", tuple(range(2720, 2730))),
                 KH=("REG3_KEYHINGE_D", tuple(range(2720, 2730))), FQ=("REG3_FREQHINGE_D", tuple(range(2720, 2730))))
S87_HEBB = dict(B=("BASELINE", tuple(range(2700, 2720))), KH=("REG3_KEYHINGE", tuple(range(2700, 2720))),
                FQ=("REG3_FREQHINGE", tuple(range(2700, 2720))))
say(f"  store meta sha {sha}, cpu {st87['meta']['cpu']}; last commit touching it: "
    f"{git('log', '-1', '--format=%h %ad', '--date=short', G2, '--', 'explore_out/G/s87_results.json').strip()}")
for tag, spec in (("Hebbian 2700-2719", S87_HEBB), ("delta 2720-2729", S87_DELTA)):
    R = {k: recorded(a, s) for k, (a, s) in spec.items()}
    seeds = next(iter(spec.values()))[1]
    rd = readings(R, seeds, "lite", label={k: a for k, (a, _) in spec.items()})
    say(f"  {tag}: " + " | ".join(f"{k}: {v}" for k, v in rd.items() if k != "validity"))
    pat = Counter((hp.routed_bound(R["KH"][s]), hp.routed_bound(R["FQ"][s]), hp.routed_bound(R["B"][s])) for s in seeds)
    say(f"    per-seed (KH, FQ, B) ROUTED-BOUND patterns {dict(pat)}; FQ's failures are KH successes (nested): "
        f"{all(hp.routed_bound(R['KH'][s]) for s in seeds if not hp.routed_bound(R['FQ'][s]))}")
for arm, seed in (("REG3_FREQHINGE_D", 2720), ("REG3_KEYHINGE_D", 2720), ("BASELINE_D", 2720), ("REG3_D", 2720),
                  ("REG3_FREQHINGE", 2700), ("REG3_KEYHINGE", 2700), ("BASELINE", 2700)):
    pre = f"{arm}|{seed}|{sha}|"
    hits = [k for k, v in st87["runs"].items() if k.startswith(pre) and v.get("ok")]
    r = st87["runs"][hits[0]]
    say(f"  checks_repro record {arm}|{seed}: {len(hits)} hit(s) under the prefix (child:504-507 asserts 1); recorded on "
        f"{hits[0][len(pre):]}; curve through 2400 {[c[:2] for c in r['curve'] if c[0] <= 2400]}; transition {r['transition']}")

# ══ §5 operating characteristics at the new seed counts ═════════════════════════════════════════════════════════════
say("\n§5 OPERATING CHARACTERISTICS at n = 20 (exact multinomial over the 8 per-seed patterns, names from §3's run of the "
    "parent's readings(); 'indep.' = B, KH, A independent; 'nested' = KH and A comonotone (one uniform per seed), B independent)")
C20 = np.array(list(compositions(20, 8)), dtype=np.int64)
N20 = np.array(CLASSIFY[20])
LOGF = np.array([math.lgamma(i + 1) for i in range(64)])


def oc(pB, pK, pA, nested, n=20):
    probs = []
    for i in range(8):
        K, A, Bv = bool(i & 4), bool(i & 2), bool(i & 1)
        pb = pB if Bv else 1 - pB
        if nested:
            lo, hi = min(pK, pA), max(pK, pA)
            pka = {(1, 1): lo, (0, 0): 1 - hi, (1, 0): max(pK - pA, 0.0), (0, 1): max(pA - pK, 0.0)}[(K, A)]
        else:
            pka = (pK if K else 1 - pK) * (pA if A else 1 - pA)
        probs.append(pb * pka)
    lp = np.array([math.log(p) if p > 0 else 0.0 for p in probs])
    zero = np.array([p <= 0 for p in probs])
    logm = LOGF[n] - LOGF[C20].sum(1) + (C20 * lp).sum(1)
    w = np.where((C20[:, zero] > 0).any(1), 0.0, np.exp(logm))
    return {k: float(w[N20 == k].sum()) for k in set(N20)}, float(w.sum())


short = {"does not need": "noneed", "fails here": "fails", NAMES["FQ"][0]: "SURV", NAMES["FQ"][1]: "weaker",
         NAMES["FQ"][2]: "needs", NAMES["FQ"][3]: "nothing"}
say(f"  {'pB':>4} {'pK':>5} {'pA':>5} {'coupling':>7} | " + " ".join(f"{v:>7}" for v in short.values()))
for pK in (1.0, 0.95, 0.9):
    for pB in (0.0, 0.1, 0.3):
        for pA in (1.0, 0.95, 0.9, 0.8, 0.7, 0.5, 0.2, 0.0):
            for nested in (False, True):
                if pK == 1.0 and nested:
                    continue
                r, tot = oc(pB, pK, pA, nested)
                say(f"  {pB:>4} {pK:>5} {pA:>5} {'nested' if nested else 'indep.':>7} | "
                    + " ".join(f"{r.get(k, 0.0):7.3f}" for k in short) + f"   (sum {tot:.6f})")
say(f"  validity gate (ORACLE_D 2/2 on 2840-2841): P(valid) = p^2: p 1.0 -> 1.000, 0.95 -> {0.95 ** 2:.3f}, 0.9 -> {0.81:.3f}, "
    f"0.8 -> {0.64:.3f}")
say(f"  'task does not need the hinge' needs BASELINE_D >= 16/20: P at pB 0.3 {sum(math.comb(20, k) * 0.3 ** k * 0.7 ** (20 - k) for k in range(16, 21)):.2e}, "
    f"pB 0.6 {sum(math.comb(20, k) * 0.6 ** k * 0.4 ** (20 - k) for k in range(16, 21)):.3f}, pB 0.8 "
    f"{sum(math.comb(20, k) * 0.8 ** k * 0.2 ** (20 - k) for k in range(16, 21)):.3f}")

# ══ §6 the generators: cuts, hinge positions, coincidence with the key term, chance and perfect-gate eta^2 ═════════
say("\n§6 HINGES ON A SAMPLE (numpy replica of RandHeaderTask(P=4, nb=2, lmin=1, lmax=3).make_batch and F's _relabel; "
    "4096 sequences; distribution as F's, draws not torch's)")
S, P, NV, NB = 2, 4, 16, 2
COMPS = [c for c in itertools.product(range(1, 4), repeat=NB) if sum(c) == P]
K_POOL, N_CTX = 32, 16


def lite_batch(B, rng):
    vals = rng.random((B, P, NV)).argsort(-1)[..., :S]
    comp = rng.integers(len(COMPS), size=(B, S))
    korder = rng.random((B, S, P)).argsort(-1)
    border = rng.random((B, S * NB)).argsort(-1)
    q = rng.random((B, S * P)).argsort(-1)[:, :1]
    rows, roles, strm = [], [], []
    for b in range(B):
        seq, rl, sl = [], [], []
        for blk in border[b]:
            s, i = divmod(int(blk), NB)
            ln = COMPS[comp[b, s]]
            st = sum(ln[:i])
            seq.append(s); rl.append("CTX"); sl.append(s)
            for kk in korder[b, s, st:st + ln[i]]:
                seq += [S + int(kk), S + P + int(vals[b, kk, s])]; rl += ["KEY", "VAL"]; sl += [s, s]
        qs, qk = int(q[b, 0]) // P, int(q[b, 0]) % P
        seq += [qs, S + qk, S + P + int(vals[b, qk, qs])]; rl += ["CTX", "KEY", "VAL"]; sl += [qs, qs, qs]
        rows.append(seq); roles.append(rl); strm.append(sl)
    return np.array(rows), np.array(roles), np.array(strm)


def relabel(x, strm, rng, allpool):
    B = x.shape[0]
    keyset = rng.random((B, K_POOL)).argsort(-1)[:, :P]
    n_ctx = N_CTX if allpool else S
    key0, val0 = n_ctx, n_ctx + K_POOL
    out = x.copy()
    isk, isv, isc = (x >= S) & (x < S + P), x >= S + P, x < S
    out[isk] = key0 + np.take_along_axis(keyset, np.clip(x - S, 0, P - 1), 1)[isk]
    out[isv] = val0 + (x - S - P)[isv]
    lab = strm.copy()
    if allpool:
        pair = rng.random((B, N_CTX)).argsort(-1)[:, :S]
        first = x[:, 0]
        ctxset = np.empty_like(pair)
        ctxset[np.arange(B), first] = pair[:, 0]
        ctxset[np.arange(B), 1 - first] = pair[:, 1]
        out[isc] = np.take_along_axis(ctxset, np.clip(x, 0, S - 1), 1)[isc]
        lab = (strm != first[:, None]).astype(int)          # stream index by first appearance
    return out, lab, dict(n_ctx=n_ctx, key0=key0, val0=val0)


def largest_ratio_cut(tok):
    cnt = np.bincount(tok.reshape(-1)).astype(float)
    fr = cnt / cnt.sum()
    order = [int(t) for t in np.argsort(-fr, kind="stable") if fr[t] > 0]
    vals = [fr[t] for t in order]
    cut = max(range(len(vals) - 1), key=lambda i: vals[i] / vals[i + 1])
    return order[:cut + 1], vals


def repeat_mask(tok):
    V = int(tok.max()) + 1
    cnt = np.zeros((tok.shape[0], V), dtype=int)
    for b in range(tok.shape[0]):
        np.add.at(cnt[b], tok[b], 1)
    return np.take_along_axis(cnt, tok, 1) > 1


def eta2_pool(x, lab):                 # explore_b18_gates.eta2_pool (x (N, k), lab (N,)), eps 0
    mu = x.mean(0)
    tot = ((x - mu) ** 2).sum()
    between = 0.0
    for g in np.unique(lab):
        m = lab == g
        between += m.sum() * ((x[m].mean(0) - mu) ** 2).sum()
    return between / tot if tot > 0 else 0.0


rng = np.random.default_rng(87_000)
lite, roles, strm = lite_batch(4096, rng)
inp_roles, inp_lab = roles[:, :-1], strm[:, :-1]
body_len = S * (NB + 2 * P)
keyterm = np.zeros_like(inp_roles, dtype=bool)
keyterm[:, :body_len] = inp_roles[:, :body_len] == "KEY"
tasks = {"lite": (lite, strm, dict(n_ctx=S, key0=S, val0=S + P))}
xp, lp, vp = relabel(lite, strm, np.random.default_rng(1), False)
tasks["pool"] = (xp, lp, vp)
xa, la, va = relabel(lite, strm, np.random.default_rng(2), True)
tasks["allpool"] = (xa, la, va)
rep_lite = repeat_mask(lite[:, :-1])
masks = {}
for name, (x, lab, voc) in tasks.items():
    xin = x[:, :-1]
    sel, vals = largest_ratio_cut(xin)
    selroles = Counter("CTX" if t < voc["n_ctx"] else "KEY" if t < voc["val0"] else "VAL" for t in sel)
    fm, rm = np.isin(xin, sel), repeat_mask(xin)
    km = keyterm
    masks[name] = dict(key=km, freq=fm, repeat=rm)
    say(f"  {name}: largest-ratio cut {len(sel)} types {dict(selroles)} (ratio {vals[len(sel) - 1] / vals[len(sel)]:.2f})")
    for hname, m in (("key term", km), ("frequency hinge", fm), ("repeat hinge", rm)):
        per = {r: (m & (inp_roles == r)).sum(1).mean() for r in ("CTX", "KEY", "VAL")}
        both = (m & keyterm).sum(1)
        say(f"    {hname}: per sequence {m.sum(1).mean():.2f} positions (CTX {per['CTX']:.2f}, KEY {per['KEY']:.2f}, VAL "
            f"{per['VAL']:.2f}); of the key term's 8: {both.mean():.2f} (min {both.min()}, max {both.max()}); exactly the key "
            f"term's 8 in {int((m == keyterm).all(1).sum())} of 4096 sequences")
    if name != "lite":
        say(f"    repeat mask identical to lite's at every position: {bool((rm == rep_lite).all())}")

say("  eta^2 the hinge computes (explore_b18_gates.eta2_pool over a 32-sequence training batch, grouped by the hinge's ids; "
    "mean over 128 batches) for (a) a perfectly routed gate (one-hot of the stream index at every position) and (b) a "
    "label-free random gate (iid uniform channel-0 probability); TAU 0.2, so the hinge's value is relu(eta^2 - 0.2):")
rg = np.random.default_rng(5)
for name, (x, lab, voc) in tasks.items():
    xin, lin = x[:, :-1], lab[:, :-1]
    row = []
    for hname in ("key", "freq", "repeat"):
        m = masks[name][hname]
        perf, rnd, ngr = [], [], []
        for bi in range(128):
            sl = slice(32 * bi, 32 * bi + 32)
            mm = m[sl]
            if hname == "key":
                ids = (xin[sl] - voc["key0"])[mm]
            else:
                ids = xin[sl][mm]
            oh = np.eye(2)[lin[sl][mm]]
            p = rg.random(int(mm.sum()))
            perf.append(eta2_pool(oh, ids))
            rnd.append(eta2_pool(np.stack([p, 1 - p], 1), ids))
            ngr.append(len(np.unique(ids)))
        pe, re_ = float(np.mean(perf)), float(np.mean(rnd))
        row.append(f"{hname}: perfect gate {pe:.3f} (hinge {max(pe - 0.2, 0):.3f}), random gate {re_:.3f}, groups "
                   f"{np.mean(ngr):.1f} over {int(m[:32].sum())} positions")
    say(f"    {name}: " + "; ".join(row))

# ══ §7 ALLPOOL: the within-sequence outcome, swap invariance; the outcome grid ═══════════════════════════════════════
say("\n§7 THE ALLPOOL OUTCOME (numpy replicas of F's eta2_rows / inv_from_gate and pool_routing_stats' margin; F's "
    "pool_outcome exec'd from b0c6332; G's inv_view and hp.routed_bound exec'd from db14e9b9)")
x, lab, voc = tasks["allpool"]
xin, lin = x[:, :-1], lab[:, :-1]
B = 2048
xin, lin, rl = xin[:B], lin[:B], inp_roles[:B]
kpos = np.array([np.where(r[:body_len] == "KEY")[0] for r in rl])
vpos = kpos + 1


def eta2_rows(v, g):
    mu = v.mean(1)
    tot = ((v - mu[:, None]) ** 2).sum(1)
    between = np.zeros_like(tot)
    for c in np.unique(g):
        m = (g == c).astype(float)
        cnt = m.sum(1)
        mg = (v * m).sum(1) / np.maximum(cnt, 1)
        between += np.where(cnt > 0, cnt * (mg - mu) ** 2, 0.0)
    return np.where(tot > 0, between / np.where(tot > 0, tot, 1.0), 0.0)


def stats_of(gr, gw):
    pk = np.take_along_axis(gr[..., 0], kpos, 1)
    lk = np.take_along_axis(lin, kpos, 1)
    eta_inv = eta2_rows(pk, lk)
    gv = np.take_along_axis(gr, vpos[..., None].repeat(2, -1), 1)
    lv = np.take_along_axis(lin, vpos, 1)
    means = [(gv * (lv == s)[..., None]).sum(1) / np.maximum((lv == s).sum(1), 1)[:, None] for s in range(2)]
    cos = (means[0] * means[1]).sum(-1) / (np.linalg.norm(means[0], axis=-1) * np.linalg.norm(means[1], axis=-1))
    # pooled eta^2 (test_router_layout.eta2 over all sequences' key positions) and the margin (pool_routing_stats)
    v, g = pk.reshape(-1), lk.reshape(-1)
    mu = v.mean()
    pooled = sum((g == c).sum() * (v[g == c].mean() - mu) ** 2 for c in (0, 1)) / ((v - mu) ** 2).sum()
    q = 21
    qs = lin[:, q]
    keys = np.take_along_axis(xin, kpos, 1)
    same = keys == xin[:, q][:, None]
    tgt, dis = same & (lk == qs[:, None]), same & (lk != qs[:, None])
    gwv = np.take_along_axis(gw, vpos[..., None].repeat(2, -1), 1)
    dots = (gwv * gr[:, q][:, None, :]).sum(-1)
    marg = ((dots * tgt).sum(1) - (dots * dis).sum(1)).mean()
    return dict(eta_inv=float(eta_inv.mean()), cos_inv=float(np.nanmean(cos)), pooled=float(pooled), margin=float(marg))


perfect = np.eye(2)[lin]
flip = np.random.default_rng(3).random(B) < 0.5
swapped = np.where(flip[:, None, None], perfect[..., ::-1], perfect)
ctxpos = rl == "CTX"
lo = np.where(ctxpos, xin, 10 ** 6).min(1)
latched = np.where(ctxpos, xin, -1)
cur = np.maximum.accumulate(np.where(ctxpos, np.arange(22)[None, :], -1), 1)
curtok = np.take_along_axis(xin, cur, 1)
typeord = np.eye(2)[(curtok != lo[:, None]).astype(int)]
balanced = np.eye(2)[(curtok >= 8).astype(int)]
soft = np.random.default_rng(4).random((B, 22))
softg = np.stack([soft, 1 - soft], -1)
softsw = np.where(flip[:, None, None], softg[..., ::-1], softg)
sep = float(np.mean([len(set((xin[b][ctxpos[b]] >= 8).tolist())) == 2 for b in range(B)]))
for nm, g in (("relative perfect gate", perfect), ("perfect, channels swapped in a random half of the sequences", swapped),
              ("type-ordered (lower CTX type -> channel 0)", typeord), ("balanced type-level map (types 0-7 -> ch 0)", balanced),
              ("random soft gate", softg), ("the same random gate, channels swapped per sequence", softsw)):
    s_ = stats_of(g, g)
    say(f"  {nm}: within-sequence eta^2 {s_['eta_inv']:.4f}, VAL cos {s_['cos_inv']:.4f}; pooled eta^2 {s_['pooled']:.4f}; "
        f"margin {s_['margin']:.4f}")
say(f"  the balanced map separates {sep:.4f} of these {B} sequences (F's cap 64/120 = {64 / 120:.4f})")
# the outcome grid: G's routed_bound(inv_view(rec)) vs F's pool_outcome(rec, True)
fsrc = blobs[F_NEW].decode()
fdefs = defs(ast.parse(fsrc))
fake_trr = types.ModuleType("test_router_reliability")
fake_trr.DISC_COS = 0.5
sys.modules["test_router_reliability"] = fake_trr
F_NS = {"ROUTED_MARGIN": 0.9, "ROUTED_ETA": 0.9}
exec(ast.get_source_segment(fsrc, fdefs["pool_outcome"]), F_NS)
pool_outcome = F_NS["pool_outcome"]
agree = n_ = 0
errs = Counter()
for tr, mg, e, ei in itertools.product((None, 1200), (0.5, 0.8999999, 0.9, 1.0), (0.0, 0.9, 0.9000001, 0.95),
                                       (0.0, 0.9, 0.9000001, 0.95, None)):
    rec = dict(transition=tr, val_cos=0.1, end=dict(margin=mg, eta_key_by_stream=e, eta_key_by_stream_inv=ei, val_cos_inv=0.1))
    n_ += 1
    try:
        g_ = (hp.routed_bound(rec), hp.routed_bound(inv_view(rec)))
    except TypeError as ex:
        errs[f"TypeError ({ex})"] += 1
        continue
    f_ = (pool_outcome(rec, False)["routed_bound"], pool_outcome(rec, True)["routed_bound"])
    agree += g_ == f_
say(f"  routed_bound(rec), routed_bound(inv_view(rec)) == F's pool_outcome(rec, False / True): {agree}/{n_} grid records agree "
    f"(boundaries 0.9 included); raised: {dict(errs) or 'none'}")
rec = dict(transition=1200, val_cos=0.9, end=dict(margin=1.0, eta_key_by_stream=0.0, eta_key_by_stream_inv=1.0, val_cos_inv=0.0))
say(f"  DISCOVERED on an allpool record (bound, pooled val_cos 0.9, val_cos_inv 0.0): hp.discovered-style "
    f"(bound and r['val_cos'] < 0.5) on inv_view(rec) {bool(inv_view(rec)['transition'] is not None and inv_view(rec)['val_cos'] < 0.5)}"
    f"; F's pool_outcome(rec, True)['discovered'] {pool_outcome(rec, True)['discovered']} -> inv_view keeps the pooled val_cos")

# ══ §8 seeds ════════════════════════════════════════════════════════════════════════════════════════════════════════
say("\n§8 SEEDS 2840-2899 (G's block; S89 uses 2840-2861)")
LO, HI = 2840, 2899
branches = [b.strip() for b in git("branch", "-r").splitlines() if "->" not in b]
pat_key = re.compile(rb"\|(28[4-9]\d)\|")
pat_seed = re.compile(rb'"seed"\s*:\s*(28[4-9]\d)\b')
pat_list = re.compile(rb'"seeds"\s*:\s*\[([^\]]*)\]')
pat_log = re.compile(rb"seed[ =:]+(28[4-9]\d)\b")


def seeds_in(raw):
    hits = set(int(m) for m in pat_key.findall(raw)) | set(int(m) for m in pat_seed.findall(raw))
    for m in pat_list.findall(raw):
        hits |= {int(t) for t in re.findall(rb"\b(\d{3,5})\b", m) if LO <= int(t) <= HI}
    return hits


for br in branches:
    tip = git("rev-parse", "--short", br).strip()
    files = [l for l in git("ls-tree", "-r", "--name-only", br).splitlines() if l.endswith((".json", ".jsonl", ".log"))]
    hits = {}
    for path in files:
        raw = show(br, path, binary=True)
        s = seeds_in(raw) | set(int(m) for m in pat_log.findall(raw))
        if s:
            hits[path] = sorted(s)
    say(f"  {br} @ {tip}: {len(files)} json/jsonl/log files; seeds in 2840-2899: {hits or 'none'}")
objs = git("rev-list", "--objects", "--remotes").splitlines()
paths = defaultdict(set)
for l in objs:
    parts = l.split(" ", 1)
    if len(parts) == 2 and parts[1].endswith((".json", ".jsonl", ".log", ".txt")):
        paths[parts[0]].add(parts[1])
tf = tempfile.TemporaryFile()
tf.write("".join(f"{h}\n" for h in paths).encode())
tf.seek(0)
proc = subprocess.Popen(["git", "cat-file", "--batch"], stdin=tf, stdout=subprocess.PIPE)
hist, nbytes = defaultdict(set), 0
for _ in paths:
    head = proc.stdout.readline().split()
    size = int(head[2])
    raw = proc.stdout.read(size)
    proc.stdout.read(1)
    nbytes += size
    s = seeds_in(raw) | set(int(m) for m in pat_log.findall(raw))
    if s:
        for p in paths[head[0].decode()]:
            hist[p] |= s
proc.wait()
say(f"  history: {len(paths)} distinct json/jsonl/log/txt blobs ({nbytes / 1e9:.2f} GB) reachable from {len(branches)} remote "
    f"branches; seeds 2840-2899 found in: {dict((k, sorted(v)) for k, v in hist.items()) or 'none'}")

# ══ §9 the records the pool path could be tied to ═══════════════════════════════════════════════════════════════════
say("\n§9 RECORDS ON THE ROUND-6 TASKS (explore-F) that a CHECK could name")
for f in ("s89_tasks_results.json", "s89_diag_results.json", "s89_post_results.json"):
    st = json.loads(show("origin/claude/explore-F", f"explore_out/F/{f}"))
    arms = Counter(k.split("|")[0] for k in st["runs"])
    say(f"  explore_out/F/{f} (first commit {git('log', '--format=%h', '--diff-filter=A', 'origin/claude/explore-F', '--', f'explore_out/F/{f}').strip()}): "
        f"{dict(arms)}")
    for k, v in sorted(st["runs"].items()):
        if "DELTA" in k:
            say(f"    {k.split('|')[0]}|{k.split('|')[1]}: transition {v.get('transition')}, acc {v.get('acc'):.3f}")
fc = show("origin/claude/explore-F", "explore_f_s89_tasks_child.py")
say(f"  F's delta perfect gate converts with explore_f_delta_fast.to_delta_fast (F child:{lines_of(fc, 'ff.to_delta_fast(')}); "
    f"G's ORACLE_D with explore_delta_mem.to_delta (G child:{lines_of(ch, 'dm.to_delta(')}); F's fast form is CHECKed equal "
    f"to the parallel form to 1e-5 relative in float32, not bit for bit (explore_f_delta_fast docstring)")

with open(__file__.replace(".py", "_out.txt"), "w") as fh:
    fh.write("\n".join(OUT) + "\n")
