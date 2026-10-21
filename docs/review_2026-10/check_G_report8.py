"""check_G_report8.py — Referee V's check of Session G's report 8 (S89) at claude/explore-G 8e9e3187.

Reads git objects only (git show / rev-parse / log); writes nothing; prints only. torch is not needed: G's pure parent
functions (explore_g_s89 at db14e9b9, explore_g_rhl_parent, explore_g_common's statistics, test_conv_lr.mcnemar_greater,
test_router_confirm.fisher_greater, explore_common's fail_class constants) are exec'd from blobs with stub modules, and
G's own report() is run on the committed records with its file write redirected to memory.
"""
import ast
import collections
import io
import json
import math
import os
import statistics
import subprocess
import sys
import types
import contextlib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib  # noqa: E402

TIP, FIRST = "8e9e3187", "db14e9b9"
CPU = "Intel(R) Xeon(R) Processor @ 2.10GHz"


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a, stderr=subprocess.DEVNULL).decode()


def show(ref, path):
    return git("show", f"{ref}:{path}")


def defs(src, names):
    """the top-level statements of src that define any of names (functions, classes, assignments)."""
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            keep.append(node)
        elif isinstance(node, ast.Assign):
            tg = [t.id for t in node.targets if isinstance(t, ast.Name)]
            tg += [e.id for t in node.targets if isinstance(t, ast.Tuple) for e in t.elts if isinstance(e, ast.Name)]
            if set(tg) & set(names):
                keep.append(node)
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


def H(t):
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)


# ── stores ───────────────────────────────────────────────────────────────────
STORES = {"s89": json.loads(show(TIP, "explore_out/G/s89_results.json")),
          "s87": json.loads(show(TIP, "explore_out/G/s87_results.json"))}

# ── stub modules: explore_common (ec), explore_g_common (gc) ────────────────
ec = types.ModuleType("explore_common")
# explore_common imports fail_class from test_router_layout (MARGIN_PARTIAL 0.25, ETA_SPLIT 0.5 at :198-199)
exec(defs(show(TIP, "test_router_layout.py"), {"fail_class", "MARGIN_PARTIAL", "ETA_SPLIT"}), ec.__dict__)
ec.cpu_model = lambda: CPU
ec.OUT_DIR = "/nonexistent/explore_out"

gc = types.ModuleType("explore_g_common")
gsrc = show(TIP, "explore_g_common.py")
gc.math = math
exec(defs(gsrc, {"wilson", "band", "cnt", "key", "runs"}), gc.__dict__)
_m = {}
exec(defs(show(TIP, "test_conv_lr.py"), {"mcnemar_greater"}), {"math": math}, _m)
exec(defs(show(TIP, "test_router_confirm.py"), {"fisher_greater"}), {"math": math}, _m)
gc.mcnemar_greater = _m["mcnemar_greater"]
gc.fisher_greater = _m["fisher_greater"]
gc.G_DIR = "/nonexistent/G"
gc.load = lambda name: STORES[name]
gc.ec = ec
sys.modules["explore_common"] = ec
sys.modules["explore_g_common"] = gc

hp = types.ModuleType("explore_g_rhl_parent")
hp.__dict__["__file__"] = "/nonexistent/explore_g_rhl_parent.py"
hp_src = show(TIP, "explore_g_rhl_parent.py")
exec(compile(hp_src, "explore_g_rhl_parent.py", "exec"), hp.__dict__)
sys.modules["explore_g_rhl_parent"] = hp

VALIDITY = json.loads(show(TIP, "explore_out/G/s89_validity.json"))


class _Sink(io.StringIO):
    def close(self):
        pass


SINK = {}


def fake_open(p, mode="r", *a, **k):
    if "w" in mode:
        SINK[p] = _Sink()
        return SINK[p]
    raise RuntimeError(f"no file reads from disk: {p}")


def load_parent(ref):
    """explore_g_s89 at ref, exec'd statement by statement; _load reads the validity file from git; open() for writing
    goes to memory."""
    src = show(ref, "explore_g_s89.py")
    m = types.ModuleType("explore_g_s89")
    m.__dict__.update(__file__="/nonexistent/explore_g_s89.py", __name__="explore_g_s89", open=fake_open)
    sys.modules["explore_g_s89"] = m
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.If):      # if __name__ == "__main__"
            continue
        exec(compile(ast.Module(body=[node], type_ignores=[]), "explore_g_s89.py", "exec"), m.__dict__)
        if isinstance(node, ast.FunctionDef) and node.name == "_load":
            m._load = lambda p: VALIDITY if p.endswith("s89_validity.json") else None
    return m


P = load_parent(FIRST)

# ═══ 0. the commits, the code blobs, the record timeline ═════════════════════
H("0. commits db14e9b9..8e9e3187: code blobs and records per commit")
revs = git("rev-list", "--reverse", f"{FIRST}^..{TIP}").split()
CODE = ("explore_g_s89.py", "explore_g_s89_child.py", "explore_g_rhl_parent.py", "explore_g_common.py",
        "explore_f_tasks.py", "explore_common16.py", "explore_common.py")
blobs = collections.defaultdict(set)
first_learned = None
rows0 = []
for c in revs:
    t = git("log", "-1", "--format=%ci", c).strip()
    for f in CODE:
        try:
            blobs[f].add(git("rev-parse", f"{c}:{f}").strip()[:10])
        except subprocess.CalledProcessError:
            blobs[f].add(None)
    try:
        st = json.loads(show(c, "explore_out/G/s89_results.json"))
        ks = list(st["runs"])
    except subprocess.CalledProcessError:
        ks = []
    learned = [k for k in ks if "ORACLE" not in k]
    if learned and first_learned is None:
        first_learned = (c[:8], t, learned)
    rows0.append((c[:8], t, len(ks), len(learned)))
for r in rows0:
    print(f"  {r[0]} {r[1]}  records {r[2]:3d}  learned {r[3]:3d}")
print("  code blobs over the range (one value = unchanged):")
for f, b in blobs.items():
    print(f"    {f}: {sorted(x for x in b if x) or b}")
changed = git("diff", "--name-only", FIRST, TIP).split()
print(f"  files changed {FIRST}..{TIP}: {changed}")
print(f"  .py files changed: {[f for f in changed if f.endswith('.py')]}")
print(f"  first learned record(s) at {first_learned[0]} {first_learned[1]}: {first_learned[2]}")
v2 = git("log", "--format=%h %ci", "--", "docs/review_2026-10/screen_review_s89_G_v2.md").split("\n")[-2:]
print(f"  V's screen_review_s89_G_v2.md committed (review-V): {v2}")
before_v2 = [r for r in rows0 if r[1] < "2026-10-20 22:26:05"]
print(f"  records on G's branch at the last commit before 22:26:05: {before_v2[-1]}")

# ═══ 1. the records ═══════════════════════════════════════════════════════════
H("1. the records at 8e9e3187")
S = STORES["s89"]
runs = S["runs"]
print(f"  n records {len(runs)}; meta sha {S['meta']['sha'][:12]}; provenance {S['meta']['provenance']}")
print(f"  key SHAs {collections.Counter(k.split('|')[2] for k in runs)}; record sha {collections.Counter(r['sha'] for r in runs.values())}")
print(f"  CPUs {collections.Counter(r['cpu'] for r in runs.values())}")
print(f"  ok {sum(r['ok'] for r in runs.values())}; attempts {collections.Counter(r['attempt'] for r in runs.values())}")
print(f"  CPU-hours (sum secs) {sum(r['secs'] for r in runs.values()) / 3600:.2f}; wall-hours (sum secs_wall) "
      f"{sum(r['secs_wall'] for r in runs.values()) / 3600:.2f}")
print(f"  per arm: {dict(sorted(collections.Counter(k.split('|')[0] for k in runs).items()))}")
print(f"  validity file: {VALIDITY}")

# checks files
cf = [f for f in git("ls-tree", "--name-only", TIP, "explore_out/G/").split() if "checks_s89" in f]
for f in cf:
    d = json.loads(show(TIP, f))
    print(f"  {f}: rows {len(d['rows'])}, ok {sum(1 for _, o in d['rows'] if o)}, git {d['provenance'].get('git')}")
log = show(TIP, "explore_out/G/s89.log")
print(f"  log: '=== RESTART' lines {log.count('=== RESTART')}: "
      f"{[l for l in log.splitlines() if l.startswith('=== RESTART')]}")
print(f"  log: 'CHECK S89:' lines {log.count('CHECK S89:')}, ' FAIL' among them "
      f"{sum(1 for l in log.splitlines() if 'CHECK S89:' in l and l.endswith('FAIL'))}")
print(f"  log: code SHA banners {collections.Counter(l.split('code SHA ')[1].split()[0] for l in log.splitlines() if 'code SHA' in l)}")

# ═══ 2. G's report() on the records, against the log ════════════════════════
H("2. G's own report() (db14e9b9 blob) on the committed records, compared with s89.log")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    P.report()
out = buf.getvalue().rstrip("\n").splitlines()
lg = log.rstrip("\n").splitlines()
i0 = next(i for i, l in enumerate(lg) if l.startswith("OUTCOME FORM"))
tail = lg[i0:]
diff = [(i, a, b) for i, (a, b) in enumerate(zip(out, tail)) if a != b]
print(f"  report() lines {len(out)}, log lines from 'OUTCOME FORM' {len(tail)}, differing lines {len(diff)}")
for i, a, b in diff[:10]:
    print(f"    line {i}:\n      mine: {a[:300]}\n      log : {b[:300]}")
summ = json.loads(SINK[[k for k in SINK if k.endswith("s89_summary.json")][0]].getvalue())
summ_c = json.loads(show(TIP, "explore_out/G/s89_summary.json"))
print(f"  s89_summary.json from report() == committed: {summ == summ_c}")

# ═══ 3. counts: G's functions and rlib ════════════════════════════════════════
H("3. counts per task and arm (G's hp.routed_bound / inv_view; rlib cross-check)")
ARMS = ("ORACLE_D",) + P.LEARNED


def recs(task, arm):
    out = {}
    for k, r in runs.items():
        a, s = k.split("|")[:2]
        if a == f"{task}.{arm}":
            out[int(s)] = r
    return out


def rl_routed(r, inv):
    e = r["end"]
    eta = e.get("eta_key_by_stream_inv" if inv else "eta_key_by_stream")
    return rlib.bound(r) and e["margin"] >= 0.9 and eta > 0.9


def rl_disc(r, inv):
    vc = r["end"]["val_cos_inv"] if inv else r["val_cos"]
    return rlib.bound(r) and vc < 0.5


R = {}
for t in P.TASK_NAMES:
    for a in ARMS:
        rr = recs(t, a)
        R[(t, a)] = rr
        g_rb = sum(hp.routed_bound(P.inv_view(r) if t == "allpool" else r) for r in rr.values())
        g_rb_other = sum(hp.routed_bound(r if t == "allpool" else P.inv_view(r)) for r in rr.values())
        rl_rb = sum(rl_routed(r, t == "allpool") for r in rr.values())
        bd = sum(hp.bound(r) for r in rr.values())
        bd_rl = sum(rlib.bound(r) for r in rr.values())
        dp = sum(rl_disc(r, False) for r in rr.values())
        di = sum(rl_disc(r, True) for r in rr.values())
        tr = sorted(r["transition"] for r in rr.values() if r["transition"] is not None)
        trs = f"median {statistics.median(tr):.0f} ({tr[0]}-{tr[-1]})" if tr else "-"
        cls = collections.Counter(hp.outcome(P.inv_view(r) if t == "allpool" else r) for r in rr.values()
                                  if not hp.bound(r))
        lo, hi = rlib.wilson(g_rb, len(rr))
        print(f"  {t}.{a:<19} n {len(rr):2d}  ROUTED-BOUND G {g_rb:2d} [{lo:.2f}, {hi:.2f}] {rlib.band(g_rb, len(rr))}"
              f" | rlib {rl_rb:2d} | other form {g_rb_other:2d} | BOUND G {bd:2d} rlib {bd_rl:2d} | DISC pooled {dp} inv {di}"
              f" | trans {trs} | unbound {dict(cls)}")

# lite references
print("  lite (S87 delta 2720-2729), G's recorded() + routed_bound:",
      {k: sum(hp.routed_bound(r) for r in P.recorded("s87", a, s).values()) for k, (a, s) in P.S87_DELTA.items()})
s82 = json.loads(show(TIP, "explore_out/G/s82_devices_results.json"))
r82 = {k: v for k, v in s82["runs"].items() if k.split("|")[0] == "REG3"}
print(f"  S82 Hebbian lite REG3: n {len(r82)}, ROUTED-BOUND {sum(hp.routed_bound(r) for r in r82.values())}, "
      f"BOUND {sum(hp.bound(r) for r in r82.values())}")

# ═══ 4. pairs ════════════════════════════════════════════════════════════════
H("4. paired comparisons on pool (b = first only, c = second only; one-sided exact McNemar p(first >))")
rbv = hp.routed_bound
pool = {a: R[("pool", a)] for a in P.LEARNED}
names = dict(B="BASELINE_D", R="REG3_D", KH="REG3_KEYHINGE_D", FQ="REG3_FREQHINGE_D", RP="REG3_REPEATHINGE_D")
for x, y in (("KH", "B"), ("FQ", "B"), ("RP", "B"), ("KH", "FQ"), ("KH", "RP"), ("RP", "KH"), ("KH", "R"),
             ("RP", "R"), ("R", "FQ"), ("FQ", "R"), ("R", "B"), ("B", "R")):
    b, c, p, n = hp.paired(pool[names[x]], pool[names[y]], rbv)
    b2, c2, n2 = rlib.pairs(pool[names[x]], pool[names[y]], lambda r: rl_routed(r, False))
    print(f"  {x} vs {y}: {x} only {b}, {y} only {c}, p {p:.3g}, n {n}  (rlib {b2}/{c2}, two-sided p {rlib.mcnemar2(b, c):.3g})")
for t in P.TASK_NAMES:
    print(f"  {t} seeds where REG3_D is not ROUTED-BOUND: "
          f"{[s for s, r in sorted(R[(t, 'REG3_D')].items()) if not rbv(P.inv_view(r) if t == 'allpool' else r)]}")
fail_r = [s for s, r in sorted(pool["REG3_D"].items()) if not rbv(r)]
for s in fail_r:
    r = pool["REG3_D"][s]
    print(f"    pool REG3_D {s}: class {hp.outcome(r)}, acc {r['acc']:.3f}; KH trans {pool['REG3_KEYHINGE_D'][s]['transition']}, "
          f"RP trans {pool['REG3_REPEATHINGE_D'][s]['transition']}, FQ {'RB' if rbv(pool['REG3_FREQHINGE_D'][s]) else 'no'}")
same = sum(pool["REG3_KEYHINGE_D"][s]["transition"] == pool["REG3_REPEATHINGE_D"][s]["transition"] for s in P.S20)
print(f"  KH and RP: same transition on {same}/20 seeds; KH {[pool['REG3_KEYHINGE_D'][s]['transition'] for s in P.S20]}")
print(f"                                              RP {[pool['REG3_REPEATHINGE_D'][s]['transition'] for s in P.S20]}")
print(f"                                              R  {[pool['REG3_D'][s]['transition'] for s in P.S20]}")
eq_acc = sum(pool["REG3_KEYHINGE_D"][s]["acc"] == pool["REG3_REPEATHINGE_D"][s]["acc"] for s in P.S20)
print(f"  KH and RP identical final acc on {eq_acc}/20 seeds")
for a in ("REG3_FREQHINGE_D",):
    rb7 = [s for s, r in sorted(pool[a].items()) if rbv(r)]
    print(f"  FQ ROUTED-BOUND seeds {rb7}; of them REG3_D ROUTED-BOUND: {[s for s in rb7 if rbv(pool['REG3_D'][s])]}")

# ═══ 5. the repeat hinge's name: the committed rule, and inert-hinge cases ══
H("5. 'the repeat hinge replaces the key hinge': the committed rule (db14e9b9 arm_name) and counterfactual inputs")
print("  arm_name source (db14e9b9):")
print("    " + "\n    ".join(ast.get_source_segment(show(FIRST, "explore_g_s89.py"),
                                                   next(n for n in ast.parse(show(FIRST, "explore_g_s89.py")).body
                                                        if isinstance(n, ast.FunctionDef) and n.name == "arm_name")).splitlines()))
KH, B, Rr = pool["REG3_KEYHINGE_D"], pool["BASELINE_D"], pool["REG3_D"]
print(f"  actual RP: {P.arm_name(KH, pool['REG3_REPEATHINGE_D'], B, 'RP', 'pool')}")
print(f"  an inert hinge identical to REG3_D seed by seed (RP := REG3_D): {P.arm_name(KH, Rr, B, 'RP', 'pool')}")
print(f"  KH := REG3_D vs RP (would the register with no hinge 'replace' the key hinge?): "
      f"{P.arm_name(Rr, pool['REG3_REPEATHINGE_D'], B, 'RP', 'pool')}")
# an inert hinge as an independent draw at REG3_D's rate, KH at 20/20 nested
def p_replaces(pa, n=20):
    # KH succeeds on all n; arm succeeds on m ~ Bin(n, pa); KH only = n - m, arm only 0; B 0/20 (disc iff m >= 5)
    tot = 0.0
    for m in range(n + 1):
        pm = math.comb(n, m) * pa ** m * (1 - pa) ** (n - m)
        kh_only = n - m
        needs = gc.mcnemar_greater(kh_only, 0) <= P.P4
        disc = gc.mcnemar_greater(m, 0) < 0.05
        if (not needs) and disc and kh_only <= 1:
            tot += pm
    return tot
lo, hi = rlib.wilson(16, 20)
for pa in (0.58, 0.7, 0.8, 0.9, round(hi, 3), 0.95, 1.0):
    print(f"  P('replaces' | KH 20/20, BASELINE 0/20, arm iid at rate {pa}) = {p_replaces(pa):.3f}")
# V's fix 1 (2b): KH beats REG3_D?
kb, kc, kp, kn = hp.paired(KH, Rr, rbv)
print(f"  V's v2 fix 1 / reading 2b: KH beats REG3_D? KH only {kb}, REG3_D only {kc}, p {kp:.4g} -> "
      f"{'beats' if kp < 0.05 else 'does NOT beat: the register does not need the hinge here'}")
rb_, rc_, rp_, _ = hp.paired(pool["REG3_REPEATHINGE_D"], Rr, rbv)
print(f"  RP vs REG3_D: RP only {rb_}, REG3_D only {rc_}, p {rp_:.4g}; FQ vs REG3_D: "
      f"{hp.paired(pool['REG3_FREQHINGE_D'], Rr, rbv)[:3]}; REG3_D vs FQ: {hp.paired(Rr, pool['REG3_FREQHINGE_D'], rbv)[:3]}")
# is REG3_D 'discovered' against BASELINE by the same rule? and what would it be named?
print(f"  REG3_D vs BASELINE_D: {hp.paired(Rr, B, rbv)[:3]} (discovered = p < 0.05)")

# ═══ 6. ALLPOOL ═══════════════════════════════════════════════════════════════
H("6. ALLPOOL: per run (within-sequence fields; G's outcome on inv_view)")
for a in P.LEARNED:
    rr = R[("allpool", a)]
    print(f"  allpool.{a}:")
    srf = []
    for s, r in sorted(rr.items()):
        e = r["end"]
        o = hp.outcome(P.inv_view(r))
        srf.append(e["seq_routed_frac"])
        flag = " <- cap band" if 0.4 <= e["seq_routed_frac"] <= 0.6 else ""
        print(f"    {s} {o:<15} acc {r['acc']:.3f} trans {r['transition']} margin {e['margin']:.3f} "
              f"eta_inv {e['eta_key_by_stream_inv']:.3f} eta_pooled {e['eta_key_by_stream']:.3f} seq_routed {e['seq_routed_frac']:.3f} "
              f"val_cos_inv {e['val_cos_inv']:.3f} val_cos {r['val_cos']:.3f} eta_by_key {e['eta_key_by_key']:.3f} "
              f"probeL3 {r['probe']['L3|K']['acc']:.2f}/{r['probe']['L3|V']['acc']:.2f}{flag}")
cap = 64 / 120
allr = [(a, s, r) for a in P.LEARNED for s, r in R[("allpool", a)].items()]
print(f"  cap 64/120 = {cap:.4f}; max margin over learned allpool runs {max(r['end']['margin'] for _, _, r in allr):.3f}; "
      f"max seq_routed_frac {max(r['end']['seq_routed_frac'] for _, _, r in allr):.3f}; "
      f"max eta_inv {max(r['end']['eta_key_by_stream_inv'] for _, _, r in allr):.3f}")
capb = [(a, s, r) for a, s, r in allr if 0.4 <= r["end"]["seq_routed_frac"] <= 0.6]
print(f"  runs with seq_routed_frac in [0.4, 0.6]: {collections.Counter(a for a, _, _ in capb)}; range "
      f"{min(r['end']['seq_routed_frac'] for *_, r in capb):.3f}-{max(r['end']['seq_routed_frac'] for *_, r in capb):.3f}; "
      f"margin {min(r['end']['margin'] for *_, r in capb):.3f}-{max(r['end']['margin'] for *_, r in capb):.3f}; "
      f"acc {min(r['acc'] for *_, r in capb):.3f}-{max(r['acc'] for *_, r in capb):.3f}; "
      f"val_cos_inv {min(r['end']['val_cos_inv'] for *_, r in capb):.3f}-{max(r['end']['val_cos_inv'] for *_, r in capb):.3f}; "
      f"probe L3|K {min(r['probe']['L3|K']['acc'] for *_, r in capb):.2f}-{max(r['probe']['L3|K']['acc'] for *_, r in capb):.2f}; "
      f"bound {sum(hp.bound(r) for *_, r in capb)}")
ks = [r for a, s, r in allr if hp.outcome(P.inv_view(r)) == "KEY"]
print(f"  KEY-class runs {len(ks)}: eta_key_by_key {min(r['end']['eta_key_by_key'] for r in ks):.3f}-"
      f"{max(r['end']['eta_key_by_key'] for r in ks):.3f}")
bd = [(a, s, r) for a, s, r in allr if hp.bound(r)]
for a, s, r in bd:
    print(f"  bound allpool run {a} {s}: trans {r['transition']}, margin {r['end']['margin']:.3f}, eta_by_key "
          f"{r['end']['eta_key_by_key']:.3f}, seq_routed {r['end']['seq_routed_frac']:.3f}, val_cos_inv {r['end']['val_cos_inv']:.3f}"
          f" -> DISCOVERED inv form {r['end']['val_cos_inv'] < 0.5}")
# any learned run 'DISCOVERED' (within-sequence form) while not ROUTED?
print(f"  learned allpool runs DISCOVERED (inv form) and not ROUTED-BOUND: "
      f"{[(a, s) for a, s, r in allr if rl_disc(r, True) and not rbv(P.inv_view(r))]}")
print(f"  learned allpool runs with val_cos_inv < 0.5 (bound or not): "
      f"{[(a, s, round(r['end']['val_cos_inv'], 3)) for a, s, r in allr if r['end']['val_cos_inv'] < 0.5]}")
for t in P.TASK_NAMES:
    o = R[(t, "ORACLE_D")]
    for s, r in sorted(o.items()):
        e = r["end"]
        print(f"  {t}.ORACLE_D {s}: trans {r['transition']}, margin {e['margin']:.3f}, eta pooled {e['eta_key_by_stream']:.3f}, "
              f"inv {e['eta_key_by_stream_inv']:.3f}, seq_routed {e['seq_routed_frac']:.3f}, val_cos {r['val_cos']:.3f}, "
              f"val_cos_inv {e['val_cos_inv']:.3f}")
# the perfect gate used on allpool (child at db14e9b9)
ch = show(FIRST, "explore_g_s89_child.py").splitlines()
print("  child lines mentioning perfect_gate:")
for i, l in enumerate(ch, 1):
    if "perfect_gate" in l and "def " not in l[:8]:
        print(f"    child:{i}: {l.strip()[:150]}")

# ═══ 7. pool: CTX positions in the routed learned runs (item 5) ═══════════════
H("7. pool: what the routed learned runs do at CTX positions (end, per run; |g(s0,ch0) - g(s1,ch0)| at L1/L3)")
for a in ("REG3_D", "REG3_KEYHINGE_D", "REG3_REPEATHINGE_D", "REG3_FREQHINGE_D", "BASELINE_D"):
    rr = pool[a]
    sep = {}
    for l in ("L1", "L3"):
        sep[l] = [abs(r["end"]["layers"][l]["table"]["CTX"][0][0] - r["end"]["layers"][l]["table"]["CTX"][1][0])
                  for s, r in sorted(rr.items()) if rbv(r) or a in ("BASELINE_D", "REG3_FREQHINGE_D")]
    print(f"  pool.{a}: CTX separation L1 median {statistics.median(sep['L1']):.2f} (min {min(sep['L1']):.2f}, max "
          f"{max(sep['L1']):.2f}); L3 median {statistics.median(sep['L3']):.2f} (n {len(sep['L3'])})")
fq = pool["REG3_FREQHINGE_D"]
for s, r in sorted(fq.items()):
    v = r["term"]["value_at"]
    print(f"    FQ {s}: {'RB' if rbv(r) else hp.outcome(r):<15} acc {r['acc']:.3f} fired {r['term']['fired']} value@1 "
          f"{statistics.fmean(v['1']):.3f} spread@2400 {P.commit(r):.3f} REG3_D {'RB' if rbv(pool['REG3_D'][s]) else 'no'}")

# ═══ 8. readings with G's code, re-run directly ═══════════════════════════════
H("8. readings() (db14e9b9) directly, both tasks")
for t in P.TASK_NAMES:
    RR = {k: (R[(t, a)] if t == "pool" else {s: P.inv_view(r) for s, r in R[(t, a)].items()}) for k, a in P.ROLE.items()}
    rd = P.readings(RR, P.S20, t, valid=VALIDITY[t]["valid"], valid_note="")
    for k, v in rd.items():
        print(f"  {t}: {k}: {v}")

# ═══ 9. remaining numbers quoted in report 8 / the README ═════════════════════
H("9. remaining numbers: allpool classes and plateaus, pool BASELINE, KH / RP / FQ descriptives")
allc = collections.Counter()
for a, s, r in allr:
    allc[hp.outcome(P.inv_view(r))] += 1
print(f"  allpool learned runs (100) by outcome: {dict(allc)}")
kk = [r["end"]["eta_key_by_key"] for a, s, r in allr if hp.outcome(P.inv_view(r)) == "KEY"]
print(f"  KEY-class: {len(kk)} runs, eta_key_by_key >= 0.95 on {sum(x >= 0.95 for x in kk)}, min {min(kk):.3f}")
for a in P.LEARNED:
    unb = [r["acc"] for r in R[("allpool", a)].values() if not hp.bound(r)]
    print(f"  allpool.{a}: unbound {len(unb)}, median acc {statistics.median(unb):.3f}; all-run median "
          f"{statistics.median(r['acc'] for r in R[('allpool', a)].values()):.3f}")
b = pool["BASELINE_D"]
print(f"  pool.BASELINE_D: BOUND {sum(hp.bound(r) for r in b.values())}, ROUTED-BOUND {sum(rbv(r) for r in b.values())}, "
      f"G's 'BOUND ROUTED (mine)' {sum(hp.bound_routed_mine(r) for r in b.values())}; margin median "
      f"{statistics.median(r['end']['margin'] for r in b.values()):.3f}")
for a in ("REG3_KEYHINGE_D", "REG3_REPEATHINGE_D", "REG3_FREQHINGE_D"):
    rr = pool[a]
    f = [r["term"]["fired"] for r in rr.values()]
    ends = [r["end"]["beta"]["CTX"] for r in rr.values()]
    print(f"  pool.{a}: fired median {statistics.median(f)} ({min(f)}-{max(f)}); spread@2400 median "
          f"{statistics.median(P.commit(r) for r in rr.values()):.4f}; spread@3600 median "
          f"{statistics.median(P.commit(r, 3600) for r in rr.values() if P.commit(r, 3600) is not None):.4f}; "
          f"beta^R CTX end median {statistics.median(ends):.2f}")
# the operating characteristic V's v2 l.45 / l.269 relies on, at the observed REG3_D rate
def p_name(pa, n=20):
    out = collections.Counter()
    for m in range(n + 1):
        pm = math.comb(n, m) * pa ** m * (1 - pa) ** (n - m)
        kh_only = n - m
        needs = gc.mcnemar_greater(kh_only, 0) <= P.P4
        disc = gc.mcnemar_greater(m, 0) < 0.05
        nm = "needs" if needs else "replaces" if disc and kh_only <= 1 else "weaker" if disc else "does nothing"
        out[nm] += pm
    return {k: round(v, 3) for k, v in out.items()}
for pa in (0.8, 0.85, 0.9, 0.95):
    print(f"  names of an inert (REG3-rate {pa}) hinge arm with KH 20/20 nested, BASELINE 0/20: {p_name(pa)}")
print(f"  Fisher one-sided, 20/20 vs 16/20 (RP or KH vs REG3_D, unpaired): {gc.fisher_greater(20, 20, 16, 20):.4f}")
