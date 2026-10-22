"""check_G_report9.py — Referee V's check of Session G's report 9 (S89H) at claude/explore-G 996e38b7.

Reads git objects only (git show / rev-parse / log / diff); writes nothing; prints only. torch is not needed: G's pure
parent functions (explore_g_s89h at d5d48eeb, explore_g_s89 at d5d48eeb, explore_g_rhl_parent, explore_g_common's
statistics, test_conv_lr.mcnemar_greater, test_router_confirm.fisher_greater, test_router_layout's fail_class constants)
are exec'd from blobs with stub modules, as check_G_report8.py does, and G's own report() is run on the committed records
with its one file write sent to memory. rlib.py (the referee's own definitions) cross-checks the counts.
"""
import ast
import collections
import contextlib
import io
import json
import math
import os
import statistics
import subprocess
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib  # noqa: E402

TIP, FIRST, PREV = "996e38b7", "d5d48eeb", "8e9e3187"
CPU = "Intel(R) Xeon(R) Processor @ 2.10GHz"


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a, stderr=subprocess.DEVNULL).decode()


def show(ref, path):
    return git("show", f"{ref}:{path}")


def defs(src, names):
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


STORES = {n: json.loads(show(TIP, f"explore_out/G/{n}_results.json")) for n in ("s89h", "s89", "s87")}
VALID = {"s89_validity.json": json.loads(show(TIP, "explore_out/G/s89_validity.json")),
         "s89h_validity.json": json.loads(show(TIP, "explore_out/G/s89h_validity.json"))}

# ── stub modules ─────────────────────────────────────────────────────────────
ec = types.ModuleType("explore_common")
exec(defs(show(TIP, "test_router_layout.py"), {"fail_class", "MARGIN_PARTIAL", "ETA_SPLIT"}), ec.__dict__)
ec.cpu_model = lambda: CPU
ec.OUT_DIR = "/nonexistent/explore_out"
gc = types.ModuleType("explore_g_common")
gc.math = math
exec(defs(show(TIP, "explore_g_common.py"), {"wilson", "band", "cnt", "key", "runs"}), gc.__dict__)
_m = {}
exec(defs(show(TIP, "test_conv_lr.py"), {"mcnemar_greater"}), {"math": math}, _m)
exec(defs(show(TIP, "test_router_confirm.py"), {"fisher_greater"}), {"math": math}, _m)
gc.mcnemar_greater, gc.fisher_greater = _m["mcnemar_greater"], _m["fisher_greater"]
gc.G_DIR = "/nonexistent/G"
gc.load = lambda name: STORES[name]
gc.ec = ec
sys.modules["explore_common"] = ec
sys.modules["explore_g_common"] = gc
hp = types.ModuleType("explore_g_rhl_parent")
hp.__dict__["__file__"] = "/nonexistent/explore_g_rhl_parent.py"
exec(compile(show(TIP, "explore_g_rhl_parent.py"), "explore_g_rhl_parent.py", "exec"), hp.__dict__)
sys.modules["explore_g_rhl_parent"] = hp


class _Sink(io.StringIO):
    def close(self):
        pass


SINK = {}


def fake_open(p, mode="r", *a, **k):
    if "w" in mode:
        SINK[p] = _Sink()
        return SINK[p]
    raise RuntimeError(f"no file reads from disk: {p}")


def load_mod(ref, path, name):
    src = show(ref, path)
    m = types.ModuleType(name)
    m.__dict__.update(__file__=f"/nonexistent/{path}", __name__=name, open=fake_open)
    sys.modules[name] = m
    for node in ast.parse(src).body:
        if isinstance(node, ast.If):
            continue
        exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), m.__dict__)
        if isinstance(node, ast.FunctionDef) and node.name == "_load":
            m._load = lambda p: next((v for k, v in VALID.items() if p.endswith("/" + k)), None)
    return m


S89 = load_mod(FIRST, "explore_g_s89.py", "explore_g_s89")
P = load_mod(FIRST, "explore_g_s89h.py", "explore_g_s89h")
rbv = hp.routed_bound

# ═══ 0. commits ══════════════════════════════════════════════════════════════
H(f"0. commits {FIRST}..{TIP}: code blobs, records per commit")
revs = git("rev-list", "--reverse", f"{FIRST}^..{TIP}").split()
CODE = ("explore_g_s89h.py", "explore_g_s89h_child.py", "explore_g_s89.py", "explore_g_s89_child.py",
        "explore_g_rhl_parent.py", "explore_g_rhl_child.py", "explore_g_common.py", "explore_f_tasks.py",
        "explore_common16.py", "explore_common.py", "test_short_conv.py", "test_multilayer_binding.py",
        "explore_out/G/refs/f_s89_post_conv_records.json")
blobs = collections.defaultdict(set)
rows0, first_rec, first_learned = [], None, None
for c in revs:
    t = git("log", "-1", "--format=%ci", c).strip()
    for f in CODE:
        try:
            blobs[f].add(git("rev-parse", f"{c}:{f}").strip()[:10])
        except subprocess.CalledProcessError:
            blobs[f].add(None)
    try:
        ks = list(json.loads(show(c, "explore_out/G/s89h_results.json"))["runs"])
    except subprocess.CalledProcessError:
        ks = []
    learned = [k for k in ks if "PERFECT" not in k]
    if ks and first_rec is None:
        first_rec = (c[:8], t, ks)
    if learned and first_learned is None:
        first_learned = (c[:8], t, learned)
    files = git("diff-tree", "--no-commit-id", "--name-only", "-r", c).split()
    rows0.append((c[:8], t, len(ks), len(learned), [f.split("/")[-1] for f in files if not f.endswith("results.json")]))
for r in rows0:
    print(f"  {r[0]} {r[1]}  records {r[2]:3d}  learned {r[3]:3d}  other files {r[4]}")
print("  code blobs over the range (one value = unchanged):")
for f, b in blobs.items():
    print(f"    {f}: {sorted(x for x in b if x) or b}")
changed = git("diff", "--name-only", FIRST, TIP).split()
print(f"  files changed {FIRST}..{TIP}: {changed}")
print(f"  .py files changed: {[f for f in changed if f.endswith('.py')]}")
print(f"  .py files changed {PREV}..{FIRST}: {[f for f in git('diff', '--name-only', PREV, FIRST).split() if f.endswith('.py')]}")
print(f"  .py files changed db14e9b9..{FIRST} (S89's code): "
      f"{[f for f in git('diff', '--name-only', 'db14e9b9', FIRST).split() if f.endswith('.py')]}")
print(f"  first record(s) at {first_rec[0]} {first_rec[1]}: {first_rec[2]}")
print(f"  first learned record(s) at {first_learned[0]} {first_learned[1]}: {first_learned[2]}")
s89_last = git("log", "-1", "--format=%h %ci %s", "d745d869").strip()[:80]
print(f"  S89's last progress commit: {s89_last}; S89 records then: "
      f"{len(json.loads(show('d745d869', 'explore_out/G/s89_results.json'))['runs'])}")
print(f"  report 8: {git('log', '-1', '--format=%h %ci', PREV).strip()}")
for c in ("c42ace84", "b2169dc0"):
    print(f"  review-V {git('log', '-1', '--format=%h %ci %s', c).strip()[:120]}")
print(f"  restart banner's git cd7a00b is an ancestor of {TIP}: "
      f"{subprocess.call(('git', '-C', ROOT, 'merge-base', '--is-ancestor', 'cd7a00b', TIP)) == 0}")

# ═══ 1. records ═══════════════════════════════════════════════════════════════
H("1. the S89H records at 996e38b7")
S = STORES["s89h"]
runs = S["runs"]
print(f"  n {len(runs)}; meta sha {S['meta']['sha'][:12]}; provenance {S['meta']['provenance']}; sha_rows {len(S['meta']['sha_rows'])}")
print(f"  key SHAs {collections.Counter(k.split('|')[2] for k in runs)}; record sha {collections.Counter(r['sha'] for r in runs.values())}")
print(f"  CPUs {collections.Counter(r['cpu'] for r in runs.values())}; ok {sum(r['ok'] for r in runs.values())}; "
      f"attempts {collections.Counter(r['attempt'] for r in runs.values())}")
print(f"  CPU-hours (sum secs) {sum(r['secs'] for r in runs.values()) / 3600:.2f}; sum secs_wall "
      f"{sum(r['secs_wall'] for r in runs.values()) / 3600:.2f}")
print("  BUDGET actually run, per task (record 'iters', max 'stopped_at', last curve step):")
for t in P.TASK_NAMES:
    rr = [r for k, r in runs.items() if k.startswith(t + ".")]
    print(f"    {t}: parent ITERS {P.ITERS[t]}; meta arms iters {sorted(set(a['iters'] for k, a in S['meta']['arms'].items() if k.startswith(t + '.')))}; "
          f"record iters {collections.Counter(r['iters'] for r in rr)}; max stopped_at {max(r['stopped_at'] for r in rr)}; "
          f"max last curve step {max(r['curve'][-1][0] for r in rr)}; unbound runs stopped at "
          f"{collections.Counter(r['stopped_at'] for r in rr if r['transition'] is None)}")
print(f"  S89 delta records: iters {collections.Counter((k.split('.')[0], r['iters']) for k, r in STORES['s89']['runs'].items())}")
print(f"  validity file: {VALID['s89h_validity.json']}")
cf = [f for f in git("ls-tree", "--name-only", TIP, "explore_out/G/").split() if "checks_s89h" in f]
for f in cf:
    d = json.loads(show(TIP, f))
    print(f"  {f}: rows {len(d['rows'])}, ok {sum(1 for _, o in d['rows'] if o)}, '(S89)' rows "
          f"{sum(1 for n, _ in d['rows'] if n.startswith('(S89)'))}, git {d['provenance'].get('git')}")
    if f == cf[0]:
        for n, o in d["rows"]:
            if "readings hand case" in n or "inv_view" in n:
                print(f"      {o} {n[:170]}")
log = show(TIP, "explore_out/G/s89h.log")
L = log.splitlines()
print(f"  log: '=== RESTART' {[l for l in L if l.startswith('=== RESTART')]}; 'CHECK S89H:' lines "
      f"{sum('CHECK S89H:' in l for l in L)}, FAIL {sum(1 for l in L if 'CHECK S89H:' in l and l.endswith('FAIL'))}")
print(f"  log banners: code SHA {collections.Counter(l.split('code SHA ')[1].split()[0] for l in L if 'code SHA' in l)}; "
      f"git {[l.split('git ')[1].split()[0] for l in L if '| git ' in l]}")
print(f"  learned records before the restart (progress lines before it): "
      f"{max(int(l.split('[')[1].split('/')[0]) for l in L[:L.index(next(x for x in L if x.startswith('=== RESTART')))] if l.strip().startswith('[') and '/200]' in l)}")

# ═══ 2. report() ═════════════════════════════════════════════════════════════
H("2. G's own report() (d5d48eeb blob) on the committed records, compared with s89h.log")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    P.report()
out = buf.getvalue().rstrip("\n").splitlines()
i0 = max(i for i, l in enumerate(L) if l.startswith("| arm | n | BOUND"))
tail = [l for l in L[i0:] if l != "SCREEN_DONE"]
diff = [(i, a, b) for i, (a, b) in enumerate(zip(out, tail)) if a != b]
print(f"  report() lines {len(out)}, log lines from the count table {len(tail)} (+ SCREEN_DONE), differing {len(diff)}")
for i, a, b in diff[:10]:
    print(f"    line {i}:\n      mine: {a[:300]}\n      log : {b[:300]}")
summ = json.loads(SINK[[k for k in SINK if k.endswith("s89h_summary.json")][0]].getvalue())
print(f"  s89h_summary.json from report() == committed: {summ == json.loads(show(TIP, 'explore_out/G/s89h_summary.json'))}")

# ═══ 3. counts ═══════════════════════════════════════════════════════════════
H("3. counts per task and arm (G's routed_bound / inv_view; rlib cross-check)")


def recs(store, task, arm):
    out = {}
    for k, r in STORES[store]["runs"].items():
        a, s = k.split("|")[:2]
        if a == f"{task}.{arm}":
            out[int(s)] = r
    return out


def rl_routed(r, inv):
    e = r["end"]
    return rlib.bound(r) and e["margin"] >= 0.9 and e["eta_key_by_stream_inv" if inv else "eta_key_by_stream"] > 0.9


def view(t, rr):
    return {s: P.inv_view(r) for s, r in rr.items()} if t == "allpool" else rr


R, RD = {}, {}
for t in P.TASK_NAMES:
    for a in ("PERFECT_H",) + P.LEARNED:
        rr = recs("s89h", t, a)
        R[(t, a)] = rr
        g = sum(rbv(P.inv_view(r) if t == "allpool" else r) for r in rr.values())
        go = sum(rbv(r if t == "allpool" else P.inv_view(r)) for r in rr.values())
        rl = sum(rl_routed(r, t == "allpool") for r in rr.values())
        bd, bdr = sum(hp.bound(r) for r in rr.values()), sum(rlib.bound(r) for r in rr.values())
        tr = sorted(r["transition"] for r in rr.values() if r["transition"] is not None)
        trs = f"median {statistics.median(tr):.0f} ({tr[0]}-{tr[-1]})" if tr else "-"
        trr = sorted(r["transition"] for r in rr.values() if rbv(P.inv_view(r) if t == "allpool" else r))
        cls = collections.Counter(hp.outcome(P.inv_view(r) if t == "allpool" else r) for r in rr.values() if not hp.bound(r))
        lo, hi = rlib.wilson(g, len(rr))
        print(f"  {t}.{a:<19} n {len(rr):2d} RB G {g:2d} [{lo:.2f}, {hi:.2f}] {rlib.band(g, len(rr))} | rlib {rl:2d} | other form {go:2d}"
              f" | BOUND {bd:2d} rlib {bdr:2d} | trans(bound) {trs} | trans(RB) {('median %d' % statistics.median(trr)) if trr else '-'}"
              f" | seeds {sorted(rr)[0] if rr else '-'}-{sorted(rr)[-1] if rr else '-'} | unbound {dict(cls)}")
    for k, a in P.ROLE_D.items():
        RD[(t, k)] = recs("s89", t, a)
    RD[(t, "O")] = recs("s89", t, "ORACLE_D")
print("  S89 delta (same seeds):", {f"{t}.{k}": sum(rbv(P.inv_view(r) if t == "allpool" else r) for r in RD[(t, k)].values())
                                   for t in P.TASK_NAMES for k in list(P.ROLE_D) + ["O"]})
print("  lite S87 Hebbian 2700-2719:", {a: sum(rbv(r) for r in S89.recorded("s87", a, range(2700, 2720)).values())
                                       for a in ("BASELINE", "REG3_KEYHINGE", "REG3_FREQHINGE")})
pool_b = R[("pool", "BASELINE_H")]
bb = [r for r in pool_b.values() if hp.bound(r)]
print(f"  pool BASELINE_H bound {len(bb)}: margins {[round(r['end']['margin'], 3) for r in bb]}; transitions "
      f"{[r['transition'] for r in bb]}; all-run median margin {statistics.median(r['end']['margin'] for r in pool_b.values()):.3f}")
kh = R[("pool", "REG3_KEYHINGE_H")]
print(f"  pool KH not RB: {[(s, hp.outcome(r), r['transition'], round(r['end']['margin'], 3), round(r['end']['eta_key_by_stream'], 3)) for s, r in sorted(kh.items()) if not rbv(r)]}")
for a in P.LEARNED:
    print(f"  pool {a} not RB seeds: {[s for s, r in sorted(R[('pool', a)].items()) if not rbv(r)]}")

# ═══ 4. the 2b reading ═══════════════════════════════════════════════════════
H("4. reading 2b and every hinge vs the register (b = arm only, c = REG3 only)")
NM = dict(B="BASELINE", R="REG3", KH="KH", FQ="FREQ", RP="REPEAT")
for t in P.TASK_NAMES:
    Rr = {k: view(t, R[(t, a)]) for k, a in P.ROLE.items()}
    for x in ("KH", "FQ", "RP", "B"):
        b, c, p, n = hp.paired(Rr[x], Rr["R"], rbv)
        p2 = hp.paired(Rr["R"], Rr[x], rbv)[2]
        print(f"  S89H {t}: {NM[x]} vs REG3: {b}/{c}, p(arm >) {p:.3g}, p(REG3 >) {p2:.3g}, two-sided {rlib.mcnemar2(b, c):.3g}, n {n}"
              f"  (rlib {rlib.pairs(Rr[x], Rr['R'], lambda r: rl_routed(r, False))[:2]})")
    for x in ("KH", "FQ", "RP"):
        b, c, p, n = hp.paired(view(t, RD[(t, x)]), view(t, RD[(t, "R")]), rbv)
        print(f"  S89 delta {t}: {NM[x]}_D vs REG3_D: {b}/{c}, p {p:.3g}, two-sided {rlib.mcnemar2(b, c):.3g}")
    for x, y in (("KH", "FQ"), ("KH", "RP"), ("FQ", "B"), ("RP", "B"), ("KH", "B")):
        b, c, p, n = hp.paired(Rr[x], Rr[y], rbv)
        print(f"  S89H {t}: {NM[x]} vs {NM[y]}: {b}/{c}, p {p:.3g}")
# lite: S82 (Hebbian, delta) and S87 (delta)
s82 = json.loads(show(TIP, "explore_out/G/s82_devices_results.json"))


def r82(a):
    return {int(k.split("|")[1]): v for k, v in s82["runs"].items() if k.split("|")[0] == a}


for (st, kha, ra) in (("S82 Hebbian lite 2100-2119", r82("REG3_KEYHINGE"), r82("REG3")),
                      ("S82 delta lite 2100-2109", r82("REG3_KEYHINGE_D"), r82("REG3_D")),
                      ("S87 delta lite 2720-2729", S89.recorded("s87", "REG3_KEYHINGE_D", range(2720, 2730)),
                       S89.recorded("s87", "REG3_D", range(2720, 2730))),
                      ("S87 delta lite FREQ_D", S89.recorded("s87", "REG3_FREQHINGE_D", range(2720, 2730)),
                       S89.recorded("s87", "REG3_D", range(2720, 2730)))):
    b, c, p, n = hp.paired(kha, ra, rbv)
    print(f"  {st}: hinge {sum(rbv(r) for r in kha.values())}/{len(kha)}, REG3 {sum(rbv(r) for r in ra.values())}/{len(ra)}; "
          f"hinge only {b}, REG3 only {c}, p {p:.3g}, two-sided {rlib.mcnemar2(b, c):.3g}, n {n}")
print(f"  S82 recipes: {collections.Counter((k.split('|')[0], v.get('recipe')) for k, v in s82['runs'].items() if k.split('|')[0] in ('REG3', 'REG3_KEYHINGE'))}")
print(f"  S87 Hebbian lite has a REG3 arm: {any(k.split('|')[0] == 'REG3' for k in STORES['s87']['runs'])}")
# the names G's rule gives to counterfactual arms on POOL
Rp = {k: R[("pool", a)] for k, a in P.ROLE.items()}
print(f"  arm_name(actual RP): {S89.arm_name(Rp['KH'], Rp['RP'], Rp['B'], 'RP', 'pool')[0]}")
print(f"  arm_name(an inert hinge = REG3_H's records): {S89.arm_name(Rp['KH'], Rp['R'], Rp['B'], 'RP', 'pool')}")
khs = [rbv(Rp["KH"][s]) for s in P.S20]


def p_names(pa):
    """arm iid Bernoulli(pa) per seed, KH's and BASELINE's observed patterns fixed: P(each name)."""
    out = collections.Counter()
    nk = sum(khs)
    nn = len(khs) - nk
    for i in range(nk + 1):            # arm successes on KH seeds
        for j in range(nn + 1):        # arm successes on non-KH seeds
            pr = (math.comb(nk, i) * pa ** i * (1 - pa) ** (nk - i)) * (math.comb(nn, j) * pa ** j * (1 - pa) ** (nn - j))
            b, c = nk - i, j          # KH only, arm only
            disc = gc.mcnemar_greater(i + j, 0) < 0.05
            nm = ("needs" if gc.mcnemar_greater(b, c) <= P.P4 else "replaces" if disc and b - c <= 1 else
                  "weaker" if disc else "does nothing")
            out[nm] += pr
    return {k: round(v, 4) for k, v in sorted(out.items())}


lo, hi = rlib.wilson(9, 20)
for pa in (0.45, round(hi, 3), 0.8, 0.9, 0.95):
    print(f"  names of a hinge arm drawn iid at rate {pa} (KH 18/20 and BASELINE 0/20 as observed): {p_names(pa)}")
print("  readings() (d5d48eeb) per task:")
for t in P.TASK_NAMES:
    rd = P.readings({k: view(t, R[(t, a)]) for k, a in P.ROLE.items()}, P.S20, t, valid=True, valid_note="")
    for k, v in rd.items():
        print(f"    {t}: {k}: {v}")

# ═══ 5. across memories ══════════════════════════════════════════════════════
H("5. across memories: S89H (Hebbian + conv) vs S89 (delta), per task and arm")
for t in P.TASK_NAMES:
    for k, a in P.ROLE.items():
        h, d = view(t, R[(t, a)]), view(t, RD[(t, k)])
        b, c, p, n = hp.paired(h, d, rbv)
        p2 = hp.paired(d, h, rbv)[2]
        nh, nd = sum(rbv(r) for r in h.values()), sum(rbv(r) for r in d.values())
        fis = gc.fisher_greater(nh, 20, nd, 20) if nh >= nd else gc.fisher_greater(nd, 20, nh, 20)
        print(f"  {t}.{a} vs {P.ROLE_D[k]}: H {nh}/{len(h)}, D {nd}/{len(d)}; same seed set {set(h) == set(d)}; "
              f"H only {b}, D only {c}; p(H >) {p:.3g}, p(D >) {p2:.3g}, two-sided {rlib.mcnemar2(b, c):.3g}; "
              f"unpaired Fisher one-sided {fis:.3g}; both {sum(rbv(h[s]) and rbv(d[s]) for s in P.S20)}, neither "
              f"{sum(not rbv(h[s]) and not rbv(d[s]) for s in P.S20)}")
    d_it = collections.Counter(r["iters"] for k_ in P.ROLE for r in RD[(t, k_)].values())
    h_it = collections.Counter(r["iters"] for a in P.LEARNED for r in R[(t, a)].values())
    print(f"  {t}: budgets run: S89H {dict(h_it)}, S89 {dict(d_it)}")
# across tasks
for a in ("BASELINE_H", "REG3_H", "REG3_KEYHINGE_H", "REG3_REPEATHINGE_H"):
    b, c, p, n = hp.paired(R[("pool", a)], view("allpool", R[("allpool", a)]), rbv)
    print(f"  across tasks {a}: pool only {b}, allpool only {c}, p {p:.3g}")

# ═══ 6. ALLPOOL ══════════════════════════════════════════════════════════════
H("6. ALLPOOL under S89H: per run (within-sequence fields; outcome on inv_view)")
cap = 64 / 120
allr = [(a, s, r) for a in P.LEARNED for s, r in sorted(R[("allpool", a)].items())]
for a in P.LEARNED:
    rr = R[("allpool", a)]
    e = [r["end"] for r in rr.values()]
    srf = [x["seq_routed_frac"] for x in e]
    print(f"  allpool.{a}: classes {dict(collections.Counter(hp.outcome(P.inv_view(r)) for r in rr.values()))}; "
          f"eta_inv median {statistics.median(x['eta_key_by_stream_inv'] for x in e):.3f} "
          f"(range {min(x['eta_key_by_stream_inv'] for x in e):.3f}-{max(x['eta_key_by_stream_inv'] for x in e):.3f}); "
          f"seq_routed median {statistics.median(srf):.3f} range {min(srf):.3f}-{max(srf):.3f}; "
          f"in [0.33, 0.57]: {sum(0.33 <= x <= 0.57 for x in srf)}; in [0.40, 0.60]: {sum(0.40 <= x <= 0.60 for x in srf)}; "
          f">= 0.30: {sum(x >= 0.30 for x in srf)}; margin max {max(x['margin'] for x in e):.3f}; "
          f"val_cos_inv < 0.5: {sum(x['val_cos_inv'] < 0.5 for x in e)}; acc median {statistics.median(r['acc'] for r in rr.values()):.3f}; "
          f"commit@2400 {S89.med_commit(rr):.4f}")
    for s, r in sorted(rr.items()):
        x = r["end"]
        print(f"    {s} {hp.outcome(P.inv_view(r)):<15} acc {r['acc']:.3f} margin {x['margin']:.3f} eta_inv {x['eta_key_by_stream_inv']:.3f} "
              f"eta_pooled {x['eta_key_by_stream']:.3f} seq_routed {x['seq_routed_frac']:.3f} val_cos_inv {x['val_cos_inv']:.3f} "
              f"eta_by_key {x['eta_key_by_key']:.3f} stopped {r['stopped_at']}")
print(f"  cap 64/120 = {cap:.4f}; learned allpool runs: max margin {max(r['end']['margin'] for *_, r in allr):.3f}, max seq_routed "
      f"{max(r['end']['seq_routed_frac'] for *_, r in allr):.3f}, max eta_inv {max(r['end']['eta_key_by_stream_inv'] for *_, r in allr):.3f}; "
      f"bound {sum(hp.bound(r) for *_, r in allr)}; above cap (seq_routed > 0.5334) "
      f"{[(a, s, round(r['end']['seq_routed_frac'], 3)) for a, s, r in allr if r['end']['seq_routed_frac'] > cap + 1e-4]}")
print(f"  classes over the 100 learned runs: {dict(collections.Counter(hp.outcome(P.inv_view(r)) for *_, r in allr))}")
mg = [(a, s, r) for a, s, r in allr if hp.outcome(P.inv_view(r)) == "MERGED"]
print(f"  MERGED runs {len(mg)}: seq_routed {min(r['end']['seq_routed_frac'] for *_, r in mg):.3f}-{max(r['end']['seq_routed_frac'] for *_, r in mg):.3f}, "
      f"margin {min(r['end']['margin'] for *_, r in mg):.3f}-{max(r['end']['margin'] for *_, r in mg):.3f}, acc "
      f"{min(r['acc'] for *_, r in mg):.3f}-{max(r['acc'] for *_, r in mg):.3f}")
for t in P.TASK_NAMES:
    for s, r in sorted(R[(t, "PERFECT_H")].items()):
        e = r["end"]
        print(f"  {t}.PERFECT_H {s}: trans {r['transition']}, stopped {r['stopped_at']}, margin {e['margin']:.3f}, eta pooled "
              f"{e['eta_key_by_stream']:.3f} inv {e['eta_key_by_stream_inv']:.3f}, seq_routed {e['seq_routed_frac']:.3f}")
ch = show(FIRST, "explore_g_s89h_child.py").splitlines()
print("  s89h child lines mentioning perfect_gate:")
for i, l in enumerate(ch, 1):
    if "perfect_gate" in l:
        print(f"    child:{i}: {l.strip()[:150]}")

# ═══ 7. remaining numbers ════════════════════════════════════════════════════
H("7. remaining numbers in report 9")
pr = R[("pool", "REG3_H")]


def beta_at(r, step, role="CTX"):
    for s_ in r["stats"]:
        if s_["step"] == step and s_.get("beta"):
            return s_["beta"][role]
    return None


fails = [r for r in pr.values() if not rbv(r)]
for st_ in (0, 1200, 2400, 4800, 24000):
    v = [beta_at(r, st_) for r in fails if beta_at(r, st_) is not None]
    print(f"  pool REG3_H failures ({len(fails)}): beta^R CTX at {st_} median {statistics.median(v):.3f} (n {len(v)})")
print(f"  pool REG3_H failures: end beta CTX median {statistics.median(r['end']['beta']['CTX'] for r in fails):.3f}; "
      f"KEY {statistics.median(r['end']['beta']['KEY'] for r in fails):.3f}; classes {collections.Counter(hp.outcome(r) for r in fails)}")
for a in ("REG3_KEYHINGE_H", "REG3_REPEATHINGE_H", "REG3_FREQHINGE_H"):
    rr = R[("pool", a)]
    c12 = statistics.median(S89.commit(r, 1200) for r in rr.values())
    c24 = statistics.median(S89.commit(r, 2400) for r in rr.values())
    c36 = statistics.median(S89.commit(r, 3600) for r in rr.values() if S89.commit(r, 3600) is not None)
    b = {st_: statistics.median(x for x in (beta_at(r, st_) for r in rr.values()) if x is not None) for st_ in (2400, 4800)}
    l1 = rr and statistics.median(r["end"]["layers"]["L1"]["table"]["KEY"][0][0] for r in rr.values())
    print(f"  pool {a}: commit 1200 {c12:.4f}, 2400 {c24:.4f}, 3600 {c36:.4f}; beta CTX 2400 {b[2400]:.3f}, 4800 {b[4800]:.3f}, "
          f"end {statistics.median(r['end']['beta']['CTX'] for r in rr.values()):.3f}; L1 KEY g(s0,ch0) median {l1:.3f}")
for a in ("BASELINE_H", "REG3_H"):
    print(f"  allpool {a}: commit@2400 {S89.med_commit(R[('allpool', a)]):.4f}")

# ═══ 8. F's records (vendored and quoted) ════════════════════════════════════
H("8. F's conv records: G's vendored copy against F's 129215e store; O_TYPE_HEBB_ALLPOOL")
FREC = json.loads(show("129215e", "explore_out/F/s89_post_results.json"))["runs"]
VEN = json.loads(show(TIP, "explore_out/G/refs/f_s89_post_conv_records.json"))
print(f"  F store blob {git('rev-parse', '129215e:explore_out/F/s89_post_results.json').strip()[:12]}; vendored source: {VEN['source']}")
for k, r in VEN["records"].items():
    fk = [x for x in FREC if x.startswith(k + "|")]
    print(f"  vendored {k}: F key {fk[0][:40] if fk else None}; record equal {bool(fk) and FREC[fk[0]] == r}")
for k, r in sorted(FREC.items()):
    if k.split("|")[0] in ("O_HEBB_CONV_POOL", "O_HEBB_CONV_ALLPOOL", "O_HEBB_CONV_ALLPOOL_48K", "O_TYPE_HEBB_ALLPOOL"):
        print(f"  F {k.split('|')[0]}|{k.split('|')[1]}: iters {r.get('iters')}, transition {r.get('transition')}, stopped {r.get('stopped_at')}")
