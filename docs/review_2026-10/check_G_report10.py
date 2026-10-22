"""check_G_report10.py — Referee V's check of Session G's report 10 (S90, S91) at claude/explore-G cbca3e36.

Reads git objects only (git show / rev-parse / log / ls-tree / grep); writes nothing; prints only. torch is not needed:
G's pure parent functions (explore_g_s90 at b57c1867, explore_g_s91 at 60b1932a, explore_g_s89h and explore_g_s89,
explore_g_rhl_parent, explore_g_common's store helpers, test_conv_lr.mcnemar_greater, test_router_layout's fail_class
constants) are exec'd from blobs statement by statement with stub modules, as check_G_report9.py does. G's own report()
of each screen is run on the committed records with its one file write sent to memory, and its output is compared with
the committed log line by line.
"""
import ast
import collections
import io
import json
import math
import statistics
import subprocess
import sys
import types

ROOT = "/home/user/clankers"
TIP = "cbca3e36"            # report 10
S90C, S90D = "b57c1867", "2c1913b2"   # S90 code commit; S90 done
SPEC2, S91C = "6fc55b17", "60b1932a"  # spec rev 2; S91 code commit
CPU = "Intel(R) Xeon(R) Processor @ 2.10GHz"


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a, stderr=subprocess.DEVNULL).decode()


def show(ref, path):
    return git("show", f"{ref}:{path}")


def H(t):
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)


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


def blob(ref, path):
    try:
        return git("rev-parse", f"{ref}:{path}").strip()[:10]
    except subprocess.CalledProcessError:
        return "-"


# ═══ 0. commits, times, blobs, records per commit ═══════════════════════════
H("0. commits b57c1867^..cbca3e36 (UTC): files, S90 / S91 records per commit, code blobs")
print(f"origin/claude/explore-G now at {git('rev-parse', 'origin/claude/explore-G').strip()[:8]} (read here: {TIP})")
revs = git("rev-list", "--reverse", f"{S90C}^..{TIP}").split()
CODE = ("explore_g_s90.py", "explore_g_s90_child.py", "explore_g_s91.py", "explore_g_s91_child.py", "explore_g_s89h.py",
        "explore_g_s89h_child.py", "explore_g_s89.py", "explore_g_s89_child.py", "explore_g_rhl_parent.py",
        "explore_g_rhl_child.py", "explore_g_common.py", "explore_common16.py", "explore_common.py",
        "explore_f_tasks.py", "test_short_conv.py")
blobs = collections.defaultdict(list)
for c in revs:
    t = git("log", "-1", "--format=%ci", c).strip()[:19]
    n = {}
    for s in ("s90", "s91"):
        try:
            n[s] = len(json.loads(show(c, f"explore_out/G/{s}_results.json"))["runs"])
        except subprocess.CalledProcessError:
            n[s] = None
    files = git("diff-tree", "--no-commit-id", "--name-only", "-r", c).split()
    print(f"  {c[:8]} {t}  s90 {n['s90']}  s91 {n['s91']}  files {[f.split('/')[-1] for f in files]}")
    for f in CODE:
        b = blob(c, f)
        if not blobs[f] or blobs[f][-1][0] != b:
            blobs[f].append((b, c[:8]))
print("  code blobs over the range ((blob, first commit) per change; one entry = unchanged):")
for f, b in blobs.items():
    print(f"    {f:<26} {b}")
print(f"  .py changed {S90C}..{TIP}: {[f for f in git('diff', '--name-only', S90C, TIP).split() if f.endswith('.py')]}")
print(f"  .py changed {S90C}..{S91C}^: {[f for f in git('diff', '--name-only', S90C, S91C + '^').split() if f.endswith('.py')]}")
print(f"  .py changed {S91C}..{TIP}: {[f for f in git('diff', '--name-only', S91C, TIP).split() if f.endswith('.py')]}")
for br, path in (("origin/claude/review-V", "docs/review_2026-10/screen_review_s90_G.md"),
                 ("origin/claude/review-V", "docs/review_2026-10/check_G_report9.md")):
    print(f"  review-V {path.split('/')[-1]}: {git('log', '--format=%h %ci', br, '--', path).split(chr(10))[-2] if False else git('log', '--reverse', '--format=%h %ci', br, '--', path).splitlines()[0]}")
print(f"  review-V files mentioning s91: {[f for f in git('ls-tree', '--name-only', 'origin/claude/review-V', 'docs/review_2026-10/').split() if 's91' in f.lower()]}")
print("  explore_g_common.py:160-166 at S91's commit (the budget line, V's fix 8):")
for i, l in enumerate(show(S91C, "explore_g_common.py").splitlines()[159:166], 160):
    print(f"    {i}: {l}")
print("  explore_g_s91_child.py:229 at S91's commit:")
print("    229:", show(S91C, "explore_g_s91_child.py").splitlines()[228].strip())

# ═══ stubs, as check_G_report9.py ════════════════════════════════════════════
STORES = {n: json.loads(show(TIP, f"explore_out/G/{n}_results.json")) for n in ("s90", "s91", "s87", "s89h", "s89")}
VALIDJ = {"s89_validity.json": json.loads(show(TIP, "explore_out/G/s89_validity.json")),
          "s89h_validity.json": json.loads(show(TIP, "explore_out/G/s89h_validity.json")),
          "s91_validity.json": json.loads(show(TIP, "explore_out/G/s91_validity.json"))}
ec = types.ModuleType("explore_common")
exec(defs(show(TIP, "test_router_layout.py"), {"fail_class", "MARGIN_PARTIAL", "ETA_SPLIT"}), ec.__dict__)
ec.cpu_model = lambda: CPU
gc = types.ModuleType("explore_g_common")
gc.math = math
exec(defs(show(S91C, "explore_g_common.py"), {"wilson", "band", "cnt", "key", "runs"}), gc.__dict__)
_m = {}
exec(defs(show(TIP, "test_conv_lr.py"), {"mcnemar_greater"}), {"math": math}, _m)
gc.mcnemar_greater = _m["mcnemar_greater"]
gc.fisher_greater = lambda *a: float("nan")
gc.G_DIR = "/nonexistent/G"
gc.WORKERS = 4
gc.load = lambda name: STORES[name]
gc.ec = ec
sys.modules["explore_common"] = ec
sys.modules["explore_g_common"] = gc
hp = types.ModuleType("explore_g_rhl_parent")
hp.__dict__["__file__"] = "/nonexistent/explore_g_rhl_parent.py"
exec(compile(show(TIP, "explore_g_rhl_parent.py"), "explore_g_rhl_parent.py", "exec"), hp.__dict__)
sys.modules["explore_g_rhl_parent"] = hp
SINK = {}


def fake_open(p, mode="r", *a, **k):
    if "w" in mode:
        SINK[p] = io.StringIO()
        SINK[p].close = lambda: None
        return SINK[p]
    raise RuntimeError(f"no file reads from disk: {p}")


def load_mod(ref, path, name, swap=None):
    src = show(ref, path)
    m = types.ModuleType(name)
    m.__dict__.update(__file__=f"/nonexistent/{path}", __name__=name, open=fake_open)
    sys.modules[name] = m
    for node in ast.parse(src).body:
        if isinstance(node, ast.If):
            continue
        exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), m.__dict__)
        if isinstance(node, ast.FunctionDef) and node.name == "_load":
            m._load = lambda p: next((v for k, v in VALIDJ.items() if p.endswith("/" + k)), None)
        if swap and isinstance(node, ast.FunctionDef) and node.name in swap:
            setattr(m, node.name, swap[node.name](m))
    return m


S89 = load_mod(S90C, "explore_g_s89.py", "explore_g_s89")
S89H = load_mod(S90C, "explore_g_s89h.py", "explore_g_s89h")
S90 = load_mod(S90C, "explore_g_s90.py", "explore_g_s90")
rb = hp.routed_bound
Q = S89H.Q


def run_report(mod):
    buf, old = io.StringIO(), sys.stdout
    try:
        sys.stdout = buf
        mod.report()
    finally:
        sys.stdout = old
    return buf.getvalue().splitlines()


def compare(out, log, first):
    L = log.splitlines()
    i = next(k for k, l in enumerate(L) if l.startswith(first))
    tail = [l for l in L[i:] if l != "SCREEN_DONE"]
    d = [(k, a, b) for k, (a, b) in enumerate(zip(out, tail)) if a != b]
    print(f"  regenerated {len(out)} lines; log from the count table {len(tail)} lines (SCREEN_DONE dropped); "
          f"differing lines {len(d)}; length equal {len(out) == len(tail)}")
    for k, a, b in d[:5]:
        print(f"    line {k}:\n      mine {a[:200]}\n      log  {b[:200]}")


# ═══ 1. S90 records ════════════════════════════════════════════════════════════
H("1. S90 records at cbca3e36 (and S87's paired records)")
S = STORES["s90"]
runs = S["runs"]
print(f"  n {len(runs)}; meta sha {S['meta']['sha'][:12]}; sha_rows {len(S['meta']['sha_rows'])}; provenance {S['meta']['provenance']}")
print(f"  key SHAs {collections.Counter(k.split('|')[2] for k in runs)}; ok {sum(r['ok'] for r in runs.values())}; "
      f"attempts {collections.Counter(r['attempt'] for r in runs.values())}; CPUs {collections.Counter(r['cpu'] for r in runs.values())}")
print(f"  iters {collections.Counter(r['iters'] for r in runs.values())}; tasks {collections.Counter(r.get('task') for r in runs.values())}; "
      f"recipes {collections.Counter(r.get('recipe') for r in runs.values())}; k {collections.Counter(r.get('k') for r in runs.values())}; "
      f"max stopped_at {max(r['stopped_at'] for r in runs.values())}")
print(f"  modules in sha_rows include: s90 parent {any('explore_g_s90.py' in str(x) for x in S['meta']['sha_rows'])}, "
      f"explore_g_common {any('explore_g_common' in str(x) for x in S['meta']['sha_rows'])}")
print(f"  s90 store identical at {S90D} and {TIP}: {show(S90D, 'explore_out/G/s90_results.json') == show(TIP, 'explore_out/G/s90_results.json')}")
R = S90.records()
for k in ("B", "R", "KH", "FQ", "RP"):
    rr = R[k]
    ok = [s for s in S90.S20 if s in rr and rb(rr[s])]
    tr = [rr[s]["transition"] for s in ok]
    print(f"  {S90.LABEL[k]:<22} n {len(rr)} ROUTED-BOUND {len(ok)}/20 Wilson {gc.wilson(len(ok), 20)}; fail {[s for s in S90.S20 if s not in ok]}; "
          f"fail classes {collections.Counter(ec.fail_class(rr[s]) for s in rr if not rb(rr[s]))}; trans median "
          f"{statistics.median(tr) if tr else None} ({min(tr) if tr else None}-{max(tr) if tr else None}); iters {collections.Counter(r['iters'] for r in rr.values())}")
print(f"  S90 run wall: sum secs_wall {sum(r['secs_wall'] for r in runs.values()) / 3600:.2f} h")

H("1b. paired tables (exact one-sided McNemar, G's hp.paired)")
for a, b in (("KH", "RP"), ("RP", "KH"), ("KH", "FQ"), ("KH", "R"), ("RP", "R"), ("R", "RP"), ("FQ", "R"),
             ("FQ", "RP"), ("RP", "FQ"), ("RP", "B"), ("KH", "B")):
    x = hp.paired(R[a], R[b], rb)
    print(f"  {a:>2} vs {b:<2}: {a} only {x[0]}, {b} only {x[1]}, p({a} >) {x[2]:.4g}, n {x[3]}")
print(f"  two-sided p for 3/0: {min(1, 2 * hp.paired(R['KH'], R['RP'], rb)[2]):.3g}")

H("2. G's S90 report() (explore_g_s90.py at b57c1867) on the store at cbca3e36; compared with s90.log")
out = run_report(S90)
compare(out, show(TIP, "explore_out/G/s90.log"), "| arm | n |")
summ = json.loads(SINK["/nonexistent/G/s90_summary.json"].getvalue())
print(f"  summary == committed s90_summary.json: {summ == json.loads(show(TIP, 'explore_out/G/s90_summary.json'))}")
rd = summ["readings"]
for k in ("first", "2b", "question", "FOR THE SPEC"):
    print(f"  {k}: {rd.get(k)}")
log90 = show(TIP, "explore_out/G/s90.log")
print(f"  'FOR THE SPEC' lines in the log: {sum('FOR THE SPEC' in l for l in log90.splitlines())}")
for c in revs:
    try:
        lg = show(c, "explore_out/G/s90.log")
    except subprocess.CalledProcessError:
        continue
    if "FOR THE SPEC" in lg:
        print(f"  first commit whose s90.log has 'FOR THE SPEC': {c[:8]} {git('log', '-1', '--format=%ci', c).strip()} "
              f"(records then {len(json.loads(show(c, 'explore_out/G/s90_results.json'))['runs'])})")
        break


def spec_fixed(rd_, nR):
    """V's fixes 1 and 2 (screen_review_s90_G.md)."""
    q = rd_["question"]
    if nR < 18:
        q = Q[3]
    if q == Q[0]:
        return "the spec's label-free arm is the repeat hinge"
    if q == Q[3]:
        return "UNTESTED: no spec decision (S90 could not be asked); the spec waits"
    return "the repeat hinge falls short: the spec carries the key hinge only"


print(f"  committed rule (s90:99-101) on the records: {S90.spec_decision(rd)}")
print(f"  V's fixed rule (fixes 1 + 2) on the records: {spec_fixed(rd, len(R['R']))}; REG3 completed {len(R['R'])}/20")

H("2b. S90 descriptive numbers in report 10 (from the log lines regenerated above)")
for l in out:
    if any(s in l for s in ("commitment (spread) by step", "write rate", "contingency L1", "plateau", "firings median")):
        print("  " + l.strip()[:230])
for k in ("RP", "R"):
    rr = R[k]
    ws = [r for s, r in rr.items() if not rb(r)]
    print(f"  {S90.LABEL[k]} unbound end margins {sorted(round(r['end']['margin'], 2) for r in ws)}; "
          f"eta key@KEY {sorted(round(r['end'].get('eta_key_by_key', r['end'].get('eta_key_by_key_id', -1)), 2) for r in ws) if ws else []}")
print("  end keys available:", sorted(next(iter(R['RP'].values()))['end'].keys())[:30])

# ═══ 3. S91 ═════════════════════════════════════════════════════════════════════
H("3. S91 records at cbca3e36: budget, validity, counts")
S91J = STORES["s91"]
r91 = S91J["runs"]
print(f"  n {len(r91)}; meta sha {S91J['meta']['sha'][:12]}; sha_rows {len(S91J['meta']['sha_rows'])}; provenance {S91J['meta']['provenance']}")
print(f"  key SHAs {collections.Counter(k.split('|')[2] for k in r91)}; ok {sum(r['ok'] for r in r91.values())}; "
      f"attempts {collections.Counter(r['attempt'] for r in r91.values())}")
print(f"  modules in sha_rows: s91 parent {any('explore_g_s91.py' in str(x) for x in S91J['meta']['sha_rows'])}, "
      f"s91 child {any('explore_g_s91_child' in str(x) for x in S91J['meta']['sha_rows'])}, "
      f"explore_g_common {any('explore_g_common' in str(x) for x in S91J['meta']['sha_rows'])}")
print("  BUDGET per arm: meta iters | record iters | max stopped_at | unbound runs' stopped_at | last curve step")
byarm = collections.defaultdict(dict)
for k, r in r91.items():
    byarm[k.split("|")[0]][int(k.split("|")[1])] = r
for a, rr in sorted(byarm.items()):
    print(f"    {a:<30} {S91J['meta']['arms'][a]['iters']} | {collections.Counter(r['iters'] for r in rr.values())} | "
          f"{max(r['stopped_at'] for r in rr.values())} | {collections.Counter(r['stopped_at'] for r in rr.values() if r['transition'] is None)} | "
          f"{collections.Counter(r['curve'][-1][0] for r in rr.values())}")
print(f"  learned Hebbian arms in store: {[a for a in byarm if a.endswith('_H') and 'PERFECT' not in a]}; in meta arms: "
      f"{[a for a in S91J['meta']['arms'] if a.endswith('_H') and 'PERFECT' not in a]}")
print(f"  validity file: {VALIDJ['s91_validity.json']}")
for s in (2960, 2961):
    r = byarm["allpool.PERFECT_H"][s]
    print(f"  PERFECT_H|{s}: transition {r['transition']} stopped_at {r['stopped_at']} iters {r['iters']} bound {hp.bound(r)}; "
          f"last 10 evaluations {[(c[0], round(c[1], 4)) for c in r['curve'][-10:]]}")
    print(f"     max acc {max(c[1] for c in r['curve']):.4f}; evaluations >= 0.95: {[c[0] for c in r['curve'] if c[1] >= 0.95]}; "
          f"class {ec.fail_class(S89H.inv_view(r)) if r['transition'] is None else '-'}")
for s in (2960, 2961):
    r = byarm["allpool.ORACLE_D"][s]
    print(f"  ORACLE_D|{s}: transition {r['transition']} iters {r['iters']}")
# S89H's and F's Hebbian + conv ALLPOOL perfect gates
for k, r in STORES["s89h"]["runs"].items():
    if "PERFECT" in k:
        print(f"  S89H {k.split('|')[0]}|{k.split('|')[1]}: transition {r['transition']} iters {r['iters']} stopped {r['stopped_at']}")
try:
    F = json.loads(show("129215e", "explore_out/F/s89_post_results.json"))
    for k, r in F["runs"].items():
        if k.startswith("O_HEBB_CONV_ALLPOOL"):
            print(f"  F 129215e {k.split('|')[0]}|{k.split('|')[1]}: transition {r.get('transition')} iters {r.get('iters')} stopped {r.get('stopped_at')}")
except subprocess.CalledProcessError:
    print("  F's store 129215e not readable here")

H("3b. G's S91 report() (explore_g_s91.py at 60b1932a) on the store at cbca3e36; compared with s91.log")


def _valid_swap(m):
    def _valid():
        v = VALIDJ["s91_validity.json"]
        return v if v and v.get("sha") == m._sha() else None
    return _valid


S89H91 = load_mod(S91C, "explore_g_s89h.py", "explore_g_s89h")
print(f"  explore_g_s89h blob at {S90C} / {S91C}: {blob(S90C, 'explore_g_s89h.py')} / {blob(S91C, 'explore_g_s89h.py')}")
S91 = load_mod(S91C, "explore_g_s91.py", "explore_g_s91", swap={"_valid": _valid_swap})
S91.ARMS = S91._arms()
print(f"  ARMS rebuilt with G's _arms() and the committed validity: {sorted(S91.ARMS)}")
out91 = run_report(S91)
compare(out91, show(TIP, "explore_out/G/s91.log"), "| arm | n |")
sm91 = json.loads(SINK["/nonexistent/G/s91_summary.json"].getvalue())
print(f"  summary == committed s91_summary.json: {sm91 == json.loads(show(TIP, 'explore_out/G/s91_summary.json'))}")
print(f"  H readings: {sm91['readings']['H']}")
print(f"  D readings: {sm91['readings']['D']['READINGS']}")

H("3c. S91 delta numbers in report 10")
rbi = S91.rb
D = {a.split(".")[1][:-2]: rr for a, rr in byarm.items() if a.endswith("_D") and "ORACLE" not in a}
for a, rr in D.items():
    fr = sorted(round(r["end"]["seq_routed_frac"], 4) for r in rr.values())
    tm = sorted(round(r["type_map"], 3) for r in rr.values())
    bnd = {s: r for s, r in rr.items() if hp.bound(r)}
    print(f"  {a:<18} ROUTED-BOUND(within) {sum(rbi(r) for r in rr.values())}/10; BOUND {len(bnd)}; "
          f"classes(inv_view, unbound) {collections.Counter(ec.fail_class(S89H.inv_view(r)) for r in rr.values() if r['transition'] is None)}")
    print(f"      seq_routed_frac {fr}; >0.533: {sum(f > 64 / 120 for f in fr)}; in 256ths: {[round(f * 256) for f in fr if f > 0.5]}")
    print(f"      type_map {tm}; > 0.5: {sum(t > 0.5 for t in tm)}")
    for s, r in sorted(bnd.items()):
        e = S89H.inv_view(r)["end"]
        print(f"      bound {s}: transition {r['transition']}, end margin {r['end']['margin']:.3f}, eta stream inv {r['end'].get('eta_key_by_stream_inv')}, "
              f"class(as unbound) {ec.fail_class(dict(S89H.inv_view(r), transition=None))}")
    if a != "REG3_REPEATHINGE":
        b, c, p, n = hp.paired(rr, D["REG3_REPEATHINGE"], rbi)
        print(f"      vs control: {b}/{c}, p {p:.3g}, n {n}")
print("  type-level map counts control vs MATCHGATE (2/10 vs 0/10): one-sided Fisher p (control >) "
      f"{sum(math.comb(10, k) * math.comb(10, 2 - k) for k in range(2, 3)) / math.comb(20, 2):.3f}")
print(f"  S91 CPU-hours (secs_wall) {sum(r['secs_wall'] for r in r91.values()) / 3600:.2f}; log 'runs done' "
      f"{[l.strip() for l in show(TIP, 'explore_out/G/s91.log').splitlines() if 'runs done' in l]}")

H("4. CHECK files")
for f in ("explore_out/G/checks_s90_20261022_050720.json", "explore_out/G/checks_s91_20261022_061400.json"):
    d = json.loads(show(TIP, f))
    rows = d["rows"]
    print(f"  {f.split('/')[-1]}: rows {len(rows)}, ok {sum(1 for _, o in rows if o)}, git {d['provenance'].get('git')}, "
          f"secs {d.get('secs', 0):.0f}")
    print(f"    rows mentioning iters/budget/48000: {[n[:120] for n, _ in rows if any(w in n.lower() for w in ('iters', 'budget', '48000', '48 000'))]}")
    for n, o in rows:
        if any(w in n for w in ("gradient", "seed", "2960", "bit for bit through", "type_map", "ROUTED*")):
            print(f"      {o} {n[:220]}")

H("5. seeds 2960-2991 in other stores (branch heads now; descriptive)")
import re  # noqa: E402
pat = re.compile(r"\|29(6\d|7\d|8\d|9[01])\|")
heads = [h for h in git("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin").split() if h != "origin/HEAD"]
hits = {}
for h in heads:
    for f in git("ls-tree", "-r", "--name-only", h).split():
        if f.endswith("results.json") and not f.endswith("G/s91_results.json"):
            try:
                ks = json.loads(show(h, f)).get("runs", {})
            except Exception:                                        # noqa: BLE001
                continue
            m = [k for k in (ks if isinstance(ks, dict) else []) if pat.search("|" + k.split("|", 1)[-1] if "|" in k else k)]
            if m:
                hits[(h, f)] = len(m)
print(f"  {len(heads)} remote heads; stores holding a 2960-2991 seed (other than G's s91): {hits or 'none'}")
