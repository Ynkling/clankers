"""check_G_report11.py — Referee V's check of Session G's report 11 (S91b) at claude/explore-G 330290dd.

Reads git objects only (git show / rev-parse / rev-list / log / diff-tree / cat-file / ls-tree / for-each-ref);
writes nothing; prints only. torch is not needed: G's pure parent functions (explore_g_s91b at 9949e495, the version that
ran; explore_g_s89h, explore_g_s89, explore_g_rhl_parent, explore_g_common's store and statistics helpers,
test_router_layout's fail_class) and F's pure functions (explore_f_tasks.pool_outcome, _esym, _partitions,
type_map_cap) are exec'd from blobs with stub modules, as check_G_report10.py does. G's own report() is run on the
committed records with its one file write sent to memory, and its output is compared with the committed log line by
line. rlib (docs/review_2026-10/rlib.py) is the referee's independent transition, fail_class, wilson, fisher1.
"""
import ast
import collections
import hashlib
import io
import json
import math
import os
import re
import statistics
import subprocess
import sys
import types

ROOT = "/home/user/clankers"
sys.path.insert(0, os.path.join(ROOT, "docs/review_2026-10"))
import rlib  # noqa: E402

TIP = "330290dd"            # report 11
C, P = "b981c3fa", "9949e495"   # S91b code commit; the parent change
FIX = "4314e35c"            # explore_g_common's budget fix
CPU = "Intel(R) Xeon(R) Processor @ 2.10GHz"
SHA12 = "a0134e14a262"


def git(*a):
    return subprocess.check_output(("git", "-C", ROOT) + a, stderr=subprocess.DEVNULL)


def gs(*a):
    return git(*a).decode()


def show(ref, path):
    return gs("show", f"{ref}:{path}")


def blob(ref, path):
    try:
        return gs("rev-parse", f"{ref}:{path}").strip()[:10]
    except subprocess.CalledProcessError:
        return "-"


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


def fsrc(ref, path, name):
    for node in ast.parse(show(ref, path)).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    return None


# ═══ 0. commits, times, blobs, records per commit ═══════════════════════════
H("0. commits b981c3fa^..330290dd (UTC): files, S91b records per commit, code blobs")
print(f"origin/claude/explore-G now at {gs('rev-parse', 'origin/claude/explore-G').strip()[:8]} (read here: {TIP})")
revs = gs("rev-list", "--reverse", f"{C}^..{TIP}").split()
CODE = ("explore_g_s91b.py", "explore_g_s91b_child.py", "explore_g_common.py", "explore_f_tasks.py", "explore_g_s89.py",
        "explore_g_s89h.py", "explore_g_s89_child.py", "explore_g_s89h_child.py", "explore_g_rhl_parent.py",
        "explore_common16.py", "explore_common.py", "test_router_reliability.py")
blobs = collections.defaultdict(list)
prev_n = None
first_rec = None
for c in revs:
    t = gs("log", "-1", "--format=%ci", c).strip()[:19]
    try:
        txt = show(c, "explore_out/G/s91b_results.json")
        n = txt.count(f"|{SHA12}|Intel")
    except subprocess.CalledProcessError:
        n = None
    files = [f.split("/")[-1] for f in gs("diff-tree", "--no-commit-id", "--name-only", "-r", c).split()]
    subj = gs("log", "-1", "--format=%s", c).strip()
    if n != prev_n or "progress" not in subj:
        print(f"  {c[:8]} {t}  s91b records {n}  files {files}  | {subj[:70]}")
    if n and first_rec is None:
        first_rec = (c[:8], t, n)
    prev_n = n
    for f in CODE:
        b = blob(c, f)
        if not blobs[f] or blobs[f][-1][0] != b:
            blobs[f].append((b, c[:8]))
print(f"  first commit holding S91b records: {first_rec}")
print("  code blobs over the range ((blob, first commit) per change; one entry = unchanged):")
for f, b in blobs.items():
    print(f"    {f:<28} {b}")
print(f"  .py changed {C}..{TIP}: {[f for f in gs('diff', '--name-only', C, TIP).split() if f.endswith('.py')]}")
print(f"  .py changed {P}..{TIP}: {[f for f in gs('diff', '--name-only', P, TIP).split() if f.endswith('.py')]}")
print(f"  files changed in {P}: {gs('diff-tree', '--no-commit-id', '--name-only', '-r', P).split()}")
print("  explore_g_s91b.py functions at b981c3fa vs 9949e495 (ast source equal?):")
for fn in ("rbk", "type_routed", "_sha", "_valid", "_arms", "counterpart", "describe", "readings", "check_readings",
           "report", "run_checks", "main"):
    print(f"    {fn:<15} {fsrc(C, 'explore_g_s91b.py', fn) == fsrc(P, 'explore_g_s91b.py', fn)}")
d = gs("diff", C, P, "--", "explore_g_s91b.py")
print("  diff b981c3fa..9949e495 explore_g_s91b.py (+/- lines):")
for l in d.splitlines():
    if l.startswith(("+", "-")) and not l.startswith(("+++", "---")):
        print("    " + l[:160])
print(f"  explore_g_common.py: last change {gs('log', '-1', '--format=%h %ci', C, '--', 'explore_g_common.py').strip()}; "
      f"blob at {C} {blob(C, 'explore_g_common.py')}")
ln = show(C, "explore_g_common.py").splitlines()
for i in range(150, 175):
    if "iters" in ln[i]:
        print(f"    explore_g_common.py:{i + 1}: {ln[i].strip()[:150]}")
ch = show(C, "explore_g_s91b_child.py").splitlines()
print(f"  child l.158: {ch[157].strip()[:150]}")
rcsrc = show(C, "explore_g_rhl_child.py")
print(f"  explore_g_rhl_child ITERS: {[l.strip() for l in rcsrc.splitlines() if l.startswith('ITERS')]}")
print("  explore_g_s91b.py:110,117 (the arms' iters):")
pl = show(C, "explore_g_s91b.py").splitlines()
for i in (109, 116):
    print(f"    {i + 1}: {pl[i].strip()}")

H("0b. the vendored module: G's explore_f_tasks.py vs F's 9017b82; the b0c6332 (e24d5e1) prefix")
gb = gs("rev-parse", f"{C}:explore_f_tasks.py").strip()
fb = gs("rev-parse", "9017b823:explore_f_tasks.py").strip()
pre = git("cat-file", "blob", "e24d5e10815eb1234d1947aa6b74b128d2696882")
new = git("cat-file", "blob", gb)
print(f"  G at {C}: {gb}; F at 9017b823: {fb}; equal {gb == fb}; G's parent {FIX}: {blob(FIX, 'explore_f_tasks.py')}")
print(f"  e24d5e1 bytes {len(pre)}; G's file bytes {len(new)}; startswith {new.startswith(pre)}; child constants "
      f"{[l.strip()[:120] for l in ch if l.startswith('F_BLOB')]}")
print(f"  G's file blob at every commit of the range: {sorted(set(blob(c, 'explore_f_tasks.py') for c in revs))}")

H("0c. the code SHA: recompute sha1(json.dumps(sha_rows)) and each row's source hash at 9949e495")
S = json.loads(show(TIP, "explore_out/G/s91b_results.json"))
meta = S["meta"]
rows = meta["sha_rows"]
print(f"  meta sha {meta['sha']}; recomputed {hashlib.sha1(json.dumps(rows).encode()).hexdigest()}; rows {len(rows)}")
names = [r[0] for r in rows]
print(f"  modules: {names}")
print(f"  parent explore_g_s91b in rows: {'explore_g_s91b' in names}; child: {'explore_g_s91b_child' in names}; "
      f"explore_g_common: {'explore_g_common' in names}; explore_f_tasks: {'explore_f_tasks' in names}")
bad = []
for nm, h, origin in rows:
    path = nm.replace(".", "/") + ".py"
    try:
        src = git("show", f"{P}:{path}")
    except subprocess.CalledProcessError:
        bad.append((nm, "not in git"))
        continue
    if hashlib.sha1(src).hexdigest() != h and gs("rev-parse", f"{P}:{path}").strip() != h:
        bad.append((nm, "differs"))
print(f"  rows whose hash != sha1(source at {P}) or its git blob: {bad or 'none'}")
print(f"  validity file sha: {json.loads(show(TIP, 'explore_out/G/s91b_validity.json'))['sha']}")
print(f"  meta provenance: {meta['provenance']}")

# ═══ stubs ════════════════════════════════════════════════════════════════════
STORES = {n: json.loads(show(TIP, f"explore_out/G/{n}_results.json")) for n in ("s91b", "s89h", "s89")}
VALIDJ = {"s91b_validity.json": json.loads(show(TIP, "explore_out/G/s91b_validity.json")),
          "s89_validity.json": json.loads(show(TIP, "explore_out/G/s89_validity.json")),
          "s89h_validity.json": json.loads(show(TIP, "explore_out/G/s89h_validity.json"))}
ec = types.ModuleType("explore_common")
exec(defs(show(C, "test_router_layout.py"), {"fail_class", "MARGIN_PARTIAL", "ETA_SPLIT"}), ec.__dict__)
ec.cpu_model = lambda: CPU
ec.OUT_DIR = "/nonexistent"
gc = types.ModuleType("explore_g_common")
gc.math = math
exec(defs(show(C, "explore_g_common.py"), {"wilson", "band", "cnt", "mcnemar_greater", "fisher_greater", "key", "runs",
                                           "iters_check"}), gc.__dict__)
gc.G_DIR = "/nonexistent/G"
gc.WORKERS = 4
gc.load = lambda name: STORES[name]
gc.ec = ec
for _nm, _path, _fn in (("test_router_confirm", "test_router_confirm.py", "fisher_greater"),
                        ("test_conv_lr", "test_conv_lr.py", "mcnemar_greater")):
    _m = types.ModuleType(_nm)
    _m.math = math
    exec("import math\n" + defs(show(C, _path), {_fn}), _m.__dict__)
    sys.modules[_nm] = _m
sys.modules["explore_common"] = ec
sys.modules["explore_g_common"] = gc
hp = types.ModuleType("explore_g_rhl_parent")
hp.__dict__["__file__"] = "/nonexistent/explore_g_rhl_parent.py"
exec(compile(show(C, "explore_g_rhl_parent.py"), "explore_g_rhl_parent.py", "exec"), hp.__dict__)
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


S89 = load_mod(C, "explore_g_s89.py", "explore_g_s89")
S89H = load_mod(C, "explore_g_s89h.py", "explore_g_s89h")


def _valid_swap(m):
    def _valid():
        v = VALIDJ["s91b_validity.json"]
        return v if v and v.get("sha") == m._sha() else None
    return _valid


S91B = load_mod(P, "explore_g_s91b.py", "explore_g_s91b", swap={"_valid": _valid_swap})
print(f"\n  S91b ARMS rebuilt with G's _arms() and the committed validity: {len(S91B.ARMS)} arms")
FT = types.ModuleType("ftasks")
exec("import math\n" + defs(show(C, "explore_f_tasks.py"), {"_esym", "_partitions", "type_map_cap", "pool_outcome",
                                                             "ROUTED_MARGIN", "ROUTED_ETA"}), FT.__dict__)
_trr = types.ModuleType("test_router_reliability")
_trr.DISC_COS = 0.5
sys.modules["test_router_reliability"] = _trr

# ═══ 1. records ════════════════════════════════════════════════════════════════
H("1. S91b records at 330290dd: n, ok, attempts, budget, transitions")
runs = S["runs"]
print(f"  n {len(runs)}; key SHAs {collections.Counter(k.split('|')[2] for k in runs)}; CPUs {collections.Counter(k.split('|')[3] for k in runs)}")
print(f"  ok {sum(bool(r.get('ok')) for r in runs.values())}; attempts {collections.Counter(r.get('attempt') for r in runs.values())}")
print(f"  iters {collections.Counter(r.get('iters') for r in runs.values())}; meta arms iters "
      f"{collections.Counter(a['iters'] for a in meta['arms'].values())}; arms {len(meta['arms'])}")
print(f"  record iters == meta ARMS iters: {sum(r['iters'] == meta['arms'][k.split('|')[0]]['iters'] for k, r in runs.items())}/{len(runs)}")
unb = [r for r in runs.values() if r["transition"] is None]
print(f"  unbound runs {len(unb)}: stopped_at {collections.Counter(r['stopped_at'] for r in unb)}; last curve step "
      f"{collections.Counter(r['curve'][-1][0] for r in unb)}")
print(f"  record transition == rlib.transition(curve): {sum(r['transition'] == rlib.transition(r['curve']) for r in runs.values())}/{len(runs)}")
print(f"  every record 'k' {collections.Counter(r.get('k') for r in runs.values())}; tasks {collections.Counter(r.get('task') for r in runs.values())}; "
      f"recipe {collections.Counter(r.get('recipe') for r in runs.values())}")
seeds_by = collections.defaultdict(set)
for k in runs:
    seeds_by[k.split("|")[0]].add(int(k.split("|")[1]))
print(f"  seeds: gates {sorted(set().union(*[v for a, v in seeds_by.items() if 'PIDX' in a or 'PTYPE' in a]))}; learned "
      f"{sorted(set().union(*[v for a, v in seeds_by.items() if 'REG3' in a]))}")
print(f"  CPU-hours (secs_wall) {sum(r['secs_wall'] for r in runs.values()) / 3600:.2f}")

# ═══ 2. counts ═════════════════════════════════════════════════════════════════
H("2. counts per task x memory x arm (k-general ROUTED_k = the reading's rbk; k = 2 forms beside it)")
TASKS, MEMS, GATES, LEARNED = S91B.TASKS, S91B.MEMS, S91B.GATES, S91B.LEARNED
R = {a: gc.runs(S91B, a) for a in S91B.ARMS}
rbk, tr_ = S91B.rbk, S91B.type_routed


def med(x):
    return statistics.median(x) if x else float("nan")


def cls(rr, f):
    return dict(collections.Counter(f(r) for r in rr.values() if r["transition"] is None))


for m in MEMS:
    for t in TASKS:
        for a in GATES + LEARNED:
            rr = R[f"{t}.{a}_{m}"]
            n = len(rr)
            nb = sum(hp.bound(r) for r in rr.values())
            k = sum(rbk(r) for r in rr.values())
            tr = sorted(r["transition"] for r in rr.values() if r["transition"] is not None)
            inv = sum(hp.routed_bound(S89H.inv_view(r)) for r in rr.values())
            pooled = sum(hp.routed_bound(r) for r in rr.values())
            fpo_i = sum(FT.pool_outcome(r, True)["routed_bound"] for r in rr.values())
            fpo_p = sum(FT.pool_outcome(r, False)["routed_bound"] for r in rr.values())
            rl = sum(rlib.bound(r) for r in rr.values())
            e = [r["end"] for r in rr.values()]
            print(f"  {t}.{a}_{m}: n {n} BOUND {nb} (rlib {rl}) ROUTED_k {k}/{n} W{tuple(round(x, 2) for x in rlib.wilson(k, n))} "
                  f"TYPE-ROUTED {sum(tr_(r) for r in rr.values())} | k=2 within (inv_view) {inv}, pooled {pooled}, "
                  f"F pool_outcome inv/pooled {fpo_i}/{fpo_p} | trans {tr}")
            print(f"      type_map_k med {med([r['type_map_k'] for r in rr.values()]):.3f}; eff_ch med {med([x['eff_ch'] for x in e]):.2f} "
                  f"({min(x['eff_ch'] for x in e):.1f}-{max(x['eff_ch'] for x in e):.1f}); sep med {med([x['seq_separated_frac'] for x in e]):.3f} "
                  f"max {max(x['seq_separated_frac'] for x in e):.3f}; margin med {med([x['margin'] for x in e]):.3f} max "
                  f"{max(x['margin'] for x in e):.3f}; acc med {med([r['acc'] for r in rr.values()]):.3f}; "
                  f"sep_key_inv med {med([x['sep_key_inv'] for x in e]):.3f}; val_cos_inv_k med {med([x['val_cos_inv_k'] for x in e]):.3f}")
            if a in LEARNED:
                print(f"      unbound classes: rec['fail'] {cls(rr, lambda r: r.get('fail'))}; fail_class(pooled) "
                      f"{cls(rr, ec.fail_class)}; hp.outcome {cls(rr, hp.outcome)}; fail_class(inv_view) "
                      f"{cls(rr, lambda r: ec.fail_class(S89H.inv_view(r)))}; hp.outcome(inv_view) {cls(rr, lambda r: hp.outcome(S89H.inv_view(r)))}; "
                      f"rlib.fail_class {cls(rr, rlib.fail_class)}")
print("  k = 2 counterparts (S89H / S89 allpool, 2842-2861, within-sequence):",
      {(a, m_): S91B.counterpart(a, m_) for a in LEARNED for m_ in MEMS})
print(f"  Fisher one-sided p(0/10 vs 0/20): G {gc.fisher_greater(0, 10, 0, 20)}, rlib {rlib.fisher1(0, 10, 0, 20)}; "
      f"(7/10 vs 0/20) rlib {rlib.fisher1(7, 10, 0, 20):.3g}")
print("  -ctxpool vs -allpool paired (ROUTED_k), every learned arm and memory:",
      {(a, m_): hp.paired(R[f"ctxpool.{a}_{m_}"], R[f"allpool.{a}_{m_}"], rbk)[:2] for a in LEARNED for m_ in MEMS})

H("2b. the perfect gates: PIDX vs PTYPE seed for seed; eff_ch; type maps")
for m in MEMS:
    for t in TASKS:
        for s in S91B.S2:
            a, b = R[f"{t}.PIDX_{m}"][s], R[f"{t}.PTYPE_{m}"][s]
            print(f"  {t} {m} {s}: trans {a['transition']}/{b['transition']} acc {a['acc']:.4f}/{b['acc']:.4f} curve equal "
                  f"{a['curve'] == b['curve']}; margin {a['end']['margin']:.3f}/{b['end']['margin']:.3f} sep "
                  f"{a['end']['seq_separated_frac']:.3f}/{b['end']['seq_separated_frac']:.3f} type_map {a['type_map_k']:.3f}/"
                  f"{b['type_map_k']:.3f} eff_ch {a['end']['eff_ch']:.2f}/{b['end']['eff_ch']:.2f}")

# ═══ 3. G's report() ═══════════════════════════════════════════════════════════


def run_report(mod):
    buf, old = io.StringIO(), sys.stdout
    try:
        sys.stdout = buf
        mod.report()
    finally:
        sys.stdout = old
    return buf.getvalue().splitlines()


H("3. G's report() (explore_g_s91b.py at 9949e495) on the store at 330290dd; compared with s91b.log")
out = run_report(S91B)
LOG = show(TIP, "explore_out/G/s91b.log")
L = LOG.splitlines()
i0 = [k for k, l in enumerate(L) if l.startswith("| arm | n |")]
print(f"  count-table headers in the log at lines {[k + 1 for k in i0]}")
tail = [l for l in L[i0[-1]:] if not l.startswith("SCREEN_DONE")]
dd = [(k, a, b) for k, (a, b) in enumerate(zip(out, tail)) if a != b]
print(f"  regenerated {len(out)} lines; log from the count table {len(tail)} lines (SCREEN_DONE* dropped); differing {len(dd)}; "
      f"length equal {len(out) == len(tail)}")
for k, a, b in dd[:5]:
    print(f"    line {k}:\n      mine {a[:200]}\n      log  {b[:200]}")
summ = json.loads(SINK["/nonexistent/G/s91b_summary.json"].getvalue())
print(f"  summary == committed s91b_summary.json: {summ == json.loads(show(TIP, 'explore_out/G/s91b_summary.json'))}")
for m in MEMS:
    print(f"  {m} READING: {summ['readings'][m]['READING']}")

# ═══ 4. the rule ═══════════════════════════════════════════════════════════════
H("4. 'the channels do not suffice': the committed rule (b981c3fa) and an independent recomputation")
for i in (35, 36, 37, 38, 39, 40, 41, 42, 43):
    print(f"  s91b:{i + 1}: {pl[i]}")
for i in range(139, 171):
    if any(w in pl[i] for w in ("nb =", "routes", "do not suffice", "k <= 3", "disc =", "if nb", "missing")):
        print(f"  s91b:{i + 1}: {pl[i]}")
for m in MEMS:
    Rm = {a.rsplit("_", 1)[0]: rr for a, rr in R.items() if a.endswith(f"_{m}")}
    nb = (sum(rlib.bound(r) for r in Rm["ctxpool.PIDX"].values()), sum(rlib.bound(r) for r in Rm["ctxpool.PTYPE"].values()))
    ks = {a: sum(1 for r in Rm[f"ctxpool.{a}"].values() if rlib.bound(r) and r["end"]["margin"] >= 0.9
                 and r["end"]["seq_separated_frac"] > 0.9) for a in LEARNED}
    routes = any(k >= 7 and rlib.fisher1(k, 10, 0, 20) < 0.05 for k in ks.values())
    mine = ("the gate routes by type when it has the channels" if routes else
            "the channels do not suffice" if all(k <= 3 for k in ks.values()) and nb == (2, 2) else "no reading applies")
    print(f"  {m}: perfect gates BOUND (rlib) PIDX/PTYPE {nb}; ROUTED_k {ks}; independent: {mine}; G's readings(): "
          f"{S91B.readings(Rm, m)['READING']}")

# ═══ 5. the key hinge ══════════════════════════════════════════════════════════
H("5. the learned gates per run: type_map_k, eff_ch, seq_separated_frac, margin (closest to ROUTED_k)")
F = ("type_map_k", "eff_ch", "seq_separated_frac", "seq_routed_frac", "margin", "sep_key_inv", "val_cos_inv_k",
     "eta_key_by_stream_inv", "eta_key_by_key")
for arm in ("ctxpool.REG3_KEYHINGE_H", "ctxpool.REG3_REPEATHINGE_H", "allpool.REG3_KEYHINGE_H", "ctxpool.REG3_KEYHINGE_D",
            "ctxpool.REG3_H"):
    print(f"  {arm}:  seed | " + " ".join(f"{f[:10]:>10}" for f in F) + " |  acc  trans  class(pooled/inv)")
    for s, r in sorted(R[arm].items()):
        e = r["end"]
        v = [r["type_map_k"]] + [e[f] for f in F[1:]]
        print(f"      {s} | " + " ".join(f"{x:>10.3f}" for x in v) + f" | {r['acc']:.3f} {str(r['transition']):>5} "
              f"{ec.fail_class(r) if r['transition'] is None else '-'}/{ec.fail_class(S89H.inv_view(r)) if r['transition'] is None else '-'}")
lr = [(k.split("|")[0], int(k.split("|")[1]), r) for k, r in runs.items() if "REG3" in k]
close = sorted(lr, key=lambda x: -min(x[2]["end"]["margin"] / 0.9, x[2]["end"]["seq_separated_frac"] / 0.9))[:8]
print("  closest learned runs to ROUTED_k (by min(margin/0.9, separated/0.9)):")
for a, s, r in close:
    print(f"    {a}|{s}: margin {r['end']['margin']:.3f} separated {r['end']['seq_separated_frac']:.3f} type_map_k "
          f"{r['type_map_k']:.3f} eff_ch {r['end']['eff_ch']:.2f} bound {r['transition']}")
print(f"  learned runs with margin >= 0.9: {sum(r['end']['margin'] >= 0.9 for _, _, r in lr)}; with separated > 0.9: "
      f"{sum(r['end']['seq_separated_frac'] > 0.9 for _, _, r in lr)}; with type_map_k > 0.9: "
      f"{sum(r['type_map_k'] > 0.9 for _, _, r in lr)} of {len(lr)}")
print("  type_map_k > 0.9 by arm:", dict(collections.Counter(a for a, s, r in lr if r["type_map_k"] > 0.9)))
print("  the cap of a fixed type -> channel map using m channels (F's type_map_cap(16, 2, m)):")
print("    " + ", ".join(f"m={m_}: {FT.type_map_cap(16, 2, m_)[0]:.3f}" for m_ in (2, 4, 5, 6, 7, 8, 10, 12, 16)))
print("  KH_H on -ctxpool, the course of the evaluations (median over 10 runs) of separated, margin, eff_ch, acc:")
kh = R["ctxpool.REG3_KEYHINGE_H"]
steps = sorted({st_["step"] for r in kh.values() for st_ in r["stats"] if isinstance(st_.get("step"), int)})
for st in steps[-8:]:
    vals = [next((x for x in r["stats"] if x.get("step") == st), None) for r in kh.values()]
    vals = [v for v in vals if v]
    acc = [next((c[1] for c in r["curve"] if c[0] == st), None) for r in kh.values()]
    print(f"    {st}: separated {med([v['seq_separated_frac'] for v in vals]):.3f} (max {max(v['seq_separated_frac'] for v in vals):.3f}) "
          f"eff_ch {med([v['eff_ch'] for v in vals]):.2f} acc {med([a_ for a_ in acc if a_ is not None]):.3f} (n {len(vals)}; the "
          f"per-evaluation stats carry no margin)")
for arm in ("ctxpool.REG3_KEYHINGE_H", "allpool.REG3_KEYHINGE_H", "ctxpool.REG3_REPEATHINGE_H"):
    rr = R[arm]
    d_sep = [(s, round(next(x for x in r["stats"] if x.get("step") == 18000)["seq_separated_frac"], 3),
              round(r["end"]["seq_separated_frac"], 3), round(r["curve"][-1][1], 3)) for s, r in sorted(rr.items())
             if r["transition"] is None]
    print(f"  {arm} unbound runs (seed, separated at 18000, at the end, last acc): {d_sep}")

# ═══ 6. power ══════════════════════════════════════════════════════════════════
H("6. what 0/10 excludes; the rule's operating characteristic")
print(f"  0/10: Clopper-Pearson one-sided 95% upper {1 - 0.05 ** 0.1:.4f}; two-sided 95% upper {1 - 0.025 ** 0.1:.4f}; "
      f"Wilson {tuple(round(x, 4) for x in rlib.wilson(0, 10))}")
for p in (0.05, 0.1, 0.2, 0.26, 0.3, 0.4, 0.5, 0.7):
    le3 = sum(math.comb(10, i) * p ** i * (1 - p) ** (10 - i) for i in range(4))
    print(f"  true rate {p}: P(0/10) {(1 - p) ** 10:.3f}; P(<= 3/10) {le3:.3f}")
print(f"  pooled 0/30 per memory on -ctxpool (three arms, not one arm): one-sided 95% upper {1 - 0.05 ** (1 / 30):.4f}; "
      f"0/60 over both memories {1 - 0.05 ** (1 / 60):.4f}")

# ═══ 7. CHECK files ════════════════════════════════════════════════════════════
H("7. CHECK files")
for f in sorted(x for x in gs("ls-tree", "--name-only", TIP, "explore_out/G/").split() if "checks_s91b" in x):
    dch = json.loads(show(TIP, f))
    rw = dch["rows"]
    print(f"  {f.split('/')[-1]}: rows {len(rw)}, ok {sum(1 for _, o in rw if o)}, git {dch['provenance'].get('git')}, "
          f"started {dch['provenance'].get('started')}, secs {dch.get('secs', 0):.0f}")
    print(f"    failing rows: {[n[:150] for n, o in rw if not o]}")
    print(f"    rows on iters/budget: {[n[:100] for n, o in rw if 'iters' in n.lower()]}")
    if f.endswith("224522.json"):
        for n, o in rw:
            if "untrained" in n or "first 49923" in n or "cap at k = 16" in n:
                print(f"      {o} {n[:200]}")
print(f"  log lines 'CHECK: each queued payload': {[l.strip()[:60] for l in L if 'each queued payload' in l]}")
print(f"  log banners: {[l.strip() for l in L if '| git ' in l]}")
print(f"  log markers: {[l.strip() for l in L if l.startswith('===') and 'RESTART' in l or 'relaunch' in l]}")

# ═══ 8. seeds ═════════════════════════════════════════════════════════════════
H("8. seeds 2992-3019 in other stores (branch heads now; descriptive)")
pat = re.compile(r"\|(299[2-9]|300\d|301\d)\|")
heads = [h for h in gs("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin").split() if h != "origin/HEAD"]
hits = {}
for h in heads:
    for f in gs("ls-tree", "-r", "--name-only", h).split():
        if f.endswith("results.json") and not f.endswith("G/s91b_results.json"):
            try:
                ks = json.loads(show(h, f)).get("runs", {})
            except Exception:                                        # noqa: BLE001
                continue
            mm = [k for k in (ks if isinstance(ks, dict) else []) if pat.search("|" + k.split("|", 1)[-1] if "|" in k else k)]
            if mm:
                hits[(h, f)] = len(mm)
print(f"  {len(heads)} remote heads; stores holding a 2992-3019 seed (other than G's s91b): {hits or 'none'}")
