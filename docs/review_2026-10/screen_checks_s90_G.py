"""screen_checks_s90_G.py — Referee V's pre-run screen of Session G's S90 (claude/explore-G b57c1867) and the
progress commits after it.

Reads git objects only (git show / rev-parse / log / ls-tree); writes nothing; prints only (the output is committed
as screen_checks_s90_G_out.txt). torch is not needed: G's pure parent functions (explore_g_s90, explore_g_s89h and
explore_g_s89 at b57c1867, explore_g_rhl_parent, explore_g_common's store helpers, test_conv_lr.mcnemar_greater,
test_router_layout's fail_class constants) are exec'd from blobs statement by statement with stub modules, as
check_G_report9.py does. Synthetic records carry only the fields G's readings read.
"""
import ast
import io
import itertools
import json
import math
import subprocess
import sys
import types

ROOT = "/home/user/clankers"
CODE = "b57c1867"           # the S90 commit
TIP = "afe7a1e6"            # the last progress commit read here
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


# ═══ 0. commits, times, blobs ═══════════════════════════════════════════════
H("0. commits and times (UTC)")
tip_now = git("rev-parse", "origin/claude/explore-G").strip()
print(f"origin/claude/explore-G now at {tip_now[:8]} (read here: {TIP})")
for ref, what in (("d5d48eeb", "S89H readings commit"), ("d60e8668", "S89H last progress (201 records)"),
                  ("996e38b7", "G report 9"), (CODE, "S90 commit"), ("00348e77", "S90 progress 1"),
                  ("5c04a858", "S90 progress 2"), ("82d37ae0", "S90 progress 3"), ("afe7a1e6", "S90 progress 4")):
    print(f"  {git('log', '-1', '--format=%ad', '--date=iso', ref).strip()}  {ref}  {what}")
for br, path, what in (("origin/claude/bdh-growth-hebbian-inference-w90069", "docs/reading/parallel_sessions_prompts.md",
                        "round-7 prompt"),
                       ("origin/claude/review-V", "docs/review_2026-10/check_G_report9.md", "V's check_G_report9")):
    print(f"  {git('log', '-1', '--format=%ad %h', '--date=iso', br, '--', path).strip()}  {what}")
print("\nfiles changed b57c1867..TIP:", git("diff", "--name-only", CODE, TIP).split())
print("\nblob of each module S90 imports (d5d48eeb / b57c1867 / TIP):")
MODS = ("explore_g_s90.py", "explore_g_s90_child.py", "explore_g_s89h.py", "explore_g_s89.py", "explore_g_s89_child.py",
        "explore_g_rhl_parent.py", "explore_g_rhl_child.py", "explore_g_common.py", "explore_g_s87_child.py",
        "explore_g_devices_child.py", "explore_f_tasks.py", "test_router_reliability.py", "test_binding_onset.py")
for m in MODS:
    b = []
    for r in ("d5d48eeb", CODE, TIP):
        try:
            b.append(git("rev-parse", f"{r}:{m}").strip()[:10])
        except subprocess.CalledProcessError:
            b.append("-")
    print(f"  {m:<30} {' / '.join(b)}")

# ═══ stubs, as check_G_report9.py ════════════════════════════════════════════
STORES = {n: json.loads(show(TIP, f"explore_out/G/{n}_results.json")) for n in ("s90", "s87", "s89h", "s89")}
VALID = {"s89_validity.json": json.loads(show(TIP, "explore_out/G/s89_validity.json")),
         "s89h_validity.json": json.loads(show(TIP, "explore_out/G/s89h_validity.json"))}
ec = types.ModuleType("explore_common")
exec(defs(show(CODE, "test_router_layout.py"), {"fail_class", "MARGIN_PARTIAL", "ETA_SPLIT"}), ec.__dict__)
ec.cpu_model = lambda: CPU
gc = types.ModuleType("explore_g_common")
gc.math = math
exec(defs(show(CODE, "explore_g_common.py"), {"wilson", "band", "cnt", "key", "runs"}), gc.__dict__)
_m = {}
exec(defs(show(CODE, "test_conv_lr.py"), {"mcnemar_greater"}), {"math": math}, _m)
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
exec(compile(show(CODE, "explore_g_rhl_parent.py"), "explore_g_rhl_parent.py", "exec"), hp.__dict__)
sys.modules["explore_g_rhl_parent"] = hp
SINK = {}


def fake_open(p, mode="r", *a, **k):
    if "w" in mode:
        SINK[p] = io.StringIO()
        SINK[p].close = lambda: None
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


S89 = load_mod(CODE, "explore_g_s89.py", "explore_g_s89")
S89H = load_mod(CODE, "explore_g_s89h.py", "explore_g_s89h")
S90 = load_mod(CODE, "explore_g_s90.py", "explore_g_s90")
rb = hp.routed_bound
Q = S89H.Q
S20 = S90.S20

# ═══ 1. S87's records that S90 pairs with ════════════════════════════════════
H("1. S87's records (s87_results.json at TIP) read by S90's records(): explore_g_s89.recorded('s87', ...)")
st87 = STORES["s87"]
print(f"S87 store: sha {st87['meta']['sha'][:12]}, provenance {st87['meta']['provenance']}")
REC = {k: S89.recorded("s87", a, S20) for k, a in (("B", "BASELINE"), ("KH", "REG3_KEYHINGE"), ("FQ", "REG3_FREQHINGE"))}
for k, rr in REC.items():
    rbs = [s for s in S20 if s in rr and rb(rr[s])]
    print(f"  {k}: {len(rr)}/20 records; ROUTED-BOUND {len(rbs)}/20; failing seeds {[s for s in S20 if s not in rbs]}; "
          f"iters {sorted({r['iters'] for r in rr.values()})}; max stopped_at {max(r['stopped_at'] for r in rr.values())}; "
          f"recipe {sorted({r['recipe'] for r in rr.values()})}; k {sorted({r.get('k') for r in rr.values()})}")
kh2700 = REC["KH"][2700]
print(f"  KH|2700 (the CHECKed record): transition {kh2700['transition']}, stopped {kh2700['stopped_at']} -> the CHECK "
      f"(through 2400) ends before its transition")
print(f"  FQ|2700: transition {REC['FQ'][2700]['transition']}, stopped {REC['FQ'][2700]['stopped_at']} (unbound; CHECKed "
      f"through 2400 of 24000)")
o = S89.recorded("s87", "ORACLE_ALL", (2700, 2701))
print(f"  ORACLE_ALL: {sum(hp.bound(r) for r in o.values())}/{len(o)} bound; transitions {[r['transition'] for r in o.values()]}")

# ═══ 2. what S90 has run ═════════════════════════════════════════════════════
H("2. what S90 has run (TIP)")
st = STORES["s90"]
print(f"meta: sha {st['meta']['sha'][:12]}, cpu {st['meta']['cpu']}, arms "
      f"{ {k: (v['iters'], len(v['seeds'])) for k, v in st['meta']['arms'].items()} }")
for k, r in st["runs"].items():
    print(f"  {k}: ok {r.get('ok')} attempt {r.get('attempt')} iters {r.get('iters')} stopped {r.get('stopped_at')} "
          f"transition {r.get('transition')} ROUTED-BOUND {rb(r)} curve steps {[c[0] for c in r['curve']]} k {r.get('k')} "
          f"recipe {r.get('recipe')} task {r.get('task')} term fired {r.get('term', {}).get('fired')} weights "
          f"{r.get('term', {}).get('weights')}")
ck = json.loads(show(TIP, "explore_out/G/checks_s90_20261022_050720.json"))
print(f"CHECK file: {len(ck['rows'])} rows, {sum(1 for _, x in ck['rows'] if x)} ok; provenance git {ck['provenance']['git']}, "
      f"started {ck['provenance']['started']}, torch {ck['provenance']['torch']}")
print("CHECK rows mentioning iters / budget / 24000:",
      [n[:80] for n, _ in ck["rows"] if "iters" in n or "budget" in n or "24000" in n] or "none")
# the order run_screen queues the 40 jobs (explore_g_common.py:146-153)
jobs = sorted([(a.get("prio", 1), a["iters"], arm, s) for arm, a in S90.ARMS.items() for s in a["seeds"]],
              key=lambda j: (-j[0], -j[1], j[3]))
print("queue order (first 8):", [(j[2].split('.')[1], j[3]) for j in jobs[:8]])

# ═══ 3. the budget ═══════════════════════════════════════════════════════════
H("3. the budget: does each run get the intended iters?")
src = show(CODE, "explore_g_common.py")
for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "run_child" and node.lineno > 136:
        print(f"  explore_g_common.py:{node.lineno}: {ast.unparse(node)}")
c89 = show(CODE, "explore_g_s89_child.py")
for i, l in enumerate(c89.splitlines(), 1):
    if "rc.ITERS" in l:
        print(f"  explore_g_s89_child.py:{i}: {l.strip()}")
for i, l in enumerate(show(CODE, "explore_g_rhl_child.py").splitlines(), 1):
    if l.startswith("ITERS"):
        print(f"  explore_g_rhl_child.py:{i}: {l.strip()}")
print(f"  S90 intends {sorted({a['iters'] for a in S90.ARMS.values()})} (explore_g_s90.py:85); the child default is "
      f"24000: equal, so the defect is latent here. Records so far: iters {sorted({r['iters'] for r in st['runs'].values()})}")

# ═══ 4. the readings: exhaustive on every outcome ════════════════════════════
H("4. readings(): explore_g_s89h.readings (imported, explore_g_s90.py:109) on every outcome of the two new arms, "
  "with S87's B / KH / FQ records fixed")
GOOD = dict(transition=1200, end=dict(margin=1.0, eta_key_by_stream=1.0), stats=[dict(step=2400, spread=0.5)])
BAD = dict(transition=None, end=dict(margin=0.0, eta_key_by_stream=0.0), stats=[dict(step=2400, spread=0.5)])
fq_fail = [s for s in S20 if not rb(REC["FQ"][s])]
fq_pass = [s for s in S20 if rb(REC["FQ"][s])]


def comps(n, k=4):
    for c in itertools.combinations_with_replacement(range(k), n):
        yield [c.count(i) for i in range(k)]


def assign(seeds, cnt):
    out, i = [], 0
    for st_, n in enumerate(cnt):        # state: bit 1 = REG3, bit 0 = RP
        out += [(seeds[i + j], st_) for j in range(n)]
        i += n
    return out


def run_readings(R_, RP_):
    R = dict(B=REC["B"], R=R_, KH=REC["KH"], FQ=REC["FQ"], RP=RP_)
    rd = S89H.readings(R, S20, "lite", valid=True, label=S90.LABEL, valid_note="")
    return rd, S90.spec_decision(rd)


bymarg = {}
tot = bad = 0
qcount = {}
for ca in comps(len(fq_pass)):
    for cf in comps(len(fq_fail)):
        pairs = assign(fq_pass, ca) + assign(fq_fail, cf)
        Rr = {s: (GOOD if st_ & 2 else BAD) for s, st_ in pairs}
        Rp = {s: (GOOD if st_ & 1 else BAD) for s, st_ in pairs}
        rd, spec = run_readings(Rr, Rp)
        r, c = sum(rb(x) for x in Rr.values()), sum(rb(x) for x in Rp.values())
        q = rd["question"]
        tot += 1
        bad += q not in Q
        bymarg.setdefault((r, c), set()).add((q, spec, rd[S90.LABEL["RP"]].split(" (")[0]))
        qcount[q] = qcount.get(q, 0) + 1
print(f"{tot} paired configurations of (REG3, REPEAT) over the 20 seeds (by counts within S87-FQ-pass / FQ-fail seeds); "
      f"question outside Q: {bad}")
print("question counts:", qcount)
print(f"does the question / spec decision / REPEAT name depend on anything but the marginal counts (r = REG3, c = REPEAT)? "
      f"{'no' if all(len(v) == 1 for v in bymarg.values()) else 'YES'}")
closed = all(next(iter(v))[0] == (Q[0] if c >= 19 and r <= 15 else Q[1] if c >= 19 else Q[2]) for (r, c), v in bymarg.items())
print(f"closed form holds on every (r, c): Q0 iff REPEAT >= 19 and REG3 <= 15; Q1 iff REPEAT >= 19 and REG3 >= 16; else Q2: {closed}")
names_by_c = {}
for (r, c), v in bymarg.items():
    names_by_c.setdefault(c, set()).add(next(iter(v))[2])
print("REPEAT's name by its count c:", {c: sorted(v) for c, v in sorted(names_by_c.items())})
print("spec decision by question:", sorted({(next(iter(v))[0][:20], next(iter(v))[1][:45]) for v in bymarg.values()}))

print("\nmissing-run cases (spec_decision, explore_g_s90.py:99-101):")
allg = {s: GOOD for s in S20}
allb = {s: BAD for s in S20}
cases = [("RP 20/20, REG3 0/20 (complete)", allb, allg),
         ("RP one seed missing (19 ok, all bound)", allb, {s: GOOD for s in S20[1:]}),
         ("RP 20/20, REG3: no completed run", {}, allg),
         ("RP 20/20, REG3: 4 completed, all unbound", {s: BAD for s in S20[:4]}, allg),
         ("RP 20/20, REG3: 5 completed, all unbound", {s: BAD for s in S20[:5]}, allg),
         ("RP 20/20, REG3: 15 completed, 11 bound", {s: (GOOD if i < 11 else BAD) for i, s in enumerate(S20[:15])}, allg)]
for nm, Rr, Rp in cases:
    rd, spec = run_readings(Rr, Rp)
    print(f"  {nm}: first '{rd['first'][:45]}'; 2b '{rd.get('2b', '-')[:60]}'; question '{rd['question'][:40]}'; "
          f"spec '{spec}'")

# ═══ 5. operating characteristics at S87's rates ════════════════════════════
H("5. operating characteristics (exact; KH 20/20 and BASELINE 0/20 fixed as recorded, so every decisive test is marginal)")


def binom_ge(n, p, k):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def P_q0(p_rp, p_r):
    return sum(math.comb(20, c) * p_rp ** c * (1 - p_rp) ** (20 - c) * math.comb(20, r) * p_r ** r * (1 - p_r) ** (20 - r)
               for (r, c), v in bymarg.items() if next(iter(v))[0] == Q[0])


print("P(REPEAT 'replaces') = P(c >= 19):", {p: round(binom_ge(20, p, 19), 4) for p in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7)})
print("P(REPEAT 'is weaker') = P(c in {17, 18}):",
      {p: round(binom_ge(20, p, 17) - binom_ge(20, p, 19), 4) for p in (1.0, 0.95, 0.9, 0.85, 0.8)})
print("P(2b applies) = P(REG3 >= 16):", {p: round(binom_ge(20, p, 16), 4) for p in (0.1, 0.45, 0.6, 0.8, 0.9)})
print("P(spec's label-free arm = the repeat hinge) by (p_REPEAT, p_REG3), from the enumerated readings:")
print("  p_RP \\ p_REG3 " + " ".join(f"{q:>7}" for q in (0.1, 0.45, 0.8)))
for p in (1.0, 0.95, 0.9, 0.85, 0.8):
    print(f"  {p:<14} " + " ".join(f"{P_q0(p, q):7.4f}" for q in (0.1, 0.45, 0.8)))
print("  (S82's lite Hebbian REG3: 2/20 on 2100-2119 -> 0.1; S89H POOL REG3_H 9/20 -> 0.45; S87's FQ on these seeds "
      "17/20 -> 0.85)")
rd87 = run_readings(allb, {s: REC["FQ"][s] for s in S20})[0]
print(f"S87's FQ records read as if they were REPEAT's (with REG3 0/20): name '{rd87[S90.LABEL['RP']].split(' (')[0]}', "
      f"question '{rd87['question']}'")

# ═══ 6. the repeat set on lite against S87's frequency set (from G's CHECK rows) ═══
H("6. the repeat hinge's positions vs S87's frequency hinge on lite (CHECK rows, checks_s90_20261022_050720.json)")
for n, ok in ck["rows"]:
    if n.startswith("lite: the freq hinge selects") or n.startswith("lite: the repeat hinge selects") \
            or n.startswith("lite: the frequent types"):
        print(f"  [{ok}] {n[:170]}")
print("  CTX + KEY + QUERY of the frequency hinge = 4 + 8 + 2 = 14 (every CTX and KEY position: the frequent types are "
      "all 6 CTX/KEY ids); the repeat hinge has the same 14 at their maxima (mean = max, so on every sample sequence) "
      "plus VAL 2.63: on lite the repeat set is the frequency set plus the chance-repeated values, grouped by the same "
      "token id.")

# ═══ 7. token ids only (AST of the statistic) ═══════════════════════════════
H("7. what the repeat statistic reads (AST of explore_g_s89_child at b57c1867)")
tree = ast.parse(c89)
for fn in tree.body:
    if isinstance(fn, ast.FunctionDef) and fn.name in ("repeat_mask", "hinge_mask", "term_value", "_assert_tokens"):
        args = [a.arg for a in fn.args.args]
        names = sorted({n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} - set(args))
        attrs = sorted({ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Attribute)
                        and isinstance(n.value, ast.Name) and n.value.id in args})
        print(f"  {fn.name} (l.{fn.lineno}-{fn.end_lineno}): args {args}; free names {names}; attributes of args {attrs}")
for cl in tree.body:
    if isinstance(cl, ast.ClassDef) and cl.name == "Term89":
        for node in ast.walk(cl):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "term_value":
                print(f"  Term89.forward l.{node.lineno}: {ast.unparse(node)} (tokens = forward's input; _assert_tokens "
                      f"requires width L - 1)")


def repeat_mask_np(tok):                       # numpy replica of explore_g_s89_child.repeat_mask
    import numpy as np
    tok = np.asarray(tok)
    V = int(tok.max()) + 1
    cnt = np.zeros((tok.shape[0], V), dtype=int)
    for b in range(tok.shape[0]):
        np.add.at(cnt[b], tok[b], 1)
    return np.take_along_axis(cnt, tok, 1) > 1


print("  numpy replica on [[3, 3, 7, 5, 9, 1, 5]]:", repeat_mask_np([[3, 3, 7, 5, 9, 1, 5]]).tolist(),
      "(G's hand case reads [[True, True, False, True, False, False, True]])")

# ═══ 8. G's report() on the current store (memory only) ══════════════════════
H("8. G's report() (explore_g_s90.py:104-129) on the store at TIP, file write to memory")
buf = io.StringIO()
old = sys.stdout
try:
    sys.stdout = buf
    S90.report()
    sys.stdout = old
    out = buf.getvalue().splitlines()
    print("\n".join(l for l in out if "READINGS" in l or l.startswith("  first") or l.startswith("  question")
                    or l.startswith("  FOR THE SPEC") or l.startswith("  counts")))
except Exception as e:                          # noqa: BLE001
    sys.stdout = old
    print(f"  report() raised {type(e).__name__}: {e}")
