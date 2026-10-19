#!/usr/bin/env python
"""
screen_checks_s89_F.py — Referee V's checks for the review of Session F's round-6 screen (claude/explore-F at 8ba99bb,
task module explore_f_tasks.py at 50dc8e4). Reads git objects only (git show / cat-file / ls-tree / rev-list); trains
nothing; needs numpy (no torch: the generators are replicated in numpy for their distributions, and the replica is
checked against F's own printed frequencies).

  python docs/review_2026-10/screen_checks_s89_F.py > docs/review_2026-10/screen_checks_s89_F_out.txt

Sections
  A  the readings: explore_f_s89_tasks.decide (exec'd from 8ba99bb by ast, with stub stores) and explore_f_reports6's
     reading expression, on every combination of phase-0 / 24K / 48K outcomes per task; failed records
  B  the outcome functions: ast comparison of routed_end (reports6), routed_star_end (explore_f_reports), G's
     routed_star, the main line's test_stream_recipe.routed and fail_class_v2.outcome_v2; blob identity of run_one
  C  the seeds: every remote branch tip, then every json/jsonl/log blob in the history of every remote branch; the main
     line's seed constants (with attempt seeds s + 1000 j)
  D  the reference records the CHECKs read (which commit first holds them)
  E  the generators (numpy replica): frequencies and cuts, repeats by role, context distance, the no-memory Bayes bound,
     the 16-context collision bound, eta^2 by stream and VAL cos under a per-sequence channel swap, positional residue
  F  operating characteristics of "within 1 pair" / "survives" (exact; independent arms and the full range of pairing)
"""
import ast
import itertools
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import types

import numpy as np

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
F_SHA, T_SHA, MAIN, G_REF, G_SHA = "8ba99bb", "50dc8e4", "7c4ceb8", "origin/claude/explore-G", "e46799a"


def git(*a, binary=False):
    r = subprocess.run(["git", "-C", ROOT, *a], capture_output=True)
    return r.stdout if binary else r.stdout.decode("utf-8", "replace")


def show(ref, path):
    return git("show", f"{ref}:{path}")


def head(s):
    print("\n" + "=" * 110 + f"\n{s}\n" + "=" * 110)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
head("A  The readings (explore_f_s89_tasks.decide at 8ba99bb, exec'd; explore_f_reports6's reading expression)")
src = show(F_SHA, "explore_f_s89_tasks.py")
tree = ast.parse(src)
keep = {"decision", "_arms", "_bound", "decide"}
consts = {"P0_SEEDS", "P1_SEEDS", "P2_SEEDS", "TASKS", "NOMEM_THR", "LR"}
body = []
for n in tree.body:
    if isinstance(n, ast.FunctionDef) and n.name in keep:
        body.append(n)
    elif isinstance(n, ast.Assign) and any(isinstance(t, (ast.Name, ast.Tuple)) and
                                           ({t.id} if isinstance(t, ast.Name) else {e.id for e in t.elts}) & consts
                                           for t in n.targets):
        body.append(n)
mod_src = ast.unparse(ast.Module(body=body, type_ignores=[]))
rsrc = show(F_SHA, "explore_f_reports6.py")
rexpr = None
for n in ast.walk(ast.parse(rsrc)):
    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript) and ast.unparse(n.targets[0]) == "rd[t]" \
            and isinstance(n.value, ast.IfExp):
        rexpr = ast.unparse(n.value)
print("reports6 reading expression (l.73-74):", rexpr)


def make_ns(store, dfile):
    name = "s89_stub"
    m = types.ModuleType(name)
    sys.modules[name] = m
    fc = types.SimpleNamespace(setup=lambda mods: None, F_DIR=os.path.dirname(dfile))

    def runs(me, arm):
        a = m.ARMS[arm]
        # explore_common16.pick: only records with ok are returned
        return {s: store[(arm, s)] for s in a["seeds"] if (arm, s) in store and store[(arm, s)].get("ok")}
    fc.runs = runs
    m.__dict__.update(json=json, os=os, sys=sys, fc=fc, __name__=name)
    exec(compile(mod_src, "explore_f_s89_tasks@8ba99bb", "exec"), m.__dict__)
    m.DECISION = dfile
    return m


def pipeline(outcome):
    """outcome[task] = dict(p0=[4 accs or None for failed], h24=[b,b], d24=[b,b], h48=[..], d48=[..]) with b in
    {True, False, 'fail'}; runs the stages of run_s89.sh (each stage re-imports, so ARMS is recomputed) and returns
    {task: reading}."""
    d = tempfile.mkdtemp()
    dfile = os.path.join(d, "s89_decision.json")
    store = {}

    def rec_bound(b):
        if b == "fail":
            return dict(ok=False, error="x")
        return dict(ok=True, transition=1200 if b else None, acc=0.97 if b else 0.5)
    errors = []
    for stage in ("phase0", "phase1", "phase1b", "phase2"):
        m = make_ns(store, dfile)
        m.ARMS = m._arms()
        for arm, a in m.ARMS.items():
            t = arm.split("_")[-1] if not arm.endswith("_48K") else arm.split("_")[-2]
            for i, s in enumerate(a["seeds"]):
                if (arm, s) in store:
                    continue
                if arm.startswith("NOMEM"):
                    if t == "LITE":
                        store[(arm, s)] = dict(ok=True, acc=0.2, transition=None)
                        continue
                    acc = outcome[t]["p0"][(0 if arm.startswith("NOMEM_SELF") else 2) + i]
                    store[(arm, s)] = dict(ok=False) if acc is None else dict(ok=True, acc=acc, transition=None)
                elif arm.startswith("O_"):
                    mem = "h" if "HEBB" in arm else "d"
                    key = mem + ("48" if arm.endswith("_48K") else "24")
                    store[(arm, s)] = rec_bound(outcome[t][key][i])
                else:
                    store[(arm, s)] = dict(ok=True, transition=None, acc=0.3)
        m = make_ns(store, dfile)
        m.ARMS = m._arms()
        try:
            import io
            import contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                m.decide()
        except Exception as e:                                   # noqa
            errors.append(f"{stage}: {type(e).__name__} {e}")
            break
    dd = json.load(open(dfile)) if os.path.exists(dfile) else {}
    out = {}
    for t in ("POOL", "ALLPOOL"):
        if dd.get("phase0", {}).get(t, {}).get("stopped"):
            out[t] = "solvable without memory (stopped)"
        else:
            b = dd.get("budget", {}).get(t)
            out[t] = eval(rexpr, {"b": b})
    return out, errors, dd


def rule(o):
    """The docstring's rules (l.40-43, l.52-56), applied independently."""
    if any(a is not None and a > 0.6 for a in o["p0"]):
        return "solvable without memory (stopped)"
    if all(x is True for x in o["h24"] + o["d24"]):
        return "the task is valid"
    if all(x is True for x in o["h48"] + o["d48"]):
        return "the task needs the longer budget"
    return "untested"


VALID = dict(p0=[0.2] * 4, h24=[True, True], d24=[True, True], h48=[True, True], d48=[True, True])
tally, mism, names = {}, [], {}
B2 = [True, False]
for target in ("POOL", "ALLPOOL"):
    other = "ALLPOOL" if target == "POOL" else "POOL"
    for p0 in itertools.product([0.3, 0.6, 0.61], repeat=4):
        if sum(a > 0.6 for a in p0) > 1:
            continue
        for h24 in itertools.product(B2, repeat=2):
            for d24 in itertools.product(B2, repeat=2):
                for h48 in itertools.product(B2, repeat=2):
                    for d48 in itertools.product(B2, repeat=2):
                        o = dict(p0=list(p0), h24=list(h24), d24=list(d24), h48=list(h48), d48=list(d48))
                        got, err, _ = pipeline({target: o, other: VALID})
                        want = rule(o)
                        names[got[target]] = names.get(got[target], 0) + 1
                        if err or got[target] != want or got[other] != "the task is valid":
                            mism.append((target, o, got, want, err))
print(f"combinations run: {sum(names.values())} (phase-0 accs in {{0.3, 0.6, 0.61}} with at most one above 0.6, x 2^4 "
      f"24K x 2^4 48K patterns, each task as target with the other valid)")
print("readings given:", names)
print(f"combinations where the pipeline's reading differs from the docstring's rule, or an error: {len(mism)}")
for x in mism[:5]:
    print("  ", x)
# a failed record
o = dict(VALID, h24=["fail", True])
got, err, dd = pipeline({"POOL": o, "ALLPOOL": VALID})
print(f"a failed 24K pilot record (O_HEBB_POOL seed 2800 ok False): reading {got['POOL']!r}; decision {dd.get('pilots')}, "
      f"budget {dd.get('budget')}; errors {err}")
o = dict(VALID, p0=[None, 0.2, 0.2, 0.2])
got, err, dd = pipeline({"POOL": o, "ALLPOOL": VALID})
print(f"a failed phase-0 record (NOMEM_SELF_POOL seed 2800 ok False): readings {got}; phase0 recorded: {'phase0' in dd}; "
      f"errors {err}")
o = dict(VALID, h24=[True, False], h48=["fail", True])
got, err, dd = pipeline({"POOL": o, "ALLPOOL": VALID})
print(f"a failed 48K record: reading {got['POOL']!r}; budget {dd.get('budget')}")
# the validity rule's operating characteristic
print("\nP(the task is valid) = p_H^2 p_D^2 if each pilot run binds independently with probability p:")
for p in (1.0, 0.95, 0.9, 0.8, 0.7):
    print(f"  p = {p:.2f}: P(valid) = {p ** 4:.3f}; P(at least one of the 4 unbound) = {1 - p ** 4:.3f}")

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
head("B  The outcome functions (ast)")


def fn_src(ref, path, name):
    s = show(ref, path)
    for n in ast.parse(s).body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.unparse(n), n.lineno
    return None, None


cands = [(F_SHA, "explore_f_reports6.py", "routed_end"), (F_SHA, "explore_f_reports.py", "routed_star_end"),
         (G_REF, "explore_g_rhl_parent.py", "routed_star"), (MAIN, "test_stream_recipe.py", "routed"),
         (MAIN, "test_stream_recipe.py", "outcome"), (MAIN, "fail_class_v2.py", "outcome_v2"),
         (MAIN, "test_window_gate.py", "success"), (F_SHA, "explore_common.py", "routed"),
         (F_SHA, "explore_common.py", "discovered")]
for ref, path, name in cands:
    s, ln = fn_src(ref, path, name)
    print(f"--- {ref}:{path} l.{ln} {name}\n{s}")
a, _ = fn_src(F_SHA, "explore_f_reports6.py", "routed_end")
b, _ = fn_src(F_SHA, "explore_f_reports.py", "routed_star_end")
c, _ = fn_src(G_REF, "explore_g_rhl_parent.py", "routed_star")
print("\nrouted_end (reports6) == routed_star_end (explore_f_reports) as ast, name aside:",
      a.replace("routed_end", "X") == b.replace("routed_star_end", "X"))
print("routed_end == G's routed_star, name aside:", a.replace("routed_end", "X") == c.replace("routed_star", "X"))
for f in ("test_router_reliability.py", "test_router_discovery.py", "test_multilayer_binding.py", "test_router_layout.py",
          "test_binding_onset.py", "test_instrument_v2.py", "test_channel_binding.py"):
    x, y = git("rev-parse", f"{MAIN}:{f}").strip(), git("rev-parse", f"{F_SHA}:{f}").strip()
    print(f"blob {f}: main line {x[:10]} / F {y[:10]} {'identical' if x == y else 'DIFFERENT'}")
for path in ("explore_f_s89_tasks_child.py", "explore_f_reports6.py", "explore_f_s89_tasks.py"):
    s = show(F_SHA, path)
    print(f"{path}: mentions fail_class_v2 / outcome_v2: {'fail_class_v2' in s or 'outcome_v2' in s}; "
          f"ec.fail_class: {'fail_class(' in s}; one_to_one / stream_acc: {'one_to_one' in s or 'stream_acc' in s}")

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
head("C  The seeds 2800-2839")
SEEDS = set(range(2800, 2840))
pat_key = re.compile(r'"([A-Za-z0-9_]+)\|(28[0-3]\d)\|')
pat_seed = re.compile(r'"seed"\s*:\s*(28[0-3]\d)\b')
pat_log = re.compile(r'\bseed[s]?\s*[=:]?\s*(28[0-3]\d)\b')
refs = [r for r in git("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin").split() if r != "origin/HEAD"
        and r != "origin"]
EXT = (".json", ".jsonl", ".log")


def scan_blob(text):
    h = set()
    for m in pat_key.finditer(text):
        h.add(f"key {m.group(1)}|{m.group(2)}|")
    for m in pat_seed.finditer(text):
        h.add(f'"seed": {m.group(1)}')
    for m in pat_log.finditer(text):
        h.add(f"seed {m.group(1)}")
    return h


print("branch tips:")
for r in refs:
    files = [p for p in git("ls-tree", "-r", "--name-only", r).split("\n") if p.endswith(EXT)]
    hits = {}
    for p in files:
        h = scan_blob(show(r, p))
        if h:
            hits[p] = sorted(h)
    tip = git("rev-parse", "--short", r).strip()
    print(f"  {r} ({tip}): {len(files)} json/jsonl/log files; hits: {hits if hits else 'none'}")
# history: every blob at a json/jsonl/log path reachable from any remote branch
objs = git("rev-list", "--objects", "--remotes=origin").split("\n")
blobs = {}
for line in objs:
    parts = line.split(" ", 1)
    if len(parts) == 2 and parts[1].endswith(EXT):
        blobs[parts[0]] = parts[1]
inp = ("\n".join(blobs) + "\n").encode()
out = subprocess.run(["git", "-C", ROOT, "cat-file", "--batch"], input=inp, capture_output=True).stdout
i, hist = 0, {}
while i < len(out):
    nl = out.index(b"\n", i)
    hdr = out[i:nl].decode().split()
    size = int(hdr[2])
    data = out[nl + 1: nl + 1 + size].decode("utf-8", "replace")
    i = nl + 1 + size + 1
    h = scan_blob(data)
    if h:
        hist.setdefault(blobs[hdr[0]], set()).update(h)
print(f"history: {len(blobs)} distinct json/jsonl/log blobs reachable from {len(refs)} remote branches; paths with a hit: "
      f"{ {p: sorted(v) for p, v in hist.items()} if hist else 'none'}")
# the main line's seed constants
mainfiles = [p for p in git("ls-tree", "--name-only", MAIN).split() if p.startswith("test_") and p.endswith(".py")]
used = set()
for p in mainfiles:
    s = show(MAIN, p)
    for n in ast.walk(ast.parse(s)):
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "range" and all(isinstance(x, ast.Constant)
                                                                                      for x in n.args):
            args = [x.value for x in n.args]
            if all(isinstance(v, int) for v in args) and len(args) in (1, 2) and max(args) < 10_000:
                used |= set(range(*args))
att = {s + 1000 * j for s in used for j in range(5)}
print(f"main line {MAIN}: every literal range() in {len(mainfiles)} test_*.py files covers {len(used)} integers; with attempt "
      f"seeds s + 1000 j (j < 5): {len(att)}; intersection with 2800-2839: {sorted(att & SEEDS) or 'none'} "
      f"(bases that would reach it: 800-839, 1800-1839: {sorted(used & (set(range(800, 840)) | set(range(1800, 1840)))) or 'none'})")
prm = show(MAIN, "docs/reading/parallel_sessions_prompts.md")
print("prompts: blocks named for round 6:", re.findall(r"Seed blocks: [^\n]+", prm)[-1:])

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
head("D  The reference records the CHECKs read")


def first_with(ref, path, key):
    for h in git("log", "--reverse", "--format=%h", ref, "--", path).split():
        try:
            d = json.loads(show(h, path))
        except Exception:                                        # noqa
            continue
        if key in d.get("runs", {}):
            return h, d["runs"][key]
    return None, None


for ref, path, key in ((G_REF, "explore_out/G/s87_results.json", "BASELINE|2700|f205db31dfed|Intel(R) Xeon(R) Processor @ 2.10GHz"),
                       ("origin/claude/explore-F", "explore_out/F/rh_lite_results.json",
                        "S70_DELTA|330|ff908bdbc0b2|Intel(R) Xeon(R) Processor @ 2.10GHz")):
    h, r = first_with(ref, path, key)
    tipr = json.loads(show(ref, path))["runs"].get(key)
    same = tipr == r
    print(f"{key}\n  first held by {h} ({git('log', '-1', '--format=%ad %s', '--date=iso', h).strip()[:90]})\n"
          f"  curve through 2400 {[c for c in r['curve'] if c[0] <= 2400]}; grad steps {sorted(r['grad'], key=int)}; "
          f"stats steps <= 2400 {[s['step'] for s in r['stats'] if s['step'] != 'end' and s['step'] <= 2400]}; "
          f"identical at {ref}'s tip: {same}")
s = show(F_SHA, "explore_f_s89_tasks_child.py")
print("child l.51 reads G's record from", re.search(r'G_REF, G_SHA, G_FILE = ([^\n]+)', s).group(1),
      "-> the record is read from the moving tip G_REF; the module from G_SHA")
print("child l.255 reads S70's record with ec.load_store('F/rh_lite'): the working tree's file, no commit")

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
head("E  The generators (numpy replica of explore_f_tasks at 50dc8e4: lite, pool, allpool)")
S, P, NV, NB, KP, NC = 2, 4, 16, 2, 32, 16
COMPS = [c for c in itertools.product(range(1, 4), repeat=2) if sum(c) == P]
rng = np.random.default_rng(89)


def lite_seq():
    vals = [rng.permutation(NV)[:S] for _ in range(P)]
    comp = [COMPS[rng.integers(len(COMPS))] for _ in range(S)]
    korder = [rng.permutation(P) for _ in range(S)]
    border = rng.permutation(S * NB)
    q = rng.integers(S * P)
    seq, role, strm, pairs = [], [], [], []
    for blk in border:
        s_, i_ = divmod(int(blk), NB)
        st = sum(comp[s_][:i_])
        seq.append(("C", s_)); role.append("CTX"); strm.append(s_)
        for kk in korder[s_][st:st + comp[s_][i_]]:
            pairs.append((len(seq), s_, int(kk)))
            seq.append(("K", int(kk))); role.append("KEY"); strm.append(s_)
            seq.append(("V", int(vals[kk][s_]))); role.append("VAL"); strm.append(s_)
    qs, qk = q // P, q % P
    seq += [("C", qs), ("K", qk), ("V", int(vals[qk][qs]))]
    role += ["CTX", "KEY", "VAL"]
    strm += [qs, qs, qs]
    return seq, role, strm, pairs, (qs, qk)


def relabel(seq, kind):
    keyset = rng.permutation(KP)[:P]
    if kind == "allpool":
        pair = rng.permutation(NC)[:S]
        first = seq[0][1]
        ctxset = {first: int(pair[0]), 1 - first: int(pair[1])}
    out = []
    for r_, v in seq:
        if r_ == "K":
            out.append(("K", int(keyset[v])) if kind != "lite" else ("K", v))
        elif r_ == "C" and kind == "allpool":
            out.append(("C", ctxset[v]))
        else:
            out.append((r_, v))
    return out


def sample(kind, n):
    rows = []
    for _ in range(n):
        seq, role, strm, pairs, q = lite_seq()
        toks = relabel(seq, kind)
        first = seq[0][1]
        idx = [0 if s_ == first else 1 for s_ in strm]           # stream index by first appearance (allpool's label)
        rows.append(dict(toks=toks, role=role, strm=strm, idx=idx, pairs=pairs, q=q, lite=seq))
    return rows


N = 20000
data = {k: sample(k, N) for k in ("lite", "pool", "allpool")}
INP = 22                                                         # input tokens tokens[:, :-1]
for k, rows in data.items():
    cnt = {}
    for r in rows:
        for t in r["toks"][:INP]:
            cnt[t] = cnt.get(t, 0) + 1
    tot = sum(cnt.values())
    fr = sorted(((v / tot, t) for t, v in cnt.items()), reverse=True)
    by = {}
    for f, t in fr:
        by.setdefault({"C": "CTX", "K": "KEY", "V": "VAL"}[t[0]], []).append(f)
    vals = [f for f, _ in fr]
    cut = max(range(len(vals) - 1), key=lambda i: vals[i] / vals[i + 1])
    lr_roles = [fr[i][1][0] for i in range(cut + 1)]
    acc, hm = 0.0, []
    for f, t in fr:
        hm.append(t[0]); acc += f
        if acc >= 0.5:
            break
    summ = "; ".join(f"{r_} {len(v)} types {min(v):.4f}-{max(v):.4f}" for r_, v in by.items())
    print(f"{k:8s} {summ}; largest-ratio cut {dict((x, lr_roles.count(x)) for x in 'CKV' if lr_roles.count(x))}; "
          f"half-mass cut {dict((x, hm.count(x)) for x in 'CKV' if hm.count(x))}")
print("  (F's docstring l.16-25: lite CTX 0.114, KEY 0.102, VAL 0.023; pool KEY 0.013; allpool CTX 0.014, KEY 0.013; the cuts:\n"
      "   pool LR = 2 CTX, half-mass 2 CTX + 12 VAL; allpool LR = 16 VAL, half-mass 16 VAL + 10 CTX — the replica agrees\n"
      "   up to sampling where the half-mass cut falls inside a tied class)")

print("\nRepeats within a sequence (input tokens, 22 positions; types occurring more than once):")
for k, rows in data.items():
    tot = {"CTX": 0, "KEY": 0, "VAL": 0}
    rep = {"CTX": 0, "KEY": 0, "VAL": 0}
    val_rep_seq, nonkey = 0, []
    body_rep = {"CTX": 0, "KEY": 0, "VAL": 0}
    for r in rows:
        toks = r["toks"][:INP]
        c = {}
        for t in toks:
            c[t] = c.get(t, 0) + 1
        bc = {}
        for t in toks[:20]:
            bc[t] = bc.get(t, 0) + 1
        nk = 0
        for pos, (t, ro) in enumerate(zip(toks, r["role"][:INP])):
            tot[ro] += 1
            if c[t] > 1:
                rep[ro] += 1
                if ro != "KEY":
                    nk += 1
            if pos < 20 and bc[t] > 1:
                body_rep[ro] += 1
        nonkey.append(nk)
        vv = [t for t in toks[:20] if t[0] == "V"]
        val_rep_seq += len(set(vv)) < len(vv)
    n = len(rows)
    print(f"  {k:8s} per sequence: repeated positions CTX {rep['CTX'] / n:.2f} of {tot['CTX'] / n:.0f}, KEY {rep['KEY'] / n:.2f} "
          f"of {tot['KEY'] / n:.0f}, VAL {rep['VAL'] / n:.2f} of {tot['VAL'] / n:.0f}; in the body (20 positions) CTX "
          f"{body_rep['CTX'] / n:.2f}, KEY {body_rep['KEY'] / n:.2f}, VAL {body_rep['VAL'] / n:.2f}; P(some value type "
          f"twice in the body) {val_rep_seq / n:.3f}; non-key repeated positions per sequence: mean {np.mean(nonkey):.2f}, "
          f"min {min(nonkey)}; sequences where the repeated positions are exactly the key positions: "
          f"{np.mean([x == 0 for x in nonkey]):.4f}")

print("\nContext distance (KEY position minus its block's CTX position) and the LOCAL3 window (t-2..t):")
for k in ("pool",):
    dist = {}
    for r in data[k]:
        last = None
        for pos, ro in enumerate(r["role"][:20]):
            if ro == "CTX":
                last = pos
            elif ro == "KEY":
                dd_ = pos - last
                dist[dd_] = dist.get(dd_, 0) + 1
    tot = sum(dist.values())
    print(f"  {k}: " + ", ".join(f"{d_}: {v / tot:.3f}" for d_, v in sorted(dist.items()))
          + f"; keys whose CTX is inside a width-3 window: {sum(v for d_, v in dist.items() if d_ <= 2) / tot:.3f}"
          " (the same on lite and allpool: the layout is lite's)")
    gap = []
    for r in data[k]:
        pos = {}
        for p_, s_, kk in r["pairs"]:
            pos.setdefault(kk, []).append(p_)
        gap += [abs(a_ - b_) for a_, b_ in pos.values()]
    print(f"  distance between a key's two body occurrences: min {min(gap)}, median {int(np.median(gap))}, max {max(gap)}")

print("\nPhase 0's no-memory models: a Bayes upper bound on held-out accuracy at the query (input position 21):")
for k in ("lite", "pool", "allpool"):
    for name, reach in (("NOMEM (last 4 tokens)", 4), ("NOMEM_SELF (last 10 tokens)", 10)):
        lo = INP - reach
        seen = 0
        for r in data[k]:
            qs, qk = r["q"]
            p_ = [p for p, s_, kk in r["pairs"] if s_ == qs and kk == qk][0]
            seen += p_ >= lo
        ps = seen / len(data[k])
        ub = ps + (1 - ps) / 15
        print(f"  {k:8s} {name:28s}: P(target KEY and VAL inside the reach) {ps:.3f}; bound {ub:.3f} (rule: stop if > 0.6)")

print("\nALLPOOL's contexts: no fixed CTX type -> stream map")
best = 64 / 120
print(f"  two distinct contexts from 16 types: a fixed map of types to 2 channels separates them in at most "
      f"8*8/C(16,2) = {best:.4f} of sequences (balanced map); collision {1 - best:.4f}")
sep_ = np.mean([(r["toks"][0][1] < 8) != ([t for t in r["toks"][:20] if t[0] == "C" and t != r["toks"][0]][0][1] < 8)
                for r in data["allpool"]])
print(f"  replica, map = type < 8: separated {sep_:.4f}")


def eta2(x, lab):
    x, lab = np.asarray(x, float), np.asarray(lab)
    mu = x.mean()
    tot = ((x - mu) ** 2).sum()
    if tot <= 0:
        return 0.0
    return float(sum((lab == g).sum() * (x[lab == g].mean() - mu) ** 2 for g in np.unique(lab)) / tot)


def cos(a_, b_):
    return float(np.dot(a_, b_) / np.linalg.norm(a_) / np.linalg.norm(b_))


print("  A gate that routes every separated sequence perfectly, channel = map(CTX type of the block), measured as\n"
      "  pool_routing_stats / role_cosines measure it (pooled over sequences, stream = allpool's first-appearance index):")
for k in ("pool", "allpool"):
    p0, labs, vg0, vg1, marg = [], [], [], [], []
    for r in data[k]:
        ctx = [t for t in r["toks"][:20] if t[0] == "C"]
        if k == "allpool" and (ctx[0][1] < 8) == ([c for c in ctx if c != ctx[0]][0][1] < 8):
            continue                                             # keep only the sequences the map separates
        ch = None
        lab = r["idx"] if k == "allpool" else r["strm"]
        for pos, (t, ro) in enumerate(zip(r["toks"][:20], r["role"][:20])):
            if ro == "CTX":
                ch = (t[1] < 8) if k == "allpool" else (t[1] == 0)
            g = np.array([1.0, 0.0]) if ch else np.array([0.0, 1.0])
            if ro == "KEY":
                p0.append(g[0]); labs.append(lab[pos])
            if ro == "VAL":
                (vg0 if lab[pos] == 0 else vg1).append(g)
        marg.append(1.0)
    print(f"    {k:8s}: margin {np.mean(marg):.3f}; eta^2 by stream at KEY {eta2(p0, labs):.4f}; VAL cos between the "
          f"streams' mean gates {cos(np.mean(vg0, 0), np.mean(vg1, 0)):.4f} -> ROUTED*@end "
          f"{np.mean(marg) >= 0.9 and eta2(p0, labs) > 0.9}, DISCOVERED-if-bound {cos(np.mean(vg0, 0), np.mean(vg1, 0)) < 0.5}")
    # the same with per-sequence alignment (channel of stream 0 := channel 0)
print("  Positional residue of the first-appearance index (allpool) vs the CTX id (pool): eta^2 by stream at KEY of the\n"
      "  best position-only gate p(t) = P(stream label 0 | t):")
for k in ("pool", "allpool"):
    cnt0 = np.zeros(20); cntn = np.zeros(20)
    for r in data[k]:
        lab = r["idx"] if k == "allpool" else r["strm"]
        for pos in range(20):
            cntn[pos] += 1; cnt0[pos] += lab[pos] == 0
    pt = cnt0 / cntn
    xs, ls = [], []
    for r in data[k][:5000]:
        lab = r["idx"] if k == "allpool" else r["strm"]
        for pos, ro in enumerate(r["role"][:20]):
            if ro == "KEY":
                xs.append(pt[pos]); ls.append(lab[pos])
    print(f"    {k:8s}: P(label 0 | position) at positions 0-5: {np.round(pt[:6], 3).tolist()}; eta^2 {eta2(xs, ls):.4f}")

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
head("F  Operating characteristics: 'within 1 pair of the labelled ceiling' and 'survives'")


def mcn(b, c):
    n = b + c
    return 1.0 if n == 0 else sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n


def dist_pairs(n, pc, pl, q):
    """distribution of (C only, L only) over n seeds, P(C and L) = q per seed."""
    a_, b_, c_ = pc - q, pl - q, q
    d_ = 1 - a_ - b_ - c_
    assert min(a_, b_, c_, d_) > -1e-12, (pc, pl, q)
    out = {}
    for x in range(n + 1):
        for y in range(n - x + 1):
            rest = n - x - y
            # x C-only, y L-only, rest split both/neither (summed)
            pr = math.comb(n, x) * math.comb(n - x, y) * a_ ** x * b_ ** y * (c_ + d_) ** rest
            out[(x, y)] = pr
    return out


def p_within(n, pc, pl, q):
    return sum(v for (x, y), v in dist_pairs(n, pc, pl, q).items() if x - y <= 1)


def q_range(pc, pl):
    return max(0.0, pc + pl - 1), min(pc, pl)


print("P(within 1 pair) = P((ceiling only) - (learned only) <= 1), G's definition (explore_g_s87 l.27):")
print(" n   ceiling learned | independent | min..max over every pairing (q = P(both) from max(0,pc+pl-1) to min(pc,pl))")
for n in (20, 10):
    for pc in (0.9, 1.0):
        for pl in (0.9, 0.8, 0.7, 0.5):
            if pl > pc:
                continue
            lo_, hi_ = q_range(pc, pl)
            vals = [p_within(n, pc, pl, lo_ + (hi_ - lo_) * i / 50) for i in range(51)]
            print(f" {n:2d}  {pc:.2f}    {pl:.2f}    |   {p_within(n, pc, pl, pc * pl):.3f}     | {min(vals):.3f} .. {max(vals):.3f}")


def p_survives(n, pc, pl, pb, q, need_beats=None):
    """G: learned discovered vs BASELINE (one-sided exact McNemar p < 0.05) AND within 1 pair of the ceiling.
    BASELINE independent of (C, L). DP over seeds: state (C only - L only, L only vs B, B only vs L)."""
    pcl = {(1, 1): q, (1, 0): pc - q, (0, 1): pl - q, (0, 0): 1 - pc - pl + q}
    per = {}
    for (cc, ll), pr in pcl.items():
        for bb in (0, 1):
            per[(cc, ll, bb)] = pr * (pb if bb else 1 - pb)
    st = {(0, 0, 0): 1.0}
    for _ in range(n):
        nx = {}
        for (d_, x, y), pr in st.items():
            for (cc, ll, bb), pp in per.items():
                if pp <= 0:
                    continue
                k_ = (d_ + (cc - ll), x + (ll and not bb), y + (bb and not ll))
                nx[k_] = nx.get(k_, 0.0) + pr * pp
        st = nx
    tot = 0.0
    for (d_, x, y), pr in st.items():
        ok = d_ <= 1 and (mcn(x, y) < 0.05 if need_beats is None else mcn(x, y) <= need_beats)
        tot += pr * ok
    return tot


print("\nG's 'the frequency (or repeat) hinge survives' = discovered against BASELINE (p < 0.05) AND within 1 pair of\n"
      "REG3_KEYHINGE; 20 seeds; BASELINE rate pb (S87 on lite: BASELINE 0/20 ROUTED-BOUND; REG3 2/20):")
print(" pb   ceiling learned | independent | min..max over pairing")
for pb in (0.0, 0.1):
    for pc in (0.9, 1.0):
        for pl in (0.9, 0.8, 0.7, 0.5):
            lo_, hi_ = q_range(pc, pl)
            vals = [p_survives(20, pc, pl, pb, lo_ + (hi_ - lo_) * i / 10) for i in range(11)]
            print(f" {pb:.1f}  {pc:.2f}    {pl:.2f}    |   {p_survives(20, pc, pl, pb, pc * pl):.3f}     | "
                  f"{min(vals):.3f} .. {max(vals):.3f}")
print("\nH's 'the all-position (repeat) trigger survives' = within 1 pair of BASE AND beats NOSPLIT by >= 4 vs 0\n"
      "(one-sided p <= 0.0625); 10 seeds; NOSPLIT rate pb:")
print(" pb   ceiling learned | independent | min..max over pairing")
for pb in (0.0, 0.3):
    for pc in (0.9, 1.0):
        for pl in (0.9, 0.8, 0.7, 0.5):
            lo_, hi_ = q_range(pc, pl)
            vals = [p_survives(10, pc, pl, pb, lo_ + (hi_ - lo_) * i / 10, need_beats=0.0625) for i in range(11)]
            print(f" {pb:.1f}  {pc:.2f}    {pl:.2f}    |   {p_survives(10, pc, pl, pb, pc * pl, need_beats=0.0625):.3f}     | "
                  f"{min(vals):.3f} .. {max(vals):.3f}")
print("\nThe complementary rule 'the ceiling beats it by >= 4 vs 0' (G: 'needs frequent keys'), 20 seeds, independent:")
for pc in (0.9, 1.0):
    for pl in (0.9, 0.8, 0.7, 0.5):
        pr = sum(v for (x, y), v in dist_pairs(20, pc, pl, pc * pl).items() if mcn(x, y) <= 0.0625)
        pw = p_within(20, pc, pl, pc * pl)
        print(f"  ceiling {pc:.2f} learned {pl:.2f}: P(beats by >= 4 vs 0) {pr:.3f}; P(within 1 pair) {pw:.3f}; "
              f"P(neither: 'no reading applies') >= {max(0.0, 1 - pr - pw):.3f}")
print("\ndone")
