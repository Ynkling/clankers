#!/usr/bin/env python3
"""
screen_checks_s89_G.py -- Referee V, 19 October 2026. Pre-run checks of Session G's S89 screen (claude/explore-G at
7ab98c1: explore_g_s89.py, explore_g_s89_child.py, the vendored explore_f_tasks.py) for screen_review_s89_G.md.

Reads git objects only (git show / cat-file / ls-tree); trains nothing; imports nothing from the repository (torch is
not installed in this container, so G's torch functions are not executed: the generator, repeat_mask, freq_types and
the cut are re-implemented line for line in numpy below and run on a sample drawn from the same DISTRIBUTION as
RandHeaderTask(P=4, nb=2, lmin=1, lmax=3) and F's pool / allpool relabelling; numbers are distributional, not
torch-bit-identical; F's own printed tables (claude/explore-F explore_out/F/s89.log) are quoted beside them).

Output: screen_checks_s89_G_out.txt (sections §1-§7).
"""
import ast
import difflib
import itertools
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np

G, F_MOD, F_OLD = "7ab98c1", "50dc8e4", "a199d5e"
MAIN = "7c4ceb8"
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)


def git(*args, binary=False):
    r = subprocess.run(["git", *args], capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode("utf-8", "replace")


def show(rev, path, binary=False):
    return git("show", f"{rev}:{path}", binary=binary)


def lines_of(src, needle, start=1):
    return [i for i, l in enumerate(src.splitlines(), 1) if needle in l and i >= start]


# ══ §1 the vendored module ══════════════════════════════════════════════════════════════════════════════════════════
say("§1 VENDORED explore_f_tasks.py")
blobs = {}
for rev in (G, F_MOD, F_OLD, "origin/claude/explore-F"):
    b = show(rev, "explore_f_tasks.py", binary=True)
    blobs[rev] = b
    say(f"  {rev:24s} blob {git('rev-parse', f'{rev}:explore_f_tasks.py').strip()}  {len(b)} bytes")
g, f = blobs[G], blobs[F_MOD]
say(f"  G's copy == F's 50dc8e4: {g == f};  G's copy == a199d5e: {g == blobs[F_OLD]};  G's copy is a byte prefix of F's "
    f"50dc8e4 file: {f.startswith(g)} ({len(g)} of {len(f)} bytes); F's tip == 50dc8e4: {blobs['origin/claude/explore-F'] == f}")
fsrc = f.decode()
for cls in ("class RandHeaderTaskPool", "class RandHeaderTaskAllPool", "def largest_ratio_cut", "def pool_stats"):
    say(f"  {cls!r}: in G's copy {cls in g.decode()}; in F's 50dc8e4 at line {lines_of(fsrc, cls)}")

# ══ §2 what G's files contain (ast) ══════════════════════════════════════════════════════════════════════════════════
say("\n§2 THE SCREEN'S CODE (ast)")
par = show(G, "explore_g_s89.py")
pt = ast.parse(par)
say(f"  explore_g_s89.py: {len(par.splitlines())} lines; top-level statements {[type(n).__name__ for n in pt.body]} "
    f"(a docstring and nothing else: no ARMS, seeds, queue, validity gate, reading or report code)")
ch = show(G, "explore_g_s89_child.py")
ct = ast.parse(ch)
fns = {n.name: n for n in ast.walk(ct) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
say(f"  explore_g_s89_child.py: {len(ch.splitlines())} lines; defs {sorted(fns)}")
tasks_line = lines_of(ch, "TASKS = ")
say(f"  TASKS (child:{tasks_line}): {ch.splitlines()[tasks_line[0] - 1].strip()}")
chk_src = ast.get_source_segment(ch, fns["checks"])
say(f"  checks() mentions the vendored blob / F's commit / a hash: "
    f"{any(w in chk_src for w in ('50dc8e4', 'blob', 'hash', 'sha', 'explore_f_tasks.py'))}")
for name in ("repeat_mask", "frequent", "hinge_mask", "term_value", "selected"):
    node = fns[name]
    attrs = sorted({ast.unparse(n) for n in ast.walk(node) if isinstance(n, ast.Attribute)
                    and not ast.unparse(n).startswith(("torch.", "F.", "st[", "self."))})
    calls = sorted({ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)})
    say(f"  {name} (child:{node.lineno}-{node.end_lineno}): attributes read {attrs}; calls {calls}")

# Term89.forward vs S87's Term87.forward; term_value's freq branch vs S87's
c87 = show(G, "explore_g_s87_child.py")
t87 = ast.parse(c87)
f87 = {n.name: n for n in ast.walk(t87) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def method(tree_fns, cls, m):
    return next(n for n in tree_fns[cls].body if isinstance(n, ast.FunctionDef) and n.name == m)


a = ast.unparse(method(fns, "Term89", "forward")).splitlines()
b = ast.unparse(method(f87, "Term87", "forward")).splitlines()
say("  ast diff Term87.forward (S87, explore_g_s87_child.py:127-162) -> Term89.forward (child:100-133):")
for l in difflib.unified_diff(b, a, "Term87.forward", "Term89.forward", n=0, lineterm=""):
    if not l.startswith(("---", "+++")):
        say("    " + l)
say("  S89 imports S87's freq_types / FREQ_SAMPLE / FREQ_SEED (child:76) and S82's key_eta (child:91): "
    f"{'s87.freq_types' in ch and 'dc.key_eta' in ch}; repeat_mask and the hinge mask are new (child:62-67, 80-86)")

# the outcome functions
rhp = show(G, "explore_g_rhl_parent.py")
say(f"  ROUTED-BOUND as a named function in G's committed code: explore_g_rhl_parent.routed_bound at line "
    f"{lines_of(rhp, 'def routed_bound')}, outcome() at {lines_of(rhp, 'def outcome')}, paired/beats/not_worse_1 at "
    f"{lines_of(rhp, 'def paired') + lines_of(rhp, 'def beats') + lines_of(rhp, 'def not_worse_1')}; S87's parent uses it "
    f"(explore_g_s87.py: {lines_of(show(G, 'explore_g_s87.py'), 'hp.routed_bound')}); S89 imports it: "
    f"{'routed_bound' in par or 'routed_bound' in ch}")
main_hits = git("grep", "-n", "def routed_bound", MAIN, "--", "*.py").strip()
say(f"  main line {MAIN}: 'def routed_bound' in a .py outside docs/review: "
    f"{[h for h in main_hits.splitlines() if 'docs/review' not in h] or 'none'}")
ec_src = show(G, "explore_common.py")
say(f"  name clash: explore_common.discovered (line {lines_of(ec_src, 'def discovered')}) = bound and val_cos < 0.5; "
    f"explore_g_rhl_parent.discovered (line {lines_of(rhp, 'def discovered')}) the same; S89's 'discovered' "
    f"(explore_g_s89.py:40) = beats BASELINE p < 0.05")

# lite-only code on the S89 run path
say("  lite-vocabulary code on the S89 run path (reads token ids S..S+P-1 as keys, or the module-level body_positions):")
fmod_old = g.decode()
devc = show(G, "explore_g_devices_child.py")
rcs = show(G, "explore_g_rhl_child.py")
gcs = show(G, "explore_g_common.py")
for where, src, needles in (
        ("explore_f_tasks.py (module) body_positions", fmod_old, ("def body_positions", "is_key = (body >= S) & (body < S + P)")),
        ("explore_g_devices_child.key_eta", devc, ("def key_eta", "kid = tokens.gather(1, kpos) - task.S", "eta2_pool(gk, kid, task.P)")),
        ("explore_g_rhl_child STATS_F / layer_stats / probe_rh / acc_by_cell / roles", rcs,
         ("STATS_F = ft.rand_header_stats", "kpos, vpos = ft.body_positions(task, pinp)", "kpos, vpos = ft.body_positions(task, x)",
          "elif S <= tok < S + task.P:", "KEY=(x >= S) & (x < S + P) & body")),
        ("explore_g_common.stream_acc", gcs, ("qs = data[:, task.qpos[0] - 1]",)),
        ("explore_g_s89_child.selected", ch, ("kpos, _ = ft.body_positions(t, x)", "is_ctx, is_key = x < S, (x >= S) & (x < S + t.P)")),
        ("explore_g_rhl_child ORACLE_ALL", rcs, ("CEIL = ", "m = build(task, CEIL, ARCH, seed)"))):
    say(f"    {where}: " + "; ".join(f"{n!r} at {lines_of(src, n)}" for n in needles))

# ══ §3 the generator's distribution, the cuts, the hinges' positions ════════════════════════════════════════════════
say("\n§3 FREQUENCY CUT AND REPEAT MASK ON A SAMPLE (numpy re-implementation of RandHeaderTask(P=4, nb=2, lmin=1, "
    "lmax=3).make_batch and F's _relabel; 4096 sequences; distribution as F's, draws not torch's)")
S, P, NV, NB = 2, 4, 16, 2
COMPS = [c for c in itertools.product(range(1, 4), repeat=NB) if sum(c) == P]
K_POOL, N_CTX = 32, 16


def lite_batch(B, rng):
    vals = rng.random((B, P, NV)).argsort(-1)[..., :S]
    comp = rng.integers(len(COMPS), size=(B, S))
    korder = rng.random((B, S, P)).argsort(-1)
    border = rng.random((B, S * NB)).argsort(-1)
    q = rng.random((B, S * P)).argsort(-1)[:, :1]
    rows, roles, keypos = [], [], []
    for b in range(B):
        seq, rl = [], []
        for blk in border[b]:
            s, i = divmod(int(blk), NB)
            ln = COMPS[comp[b, s]]
            st = sum(ln[:i])
            seq.append(s); rl.append("CTX")
            for kk in korder[b, s, st:st + ln[i]]:
                seq += [S + int(kk), S + P + int(vals[b, kk, s])]; rl += ["KEY", "VAL"]
        qs, qk = int(q[b, 0]) // P, int(q[b, 0]) % P
        seq += [qs, S + qk, S + P + int(vals[b, qk, qs])]; rl += ["CTX", "KEY", "VAL"]
        rows.append(seq); roles.append(rl)
    return np.array(rows), np.array(roles)


def relabel(x, rng, allpool):
    B = x.shape[0]
    keyset = rng.random((B, K_POOL)).argsort(-1)[:, :P]
    n_ctx = N_CTX if allpool else S
    key0, val0 = n_ctx, n_ctx + K_POOL
    out = x.copy()
    isk, isv, isc = (x >= S) & (x < S + P), x >= S + P, x < S
    out[isk] = key0 + np.take_along_axis(keyset, np.clip(x - S, 0, P - 1), 1)[isk]
    out[isv] = val0 + (x - S - P)[isv]
    if allpool:
        pair = rng.random((B, N_CTX)).argsort(-1)[:, :S]
        first = x[:, 0]
        ctxset = np.empty_like(pair)
        ctxset[np.arange(B), first] = pair[:, 0]
        ctxset[np.arange(B), 1 - first] = pair[:, 1]
        out[isc] = np.take_along_axis(ctxset, np.clip(x, 0, S - 1), 1)[isc]
    return out, dict(n_ctx=n_ctx, key0=key0, val0=val0)


def largest_ratio_cut(tok):                      # S87's freq_types / F's largest_ratio_cut, line for line in numpy
    cnt = np.bincount(tok.reshape(-1)).astype(float)
    fr = cnt / cnt.sum()
    order = [int(t) for t in np.argsort(-fr, kind="stable") if fr[t] > 0]
    vals = [fr[t] for t in order]
    cut = max(range(len(vals) - 1), key=lambda i: vals[i] / vals[i + 1])
    return order[:cut + 1], [(t, round(v, 5)) for t, v in zip(order, vals)]


def half_mass(tok):
    cnt = np.bincount(tok.reshape(-1)).astype(float)
    fr = cnt / cnt.sum()
    acc, sel = 0.0, []
    for t in np.argsort(-fr, kind="stable"):
        sel.append(int(t)); acc += fr[t]
        if acc >= 0.5:
            break
    return sel


def repeat_mask(tok):                             # G's repeat_mask (child:62-67) in numpy
    V = int(tok.max()) + 1
    cnt = np.zeros((tok.shape[0], V), dtype=int)
    for b in range(tok.shape[0]):
        np.add.at(cnt[b], tok[b], 1)
    return np.take_along_axis(cnt, tok, 1) > 1


rng = np.random.default_rng(87_000)
lite, roles = lite_batch(4096, rng)
inp_roles = roles[:, :-1]
body_len = S * (NB + 2 * P)
keyterm = np.zeros_like(inp_roles, dtype=bool)
keyterm[:, :body_len] = inp_roles[:, :body_len] == "KEY"           # the key term's 8 body KEY positions
tasks = {"lite": (lite, dict(n_ctx=S, key0=S, val0=S + P))}
tasks["pool"] = relabel(lite, np.random.default_rng(1), False)
tasks["allpool"] = relabel(lite, np.random.default_rng(2), True)
F_LOG = {"lite": "LRC 6 types [CTX, KEY]; half-mass 5 {CTX 2, KEY 3}",
         "pool": "LRC 2 types [CTX]; half-mass 14 {CTX 2, VAL 12}; KEY 0.0120-0.0138, VAL 0.0220-0.0236",
         "allpool": "LRC 16 types [VAL]; half-mass 26 {CTX 10, VAL 16}; VAL 0.0220-0.0236, CTX 0.0128-0.0151, KEY 0.0120-0.0138"}
rep_lite = None
for name, (x, voc) in tasks.items():
    xin = x[:, :-1]
    role = np.where(xin < voc["n_ctx"], "CTX", np.where(xin < voc["val0"], "KEY", "VAL"))
    assert (role == inp_roles).all()
    sel, tab = largest_ratio_cut(xin)
    fr = dict(tab)
    by = defaultdict(list)
    for t, v in tab:
        by["CTX" if t < voc["n_ctx"] else "KEY" if t < voc["val0"] else "VAL"].append(v)
    rng_txt = ", ".join(f"{r} {len(v)} types {min(v):.4f}-{max(v):.4f}" for r, v in by.items())
    selroles = Counter("CTX" if t < voc["n_ctx"] else "KEY" if t < voc["val0"] else "VAL" for t in sel)
    hm = Counter("CTX" if t < voc["n_ctx"] else "KEY" if t < voc["val0"] else "VAL" for t in half_mass(xin))
    fm = np.isin(xin, sel)
    rm = repeat_mask(xin)
    if name == "lite":
        rep_lite = rm
    say(f"  {name}: {rng_txt}")
    say(f"    largest-ratio cut: {len(sel)} types {dict(selroles)}; half-mass cut {dict(hm)}   [F's s89.log: {F_LOG[name]}]")
    for hname, m in (("frequency hinge", fm), ("repeat hinge", rm)):
        per = {r: (m & (inp_roles == r)).sum(1).mean() for r in ("CTX", "KEY", "VAL")}
        both = (m & keyterm).sum(1)
        say(f"    {hname}: per sequence {m.sum(1).mean():.2f} positions (CTX {per['CTX']:.2f}, KEY {per['KEY']:.2f}, VAL "
            f"{per['VAL']:.2f}); of the key term's 8 positions {both.mean():.2f} (min {both.min()}, max {both.max()}); "
            f"key-term share of the selected {both.sum() / max(m.sum(), 1):.3f}")
    if name != "lite":
        say(f"    repeat mask identical to lite's at every position of every sequence: {bool((rm == rep_lite).all())}")
        body = x[:, :body_len]
        lite_key = (body >= S) & (body < S + P)
        ok = (lite_key.sum(1) == S * P)
        kidmax = int((x[:, :body_len][inp_roles[:, :body_len] == "KEY"] - S).max())
        say(f"    module-level ft.body_positions(task, x) (explore_f_tasks.py:95-103): its assert (8 tokens in S..S+P-1 per "
            f"body) holds in {int(ok.sum())} of 4096 sequences -> P(a 32-sequence batch passes) = {ok.mean() ** 32:.3g}; "
            f"dc.key_eta's key ids tokens - S reach {kidmax} against one_hot(.., task.P = 4)")
        # what selected() would label as CTX / KEY / VAL (child:173-176) against the true roles
        is_ctx, is_key = xin < S, (xin >= S) & (xin < S + P)
        say(f"    selected()'s role masks (child:174) on this vocabulary: 'ctx' = true CTX {(is_ctx & (inp_roles == 'CTX')).sum(1).mean():.2f}"
            f" of 5 per sequence; 'key' = true KEY {(is_key & (inp_roles == 'KEY')).sum(1).mean():.2f} of 9, true CTX "
            f"{(is_key & (inp_roles == 'CTX')).sum(1).mean():.2f}")
# G's docstring arithmetic (explore_g_s89.py:24-27)
say(f"  analytic per-type frequencies (22 input tokens: 5 CTX, 9 KEY, 8 VAL): pool CTX {5 / 22 / 2:.4f}, VAL {8 / 22 / 16:.4f}, "
    f"KEY {9 / 22 / 32:.4f}; allpool CTX {5 / 22 / 16:.4f} (G's docstring: 0.028), VAL {8 / 22 / 16:.4f}, KEY {9 / 22 / 32:.4f}; "
    f"allpool ratios VAL/CTX {(8 / 16) / (5 / 16):.2f}, CTX/KEY {(5 / 16) / (9 / 32):.2f} -> the cut falls after the 16 VAL types")

# ══ §4 the readings, enumerated ═════════════════════════════════════════════════════════════════════════════════════
say("\n§4 READINGS (explore_g_s89.py:39-52, implemented here from the prose: G committed no reading code)")


def p_mc(b, c):                                   # test_conv_lr.mcnemar_greater
    n = b + c
    return 1.0 if n == 0 else sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n


P4 = p_mc(4, 0)
PAT = list(itertools.product((0, 1), repeat=3))   # (BASE, KH, ARM) per seed


def compositions(n, k):
    for c in itertools.combinations(range(n + k - 1), k - 1):
        prev, out = -1, []
        for x in c:
            out.append(x - prev - 1); prev = x
        out.append(n + k - 2 - prev)
        yield out


import functools


@functools.lru_cache(None)
def stats_arrays(n):
    C = np.array(list(compositions(n, 8)), dtype=np.int64)
    idx = {p: i for i, p in enumerate(PAT)}
    cell = lambda cond: sum(C[:, idx[p]] for p in PAT if cond(*p))
    b1, c1 = cell(lambda B, K, A: K and not B), cell(lambda B, K, A: B and not K)
    b2, c2 = cell(lambda B, K, A: A and not B), cell(lambda B, K, A: B and not A)
    b3, c3 = cell(lambda B, K, A: K and not A), cell(lambda B, K, A: A and not K)
    return C, b1, c1, b2, c2, b3, c3


PV = {}


def pv(b, c):
    key = (b, c)
    if key not in PV:
        PV[key] = p_mc(b, c)
    return PV[key]


vp = np.vectorize(pv)


@functools.lru_cache(None)
def names(n, commit_below, kind):
    C, b1, c1, b2, c2, b3, c3 = stats_arrays(n)
    fails = vp(b1, c1) >= 0.05
    disc = vp(b2, c2) < 0.05
    within = (b3 - c3) <= 1
    needs = (vp(b3, c3) <= P4) if kind == "freq" else np.zeros(len(C), bool)
    clamps = ((b2 - c2) <= 1) & commit_below
    surv = disc & within
    return C, dict(fails=fails, surv=~fails & surv, needs=~fails & needs, clamps=~fails & clamps,
                   none=~fails & ~surv & ~needs & ~clamps, b3=b3, c3=c3, disc=disc, within=within, b1=b1, c1=c1, b2=b2, c2=c2)


for n in (20, 10):
    for kind in ("freq", "repeat"):
        for cb in (False, True):
            C, R = names(n, cb, kind)
            ov = {"survives & needs": int((R["surv"] & R["needs"]).sum()), "survives & clamps": int((R["surv"] & R["clamps"]).sum()),
                  "needs & clamps": int((R["needs"] & R["clamps"]).sum())}
            say(f"  n {n}, arm {kind}, commitment below BASELINE's {cb}: {len(C)} outcome classes (counts of the 8 per-seed "
                f"patterns); 'fails here' {int(R['fails'].sum())}; survives/replaces {int(R['surv'].sum())}; needs "
                f"{int(R['needs'].sum())}; clamps {int(R['clamps'].sum())}; NO NAME for this arm (KH discovered) "
                f"{int(R['none'].sum())}; overlaps {ov}")
# the gap's shape and S87's own record as a point in it
C, R = names(20, False, "freq")
gap = R["none"]
sub = Counter(("discovered" if d else "not discovered", "within 1" if w else f"KH-arm {int(b - c)}")
              for d, w, b, c in zip(R["disc"][gap], R["within"][gap], R["b3"][gap], R["c3"][gap]))
say(f"  freq, n 20, unnamed classes by (discovered?, KH only - arm only): {dict(sorted(Counter((k[0], k[1] if k[1] == 'within 1' else 'KH-arm >= 2') for k in sub.elements()).items()))}")
say(f"  S87's record (lite; BASELINE 0/20, KH 20/20, FREQ 17/20 nested): KH vs BASE 20/0 p {p_mc(20, 0):.2g}; FREQ vs BASE 17/0 "
    f"p {p_mc(17, 0):.2g}; KH vs FREQ 3/0 p {p_mc(3, 0):.3f} > {P4}: survives no (3 > 1), needs no -> under S89's rules "
    f"FREQ has no name, and 'no reading applies' is suppressed whenever any other arm's reading applies")
say(f"  smallest wins: discovered (p < 0.05) needs b - c >= 5 at c = 0 ({p_mc(5, 0):.4f}), (7, 1) {p_mc(7, 1):.4f}, "
    f"(8, 2) {p_mc(8, 2):.4f}; 'beats by >= 4 vs 0' (p <= {P4}): (4, 0) {p_mc(4, 0):.4f}, (5, 1) {p_mc(5, 1):.4f}, "
    f"(6, 1) {p_mc(6, 1):.4f}, (7, 2) {p_mc(7, 2):.4f}")
for nb in (0, 3, 6, 10, 15, 16):
    say(f"  BASELINE at {nb}/20 (independent of KH; KH 20/20): KH's best case vs BASE {20 - nb}/0 p {p_mc(20 - nb, 0):.3g} "
        f"-> 'the labelled hinge itself fails here' {'certain' if p_mc(20 - nb, 0) >= 0.05 else 'avoidable'}")

# ══ §5 operating characteristics ════════════════════════════════════════════════════════════════════════════════════
say("\n§5 OPERATING CHARACTERISTICS (exact: multinomial over the 8 per-seed patterns; 'independent' = BASE, KH, ARM independent "
    "per seed; 'paired' = KH and ARM comonotone on one uniform per seed (ARM's successes nested in KH's or vice versa), BASE "
    "independent)")
LOGF = [math.lgamma(i + 1) for i in range(64)]


def scenario(n, pB, pK, pA, coupled, kind="freq"):
    C, R = names(n, False, kind)
    probs = []
    for Bv, Kv, Av in PAT:
        pb = pB if Bv else 1 - pB
        if coupled:
            lo, hi = min(pK, pA), max(pK, pA)
            pka = {(1, 1): lo, (0, 0): 1 - hi, (1, 0): (pK - pA) if pK > pA else 0.0, (0, 1): (pA - pK) if pA > pK else 0.0}[(Kv, Av)]
        else:
            pka = (pK if Kv else 1 - pK) * (pA if Av else 1 - pA)
        probs.append(pb * pka)
    lp = np.array([math.log(p) if p > 0 else -np.inf for p in probs])
    with np.errstate(invalid="ignore"):
        logm = LOGF[n] - np.asarray(LOGF)[C].sum(1) + (C * np.where(np.isfinite(lp), lp, 0)).sum(1)
    zero = ((C > 0) & ~np.isfinite(lp)[None, :]).any(1)
    w = np.where(zero, 0.0, np.exp(logm))
    out = {k: float(w[R[k]].sum()) for k in ("fails", "surv", "needs", "none")}
    out["disc"] = float(w[R["disc"]].sum())
    out["tot"] = float(w.sum())
    return out


say(f"  {'n':>2} {'pB':>4} {'pK':>4} {'pA':>4} {'coupling':>11} | P(KH fails) P(A discovered) P(survives) P(needs) P(unnamed)")
for n in (20, 10):
    for pB in (0.0, 0.1):
        for pK in (0.9, 0.85):
            for pA in (0.9, 0.8, 0.7, 0.5):
                for cp in (False, True):
                    r = scenario(n, pB, pK, pA, cp)
                    say(f"  {n:>2} {pB:>4} {pK:>4} {pA:>4} {'paired' if cp else 'independent':>11} | {r['fails']:.3f}       "
                        f"{r['disc']:.3f}           {r['surv']:.3f}       {r['needs']:.3f}    {r['none']:.3f}   (sum {r['tot']:.6f})")
say("  (pB = 0.1 rows: BASELINE independent of both arms. 'unnamed' = KH discovered and none of survives / needs / clamps; clamps "
    "is 0 at these rates.)")

# ══ §6 seeds ════════════════════════════════════════════════════════════════════════════════════════════════════════
say("\n§6 SEEDS 2840-2899 (G's block; S89 uses 2840-2871)")
LO, HI = 2840, 2899
branches = [b.strip() for b in git("branch", "-r").splitlines() if "->" not in b]
pat_key = re.compile(rb"\|(28[4-9]\d)\|")
pat_seed = re.compile(rb'"seed"\s*:\s*(28[4-9]\d)\b')
pat_list = re.compile(rb'"seeds"\s*:\s*\[([^\]]*)\]')


def seeds_in(raw):
    hits = set(int(m) for m in pat_key.findall(raw)) | set(int(m) for m in pat_seed.findall(raw))
    for m in pat_list.findall(raw):
        hits |= {int(t) for t in re.findall(rb"\b(\d{3,5})\b", m) if LO <= int(t) <= HI}
    return hits


tip_hits = {}
for br in branches:
    tip = git("rev-parse", "--short", br).strip()
    files = [l for l in git("ls-tree", "-r", "--name-only", br).splitlines() if l.endswith((".json", ".jsonl"))]
    hits = {}
    for path in files:
        raw = show(br, path, binary=True)
        s = seeds_in(raw)
        try:                                        # full parse of every json / jsonl as a second route
            docs = [json.loads(raw)] if path.endswith(".json") else [json.loads(l) for l in raw.splitlines() if l.strip()]
        except Exception:
            docs = []

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    for part in str(k).split("|"):
                        if part.isdigit() and LO <= int(part) <= HI:
                            s.add(int(part))
                    if k in ("seed",) and isinstance(v, int) and LO <= v <= HI:
                        s.add(v)
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        for d in docs:
            walk(d)
        if s:
            hits[path] = sorted(s)
    tip_hits[br] = hits
    say(f"  {br} @ {tip}: {len(files)} json/jsonl files; seeds in 2840-2899: {hits or 'none'}")

# history: every json / jsonl / log / txt blob reachable from any remote branch
objs = git("rev-list", "--objects", "--remotes").splitlines()
paths = defaultdict(set)
for l in objs:
    parts = l.split(" ", 1)
    if len(parts) == 2 and parts[1].endswith((".json", ".jsonl", ".log", ".txt")):
        paths[parts[0]].add(parts[1])
proc = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
proc.stdin.write("".join(f"{h}\n" for h in paths).encode()); proc.stdin.close()
hist, nbytes = defaultdict(set), 0
pat_log = re.compile(rb"seed[ =:]+(28[4-9]\d)\b")
for _ in paths:
    head = proc.stdout.readline().split()
    size = int(head[2])
    raw = proc.stdout.read(size); proc.stdout.read(1)
    nbytes += size
    s = seeds_in(raw) | set(int(m) for m in pat_log.findall(raw))
    if s:
        for p in paths[head[0].decode()]:
            hist[p] |= s
proc.wait()
say(f"  history: {len(paths)} distinct json/jsonl/log/txt blobs ({nbytes / 1e9:.2f} GB) reachable from {len(branches)} remote "
    f"branches; seeds 2840-2899 found in: {dict((k, sorted(v)) for k, v in hist.items()) or 'none'}")
src_hits = []
for br in branches:
    try:
        o = git("grep", "-n", "-E", r"range\(28[4-9][0-9]|seed[s]? *[=:] *28[4-9][0-9]|\b28[4-9][0-9] *- *28[4-9][0-9]\b", br,
                "--", "*.py", "*.md", "*.sh")
    except subprocess.CalledProcessError:
        o = ""
    src_hits += [h for h in o.splitlines()]
say(f"  sources (.py/.md/.sh at every tip) naming 28xx seed ranges: {len(src_hits)} lines")
for h in sorted(set(h.split(':', 1)[1] if True else h for h in src_hits))[:40]:
    say(f"    {h[:200]}")

# ══ §7 the S87 record the CHECK names ═══════════════════════════════════════════════════════════════════════════════
say("\n§7 THE RECORD THE REPRODUCTION CHECK NAMES (child:218-234)")
st = json.loads(show(G, "explore_out/G/s87_results.json"))
k = [k for k in st["runs"] if k.startswith("REG3_FREQHINGE|2700|")]
r = st["runs"][k[0]]
say(f"  key {k}; store sha {st['meta']['sha'][:12]}, provenance git {st['meta']['provenance']['git']}, cpu {st['meta']['cpu']}")
say(f"  curve through 2400 {[c for c in r['curve'] if c[0] <= 2400]}; stats steps through 2400 "
    f"{[s['step'] for s in r['stats'] if isinstance(s['step'], int) and s['step'] <= 2400]}; transition {r.get('transition')}")
say(f"  s87_record() builds the key with ec.cpu_model() (child:221): on any CPU other than '{st['meta']['cpu']}' the CHECK "
    f"raises KeyError instead of printing; other S87 records on 2700 usable as references: "
    f"{sorted(x.split('|')[0] for x in st['runs'] if x.split('|')[1] == '2700')}")
for line in lines_of(git("log", "--format=%h %s", "-3", G, "--", "explore_out/G/s87_results.json"), " "):
    pass
say(f"  committed in: {git('log', '--format=%h %ad %s', '--date=short', '-1', G, '--', 'explore_out/G/s87_results.json').strip()[:120]}")

with open(__file__.replace(".py", "_out.txt"), "w") as fh:
    fh.write("\n".join(OUT) + "\n")
