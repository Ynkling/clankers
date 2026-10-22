#!/usr/bin/env python
"""
screen_checks_s90_F.py — Referee V's checks for screen_review_s90_F.md (Session F round 7: explore_f_tasks at 9017b823,
explore_f_s90_ctx / _child / explore_f_reports7 at 73749e8a). Reads git objects only; trains nothing; writes nothing but
its stdout (screen_checks_s90_F_out.txt). No torch here: the generators are replicated in numpy (structure only, own RNG),
and F's pure decision functions (decide, _arms, outcomes, reading_task, reading_screen) are exec'd by ast from 73749e8a on
in-memory stubs.

  A  the module: blobs, byte prefix, AST (old top-level definitions identical; nothing appended rebinds them)
  B  the generators (numpy replica): frequencies, both cuts, the repeat set, the positional residue, the last-write share,
     the no-memory bounds
  C  the cap of a fixed type -> channel map (exact over every split; simulated)
  D  the within-sequence statistics on synthetic gates (numpy replica of inv_from_gate_k and the margin)
  E  the readings: decide()/reading_task()/reading_screen() from 73749e8a on every outcome, stage by stage as run_s90.sh
  F  operating characteristics of 7/10 and 3/10, and of the 2-seed validity rule
  G  seeds 3060-3099 on every remote branch (tips and history) and the main line's seed list
  H  modules the S90 child imports: branch blob vs main line
"""
import ast
import itertools as it
import json
import math
import re
import subprocess
import sys
from collections import Counter

import numpy as np

REPO = "/home/user/clankers"
F_TASKS, F_SCREEN, OLD = "9017b823", "73749e8a", "b0c6332"
MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"


def git(*a, binary=False):
    r = subprocess.run(["git", "-C", REPO, *a], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode())
    return r.stdout if binary else r.stdout.decode()


def hdr(s):
    print("\n" + "=" * 100 + f"\n{s}\n" + "=" * 100)


# ══ A ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
hdr("A. The module: explore_f_tasks.py at 9017b823 against b0c6332")
old_b = git("show", f"{OLD}:explore_f_tasks.py", binary=True)
new_b = git("show", f"{F_TASKS}:explore_f_tasks.py", binary=True)
print(f"blob {OLD}: {git('rev-parse', OLD + ':explore_f_tasks.py').strip()}  ({len(old_b)} bytes, "
      f"{len(old_b.decode())} characters, {old_b.count(bytes([10]))} lines)")
print(f"blob {F_TASKS}: {git('rev-parse', F_TASKS + ':explore_f_tasks.py').strip()}  ({len(new_b)} bytes, "
      f"{new_b.count(bytes([10]))} lines)")
print(f"parent of {F_TASKS}: {git('rev-parse', F_TASKS + '^:explore_f_tasks.py').strip()}")
for c in (F_SCREEN, "origin/claude/explore-F"):
    print(f"blob at {c}: {git('rev-parse', c + ':explore_f_tasks.py').strip()}")
print(f"byte prefix: new.startswith(old) = {new_b.startswith(old_b)}; appended {len(new_b) - len(old_b)} bytes")
to = ast.parse(old_b.decode())
tn = ast.parse(new_b.decode())
old_src, new_src = old_b.decode(), new_b.decode()
seg = lambda src, n: ast.get_source_segment(src, n)
old_defs = {}
for n in to.body:
    nm = getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                      else (n.targets[0].elts[0].id if isinstance(n, ast.Assign) and isinstance(
                                          n.targets[0], ast.Tuple) else None))
    if nm:
        old_defs[nm] = seg(old_src, n)
n_old = len(to.body)
same_prefix_nodes = all(seg(old_src, a) == seg(new_src, b) for a, b in zip(to.body, tn.body[:n_old]))
print(f"top-level statements: old {n_old}, new {len(tn.body)}; the first {n_old} identical by source segment: "
      f"{same_prefix_nodes}")
appended = tn.body[n_old:]
bound = set()
for n in appended:
    for x in ast.walk(n):
        if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x in appended:
            bound.add(x.name)
    if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
        bound.add(n.name)
    elif isinstance(n, ast.Assign):
        for t in n.targets:
            for x in ast.walk(t):
                if isinstance(x, ast.Name):
                    bound.add(x.id)
print(f"appended top-level names ({len(bound)}): {sorted(bound)}")
print(f"  of them already defined above: {sorted(bound & set(old_defs))}")
# any appended code that assigns to an attribute of an old name, or calls setattr, at module level
mod_level_attr = []
for n in appended:
    if isinstance(n, (ast.Assign, ast.AugAssign, ast.Expr)):
        for x in ast.walk(n):
            if isinstance(x, ast.Attribute) and isinstance(x.ctx, ast.Store):
                mod_level_attr.append(seg(new_src, n))
            if isinstance(x, ast.Call) and getattr(x.func, "id", "") == "setattr":
                mod_level_attr.append(seg(new_src, n))
print(f"  module-level attribute stores / setattr in the appended code: {mod_level_attr or 'none'}")
# global statements in appended functions
glob = [x.names for n in appended for x in ast.walk(n) if isinstance(x, ast.Global)]
print(f"  'global' statements in the appended code: {glob or 'none'}")
# where the child's CHECK compares
ch = git("show", f"{F_SCREEN}:explore_f_s90_ctx_child.py")
i = ch.index('old = s89._git("show", "9017b82')
print("child CHECK (l.191-195):", "\n   ".join(ch[i:i + 520].splitlines()[:6]))

# ══ B ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
hdr("B. The generators (numpy replica of RandHeaderTask.make_batch + the context renaming; 20,000 sequences per task)")
P, NV, NB, NCTX = 4, 16, 2, 16
COMPS = [c for c in it.product(range(1, 4), repeat=NB) if sum(c) == P]


def make(S, B, rng):
    """lite's structure at S streams, then the S contexts renamed to distinct types of 16 by first appearance.
    Returns tokens (B, L) in the new vocabulary (CTX 0..15, KEY 16..19, VAL 20..35), the lite stream per position
    (body; -1 at the query), the index by first appearance per position, qpos."""
    body_len = S * (NB + 2 * P)
    L = body_len + 3
    toks = np.zeros((B, L), np.int64)
    lab = np.full((B, L), -1, np.int64)
    qstream = np.zeros(B, np.int64)
    for b in range(B):
        vals = np.stack([rng.permutation(NV)[:S] for _ in range(P)])          # (P, S)
        comp = [COMPS[rng.integers(len(COMPS))] for _ in range(S)]
        kord = [rng.permutation(P) for _ in range(S)]
        bord = rng.permutation(S * NB)
        q = rng.integers(S * P)
        seq, sl = [], []
        for blk in bord:
            s, i = divmod(int(blk), NB)
            st = sum(comp[s][:i])
            seq.append(("C", s)); sl.append(s)
            for kk in kord[s][st:st + comp[s][i]]:
                seq += [("K", int(kk)), ("V", int(vals[kk, s]))]
                sl += [s, s]
        qs, qk = divmod(int(q), P)
        seq += [("C", qs), ("K", qk), ("V", int(vals[qk, qs]))]
        sl += [qs, qs, qs]
        # first appearance rank of lite streams
        first = []
        for r, s in seq[:body_len]:
            if r == "C" and s not in first:
                first.append(s)
        rank = {s: first.index(s) for s in range(S)}
        pair = rng.permutation(NCTX)[:S]
        ctx_of = {s: int(pair[rank[s]]) for s in range(S)}
        for t, ((r, v), s) in enumerate(zip(seq, sl)):
            toks[b, t] = ctx_of[v] if r == "C" else (NCTX + v if r == "K" else NCTX + P + v)
            lab[b, t] = rank[s]
        qstream[b] = rank[qs]
    return toks, lab, qstream, body_len + 1


N = 20000
rng = np.random.default_rng(90)
DATA = {}
for S in (2, 3, 4):
    DATA[S] = make(S, N, rng)

role = lambda t: np.where(t < NCTX, 0, np.where(t < NCTX + P, 1, 2))      # 0 CTX, 1 KEY, 2 VAL
RN = ("CTX", "KEY", "VAL")


def largest_ratio(tok):
    cnt = np.bincount(tok.reshape(-1), minlength=36).astype(float)
    fr = cnt / cnt.sum()
    order = [int(t) for t in np.argsort(-fr, kind="stable") if fr[t] > 0]
    v = [fr[t] for t in order]
    cut = max(range(len(v) - 1), key=lambda i: v[i] / v[i + 1])
    return order[:cut + 1], fr, v[cut] / v[cut + 1]


def half_mass(tok):
    cnt = np.bincount(tok.reshape(-1), minlength=36).astype(float)
    fr = cnt / cnt.sum()
    order = [int(t) for t in np.argsort(-fr, kind="stable") if fr[t] > 0]
    acc, sel = 0.0, []
    for t in order:
        sel.append(t)
        acc += fr[t]
        if acc >= 0.5:
            break
    return sel, acc


for S in (2, 3, 4):
    toks, lab, qs, qp = DATA[S]
    inp = toks[:, :-1]
    T = inp.shape[1]
    nm = "CTXPOOL" if S == 2 else f"DOCS{S}"
    cnt = np.bincount(inp.reshape(-1), minlength=36) / inp.size
    print(f"\n{nm} (S = D = {S}): L {toks.shape[1]}, input {T} tokens: per sequence CTX {S * NB + 1}, KEY {S * P + 1}, VAL {S * P}")
    for r, (lo, hi) in enumerate(((0, NCTX), (NCTX, NCTX + P), (NCTX + P, 36))):
        exp = [(S * NB + 1) / T / NCTX, (S * P + 1) / T / P, S * P / T / NV][r]
        print(f"  {RN[r]}: {hi - lo} types, frequency {cnt[lo:hi].min():.4f}-{cnt[lo:hi].max():.4f} (expected {exp:.4f} each)")
    lr, fr, ratio = largest_ratio(inp)
    print(f"  largest-ratio cut: {len(lr)} types, roles {Counter(RN[int(role(np.array(t)))] for t in lr)}, ratio at the cut "
          f"{ratio:.2f}; its positions = the KEY positions in {np.mean([set(np.nonzero(np.isin(x, lr))[0]) == set(np.nonzero(role(x) == 1)[0]) for x in inp[:2000]]):.3f} of sequences")
    hm, mass = half_mass(inp)
    print(f"  half-mass cut: {len(hm)} types, roles {Counter(RN[int(role(np.array(t)))] for t in hm)}, mass {mass:.4f}; "
          f"expected mass of 4 KEY + 4 VAL = {(S * P + 1) / T + 4 * S * P / T / NV:.6f} (exactly 0.5)")
    # the half-mass cut's stability over independent 4096-sequence samples (F's sample size)
    nval = Counter()
    for rep in range(200):
        idx = np.random.default_rng(1000 + rep).choice(N, 4096, replace=False)
        h, _ = half_mass(inp[idx])
        nval[sum(1 for t in h if t >= NCTX + P)] += 1
    print(f"  half-mass cut over 200 resampled 4096-sequence samples: number of VAL types selected {dict(sorted(nval.items()))}"
          f" (which VAL types: the most frequent by sampling noise; all 16 have the same expected frequency)")
    # the repeat set
    rep_pos = np.zeros_like(inp, bool)
    for b in range(N):
        c = Counter(inp[b].tolist())
        rep_pos[b] = [c[t] >= 2 for t in inp[b]]
    rl = role(inp)
    per = {RN[r]: (rep_pos & (rl == r)).sum(1).mean() for r in range(3)}
    tot = {RN[r]: (rl == r).sum(1).mean() for r in range(3)}
    exact_keys = np.mean([(rep_pos[b] == (rl[b] == 1)).all() for b in range(N)])
    keys_ctx = np.mean([(rep_pos[b] == (rl[b] <= 1)).all() for b in range(N)])
    anyval = np.mean((rep_pos & (rl == 2)).any(1))
    print(f"  repeat set (type occurs >= 2 times in the input): CTX {per['CTX']:.2f} of {tot['CTX']:.0f}, KEY {per['KEY']:.2f} "
          f"of {tot['KEY']:.0f}, VAL {per['VAL']:.2f} of {tot['VAL']:.0f} positions; exactly the KEY positions in "
          f"{exact_keys:.4f} of sequences; exactly KEY + CTX in {keys_ctx:.4f}; a repeated VAL type in {anyval:.4f}")
    # positional residue of the index
    p0 = [(lab[:, t] == 0).mean() for t in range(T)]
    print(f"  P(index 0 | position) for positions 0-7: {[round(float(x), 3) for x in p0[:8]]}; index 0 at position 0 "
          f"always: {bool((lab[:, 0] == 0).all())}")
    # body KEY positions: the share of index-0 keys by order of appearance
    # last write share and no-memory bounds
    qk = inp[:, qp]
    tgt_kpos, last_kpos, val_in = [], [], []
    for b in range(N):
        body = inp[b, :qp - 1]
        hits = [t for t in range(qp - 1) if body[t] == qk[b]]
        tk = [t for t in hits if lab[b, t] == lab[b, qp]][0]
        tgt_kpos.append(tk)
        last_kpos.append(max(hits))
    tgt_kpos, last_kpos = np.array(tgt_kpos), np.array(last_kpos)
    print(f"  queries asking the last write of their key: {np.mean(tgt_kpos == last_kpos):.4f} (1/S = {1 / S:.4f}); "
          f"a last-write-only memory scores about this")
    for reach in (4, 10):
        pin = np.mean(tgt_kpos >= qp - reach + 1)
        pin_v = np.mean(tgt_kpos + 1 >= qp - reach + 1)
        print(f"  reach {reach}: P(target KEY and VAL inside) {pin:.3f} -> bound {pin + (1 - pin) / (NV - S + 1):.3f}; "
              f"P(target VAL inside) {pin_v:.3f} -> bound {pin_v + (1 - pin_v) / (NV - S + 1):.3f}; "
              f"chance-only floor 1/{NV} = {1 / NV:.4f}")

# ══ C ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
hdr("C. The cap of a fixed type -> channel map (exact over every split of the 16 types; balanced map simulated)")


def esym(sizes, D):
    e = [1] + [0] * D
    for n in sizes:
        for j in range(D, 0, -1):
            e[j] += e[j - 1] * n
    return e[D]


def comps(n, k):
    if k == 1:
        yield (n,)
        return
    for a in range(n + 1):
        for r in comps(n - a, k - 1):
            yield (a,) + r


for D in (2, 3, 4):
    tot = math.comb(16, D)
    for k in sorted({2, D, 2 * D, 16}):
        if k < D:
            print(f"  D {D}, k {k:>2}: cap 0 (fewer channels than documents)")
            continue
        if k <= 8:
            best = max(comps(16, k), key=lambda s: esym(s, D))
        else:
            best = tuple([1] * 16 + [0] * (k - 16)) if k >= 16 else None
        c = esym(best, D)
        bal = [sum(1 for t in range(16) if t * k // 16 == j) for j in range(k)]
        print(f"  D {D}, k {k:>2}: cap {c}/{tot} = {c / tot:.4f} (best split {sorted(best, reverse=True)}; balanced "
              f"{esym(bal, D)}/{tot})")
    # brute force at k = 2 over all 2^16 maps
    if D == 2:
        bestbf = max(sum(1 for a, b in it.combinations(range(16), 2) if ((m >> a) & 1) != ((m >> b) & 1))
                     for m in range(1 << 15))
        print(f"  D 2, k 2: brute force over all 2^15 maps (type 15 fixed): best {bestbf}/120")
    toks = DATA[D][0]
    bal = np.array([t * D // 16 for t in range(16)])
    sep = np.mean([len(set(bal[np.unique(x[x < 16])])) == D for x in toks])
    print(f"  D {D}: the balanced map separates {sep:.4f} of {N} replica sequences (cap {esym([sum(1 for t in range(16) if t * D // 16 == j) for j in range(D)], D) / tot:.4f})")
print("F's printed caps (s90 l.28-29, log CHECK lines): 0.5333 / 0.2679 / 0.1407 at k = D; k = 2D 0.800 / 0.664 / 0.615")

# ══ D ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
hdr("D. Within-sequence statistics (numpy replica of explore_f_tasks.inv_from_gate_k l.976-1000, ROUTED_ETA 0.9) and "
    "the margin (pool_routing_stats l.437-447) on synthetic gates, 20,000 replica sequences")


def inv_k(toks, lab, qp, gate, S):
    """gate: (B, T, k) read = write gate on the input tokens[:, :-1]."""
    inp = toks[:, :-1]
    B = inp.shape[0]
    kmask = role(inp[:, :qp - 1]) == 1
    tv, cs, sepf, marg = [], [], [], []
    for b in range(B):
        kp = np.nonzero(kmask[b])[0]
        vp = kp + 1
        lk, lv = lab[b, kp], lab[b, vp]
        mk = np.stack([gate[b, kp[lk == s]].mean(0) for s in range(S)])
        mv = np.stack([gate[b, vp[lv == s]].mean(0) for s in range(S)])
        pr = list(it.combinations(range(S), 2))
        tv.append(np.mean([0.5 * np.abs(mk[a] - mk[c]).sum() for a, c in pr]))
        cs.append(np.mean([mv[a] @ mv[c] / (np.linalg.norm(mv[a]) * np.linalg.norm(mv[c]) + 1e-30) for a, c in pr]))
        am = mk.argmax(1)
        sepf.append(len(set(am.tolist())) == S and (mk.max(1) >= 0.9).all())
        # margin: query read gate (at qp) against the write gate at the VAL of the target and the S-1 same-key distractors
        qk = inp[b, qp]
        same = [t for t in kp if inp[b, t] == qk]
        tg = [t for t in same if lab[b, t] == lab[b, qp]]
        ds = [t for t in same if lab[b, t] != lab[b, qp]]
        g = gate[b, qp]
        marg.append(gate[b, tg[0] + 1] @ g - np.mean([gate[b, t + 1] @ g for t in ds]))
    return dict(sep_key_inv=np.mean(tv), seq_separated_frac=np.mean(sepf), val_cos_inv_k=np.mean(cs), margin=np.mean(marg))


def onehot(ch, k):
    return np.eye(k)[ch]


for S in (2, 3, 4):
    toks, lab, qs, qp = DATA[S]
    inp = toks[:, :-1]
    M = 5000
    t_, l_ = toks[:M], lab[:M]
    i_ = inp[:M]
    B, T = i_.shape
    # latched CTX type per position
    lat = np.zeros_like(i_)
    for b in range(B):
        cur = i_[b, 0]
        for t in range(T):
            if i_[b, t] < 16:
                cur = i_[b, t]
            lat[b, t] = cur
    lab_in = l_[:, :-1]
    rngd = np.random.default_rng(7)
    perm = np.stack([rngd.permutation(S) for _ in range(B)])
    randmap = rngd.integers(S, size=16)
    pos_ch = np.minimum((np.arange(T) * S) // (qp - 1), S - 1)
    gates = {
        "relative (the perfect gate: one-hot of the index)": onehot(lab_in, S),
        "relative, channels permuted per sequence": onehot(np.take_along_axis(perm, lab_in, 1), S),
        f"balanced type-level map (k = {S})": onehot(np.array([t * S // 16 for t in range(16)])[lat], S),
        f"a random fixed type map (k = {S}; drawn once)": onehot(randmap[lat], S),
        "one channel per type (k = 16)": onehot(lat, 16),
        "all on one channel": onehot(np.zeros_like(lab_in), S),
        "position only (channel = body position x S // body length)": onehot(np.broadcast_to(pos_ch, lab_in.shape), S),
        "first document vs the rest (index 0 -> ch 0, else ch 1)": onehot(np.minimum(lab_in, 1), S),
        "uniform 1/k everywhere": np.full(lab_in.shape + (S,), 1.0 / S),
    }
    print(f"\n{'CTXPOOL' if S == 2 else 'DOCS' + str(S)} (S = {S}; {B} sequences): sep_key_inv / seq_separated_frac / "
          f"val_cos_inv_k / margin;  'DISCOVERED' = val_cos_inv_k < 0.5")
    for nm, g in gates.items():
        r = inv_k(t_, l_, qp, g, S)
        print(f"  {nm:<62} {r['sep_key_inv']:.3f} / {r['seq_separated_frac']:.3f} / {r['val_cos_inv_k']:.3f} / "
              f"{r['margin']:.3f}   {'DISCOVERED-form' if r['val_cos_inv_k'] < 0.5 else ''}")
    # expected values for a one-hot fixed type map with sizes n_j: P(two distinct types share a channel)
    sizes = [sum(1 for t in range(16) if t * S // 16 == j) for j in range(S)]
    coll = sum(n * (n - 1) for n in sizes) / (16 * 15)
    print(f"  expected for the balanced map: val_cos_inv_k = P(two of the sequence's types share a channel) = {coll:.4f}; "
          f"sep_key_inv = 1 - that = {1 - coll:.4f}; separated = cap")

# ══ E ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
hdr("E. The readings: decide / _arms / outcomes / reading_task / reading_screen exec'd from explore_f_s90_ctx.py at 73749e8a")
src = git("show", f"{F_SCREEN}:explore_f_s90_ctx.py")
tree = ast.parse(src)
WANT_F = {"decision", "_arms", "outcomes", "decide", "reading_task", "reading_screen"}
WANT_A = {"LR", "P0_SEEDS", "P2_SEEDS", "TASKS", "MEMS", "NOMEM_THR", "SCREEN_HI", "SCREEN_LO", "SLOW"}
code = []
for n in tree.body:
    if isinstance(n, ast.FunctionDef) and n.name in WANT_F:
        code.append(ast.get_source_segment(src, n))
    if isinstance(n, ast.Assign):
        names = {x.id for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name)}
        if names & WANT_A:
            code.append(ast.get_source_segment(src, n))
CODE = compile("\n\n".join(code), "explore_f_s90_ctx@73749e8a", "exec")
print(f"exec'd {len(code)} statements: " + ", ".join(sorted(WANT_F | WANT_A)))
import os
import tempfile
import types

TMP = tempfile.mkdtemp(prefix="s90F_")
STORE = {}


class _FC:
    F_DIR = TMP

    @staticmethod
    def setup(mods):
        pass

    @staticmethod
    def load(me):
        return {"runs": STORE}

    @staticmethod
    def runs(me, arm, seeds=None):
        return {int(k.split("|")[1]): r for k, r in STORE.items() if k.startswith(arm + "|") and r.get("ok")}


class _C16:
    MAX_ATTEMPTS = 3
    STATE = {}


def fresh_ns():
    ns = {"json": json, "os": os, "fc": _FC, "c16": _C16, "__name__": "s90stub"}
    me = types.SimpleNamespace()
    ns["sys"] = types.SimpleNamespace(modules={"s90stub": me})
    exec(CODE, ns)
    ns["DECISION"] = os.path.join(TMP, "s90_decision.json")
    ns["accept_other_cpus"] = lambda: []
    return ns


def put(arm, seed, outcome, acc=None):
    """outcome: 'B' bound, 'U' unbound, 'F' failed 3 times; for phase 0 an accuracy or 'F'."""
    k = f"{arm}|{seed}|sha|cpu"
    if outcome == "F":
        STORE[k] = dict(ok=False, attempts=3, error="x")
    elif outcome in ("B", "U"):
        STORE[k] = dict(ok=True, transition=1200 if outcome == "B" else None, acc=0.99 if outcome == "B" else 0.5)
    else:
        STORE[k] = dict(ok=True, transition=None, acc=float(outcome))


def stage(ns, gen):
    """run_s90.sh's stage: import (ARMS), run every queued (arm, seed) from gen, then --decide in a fresh import."""
    ns["ARMS"] = ns["_arms"]()
    for arm, a in ns["ARMS"].items():
        for s in a["seeds"]:
            if not any(k.startswith(f"{arm}|{s}|") for k in STORE):
                put(arm, s, gen(arm, s))
    arms_at_run = dict(ns["ARMS"])
    ns["ARMS"] = ns["_arms"]()
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        d = ns["decide"]()
    return arms_at_run


def pipeline(gen):
    STORE.clear()
    if os.path.exists(os.path.join(TMP, "s90_decision.json")):
        os.remove(os.path.join(TMP, "s90_decision.json"))
    ns = fresh_ns()
    hist = []
    for st in ("phase0", "phase1", "phase1b", "phase2", "final"):
        ns = fresh_ns()
        hist.append((st, stage(ns, gen)))
    ns = fresh_ns()
    ns["ARMS"] = ns["_arms"]()
    d = ns["decision"]()
    return ns, d, hist


TASKS3, MEMS2 = ("CTXPOOL", "DOCS3", "DOCS4"), ("HEBBCONV", "DELTA")


def indep_task(p0, p24, p48):
    if any(x != "F" and float(x) > 0.6 for x in p0):
        return "solvable without memory"
    if all(x == "B" for x in p24):
        return "the task is valid"
    if all(x == "B" for x in p48):
        return "the task needs the longer budget"
    return "untested"


tally, dis, errs = Counter(), 0, 0
budget_ok = True
single_stage = Counter()
P0V = ("0.6", "0.61", "F")
P1V = ("B", "U", "F")
for T in TASKS3:
    for Mm in MEMS2:
        for p0 in it.product(P0V, repeat=4):
            for p24 in it.product(P1V, repeat=2):
                for p48 in (it.product(P1V, repeat=2) if not all(x == "B" for x in p24) else [("B", "B")]):
                    def gen(arm, s, T=T, Mm=Mm, p0=p0, p24=p24, p48=p48):
                        sidx = s - 3060
                        if arm.startswith("NOMEM"):
                            if arm.endswith("_" + T):
                                return p0[(0 if arm.startswith("NOMEM_SELF") else 2) + sidx]
                            return "0.3"
                        if arm == f"O_{Mm}_{T}":
                            return p24[sidx]
                        if arm == f"O_{Mm}_{T}_48K":
                            return p48[sidx]
                        return "B"
                    try:
                        ns, d, hist = pipeline(gen)
                        got = ns["reading_task"](d, T, Mm)
                    except Exception as e:  # noqa
                        errs += 1
                        got = f"ERROR {e}"
                    want = indep_task(p0, p24, p48)
                    tally[(got)] += 1
                    dis += got != want
                    # the budget reaching the screen's arm, and when it is queued
                    b = d.get("budget", {}).get(T, {}).get(Mm)
                    a = f"SINGLE_{Mm}_{T}"
                    if isinstance(b, int):
                        budget_ok &= ns["ARMS"][a]["iters"] == b
                        for st, arms in hist:
                            if a in arms:
                                single_stage[(b, st)] += 1
                                break
                    else:
                        budget_ok &= a not in ns["ARMS"]
print(f"task readings over every outcome (3 tasks x 2 memories; phase-0 accuracies in {P0V} on the target's 4 runs; pilot "
      f"runs bound / unbound / failed x3 at 24K and at 48K): {sum(tally.values())} combinations")
for k, v in sorted(tally.items()):
    print(f"  {k:<36} {v}")
print(f"  disagreements with the docstring's rule (s90 l.49-56), applied independently: {dis}; errors {errs}")
print(f"  SINGLE_<M>_<T> exists iff the reading is valid / needs the longer budget, with iters = that budget: {budget_ok}")
print(f"  stage at which SINGLE_<M>_<T> is first queued: {dict(single_stage)}")

# the screen reading on every SINGLE_DELTA_CTXPOOL outcome, at each budget state of (CTXPOOL, DELTA)
scr = Counter()
dis2 = 0
ns = fresh_ns()
for bstate in (24000, 48000, "UNTESTED", "STOPPED (solvable without memory)", None):
    d = {"budget": {"CTXPOOL": {"DELTA": bstate}}} if bstate is not None else {"budget": {}}
    ns["ARMS"] = {"SINGLE_DELTA_CTXPOOL": dict(seeds=ns["P2_SEEDS"], opt="adam", iters=bstate)} if isinstance(bstate, int) else {}
    pats = (list(it.product(("B", "U", "F"), repeat=10)) + [tuple(None if j == i else "B" for j in range(10)) for i in range(10)]
            + [(None,) * 10]) if isinstance(bstate, int) else [(None,) * 10]
    for pat in pats:
        STORE.clear()
        for s, o in zip(ns["P2_SEEDS"], pat):
            if o is not None:
                put("SINGLE_DELTA_CTXPOOL", s, o)
        got = ns["reading_screen"](d)
        if not isinstance(bstate, int):
            want = "INCOMPLETE, no reading" if bstate is None else "untested"
        elif any(o is None for o in pat):
            want = "INCOMPLETE, no reading"
        else:
            n = sum(o == "B" for o in pat)
            want = ("one channel binds without a context identity" if n >= 7 else
                    "the partition's claim survives the context pool" if n <= 3 else "no reading applies")
        scr[(str(bstate), got)] += 1
        dis2 += got != want
print(f"\nscreen reading over every outcome of SINGLE_DELTA_CTXPOOL's 10 runs (bound / unbound / failed x3; or missing), "
      f"at each (CTXPOOL, DELTA) budget state (all 3^10 complete patterns, 10 with one record missing, 1 empty): {sum(scr.values())} combinations")
for k, v in sorted(scr.items()):
    print(f"  budget {k[0]:<36} {k[1]:<50} {v}")
print(f"  disagreements with the docstring (s90 l.67-72): {dis2}")
print("  note: a STOPPED (solvable-without-memory) CTXPOOL reads 'untested' in reading_screen (l.296-298), not "
      "'solvable without memory'")

# ══ F ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
hdr("F. Operating characteristics (exact binomial, 10 independent seeds; the 2-seed validity rule)")
binom = lambda n, k, p: math.comb(n, k) * p ** k * (1 - p) ** (n - k)
print("  p     P(>= 7/10) 'binds'   P(<= 3/10) 'survives'   P(4-6) 'no reading'   P(valid: 2/2)   P(valid or 48K, p48 = p)")
for p in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    hi = sum(binom(10, k, p) for k in range(7, 11))
    lo = sum(binom(10, k, p) for k in range(0, 4))
    print(f"  {p:<5} {hi:>9.4f}            {lo:>9.4f}              {1 - hi - lo:>9.4f}         {p * p:>8.4f}      "
          f"{p * p + (1 - p * p) * p * p:>8.4f}")
# the probability that the reading is 'survives' is P(<=3/10) * P(delta pilot valid or 48K)
print("  'survives' also needs the delta perfect gate valid (2/2 at 24K or at 48K); at a perfect-gate rate q the "
      "conjunct holds with q^2 + (1 - q^2) q48^2")

# ══ G ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
hdr("G. Seeds 3060-3099: every remote branch tip and history; the main line's seed list")
SEEDS = range(3060, 3100)
pat = re.compile(r'(?:"[A-Za-z0-9_]+\|(30[6-9]\d)\|)|(?:"seed":\s*(30[6-9]\d)\b)|(?:\bseed[s]?\s+(30[6-9]\d)\b)')
branches = [b.strip() for b in git("branch", "-r").splitlines() if "->" not in b]
print(f"{len(branches)} remote branches: {', '.join(branches)}")
hits = Counter()
for br in branches:
    files = [f for f in git("ls-tree", "-r", "--name-only", br).splitlines() if f.endswith((".json", ".jsonl", ".log"))]
    for f in files:
        try:
            txt = git("show", f"{br}:{f}")
        except Exception:  # noqa
            continue
        for m in pat.finditer(txt):
            hits[(br, f)] += 1
print("tip hits (branch, file, count):")
for (br, f), c in sorted(hits.items()):
    print(f"  {br} {f} {c}")
if not hits:
    print("  none")
# history: every json/jsonl/log blob reachable from any remote branch
objs = git("rev-list", "--objects", "--remotes").splitlines()
blobs = {}
for ln in objs:
    parts = ln.split(" ", 1)
    if len(parts) == 2 and parts[1].endswith((".json", ".jsonl", ".log")):
        blobs.setdefault(parts[0], parts[1])
print(f"history: {len(blobs)} distinct json/jsonl/log blobs reachable from the remotes")
p = subprocess.Popen(["git", "-C", REPO, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
out, _ = p.communicate(("\n".join(blobs) + "\n").encode())
hist_hits = Counter()
pos = 0
for sha in blobs:
    header_end = out.index(b"\n", pos)
    hdr_ = out[pos:header_end].split()
    size = int(hdr_[2])
    body = out[header_end + 1:header_end + 1 + size].decode("utf-8", "replace")
    pos = header_end + 1 + size + 1
    n = len(pat.findall(body))
    if n:
        hist_hits[blobs[sha]] += n
print("history hits (path: count over blobs): " + (", ".join(f"{k}: {v}" for k, v in sorted(hist_hits.items())) or "none"))
# the main line's literal range() seeds in test_*.py, and the attempt seeds s + 1000 j (j < 5)
tests = [f for f in git("ls-tree", "--name-only", MAIN).splitlines() if re.match(r"test_.*\.py$", f)]
lit = set()
for f in tests:
    t = ast.parse(git("show", f"{MAIN}:{f}"))
    for x in ast.walk(t):
        if isinstance(x, ast.Call) and getattr(x.func, "id", "") == "range" and all(isinstance(a, ast.Constant) for a in x.args):
            try:
                lit |= set(range(*[a.value for a in x.args]))
            except TypeError:
                pass
att = {s + 1000 * j for s in lit for j in range(5)}
print(f"main line {MAIN} ({git('rev-parse', '--short', MAIN).strip()}): {len(tests)} test_*.py; literal range() integers "
      f"{len(lit)}; with s + 1000 j (j < 5) {len(att)}; in 3060-3099: {sorted(set(SEEDS) & att) or 'none'}")
print("restart bases in the main line's restart tests: 160-189 (test_load_curriculum l.259-261), 200-219 "
      "(test_curriculum_confirm l.254-257), 240-259 (test_stream_recipe l.278-280), 260-279 (test_stream_curriculum "
      "l.268-269); test_router_reliability's trials 1000 + 5i + j (l.62, l.214): none reaches 3060-3099")
# G and H round-7 blocks
print("round-7 blocks (prompt l.1274): G 2960-3019, H 3020-3059, F 3060-3099: disjoint")

# ══ H ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
hdr("H. Modules the S90 child's code SHA covers (log: 27 modules): F's blob at 73749e8a vs the main line")
mods = ("bdh bdh_recurrent explore_b16_gates explore_common explore_common2 explore_delta_mem explore_f_delta_fast "
        "explore_f_s89_tasks_child explore_f_s90_ctx_child explore_f_tasks explore_f_window_delta_child explore_local_gate "
        "explore_window_recipe_child test_binding_capacity test_binding_onset test_binding_recipe test_channel_binding "
        "test_instrument_v2 test_interference test_multilayer_binding test_readout_path test_router_confirm "
        "test_router_curriculum test_router_discovery test_router_layout test_router_reliability test_short_conv").split()
for m_ in mods:
    a = git("rev-parse", f"{F_SCREEN}:{m_}.py").strip()
    try:
        b = git("rev-parse", f"{MAIN}:{m_}.py").strip()
    except Exception:  # noqa
        b = "absent"
    print(f"  {m_:<32} F {a[:10]}  main {b[:10]}  {'same' if a == b else ('branch-only' if b == 'absent' else 'DIFFERENT')}")
for m_ in ("explore_f_s89_tasks_child", "explore_f_s90_ctx_child", "explore_f_s90_ctx", "explore_f_reports7"):
    log = git("log", "--format=%h %ci", "origin/claude/explore-F", "--", f"{m_}.py").splitlines()
    print(f"  commits touching {m_}.py on explore-F: {log[:4]}")
print("\ndone")
