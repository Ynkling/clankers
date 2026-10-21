#!/usr/bin/env python
"""screen_checks_s89_F_v2.py — Referee V: checks for screen_review_s89_F_v2.md (Session F's S89-F revision b0c6332 /
3dd5b74 and the follow-up pilots explore_f_s89_post, claude/explore-F up to 5357889).

Reads git objects only, trains nothing, writes nothing (it prints; run with stdout to screen_checks_s89_F_v2_out.txt).
No torch here: the swap-invariant statistics (explore_f_tasks.eta2_rows / inv_from_gate at b0c6332), the pooled ones
(pool_routing_stats' eta_key_by_stream and margin, role_cosines' VAL cos) and the generators are replicated line by
line in numpy. F's pure functions pool_outcome (b0c6332) and reading (explore_f_s89_post at 3dd5b74) are extracted by
ast and exec'd on the records / synthetic inputs, with a stub module carrying DISC_COS read from the source."""
import ast
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import types

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rlib  # noqa: E402

ROOT = rlib.ROOT
F = "origin/claude/explore-F"
REV_MOD, OLD_MOD = "b0c6332b", "50dc8e4"
REV, Q, REC, TIP = "3dd5b746", "4c8a343d", "73035d37", "53578896"
S89_TIP = "52e2ea70"


def git(*a, text=True):
    out = subprocess.check_output(("git", "-C", ROOT) + a, stderr=subprocess.DEVNULL)
    return out.decode() if text else out


def show(ref, path, text=True):
    return git("show", f"{ref}:{path}", text=text)


def H(t):
    print("\n" + "=" * 112 + f"\n{t}\n" + "=" * 112)


def extract(src, names):
    tree = ast.parse(src)
    out = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            out.append(ast.get_source_segment(src, n))
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets):
            out.append(ast.get_source_segment(src, n))
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Tuple) and any(getattr(e, "id", None) in names
                                                                                 for e in t.elts) for t in n.targets):
            out.append(ast.get_source_segment(src, n))
    return "\n\n".join(out)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
H("A  The module change: b0c6332's explore_f_tasks.py against 50dc8e4's (byte prefix)")
old = show(OLD_MOD, "explore_f_tasks.py", text=False)
new = show(REV_MOD, "explore_f_tasks.py", text=False)
print(f"  50dc8e4 file: {len(old)} bytes = {len(old.decode())} characters; b0c6332 file: {len(new)} bytes = "
      f"{len(new.decode())} characters")
print(f"  b0c6332's first {len(old)} bytes equal 50dc8e4's file: {new[:len(old)] == old}; appended {len(new) - len(old)} bytes "
      f"= {len(new.decode()) - len(old.decode())} characters, {new[len(old):].count(b'\n')} lines")
print(f"  (F's CHECK label prints len(str): 'first 34603 bytes' and 'appended 13854 bytes' are characters)")
print(f"  blobs: 50dc8e4 {git('rev-parse', OLD_MOD + ':explore_f_tasks.py').strip()[:10]}, b0c6332 "
      f"{git('rev-parse', REV_MOD + ':explore_f_tasks.py').strip()[:10]}")
for ref in (REV, Q, REC, "2279ca1f", TIP):
    print(f"  explore_f_tasks.py at {ref[:7]}: blob {git('rev-parse', ref + ':explore_f_tasks.py').strip()[:10]}")
src_new = new.decode()
tree = ast.parse(src_new)
tree_old = ast.parse(old.decode())
names_old = {getattr(n, 'name', None) for n in tree_old.body} - {None}
names_new = [getattr(n, 'name', None) for n in tree.body if getattr(n, 'name', None) and n.name not in names_old]
redef = [getattr(n, 'name', None) for n in tree.body[len(tree_old.body):]
         if getattr(n, 'name', None) in names_old]
asg_old = {t.id for n in tree_old.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
asg_new = [t.id for n in tree.body[len(tree_old.body):] if isinstance(n, ast.Assign) for t in n.targets
           if isinstance(t, ast.Name)]
print(f"  appended top-level defs: {names_new}; appended names that redefine an old one: {redef}; "
      f"appended assignments {asg_new}, of which rebind an old name: {[a for a in asg_new if a in asg_old]}")
print("  G's tip holds the b0c6332 blob:", git("rev-parse", "origin/claude/explore-G:explore_f_tasks.py").strip()[:10],
      "; H's tip has explore_f_tasks.py:", bool(subprocess.run(("git", "-C", ROOT, "cat-file", "-e",
                                                                 "origin/claude/explore-H:explore_f_tasks.py"),
                                                                stderr=subprocess.DEVNULL).returncode == 0))

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
H("B  The swap-invariant statistics on synthetic gates (numpy replica of b0c6332 l.663-710, pool_routing_stats l.414-447,"
  "\n   role_cosines; generators of 50dc8e4, as screen_checks_s89_F.py §E)")
S, P, NV, NB, KP = 2, 4, 16, 2, 32
COMPS = [c for c in itertools.product(range(1, 4), repeat=2) if sum(c) == P]
rng = np.random.default_rng(8902)


def lite_seq():
    vals = [rng.permutation(NV)[:S] for _ in range(P)]
    comp = [COMPS[rng.integers(len(COMPS))] for _ in range(S)]
    korder = [rng.permutation(P) for _ in range(S)]
    border = rng.permutation(S * NB)
    q = rng.integers(S * P)
    seq = []                                  # (role, lite id, lite stream)
    for blk in border:
        s_, i_ = divmod(int(blk), NB)
        st = sum(comp[s_][:i_])
        seq.append(("C", s_, s_))
        for kk in korder[s_][st:st + comp[s_][i_]]:
            seq.append(("K", int(kk), s_))
            seq.append(("V", int(vals[kk][s_]), s_))
    qs, qk = q // P, q % P
    seq += [("C", qs, qs), ("K", qk, qs)]     # the input ends at the query KEY (22 tokens)
    return seq


def batch(kind, n):
    """token ids in the task's vocabulary (n, 22), and the label array as the task defines it."""
    nctx = 16 if kind == "allpool" else 2
    key0, val0 = nctx, nctx + KP
    X = np.zeros((n, 22), int)
    for b in range(n):
        seq = lite_seq()
        keyset = rng.permutation(KP)[:P]
        if kind == "allpool":
            pair = rng.permutation(16)[:S]
            first = seq[0][1]
            ctxset = {first: int(pair[0]), 1 - first: int(pair[1])}
        for t, (r_, v, _) in enumerate(seq):
            if r_ == "C":
                X[b, t] = ctxset[v] if kind == "allpool" else v
            elif r_ == "K":
                X[b, t] = key0 + keyset[v]
            else:
                X[b, t] = val0 + v
    return X, nctx


def latest_ctx(X, nctx):
    isc = X < nctx
    idx = np.where(isc, np.arange(X.shape[1])[None, :], -1)
    last = np.maximum.accumulate(idx, axis=1)
    return np.take_along_axis(X, np.clip(last, 0, None), 1), last


def stream_labels(X, nctx, kind):
    cur, last = latest_ctx(X, nctx)
    lab = (cur != X[:, :1]).astype(int) if kind == "allpool" else cur
    return np.where(last >= 0, lab, -1)


def positions(X, nctx):
    body = X[:, :20]
    isk = (body >= nctx) & (body < nctx + KP)
    assert (isk.sum(1) == S * P).all()
    kpos = np.sort(np.where(isk, np.arange(20)[None, :], 10 ** 9), 1)[:, :S * P]
    return kpos, kpos + 1


def eta2_rows(x, lab):                         # b0c6332 l.663-674, line for line
    x = x.astype(np.float64)
    mu = x.mean(1)
    tot = ((x - mu[:, None]) ** 2).sum(1)
    between = np.zeros_like(tot)
    for g in np.unique(lab):
        m = (lab == g).astype(np.float64)
        cnt = m.sum(1)
        mg = (x * m).sum(1) / np.clip(cnt, 1, None)
        between = between + np.where(cnt > 0, cnt * (mg - mu) ** 2, 0.0)
    return np.where(tot > 0, between / np.where(tot > 0, tot, 1.0), 0.0)


def eta2(x, lab):                              # test_router_layout.eta2 (pooled)
    x, lab = np.asarray(x, np.float64).ravel(), np.asarray(lab).ravel()
    mu = x.mean()
    tot = ((x - mu) ** 2).sum()
    if tot <= 0:
        return 0.0
    return float(sum((lab == g).sum() * (x[lab == g].mean() - mu) ** 2 for g in np.unique(lab)) / tot)


def cosv(a, b):
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    return (a * b).sum(-1) / np.maximum(na * nb, 1e-8)       # F.cosine_similarity, eps 1e-8


def stats(G, X, nctx, kind):
    """pooled: eta_key_by_stream, VAL cos (role_cosines), margin; within-sequence: inv_from_gate."""
    lab = stream_labels(X, nctx, kind)
    kpos, vpos = positions(X, nctx)
    p = G[..., 0]
    ctx = np.take_along_axis(lab, kpos, 1)
    out = dict(eta_pooled=eta2(np.take_along_axis(p, kpos, 1), ctx))
    isv = X >= nctx + KP
    a = G[(lab == 0) & isv].mean(0)
    b = G[(lab == 1) & isv].mean(0)
    out["cos_pooled"] = float(cosv(a, b))
    # margin (pool_routing_stats l.437-446)
    qpos = 21
    qs, qk = lab[:, qpos], X[:, qpos]
    key = np.take_along_axis(X, kpos, 1)
    same = key == qk[:, None]
    tgt, dis = same & (ctx == qs[:, None]), same & (ctx != qs[:, None])
    assert (tgt.sum(1) == 1).all() and (dis.sum(1) == 1).all()
    gwv = np.take_along_axis(G, vpos[..., None].repeat(2, -1), 1)
    dots = (gwv * G[:, qpos][:, None, :]).sum(-1)
    out["margin"] = float(((dots * tgt).sum(1) - (dots * dis).sum(1)).mean())
    # within-sequence (inv_from_gate l.681-696)
    eta = eta2_rows(np.take_along_axis(p, kpos, 1), ctx)
    gv = np.take_along_axis(G, vpos[..., None].repeat(2, -1), 1)
    lv = np.take_along_axis(lab, vpos, 1)
    means = []
    for s in range(2):
        m = (lv == s).astype(float)[..., None]
        means.append((gv * m).sum(1) / np.clip(m.sum(1), 1, None))
    c = cosv(means[0], means[1])
    out.update(eta_inv=float(eta.mean()), cos_inv=float(c.mean()), seq_routed=float((eta > 0.9).mean()))
    out["routed_pooled"] = out["margin"] >= 0.9 and out["eta_pooled"] > 0.9
    out["routed_inv"] = out["margin"] >= 0.9 and out["eta_inv"] > 0.9
    out["disc_pooled"] = out["cos_pooled"] < 0.5
    out["disc_inv"] = out["cos_inv"] < 0.5
    return out


def onehot(ch):
    return np.stack([1.0 - ch, ch], -1).astype(float)


N = 20000
data = {k: batch(k, N) for k in ("pool", "allpool")}
gates = {}


def mk(kind):
    X, nctx = data[kind]
    lab = stream_labels(X, nctx, kind)
    cur, last = latest_ctx(X, nctx)
    B, T = X.shape
    flip = rng.random(B) < 0.5
    lo = np.where(X < nctx, X, 10 ** 6).min(1)
    nctx_seen = np.cumsum(X < nctx, 1)
    isk = (X >= nctx) & (X < nctx + KP)
    kidx = np.where(isk, np.arange(T)[None, :], -1)
    lastk = np.maximum.accumulate(kidx, 1)
    curk = np.take_along_axis(X, np.clip(lastk, 0, None), 1)
    pos_p0 = np.array([(lab[:, t] == 0).mean() for t in range(T)])
    g = {}
    g["relative (task.perfect_gate)"] = onehot(lab)
    g["relative, channels swapped in a random half"] = onehot(np.where(flip[:, None], 1 - lab, lab))
    g["type-ordered (lower CTX id -> ch 0)"] = onehot((cur != lo[:, None]).astype(int))
    bm = np.array([int(t >= 8) for t in range(16)]) if kind == "allpool" else np.array([0, 1])
    g["type-level, balanced map (BALANCED_MAP)"] = onehot(bm[cur])
    g["type-level soft map (0.8 / 0.2)"] = 0.2 + 0.6 * onehot(bm[cur])
    # source-blind
    g["BLIND uniform 0.5"] = np.full((B, T, 2), 0.5)
    g["BLIND position half (t < 10 -> ch 0)"] = onehot((np.arange(T)[None, :] >= 10).astype(int).repeat(B, 0))
    g["BLIND block parity (# CTX seen mod 2)"] = onehot((nctx_seen + 1) % 2)
    g["BLIND latest key-type parity"] = np.where((lastk >= 0)[..., None], onehot(curk % 2), 0.5)
    g["BLIND one channel per sequence, random"] = onehot(np.repeat((rng.random(B) < 0.5).astype(int)[:, None], T, 1))
    g["BLIND iid random one-hot per position"] = onehot((rng.random((B, T)) < 0.5).astype(int))
    g["BLIND best position-only soft gate P(label 0 | t)"] = np.stack([pos_p0, 1 - pos_p0], -1)[None].repeat(B, 0)
    # partial / noisy routing
    for f_ in (0.85, 0.9, 0.95):
        sel = rng.random(B) < f_
        g[f"relative in {f_:.0%} of sequences, uniform elsewhere"] = np.where(sel[:, None, None], onehot(lab), 0.5)
    for sd in (0.10, 0.20, 0.30):
        p0 = np.clip(np.where(lab == 0, 0.8, 0.2) + rng.normal(0, sd, (B, T)), 0, 1)
        g[f"relative 0.8/0.2 + iid N(0, {sd:.2f}) per position"] = np.stack([p0, 1 - p0], -1)
    for sd in (0.05, 0.15):
        p0 = np.clip(np.where(lab == 0, 1.0, 0.0) + np.where(lab == 0, -1, 1) * np.abs(rng.normal(0, sd, (B, T))), 0, 1)
        g[f"relative one-hot with a |N(0, {sd:.2f})| leak per position"] = np.stack([p0, 1 - p0], -1)
    off = rng.uniform(-0.25, 0.25, B)[:, None]
    p0 = np.clip(np.where(lab == 0, 0.8, 0.2) + off, 0, 1)
    g["relative 0.8/0.2 + a per-sequence offset U(-0.25, 0.25)"] = np.stack([p0, 1 - p0], -1)
    return g


fmt = lambda v: f"{v:.4f}"
for kind in ("pool", "allpool"):
    X, nctx = data[kind]
    print(f"\n  {kind.upper()} ({N} sequences, numpy replica; stream index = "
          f"{'first appearance' if kind == 'allpool' else 'CTX id'})")
    print(f"  {'gate':58s} {'margin':>7} {'eta pool':>8} {'eta inv':>8} {'cos pool':>8} {'cos inv':>8} {'seq>0.9':>7} "
          f" ROUTED pooled/inv  DISC-if-bound pooled/inv")
    for nm, G in mk(kind).items():
        o = stats(G, X, nctx, kind)
        print(f"  {nm:58s} {o['margin']:7.4f} {o['eta_pooled']:8.4f} {o['eta_inv']:8.4f} {o['cos_pooled']:8.4f} "
              f"{o['cos_inv']:8.4f} {o['seq_routed']:7.4f}  {str(o['routed_pooled']):>6}/{str(o['routed_inv']):<6}"
              f"      {str(o['disc_pooled']):>6}/{str(o['disc_inv']):<6}")

print("\n  Within-sequence eta^2 on 8 points (4 per stream) under the null (a gate independent of the stream): mean")
null = eta2_rows(rng.random((200000, 8)), np.array([0, 1] * 4)[None].repeat(200000, 0))
print(f"  {null.mean():.4f} (theory (g-1)/(n-1) = 1/7 = {1 / 7:.4f}); P(eta^2_seq > 0.9) {np.mean(null > 0.9):.5f}")
print("  Positive small-sample bias: a gate with the same signal-to-noise everywhere reads higher within-sequence than pooled")
print("  (rows 'relative 0.8/0.2 + iid N(0, sd)' above); the per-sequence offset row reads lower pooled.")

print("\n  ALLPOOL's cap: max over all 2^16 maps of 16 CTX types to 2 channels of the fraction of the C(16,2) pairs separated")
best = max(a * (16 - a) for a in range(17)) / 120
X, nctx = data["allpool"]
cur, _ = latest_ctx(X, nctx)
c0 = X[:, 0]
c1 = np.array([[t for t in row[:20] if t < 16 and t != row[0]][0] for row in X])
sep = np.mean((c0 >= 8) != (c1 >= 8))
print(f"  max a(16-a)/120 = {best:.4f} (64/120 = {64 / 120:.4f}); replica, BALANCED_MAP separates {sep:.4f}; F's probe 256 "
      f"(seed 2800) 0.5117 = 131/256, record 2801 0.4922 = 126/256; binomial sd at 256: "
      f"{math.sqrt(best * (1 - best) / 256):.4f}")
print(f"  (z of 131/256: {(131 / 256 - best) / math.sqrt(best * (1 - best) / 256):+.2f}, of 126/256: "
      f"{(126 / 256 - best) / math.sqrt(best * (1 - best) / 256):+.2f})")

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
H("C  pool_outcome (b0c6332) and reading (explore_f_s89_post at 3dd5b74), exec'd by ast")
trd = show(REV, "test_router_discovery.py")
disc = re.search(r"^DISC_COS\s*=\s*([0-9.]+)", trd, re.M).group(1)
stub = types.ModuleType("test_router_reliability")
stub.DISC_COS = float(disc)
sys.modules["test_router_reliability"] = stub
ns = {"ROUTED_MARGIN": 0.9, "ROUTED_ETA": 0.9}
exec(extract(src_new, {"ROUTED_MARGIN", "ROUTED_ETA"}), ns)
exec(extract(src_new, {"pool_outcome"}), ns)
pool_outcome = ns["pool_outcome"]
print(f"  DISC_COS (test_router_discovery at 3dd5b74) = {disc}; ROUTED_MARGIN, ROUTED_ETA = {ns['ROUTED_MARGIN']}, {ns['ROUTED_ETA']}")
print("  None guard: pool_outcome({}) ->", pool_outcome({}, True), "; end with margin None ->",
      pool_outcome(dict(transition=1, end=dict(margin=None, eta_key_by_stream_inv=1.0, val_cos_inv=0.0)), True)["routed_end"])
post_src = show(REV, "explore_f_s89_post.py")
ns2 = {}
exec(extract(post_src, {"CONV_ARMS", "SEEDS"}), ns2)
exec(extract(post_src, {"reading"}), ns2)
reading = ns2["reading"]
print("  reading(n_bound, n_failed, n_ok) over every complete case (n_ok + n_failed = 4):")
for nf in range(5):
    print("   ", "; ".join(f"{nb} bound, {nf} failed -> {reading(nb, nf, 4 - nf)}" for nb in range(0, 5 - nf)))
print("  incomplete:", reading(3, 0, 3))
print("\n  Operating characteristic of the 4-run reading (each run binds independently with probability p):")
print("  " + "  ".join(f"p={p:.2f}: restores {p ** 4:.3f}, partly {4 * p ** 3 * (1 - p):.3f}, does not "
                       f"{1 - p ** 4 - 4 * p ** 3 * (1 - p):.3f}" for p in (0.95, 0.9, 0.75, 0.5)))

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
H("D  The follow-up records (s89_post_results.json at 73035d3) against the addendum, the README and the tables")
post = json.loads(show(REC, "explore_out/F/s89_post_results.json"))["runs"]
s89 = json.loads(show(S89_TIP, "explore_out/F/s89_tasks_results.json"))["runs"]


def get(store, arm, seed):
    ks = [k for k in store if k.startswith(f"{arm}|{seed}|")]
    assert len(ks) == 1, (arm, seed, ks)
    return store[ks[0]]


print("  meta: code_sha", {r["code_sha"][:12] for r in post.values()}, "cpu", {r["cpu"] for r in post.values()},
      "torch", {r["torch"] for r in post.values()}, "git", sorted({r["git"] for r in post.values()}),
      "attempts", {r["attempts"] for r in post.values()}, "ok", {r["ok"] for r in post.values()})
tot_wall = sum(r["secs_wall"] for r in post.values())
print(f"  run time: sum secs_wall {tot_wall:.0f} s = {tot_wall / 3600:.3f} h; sum secs {sum(r['secs'] for r in post.values()) / 3600:.3f} h")
ARMS = ("O_TYPE_HEBB_ALLPOOL", "O_HEBB_CONV_POOL", "O_HEBB_CONV_ALLPOOL")
nb_conv = 0
for arm in ARMS:
    for sd in (2800, 2801):
        r = get(post, arm, sd)
        tr = rlib.transition(r["curve"])
        e = r["end"]
        w = r.get("write_order") or {}
        po, pi = pool_outcome(r, False), pool_outcome(r, True)
        same_po = po == r["outcome_pooled"]
        same_pi = pi == r["outcome_inv"]
        if arm != "O_TYPE_HEBB_ALLPOOL":
            nb_conv += tr is not None
        cv = [c[1] for c in r["curve"]]
        print(f"  {arm}|{sd}: transition record {r['transition']} recomputed {tr}; final acc {r['acc']:.4f}; stopped_at "
              f"{r.get('stopped_at')}; margin {e['margin']:.4f}; eta pooled {e['eta_key_by_stream']:.4f} inv "
              f"{e['eta_key_by_stream_inv']:.4f}; VAL cos pooled {r['val_cos']:.4f} inv {e['val_cos_inv']:.4f}; seq routed "
              f"{e['seq_routed_frac']:.4f}; last/not-last {w.get('acc_last', float('nan')):.4f}/"
              f"{w.get('acc_notlast', float('nan')):.4f}; outcome_pooled == pool_outcome {same_po}, outcome_inv == {same_pi}; "
              f"curve min/max from 2400 {min(cv[1:]):.4f}/{max(cv[1:]):.4f}, first eval {cv[0]:.4f}")
        if arm == "O_TYPE_HEBB_ALLPOOL":
            print(f"      margin = eta_inv = seq_routed = 1 - val_cos_inv exactly: {e['margin'] == e['eta_key_by_stream_inv'] == e['seq_routed_frac'] == 1 - e['val_cos_inv']}"
                  f" ({e['margin'] * 256:.1f}/256); key_sep {e['key_sep']}, key_part {e['key_part']} (lite ids = CTX types 2-5 here)")
print(f"  conv pilots bound: {nb_conv}/4; Wilson {rlib.wilson(nb_conv, 4)}; band {rlib.band(nb_conv, 4)}; reading -> "
      f"{reading(nb_conv, 0, 4)!r}")
for arm, n in (("O_HEBB_CONV_POOL", 2), ("O_HEBB_CONV_ALLPOOL", 1), ("O_TYPE_HEBB_ALLPOOL", 0)):
    print(f"  {arm}: {n}/2 Wilson {tuple(round(x, 2) for x in rlib.wilson(n, 2))} band {rlib.band(n, 2)}")
c = get(post, "O_HEBB_CONV_ALLPOOL", 2801)["curve"]
print("  O_HEBB_CONV_ALLPOOL|2801 curve:", [(s, round(a, 3)) for s, a, *_ in c])
print("  every 2400 (tables):", " ".join(f"{a:.2f}" for s, a, *_ in c[1::2]))
print("  max 9600..20400:", max(a for s, a, *_ in c if 9600 <= s <= 20400), "min:", min(a for s, a, *_ in c if 9600 <= s <= 20400))
for arm, sds in (("O_TYPE_HEBB_ALLPOOL", (2800, 2801)),):
    for sd in sds:
        c = get(post, arm, sd)["curve"]
        print(f"  {arm}|{sd} curve range all evaluations {min(a for _, a, *_ in c):.4f}-{max(a for _, a, *_ in c):.4f}; "
              f"final {c[-1][1]:.4f} -> 2 decimals {round(c[-1][1], 2):.2f}")
print("  S89 pilots for reference (52e2ea7):")
for arm in ("O_HEBB_POOL", "O_HEBB_ALLPOOL", "O_HEBB_POOL_48K", "O_HEBB_ALLPOOL_48K", "O_DELTA_POOL", "O_DELTA_ALLPOOL"):
    for sd in (2800, 2801):
        try:
            r = get(s89, arm, sd)
            print(f"    {arm}|{sd}: transition {r['transition']} acc {r['acc']:.4f}")
        except AssertionError:
            pass
b3 = [get(post, a, s)["acc"] for a, s in (("O_HEBB_CONV_POOL", 2800), ("O_HEBB_CONV_POOL", 2801), ("O_HEBB_CONV_ALLPOOL", 2800))]
print(f"  three bound conv runs final acc {[round(x, 4) for x in b3]}")
print(f"  conv vs the S89 pilot on the same 4 (task, seed): conv only {nb_conv}, pilot only 0 -> exact one-sided sign test "
      f"p = {0.5 ** nb_conv:.3f} (rlib.mcnemar1 {rlib.mcnemar1(nb_conv, 0):.3f}); unpaired Fisher 3/4 vs 0/4 one-sided "
      f"{rlib.fisher1(nb_conv, 4, 0, 4):.3f} (and across CPUs: 2.10 vs 2.80 GHz)")
tab = show(REC, "explore_out/F/s89_post_tables.md")
print("  s89_post_tables.md: 'READING' line:", [l for l in tab.splitlines() if "READING" in l])

# H's oracle cited in the hypothesis
hh = git("log", "--format=%h", "--before=2026-10-20 21:47:35", "-1", "origin/claude/explore-H", "--",
         "explore_out/H/s89_results.json").strip()
hr = json.loads(show(hh, "explore_out/H/s89_results.json"))["runs"]
for k, r in hr.items():
    if k.startswith("ORACLE|"):
        print(f"  H at {hh}: {k.split('|')[0]}|{k.split('|')[1]}: transition {r['transition']} iters {r['iters']} sched "
              f"{r['sched']} k {r['k']} pool {r.get('pool')} conv taps recorded {'conv_absmean' in (r.get('st1') or {})}")

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
H("E  Order, code SHA, files")
for c in git("rev-list", "--reverse", f"{S89_TIP}..{TIP}").split():
    print("  ", git("log", "-1", "--format=%h %ad %s", "--date=iso", c).strip()[:150])
MODS27 = ("bdh bdh_recurrent explore_b16_gates explore_common explore_common2 explore_delta_mem explore_f_delta_fast "
          "explore_f_s89_post_child explore_f_s89_tasks_child explore_f_tasks explore_f_window_delta_child explore_local_gate "
          "explore_window_recipe_child test_binding_capacity test_binding_onset test_binding_recipe test_channel_binding "
          "test_instrument_v2 test_interference test_multilayer_binding test_readout_path test_router_confirm "
          "test_router_curriculum test_router_discovery test_router_layout test_router_reliability test_short_conv").split()


def code_sha(ref, mods):
    rows = []
    for m in sorted(mods):
        rows.append([m, hashlib.sha1(show(ref, m + ".py", text=False).decode().encode()).hexdigest(), "branch"])
    return hashlib.sha1(json.dumps(rows).encode()).hexdigest()


rec_sha = {r["code_sha"] for r in post.values()}
for ref in (REV, "b036d914", "530240cf", "e929827c", "209a2713", REC, TIP):
    s = code_sha(ref, MODS27)
    print(f"  27-module code SHA at {ref[:7]}: {s[:16]}  {'MATCH' if s in rec_sha else 'DIFFERENT'} (records {sorted(rec_sha)[0][:16]})")
FILES = ["explore_f_tasks.py", "explore_f_s89_tasks.py", "explore_f_s89_tasks_child.py", "explore_f_s89_post.py",
         "explore_f_s89_post_child.py", "explore_f_reports6.py", "explore_f_common.py", "explore_common16.py"]
commits = git("rev-list", "--reverse", f"{REV}~1..{TIP}").split()
for f in FILES:
    ids = []
    for c in commits:
        try:
            ids.append((c[:7], git("rev-parse", f"{c}:{f}").strip()[:10]))
        except subprocess.CalledProcessError:
            ids.append((c[:7], "-"))
    ch = [ids[0]] + [ids[i] for i in range(1, len(ids)) if ids[i][1] != ids[i - 1][1]]
    print(f"  {f:30s} blob changes over {commits[0][:7]}..{TIP[:7]}: {ch}")
log = show(REC, "explore_out/F/s89post.log")
print("  log start:", [l for l in log.splitlines() if "S89 post start" in l or "started 2026" in l])
print("  timings measured (runtime json):", json.loads(show("b036d914", "explore_out/F/s89post_runtime.json"))["measured"])
print("  CHECK lines: s89_tasks", log.count("CHECK F/s89_tasks:"), "s89_post", log.count("CHECK F/s89_post:"),
      "FAIL", log.count(": FAIL"), "; delta checks:", [l.strip() for l in log.splitlines() if "ALL CHECKS PASSED" in l])
for ref in ("530240cf", "e929827c", "209a2713", REC):
    st = json.loads(show(ref, "explore_out/F/s89_post_results.json"))["runs"]
    print(f"  records at {ref[:7]} ({git('log', '-1', '--format=%ad', '--date=iso', ref).strip()}): "
          f"{sorted(k.split('|')[0] + '|' + k.split('|')[1] for k in st)}")
print("  first appearance of 'O_HEBB_CONV' in any commit (git log --all -S):",
      git("log", "--all", "--reverse", "--format=%h %ad", "--date=iso", "-S", "O_HEBB_CONV").splitlines()[:1])

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
H("F  Seeds 2800-2839 on every remote tip (json / jsonl / log), outside explore_out/F")
pat = r'\|28[0-3][0-9]\||"seed": 28[0-3][0-9]\b|seed 28[0-3][0-9]\b'
tips = [b.strip() for b in git("branch", "-r").splitlines() if "HEAD" not in b]
for b in tips:
    try:
        out = git("grep", "-l", "-E", pat, b, "--", "*.json", "*.jsonl", "*.log")
    except subprocess.CalledProcessError:
        out = ""
    hits = [l.split(":", 1)[1] for l in out.splitlines()]
    other = [h for h in hits if not h.startswith("explore_out/F/")]
    print(f"  {b:52s} files with a hit: {len(hits)}; outside explore_out/F: {other}")
