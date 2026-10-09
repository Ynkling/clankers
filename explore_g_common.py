#!/usr/bin/env python
"""
explore_g_common.py — EXPLORATORY, not a result. Session G's harness (branch claude/explore-G). It imports explore_common,
explore_common16 and the test modules unchanged.

  - CACHE KEY "arm|seed|code SHA (12 hex)|CPU model". A stored run is reused only when its code SHA is the screen's current
    one AND it ran on this container's CPU (pairing bit for bit only within this CPU; other CPUs' records: counts only).
  - CODE SHA: explore_common16.code_sha (SHA-1 over the sources of every repository module the screen's child imports, in
    a fresh child of the same kind as the runs, this branch's modules only).
  - RETRY: a run with ok = False (or a crashed child) is stored with its error and re-queued, up to 3 attempts.
  - POOL: 4 threads, each driving one child process with 1 torch thread (OMP/MKL 1); jobs longest-first.
Outputs: explore_out/G/<screen>_results.json.

Statistics (fixed for every Session G screen): Wilson 95% (z = 1.96) with every count; bands on n runs: RELIABLE >=
ceil(0.9 n), MAJORITY >= ceil(0.5 n), MINORITY >= 1, NEVER 0; exact one-sided McNemar on paired seeds
(test_conv_lr.mcnemar_greater: P(Bin(b + c, 1/2) >= b)); one-sided Fisher (test_router_confirm.fisher_greater) unpaired.

Model-side helpers (used inside children):
  residuals(model, x)  the residual stream of a no-conv MultiBDH-family model: the raw embedding, the stream entering
                       layers 1..n_layer (layer 1's is LN(embedding)), and the final layer's output (the lm_head input).
                       It calls model(x) once (which sets attn.G from the model's own gate), then runs bdh.BDH.forward's
                       layer loop line for line, recording x; it returns the logits too (CHECK: equal to model(x)'s).
  probe(...)           the logistic-regression probe (sklearn LogisticRegression, lbfgs, on standardised features) from
                       those features at the body's KEY and VAL positions to the true stream (the block's CTX token), fit
                       on 80% of the probe sequences and scored on the other 20% (split by sequence, never by position).
"""

import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec

G_DIR = os.path.join(ec.OUT_DIR, "G")
WORKERS = 4
MAX_ATTEMPTS = 3


# ── Statistics ───────────────────────────────────────────────────────────────
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def band(k, n):
    if n == 0:
        return "--"
    if k >= math.ceil(0.9 * n):
        return "RELIABLE"
    if k >= math.ceil(0.5 * n):
        return "MAJORITY"
    return "MINORITY" if k >= 1 else "NEVER"


def cnt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} [{lo:.2f}, {hi:.2f}] {band(k, n)}"


def mcnemar_greater(b, c):
    from test_conv_lr import mcnemar_greater as m
    return m(b, c)


def fisher_greater(a, n1, b, n2):
    from test_router_confirm import fisher_greater as f
    return f(a, n1, b, n2)


# ── Store, keys, children ────────────────────────────────────────────────────
def store_path(name):
    return os.path.join(G_DIR, f"{name}_results.json")


def load(name):
    p = store_path(name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"meta": {}, "runs": {}}


_LOCK = threading.Lock()


def save(name, st):
    os.makedirs(G_DIR, exist_ok=True)
    p = store_path(name)
    with open(p + ".tmp", "w") as f:
        json.dump(st, f, indent=1, sort_keys=True)
    os.replace(p + ".tmp", p)


def key(arm, seed, sha, cpu):
    return f"{arm}|{seed}|{sha[:12]}|{cpu}"


def screen_sha(child_module):
    import explore_common16 as c16
    return c16.code_sha(child_module, main=False)


def run_child(module, func, payload, timeout=None):
    import explore_common16 as c16
    return c16.run_child(module, func, payload, main=False, timeout=timeout)


def runs(screen, arm, st=None):
    """{seed: record} of an arm's usable runs (this code SHA, this CPU)."""
    st = load(screen.NAME) if st is None else st
    sha, cpu = st["meta"]["sha"], ec.cpu_model()
    out = {}
    for s in screen.ARMS[arm]["seeds"]:
        r = st["runs"].get(key(arm, s, sha, cpu))
        if r is not None and r.get("ok"):
            out[s] = r
    return out


def run_screen(screen, workers=WORKERS, only=None):
    """Every (arm, seed) of the screen not stored ok under this SHA and CPU, retried up to MAX_ATTEMPTS."""
    st = load(screen.NAME)
    sha, rows = screen_sha(screen.CHILD)
    cpu = ec.cpu_model()
    st["meta"].update(sha=sha, sha_rows=rows, cpu=cpu, provenance=ec.provenance(),
                      arms={k: dict(a, seeds=list(a["seeds"])) for k, a in screen.ARMS.items()})
    save(screen.NAME, st)
    print(f"  {screen.NAME}: code SHA {sha[:12]} ({len(rows)} modules), cpu {cpu}", flush=True)
    jobs = []
    for arm, a in screen.ARMS.items():
        if only is not None and arm not in only:
            continue
        for s in a["seeds"]:
            r = st["runs"].get(key(arm, s, sha, cpu))
            if r is None or not r.get("ok"):
                jobs.append((a.get("prio", 1), a["iters"], arm, s))
    jobs.sort(key=lambda j: (-j[0], -j[1], j[3]))
    print(f"  {len(jobs)} runs queued on {workers} workers", flush=True)
    t0 = time.time()
    done = [0]

    def one(job):
        _, _, arm, s = job
        rec = None
        for att in range(1, MAX_ATTEMPTS + 1):
            ts = time.time()
            try:
                rec = run_child(screen.CHILD, "run", dict(arm=arm, seed=s))
            except Exception as e:                                         # noqa: BLE001
                rec = dict(ok=False, error=f"{type(e).__name__}: {e}")
            rec.update(attempt=att, secs_wall=time.time() - ts, cpu=cpu, sha=sha[:12])
            with _LOCK:
                cur = load(screen.NAME)
                cur["runs"][key(arm, s, sha, cpu)] = rec
                save(screen.NAME, cur)
            if rec.get("ok"):
                break
        with _LOCK:
            done[0] += 1
            print(f"  [{done[0]:>3}/{len(jobs)}] {(time.time() - t0) / 60:6.1f} min  {arm:<12} seed {s}  "
                  f"{screen.line(arm, rec)}", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(one, j) for j in jobs]):
            f.result()
    return load(screen.NAME)


# ── Model-side helpers (children) ────────────────────────────────────────────
def residuals(model, x):
    """(feats, logits): feats = dict(emb, L1..Ln, final), each (B, T, D); see the module docstring."""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        was = model.training
        model.eval()
        try:
            ref = model(x)[0]
            assert getattr(model, "conv", None) is None
            C = model.config
            B, T = x.size()
            D = C.n_embd
            nh = C.n_head
            N = D * C.mlp_internal_dim_multiplier // nh
            v = model.embed(x)
            feats = {"emb": v}
            h = model.ln(v.unsqueeze(1))
            for level in range(C.n_layer):
                feats[f"L{level + 1}"] = h.view(B, T, D)
                x_latent = h @ model.encoder
                x_sparse = F.relu(x_latent)
                yKV = model.attn(Q=x_sparse, K=x_sparse, V=h)
                yKV = model.ln(yKV)
                y_latent = yKV @ model.encoder_v
                y_sparse = F.relu(y_latent)
                xy_sparse = x_sparse * y_sparse
                xy_sparse = model.drop(xy_sparse)
                yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model.decoder
                y = model.ln(yMLP)
                h = model.ln(h + y)
            feats["final"] = h.view(B, T, D)
            logits = h.view(B, T, D) @ model.lm_head
        finally:
            model.train(was)
    return feats, logits, ref


def header_positions(task):
    """KEY positions of a fixed-header task in body order, VAL one later, and each pair's index within its block."""
    import torch
    S, P, blk = task.S, task.P, task.blk
    kpos = torch.tensor([b * blk + 1 + 2 * i for b in range(S) for i in range(P)])
    within = torch.tensor([i for b in range(S) for i in range(P)])
    return kpos, kpos + 1, within


def probe_data(task, n=2000, seed=55_000):
    import torch
    x = task.make_batch(n, torch.Generator().manual_seed(seed))[0][:, :-1]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 1))
    ntr = int(round(0.8 * n))
    return x, perm[:ntr], perm[ntr:]


def fit_score(Xtr, ytr, Xte, yte):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    return pred == yte, float(clf.score(sc.transform(Xtr), ytr))


def probe(model, task, data, layers=None):
    """Held-out probe accuracy per (feature, position type), with n, Wilson, and accuracy by pair index within block."""
    import numpy as np
    x, tr, te = data
    feats, logits, ref = residuals(model, x)
    kpos, vpos, within = header_positions(task)
    lab = task.stream_labels(x)
    out = dict(logits_equal=bool((logits == ref).all()))
    for name, f in feats.items():
        if layers is not None and name not in layers:
            continue
        for role, pos in (("K", kpos), ("V", vpos)):
            X = f[:, pos].double().numpy()                       # (n, npairs, D)
            y = lab[:, pos].numpy()
            Xtr, ytr = X[tr.numpy()].reshape(-1, X.shape[-1]), y[tr.numpy()].reshape(-1)
            Xte, yte = X[te.numpy()].reshape(-1, X.shape[-1]), y[te.numpy()].reshape(-1)
            hit, tracc = fit_score(Xtr, ytr, Xte, yte)
            k, n = int(hit.sum()), int(hit.size)
            hit2 = hit.reshape(len(te), -1)
            by_i = {int(i): float(hit2[:, (within == i).numpy()].mean()) for i in within.unique()}
            out[f"{name}|{role}"] = dict(acc=k / n, k=k, n=n, wilson=wilson(k, n), train_acc=tracc, by_index=by_i)
    return out


def stream_acc(model, task, data):
    """Held-out accuracy per queried stream (explore_keysplits_child.stream_acc's computation)."""
    import torch
    import test_binding_onset as tbo
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            ql, qt, _ = task.select(tbo.logits_of(model, data[:, :-1]), data[:, 1:], None)
    finally:
        model.train(was)
    qs = data[:, task.qpos[0] - 1]
    hit = (ql.argmax(-1) == qt).float()
    return [hit[qs == s].mean().item() for s in range(task.S)]
