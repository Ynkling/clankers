#!/usr/bin/env python
"""
explore_f_probe_common.py — EXPLORATORY, not a result. Session G's S55 probe, COPIED UNCHANGED from claude/explore-G at
3932db7a77a25f121087fa8a789c6e3000d697b7 (explore_g_common.py: residuals, header_positions, probe_data, fit_score, probe,
and its wilson), so Session F's S77 uses exactly G's probe: sklearn LogisticRegression (lbfgs, max_iter 5000) on
standardised features, fit on 80% of 2000 probe sequences (seed 55000; split by sequence) and scored on the other 20%,
from the raw embedding, the residual entering layers 1..3, and the final layer, at the body's KEY and VAL positions, to
the true stream (the block's CTX). Only the header-layout tasks (fixed block positions) are supported, as in G.
"""
import math


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


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


