#!/usr/bin/env python
"""
explore_g_probe_child.py — EXPLORATORY, not a result. S55's run code (Session G). It runs INSIDE a child process on this
branch's own modules (no main-line hook), 1 torch thread. Parent and readings: explore_g_probe.

  FAR_L3_P4  batch 2's far_L3 (explore_far_cue): LOCAL3 (explore_local_gate.builder) on explore_far_cue.HEADER (P=4, S=2),
             Adam 1e-3 throughout, 24000, explore_far_cue.header_stats; through test_router_reliability.run_one exactly as
             explore_far_cue.run_job. A reproduction: the curve must equal batch 2's record (rec["repro"]).
  L3S_P8     LOCAL3 + SLOW (S43's recipe: explore_window_recipe_child's groups and lr switch; gate W_in, W_g, window 1e-3
             throughout, the rest 1e-4 for updates 1-2400 and 1e-3 after; W_h frozen; no hinge) on
             explore_far_cue.HeaderTask(8, S=2, n_vals=16, n_q=1) (L = 37), 24000. The model: build(task, arm A, ARCH,
             seed) converted by explore_b16_gates.to_local(m, seed) (batch 1's window for the seed).
  ORACLE_P8  the perfect gate (test_channel_binding's 'ceiling' arm for the task), Adam 1e-3 throughout, 24000.
Statistics at every evaluation: explore_far_cue.header_stats (gate_stats_k; header_routing_stats at 1200/2400/3600/end:
eta^2 at KEY and VAL positions by stream/key/half/index, the routing margin from the write gate at VAL positions) plus
header_map at the end (the stream -> channel map from the mean read gate at VAL and at KEY positions; one_to_one) and
stream_acc (held-out accuracy per queried stream).
After the run: explore_g_common.probe on the trained model and on the same seed's model at initialisation.
"""

import json
import os

import torch

import explore_common as ec
import explore_far_cue as s6
import explore_local_gate as lg
import explore_window_recipe_child as s43c
import explore_b16_gates as g16
import explore_g_common as gc
import test_short_conv as tsc
import test_binding_onset as tbo
from test_multilayer_binding import build
from test_channel_binding import ARCH, arms as cb_arms
from test_router_reliability import run_one

ITERS = 24000
LR = 1e-3
H8 = s6.HeaderTask(8, S=2, n_vals=16, n_q=1)
CEIL8 = dict({a["key"]: a for a in cb_arms(H8)}["ceiling"], key="ORACLE_P8")


def far_record(seed):
    with open(os.path.join(ec.OUT_DIR, "far_cue_results.json")) as f:
        return json.load(f)["runs"][f"far_L3|{seed}"]


@torch.no_grad()
def header_map(model, task, probe):
    was = model.training
    model.eval()
    try:
        pinp = probe[0]
        gr = model(pinp)[2]
    finally:
        model.train(was)
    kpos, vpos, _ = gc.header_positions(task)
    lab = task.stream_labels(pinp)
    out = {}
    for role, pos in (("val", vpos), ("key", kpos)):
        g, l = gr[:, pos], lab[:, pos]
        means = [g[l == s].mean(0) for s in range(task.S)]
        cmap = [int(m.argmax()) for m in means]
        out[f"ch_map_{role}"] = cmap
        out[f"one_to_one_{role}"] = len(set(cmap)) == task.S
        out[f"stream_gate_{role}"] = [[round(v, 4) for v in m.tolist()] for m in means]
    return out


def stats_for(task):
    def st(model, t, probe, step):
        out = s6.header_stats(model, t, probe, step)
        if step == "end" and model.n_ch > 1:
            out.update(header_map(model, t, probe))
        return out
    return st


def l3s_builder(task):
    def b(a, seed):
        def make():
            return g16.to_local(build(task, ec.ARM_A, ARCH, seed), seed)
        return make
    return b


def spec(arm):
    if arm == "FAR_L3_P4":
        return s6.ARMS["far_L3"], s6.HEADER, lg.builder, {}, None
    if arm == "L3S_P8":
        holder = {}
        return (dict(ec.ARM_A, key="L3S_P8"), H8, l3s_builder(H8), dict(param_groups=s43c.groups(holder)),
                s43c.make_check(True, holder, None, None))
    if arm == "ORACLE_P8":
        return CEIL8, H8, (lambda a, seed: (lambda: build(H8, a, ARCH, seed))), {}, None
    raise KeyError(arm)


def run(p):
    arm, seed = p["arm"], p["seed"]
    a, task, builder, run_kw, check = spec(arm)
    keep = {"at": -1}
    rec = run_one(a, seed, p.get("iters", ITERS), task=task, stats_fn=stats_for(task), grad_fn=tsc.conv_grad_norms,
                  lr=LR, builder=builder, run_kw=run_kw, check=check, keep=keep)
    if not rec.get("ok"):
        return json.loads(json.dumps(rec, default=float))
    m = keep["model"]
    data = gc.probe_data(task)
    rec["stream_acc"] = gc.stream_acc(m, task, tbo.eval_batch(task))
    rec["probe"] = gc.probe(m, task, data)
    rec["probe_init"] = gc.probe(builder(a, seed)(), task, data)
    if arm == "FAR_L3_P4" and "iters" not in p:
        r0 = far_record(seed)
        rec["repro"] = rec["curve"] == r0["curve"]
    rec.update(arm=arm, tag=ec.tag(rec), fail=None if rec["transition"] is not None or a["n_ch"] < 2
               else ec.fail_class(rec))
    return json.loads(json.dumps(rec, default=float))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def checks(p):
    import numpy as np
    rows = []
    # far_L3 short reproduction: the first 2400 steps of batch 2's record
    r = run(dict(arm="FAR_L3_P4", seed=160, iters=2400))
    want = [c for c in far_record(160)["curve"] if c[0] <= 2400]
    rows.append((f"FAR_L3_P4|160 through 2400 reproduces batch 2's far_L3 record bit for bit ({r['curve']} vs {want})",
                 r["curve"] == want))
    # residual extraction equals the model's own forward, for each model kind
    for arm in ("FAR_L3_P4", "L3S_P8", "ORACLE_P8"):
        a, task, builder, _, _ = spec(arm)
        m = builder(a, 360)()
        x = task.make_batch(64, torch.Generator().manual_seed(3))[0][:, :-1]
        feats, logits, ref = gc.residuals(m, x)
        ok = torch.equal(logits, ref) and set(feats) == {"emb", "L1", "L2", "L3", "final"}
        ok &= torch.allclose(feats["L1"], torch.nn.functional.layer_norm(feats["emb"], (feats["emb"].shape[-1],)), atol=1e-6)
        rows.append((f"{arm}: the residual loop's logits equal model(x)'s bit for bit; layer-1 input = LN(embedding)", ok))
    # labels and split
    x, tr, te = gc.probe_data(H8)
    kpos, vpos, within = gc.header_positions(H8)
    lab = H8.stream_labels(x)
    hdr = torch.stack([x[:, b * H8.blk] for b in range(H8.S)], 1).repeat_interleave(H8.P, 1)
    rows.append(("probe labels at KEY and VAL = the block's CTX token; 2000 sequences, split 1600/400 by sequence, disjoint",
                 torch.equal(lab[:, kpos], hdr) and torch.equal(lab[:, vpos], hdr) and len(set(tr.tolist()) & set(te.tolist())) == 0
                 and len(tr) == 1600 and len(te) == 400 and bool((x[:, kpos] >= H8.S).all())
                 and bool((x[:, kpos] < H8.S + H8.P).all()) and bool((x[:, vpos] >= H8.S + H8.P).all())))
    # synthetic probe: a noisy one-hot stream feature decodes; a random feature does not
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    Xs = rng.normal(size=(4000, 8)); Xs[:, 0] += 3 * (2 * y - 1)
    hit, _ = gc.fit_score(Xs[:3200], y[:3200], Xs[3200:], y[3200:])
    Xr = rng.normal(size=(4000, 8))
    hit_r, _ = gc.fit_score(Xr[:3200], y[:3200], Xr[3200:], y[3200:])
    rows.append((f"synthetic probe: informative feature {hit.mean():.3f} >= 0.99; random feature {hit_r.mean():.3f} in [0.4, 0.6]",
                 hit.mean() >= 0.99 and 0.4 <= hit_r.mean() <= 0.6))
    # L3S_P8's model and groups
    m = l3s_builder(H8)(ec.ARM_A, 360)()
    ref = build(H8, ec.ARM_A, ARCH, 360)
    pr = dict(ref.named_parameters())
    rows.append(("L3S_P8: arm A's initial parameters for the seed, batch 1's window (window_init), W_h frozen",
                 all(torch.equal(q, pr[n]) for n, q in m.named_parameters() if n in pr)
                 and torch.equal(m.gate_conv.conv_w.detach(), g16.window_init(360)) and not m.W_h.requires_grad))
    holder = {}
    s43c.groups(holder)(m)
    names = [n for n, q in m.named_parameters() if q.requires_grad]
    rows.append((f"L3S_P8 groups: gate {holder['names'][0]} at 1e-3, the other {len(holder['names'][1])} at 1e-4, every "
                 f"trainable tensor once", sorted(holder["names"][0] + holder["names"][1]) == sorted(names)
                 and holder["groups"][0]["lr"] == 1e-3 and holder["groups"][1]["lr"] == 1e-4))
    rows.append((f"H8: L = {H8.L} (37), qpos {H8.qpos} ([35]), vocab {H8.vocab} (26)", H8.L == 37 and H8.qpos == [35] and H8.vocab == 26))
    return [[n, bool(v)] for n, v in rows]
