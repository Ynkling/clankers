#!/usr/bin/env python
"""
explore_i_common.py — EXPLORATORY, not a result. Session I's harness: models, recipes, the training loop, ICMC's metrics,
the split, the nudge, and a driver (4 worker processes, 1 torch thread each, cached records). It changes no existing
module; explore_f_tasks.py, explore_b16_gates.py and explore_g_regmodel.py are copied unchanged from claude/explore-G.

MODELS (every model is the project's BDH: test_multilayer_binding.MultiBDH via build(task, arm, ARCH, seed); ARCH = 3
layers, positional "decay" at 0.95; mult 8 -> N = 256; D = 32; one head; conv "layer", width 4, identity init — the
eight-stream tests' convolution, test_short_conv.LAYER; k channels; read gate = write gate)
  perfect  test_short_conv's 'ceiling_conv' arm with n_ch = k (the perfect gate: one-hot on the latest SRC token's source).
  none     test_short_conv's 'B_conv' arm (k = 1, no gate): SINGLE and NOMIX.
  win3     LOCAL3: the A_conv arm (recurrent gate, n_ch = k) converted by explore_b16_gates.to_local (batch 1's width-3
           window; W_h frozen, unused, in no group).
  wide     the WIDE window gate: the same construction at width W = LMAX + 2 (to_window: a CausalConv(D, W) drawn uniform
           +-1/sqrt(W) from the generator seeded seed + 31_337, as window_init; at W = 3 equal to to_local, CHECKed). It
           sees tokens t .. t-(LMAX+1), so the block's SRC token at every word of a block (CHECKed: t-(LMAX+1) seen,
           t-(LMAX+2) not).
  reg3     Session G's REG3 (explore_g_rhl_child.Reg3BDH, S72): explore_g_regmodel.to_reg with gated = res_layers =
           (1, 2, 3), SG, the register on, inp = "window" (each layer's gate reads LOCAL3's window of the raw embeddings,
           LN of the residual entering the layer, and R_{t-1}); A_v = W_in and W_g2 = W_g copies at init; no nudge unless
           the arm says so. Reg3Conv adds the "layer" convolution to RegBDH.forward's loop exactly as MultiBDH.forward_conv
           inserts it (the gate reads the unconvolved residual); with conv None its logits equal RegBDH.forward's bit for
           bit (CHECKed). to_reg3 calls the copied to_reg unchanged (its no-conv assertion is satisfied by hiding the conv
           attribute during the call).
RECIPES (Adam, lr 1e-3; batch 32)
  ADAM   one Adam at 1e-3 on every parameter (the validity/single-channel recipe): ORACLE, SINGLE, NOMIX.
  SLOW   the gate group (W_in, W_g, gate_conv.conv_w; for REG3 explore_g_regmodel.gate_names: also A_v, A_r, W_g2, A_R,
         W_R, w_R, b_R) at 1e-3 throughout; every other trainable parameter at 1e-4 for updates 1-2400 and 1e-3 after (the
         switch right after the evaluation at 2400).
  SPLIT  (KEYMASS, batch 17's W_SPLIT rule as test_window_gate's Part D, transposed to ICMC): checks after the evaluations
         at 2400 (reference), 4800, 7200, ... through total - 2400, after the lr switch, on a fixed 64-sequence probe
         (generator 12345); fire if the probe's SET accuracy < SPLIT_THR and it rose < 0.02 since the previous check; at
         most 3 splits, none within 4800 updates of the previous; c* / c0 = the channels with the largest / smallest mean
         read-gate mass at the probe's KEY POSITIONS, which here are ALL WORD POSITIONS (ICMC has no keys: every word is
         both a write and a read); W_g[c0] = w + n2, W_g[c*] = w + n1 (w = W_g[c*], noise 0.1 std(w), generator
         12_000_000 + 1000 seed + split count); W_g's Adam state zeroed (split_op, noise_for, zero_state transcribed from
         test_window_gate, CHECKed equal). SPLIT_THR is set by the screen (explore_i_dense).
  NUDGE  explore_g_register_child.nudge_reg, transcribed: inside the model build, 300 Adam steps at lr 0.01 on the new
         gate / register parameters (explore_g_regmodel.GATE_NEW) toward (1 - 0.05)/k + 0.05 onehot(source) for the
         gates of all three layers at every position (MSE; the label is ICMC's stream_labels), batches of 64 from a
         generator seeded seed + 55_555.
TRAINING LOOP (train): test_binding_onset.onset_run's order line for line — torch.manual_seed(seed); the model; Adam;
  a batch generator seeded seed + 10_000; per update make_batch(32, rng), logits = logits_of(model, tokens[:, :-1]), the
  loss, zero_grad, backward, step — with the loss either ICMC's masked cross-entropy ("icmc": mean over the mask's
  positions, the full-vocabulary softmax) or BindTask's query loss ("select", for the reproduction CHECK). No early stop
  under "icmc" (a fixed budget; the outcomes are read at the end).
EVALUATION (every 1200 updates and at the end; held-out 1024 sequences from a generator seeded 424_242; eval mode):
  loss       the masked cross-entropy, overall and by target bin (1, 2, 3-4, 5+: the target word's index in its block)
  set        SET accuracy (explore_i_task.set_hits), overall, by bin and by source
  routing    per gate layer (one entry "L1" for a shared gate; L1..L3 for REG3), at WORD POSITIONS (input positions whose
             token is a word; the gate there writes that word and reads for the next): eta^2 by source of the gate vector
             (test_scale_axes.eta2_multi), also restricted to the word's index in its block 1-2 and 5+; the source ->
             channel map (argmax of the mean gate per source) and one_to_one; the mean gate mass per channel; the
             routing margin = mean over word positions t of [mean_{s<t, word, same source} g_t.g_s - mean_{s<t, word,
             other source} g_t.g_s] (positions with both sets non-empty; 1 for a one-hot partition, 0 for no routing).
             On the first 256 held-out sequences.
  probe      Session G's S55 probe (explore_g_common.fit_score: sklearn LogisticRegression, lbfgs, max_iter 5000, on
             standardised features) from the residual entering layers 1, 2, 3 and the final layer's output (the lm_head
             input) to the source, at word positions, on 256 probe sequences (generator 55_000) split 80/20 by sequence
             (generator 55_001); at most 20000 training and 10000 test positions (a fixed subsample, generator 55_002).
             The residuals are recorded by replaying the model's own layer loop with its own gates; the replay's logits
             equal the model's bit for bit (asserted at every call).
OUTCOMES (explore_i_dense defines ROUTED and GAP from these records).
CACHE: explore_out/I/<screen>_results.json, key "arm|seed|code SHA (12 hex)|CPU model"; a record is reused only with the
screen's current code SHA (SHA-1 over the source of every repository module loaded by the run code) and this CPU; ok =
False is retried (up to 3 attempts). Atomic writes after every run.
"""

import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F

from bdh import BDH
import explore_i_task as it
import explore_b16_gates as g16
import explore_g_regmodel as rm
import test_short_conv as tsc
import test_binding_onset as tbo
from test_multilayer_binding import build, CausalConv, D, MultiBDH
from test_channel_binding import ARCH
from test_scale_axes import eta2_multi
from test_instrument_v2 import TAU_END

OUT_DIR = os.path.join(HERE, "explore_out", "I")
LR, LR_WARM, WARM = 1e-3, 1e-4, 2400
BATCH = 32
EVAL_EVERY = 1200
EVAL_N, EVAL_SEED = 1024, 424_242
ROUTE_N = 256
PROBE_N, PROBE_SEED, PROBE_TR, PROBE_TE = 256, 55_000, 20000, 10000
SPLIT_PROBE_N, SPLIT_PROBE_SEED = 64, 12345
SPLIT_EVERY, SPLIT_FIRST, SPLIT_REF = 2400, 4800, 2400
SPLIT_RISE, SPLIT_MAX, SPLIT_GAP = 0.02, 3, 4800
NOISE_REL, NOISE_BASE = 0.1, 12_000_000
NUDGE_STEPS, NUDGE_LR, NUDGE_B, NUDGE_AMP, NUDGE_OFFSET = 300, 1e-2, 64, 0.05, 55_555
GATE_L = ("W_in", "W_g", "gate_conv.conv_w")
ARM_CEIL, ARM_A, ARM_B = tsc.ARM["ceiling_conv"], tsc.ARM["A_conv"], tsc.ARM["B_conv"]


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


# ── gates ─────────────────────────────────────────────────────────────────────
def window_init_w(seed, width):
    """batch 1's window construction at any width: uniform +-1/sqrt(width), generator seeded seed + 31_337."""
    g = torch.Generator().manual_seed(int(seed) + g16.GCONV_OFFSET)
    return (torch.rand(D, width, generator=g) * 2 - 1) * (1.0 / math.sqrt(width))


def to_window(m, seed, width):
    """explore_b16_gates.to_local at another width (the class is LocalBDH; its forward is unchanged)."""
    assert type(m) is MultiBDH and m.gate_kind == "recurrent", (type(m), m.gate_kind)
    assert not m.gate_to_readout and not m.gate_ln and m.gate_noise == 0.0
    m.__class__ = g16.LocalBDH
    m.local = True
    m.gate_conv = CausalConv(D, width)
    with torch.no_grad():
        m.gate_conv.conv_w.copy_(window_init_w(seed, width))
    m.W_h.requires_grad_(False)
    return m


class Reg3Conv(rm.RegBDH):
    """Session G's Reg3BDH (inp = "window") with the "layer" convolution inserted as MultiBDH.forward_conv does."""

    def layer_gate(self, level, h, v, Rprev):
        x_in = self.gate_conv(v) if self.inp == "window" else v
        return rm.RegBDH.layer_gate(self, level, h, x_in, Rprev)

    def forward(self, tokens, tau=None, track_sat=False):
        if self.conv is None:
            return rm.RegBDH.forward(self, tokens, tau, track_sat)
        assert self.conv == "layer" and self.mem == "hebb"
        C = self.config
        B, T = tokens.size()
        D_ = C.n_embd
        nh = C.n_head
        N = D_ * C.mlp_internal_dim_multiplier // nh
        v = self.embed(tokens)
        g1, _ = self.local_gates(v)
        Rprev, beta = self.register(v) if self.reg else (None, None)
        gates = []
        G1 = torch.einsum("btk,bsk->bts", g1, g1).unsqueeze(1)
        x = self.embed(tokens).unsqueeze(1)
        x = self.ln(x)
        for level in range(C.n_layer):
            lv = level + 1
            g = self.layer_gate(lv, x.view(B, T, D_), v, Rprev) if lv in self.gated else g1
            gates.append(g)
            self.attn.G = G1 if g is g1 else torch.einsum("btk,bsk->bts", g, g).unsqueeze(1)
            xc = self.short_conv(x)
            x_latent = xc @ self.encoder
            x_sparse = F.relu(x_latent)
            yKV = self.attn(Q=x_sparse, K=x_sparse, V=xc)
            yKV = self.ln(yKV)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            y = self.ln(yMLP)
            x = self.ln(x + y)
        logits = x.view(B, T, D_) @ self.lm_head
        if self.keep_diag:
            self.diag = dict(gates=[g.detach() for g in gates], beta=None if beta is None else beta.detach())
        gl = gates[max(self.gated) - 1] if self.gated else g1
        return logits, None, gl, gl


def to_reg3(m, seed):
    """explore_g_rhl_child.make_reg3(task, seed, "hebb") on a LocalBDH that may carry the "layer" convolution."""
    assert type(m) is g16.LocalBDH, type(m)
    conv = m.conv
    m.conv = None                                    # to_reg asserts a no-conv model; restored right after
    try:
        rm.to_reg(m, seed, sg=True, reg=True, gated=(1, 2, 3), res_layers=(1, 2, 3), mem="hebb")
    finally:
        m.conv = conv
    m.__class__ = Reg3Conv
    m.inp = "window"
    return m


def live_gates(m, x):
    """Every layer's gate with gradient (explore_g_register_child.live_gates)."""
    keep = []
    old = m.layer_gate

    def lg(level, h, v, Rprev):
        g = old(level, h, v, Rprev)
        keep.append((level, g))
        return g
    m.layer_gate = lg
    try:
        m(x)
    finally:
        del m.layer_gate
    out = [None] * m.config.n_layer
    for level, g in keep:
        out[level - 1] = g
    return out


def nudge_reg(m, task, seed):
    """explore_g_register_child.nudge_reg with ICMC's labels and target (1 - AMP)/k + AMP onehot (0.475/0.525 at k=2)."""
    g = torch.Generator().manual_seed(seed + NUDGE_OFFSET)
    params = [getattr(m, n) for n in rm.GATE_NEW if hasattr(m, n)]
    opt = torch.optim.Adam(params, lr=NUDGE_LR)
    k = m.n_ch
    for _ in range(NUDGE_STEPS):
        x = task.make_batch(NUDGE_B, g)[0][:, :-1]
        lab = task.stream_labels(x)
        tgt = (1 - NUDGE_AMP) / k + NUDGE_AMP * F.one_hot(lab.clamp(min=0), k).float()
        gates = live_gates(m, x)
        loss = sum(F.mse_loss(gates[l - 1], tgt) for l in m.gated)
        opt.zero_grad(); loss.backward(); opt.step()
    m.zero_grad(set_to_none=True)


# ── models ────────────────────────────────────────────────────────────────────
def make_model(spec, task, seed):
    """spec: dict(gate in perfect/none/win3/wide/reg3, k, nudge=False)."""
    gate, k = spec["gate"], int(spec.get("k", 1))
    if gate == "perfect":
        return build(task, dict(ARM_CEIL, n_ch=k), ARCH, seed)
    if gate == "none":
        assert k == 1
        return build(task, ARM_B, ARCH, seed)
    m = build(task, dict(ARM_A, n_ch=k), ARCH, seed)
    if gate == "win3":
        return g16.to_local(m, seed)
    if gate == "wide":
        return to_window(m, seed, task.lmax + 2)
    if gate == "reg3":
        m = to_reg3(g16.to_local(m, seed), seed)
        if spec.get("nudge"):
            nudge_reg(m, task, seed)
        return m
    raise KeyError(gate)


def gate_names(m):
    if isinstance(m, rm.RegBDH):
        return rm.gate_names(m)
    names = [n for n, _ in m.named_parameters()]
    return [n for n in GATE_L if n in names]


def slow_groups(holder):
    def param_groups(model):
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        nm = dict(named)
        gate = gate_names(model)
        rest = [n for n, _ in named if n not in gate]
        holder["names"] = [gate, rest]
        holder["groups"] = [dict(params=[nm[n] for n in gate], lr=LR), dict(params=[nm[n] for n in rest], lr=LR_WARM)]
        return holder["groups"]
    return param_groups


# ── training loop ─────────────────────────────────────────────────────────────
def train(task, make, seed, iters, loss_kind="icmc", param_groups=None, on_eval=None, grad_hook=None,
          eval_every=EVAL_EVERY, lr=LR):
    """onset_run's order (see the docstring). on_eval(model, step, opt) after every eval_every updates."""
    torch.manual_seed(seed)
    model = make()
    opt = torch.optim.Adam(model.parameters() if param_groups is None else param_groups(model), lr=lr)
    rng = torch.Generator(); rng.manual_seed(seed + 10_000)
    for step in range(1, iters + 1):
        tokens, _ = task.make_batch(BATCH, rng)
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        logits = tbo.logits_of(model, inp)
        if loss_kind == "select":
            ql, qt, _ = task.select(logits, tgt, None)
            loss = F.cross_entropy(ql, qt)
        else:
            mask = task.loss_mask(tokens)
            loss = F.cross_entropy(logits[mask], tgt[mask])
        opt.zero_grad(); loss.backward()
        if grad_hook is not None:
            grad_hook(model, step)
        opt.step()
        if on_eval is not None and step % eval_every == 0:
            on_eval(model, step, opt)
    return model, opt


# ── evaluation ────────────────────────────────────────────────────────────────
def eval_set(task, n=EVAL_N, seed=EVAL_SEED):
    tokens, meta = task.make_batch(n, torch.Generator().manual_seed(seed))
    ann = task.annotate(tokens)
    return dict(tokens=tokens, tables=meta["tables"], ann=ann, sets=task.true_sets(tokens, meta["tables"], ann),
                bins=task.bin_of(ann["bpos"]))


def probe_set(task):
    x = task.make_batch(PROBE_N, torch.Generator().manual_seed(PROBE_SEED))[0][:, :-1]
    perm = torch.randperm(PROBE_N, generator=torch.Generator().manual_seed(PROBE_SEED + 1))
    ntr = int(round(0.8 * PROBE_N))
    return x, perm[:ntr], perm[ntr:]


def model_gates(model, x):
    """[g per gate layer] (B, T, k): one shared gate -> [g]; REG3 -> [g1, g2, g3]; eval mode, no grad."""
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            if isinstance(model, rm.RegBDH):
                model.keep_diag = True
                model(x)
                model.keep_diag = False
                return list(model.diag["gates"])
            return [model(x, TAU_END)[2]]
    finally:
        model.train(was)


@torch.no_grad()
def logits_chunked(model, x, chunk=128):
    was = model.training
    model.eval()
    try:
        return torch.cat([tbo.logits_of(model, x[i:i + chunk]) for i in range(0, x.shape[0], chunk)], 0)
    finally:
        model.train(was)


def margin_of(g, src, word, chunk=64):
    """Mean over word positions t of [mean same-source earlier word g_t.g_s - mean other-source earlier word g_t.g_s]."""
    tot, n = 0.0, 0
    T = g.shape[1]
    tri = torch.ones(T, T, dtype=torch.bool).tril(-1)
    for i in range(0, g.shape[0], chunk):
        gg, ss, ww = g[i:i + chunk].double(), src[i:i + chunk], word[i:i + chunk]
        dots = gg @ gg.transpose(1, 2)                                      # (b, T, T)
        base = tri & ww[:, :, None] & ww[:, None, :]
        same = base & (ss[:, :, None] == ss[:, None, :])
        diff = base & (ss[:, :, None] != ss[:, None, :])
        ns, nd = same.sum(-1), diff.sum(-1)
        ok = (ns > 0) & (nd > 0)
        ms = (dots * same).sum(-1) / ns.clamp(min=1)
        md = (dots * diff).sum(-1) / nd.clamp(min=1)
        tot += float((ms - md)[ok].sum())
        n += int(ok.sum())
    return tot / max(n, 1)


def routing_layer(g, x, task):
    S = task.S
    src = task.stream_labels(x)
    word = x >= S
    idx = torch.zeros_like(x)
    run = torch.zeros(x.shape[0], dtype=torch.long)
    for t in range(x.shape[1]):
        run = torch.where(word[:, t], run + 1, torch.zeros_like(run))
        idx[:, t] = run
    gw, sw = g[word], src[word]
    out = dict(eta=eta2_multi(gw, sw))
    early, late = word & (idx <= 2), word & (idx >= 5)
    out["eta_pos12"] = eta2_multi(g[early], src[early]) if early.any() else None
    out["eta_pos5p"] = eta2_multi(g[late], src[late]) if late.any() else None
    means = [gw[sw == s].mean(0) for s in range(S)]
    cmap = [int(m_.argmax()) for m_ in means]
    out.update(ch_map=cmap, one_to_one=len(set(cmap)) == S,
               src_gate=[[round(v, 4) for v in m_.tolist()] for m_ in means],
               mass=[round(v, 5) for v in gw.mean(0).tolist()])
    out["margin"] = margin_of(g, src, word) if g.shape[-1] > 1 and S > 1 else 0.0
    return out


@torch.no_grad()
def residuals(model, x):
    """(feats {L1, L2, L3, final}: (B, T, D), logits) by replaying the model's layer loop with its own gates."""
    was = model.training
    model.eval()
    try:
        if isinstance(model, rm.RegBDH):
            model.keep_diag = True
            logits = model(x)[0]
            model.keep_diag = False
            Gs = [torch.einsum("btk,bsk->bts", g, g).unsqueeze(1) for g in model.diag["gates"]]
        else:
            logits = model(x, TAU_END)[0]
            Gs = [model.attn.G] * model.config.n_layer
        C = model.config
        B, T = x.shape
        D_ = C.n_embd
        nh = C.n_head
        N = D_ * C.mlp_internal_dim_multiplier // nh
        h = model.embed(x).unsqueeze(1)
        if model.conv == "in":
            h = model.short_conv(h)
        h = model.ln(h)
        feats = {}
        for level in range(C.n_layer):
            feats[f"L{level + 1}"] = h.reshape(B, T, D_)
            model.attn.G = Gs[level]
            hc = model.short_conv(h) if model.conv == "layer" else h
            x_sparse = F.relu(hc @ model.encoder)
            yKV = model.ln(model.attn(Q=x_sparse, K=x_sparse, V=hc))
            y_sparse = F.relu(yKV @ model.encoder_v)
            xy = model.drop(x_sparse * y_sparse)
            y = model.ln(xy.transpose(1, 2).reshape(B, 1, T, N * nh) @ model.decoder)
            h = model.ln(h + y)
        feats["final"] = h.reshape(B, T, D_)
        lg = h.reshape(B, T, D_) @ model.lm_head
        assert torch.equal(lg, logits), "residual replay differs from the model's forward"
        return feats, logits
    finally:
        model.train(was)


def fit_score(Xtr, ytr, Xte, yte):
    """explore_g_common.fit_score (Session G's S55 probe), transcribed."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    return pred == yte, float(clf.score(sc.transform(Xtr), ytr))


def probe_source(model, task, pdata):
    x, tr, te = pdata
    if task.S < 2:
        return None
    feats, _ = residuals(model, x)
    lab = task.stream_labels(x)
    word = x >= task.S
    sub = torch.Generator().manual_seed(PROBE_SEED + 2)
    out = {}
    for name, f in feats.items():
        Xtr, ytr = f[tr][word[tr]], lab[tr][word[tr]]
        Xte, yte = f[te][word[te]], lab[te][word[te]]
        ptr = torch.randperm(Xtr.shape[0], generator=sub)[:PROBE_TR]
        pte = torch.randperm(Xte.shape[0], generator=sub)[:PROBE_TE]
        hit, tracc = fit_score(Xtr[ptr].double().numpy(), ytr[ptr].numpy(), Xte[pte].double().numpy(), yte[pte].numpy())
        out[name] = dict(acc=float(hit.mean()), n=int(hit.size), train_acc=tracc)
    return out


def evaluate(model, task, ev, pdata=None, with_probe=True):
    x, tok = ev["tokens"][:, :-1], ev["tokens"]
    tgt = tok[:, 1:]
    lg = logits_chunked(model, x)
    m, bins, sets, src = ev["ann"]["mask"], ev["bins"], ev["sets"], ev["ann"]["src"]
    ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
    hit = it.set_hits(task, lg, sets)
    out = dict(loss=float(ce[m].mean()), set=float(hit[m].float().mean()))
    for i, name in enumerate(it.BIN_NAMES):
        mm = m & (bins == i)
        out[f"loss_{name}"] = float(ce[mm].mean()) if mm.any() else None
        out[f"set_{name}"] = float(hit[mm].float().mean()) if mm.any() else None
    out["set_by_src"] = [float(hit[m & (src == s)].float().mean()) for s in range(task.S)]
    if model.n_ch > 1:
        xr = x[:ROUTE_N]
        out["routing"] = {f"L{i + 1}": routing_layer(g, xr, task) for i, g in enumerate(model_gates(model, xr))}
    if with_probe and pdata is not None:
        out["probe"] = probe_source(model, task, pdata)
    return out


# ── the split (KEYMASS on ICMC) ───────────────────────────────────────────────
def noise_for(w, seed):
    """test_window_gate.noise_for, transcribed (CHECKed equal)."""
    sd = NOISE_REL * w.std()
    g = torch.Generator().manual_seed(seed)
    n1 = torch.randn(w.shape, generator=g, dtype=w.dtype) * sd                   # row c*
    n2 = torch.randn(w.shape, generator=g, dtype=w.dtype) * sd                   # row c0
    return sd, n1, n2


def zero_state(o, W):
    """test_window_gate.zero_state, transcribed."""
    st = o.state.get(W)
    zeroed = []
    if st:
        for k, v in st.items():
            if torch.is_tensor(v):
                v.zero_()
                zeroed.append(k)
    return zeroed


def split_op(m, o, cs, c0, seed):
    """test_window_gate.split_op, transcribed: the row copy with noise, then W_g's optimizer state zeroed."""
    assert not any(n.startswith("b_g") or n == "W_g_bias" for n, _ in m.named_parameters())
    W = m.W_g
    with torch.no_grad():
        w = W[cs].clone()
        before = float((W[cs] - W[c0]).norm())
        sd, n1, n2 = noise_for(w, seed)
        W[c0] = w + n2
        W[cs] = w + n1
    zeroed = zero_state(o, W)
    return dict(noise_sd=float(sd), noise_seed=seed, row_dist_before=before,
                row_dist_after=float((W[cs] - W[c0]).detach().norm()), state_zeroed=zeroed)


def make_split(task, seed, last, thr, rise=SPLIT_RISE, max_splits=SPLIT_MAX, gap=SPLIT_GAP):
    """The trigger: called after each evaluation (and the lr switch) with (model, step, opt); rows in .rows."""
    pe = eval_set(task, SPLIT_PROBE_N, SPLIT_PROBE_SEED)
    px = pe["tokens"][:, :-1]
    st = dict(prev=None, n=0, last=None)
    rows = []

    def trig(m, step, opt):
        if step % SPLIT_EVERY or step < SPLIT_REF or step > last:
            return
        lg = logits_chunked(m, px)
        acc = float(it.set_hits(task, lg, pe["sets"])[pe["ann"]["mask"]].float().mean())
        g = model_gates(m, px)[0]
        word = px >= task.S
        km = g[word].mean(0)
        cs, c0 = int(km.argmax()), int(km.argmin())
        cmap = routing_layer(g, px, task)["ch_map"]
        row = dict(step=step, acc=acc, prev=st["prev"], map_before=cmap, masses=[round(float(v), 5) for v in km],
                   cs=cs, c0=c0, on_target=cmap.count(cs) >= 2 and cmap.count(c0) == 0, fired=False, blocked=False)
        eligible = step >= SPLIT_FIRST and st["prev"] is not None and acc < thr and acc - st["prev"] < rise
        capped = st["n"] >= max_splits or (st["last"] is not None and step - st["last"] < gap)
        row["eligible"] = eligible
        if eligible and capped:
            row.update(blocked=True, blocked_by="cap" if st["n"] >= max_splits else "gap")
        elif eligible:
            res = split_op(m, opt, cs, c0, NOISE_BASE + 1000 * seed + st["n"])
            st["n"] += 1
            st["last"] = step
            g2 = model_gates(m, px)[0]
            row.update(fired=True, map_after=routing_layer(g2, px, task)["ch_map"], **res)
        st["prev"] = acc
        rows.append(row)
    trig.rows = rows
    return trig


# ── one run ───────────────────────────────────────────────────────────────────
def run_one(task, spec, seed, iters, split_thr=None, with_probe=True, tag=None):
    """One record. spec: gate, k, recipe ("adam"/"slow"), split (bool), nudge (bool)."""
    ts = time.time()
    holder = {}
    ev, pdata = eval_set(task), probe_set(task)
    curve = []
    trig = make_split(task, seed, iters - SPLIT_EVERY, split_thr) if spec.get("split") else None
    slow = spec.get("recipe") == "slow"

    def on_eval(m, step, opt):
        e = evaluate(m, task, ev, pdata, with_probe)
        e["step"] = step
        e["secs"] = round(time.time() - ts, 1)
        curve.append(e)
        if tag is not None:
            rt = " ".join(f"{L} eta {r['eta']:.2f} m {r['margin']:.2f}" for L, r in (e.get("routing") or {}).items())
            pr = " ".join(f"{k} {v['acc']:.2f}" for k, v in (e.get("probe") or {}).items())
            print(f"    .. {tag} step {step}: SET {e['set']:.3f} loss {e['loss']:.3f} {rt} probe {pr} "
                  f"({e['secs'] / 60:.1f} min)", flush=True)
        if slow and step >= WARM:
            holder["groups"][1]["lr"] = LR
        if trig is not None:
            trig(m, step, opt)

    try:
        model, _ = train(task, lambda: make_model(spec, task, seed), seed, iters,
                         param_groups=slow_groups(holder) if slow else None, on_eval=on_eval)
        if not all(math.isfinite(c["loss"]) for c in curve):
            raise ValueError("non-finite loss in the curve")
        rec = dict(ok=True, seed=seed, curve=curve, end=curve[-1], iters=iters, k=model.n_ch,
                   group_names=holder.get("names"), secs=time.time() - ts)
        if trig is not None:
            rec.update(checks=trig.rows, splits=sum(r["fired"] for r in trig.rows))
        if isinstance(model, rm.RegBDH):
            with torch.no_grad():
                x = ev["tokens"][:ROUTE_N, :-1]
                beta = torch.sigmoid(model.embed(x) @ model.w_R + model.b_R)
                rec["beta_src"] = float(beta[x < task.S].mean())
                rec["beta_word"] = float(beta[x >= task.S].mean())
    except Exception as e:
        rec = dict(ok=False, seed=seed, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:],
                   secs=time.time() - ts)
    return json.loads(json.dumps(rec, default=float))


# ── driver ────────────────────────────────────────────────────────────────────
def code_sha_here():
    """SHA-1 over the sources of every repository module loaded in this process (sorted by name)."""
    h = hashlib.sha1()
    mods = sorted((n, m) for n, m in list(sys.modules.items())
                  if isinstance(getattr(m, "__file__", None), str) and os.path.isabs(m.__file__)
                  and os.path.dirname(m.__file__) == HERE)
    for n, m in mods:
        h.update(n.encode())
        with open(m.__file__, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12], [n for n, _ in mods]


def code_sha(module):
    """code_sha_here in a fresh child that imports only the run code's module (as the runs' workers do)."""
    import subprocess
    out = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {HERE!r}); import json, {module}, "
                          f"explore_i_common as ic; print(json.dumps(ic.code_sha_here()))"],
                         capture_output=True, text=True, check=True, cwd=HERE).stdout
    sha, mods = json.loads(out.strip().splitlines()[-1])
    return sha, mods


def store_path(name):
    return os.path.join(OUT_DIR, f"{name}_results.json")


def load(name):
    p = store_path(name)
    if not os.path.exists(p):
        return {"meta": {}, "runs": {}}
    with open(p) as f:
        return json.load(f)


def save(name, st):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = store_path(name)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, p)


def key(arm, seed, sha, cpu):
    return f"{arm}|{seed}|{sha}|{cpu}"


def _worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _job(payload):
    _worker_init()
    module = __import__(payload["module"])
    return module.job(payload)


def run_jobs(name, module, jobs, workers=4, log=print, max_attempts=3):
    """jobs: list of dict(arm, seed, est (seconds, for longest-first), ...) passed to module.job; records cached."""
    from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
    import multiprocessing as mp
    sha, mods = code_sha(module)
    cpu = cpu_model()
    st = load(name)
    st["meta"].setdefault("started", time.strftime("%Y-%m-%d %H:%M:%S"))
    st["meta"].update(sha=sha, cpu=cpu, torch=torch.__version__, modules=mods, workers=workers)
    todo = [j for j in jobs if not (st["runs"].get(key(j["arm"], j["seed"], sha, cpu)) or {}).get("ok")]
    todo.sort(key=lambda j: -j.get("est", 0))
    log(f"[{name}] code SHA {sha}, CPU {cpu}; {len(jobs) - len(todo)} cached, {len(todo)} to run on {workers} workers")
    save(name, st)
    attempts = {}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        pending = {}
        queue = list(todo)

        def submit():
            while queue and len(pending) < workers:
                j = queue.pop(0)
                pending[pool.submit(_job, dict(j, module=module))] = j
        submit()
        while pending:
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for f in done:
                j = pending.pop(f)
                k = key(j["arm"], j["seed"], sha, cpu)
                try:
                    rec = f.result()
                except Exception as e:
                    rec = dict(ok=False, seed=j["seed"], error=f"worker: {type(e).__name__}: {e}")
                rec.update(arm=j["arm"], cpu=cpu, sha=sha)
                st = load(name)
                st["runs"][k] = rec
                save(name, st)
                attempts[k] = attempts.get(k, 0) + 1
                if rec.get("ok"):
                    e = rec["end"]
                    rt = " ".join(f"{L}: eta {r['eta']:.2f} map {r['ch_map']} m {r['margin']:.2f}"
                                  for L, r in (e.get("routing") or {}).items())
                    log(f"  {j['arm']:<12} {j['seed']}  SET {e['set']:.3f} loss {e['loss']:.3f} "
                        f"bins {[round(e['set_' + b], 3) for b in it.BIN_NAMES]} {rt}"
                        f"{' splits ' + str(rec.get('splits')) if 'splits' in rec else ''}  {rec['secs'] / 60:.1f} min")
                else:
                    log(f"  {j['arm']:<12} {j['seed']}  FAILED ({rec.get('error')})")
                    if attempts[k] < max_attempts:
                        queue.append(j)
            submit()
    return load(name)


def runs(name, arm, st=None):
    """{seed: record}: the arm's ok records under the store's current code SHA (and, by the key, this CPU)."""
    st = load(name) if st is None else st
    out = {}
    for k, r in st["runs"].items():
        a, s = k.split("|")[:2]
        if a == arm and r.get("ok") and r.get("sha") == st["meta"].get("sha"):
            out[int(s)] = r
    return out


# ── statistics ────────────────────────────────────────────────────────────────
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
        return "-"
    if k >= math.ceil(0.9 * n):
        return "RELIABLE"
    if k >= math.ceil(0.5 * n):
        return "MAJORITY"
    return "MINORITY" if k >= 1 else "NEVER"


def mcnemar_greater(b, c):
    """Exact one-sided McNemar: P(Bin(b + c, 1/2) >= b)."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n
