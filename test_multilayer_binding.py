#!/usr/bin/env python
"""
test_multilayer_binding.py — can multilayer BDH bind a key to the value that follows it,
and if it can, do multiple channels beat single-channel alternatives?

Run it directly:

    python test_multilayer_binding.py                   # Gate 0, then everything
    python test_multilayer_binding.py --stage gate0     # Gate 0 only: grid, then stop
    python test_multilayer_binding.py --seeds 3 --force --threads 4

WHY. test_binding_capacity.py showed the one-layer instrument cannot bind: at every
position its memory stores (x_s -> v_s) for the SAME token, so reading at KEY_q returns
KEY_q's own embedding and never the value after it. Its perfect-separation ceiling at P=4
was 0.29 ~ 1/P against a floor of ~1/(S*P): separation buys the factor S, and binding,
which that architecture lacks, would buy the factor P. The old interference task never
needed binding (one flip bit per sequence). The reference bdh.py runs n_layer=6 with
shared weights and RoPE as its positional operator; stacking layers with RoPE is the
native way BDH could form a previous-token circuit and bind. Two questions, in order:
  (1) can multilayer BDH bind on this task at all?        -> GATE 0, which can end the test
  (2) if so, do channels beat single-channel alternatives? -> the arm comparison

MODEL, built on bdh.py rather than the one-layer instrument:
  * bdh.BDH is subclassed and its forward() — the layer loop, shared weights across
    layers, LayerNorms, sparse ReLU codes, decoder — is called UNMODIFIED. The only
    substitution is self.attn, replaced by GatedAttention (a bdh.Attention subclass).
  * GatedAttention mirrors bdh.Attention.forward's four RoPE lines exactly, using bdh's own
    rope() and freqs buffer; the lines are mirrored rather than called because bdh never
    exposes the score matrix. CHECK 2 proves them equal. Channels enter as
        scores[t,s] = RoPE-score[t,s] * (g_read[t] . g_write[s]),   strictly causal
    applied identically at EVERY layer (CHECK 6 observes it inside each layer).
  * The gate is the instrument's own code: MultiBDH calls test_instrument_v2.Instrument
    .gates() on itself, so the recurrent gate h_t = tanh(W_in v_t + W_h h_{t-1}),
    g_t = softmax(W_g h_t) over the raw embeddings — and the uniform and perfect gates —
    are IMPORTED, not mirrored. CHECK 5 compares against an Instrument bitwise.
  * gate_to_readout adds W_ro @ h before lm_head. lm_head is linear, so this is computed
    as logits + (h W_ro^T) lm_head, which is exactly that.
  * positional = 'rope' (bdh's operator) or 'decay' (the instrument's scalar decay mask at
    OPERATING_DECAY), the latter for Gate 0 only.
  * n_head=1, dropout=0, n_embd=32, mlp_internal_dim_multiplier=2 so N=64 (the
    instrument's scale). n_layer is a knob. Loss and accuracy on query value positions only.

TASK. test_binding_capacity.BindTask, IMPORTED, not rewritten (CHECK 4): S streams, S
distinct values per key from n_vals, n_q queries without replacement, randomized order.

GATE 0 — CAN IT BIND? The step this project skipped for six experiments, so it runs first
and can end the test. S=1 (one stream, no interference), P=4, n_vals=16, n_q=4, one
channel, no gate. n_layer in {1,2,3} x positional in {rope, decay}, 3 seeds each, ITERS
climbing 1200 -> 2400 -> 4800 -> 9600 until all 3 seeds reach >= 0.95 or the cap is hit.
The full grid is printed. PRE-REGISTERED choice among combinations that bind: the
smallest n_layer, then the smallest ITERS, then rope before decay. That (n_layer,
positional, ITERS) is used for everything below. If nothing binds the verdict is
UNTESTED and the script STOPS — the channel comparison is never run on an architecture
that cannot bind.

GATE 1, per config: the perfect gate (k=S) must reach >= 0.95 on every seed, and the
recurrent gate must fit the perfect routing (argmax match >= 0.99). Floors are measured
per seed; seeds more than FLOOR_FLAG above 1/S are flagged. Failing configs are excluded
and reported as excluded; arms run on valid configs only.

ARMS, per valid config (k = S where applicable):
  floor    uniform gate, k=S          ceiling  perfect gate, k=S
  A        recurrent gate, k=S, no readout
  A_ro     recurrent gate, k=S, + readout
  B        plain single-channel multilayer BDH — a serious baseline: multilayer attention
           with RoPE might resolve interference without channels, e.g. by attending to
           the KEY that follows CTX_q
  E        1 channel + recurrent-gate readout (h_gate 32; also the h=32 point of E_h)
  E_mem    1 channel, sparse width N*S (A's memory), + readout
  E_h64, E_h128   1 channel + readout with h_gate 64 / 128. A width sweep, not a
           parameter match: channels share their projections, so a match is degenerate.
CONFIGS. S=2, n_vals=16, n_q=4, P in {4, 8, 16, 32}; 5 seeds; same budget for every arm.

PRE-REGISTERED VERDICT, EPS = 0.10, printed before any interpretation. Exactly one of:
  UNTESTED       Gate 0 fails, or every config fails Gate 1.
  SUPPORTED      At some valid config, max(A, A_ro) beats max(B, E, E_mem, E_h*) by >= EPS
                 in mean accuracy, beats that control on every seed, AND the winning
                 channel arm separates (VAL gate cosine between streams < 0.5, every seed).
  NOT SUPPORTED  At least one valid config, and none meets that bar.

DIAGNOSTICS, not part of the verdict: accuracy vs P for every arm; the P at which each
E_h width falls below 0.9 (a shift with width indicates an RNN capacity limit); whether B
falls with P, and where.

LOGISTICS. Gate 0 runs alone first and its grid is reported before anything else. Then
every arm is timed on every config and the projected wall clock printed; over
WALL_LIMIT_H, seeds 0-2 run first with an interim report, then the rest. Reporting follows
test_phase2_seeds.py: per-seed raw values before any aggregate, population std,
"identical: X", failures recorded per seed and never dropped, PARTIAL labels. Results
persist atomically to multilayer_binding_results.json.

(Results are recorded at the bottom of this docstring after the run.)
"""

import argparse
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F

import bdh
from bdh import BDH, BDHConfig, Attention, get_freqs
import test_instrument_v2 as tiv
from test_instrument_v2 import (
    Instrument, OPERATING_DECAY, LR, BATCH, D, N, H_GATE, PROBE_B, N_CKPT, TAU_END,
    SEP_COS, decay_mask, perfect_gate_general,
)
import test_binding_capacity as tbc
from test_binding_capacity import BindTask, stats, cell, save_results

# ── Settings ─────────────────────────────────────────────────────────────────
EPS = 0.10                        # PRE-REGISTERED win margin
MULT = 2                          # mlp_internal_dim_multiplier: N = MULT * D = 64
N_SEEDS = 5
S_STREAMS, N_VALS, N_Q = 2, 16, 4
P_LIST = [4, 8, 16, 32]
LADDER = (1200, 2400, 4800, 9600)
BIND_PASS = 0.95                  # Gate 0: "binds"
CEIL_PASS = 0.95                  # Gate 1: ceiling
FIT_PASS = 0.99                   # Gate 1: recurrent gate fits perfect routing
FIT_ITERS = 400
FLOOR_FLAG = 0.05
E_DROP = 0.90
G0_LAYERS = (1, 2, 3)
G0_POS = ("rope", "decay")
G0_SEEDS = (0, 1, 2)
E_H_WIDTHS = (64, 128)            # plus E itself at h_gate = H_GATE = 32
WALL_LIMIT_H = 3.0
THREADS = 1
TIME_ITERS = (10, 30)
RESULTS_FILE = "multilayer_binding_results.json"


# ── Model ────────────────────────────────────────────────────────────────────
class GatedAttention(Attention):
    """
    bdh.Attention with two additions, both off by default:
      * G, a (B,1,T,T) channel-gate matrix g_read[t].g_write[s], multiplies the score
        before the strictly-causal mask. G=None reproduces bdh.Attention bit for bit.
      * positional='decay' replaces RoPE by the instrument's scalar decay mask.
    The RoPE branch mirrors bdh.Attention.forward line for line (CHECK 2).
    """

    def __init__(self, config, positional="rope", decay=OPERATING_DECAY):
        super().__init__(config)
        assert positional in ("rope", "decay")
        self.positional, self.decay = positional, decay
        self.G = None
        self.record = None            # CHECK 6 only: collects each layer's score matrix

    def forward(self, Q, K, V):
        assert self.freqs.dtype == torch.float32
        assert K is Q
        _, _, T, _ = Q.size()
        if self.positional == "rope":
            r_phases = (
                torch.arange(0, T, device=self.freqs.device,
                             dtype=self.freqs.dtype).view(1, 1, -1, 1)
            ) * self.freqs
            QR = self.rope(r_phases, Q)
            KR = QR
            scores = QR @ KR.mT
        else:
            scores = (Q @ K.mT) * decay_mask(T, self.decay, Q.dtype).to(Q.device)
        if self.G is not None:
            scores = scores * self.G
        scores = scores.tril(diagonal=-1)
        if self.record is not None:
            self.record.append(scores.detach())
        return scores @ V


class MultiBDH(BDH):
    """bdh.BDH with a channel gate. forward() calls BDH.forward unmodified."""

    def __init__(self, vocab, n_layer, gate="none", n_ch=1, positional="rope",
                 gate_to_readout=False, h_gate=None, mult=MULT, ctx_tokens=None,
                 decay=OPERATING_DECAY):
        cfg = BDHConfig(n_layer=n_layer, n_embd=D, dropout=0.0, n_head=1,
                        mlp_internal_dim_multiplier=mult, vocab_size=vocab)
        super().__init__(cfg)
        self.attn = GatedAttention(cfg, positional, decay)
        # Attributes read by test_instrument_v2.Instrument.gates, which is called below.
        self.gate_kind, self.mode, self.n_ch = gate, "sym", n_ch
        self.ctx_tokens = None if ctx_tokens is None else tuple(ctx_tokens)
        self.h_gate = H_GATE if h_gate is None else int(h_gate)
        self.gate_to_readout = bool(gate_to_readout)
        self.n_feat, self.positional, self.n_layers = cfg.mlp_internal_dim_multiplier * D, \
            positional, n_layer
        hg = self.h_gate
        if gate == "recurrent":
            self.W_in = nn.Parameter(torch.randn(hg, D) * 0.1)
            self.W_h = nn.Parameter(torch.randn(hg, hg) * 0.1)
            self.W_g = nn.Parameter(torch.randn(n_ch, hg) * 0.1)
        elif gate == "token":
            self.W_tok = nn.Parameter(torch.randn(D, n_ch) * 0.1)
        if self.gate_to_readout:
            assert gate == "recurrent", "gate_to_readout needs the recurrent gate"
            self.W_ro = nn.Parameter(torch.randn(D, hg) * 0.1)

    def forward(self, tokens, tau=None, track_sat=False):
        v = self.embed(tokens)
        gr, gw = Instrument.gates(self, tokens, v, tau)          # the instrument's own code
        self.attn.G = (None if self.gate_kind == "none"
                       else torch.einsum("btk,bsk->bts", gr, gw).unsqueeze(1))
        logits, _ = BDH.forward(self, tokens)                     # bdh's own layer loop
        if self.gate_to_readout:
            logits = logits + (self._h_seq @ self.W_ro.T) @ self.lm_head
        return logits, None, gr, gw


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def n_state(m):
    """Fast-weight floats: each layer's linear-attention memory, n_ch x (N x D) apiece."""
    return m.n_layers * m.n_ch * m.n_feat * D


# ── Training: test_binding_capacity.run_bind's structure, for MultiBDH ──────
def run_ml(task, arch, gate, n_ch, seed, iters, gate_to_readout=False, h_gate=None,
           mult=MULT, ckpts=None):
    torch.manual_seed(seed)
    model = MultiBDH(task.vocab, arch["n_layer"], gate, n_ch, arch["positional"],
                     gate_to_readout=gate_to_readout, h_gate=h_gate, mult=mult,
                     ctx_tokens=task.ctx_tokens)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = torch.Generator(); rng.manual_seed(seed + 10_000)
    prng = torch.Generator(); prng.manual_seed(seed + 99_000)
    ptok, _ = task.make_batch(PROBE_B, prng)
    pinp, plab = ptok[:, :-1], task.stream_labels(ptok[:, :-1])
    if ckpts is None:
        ckpts = sorted({int(i * (iters - 1) / (N_CKPT - 1)) for i in range(N_CKPT)})
    trace = []
    prole = task.role_masks(pinp)

    def diag(step):
        with torch.no_grad():
            _, _, gr, _ = model(pinp)
            d = {"step": step}
            for rn, rm in prole.items():
                a = gr[(plab == 0) & rm].mean(0)
                b = gr[(plab == 1) & rm].mean(0)
                d[f"{rn}_cos"] = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            return d

    for step in range(iters):
        if step in ckpts and n_ch > 1:
            trace.append(diag(step))
        tokens, _ = task.make_batch(BATCH, rng)
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        logits, _, _, _ = model(inp, TAU_END)
        ql, qt, _ = task.select(logits, tgt, None)
        loss = F.cross_entropy(ql, qt)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    if n_ch > 1:
        trace.append(diag(iters))
    with torch.no_grad():
        tokens, qs = task.make_batch(512, rng)
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        logits, _, _, _ = model(inp, TAU_END)
        ql, qt, qs = task.select(logits, tgt, qs)
        corr = (ql.argmax(-1) == qt)
        accs = [corr[qs == s].float().mean().item() for s in range(task.S)]
        losses = [F.cross_entropy(ql[qs == s], qt[qs == s]).item() for s in range(task.S)]
    return dict(accs=accs, losses=losses, acc=sum(accs) / len(accs), trace=trace)


def gate_fit(task, arch):
    """Mirror of test_binding_capacity.gate_fit on THIS model's gate and embeddings."""
    g = torch.Generator().manual_seed(11)
    tok, _ = task.make_batch(64, g)
    x = tok[:, :-1]
    tgt = perfect_gate_general(x, task.ctx_tokens)
    torch.manual_seed(3)
    m = MultiBDH(task.vocab, arch["n_layer"], "recurrent", task.S, arch["positional"],
                 ctx_tokens=task.ctx_tokens)
    opt = torch.optim.Adam([m.W_in, m.W_h, m.W_g, m.embed.weight], lr=1e-2)
    for _ in range(FIT_ITERS):
        gr, _ = Instrument.gates(m, x, m.embed(x))
        loss = F.mse_loss(gr, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        gr, _ = Instrument.gates(m, x, m.embed(x))
        return F.mse_loss(gr, tgt).item(), (gr.argmax(-1) == tgt.argmax(-1)).float().mean().item()


# ── Persistence and one recorded run ─────────────────────────────────────────
def load_results(path):
    blank = {"meta": {}, "runs": {}, "gate0": {}, "gate1": {}}
    if not os.path.exists(path):
        return blank
    try:
        with open(path) as f:
            d = json.load(f)
        for kk in blank:
            d.setdefault(kk, {})
        return d
    except Exception as e:
        print(f"  ! could not read {path} ({e}); starting fresh")
        return blank


def rkey(cfg, arm, arch, iters):
    return f"{cfg}|{arm}|L{arch['n_layer']}{arch['positional']}|{iters}"


def get_run(store, cfg, arm, arch, iters, seed):
    return store["runs"].get(rkey(cfg, arm, arch, iters), {}).get(str(seed))


def do_run(store, path, task, arm, arch, iters, seed, force, t0, tag=""):
    rk = rkey(task.key, arm["key"], arch, iters)
    store["runs"].setdefault(rk, {})
    cached = store["runs"][rk].get(str(seed))
    if cached is not None and not force:
        print(f"   {task.key:>4} {arm['key']:>7} L{arch['n_layer']}-{arch['positional']:<5} "
              f"@{iters:<4} seed {seed}  cached ({'ok' if cached.get('ok') else 'FAILED'}){tag}")
        return cached
    ts = time.time()
    try:
        r = run_ml(task, arch, arm["gate"], arm["n_ch"], seed, iters,
                   gate_to_readout=arm["g2r"], h_gate=arm["h_gate"], mult=arm["mult"])
        vals = r["accs"] + r["losses"]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            raise ValueError(f"non-finite result: accs={r['accs']} losses={r['losses']}")
        last = r["trace"][-1] if r["trace"] else {}
        rec = dict(ok=True, seed=seed, iters=iters, accs=r["accs"], losses=r["losses"],
                   acc=r["acc"], key_cos=last.get("key_cos"), val_cos=last.get("val_cos"),
                   ctx_cos=last.get("ctx_cos"), secs=time.time() - ts)
    except Exception as e:
        rec = dict(ok=False, seed=seed, iters=iters, error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc()[-1500:], secs=time.time() - ts)
    store["runs"][rk][str(seed)] = rec
    save_results(path, store)
    el = (time.time() - t0) / 60
    if rec["ok"]:
        vc = "" if rec["val_cos"] is None else f"  VALcos={rec['val_cos']:.4f}"
        print(f"   {task.key:>4} {arm['key']:>7} L{arch['n_layer']}-{arch['positional']:<5} "
              f"@{iters:<4} seed {seed}  acc={rec['acc']:.4f}{vc}   {rec['secs']:.0f}s  "
              f"(elapsed {el:.1f}m){tag}")
    else:
        print(f"   {task.key:>4} {arm['key']:>7} L{arch['n_layer']}-{arch['positional']:<5} "
              f"@{iters:<4} seed {seed}  *** FAILED: {rec['error']}   (elapsed {el:.1f}m){tag}")
    return rec


# ── Arms ─────────────────────────────────────────────────────────────────────
def arm(key, label, gate, n_ch, g2r=False, h_gate=None, mult=MULT):
    return dict(key=key, label=label, gate=gate, n_ch=n_ch, g2r=g2r, h_gate=h_gate,
                mult=mult)


def arms_for(task):
    S = task.S
    out = [
        arm("floor", "floor    uniform gate, k=S", "uniform", S),
        arm("ceiling", "ceiling  perfect gate, k=S", "perfect", S),
        arm("A", "A        recurrent k=S, no readout", "recurrent", S),
        arm("A_ro", "A_ro     recurrent k=S + readout", "recurrent", S, g2r=True),
        arm("B", "B        plain single channel", "none", 1),
        arm("E", "E        1ch + readout (h=32)", "recurrent", 1, g2r=True),
        arm("E_mem", "E_mem    1ch, width N*S + readout", "recurrent", 1, g2r=True,
            mult=MULT * S),
    ]
    for h in E_H_WIDTHS:
        out.append(arm(f"E_h{h}", f"E_h{h:<4}  1ch + readout, h_gate={h}", "recurrent", 1,
                       g2r=True, h_gate=h))
    return out


G0_ARM = arm("B", "single channel, no gate", "none", 1)
CHANNEL_ARMS = ("A", "A_ro")
CONTROL_ARMS = ("B", "E", "E_mem") + tuple(f"E_h{h}" for h in E_H_WIDTHS)


def build(task, a, arch, seed=0):
    torch.manual_seed(seed)
    return MultiBDH(task.vocab, arch["n_layer"], a["gate"], a["n_ch"], arch["positional"],
                    gate_to_readout=a["g2r"], h_gate=a["h_gate"], mult=a["mult"],
                    ctx_tokens=task.ctx_tokens)


# ── Verification ─────────────────────────────────────────────────────────────
def verify():
    print("=" * 100)
    print("VERIFICATION — printed before any training")
    print("=" * 100)
    ok = True
    Vt = 21
    g = torch.Generator().manual_seed(1)
    tok = torch.randint(0, Vt, (4, 24), generator=g)

    print("CHECK 1  k=1, no gate, RoPE, weights copied from a bdh.BDH instance (n_head=1,")
    print("         dropout=0): MultiBDH reproduces bdh.BDH's logits, per n_layer. The copy")
    print("         uses load_state_dict(strict=True), so the parameter sets are identical.")
    worst = 0.0
    for L in (1, 2, 3):
        cfg = BDHConfig(n_layer=L, n_embd=D, dropout=0.0, n_head=1,
                        mlp_internal_dim_multiplier=MULT, vocab_size=Vt)
        torch.manual_seed(7)
        ref = BDH(cfg).eval()
        with torch.no_grad():
            lr, _ = ref(tok)
        for gname in ("none", "uniform"):
            m = MultiBDH(Vt, L, gname, 1, "rope").eval()
            m.load_state_dict(ref.state_dict(), strict=True)
            with torch.no_grad():
                lm, *_ = m(tok)
            d = (lr - lm).abs().max().item()
            worst = max(worst, d)
            path = "G=None" if gname == "none" else "G=ones via uniform k=1"
            print(f"         n_layer={L}  {path:<24} max |logits - bdh.BDH| = {d:.2e}")
    ok &= worst < 1e-6
    print(f"         -> {'REPRODUCES bdh.BDH' if worst < 1e-6 else 'FAILED'}\n")

    print("CHECK 2  GatedAttention's RoPE scores equal bdh.Attention's on random input:")
    cfg = BDHConfig(n_layer=1, n_embd=D, dropout=0.0, n_head=1,
                    mlp_internal_dim_multiplier=MULT, vocab_size=Vt)
    T = 30
    Q = F.relu(torch.randn(3, 1, T, N, generator=g))
    Vr = torch.randn(3, 1, T, D, generator=g)
    eye = torch.eye(T).expand(3, 1, T, T)
    ref_att, my_att = Attention(cfg), GatedAttention(cfg, "rope")
    ds = (ref_att(Q, Q, eye) - my_att(Q, Q, eye)).abs().max().item()
    do = (ref_att(Q, Q, Vr) - my_att(Q, Q, Vr)).abs().max().item()
    print(f"         scores (V = identity): max diff = {ds:.2e}")
    print(f"         outputs (random V)   : max diff = {do:.2e}")
    ok &= ds == 0.0 and do == 0.0
    print(f"         -> {'IDENTICAL' if ds == 0.0 and do == 0.0 else 'FAILED'}\n")

    print("CHECK 3  gate algebra: a uniform gate over k channels gives g_r.g_w = k*(1/k)^2 =")
    print("         1/k, so the multichannel score must equal the single-channel score / k:")
    worst = 0.0
    for kk in (2, 3, 4):
        gu = torch.full((3, T, kk), 1.0 / kk)
        my_att.G = torch.einsum("btk,bsk->bts", gu, gu).unsqueeze(1)
        s_k = my_att(Q, Q, eye)
        my_att.G = None
        s_1 = my_att(Q, Q, eye)
        d = (s_k - s_1 / kk).abs().max().item()
        worst = max(worst, d)
        print(f"         k={kk}: max |score_k - score_1/k| = {d:.2e}")
    ok &= worst < 1e-5
    print(f"         -> {'CONFIRMED' if worst < 1e-5 else 'FAILED'}")
    torch.manual_seed(9); mb = MultiBDH(Vt, 2, "none", 1, "rope").eval()
    torch.manual_seed(9); mu = MultiBDH(Vt, 2, "uniform", 2, "rope").eval()
    mu.load_state_dict(mb.state_dict(), strict=True)
    with torch.no_grad():
        dd = (mb(tok)[0] - mu(tok)[0]).abs().max().item()
    print(f"         model level (informative): uniform k=2 vs single channel, same weights,")
    print(f"         max |logits diff| = {dd:.2e}. bdh.py LayerNorms yKV right after the")
    print(f"         attention, which cancels the 1/k up to LayerNorm's eps — so the uniform-")
    print(f"         gate floor and the plain single channel B are nearly the same function.\n")

    print("CHECK 4  the task generator is REUSED from test_binding_capacity, not rewritten:")
    same = BindTask.make_batch is tbc.BindTask.make_batch
    print(f"         BindTask.__module__ = {BindTask.__module__!r}")
    print(f"         BindTask.make_batch is test_binding_capacity.BindTask.make_batch: {same}")
    t1 = BindTask(4, S=1, n_vals=N_VALS, n_q=N_Q)
    gg = torch.Generator().manual_seed(2)
    tk, qs = t1.make_batch(256, gg)
    body = tk[:, :3 * t1.P].reshape(256, t1.P, 3)
    q = tk[:, 3 * t1.P:].reshape(256, t1.n_q, 3)
    bound_ok = all(
        {int(body[b, i, 1]): int(body[b, i, 2]) for i in range(t1.P)}[int(q[b, j, 1])]
        == int(q[b, j, 2]) for b in range(64) for j in range(t1.n_q))
    one_stream = bool((tk[:, 0::3] == 0).all())
    print(f"         at S=1 (Gate 0): one context token throughout = {one_stream}; every query")
    print(f"         answered by its key's bound value = {bound_ok}")
    ok &= same and bound_ok and one_stream
    print(f"         -> {'REUSED' if same else 'NOT REUSED'}\n")

    print("CHECK 5  the gate is test_instrument_v2.Instrument.gates itself: MultiBDH vs an")
    print("         Instrument holding the same gate weights and embeddings, bitwise:")
    t2 = BindTask(8)
    x = t2.make_batch(16, torch.Generator().manual_seed(4))[0][:, :-1]
    worst = 0.0
    for gname in ("recurrent", "perfect", "uniform"):
        torch.manual_seed(5); mm = MultiBDH(t2.vocab, 2, gname, 2, ctx_tokens=t2.ctx_tokens)
        torch.manual_seed(5); ii = Instrument(gname, "sym", 2, 1.0, None, vocab=t2.vocab,
                                              ctx_tokens=t2.ctx_tokens)
        with torch.no_grad():
            ii.embed.weight.copy_(mm.embed.weight)
            if gname == "recurrent":
                ii.W_in.copy_(mm.W_in); ii.W_h.copy_(mm.W_h); ii.W_g.copy_(mm.W_g)
            a = Instrument.gates(mm, x, mm.embed(x))[0]
            b = ii.gates(x, ii.embed(x))[0]
        d = (a - b).abs().max().item()
        worst = max(worst, d)
        print(f"         {gname:>9}: max |MultiBDH gate - Instrument gate| = {d:.2e}")
    ok &= worst == 0.0
    print(f"         -> {'IDENTICAL' if worst == 0.0 else 'FAILED'}\n")

    print("CHECK 6  one gate, applied identically at EVERY layer: with the perfect gate, every")
    print("         cross-stream score must be exactly zero inside each layer's attention:")
    torch.manual_seed(6)
    mp = MultiBDH(t2.vocab, 3, "perfect", 2, ctx_tokens=t2.ctx_tokens).eval()
    mp.attn.record = []
    with torch.no_grad():
        mp(x)
    lab = t2.stream_labels(x)
    cross = (lab.unsqueeze(2) != lab.unsqueeze(1)) & (lab.unsqueeze(2) >= 0) & (lab.unsqueeze(1) >= 0)
    maxc = [s[:, 0][cross].abs().max().item() for s in mp.attn.record]
    live = [s[:, 0][~cross].abs().max().item() for s in mp.attn.record]
    print(f"         layers seen: {len(mp.attn.record)} (n_layer=3)")
    for i, (c, l) in enumerate(zip(maxc, live)):
        print(f"         layer {i}: max |cross-stream score| = {c:.2e}   "
              f"max |same-stream score| = {l:.2e}")
    mp.attn.record = None
    ok &= len(maxc) == 3 and max(maxc) == 0.0 and min(live) > 0
    print(f"         -> {'APPLIED AT EVERY LAYER' if ok else 'FAILED'}\n")

    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"


# ── Gate 0 ───────────────────────────────────────────────────────────────────
def gate0(store, path, force, t0):
    task = BindTask(4, S=1, n_vals=N_VALS, n_q=N_Q)
    task.key = "G0"
    print("=" * 100)
    print("GATE 0 — CAN MULTILAYER BDH BIND?  S=1 (no interference), P=4, n_vals=16, n_q=4,")
    print(f"one channel, no gate. Binds = every seed >= {BIND_PASS}. 1/P = {1 / task.P:.3f}.")
    print("=" * 100)
    grid = {}
    for L in G0_LAYERS:
        for pos in G0_POS:
            arch = dict(n_layer=L, positional=pos)
            grid[(L, pos)] = {}
            for it in LADDER:
                accs = []
                for s in G0_SEEDS:
                    r = do_run(store, path, task, G0_ARM, arch, it, s, force, t0,
                               tag="  [gate 0]")
                    accs.append(r["acc"] if r.get("ok") else None)
                grid[(L, pos)][it] = accs
                if all(a is not None and a >= BIND_PASS for a in accs):
                    break
    print()
    print("  GATE 0 GRID — query accuracy per seed, by rung (a combination stops climbing")
    print("  once all three seeds bind; '·' = rung not needed)")
    print(f"  {'n_layer':>7} {'pos':>6} | " + " | ".join(f"{'@' + str(it):^22}" for it in LADDER)
          + " | binds at")
    binds = []
    for L in G0_LAYERS:
        for pos in G0_POS:
            row, at = [], None
            for it in LADDER:
                accs = grid[(L, pos)].get(it)
                if accs is None:
                    row.append(f"{'·':^22}")
                    continue
                row.append(" ".join("--" if a is None else f"{a:.3f}" for a in accs).center(22))
                if at is None and all(a is not None and a >= BIND_PASS for a in accs):
                    at = it
            if at is not None:
                binds.append((L, at, 0 if pos == "rope" else 1, pos))
            print(f"  {L:>7} {pos:>6} | " + " | ".join(row) + f" | {at if at else 'never'}")
    print()
    chosen = None
    if binds:
        L, it, _, pos = min(binds)
        chosen = dict(n_layer=L, positional=pos, iters=it)
        print(f"  -> BINDS. Pre-registered choice (smallest n_layer, then ITERS, then rope):")
        print(f"     n_layer={L}, positional={pos}, ITERS={it} — used for everything below.")
    else:
        print("  -> UNTESTED: multilayer BDH at this scale does not bind within budget.")
    store["gate0"] = dict(chosen=chosen, grid={f"{L}|{p}|{it}": v for (L, p), d in grid.items()
                                               for it, v in d.items()})
    save_results(path, store)
    print()
    return chosen


# ── Reporting ────────────────────────────────────────────────────────────────
def accs(store, cfg, key, arch, it, seeds):
    out = []
    for s in seeds:
        r = get_run(store, cfg, key, arch, it, s)
        out.append(r["acc"] if (r and r.get("ok")) else None)
    return out


def report(store, tasks, seeds, arch, it, g1, info, wall, path, partial_note=""):
    print("#" * 100)
    print(("INTERIM REPORT" + partial_note) if partial_note else "FINAL REPORT")
    print("#" * 100)
    print(f"  architecture from Gate 0: n_layer={arch['n_layer']}, "
          f"positional={arch['positional']}, ITERS={it}")
    print()
    valid = [t for t in tasks if g1[t.key]["valid"]]
    for t in tasks:
        ex = g1[t.key]
        print("=" * 100)
        print(f"{t.label}   {'VALID' if ex['valid'] else 'EXCLUDED — ' + ex['reason']}")
        print("=" * 100)
        keys = [a["key"] for a in arms_for(t)]
        W = 9
        print("  PER-SEED RAW ACCURACY — every value, before any aggregate")
        print("  seed " + "".join(f"{kk:>{W}}" for kk in keys))
        for s in seeds:
            row = ""
            for kk in keys:
                r = get_run(store, t.key, kk, arch, it, s)
                row += f"{'--' if r is None else ('FAIL' if not r.get('ok') else format(r['acc'], '.4f')):>{W}}"
            print(f"  {s:>4} " + row)
        if ex["valid"]:
            print("  per-seed VAL gate cosine between streams (channel arms):")
            for kk in CHANNEL_ARMS:
                vc = [get_run(store, t.key, kk, arch, it, s) for s in seeds]
                vc = [None if (r is None or not r.get("ok")) else r.get("val_cos") for r in vc]
                print(f"    {kk:>5}: " + "  ".join("--" if v is None else f"{v:.4f}" for v in vc))
        chance = 1.0 / t.S
        fl = accs(store, t.key, "floor", arch, it, seeds)
        flags = []
        for s, f in zip(seeds, fl):
            if f is None:
                continue
            if f > chance + FLOOR_FLAG:
                flags.append(f"seed {s}: floor {f:.4f} RISES more than {FLOOR_FLAG} above 1/S")
        print(f"  floor vs 1/S = {chance:.3f}: {'no seed rises above it' if not flags else ''}")
        for f in flags:
            print(f"    FLAG {f}")
        print()
        print(f"  {'arm':<36} {'params':>7} {'state':>7} {'accuracy mean +- std':>24} "
              f"{'min':>7} {'max':>7} {'n':>3}")
        for a in arms_for(t):
            kk = a["key"]
            xs = accs(store, t.key, kk, arch, it, seeds)
            nf = sum(1 for s in seeds if get_run(store, t.key, kk, arch, it, s) is not None
                     and not get_run(store, t.key, kk, arch, it, s).get("ok"))
            st = stats([x for x in xs if x is not None])
            p, sf = info[t.key][kk]
            if st is None:
                body = f"{'not run' if not ex['valid'] else 'no completed seeds':>24}"
                print(f"  {a['label']:<36} {p:>7} {sf:>7} {body}"
                      + (f"   [{nf} FAILED]" if nf else ""))
            else:
                print(f"  {a['label']:<36} {p:>7} {sf:>7} {cell(st):>24} "
                      f"{st['min']:>7.4f} {st['max']:>7.4f} {st['n']:>3}"
                      + (f"   [{nf} FAILED]" if nf else ""))
        print()

    # ── Verdict ─────────────────────────────────────────────────────────────
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"  PRE-REGISTERED (set before any result was seen), EPS = {EPS}. Exactly one of:")
    print("    UNTESTED       Gate 0 fails, or every config fails Gate 1.")
    print("    SUPPORTED      At some valid config, max(A, A_ro) beats max(B, E, E_mem, E_h*)")
    print("                   by >= EPS in mean accuracy, beats that control on every seed, AND")
    print("                   the winning channel arm separates (VAL cos < 0.5, every seed).")
    print("    NOT SUPPORTED  At least one valid config, and none meets that bar.")
    print()
    attempted = completed = 0
    for t in valid:
        for a in arms_for(t):
            for s in seeds:
                attempted += 1
                r = get_run(store, t.key, a["key"], arch, it, s)
                completed += bool(r and r.get("ok"))
    print(f"  {completed}/{attempted} arm runs on valid configs completed"
          + ("   *** PARTIAL SWEEP ***" if (completed < attempted or partial_note) else ""))
    if completed < attempted:
        for t in valid:
            for a in arms_for(t):
                for s in seeds:
                    r = get_run(store, t.key, a["key"], arch, it, s)
                    if r is None:
                        print(f"    {t.key} {a['key']} seed {s}: NOT RUN")
                    elif not r.get("ok"):
                        print(f"    {t.key} {a['key']} seed {s}: FAILED — {r['error']}")
    print()
    wins = []
    for t in tasks:
        if not g1[t.key]["valid"]:
            print(f"  {t.key:>4}: EXCLUDED — {g1[t.key]['reason']}")
            continue
        m = lambda kk: stats([x for x in accs(store, t.key, kk, arch, it, seeds) if x is not None])
        ch = max(CHANNEL_ARMS, key=lambda kk: (m(kk) or {"mean": -1})["mean"])
        ct = max(CONTROL_ARMS, key=lambda kk: (m(kk) or {"mean": -1})["mean"])
        mc, mt = m(ch), m(ct)
        if mc is None or mt is None:
            print(f"  {t.key:>4}: no verdict (incomplete)")
            continue
        margin = mc["mean"] - mt["mean"]
        pairs = [(a, b) for a, b in zip(accs(store, t.key, ch, arch, it, seeds),
                                        accs(store, t.key, ct, arch, it, seeds))
                 if a is not None and b is not None]
        every = bool(pairs) and all(a > b for a, b in pairs)
        vcs = [get_run(store, t.key, ch, arch, it, s) for s in seeds]
        vcs = [r.get("val_cos") for r in vcs if r and r.get("ok")]
        sep = bool(vcs) and all(v is not None and v < SEP_COS for v in vcs)
        win = margin >= EPS and every and sep
        wins += [t.key] if win else []
        print(f"  {t.key:>4}: best channel {ch:<5} {mc['mean']:.4f}   best control {ct:<7} "
              f"{mt['mean']:.4f}   margin {margin:+.4f} ({'>=' if margin >= EPS else '<'} EPS)")
        print(f"        beats that control on every seed: {every}   winning channel arm "
              f"separates: {sep}   -> {'CHANNELS WIN' if win else 'no win'}")
    print()
    if not valid:
        verdict = "UNTESTED"
        why = "every config failed Gate 1."
    elif wins:
        verdict = "SUPPORTED"
        why = f"channels win at {wins}."
    else:
        verdict = "NOT SUPPORTED"
        why = f"valid configs {[t.key for t in valid]}, and none meets the bar."
    print(f"  *** {verdict}: {why} ***")

    # ── Diagnostics ─────────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    if not valid:
        print("  none: no valid config.")
    else:
        keys = [a["key"] for a in arms_for(valid[0])]
        print("  mean accuracy vs P (valid configs only):")
        print(f"  {'arm':>8}" + "".join(f"{'P=' + str(t.P):>10}" for t in valid))
        means = {}
        for kk in keys:
            row = []
            for t in valid:
                st = stats([x for x in accs(store, t.key, kk, arch, it, seeds) if x is not None])
                row.append(None if st is None else st["mean"])
            means[kk] = row
            print(f"  {kk:>8}" + "".join(f"{('--' if v is None else format(v, '.4f')):>10}"
                                         for v in row))
        print(f"  {'1/S':>8}" + "".join(f"{1 / t.S:>10.4f}" for t in valid))
        print()

        def first_below(kk):
            for t, v in zip(valid, means[kk]):
                if v is not None and v < E_DROP:
                    return t.P
            return None
        print(f"  first P at which each readout width falls below {E_DROP}:")
        widths = [("E", H_GATE)] + [(f"E_h{h}", h) for h in E_H_WIDTHS]
        drops = []
        for kk, h in widths:
            p = first_below(kk)
            drops.append(p)
            print(f"    h_gate={h:>3}: {p if p is not None else 'never, within the valid P range'}")
        known = [(h, p) for (_, h), p in zip(widths, drops) if p is not None]
        if len(set(p for _, p in known)) > 1:
            print("    the drop point SHIFTS with width -> consistent with an RNN capacity limit.")
        elif known:
            print("    the drop point does not shift with width over the widths tried.")
        pb = first_below("B")
        print(f"  B (plain single channel) falls below {E_DROP} at P = "
              f"{pb if pb is not None else 'never, within the valid P range'}"
              f"; B by P: " + ", ".join(f"P={t.P}:{'--' if v is None else format(v, '.3f')}"
                                         for t, v in zip(valid, means['B'])))
    print()
    print(f"  total wall clock this invocation: {wall / 60:.1f} min ({wall:.0f}s)")
    print(f"  raw records: {path}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def time_arm(task, a, arch):
    ts = []
    for n in TIME_ITERS:
        t0 = time.time()
        run_ml(task, arch, a["gate"], a["n_ch"], 0, n, gate_to_readout=a["g2r"],
               h_gate=a["h_gate"], mult=a["mult"])
        ts.append(time.time() - t0)
    per = max((ts[1] - ts[0]) / (TIME_ITERS[1] - TIME_ITERS[0]), 1e-6)
    return per, max(ts[0] - TIME_ITERS[0] * per, 0.0)


def main():
    ap = argparse.ArgumentParser(description="Multilayer BDH binding and channel test.")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--stage", choices=("gate0", "all"), default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--results", default=RESULTS_FILE)
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.threads))
    seeds = list(range(args.seeds))

    print("=" * 100)
    print("Multilayer BDH binding: can it bind, and do channels then beat single-channel?")
    print(f"  built on bdh.py (n_head=1, dropout=0, n_embd={D}, N={MULT * D})   EPS={EPS}")
    print(f"  Gate 0: layers {G0_LAYERS} x {G0_POS}, seeds {G0_SEEDS}, ladder {LADDER}")
    print(f"  configs: S={S_STREAMS} n_vals={N_VALS} n_q={N_Q} P={P_LIST}  seeds={seeds}  "
          f"threads={args.threads}  decay(for 'decay')={OPERATING_DECAY}")
    print("=" * 100)
    print()
    verify()
    store = load_results(args.results)
    store["meta"] = dict(eps=EPS, seeds=seeds, p_list=P_LIST, ladder=list(LADDER),
                         started=time.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()

    chosen = gate0(store, args.results, args.force, t0)
    if chosen is None:
        print("=" * 100)
        print("VERDICT")
        print("=" * 100)
        print(f"  PRE-REGISTERED, EPS = {EPS}. Gate 0 failed, so:")
        print("  *** UNTESTED: multilayer BDH at this scale does not bind within budget. ***")
        print("  STOPPED — the channel comparison is not run on an architecture that cannot bind.")
        return
    if args.stage == "gate0":
        print("--stage gate0: stopping after Gate 0 as requested.")
        return
    arch = dict(n_layer=chosen["n_layer"], positional=chosen["positional"])
    it = chosen["iters"]
    tasks = [BindTask(P) for P in P_LIST]

    print("=" * 100)
    print(f"RESOURCES — architecture n_layer={arch['n_layer']}, {arch['positional']}; "
          f"state = n_layer * n_ch * N * D")
    print("=" * 100)
    info = {}
    for t in tasks:
        info[t.key] = {}
        print(f"  {t.label}  (T={t.T}, {t.info_bits:.0f} bits)")
        for a in arms_for(t):
            m = build(t, a, arch)
            info[t.key][a["key"]] = (n_params(m), n_state(m))
            print(f"    {a['label']:<36} n_ch={a['n_ch']} N={m.n_feat:<4} "
                  f"h_gate={m.h_gate if a['gate'] == 'recurrent' else '-':<4} "
                  f"params={info[t.key][a['key']][0]:>6}  state={info[t.key][a['key']][1]:>6}")
        sa, sm = info[t.key]["A"][1], info[t.key]["E_mem"][1]
        print(f"    E_mem memory vs A: {sm} vs {sa} -> residual {(sm - sa) / sa:+.4%}")
    print()

    print("=" * 100)
    print(f"TIMING — every arm on every config, per-iteration cost from {TIME_ITERS} steps")
    print("=" * 100)
    timing = {}
    for t in tasks:
        timing[t.key] = {}
        row = []
        for a in arms_for(t):
            timing[t.key][a["key"]] = time_arm(t, a, arch)
            row.append(f"{a['key']}={timing[t.key][a['key']][0] * 1000:.0f}ms")
        print(f"  {t.key:>4} (T={t.T:>3}): " + "  ".join(row))
    est = lambda cfg, kk: timing[cfg][kk][1] + it * timing[cfg][kk][0]
    total = sum(est(t.key, a["key"]) for t in tasks for a in arms_for(t) for _ in seeds)
    print(f"  projected wall clock, every arm x config x seed at ITERS={it}: {total / 3600:.2f} h")
    print()

    print("=" * 100)
    print(f"GATE 1 — per config, at ITERS={it}")
    print("=" * 100)
    g1 = {}
    for t in tasks:
        arms = {a["key"]: a for a in arms_for(t)}
        mse, am = gate_fit(t, arch)
        fit_ok = am >= FIT_PASS
        print(f"  {t.label}")
        print(f"    recurrent gate fit to perfect routing: MSE={mse:.4f} argmax={am:.4f} "
              f"({'PASS' if fit_ok else 'FAIL'} >= {FIT_PASS})")
        ceil_ok, cvals = True, []
        for s in seeds:
            r = do_run(store, args.results, t, arms["ceiling"], arch, it, s, args.force, t0,
                       tag="  [gate 1]")
            a = r["acc"] if r.get("ok") else None
            cvals.append(a)
            if a is None or a < CEIL_PASS:
                ceil_ok = False
                break
        for s in seeds:
            do_run(store, args.results, t, arms["floor"], arch, it, s, args.force, t0,
                   tag="  [gate 1 floor]")
        print(f"    ceiling per seed: {['--' if a is None else round(a, 4) for a in cvals]} "
              f"-> {'PASS on every seed' if ceil_ok else 'FAIL (stopped at first failing seed)'}")
        valid = fit_ok and ceil_ok
        reason = ("" if valid else
                  (f"ceiling < {CEIL_PASS} on a seed" if not ceil_ok else
                   f"recurrent gate cannot fit perfect routing (argmax {am:.4f})"))
        g1[t.key] = dict(valid=valid, reason=reason, fit_argmax=am, fit_mse=mse)
        store["gate1"][t.key] = g1[t.key]
        save_results(args.results, store)
        print(f"    -> {'VALID' if valid else 'EXCLUDED: ' + reason}")
        print()

    valid = [t for t in tasks if g1[t.key]["valid"]]
    if not valid:
        report(store, tasks, seeds, arch, it, g1, info, time.time() - t0, args.results)
        return
    proj = sum(est(t.key, a["key"]) for t in valid for a in arms_for(t) for s in seeds
               if get_run(store, t.key, a["key"], arch, it, s) is None or args.force)
    split = proj / 3600 > WALL_LIMIT_H and len(seeds) > 3
    print("=" * 100)
    print(f"PROJECTED WALL CLOCK for the remaining arm sweep on valid configs: "
          f"{proj / 3600:.2f} h (limit {WALL_LIMIT_H} h)")
    print(f"  -> {'SPLIT: seeds 0-2 first, interim report, then the rest' if split else 'single pass'}")
    print("=" * 100)
    print()
    phases = [seeds[:3], seeds[3:]] if split else [seeds]
    for pi, ph in enumerate(phases):
        print(f"-- sweep phase {pi + 1}/{len(phases)}: seeds {ph} --")
        for t in valid:
            for a in arms_for(t):
                for s in ph:
                    do_run(store, args.results, t, a, arch, it, s, args.force, t0)
        print()
        done = [s for p in phases[:pi + 1] for s in p]
        note = (f" — PARTIAL: seeds {done} of {seeds}" if pi < len(phases) - 1 else "")
        report(store, tasks, done if note else seeds, arch, it, g1, info,
               time.time() - t0, args.results, partial_note=note)


if __name__ == "__main__":
    main()
