#!/usr/bin/env python
"""
explore_local_gate.py — EXPLORATORY, not a result. Screen S1: compute the channel gate from a
short causal convolution of the token embeddings, with no recurrence (Mamba / H3 / RWKV style),
instead of arm A's recurrent gate.

IDEA. In Mamba (Gu & Dao 2023) the selective gates (Delta, B, C) are computed from the output of
a short causal depthwise convolution (d_conv=4) of the input, not from a long recurrence. H3's
shift-SSM and RWKV's token shift do the same. A gate built that way sees only the last few tokens.

WHY HERE. Arm A's failures are mostly POSITION splits: the recurrent gate drifts along the
sequence and sends early key groups to one channel and later ones to the other (test_router_layout:
6 of 8 grouped failures; test_short_conv's A on seeds 160-199: 18 of 28; test_p_scaling's A_conv8:
15 of 26). A gate that sees only tokens t-2..t cannot express a position split (except at the
first two positions), but it can express the stream routing: every triple is [CTX, KEY, VAL], so a
window of width 3 at a KEY or VAL position, and at the query's KEY position, contains that triple's
CTX token. It can still express a KEY split or a role split. Prediction, fixed before any run:
POSITION failures vanish; DISCOVERED rises above arm A's on the same seeds if they are not replaced
by KEY failures.

THE CHANGE (one arm, LOCAL3). MultiBDH with gate='recurrent', k=2, built exactly as arm A (same
seed, so every BDH parameter and W_in, W_h, W_g are bit-identical to arm A's), plus a depthwise
causal convolution of width 3 over the raw embeddings, drawn from its own generator (seed +
GCONV_OFFSET), init as nn.Conv1d's default for a depthwise kernel (uniform +-1/sqrt(width)), no
bias. The gate is
    u_t = sum_{j<3} w_j * v_{t-j}  (zero-padded),   h_t = tanh(W_in u_t),   g_t = softmax(W_g h_t)
i.e. arm A's gate with W_h dropped and u_t in place of v_t. W_h exists but is unused (no gradient,
Adam skips it). Read and write gates are the same, as in arm A. Everything else is arm A's.

CHECKS (printed before any run): with local=False the subclass equals arm A exactly (parameters
and logits); with local=True the shared parameters equal arm A's; the gate at t does not change
when token t-3 changes and does change when token t-2 changes.

Run:  python explore_local_gate.py            (or via explore_batch1.py)
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F

import explore_common as ec
from test_multilayer_binding import MultiBDH, CausalConv, BDH, D, build
from test_channel_binding import ARCH

NAME = "local_gate"
IDEA = "gate from a width-3 causal conv of the embeddings, no recurrence (cannot split by position)"
SOURCE = "Mamba (Gu & Dao 2023) conv-before-selective-gates; H3 shift-SSM; RWKV token shift"
CHANGE = ("arm A with h_t = tanh(W_in conv3(v)_t) instead of tanh(W_in v_t + W_h h_{t-1}); "
          "conv init uniform +-1/sqrt(3) from its own generator; all else arm A")
WIDTH = 3
GCONV_OFFSET = 31_337

ARMS = {
    "LOCAL3": dict(ec.ARM_A, key="LOCAL3", lr=ec.SUB_LR, iters=ec.MAX_ITERS, seeds=ec.SEEDS,
                   label="arm A, gate from a width-3 causal conv (no recurrence)", prio=1),
}


class LocalGateBDH(MultiBDH):
    def __init__(self, *args, local=True, width=WIDTH, gconv_seed=0, **kw):
        super().__init__(*args, **kw)                    # arm A's parameters, same RNG draws
        self.local = bool(local)
        if self.local:
            assert self.gate_kind == "recurrent" and not self.gate_to_readout
            assert not self.gate_ln and self.gate_noise == 0.0
            g = torch.Generator().manual_seed(int(gconv_seed))
            self.gate_conv = CausalConv(D, int(width))
            bound = 1.0 / math.sqrt(width)
            with torch.no_grad():
                self.gate_conv.conv_w.copy_((torch.rand(D, width, generator=g) * 2 - 1) * bound)

    def local_gates(self, v):
        u = self.gate_conv(v)                                         # (B, T, D), causal
        h = torch.tanh(u @ self.W_in.T)
        g = F.softmax(h @ self.W_g.T, dim=-1)
        return g, g

    def forward(self, tokens, tau=None, track_sat=False):
        if not self.local:
            return super().forward(tokens, tau, track_sat)
        gr, gw = self.local_gates(self.embed(tokens))
        self.attn.G = torch.einsum("btk,bsk->bts", gr, gw).unsqueeze(1)
        if self.conv is None:
            logits, _ = BDH.forward(self, tokens)
        else:
            logits = self.forward_conv(tokens)
        return logits, None, gr, gw


def make_model(a, seed, local=True):
    task = ec.TASK
    torch.manual_seed(seed)                              # build()'s seeding, then the same ctor
    return LocalGateBDH(task.vocab, ARCH["n_layer"], a["gate"], a["n_ch"], ARCH["positional"],
                        gate_to_readout=a["g2r"], h_gate=a["h_gate"], mult=a["mult"],
                        ctx_tokens=task.ctx_tokens, **a.get("model_kw", {}),
                        local=local, width=WIDTH, gconv_seed=seed + GCONV_OFFSET)


def builder(a, seed):
    return lambda: make_model(a, seed)


def run_job(arm, seed):
    a = ARMS[arm]
    return ec.run_recipe(a, seed, builder=builder, lr=a["lr"], iters=a["iters"])


def check():
    ok = True
    seed = 160
    ref = build(ec.TASK, ec.ARM_A, ARCH, seed)
    off = make_model(ec.ARM_A, seed, local=False)
    on = make_model(ec.ARM_A, seed, local=True)
    pr = dict(ref.named_parameters())
    same_off = all(torch.equal(p, pr[n]) for n, p in off.named_parameters())
    same_on = all(torch.equal(p, pr[n]) for n, p in on.named_parameters() if n in pr)
    g = torch.Generator().manual_seed(5)
    x, _ = ec.TASK.make_batch(4, g)
    x = x[:, :-1]
    ref.eval(); off.eval(); on.eval()
    with torch.no_grad():
        same_logits = torch.equal(ref(x)[0], off(x)[0])
        t = 12
        g0 = on(x)[2]
        x3 = x.clone(); x3[:, t - 3] = (x3[:, t - 3] + 1) % ec.TASK.vocab
        x2 = x.clone(); x2[:, t - 2] = (x2[:, t - 2] + 1) % ec.TASK.vocab
        blind = torch.equal(on(x3)[2][:, t], g0[:, t])
        sees = not torch.equal(on(x2)[2][:, t], g0[:, t])
    for name, v in (("local=False equals arm A (parameters)", same_off),
                    ("local=False equals arm A (logits)", same_logits),
                    ("local=True shares arm A's parameters", same_on),
                    ("gate at t ignores token t-3", blind),
                    ("gate at t sees token t-2", sees)):
        print(f"  CHECK {NAME}: {name}: {'ok' if v else 'FAIL'}")
        ok &= bool(v)
    return ok


def report(store):
    ec.print_screen_header(sys.modules[__name__])
    ec.per_seed_table(sys.modules[__name__], store)
    s = ec.SEEDS
    print("  paired with X's recorded arm A (same seeds, initial parameters and batches):")
    d = ec.paired_vs_recorded(store, "LOCAL3", s, ec.discovered, label="DISCOVERED")
    ec.paired_vs_recorded(store, "LOCAL3", s, ec.bound, label="BOUND")
    print(f"    failures LOCAL3 {ec.fail_counts(store, 'LOCAL3', s)}   "
          f"X arm A {ec.recorded_fail_counts('A', s)}")
    return d


if __name__ == "__main__":
    pv = ec.print_banner(NAME)
    assert check()
    ec.print_screen_header(sys.modules[__name__])
    ec.run_jobs([sys.modules[__name__]], pv=pv)
    report(ec.load_store(NAME))
