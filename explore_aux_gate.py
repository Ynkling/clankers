#!/usr/bin/env python
"""
explore_aux_gate.py — EXPLORATORY, not a result. Screen S3: give the gate's recurrent state its
own self-supervised signal — predict the next token — as an auxiliary loss (UNREAL / CPC style).

IDEA. Jaderberg et al. (ICLR 2017, UNREAL) and van den Oord et al. (2018, CPC): auxiliary
prediction losses on a recurrent state shape it to carry the structure of the input stream, which
the main objective then reads.

WHY HERE. Arm A's gate reads h_t = tanh(W_in v_t + W_h h_{t-1}); from the task loss alone its
gradient is ~1e-5 at step 1 and it commits within 1200 steps, often to a POSITION or KEY split. The
token stream itself says which stream a triple belongs to: in the grouped layout the second CTX of
a key group is always the other stream, and its KEY repeats the first triple's. To predict the next
token, h must therefore carry the current triple's stream (at VAL positions) and the current key
(at CTX positions). If the stream becomes a leading direction of h early, W_g's first moves should
lean toward it, the way the labelled 5% nudge led to routing on 56/60 arm-A seeds. Counter-risk,
from test_readout_path: a non-routing gradient into h (the readout's) moved the gate onto wrong
splits (A_ro 2/30). Direction of the effect is the open question.

THE CHANGE. Arm A, plus in training forwards only: h_aux = the gate's own recurrence (W_in, W_h)
run on the DETACHED embeddings; aux = LAMBDA * CE(h_aux[:, :-1] @ W_aux.T, tokens[:, 1:]) over every
input position; W_aux (vocab x 32) is new, drawn from its own generator (seed + AUX_OFFSET, randn
* 0.1). The aux gradient reaches W_in, W_h and W_aux only (not the embeddings, not the BDH stack,
not W_g); it enters through loss.backward() by a gradient-injection node on the logits, so
onset_run is unchanged. LAMBDA = 1.0. No stream labels are used. Evaluation forwards are arm A's.

CHECKS: parameters equal arm A's (plus W_aux); eval-mode logits equal arm A's exactly; training-mode
logits equal arm A's exactly; the aux gradient reaches W_in, W_h, W_aux and nothing else (embed,
encoder, W_g get none from it).

Run:  python explore_aux_gate.py            (or via explore_batch1.py)
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F

import explore_common as ec
from test_multilayer_binding import MultiBDH, build
from test_channel_binding import ARCH

NAME = "aux_gate"
IDEA = "self-supervised next-token prediction from the gate's recurrent state (aux loss, weight 1)"
SOURCE = "UNREAL auxiliary tasks (Jaderberg et al. 2017); CPC (van den Oord et al. 2018)"
CHANGE = ("arm A + LAMBDA*CE(next token | h_aux), h_aux = gate recurrence on detached embeddings; "
          "gradient to W_in, W_h, W_aux only")
LAMBDA = 1.0
AUX_OFFSET = 42_424

ARMS = {
    "AUX1": dict(ec.ARM_A, key="AUX1", lr=ec.SUB_LR, iters=ec.MAX_ITERS, seeds=ec.SEEDS,
                 label="arm A + next-token aux on the gate state (lambda 1)", prio=1),
}


class _Inject(torch.autograd.Function):
    """Identity on the logits; in backward, also sends gradient 1 into the aux loss."""

    @staticmethod
    def forward(ctx, logits, aux):
        return logits.clone()

    @staticmethod
    def backward(ctx, g):
        return g, torch.ones((), dtype=g.dtype)


class AuxGateBDH(MultiBDH):
    def __init__(self, *args, aux_lambda=0.0, aux_seed=0, **kw):
        super().__init__(*args, **kw)                   # arm A's parameters, same RNG draws
        self.aux_lambda = float(aux_lambda)
        assert self.gate_kind == "recurrent" and not self.gate_to_readout
        g = torch.Generator().manual_seed(int(aux_seed))
        self.W_aux = nn.Parameter(torch.randn(self.embed.num_embeddings, self.h_gate,
                                              generator=g) * 0.1)
        self.aux_last = None

    def aux_loss(self, tokens):
        v = self.embed(tokens).detach()
        B, T, _ = v.shape
        h = torch.zeros(B, self.h_gate, dtype=v.dtype)
        hs = []
        for t in range(T):                               # Instrument.gates' recurrence
            h = torch.tanh(v[:, t] @ self.W_in.T + h @ self.W_h.T)
            hs.append(h)
        hs = torch.stack(hs, dim=1)
        lg = hs[:, :-1] @ self.W_aux.T
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tokens[:, 1:].reshape(-1))

    def forward(self, tokens, tau=None, track_sat=False):
        logits, sat, gr, gw = super().forward(tokens, tau, track_sat)
        if self.training and self.aux_lambda > 0 and torch.is_grad_enabled():
            aux = self.aux_loss(tokens)
            self.aux_last = float(aux.detach())
            logits = _Inject.apply(logits, self.aux_lambda * aux)
        return logits, sat, gr, gw


def make_model(a, seed, aux_lambda=LAMBDA):
    task = ec.TASK
    torch.manual_seed(seed)
    return AuxGateBDH(task.vocab, ARCH["n_layer"], a["gate"], a["n_ch"], ARCH["positional"],
                      gate_to_readout=a["g2r"], h_gate=a["h_gate"], mult=a["mult"],
                      ctx_tokens=task.ctx_tokens, **a.get("model_kw", {}),
                      aux_lambda=aux_lambda, aux_seed=seed + AUX_OFFSET)


def builder(a, seed):
    return lambda: make_model(a, seed)


def stats_fn(model, task, probe, step):
    out = ec.tsc.conv_stats(model, task, probe, step)
    out["aux_loss_last_train"] = model.aux_last
    return out


def run_job(arm, seed):
    a = ARMS[arm]
    return ec.run_one(a, seed, a["iters"], task=ec.TASK, stats_fn=stats_fn,
                      grad_fn=ec.tsc.conv_grad_norms, lr=a["lr"], builder=builder)


def check():
    ok = True
    seed = 160
    ref = build(ec.TASK, ec.ARM_A, ARCH, seed)
    on = make_model(ec.ARM_A, seed)
    pr = dict(ref.named_parameters())
    same = all(torch.equal(p, pr[n]) for n, p in on.named_parameters() if n != "W_aux")
    g = torch.Generator().manual_seed(5)
    x, _ = ec.TASK.make_batch(4, g)
    x = x[:, :-1]
    ref.eval(); on.eval()
    with torch.no_grad():
        eq_eval = torch.equal(ref(x)[0], on(x)[0])
    ref.train(); on.train()
    eq_train = torch.equal(ref(x)[0].detach(), on(x)[0].detach())
    on.zero_grad()
    aux = on.aux_loss(x)
    aux.backward()
    got = {n for n, p in on.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    reach = got == {"W_in", "W_h", "W_aux"}
    # the injection: the task loss's gradient plus the aux gradient, through one backward
    on.zero_grad()
    lg = on(x)[0]
    lg[:, -1].sum().backward()
    inj = on.W_aux.grad is not None and on.W_aux.grad.abs().sum() > 0
    for name, v in (("parameters equal arm A's (plus W_aux)", same),
                    ("eval-mode logits equal arm A's", eq_eval),
                    ("training-mode logits equal arm A's", eq_train),
                    (f"aux gradient reaches exactly W_in, W_h, W_aux (got {sorted(got)})", reach),
                    ("aux gradient injected through the task loss's backward", inj)):
        print(f"  CHECK {NAME}: {name}: {'ok' if v else 'FAIL'}")
        ok &= bool(v)
    return ok


def report(store):
    ec.print_screen_header(sys.modules[__name__])
    ec.per_seed_table(sys.modules[__name__], store)
    s = ec.SEEDS
    print("  paired with X's recorded arm A (same seeds, initial parameters and batches):")
    d = ec.paired_vs_recorded(store, "AUX1", s, ec.discovered, label="DISCOVERED")
    ec.paired_vs_recorded(store, "AUX1", s, ec.bound, label="BOUND")
    print(f"    failures AUX1 {ec.fail_counts(store, 'AUX1', s)}   "
          f"X arm A {ec.recorded_fail_counts('A', s)}")
    aux = [r["stats"][1].get("aux_loss_last_train") for sd in s
           if (r := store["runs"].get(f"AUX1|{sd}")) and r.get("ok") and len(r["stats"]) > 1]
    if aux:
        print(f"    aux loss at step 1200 (last training batch): "
              f"{' '.join(f'{a:.2f}' for a in aux)}")
    return d


if __name__ == "__main__":
    pv = ec.print_banner(NAME)
    assert check()
    ec.print_screen_header(sys.modules[__name__])
    ec.run_jobs([sys.modules[__name__]], pv=pv)
    report(ec.load_store(NAME))
