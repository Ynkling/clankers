#!/usr/bin/env python
"""
explore_kwta.py — EXPLORATORY, not a result. Screen S2: k-winners-take-all on BDH's sparse code
(the fly olfactory circuit's random expansion + kWTA, "FlyHash").

IDEA. Dasgupta, Stevens & Navlakha (Science 2017): a random expansion followed by keeping only the
top ~5% of units gives near-orthogonal codes for different inputs. Ahmad & Hawkins (2019, "How can
we be so dense?") use the same k-WTA to cut interference between stored patterns.

WHY HERE. BDH already expands D=32 to N=256 through a random encoder and a ReLU, which leaves about
half the units active: two unrelated tokens' codes overlap with cosine ~1/pi ~ 0.32 at init. The
learned gate's wrong splits (KEY, POSITION) both reduce interference BETWEEN KEYS; the stream split
reduces interference between the two streams of one key. If key codes start near-orthogonal, the
between-key interference the wrong splits would relieve is small from step 1, and the stream
conflict is the main thing a gate can fix. Prediction, fixed before any run: fewer KEY/POSITION
failures, DISCOVERED above arm A's on the same seeds, provided the ceiling still binds.

THE CHANGE. Arm A, and at every layer x_sparse = kWTA(relu(x @ encoder), K_TOP) — the top
K_TOP = 16 of N = 256 units (6.25%) per position are kept, the rest set to 0 (gradient flows through
the kept units). y_sparse is unchanged. Arms:
  KWTA16        arm A + kWTA                   seeds 160-179
  KWTA16_ceil   perfect gate + kWTA (validity)  seeds 160-162: kWTA must still let the model bind
Every parameter is drawn exactly as arm A's (the ceiling's as test_channel_binding's ceiling).

CHECKS: with K_TOP = N the kWTA loop's logits equal bdh.BDH.forward's exactly; parameters equal
arm A's; with K_TOP = 16 exactly 16 units (or fewer, when fewer are positive) are non-zero.

Run:  python explore_kwta.py            (or via explore_batch1.py)
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F

import explore_common as ec
from test_multilayer_binding import MultiBDH, BDH, build
from test_instrument_v2 import Instrument
from test_channel_binding import ARCH, arms as cb_arms

NAME = "kwta"
IDEA = "keep only the top 16 of 256 units of x_sparse at every layer (near-orthogonal key codes)"
SOURCE = "FlyHash (Dasgupta et al., Science 2017); k-WTA (Ahmad & Hawkins 2019)"
CHANGE = "arm A with x_sparse = topk-mask(relu(x @ encoder), 16) at every layer; all else arm A"
K_TOP = 16

_CEIL = {a["key"]: a for a in cb_arms(ec.TASK)}["ceiling"]
ARMS = {
    "KWTA16_ceil": dict(_CEIL, key="KWTA16_ceil", lr=ec.SUB_LR, iters=ec.MAX_ITERS,
                        seeds=(160, 161, 162), label="perfect gate + kWTA (validity)", prio=2),
    "KWTA16": dict(ec.ARM_A, key="KWTA16", lr=ec.SUB_LR, iters=ec.MAX_ITERS, seeds=ec.SEEDS,
                   label="arm A + kWTA 16/256 at every layer", prio=1),
}


def kwta(x, k):
    idx = x.topk(k, dim=-1).indices
    return x * torch.zeros_like(x).scatter_(-1, idx, 1.0)


class KWTABDH(MultiBDH):
    def __init__(self, *args, k_top=None, **kw):
        super().__init__(*args, **kw)
        self.k_top = k_top

    def forward(self, tokens, tau=None, track_sat=False):
        if self.k_top is None:
            return super().forward(tokens, tau, track_sat)
        assert self.conv is None and not self.gate_to_readout and self.gate_noise == 0.0
        v = self.embed(tokens)
        vg = F.layer_norm(v, (v.shape[-1],)) if self.gate_ln else v
        gr, gw = Instrument.gates(self, tokens, vg, tau)
        self.attn.G = (None if self.gate_kind == "none"
                       else torch.einsum("btk,bsk->bts", gr, gw).unsqueeze(1))
        return self.forward_kwta(tokens), None, gr, gw

    def forward_kwta(self, idx):
        """bdh.BDH.forward's layer loop, line for line, with kWTA on x_sparse."""
        C = self.config
        B, T = idx.size()
        D_ = C.n_embd
        nh = C.n_head
        N = D_ * C.mlp_internal_dim_multiplier // nh
        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)
        for level in range(C.n_layer):
            x_latent = x @ self.encoder
            x_sparse = kwta(F.relu(x_latent), self.k_top)
            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            y = self.ln(yMLP)
            x = self.ln(x + y)
        return x.view(B, T, D_) @ self.lm_head


def make_model(a, seed, k_top=K_TOP):
    task = ec.TASK
    torch.manual_seed(seed)
    return KWTABDH(task.vocab, ARCH["n_layer"], a["gate"], a["n_ch"], ARCH["positional"],
                   gate_to_readout=a["g2r"], h_gate=a["h_gate"], mult=a["mult"],
                   ctx_tokens=task.ctx_tokens, **a.get("model_kw", {}), k_top=k_top)


def builder(a, seed):
    return lambda: make_model(a, seed)


def run_job(arm, seed):
    a = ARMS[arm]
    return ec.run_recipe(a, seed, builder=builder, lr=a["lr"], iters=a["iters"])


def check():
    ok = True
    seed = 160
    ref = build(ec.TASK, ec.ARM_A, ARCH, seed)
    full = make_model(ec.ARM_A, seed, k_top=ref.n_feat)
    on = make_model(ec.ARM_A, seed)
    pr = dict(ref.named_parameters())
    same = all(torch.equal(p, pr[n]) for n, p in on.named_parameters())
    g = torch.Generator().manual_seed(5)
    x, _ = ec.TASK.make_batch(4, g)
    x = x[:, :-1]
    with torch.no_grad():
        eq = torch.equal(ref(x)[0], full(x)[0])
        z = F.relu(ref.ln(ref.embed(x).unsqueeze(1)) @ ref.encoder)
        nz = (kwta(z, K_TOP) != 0).sum(-1)
        cnt = bool((nz == torch.minimum((z > 0).sum(-1), torch.tensor(K_TOP))).all())
    for name, v in (("K_TOP=N equals arm A (logits through bdh.BDH.forward)", eq),
                    ("parameters equal arm A's", same),
                    (f"kWTA keeps min(#positive, {K_TOP}) units", cnt)):
        print(f"  CHECK {NAME}: {name}: {'ok' if v else 'FAIL'}")
        ok &= bool(v)
    return ok


def report(store):
    ec.print_screen_header(sys.modules[__name__])
    ec.per_seed_table(sys.modules[__name__], store)
    ceil = [store["runs"].get(f"KWTA16_ceil|{s}") for s in ARMS["KWTA16_ceil"]["seeds"]]
    nb = sum(ec.bound(r) for r in ceil)
    print(f"  validity: KWTA16_ceil bound {nb}/{len(ceil)} "
          f"(transitions {[r['transition'] if r and r.get('ok') else None for r in ceil]})")
    s = ec.SEEDS
    print("  paired with X's recorded arm A (same seeds, initial parameters and batches):")
    d = ec.paired_vs_recorded(store, "KWTA16", s, ec.discovered, label="DISCOVERED")
    ec.paired_vs_recorded(store, "KWTA16", s, ec.bound, label="BOUND")
    print(f"    failures KWTA16 {ec.fail_counts(store, 'KWTA16', s)}   "
          f"X arm A {ec.recorded_fail_counts('A', s)}")
    d["valid"] = nb >= 2
    return d


if __name__ == "__main__":
    pv = ec.print_banner(NAME)
    assert check()
    ec.print_screen_header(sys.modules[__name__])
    ec.run_jobs([sys.modules[__name__]], pv=pv)
    report(ec.load_store(NAME))
