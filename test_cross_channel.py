"""
test_cross_channel.py — Bridge task: does per-channel Hebbian isolation hurt cross-context recall?

Task (byte-level, self-contained, no downloads):
  Phase A  presents P shuffled key→value associations using "phase-A key" tokens (k^A_*).
  Phase B  queries one association using a DIFFERENT token set (k^B_*) for the same keys.
  The correct answer (a value token) requires reading phase-A memory during phase-B.

Because k^A and k^B are different vocab items, the MC gate CAN learn to route them to
different channels — which would isolate the memories written in phase A from the reads
in phase B, causing failure. The question is whether it does.

Sanity (zero-memory lower bound):
  A model with NO access to phase-A memory must guess uniformly among P values.
  CE loss >= log(P) ≈ 1.386 for P=4.  Any model scoring below this has cross-context recall.

Conditions:
  1. single-channel BDH      — baseline, no isolation possible
  2. MC k=2, learned gate    — gate CAN create isolation; does it?
  3. MC k=2, uniform gate    — MC parameterization but g=[0.5,0.5] always; no isolation
     (After LayerNorm the uniform-gate MC is functionally equivalent to single-channel,
      so conditions 1 and 3 serve as paired controls: any gap with condition 2 is from gating.)
"""

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from bdh_recurrent import bdh_layer_parallel, bdh_layer_mc_parallel

# ── Hyperparameters ────────────────────────────────────────────────────────────
BLOCK  = 12      # sequence length: 2P + 4  for P=4
BATCH  = 32
ITERS  = 1200
k      = 2       # channels for MC models
SEEDS  = [0, 1, 2]

P  = 4           # number of key-value pairs (small: chance = log(4)≈1.386)
D  = 24          # embedding dim
N  = 48          # sparse feature dim
LR = 4e-3

# ── Vocabulary (size = 3P+2 = 14) ────────────────────────────────────────────
#   0 .. P-1      : phase-A keys  k^A_i
#   P .. 2P-1     : values        v_i
#   2P .. 3P-1    : phase-B keys  k^B_i  (same semantics, different tokens)
#   3P            : WRITE_MARK    (phase-A context marker)
#   3P+1          : QUERY_MARK    (phase-B context marker)
V          = 3 * P + 2
WRITE_MARK = 3 * P
QUERY_MARK = 3 * P + 1

# Sequence layout (length = BLOCK = 12):
#   [WRITE_MARK, k^A_{π0}, v_{π0}, k^A_{π1}, v_{π1}, ..., k^A_{π3}, v_{π3},
#    QUERY_MARK, k^B_q,  v_q  <- TARGET]
# inp = seq[0..10], tgt = seq[1..11]; loss at position 10 predicts v_q.

assert BLOCK == 2 * P + 4, "BLOCK must equal 2P+4"
CHANCE_LOSS = math.log(P)   # ≈ 1.386 — CE loss at pure chance over P values


# ── Data ──────────────────────────────────────────────────────────────────────
def make_batch(B: int, rng: torch.Generator) -> torch.Tensor:
    """Return (B, BLOCK) token sequences."""
    seqs = []
    for _ in range(B):
        perm = torch.randperm(P, generator=rng).tolist()
        q    = torch.randint(P, (1,), generator=rng).item()
        seq  = [WRITE_MARK]
        for i in range(P):
            seq.append(perm[i])         # k^A_{perm[i]}  in [0, P)
            seq.append(P + perm[i])     # v_{perm[i]}    in [P, 2P)
        seq.append(QUERY_MARK)
        seq.append(2 * P + q)           # k^B_q          in [2P, 3P)
        seq.append(P + q)               # v_q  ← TARGET  in [P, 2P)
        assert len(seq) == BLOCK
        seqs.append(seq)
    return torch.tensor(seqs, dtype=torch.long)  # (B, BLOCK)


# ── BDH wrappers (import the verified functions, wrap for batch + grad) ────────
def _bdh_single_forward(embed, Dx, Dy, E, head, tokens):
    """tokens: (B, T) → logits: (B, T, V)  [single-channel]"""
    B, T = tokens.shape
    v = embed(tokens)                        # (B, T, D)
    # process each example through the recurrent form (parallel is equivalent)
    out = []
    for b in range(B):
        v_next, *_ = bdh_layer_parallel(v[b], Dx, Dy, E)
        out.append(v_next)
    v_next = torch.stack(out)               # (B, T, D)
    return head(v_next)                     # (B, T, V)


def _bdh_mc_forward(embed, Dx, Dy, E, W_g, head, tokens):
    """tokens: (B, T) → logits: (B, T, V)  [multi-channel]"""
    B, T = tokens.shape
    v = embed(tokens)                        # (B, T, D)
    out = []
    for b in range(B):
        v_next, *_ = bdh_layer_mc_parallel(v[b], Dx, Dy, E, W_g)
        out.append(v_next)
    v_next = torch.stack(out)               # (B, T, D)
    return head(v_next)                     # (B, T, V)


class BDHSingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.Dx    = nn.Parameter(torch.randn(D, N) * 0.1)
        self.Dy    = nn.Parameter(torch.randn(D, N) * 0.1)
        self.E     = nn.Parameter(torch.randn(N, D) * 0.1)
        self.head  = nn.Linear(D, V, bias=False)

    def forward(self, tokens):
        return _bdh_single_forward(self.embed, self.Dx, self.Dy, self.E,
                                   self.head, tokens)


class BDHMCMC(nn.Module):
    def __init__(self, uniform_gate: bool = False):
        super().__init__()
        self.embed        = nn.Embedding(V, D)
        self.Dx           = nn.Parameter(torch.randn(D, N) * 0.1)
        self.Dy           = nn.Parameter(torch.randn(D, N) * 0.1)
        self.E            = nn.Parameter(torch.randn(N, D) * 0.1)
        self.head         = nn.Linear(D, V, bias=False)
        self.uniform_gate = uniform_gate
        if uniform_gate:
            # fixed buffer: gate = [1/k, 1/k] for every token, no gradient
            self.register_buffer('W_g', torch.zeros(D, k))
        else:
            self.W_g = nn.Parameter(torch.randn(D, k) * 0.1)

    def forward(self, tokens):
        return _bdh_mc_forward(self.embed, self.Dx, self.Dy, self.E,
                               self.W_g, self.head, tokens)


# ── Training + evaluation ─────────────────────────────────────────────────────
def run(make_model, seed: int):
    torch.manual_seed(seed)
    model = make_model()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    rng   = torch.Generator()
    rng.manual_seed(seed + 10_000)

    for _ in range(ITERS):
        tokens = make_batch(BATCH, rng)    # (B, 12)
        inp    = tokens[:, :-1]            # (B, 11)  input tokens
        tgt    = tokens[:, 1:]             # (B, 11)  next-token targets

        logits = model(inp)                # (B, 11, V)

        # Train ONLY on the query position (pos 10 in the 11-length output):
        #   inp[:, 10] = k^B_q  →  target = v_q = tgt[:, 10]
        # This gives the model a direct gradient signal for cross-context recall
        # and avoids the majority of positions that are unresolvable (random permutations).
        loss = F.cross_entropy(logits[:, -1, :], tgt[:, -1])
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Evaluate on a fresh held-out batch (same distribution, no overfitting possible
    # since data is generated i.i.d. — but we use a separate rng state for clarity)
    model.eval()
    with torch.no_grad():
        tokens = make_batch(512, rng)
        inp    = tokens[:, :-1]
        tgt    = tokens[:, 1:]
        logits = model(inp)                # (B, 11, V)

        q_logits = logits[:, -1, :]       # (B, V)  — query position only
        q_tgt    = tgt[:, -1]             # (B,)

        eval_loss = F.cross_entropy(q_logits, q_tgt).item()
        preds     = q_logits.argmax(-1)
        accuracy  = (preds == q_tgt).float().mean().item()

    return eval_loss, accuracy


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Bridge task  P={P} pairs  D={D}  N={N}  k={k}  "
          f"BATCH={BATCH}  ITERS={ITERS}  SEEDS={SEEDS}")
    print(f"Chance loss (zero phase-A memory) = log({P}) = {CHANCE_LOSS:.4f} nats")
    print(f"Vocab size = {V}  |  Sequence length = {BLOCK}")
    print()

    # Sanity: confirm chance level — a "model" that always predicts uniform over
    # the value tokens [P..2P) achieves exactly CHANCE_LOSS on q_tgt.
    rng_sanity = torch.Generator(); rng_sanity.manual_seed(99)
    toks = make_batch(1024, rng_sanity)
    q_tgt = toks[:, -1]   # v_q in [P, 2P)
    uniform_logits = torch.zeros(len(q_tgt), V)
    uniform_logits[:, P:2*P] = 0.0   # equal logits for value tokens, -inf elsewhere (close enough with 0 elsewhere)
    sanity_loss = F.cross_entropy(uniform_logits, q_tgt).item()
    # exact chance among all V tokens is log(V); among P values is log(P)
    print(f"Sanity — uniform-over-values CE loss = {sanity_loss:.4f}  "
          f"(expected ≈ {math.log(V):.4f} if uniform over V, "
          f"≈ {CHANCE_LOSS:.4f} if uniform over P values)\n")

    conditions = [
        ("single-channel",      lambda: BDHSingle()),
        ("MC k=2 learned gate", lambda: BDHMCMC(uniform_gate=False)),
        ("MC k=2 uniform gate", lambda: BDHMCMC(uniform_gate=True)),
    ]

    results = {}
    for name, make_model in conditions:
        losses, accs = [], []
        for seed in SEEDS:
            loss, acc = run(make_model, seed)
            losses.append(loss)
            accs.append(acc)
        mean_l = sum(losses) / len(losses)
        std_l  = (sum((l - mean_l) ** 2 for l in losses) / len(losses)) ** 0.5
        mean_a = sum(accs) / len(accs)
        std_a  = (sum((a - mean_a) ** 2 for a in accs) / len(accs)) ** 0.5
        results[name] = dict(losses=losses, accs=accs,
                             mean_l=mean_l, std_l=std_l,
                             mean_a=mean_a, std_a=std_a)
        print(f"  {name:<24}  loss {mean_l:.4f} ± {std_l:.4f}   "
              f"acc {mean_a:.4f} ± {std_a:.4f}   "
              f"(seeds: {[round(l,3) for l in losses]})")

    print()
    print(f"Chance baseline:             loss {CHANCE_LOSS:.4f}             acc {1/P:.4f}")
    print()

    # ── Interpret ────────────────────────────────────────────────────────────
    mc_l   = results["MC k=2 learned gate"]["mean_l"]
    base_l = results["single-channel"]["mean_l"]
    unif_l = results["MC k=2 uniform gate"]["mean_l"]
    gap    = mc_l - min(base_l, unif_l)
    thresh = 0.05   # nats; anything above this is a meaningful gap

    print("── Finding ─────────────────────────────────────────────────────────")
    if gap > thresh:
        print(f"ISOLATION HAS A COST: MC learned gate is worse than both controls by "
              f"{gap:.4f} nats (>{thresh} threshold).")
        print("The g@g.T term is actively hurting cross-context recall on this task.")
    else:
        print(f"ISOLATION IS FREE (or beneficial): MC learned gate matches/beats controls "
              f"(gap = {gap:+.4f} nats, within ±{thresh} threshold).")
        print("The gate learned NOT to isolate phases when cross-context recall is required.")
