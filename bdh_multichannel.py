"""Multi-channel Hebbian-recurrent BDH.

k independent Hebbian states S: (k, D, D).
Gate:  g_t = softmax(h_t @ W_gate)  in R^k — routing signal from recurrent context.
Output logits come from the gated Hebbian READ (r_t), not directly from h_t.
This forces the model to rely on Hebbian retrieval; h_t provides only the gate.

Bigram write (outer(value, key) for content-addressable memory):
    S_c += g_t[c] * outer(W_v @ embed(x_t),  W_k @ embed(x_{t-1}))

Gated read:
    r_t = sum_c g_t[c] * S_c_{t-1} @ (W_q @ embed(x_{t-1}))
    logits_t = lm_head(r_t)

Effective attention score between write position j and read position i:
    (x_i · x_j) * (g_i · g_j)   — content similarity × gate similarity.

gate_override: (B, T, k) tensor supplied by caller for conditions 3 (perfect) and 4 (uniform).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BDHMultiChannelConfig:
    def __init__(self, vocab_size, d_model=32, n_channels=2, dropout=0.0):
        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.n_channels = n_channels
        self.dropout    = dropout


class BDHMultiChannel(nn.Module):
    def __init__(self, config: BDHMultiChannelConfig):
        super().__init__()
        D = config.d_model
        k = config.n_channels
        self.config  = config
        self.embed   = nn.Embedding(config.vocab_size, D)
        self.W_in    = nn.Linear(D, D, bias=False)
        self.W_h     = nn.Linear(D, D, bias=False)
        self.W_k     = nn.Linear(D, D, bias=False)
        self.W_v     = nn.Linear(D, D, bias=False)
        self.W_q     = nn.Linear(D, D, bias=False)
        self.W_gate  = nn.Linear(D, k,  bias=False)
        self.drop    = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(D, config.vocab_size, bias=False)

    def forward(self, idx, targets=None, loss_mask=None, gate_override=None):
        B, T = idx.shape
        D    = self.config.d_model
        k    = self.config.n_channels

        h        = torch.zeros(B, D,    device=idx.device)
        S        = torch.zeros(B, k, D, D, device=idx.device)
        emb_prev = torch.zeros(B, D,    device=idx.device)

        logits_list = []
        gates_list  = []

        for t in range(T):
            x_t = self.drop(self.embed(idx[:, t]))
            h   = torch.tanh(self.W_in(x_t) + self.W_h(h))

            # Gate: from caller or learned via h
            if gate_override is not None:
                g = gate_override[:, t, :]                  # (B, k)
            else:
                g = torch.softmax(self.W_gate(h), dim=-1)  # (B, k)
            gates_list.append(g)

            # Gated read BEFORE write
            q_t = self.W_q(emb_prev)                           # (B, D)
            Sq  = torch.einsum('bkij,bj->bki', S, q_t)         # (B, k, D)
            r_t = torch.einsum('bk,bki->bi', g, Sq)            # (B, D)

            # Logits from Hebbian read only
            logits_list.append(self.lm_head(r_t))

            # Gated bigram write: outer(value, key)
            k_t   = self.W_k(emb_prev)
            v_t   = self.W_v(x_t)
            write = torch.einsum('bi,bj->bij', v_t, k_t)       # outer(val, key) (B,D,D)
            S     = S + torch.einsum('bk,bij->bkij', g, write)

            emb_prev = x_t

        logits = torch.stack(logits_list, dim=1)  # (B, T, V)
        gates  = torch.stack(gates_list,  dim=1)  # (B, T, k)

        loss = None
        if targets is not None:
            if loss_mask is not None:
                flat_l = logits.reshape(-1, logits.size(-1))
                flat_t = targets.reshape(-1)
                mask   = loss_mask.reshape(-1).bool()
                loss   = F.cross_entropy(flat_l[mask], flat_t[mask])
            else:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       targets.reshape(-1))
        return logits, loss, gates
