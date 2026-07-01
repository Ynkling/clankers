"""Single-channel Hebbian-recurrent BDH.

Truly recurrent model.  Information for the OUTPUT flows only through the
Hebbian state S — the recurrent hidden state h is kept for internal context
tracking but does NOT directly contribute to logits.  This forces the model
to rely on Hebbian retrieval, not on shortcuts via h.

Bigram write (outer(value, key) convention for content-addressable memory):
    at position t,  S += outer(W_v @ embed(x_t), W_k @ embed(x_{t-1}))
    so that  S @ W_q @ embed(x_{t-1})  retrieves  W_v @ embed(x_t).

Read:
    r_t = S_{t-1} @ (W_q @ embed(x_{t-1}))
    logits_t = lm_head(r_t)

Gate (context tracking, not used for logits):
    h_t = tanh(W_in @ embed(x_t) + W_h @ h_{t-1})
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BDHRecurrentConfig:
    def __init__(self, vocab_size, d_model=32, dropout=0.0):
        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.dropout    = dropout


class BDHRecurrent(nn.Module):
    def __init__(self, config: BDHRecurrentConfig):
        super().__init__()
        D = config.d_model
        self.config  = config
        self.embed   = nn.Embedding(config.vocab_size, D)
        self.W_in    = nn.Linear(D, D, bias=False)   # input → hidden
        self.W_h     = nn.Linear(D, D, bias=False)   # hidden → hidden
        self.W_k     = nn.Linear(D, D, bias=False)   # key projection  (write address)
        self.W_v     = nn.Linear(D, D, bias=False)   # value projection (write content)
        self.W_q     = nn.Linear(D, D, bias=False)   # query projection (read address)
        self.drop    = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(D, config.vocab_size, bias=False)

    def forward(self, idx, targets=None, loss_mask=None):
        B, T = idx.shape
        D    = self.config.d_model

        h        = torch.zeros(B, D, device=idx.device)
        S        = torch.zeros(B, D, D, device=idx.device)
        emb_prev = torch.zeros(B, D, device=idx.device)

        logits_list = []
        for t in range(T):
            x_t = self.drop(self.embed(idx[:, t]))          # (B, D)
            h   = torch.tanh(self.W_in(x_t) + self.W_h(h)) # (B, D) — context only

            # Read BEFORE write (strictly causal), addressing with previous token
            q_t = self.W_q(emb_prev)                                    # (B, D)
            r_t = torch.bmm(S, q_t.unsqueeze(-1)).squeeze(-1)           # (B, D)

            # Logits from Hebbian read only — no h shortcut
            logits_list.append(self.lm_head(r_t))

            # Bigram write: outer(value, key) → S @ key = value
            k_t = self.W_k(emb_prev)
            v_t = self.W_v(x_t)
            S   = S + torch.bmm(v_t.unsqueeze(-1), k_t.unsqueeze(1))   # (B, D, D)

            emb_prev = x_t

        logits = torch.stack(logits_list, dim=1)   # (B, T, V)
        loss   = None
        if targets is not None:
            if loss_mask is not None:
                flat_l = logits.reshape(-1, logits.size(-1))
                flat_t = targets.reshape(-1)
                mask   = loss_mask.reshape(-1).bool()
                loss   = F.cross_entropy(flat_l[mask], flat_t[mask])
            else:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       targets.reshape(-1))
        return logits, loss
