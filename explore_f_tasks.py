#!/usr/bin/env python
"""
explore_f_tasks.py — EXPLORATORY, not a result. Session F's task-side helpers (branch modules only; nothing here changes
test_binding_capacity — the new layout subclasses BindTask, as explore_far_cue.HeaderTask does).

RandHeaderTask (S54(d); Session G copies this definition). The randomised header layout.
  Content: as BindTask / HeaderTask — S streams, P keys, n_vals values; for every key, S distinct values (one per
    stream); every (stream, key) pair appears exactly once in the body; n_q queries [CTX_s, KEY_k, VAL] at the end,
    drawn without replacement from the S·P pairs (as make_batch).
  Blocks: each stream's P (key, value) pairs are split into NB consecutive blocks; the block lengths (in pairs) are one
    composition of P into NB parts, each part in {LMIN..LMAX}, drawn uniformly per sequence and stream from the list of
    all such compositions. Each block is [CTX_s, KEY, VAL, KEY, VAL, ...]: exactly one CTX token, at its start. The S·NB
    blocks appear in a uniformly random order (blocks of one stream need not alternate with other streams' blocks). Within
    a stream the keys are in a uniformly random order, cut into its blocks in that order.
  Defaults (S54(d)): P = 8, NB = 2, LMIN..LMAX = 2..6. The compositions of 8 into 2 parts in 2..6 are (2,6), (3,5),
    (4,4), (5,3), (6,2): each block's length is uniform on {2,3,4,5,6}, a stream's two lengths sum to 8. P = 8 with NB = 2
    is the smallest fixed-length configuration in which every block length 2..6 occurs (a sequence's length must not
    vary: the model has no padding). L = S·(NB + 2P) + 3·n_q (S = 2, n_q = 1: 39); qpos = S·(NB + 2P) + 3j + 1.
  Why: in HeaderTask the block of each stream sits at fixed positions, so a gate that counts positions can route
    (distant_cues §3(d)); here a token's stream is the stream of the latest CTX token, at a distance of 1 to 2·LMAX
    tokens, and no position determines it.
  stream_labels (BindTask's: the latest CTX token) gives every body position its block's stream.
  CHECKS (check_rand_header): every (stream, key) once; S distinct values per key; each query answered by its own
    stream's value; exactly one CTX per block, at the block's start, and S·NB CTX tokens in the body; every stream
    has NB blocks and its block lengths are a composition of P with parts in LMIN..LMAX; over 4096 sequences every length
    LMIN..LMAX occurs, each with frequency within 0.03 of 1/5; the block order is not fixed (more than one stream order
    of the blocks occurs); stream_labels equals the latest CTX's stream at every body position; L, qpos, select.

rand_header_routing_stats: test_router_layout.routing_stats with the positions read from each sequence (the body's
  key positions, ordered by appearance; values one after; labels stream = stream_labels at the key, key = token − S,
  index = order of appearance among the S·P pairs, half = index >= S·P/2). The margin: the query's read gate against
  the write gate at the VAL position of the target pair and of its S − 1 same-key distractors, as routing_stats.
  Measured at KEY and at VAL positions (audit §3.2).

write_order_acc (S53's "does a single delta channel keep exactly the last-written stream?"): on the held-out set, the
  accuracy over queries whose (stream, key) was the LAST of the key's S writes in the body, and over the others; for the
  others, the fraction answered with the last-written stream's value; and accuracy by query stream.
"""

import itertools

import torch

from test_binding_capacity import BindTask
from test_router_layout import eta2
from test_instrument_v2 import TAU_END


def compositions(P, nb, lmin, lmax):
    return [c for c in itertools.product(range(lmin, lmax + 1), repeat=nb) if sum(c) == P]


class RandHeaderTask(BindTask):
    def __init__(self, P=8, S=2, n_vals=16, n_q=1, nb=2, lmin=2, lmax=6):
        super().__init__(P, S=S, n_vals=n_vals, n_q=n_q)        # grouped, then re-laid out below
        self.layout = "randheader"
        self.nb, self.lmin, self.lmax = int(nb), int(lmin), int(lmax)
        self.comps = compositions(P, nb, lmin, lmax)
        assert self.comps, (P, nb, lmin, lmax)
        self.key = f"P{P}rh{nb}"
        self.label += f", layout=randheader (NB={nb}, block lengths {lmin}..{lmax})"
        self.body_len = S * (nb + 2 * P)
        self.L = self.body_len + 3 * n_q
        self.T = self.L - 1
        self.qpos = [self.body_len + 3 * j + 1 for j in range(n_q)]

    def make_batch(self, B, gen):
        S, P, nv, nq, nb = self.S, self.P, self.n_vals, self.n_q, self.nb
        vals = torch.rand(B, P, nv, generator=gen).argsort(-1)[..., :S]        # (B,P,S) distinct per key
        comp = torch.randint(len(self.comps), (B, S), generator=gen)            # composition per (seq, stream)
        korder = torch.rand(B, S, P, generator=gen).argsort(-1)                 # key order within each stream
        border = torch.rand(B, S * nb, generator=gen).argsort(-1)               # block order: block id = s*nb + i
        q = torch.rand(B, S * P, generator=gen).argsort(-1)[:, :nq]             # no replacement
        comps = torch.tensor(self.comps)                                        # (n_comp, nb)
        lens = comps[comp]                                                      # (B, S, nb)
        starts = torch.cat([torch.zeros(B, S, 1, dtype=torch.long), lens.cumsum(-1)[..., :-1]], -1)
        rows = []
        V, K, Ln, St, Bo = vals.tolist(), korder.tolist(), lens.tolist(), starts.tolist(), border.tolist()
        for b in range(B):
            seq = []
            for blk in Bo[b]:
                s, i = divmod(blk, nb)
                seq.append(s)
                for kk in K[b][s][St[b][s][i]:St[b][s][i] + Ln[b][s][i]]:
                    seq.extend((S + kk, S + P + V[b][kk][s]))
            rows.append(seq)
        body = torch.tensor(rows, dtype=torch.long)
        qs, qk = q // P, q % P
        qv = vals.reshape(B, P * S).gather(1, qk * S + qs)
        queries = torch.stack([qs, S + qk, S + P + qv], dim=-1).reshape(B, -1)
        return torch.cat([body, queries], dim=1), qs


# ── Positions and routing statistics for header-type layouts (positions read per sequence) ─────────────────────────
def body_positions(task, x):
    """(kpos, vpos): (B, S·P) positions of the body's KEY tokens in order of appearance, and their VAL positions."""
    S, P = task.S, task.P
    body = x[:, :task.qpos[0] - 1]
    is_key = (body >= S) & (body < S + P)
    assert bool((is_key.sum(1) == S * P).all())
    idx = torch.arange(body.shape[1]).expand_as(body)
    kpos = torch.where(is_key, idx, torch.full_like(idx, 10 ** 9)).sort(1).values[:, :S * P]
    return kpos, kpos + 1


@torch.no_grad()
def rand_header_routing_stats(model, task, probe):
    assert task.n_q == 1
    was = model.training
    model.eval()
    try:
        pinp = probe[0]
        _, _, gr, gw = model(pinp, TAU_END)
    finally:
        model.train(was)
    S, P, n = task.S, task.P, task.S * task.P
    B = pinp.shape[0]
    kpos, vpos = body_positions(task, pinp)
    lab = task.stream_labels(pinp)
    j = torch.arange(n)
    ctx = lab.gather(1, kpos)
    key = pinp.gather(1, kpos) - S
    labels = dict(stream=ctx, key=key, half=(j >= n // 2).long().expand(B, n), index=j.expand(B, n))
    p = gr[..., 0]
    out = {}
    for role, pos in (("key", kpos), ("val", vpos)):
        for name, lb in labels.items():
            out[f"eta_{role}_by_{name}"] = eta2(p.gather(1, pos), lb)
    qpos = task.qpos[0]
    qs, qk = pinp[:, qpos - 1], pinp[:, qpos] - S
    same_key = key == qk[:, None]
    tgt = same_key & (ctx == qs[:, None])
    dis = same_key & (ctx != qs[:, None])
    assert bool((tgt.sum(1) == 1).all()) and bool((dis.sum(1) == S - 1).all())
    gwv = gw.gather(1, vpos.unsqueeze(-1).expand(-1, -1, gw.shape[-1]))
    dots = (gwv * gr[:, qpos][:, None, :]).sum(-1)
    marg = (dots * tgt).sum(1) - (dots * dis).sum(1) / (S - 1)
    out["margin"] = marg.mean().item()
    return out


def rand_header_stats(route_at):
    """run_one's stats_fn for k = 2 on RandHeaderTask: gate_stats_k at every evaluation, the routing statistics at
    route_at (as explore_far_cue.header_stats)."""
    from test_router_reliability import gate_stats_k

    @torch.no_grad()
    def stats(model, task, probe, step):
        out = gate_stats_k(model, task, probe, step)
        if step in route_at:
            out.update(rand_header_routing_stats(model, task, probe))
        return out
    return stats


# ── Write order (S53) ────────────────────────────────────────────────────────
@torch.no_grad()
def write_order_acc(model, task, data):
    """Held-out accuracy split by whether the query's pair was the last of its key's S writes (any layout whose body is
    [.., KEY, VAL, ..] pairs with the CTX before)."""
    from test_binding_onset import logits_of
    was = model.training
    model.eval()
    try:
        inp, tgt = data[:, :-1], data[:, 1:]
        lg = logits_of(model, inp)
    finally:
        model.train(was)
    S, P = task.S, task.P
    qpos = task.qpos[0]
    pred = lg[:, qpos].argmax(-1)
    ans = tgt[:, qpos]
    qs, qk = inp[:, qpos - 1], inp[:, qpos]
    body = inp[:, :qpos - 1]
    lab = task.stream_labels(inp)[:, :qpos - 1]
    idx = torch.arange(body.shape[1]).expand_as(body)
    hit = body == qk[:, None]                                           # the key's S occurrences in the body
    assert bool((hit.sum(1) == S).all())
    last_pos = torch.where(hit, idx, torch.full_like(idx, -1)).max(1).values
    last_stream = lab.gather(1, last_pos[:, None])[:, 0]
    last_val = inp.gather(1, (last_pos + 1)[:, None])[:, 0]
    is_last = last_stream == qs
    ok = pred == ans
    out = dict(n=int(ok.numel()), acc=float(ok.float().mean()),
               n_last=int(is_last.sum()), acc_last=float(ok[is_last].float().mean()) if is_last.any() else None,
               n_notlast=int((~is_last).sum()),
               acc_notlast=float(ok[~is_last].float().mean()) if (~is_last).any() else None,
               lastval_notlast=float((pred[~is_last] == last_val[~is_last]).float().mean()) if (~is_last).any() else None)
    out["acc_by_stream"] = [float(ok[qs == s].float().mean()) if (qs == s).any() else None for s in range(S)]
    return out


# ── CHECKs ───────────────────────────────────────────────────────────────────
def check_rand_header(task=None, B=4096, seed=7):
    t = RandHeaderTask() if task is None else task
    S, P, nb = t.S, t.P, t.nb
    x, qs_ret = t.make_batch(B, torch.Generator().manual_seed(seed))
    body, qry = x[:, :t.body_len], x[:, t.body_len:]
    rows = []
    kpos, vpos = body_positions(t, x[:, :-1])
    lab = t.stream_labels(x[:, :-1])
    st, keys, vals = lab.gather(1, kpos), x.gather(1, kpos) - S, x.gather(1, vpos)
    once = all(sorted(zip(st[b].tolist(), keys[b].tolist())) == [(s, k) for s in range(S) for k in range(P)]
               for b in range(B))
    rows.append(("every (stream, key) appears exactly once in the body", once))
    vtab = torch.full((B, S, P), -1, dtype=torch.long)
    vtab[torch.arange(B)[:, None], st, keys] = vals
    distinct = bool((vtab >= S + P).all()) and all(len(set(vtab[b, :, k].tolist())) == S for b in range(B) for k in range(P))
    rows.append(("S distinct values per key", distinct))
    qsx, qk, qv = qry[:, 0], qry[:, 1] - S, qry[:, 2]
    rows.append(("each query is answered by its own stream's value; make_batch returns the query streams",
                 bool((vtab[torch.arange(B), qsx, qk] == qv).all()) and bool((qs_ret[:, 0] == qsx).all())))
    # blocks: parse the body
    lens_ok, one_ctx, lens_seen, orders = True, True, [], set()
    for b in range(B):
        seq = body[b].tolist()
        blocks, cur = [], None
        for pos, tok in enumerate(seq):
            if tok < S:
                cur = [tok, 0]
                blocks.append(cur)
            else:
                one_ctx &= cur is not None
                if S <= tok < S + P:
                    cur[1] += 1
        one_ctx &= len(blocks) == S * nb and seq[0] < S
        per = {s: [ln for ss, ln in blocks if ss == s] for s in range(S)}
        lens_ok &= all(len(per[s]) == nb and tuple(per[s]) in t.comps for s in range(S))
        lens_seen += [ln for _, ln in blocks]
        orders.add(tuple(ss for ss, _ in blocks))
    rows.append((f"exactly one CTX per block, at its start; {S * nb} blocks per sequence", one_ctx))
    rows.append((f"each stream has {nb} blocks whose lengths are a composition of P = {P} with parts in {t.lmin}..{t.lmax}",
                 lens_ok))
    freq = {ln: lens_seen.count(ln) / len(lens_seen) for ln in range(t.lmin, t.lmax + 1)}
    unif = 1.0 / (t.lmax - t.lmin + 1)
    rows.append((f"block-length frequencies {dict((k, round(v, 3)) for k, v in freq.items())} each within 0.03 of {unif:.2f}",
                 all(abs(v - unif) <= 0.03 for v in freq.values())))
    rows.append((f"the stream order of the blocks varies ({len(orders)} distinct orders over {B} sequences)", len(orders) > 1))
    inp = x[:, :-1]
    idx = torch.arange(inp.shape[1]).expand_as(inp)
    is_ctx = inp < S
    last = torch.where(is_ctx, idx, torch.full_like(idx, -1)).cummax(1).values
    want = inp.gather(1, last.clamp(min=0))
    rows.append(("stream_labels = the latest CTX token's stream at every position", bool((t.stream_labels(inp) == want).all())))
    ql, qt, _ = t.select(torch.zeros(B, inp.shape[1], t.vocab), x[:, 1:], None)
    rows.append((f"L = {t.L} = S(NB + 2P) + 3 n_q, qpos = {t.qpos}; select reads the query's KEY position and targets its value",
                 x.shape[1] == t.L == S * (nb + 2 * P) + 3 * t.n_q and bool((qt == qv).all())
                 and bool((inp[:, t.qpos[0]] == S + qk).all())))
    return rows
