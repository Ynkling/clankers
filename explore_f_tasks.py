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


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Round 6 (19 October 2026): the lite layout without marked keys. Everything above this line is byte-identical to the
# file at a199d5e (and to the copy Session G vendored for S71-S87); everything below is new.
#
# RandHeaderTaskPool ("RandHeaderTask-pool"): S70's RandHeaderTask-lite layout exactly (S = 2 streams, P = 4 keys per
#   stream in NB = 2 blocks of 1-3 pairs, the 4 blocks in random order, one CTX per block, one query; L = 23), with the
#   4 keys of each sequence drawn WITHOUT REPLACEMENT from a pool of K_POOL = 32 key types (the same 4 keys serve both
#   streams, as in lite: every key is bound once in each stream, to S distinct values) and the values from the pool of
#   n_vals = 16 as before; the 2 CTX types as before (ids 0, 1). Vocabulary: CTX 0..S-1, KEY S..S+K_POOL-1, VAL
#   S+K_POOL..S+K_POOL+n_vals-1 (50).
#   make_batch draws lite's batch first (RandHeaderTask.make_batch, the same generator calls in the same order: the
#   values, compositions, key orders, block order and query), then the sequence's key types (one draw of B x K_POOL
#   uniforms, argsorted), and relabels lite's key i to the sequence's i-th key type; values and CTX keep their lite
#   identities (shifted into the new vocabulary). So for a given generator state the pool batch IS lite's batch with
#   the key identities renamed (CHECK), and a key's stream, block, position, value and the query are lite's.
# RandHeaderTaskAllPool ("RandHeaderTask-allpool"): the same, with the 2 CTX tokens of each sequence drawn without
#   replacement from a pool of N_CTX = 16 CTX types (one more draw after the key types). Vocabulary: CTX 0..15, KEY
#   16..47, VAL 48..63 (64). Nothing is then frequent by role: per sequence there are 5 CTX tokens over 16 types, 9 key
#   tokens over 32, 8 value tokens over 16. The only structure left is positional (a CTX token opens its block) and
#   relational (each key occurs twice, once under each of the sequence's two contexts, with different values).
#   STREAM INDEX: there is no fixed CTX-type -> stream map. The stream of a position is defined within its sequence by
#   order of first appearance: stream 0 = the context of the first block (position 0 is always a CTX token), stream 1 =
#   the other. stream_labels, the query's stream (make_batch's second output) and perfect_gate use this index; lite's
#   draws decide which of lite's streams opens the sequence, and that stream gets the first drawn CTX type.
#   This is the two-document associative-recall layout: MQAR (multi-query associative recall, Arora et al. 2023) with a
#   context token per document, at our vocabulary size: two "documents" (streams), each a set of key-value pairs under
#   its own context token, the same keys in both documents, and a query naming a document and a key. What MQAR proper
#   adds that this does not: many documents rather than two (here exactly two, split into two blocks each); queries
#   interleaved with the pairs and many per sequence (here one, at the end); and no context token at all (MQAR's keys
#   are unique within a sequence, so a key alone names its value; here a key names two values and only the context
#   disambiguates). It also has a far larger vocabulary and longer sequences.
#   The model's built-in perfect gate does not apply (test_instrument_v2.perfect_gate_general maps a fixed CTX token to
#   a fixed stream); use task.perfect_gate(tokens), a one-hot of the stream index (ctx_tokens is set to all 16 CTX types,
#   so the built-in gate gives 16 columns and fails loudly against k = 2 channels where shapes must match).
# Shared helpers: role_masks / key_mask / key_index / body_positions read the new vocabulary; pool_routing_stats is
#   rand_header_routing_stats with the keys found by task.key_mask and labelled by key type; pool_stats(route_at) is
#   rand_header_stats's composition (test_router_reliability.gate_stats_k at every evaluation — its key_sep / key_part read
#   token ids S..S+P-1 and are not meaningful on these tasks — plus pool_routing_stats at route_at).
# Frequency tables (Session G's S87 rule, copied): type_frequencies, largest_ratio_cut (S87's freq_types, line for line)
#   and half_mass_cut (the smallest K whose most frequent types cover half the token mass), on freq_sample(task): 4096
#   sequences from torch.Generator(87_000), input tokens tokens[:, :-1], every position.
# CHECKS (check_pool_task): the pool batch is lite's batch relabelled (the pool -> lite map recovers lite's batch from the
#   same generator state); each sequence's 4 key types are distinct and from the pool; every key occurs exactly S times,
#   once under each stream (so in one block of each stream); one value per (stream, key), distinct across the streams;
#   the query's (stream, key) occurs exactly once in the body and its target is that pair's value (the answer is
#   determined); one CTX per block at its start; allpool: each sequence's 2 CTX types distinct, position 0 is CTX, stream
#   labels by first appearance, perfect_gate one-hot of them; pool: perfect_gate equals perfect_gate_general(tokens,
#   (0, 1)) and stream_labels equal BindTask's; the frequency table's maxima as the round's CHECKs ask (-pool: no key
#   type above any value type's frequency; -allpool: no type above 1/8).
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
K_POOL, N_CTX_POOL = 32, 16
FREQ_SAMPLE, FREQ_SEED = 4096, 87_000


class RandHeaderTaskPool(RandHeaderTask):
    def __init__(self, P=4, S=2, n_vals=16, n_q=1, nb=2, lmin=1, lmax=3, k_pool=K_POOL):
        super().__init__(P, S=S, n_vals=n_vals, n_q=n_q, nb=nb, lmin=lmin, lmax=lmax)
        self.k_pool = int(k_pool)
        assert self.k_pool >= P
        self.n_ctx = S
        self._vocab_ids()
        self.key = f"P{P}rh{nb}pool{self.k_pool}"
        self.label += f", keys from a pool of {self.k_pool}"

    def _vocab_ids(self):
        self.key0 = self.n_ctx
        self.val0 = self.n_ctx + self.k_pool
        self.vocab = self.val0 + self.n_vals

    # the lite batch, relabelled
    def _lite_batch(self, B, gen):
        return RandHeaderTask.make_batch(self, B, gen)

    def _relabel(self, x, keyset, ctxset=None):
        """lite token ids -> this task's: KEY S+i -> key0 + keyset[b, i]; VAL S+P+j -> val0 + j; CTX s -> ctxset[b, s]
        (or s)."""
        S, P = self.S, self.P
        out = x.clone()
        isk = (x >= S) & (x < S + P)
        isv = x >= S + P
        out[isk] = self.key0 + keyset.gather(1, (x - S).clamp(0, P - 1))[isk]
        out[isv] = self.val0 + (x - S - P)[isv]
        if ctxset is not None:
            isc = x < S
            out[isc] = ctxset.gather(1, x.clamp(0, S - 1))[isc]
        return out

    def make_batch(self, B, gen):
        x, qs = self._lite_batch(B, gen)
        keyset = torch.rand(B, self.k_pool, generator=gen).argsort(-1)[:, :self.P]   # (B, P) distinct key types
        return self._relabel(x, keyset), qs

    # roles and positions in the new vocabulary
    def ctx_mask(self, x):
        return x < self.n_ctx

    def key_mask(self, x):
        return (x >= self.key0) & (x < self.val0)

    def val_mask(self, x):
        return x >= self.val0

    def key_index(self, x):
        """the key type (0..k_pool-1) at key positions, -1 elsewhere."""
        return torch.where(self.key_mask(x), x - self.key0, torch.full_like(x, -1))

    def role_masks(self, x):
        return {'ctx': self.ctx_mask(x), 'key': self.key_mask(x), 'val': self.val_mask(x)}

    def body_positions(self, x):
        """(kpos, vpos): (B, S·P) body KEY positions in order of appearance, and their VAL positions."""
        S, P = self.S, self.P
        body = x[:, :self.qpos[0] - 1]
        is_key = self.key_mask(body)
        assert bool((is_key.sum(1) == S * P).all())
        idx = torch.arange(body.shape[1]).expand_as(body)
        kpos = torch.where(is_key, idx, torch.full_like(idx, 10 ** 9)).sort(1).values[:, :S * P]
        return kpos, kpos + 1

    def perfect_gate(self, tokens, dtype=torch.float32):
        """One channel per stream: the one-hot of stream_labels (uniform 1/S before the first CTX token)."""
        lab = self.stream_labels(tokens)
        oh = torch.nn.functional.one_hot(lab.clamp(min=0), self.S).to(dtype)
        return torch.where((lab >= 0).unsqueeze(-1), oh, torch.full_like(oh, 1.0 / self.S))


class RandHeaderTaskAllPool(RandHeaderTaskPool):
    def __init__(self, P=4, S=2, n_vals=16, n_q=1, nb=2, lmin=1, lmax=3, k_pool=K_POOL, n_ctx=N_CTX_POOL):
        super().__init__(P, S=S, n_vals=n_vals, n_q=n_q, nb=nb, lmin=lmin, lmax=lmax, k_pool=k_pool)
        assert int(n_ctx) >= S
        self.n_ctx = int(n_ctx)
        self._vocab_ids()
        self.ctx_tokens = tuple(range(self.n_ctx))
        self.key = f"P{P}rh{nb}pool{self.k_pool}ctx{self.n_ctx}"
        self.label += f", contexts from a pool of {self.n_ctx}"

    def make_batch(self, B, gen):
        x, qs = self._lite_batch(B, gen)
        keyset = torch.rand(B, self.k_pool, generator=gen).argsort(-1)[:, :self.P]
        pair = torch.rand(B, self.n_ctx, generator=gen).argsort(-1)[:, :self.S]      # (B, S) distinct CTX types
        first = x[:, 0]                                                               # lite stream of the first block
        # lite stream s -> stream index by first appearance; the first-appearing stream gets pair[:, 0]
        order = torch.stack([first, 1 - first], 1) if self.S == 2 else None
        assert order is not None, "first-appearance relabelling is written for S = 2"
        ctxset = torch.empty_like(pair)
        ctxset.scatter_(1, order, pair)                                               # ctxset[b, lite s] = CTX type
        qidx = (qs != first[:, None]).long()                                          # the query's stream index
        return self._relabel(x, keyset, ctxset), qidx

    def stream_labels(self, x):
        """the stream index by first appearance (0 = the first block's context), -1 before any CTX token."""
        B, T = x.shape
        idx = torch.arange(T).expand(B, T)
        isc = self.ctx_mask(x)
        last = torch.where(isc, idx, torch.full_like(idx, -1)).cummax(dim=1).values
        cur = x.gather(1, last.clamp(min=0))
        lab = (cur != x[:, :1]).long()
        return torch.where(last >= 0, lab, torch.full_like(lab, -1))


# ── Routing statistics on the pool tasks ─────────────────────────────────────
@torch.no_grad()
def pool_routing_stats(model, task, probe):
    """rand_header_routing_stats with the keys found by task.key_mask and labelled by key type (key = token − key0)."""
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
    kpos, vpos = task.body_positions(pinp)
    lab = task.stream_labels(pinp)
    j = torch.arange(n)
    ctx = lab.gather(1, kpos)
    key = pinp.gather(1, kpos) - task.key0
    labels = dict(stream=ctx, key=key, half=(j >= n // 2).long().expand(B, n), index=j.expand(B, n))
    p = gr[..., 0]
    out = {}
    for role, pos in (("key", kpos), ("val", vpos)):
        for name, lb in labels.items():
            out[f"eta_{role}_by_{name}"] = eta2(p.gather(1, pos), lb)
    qpos = task.qpos[0]
    qs, qk = lab[:, qpos], pinp[:, qpos] - task.key0
    same_key = key == qk[:, None]
    tgt = same_key & (ctx == qs[:, None])
    dis = same_key & (ctx != qs[:, None])
    assert bool((tgt.sum(1) == 1).all()) and bool((dis.sum(1) == S - 1).all())
    gwv = gw.gather(1, vpos.unsqueeze(-1).expand(-1, -1, gw.shape[-1]))
    dots = (gwv * gr[:, qpos][:, None, :]).sum(-1)
    marg = (dots * tgt).sum(1) - (dots * dis).sum(1) / (S - 1)
    out["margin"] = marg.mean().item()
    return out


def pool_stats(route_at):
    """run_one's stats_fn for k >= 2 on the pool tasks (rand_header_stats's composition)."""
    from test_router_reliability import gate_stats_k

    @torch.no_grad()
    def stats(model, task, probe, step):
        out = gate_stats_k(model, task, probe, step)
        if step in route_at:
            out.update(pool_routing_stats(model, task, probe))
        return out
    return stats


# ── Token-type frequencies and the two cuts (Session G's S87 rule) ───────────
def freq_sample(task, n=FREQ_SAMPLE, seed=FREQ_SEED):
    """S87's sample: n sequences from torch.Generator(seed), the input tokens tokens[:, :-1]."""
    return task.make_batch(n, torch.Generator().manual_seed(seed))[0][:, :-1]


def type_frequencies(tokens):
    """[(type id, frequency)] sorted by frequency, descending, over every position of a LongTensor of token ids."""
    assert isinstance(tokens, torch.Tensor) and tokens.dtype == torch.long
    cnt = torch.bincount(tokens.reshape(-1)).double()
    fr = cnt / cnt.sum()
    order = [int(t) for t in fr.argsort(descending=True) if fr[t] > 0]
    return [(t, float(fr[t])) for t in order]


def largest_ratio_cut(tokens):
    """Session G's S87 freq_types (explore_g_s87_child at e46799a), line for line: (frequent type ids, table), the
    frequencies sorted and cut at the largest ratio between consecutive frequencies; the types above the cut."""
    assert isinstance(tokens, torch.Tensor) and tokens.dtype == torch.long
    cnt = torch.bincount(tokens.reshape(-1)).double()
    fr = cnt / cnt.sum()
    order = [int(t) for t in fr.argsort(descending=True) if fr[t] > 0]
    vals = [float(fr[t]) for t in order]
    cut = max(range(len(vals) - 1), key=lambda i: vals[i] / vals[i + 1])
    return order[:cut + 1], [(t, round(v, 5)) for t, v in zip(order, vals)]


def half_mass_cut(tokens):
    """The smallest K whose K most frequent types cover at least half the token mass: (those type ids, their mass)."""
    tab = type_frequencies(tokens)
    acc, sel = 0.0, []
    for t, f in tab:
        sel.append(t)
        acc += f
        if acc >= 0.5:
            break
    return sel, acc


def role_of(task, t):
    """'CTX' / 'KEY' / 'VAL' for a token id of a lite or pool task."""
    if isinstance(task, RandHeaderTaskPool):
        return "CTX" if t < task.n_ctx else "KEY" if t < task.val0 else "VAL"
    S, P = task.S, task.P
    return "CTX" if t < S else "KEY" if t < S + P else "VAL"


def freq_report(task, tokens=None):
    """The frequency table by role and both cuts, as a dict (for printing)."""
    tokens = freq_sample(task) if tokens is None else tokens
    tab = type_frequencies(tokens)
    by_role = {}
    for t, f in tab:
        by_role.setdefault(role_of(task, t), []).append(f)
    lr, _ = largest_ratio_cut(tokens)
    hm, mass = half_mass_cut(tokens)
    summ = {r: dict(n=len(v), min=min(v), max=max(v), mean=sum(v) / len(v), mass=sum(v)) for r, v in by_role.items()}
    return dict(table=tab, by_role=summ, largest_ratio=lr, largest_ratio_roles=[role_of(task, t) for t in lr],
                half_mass=hm, half_mass_roles=[role_of(task, t) for t in hm], half_mass_mass=mass)


# ── CHECKs for the pool tasks ────────────────────────────────────────────────
def check_pool_task(task, B=4096, seed=7):
    """[(name, ok)] for a RandHeaderTaskPool / RandHeaderTaskAllPool (module comment above)."""
    from test_instrument_v2 import perfect_gate_general
    rows = []
    S, P, nb = task.S, task.P, task.nb
    allpool = isinstance(task, RandHeaderTaskAllPool)
    lite = RandHeaderTask(P, S=S, n_vals=task.n_vals, n_q=task.n_q, nb=nb, lmin=task.lmin, lmax=task.lmax)
    x, qs = task.make_batch(B, torch.Generator().manual_seed(seed))
    xl, qsl = lite.make_batch(B, torch.Generator().manual_seed(seed))
    rows.append((f"shape: L {task.L} (lite {lite.L}), vocab {task.vocab}; batch {tuple(x.shape)}",
                 x.shape == xl.shape and task.L == lite.L and x.shape[1] == task.L))
    # the pool batch is lite's relabelled: the same roles at the same positions, the same value ids, the map back
    same_roles = (torch.equal(task.ctx_mask(x), xl < S) and torch.equal(task.key_mask(x), (xl >= S) & (xl < S + P))
                  and torch.equal(task.val_mask(x), xl >= S + P))
    same_vals = torch.equal(x[task.val_mask(x)] - task.val0, xl[xl >= S + P] - S - P)
    back_keys = True
    for b in range(0, B, 97):
        m = {}
        for t, tl in zip(x[b].tolist(), xl[b].tolist()):
            if S <= tl < S + P:
                back_keys &= m.setdefault(t, tl) == tl
        back_keys &= len(set(m.values())) == len(m) == P
    rows.append((f"the batch is lite's batch with the identities renamed: roles at the same positions ({same_roles}), the same "
                 f"value ids ({same_vals}), a one-to-one key map per sequence ({back_keys})", same_roles and same_vals and back_keys))
    # keys: distinct per sequence, from the pool; every key S times, once under each stream; one value per (stream, key)
    body = x[:, :task.qpos[0] - 1]
    lab = task.stream_labels(x)
    kidx = task.key_index(body)
    ok_k = ok_both = ok_val = True
    for b in range(B):
        ks = kidx[b][kidx[b] >= 0].tolist()
        types = set(ks)
        ok_k &= len(types) == P and all(0 <= k < task.k_pool for k in types)
        for k in types:
            pos = (kidx[b] == k).nonzero()[:, 0]
            ok_both &= len(pos) == S and sorted(lab[b, pos].tolist()) == list(range(S))
            vals = x[b, pos + 1].tolist()
            ok_val &= len(set(vals)) == S and all(v >= task.val0 for v in vals)
    rows.append((f"each sequence's {P} key types are distinct and from the pool of {task.k_pool} ({ok_k}); every key occurs "
                 f"exactly {S} times, once under each stream, i.e. in one block of each stream ({ok_both}); its {S} values are "
                 f"distinct ({ok_val})", ok_k and ok_both and ok_val))
    # the query is answered by exactly one body pair, and its target is that pair's value
    qp = task.qpos[0]
    qk, qst = x[:, qp], lab[:, qp]
    hit = (body == qk[:, None]) & (lab[:, :qp - 1] == qst[:, None])
    one = bool((hit.sum(1) == 1).all())
    kp = torch.where(hit, torch.arange(qp - 1).expand_as(hit), torch.full_like(hit, -1, dtype=torch.long)).max(1).values
    ans = x[:, qp + 1]
    rows.append((f"the query's (stream, key) occurs exactly once in the body ({one}) and its target is that pair's value "
                 f"({bool(torch.equal(x.gather(1, (kp + 1)[:, None])[:, 0], ans))}); the query stream from make_batch equals "
                 f"stream_labels at the query ({bool(torch.equal(qs.reshape(-1), qst))})",
                 one and bool(torch.equal(x.gather(1, (kp + 1)[:, None])[:, 0], ans)) and bool(torch.equal(qs.reshape(-1), qst))))
    # one CTX per block at its start: S·NB CTX tokens in the body, each followed by a key
    nctx = task.ctx_mask(body).sum(1)
    cpos = task.ctx_mask(body)
    after = torch.zeros_like(cpos)
    after[:, :-1] = cpos[:, :-1] & task.key_mask(body[:, 1:])
    rows.append((f"{S * nb} CTX tokens in each body, each followed by a key, position 0 a CTX",
                 bool((nctx == S * nb).all()) and torch.equal(after.sum(1), nctx) and bool(task.ctx_mask(x[:, :1]).all())))
    # the stream index and the perfect gate
    pg = task.perfect_gate(x[:, :-1])
    if allpool:
        two = True
        for b in range(0, B, 13):
            c = set(x[b][task.ctx_mask(x[b])].tolist())
            two &= len(c) == S
        lab0 = bool((lab[:, 0] == 0).all())
        rows.append((f"allpool: each sequence's {S} CTX types are distinct ({two}); the stream index is by first appearance "
                     f"(stream 0 at position 0: {lab0}); the perfect gate is its one-hot "
                     f"({bool(torch.equal(pg.argmax(-1), lab[:, :-1].clamp(min=0)))})",
                     two and lab0 and bool(torch.equal(pg.argmax(-1), lab[:, :-1].clamp(min=0)))))
    else:
        bl = BindTask.stream_labels(task, x)
        rows.append((f"pool: stream_labels equal BindTask's (CTX id) and perfect_gate equals perfect_gate_general(tokens, (0, 1))",
                     torch.equal(bl, lab) and torch.equal(pg, perfect_gate_general(x[:, :-1], (0, 1)))))
    # the frequency maxima the round asks for
    fr = freq_report(task)
    br = fr["by_role"]
    if allpool:
        top = max(f for _, f in fr["table"])
        rows.append((f"allpool: no token type above 1/8 (max {top:.4f}; CTX {br['CTX']['max']:.4f}, KEY {br['KEY']['max']:.4f}, "
                     f"VAL {br['VAL']['max']:.4f})", top < 1 / 8))
    else:
        rows.append((f"pool: no key type above any value type's frequency (max KEY {br['KEY']['max']:.4f} < min VAL "
                     f"{br['VAL']['min']:.4f})", br["KEY"]["max"] < br["VAL"]["min"]))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Round 6, revision of 20 October (referee V's review of S89-F, claude/review-V 3ae815b, fixes 1 and 2). APPENDED ONLY:
# everything above is byte-identical to 50dc8e4, the commit G and H vendored first, so both classes and every helper
# above behave exactly as there.
#
# SWAP-INVARIANT ROUTING STATISTICS (fix 1). On RandHeaderTask-allpool the stream index is a per-sequence convention
#   (first appearance). The pooled statistics, eta^2 by stream at KEY (pool_routing_stats) and VAL cos (gate_stats_k's,
#   from test_multilayer_binding.role_cosines), pool positions across sequences under that index, so they measure the
#   labelling as well as the routing: a gate that separates the two streams of every sequence but puts the first block's
#   context on channel 0 in only half of them scores eta^2 ~0 and VAL cos ~1 (check_inv_stats). The within-sequence
#   forms (pool_routing_stats_inv):
#     eta_key_by_stream_inv   eta^2 of the read gate's channel-0 probability over the sequence's S·P body KEY positions,
#                             grouped by stream (test_router_layout.eta2 on each sequence alone: 0 where the sequence's
#                             values are constant), averaged over the probe's sequences;
#     val_cos_inv             the cosine between the two streams' mean read gates (k-vectors) over the sequence's S·P body
#                             VAL positions (role_masks' 'val' of the probe input), per sequence, averaged over the probe;
#     seq_routed_frac         the fraction of probe sequences whose eta^2 alone exceeds ROUTED_ETA (descriptive).
#   Both are invariant to any permutation of the channels within a sequence (eta^2 of p is invariant to p -> 1 - p; a
#   common permutation of two vectors' coordinates leaves their cosine unchanged), so neither reads the labelling.
#   pool_stats_inv(route_at): pool_stats(route_at) plus the three fields at every evaluation (one more noise-free
#   forward on the probe; no generator is drawn, so training is unchanged).
#   pool_outcome(rec, invariant): ROUTED*@end (end margin >= 0.9 and eta^2 by stream at KEY > 0.9; S70 / S87's
#   exploratory definition, not a main-line outcome), ROUTED-BOUND (bound and ROUTED*@end) and DISCOVERED (bound and end
#   VAL cos < test_router_reliability.DISC_COS = 0.5), from the pooled fields (invariant False: eta_key_by_stream,
#   rec["val_cos"]) or the within-sequence ones (True: eta_key_by_stream_inv, end val_cos_inv). On ALLPOOL the readings use
#   the within-sequence fields; on POOL both are recorded and printed (they agree for a gate that is the same function of
#   the CTX type in every sequence; check_inv_stats). The margin is already within-sequence (the query's read gate
#   against its target's and distractor's write gates) and serves both.
# THE ROUTING CAP ON ALLPOOL (fix 2). No fixed map from CTX type to channel separates a sequence's two contexts in more
#   than 64/120 = 0.533 of sequences: with the 16 types split a / 16 - a over two channels, a sequence's 2 distinct types
#   (one of C(16, 2) = 120 unordered pairs, uniform) fall on different channels in a·(16 - a) pairs, at most 8·8 = 64.
#   A gate that reads token identity (LOCAL3, the register gates) can route the remaining sequences only by comparing
#   the contexts within the sequence. task.perfect_gate is handed that comparison (the index by first appearance), so a
#   valid perfect gate on ALLPOOL says the memory can use a relative routing, not that a gate under test can express
#   one; a null on ALLPOOL does not separate "keys are not frequent" from "contexts have no fixed identity".
#   type_level_gate(task, tokens, ch_of_type): the type-level perfect gate, the one-hot of ch_of_type[the latched CTX
#   type] (uniform 1/k before the first CTX token). BALANCED_MAP: CTX types 0-7 -> channel 0, 8-15 -> channel 1 (a
#   balanced map, at the cap). On POOL, with ch_of_type = (0, 1), it is task.perfect_gate.
# check_inv_stats(task): the synthetic-gate CHECK (no model): the relative perfect gate; the same with the channels
#   swapped in a random half of the sequences; a gate that sends each sequence's lower CTX type to channel 0 (a
#   context-type gate that separates every sequence, whichever channel the first block's context takes); on ALLPOOL the
#   balanced type-level gate. The within-sequence forms score eta^2 1 and cos 0 on the first three on both tasks; on
#   ALLPOOL the pooled forms score ~0 and ~1 on the swapped and the type-ordered gates; the balanced type-level gate's
#   within-sequence eta^2 equals the probe's fraction of separated sequences exactly; the vectorised per-sequence eta^2
#   equals test_router_layout.eta2 applied to each sequence alone.
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
ROUTED_MARGIN, ROUTED_ETA = 0.9, 0.9
BALANCED_MAP = tuple(int(t >= N_CTX_POOL // 2) for t in range(N_CTX_POOL))


def eta2_rows(x, lab):
    """test_router_layout.eta2 on each row of x (B, n) grouped by lab (B, n): (B,) float64; 0 where a row is constant."""
    x = x.double()
    mu = x.mean(1)
    tot = ((x - mu[:, None]) ** 2).sum(1)
    between = torch.zeros_like(tot)
    for g in lab.unique().tolist():
        m = (lab == g).double()
        cnt = m.sum(1)
        mg = (x * m).sum(1) / cnt.clamp(min=1)
        between = between + torch.where(cnt > 0, cnt * (mg - mu) ** 2, torch.zeros_like(cnt))
    return torch.where(tot > 0, between / torch.where(tot > 0, tot, torch.ones_like(tot)), torch.zeros_like(tot))


def _positions(task, x):
    return task.body_positions(x) if hasattr(task, "body_positions") else body_positions(task, x)


def inv_from_gate(task, pinp, gr):
    """The within-sequence forms from a read gate gr (B, T, k) on the probe input pinp (module comment above)."""
    import torch.nn.functional as F
    kpos, vpos = _positions(task, pinp)
    lab = task.stream_labels(pinp)
    eta = eta2_rows(gr[..., 0].gather(1, kpos), lab.gather(1, kpos))
    k = gr.shape[-1]
    gv = gr.gather(1, vpos.unsqueeze(-1).expand(-1, -1, k))
    lv = lab.gather(1, vpos)
    means = []
    for s in range(task.S):
        m = (lv == s).to(gv.dtype).unsqueeze(-1)
        means.append((gv * m).sum(1) / m.sum(1).clamp(min=1))
    cos = F.cosine_similarity(means[0], means[1], dim=-1)
    return dict(eta_key_by_stream_inv=eta.mean().item(), val_cos_inv=cos.double().mean().item(),
                seq_routed_frac=(eta > ROUTED_ETA).double().mean().item())


@torch.no_grad()
def pool_routing_stats_inv(model, task, probe):
    """The within-sequence forms on the probe (module comment above), noise-free and in eval mode."""
    assert task.S == 2, "the within-sequence forms are written for S = 2"
    was = model.training
    model.eval()
    try:
        pinp = probe[0]
        gr = model(pinp, TAU_END)[2]
    finally:
        model.train(was)
    return inv_from_gate(task, pinp, gr)


def pool_stats_inv(route_at):
    """pool_stats(route_at) plus the within-sequence forms at every evaluation."""
    base = pool_stats(route_at)

    @torch.no_grad()
    def stats(model, task, probe, step):
        out = base(model, task, probe, step)
        out.update(pool_routing_stats_inv(model, task, probe))
        return out
    return stats


def pool_outcome(rec, invariant):
    """ROUTED*@end, ROUTED-BOUND and DISCOVERED from a run_one record (module comment above). None fields read False."""
    from test_router_reliability import DISC_COS
    rec = rec or {}
    e = rec.get("end") or {}
    bound = rec.get("transition") is not None
    eta = e.get("eta_key_by_stream_inv" if invariant else "eta_key_by_stream")
    vc = e.get("val_cos_inv") if invariant else rec.get("val_cos")
    mg = e.get("margin")
    routed = mg is not None and eta is not None and mg >= ROUTED_MARGIN and eta > ROUTED_ETA
    return dict(form="within-sequence" if invariant else "pooled", bound=bound, routed_end=bool(routed),
                routed_bound=bool(bound and routed), discovered=bool(bound and vc is not None and vc < DISC_COS),
                margin=mg, eta_key_by_stream=eta, val_cos=vc)


def type_level_gate(task, tokens, ch_of_type=BALANCED_MAP, dtype=torch.float32):
    """The type-level perfect gate: one-hot of ch_of_type[the latched CTX type]; uniform 1/k before the first CTX."""
    B, T = tokens.shape
    idx = torch.arange(T).expand(B, T)
    last = torch.where(task.ctx_mask(tokens), idx, torch.full_like(idx, -1)).cummax(dim=1).values
    cur = tokens.gather(1, last.clamp(min=0))
    cmap = torch.tensor(ch_of_type, dtype=torch.long)
    assert len(ch_of_type) == task.n_ctx
    k = int(cmap.max()) + 1
    oh = torch.nn.functional.one_hot(cmap[cur.clamp(0, task.n_ctx - 1)], k).to(dtype)
    return torch.where((last >= 0).unsqueeze(-1), oh, torch.full_like(oh, 1.0 / k))


class _GateStub(torch.nn.Module):
    """A model stand-in whose read and write gates are gate_fn(tokens): for the statistics' CHECKs only."""

    def __init__(self, gate_fn):
        super().__init__()
        self.gate_fn = gate_fn

    def forward(self, tokens, tau=None, track_sat=False):
        g = self.gate_fn(tokens)
        return None, None, g, g


def check_inv_stats(task, seed=2800):
    """[(name, ok)]: the synthetic-gate CHECK (module comment above), on test_multilayer_binding.probe_batch(task, seed)."""
    from test_multilayer_binding import probe_batch
    probe = probe_batch(task, seed)
    pinp = probe[0]
    B = pinp.shape[0]
    allpool = isinstance(task, RandHeaderTaskAllPool)
    rel = lambda x: task.perfect_gate(x)
    flip = torch.rand(B, generator=torch.Generator().manual_seed(seed + 1)) < 0.5

    def swapped(x):
        g = task.perfect_gate(x)
        return torch.where(flip[:x.shape[0], None, None], g.flip(-1), g)

    def type_ordered(x):
        isc = task.ctx_mask(x)
        lo = torch.where(isc, x, torch.full_like(x, 10 ** 6)).min(1).values
        # channel 0 for the sequence's lower CTX type, channel 1 for the other
        idx = torch.arange(x.shape[1]).expand_as(x)
        last = torch.where(isc, idx, torch.full_like(idx, -1)).cummax(dim=1).values
        cur = x.gather(1, last.clamp(min=0))
        oh = torch.nn.functional.one_hot((cur != lo[:, None]).long(), 2).float()
        return torch.where((last >= 0).unsqueeze(-1), oh, torch.full_like(oh, 0.5))

    gates = dict(relative=rel, swapped=swapped, type_ordered=type_ordered)
    if allpool:
        gates["type_balanced"] = lambda x: type_level_gate(task, x, BALANCED_MAP)
    pooled_fn = pool_stats(("end",))
    rows, res = [], {}
    for nm, fn in gates.items():
        stub = _GateStub(fn)
        po = pooled_fn(stub, task, probe, "end")
        iv = pool_routing_stats_inv(stub, task, probe)
        res[nm] = (po, iv)
    for nm in ("relative", "swapped", "type_ordered"):
        po, iv = res[nm]
        ok = abs(iv["eta_key_by_stream_inv"] - 1) < 1e-9 and abs(iv["val_cos_inv"]) < 1e-6 and abs(po["margin"] - 1) < 1e-6
        rows.append((f"{task.key}: synthetic gate '{nm}': within-sequence eta^2 {iv['eta_key_by_stream_inv']:.6f} (1), VAL cos "
                     f"{iv['val_cos_inv']:.6f} (0), margin {po['margin']:.6f} (1); pooled eta^2 {po['eta_key_by_stream']:.4f}, "
                     f"VAL cos {po['val_cos']:.4f}", ok))
    po, _ = res["relative"]
    rows.append((f"{task.key}: the relative perfect gate's pooled forms are eta^2 1 and VAL cos 0 ({po['eta_key_by_stream']:.6f}, "
                 f"{po['val_cos']:.6f})", abs(po["eta_key_by_stream"] - 1) < 1e-9 and abs(po["val_cos"]) < 1e-6))
    for nm in ("swapped", "type_ordered"):
        po, _ = res[nm]
        if allpool or nm == "swapped":
            ok = po["eta_key_by_stream"] < 0.05 and po["val_cos"] > 0.95
            rows.append((f"{task.key}: gate '{nm}' (separates every sequence): the pooled forms read the labelling, eta^2 "
                         f"{po['eta_key_by_stream']:.4f} (~0) and VAL cos {po['val_cos']:.4f} (~1)", ok))
        else:
            ok = abs(po["eta_key_by_stream"] - 1) < 1e-9 and abs(po["val_cos"]) < 1e-6
            rows.append((f"{task.key}: gate '{nm}' is the fixed CTX -> channel map here: the pooled forms agree with the "
                         f"within-sequence ones (eta^2 {po['eta_key_by_stream']:.6f}, VAL cos {po['val_cos']:.6f})", ok))
    if allpool:
        po, iv = res["type_balanced"]
        cm = torch.tensor(BALANCED_MAP)
        isc = task.ctx_mask(pinp)
        sep = torch.tensor([len(set(cm[pinp[b][isc[b]]].tolist())) == 2 for b in range(B)]).double().mean().item()
        ok = abs(iv["eta_key_by_stream_inv"] - sep) < 1e-9 and abs(iv["val_cos_inv"] - (1 - sep)) < 1e-6
        rows.append((f"{task.key}: the balanced type-level gate separates {sep:.4f} of the probe's {B} sequences (cap 64/120 = "
                     f"0.5333): within-sequence eta^2 {iv['eta_key_by_stream_inv']:.4f} equals it, VAL cos "
                     f"{iv['val_cos_inv']:.4f} equals 1 minus it; pooled eta^2 {po['eta_key_by_stream']:.4f}, VAL cos "
                     f"{po['val_cos']:.4f}, margin {po['margin']:.4f}", ok))
    g = torch.rand(64, pinp.shape[1], 2, generator=torch.Generator().manual_seed(seed + 2)).softmax(-1)
    kpos, _ = _positions(task, pinp[:64])
    lab = task.stream_labels(pinp[:64]).gather(1, kpos)
    v = eta2_rows(g[..., 0].gather(1, kpos), lab)
    ref = torch.tensor([eta2(g[b, :, 0].gather(0, kpos[b]), lab[b]) for b in range(64)], dtype=torch.float64)
    d = (v - ref).abs().max().item()
    rows.append((f"{task.key}: the vectorised per-sequence eta^2 equals test_router_layout.eta2 on each sequence alone (max "
                 f"|diff| {d:.1e} on 64 sequences, a random gate)", d < 1e-12))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Round 7 (22 October): the context-pool family (S90-F). APPENDED ONLY: everything above is byte-identical to b0c6332 (the
# commit G and H vendored for round 6), so every class and helper above behaves exactly as there.
#
# Round 6 showed that on -allpool the contexts' missing identity, not the keys, is what no learned gate crossed. -allpool
# changes two things at once against lite (V's screen review of S89-F, fix 2; G's caveat): the keys stop being frequent
# AND the contexts stop having a fixed identity. This family changes only the second.
# RandHeaderTaskCtxPool ("RandHeaderTask-ctxpool"): lite (S = 2 streams, P = 4 keys per stream in NB = 2 blocks of 1-3
#   pairs, the 4 blocks in random order, one CTX per block, one query; L = 23) with lite's 4 key types KEPT (lite's key i
#   is key type i, every sequence) and only the S CTX tokens of each sequence drawn without replacement from a pool of
#   N_CTX_POOL = 16 CTX types. Vocabulary: CTX 0..15, KEY 16..19, VAL 20..35 (36). make_batch draws lite's batch (the
#   same generator calls in the same order) and then the S CTX types (one draw of B x 16 uniforms, argsorted; no key draw),
#   and renames lite's contexts only: the batch IS lite's batch with the contexts renamed (CHECK).
#   STREAM INDEX (any S): by order of first appearance in the sequence, stream 0 = the first block's context, stream 1 =
#   the next new context, ...; stream_labels, the query's stream (make_batch's second output, (B, n_q)) and perfect_gate
#   use it. At S = 2 it is -allpool's convention (CHECK).
# RandHeaderTaskDocs ("RandHeaderTask-docs", D documents): RandHeaderTaskCtxPool at S = D streams ("documents"), D = 3 or
#   4: each document's 4 pairs in 2 blocks of 1-3, the 2·D blocks in random order, one CTX per block, the D contexts of a
#   sequence distinct and drawn from the 16, every key once in every document with D distinct values, one query naming a
#   (document, key). L = D·(NB + 2P) + 3 (D = 3: 33; D = 4: 43). The MQAR-with-contexts layout at D documents (one query;
#   MQAR proper has many queries and no context token; see the round-6 section).
# THE ROUTING CAP for a fixed type -> channel map: with k channels and a map putting n_1..n_k of the 16 CTX types on each
#   channel, a sequence's D contexts (a uniform D-subset of the 16) land on D distinct channels in e_D(n_1..n_k) of the
#   C(16, D) subsets (e_D the elementary symmetric polynomial). type_map_cap(16, D, k) maximises this exactly over all
#   channel sizes (the partitions of 16 into at most k parts); the maximum is the balanced split (balanced_map). At k = D
#   (one channel per document, the perfect gate's k): D = 2: 64/120 = 0.533; D = 3: 150/560 = 0.268; D = 4: 256/1820 =
#   0.141. With k = 16 channels (one per type) the cap is 1: the cap is a property of k = D channels, which is what the
#   perfect gate and the type-level gate here use. A gate that reads context identity at k = D can route the rest only
#   by comparing contexts within the sequence; the perfect gate is handed that comparison (the index).
# WITHIN-SEQUENCE STATISTICS at any S and k (inv_from_gate_k): per sequence, each stream's mean read gate over its P body
#   KEY positions and over its P body VAL positions;
#     sep_key_inv          the mean over stream pairs of the total-variation distance between their mean KEY gates;
#     seq_separated_frac   the fraction of sequences whose S mean KEY gates have S distinct argmax channels, each with
#                          max >= ROUTED_ETA (0.9);
#     val_cos_inv_k        the mean over stream pairs of the cosine between their mean VAL gates.
#   All invariant to a per-sequence permutation of the channels. (eta^2 of a k-vector by stream is not used at S > 2: it
#   is 1 whenever each stream's gate is constant, even when two streams share a channel.) At S = k = 2, val_cos_inv_k
#   equals inv_from_gate's val_cos_inv (CHECK). ctx_stats(route_at): pool_stats plus inv_from_gate_k at every
#   evaluation (and, at S = 2, pool_routing_stats_inv); gate_stats_k's key_sep and key_part, which read token ids
#   S..S+P-1 (CTX types here), are set to NaN.
# ERRATUM (22 October): write_order_acc reads the query's stream as the CTX TOKEN before the query's key and compares it
#   with stream_labels. That is right where the CTX ids are the stream indices (lite, -pool) and WRONG on -allpool, where
#   the query's CTX token is a type 0..15 and stream_labels is the index 0..1: its last / not-last split and acc_by_stream
#   on -allpool are invalid (round 6's report 9 printed them for -allpool runs; withdrawn there). write_order_acc_idx reads
#   the query's stream as stream_labels at the query's CTX position; it equals write_order_acc on -pool (CHECK) and is
#   correct on every task here (its last-write mask equals the one computed on lite's own batch, CHECK).
# check_ctx_task(task): the structure CHECKs (below); check_inv_stats_k(task): the synthetic-gate CHECK at any S.
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
DOCS_D = (3, 4)


class RandHeaderTaskCtxPool(RandHeaderTaskPool):
    def __init__(self, P=4, S=2, n_vals=16, n_q=1, nb=2, lmin=1, lmax=3, n_ctx=N_CTX_POOL):
        super().__init__(P, S=S, n_vals=n_vals, n_q=n_q, nb=nb, lmin=lmin, lmax=lmax, k_pool=P)
        assert int(n_ctx) >= S
        self.n_ctx = int(n_ctx)
        self._vocab_ids()
        self.ctx_tokens = tuple(range(self.n_ctx))
        self.key = f"P{P}rh{nb}S{S}ctx{self.n_ctx}"
        self.label = (f"P={P}: RandHeaderTask-ctxpool, S={S}, n_vals={n_vals}, n_q={n_q}, layout=randheader (NB={nb}, block "
                      f"lengths {lmin}..{lmax}), lite's {P} key types, contexts from a pool of {self.n_ctx}")

    def first_rank(self, x):
        """(B, S): the order of first appearance of each LITE stream s (ids 0..S-1) in lite's batch x."""
        B, T = x.shape
        S = self.S
        idx = torch.arange(T).expand(B, T)
        pos = torch.stack([torch.where(x == s, idx, torch.full_like(idx, T)).min(1).values for s in range(S)], 1)
        return pos.argsort(1).argsort(1)

    def make_batch(self, B, gen):
        x, qs = self._lite_batch(B, gen)
        pair = torch.rand(B, self.n_ctx, generator=gen).argsort(-1)[:, :self.S]     # (B, S) distinct, by first appearance
        rank = self.first_rank(x[:, :self.body_len])
        ctxset = pair.gather(1, rank)                                                # ctxset[b, lite s] = CTX type
        keyset = torch.arange(self.P).expand(B, self.P)                              # lite's keys, unchanged
        return self._relabel(x, keyset, ctxset), rank.gather(1, qs)

    def stream_labels(self, x):
        """the stream index by first appearance of the latched CTX type (0 = the first block's), -1 before any CTX."""
        B, T = x.shape
        idx = torch.arange(T).expand(B, T)
        isc = self.ctx_mask(x)
        last = torch.where(isc, idx, torch.full_like(idx, -1)).cummax(dim=1).values
        cur = x.gather(1, last.clamp(min=0))
        first = torch.full((B, self.n_ctx), T, dtype=torch.long)
        first.scatter_reduce_(1, torch.where(isc, x, torch.zeros_like(x)), torch.where(isc, idx, torch.full_like(idx, T)),
                              reduce="amin")
        rank_of_type = (first[:, None, :] < first[:, :, None]).sum(-1)              # (B, n_ctx)
        lab = rank_of_type.gather(1, cur.clamp(0, self.n_ctx - 1))
        return torch.where(last >= 0, lab, torch.full_like(lab, -1))


class RandHeaderTaskDocs(RandHeaderTaskCtxPool):
    def __init__(self, D=3, P=4, n_vals=16, n_q=1, nb=2, lmin=1, lmax=3, n_ctx=N_CTX_POOL):
        super().__init__(P, S=D, n_vals=n_vals, n_q=n_q, nb=nb, lmin=lmin, lmax=lmax, n_ctx=n_ctx)
        self.D = int(D)
        self.key = f"P{P}rh{nb}docs{self.D}ctx{self.n_ctx}"
        self.label = self.label.replace("RandHeaderTask-ctxpool", f"RandHeaderTask-docs (D={self.D} documents)")


# ── The routing cap of a fixed type -> channel map ───────────────────────────
def balanced_map(n_ctx, k):
    """CTX type t -> channel t·k // n_ctx: sizes differ by at most 1."""
    return tuple(t * k // n_ctx for t in range(n_ctx))


def _esym(sizes, D):
    e = [1] + [0] * D
    for n in sizes:
        for j in range(D, 0, -1):
            e[j] += e[j - 1] * n
    return e[D]


def _partitions(n, k, top=None):
    top = n if top is None else top
    if n == 0:
        yield ()
        return
    if k == 0:
        return
    for first in range(min(n, top), 0, -1):
        for rest in _partitions(n - first, k - 1, first):
            yield (first,) + rest


def type_map_cap(n_ctx, D, k):
    """(cap, best sizes, count, C(n_ctx, D)): the largest fraction of a sequence's D contexts (a uniform D-subset) that
    a fixed map of the n_ctx types to k channels sends to D distinct channels, over every split of the types (exact)."""
    import math
    tot = math.comb(n_ctx, D)
    best = max(_partitions(n_ctx, k), key=lambda s: _esym(s, D))
    return _esym(best, D) / tot, best, _esym(best, D), tot


# ── Within-sequence statistics at any S and k ────────────────────────────────
def inv_from_gate_k(task, pinp, gr):
    """sep_key_inv, seq_separated_frac, val_cos_inv_k from a read gate gr (B, T, k) on the probe input (module comment)."""
    import itertools as it
    import torch.nn.functional as F
    kpos, vpos = _positions(task, pinp)
    lab = task.stream_labels(pinp)
    k = gr.shape[-1]
    means = {}
    for role, pos in (("key", kpos), ("val", vpos)):
        g = gr.gather(1, pos.unsqueeze(-1).expand(-1, -1, k))
        lb = lab.gather(1, pos)
        ms = []
        for s in range(task.S):
            m = (lb == s).to(g.dtype).unsqueeze(-1)
            ms.append((g * m).sum(1) / m.sum(1).clamp(min=1))
        means[role] = torch.stack(ms, 1)                                     # (B, S, k)
    pairs = list(it.combinations(range(task.S), 2))
    mk, mv = means["key"].double(), means["val"]
    tvd = torch.stack([0.5 * (mk[:, a] - mk[:, b]).abs().sum(-1) for a, b in pairs], 1).mean(1)
    cos = torch.stack([F.cosine_similarity(mv[:, a], mv[:, b], dim=-1) for a, b in pairs], 1).double().mean(1)
    mx, am = mk.max(-1)
    distinct = torch.tensor([len(set(r)) == task.S for r in am.tolist()])
    sep = distinct & (mx >= ROUTED_ETA).all(1)
    return dict(sep_key_inv=tvd.mean().item(), seq_separated_frac=sep.double().mean().item(),
                val_cos_inv_k=cos.mean().item())


@torch.no_grad()
def pool_routing_stats_inv_k(model, task, probe):
    was = model.training
    model.eval()
    try:
        pinp = probe[0]
        gr = model(pinp, TAU_END)[2]
    finally:
        model.train(was)
    return inv_from_gate_k(task, pinp, gr)


def ctx_stats(route_at):
    """run_one's stats_fn on the context-pool tasks (module comment above)."""
    base = pool_stats(route_at)
    nan = float("nan")

    @torch.no_grad()
    def stats(model, task, probe, step):
        out = base(model, task, probe, step)
        for f in ("key_sep", "key_part"):
            if f in out:
                out[f] = nan
        if task.S == 2:
            out.update(pool_routing_stats_inv(model, task, probe))
        out.update(pool_routing_stats_inv_k(model, task, probe))
        return out
    return stats


# ── The write order with the query's stream index (ERRATUM above) ────────────
def last_write_mask(task, inp, qstream):
    """(B,) bool: the query's pair is the last of its key's S writes; qstream (B,) the query's stream in the same
    convention as task.stream_labels."""
    qpos = task.qpos[0]
    body = inp[:, :qpos - 1]
    lab = task.stream_labels(inp)[:, :qpos - 1]
    idx = torch.arange(body.shape[1]).expand_as(body)
    hit = body == inp[:, qpos][:, None]
    assert bool((hit.sum(1) == task.S).all())
    last_pos = torch.where(hit, idx, torch.full_like(idx, -1)).max(1).values
    return lab.gather(1, last_pos[:, None])[:, 0] == qstream, last_pos


@torch.no_grad()
def write_order_acc_idx(model, task, data):
    """write_order_acc with the query's stream as stream_labels at the query's CTX position (ERRATUM above)."""
    from test_binding_onset import logits_of
    was = model.training
    model.eval()
    try:
        inp, tgt = data[:, :-1], data[:, 1:]
        lg = logits_of(model, inp)
    finally:
        model.train(was)
    S = task.S
    qpos = task.qpos[0]
    pred = lg[:, qpos].argmax(-1)
    ans = tgt[:, qpos]
    qs = task.stream_labels(inp)[:, qpos - 1]
    is_last, last_pos = last_write_mask(task, inp, qs)
    last_val = inp.gather(1, (last_pos + 1)[:, None])[:, 0]
    ok = pred == ans
    out = dict(n=int(ok.numel()), acc=float(ok.float().mean()),
               n_last=int(is_last.sum()), acc_last=float(ok[is_last].float().mean()) if is_last.any() else None,
               n_notlast=int((~is_last).sum()),
               acc_notlast=float(ok[~is_last].float().mean()) if (~is_last).any() else None,
               lastval_notlast=float((pred[~is_last] == last_val[~is_last]).float().mean()) if (~is_last).any() else None)
    out["acc_by_stream"] = [float(ok[qs == s].float().mean()) if (qs == s).any() else None for s in range(S)]
    return out


# ── CHECKs for the context-pool tasks ────────────────────────────────────────
def check_ctx_task(task, B=4096, seed=7):
    """[(name, ok)] for a RandHeaderTaskCtxPool / RandHeaderTaskDocs (module comment above)."""
    rows = []
    S, P, nb = task.S, task.P, task.nb
    lite = RandHeaderTask(P, S=S, n_vals=task.n_vals, n_q=task.n_q, nb=nb, lmin=task.lmin, lmax=task.lmax)
    x, qidx = task.make_batch(B, torch.Generator().manual_seed(seed))
    xl, qsl = lite.make_batch(B, torch.Generator().manual_seed(seed))
    rows.append((f"{task.key}: shape: L {task.L} (lite at S = {S}: {lite.L}), vocab {task.vocab} = {task.n_ctx} + {P} + "
                 f"{task.n_vals}; batch {tuple(x.shape)}", x.shape == xl.shape and task.L == lite.L == x.shape[1]
                 and task.vocab == task.n_ctx + P + task.n_vals))
    for nm, v in check_rand_header(lite, B=B, seed=seed):
        rows.append((f"{task.key}: lite's layout at S = {S}: {nm}", v))
    same_roles = (torch.equal(task.ctx_mask(x), xl < S) and torch.equal(task.key_mask(x), (xl >= S) & (xl < S + P))
                  and torch.equal(task.val_mask(x), xl >= S + P))
    same_keys = torch.equal(x[task.key_mask(x)] - task.key0, xl[(xl >= S) & (xl < S + P)] - S)
    same_vals = torch.equal(x[task.val_mask(x)] - task.val0, xl[xl >= S + P] - S - P)
    cmap_ok, distinct = True, True
    for b in range(0, B, 7):
        m = {}
        for t, tl in zip(x[b].tolist(), xl[b].tolist()):
            if tl < S:
                cmap_ok &= m.setdefault(tl, t) == t
        distinct &= len(set(m.values())) == len(m) == S and all(0 <= t < task.n_ctx for t in m.values())
    rows.append((f"{task.key}: the batch is lite's batch with only the contexts renamed: roles at the same positions "
                 f"({same_roles}), lite's key types ({same_keys}) and values ({same_vals}), one CTX type per lite stream "
                 f"({cmap_ok}), the sequence's {S} types distinct and from the pool of {task.n_ctx} ({distinct})",
                 same_roles and same_keys and same_vals and cmap_ok and distinct))
    lab = task.stream_labels(x)
    rank = task.first_rank(xl[:, :task.body_len])
    ll = BindTask.stream_labels(lite, xl)
    want = torch.where(ll >= 0, rank.gather(1, ll.clamp(min=0)), ll)
    firsts = True
    for b in range(0, B, 11):
        seen = []
        for v in lab[b, :task.body_len].tolist():
            if v not in seen:
                seen.append(v)
        firsts &= seen == list(range(S))
    rows.append((f"{task.key}: the stream index is lite's stream ranked by first appearance ({torch.equal(lab, want)}); "
                 f"the indices first appear in the order 0..{S - 1} ({firsts}); position 0 is a CTX of stream 0 "
                 f"({bool(task.ctx_mask(x[:, :1]).all() and (lab[:, 0] == 0).all())})",
                 torch.equal(lab, want) and firsts and bool(task.ctx_mask(x[:, :1]).all() and (lab[:, 0] == 0).all())))
    if S == 2:
        cur_first = (x.gather(1, torch.where(task.ctx_mask(x), torch.arange(x.shape[1]).expand_as(x),
                                             torch.full_like(x, -1)).cummax(1).values.clamp(min=0)) != x[:, :1]).long()
        rows.append((f"{task.key}: at S = 2 the index is -allpool's convention (the latched CTX differs from position 0's)",
                     torch.equal(lab[:, 1:], cur_first[:, 1:])))
    body = x[:, :task.qpos[0] - 1]
    kidx = task.key_index(body)
    ok_both = ok_val = True
    for b in range(0, B, 3):
        for k in range(P):
            pos = (kidx[b] == k).nonzero()[:, 0]
            ok_both &= len(pos) == S and sorted(lab[b, pos].tolist()) == list(range(S))
            vals = x[b, pos + 1].tolist()
            ok_val &= len(set(vals)) == S and all(v >= task.val0 for v in vals)
    rows.append((f"{task.key}: every key occurs exactly {S} times, once under each stream ({ok_both}); its {S} values are "
                 f"distinct ({ok_val})", ok_both and ok_val))
    qp = task.qpos[0]
    qk, qst = x[:, qp], lab[:, qp]
    hit = (body == qk[:, None]) & (lab[:, :qp - 1] == qst[:, None])
    one = bool((hit.sum(1) == 1).all())
    kp = torch.where(hit, torch.arange(qp - 1).expand_as(hit), torch.full_like(hit, -1, dtype=torch.long)).max(1).values
    tgt_ok = bool(torch.equal(x.gather(1, (kp + 1)[:, None])[:, 0], x[:, qp + 1]))
    q_ok = bool(torch.equal(qidx.reshape(-1), qst))
    rows.append((f"{task.key}: the query's (stream, key) occurs exactly once in the body ({one}), its target is that pair's "
                 f"value ({tgt_ok}); make_batch's query stream equals stream_labels at the query ({q_ok})",
                 one and tgt_ok and q_ok))
    pg = task.perfect_gate(x[:, :-1])
    rows.append((f"{task.key}: the perfect gate is the one-hot of the index, k = {S}",
                 pg.shape[-1] == S and bool(torch.equal(pg.argmax(-1), lab[:, :-1].clamp(min=0)))
                 and bool(((pg == 0) | (pg == 1)).all())))
    fr = freq_report(task)
    br = fr["by_role"]
    rows.append((f"{task.key}: frequencies: CTX {br['CTX']['n']} types {br['CTX']['min']:.4f}-{br['CTX']['max']:.4f}, KEY "
                 f"{br['KEY']['n']} {br['KEY']['min']:.4f}-{br['KEY']['max']:.4f}, VAL {br['VAL']['n']} "
                 f"{br['VAL']['min']:.4f}-{br['VAL']['max']:.4f}; no CTX type as frequent as a value type (max CTX < min "
                 f"VAL)", br["CTX"]["max"] < br["VAL"]["min"]))
    cap, best, cnt, tot = type_map_cap(task.n_ctx, S, S)
    sizes = [balanced_map(task.n_ctx, S).count(c) for c in range(S)]
    bal = _esym(sizes, S) / tot
    cm = torch.tensor(balanced_map(task.n_ctx, S))
    isc = task.ctx_mask(x)
    emp = torch.tensor([len(set(cm[x[b][isc[b]]].tolist())) == S for b in range(B)]).double().mean().item()
    rows.append((f"{task.key}: the cap of a fixed type -> channel map at k = {S}: {cnt}/{tot} = {cap:.4f} (best split "
                 f"{list(best)}; the balanced map's {sizes}: {bal:.4f}); the balanced map separates {emp:.4f} of {B} "
                 f"sequences", abs(bal - cap) < 1e-12 and abs(emp - cap) < 4 * (cap * (1 - cap) / B) ** 0.5 + 1e-9))
    return rows


def check_inv_stats_k(task, seed=3060):
    """[(name, ok)]: synthetic gates at any S (module comment above), on probe_batch(task, seed)."""
    from test_multilayer_binding import probe_batch
    probe = probe_batch(task, seed)
    pinp = probe[0]
    B, S = pinp.shape[0], task.S
    perm = torch.rand(B, S, generator=torch.Generator().manual_seed(seed + 1)).argsort(-1)       # a channel order per seq
    rel = lambda x: task.perfect_gate(x)
    permuted = lambda x: task.perfect_gate(x).gather(-1, perm[:x.shape[0], None, :].expand(-1, x.shape[1], -1))
    cm = balanced_map(task.n_ctx, S)
    typ = lambda x: type_level_gate(task, x, cm)
    rows = []
    for nm, fn in (("relative", rel), ("permuted per sequence", permuted)):
        iv = inv_from_gate_k(task, pinp, fn(pinp))
        ok = abs(iv["sep_key_inv"] - 1) < 1e-9 and abs(iv["seq_separated_frac"] - 1) < 1e-12 and abs(iv["val_cos_inv_k"]) < 1e-6
        rows.append((f"{task.key}: synthetic gate '{nm}': sep_key_inv {iv['sep_key_inv']:.6f} (1), separated "
                     f"{iv['seq_separated_frac']:.4f} (1), VAL cos {iv['val_cos_inv_k']:.6f} (0)", ok))
    iv = inv_from_gate_k(task, pinp, typ(pinp))
    cmt = torch.tensor(cm)
    isc = task.ctx_mask(pinp)
    sep = torch.tensor([len(set(cmt[pinp[b][isc[b]]].tolist())) == S for b in range(B)]).double().mean().item()
    rows.append((f"{task.key}: the balanced type-level gate (k = {S}) separates {sep:.4f} of the probe's {B} sequences: "
                 f"seq_separated_frac {iv['seq_separated_frac']:.4f} equals it; sep_key_inv {iv['sep_key_inv']:.4f}, VAL cos "
                 f"{iv['val_cos_inv_k']:.4f}", abs(iv["seq_separated_frac"] - sep) < 1e-12))
    if S == 2:
        g = torch.rand(B, pinp.shape[1], 2, generator=torch.Generator().manual_seed(seed + 2)).softmax(-1)
        a, b = inv_from_gate(task, pinp, g), inv_from_gate_k(task, pinp, g)
        d = abs(a["val_cos_inv"] - b["val_cos_inv_k"])
        rows.append((f"{task.key}: at S = k = 2, val_cos_inv_k equals inv_from_gate's val_cos_inv on a random gate "
                     f"(|diff| {d:.1e})", d < 1e-12))
    return rows


def check_write_order_idx(task, B=2048, seed=11):
    """[(name, ok)]: the corrected write-order mask equals the one computed on lite's own batch (the same draws)."""
    S, P = task.S, task.P
    lite = RandHeaderTask(P, S=S, n_vals=task.n_vals, n_q=task.n_q, nb=task.nb, lmin=task.lmin, lmax=task.lmax)
    x = task.make_batch(B, torch.Generator().manual_seed(seed))[0][:, :-1]
    xl = lite.make_batch(B, torch.Generator().manual_seed(seed))[0][:, :-1]
    qp = task.qpos[0]
    mine, _ = last_write_mask(task, x, task.stream_labels(x)[:, qp - 1])
    ref, _ = last_write_mask(lite, xl, xl[:, qp - 1])
    return [(f"{task.key}: write_order_acc_idx's last-write mask equals the one on lite's own batch ({int(mine.sum())} of "
             f"{B} queries ask the last write)", bool(torch.equal(mine, ref)))]
