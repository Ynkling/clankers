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
