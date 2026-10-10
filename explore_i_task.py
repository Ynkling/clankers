#!/usr/bin/env python
"""
explore_i_task.py — EXPLORATORY, not a result. Session I's task: ICMC, in-context Markov chains with sources. A new task
class (not a BindTask subclass) keeping the harness conventions: a fixed length L; make_batch(B, gen) -> (tokens (B, L),
meta); stream_labels(x) giving every position its source; a loss mask. The model reads x = tokens[:, :-1] (T = L - 1
positions) and predicts tokens[:, 1:].

THE TASK (Bietti et al. 2023's in-context bigrams and Edelman et al. 2024's in-context Markov chains, with several sources
interleaved in blocks; defined fully here).
  Vocabulary  S source tokens SRC_s = s (s = 0..S-1), then V word tokens S + w (w = 0..V-1); vocab = S + V.
  Tables      per sequence, each source s draws its own table T_s: for every word w a set T_s[w] of N_SUCC = 3 successors
              drawn uniformly without replacement from the V words (w itself may be among them); the next word is
              uniform over the 3. Different sources' tables are independent; tables are fresh per sequence (nothing to
              memorise in the weights: the sets must be learned in context).
  Layout      NB blocks per source, each [SRC_s, w_1, ..., w_n], n in {LMIN..LMAX}; the S·NB blocks in a uniformly random
              order (a uniformly random permutation of the block ids s·NB + i; a block's source is id // NB; a source's
              blocks need not alternate with other sources'). A block continues its source's chain from the last word of
              that source's previous block (previous in sequence order); the first word of a source's first block is
              uniform on the V words.
  Fixed L     every source has exactly M words; its NB block lengths (in the order its blocks appear) are one composition
              of M into NB parts, each in LMIN..LMAX, drawn uniformly per (sequence, source) from all such compositions —
              explore_f_tasks.RandHeaderTask's rule (that file is copied unchanged from claude/explore-G, which copied it
              from claude/explore-F at 9a6c2bf; its compositions() enumerates them). Enumeration is too large here (e.g. 88
              into 8 parts), so the composition is drawn part by part with exact counts (DP), which is the same uniform
              distribution over the same list (CHECKed against compositions() where enumerable). L = S·(NB + M).
  Loss mask   cross-entropy at every input position t whose target tokens[t + 1] is a word with a known same-source
              predecessor: masked (excluded) are targets that are SRC tokens (the prediction OF each SRC token: switches
              are unpredictable) and the first word of each source's first block. Per sequence S·M − S positions count.
  Annotation  annotate(tokens) gives, per input position t: src (the target's source = the latest SRC at <= t + 1), prev
              (the target's same-source predecessor word, -1 if none), bpos (the target word's index within its block,
              1..n; 0 for SRC targets) and mask. The true successor set at t is T_src[prev].
  SET accuracy the fraction of masked positions whose predicted top-3 set (the 3 largest logits among the V WORD tokens;
              SRC tokens are never successors, so they are excluded from the ranking) equals T_src[prev] as a set.
              Position bins (the target word's index within its block): 1, 2, 3-4, 5+. Bin 1 is the target predicted AT
              the SRC token (its predecessor is the last word of the source's previous block, further back).

CHECKS (check_task, asserted by every screen before any run): every masked target is in its source's set T_src[prev];
each source's chain, read across its blocks in order, has exactly M words, every word after the first in T_s of the one
before (the continuation across blocks included); the mask excludes exactly the SRC targets and each source's first word
(S·M − S counted per sequence); every block starts with exactly one SRC token, NB blocks per source; every source's block
lengths are a composition of M with parts in LMIN..LMAX, and the DP sampler's part-length frequencies match the exact
marginals within max(0.02, 4 binomial sd) over 4096 sequences (where compositions() can enumerate the list, <= 200, every
composition's frequency is within 4 sd of uniform); more than one block order
occurs and the first block's source is uniform within 0.03; stream_labels equals the latest SRC's source at every
position; SET accuracy of logits built from the true sets is 1.0 and of a random ranking ~ 1/C(V, 3); tables differ
between sequences; L, T and the vocabulary.

IDEAL LEARNERS (ideal_set_acc; descriptive, used by the pilot to choose constants before training anything): expected SET
accuracy of (a) PER-SOURCE: the successors of prev observed earlier in the same source's chain (exact when all 3 were seen,
else the observed ones plus a uniformly random completion), (b) MIXED: the same counts pooled over all sources, top 3 by
count with random tie-breaking, (c) each with a fading memory: an observation at distance d tokens weighs DECAY**d and
counts only if its summed weight is >= EPS (DECAY = 0.95, the model's; EPS = 0.05, about 58 tokens for one observation).
"""

import itertools
import math

import numpy as np
import torch

import explore_f_tasks as ft

N_SUCC = 3
BINS = ((1, 1), (2, 2), (3, 4), (5, 10 ** 9))
BIN_NAMES = ("1", "2", "3-4", "5+")


def _ways(M, NB, lmin, lmax):
    """w[j][m] = number of compositions of m into j parts, each in lmin..lmax (exact ints)."""
    w = [[0] * (M + 1) for _ in range(NB + 1)]
    w[0][0] = 1
    for j in range(1, NB + 1):
        for m in range(M + 1):
            w[j][m] = sum(w[j - 1][m - l] for l in range(lmin, min(lmax, m) + 1))
    return w


class ICMCTask:
    def __init__(self, S=2, V=16, NB=3, lmin=6, lmax=14, M=None, L=None):
        self.S, self.V, self.NB, self.lmin, self.lmax = int(S), int(V), int(NB), int(lmin), int(lmax)
        if M is None:
            assert L is not None and (L - S * NB) % S == 0, (L, S, NB)
            M = (L - S * NB) // S
        self.M = int(M)
        self.L = self.S * (self.NB + self.M)
        assert L is None or L == self.L, (L, self.L)
        self.T = self.L - 1
        self.vocab = self.S + self.V
        self.ctx_tokens = tuple(range(self.S))
        self.n_ctx = self.S
        self.ways = _ways(self.M, self.NB, self.lmin, self.lmax)
        assert self.ways[self.NB][self.M] > 0, f"no composition of M={M} into NB={NB} parts in {lmin}..{lmax}"
        self.layout = "icmc"
        self.key = f"icmcS{S}V{V}NB{NB}l{lmin}-{lmax}M{self.M}"
        self.label = (f"ICMC S={S} V={V} NB={NB} block words {lmin}..{lmax} M={self.M} words/source L={self.L}")

    # ── generation ─────────────────────────────────────────────────────────────
    def _lengths(self, u):
        """One composition of M into NB parts in lmin..lmax from NB uniforms u (exactly uniform over compositions)."""
        out, m = [], self.M
        for j in range(self.NB, 0, -1):
            tot = self.ways[j][m]
            r = u[self.NB - j] * tot
            acc = 0
            for l in range(self.lmin, min(self.lmax, m) + 1):
                acc += self.ways[j - 1][m - l]
                if r < acc:
                    break
            else:
                l = max(x for x in range(self.lmin, min(self.lmax, m) + 1) if self.ways[j - 1][m - x] > 0)
            out.append(l)
            m -= l
        assert m == 0
        return out

    def make_batch(self, B, gen):
        S, V, NB, M = self.S, self.V, self.NB, self.M
        tables = torch.rand(B, S, V, V, generator=gen).argsort(-1)[..., :N_SUCC]       # (B,S,V,3)
        ucomp = torch.rand(B, S, NB, generator=gen, dtype=torch.float64)
        border = torch.rand(B, S * NB, generator=gen).argsort(-1)                       # block order
        first = torch.randint(V, (B, S), generator=gen)
        choice = torch.randint(N_SUCC, (B, S, M), generator=gen)
        Tl, U, Bo, F0, C = tables.tolist(), ucomp.tolist(), border.tolist(), first.tolist(), choice.tolist()
        rows = []
        for b in range(B):
            words = []
            for s in range(S):
                w = [F0[b][s]]
                for i in range(1, M):
                    w.append(Tl[b][s][w[-1]][C[b][s][i]])
                words.append(w)
            lens = [self._lengths(U[b][s]) for s in range(S)]
            occ, pos, seq = [0] * S, [0] * S, []
            for blk in Bo[b]:
                s = blk // NB
                n = lens[s][occ[s]]
                occ[s] += 1
                seq.append(s)
                seq.extend(S + x for x in words[s][pos[s]:pos[s] + n])
                pos[s] += n
            rows.append(seq)
        tokens = torch.tensor(rows, dtype=torch.long)
        assert tokens.shape == (B, self.L)
        return tokens, dict(tables=tables)

    # ── labels, mask, annotation ───────────────────────────────────────────────
    def stream_labels(self, x):
        """The source of every position: the latest SRC token at or before it (positions before any SRC: 0; never
        occurs, every sequence starts with a SRC token)."""
        B, T = x.shape
        is_src = x < self.S
        idx = torch.arange(T).expand(B, T)
        last = torch.where(is_src, idx, torch.full_like(idx, -1)).cummax(1).values
        return x.gather(1, last.clamp(min=0)).clamp(max=self.S - 1) * (last >= 0)

    def annotate(self, tokens):
        """Per input position t (0..L-2), for the target tokens[:, t + 1]: src, prev (-1 if none), bpos, mask."""
        B, L = tokens.shape
        S = self.S
        lab = self.stream_labels(tokens)                         # source of every token position
        src = lab[:, 1:]
        prev = torch.full((B, L - 1), -1, dtype=torch.long)
        bpos = torch.zeros((B, L - 1), dtype=torch.long)
        last = torch.full((B, S), -1, dtype=torch.long)
        run = torch.zeros(B, dtype=torch.long)
        ar = torch.arange(B)
        for p in range(L):
            tok = tokens[:, p]
            word = tok >= S
            s = lab[:, p]
            if p >= 1:
                prev[:, p - 1] = torch.where(word, last[ar, s], torch.full_like(tok, -1))
            run = torch.where(word, run + 1, torch.zeros_like(run))
            if p >= 1:
                bpos[:, p - 1] = run
            last[ar, s] = torch.where(word, tok - S, last[ar, s])
        mask = (tokens[:, 1:] >= S) & (prev >= 0)
        return dict(src=src, prev=prev, bpos=bpos, mask=mask)

    def loss_mask(self, tokens):
        return self.annotate(tokens)["mask"]

    def true_sets(self, tokens, tables, ann=None):
        """(B, L-1, 3) the true successor set T_src[prev] at every input position (garbage where mask is False)."""
        ann = self.annotate(tokens) if ann is None else ann
        B = tokens.shape[0]
        bi = torch.arange(B)[:, None].expand_as(ann["src"])
        return tables[bi, ann["src"], ann["prev"].clamp(min=0)]

    def bin_of(self, bpos):
        out = torch.full_like(bpos, -1)
        for i, (lo, hi) in enumerate(BINS):
            out = torch.where((bpos >= lo) & (bpos <= hi), torch.full_like(bpos, i), out)
        return out


# ── SET accuracy ───────────────────────────────────────────────────────────────
def set_hits(task, logits, sets):
    """(B, T) bool: the top-3 of the word logits equals the true set (as a set). logits (B, T, vocab); sets (B, T, 3)."""
    S, V = task.S, task.V
    top = logits[..., S:S + V].topk(N_SUCC, dim=-1).indices                  # word ids
    a = torch.zeros(*top.shape[:-1], V, dtype=torch.bool)
    a.scatter_(-1, top, True)
    b = torch.zeros(*sets.shape[:-1], V, dtype=torch.bool)
    b.scatter_(-1, sets, True)
    return (a == b).all(-1)


# ── ideal learners (pilot design aid; numpy) ──────────────────────────────────
def ideal_set_acc(task, n=256, seed=7, decay=0.95, eps=0.05, rng_seed=0):
    """Expected SET accuracy of the ideal per-source and mixed learners, unlimited and fading; overall and by bin."""
    tokens, meta = task.make_batch(n, torch.Generator().manual_seed(seed))
    ann = task.annotate(tokens)
    tab = meta["tables"].numpy()
    tk, src, prev, bpos, mask = (tokens.numpy(), ann["src"].numpy(), ann["prev"].numpy(), ann["bpos"].numpy(),
                                 ann["mask"].numpy())
    S, V = task.S, task.V
    rng = np.random.default_rng(rng_seed)
    names = ("own", "mixed", "own_fade", "mixed_fade")
    tot = {k: np.zeros(len(BINS) + 1) for k in names}
    cnt = np.zeros(len(BINS) + 1)
    binof = task.bin_of(ann["bpos"]).numpy()

    def score(w, true):
        """Expected hit of 'top 3 by weight w (V,), random ties, random completion among zeros'."""
        pos = np.flatnonzero(w > 0)
        if len(pos) <= N_SUCC:
            k = len(pos)
            if not set(pos.tolist()) <= true:
                return 0.0
            return 1.0 / math.comb(V - k, N_SUCC - k)
        hits = 0
        for _ in range(8):
            jitter = w + rng.random(V) * 1e-9
            hits += set(np.argsort(-jitter)[:N_SUCC].tolist()) == true
        return hits / 8

    for b in range(n):
        obs = []                                   # (pos of target, source, pred word, succ word)
        for t in range(task.T):
            p = t + 1
            if mask[b, t]:
                s, w = src[b, t], prev[b, t]
                true = set(tab[b, s, w].tolist())
                ws = {k: np.zeros(V) for k in names}
                for (q, s2, w2, x2) in obs:
                    if w2 != w:
                        continue
                    f = decay ** (p - q)
                    ws["mixed"][x2] += 1
                    ws["mixed_fade"][x2] += f
                    if s2 == s:
                        ws["own"][x2] += 1
                        ws["own_fade"][x2] += f
                for k in ("own_fade", "mixed_fade"):
                    ws[k] = np.where(ws[k] >= eps, ws[k], 0.0)
                bi = binof[b, t]
                cnt[bi] += 1
                cnt[-1] += 1
                for k in names:
                    v = score(ws[k], true)
                    tot[k][bi] += v
                    tot[k][-1] += v
            if mask[b, t] or (tk[b, p] >= S and prev[b, t] >= 0):
                obs.append((p, src[b, t], prev[b, t], tk[b, p] - S))
    out = {k: dict(all=tot[k][-1] / cnt[-1], **{BIN_NAMES[i]: tot[k][i] / max(cnt[i], 1) for i in range(len(BINS))})
           for k in names}
    out["n_pos"] = dict(all=int(cnt[-1]), **{BIN_NAMES[i]: int(cnt[i]) for i in range(len(BINS))})
    return out


# ── CHECKs ────────────────────────────────────────────────────────────────────
def check_task(task, n=4096, seed=4242):
    S, V, NB, M = task.S, task.V, task.NB, task.M
    g = torch.Generator().manual_seed(seed)
    tokens, meta = task.make_batch(n, g)
    tab = meta["tables"]
    ok = {}
    ok["shape/vocab/T"] = (tokens.shape == (n, task.L) and int(tokens.max()) < task.vocab and int(tokens.min()) >= 0
                           and task.T == task.L - 1)
    ann = task.annotate(tokens)
    sets = task.true_sets(tokens, tab, ann)
    tgt = tokens[:, 1:] - S
    m = ann["mask"]
    ok["every masked target in T_src[prev]"] = bool((sets == tgt.unsqueeze(-1)).any(-1)[m].all())
    # chains read across blocks, independently of annotate
    chain_ok, len_ok, blk_ok, comp_ok = True, True, True, True
    lens_seen = []
    T_ = tab.tolist()
    tl = tokens.tolist()
    first_src = []
    orders = set()
    for b in range(n):
        seq = tl[b]
        assert seq[0] < S
        chains = {s: [] for s in range(S)}
        blens = {s: [] for s in range(S)}
        cur = None
        order = []
        for x in seq:
            if x < S:
                cur = x
                blens[x].append(0)
                order.append(x)
            else:
                chains[cur].append(x - S)
                blens[cur][-1] += 1
        first_src.append(order[0])
        orders.add(tuple(order))
        for s in range(S):
            c = chains[s]
            len_ok &= len(c) == M
            chain_ok &= all(c[i] in T_[b][s][c[i - 1]] for i in range(1, len(c)))
            blk_ok &= len(blens[s]) == NB
            comp_ok &= sum(blens[s]) == M and all(task.lmin <= l <= task.lmax for l in blens[s])
            lens_seen.append(blens[s])
    ok["chains: M words per source, each in T_s of the one before (across blocks)"] = chain_ok and len_ok
    ok["NB blocks per source, one SRC at each block's start"] = blk_ok and int((tokens < S).sum()) == n * S * NB
    ok["block lengths: compositions of M in lmin..lmax"] = comp_ok
    # composition marginals vs exact DP marginals
    w = task.ways
    marg_ok = True
    arr = np.array(lens_seen)
    for j in range(NB):
        # exact marginal of part j: sum over prefixes is messy; part 0 exact, and by symmetry every part has the same
        # marginal (compositions are exchangeable under permutation of parts)
        exact = {l: w[NB - 1][M - l] / w[NB][M] for l in range(task.lmin, task.lmax + 1) if M - l >= 0}
        emp = {l: float((arr[:, j] == l).mean()) for l in exact}
        marg_ok &= all(abs(emp[l] - exact[l]) <= max(0.02, 4 * math.sqrt(exact[l] * (1 - exact[l]) / len(arr)))
                       for l in exact)
    ok["DP sampler's part marginals within max(0.02, 4 sd) of the exact marginal"] = marg_ok
    if w[NB][M] <= 200:
        comps = ft.compositions(M, NB, task.lmin, task.lmax)
        freq = {c: 0 for c in comps}
        for x in lens_seen:
            freq[tuple(x)] += 1
        p0 = 1 / len(comps)
        ok["composition list = compositions(); each drawn with frequency within 4 sd of uniform"] = (
            len(comps) == w[NB][M] and all(abs(v / len(lens_seen) - p0) <= 4 * math.sqrt(p0 * (1 - p0) / len(lens_seen))
                                           for v in freq.values()))
    ok["block order: >1 order occurs; first block's source uniform within 0.03"] = (
        len(orders) > 1 and all(abs(first_src.count(s) / n - 1 / S) <= 0.03 for s in range(S)))
    # mask
    is_src_t = tokens[:, 1:] < S
    n_mask = m.sum(1)
    ok["mask: excludes SRC targets and each source's first word (S*M - S per sequence)"] = (
        bool((~m[is_src_t]).all()) and bool((n_mask == S * M - S).all()))
    # stream labels: independent latch
    lab = task.stream_labels(tokens)
    lab_ok = True
    for b in range(min(n, 256)):
        cur = None
        for p, x in enumerate(tl[b]):
            if x < S:
                cur = x
            lab_ok &= int(lab[b, p]) == cur
    ok["stream_labels = latest SRC's source at every position"] = lab_ok
    # SET accuracy of the true sets
    logits = torch.full((n, task.T, task.vocab), -1.0)
    logits[..., S:].scatter_(-1, sets, 1.0)
    logits[..., :S] = 5.0                         # SRC logits are excluded from the ranking
    h = set_hits(task, logits, sets)
    ok["SET accuracy of the true sets = 1.0"] = bool(h[m].all())
    rnd = set_hits(task, torch.randn(n, task.T, task.vocab, generator=torch.Generator().manual_seed(1)), sets)
    racc = float(rnd[m].float().mean())
    chance = 1 / math.comb(V, N_SUCC)
    ok[f"SET accuracy of a random ranking ~ 1/C(V,3) = {chance:.4f} (got {racc:.4f})"] = abs(racc - chance) < max(
        0.01, 3 * math.sqrt(chance / max(int(m.sum()), 1)) + 0.002)
    ok["tables fresh per sequence"] = not torch.equal(tab[0], tab[1])
    ok["bpos: 1..n at words, 0 at SRC targets"] = bool((ann["bpos"][~is_src_t] >= 1).all()) and bool(
        (ann["bpos"][is_src_t] == 0).all())
    return ok


if __name__ == "__main__":
    import sys
    for t in (ICMCTask(S=2, V=8, NB=3, lmin=2, lmax=5, M=10), ICMCTask(S=4, V=6, NB=4, lmin=6, lmax=14, M=40)):
        r = check_task(t, n=1024)
        print(t.label)
        for k, v in r.items():
            print(f"  {'ok ' if v else 'FAIL'} {k}")
        assert all(r.values())
