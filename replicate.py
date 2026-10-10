"""Independent replication, EXPLORATORY, not a result.

Re-implementation, from the Revision 8 report (multichannel_hebbian_report_v8.pdf) and
Pathway's public bdh.py only, of the eight-stream test of Section 5 (Table 2, Parts C/D):
WIN3_SLOW_D8 (window gate + SLOW, no split) against WIN3_SPLIT_D8 (window gate + SLOW + SPLIT),
with the perfect gate (ORACLE) as the validity arm. Every detail the report does not determine
is logged in UNSTATED.md.

PRE-REGISTERED DESIGN AND READINGS (fixed and committed before any Phase 2 run)
------------------------------------------------------------------------------
Configuration: S=8 streams, P=4 keys, 16 value tokens, grouped layout, one query; BDH with
N=256, D=32, one head, three applications of the shared layer, decay 0.95, the width-4 "conv",
k=16 channels, batch 32, budget 43,200 updates, evaluation every 1200 updates on 2048
held-out queries, early stop after three consecutive evaluations >= 0.95.

Arms and seeds:
  ORACLE         perfect gate (one-hot on the stream), seeds 900-901.  Validity: both must bind.
  WIN3_SLOW_D8   window gate + SLOW,          seeds 900-919.
  WIN3_SPLIT_D8  window gate + SLOW + SPLIT,  seeds 900-919 (paired with WIN3_SLOW_D8 by seed).

Primary outcome: "bound routed" (bound; one-to-one stream-to-channel map at value positions;
every stream's accuracy >= 0.9).

Readings (WIN3_SPLIT_D8 bound-routed count x/20; McNemar on the 20 paired seeds,
b = seeds where SPLIT is bound routed and SLOW is not, c = the reverse,
p = exact one-sided P[Binomial(b+c, 1/2) >= b]):
  "replicated"            if x >= 18 AND p < 0.05
  "partially replicated"  if x >= 12 (and "replicated" does not apply)
  "not replicated"        otherwise
If the ORACLE validity arm fails (either seed unbound) the reading is reported as
"INVALID (oracle did not bind)" next to the mechanical reading.
Every count is reported with its Wilson 95% interval.

RESULT (filled in after the run; see records/phase2.jsonl, records/phase2_table.md, analyze.py)
------------------------------------------------------------------------------
Run 2026-10-10, one container, 4 workers x 1 thread, PyTorch 2.14.1 (CPU), 37.4 h of training.
  ORACLE         bound 2/2 (transitions 10,800 and 6000)            -> validity PASS
  WIN3_SPLIT_D8  bound routed 17/20, Wilson 95% [0.640, 0.948]; median transition 20,400
  WIN3_SLOW_D8   bound routed  7/20, Wilson 95% [0.181, 0.567]; median transition 22,800
  McNemar SPLIT vs SLOW: 10 vs 0, exact one-sided p = 0.00098
  45 splits in 20 runs (1-3 per run); 44 had >= 2 streams on c* and none on c0 (labels, diagnostic).
  Every one of the 16 failures (3 SPLIT, 13 SLOW) is a merge of 2 or 3 streams on one channel.
READING (mechanical): "partially replicated" (17 < 18; McNemar p < 0.05).
Independent replication, EXPLORATORY, not a result.
"""

import dataclasses
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------- task

N_VALUES = 16
ROLE_CTX, ROLE_KEY, ROLE_VAL, ROLE_QMARK = 0, 1, 2, 3


@dataclasses.dataclass(frozen=True)
class Task:
    S: int = 8
    P: int = 4

    @property
    def vocab(self):
        # values 0..15, keys, context tokens, '?'
        return N_VALUES + self.P + self.S + 1

    def key_tok(self, i):
        return N_VALUES + i

    def ctx_tok(self, s):
        return N_VALUES + self.P + s

    @property
    def qmark(self):
        return N_VALUES + self.P + self.S

    @property
    def T(self):
        return 3 * self.S * self.P + 3

    @property
    def query_key_pos(self):
        return self.T - 2  # prediction is made at the query's KEY position

    def sample(self, B, gen):
        """Grouped layout. Returns dict with idx (B,T), target (B,), stream (B,T),
        role (B,T), qstream (B,)."""
        S, P = self.S, self.P
        # for each key, S distinct values out of 16
        vals = torch.rand(B, P, N_VALUES, generator=gen).argsort(-1)[..., :S]  # B,P,S
        key_order = torch.rand(B, P, generator=gen).argsort(-1)  # B,P
        stream_order = torch.rand(B, P, S, generator=gen).argsort(-1)  # B,P,S
        qs = torch.randint(0, S, (B,), generator=gen)
        qk = torch.randint(0, P, (B,), generator=gen)

        keys = key_order.unsqueeze(-1).expand(B, P, S)  # key id of each triple
        streams = stream_order  # stream id of the j-th triple in each key group
        tv = torch.gather(vals, 1, key_order.unsqueeze(-1).expand(B, P, S))  # vals of the key, per stream
        tv = torch.gather(tv, 2, streams)  # value of (key, stream) for each triple
        ctx = N_VALUES + P + streams
        key = N_VALUES + keys
        body = torch.stack([ctx, key, tv], -1).reshape(B, 3 * S * P)
        q = torch.stack([N_VALUES + P + qs, N_VALUES + qk, torch.full_like(qs, self.qmark)], -1)
        idx = torch.cat([body, q], 1)
        target = vals[torch.arange(B), qk, qs]

        stream = torch.cat([streams.reshape(B, S * P).repeat_interleave(3, 1), qs.unsqueeze(1).expand(B, 3)], 1)
        role = torch.tensor([ROLE_CTX, ROLE_KEY, ROLE_VAL] * (S * P) + [ROLE_CTX, ROLE_KEY, ROLE_QMARK])
        role = role.unsqueeze(0).expand(B, -1)
        return dict(idx=idx, target=target, stream=stream, role=role, qstream=qs)


    def sample_theirs(self, B, gen):
        """PHASE 3 ONLY: the main line's BindTask.make_batch (grouped, n_q=1), draw for draw, re-expressed in this
        file's vocabulary (their CTX s, KEY i, VAL v -> ctx_tok(s), key_tok(i), v; the query's VAL becomes '?')."""
        S, P = self.S, self.P
        vals = torch.rand(B, P, N_VALUES, generator=gen).argsort(-1)[..., :S]
        korder = torch.rand(B, P, generator=gen).argsort(-1)
        sorder = torch.rand(B, P, S, generator=gen).argsort(-1)
        vg = vals.gather(1, korder.unsqueeze(-1).expand(B, P, S)).gather(2, sorder)
        q = torch.rand(B, S * P, generator=gen).argsort(-1)[:, :1]
        qs, qk = (q // P).squeeze(1), (q % P).squeeze(1)
        body = torch.stack([N_VALUES + P + sorder, (N_VALUES + korder).unsqueeze(-1).expand(B, P, S), vg], -1)
        body = body.reshape(B, 3 * S * P)
        qrow = torch.stack([N_VALUES + P + qs, N_VALUES + qk, torch.full_like(qs, self.qmark)], -1)
        idx = torch.cat([body, qrow], 1)
        target = vals.reshape(B, P * S).gather(1, (qk * S + qs).unsqueeze(1)).squeeze(1)
        stream = torch.cat([sorder.reshape(B, S * P).repeat_interleave(3, 1), qs.unsqueeze(1).expand(B, 3)], 1)
        role = torch.tensor([ROLE_CTX, ROLE_KEY, ROLE_VAL] * (S * P) + [ROLE_CTX, ROLE_KEY, ROLE_QMARK])
        return dict(idx=idx, target=target, stream=stream, role=role.unsqueeze(0).expand(B, -1), qstream=qs)


# ----------------------------------------------------------------------------- model


class WindowGate(nn.Module):
    """g_t = softmax(W_g tanh(W_in sum_{j<3} w_j * v_{t-j})); eq. (3)."""

    def __init__(self, D, k, H=32, width=3):
        super().__init__()
        self.width = width
        # depthwise causal conv, no bias; PyTorch's default init is U(+-1/sqrt(width))
        self.conv = nn.Conv1d(D, D, width, groups=D, bias=False)
        self.W_in = nn.Parameter(torch.randn(H, D) * 0.1)
        self.W_g = nn.Parameter(torch.randn(k, H) * 0.1)

    def logits(self, v):  # v: B,T,D raw embeddings
        u = F.pad(v.transpose(1, 2), (self.width - 1, 0))
        u = self.conv(u).transpose(1, 2)
        h = torch.tanh(u @ self.W_in.T)
        return h @ self.W_g.T

    def forward(self, v):
        return torch.softmax(self.logits(v), -1)


class CausalConv(nn.Module):
    """The report's "conv": depthwise causal filter of width 4, identity init, feeding Q(=K) and V."""

    def __init__(self, D, width=4):
        super().__init__()
        self.width = width
        w = torch.zeros(D, 1, width)
        w[:, 0, -1] = 1.0  # last tap = current token -> identity
        self.weight = nn.Parameter(w)

    def forward(self, x):  # B,T,D
        u = F.pad(x.transpose(1, 2), (self.width - 1, 0))
        return F.conv1d(u, self.weight, groups=x.shape[-1]).transpose(1, 2)


class MCBDH(nn.Module):
    """Pathway BDH (shared layer, Q=K, LN after attention) with k memory channels:
    score(t,s) = (x_t . x_s) (g_t . g_s) decay^(t-s), s<t."""

    def __init__(self, vocab, D=32, N=256, n_layer=3, k=16, decay=0.95, conv=True, gate="window"):
        super().__init__()
        self.D, self.N, self.n_layer, self.k, self.decay = D, N, n_layer, k, decay
        self.embed = nn.Embedding(vocab, D)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.encoder = nn.Parameter(torch.randn(D, N) * 0.02)
        self.encoder_v = nn.Parameter(torch.randn(D, N) * 0.02)
        self.decoder = nn.Parameter(torch.randn(N, D) * 0.02)
        self.lm_head = nn.Parameter(torch.randn(D, vocab) * 0.02)
        self.conv = CausalConv(D) if conv else None
        self.gate_kind = gate
        self.gate = WindowGate(D, k) if gate == "window" else None

    @staticmethod
    def ln(x):
        return F.layer_norm(x, x.shape[-1:])

    def gate_params(self):
        return list(self.gate.parameters()) if self.gate is not None else []

    def forward(self, idx, stream=None, return_gate=False):
        B, T = idx.shape
        v = self.embed(idx)
        g = self.gate(v) if self.gate_kind == "window" else F.one_hot(stream, self.k).float()
        t = torch.arange(T)
        diff = (t.view(-1, 1) - t.view(1, -1)).float()
        dec = torch.where(diff > 0, self.decay ** diff.clamp(min=0), torch.zeros(()))
        M = (g @ g.transpose(1, 2)) * dec  # B,T,T  context-gate match x decay, strictly causal
        x = self.ln(v)
        for _ in range(self.n_layer):
            xc = self.conv(x) if self.conv is not None else x
            xs = F.relu(xc @ self.encoder)  # B,T,N
            scores = (xs @ xs.transpose(1, 2)) * M
            yKV = self.ln(scores @ xc)
            ys = F.relu(yKV @ self.encoder_v)
            y = self.ln((xs * ys) @ self.decoder)
            x = self.ln(x + y)
        logits = x @ self.lm_head
        return (logits, g) if return_gate else logits


# ----------------------------------------------------------------------------- outcomes

BIND_THR = 0.95


def bound_and_transition(evals, budget, every=1200):
    """evals: list of (update, acc) in order. Bound: an evaluation reaches 0.95 and every later
    one stays there; the transition is the first such step. A run that reaches its budget is
    bound on a single final evaluation >= 0.95."""
    if not evals or evals[-1][1] < BIND_THR:
        return False, None
    i = len(evals) - 1
    while i > 0 and evals[i - 1][1] >= BIND_THR:
        i -= 1
    trailing = len(evals) - i
    if trailing >= 3 or evals[-1][0] >= budget:
        return True, evals[i][0]
    return False, None


def stream_channel_map(g, stream, role, S):
    """Map stream -> argmax channel of the mean gate at value positions."""
    m = role == ROLE_VAL
    out = []
    for s in range(S):
        sel = m & (stream == s)
        out.append(int(g[sel].mean(0).argmax()))
    return out


def bound_routed(bound, cmap, per_stream_acc, thr=0.9):
    one_to_one = len(set(cmap)) == len(cmap)
    return bool(bound and one_to_one and min(per_stream_acc) >= thr)


def failure_class(bound, cmap, final_acc, per_stream_acc):
    if final_acc < 0.15:
        return "collapsed"
    counts = {}
    for c in cmap:
        counts[c] = counts.get(c, 0) + 1
    j = max(counts.values())
    if j > 1:
        return f"merged({j} share)"
    if not bound:
        return "unbound-routed"
    if min(per_stream_acc) < 0.9:
        return "stream-acc<0.9"
    return "bound-routed"


# ----------------------------------------------------------------------------- split


class Split:
    """SPLIT (KEYMASS). Checks every `every` updates from `first`; fires if probe acc < 0.95 and
    rose by < 0.02 since the previous check; at most `max_splits`, at least `min_gap` apart.
    The probe is also read at `first - every` so that the first check has a previous value."""

    def __init__(self, first=4800, every=2400, max_splits=3, min_gap=4800, noise=0.1):
        self.first, self.every, self.max_splits, self.min_gap, self.noise = first, every, max_splits, min_gap, noise
        self.prev_acc = None
        self.splits = []  # dicts

    def is_probe_step(self, u):
        return u >= self.first - self.every and (u - self.first) % self.every == 0

    def decide(self, u, acc):
        """Record the probe accuracy at update u; return True if a split should fire now."""
        prev, self.prev_acc = self.prev_acc, acc
        if u < self.first or prev is None:
            return False
        if len(self.splits) >= self.max_splits:
            return False
        if self.splits and u - self.splits[-1]["update"] < self.min_gap:
            return False
        return acc < BIND_THR and (acc - prev) < 0.02

    @staticmethod
    def targets(key_mass):
        """key_mass: (k,) mean read-gate mass at key positions."""
        return int(key_mass.argmax()), int(key_mass.argmin())

    def apply(self, W_g, opt, c_star, c0, gen):
        with torch.no_grad():
            row = W_g[c_star].clone()
            sd = row.std()
            W_g[c0] = row + self.noise * sd * torch.randn(row.shape, generator=gen)
            W_g[c_star] = row + self.noise * sd * torch.randn(row.shape, generator=gen)
        st = opt.state.get(W_g)
        if st:
            for key, val in st.items():
                if torch.is_tensor(val):
                    val.zero_()


# ----------------------------------------------------------------------------- training

ARMS = {
    # name: (gate, split)
    "ORACLE": ("perfect", False),
    "WIN3_SLOW_D8": ("window", False),
    "WIN3_SPLIT_D8": ("window", True),
    # Phase 3 reruns: WIN3_SPLIT_D8 with ONE discrepancy (DIFF.md) set to the main line's choice
    "SPLIT_R1_PROBE": ("window", True),     # D-1: the probe is the main line's, generator 12345, same for every seed
    "SPLIT_R2_BODYKEYS": ("window", True),  # D-2: key mass over the 32 body key positions only (query key excluded)
}

EVAL_SEED = 123_457
PROBE_SEED_OFFSET = 1_000_003
NOISE_SEED_OFFSET = 2_000_003


def make_eval_set(task, n=2048, seed=EVAL_SEED):
    g = torch.Generator().manual_seed(seed)
    return task.sample(n, g)


@torch.no_grad()
def evaluate(model, data, S, chunk=512):
    model.eval()
    preds, gs = [], []
    n = data["idx"].shape[0]
    for i in range(0, n, chunk):
        logits, g = model(data["idx"][i:i + chunk], data["stream"][i:i + chunk], return_gate=True)
        preds.append(logits[:, -2].argmax(-1))
        gs.append(g)
    model.train()
    pred = torch.cat(preds)
    g = torch.cat(gs)
    correct = (pred == data["target"]).float()
    per_stream = [float(correct[data["qstream"] == s].mean()) for s in range(S)]
    return float(correct.mean()), per_stream, g


def run(arm, seed, S=8, P=4, budget=43_200, batch=32, eval_every=1200, n_eval=2048, k=16,
        slow_until=2400, lr=1e-3, slow_lr=1e-4, split_kw=None, log=None, stop_after=None, ckpt=None):
    torch.set_num_threads(1)
    gate, use_split = ARMS[arm]
    task = Task(S=S, P=P)
    torch.manual_seed(seed)
    model = MCBDH(task.vocab, k=k, gate=gate)
    gate_ids = {id(p) for p in model.gate_params()}
    others = [p for p in model.parameters() if id(p) not in gate_ids]
    groups = [{"params": others, "lr": slow_lr, "name": "memory"}]
    if model.gate_params():
        groups.append({"params": model.gate_params(), "lr": lr, "name": "gate"})
    opt = torch.optim.Adam(groups)
    data_gen = torch.Generator().manual_seed(seed)
    probe = task.sample(64, torch.Generator().manual_seed(seed + PROBE_SEED_OFFSET))
    if arm == "SPLIT_R1_PROBE":
        probe = task.sample_theirs(64, torch.Generator().manual_seed(12345))
    key_role = (probe["role"] == ROLE_KEY)
    if arm == "SPLIT_R2_BODYKEYS":
        key_role = key_role.clone()
        key_role[:, -3:] = False
    noise_gen = torch.Generator().manual_seed(seed + NOISE_SEED_OFFSET)
    evalset = make_eval_set(task, n_eval)
    split = Split(**(split_kw or {})) if use_split else None

    evals, probes = [], []
    t0 = time.time()
    u = 0
    streak = 0
    start, prior_wall = 1, 0.0
    if ckpt and os.path.exists(ckpt):  # resume bit-for-bit after a container restart
        st = torch.load(ckpt, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        data_gen.set_state(st["data_gen"])
        noise_gen.set_state(st["noise_gen"])
        evals, probes, streak, prior_wall = st["evals"], st["probes"], st["streak"], st["wall"]
        if split is not None:
            split.prev_acc, split.splits = st["split_prev"], st["splits"]
        start = st["u"] + 1
    for u in range(start, budget + 1):
        opt.param_groups[0]["lr"] = slow_lr if u <= slow_until else lr
        b = task.sample(batch, data_gen)
        logits = model(b["idx"], b["stream"])
        loss = F.cross_entropy(logits[:, task.query_key_pos], b["target"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if u % eval_every == 0 or u == budget:
            acc, _, _ = evaluate(model, evalset, S)
            evals.append((u, acc))
            streak = streak + 1 if acc >= BIND_THR else 0
            if log:
                log(f"{arm} seed={seed} u={u} acc={acc:.3f} loss={loss.item():.3f}")
            if streak >= 3:
                break
            save_ckpt = True
        else:
            save_ckpt = False
        if split is not None and split.is_probe_step(u):
            pacc, _, pg = evaluate(model, probe, S)
            probes.append((u, pacc))
            if split.decide(u, pacc):
                key_mass = pg[key_role].mean(0)
                c_star, c0 = Split.targets(key_mass)
                # diagnostics only (labels are not used by the rule): streams on each channel
                cmap_now = stream_channel_map(pg, probe["stream"], probe["role"], S)
                split.apply(model.gate.W_g, opt, c_star, c0, noise_gen)
                split.splits.append(dict(update=u, c_star=c_star, c0=c0, probe_acc=pacc,
                                         streams_on_c_star=cmap_now.count(c_star),
                                         streams_on_c0=cmap_now.count(c0)))
                if log:
                    log(f"{arm} seed={seed} SPLIT u={u} c*={c_star} c0={c0} probe={pacc:.3f}")
        if ckpt and save_ckpt:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), data_gen=data_gen.get_state(),
                            noise_gen=noise_gen.get_state(), evals=evals, probes=probes, streak=streak,
                            wall=prior_wall + time.time() - t0, u=u,
                            split_prev=split.prev_acc if split else None,
                            splits=split.splits if split else []), ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
        if stop_after is not None and u >= stop_after:
            break
    wall = prior_wall + time.time() - t0
    final_acc, per_stream, g = evaluate(model, evalset, S)
    bound, transition = bound_and_transition(evals, budget, eval_every)
    cmap = stream_channel_map(g, evalset["stream"], evalset["role"], S)
    br = bound_routed(bound, cmap, per_stream)
    return dict(arm=arm, seed=seed, S=S, P=P, k=k, budget=budget, updates=u, bound=bound,
                transition=transition, bound_routed=br, map=cmap, final_acc=final_acc,
                per_stream_acc=per_stream, failure=failure_class(bound, cmap, final_acc, per_stream),
                splits=split.splits if split else [], evals=evals, probes=probes, wall_s=wall,
                torch=torch.__version__)


CKPT_DIR = os.environ.get("REPLICATE_CKPT_DIR", "ckpt")


def _job(args):
    arm, seed = args
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"{arm}_{seed}.pt")
    r = run(arm, seed, ckpt=path)
    return r, path


def main():
    import argparse
    import multiprocessing as mp

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="records/phase2.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--phase", type=int, default=2)
    a = ap.parse_args()
    if a.phase == 2:
        jobs = [("ORACLE", 900), ("ORACLE", 901)]
        for s in range(900, 920):
            jobs += [("WIN3_SPLIT_D8", s), ("WIN3_SLOW_D8", s)]
    else:  # Phase 3: seeds 900-909, one discrepancy fixed at a time
        jobs = [(arm, s) for arm in ("SPLIT_R1_PROBE", "SPLIT_R2_BODYKEYS") for s in range(900, 910)]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["arm"], r["seed"]))
    jobs = [j for j in jobs if j not in done]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    print(f"{len(jobs)} jobs to run", flush=True)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for r, path in pool.imap_unordered(_job, jobs):
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
            if os.path.exists(path):
                os.remove(path)
            print(f"{r['arm']} {r['seed']} bound={r['bound']} br={r['bound_routed']} "
                  f"T={r['transition']} acc={r['final_acc']:.3f} splits={len(r['splits'])} "
                  f"wall={r['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
