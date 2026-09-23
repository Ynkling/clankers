#!/usr/bin/env python
"""
test_binding_capacity.py — do multiple channels beat a single-channel recurrent readout
once the task actually requires associative memory?

Run it directly:

    python test_binding_capacity.py                 # all configs, 5 seeds
    python test_binding_capacity.py --seeds 3
    python test_binding_capacity.py --force         # ignore cache, retrain
    python test_binding_capacity.py --threads 4

WHY. test_capacity_control.py found arm E (one channel + recurrent-gate readout) ties the
k=2 model. The root cause is the task: test_instrument_v2.make_seq draws ONE flip bit per
sequence, every key in a stream shares that stream's value, and the answer depends only
on (flip, query stream), never on the key. The task stores one bit, so a 32-unit RNN
readout solves it without associative memory. This test replaces it with a task that
stores many independent, conflicting bindings, so memory capacity and interference both
matter.

NEW TASK. Streams s = 0..S-1 with context tokens CTX_s. For each key i, S DISTINCT values
are drawn from n_vals value tokens, so every key conflicts across streams. The body is
S*P triples [CTX_s, KEY_i, VAL_{i,s}], grouped per key, keys in random order, stream
order randomized within each group (generalizing the randomized block order). It ends
with n_q queries [CTX_q, KEY_q, VAL_{q_key,q}], whose (stream, key) pairs are drawn
WITHOUT replacement — otherwise an earlier query would reveal a later query's answer.
Loss and accuracy are on every query's value position. Vocabulary: CTX_s = s,
KEY_i = S + i, VAL_j = S + P + j.

DISCIPLINE. The old task path is untouched: make_seq, make_batch and run() are unchanged.
Three knobs were added to test_instrument_v2.Instrument — vocab, h_gate, ctx_tokens —
none of which creates a parameter; at their defaults every shape and every random draw is
as before. perfect_gate_general() was added for the S-context perfect gate. Where run()
cannot express the new task, run_bind() below mirrors it line for line, parameterized by
a task object; lines that differ are marked [bind]. The CHECKs prove all of it:
  CHECK 0  the pre-change file (git LEGACY_SHA) and the current one give bitwise-identical
           parameters and a bitwise-identical seed-0 headline run.
  CHECK 1  every new knob is inert at its default (implicit vs explicit defaults).
  CHECK 2  run_bind(old task) reproduces run() to the last bit on seed 0.
  CHECK 3  perfect_gate_general equals _perfect_gate exactly at S=2.
  CHECK 4  the new task is built as specified.

ARMS, at every config, k = S channels where applicable:
  floor     uniform gate, k=S              ceiling   perfect gate, k=S
  B         plain single channel (sanity)
  A         recurrent gate, k=S, no readout
  A_ro      recurrent gate, k=S, + gate_to_readout
  E         1 channel + gate_to_readout                 <- the baseline to beat
  E_mem     1 channel, n_feat = S*N (A's memory), + gate_to_readout
  E_wide    1 channel + gate_to_readout, h_gate widened to match A_ro's parameters

CONFIGS. C0 = the OLD task via make_batch — a sanity reproduction of
test_capacity_control, where E and A should both sit at the ceiling. Then the new task
with S=2, n_vals=16, n_q=4, P in {4, 8, 16, 32}. Decay 1.0 throughout: the 2x2 showed U
does not decide the outcome, and decay 0.95 would erase early bindings at these lengths;
decay^T is printed per config to show it. 5 seeds per config.

EXPRESSIBILITY FIRST, per config, before any arm is trusted:
  * the ceiling (perfect gate) must reach >= CEIL_PASS on every seed. ITERS climbs the
    ladder 1200 -> 2400 -> 4800, equal for all arms. A rung is abandoned at its first
    failing seed, which decides pass/fail exactly as testing every seed would, at a
    fraction of the cost. A config whose ceiling fails at every rung is outside the
    architecture's reach: it is EXCLUDED from the verdict, said so, and its arms still
    run at the top rung (4800) so the diagnostics are not a training-budget artifact.
  * the recurrent gate must fit the perfect gate's routing (CHECK 5 analog,
    argmax match >= FIT_PASS).
  * the floor is measured per seed, and any seed more than FLOOR_FLAG above 1/S is
    flagged. A floor far BELOW 1/S is flagged too: it means the model cannot even narrow
    a query to its key's S candidates.
Information content (P*S*log2(n_vals) bits vs the old task's 1 bit) is printed next to
the RNN width so the design intent is visible.

PRE-REGISTERED VERDICT, EPS = 0.10, printed before any interpretation. Per valid config:
best channel arm = max(A, A_ro), best control = max(E, E_mem, E_wide), by mean accuracy.
CHANNELS WIN at that config only if (best channel - best control) >= EPS, the channel arm
beats that control on every seed, AND the channel arm separates (VAL gate cosine between
streams < SEP_COS on every seed). Overall: SUPPORTED if any valid config is a win,
NOT SUPPORTED otherwise. C0 is the task this test replaces; it is evaluated and printed
like any config, and since E ties A there it cannot produce a win.

DIAGNOSTICS, printed separately and not part of the verdict: accuracy vs P for every arm;
the P at which E drops below 0.9; whether E_wide moves that point; a flag if E is below
the ceiling even at P=4.

LOGISTICS. Every arm is timed on every config before the sweep, and the projected wall
clock is printed. If it exceeds WALL_LIMIT_H, seeds 0-2 run across all configs first, an
interim report is printed, then the remaining seeds run. Reporting follows
test_phase2_seeds.py: per-seed raw values before any aggregate, population std,
"identical: X" instead of "+- 0.000", failures (exception or non-finite) recorded per
seed and never dropped, PARTIAL labels on incomplete sweeps. Results persist atomically
to binding_capacity_results.json after every run.

(Results are recorded at the bottom of this docstring after the run.)
"""

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F

import test_instrument_v2 as tiv
from test_instrument_v2 import (
    Instrument, OPERATING_DECAY, ITERS, BATCH, LR, D, N, k, V, H_GATE, PROBE_B, N_CKPT,
    TAU_START, TAU_END, SEP_COS, BLOCK, CTX1, CTX2, perfect_gate_general,
)
from test_interference import _perfect_gate

# ── Settings ─────────────────────────────────────────────────────────────────
EPS = 0.10                        # PRE-REGISTERED win margin
N_SEEDS = 5
S_STREAMS = 2
N_VALS = 16
N_Q = 4
P_LIST = [4, 8, 16, 32]
DECAY_BIND = 1.0
ITERS_LADDER = (1200, 2400, 4800)
CEIL_PASS = 0.95
FIT_PASS = 0.99
FIT_ITERS = 400
FLOOR_FLAG = 0.05
E_DROP = 0.90
WALL_LIMIT_H = 3.0
THREADS = 1
RESULTS_FILE = "binding_capacity_results.json"
LEGACY_SHA = "df5f036"            # head before this test's changes to test_instrument_v2
TIME_ITERS = (20, 60)             # timing: per-iter cost from the difference of two runs


# ── Tasks ────────────────────────────────────────────────────────────────────
class OldTask:
    """test_instrument_v2's task, unchanged. select() is run()'s own expression."""
    key, sanity = "C0", True
    S, P, n_vals, n_q = 2, 4, 2, 1
    vocab, ctx_tokens = None, None
    T = BLOCK - 1
    info_bits = 1.0
    label = "C0: OLD task via make_batch (sanity)"

    def make_batch(self, B, gen):
        return tiv.make_batch(B, gen)

    def stream_labels(self, x):
        return tiv.stream_labels(x)

    def role_masks(self, x):
        return tiv.role_masks(x)

    def select(self, logits, tgt, meta):
        return logits[:, -1, :], tgt[:, -1], meta


class BindTask:
    sanity = False

    def __init__(self, P, S=S_STREAMS, n_vals=N_VALS, n_q=N_Q):
        assert n_q <= S * P and S <= n_vals
        self.S, self.P, self.n_vals, self.n_q = S, P, n_vals, n_q
        self.key = f"P{P}"
        self.label = f"P={P}: new task, S={S}, n_vals={n_vals}, n_q={n_q}"
        self.vocab = S + P + n_vals
        self.ctx_tokens = tuple(range(S))
        self.L = 3 * (S * P + n_q)
        self.T = self.L - 1
        self.qpos = [3 * (S * P + j) + 1 for j in range(n_q)]
        self.info_bits = P * S * math.log2(n_vals)

    def make_batch(self, B, gen):
        S, P, nv, nq = self.S, self.P, self.n_vals, self.n_q
        vals = torch.rand(B, P, nv, generator=gen).argsort(-1)[..., :S]    # (B,P,S) distinct
        korder = torch.rand(B, P, generator=gen).argsort(-1)                # key order
        sorder = torch.rand(B, P, S, generator=gen).argsort(-1)             # stream order
        vg = vals.gather(1, korder.unsqueeze(-1).expand(B, P, S)).gather(2, sorder)
        keys = (S + korder).unsqueeze(-1).expand(B, P, S)
        body = torch.stack([sorder, keys, S + P + vg], dim=-1).reshape(B, -1)
        q = torch.rand(B, S * P, generator=gen).argsort(-1)[:, :nq]         # no replacement
        qs, qk = q // P, q % P
        qv = vals.reshape(B, P * S).gather(1, qk * S + qs)
        queries = torch.stack([qs, S + qk, S + P + qv], dim=-1).reshape(B, -1)
        return torch.cat([body, queries], dim=1), qs

    def stream_labels(self, x):
        B, T = x.shape
        idx = torch.arange(T).expand(B, T)
        cid = torch.where(x < self.S, x, torch.full_like(x, -1))
        last = torch.where(cid >= 0, idx, torch.full_like(idx, -1)).cummax(dim=1).values
        lab = cid.gather(1, last.clamp(min=0))
        return torch.where(last >= 0, lab, torch.full_like(lab, -1))

    def role_masks(self, x):
        S, P = self.S, self.P
        return {'ctx': x < S, 'key': (x >= S) & (x < S + P), 'val': x >= S + P}

    def select(self, logits, tgt, meta):
        ql = logits[:, self.qpos, :].reshape(-1, logits.shape[-1])
        qt = tgt[:, self.qpos].reshape(-1)
        return ql, qt, (None if meta is None else meta.reshape(-1))


# ── run_bind: test_instrument_v2.run() line for line; differences marked [bind] ─
def run_bind(task, gate, mode, n_ch, decay, C, seed, iters, anneal=False, ckpts=None,
             n_feat=None, gate_to_readout=False, h_gate=None):
    torch.manual_seed(seed)
    model = Instrument(gate, mode, n_ch, decay, C,
                       n_feat=n_feat, gate_to_readout=gate_to_readout,
                       vocab=task.vocab, h_gate=h_gate,                     # [bind]
                       ctx_tokens=task.ctx_tokens)                          # [bind]
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = torch.Generator(); rng.manual_seed(seed + 10_000)
    prng = torch.Generator(); prng.manual_seed(seed + 99_000)
    ptok, _ = task.make_batch(PROBE_B, prng)                                # [bind]
    pinp, plab = ptok[:, :-1], task.stream_labels(ptok[:, :-1])              # [bind]
    if ckpts is None:
        ckpts = sorted({int(i * (iters - 1) / (N_CKPT - 1)) for i in range(N_CKPT)})  # [bind]
    else:
        ckpts = sorted(set(ckpts))
    trace = []

    prole = task.role_masks(pinp)                                           # [bind]

    def diag(step, tau):
        with torch.no_grad():
            _, _, gr, gw = model(pinp, tau)
            d = {'step': step}
            for nm, g in (('r', gr), ('w', gw)):
                g0, g1 = g[plab == 0].mean(0), g[plab == 1].mean(0)
                d[f'{nm}_s1'], d[f'{nm}_s2'] = g0.tolist(), g1.tolist()
                d[f'{nm}_cos'] = F.cosine_similarity(g0.unsqueeze(0), g1.unsqueeze(0)).item()
            for rn, rm in prole.items():
                a = gr[(plab == 0) & rm].mean(0)
                b = gr[(plab == 1) & rm].mean(0)
                d[f'{rn}_cos'] = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
                d[f'{rn}_s1'], d[f'{rn}_s2'] = a.tolist(), b.tolist()
            return d

    for step in range(iters):                                               # [bind]
        tau = (TAU_START * (TAU_END / TAU_START) ** (step / (iters - 1))) if anneal else TAU_END
        if step in ckpts and n_ch > 1:
            trace.append(diag(step, tau))
        tokens, _ = task.make_batch(BATCH, rng)                             # [bind]
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        logits, _, _, _ = model(inp, tau)
        ql, qt, _ = task.select(logits, tgt, None)                          # [bind]
        loss = F.cross_entropy(ql, qt)                                      # [bind]
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    if n_ch > 1:
        trace.append(diag(iters, TAU_END))                                  # [bind]
    with torch.no_grad():
        tokens, qs = task.make_batch(512, rng)                              # [bind]
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        logits, _, _, _ = model(inp, TAU_END)
        ql, qt, qs = task.select(logits, tgt, qs)                           # [bind]
        corr = (ql.argmax(-1) == qt)
        m1, m2 = qs == 0, qs == 1
        out = dict(l1=F.cross_entropy(ql[m1], qt[m1]).item(),
                   l2=F.cross_entropy(ql[m2], qt[m2]).item(),
                   a1=corr[m1].float().mean().item(),
                   a2=corr[m2].float().mean().item(), trace=trace, sat=[0.0] * n_ch)
        if C is not None:
            _, sat, _, _ = model(pinp[:64], TAU_END, track_sat=True)
            out['sat'] = sat.tolist()
    return out


# ── Arms ─────────────────────────────────────────────────────────────────────
def build(task, arm, seed=0):
    torch.manual_seed(seed)
    return Instrument(arm["gate"], "sym", arm["n_ch"], DECAY_BIND, None,
                      n_feat=arm["n_feat"], gate_to_readout=arm["g2r"],
                      vocab=task.vocab, h_gate=arm["h_gate"], ctx_tokens=task.ctx_tokens)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def n_state(m):
    return m.n_ch * m.Dx.shape[1] * D


def arms_for(task):
    S = task.S
    base = [
        dict(key="floor", label="floor   uniform gate, k=S", gate="uniform", n_ch=S),
        dict(key="ceiling", label="ceiling perfect gate, k=S", gate="perfect", n_ch=S),
        dict(key="B", label="B       plain single channel", gate="none", n_ch=1),
        dict(key="A", label="A       recurrent gate, k=S", gate="recurrent", n_ch=S),
        dict(key="A_ro", label="A_ro    recurrent k=S + readout", gate="recurrent", n_ch=S,
             g2r=True),
        dict(key="E", label="E       1ch + readout (to beat)", gate="recurrent", n_ch=1,
             g2r=True),
        dict(key="E_mem", label="E_mem   1ch, n_feat=S*N + readout", gate="recurrent",
             n_ch=1, g2r=True, n_feat=S * N),
        dict(key="E_wide", label="E_wide  1ch + readout, wide h_gate", gate="recurrent",
             n_ch=1, g2r=True),
    ]
    for a in base:
        a.setdefault("g2r", False)
        a.setdefault("n_feat", None)
        a.setdefault("h_gate", None)
    by = {a["key"]: a for a in base}
    target = n_params(build(task, by["A_ro"]))
    best = None
    for hg in range(H_GATE + 1, 4 * H_GATE + 1):       # "widened": strictly above H_GATE
        a = dict(by["E_wide"], h_gate=hg)
        res = n_params(build(task, a)) - target
        if best is None or abs(res) < abs(best[1]):
            best = (hg, res)
    by["E_wide"]["h_gate"] = best[0]
    return base


# ── Persistence ──────────────────────────────────────────────────────────────
def load_results(path):
    if not os.path.exists(path):
        return {"meta": {}, "runs": {}, "express": {}}
    try:
        with open(path) as f:
            d = json.load(f)
        for kk in ("meta", "runs", "express"):
            d.setdefault(kk, {})
        return d
    except Exception as e:
        print(f"  ! could not read {path} ({e}); starting fresh")
        return {"meta": {}, "runs": {}, "express": {}}


def save_results(path, store):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(store, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def rkey(cfg, arm, iters):
    return f"{cfg}|{arm}|{iters}"


def get_run(store, cfg, arm, iters, seed):
    return store["runs"].get(rkey(cfg, arm, iters), {}).get(str(seed))


def do_run(store, path, task, arm, iters, seed, force, t0, tag=""):
    """Run one (config, arm, iters, seed) unless cached; persist; never drop a failure."""
    rk = rkey(task.key, arm["key"], iters)
    store["runs"].setdefault(rk, {})
    cached = store["runs"][rk].get(str(seed))
    if cached is not None and not force:
        print(f"   {task.key:>3} {arm['key']:>7} @{iters:<4} seed {seed}  cached "
              f"({'ok' if cached.get('ok') else 'FAILED'}){tag}")
        return cached
    ts = time.time()
    try:
        r = run_bind(task, arm["gate"], "sym", arm["n_ch"], DECAY_BIND, None, seed, iters,
                     n_feat=arm["n_feat"], gate_to_readout=arm["g2r"], h_gate=arm["h_gate"])
        vals = [r["l1"], r["l2"], r["a1"], r["a2"]]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            raise ValueError(f"non-finite result: l1={r['l1']} l2={r['l2']} "
                             f"a1={r['a1']} a2={r['a2']}")
        last = r["trace"][-1] if r["trace"] else {}
        rec = dict(ok=True, seed=seed, iters=iters, l1=r["l1"], l2=r["l2"], a1=r["a1"],
                   a2=r["a2"], acc=(r["a1"] + r["a2"]) / 2.0,
                   key_cos=last.get("key_cos"), val_cos=last.get("val_cos"),
                   ctx_cos=last.get("ctx_cos"), secs=time.time() - ts)
    except Exception as e:
        rec = dict(ok=False, seed=seed, iters=iters, error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc()[-1500:], secs=time.time() - ts)
    store["runs"][rk][str(seed)] = rec
    save_results(path, store)
    el = (time.time() - t0) / 60
    if rec["ok"]:
        vc = "" if rec["val_cos"] is None else f"  VALcos={rec['val_cos']:.4f}"
        print(f"   {task.key:>3} {arm['key']:>7} @{iters:<4} seed {seed}  "
              f"acc={rec['acc']:.4f}{vc}   {rec['secs']:.0f}s  (elapsed {el:.1f}m){tag}")
    else:
        print(f"   {task.key:>3} {arm['key']:>7} @{iters:<4} seed {seed}  *** FAILED: "
              f"{rec['error']}   (elapsed {el:.1f}m){tag}")
    return rec


# ── Statistics ───────────────────────────────────────────────────────────────
def stats(xs):
    """mean, POPULATION std (as used throughout this series), min, max, n."""
    xs = [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)
          and math.isfinite(x)]
    if not xs:
        return None
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    return dict(mean=m, std=sd, min=min(xs), max=max(xs), n=len(xs),
                identical=(max(xs) - min(xs) < 1e-12))


def cell(st, prec=4, signed=False):
    if st is None:
        return "no data"
    sg = "+" if signed else ""
    if st["identical"]:
        return f"identical: {st['mean']:{sg}.{prec}f}"
    return f"{st['mean']:{sg}.{prec}f} +- {st['std']:.{prec}f}"


# ── Verification ─────────────────────────────────────────────────────────────
def load_legacy():
    """test_instrument_v2.py as of LEGACY_SHA, imported under a separate module name."""
    src = subprocess.run(["git", "-C", HERE, "show", f"{LEGACY_SHA}:test_instrument_v2.py"],
                         capture_output=True, text=True, check=True).stdout
    d = tempfile.mkdtemp(prefix="tiv_legacy_")
    p = os.path.join(d, f"_tiv_legacy_{LEGACY_SHA}.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(f"_tiv_legacy_{LEGACY_SHA}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify():
    print("=" * 100)
    print("VERIFICATION — printed before any training")
    print("=" * 100)
    ok = True
    gates = [("recurrent", 2), ("token", 2), ("uniform", 2), ("perfect", 2), ("none", 1)]

    # CHECK 0 — old path untouched: pre-change file vs current, same process, bitwise
    print(f"CHECK 0  THE OLD PATH IS UNTOUCHED. test_instrument_v2.py as of {LEGACY_SHA}")
    print("         (before this test) vs now, in one process, bitwise:")
    try:
        L = load_legacy()
        bad = 0
        for g, nc in gates:
            torch.manual_seed(5); mo = L.Instrument(g, "sym", nc, OPERATING_DECAY, None)
            torch.manual_seed(5); mn = Instrument(g, "sym", nc, OPERATING_DECAY, None)
            po, pn = dict(mo.named_parameters()), dict(mn.named_parameters())
            same = sorted(po) == sorted(pn) and all(torch.equal(po[n], pn[n]) for n in po)
            bad += not same
            print(f"         {g:>9} k={nc}: parameters bitwise equal = {same}")
        ro = L.run("recurrent", "sym", k, OPERATING_DECAY, None, 0)
        rn = tiv.run("recurrent", "sym", k, OPERATING_DECAY, None, 0)
        print("         seed-0 headline (recurrent k=2, decay 0.95, 1200 steps) re-run:")
        for f in ("l1", "l2", "a1", "a2"):
            same = ro[f] == rn[f]; bad += not same
            print(f"           {f:>8}: {ro[f]!r} vs {rn[f]!r}  "
                  f"{'IDENTICAL' if same else '*** DIFFERS ***'}")
        for f in ("key_cos", "val_cos", "ctx_cos", "r_cos"):
            a, b = ro["trace"][-1][f], rn["trace"][-1][f]
            same = a == b; bad += not same
            print(f"           {f:>8}: {a!r} vs {b!r}  "
                  f"{'IDENTICAL' if same else '*** DIFFERS ***'}")
        ok &= bad == 0
        print(f"         -> {'UNTOUCHED to the last bit' if bad == 0 else 'FAILED'}\n")
    except Exception as e:
        print(f"         SKIPPED: could not load {LEGACY_SHA} from git ({type(e).__name__}: "
              f"{e}).")
        print("         CHECKs 1 and 2 still prove the default path and run_bind.\n")

    # CHECK 1 — new knobs inert at their defaults
    print("CHECK 1  vocab / h_gate / ctx_tokens are INERT at their defaults (implicit vs")
    print("         explicit defaults, bitwise):")
    bad = 0
    for g, nc in gates:
        torch.manual_seed(5); m0 = Instrument(g, "sym", nc, OPERATING_DECAY, None)
        torch.manual_seed(5); m1 = Instrument(g, "sym", nc, OPERATING_DECAY, None,
                                              vocab=V, h_gate=H_GATE, ctx_tokens=None)
        p0, p1 = dict(m0.named_parameters()), dict(m1.named_parameters())
        same = sorted(p0) == sorted(p1) and all(torch.equal(p0[n], p1[n]) for n in p0)
        bad += not same
        print(f"         {g:>9} k={nc}: bitwise equal = {same}   ({n_params(m0)} params)")
    ok &= bad == 0
    print(f"         -> {'INERT' if bad == 0 else 'FAILED'}\n")

    # CHECK 2 — run_bind(old task) == run() to the last bit, seed 0
    print("CHECK 2  run_bind(old task) reproduces run() to the last bit, seed 0, 1200 steps:")
    old = OldTask()
    bad = 0
    for g, nc, g2r in (("recurrent", 2, False), ("recurrent", 1, True), ("perfect", 2, False)):
        a = tiv.run(g, "sym", nc, OPERATING_DECAY, None, 0, gate_to_readout=g2r)
        b = run_bind(old, g, "sym", nc, OPERATING_DECAY, None, 0, ITERS, gate_to_readout=g2r)
        fl = [(f, a[f], b[f]) for f in ("l1", "l2", "a1", "a2")]
        if a["trace"]:
            fl += [(f, a["trace"][-1][f], b["trace"][-1][f]) for f in ("key_cos", "val_cos")]
        same = all(x == y for _, x, y in fl)
        bad += not same
        print(f"         {g:>9} k={nc} readout={str(g2r):<5}: "
              + "  ".join(f"{f} {'==' if x == y else '!='}" for f, x, y in fl)
              + f"   -> {'IDENTICAL' if same else '*** DIFFERS ***'}")
        print(f"           e.g. l1: {a['l1']!r} vs {b['l1']!r}")
    ok &= bad == 0
    print(f"         -> {'run_bind MIRRORS run()' if bad == 0 else 'FAILED'}\n")

    # CHECK 3 — the generalized perfect gate equals _perfect_gate at S=2
    print("CHECK 3  perfect_gate_general == _perfect_gate exactly at S=2:")
    worst = 0.0
    for name, task in (("old task", OldTask()), ("new task P=8", BindTask(8))):
        g = torch.Generator().manual_seed(3)
        tok, _ = task.make_batch(64, g)
        x = tok[:, :-1]
        ref = torch.stack([_perfect_gate(x[b]) for b in range(x.shape[0])])
        gen = perfect_gate_general(x, (CTX1, CTX2))
        d = (gen - ref).abs().max().item()
        worst = max(worst, d)
        print(f"         {name:<13}: max |general - _perfect_gate| = {d:.2e}")
    ok &= worst == 0.0
    print(f"         -> {'IDENTICAL' if worst == 0.0 else 'FAILED'}\n")

    # CHECK 4 — the new task is built as specified
    print("CHECK 4  the new task is built as specified (P=8, 512 sequences):")
    t = BindTask(8)
    g = torch.Generator().manual_seed(4)
    tok, qs = t.make_batch(512, g)
    S, P = t.S, t.P
    body = tok[:, :3 * S * P].reshape(512, P, S, 3)
    same_key = (body[..., 1] == body[..., :1, 1]).all().item()
    vals = body[..., 2]
    distinct = (vals[..., 0] != vals[..., 1]).all().item()
    streams_ok = (body[..., 0].sort(-1).values == torch.arange(S)).all().item()
    s0_first = (body[:, :, 0, 0] == 0).float().mean().item()
    q = tok[:, 3 * S * P:].reshape(512, t.n_q, 3)
    bound = {}
    good_q = True
    for b in range(64):
        m = {(int(body[b, gi, j, 0]), int(body[b, gi, j, 1])): int(body[b, gi, j, 2])
             for gi in range(P) for j in range(S)}
        pairs = [(int(q[b, j, 0]), int(q[b, j, 1])) for j in range(t.n_q)]
        good_q &= len(set(pairs)) == t.n_q
        good_q &= all(m[p] == int(q[b, j, 2]) for j, p in enumerate(pairs))
    checks = [("each group repeats one key across its S triples", same_key),
              ("every key has S DISTINCT values (conflict)", distinct),
              ("every group contains each stream exactly once", streams_ok),
              ("queries are distinct (stream,key) pairs with the BOUND value", good_q),
              (f"stream order randomized: P(stream 0 first) = {s0_first:.3f}",
               abs(s0_first - 0.5) < 0.05)]
    for msg, c in checks:
        print(f"         {'ok ' if c else 'BAD'} {msg}")
        ok &= bool(c)
    ex = tok[0].tolist()
    dec = lambda x: (f"C{x}" if x < S else (f"K{x - S}" if x < S + P else f"v{x - S - P}"))
    print("         example: " + " ".join(dec(x) for x in ex[:12]) + " ... | queries: "
          + " ".join(dec(x) for x in ex[3 * S * P:]))
    print()

    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"


def gate_fit(task):
    """CHECK 5 analog: can the recurrent gate fit the perfect gate's routing?"""
    g = torch.Generator().manual_seed(11)
    tok, _ = task.make_batch(64, g)
    x = tok[:, :-1]
    tgt = (tiv.perfect_gate_batched(x) if task.ctx_tokens is None
           else perfect_gate_general(x, task.ctx_tokens))
    torch.manual_seed(3)
    m = Instrument("recurrent", "sym", task.S, DECAY_BIND, None, vocab=task.vocab,
                   ctx_tokens=task.ctx_tokens)
    opt = torch.optim.Adam([m.W_in, m.W_h, m.W_g, m.embed.weight], lr=1e-2)
    for _ in range(FIT_ITERS):
        gr, _ = m.gates(x, m.embed(x))
        loss = F.mse_loss(gr, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        gr, _ = m.gates(x, m.embed(x))
        return F.mse_loss(gr, tgt).item(), (gr.argmax(-1) == tgt.argmax(-1)).float().mean().item()


def time_arm(task, arm):
    """Per-iteration cost and fixed overhead from the difference of two short runs."""
    ts = []
    for it in TIME_ITERS:
        t0 = time.time()
        run_bind(task, arm["gate"], "sym", arm["n_ch"], DECAY_BIND, None, 0, it,
                 n_feat=arm["n_feat"], gate_to_readout=arm["g2r"], h_gate=arm["h_gate"])
        ts.append(time.time() - t0)
    per = max((ts[1] - ts[0]) / (TIME_ITERS[1] - TIME_ITERS[0]), 1e-6)
    return per, max(ts[0] - TIME_ITERS[0] * per, 0.0)


# ── Reporting ────────────────────────────────────────────────────────────────
def acc_list(store, cfg, arm, iters, seeds):
    out = []
    for s in seeds:
        r = get_run(store, cfg, arm, iters, s)
        out.append(r["acc"] if (r and r.get("ok")) else None)
    return out


def report(store, tasks, seeds, final_iters, express, info, wall, path, partial_note=""):
    title = "INTERIM REPORT" if partial_note else "FINAL REPORT"
    print("#" * 100)
    print(f"{title}{partial_note}")
    print("#" * 100)
    print()
    arm_keys = [a["key"] for a in arms_for(tasks[0])]

    for task in tasks:
        cfg, it = task.key, final_iters[task.key]
        ex = express[task.key]
        print("=" * 100)
        print(f"{task.label}   ITERS={it}   "
              f"{'VALID' if ex['valid'] else 'EXCLUDED — ' + ex['reason']}")
        print("=" * 100)
        W = 10
        print("  PER-SEED RAW ACCURACY — every value, before any aggregate")
        print("  seed  " + "".join(f"{kk:>{W}}" for kk in arm_keys))
        for s in seeds:
            row = ""
            for kk in arm_keys:
                r = get_run(store, cfg, kk, it, s)
                row += f"{'--' if r is None else ('FAIL' if not r.get('ok') else format(r['acc'], '.4f')):>{W}}"
            print(f"  {s:>4}  " + row)
        print("  per-seed VAL gate cosine between streams (channel arms):")
        for kk in ("A", "A_ro"):
            vc = []
            for s in seeds:
                r = get_run(store, cfg, kk, it, s)
                vc.append(None if (r is None or not r.get("ok")) else r.get("val_cos"))
            print(f"    {kk:>5}: " + "  ".join("--" if v is None else f"{v:.4f}" for v in vc))
        floor = acc_list(store, cfg, "floor", it, seeds)
        chance = 1.0 / task.S
        flags = []
        for s, f in zip(seeds, floor):
            if f is None:
                continue
            if f > chance + FLOOR_FLAG:
                flags.append(f"seed {s}: floor {f:.4f} RISES > {chance:.3f}+{FLOOR_FLAG}")
            elif f < chance - FLOOR_FLAG:
                flags.append(f"seed {s}: floor {f:.4f} is BELOW 1/S={chance:.3f}")
        print(f"  floor vs 1/S = {chance:.3f}: "
              + ("no flags" if not flags else "")
              )
        for fl in flags:
            print(f"    FLAG {fl}")
        print()
        print(f"  {'arm':<34} {'params':>7} {'state':>6} {'accuracy mean +- std':>24} "
              f"{'min':>7} {'max':>7} {'n':>3}")
        for a in arms_for(task):
            kk = a["key"]
            xs = acc_list(store, cfg, kk, it, seeds)
            nf = sum(1 for s in seeds if get_run(store, cfg, kk, it, s) is not None
                     and not get_run(store, cfg, kk, it, s).get("ok"))
            st = stats([x for x in xs if x is not None])
            p, sf = info[cfg][kk]
            if st is None:
                print(f"  {a['label']:<34} {p:>7} {sf:>6} {'no completed seeds':>24}"
                      + (f"   [{nf} FAILED]" if nf else ""))
            else:
                print(f"  {a['label']:<34} {p:>7} {sf:>6} {cell(st):>24} "
                      f"{st['min']:>7.4f} {st['max']:>7.4f} {st['n']:>3}"
                      + (f"   [{nf} FAILED]" if nf else ""))
        print()

    # ── Verdict ─────────────────────────────────────────────────────────────
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"  PRE-REGISTERED CRITERION (set before any result was seen), EPS = {EPS}:")
    print("    Per valid config: best channel arm = max(A, A_ro), best control =")
    print("    max(E, E_mem, E_wide), by mean accuracy. CHANNELS WIN at that config only if")
    print(f"    (best channel - best control) >= EPS, the channel arm beats that control on")
    print(f"    EVERY seed, AND the channel arm separates (VAL cosine < {SEP_COS} on every")
    print("    seed). Overall: SUPPORTED if any valid config is a win, NOT SUPPORTED otherwise.")
    print()
    attempted = completed = 0
    for task in tasks:
        for kk in arm_keys:
            for s in seeds:
                attempted += 1
                r = get_run(store, task.key, kk, final_iters[task.key], s)
                completed += bool(r and r.get("ok"))
    print(f"  {completed}/{attempted} runs completed"
          + ("   *** PARTIAL SWEEP ***" if completed < attempted or partial_note else ""))
    if completed < attempted:
        for task in tasks:
            for kk in arm_keys:
                for s in seeds:
                    r = get_run(store, task.key, kk, final_iters[task.key], s)
                    if r is None:
                        print(f"    {task.key} {kk} seed {s}: NOT RUN")
                    elif not r.get("ok"):
                        print(f"    {task.key} {kk} seed {s}: FAILED — {r['error']}")
    print()
    wins, valid = [], []
    for task in tasks:
        cfg, it, ex = task.key, final_iters[task.key], express[task.key]
        if not ex["valid"]:
            print(f"  {cfg:>3}: EXCLUDED — {ex['reason']}")
            continue
        valid.append(cfg)
        m = lambda kk: stats([x for x in acc_list(store, cfg, kk, it, seeds) if x is not None])
        ch = max(("A", "A_ro"), key=lambda kk: (m(kk) or {"mean": -1})["mean"])
        ct = max(("E", "E_mem", "E_wide"), key=lambda kk: (m(kk) or {"mean": -1})["mean"])
        mc, mt = m(ch), m(ct)
        if mc is None or mt is None:
            print(f"  {cfg:>3}: no verdict (incomplete)")
            continue
        margin = mc["mean"] - mt["mean"]
        per = [(a, b) for a, b in zip(acc_list(store, cfg, ch, it, seeds),
                                      acc_list(store, cfg, ct, it, seeds))
               if a is not None and b is not None]
        every = bool(per) and all(a > b for a, b in per)
        vcs = [get_run(store, cfg, ch, it, s) for s in seeds]
        vcs = [r.get("val_cos") for r in vcs if r and r.get("ok")]
        sep = bool(vcs) and all(v is not None and v < SEP_COS for v in vcs)
        win = margin >= EPS and every and sep
        wins += [cfg] if win else []
        tag = " (sanity: the task this test replaces)" if task.sanity else ""
        print(f"  {cfg:>3}{tag}")
        print(f"       best channel {ch:<5} {mc['mean']:.4f}   best control {ct:<6} "
              f"{mt['mean']:.4f}   margin {margin:+.4f} "
              f"({'>=' if margin >= EPS else '<'} EPS)")
        print(f"       beats control on every seed: {every}   "
              f"separates (VAL cos < {SEP_COS} every seed): {sep}   -> "
              f"{'CHANNELS WIN' if win else 'no win'}")
    print()
    new_valid = [c for c in valid if c != "C0"]
    print(f"  valid configs: {valid if valid else 'NONE'}   "
          f"(new-task configs valid: {new_valid if new_valid else 'NONE'})")
    if wins:
        print(f"  *** SUPPORTED: channels win at {wins}. ***")
    else:
        print("  *** NOT SUPPORTED: no valid config is a channel win. ***")
        if not new_valid:
            print("  Every new-task config failed the expressibility gate, so the question")
            print("  this test was built to ask — do channels beat a single-channel readout")
            print("  once the task requires memory? — was not tested on this architecture.")

    # ── Diagnostics ─────────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    bind = [t for t in tasks if not t.sanity]
    print("  mean accuracy vs P (new task; EXCLUDED configs marked *):")
    print(f"  {'arm':>8}" + "".join(f"{('P=' + str(t.P) + ('*' if not express[t.key]['valid'] else '')):>11}"
                                    for t in bind))
    means = {}
    for kk in arm_keys:
        row = []
        for t in bind:
            st = stats([x for x in acc_list(store, t.key, kk, final_iters[t.key], seeds)
                        if x is not None])
            row.append(None if st is None else st["mean"])
        means[kk] = row
        print(f"  {kk:>8}" + "".join(f"{('--' if v is None else format(v, '.4f')):>11}"
                                     for v in row))
    print(f"  {'1/S':>8}" + "".join(f"{1 / t.S:>11.4f}" for t in bind))
    print(f"  {'1/(S*P)':>8}" + "".join(f"{1 / (t.S * t.P):>11.4f}" for t in bind))
    print()

    def drop(kk):
        for t, v in zip(bind, means[kk]):
            if v is not None and v < E_DROP:
                return t.P
        return None
    de, dw = drop("E"), drop("E_wide")
    print(f"  E drops below {E_DROP} at P = {de if de is not None else 'never (within P_LIST)'}")
    print(f"  E_wide drops below {E_DROP} at P = {dw if dw is not None else 'never (within P_LIST)'}")
    if de is not None and dw is not None:
        moved = dw > de
        print(f"  E_wide {'MOVES' if moved else 'does NOT move'} the drop point to larger P.")
    hg_e, hg_w = H_GATE, arms_for(bind[0])[-1]["h_gate"]
    print(f"  NOTE: E_wide is only {hg_w - hg_e} hidden unit(s) wider than E (h_gate {hg_w} vs "
          f"{hg_e}), because A_ro's")
    print("  parameter advantage over E is a single extra W_g row. A parameter-matched")
    print("  widening therefore CANNOT test an RNN-capacity bottleneck; 'does not move' here")
    print("  is not evidence against one.")
    if bind and means["E"][0] is not None and means["ceiling"][0] is not None:
        e4, c4 = means["E"][0], means["ceiling"][0]
        if e4 < c4 - 0.05 or e4 < E_DROP:
            print(f"  FLAG: E is {e4:.4f} at P={bind[0].P}, below the ceiling ({c4:.4f}) and")
            print(f"  below {E_DROP}, already at the smallest config. At ITERS="
                  f"{final_iters[bind[0].key]} this could be a training-budget artifact")
            print("  rather than a capacity limit.")
    print()
    print(f"  total wall clock this invocation: {wall / 60:.1f} min ({wall:.0f}s)")
    print(f"  raw records: {path}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Binding-capacity test for multi-channel BDH.")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--results", default=RESULTS_FILE)
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.threads))
    seeds = list(range(args.seeds))
    tasks = [OldTask()] + [BindTask(P) for P in P_LIST]

    print("=" * 100)
    print("Binding capacity: do channels beat a single-channel recurrent readout once the")
    print("task requires memory?   instrument imported from test_instrument_v2")
    print(f"  S={S_STREAMS} n_vals={N_VALS} n_q={N_Q} P={P_LIST} decay={DECAY_BIND} "
          f"EPS={EPS} ladder={ITERS_LADDER} seeds={seeds} threads={args.threads}")
    print("=" * 100)
    print()
    verify()

    store = load_results(args.results)
    store["meta"] = dict(eps=EPS, seeds=seeds, s=S_STREAMS, n_vals=N_VALS, n_q=N_Q,
                         p_list=P_LIST, decay=DECAY_BIND, ladder=list(ITERS_LADDER),
                         started=time.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()

    # ── Per-config design and resource accounting ───────────────────────────
    info = {}
    print("=" * 100)
    print("CONFIGS — information content vs RNN width, sequence length, decay^T")
    print("=" * 100)
    print(f"  {'config':<48} {'bits':>6} {'RNN':>4} {'T':>4} {'1.0^T':>6} {'0.95^T':>9}")
    for t in tasks:
        print(f"  {t.label:<48} {t.info_bits:>6.0f} {H_GATE:>4} {t.T:>4} "
              f"{DECAY_BIND ** t.T:>6.3f} {0.95 ** t.T:>9.2e}")
    print("  (bits = P*S*log2(n_vals) stored bindings; the old task stores 1 bit. 0.95^T is")
    print("   what decay 0.95 would leave of the earliest binding — why decay is 1.0 here.)")
    print()

    print("=" * 100)
    print("RESOURCES — trainable params and fast-weight state floats per arm, per config")
    print("=" * 100)
    for t in tasks:
        info[t.key] = {}
        arms = arms_for(t)
        print(f"  {t.label}")
        for a in arms:
            m = build(t, a)
            info[t.key][a["key"]] = (n_params(m), n_state(m))
            print(f"    {a['label']:<34} n_ch={a['n_ch']} n_feat={m.n_feat:<4} "
                  f"h_gate={m.h_gate if a['gate'] == 'recurrent' else '-':<4} "
                  f"params={info[t.key][a['key']][0]:>6}  state={info[t.key][a['key']][1]:>5}")
        pa = info[t.key]["A_ro"][0]
        pw = info[t.key]["E_wide"][0]
        sa = info[t.key]["A"][1]
        sm = info[t.key]["E_mem"][1]
        print(f"    E_mem memory vs A:    {sm} vs {sa} floats -> residual "
              f"{(sm - sa) / sa:+.4%}")
        print(f"    E_wide params vs A_ro: {pw} vs {pa} -> residual {pw - pa:+d} "
              f"({(pw - pa) / pa:+.2%}); nearest match with h_gate strictly widened")
    print()

    # ── Timing ──────────────────────────────────────────────────────────────
    print("=" * 100)
    print(f"TIMING — each arm on each config, per-iteration cost from {TIME_ITERS} steps")
    print("=" * 100)
    timing = {}
    for t in tasks:
        timing[t.key] = {}
        row = []
        for a in arms_for(t):
            per, over = time_arm(t, a)
            timing[t.key][a["key"]] = (per, over)
            row.append(f"{a['key']}={per * 1000:.0f}ms")
        print(f"  {t.key:>3} (T={t.T:>3}): " + "  ".join(row))
    est = lambda cfg, arm, it: timing[cfg][arm][1] + it * timing[cfg][arm][0]
    worst_ex = sum(est(t.key, "ceiling", i) for t in tasks for i in ITERS_LADDER)
    print(f"  expressibility gate, worst case (every rung, seed 0 only): {worst_ex / 60:.0f} min")
    print()

    # ── Expressibility ──────────────────────────────────────────────────────
    print("=" * 100)
    print("EXPRESSIBILITY — per config, before any arm is trusted")
    print("=" * 100)
    express, final_iters = {}, {}
    for t in tasks:
        arms = {a["key"]: a for a in arms_for(t)}
        mse, am = gate_fit(t)
        fit_ok = am >= FIT_PASS
        print(f"  {t.label}")
        print(f"    recurrent gate fit to perfect routing: MSE={mse:.4f} argmax match={am:.4f}"
              f"  ({'PASS' if fit_ok else 'FAIL'} >= {FIT_PASS})")
        chosen = None
        for it in ITERS_LADDER:
            accs, rung_ok = [], True
            for s in seeds:
                r = do_run(store, args.results, t, arms["ceiling"], it, s, args.force, t0,
                           tag="  [ceiling ladder]")
                a = r["acc"] if r.get("ok") else None
                accs.append(a)
                if a is None or a < CEIL_PASS:
                    rung_ok = False
                    break
            print(f"    ceiling @ITERS={it}: per seed {['--' if a is None else round(a, 4) for a in accs]}"
                  f"  -> {'PASS on every seed' if rung_ok else 'FAIL (stopped at first failing seed)'}")
            if rung_ok:
                chosen = it
                break
        valid = fit_ok and chosen is not None
        reason = ""
        if chosen is None:
            reason = (f"ceiling < {CEIL_PASS} at every rung up to {ITERS_LADDER[-1]}: "
                      f"outside the architecture's reach")
        elif not fit_ok:
            reason = f"recurrent gate cannot fit perfect routing (argmax {am:.4f})"
        final_iters[t.key] = chosen if chosen is not None else ITERS_LADDER[-1]
        express[t.key] = dict(valid=valid, reason=reason, fit_mse=mse, fit_argmax=am,
                              iters=final_iters[t.key])
        store["express"][t.key] = express[t.key]
        save_results(args.results, store)
        print(f"    -> {'VALID' if valid else 'EXCLUDED: ' + reason};  all arms run at "
              f"ITERS={final_iters[t.key]}")
        print()

    # ── Projection and phase split ──────────────────────────────────────────
    proj = 0.0
    for t in tasks:
        for kk in [a["key"] for a in arms_for(t)]:
            for s in seeds:
                if get_run(store, t.key, kk, final_iters[t.key], s) is None or args.force:
                    proj += est(t.key, kk, final_iters[t.key])
    print("=" * 100)
    print(f"PROJECTED WALL CLOCK for the arm sweep: {proj / 3600:.2f} h "
          f"(limit {WALL_LIMIT_H} h)")
    split = proj / 3600 > WALL_LIMIT_H and len(seeds) > 3
    phases = [seeds[:3], seeds[3:]] if split else [seeds]
    print(f"  -> {'SPLIT: seeds 0-2 across all configs first, interim report, then the rest' if split else 'single pass'}")
    print("=" * 100)
    print()

    # ── Sweep ───────────────────────────────────────────────────────────────
    for pi, ph in enumerate(phases):
        print(f"-- sweep phase {pi + 1}/{len(phases)}: seeds {ph} --")
        for t in tasks:
            for a in arms_for(t):
                for s in ph:
                    do_run(store, args.results, t, a, final_iters[t.key], s, args.force, t0)
        print()
        done_seeds = [s for p in phases[:pi + 1] for s in p]
        note = (f" — PARTIAL: seeds {done_seeds} of {seeds}" if pi < len(phases) - 1 else "")
        report(store, tasks, done_seeds if note else seeds, final_iters, express, info,
               time.time() - t0, args.results, partial_note=note)


if __name__ == "__main__":
    main()
