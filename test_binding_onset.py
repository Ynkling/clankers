#!/usr/bin/env python
"""
test_binding_onset.py — how reliably, and how fast, does multilayer BDH start binding?

Run it directly:

    python test_binding_onset.py                    # everything, 5 seeds
    python test_binding_onset.py --seeds 3 --force --threads 4

WHY. A rerun of test_multilayer_binding.py's Gate 0 produced the first real binding in
this project: n_layer=3 with the 'decay' positional operator, seed 0, reached 2048/2048 at
9600 steps (loss 0.0019), up from 0.587 at 4800. It answered every FIRST query correctly,
which elimination cannot do. The other two seeds stayed on the elimination plateau
(~0.56), and the same seed had not bound in the first run on another machine. So BDH at
N=64 CAN bind, by a late, abrupt jump sitting at the edge of the 9600-step budget; RoPE
never left its 0.55 plateau. This test measures when binding appears and how reliably,
in order to pick a configuration where it happens on EVERY seed — which the channel
comparison needs. It runs no channel comparison.

REUSE. MultiBDH, the six verification CHECKs, BindTask, run_ml and the reporting helpers
(stats, cell, save_results) all come from test_multilayer_binding.py; none is rewritten.

TASK. BindTask with S=1, P=4, n_vals=16, one channel, no gate.
  primary    n_q=1: one query per sequence, so elimination is impossible and the
             training signal is exactly what is scored.
  secondary  n_q=4, the original Gate 0 setting, n_layer=3 'decay' only: does the
             elimination shortcut speed up or delay binding?

TRAINING. One long run per (config, seed) instead of a ladder of separate runs: up to
MAX_ITERS steps, evaluating every EVAL_EVERY steps on a fixed held-out batch drawn from
its own generator. A run stops early once accuracy >= BIND_PASS at STOP_AFTER consecutive
evaluations. The training recipe is Gate 0's exactly: CHECK 7 proves this loop reproduces
test_multilayer_binding.run_ml bit for bit, and CHECK 8 proves evaluating does not
perturb training. The transition step is the first evaluation reaching >= BIND_PASS that
holds to the end of the run; a run is BOUND if it has one.

GRID (primary): n_layer in {2, 3, 4} x positional in {decay, rope}, 5 seeds.
REPRODUCTION (informational): the exact Gate 0 code path, n_layer=3 'decay' seed 0 at
  9600 steps — does this machine reproduce 2048/2048? A mismatch is expected to be
  machine-dependent and is only reported.
REFERENCE (informational, not a gate): a 2-layer, d_model=32, 2-head causal softmax
  transformer on the primary task with the same recipe and evaluation schedule, 5 seeds
  (CHECK 9 proves the causal mask works).

SUMMARY, mechanical: per (n_layer, positional) the seeds bound by MAX_ITERS (k/5) and the
median and max transition step. RELIABLE CONFIG = the smallest n_layer, decay before rope
on a tie, where all 5 seeds bind, with RECOMMENDED BUDGET = ceil(1.25 x max transition /
EVAL_EVERY) x EVAL_EVERY — the setting to rerun the multichannel test at. Otherwise
NO RELIABLE CONFIG, with the config that bound the most seeds and its rate. Then one
compact accuracy-vs-step row per run.

LOGISTICS. Wall clock is projected before the sweep; over WALL_LIMIT_H, n_layer=4 is
dropped and that is said. Failures (exception or non-finite) are recorded per run and
never dropped. Results persist atomically to binding_onset_results.json after every run.

(Results are recorded at the bottom of this docstring after the run.)
"""

import argparse
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F

import test_multilayer_binding as tmb
from test_multilayer_binding import (
    MultiBDH, BindTask, stats, cell, save_results, run_ml, n_params, MULT,
)
from test_instrument_v2 import LR, BATCH, D, TAU_END

# ── Settings ─────────────────────────────────────────────────────────────────
MAX_ITERS = 38400
EVAL_EVERY = 1200
BIND_PASS = 0.95
STOP_AFTER = 3
EVAL_QUERIES = 2048               # held-out queries per evaluation
EVAL_SEED = 424_242               # the held-out batch's own generator
P, N_VALS = 4, 16
LAYERS = (2, 3, 4)
POSITIONALS = ("decay", "rope")   # decay first: the tie-break order
N_SEEDS = 5
TF_LAYERS, TF_HEADS = 2, 2
REPRO = dict(n_layer=3, positional="decay", seed=0, iters=9600)
WALL_LIMIT_H = 4.0
THREADS = 1
RESULTS_FILE = "binding_onset_results.json"

PRIMARY = BindTask(P, S=1, n_vals=N_VALS, n_q=1)
PRIMARY.key = "Q1"
SECONDARY = BindTask(P, S=1, n_vals=N_VALS, n_q=4)
SECONDARY.key = "Q4"


# ── Reference: a small causal softmax transformer ────────────────────────────
class CausalTransformer(nn.Module):
    """2-layer pre-norm causal softmax transformer with learned positions."""

    def __init__(self, vocab, T, d=D, n_head=TF_HEADS, n_layer=TF_LAYERS):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(T, d)
        layer = nn.TransformerEncoderLayer(d, n_head, dim_feedforward=4 * d, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.use_mask = True                      # CHECK 9 switches it off to show teeth

    def forward(self, x):
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T))
        if self.use_mask:
            h = self.enc(h, mask=nn.Transformer.generate_square_subsequent_mask(T),
                         is_causal=True)
        else:
            h = self.enc(h)
        return (self.head(self.ln(h)),)


def make_bdh(task, n_layer, positional):
    """Exactly the model run_ml builds for Gate 0's single-channel, ungated arm."""
    return lambda: MultiBDH(task.vocab, n_layer, "none", 1, positional,
                            gate_to_readout=False, h_gate=None, mult=MULT,
                            ctx_tokens=task.ctx_tokens)


def make_tf(task):
    return lambda: CausalTransformer(task.vocab, task.T)


# ── The long run ─────────────────────────────────────────────────────────────
def logits_of(model, x):
    """MultiBDH is called exactly as run_ml calls it; the transformer takes tokens only."""
    return model(x, TAU_END)[0] if isinstance(model, MultiBDH) else model(x)[0]


def eval_batch(task):
    g = torch.Generator().manual_seed(EVAL_SEED)
    return task.make_batch(EVAL_QUERIES // task.n_q, g)[0]


@torch.no_grad()
def evaluate(model, task, data):
    was = model.training
    model.eval()
    ql, qt, _ = task.select(logits_of(model, data[:, :-1]), data[:, 1:], None)
    acc = (ql.argmax(-1) == qt).float().mean().item()
    loss = F.cross_entropy(ql, qt).item()
    model.train(was)
    return acc, loss


def onset_run(task, make_model, seed, max_iters=MAX_ITERS, eval_every=EVAL_EVERY,
              data=None, early_stop=True):
    """run_ml's training recipe, run long, with periodic held-out evaluation."""
    torch.manual_seed(seed)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = torch.Generator(); rng.manual_seed(seed + 10_000)
    curve, streak = [], 0
    for step in range(1, max_iters + 1):
        tokens, _ = task.make_batch(BATCH, rng)
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        ql, qt, _ = task.select(logits_of(model, inp), tgt, None)
        loss = F.cross_entropy(ql, qt)
        opt.zero_grad(); loss.backward(); opt.step()
        if data is not None and step % eval_every == 0:
            acc, el = evaluate(model, task, data)
            curve.append([step, acc, el])
            streak = streak + 1 if acc >= BIND_PASS else 0
            if early_stop and streak >= STOP_AFTER:
                break
    return model, curve, rng


def transition(curve):
    """First evaluation >= BIND_PASS that holds to the end of the run, else None."""
    t = None
    for step, acc, _ in reversed(curve):
        if acc >= BIND_PASS:
            t = step
        else:
            break
    return t


# ── Persistence and one recorded run ─────────────────────────────────────────
def load_results(path):
    blank = {"meta": {}, "runs": {}, "repro": {}}
    if not os.path.exists(path):
        return blank
    try:
        with open(path) as f:
            d = json.load(f)
        for kk in blank:
            d.setdefault(kk, {})
        return d
    except Exception as e:
        print(f"  ! could not read {path} ({e}); starting fresh")
        return blank


def do_run(store, path, group, name, task, make_model, seed, data, force, t0):
    key = f"{group}|{name}|{seed}"
    cached = store["runs"].get(key)
    if cached is not None and not force:
        print(f"   {group} {name:<10} seed {seed}  cached "
              f"({'ok' if cached.get('ok') else 'FAILED'})")
        return cached
    ts = time.time()
    try:
        model, curve, _ = onset_run(task, make_model, seed, data=data)
        if not all(math.isfinite(a) and math.isfinite(l) for _, a, l in curve):
            raise ValueError("non-finite accuracy or loss in the curve")
        rec = dict(ok=True, seed=seed, curve=curve, transition=transition(curve),
                   stopped_at=curve[-1][0] if curve else 0, params=n_params(model),
                   secs=time.time() - ts)
    except Exception as e:
        rec = dict(ok=False, seed=seed, error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc()[-1500:], secs=time.time() - ts)
    store["runs"][key] = rec
    save_results(path, store)
    el = (time.time() - t0) / 60
    if rec["ok"]:
        last = rec["curve"][-1] if rec["curve"] else [0, float("nan"), float("nan")]
        tr = rec["transition"]
        print(f"   {group} {name:<10} seed {seed}  "
              f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} last acc={last[1]:.4f} "
              f"@{last[0]:<5} {rec['secs']:.0f}s  (elapsed {el:.1f}m)")
    else:
        print(f"   {group} {name:<10} seed {seed}  *** FAILED: {rec['error']}   "
              f"(elapsed {el:.1f}m)")
    return rec


# ── Verification ─────────────────────────────────────────────────────────────
def verify():
    tmb.verify()            # CHECKs 1-6: MultiBDH, RoPE, gate algebra, BindTask, gate, layers
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True

    print("CHECK 7  the long-run training loop IS Gate 0's: onset_run with no evaluation,")
    print("         then run_ml's own final evaluation (512 sequences drawn from the")
    print("         training generator), vs test_multilayer_binding.run_ml itself:")
    for L, pos in ((3, "decay"), (2, "rope")):
        arch = dict(n_layer=L, positional=pos)
        ref = run_ml(SECONDARY, arch, "none", 1, 0, 1200)
        model, _, rng = onset_run(SECONDARY, make_bdh(SECONDARY, L, pos), 0, max_iters=1200)
        with torch.no_grad():
            model.eval()
            tok, _ = SECONDARY.make_batch(512, rng)
            ql, qt, _ = SECONDARY.select(logits_of(model, tok[:, :-1]), tok[:, 1:], None)
            mine = (ql.argmax(-1) == qt).float().mean().item()
        same = mine == ref["acc"]
        ok &= same
        print(f"         n_layer={L} {pos:<5} seed 0, 1200 steps: run_ml acc={ref['acc']!r}  "
              f"onset_run acc={mine!r}  -> {'IDENTICAL' if same else '*** DIFFERS ***'}")
    print()

    print("CHECK 8  evaluation does not perturb training: 4800 steps with an evaluation every")
    print(f"         {EVAL_EVERY} steps vs none, n_layer=3 decay, seed 0, primary task:")
    data = eval_batch(PRIMARY)
    m_eval, curve, _ = onset_run(PRIMARY, make_bdh(PRIMARY, 3, "decay"), 0, max_iters=4800,
                                 data=data, early_stop=False)
    m_none, _, _ = onset_run(PRIMARY, make_bdh(PRIMARY, 3, "decay"), 0, max_iters=4800)
    pa, pb = dict(m_eval.named_parameters()), dict(m_none.named_parameters())
    wsame = sorted(pa) == sorted(pb) and all(torch.equal(pa[n], pb[n]) for n in pa)
    a_eval, _ = evaluate(m_eval, PRIMARY, data)
    a_none, _ = evaluate(m_none, PRIMARY, data)
    print(f"         evaluations made: {len(curve)}   final weights bitwise equal: {wsame}")
    print(f"         held-out accuracy: with evals {a_eval!r}  without {a_none!r}  "
          f"-> {'IDENTICAL' if a_eval == a_none else '*** DIFFERS ***'}")
    ok &= wsame and a_eval == a_none
    print()

    print("CHECK 9  the reference transformer's causal mask works: changing every token after")
    print("         position t must leave logits at positions <= t untouched (train mode, as")
    print("         trained, and eval mode, as evaluated):")
    torch.manual_seed(1)
    tf = CausalTransformer(PRIMARY.vocab, PRIMARY.T)
    g = torch.Generator().manual_seed(9)
    x = PRIMARY.make_batch(8, g)[0][:, :-1]
    for mode in ("train", "eval"):
        tf.train(mode == "train")
        worst_masked, worst_open = 0.0, 0.0
        with torch.no_grad():
            for t in (2, 6, 10):
                y = x.clone()
                y[:, t + 1:] = torch.randint(0, PRIMARY.vocab, y[:, t + 1:].shape, generator=g)
                worst_masked = max(worst_masked,
                                   (tf(x)[0][:, :t + 1] - tf(y)[0][:, :t + 1]).abs().max().item())
                tf.use_mask = False
                worst_open = max(worst_open,
                                 (tf(x)[0][:, :t + 1] - tf(y)[0][:, :t + 1]).abs().max().item())
                tf.use_mask = True
        good = worst_masked < 1e-5 and worst_open > 1e-3
        ok &= good
        print(f"         {mode:<5}  with the mask: max |change at positions <= t| = "
              f"{worst_masked:.2e}   without it: {worst_open:.2e} (the check has teeth)"
              f"  -> {'CAUSAL' if good else 'FAILED'}")
    print()

    print("CHECK 10 the primary task leaves nothing to eliminate: one query per sequence,")
    print("         answered by its key's bound value:")
    tok = PRIMARY.make_batch(512, torch.Generator().manual_seed(4))[0]
    body = tok[:, :3 * P].reshape(512, P, 3)
    q = tok[:, 3 * P:].reshape(512, 1, 3)
    bound = all({int(body[b, i, 1]): int(body[b, i, 2]) for i in range(P)}[int(q[b, 0, 1])]
                == int(q[b, 0, 2]) for b in range(512))
    print(f"         query positions scored per sequence: {len(PRIMARY.qpos)}   bound value "
          f"answers every query: {bound}   sequence length: {PRIMARY.L}")
    ok &= len(PRIMARY.qpos) == 1 and bound
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"


def time_per_step(task, make_model, steps=300):
    data = eval_batch(task)
    t0 = time.time()
    onset_run(task, make_model, 0, max_iters=steps, eval_every=steps, data=data,
              early_stop=False)
    return (time.time() - t0) / steps


# ── Reporting ────────────────────────────────────────────────────────────────
def row_stats(store, group, name, seeds):
    recs = [store["runs"].get(f"{group}|{name}|{s}") for s in seeds]
    done = [r for r in recs if r is not None and r.get("ok")]
    failed = [r for r in recs if r is not None and not r.get("ok")]
    trs = [r["transition"] for r in done if r["transition"] is not None]
    st = stats(trs)
    med = None
    if trs:
        s = sorted(trs)
        med = s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2
    finals = stats([r["curve"][-1][1] for r in done if r["curve"]])
    return dict(bound=len(trs), done=len(done), failed=len(failed), total=len(seeds),
                med=med, max=(st["max"] if st else None), trs=trs, finals=finals)


def fmt_step(x):
    return "--" if x is None else f"{x:.0f}"


def report(store, seeds, layers, wall, path):
    print("#" * 100)
    print("SUMMARY — printed mechanically")
    print("#" * 100)
    print(f"  bound = an evaluation >= {BIND_PASS} that holds to the end of the run, within "
          f"MAX_ITERS={MAX_ITERS}")
    print()
    print(f"  {'config':<26} {'bound':>7} {'median':>7} {'max':>7}   {'per-seed transition':<36}"
          f" final held-out acc")
    rows = {}
    order = [("Q1", f"L{L}-{pos}", f"primary n_q=1   L{L} {pos}") for L in layers
             for pos in POSITIONALS]
    extra = [("Q4", "L3-decay", "secondary n_q=4 L3 decay"),
             ("TF", "L2-softmax", "reference transformer")]
    for group, name, label in order + extra:
        rs = row_stats(store, group, name, seeds)
        rows[(group, name)] = rs
        per = []
        for s in seeds:
            r = store["runs"].get(f"{group}|{name}|{s}")
            per.append("--" if r is None else ("FAIL" if not r.get("ok") else
                       (str(r["transition"]) if r["transition"] else "never")))
        print(f"  {label:<26} {rs['bound']:>3}/{rs['total']:<3} {fmt_step(rs['med']):>7} "
              f"{fmt_step(rs['max']):>7}   {' '.join(per):<36} {cell(rs['finals'])}"
              + (f"   [{rs['failed']} FAILED]" if rs["failed"] else ""))
    print()

    reliable = None
    for L in layers:
        for pos in POSITIONALS:
            rs = rows[("Q1", f"L{L}-{pos}")]
            if rs["bound"] == len(seeds) and reliable is None:
                reliable = (L, pos, rs)
        if reliable:
            break
    if reliable:
        L, pos, rs = reliable
        budget = math.ceil(1.25 * rs["max"] / EVAL_EVERY) * EVAL_EVERY
        print(f"  RELIABLE CONFIG: n_layer={L}, positional={pos} — {rs['bound']}/{len(seeds)} "
              f"seeds bound, max transition {rs['max']:.0f}")
        print(f"  RECOMMENDED BUDGET = ceil(1.25 x {rs['max']:.0f} / {EVAL_EVERY}) x "
              f"{EVAL_EVERY} = {budget} steps. This is the setting to rerun the multichannel "
              f"test at.")
    else:
        best = None
        for L in layers:
            for pos in POSITIONALS:
                rs = rows[("Q1", f"L{L}-{pos}")]
                if best is None or rs["bound"] > best[2]["bound"]:
                    best = (L, pos, rs)
        L, pos, rs = best
        print("  NO RELIABLE CONFIG: no primary config bound all "
              f"{len(seeds)} seeds within {MAX_ITERS} steps.")
        print(f"  Most seeds bound: n_layer={L}, positional={pos} — {rs['bound']}/{len(seeds)} "
              f"= {rs['bound'] / len(seeds):.0%}"
              + (f", transitions {sorted(rs['trs'])}" if rs["trs"] else ""))
    print()

    q4, q1 = rows[("Q4", "L3-decay")], rows[("Q1", "L3-decay")]
    print("  Secondary (informational): does the elimination shortcut change binding onset?")
    print(f"    n_layer=3 decay  n_q=1 (primary): {q1['bound']}/{q1['total']} bound, median "
          f"{fmt_step(q1['med'])}, max {fmt_step(q1['max'])}")
    print(f"    n_layer=3 decay  n_q=4 (Gate 0):  {q4['bound']}/{q4['total']} bound, median "
          f"{fmt_step(q4['med'])}, max {fmt_step(q4['max'])}")
    tf = rows[("TF", "L2-softmax")]
    print(f"  Reference (informational): 2-layer causal softmax transformer, n_q=1: "
          f"{tf['bound']}/{tf['total']} bound, median {fmt_step(tf['med'])}, "
          f"max {fmt_step(tf['max'])}")
    rp = store.get("repro", {})
    if rp:
        print(f"  Reproduction (informational): Gate 0 code path, n_layer=3 decay seed 0 @9600 "
              f"-> acc {rp['acc']:.4f} = {rp['correct']}/2048 "
              f"({'REPRODUCES 2048/2048' if rp['correct'] == 2048 else 'does NOT reproduce 2048/2048'})")
    print()

    print("=" * 100)
    print(f"CURVES — held-out accuracy x100 at every {EVAL_EVERY} steps; '|' marks the "
          f"transition; runs stop after {STOP_AFTER} consecutive evaluations >= {BIND_PASS}")
    print("=" * 100)
    for group, name, label in order + extra:
        for s in seeds:
            r = store["runs"].get(f"{group}|{name}|{s}")
            if r is None:
                print(f"  {group} {name:<10} s{s}  NOT RUN")
                continue
            if not r.get("ok"):
                print(f"  {group} {name:<10} s{s}  FAILED — {r['error']}")
                continue
            cells = []
            for step, acc, _ in r["curve"]:
                mark = "|" if r["transition"] == step else " "
                cells.append(f"{mark}{round(acc * 100):>3}")
            tr = r["transition"]
            print(f"  {group} {name:<10} s{s} " + "".join(cells)
                  + f"   -> {'bound @' + str(tr) if tr else 'not bound'}")
    print()
    print(f"  total wall clock this invocation: {wall / 60:.1f} min ({wall:.0f}s)")
    print(f"  raw records: {path}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="When and how reliably does BDH bind?")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--results", default=RESULTS_FILE)
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.threads))
    seeds = list(range(args.seeds))

    print("=" * 100)
    print("Binding onset: how reliably and how fast does multilayer BDH bind?")
    print(f"  S=1 P={P} n_vals={N_VALS}  primary n_q=1, secondary n_q=4  MAX_ITERS={MAX_ITERS} "
          f"eval every {EVAL_EVERY} on {EVAL_QUERIES} held-out queries")
    print(f"  grid: n_layer {LAYERS} x {POSITIONALS}, seeds {seeds}   torch {torch.__version__}"
          f"   threads {args.threads}")
    print("=" * 100)
    print()
    verify()

    store = load_results(args.results)
    store["meta"] = dict(max_iters=MAX_ITERS, eval_every=EVAL_EVERY, bind_pass=BIND_PASS,
                         stop_after=STOP_AFTER, seeds=seeds, torch=torch.__version__,
                         started=time.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()

    print("=" * 100)
    print("PROJECTION — per-step cost from 300-step runs, worst case = every run to MAX_ITERS")
    print("=" * 100)
    per = {}
    for L in LAYERS:
        for pos in POSITIONALS:
            per[(L, pos)] = time_per_step(PRIMARY, make_bdh(PRIMARY, L, pos))
    per_q4 = time_per_step(SECONDARY, make_bdh(SECONDARY, 3, "decay"))
    per_tf = time_per_step(PRIMARY, make_tf(PRIMARY))
    for (L, pos), v in per.items():
        print(f"  primary L{L} {pos:<5}: {v * 1000:.1f} ms/step")
    print(f"  secondary L3 decay: {per_q4 * 1000:.1f} ms/step   transformer: "
          f"{per_tf * 1000:.1f} ms/step")
    n = len(seeds)
    fixed = n * MAX_ITERS * (per_q4 + per_tf) + REPRO["iters"] * per[(3, "decay")]
    full = n * MAX_ITERS * sum(per.values()) + fixed
    no4 = n * MAX_ITERS * sum(v for (L, _), v in per.items() if L != 4) + fixed
    print(f"  worst case with n_layer=4: {full / 3600:.2f} h;  without: {no4 / 3600:.2f} h "
          f"(limit {WALL_LIMIT_H} h)")
    layers = LAYERS
    if full / 3600 > WALL_LIMIT_H:
        layers = tuple(L for L in LAYERS if L != 4)
        print(f"  -> OVER THE LIMIT: n_layer=4 DROPPED. Grid is n_layer {layers}.")
    else:
        print("  -> within the limit: full grid.")
    print()

    print("=" * 100)
    print("GRID — primary task (n_q=1)")
    print("=" * 100)
    data1 = eval_batch(PRIMARY)
    for L in layers:
        for pos in POSITIONALS:
            for s in seeds:
                do_run(store, args.results, "Q1", f"L{L}-{pos}", PRIMARY,
                       make_bdh(PRIMARY, L, pos), s, data1, args.force, t0)
    print()
    print("=" * 100)
    print("SECONDARY — original Gate 0 setting (n_q=4), n_layer=3 decay")
    print("=" * 100)
    data4 = eval_batch(SECONDARY)
    for s in seeds:
        do_run(store, args.results, "Q4", "L3-decay", SECONDARY, make_bdh(SECONDARY, 3, "decay"),
               s, data4, args.force, t0)
    print()
    print("=" * 100)
    print("REPRODUCTION (informational) — the exact Gate 0 code path, "
          f"n_layer={REPRO['n_layer']} {REPRO['positional']} seed {REPRO['seed']} "
          f"@{REPRO['iters']}")
    print("=" * 100)
    if not store["repro"] or args.force:
        r = run_ml(SECONDARY, dict(n_layer=REPRO["n_layer"], positional=REPRO["positional"]),
                   "none", 1, REPRO["seed"], REPRO["iters"])
        store["repro"] = dict(acc=r["acc"], correct=round(r["acc"] * 2048),
                              loss=r["losses"][0], torch=torch.__version__)
        save_results(args.results, store)
    rp = store["repro"]
    print(f"  acc {rp['acc']:.4f} = {rp['correct']}/2048, loss {rp['loss']:.4f}   (the other "
          f"machine: 2048/2048, loss 0.0019; this project's first run: 0.5088)")
    print(f"  -> {'REPRODUCES' if rp['correct'] == 2048 else 'does NOT reproduce'} 2048/2048 on "
          f"this machine (torch {rp['torch']}). Binding onset at 9600 steps is machine-dependent"
          f" if these differ.")
    print()
    print("=" * 100)
    print("REFERENCE (informational) — 2-layer, d_model=32, 2-head causal softmax transformer")
    print("=" * 100)
    for s in seeds:
        do_run(store, args.results, "TF", "L2-softmax", PRIMARY, make_tf(PRIMARY), s, data1,
               args.force, t0)
    print()
    report(store, seeds, layers, time.time() - t0, args.results)


if __name__ == "__main__":
    main()
