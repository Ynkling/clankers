#!/usr/bin/env python
"""
test_capacity_control.py — does the multi-channel IDEA beat simply having more
resources?

Run it directly:

    python test_capacity_control.py                 # 5 arms + floor/ceiling, 10 seeds
    python test_capacity_control.py --seeds 20
    python test_capacity_control.py --force         # ignore cache, retrain
    python test_capacity_control.py --threads 4

SCIENTIFIC QUESTION. test_instrument_v2.py showed the k=2 recurrent-gate model reaches
the ceiling on the interference task. But that task is built to force separation, and
going from one channel to k channels buys two things at once: more trainable parameters
AND more fast-weight memory. If a SINGLE-channel BDH with matched resources solves the
same task, then "channels" is a re-parameterization of "more memory" and the distinctive
multi-channel claim fails — even though the apparatus demonstrably works.

NO SECOND IMPLEMENTATION. The model, task, training loop, floor/ceiling, decay,
randomized block order and every hyperparameter are imported from test_instrument_v2, as
test_phase2_seeds.py does. The two controls that need a wider state or extra parameters
reach it through knobs added to the IMPORTED model, not a second model:

  * Instrument(n_feat=...) sets the sparse feature width, so one channel's fast-weight
    state is n_feat x D. Defaults to N, drawing the same random numbers in the same
    order as before.
  * Instrument(gate_to_readout=True) adds one projection of the recurrent gate's hidden
    state into the residual before the head. The parameter is created after all
    pre-existing ones and only when the flag is set, so the default RNG stream is
    untouched.

CHECK 0 proves the refactor is inert: default-path parameters and a full 1200-step run
are bitwise identical with and without the new knobs.

ARMS (all: interference task, decay 0.95 = U!=I, randomized order, symmetric soft gate,
per-seed lift against that seed's own floor and ceiling):

  A  k=2 recurrent-state gate       the headline model, reused as-is.
  B  single-channel BDH             the plain baseline: fewer params AND less memory.
  C  single-channel, MEMORY-matched one channel whose fast-weight state holds the same
                                    number of floats as A's two channels combined
                                    (n_feat doubled). Rules out "the win is just more
                                    stored state". C has MORE parameters than A.
  D  single-channel, PARAM-matched  one channel with the same trainable parameter count
                                    as A (n_feat=86). Rules out "the win is just more
                                    weights". D has LESS memory than A.
  E  single-channel + gate features A's recurrent gate runs and feeds the readout, but
                                    there is only ONE channel, so there is nothing to
                                    route to. Rules out "the recurrent gate is a small
                                    RNN that helps on its own, independent of routing".

No single control is decisive, which is why all three are run: C and D pull apart
(C over-matches parameters while matching memory, D matches parameters while
under-matching memory) and E removes routing while keeping the gate machinery.

PRE-REGISTERED KILL CRITERION (printed in the verdict BEFORE any interpretation):
the multi-channel idea FAILS to clear its own capacity baseline on this task if the best
matched single-channel control (C, D or E) comes within EPS = 0.05 raw query accuracy of
arm A. The verdict prints this mechanically. A fail is a fail, not a partial.

PRIMARY AXIS: raw query accuracy, arm A minus best-matched-control, per seed. Lift is
secondary and reported against each seed's own floor/ceiling.

REPORTING follows test_phase2_seeds.py: every per-seed raw accuracy before any
aggregate; population std; a condition whose seeds all agree is labelled "identical: X"
rather than "+- 0.000"; failures (exception or non-finite result) are recorded per seed
with the error text, shown in the tables and never dropped, and any incomplete sweep is
labelled PARTIAL. A run that merely scores badly is a RESULT, not a failure — there is
no convergence threshold that could quietly discard an inconvenient seed. Results
persist atomically to capacity_control_results.json after every run; cached seeds are
skipped unless --force.
"""

import argparse
import json
import math
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import test_instrument_v2 as tiv
from test_instrument_v2 import OPERATING_DECAY, ITERS, D, N, k, H_GATE

# ── Settings ─────────────────────────────────────────────────────────────────
N_SEEDS = 10
THREADS = 1
RESULTS_FILE = "capacity_control_results.json"

# PRE-REGISTERED kill criterion. If the best matched single-channel control lands within
# EPS raw accuracy of arm A, the multi-channel idea has not cleared its capacity baseline.
EPS = 0.05

# n_feat for the two matched controls. Both are exact with the available knobs; CHECK 1
# prints the residuals rather than asserting the match silently.
#   memory:    A holds k * N * D = 2*64*32 = 4096 state floats; 1 channel needs
#              n_feat*D = 4096 -> n_feat = 128.
#   parameter: A has 8768 trainable params; a gate-free single channel has
#              512 + 3*D*n_feat -> 512 + 96*n_feat = 8768 -> n_feat = 86.
NFEAT_MEMORY_MATCHED = 128
NFEAT_PARAM_MATCHED = 86
MATCH_TOL = 0.02          # fractional tolerance a "match" must land inside

ARMS = [
    dict(key="A_mc_recurrent", short="A k=2", label="A. k=2 recurrent-state gate (headline)",
         gate="recurrent", n_ch=k, n_feat=None, g2r=False, matched=False),
    dict(key="B_single", short="B 1ch", label="B. single-channel BDH (plain baseline)",
         gate="none", n_ch=1, n_feat=None, g2r=False, matched=False),
    dict(key="C_single_mem", short="C mem", label="C. single-channel, MEMORY-matched",
         gate="none", n_ch=1, n_feat=NFEAT_MEMORY_MATCHED, g2r=False, matched=True),
    dict(key="D_single_par", short="D par", label="D. single-channel, PARAMETER-matched",
         gate="none", n_ch=1, n_feat=NFEAT_PARAM_MATCHED, g2r=False, matched=True),
    dict(key="E_single_gate", short="E gate", label="E. single-channel + gate features",
         gate="recurrent", n_ch=1, n_feat=None, g2r=True, matched=True),
]
REFS = [
    dict(key="floor_uniform", short="floor", label="floor: MC k=2, uniform gate",
         gate="uniform", n_ch=k, n_feat=None, g2r=False, matched=False),
    dict(key="ceil_perfect", short="ceil", label="ceiling: MC k=2, perfect gate",
         gate="perfect", n_ch=k, n_feat=None, g2r=False, matched=False),
]
ALL = ARMS + REFS
BY_KEY = {c["key"]: c for c in ALL}
HEADLINE = "A_mc_recurrent"
MATCHED = [c["key"] for c in ARMS if c["matched"]]
FLOOR_K, CEIL_K = "floor_uniform", "ceil_perfect"


# ── Resource accounting ──────────────────────────────────────────────────────
def build(cfg, seed=0):
    torch.manual_seed(seed)
    return tiv.Instrument(cfg["gate"], "sym", cfg["n_ch"], OPERATING_DECAY, None,
                          n_feat=cfg["n_feat"], gate_to_readout=cfg["g2r"])


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def n_state(m):
    """Fast-weight floats held per sequence: n_ch channels of an (n_feat x D) matrix."""
    return m.n_ch * m.Dx.shape[1] * D


# ── Persistence ──────────────────────────────────────────────────────────────
def load_results(path):
    if not os.path.exists(path):
        return {"meta": {}, "results": {}}
    try:
        with open(path) as f:
            d = json.load(f)
        d.setdefault("meta", {})
        d.setdefault("results", {})
        return d
    except Exception as e:
        print(f"  ! could not read {path} ({e}); starting fresh")
        return {"meta": {}, "results": {}}


def save_results(path, store):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(store, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


# ── One (arm, seed) run ──────────────────────────────────────────────────────
def run_one(key, seed):
    c = BY_KEY[key]
    t0 = time.time()
    try:
        r = tiv.run(c["gate"], "sym", c["n_ch"], OPERATING_DECAY, None, seed,
                    anneal=False, n_feat=c["n_feat"], gate_to_readout=c["g2r"])
        vals = [r["l1"], r["l2"], r["a1"], r["a2"]]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            raise ValueError(f"non-finite result: l1={r['l1']} l2={r['l2']} "
                             f"a1={r['a1']} a2={r['a2']}")
        return dict(ok=True, seed=seed, l1=r["l1"], l2=r["l2"], a1=r["a1"], a2=r["a2"],
                    acc=(r["a1"] + r["a2"]) / 2.0, secs=time.time() - t0)
    except Exception as e:
        return dict(ok=False, seed=seed, error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc()[-1500:], secs=time.time() - t0)


# ── Statistics ───────────────────────────────────────────────────────────────
def stats(xs):
    """mean, POPULATION std (as used throughout this series), min, max, n."""
    xs = [x for x in xs
          if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)]
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


def get(store, key, seed):
    return store["results"].get(key, {}).get(str(seed))


def acc_of(store, key, seed):
    r = get(store, key, seed)
    return r["acc"] if (r and r.get("ok")) else None


def lift_of(store, key, seed):
    a, f, c = (acc_of(store, key, seed), acc_of(store, FLOOR_K, seed),
               acc_of(store, CEIL_K, seed))
    if a is None or f is None or c is None or (c - f) <= 1e-3:
        return None
    return (a - f) / (c - f)


# ── Verification ─────────────────────────────────────────────────────────────
def verify():
    print("=" * 100)
    print("VERIFICATION — printed before any training")
    print("=" * 100)
    ok = True

    # CHECK 0 — the two new knobs are inert on the default path
    print("CHECK 0  the knobs added to test_instrument_v2.Instrument are BACKWARD")
    print("         COMPATIBLE: defaults must draw the same weights in the same order.")
    worst = 0
    for g, nc in (("recurrent", 2), ("token", 2), ("uniform", 2), ("perfect", 2), ("none", 1)):
        torch.manual_seed(5); m0 = tiv.Instrument(g, "sym", nc, OPERATING_DECAY, None)
        torch.manual_seed(5); m1 = tiv.Instrument(g, "sym", nc, OPERATING_DECAY, None,
                                                  n_feat=None, gate_to_readout=False)
        p0, p1 = dict(m0.named_parameters()), dict(m1.named_parameters())
        same = sorted(p0) == sorted(p1) and all(torch.equal(p0[n], p1[n]) for n in p0)
        worst += (not same)
        print(f"         {g:>9} k={nc}: implicit vs explicit defaults bitwise equal = {same}"
              f"   ({n_params(m0)} params)")
    r0 = tiv.run("recurrent", "sym", k, OPERATING_DECAY, None, 0)
    r1 = tiv.run("recurrent", "sym", k, OPERATING_DECAY, None, 0,
                 n_feat=None, gate_to_readout=False)
    fields = [("l1", r0["l1"], r1["l1"]), ("l2", r0["l2"], r1["l2"]),
              ("a1", r0["a1"], r1["a1"]), ("a2", r0["a2"], r1["a2"]),
              ("val_cos", r0["trace"][-1]["val_cos"], r1["trace"][-1]["val_cos"])]
    for nm, a, b in fields:
        same = (a == b); worst += (not same)
        print(f"         full 1200-step run seed 0  {nm:>7}: {a!r} vs {b!r}  "
              f"{'IDENTICAL' if same else '*** DIFFERS ***'}")
    ok &= (worst == 0)
    print(f"         -> {'the refactor is INERT on the default path' if worst == 0 else 'FAILED'}\n")

    # CHECK 1 — resource accounting, with residuals stated
    print("CHECK 1  RESOURCE ACCOUNTING. Fast-weight state floats = n_ch * n_feat * D;")
    print(f"         D={D}, N={N}, k={k}, H_GATE={H_GATE}.")
    print(f"         {'arm':<40} {'n_ch':>5} {'n_feat':>7} {'params':>8} {'state':>8}")
    info = {}
    for c in ALL:
        m = build(c)
        info[c["key"]] = (n_params(m), n_state(m), m.n_feat)
        print(f"         {c['label']:<40} {c['n_ch']:>5} {m.n_feat:>7} "
              f"{info[c['key']][0]:>8} {info[c['key']][1]:>8}")
    print()

    pA, sA, _ = info[HEADLINE]
    pB, sB, _ = info["B_single"]
    pC, sC, _ = info["C_single_mem"]
    pD, sD, _ = info["D_single_par"]
    pE, sE, _ = info["E_single_gate"]

    print("CHECK 2  DO THE MATCHINGS HOLD? (residuals reported, not silently approximated)")
    mem_res = abs(sC - sA) / sA
    par_res = abs(pD - pA) / pA
    print(f"         C memory vs A: {sC} vs {sA} floats  -> residual {mem_res:.4%} "
          f"({'WITHIN' if mem_res <= MATCH_TOL else 'OUTSIDE'} tol {MATCH_TOL:.0%})")
    print(f"         D params vs A: {pD} vs {pA} params  -> residual {par_res:.4%} "
          f"({'WITHIN' if par_res <= MATCH_TOL else 'OUTSIDE'} tol {MATCH_TOL:.0%})")
    ok &= (mem_res <= MATCH_TOL and par_res <= MATCH_TOL)
    print("         Deliberate mismatches, stated rather than hidden — this is why three")
    print("         controls are run instead of one:")
    print(f"           C has {pC - pA:+d} params vs A ({(pC - pA) / pA:+.1%}): it OVER-matches")
    print(f"             parameters while matching memory, so if C wins it cannot be blamed")
    print(f"             on memory alone, and if C loses it is not for lack of weights.")
    print(f"           D has {sD - sA:+d} state floats vs A ({(sD - sA) / sA:+.1%}): it matches")
    print(f"             parameters exactly while holding LESS memory than A.")
    print(f"           E has {pE - pA:+d} params and {sE - sA:+d} state floats vs A: it keeps the")
    print(f"             recurrent gate (and gives it a readout path) but has ONE channel,")
    print(f"             so nothing can be routed. It is deliberately generous on params.")
    print(f"           B is the unmatched baseline: {pB - pA:+d} params, {sB - sA:+d} state.")
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"
    return info


# ── Reporting ────────────────────────────────────────────────────────────────
def report(store, seeds, info, wall, path):
    keys = [c["key"] for c in ALL]
    W = max(9, max(len(c["short"]) for c in ALL) + 2)

    print("=" * 100)
    print("PER-SEED RAW QUERY ACCURACY — every value, before any aggregate")
    print("=" * 100)
    def acc_cell(kk, s):
        r = get(store, kk, s)
        if r is None:
            return "--"
        if not r.get("ok"):
            return "FAIL"
        return f"{r['acc']:.4f}"

    print("  seed  " + "".join(f"{BY_KEY[kk]['short']:>{W}}" for kk in keys))
    for s in seeds:
        print(f"  {s:>4}  " + "".join(f"{acc_cell(kk, s):>{W}}" for kk in keys))
    print()

    print("=" * 100)
    print("PER-SEED PRIMARY AXIS — arm A minus best matched control (C, D, E), raw accuracy")
    print("=" * 100)
    print(f"  {'seed':>4}  {'A':>8}  {'bestCDE':>8}  {'which':>7}  {'A - best':>9}")
    for s in seeds:
        a = acc_of(store, HEADLINE, s)
        cands = [(acc_of(store, kk, s), kk) for kk in MATCHED]
        cands = [(v, kk) for v, kk in cands if v is not None]
        if a is None or not cands:
            print(f"  {s:>4}  {'--':>8}  {'--':>8}  {'--':>7}  {'--':>9}")
            continue
        bv, bk = max(cands)
        print(f"  {s:>4}  {a:>8.4f}  {bv:>8.4f}  {BY_KEY[bk]['short']:>7}  {a - bv:>+9.4f}")
    print()

    print("=" * 100)
    print("PER-SEED LIFT (secondary) — each seed against ITS OWN floor and ceiling")
    print("=" * 100)
    print("  seed  " + "".join(f"{BY_KEY[kk]['short']:>{W}}" for kk in keys))
    for s in seeds:
        row = ""
        for kk in keys:
            lf = lift_of(store, kk, s)
            row += f"{'--' if lf is None else f'{lf:+.3f}':>{W}}"
        print(f"  {s:>4}  " + row)
    print()

    print("=" * 100)
    print("SUMMARY — paste-ready")
    print("=" * 100)
    LB = max(40, max(len(c["label"]) for c in ALL) + 1)
    hdr = (f"  {'arm':<{LB}} {'params':>7} {'state':>6} {'accuracy mean +- std':>24} "
           f"{'min':>7} {'max':>7} {'n':>3}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for kk in keys:
        xs = [acc_of(store, kk, s) for s in seeds]
        nf = sum(1 for s in seeds
                 if get(store, kk, s) is not None and not get(store, kk, s).get("ok"))
        st = stats([x for x in xs if x is not None])
        p, stt, _ = info[kk]
        if st is None:
            print(f"  {BY_KEY[kk]['label']:<{LB}} {p:>7} {stt:>6} {'no completed seeds':>24}"
                  + (f"   [{nf} FAILED]" if nf else ""))
        else:
            print(f"  {BY_KEY[kk]['label']:<{LB}} {p:>7} {stt:>6} {cell(st):>24} "
                  f"{st['min']:>7.4f} {st['max']:>7.4f} {st['n']:>3}"
                  + (f"   [{nf} FAILED]" if nf else ""))
    print()
    print(f"  {'arm':<{LB}} {'lift mean +- std':>24} {'min':>7} {'max':>7} {'n':>3}")
    print("  " + "-" * (len(hdr) - 2))
    for kk in keys:
        st = stats([lift_of(store, kk, s) for s in seeds])
        if st is None:
            print(f"  {BY_KEY[kk]['label']:<{LB}} {'n/a':>24}")
        else:
            print(f"  {BY_KEY[kk]['label']:<{LB}} {cell(st, 3, True):>24} "
                  f"{st['min']:>7.3f} {st['max']:>7.3f} {st['n']:>3}")
    print()

    # ── Verdict ──────────────────────────────────────────────────────────────
    attempted, completed = len(keys) * len(seeds), 0
    for kk in keys:
        for s in seeds:
            r = get(store, kk, s)
            if r is not None and r.get("ok"):
                completed += 1
    partial = completed < attempted

    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"  PRE-REGISTERED KILL CRITERION (set before any result was seen):")
    print(f"    The multi-channel idea FAILS to clear its own capacity baseline on this")
    print(f"    task if the best matched single-channel control (C, D or E) comes within")
    print(f"    EPS = {EPS} raw query accuracy of arm A.")
    print()
    print(f"  {completed}/{attempted} runs completed"
          + ("   *** PARTIAL SWEEP ***" if partial else ""))
    if partial:
        for kk in keys:
            for s in seeds:
                r = get(store, kk, s)
                if r is None:
                    print(f"    {kk} seed {s}: NOT RUN")
                elif not r.get("ok"):
                    print(f"    {kk} seed {s}: FAILED — {r['error']}")
    print()

    ceil_st = stats([acc_of(store, CEIL_K, s) for s in seeds])
    print("  Which arms reached the ceiling"
          + (f" ({ceil_st['mean']:.4f})?" if ceil_st else "?"))
    for kk in [c["key"] for c in ARMS]:
        st = stats([acc_of(store, kk, s) for s in seeds])
        if st is None or ceil_st is None:
            continue
        reached = st["mean"] >= ceil_st["mean"] - EPS
        print(f"    {BY_KEY[kk]['label']:<40} acc={st['mean']:.4f}  "
              f"{'REACHED the ceiling' if reached else 'did NOT reach the ceiling'}")
    print()

    a_st = stats([acc_of(store, HEADLINE, s) for s in seeds])
    best_key, best_st = None, None
    for kk in MATCHED:
        st = stats([acc_of(store, kk, s) for s in seeds])
        if st and (best_st is None or st["mean"] > best_st["mean"]):
            best_key, best_st = kk, st
    per_seed_gap = []
    for s in seeds:
        a = acc_of(store, HEADLINE, s)
        cands = [acc_of(store, kk, s) for kk in MATCHED]
        cands = [v for v in cands if v is not None]
        if a is not None and cands:
            per_seed_gap.append(a - max(cands))
    gap_st = stats(per_seed_gap)

    if a_st is None or best_st is None or gap_st is None:
        print("  NO VERDICT: insufficient completed runs to evaluate the criterion.")
    else:
        margin = a_st["mean"] - best_st["mean"]
        fails = margin < EPS
        print(f"  arm A mean accuracy                         = {a_st['mean']:.4f}")
        print(f"  best matched control ({BY_KEY[best_key]['short']:>6}) mean accuracy = "
              f"{best_st['mean']:.4f}   [{BY_KEY[best_key]['label']}]")
        print(f"  margin (A - best matched control)           = {margin:+.4f}")
        print(f"  per-seed margin                             = {cell(gap_st, 4, True)}"
              f"  [min {gap_st['min']:+.4f}, max {gap_st['max']:+.4f}, n={gap_st['n']}]")
        print(f"  EPS                                         = {EPS}")
        print()
        n_within = sum(1 for g in per_seed_gap if g < EPS)
        if fails:
            print(f"  *** FAIL: margin {margin:+.4f} < EPS {EPS}. ***")
            print(f"  The multi-channel idea does NOT clear its own capacity baseline on this")
            print(f"  task. A matched single-channel control matches arm A, so on this task")
            print(f"  'channels' is a re-parameterization of 'more resources': the apparatus")
            print(f"  works, but the distinctive multi-channel claim is not supported here.")
            print(f"  The margin is below EPS on {n_within}/{len(per_seed_gap)} individual seeds.")
        else:
            print(f"  PASS: margin {margin:+.4f} >= EPS {EPS}.")
            print(f"  No matched single-channel control reaches arm A, so the win is not")
            print(f"  explained by parameters (D), memory (C), or the gate acting as a small")
            print(f"  RNN without routing (E). The margin clears EPS on "
                  f"{len(per_seed_gap) - n_within}/{len(per_seed_gap)} individual seeds.")
    if partial:
        print()
        print("  NOTE: this sweep is PARTIAL; the verdict covers completed seeds only.")
    print()
    print(f"  total wall clock: {wall / 60:.1f} min ({wall:.0f}s)")
    print(f"  raw per-seed records: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Capacity controls for the multi-channel claim.")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--results", default=RESULTS_FILE)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    seeds = list(range(args.seeds))

    print("=" * 100)
    print("Capacity control: does multi-channel beat matched single-channel resources?")
    print("  model, task, training loop and hyperparameters imported from test_instrument_v2")
    print(f"  decay={OPERATING_DECAY} (U!=I), randomized order, ITERS={ITERS}, "
          f"EPS={EPS}, threads={args.threads}")
    print(f"  seeds={seeds}")
    print(f"  results cache: {args.results}")
    print("=" * 100)
    print()

    info = verify()
    store = load_results(args.results)
    store["meta"] = dict(operating_decay=OPERATING_DECAY, iters=ITERS, eps=EPS,
                         seeds=seeds, nfeat_memory=NFEAT_MEMORY_MATCHED,
                         nfeat_param=NFEAT_PARAM_MATCHED,
                         started=time.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()
    total, done = len(ALL) * len(seeds), 0
    for c in ALL:
        kk = c["key"]
        store["results"].setdefault(kk, {})
        print(f"-- {c['label']}   (gate={c['gate']}, n_ch={c['n_ch']}, "
              f"n_feat={info[kk][2]}, params={info[kk][0]}, state={info[kk][1]}) --")
        for s in seeds:
            done += 1
            cached = store["results"][kk].get(str(s))
            if cached is not None and not args.force:
                print(f"   seed {s:>3}  [{done}/{total}]  cached "
                      f"({'ok' if cached.get('ok') else 'FAILED'})")
                continue
            r = run_one(kk, s)
            store["results"][kk][str(s)] = r
            save_results(args.results, store)
            el = time.time() - t0
            if r["ok"]:
                print(f"   seed {s:>3}  [{done}/{total}]  acc={r['acc']:.4f}   "
                      f"{r['secs']:.0f}s  (elapsed {el / 60:.1f}m)")
            else:
                print(f"   seed {s:>3}  [{done}/{total}]  *** FAILED: {r['error']}   "
                      f"{r['secs']:.0f}s  (elapsed {el / 60:.1f}m)")
        print()

    report(store, seeds, info, time.time() - t0, args.results)


if __name__ == "__main__":
    main()
