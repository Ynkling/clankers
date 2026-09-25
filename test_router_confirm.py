#!/usr/bin/env python
"""
test_router_confirm.py — a confirmatory test of an exploratory finding from
test_router_curriculum: on arm A, a learned channel gate reading LayerNorm'd embeddings
(A_ln), trained without stream labels, found the stream routing on 7/10 seeds. Three
questions, fixed before any result:
  C1  does a learned gate without labels beat single-channel BDH?
  C2  how reliably does it find the routing?
  C3  does the LayerNorm matter?

Run it directly:

    python test_router_confirm.py                   # CHECKs, projection, 120 runs, verdict
    python test_router_confirm.py --also other_machine/router_confirm_results.json

BACKGROUND
- Exploratory (test_router_curriculum, seeds 0-9, this machine, MAX_ITERS 24000): A_ln 7/10,
  discovering by steps 1200-6000; A 4/10; A_nudge (labelled positive control) 9/10; B 0/10.
  A_ln was a control there, not a candidate, so its 7/10 is exploratory. A_ln vs A was
  p = 0.18 (Fisher one-sided).
- Single-channel models bound 0/80 across all earlier tests, including the memory-matched
  E_mem (0/10 over both machines in test_channel_binding).
- Mechanism so far: every learned gate eventually leaves uniform; which split it settles on
  is decided in its first moves; a 5% stream-aligned lean decides it for stream routing 9/10
  times; the readout arms (A_ro) win only ~10%.
- Seed-level outcomes differ between machines. This test runs on one machine and its verdict
  is for this machine; --also pools a second machine's file later.

SUBSTRATE, TASK AND ARMS (reused, not rewritten)
  A, A_ln, A_nudge, B  exactly test_router_curriculum's arms (its run_spec and make_fn).
  E_mem                exactly test_channel_binding's: single channel, width N*S
                       (memory-matched to the k=2 arms), recurrent readout.
Task P=4, S=2, n_vals=16, n_q=1; MultiBDH n_layer=3, decay, N=256; lr 1e-3, BATCH 32;
evaluation every 1200 steps on eval_batch; early stop after 3 evaluations >= 0.95;
MAX_ITERS = 24000, as in the exploratory run.

SEEDS, all disjoint from the 0-9 used before
  A and A_ln   10-49 (40 each), paired: the same seed gives the same initial parameters and
               the same batch stream in both arms.
  A_nudge      10-29 (20).
  B and E_mem  10-19 (10 each).
Fixed before any result: the arms, the seeds, MAX_ITERS and every threshold below.

DISCOVERED = transition is not None AND final VAL cos < 0.5. B and E_mem count BOUND
(transition is not None).

PRE-REGISTERED VERDICT, three separate claims, each printed with its numbers
  INVALID  A_nudge discovers on fewer than 16/20: stop; none of C1-C3 is interpreted.
  C1 CHANNELS BEAT SINGLE CHANNEL  A_ln discovered (of 40) vs the better of B's and E_mem's
     bound counts (of 10), Fisher's exact test one-sided (A_ln higher): SUPPORTED if
     p < 0.05, else NOT SUPPORTED.
  C2 RELIABILITY of A_ln  RELIABLE >= 32/40; MAJORITY 20-31; MINORITY <= 19.
  C3 LAYERNORM EFFECT  A_ln vs A discovered (of 40 each), Fisher's exact test one-sided
     (A_ln higher): LN HELPS if p < 0.05, else NOT SHOWN.
Fisher's exact test and the exact McNemar test are implemented with math.comb.

DIAGNOSTICS, not part of the verdict: the paired A vs A_ln 2x2 table over seeds 10-49 with
an exact two-sided McNemar p; escape step per discovering run; failure modes (saddle /
locked on keys / other) per arm; collapsed runs per arm; exploratory + confirmatory pooled
counts from router_curriculum_results.json if present (not the verdict); --also pooled
counts.

LOGISTICS. Parallel single-thread workers as in test_binding_recipe.py; wall clock projected
before training (no drop rule). Results persist atomically to router_confirm_results.json
(gitignored).

(Results are recorded at the bottom of this docstring after the run.)
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch

import test_router_curriculum as trc
from test_router_curriculum import (
    TASK, ARCH, SUB_LR, make_fn, run_spec, time_arm, escape, get, count, mode_of, load_store,
    init_worker, makespan, cpu_model, same_weights, strip, save_results, fmt_step, EVAL_EVERY,
    DISC_COS,
)
from test_binding_onset import onset_run
from test_channel_binding import eval_batch, arms as cb_arms
from test_multilayer_binding import build

# ── Settings (fixed before any result) ───────────────────────────────────────
MAX_ITERS = 24000
SEEDS = dict(A=tuple(range(10, 50)), A_ln=tuple(range(10, 50)), A_nudge=tuple(range(10, 30)),
             B=tuple(range(10, 20)), E_mem=tuple(range(10, 20)))
INVALID_BELOW = 16
ALPHA = 0.05
RELIABLE_K, MAJORITY_K = 32, 20
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "router_confirm_results.json"
EXPLORATORY_FILE = "router_curriculum_results.json"
EXPLORATORY_SEEDS = tuple(range(10))

E_MEM = dict({a["key"]: a for a in cb_arms(TASK)}["E_mem"],
             label="E_mem      1ch, width N*S + readout (memory-matched)")
ARMS = [dict(trc.ARM[k]) for k in ("A", "A_ln", "A_nudge", "B")] + [E_MEM]
ARM = {a["key"]: a for a in ARMS}
CHANNEL_KEYS = ("A", "A_ln", "A_nudge")
SINGLE_KEYS = ("B", "E_mem")


# ── Exact tests, with math.comb ──────────────────────────────────────────────
def fisher_greater(a, n1, b, n2):
    """One-sided Fisher exact p that group 1's rate (a/n1) exceeds group 2's (b/n2):
    P(X >= a) for X ~ Hypergeometric(N = n1 + n2, K = a + b, draws = n1)."""
    N, K = n1 + n2, a + b
    tot = math.comb(N, n1)
    return sum(math.comb(K, x) * math.comb(N - K, n1 - x)
               for x in range(a, min(K, n1) + 1)) / tot


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p on the discordant pairs b, c: 2 * P(Bin(b+c, 1/2) <= min)."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


# ── Runs ─────────────────────────────────────────────────────────────────────
def spec(key, seed, iters):
    sp = dict(arm=key, seed=seed, iters=iters, T=None)
    if key == "E_mem":
        sp["arm_def"] = E_MEM
    return sp


# ── Verification ─────────────────────────────────────────────────────────────
def verify(pool):
    print("test_router_curriculum.py's verification (which runs test_router_discovery's, and so")
    print("on down to test_multilayer_binding's):")
    trc.verify(pool)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True
    t = TASK
    data = eval_batch(t)

    print("CHECK 32 the arms are unchanged: 60-step runs (evaluations every 20), seed 0, through")
    print("         this test's arm table vs test_router_curriculum's (A, A_ln, A_nudge, B) and")
    print("         test_channel_binding's (E_mem):")
    for key in ("A", "A_ln", "A_nudge", "B", "E_mem"):
        ref = ((lambda: build(t, {a["key"]: a for a in cb_arms(t)}["E_mem"], ARCH, 0))
               if key == "E_mem" else make_fn(trc.ARM[key], 0))
        m0, c0, _ = onset_run(t, ref, 0, max_iters=60, eval_every=20, data=data,
                              early_stop=False, lr=SUB_LR)
        m1, c1, _ = onset_run(t, make_fn(ARM[key], 0), 0, max_iters=60, eval_every=20,
                              data=data, early_stop=False, lr=SUB_LR)
        good = same_weights(m0, m1) and c0 == c1
        ok &= good
        src = "test_channel_binding" if key == "E_mem" else "test_router_curriculum"
        print(f"         {key:<8} vs {src:<22} weights equal {same_weights(m0, m1)}   curves "
              f"equal {c0 == c1}   N={m1.n_feat} k={m1.n_ch} readout={m1.gate_to_readout}  -> "
              f"{'IDENTICAL' if good else 'DIFFERS'}")
    print()

    print("CHECK 33 the seed sets, all disjoint from the exploratory seeds 0-9:")
    for key, ss in SEEDS.items():
        disj = not set(ss) & set(EXPLORATORY_SEEDS)
        ok &= disj
        print(f"         {key:<8} {len(ss):>2} seeds: {ss[0]}-{ss[-1]}   disjoint from 0-9: {disj}")
    paired = SEEDS["A"] == SEEDS["A_ln"]
    ok &= paired
    print(f"         A and A_ln use the same seeds (paired): {paired}")
    print()

    print("CHECK 34 the statistics code reproduces reference values:")
    refs = [("Fisher one-sided  7/10 vs  0/10", fisher_greater(7, 10, 0, 10), 0.00155),
            ("Fisher one-sided  7/10 vs  4/10", fisher_greater(7, 10, 4, 10), 0.18492),
            ("Fisher one-sided 15/20 vs  7/20", fisher_greater(15, 20, 7, 20), 0.01242),
            ("McNemar exact two-sided, 4 vs 1 discordant", mcnemar_exact(4, 1), 0.375)]
    for name, got, want in refs:
        good = abs(got - want) < 5e-6
        ok &= good
        print(f"         {name:<44} {got:.5f}  (reference {want})  -> {'OK' if good else 'WRONG'}")
    print()

    print("CHECK 35 the pairing: seed 10, A and A_ln start from bitwise-identical parameters and")
    print("         see bitwise-identical first 3 training batches:")
    ma, ml = make_fn(ARM["A"], 10)(), make_fn(ARM["A_ln"], 10)()
    pa, pl = dict(ma.named_parameters()), dict(ml.named_parameters())
    same_init = sorted(pa) == sorted(pl) and all(torch.equal(pa[n], pl[n]) for n in pa)
    fed = {}
    for key in ("A", "A_ln"):
        rec, mk = [], make_fn(ARM[key], 10)

        def mk2(mk=mk, rec=rec):
            mm = mk()
            mm.register_forward_pre_hook(
                lambda mod, args: rec.append(args[0].clone()) if mod.training else None)
            return mm
        onset_run(t, mk2, 10, max_iters=3, lr=SUB_LR)
        fed[key] = rec
    same_b = len(fed["A"]) == 3 and all(torch.equal(u, v) for u, v in zip(fed["A"], fed["A_ln"]))
    good = same_init and same_b and ml.gate_ln and not ma.gate_ln
    ok &= good
    print(f"         parameters identical: {same_init}   first 3 batches identical: {same_b}   "
          f"gate_ln A={ma.gate_ln} A_ln={ml.gate_ln}  -> {'PAIRED' if good else 'WRONG'}")
    print()

    print("CHECK 36 a worker's run is bit-identical to the same run here:")
    good = True
    for key in ("A_ln", "E_mem"):
        sp = spec(key, 11, 1200)
        rw = pool.submit(run_spec, sp).result()
        rh = run_spec(sp)
        g = strip(rw) == strip(rh) and rh["ok"]
        good &= g
        print(f"         {key:<6} seed 11, 1200 steps: equal {strip(rw) == strip(rh)}   acc "
              f"{rh.get('acc', float('nan')):.4f}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"


# ── Reporting ────────────────────────────────────────────────────────────────
def succ(r, key):
    """Discovered for channel arms, bound for the single-channel arms."""
    if r is None or not r.get("ok"):
        return False
    return bool(r.get("discovered")) if key in CHANNEL_KEYS else r["transition"] is not None


def tally(store, key, seeds):
    return sum(1 for s in seeds if succ(get(store, key, s), key))


def pooled_line(label, runs, key, seeds_here, store):
    here = tally(store, key, seeds_here)
    there = [(k, r) for k, r in runs.items() if k.split("|")[0] == key]
    k2 = sum(1 for _, r in there if succ(r, key))
    what = "discovered" if key in CHANNEL_KEYS else "bound"
    print(f"    {key:<8} {what:<10} here {here}/{len(seeds_here)}   {label} {k2}/{len(there)}   "
          f"pooled {here + k2}/{len(seeds_here) + len(there)}")


def report(store, wall, path, also):
    print("#" * 100)
    print("PER-SEED RAW RESULTS — every value, before any aggregate")
    print("#" * 100)
    print(f"  {'arm':<8} {'seed':>4} {'acc':>7} {'transition':>10} {'VAL cos':>8} {'sep':>7} "
          f"{'key_part':>8} {'spread':>7}  flags")
    for a in ARMS:
        k = a["key"]
        for s in SEEDS[k]:
            r = get(store, k, s)
            if r is None:
                print(f"  {k:<8} {s:>4}  NOT RUN")
                continue
            if not r.get("ok"):
                print(f"  {k:<8} {s:>4}  FAILED — {r['error']}")
                continue
            e = r.get("end")
            gs = ("      --      --       --      --" if e is None else
                  f"{e['val_cos']:>8.4f} {e['sep']:>7.4f} {e['key_part']:>8.4f} {e['spread']:>7.4f}")
            fl = (["DISCOVERED"] if r.get("discovered") else []) + (
                ["BOUND (single channel)"] if k in SINGLE_KEYS and r["transition"] else []) + (
                ["collapsed"] if r["collapsed"] else [])
            print(f"  {k:<8} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} {gs}  "
                  f"{' '.join(fl)}")
        print()

    print("=" * 100)
    print(f"COUNTS (discovered = bound AND final VAL cos < {DISC_COS}; single-channel arms count bound)")
    print("=" * 100)
    cnt = {}
    for a in ARMS:
        k, ss = a["key"], SEEDS[a["key"]]
        cnt[k] = tally(store, k, ss)
        rs = [get(store, k, s) for s in ss]
        failed = sum(1 for r in rs if r is not None and not r.get("ok"))
        done = sum(1 for r in rs if r is not None and r.get("ok"))
        what = "discovered" if k in CHANNEL_KEYS else "bound"
        print(f"  {a['label']:<56} {what:<10} {cnt[k]:>2}/{len(ss):<2}   completed {done}/{len(ss)}"
              + (f"   [{failed} FAILED]" if failed else ""))
    nA, nL, nN = len(SEEDS["A"]), len(SEEDS["A_ln"]), len(SEEDS["A_nudge"])
    print()

    print("#" * 100)
    print("PRE-REGISTERED VERDICT — three separate claims")
    print("#" * 100)
    verdict = {}
    print(f"  validity: A_nudge (positive control) {cnt['A_nudge']}/{nN}; INVALID below "
          f"{INVALID_BELOW}/{nN}.")
    if cnt["A_nudge"] < INVALID_BELOW:
        print(f"  *** INVALID: the positive control discovered on {cnt['A_nudge']}/{nN} < "
              f"{INVALID_BELOW}. None of C1-C3 is interpreted. ***")
        verdict["validity"] = "INVALID"
    else:
        verdict["validity"] = "VALID"
        best = max(SINGLE_KEYS, key=lambda k: cnt[k])
        nB = len(SEEDS[best])
        p1 = fisher_greater(cnt["A_ln"], nL, cnt[best], nB)
        c1 = "SUPPORTED" if p1 < ALPHA else "NOT SUPPORTED"
        print(f"  C1 CHANNELS BEAT SINGLE CHANNEL: A_ln {cnt['A_ln']}/{nL} vs the better single-"
              f"channel arm {best} {cnt[best]}/{nB} (B {cnt['B']}/{len(SEEDS['B'])}, E_mem "
              f"{cnt['E_mem']}/{len(SEEDS['E_mem'])}); Fisher one-sided p = {p1:.3g}")
        print(f"     *** C1: {c1} ***")
        c2 = ("RELIABLE" if cnt["A_ln"] >= RELIABLE_K else "MAJORITY" if cnt["A_ln"] >= MAJORITY_K
              else "MINORITY")
        print(f"  C2 RELIABILITY of A_ln: {cnt['A_ln']}/{nL} (RELIABLE >= {RELIABLE_K}, MAJORITY "
              f"{MAJORITY_K}-{RELIABLE_K - 1}, MINORITY <= {MAJORITY_K - 1})")
        print(f"     *** C2: {c2} ***")
        p3 = fisher_greater(cnt["A_ln"], nL, cnt["A"], nA)
        c3 = "LN HELPS" if p3 < ALPHA else "NOT SHOWN"
        print(f"  C3 LAYERNORM EFFECT: A_ln {cnt['A_ln']}/{nL} vs A {cnt['A']}/{nA}; Fisher one-sided "
              f"p = {p3:.3g}")
        print(f"     *** C3: {c3} ***")
        verdict.update(C1=c1, C1_p=p1, C2=c2, C3=c3, C3_p=p3, counts=cnt)
    store["verdict"] = verdict
    save_results(path, store)
    print()

    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    both = a_only = l_only = neither = 0
    for s in SEEDS["A"]:
        da, dl = succ(get(store, "A", s), "A"), succ(get(store, "A_ln", s), "A_ln")
        both += da and dl
        a_only += da and not dl
        l_only += dl and not da
        neither += not da and not dl
    print(f"  paired A vs A_ln over seeds {SEEDS['A'][0]}-{SEEDS['A'][-1]}:")
    print(f"                  A_ln discovered   A_ln not")
    print(f"    A discovered  {both:>15}   {a_only:>8}")
    print(f"    A not         {l_only:>15}   {neither:>8}")
    print(f"    discordant: A only {a_only}, A_ln only {l_only};  exact McNemar two-sided p = "
          f"{mcnemar_exact(a_only, l_only):.3g}")
    print()
    print(f"  escape step (first evaluation with VAL cos < {DISC_COS}) per discovering run:")
    for k in CHANNEL_KEYS:
        steps = []
        for s in SEEDS[k]:
            r = get(store, k, s)
            if r and r.get("ok") and r.get("discovered"):
                steps.append((s, escape(r), r["transition"]))
        if steps:
            print(f"    {k:<8} " + "  ".join(f"s{s}:{fmt_step(e)}/{fmt_step(tr)}" for s, e, tr in steps)
                  + "   (escape/transition)")
        else:
            print(f"    {k:<8} none")
    print()
    print("  failure modes of non-discovering channel runs (saddle: spread < 0.1; locked on keys: "
          "key_part > 0.8 and sep < 0.2; other):")
    for k in CHANNEL_KEYS:
        modes = {"saddle": 0, "locked on keys": 0, "other": 0}
        for s in SEEDS[k]:
            r = get(store, k, s)
            if r and r.get("ok") and not r.get("discovered"):
                modes[mode_of(r)] += 1
        print(f"    {k:<8} " + "   ".join(f"{m}: {c}" for m, c in modes.items()))
    print(f"  collapsed runs (final accuracy < 0.15): "
          + ", ".join(f"{a['key']} {count(store, a['key'], SEEDS[a['key']], 'collapsed')}"
                      for a in ARMS))
    print()
    print(f"  EXPLORATORY + CONFIRMATORY pooled counts (seeds 0-9 from {EXPLORATORY_FILE} + this "
          f"test) — NOT the verdict:")
    if os.path.exists(EXPLORATORY_FILE):
        with open(EXPLORATORY_FILE) as f:
            ex = json.load(f)
        runs = {k: r for k, r in ex.get("runs", {}).items()
                if k.split("|")[0] in ("A", "A_ln", "A_nudge", "B")}
        for k in ("A", "A_ln", "A_nudge", "B"):
            pooled_line("exploratory", runs, k, SEEDS[k], store)
        print(f"    E_mem    not in {EXPLORATORY_FILE}")
    else:
        print(f"    {EXPLORATORY_FILE} not present: skipped.")
    print()
    if also:
        print(f"  --also: pooled with {also}:")
        try:
            with open(also) as f:
                other = json.load(f)
            print(f"    other machine: {other.get('meta', {}).get('cpu', 'unknown')}, git "
                  f"{other.get('meta', {}).get('git')}")
            for a in ARMS:
                pooled_line("other machine", other.get("runs", {}), a["key"], SEEDS[a["key"]], store)
        except Exception as e:
            print(f"    could not read {also}: {e}")
        print()

    print("=" * 100)
    print(f"CURVES — channel runs: accuracy x100 and VAL cos x100 at each evaluation every "
          f"{EVAL_EVERY} steps")
    print("=" * 100)
    for k in CHANNEL_KEYS:
        for s in SEEDS[k]:
            r = get(store, k, s)
            if not (r and r.get("ok")):
                continue
            ev = {x["step"]: x for x in r["stats"] if isinstance(x["step"], int) and x["step"] > 0}
            accs = "".join(f"{round(acc * 100):>4}" for _, acc, _ in r["curve"])
            coss = "".join(f"{round(ev[st]['val_cos'] * 100):>4}" if st in ev else "  --"
                           for st, _, _ in r["curve"])
            print(f"  {k:<8} s{s:<2} acc {accs}")
            print(f"  {'':<8}     cos {coss}   -> "
                  f"{'bound @' + str(r['transition']) if r['transition'] else 'not bound'}"
                  f"{'  DISCOVERED' if r.get('discovered') else ''}")
    print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: "
          f"{path}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Confirm: does a label-free LN gate find the routing?")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--also", default=None,
                    help="a second machine's router_confirm_results.json, for pooled counts")
    args = ap.parse_args()
    torch.set_num_threads(1)
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()

    print("=" * 100)
    print("Router confirm: does a label-free gate on LayerNorm'd embeddings find the routing?")
    print(f"  test_router_curriculum's config (S=2 P=4 n_vals=16 n_q=1, n_layer=3 decay N=256, lr "
          f"{SUB_LR}), MAX_ITERS={MAX_ITERS}")
    print("  seeds: " + "   ".join(f"{k} {v[0]}-{v[-1]} ({len(v)})" for k, v in SEEDS.items()))
    print(f"  torch {torch.__version__}   CPU {cpu_model()}   {args.workers} worker processes x 1 "
          f"thread   git {head}")
    print("=" * 100)
    print()

    os.environ.setdefault("PYTHONWARNINGS", "ignore:Failed to initialize NumPy")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                             initializer=init_worker) as pool:
        verify(pool)
        store = load_store(args.results)
        store["meta"] = dict(torch=torch.__version__, cpu=cpu_model(), workers=args.workers,
                             git=head, max_iters=MAX_ITERS,
                             seeds={k: list(v) for k, v in SEEDS.items()},
                             started=time.strftime("%Y-%m-%d %H:%M:%S"),
                             note="seed-level outcomes differ between machines")
        save_results(args.results, store)
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — time_per_step (200 steps) inside the pool, evaluation charged once per "
              f"{EVAL_EVERY}; worst case = every run to {MAX_ITERS}")
        print("=" * 100)
        keys = [a["key"] for a in ARMS]
        costs = dict(zip(keys, pool.map(time_arm, [E_MEM if k == "E_mem" else k for k in keys])))
        for k in keys:
            per, fixed = costs[k]
            print(f"  {k:<8} {per * 1000:6.1f} ms/step" + (f"   + {fixed:.1f}s pre-fit per run"
                                                         if fixed else "")
                  + f"   x {len(SEEDS[k])} seeds")
        durs = [MAX_ITERS * costs[k][0] + costs[k][1] for k in keys for _ in SEEDS[k]]
        print(f"  {len(durs)} runs: serial {sum(durs) / 3600:.2f} h, projected wall clock on "
              f"{args.workers} workers {makespan(durs, args.workers) / 3600:.2f} h (no drop rule)")
        store["meta"]["projected_wall_h"] = makespan(durs, args.workers) / 3600
        print()

        print("=" * 100)
        print(f"RUNS — {len(durs)} runs, MAX_ITERS={MAX_ITERS}")
        print("=" * 100)
        todo = []
        for k in keys:
            for s in SEEDS[k]:
                if get(store, k, s) is not None and not args.force:
                    print(f"   {k:<8} seed {s}  cached")
                else:
                    todo.append(spec(k, s, MAX_ITERS))
        todo.sort(key=lambda sp: -costs[sp["arm"]][0])
        futs = {pool.submit(run_spec, sp): sp for sp in todo}
        for f in as_completed(futs):
            sp = futs[f]
            try:
                rec = f.result()
            except Exception as e:
                rec = dict(ok=False, seed=sp["seed"], error=f"worker {type(e).__name__}: {e}")
            store["runs"][f"{sp['arm']}|{sp['seed']}"] = rec
            save_results(args.results, store)
            el = (time.time() - t0) / 60
            if rec.get("ok"):
                e = rec.get("end")
                gs = ("" if e is None else f"  VALcos={e['val_cos']:.3f} sep={e['sep']:.3f} "
                      f"key_part={e['key_part']:.3f}")
                tr = rec["transition"]
                print(f"   {sp['arm']:<8} seed {sp['seed']:<2}  "
                      f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} "
                      f"@{rec['stopped_at']:<5}{gs}{'  DISCOVERED' if rec.get('discovered') else ''}"
                      f"  {rec['secs']:.0f}s  (elapsed {el:.1f}m)")
            else:
                print(f"   {sp['arm']:<8} seed {sp['seed']:<2}  *** FAILED: {rec['error']}   "
                      f"(elapsed {el:.1f}m)")
        print()

    report(store, time.time() - t0, args.results, args.also)


if __name__ == "__main__":
    main()
