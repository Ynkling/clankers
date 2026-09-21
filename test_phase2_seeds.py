#!/usr/bin/env python
"""
test_phase2_seeds.py — multi-seed replication of the Phase II (experiment 8) result.

Run it directly, with nothing else set up:

    python test_phase2_seeds.py                 # 5 core conditions, 10 seeds
    python test_phase2_seeds.py --seeds 20      # more seeds
    python test_phase2_seeds.py --full          # adds the U=I / U!=I 2x2
    python test_phase2_seeds.py --force         # ignore the cache, retrain everything
    python test_phase2_seeds.py --threads 4     # torch thread count

WHY. Phase II is the first positive result after seven negatives, and it is currently
reported from single runs with no seed variance. A first positive is exactly the kind of
result that should not be trusted until it replicates, so this re-runs it across a seed
set and reports the spread rather than a point estimate.

NO SECOND IMPLEMENTATION. Everything — the model, the task, the decay, the randomized
block order, the recurrent-state gate, the training loop, the hyperparameters — is
imported from test_instrument_v2.py. Nothing about the instrument is redefined here, so
this script cannot drift from the thing it is replicating.

Two things had to be refactored in test_instrument_v2.py to make that possible. Both are
backward compatible, and CHECK 0 below prints the evidence:

  * OPERATING_DECAY was hoisted from inside the __main__ block to a module-level
    constant. It was previously derived by the floor re-derivation and never exposed, so
    an importing script would have had to hard-code 0.95 and would silently diverge if
    the original ever changed. The main block still cross-checks the constant against the
    measured sweep and says so if they disagree.
  * run() gained an optional `ckpts` argument. It previously hard-coded 11 evenly spaced
    checkpoints, one every 120 steps, which cannot resolve the report's claim that
    separation emerges "between steps 40 and 80". Passing ckpts=None reproduces the old
    schedule exactly; this script passes one that is dense early.

Nothing else in test_instrument_v2.py was touched.

WHAT IS MEASURED. Per seed, per condition: per-stream query loss and accuracy, and for
gated conditions the gate cosine between streams resolved BY TOKEN ROLE. Separation only
matters on the KEY and VAL tokens, whose identity is shared between the streams so
routing is the only thing that can tell them apart; CTX tokens already mark their own
stream by identity. An aggregate cosine averaged over all roles is diluted by the CTX
tokens and previously produced a wrong verdict in this series — it read 0.7676 and
printed "not separated" for a model sitting exactly on the ceiling — so no all-role
aggregate is reported here.

Lift is computed PER SEED against that same seed's own floor and ceiling, never against
pooled references, so a seed whose floor or ceiling moved is not scored against another
seed's. The U=I conditions are scored against U=I floor/ceiling (run under --full).

Separation emergence is the first checkpoint at which the cosine drops below SEP_COS,
reported separately for KEY, for VAL, and for both-at-once, per seed.

FAILURES are never dropped. An exception, or a non-finite loss or accuracy, is recorded
against its seed with the error text, printed in the tables, and counted; a sweep that
did not complete every seed is labelled PARTIAL. Note "failure" means the run errored or
produced non-finite numbers. A run that simply scores badly is a RESULT, not a failure,
and is reported as one — this script has no convergence threshold that could quietly
discard an inconvenient seed.

PERSISTENCE. Raw per-seed records are written to phase2_seeds_results.json after every
completed run, atomically, so an interrupted sweep loses nothing and the analysis can be
recomputed without retraining. On startup, seeds already recorded for a condition are
skipped unless --force is passed.
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

# The instrument, imported whole — nothing below redefines any part of it.
import test_instrument_v2 as tiv
from test_instrument_v2 import OPERATING_DECAY, ITERS, SEP_COS

# ── Settings (all overridable on the command line) ───────────────────────────
N_SEEDS = 10
THREADS = 1                        # single-process CPU by default
RESULTS_FILE = "phase2_seeds_results.json"
REPLICATION_LIFT = 0.70            # lift a seed must exceed to count as replicating

# Checkpoints: dense where separation is claimed to emerge, coarse afterwards.
CKPTS = sorted(set(
    list(range(0, 101, 5)) + list(range(100, 301, 20)) +
    list(range(300, ITERS, 100)) + [ITERS - 1]
))

# ── Conditions. `core` ones carry the headline; the rest need --full. ────────
# gate: 'none' (single channel) | 'uniform' | 'perfect' | 'recurrent' | 'token'
# lift_ref: (floor key, ceiling key) this condition's per-seed lift is scored against.
C = [
    dict(key="floor_single", short="floor1ch", core=True, gate="none", n_ch=1,
         decay=OPERATING_DECAY, label="1. single-channel baseline (floor)",
         lift_ref=("floor_uniform", "ceil_perfect")),
    dict(key="floor_uniform", short="floorUnif", core=True, gate="uniform", n_ch=2,
         decay=OPERATING_DECAY, label="2. MC k=2, uniform gate (floor)",
         lift_ref=("floor_uniform", "ceil_perfect")),
    dict(key="ceil_perfect", short="ceilPerf", core=True, gate="perfect", n_ch=2,
         decay=OPERATING_DECAY, label="3. MC k=2, perfect gate (ceiling)",
         lift_ref=("floor_uniform", "ceil_perfect")),
    dict(key="recurrent_Uneq", short="rec[U!=I]", core=True, gate="recurrent", n_ch=2,
         decay=OPERATING_DECAY, label="4. MC k=2, recurrent-state gate [U!=I]  <-- HEADLINE",
         lift_ref=("floor_uniform", "ceil_perfect")),
    dict(key="token_Uneq", short="tok[U!=I]", core=True, gate="token", n_ch=2,
         decay=OPERATING_DECAY, label="5. MC k=2, token-identity gate [U!=I]  <-- CONTROL",
         lift_ref=("floor_uniform", "ceil_perfect")),
    # --full: the 2x2 closing the attribution gap. Conditions 4 and 5 ARE its U!=I
    # column, reused rather than retrained; these add the U=I column plus decay-matched
    # floor/ceiling so U=I lift is scored against its own references.
    dict(key="recurrent_Ueq", short="rec[U=I]", core=False, gate="recurrent", n_ch=2,
         decay=1.0, label="6. MC k=2, recurrent-state gate [U=I]",
         lift_ref=("floor_uniform_Ueq", "ceil_perfect_Ueq")),
    dict(key="token_Ueq", short="tok[U=I]", core=False, gate="token", n_ch=2,
         decay=1.0, label="7. MC k=2, token-identity gate [U=I]",
         lift_ref=("floor_uniform_Ueq", "ceil_perfect_Ueq")),
    dict(key="floor_uniform_Ueq", short="floorU=I", core=False, gate="uniform", n_ch=2,
         decay=1.0, label="8. MC k=2, uniform gate [U=I] (floor for 6/7)",
         lift_ref=("floor_uniform_Ueq", "ceil_perfect_Ueq")),
    dict(key="ceil_perfect_Ueq", short="ceilU=I", core=False, gate="perfect", n_ch=2,
         decay=1.0, label="9. MC k=2, perfect gate [U=I] (ceiling for 6/7)",
         lift_ref=("floor_uniform_Ueq", "ceil_perfect_Ueq")),
]
BY_KEY = {c["key"]: c for c in C}
HEADLINE, CONTROL = "recurrent_Uneq", "token_Uneq"


# ── Persistence ───────────────────────────────────────────────────────────────
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
    os.replace(tmp, path)          # atomic: an interrupt cannot truncate the file


# ── One (condition, seed) run ────────────────────────────────────────────────
def first_below(trace, fields):
    """First checkpoint step where every field in `fields` is below SEP_COS."""
    for p in trace:
        if all(p.get(f, 1.0) is not None and p.get(f, 1.0) < SEP_COS for f in fields):
            return p["step"]
    return None


def run_one(key, seed):
    c = BY_KEY[key]
    t0 = time.time()
    try:
        r = tiv.run(c["gate"], "sym", c["n_ch"], c["decay"], None, seed,
                    anneal=False, ckpts=CKPTS)
        vals = [r["l1"], r["l2"], r["a1"], r["a2"]]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            raise ValueError(f"non-finite result: l1={r['l1']} l2={r['l2']} "
                             f"a1={r['a1']} a2={r['a2']}")
        tr = r["trace"]
        last = tr[-1] if tr else {}
        return dict(ok=True, seed=seed, l1=r["l1"], l2=r["l2"], a1=r["a1"], a2=r["a2"],
                    acc=(r["a1"] + r["a2"]) / 2.0,
                    key_cos=last.get("key_cos"), val_cos=last.get("val_cos"),
                    ctx_cos=last.get("ctx_cos"),
                    emerge_key=first_below(tr, ["key_cos"]) if tr else None,
                    emerge_val=first_below(tr, ["val_cos"]) if tr else None,
                    emerge_both=first_below(tr, ["key_cos", "val_cos"]) if tr else None,
                    secs=time.time() - t0)
    except Exception as e:
        return dict(ok=False, seed=seed, error=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc()[-1500:], secs=time.time() - t0)


# ── Statistics ───────────────────────────────────────────────────────────────
def stats(xs):
    """mean, population std (as used throughout this series), min, max, n."""
    xs = [x for x in xs
          if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)]
    if not xs:
        return None
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    return dict(mean=m, std=sd, min=min(xs), max=max(xs), n=len(xs),
                identical=(max(xs) - min(xs) < 1e-12))


def mean_std_cell(st, prec=4, signed=False):
    """'identical: X' when every seed agreed, else 'mean +- std'."""
    if st is None:
        return "no data"
    sign = "+" if signed else ""
    if st["identical"]:
        return f"identical: {st['mean']:{sign}.{prec}f}"
    return f"{st['mean']:{sign}.{prec}f} +- {st['std']:.{prec}f}"


def per_seed_lift(store, key, seed):
    """Lift for one seed against THAT seed's own floor and ceiling."""
    fl_k, ce_k = BY_KEY[key]["lift_ref"]
    get = lambda kk: store["results"].get(kk, {}).get(str(seed))
    me, fl, ce = get(key), get(fl_k), get(ce_k)
    if not (me and fl and ce):
        return None
    if not (me.get("ok") and fl.get("ok") and ce.get("ok")):
        return None
    span = ce["acc"] - fl["acc"]
    if span <= 1e-3:
        return None                 # degenerate span: lift would carry no information
    return (me["acc"] - fl["acc"]) / span


# ── Reporting ────────────────────────────────────────────────────────────────
def report(store, conds, seeds, wall, results_path):
    keys = [c["key"] for c in conds]
    get = lambda k, s: store["results"].get(k, {}).get(str(s))
    W = max(11, max(len(c["short"]) for c in conds) + 2)

    def table(title, cell_fn, subtitle=None):
        print("=" * 100)
        print(title)
        if subtitle:
            print(subtitle)
        print("=" * 100)
        print("  seed  " + "".join(f"{BY_KEY[k]['short']:>{W}}" for k in keys))
        for s in seeds:
            print(f"  {s:>4}  " + "".join(f"{cell_fn(k, s):>{W}}" for k in keys))
        print()

    def acc_cell(k, s):
        r = get(k, s)
        if r is None:
            return "--"
        return "FAIL" if not r.get("ok") else f"{r['acc']:.4f}"

    def lift_cell(k, s):
        lf = per_seed_lift(store, k, s)
        return "--" if lf is None else f"{lf:+.3f}"

    table("PER-SEED RAW ACCURACY — every value, not just aggregates",
          acc_cell)
    table("PER-SEED LIFT — each seed scored against ITS OWN floor and ceiling",
          lift_cell)

    gated = [k for k in keys if BY_KEY[k]["gate"] in ("recurrent", "token")]
    if gated:
        print("=" * 100)
        print("PER-SEED GATE COSINE BY TOKEN ROLE")
        print("  No all-role aggregate is shown: it is diluted by CTX tokens, which need")
        print("  not split, and previously gave a wrong verdict in this series.")
        print("=" * 100)
        head = "  seed  "
        for k in gated:
            head += f"{BY_KEY[k]['short'] + ' KEY':>16}{BY_KEY[k]['short'] + ' VAL':>16}"
        print(head)
        for s in seeds:
            row = f"  {s:>4}  "
            for k in gated:
                r = get(k, s)
                for fld in ("key_cos", "val_cos"):
                    v = None if (r is None or not r.get("ok")) else r.get(fld)
                    row += f"{'--' if v is None else f'{v:.4f}':>16}"
            print(row)
        print()

        print("=" * 100)
        print(f"SEPARATION EMERGENCE — first checkpoint with cosine < {SEP_COS}")
        print(f"  (tests the single-run claim that this happens 'between steps 40 and 80')")
        print("=" * 100)
        for k in gated:
            print(f"  {BY_KEY[k]['label']}")
            for role, fld in (("KEY", "emerge_key"), ("VAL", "emerge_val"),
                              ("BOTH", "emerge_both")):
                vals, never = [], 0
                for s in seeds:
                    r = get(k, s)
                    v = None if (r is None or not r.get("ok")) else r.get(fld)
                    vals.append(v)
                    if v is None and r is not None and r.get("ok"):
                        never += 1
                st = stats([v for v in vals if v is not None])
                line = ("never separated on any seed" if st is None
                        else f"{mean_std_cell(st, 1)}  [min {st['min']:.0f}, "
                             f"max {st['max']:.0f}, n={st['n']}]")
                print(f"     {role:>4}: {line}"
                      + (f"   (never separated on {never} seed(s))" if never else ""))
                print(f"           per seed: {vals}")
            print()

    # ── Paste-ready summary ──────────────────────────────────────────────────
    print("=" * 100)
    print("SUMMARY — paste-ready")
    print("=" * 100)
    LBLW = max(46, max(len(c["label"]) for c in conds) + 1)
    hdr = f"  {'condition':<{LBLW}} {'mean +- std':>24} {'min':>8} {'max':>8} {'n':>4}"
    for title, valfn, prec, signed in (("ACCURACY", lambda k, s: (
            get(k, s)["acc"] if get(k, s) and get(k, s).get("ok") else None), 4, False),
            ("LIFT (per-seed refs)", lambda k, s: per_seed_lift(store, k, s), 3, True)):
        print(f"  [{title}]")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for k in keys:
            xs = [valfn(k, s) for s in seeds]
            nfail = sum(1 for s in seeds
                        if get(k, s) is not None and not get(k, s).get("ok"))
            st = stats([x for x in xs if x is not None])
            if st is None:
                cell, mn, mx, n = "no completed seeds", "--", "--", 0
                print(f"  {BY_KEY[k]['label']:<{LBLW}} {cell:>24} {mn:>8} {mx:>8} {n:>4}"
                      + (f"   [{nfail} FAILED]" if nfail else ""))
            else:
                sgn = "+" if signed else ""
                print(f"  {BY_KEY[k]['label']:<{LBLW}} {mean_std_cell(st, prec, signed):>24} "
                      f"{st['min']:>8.{prec}f} {st['max']:>8.{prec}f} {st['n']:>4}"
                      + (f"   [{nfail} FAILED]" if nfail else ""))
        print()

    # ── Verdict ──────────────────────────────────────────────────────────────
    attempted = len(keys) * len(seeds)
    completed = sum(1 for k in keys for s in seeds
                    if get(k, s) is not None and get(k, s).get("ok"))
    partial = completed < attempted
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"  {completed}/{attempted} runs completed"
          + ("   *** PARTIAL SWEEP — see failures below ***" if partial else ""))
    if partial:
        for k in keys:
            for s in seeds:
                r = get(k, s)
                if r is None:
                    print(f"    {k} seed {s}: NOT RUN")
                elif not r.get("ok"):
                    print(f"    {k} seed {s}: FAILED — {r['error']}")
    print()

    h_lifts = [per_seed_lift(store, HEADLINE, s) for s in seeds]
    h_ok = [x for x in h_lifts if x is not None]
    c_ok = [x for x in (per_seed_lift(store, CONTROL, s) for s in seeds) if x is not None]
    v_cos, bad = [], []
    for s in seeds:
        r = get(HEADLINE, s)
        if r and r.get("ok") and r.get("val_cos") is not None:
            v_cos.append(r["val_cos"])
    n_rep = sum(1 for x in h_ok if x > REPLICATION_LIFT)
    n_sep = sum(1 for c in v_cos if c < SEP_COS)
    for s in seeds:
        r, lf = get(HEADLINE, s), per_seed_lift(store, HEADLINE, s)
        if r is None or not r.get("ok"):
            continue
        if lf is None or lf <= REPLICATION_LIFT or (r.get("val_cos") is not None
                                                    and r["val_cos"] >= SEP_COS):
            bad.append((s, lf, r))

    h_fail = sum(1 for s in seeds
                 if get(HEADLINE, s) is not None and not get(HEADLINE, s).get("ok"))
    h_missing = len(seeds) - len(h_ok)
    if not h_ok:
        print("  NO VERDICT: the headline condition produced no scorable seed.")
    elif not bad:
        st, cst = stats(h_ok), stats(c_ok)
        caveat = ("" if h_missing == 0 else
                  f" of {len(seeds)} requested — {h_fail} failed, "
                  f"{h_missing - h_fail} unscorable; SEE ABOVE")
        print(f"  HEADLINE REPLICATES on {n_rep}/{len(h_ok)} scorable seeds{caveat}.")
        print(f"  Recurrent-state gate lift {mean_std_cell(st, 3, True)} "
              f"(min {st['min']:+.3f}), VAL cosine below {SEP_COS} on "
              f"{n_sep}/{len(v_cos)} seeds;")
        print(f"  token-gate control lift "
              f"{mean_std_cell(cst, 3, True) if cst else 'n/a'}.")
        if h_missing:
            print(f"  This is NOT a clean {len(seeds)}-seed replication: "
                  f"{h_missing} seed(s) did not contribute.")
    else:
        print(f"  HEADLINE DOES NOT FULLY REPLICATE: lift > {REPLICATION_LIFT} on "
              f"{n_rep}/{len(h_ok)} scorable seeds,")
        print(f"  gate separates (VAL cosine < {SEP_COS}) on {n_sep}/{len(v_cos)}. "
              f"It failed on {len(bad)} seed(s), as follows:")
        for s, lf, r in bad:
            print(f"    seed {s}: lift={'n/a' if lf is None else f'{lf:+.3f}'}  "
                  f"acc={r['acc']:.4f}  VALcos={r['val_cos']}  KEYcos={r['key_cos']}  "
                  f"emerge(both)={r['emerge_both']}")
    if partial:
        print()
        print("  NOTE: this sweep is PARTIAL; the verdict covers completed seeds only.")
    print()
    print(f"  total wall clock: {wall / 60:.1f} min ({wall:.0f}s)")
    print(f"  raw per-seed records: {results_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Multi-seed replication of Phase II.")
    ap.add_argument("--seeds", type=int, default=N_SEEDS, help=f"default {N_SEEDS}")
    ap.add_argument("--full", action="store_true", help="also run the U=I / U!=I 2x2")
    ap.add_argument("--force", action="store_true", help="retrain even if cached")
    ap.add_argument("--threads", type=int, default=THREADS, help=f"default {THREADS}")
    ap.add_argument("--results", default=RESULTS_FILE)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    seeds = list(range(args.seeds))
    conds = [c for c in C if c["core"] or args.full]

    print("=" * 100)
    print("Phase II multi-seed replication (experiment 8)")
    print("  instrument imported from test_instrument_v2.py — nothing redefined here")
    print(f"  decay(U!=I)={OPERATING_DECAY}  ITERS={ITERS}  SEP_COS={SEP_COS}  "
          f"threads={args.threads}")
    print(f"  seeds={seeds}")
    print(f"  conditions={len(conds)} "
          f"({'core + full 2x2' if args.full else 'core only — pass --full for the 2x2'})")
    print(f"  results cache: {args.results}")
    print("=" * 100)

    default_ckpts = sorted({int(i * (ITERS - 1) / (tiv.N_CKPT - 1))
                            for i in range(tiv.N_CKPT)})
    print("\nCHECK 0  the refactor of test_instrument_v2.py is backward compatible")
    print(f"         OPERATING_DECAY imported          = {OPERATING_DECAY}")
    print(f"         run(ckpts=None) default schedule  = {default_ckpts}")
    print(f"         schedule this script passes       = {CKPTS[:8]} ... {CKPTS[-2:]}"
          f"   ({len(CKPTS)} points)")
    print("         -> the instrument itself is unchanged; only checkpoint density and")
    print("            where the operating decay is defined differ.\n")

    store = load_results(args.results)
    store["meta"] = dict(operating_decay=OPERATING_DECAY, iters=ITERS, sep_cos=SEP_COS,
                         ckpts=CKPTS, seeds=seeds, full=bool(args.full),
                         replication_lift=REPLICATION_LIFT,
                         started=time.strftime("%Y-%m-%d %H:%M:%S"))
    t_start = time.time()
    total, done = len(conds) * len(seeds), 0

    for c in conds:
        key = c["key"]
        store["results"].setdefault(key, {})
        print(f"-- {c['label']}   (gate={c['gate']}, k={c['n_ch']}, decay={c['decay']}) --")
        for seed in seeds:
            done += 1
            cached = store["results"][key].get(str(seed))
            if cached is not None and not args.force:
                print(f"   seed {seed:>3}  [{done}/{total}]  cached "
                      f"({'ok' if cached.get('ok') else 'FAILED'})")
                continue
            r = run_one(key, seed)
            store["results"][key][str(seed)] = r
            save_results(args.results, store)       # persist after EVERY run
            el = time.time() - t_start
            if r["ok"]:
                extra = ""
                if r["val_cos"] is not None:
                    extra = (f"  KEYcos={r['key_cos']:.4f} VALcos={r['val_cos']:.4f}"
                             f"  emerge={r['emerge_both']}")
                print(f"   seed {seed:>3}  [{done}/{total}]  acc={r['acc']:.4f}{extra}"
                      f"   {r['secs']:.0f}s  (elapsed {el / 60:.1f}m)")
            else:
                print(f"   seed {seed:>3}  [{done}/{total}]  *** FAILED: {r['error']}"
                      f"   {r['secs']:.0f}s  (elapsed {el / 60:.1f}m)")
        print()

    report(store, conds, seeds, time.time() - t_start, args.results)


if __name__ == "__main__":
    main()
