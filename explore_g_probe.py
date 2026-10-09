#!/usr/bin/env python
"""
explore_g_probe.py — EXPLORATORY, not a result. Session G, screen S55: is the header-layout stream cue present in the
residual stream at the positions where the gate must act (KEY and VAL), and at which layer? Everything below was fixed and
committed before any run.

BACKGROUND. On the header layout (explore_far_cue.HeaderTask: each stream's block [CTX_s, K, V, K, V, ...]) the window gate
(LOCAL3) sees the stream token only for each block's first pair; batch 2's far_L3 bound 2/10 by key splits. Session F's S53
diagnostic found stream-tagged keys at layer 3 of a trained single delta channel (cosine of the two writes' keys 0.16 at
layer 3 vs 0.90 at layer 1), and S59 found a learned decay that clears at CTX tokens; both predict the cue is present in
the stack at layer 3. S56's per-layer gate reads LN(r^(l)), so where the cue is present decides where the gate must read.
Task (the user's instruction of 9 October): the FIXED header layout at P = 8, S = 2 (HeaderTask(8, S=2, n_vals=16, n_q=1),
L = 37), budget 24000, the Hebbian perfect gate as validity (Session F's S53: 3/3 within 48000, at 14400-24000).

ARMS (explore_g_probe_child; 1 torch thread per run; Adam; evaluation every 1200 on 2048 held-out queries; early stop
after 3 evaluations >= 0.95)
  FAR_L3_P4  batch 2's far_L3 runs reproduced (P = 4 header, LOCAL3, lr 1e-3, 24000; seeds 160-169 are batch 2's, rerun
             only to reproduce its records — no new outcome is made on them). Each rerun's curve must equal its record
             (rec["repro"]); a run that does not reproduce is reported and excluded from the readings.
  L3S_P8     LOCAL3 + SLOW (S43's recipe, no hinge) on the P = 8 header, seeds 360-369, 24000.
  ORACLE_P8  the perfect gate on the P = 8 header, seeds 360-361, 24000 (validity).

THE PROBE (explore_g_common.probe). Features: the raw embedding (emb), the residual stream entering layer l = 1, 2, 3
(L1 = LN(embedding), L2, L3; the stream x at the top of bdh.BDH.forward's loop) and the final layer's output (final, the
lm_head input), recomputed by a line-for-line copy of the loop (CHECK: its logits equal the model's bit for bit). Positions:
every body KEY position and every body VAL position (16 pairs at P = 8, 8 at P = 4). Label: the true stream (the block's
CTX token). 2000 probe sequences (torch.Generator 55_000), split by SEQUENCE 1600 / 400 (generator 55_001); sklearn
LogisticRegression (lbfgs, L2, C = 1, max_iter 5000) on standardised features; accuracy on the held-out 20% (6400 positions
at P = 8, 3200 at P = 4), with a Wilson 95% interval on positions (positions of one sequence are not independent: the
interval is a lower bound on the uncertainty). Also by the pair's index within its block (distance from the header), and
the same probe on the seed's model at initialisation (descriptive). Cosine similarity between stream-mates is not used.

READINGS (fixed now; per arm, on the median over the arm's usable runs of the held-out accuracy; the count of runs that
meet each criterion is printed with Wilson and band)
  "the cue is present at layer l"   median accuracy >= 0.9 at both K and V, for l in {L2, L3, final};
  "absent"                          median accuracy <= 0.6 at K and at V for every l in {L2, L3, final}
                                    (then the fix is upstream and S56's register is mandatory, not optional);
  otherwise                         "partial at l" for each l with both medians in (0.6, 0.9), "present at K only" /
                                    "at V only" where one side passes 0.9 and the other does not.
The arm whose reading drives S56's design is L3S_P8 (the window gate's own stack on the P = 8 layout, the configuration S56
modifies). FAR_L3_P4 and ORACLE_P8 are read the same way and printed beside it.
VALIDITY: ORACLE_P8 binds on >= 1 of its 2 seeds within 24000, else L3S_P8's training outcome is UNTESTED (the probe
readings stand: they describe the trained stack, not an outcome).
OUTCOMES (descriptive, printed): BOUND, DISCOVERED (bound and VAL cos < 0.5), ROUTED*@end (margin >= 0.9 and eta^2 by
stream at KEY > 0.9), BOUND ROUTED (bound, the stream -> channel map at VAL one-to-one, every stream's held-out accuracy >=
0.9), failure classes (test_router_layout.fail_class: STREAM-PARTIAL / KEY / POSITION / OTHER; MERGED if the map at VAL is
not one-to-one), the routing margin and eta^2 at KEY and at VAL, transitions.

CHECKS (explore_g_probe_child.checks, asserted before any run, and the harness's repro check): X's arm A seed 160 through
2400 bit for bit (explore_common.repro_check); FAR_L3_P4|160 through 2400 equals batch 2's record bit for bit; the residual
loop's logits equal model(x)'s for each model kind, layer 1's input = LN(embedding); the probe's labels and the disjoint
split; a synthetic probe decodes an informative feature (>= 0.99) and not a random one (0.4-0.6); L3S_P8's model, window,
frozen W_h and groups; H8's L, qpos, vocabulary.

Run:  python explore_g_probe.py [--report-only]
"""

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_g_common as gc
import explore_common as ec

NAME = "s55_probe"
CHILD = "explore_g_probe_child"
ARMS = {
    "ORACLE_P8": dict(iters=24000, seeds=(360, 361), prio=2, label="perfect gate, header P=8 (validity)"),
    "L3S_P8": dict(iters=24000, seeds=tuple(range(360, 370)), prio=1, label="LOCAL3 + SLOW, header P=8"),
    "FAR_L3_P4": dict(iters=24000, seeds=tuple(range(160, 170)), prio=1, label="batch 2's far_L3 reproduced (P=4)"),
}
FEATS = ("emb", "L1", "L2", "L3", "final")
READ_L = ("L2", "L3", "final")


def line(arm, r):
    if not r.get("ok"):
        return f"FAILED {r.get('error')}"
    pr = r["probe"]
    return (f"acc {r['acc']:.3f} trans {str(r['transition']):>5} {r['tag']:<15} probe L2 {pr['L2|K']['acc']:.2f}/"
            f"{pr['L2|V']['acc']:.2f} L3 {pr['L3|K']['acc']:.2f}/{pr['L3|V']['acc']:.2f} final "
            f"{pr['final|K']['acc']:.2f}/{pr['final|V']['acc']:.2f}" + (f" repro {r['repro']}" if "repro" in r else ""))


def bound_routed(r):
    e = r["end"]
    return bool(r["transition"] is not None and e.get("one_to_one_val") and min(r["stream_acc"]) >= 0.9)


def outcome(r):
    if r["transition"] is not None:
        return "BOUND ROUTED" if bound_routed(r) else "BOUND NOT routed"
    fc = ec.fail_class(r)
    if fc == "STREAM-PARTIAL" and not r["end"].get("one_to_one_val", True):
        return "MERGED"
    return fc


def reading(med):
    out = []
    for l in READ_L:
        k, v = med[f"{l}|K"], med[f"{l}|V"]
        if k >= 0.9 and v >= 0.9:
            out.append(f"present at {l}")
        elif k >= 0.9:
            out.append(f"present at K only at {l}")
        elif v >= 0.9:
            out.append(f"present at V only at {l}")
        elif k > 0.6 and v > 0.6:
            out.append(f"partial at {l}")
    if all(med[f"{l}|K"] <= 0.6 and med[f"{l}|V"] <= 0.6 for l in READ_L):
        return "absent (S56's register is mandatory)"
    return "; ".join(out) if out else "neither (below 0.6 on one side)"


def report(st=None):
    st = gc.load(NAME) if st is None else st
    out = {}
    L = []
    p = L.append
    p("| arm | runs | BOUND | BOUND ROUTED | failures | transitions | " + " | ".join(f"{f} K / V" for f in FEATS) + " |")
    p("|---|---|---|---|---|---|" + "---|" * len(FEATS))
    for arm in ("L3S_P8", "FAR_L3_P4", "ORACLE_P8"):
        rr = gc.runs(sys.modules[__name__], arm, st)
        if arm == "FAR_L3_P4":
            bad = [s for s, r in rr.items() if not r.get("repro")]
            rr = {s: r for s, r in rr.items() if r.get("repro")}
        n = len(rr)
        if not n:
            continue
        nb = sum(r["transition"] is not None for r in rr.values())
        nbr = sum(bound_routed(r) for r in rr.values())
        fails = {}
        for r in rr.values():
            if r["transition"] is None:
                o = outcome(r)
                fails[o] = fails.get(o, 0) + 1
        tr = sorted(r["transition"] for r in rr.values() if r["transition"] is not None)
        med = {f"{f}|{q}": statistics.median(r["probe"][f"{f}|{q}"]["acc"] for r in rr.values()) for f in FEATS for q in "KV"}
        medi = {f"{f}|{q}": statistics.median(r["probe_init"][f"{f}|{q}"]["acc"] for r in rr.values()) for f in FEATS for q in "KV"}
        pres = {l: sum(r["probe"][f"{l}|K"]["acc"] >= 0.9 and r["probe"][f"{l}|V"]["acc"] >= 0.9 for r in rr.values())
                for l in READ_L}
        p(f"| {arm} | {n} | {gc.cnt(nb, n)} | {gc.cnt(nbr, n)} | {fails or '-'} | {tr or '-'} | "
          + " | ".join(f"{med[f'{f}|K']:.2f} / {med[f'{f}|V']:.2f}" for f in FEATS) + " |")
        out[arm] = dict(n=n, bound=nb, bound_routed=nbr, fails=fails, transitions=tr, median=med, median_init=medi,
                        present_runs=pres, reading=reading(med),
                        excluded=bad if arm == "FAR_L3_P4" else [])
    print("\n".join(L))
    for arm, d in out.items():
        print(f"\n{arm}: reading: {d['reading']}; runs with K and V >= 0.9: "
              + ", ".join(f"{l} {gc.cnt(d['present_runs'][l], d['n'])}" for l in READ_L)
              + f"; at init (median) " + ", ".join(f"{f} {d['median_init'][f + '|K']:.2f}/{d['median_init'][f + '|V']:.2f}"
                                                    for f in READ_L)
              + (f"; excluded (not reproduced): {d['excluded']}" if d["excluded"] else ""))
    print("\nper run (held-out accuracy K / V; Wilson on positions; by index within block = V accuracy for pairs 0..P-1):")
    for arm in ("L3S_P8", "FAR_L3_P4", "ORACLE_P8"):
        for s, r in sorted(gc.runs(sys.modules[__name__], arm, st).items()):
            pr = r["probe"]
            e = r["end"]
            print(f"  {arm} {s}: {outcome(r):<16} acc {r['acc']:.3f} trans {r['transition']} margin {e.get('margin', float('nan')):.2f} "
                  f"eta_stream K/V {e.get('eta_key_by_stream', float('nan')):.2f}/{e.get('eta_val_by_stream', float('nan')):.2f} "
                  f"eta_key K/V {e.get('eta_key_by_key', float('nan')):.2f}/{e.get('eta_val_by_key', float('nan')):.2f} map V {e.get('ch_map_val')}"
                  + "".join(f"\n      {f:<5} K {pr[f + '|K']['acc']:.3f} [{pr[f + '|K']['wilson'][0]:.3f}, {pr[f + '|K']['wilson'][1]:.3f}]"
                            f"  V {pr[f + '|V']['acc']:.3f} [{pr[f + '|V']['wilson'][0]:.3f}, {pr[f + '|V']['wilson'][1]:.3f}]"
                            f"  V by index {[round(v, 2) for v in pr[f + '|V']['by_index'].values()]}" for f in READ_L))
    valid = sum(r["transition"] is not None for r in gc.runs(sys.modules[__name__], "ORACLE_P8", st).values())
    print(f"\nVALIDITY: ORACLE_P8 bound {valid}/2 -> L3S_P8's outcome {'VALID' if valid >= 1 else 'UNTESTED'}")
    with open(os.path.join(gc.G_DIR, "s55_summary.json"), "w") as f:
        json.dump(dict(arms=out, valid=valid >= 1), f, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    pv = ec.print_banner("Session G S55 probe")
    if not args.report_only:
        t0 = time.time()
        assert ec.repro_check(), "repro check failed"
        rows = gc.run_child(CHILD, "checks", {})
        for nm, v in rows:
            print(f"  CHECK S55: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        assert all(v for _, v in rows), "a CHECK failed"
        os.makedirs(gc.G_DIR, exist_ok=True)
        with open(os.path.join(gc.G_DIR, f"checks_s55_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
            json.dump(dict(provenance=pv, rows=rows, secs=time.time() - t0), f, indent=1)
        gc.run_screen(sys.modules[__name__])
        print(f"  S55 runs done in {(time.time() - t0) / 60:.1f} min", flush=True)
    report()


if __name__ == "__main__":
    main()
