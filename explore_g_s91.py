#!/usr/bin/env python
"""
explore_g_s91.py — EXPLORATORY, not a result. Session G, screen S91 (round 7): the question ALLPOOL named. Can a gate route
by comparing the current context with what the sequence has already shown, with no fixed identity for any context type?
Everything below was fixed and committed before any S91 run.

BACKGROUND. On RandHeaderTask-allpool (each sequence's 2 CTX tokens drawn from 16 types) every learned arm of S89 (delta) and
S89H (Hebbian + conv) is 0/20 ROUTED-BOUND, the key hinge included, while the perfect gate, handed the per-sequence stream
index, binds at 3600-9600. No fixed type -> channel map separates more than 0.533 of sequences, and the hinge arms' failures
sit at that cap (S89H: 0.33-0.57 of probe sequences routed in 11-17 runs per arm). A gate that routes ALLPOOL must compare.
TASK: F's RandHeaderTaskAllPool (explore_f_tasks.py at b0c6332, blob e24d5e1, vendored and asserted by S89's CHECKs).
MEMORIES: Hebbian + conv 'layer' as S89H (the primary; 48000 updates) and delta (beta 1, L2 keys) as S89 (descriptive;
24000 updates). k = 2, SLOW, the REG3 model; the repeat hinge as S89/S89H.
ARMS (explore_g_s91_child; one run path for all, the controls built by S89H's / S89's make_model):
  REG3_REPEATHINGE   S89H's arm (S89's on delta): the control, 0/20 there.
  MATCHGATE          REG3_REPEATHINGE whose shared gate input gains, at every layer, m_t a_M with the match
                     m_t = cos(W_R v_t, R_{t-1}): the current token's embedding in the register's coordinates against the
                     register's content before token t's write (what it holds since its last write). a_M (16) starts at 0;
                     the register's write is REG3's. A context token that matches the register's content is "the same
                     document", one that does not is "the other"; the gate can read that without any type having a fixed
                     channel. (The register is 16-dimensional and the embedding 32, so the comparison is made where the
                     register writes: W_R v_t is exactly what the register would hold after a full write of token t.)
  MATCHGATE_NOHINGE  the same without the hinge.
  NOVELTY            REG3_REPEATHINGE whose register write gate also receives n_t (1 if the token type at t occurred
                     earlier in the sequence: the repeat statistic, causal, per position): beta_t = sigmoid(w_R . v_t + b_R +
                     w_N n_t), w_N starting at 0.
  VALIDITY           the perfect gate (task.perfect_gate) on both memories: PERFECT_H (Hebbian + conv, 48000) and ORACLE_D
                     (delta, 24000), seeds 2960-2961, run first. A memory's learned arms are queued only if its perfect gate
                     binds 2/2; else that memory is UNTESTED here (not a negative). Written to explore_out/G/s91_validity.json
                     with the code SHA before any learned arm.
SEEDS: Hebbian arms 2962-2981 (20, paired across arms); delta arms 2982-2991 (10, descriptive); validity 2960-2961. G's
  round-7 block is 2960-3019; S90 used S87's 2700-2719; nothing else of this block is used.
OUTCOME: ROUTED-BOUND in its within-sequence form, explore_g_rhl_parent.routed_bound(explore_g_s89h.inv_view(record)) (end
  margin >= 0.9 and within-sequence eta^2 by stream at KEY > 0.9; the pooled form reads the first-appearance labelling).
  "discovered" here = beats REG3_REPEATHINGE (the control) on the paired seeds, exact one-sided McNemar, p < 0.05.
READINGS (Hebbian + conv, n = 20; fixed now; each that applies is printed; computed by readings() below):
  "a comparing gate discovers the relative routing"   MATCHGATE discovered against REG3_REPEATHINGE (p < 0.05) and
                                                       ROUTED-BOUND on >= 10/20
  "it needs the hinge"                                MATCHGATE beats MATCHGATE_NOHINGE by >= 4 vs 0 (one-sided p <= 0.0625)
  "novelty alone suffices"                            NOVELTY discovered against REG3_REPEATHINGE and (MATCHGATE only) -
                                                       (NOVELTY only) <= 1
  "no device here"                                    none of MATCHGATE, MATCHGATE_NOHINGE, NOVELTY is discovered
  "no reading applies"                                otherwise (none of the four above applies)
  A run that does not complete is retried by the harness (3 attempts); if a run of a reading's arm still fails, that
  reading is UNTESTED.
  PRINTED WITH EVERY READING, per arm: the share of probe sequences routed at the end (median, and the number of runs above
  the 0.533 line, which no fixed type map can cross), and the type-level-map count (runs whose type_map, eta^2 of the gate at
  KEY grouped by the block's CTX type, is > 0.5: the gate keyed on context identity, not on the comparison).
  DELTA (n = 10, descriptive): the same function, printed, not read.
CHECKS (asserted before any run): X's arm A bit for bit; S89's and S89H's CHECK sets as they ran there (the vendored module,
  token ids only, the task-aware code, the conv model, F's two conv records); and the child's:
  - the match by hand, m_0 = 0, its signature (the embedding, the register and W_R only: no label, no index, no position);
    novelty by hand from token ids, causal, refusing a float tensor;
  - MATCHGATE and NOVELTY strictly causal with non-zero weights;
  - at initialisation each new arm's logits equal the control's bit for bit, its parameters the control's + one;
  - with its scalar zeroed, MATCHGATE and NOVELTY equal REG3_REPEATHINGE and MATCHGATE_NOHINGE equals REG3 bit for bit
    through 2400, on both memories (curve and every statistic);
  - the new weights receive a gradient and sit in the gate group;
  - S89H's allpool.REG3_REPEATHINGE_H|2842 and S89's allpool.REG3_REPEATHINGE_D|2842 reproduce bit for bit through 2400
    through this screen's run();
  - the untrained perfect gate reads ROUTED* (within-sequence) on both memories;
  - the type_map statistic: the relative perfect gate ~0, the balanced type-level gate 1;
  - every arm end to end (20 updates) on both memories.
RUNTIME: 4 validity + 80 Hebbian (48000; S89H's ALLPOOL arms took ~7 min each) + 40 delta (24000, ~10 min unbound): about
  4-5 h on 4 workers.

Run:  python explore_g_s91.py [--report-only]
"""

import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_g_common as gc
import explore_g_rhl_parent as hp
import explore_g_s89h as s89h

NAME = "s91"
CHILD = "explore_g_s91_child"
S2, SH, SD = (2960, 2961), tuple(range(2962, 2982)), tuple(range(2982, 2992))
ITERS = dict(H=48000, D=24000)
LEARNED = ("REG3_REPEATHINGE", "MATCHGATE", "MATCHGATE_NOHINGE", "NOVELTY")
VALID_ARM = dict(H="PERFECT", D="ORACLE")
VALID = os.path.join(gc.G_DIR, "s91_validity.json")
CAP = 64 / 120
P4 = 0.0625
line = hp.line


def rb(r):
    return hp.routed_bound(s89h.inv_view(r))


def _sha():
    return gc.load(NAME)["meta"].get("sha")


def _valid():
    v = json.load(open(VALID)) if os.path.exists(VALID) else None
    return v if v and v.get("sha") == _sha() else None


def _arms():
    a = {f"allpool.{VALID_ARM[m]}_{m}": dict(iters=ITERS[m], seeds=S2, prio=2) for m in ("H", "D")}
    v = _valid()
    if v:
        for m, seeds in (("H", SH), ("D", SD)):
            if v[m]["valid"]:
                for k in LEARNED:
                    a[f"allpool.{k}_{m}"] = dict(iters=ITERS[m], seeds=seeds, prio=1)
    return a


ARMS = _arms()


def describe(rr):
    fr = [r["end"]["seq_routed_frac"] for r in rr.values()]
    tm = [r.get("type_map") for r in rr.values() if r.get("type_map") is not None]
    return (f"sequences routed median {statistics.median(fr):.2f} (max {max(fr):.2f}; {sum(f > CAP for f in fr)} runs above "
            f"the 0.533 line); type-level map {sum(t > 0.5 for t in tm)}/{len(tm)} (type_map median "
            f"{statistics.median(tm):.2f})") if fr and tm else "-"


def readings(R, seeds):
    """R: arm name -> {seed: record}. Ordered dict: the readings that apply, and the numbers."""
    n = len(seeds)
    rd = {}
    missing = {a: sorted(set(seeds) - set(R.get(a, {}))) for a in LEARNED if set(seeds) - set(R.get(a, {}))}
    C, M, MN, NV = (R.get(a, {}) for a in LEARNED)
    rd["counts"] = ", ".join(f"{a} {sum(rb(r) for r in R.get(a, {}).values())}/{len(R.get(a, {}))}" for a in LEARNED)
    disc = {}
    for a, rr in (("MATCHGATE", M), ("MATCHGATE_NOHINGE", MN), ("NOVELTY", NV)):
        if a in missing or "REG3_REPEATHINGE" in missing:
            disc[a] = None
            continue
        d, t = hp.beats(rr, C, rb)
        disc[a] = d
        rd[f"{a} vs REG3_REPEATHINGE"] = f"{a} only {t[0]}, control only {t[1]}, p {t[2]:.3g}, n {t[3]}"
    applies = []
    if disc["MATCHGATE"] is None:
        rd["a comparing gate discovers the relative routing"] = "UNTESTED (missing runs)"
    elif disc["MATCHGATE"] and sum(rb(r) for r in M.values()) >= 10:
        applies.append("a comparing gate discovers the relative routing")
    if "MATCHGATE" in missing or "MATCHGATE_NOHINGE" in missing:
        rd["it needs the hinge"] = "UNTESTED (missing runs)"
    else:
        b, c, p, nn = hp.paired(M, MN, rb)
        rd["MATCHGATE vs MATCHGATE_NOHINGE"] = f"MATCHGATE only {b}, NOHINGE only {c}, p {p:.3g}, n {nn}"
        if p <= P4:
            applies.append("it needs the hinge")
    if disc["NOVELTY"] is None or "MATCHGATE" in missing:
        rd["novelty alone suffices"] = "UNTESTED (missing runs)"
    else:
        b, c, p, nn = hp.paired(M, NV, rb)
        rd["MATCHGATE vs NOVELTY"] = f"MATCHGATE only {b}, NOVELTY only {c}, p(MATCHGATE >) {p:.3g}, n {nn}"
        if disc["NOVELTY"] and b - c <= 1:
            applies.append("novelty alone suffices")
    if all(d is False for d in disc.values()):
        applies.append("no device here")
    if not applies and not missing:
        applies.append("no reading applies")
    rd["READINGS"] = applies or ["UNTESTED (missing runs)"]
    for a in LEARNED:
        if R.get(a):
            rd[f"{a} (printed with every reading)"] = describe(R[a])
    return rd


def check_readings():
    """Hand cases on synthetic records (no torch)."""
    def rec(ok, frac=0.5, tm=0.9):
        e = dict(margin=1.0 if ok else 0.4, eta_key_by_stream=0.0, eta_key_by_stream_inv=1.0 if ok else 0.4,
                 val_cos_inv=0.0, seq_routed_frac=1.0 if ok else frac)
        return dict(transition=6000 if ok else None, end=e, type_map=0.05 if ok else tm, val_cos=0.9)

    def arm(k, n=20):
        return {s: rec(i < k) for i, s in enumerate(range(n))}
    s = list(range(20))
    rows = []
    cases = [("control 0, MATCH 14, NOHINGE 0, NOVELTY 0", dict(REG3_REPEATHINGE=arm(0), MATCHGATE=arm(14),
              MATCHGATE_NOHINGE=arm(0), NOVELTY=arm(0)), ["a comparing gate discovers the relative routing", "it needs the hinge"]),
             ("control 0, MATCH 14, NOHINGE 14, NOVELTY 14", dict(REG3_REPEATHINGE=arm(0), MATCHGATE=arm(14),
              MATCHGATE_NOHINGE=arm(14), NOVELTY=arm(14)), ["a comparing gate discovers the relative routing", "novelty alone suffices"]),
             ("all 0", dict(REG3_REPEATHINGE=arm(0), MATCHGATE=arm(0), MATCHGATE_NOHINGE=arm(0), NOVELTY=arm(0)), ["no device here"]),
             ("control 0, MATCH 6 (discovered, < 10), NOHINGE 6, NOVELTY 0", dict(REG3_REPEATHINGE=arm(0), MATCHGATE=arm(6),
              MATCHGATE_NOHINGE=arm(6), NOVELTY=arm(0)), ["no reading applies"]),
             ("a missing MATCHGATE seed", dict(REG3_REPEATHINGE=arm(0), MATCHGATE={i: rec(True) for i in range(19)},
              MATCHGATE_NOHINGE=arm(0), NOVELTY=arm(0)), None)]
    for nm, R, want in cases:
        rd = readings(R, s)
        got = rd["READINGS"]
        ok = (got == want) if want is not None else (rd["a comparing gate discovers the relative routing"].startswith("UNTESTED"))
        rows.append((f"S91 readings hand case [{nm}]: {got}", ok))
    return rows


def report():
    me = sys.modules[__name__]
    R = {a: gc.runs(me, a) for a in ARMS}
    view = {a: {s: s89h.inv_view(r) for s, r in rr.items()} for a, rr in R.items()}
    order = [a for m in ("H", "D") for a in [f"allpool.{VALID_ARM[m]}_{m}"] + [f"allpool.{k}_{m}" for k in LEARNED] if a in ARMS]
    print(hp.count_table(view, order))
    v = _valid() or {}
    out = {}
    for m, seeds in (("H", SH), ("D", SD)):
        o = R.get(f"allpool.{VALID_ARM[m]}_{m}", {})
        note = (f"{VALID_ARM[m]}_{m} on 2960-2961 bound {sum(hp.bound(r) for r in o.values())}/{len(o)}: " +
                ", ".join(f"{s} {'bound @' + str(r['transition']) if hp.bound(r) else 'unbound'}" for s, r in sorted(o.items())))
        if not v.get(m, {}).get("valid"):
            out[m] = dict(validity=note, READINGS=["UNTESTED (the perfect gate did not bind 2/2)"])
            continue
        rd = readings({k: R.get(f"allpool.{k}_{m}", {}) for k in LEARNED}, seeds)
        out[m] = dict(validity=note, **rd)
        if m == "D":
            out[m]["note"] = "descriptive (n = 10): printed, not read"
    print("\nREADINGS:")
    for m in ("H", "D"):
        print(f"  {'Hebbian + conv' if m == 'H' else 'delta'}:\n" + "\n".join(f"      {k}: {x}" for k, x in out[m].items()))
    for a in order:
        rr = R.get(a, {})
        if rr and "ORACLE" not in a and "PERFECT" not in a:
            nps = [r.get("new_params") for r in rr.values() if r.get("new_params")]
            if nps:
                k = next(iter(nps[0]))
                vals = [abs(x) for npr in nps for x in npr[k]]
                print(f"  {a}: |{k}| at the end median {statistics.median(vals):.4f} (max {max(vals):.4f})")
            fr = [r.get("term", {}).get("fired") for r in rr.values() if r.get("term")]
            if fr:
                print(f"  {a}: hinge firings median {statistics.median(fr)}")
    print("\nDIAGNOSTICS:\n" + hp.diagnostics(view, order))
    print("\nper run:\n" + hp.per_run(view, order))
    with open(os.path.join(gc.G_DIR, "s91_summary.json"), "w") as fh:
        json.dump(dict(readings=out), fh, indent=1)


def run_checks():
    jobs = [("explore_g_s89_child", "checks", {}), ("explore_g_s89h_child", "checks_static", {}),
            ("explore_g_s89h_child", "checks_repro", dict(kind="f", key="O_HEBB_CONV_ALLPOOL|2800", seed=2800, task="allpool")),
            (CHILD, "checks_static", {})]
    jobs += [(CHILD, f, dict(mem=m)) for m in ("H", "D") for f in ("checks_zero", "checks_repro", "checks_on")]
    with ThreadPoolExecutor(gc.WORKERS) as ex:
        res = list(ex.map(lambda j: gc.run_child(j[0], j[1], j[2]), jobs))
    return [[n if j[0] == CHILD else f"({j[0][10:-6].upper()}) {n}", bool(ok)] for j, rr in zip(jobs, res) for n, ok in rr]


def main():
    import argparse
    global ARMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    pv = ec.print_banner("Session G S91")
    me = sys.modules[__name__]
    if not args.report_only:
        t0 = time.time()
        assert ec.repro_check(), "repro check failed"
        rows = [list(x) for x in check_readings()] + run_checks()
        for nm, ok in rows:
            print(f"  CHECK S91: {nm}: {'ok' if ok else 'FAIL'}", flush=True)
        json.dump(dict(provenance=pv, rows=rows, secs=time.time() - t0),
                  open(os.path.join(gc.G_DIR, f"checks_s91_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)
        assert all(ok for _, ok in rows), "a CHECK failed"
        gc.run_screen(me)
        if _valid() is None:
            R = {a: gc.runs(me, a) for a in ARMS}
            v = {}
            for m in ("H", "D"):
                o = R[f"allpool.{VALID_ARM[m]}_{m}"]
                v[m] = dict(bound=sum(hp.bound(r) for r in o.values()), n=len(o))
                v[m]["valid"] = v[m]["bound"] == 2 and v[m]["n"] == 2
            v["sha"] = _sha()
            v["time"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            json.dump(v, open(VALID, "w"), indent=1)
            print(f"  S91 validity: {v}", flush=True)
        ARMS = _arms()
        gc.run_screen(me)
        print(f"  S91 runs done in {(time.time() - t0) / 60:.1f} min", flush=True)
    report()


if __name__ == "__main__":
    main()
