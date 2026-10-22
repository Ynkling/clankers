#!/usr/bin/env python
"""
explore_g_s91b.py — EXPLORATORY, not a result. Session G, screen S91b (round 7, replacing S91's remaining Hebbian half): does
the register gate route by context type when it has a channel per type? Everything below was fixed and committed before any
S91b run.

WHY. S91 asked a gate with k = 2 channels to route RandHeaderTask-allpool, whose two contexts are drawn from 16 types. F's
report 10 (claude/explore-F d9783bb) shows that the 0.533 cap both sessions printed is a property of k = D: with k = 16, one
channel per context type, a fixed type -> channel map routes every sequence (cap 1 at D = 2, 3, 4). The main line's regime
(k = 16 at eight streams) would let a gate route by type. So S91b gives the register gate k = 16. Per the user (22 October):
"This replaces S91's remaining Hebbian half; do not rerun S91."
TASKS (explore_f_tasks.py vendored from claude/explore-F at 9017b82, blob cfe3fe2, asserted; append-only over b0c6332):
  -ctxpool  RandHeaderTask-ctxpool: lite's 4 key types kept, only the 2 contexts drawn from 16 types (the cleaner task: one
            thing changed against lite). F's pilots: the k = 2 per-sequence-index perfect gate valid on both memories at
            24000 (Hebbian + conv 2400 / 4800; delta 1200 / 2400).
  -allpool  RandHeaderTask-allpool (keys from 32 types too), as S89 / S89H.
MODELS (explore_g_s91b_child; S89H's / S89's builders with n_ch = 16, CHECKed equal to them at k = 2): k = 16, the register
  gate shared by the three layers (REG3), Hebbian + conv 'layer' as S89H ("_H") and delta beta 1 / L2 as S89 ("_D"); SLOW;
  24000 updates (the budget now reaches every child: explore_g_common.run_screen sends ARMS[arm]["iters"], CHECKed).
  REG3 (no hinge), REG3_KEYHINGE (the key term), REG3_REPEATHINGE (the repeat hinge).
  PIDX (the per-sequence-index perfect gate: two of the 16 channels used) and PTYPE (the type-level perfect gate: each context
  type its own channel): validity and the two ceilings.
SEEDS (G's round-7 block 2960-3019; 2960-2991 used by S91): perfect gates 2992-2993; learned arms 2994-3003 (10), paired
  across arms, tasks and memories. 3004-3019 unused. 2 tasks x 2 memories x (2 gates x 2 + 3 arms x 10) = 136 runs.
VALIDITY: per task and memory, PIDX binds 2/2 within 24000; else that cell is UNTESTED (not a negative) and its learned arms
  are not queued. Written to explore_out/G/s91b_validity.json with the code SHA before any learned arm.
OUTCOME: ROUTED-BOUND in its within-sequence form at any k: BOUND and end margin >= 0.9 and end seq_separated_frac > 0.9
  (explore_f_tasks.inv_from_gate_k: the fraction of probe sequences whose two streams' mean KEY gates have distinct argmax
  channels, each with max >= 0.9; invariant to a per-sequence channel permutation). At k = 2 the within-sequence eta^2 by
  stream (S89 / S89H / S91's form) reads channel 0 and is not defined at k = 16; the two forms agree on synthetic k = 2 gates
  (CHECK). Printed beside it, per arm: TYPE-ROUTED = ROUTED-BOUND and type_map_k > 0.9 (the gate a fixed function of the
  context type that separates every sequence's contexts: each type on its own channel, or any map that separates the
  sequences); the median type_map_k; the median share of sequences separated; DISCOVERED (BOUND and val_cos_inv_k < 0.5).
  Exact one-sided McNemar on paired seeds; Fisher one-sided for unpaired counts.
READINGS (fixed now; per memory, on -ctxpool, n = 10; computed by readings() below):
  "the gate routes by type when it has the channels"   some learned arm is ROUTED-BOUND on >= 7/10 on -ctxpool AND is
                     discovered against its k = 2 counterpart's 0/20 (the same arm at k = 2: S89H's allpool.<ARM>_H for
                     Hebbian + conv, S89's allpool.<ARM>_D for delta, both 0/20 at 24000; one-sided Fisher p < 0.05; those
                     counterparts ran -allpool, the only k = 2 record of these arms with pooled contexts).
  "the channels do not suffice"   every learned arm <= 3/10 on -ctxpool while both perfect gates (PIDX and PTYPE) bind 2/2
                     on -ctxpool.
  "no reading applies"            otherwise.
  UNTESTED                        the cell's PIDX did not bind 2/2, or a learned run did not complete (3 attempts).
  -ALLPOOL: descriptive against -ctxpool (each arm's counts on both tasks, paired by seed, one-sided McNemar both ways; the
  same printed columns).
  Printed with every reading: TYPE-ROUTED counts, type_map_k, the separated share, the k = 2 counterpart's count and Fisher p.
NOTE (stated before the runs): F's report 10 found one delta channel (k = 1, no gate) binds -ctxpool 10/10 without any
  routing. So on delta BOUND alone says little here; ROUTED-BOUND requires the gate to separate the streams.
CHECKS (asserted before any run): X's arm A bit for bit; S89's CHECK set (as it ran there); the child's:
  - the vendored module's blob and its b0c6332 prefix (S89's own blob row, which asserts e24d5e1 itself, is printed as
    superseded by this one: added 22 October at the first CHECK pass, before any run); F's check_ctx_task (-ctxpool) and check_inv_stats_k (both tasks);
  - at k = 2 the builder equals S89H's / S89's REG3_H, REG3_REPEATHINGE_H, REG3_D, PERFECT_H and ORACLE_D (parameters and
    logits bit for bit), and S89H's / S89's allpool.REG3_REPEATHINGE_<M>|2842 reproduce bit for bit through 2400 through this
    screen's run() at k = 2 with S89's statistics;
  - at k = 16: the shapes; both perfect gates untrained read ROUTED_k, PTYPE's type_map_k 1 and PIDX's ~0; the k = 2 forms
    agree; the key term's positions on -ctxpool; each hinge off = REG3 through 2400 bit for bit (every task and memory);
    every arm end to end;
  - (harness) each queued payload carries its arm's iters; after the run, records iters == ARMS iters on n/n (printed).
RUNTIME: Hebbian + conv ~32 ms per update at k = 16, delta ~84 ms (measured): 136 runs, about 4-6 h on 4 workers.

Run:  python explore_g_s91b.py [--report-only]
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
import explore_g_s89 as s89p
import explore_g_s89h as s89h

NAME = "s91b"
CHILD = "explore_g_s91b_child"
TASKS = ("ctxpool", "allpool")
MEMS = ("H", "D")
S2, S10 = (2992, 2993), tuple(range(2994, 3004))
LEARNED = ("REG3", "REG3_KEYHINGE", "REG3_REPEATHINGE")
GATES = ("PIDX", "PTYPE")
VALID = os.path.join(gc.G_DIR, "s91b_validity.json")
line = hp.line
S20_K2 = tuple(range(2842, 2862))


def rbk(r):
    e = r["end"]
    return bool(hp.bound(r) and e.get("margin", 0.0) >= 0.9 and (e.get("seq_separated_frac") or 0.0) > 0.9)


def type_routed(r):
    return bool(rbk(r) and (r.get("type_map_k") or 0.0) > 0.9)


def _sha():
    return gc.load(NAME)["meta"].get("sha")


def _valid():
    v = json.load(open(VALID)) if os.path.exists(VALID) else None
    return v if v and v.get("sha") == _sha() else None


def _arms():
    a = {f"{t}.{g}_{m}": dict(iters=24000, seeds=S2, prio=2) for t in TASKS for m in MEMS for g in GATES}
    v = _valid()
    if v:
        for t in TASKS:
            for m in MEMS:
                if v[f"{t}.{m}"]["valid"]:
                    for k in LEARNED:
                        a[f"{t}.{k}_{m}"] = dict(iters=24000, seeds=S10, prio=1)
    return a


ARMS = _arms()


def counterpart(arm, mem):
    """The same arm at k = 2 on -allpool: S89H (Hebbian + conv) or S89 (delta), 2842-2861, within-sequence outcome."""
    rr = s89p.recorded("s89h" if mem == "H" else "s89", f"allpool.{arm}_{mem}", S20_K2)
    return sum(hp.routed_bound(s89h.inv_view(r)) for r in rr.values()), len(rr)


def describe(rr):
    if not rr:
        return "-"
    sep = [r["end"].get("seq_separated_frac") or 0.0 for r in rr.values()]
    tm = [r.get("type_map_k") or 0.0 for r in rr.values()]
    return (f"ROUTED-BOUND {sum(rbk(r) for r in rr.values())}/{len(rr)}, TYPE-ROUTED {sum(type_routed(r) for r in rr.values())}, "
            f"BOUND {sum(hp.bound(r) for r in rr.values())}; sequences separated median {statistics.median(sep):.2f} (max "
            f"{max(sep):.2f}); type_map_k median {statistics.median(tm):.2f}")


def readings(R, mem, seeds=S10, cp=None):
    """R: '<task>.<NAME>' -> {seed: record} for one memory. cp: arm -> (k = 2 count, n) (default: the records)."""
    rd = {}
    t = "ctxpool"
    pidx, ptype = R.get(f"{t}.PIDX", {}), R.get(f"{t}.PTYPE", {})
    nb = (sum(hp.bound(r) for r in pidx.values()), sum(hp.bound(r) for r in ptype.values()))
    rd["perfect gates on -ctxpool"] = f"PIDX {nb[0]}/{len(pidx)}, PTYPE {nb[1]}/{len(ptype)}"
    if nb[0] < 2 or len(pidx) < 2:
        rd["READING"] = "UNTESTED (PIDX did not bind 2/2 on -ctxpool)"
        return rd
    missing = {a: sorted(set(seeds) - set(R.get(f"{t}.{a}", {}))) for a in LEARNED if set(seeds) - set(R.get(f"{t}.{a}", {}))}
    if missing:
        rd["READING"] = f"UNTESTED (runs that did not complete: {missing})"
        return rd
    counts, routes = {}, False
    for a in LEARNED:
        rr = R[f"{t}.{a}"]
        k = sum(rbk(r) for r in rr.values())
        counts[a] = k
        c2_, n2 = cp[a] if cp else counterpart(a, mem)
        p = gc.fisher_greater(k, len(rr), c2_, n2)
        disc = p < 0.05
        rd[f"{a} on -ctxpool"] = (f"{describe(rr)}; vs its k = 2 counterpart (-allpool) {c2_}/{n2}: Fisher p {p:.3g}, "
                                  f"discovered {'yes' if disc else 'no'}")
        routes |= disc and k >= 7
    if routes:
        rd["READING"] = "the gate routes by type when it has the channels"
    elif all(k <= 3 for k in counts.values()) and nb == (2, 2):
        rd["READING"] = "the channels do not suffice"
    else:
        rd["READING"] = "no reading applies"
    return rd


def check_readings():
    def rec(ok, tm=0.95):
        return dict(transition=6000 if ok else None, end=dict(margin=1.0 if ok else 0.3, seq_separated_frac=1.0 if ok else 0.2),
                    type_map_k=tm)

    def arm(k, n=10, seeds=S10):
        return {s: rec(i < k) for i, s in enumerate(seeds)}
    gate = {s: rec(True) for s in S2}
    cp = {a: (0, 20) for a in LEARNED}
    base = {"ctxpool.PIDX": gate, "ctxpool.PTYPE": gate}
    cases = [("REPEAT 8, others 0", dict(base, **{"ctxpool.REG3": arm(0), "ctxpool.REG3_KEYHINGE": arm(0),
                                                 "ctxpool.REG3_REPEATHINGE": arm(8)}), "the gate routes by type when it has the channels"),
             ("all <= 3", dict(base, **{"ctxpool.REG3": arm(3), "ctxpool.REG3_KEYHINGE": arm(1), "ctxpool.REG3_REPEATHINGE": arm(0)}),
              "the channels do not suffice"),
             ("KH 5", dict(base, **{"ctxpool.REG3": arm(0), "ctxpool.REG3_KEYHINGE": arm(5), "ctxpool.REG3_REPEATHINGE": arm(0)}),
              "no reading applies"),
             ("all <= 3 but PTYPE 1/2", dict(base, **{"ctxpool.PTYPE": {2992: rec(True), 2993: rec(False)}, "ctxpool.REG3": arm(0),
                                                      "ctxpool.REG3_KEYHINGE": arm(0), "ctxpool.REG3_REPEATHINGE": arm(0)}),
              "no reading applies"),
             ("PIDX 1/2", dict(base, **{"ctxpool.PIDX": {2992: rec(True), 2993: rec(False)}}), "UNTESTED"),
             ("a missing KH seed", dict(base, **{"ctxpool.REG3": arm(0), "ctxpool.REG3_KEYHINGE": arm(9, seeds=S10[:9]),
                                                 "ctxpool.REG3_REPEATHINGE": arm(0)}), "UNTESTED")]
    rows = []
    for nm, R, want in cases:
        got = readings(R, "H", cp=cp)["READING"]
        rows.append((f"S91b readings hand case [{nm}]: {got}", got.startswith(want)))
    rows.append(("TYPE-ROUTED needs ROUTED-BOUND and type_map_k > 0.9", type_routed(rec(True)) and not type_routed(rec(True, 0.5))
                 and not type_routed(rec(False))))
    cps = {(a, m): counterpart(a, m) for a in LEARNED for m in MEMS}
    rows.append((f"the k = 2 counterparts on -allpool (S89H / S89, within-sequence): {cps}",
                 all(c == 0 and n == 20 for c, n in cps.values())))
    return rows


def report():
    me = sys.modules[__name__]
    R = {a: gc.runs(me, a) for a in ARMS}
    order = [f"{t}.{a}_{m}" for m in MEMS for t in TASKS for a in GATES + LEARNED if f"{t}.{a}_{m}" in ARMS]
    view = {a: R[a] for a in order}
    print(hp.count_table(view, order))
    print("NOTE: the table's ROUTED-BOUND column is the k = 2 pooled form (channel 0); the readings use ROUTED_k below.")
    nm, ok = gc.iters_check(me)
    print(f"  {nm} ({'ok' if ok else 'MISMATCH'})")
    v = _valid() or {}
    out = {}
    for m in MEMS:
        Rm = {a.rsplit("_", 1)[0]: rr for a, rr in R.items() if a.endswith(f"_{m}")}
        rd = readings(Rm, m)
        rd["validity"] = {t: v.get(f"{t}.{m}") for t in TASKS}
        allp = {}
        for a in GATES + LEARNED:
            rr = Rm.get(f"allpool.{a}", {})
            allp[a] = describe(rr)
            rc_ = Rm.get(f"ctxpool.{a}", {})
            if a in LEARNED and rr and rc_:
                b, c, p, n = hp.paired(rc_, rr, rbk)
                allp[a] += f"; -ctxpool only {b}, -allpool only {c}, p(ctxpool >) {p:.3g}, p(allpool >) {hp.paired(rr, rc_, rbk)[2]:.3g}, n {n}"
        rd["-allpool (descriptive, against -ctxpool)"] = allp
        for g in GATES:
            rd[f"ctxpool.{g} (ceiling)"] = describe(Rm.get(f"ctxpool.{g}", {}))
        out[m] = rd
    print("\nREADINGS:")
    for m in MEMS:
        print(f"  {'Hebbian + conv' if m == 'H' else 'delta'} (k = 16):")
        for k_, x in out[m].items():
            if isinstance(x, dict):
                print(f"      {k_}:\n" + "\n".join(f"          {a}: {y}" for a, y in x.items()))
            else:
                print(f"      {k_}: {x}")
    print("\nDIAGNOSTICS:\n" + hp.diagnostics(view, order))
    print("\nper run:\n" + hp.per_run(view, order))
    with open(os.path.join(gc.G_DIR, "s91b_summary.json"), "w") as fh:
        json.dump(dict(readings=out), fh, indent=1, default=str)


def run_checks():
    jobs = [("explore_g_s89_child", "checks", {}), (CHILD, "checks_static", {})]
    jobs += [(CHILD, "checks_repro", dict(mem=m)) for m in MEMS]
    jobs += [(CHILD, f, dict(task=t, mem=m)) for t in TASKS for m in MEMS for f in ("checks_off", "checks_on")]
    with ThreadPoolExecutor(gc.WORKERS) as ex:
        res = list(ex.map(lambda j: gc.run_child(j[0], j[1], j[2]), jobs))
    out = []
    for j, rr in zip(jobs, res):
        for n, ok in rr:
            if j[0] != CHILD and n.startswith("explore_f_tasks.py is F's b0c6332 blob"):
                # superseded (22 October, at the first CHECK pass, before any run): the vendored file is now F's 9017b82,
                # append-only over b0c6332; S91b's own CHECK asserts its blob and its byte prefix e24d5e1
                n, ok = f"(S89, superseded by S91b's blob + prefix CHECK) {n}", True
            out.append([n if j[0] == CHILD else (n if n.startswith("(S89") else f"(S89) {n}"), bool(ok)])
    return out


def main():
    import argparse
    global ARMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    pv = ec.print_banner("Session G S91b")
    me = sys.modules[__name__]
    if not args.report_only:
        t0 = time.time()
        assert ec.repro_check(), "repro check failed"
        rows = [list(x) for x in check_readings()] + run_checks()
        for nm, ok in rows:
            print(f"  CHECK S91b: {nm}: {'ok' if ok else 'FAIL'}", flush=True)
        json.dump(dict(provenance=pv, rows=rows, secs=time.time() - t0),
                  open(os.path.join(gc.G_DIR, f"checks_s91b_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)
        assert all(ok for _, ok in rows), "a CHECK failed"
        gc.run_screen(me)
        if _valid() is None:
            R = {a: gc.runs(me, a) for a in ARMS}
            v = {}
            for t in TASKS:
                for m in MEMS:
                    o = R[f"{t}.PIDX_{m}"]
                    v[f"{t}.{m}"] = dict(pidx_bound=sum(hp.bound(r) for r in o.values()), n=len(o),
                                         ptype_bound=sum(hp.bound(r) for r in R[f"{t}.PTYPE_{m}"].values()))
                    v[f"{t}.{m}"]["valid"] = v[f"{t}.{m}"]["pidx_bound"] == 2 and len(o) == 2
            v["sha"] = _sha()
            v["time"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            json.dump(v, open(VALID, "w"), indent=1)
            print(f"  S91b validity: {v}", flush=True)
        ARMS = _arms()
        gc.run_screen(me)
        print(f"  S91b runs done in {(time.time() - t0) / 60:.1f} min", flush=True)
    report()


if __name__ == "__main__":
    main()
