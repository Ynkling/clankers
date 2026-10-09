#!/usr/bin/env python
"""
explore_g_register.py — EXPLORATORY, not a result. Session G, screen S56: make the header-layout cue reach the gate with a
per-layer gate over the residual stream and a differentiable register. Everything below was fixed and committed before
any S56 run (and before S55's results were read).

TASK (the user's instruction of 9 October; RandHeaderTask-lite is not yet published by Session F): the FIXED header layout
at P = 8, S = 2 — explore_far_cue.HeaderTask(8, S=2, n_vals=16, n_q=1), L = 37 — k = 2, no conv, 24000 updates.
MODEL: explore_g_regmodel (layer 1 LOCAL3; layers 2-3 g = softmax(W_g2 tanh(A_r LN(r^(l)) + A_v v_t + A_R R_{t-1}));
register R_t = R_{t-1}(1 - beta_t) + beta_t W_R v_t, beta_t = sigmoid(w_R . v_t + b_R), b_R = logit(0.1); the gate at t
reads R_{t-1}). RECIPES, NUDGE, STATISTICS: explore_g_register_child's docstring.

THE READ PATH OF LAYER 2 (rule fixed now, applied to S55's L3S_P8 medians when S55 is complete; written to
explore_out/G/s56_read_path.json before any S56 run):
  - L2 present (median held-out accuracy >= 0.9 at K and V at L2)           -> res_layers = (2, 3): each gated layer reads
                                                                              its own residual (the specification);
  - else L3 present (>= 0.9 at K and V at L3)                                -> res_layers = (3,): layer 2's gate reads v_t
                                                                              and the register only, layer 3's reads LN(r^(3))
                                                                              too (the user's note: read where the stream is);
  - else (partial or absent at both)                                         -> res_layers = (2, 3) (the specification).
ARMS (Hebbian memory, seeds 370-379, paired: the same batches, the same initial parameters wherever they exist):
  ORACLE_H      the perfect gate, Adam 1e-3 throughout (seeds 370-371; validity)
  BASELINE      LOCAL3 + SLOW (S55's L3S_P8 recipe)
  RESGATE_ONLY  per-layer residual gate, no register, SG, SLOW
  REG           register + per-layer gate, SG (stop-gradient on r into the stack), SLOW
  REG_NOSG      register + per-layer gate, no stop-gradient, SLOW
  REG_NUDGE     REG + the 5% labelled nudge of the gated layers (validity of the architecture)
  SINGLE_GDN    one Hebbian channel (k = 1) with Gated DeltaNet's learned decay (Session F's S69_GDN at 24000), Adam 1e-3
                throughout: the strongest non-routing competitor on this layout
DELTA memory (explore_delta_mem copied unchanged from claude/explore-F at 9899520; beta = 1, L2 keys, tied write, decay
0.95 — F's S59 / S70 DELTA), seeds 390-399:
  ORACLE_D (390-391), BASELINE_D, REG_D, REG_NUDGE_D. Run after the Hebbian arms.
Seeds 380-389 are held back (RandHeaderTask-lite, if Session F publishes it).

OUTCOMES (per run): BOUND (transition not None); BOUND ROUTED = bound, the stream -> channel map at VAL positions
one-to-one at EVERY gated layer (layers 2 and 3; BASELINE's single gate), and every stream's held-out accuracy >= 0.9;
failure classes of unbound runs on the last gated layer's gate (the gate the harness reads): test_router_layout.fail_class
(STREAM-PARTIAL margin >= 0.25; KEY eta^2 by key at KEY >= 0.5; POSITION eta^2 by half/index >= 0.5; OTHER), MERGED if
STREAM-PARTIAL and the VAL map is not one-to-one. SINGLE_GDN (k = 1): BOUND only.
VALIDITY: an arm family (Hebbian; delta) is VALID if its oracle binds on >= 1 of its 2 seeds within 24000; otherwise its
readings are UNTESTED.

READINGS (fixed now; counts out of 10; Wilson 95% and band with every count; paired one-sided exact McNemar on 370-379)
  R1 "the register carries the cue"    REG_NUDGE BOUND ROUTED >= 9/10 (validity of the architecture)
     "architecture insufficient"       REG_NUDGE BOUND ROUTED < 6/10 (then: beta^R at CTX vs KEY/VAL — a write that does not
                                       separate CTX from K/V is the write decision; a separated write with an unrouted gate
                                       is the read path)
     otherwise                         "neither"
  R2 "discovered"                      REG or REG_NOSG BOUND >= BASELINE BOUND + 3 with that arm's KEY failures <= 2
     "not discovered"                  otherwise
  R3 vs SINGLE_GDN                     "the register beats the single GDN channel" if (best of REG, REG_NOSG) BOUND >=
                                       SINGLE_GDN BOUND + 3; "the single GDN channel is at least as good" if SINGLE_GDN
                                       BOUND >= that arm's BOUND; otherwise "neither"
  R4 "the register adds to the residual gate"  REG BOUND >= RESGATE_ONLY BOUND + 3; "it does not" if REG <= RESGATE_ONLY
  R5 "SG matters"                      |REG BOUND - REG_NOSG BOUND| >= 3 (direction printed); otherwise "SG does not matter"
  Delta: R1 and R2 on REG_NUDGE_D / REG_D vs BASELINE_D, same thresholds.
  S57 TRIGGER (explore_g_latch runs only if): R2 is "not discovered" AND REG's median (over runs) end-of-run beta^R at CTX
  tokens lies in [0.2, 0.8].
ALWAYS REPORTED: beta^R at CTX, KEY, VAL (mean, at every evaluation, median over runs); the routing margin at VAL per layer;
the CTX / KEY / VAL x stream contingency table of g at layers 2 and 3 (end, median over runs); transitions; eta^2.

CHECKS (explore_g_register_child.checks, asserted before any run): the recorded run with the new path disabled (batch 16's
S43 LOCAL3_SLOW|160 through 2400, bit for bit, curve and decoder, with RegBDH(gated=())); RegBDH(gated=()) = LOCAL3 logits
on H8; init (no existing parameter changes, A_v = W_in, W_g2 = W_g, beta^R ~ 0.1, u = W_in v with A_r = A_R = 0); the
register's causality (the gate at t reads R_{t-1}: token t's write changes nothing at <= t, changes t + 1); strict
causality of the whole model; SG blocks and NOSG passes the gate's gradient into the stack; RESGATE_ONLY has no register;
the SLOW groups; the nudge moves only the new gate parameters and leaves a stream lean; delta with beta 0 / write 1 / raw
keys = the Hebbian model (1e-5); the oracles take nothing from the other stream; SINGLE_GDN as stated.

Run:  python explore_g_register.py [--family hebb|delta] [--report-only]
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

NAME = "s56_register"
CHILD = "explore_g_register_child"
READ_PATH = os.path.join(gc.G_DIR, "s56_read_path.json")
HS, DS = tuple(range(370, 380)), tuple(range(390, 400))
HEBB = ("ORACLE_H", "BASELINE", "RESGATE_ONLY", "REG", "REG_NOSG", "REG_NUDGE", "SINGLE_GDN")
DELTA = ("ORACLE_D", "BASELINE_D", "REG_D", "REG_NUDGE_D")
ARMS = {a: dict(iters=24000, seeds=(HS[:2] if a == "ORACLE_H" else HS), prio=2 if a.startswith("ORACLE") else 1) for a in HEBB}
ARMS.update({a: dict(iters=24000, seeds=(DS[:2] if a == "ORACLE_D" else DS), prio=2 if a.startswith("ORACLE") else 0)
             for a in DELTA})


def res_layers_from_s55():
    with open(os.path.join(gc.G_DIR, "s55_summary.json")) as f:
        med = json.load(f)["arms"]["L3S_P8"]["median"]
    if med["L2|K"] >= 0.9 and med["L2|V"] >= 0.9:
        return [2, 3], "L2 present"
    if med["L3|K"] >= 0.9 and med["L3|V"] >= 0.9:
        return [3], "L2 not present, L3 present"
    return [2, 3], "neither L2 nor L3 present"


def read_path():
    if os.path.exists(READ_PATH):
        with open(READ_PATH) as f:
            return json.load(f)
    rl, why = res_layers_from_s55()
    d = dict(res_layers=rl, why=why, time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    with open(READ_PATH, "w") as f:
        json.dump(d, f, indent=1)
    return d


def line(arm, r):
    if not r.get("ok"):
        return f"FAILED {r.get('error')}"
    return f"acc {r['acc']:.3f} trans {str(r['transition']):>5} {outcome(r)}"


# ── outcomes ─────────────────────────────────────────────────────────────────
def gated_layers(r):
    return [2, 3] if r["arm"] not in ("BASELINE", "BASELINE_D") else [3]


def bound_routed(r):
    if r["k"] < 2 or r["transition"] is None:
        return False
    lay = r["end"]["layers"]
    return bool(all(lay[f"L{l}"]["one_to_one_val"] for l in gated_layers(r)) and min(r["stream_acc"]) >= 0.9)


def outcome(r):
    if r["k"] < 2:
        return "BOUND" if r["transition"] is not None else "unbound"
    if r["transition"] is not None:
        return "BOUND ROUTED" if bound_routed(r) else "BOUND NOT routed"
    fc = ec.fail_class(r)
    if fc == "STREAM-PARTIAL" and not r["end"]["layers"]["L3"]["one_to_one_val"]:
        return "MERGED"
    return fc


def mcn(ra, rb, f):
    seeds = sorted(set(ra) & set(rb))
    b = sum(1 for s in seeds if f(ra[s]) and not f(rb[s]))
    c = sum(1 for s in seeds if f(rb[s]) and not f(ra[s]))
    return b, c, gc.mcnemar_greater(b, c)


def report(family="hebb"):
    st = gc.load(NAME)
    me = sys.modules[__name__]
    arms = HEBB if family == "hebb" else DELTA
    R = {a: gc.runs(me, a, st) for a in arms}
    bnd = lambda r: r["transition"] is not None
    L = []
    p = L.append
    p("| arm | n | BOUND | BOUND ROUTED | failures (unbound) | transitions | median acc |")
    p("|---|---|---|---|---|---|---|")
    summ = {}
    for a in arms:
        rr = R[a]
        n = len(rr)
        if not n:
            continue
        nb = sum(bnd(r) for r in rr.values())
        nbr = sum(bound_routed(r) for r in rr.values())
        fails = {}
        for r in rr.values():
            if not bnd(r):
                o = outcome(r)
                fails[o] = fails.get(o, 0) + 1
        tr = sorted(r["transition"] for r in rr.values() if bnd(r))
        summ[a] = dict(n=n, bound=nb, bound_routed=nbr, fails=fails, transitions=tr)
        p(f"| {a} | {n} | {gc.cnt(nb, n)} | {gc.cnt(nbr, n) if rr[next(iter(rr))]['k'] > 1 else '--'} | {fails or '-'} | "
          f"{tr or '-'} | {statistics.median(r['acc'] for r in rr.values()):.3f} |")
    print("\n".join(L))
    out = dict(summary=summ)
    orc = "ORACLE_H" if family == "hebb" else "ORACLE_D"
    valid = summ.get(orc, {}).get("bound", 0) >= 1
    print(f"\nVALIDITY: {orc} bound {summ.get(orc, {}).get('bound', 0)}/2 -> {'VALID' if valid else 'UNTESTED'}")
    sfx = "" if family == "hebb" else "_D"
    g = lambda a, k: summ.get(a, {}).get(k, 0)
    rd = {}
    nud, reg, base = f"REG_NUDGE{sfx}", f"REG{sfx}", f"BASELINE{sfx}"
    if valid and nud in summ:
        x = g(nud, "bound_routed")
        rd["R1"] = ("the register carries the cue" if x >= 9 else "architecture insufficient" if x < 6 else "neither")
    cands = [reg] + (["REG_NOSG"] if family == "hebb" else [])
    if valid and base in summ:
        disc = [c for c in cands if c in summ and g(c, "bound") >= g(base, "bound") + 3 and g(c, "fails").get("KEY", 0) <= 2]
        rd["R2"] = f"discovered ({', '.join(disc)})" if disc else "not discovered"
        for c in cands:
            if c in R and base in R:
                b_, c_, pv = mcn(R[c], R[base], bnd)
                print(f"  {c} vs {base} BOUND: {c} only {b_}, {base} only {c_}, one-sided McNemar p = {pv:.3g}")
    if family == "hebb" and valid:
        best = max((c for c in cands if c in summ), key=lambda c: g(c, "bound"), default=None)
        if best and "SINGLE_GDN" in summ:
            rd["R3"] = ("the register beats the single GDN channel" if g(best, "bound") >= g("SINGLE_GDN", "bound") + 3 else
                        "the single GDN channel is at least as good" if g("SINGLE_GDN", "bound") >= g(best, "bound") else "neither")
            b_, c_, pv = mcn(R[best], R["SINGLE_GDN"], bnd)
            print(f"  {best} vs SINGLE_GDN BOUND: {best} only {b_}, SINGLE_GDN only {c_}, p = {pv:.3g}")
        if "REG" in summ and "RESGATE_ONLY" in summ:
            rd["R4"] = ("the register adds to the residual gate" if g("REG", "bound") >= g("RESGATE_ONLY", "bound") + 3 else
                        "it does not" if g("REG", "bound") <= g("RESGATE_ONLY", "bound") else "neither")
        if "REG" in summ and "REG_NOSG" in summ:
            d = g("REG", "bound") - g("REG_NOSG", "bound")
            rd["R5"] = f"SG matters ({'SG' if d > 0 else 'NOSG'} better by {abs(d)})" if abs(d) >= 3 else "SG does not matter"
    # beta^R at CTX / KEY / VAL by step (median over runs), the margin at VAL per layer and the contingency tables
    diag = {}
    for a in arms:
        rr = R[a]
        if not rr:
            continue
        one = next(iter(rr.values()))
        dd = {}
        if one["stats"] and one["stats"][-1].get("beta"):
            steps = [s_["step"] for s_ in one["stats"]]
            by = {}
            for i, stp in enumerate(steps):
                vals = [r["stats"][i]["beta"] for r in rr.values() if i < len(r["stats"]) and r["stats"][i].get("beta")]
                if vals:
                    by[str(stp)] = {k: statistics.median(v[k] for v in vals) for k in ("CTX", "KEY", "VAL", "QUERY")}
            ends = [r["end"]["beta"] for r in rr.values()]
            by["end"] = {k: statistics.median(v[k] for v in ends) for k in ("CTX", "KEY", "VAL", "QUERY")}
            dd["beta"] = by
        if one["k"] > 1:
            dd["margin_val"] = {f"L{l}": statistics.median(r["end"]["layers"][f"L{l}"]["margin"] for r in rr.values())
                                for l in (1, 2, 3)}
            dd["table"] = {f"L{l}": {role: [[round(statistics.median(r["end"]["layers"][f"L{l}"]["table"][role][s][c]
                                                                      for r in rr.values()), 3) for c in range(2)]
                                           for s in range(2)] for role in ("CTX", "KEY", "VAL")} for l in (2, 3)}
        diag[a] = dd
    for a, dd in diag.items():
        if "beta" in dd:
            ks = list(dd["beta"])
            sel = [k for k in ks if k in ("0", "1200", "2400", "4800", "9600", "end")]
            print(f"  {a} beta^R CTX/KEY/VAL (median over runs): " + "; ".join(
                f"{k} {dd['beta'][k]['CTX']:.2f}/{dd['beta'][k]['KEY']:.2f}/{dd['beta'][k]['VAL']:.2f}" for k in sel))
        if "margin_val" in dd:
            print(f"  {a} margin at VAL (median) " + ", ".join(f"{l} {v:.2f}" for l, v in dd["margin_val"].items())
                  + "; table (mean g, [stream0, stream1] x [ch0, ch1]) "
                  + "; ".join(f"{l}: " + ", ".join(f"{role} {t}" for role, t in tab.items()) for l, tab in dd["table"].items()))
    if family == "hebb" and "REG" in diag and "beta" in diag["REG"]:
        bc = diag["REG"]["beta"]["end"]["CTX"]
        trig = rd.get("R2") == "not discovered" and 0.2 <= bc <= 0.8
        rd["S57"] = f"{'TRIGGERED' if trig else 'not triggered'} (R2 {rd.get('R2')}; REG median beta^R at CTX at the end {bc:.2f})"
    print("\nREADINGS: " + "; ".join(f"{k}: {v}" for k, v in rd.items()))
    print("\nper run:")
    for a in arms:
        for s, r in sorted(R[a].items()):
            lay = r["end"].get("layers", {})
            print(f"  {a} {s}: {outcome(r):<16} acc {r['acc']:.3f} trans {r['transition']} stream_acc "
                  f"{[round(v, 2) for v in r['stream_acc']]}" + ("".join(
                      f" | L{l} map {lay[f'L{l}']['ch_map_val']} m {lay[f'L{l}']['margin']:.2f} etaS K/V "
                      f"{lay[f'L{l}']['eta_key_by_stream']:.2f}/{lay[f'L{l}']['eta_val_by_stream']:.2f} etaK "
                      f"{lay[f'L{l}']['eta_key_by_key']:.2f}" for l in (2, 3)) if lay else "")
                  + (f" | beta C/K/V {r['end']['beta']['CTX']:.2f}/{r['end']['beta']['KEY']:.2f}/{r['end']['beta']['VAL']:.2f}"
                     if r["end"].get("beta") else ""))
    out.update(valid=valid, readings=rd, diag=diag)
    with open(os.path.join(gc.G_DIR, f"s56_{family}_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="hebb", choices=("hebb", "delta"))
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    pv = ec.print_banner(f"Session G S56 register ({args.family})")
    if not args.report_only:
        t0 = time.time()
        rp = read_path()
        print(f"  S56 read path: res_layers {rp['res_layers']} ({rp['why']}; decided {rp['time']})", flush=True)
        assert ec.repro_check(), "repro check failed"
        rows = gc.run_child(CHILD, "checks", dict(res_layers=rp["res_layers"]))
        for nm, v in rows:
            print(f"  CHECK S56: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        assert all(v for _, v in rows), "a CHECK failed"
        with open(os.path.join(gc.G_DIR, f"checks_s56_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
            json.dump(dict(provenance=pv, rows=rows, secs=time.time() - t0), f, indent=1)
        gc.run_screen(sys.modules[__name__], only=HEBB if args.family == "hebb" else DELTA)
        print(f"  S56 ({args.family}) runs done in {(time.time() - t0) / 60:.1f} min", flush=True)
    report(args.family)


if __name__ == "__main__":
    main()
