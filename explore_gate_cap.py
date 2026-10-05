#!/usr/bin/env python
"""
explore_gate_cap.py — EXPLORATORY, not a result. Screen S41 (batch 15): cap the gate's recurrent gain (CAP), or
remove the recurrence (NOREC), at S=8, P=4, k=16, conv, all 8 streams from step 1, 28800 steps, seeds 260-269.

BACKGROUND (docstring)
- S39 (S=8, k=16), medians over seeds 260-264:
  - h decodes the stream at key positions at 0.91 at update 0 (1.00 at stream tokens).
  - Under Muon, |W_h h|@key: 0.04 (0) -> 0.19 (400) -> 1.9 (600) -> 6.9 (2400), while |W_in v|@key stays
    0.06-0.16; decodability 0.39 (400) -> 0.18 (600) -> 0.13 (2400).
  - Under Adam the jump falls between 1200 and 1600. Saturation <= 0.15. The hinge moves the loss ~200 updates
    earlier.
  - S=4 reference: |W_in v|@stream grows 0.06 -> 0.55; decodability stays 0.83.
- S40: GATE_PREV's input decodes the stream at 0.97-1.00 (Muon) but h only at 0.55-0.64. BOUND ROUTED 0/10 in both
  arms; failures moved from non-stream to MERGED on 3-4 channels; final accuracy higher in 9/10 Muon pairs.

CAP: after every optimizer step, if sigma_max(W_h) > 0.5, W_h <- W_h * 0.5 / sigma_max(W_h); optimizer state
untouched. NOREC: W_h fixed at 0 and excluded from the optimizer, so h_t = tanh(W_in v_t + W_prev v_(t-1))
(GATE_PREV's input, no recurrence). (explore_gate_cap_child.)
ARMS (main's run_sc with D8 at 9c5939e, child process):
  CAP_M         S33's D8_HINGE (Muon) + CAP; paired with S33's recorded D8_HINGE.
  PREV_CAP_M    S40's D8_PREV_M + CAP; paired with S40's recorded D8_PREV_M.
  PREV_NOREC_M  S40's D8_PREV_M with NOREC; paired with S40's recorded D8_PREV_M.
  PREV_CAP_A    S40's D8_PREV_A (Adam) + CAP; paired with S40's recorded D8_PREV_A.
MACHINE: Muon runs (bfloat16 Newton-Schulz) are CPU-specific; Adam runs are not. The first dry run and segment 1
ran on a Xeon @ 2.80GHz, where the recorded Muon runs do not reproduce; by the user's first decision the Muon
references were then rerun there (REF arms). Segment 2's container had the Xeon @ 2.10GHz of batches 1-14 again,
where the records reproduce; by the user's second decision batch 15 returned to the pairing above with the recorded
runs, on the 2.10GHz CPU: the REF arms are dropped, segment 1's 2.80GHz Muon runs are kept apart (unused;
explore_out/gate_cap_2p80_results.json), its Adam run (PREV_CAP_A|260, bit-identical on both CPUs) is kept, and the
driver's per-segment Muon check requires S33's D8_HINGE|260 to reproduce its record through 1200.
OUTCOME: BOUND ROUTED at 28800 (test_stream_recipe.outcome).
READINGS (per arm; fixed before any run): "it breaks the eight-stream stall" if >= 3/10 BOUND ROUTED; "it does not"
if 0/10; otherwise neither. (Under the runtime cut PREV_CAP_A has 5 seeds; the same counts apply, as of 5.)
DIAGNOSTICS at 600, 1200, 2400, 4800, 9600 and the end: S38's decoder of the stream from h at key and stream-token
positions and from the gate input at key positions (S38's probe, as S40); the median |W_in v_t|, |W_prev v_(t-1)|,
|W_h h_(t-1)| (and |h|) at key positions; sigma_max(W_h) (and rho(W_h)); the distinct channels of the
value-position map (S39's rule); routing_k's map (distinct channels, streams per channel) at 4800, 9600 and the
end; failure classes; the hinge's firings; sigma_max(W_h) at init and the first update the cap acts.
CHECKS (child processes): with the cap at infinity, CAP_M and PREV_CAP_M equal their paired runs bit for bit
through 1200 (CAP_M vs S33's D8_HINGE|260: curve and statistics; PREV_CAP_M vs S40's D8_PREV_M|260: curve,
statistics and S40's decodability at 1200), and PREV_CAP_A equals S40's D8_PREV_A|260 (curve and statistics); over
50 steps of each CAP arm sigma_max(W_h) <= 0.5 + 1e-6 after every step; NOREC: W_h exactly 0 at every step, no
gradient, in no optimizer group, and every other parameter's initial value equals D8_PREV_M's; the optimizer groups
cover every trainable parameter once; S38's decoder check.
RUNTIME (batch level): if the projection is over 9 h, PREV_CAP_A is cut to seeds 260-264.
"""

import os
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common2 as c2
import explore_main9c as mt
import explore_muon_scale as s33
from test_channel_binding import SUB_LR
from test_router_confirm import mcnemar_exact

NAME = "gate_cap"
IDEA = ("cap the recurrent gain of the gate (sigma_max(W_h) <= 0.5 after every step), or remove the recurrence "
        "(GATE_PREV's input only)")
SOURCE = ("batch 14's S39 (at S=8 the stream leaves h as |W_h h_(t-1)| outgrows |W_in v_t|) and S40 (GATE_PREV's input "
          "carries the stream, h only partly; 0/10 bound)")
CHANGE = ("CAP: after every optimizer step W_h <- W_h * 0.5 / sigma_max(W_h) if sigma_max(W_h) > 0.5, optimizer state "
          "untouched; NOREC: W_h = 0, frozen; nothing else")
PAIRING = ("seeds 260-269: CAP_M with S33's recorded D8_HINGE; PREV_CAP_M and PREV_NOREC_M with S40's recorded D8_PREV_M; "
           "PREV_CAP_A with S40's recorded D8_PREV_A; same initial parameters (NOREC: W_h = 0) and batches; on the 2.10GHz "
           "CPU the records were made on")
CHILD = "explore_gate_cap_child"
LR = SUB_LR
MUON_LR = 0.005
SEEDS = tuple(range(260, 270))
CUT_SEEDS = tuple(range(260, 265))
BREAK_N, NOT_N = 3, 0
EQ_AT = 1200
DIAG = ("600", "1200", "2400", "4800", "9600", "end")
MAP_AT = (4800, 9600, "end")
TIME_STEPS = 200
_mu = ("Muon lr 0.005: gate W_in/W_h/W_g{} throughout, other Muon weights x0.1 for updates 1-2400; Adam on embedding and "
       "conv 0.0001 for 1-2400, 0.001 after; + main's HINGE hinge")
_ad = ("gate W_in/W_h/W_g/W_prev 0.001 throughout; every other parameter 0.0001 for updates 1-2400, 0.001 after; + main's "
       "HINGE hinge")
_cap = "; + CAP (sigma_max(W_h) <= 0.5 after every step)"
ARMS = {
    "CAP_M": dict(key="CAP_M", opt="muon", seeds=SEEDS, lr=LR, iters=28800, muon_lr=MUON_LR, prio=2,
                  label="S33's D8_HINGE + CAP (Muon)", sched=_mu.format("") + _cap),
    "PREV_CAP_M": dict(key="PREV_CAP_M", opt="muon", seeds=SEEDS, lr=LR, iters=28800, muon_lr=MUON_LR, prio=2,
                       label="S40's D8_PREV_M + CAP (Muon)", sched=_mu.format("/W_prev") + _cap),
    "PREV_NOREC_M": dict(key="PREV_NOREC_M", opt="muon", seeds=SEEDS, lr=LR, iters=28800, muon_lr=MUON_LR, prio=2,
                         label="S40's D8_PREV_M with NOREC (W_h = 0, frozen; Muon)",
                         sched=_mu.format("/W_prev (W_h frozen at 0)")),
    "PREV_CAP_A": dict(key="PREV_CAP_A", opt="adam", seeds=SEEDS, lr=LR, iters=28800, muon_lr=None, prio=2,
                       label="S40's D8_PREV_A + CAP (Adam)", sched=_ad + _cap),
}
REF_ARMS = ()                                # the 2.80GHz plan's REF arms, dropped by the user's second decision
SELF = None
PAIRED = {"CAP_M": ("muon_scale", "D8_HINGE", "S33 D8_HINGE"), "PREV_CAP_M": ("gate_prev", "D8_PREV_M", "S40 D8_PREV_M"),
          "PREV_NOREC_M": ("gate_prev", "D8_PREV_M", "S40 D8_PREV_M"), "PREV_CAP_A": ("gate_prev", "D8_PREV_A", "S40 D8_PREV_A")}
RECORDED = {}
MUON_REF = None


def child(func, payload, timeout=None):
    return mt.run_child(CHILD, func, payload, timeout=timeout)


def run_job(arm, seed):
    a = ARMS[arm]
    return child("run", dict(arm=arm, seed=seed, iters=a["iters"], mlr=a["muon_lr"]))


def segments(arm):
    raise NotImplementedError("S41 is projected by its own child timing (explore_batch15)")


def timing(arms):
    return child("timing", dict(which=list(arms), steps=TIME_STEPS, mlr=MUON_LR))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def _stats_upto(stats, t):
    return {s["step"]: s for s in stats if s["step"] != "end" and s["step"] <= t}


def same_upto(r, ref, t, diag=False):
    """Curve and statistics through t (and, diag, every diagnostic at or before t) equal."""
    want = [c for c in ref["curve"] if c[0] <= t]
    sn, sr = _stats_upto(r["stats"], t), _stats_upto(ref["stats"], t)
    ok = r["curve"] == want and sn.keys() == sr.keys() and all(sn[k] == sr[k] for k in sr) and len(sr) >= 2
    if diag:
        keys = [k for k in (ref.get("diag") or {}) if k != "end" and int(k) <= t]
        ok = ok and len(keys) >= 1 and all(r["diag"].get(k) == ref["diag"][k] for k in keys)
    return ok, want, sorted(sr)


def _fresh(arm, cap=None):
    p = dict(arm=arm, seed=260, iters=EQ_AT, mlr=ARMS[arm]["muon_lr"])
    if cap is not None:
        p["cap"] = cap
    return child("run", p)


def check():
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_cap = ex.submit(_fresh, "CAP_M", "inf")
        f_pcap = ex.submit(_fresh, "PREV_CAP_M", "inf")
        f_acap = ex.submit(_fresh, "PREV_CAP_A", "inf")
        fc = ex.submit(child, "checks", dict(mlr=MUON_LR))
        fd = ex.submit(mt.run_child, "explore_gate_state_child", "decoder_check", {})
        rows = []
        for arm, f in (("CAP_M", f_cap), ("PREV_CAP_M", f_pcap), ("PREV_CAP_A", f_acap)):
            r = f.result()
            store, key, lab = PAIRED[arm]
            ref = ec.load_store(store)["runs"][f"{key}|260"]
            ok, want, steps = same_upto(r, ref, EQ_AT)
            extra = ""
            if arm == "PREV_CAP_M":
                d0, d1 = ref["decode"][str(EQ_AT)], r["diag"][str(EQ_AT)]
                ok = ok and all(d0[k] == d1[k] for k in ("h_key", "h_stream", "u_key"))
                extra = (f"; S40's decodability at {EQ_AT} (h@key, h@stream, input@key) "
                         f"{[d0[k] for k in ('h_key', 'h_stream', 'u_key')]} vs {[d1[k] for k in ('h_key', 'h_stream', 'u_key')]}")
            rows.append((f"{arm} with the cap at infinity equals {lab}|260 bit for bit through {EQ_AT} (curves {r['curve']} vs "
                         f"{want}; statistics at {steps}{extra}; the cap acted on {r['n_cap']} updates; sigma_max(W_h) at init "
                         f"{r['sigma0']:.4f})", ok and r["n_cap"] == 0))
        rows += [tuple(x) for x in fc.result()] + [tuple(x) for x in fd.result()]
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    print(f"  CHECK {NAME}: children at {mt.MAIN_SHA[:7]} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def compare(new, old, seeds, lab_new, lab_old):
    both = [s for s in seeds if s in new and s in old]
    b = sum(1 for s in both if new[s] and not old[s])
    c = sum(1 for s in both if old[s] and not new[s])
    p = mcnemar_exact(b, c)
    print(f"    BOUND ROUTED  {lab_new} {sum(new[s] for s in both)}/{len(both)}  {lab_old} {sum(old[s] for s in both)}/{len(both)}  "
          f"{lab_new} only {b}, {lab_old} only {c}, McNemar two-sided p = {p:.3g} (printed)")
    return dict(n=len(both), new=sum(new[s] for s in both), old=sum(old[s] for s in both), b=b, c=c, p=p)


def f2(x, w=5):
    return f"{'--':>{w}}" if x is None else f"{x:{w}.2f}"


def hinge_at(r, step):
    for s in r.get("stats") or []:
        if s["step"] == step and s.get("hs_hinge") is not None:
            return s["hs_hinge"]
    if step == "end":
        return (r.get("end") or {}).get("hs_hinge")
    return None


def med(vals):
    v = [x for x in vals if x is not None]
    return statistics.median(v) if v else None


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    got = {arm: {int(k.split("|")[1]): r for k, r in store["runs"].items() if k.startswith(arm + "|") and r.get("ok")}
           for arm in ARMS}
    out = {}
    def runs_of(store_p, key_p):
        runs = store["runs"] if store_p is SELF else ec.load_store(store_p)["runs"]
        return {int(k.split("|")[1]): r for k, r in runs.items() if k.startswith(key_p + "|") and r.get("ok", True)}
    for arm, a in ARMS.items():
        rs = got[arm]
        store_p, key_p, plab = PAIRED[arm]
        pr = runs_of(store_p, key_p)
        rec = RECORDED.get(arm)
        rr = runs_of(rec[0], rec[1]) if rec else {}
        print(f"  ARM {arm} ({a['label']}), seeds {ec.fmt_seeds(a['seeds'])}; " + (
            f"a reference (no reading); printed with {plab}" if arm in REF_ARMS else f"paired with {plab}"
            + (f"; {rec[2]} printed alongside" if rec else "")))
        print(f"    {'seed':>4} | {arm:<22} {'trans':>5} {'stop':>5} {'ch@4800':>7} {'ch@9600':>7} {'ch@end':>9} {'hinge':>9} "
              f"{'sig0':>5} {'cap@':>5} {'#capped':>7} | {plab:<22}" + (f" | {rec[2]:<22}" if rec else ""))
        for s in a["seeds"]:
            r = rs.get(s)
            if r is None:
                print(f"    {s:>4} | {'not run':<22}")
                continue
            hf = hinge_at(r, "end")
            print(f"    {s:>4} | {r['outcome']:<22} {str(r['transition']):>5} {r['stopped_at']:>5} "
                  + " ".join(f"{s33.per_channel(s33.map_at(r, t)):>{9 if t == 'end' else 7}}" for t in MAP_AT)
                  + f" {('--' if hf is None else f'{hf[0]}/{hf[1]}'):>9} {r['sigma0']:5.2f} {str(r['first_cap']):>5} "
                  f"{str(r['n_cap']):>7} | {(pr.get(s) or {}).get('outcome', '--'):<22}"
                  + (f" | {(rr.get(s) or {}).get('outcome', '--'):<22}" if rec else ""))
        br = {s: r["outcome"] == "BOUND ROUTED" for s, r in rs.items()}
        bo = {s: r.get("outcome") == "BOUND ROUTED" for s, r in pr.items()}
        d = compare(br, bo, a["seeds"], arm, plab.replace(" ", "_"))
        if rec:
            compare(br, {s: r.get("outcome") == "BOUND ROUTED" for s, r in rr.items()}, a["seeds"], arm,
                    rec[2].replace(" ", "_"))
        print(f"    failure classes at the end: {arm} {dict(Counter(r['outcome'] for r in rs.values()))}; {plab} "
              f"{dict(Counter(pr[s]['outcome'] for s in a['seeds'] if s in pr))}"
              + (f"; {rec[2]} {dict(Counter(rr[s]['outcome'] for s in a['seeds'] if s in rr))}" if rec else ""))
        print(f"    distinct channels holding the 8 streams (routing_k's map; streams per channel), per seed:")
        for t in MAP_AT:
            ms = {s: s33.map_at(r, t) for s, r in sorted(rs.items())}
            print(f"      at {str(t):>5}: " + "  ".join(f"{s}:{s33.distinct(m) if m else '--'} ({s33.per_channel(m)})"
                                                  for s, m in ms.items()))
        print(f"    the hinge's firings (cumulative training batches with a term > 0 / training batches) at 2400, 4800, 9600, end:")
        print("      " + "  ".join(f"{s}: " + ", ".join((lambda h: '--' if h is None else f'{h[0]}/{h[1]}')(hinge_at(r, t))
                                                      for t in (2400, 4800, 9600, "end")) for s, r in sorted(rs.items())))
        print(f"    S38's decodability of the stream (chance 0.125) from h at key (stream-token) positions [and from the gate "
              f"input at key positions]:")
        print(f"    {'seed':>4} | " + " | ".join(f"{t:>5}: key  (str)  [inp]" for t in DIAG))
        for s, r in sorted(rs.items()):
            dg = r.get("diag") or {}
            print(f"    {s:>4} | " + " | ".join(
                f"{'':>5}  {f2((dg.get(t) or {}).get('h_key'))} ({f2((dg.get(t) or {}).get('h_stream'))}) "
                f"[{f2((dg.get(t) or {}).get('u_key'))}]" for t in DIAG))
        print(f"    median norms at key positions |W_in v_t| / |W_prev v_(t-1)| / |W_h h_(t-1)|, and sigma_max(W_h), per seed:")
        print(f"    {'seed':>4} | " + " | ".join(f"{t:>5}: Win  Wprev  Wh   sig" for t in DIAG))
        for s, r in sorted(rs.items()):
            dg = r.get("diag") or {}
            print(f"    {s:>4} | " + " | ".join(
                f"{'':>5}  {f2((dg.get(t) or {}).get('a_key'), 4)} {f2((dg.get(t) or {}).get('p_key'), 4)} "
                f"{f2((dg.get(t) or {}).get('b_key'), 5)} {f2((dg.get(t) or {}).get('sigma'), 4)}" for t in DIAG))
        cols = (("h_key", "dec h@key"), ("h_stream", "dec h@str"), ("u_key", "dec inp@key"), ("a_key", "|Win v|@key"),
                ("p_key", "|Wprev v|"), ("b_key", "|Wh h|@key"), ("hn_key", "|h|@key"), ("sigma", "sig_max"),
                ("rho", "rho"), ("sat_key", "sat@key"), ("distinct", "distinct(v)"))
        print(f"    medians over the runs (distinct(v): the value-position map's distinct channels, S39's rule):")
        print(f"    {'at':>5} {'n':>2} | " + " ".join(f"{lab:>11}" for _, lab in cols))
        meds = {}
        for t in DIAG:
            ds = [r["diag"][t] for r in rs.values() if t in (r.get("diag") or {})]
            meds[t] = {k: med([x[k] for x in ds]) for k, _ in cols}
            print(f"    {t:>5} {len(ds):>2} | " + " ".join(f"{'--':>11}" if meds[t][k] is None else f"{meds[t][k]:11.3f}"
                                                         for k, _ in cols))
        nb = sum(br.values())
        n = len(rs)
        if arm in REF_ARMS:
            rd = f"reference (no reading): BOUND ROUTED {nb}/{n}"
            print(f"  S41 {arm}: {rd}")
        else:
            rd = ("it breaks the eight-stream stall" if nb >= BREAK_N else "it does not" if nb <= NOT_N else "neither") + \
                 f" (BOUND ROUTED {nb}/{n}; >= {BREAK_N} breaks, {NOT_N} does not)"
            print(f"  READING S41 {arm}: {rd}")
        out[arm] = dict(d, bound_routed=nb, n=n, reading=rd, complete=n == len(a["seeds"]), medians=meds,
                        classes=dict(Counter(r["outcome"] for r in rs.values())),
                        first_cap=[r.get("first_cap") for r in rs.values()], sigma0=[r.get("sigma0") for r in rs.values()])
    return out
