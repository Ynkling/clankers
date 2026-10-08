#!/usr/bin/env python
"""
explore_recipe_ablation.py — EXPLORATORY, not a result. Screen S50 (batch 18): ablations of the window recipe at eight
streams, and the split's missing reset-only control.

BACKGROUND
- Batch 17 S48: W_SPLIT (LOCAL3 + SLOW + S37's KEYMASS plateau trigger, <= 3 splits) bound and routed 8 streams on 9/10
  seeds (260-269) vs 3/10 without the split (6 vs 0, p = 0.031); all 19 splits labelled on target. Decay 0.98 alone 1/10
  (it lost the reference's three binders); 43200 updates without the split 4/10. W_NOSLOW, W_WIDTH4 and W_SPLIT_M were cut
  by the runtime rule.
- Batch 16 S44(a): LOCAL3_SLOW16_A 20/20 BOUND ROUTED at S=4, k=16. S43: LOCAL3 33/40, LOCAL3_SLOW 31/40, SLOW_HINGE 34/40
  at S=2, k=2; the window gate's failures are KEY splits that form before update 600 (S49) and STREAM-PARTIAL gates. The
  audit (docs/audit_2026-10.md, section 2.3) noted that S36/S37's split had no reset-only control; S24's NOISE arm was that
  control (1/8).
- The recipe under test: the width-3 window gate + SLOW (the memory at lr/10 for 2400 updates) + the plateau-triggered
  split. This screen ablates its parts at eight streams and adds the missing control.

DEFINITIONS (batches 16-17): LOCAL3: the gate from a width-3 causal convolution of the embeddings, h = tanh(W_in u),
g = softmax(W_g h). SLOW: gate group (W_in, W_g, window) at 1e-3 throughout; every other parameter at 1e-4 for updates
1-2400, then 1e-3 (Muon: S25's groups, Muon lr 0.005, the same x0.1 slow phase). SPLIT: S37's KEYMASS trigger as run in
batch 17 (checks every 2400 from 4800 on the 64-sequence probe; fire if probe acc < 0.95 and it rose < 0.02; <= 3 splits,
>= 4800 apart; W_g[c0] <- W_g[c*] + N(0, (0.1 std)^2) (and W_g[c*] likewise); W_g's optimizer state reset). Outcome
BOUND ROUTED (test_stream_recipe.outcome: bound, and one-to-one map + every stream's accuracy >= 0.9).

ARMS (S=8, P=4, k=16, conv, 28800 updates, seeds 260-269; explore_recipe_ablation_child; Adam lr 1e-3 = test_slow_start.LR
read in a child; Muon lr 0.005; 1 torch thread per run):
  W_RESET         LOCAL3_SLOW_D8_A + W_g's Adam-state reset at exactly the updates where W_SPLIT fired on that seed (read
                  from batch 17's records), with NO row copy: the reset-only control.
  W_SPLIT_NOSLOW  W_SPLIT with a single rate 1e-3 throughout (no slow phase).
  W_SPLIT_W4      W_SPLIT with window width 4.
  W_SPLIT_M       W_SPLIT under Muon. Its reference: batch 16's LOCAL3_SLOW_D8_M (3/10) if its seed-260 record reproduces
                  on this CPU through 1200 (decided before the projection, in the first segment), otherwise W_M_REF
                  (LOCAL3_SLOW_D8_M fresh on this CPU, a reference, no reading). Muon runs pair only within a CPU.
PAIRING: every arm with batch 17's W_SPLIT (explore_out/window_d8_results.json, code 375b30d487df; 9/10 BOUND ROUTED);
same seeds = same initial parameters (+ the window; W_SPLIT_W4's is drawn at width 4) and batches.

READINGS (fixed before any run; BOUND ROUTED counts; exact McNemar two-sided against W_SPLIT printed; the rule is on counts):
  W_RESET         "the copy does the work" if <= 4/10; "the reset does the work" if >= 8/10; otherwise neither.
  W_SPLIT_NOSLOW  "the slow phase is unnecessary at eight streams" if >= 8/10; "needed" if <= 5/10; otherwise neither.
  W_SPLIT_W4      "width 3 suffices" if within 1 of 9/10 (8-10/10); otherwise printed, and which way (<= 7/10: width 4
                  does worse).
  W_SPLIT_M       "the recipe carries to Muon" if >= 7/10 and (W_SPLIT_M - reference) >= 4; otherwise not shown.
An arm cut by the runtime rule has no reading.
DIAGNOSTICS (batch 17's S48): the map at every check (every 2400) and the end; each split's labelled target (c* holds
>= 2 streams and c0 none); W_RESET's resets (the state tensors zeroed); S38's decoder (the gate state at CTX / KEY / VAL)
at 1200, 2400, 4800, 9600 and the end; transitions; each run's agreement with its pair before they diverge (W_RESET:
batch 16's LOCAL3_SLOW_D8_A through its first reset; W_SPLIT_M: its Muon pair through its first split).
CHECKS (child and parent): the reset zeroes the same optimizer-state tensors as S36's split_op and leaves W_g unchanged;
the width-4 gate cannot see token t-4 and can see t-3; the lrs of W_SPLIT_NOSLOW / W_SPLIT_W4 / W_RESET at updates 1, 2400,
2401 through the real path; with the threshold at 0, W_SPLIT_NOSLOW|260 equals S48's single-rate recipe (W_NOSLOW) and
W_SPLIT_W4|260 equals S48's width-4 recipe (W_WIDTH4) bit for bit through 3600 (both diverge from LOCAL3_SLOW_D8_A at
update 1 by construction; these are their no-split counterparts); W_RESET|261 equals batch 16's LOCAL3_SLOW_D8_A|261
through its first reset (4800), resets the state there and differs after; W_RESET's reset updates equal batch 17's
recorded split updates on every seed; (if W_SPLIT_M runs) the trigger zeroes W_g's Muon state and, with the threshold at
0, W_SPLIT_M|260 equals its Muon pair through 3600.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_b16_report as rp
import explore_window_d8 as s48

NAME = "window_recipe_ablation"
CHILD = "explore_recipe_ablation_child"
MAIN = True
IDEA = "ablate the window recipe at eight streams (reset-only control, no slow phase, width 4, Muon)"
SOURCE = "batch 17's S48 (W_SPLIT 9/10); the audit, section 2.3 (no reset-only control); the user's batch 18"
CHANGE = ("W_RESET: the split's optimizer reset without the copy; W_SPLIT_NOSLOW: one rate; W_SPLIT_W4: width-4 window; "
          "W_SPLIT_M: Muon")
PAIRING = ("seeds 260-269 with batch 17's W_SPLIT (9/10); W_SPLIT_M also with its Muon reference on this CPU; same initial "
           "parameters (+ the window) and batches")
LR = 1e-3
MUON_LR = 0.005
ITERS = 28800
SB = tuple(range(260, 270))
B17, B17_SHA = "window_d8", "375b30d487df"
REF_CHECK_ITERS = 1200
RECIPE_CHECK_ITERS, RESET_CHECK_SEED, RESET_CHECK_ITERS, MUON_CHECK_ITERS = 3600, 261, 6000, 3600
DIAG = ("1200", "2400", "4800", "9600", "end")
_sl = (f"gate W_in/W_g/window {LR:g} throughout; every other trainable parameter {LR / 10:g} for updates 1-2400, {LR:g} "
       f"after; W_h frozen (unused); no hinge")
_one = f"every trainable parameter {LR:g} throughout; W_h frozen (unused); no hinge"
_mu = (f"Muon lr {MUON_LR:g}: W_in/W_g throughout, other Muon weights x0.1 for updates 1-2400; Adam: the window {LR:g} "
       f"throughout, embedding/conv/1-D {LR / 10:g} for 1-2400, {LR:g} after; W_h frozen; no hinge")
_trig = ("; + S37's KEYMASS trigger (every 2400 from 4800: probe acc < 0.95 and rise < 0.02; <= 3 splits, >= 4800 apart) -> "
         "S24's SPLIT of W_g's row c* onto c0, W_g's optimizer state zeroed")
ARMS = {
    "W_RESET": dict(key="W_RESET", opt="adam", lr=LR, muon_lr=None, iters=ITERS, seeds=SB,
                    label="LOCAL3_SLOW_D8_A + W_g's Adam-state reset at W_SPLIT's split updates, no copy",
                    sched=_sl + "; W_g's Adam state zeroed after the evaluations at batch 17's W_SPLIT split updates (per seed)"),
    "W_SPLIT_NOSLOW": dict(key="W_SPLIT_NOSLOW", opt="adam", lr=LR, muon_lr=None, iters=ITERS, seeds=SB,
                           label="W_SPLIT at a single rate", sched=_one + _trig),
    "W_SPLIT_W4": dict(key="W_SPLIT_W4", opt="adam", lr=LR, muon_lr=None, iters=ITERS, seeds=SB,
                       label="W_SPLIT with a width-4 window", sched=_sl + "; window width 4" + _trig),
    "W_SPLIT_M": dict(key="W_SPLIT_M", opt="muon", lr=LR, muon_lr=MUON_LR, iters=ITERS, seeds=SB,
                      label="W_SPLIT under Muon (LOCAL3_SLOW_D8_M + the trigger)", sched=_mu + _trig),
    "W_M_REF": dict(key="W_M_REF", opt="muon", lr=LR, muon_lr=MUON_LR, iters=ITERS, seeds=SB, ref=True,
                    needed_by=(("window_recipe_ablation", "W_SPLIT_M"),),
                    label="LOCAL3_SLOW_D8_M, fresh on this CPU (W_SPLIT_M's reference)", sched=_mu),
}
CUT = set()
M_REF = dict(source="W_M_REF")              # set by configure_ref: "W_M_REF" or "batch 16"


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def b17_runs(arm="W_SPLIT", seeds=SB):
    st = ec.load_store(B17)["runs"]
    out = {}
    for k, r in st.items():
        a, s, h, cpu = k.split("|", 3)
        if a == arm and h == B17_SHA and int(s) in seeds and r.get("ok"):
            out[int(s)] = r
    return out


def reset_at(seed):
    r = b17_runs(seeds=(seed,)).get(seed)
    return [c["step"] for c in (r or {}).get("checks", []) if c.get("fired")]


def run_job(arm, seed):
    a = ARMS[arm]
    p = dict(arm=arm, seed=seed, iters=a["iters"], mlr=a["muon_lr"])
    if arm == "W_RESET":
        p["reset_at"] = reset_at(seed)
    return child("run", p)


def timing(arms):
    return child("timing", dict(which=list(arms), steps=300, mlr=MUON_LR))


def n_meas(arm):
    return ARMS[arm]["iters"] // 1200


def n_extra(arm):
    return ARMS[arm]["iters"] // 2400 - 1 if "SPLIT" in arm else 0


# ── The Muon reference ───────────────────────────────────────────────────────
def muon_ref_decision():
    """Does batch 16's LOCAL3_SLOW_D8_M|260 record reproduce on this CPU through 1200 (curve, statistics, decoder)?"""
    ref = s48.b16_runs("LOCAL3_SLOW_D8_M", (260,)).get(260)
    r = child("run", dict(arm="W_M_REF", seed=260, iters=REF_CHECK_ITERS, mlr=MUON_LR))
    same = ref is not None and s48.eq_upto(r, ref, REF_CHECK_ITERS)
    return dict(reproduces=bool(same), cpu=ec.cpu_model(), record_cpu=(ref or {}).get("cpu"),
                curve=r["curve"], record_curve=[c for c in (ref or {}).get("curve", []) if c[0] <= REF_CHECK_ITERS])


def configure_ref(dec):
    if dec and dec.get("reproduces"):
        ARMS.pop("W_M_REF", None)
        M_REF["source"] = "batch 16"
    else:
        M_REF["source"] = "W_M_REF"


def m_pair(store=None):
    if M_REF["source"] == "batch 16":
        return "LOCAL3_SLOW_D8_M (batch 16)", s48.b16_runs("LOCAL3_SLOW_D8_M")
    me = sys.modules[__name__]
    return "W_M_REF", (c16.runs_of(me, "W_M_REF", store=store) if "W_M_REF" in ARMS else {})


# ── CHECKs ───────────────────────────────────────────────────────────────────
def run_checks_parent():
    from concurrent.futures import ThreadPoolExecutor
    jobs = []
    if "W_SPLIT_NOSLOW" in ARMS:
        jobs += [("NOSLOW", CHILD, dict(arm="W_SPLIT_NOSLOW", seed=260, iters=RECIPE_CHECK_ITERS, thr=0.0, total=ITERS)),
                 ("NOSLOW_REF", s48.CHILD, dict(arm="W_NOSLOW", seed=260, iters=RECIPE_CHECK_ITERS))]
    if "W_SPLIT_W4" in ARMS:
        jobs += [("W4", CHILD, dict(arm="W_SPLIT_W4", seed=260, iters=RECIPE_CHECK_ITERS, thr=0.0, total=ITERS)),
                 ("W4_REF", s48.CHILD, dict(arm="W_WIDTH4", seed=260, iters=RECIPE_CHECK_ITERS))]
    jobs.append(("RESET", CHILD, dict(arm="W_RESET", seed=RESET_CHECK_SEED, iters=RESET_CHECK_ITERS,
                                      reset_at=reset_at(RESET_CHECK_SEED))))
    if "W_SPLIT_M" in ARMS:
        jobs.append(("M", CHILD, dict(arm="W_SPLIT_M", seed=260, iters=MUON_CHECK_ITERS, thr=0.0, mlr=MUON_LR, total=ITERS)))
        if M_REF["source"] == "W_M_REF":
            jobs.append(("M_REF", CHILD, dict(arm="W_M_REF", seed=260, iters=MUON_CHECK_ITERS, mlr=MUON_LR)))
    with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as ex:
        res = dict(zip([j[0] for j in jobs], ex.map(lambda j: c16.run_child(j[1], "run", j[2], main=True), jobs)))
    rows = []
    for k, ref_k, lab in (("NOSLOW", "NOSLOW_REF", "S48's single-rate recipe W_NOSLOW"), ("W4", "W4_REF", "S48's width-4 recipe W_WIDTH4")):
        if k in res:
            r, q = res[k], res[ref_k]
            same = s48.eq_upto(r, q, RECIPE_CHECK_ITERS) and r["curve"] == q["curve"]
            rows.append((f"with the threshold at 0, {r['arm']}|260 equals {lab}|260 bit for bit through {RECIPE_CHECK_ITERS} (curve "
                         f"{r['curve']}; statistics and decoder equal {same}; trigger checks logged at {[c['step'] for c in r['checks']]}, "
                         f"{r['splits']} splits)", same and r["splits"] == 0 and len(r["checks"]) >= 1))
    r = res["RESET"]
    ref = s48.b16_runs("LOCAL3_SLOW_D8_A", (RESET_CHECK_SEED,)).get(RESET_CHECK_SEED)
    first = reset_at(RESET_CHECK_SEED)[0]
    same = ref is not None and s48.eq_upto(r, ref, first)
    rs = [x for x in r.get("resets", []) if x["step"] <= RESET_CHECK_ITERS]
    differs = ref is not None and [c for c in r["curve"] if c[0] > first] != [c for c in ref["curve"] if first < c[0] <= RESET_CHECK_ITERS]
    ok = (same and differs and len(rs) == 1 and rs[0]["step"] == first and rs[0]["w_g_unchanged"]
          and all(v > 0 for k, v in rs[0]["state_before_max"].items() if k != "step")
          and all(v == 0 for v in rs[0]["state_after_max"].values()))
    rows.append((f"W_RESET|{RESET_CHECK_SEED} (resets at {reset_at(RESET_CHECK_SEED)}) equals batch 16's LOCAL3_SLOW_D8_A|"
                 f"{RESET_CHECK_SEED} bit for bit through its first reset ({first}: {same}); the reset there zeroed "
                 f"{rs[0]['state_zeroed'] if rs else '?'} (max before {rs[0]['state_before_max'] if rs else '?'}, W_g unchanged "
                 f"{rs[0]['w_g_unchanged'] if rs else '?'}); the run differs from the record after it ({differs}); curve {r['curve']}", ok))
    table = {s: reset_at(s) for s in SB}
    rec = b17_runs()
    ok = len(rec) == len(SB) and all(table[s] == [c["step"] for c in rec[s]["checks"] if c["fired"]] and table[s] for s in SB)
    rows.append((f"W_RESET's reset updates per seed are batch 17's W_SPLIT split updates ({len(rec)} records of code {B17_SHA}): "
                 f"{table}", ok))
    if "M" in res:
        rm = res["M"]
        pm = res["M_REF"] if "M_REF" in res else s48.b16_runs("LOCAL3_SLOW_D8_M", (260,)).get(260)
        okm = pm is not None and s48.eq_upto(rm, pm, MUON_CHECK_ITERS) and rm["splits"] == 0
        rows.append((f"with the threshold at 0, W_SPLIT_M|260 equals its Muon pair ({M_REF['source']}) bit for bit through "
                     f"{MUON_CHECK_ITERS} (curve {rm['curve']}; {okm})", okm))
    return rows


def check():
    rows = [tuple(x) for x in child("checks", dict(mlr=MUON_LR, muon="W_SPLIT_M" in ARMS))]
    rows += run_checks_parent()
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def resets_block(runs):
    print("    resets (update: W_g's optimizer-state tensors zeroed; their max |value| before; W_g unchanged):")
    for s, r in sorted(runs.items()):
        print(f"      {s}: reset_at {r.get('reset_at')}; " + "; ".join(
            f"{x['step']}: {x['state_zeroed']} (before {x['state_before_max']}, W_g unchanged {x['w_g_unchanged']})"
            for x in r.get("resets") or []))


def agree_reset(runs, d8a):
    rows = {}
    for s, r in runs.items():
        if s in d8a and r.get("reset_at"):
            t = r["reset_at"][0]
            rows[s] = (t, s48.eq_upto(r, d8a[s], min(t, r["iters"])))
    print(f"    agreement before the arms diverge — equal to batch 16's LOCAL3_SLOW_D8_A through the first reset (step): {rows}")
    return rows


def reading(arm, o, ref_n=None):
    n_seeds = len(ARMS[arm]["seeds"])
    if not o["complete"]:
        return f"INCOMPLETE ({o['n']}/{n_seeds}), no reading"
    nb = o["bound_routed"]
    vs = f"vs W_SPLIT: arm only {o['b']}, W_SPLIT only {o['c']}, p = {o['p']:.3g}"
    if arm == "W_RESET":
        v = "the copy does the work" if nb <= 4 else "the reset does the work" if nb >= 8 else "neither"
        rule = "copy: <= 4/10; reset: >= 8/10"
    elif arm == "W_SPLIT_NOSLOW":
        v = "the slow phase is unnecessary at eight streams" if nb >= 8 else "needed" if nb <= 5 else "neither"
        rule = "unnecessary: >= 8/10; needed: <= 5/10"
    elif arm == "W_SPLIT_W4":
        v = ("width 3 suffices" if abs(nb - 9) <= 1 else f"width 4 does worse ({nb}/10 against 9/10)")
        rule = "suffices: within 1 of 9/10"
    else:
        hit = ref_n is not None and nb >= 7 and nb - ref_n >= 4
        v = "the recipe carries to Muon" if hit else "not shown"
        rule = f"carries: >= 7/10 and W_SPLIT_M - reference >= 4 (reference {ref_n})"
    return f"{v} (BOUND ROUTED {nb}/{o['n']}; {rule}; {vs})"


def report():
    me = sys.modules[__name__]
    import explore_common2 as c2
    c2.print_screen_header2(me)
    store = ec.load_store(NAME)
    got = {arm: c16.runs_of(me, arm, store=store) for arm in ARMS}
    ws = b17_runs()
    d8a = s48.b16_runs("LOCAL3_SLOW_D8_A")
    mlab, mruns = m_pair(store)
    print(f"  pairs: batch 17's W_SPLIT (code {B17_SHA}; {len(ws)} runs on {sorted({r['cpu'] for r in ws.values()})}), BOUND ROUTED "
          f"{sum(rp.br(r) for r in ws.values())}/{len(ws)}; Muon reference: {mlab}, BOUND ROUTED "
          f"{sum(rp.br(r) for r in mruns.values())}/{len(mruns)}")
    if CUT:
        print(f"  cut by the runtime rule: {sorted(CUT)} (no runs, no reading)")
    out = {}
    for arm, a in ARMS.items():
        rs = got[arm]
        print(f"  ARM {arm} ({a['label']}), seeds {ec.fmt_seeds(a['seeds'])}; {a['sched']}")
        plab, pr = ("batch 16's LOCAL3_SLOW_D8_M (2.80GHz)", s48.b16_runs("LOCAL3_SLOW_D8_M")) if a.get("ref") else ("W_SPLIT", ws)
        s48.maps_table(arm, a, rs, plab, pr)
        d = rp.compare(rs, pr, a["seeds"], arm, "B16_D8_M" if a.get("ref") else "W_SPLIT")
        print(f"    failure classes at the end: {arm} {rp.classes(rs)}; {plab} {rp.classes({s: pr[s] for s in a['seeds'] if s in pr})}")
        print(f"    transitions: {arm} {rp.transitions(rs)}; {plab} {rp.transitions({s: pr[s] for s in a['seeds'] if s in pr})}")
        if "SPLIT" in arm:
            s48.splits_block(rs)
        if arm == "W_RESET":
            resets_block(rs)
            agree_reset(rs, d8a)
        meds = rp.decode_block(rs, steps=DIAG, chance=1 / 8)
        dm = None
        if arm == "W_SPLIT_M":
            dm = rp.compare(rs, mruns, a["seeds"], arm, mlab.split(" ")[0])
            s48.agreement("W_SPLIT_M", rs, mruns, mlab)
        nb, n = sum(rp.br(r) for r in rs.values()), len(rs)
        out[arm] = dict(d, bound_routed=nb, n=n, complete=n == len(a["seeds"]), classes=rp.classes(rs), meds=meds,
                        bound_seeds=sorted(s for s, r in rs.items() if rp.br(r)), vs_ref=dm)
        if a.get("ref"):
            print(f"  S50 {arm}: reference (no reading): BOUND ROUTED {nb}/{n}")
        else:
            ref_n = None
            if arm == "W_SPLIT_M":
                ref_n = sum(rp.br(mruns.get(s)) for s in a["seeds"]) if len(mruns) == len(a["seeds"]) else None
            out[arm]["reading"] = reading(arm, out[arm], ref_n)
            print(f"  READING S50 {arm}: {out[arm]['reading']}")
        print()
    return out
