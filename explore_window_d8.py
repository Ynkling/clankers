#!/usr/bin/env python
"""
explore_window_d8.py — EXPLORATORY, not a result. Screen S48 (batch 17): seven one-knob changes to batch 16's eight-stream
window-gate arm LOCAL3_SLOW_D8_A (BOUND ROUTED 3/10 on seeds 260-269: 261, 263, 264, transitions 16800-21600; the seven
unbound runs ended MERGED 2-3 share on 6-7 channels) and to LOCAL3_SLOW_D8_M (3/10: 261, 264, 267, transitions
7200-9600; the seven unbound held fixed maps from 4800).

CONFIGURATION: S44(b)'s (S=8, P=4, k=16, conv, all 8 streams from step 1; test_stream_curriculum.run_sc with ARM["D8"] at
main's 9c5939e in child processes), seeds 260-269, Adam unless noted, outcome BOUND ROUTED (test_stream_recipe.outcome).
Adam lr 1e-3 = test_slow_start.LR (read in a child), the slow phase's non-gate group at 1e-4 for updates 1-2400; Muon lr
0.005 (S33's). 1 torch thread per run.

ARMS (explore_window_d8_child; each is LOCAL3_SLOW_D8_A with one change unless noted):
  W_SPLIT      S37's KEYMASS plateau trigger on: checks every 2400 from 4800 (2400 = reference), at most 3 splits per run
               (>= 4800 apart, S37's gap), 28800 updates.
  W_D98        the memory's decay 0.98 (0.95 in every eight-stream run so far).
  W_SPLIT_D98  both.
  W_LONG       43200 updates, no other change.
  W_NOSLOW     a single rate: every trainable parameter at 1e-3 throughout.
  W_WIDTH4     the window of width 4 (the gate at t sees tokens t-3 .. t).
  W_SPLIT_M    LOCAL3_SLOW_D8_M (Muon) + the trigger.
  W_M_REF      LOCAL3_SLOW_D8_M fresh on this batch's CPU (a reference for W_SPLIT_M; no reading). Runs only when the
               batch's CPU is not the CPU of batch 16's LOCAL3_SLOW_D8_M records (Muon runs in bf16 and pairs only within
               a CPU), and only while W_SPLIT_M runs.
PAIRING: every Adam arm with batch 16's LOCAL3_SLOW_D8_A records (explore_out/window_scale_results.json, code
01767493e840; Adam runs are bit-identical across the CPUs the repro check has passed on); W_SPLIT_M with
LOCAL3_SLOW_D8_M on the same CPU (batch 16's records if the batch's CPU is theirs, else W_M_REF). Same seeds = same
initial parameters (+ the window; W_WIDTH4's window is drawn at width 4) and batches.

READING per arm (fixed before any run): "improves" if BOUND ROUTED >= 6/10; "does not" if <= 3/10; otherwise neither.
McNemar (exact, two-sided) against the paired arm is printed with it. An arm cut by the runtime rule has no reading.
DIAGNOSTICS: routing_k's map (streams per channel) at every check (every 2400) and at the end; the trigger's checks, the
splits fired and their targets (KEYMASS c* -> c0, labelled: c* holds >= 2 streams and c0 none), blocked checks and why;
S38's decoder (the gate state h at CTX / KEY / VAL positions) at 1200, 2400, 4800, 9600, 19200 and the end; transitions;
failure classes; each run's agreement with its pair before the arms diverge (W_SPLIT: equal to LOCAL3_SLOW_D8_A through
the first split, or throughout if none fired; W_SPLIT_D98 to W_D98 likewise; W_SPLIT_M to its Muon pair likewise;
W_LONG: equal to LOCAL3_SLOW_D8_A through 28800).
CHECKS (explore_window_d8_child.checks and the parent): the width-4 conversion keeps every parameter and adds a (D, 4)
window from batch 1's rule; width 3 is batch 16's conversion; the gate at t ignores t-width and sees t-width+1; the
windows at [1, 0, 0] and [1, 0, 0, 0] give the same gate; decay 0.98 enters the decay mask only (S46's set_decay on the
LOCAL3 model); W_NOSLOW / W_WIDTH4 / W_D98's groups and lrs at updates 1, 2400, 2401 through the real path; the trigger
at cap 2 equals S37's row for row, at cap 3 fires at 4800, 9600, 14400 (gap-blocked between, capped after) and zeroes
W_g's optimizer state (Adam; Muon if W_SPLIT_M runs); with the threshold at 0, W_SPLIT|260 equals batch 16's
LOCAL3_SLOW_D8_A|260 bit for bit through 6000 (curve, statistics, decoder) — the recipe, the trigger's measurements and
the every-evaluation decoder are inert; (if W_SPLIT_M runs) with the threshold at 0 it equals its Muon pair through 3600.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_b16_report as rp

NAME = "window_d8"
CHILD = "explore_window_d8_child"
MAIN = True
IDEA = "one-knob changes to the eight-stream window gate (LOCAL3_SLOW_D8_A, 3/10): splits, decay 0.98, length, one rate, width 4"
SOURCE = "batch 16's S44(b) (the window gate broke the eight-stream stall at 3/10 Adam and 3/10 Muon); the user's batch 17"
CHANGE = ("LOCAL3_SLOW_D8_A + S37's KEYMASS trigger (<= 3 splits) / decay 0.98 / both / 43200 updates / one rate 1e-3 / a "
          "width-4 window; LOCAL3_SLOW_D8_M + the trigger")
PAIRING = ("seeds 260-269: the Adam arms with batch 16's LOCAL3_SLOW_D8_A records; W_SPLIT_M with LOCAL3_SLOW_D8_M on the same "
           "CPU (batch 16's records or W_M_REF); same initial parameters (+ the window) and batches")
LR = 1e-3
MUON_LR = 0.005
ITERS = 28800
LONG = 43200
SB = tuple(range(260, 270))
IMPROVES_N, NOT_N = 6, 3
B16 = "window_scale"
B16_SHA = "01767493e840"                     # batch 16's S44 code SHA (its LOCAL3_SLOW_D8_A / _M records)
SPLIT_CHECK_ITERS, MUON_CHECK_ITERS = 6000, 3600
MAP_EVERY = 2400
DIAG = ("1200", "2400", "4800", "9600", "19200", "end")
_sl = (f"gate W_in/W_g/window {LR:g} throughout; every other trainable parameter {LR / 10:g} for updates 1-2400, {LR:g} "
       f"after; W_h frozen (unused); no hinge")
_mu = (f"Muon lr {MUON_LR:g}: W_in/W_g throughout, other Muon weights x0.1 for updates 1-2400; Adam: the window {LR:g} "
       f"throughout, embedding/conv/1-D {LR / 10:g} for 1-2400, {LR:g} after; W_h frozen; no hinge")
_trig = ("; + S37's KEYMASS trigger (every 2400 from 4800: probe acc < 0.95 and rise < 0.02; <= 3 splits, >= 4800 apart) -> "
         "S24's SPLIT of W_g's row c* onto c0, W_g's optimizer state zeroed")
ARMS = {
    "W_SPLIT": dict(key="W_SPLIT", opt="adam", lr=LR, muon_lr=None, iters=ITERS, seeds=SB, pair="D8_A",
                    label="LOCAL3_SLOW_D8_A + the KEYMASS trigger (<= 3 splits)", sched=_sl + _trig),
    "W_D98": dict(key="W_D98", opt="adam", lr=LR, muon_lr=None, iters=ITERS, seeds=SB, pair="D8_A",
                  label="LOCAL3_SLOW_D8_A at decay 0.98", sched=_sl + "; decay 0.98"),
    "W_SPLIT_D98": dict(key="W_SPLIT_D98", opt="adam", lr=LR, muon_lr=None, iters=ITERS, seeds=SB, pair="D8_A",
                        label="LOCAL3_SLOW_D8_A at decay 0.98 + the KEYMASS trigger", sched=_sl + "; decay 0.98" + _trig),
    "W_LONG": dict(key="W_LONG", opt="adam", lr=LR, muon_lr=None, iters=LONG, seeds=SB, pair="D8_A",
                   label="LOCAL3_SLOW_D8_A to 43200 updates", sched=_sl),
    "W_NOSLOW": dict(key="W_NOSLOW", opt="adam", lr=LR, muon_lr=None, iters=ITERS, seeds=SB, pair="D8_A",
                     label="LOCAL3_SLOW_D8_A at a single rate", sched=f"every trainable parameter {LR:g} throughout; W_h frozen "
                                                                  f"(unused); no hinge"),
    "W_WIDTH4": dict(key="W_WIDTH4", opt="adam", lr=LR, muon_lr=None, iters=ITERS, seeds=SB, pair="D8_A",
                     label="LOCAL3_SLOW_D8_A with a width-4 window", sched=_sl + "; window width 4"),
    "W_SPLIT_M": dict(key="W_SPLIT_M", opt="muon", lr=LR, muon_lr=MUON_LR, iters=ITERS, seeds=SB, pair="D8_M",
                      label="LOCAL3_SLOW_D8_M + the KEYMASS trigger (<= 3 splits)", sched=_mu + _trig),
    "W_M_REF": dict(key="W_M_REF", opt="muon", lr=LR, muon_lr=MUON_LR, iters=ITERS, seeds=SB, pair=None, ref=True,
                    needed_by=(("window_d8", "W_SPLIT_M"),),
                    label="LOCAL3_SLOW_D8_M, fresh on this CPU (reference)", sched=_mu),
}
CUT = set()                                    # arms cut by the runtime rule (set by the driver)
M_REF_FROM = ["W_M_REF"]                        # the driver sets ["batch 16", cpu] when batch 16's records are on this CPU


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    a = ARMS[arm]
    return child("run", dict(arm=arm, seed=seed, iters=a["iters"], mlr=a["muon_lr"]))


def timing(arms):
    return child("timing", dict(which=list(arms), steps=300, mlr=MUON_LR))


def n_meas(arm):
    return ARMS[arm]["iters"] // 1200                     # the decoder after every evaluation


def n_extra(arm):
    return ARMS[arm]["iters"] // MAP_EVERY - 1 if "SPLIT" in arm else 0      # the trigger's checks, about an evaluation each


# ── Batch 16's records ───────────────────────────────────────────────────────
def b16_runs(arm, seeds=SB):
    """{seed: record} of batch 16's S44 arm (code B16_SHA), with the CPU it ran on."""
    st = ec.load_store(B16)["runs"]
    out = {}
    for k, r in st.items():
        a, s, h, cpu = k.split("|", 3)
        if a == arm and h == B16_SHA and int(s) in seeds and r.get("ok"):
            out[int(s)] = r
    return out


def b16_cpus(arm):
    return sorted({r["cpu"] for r in b16_runs(arm).values()})


def configure_ref(batch_cpu):
    """W_M_REF runs only when batch 16's LOCAL3_SLOW_D8_M records are not on this batch's CPU."""
    cpus = b16_cpus("LOCAL3_SLOW_D8_M")
    if cpus == [batch_cpu] and len(b16_runs("LOCAL3_SLOW_D8_M")) == len(SB):
        ARMS.pop("W_M_REF", None)
        M_REF_FROM[:] = ["batch 16", batch_cpu]
        return False
    return True


def m_pair(store=None):
    if M_REF_FROM[0] == "batch 16":
        return "LOCAL3_SLOW_D8_M (batch 16, same CPU)", b16_runs("LOCAL3_SLOW_D8_M")
    me = sys.modules[__name__]
    return "W_M_REF (LOCAL3_SLOW_D8_M fresh, same CPU)", (c16.runs_of(me, "W_M_REF", store=store) if "W_M_REF" in ARMS else {})


# ── CHECKs ───────────────────────────────────────────────────────────────────
def eq_upto(r, ref, t):
    """Curve, statistics and the decoder measurements both runs have equal through step t (and something to compare)."""
    c1 = [c for c in r["curve"] if c[0] <= t]
    c2 = [c for c in ref["curve"] if c[0] <= t]
    s1 = [s for s in r["stats"] if s["step"] != "end" and s["step"] <= t]
    s2 = [s for s in ref["stats"] if s["step"] != "end" and s["step"] <= t]
    d1, d2 = r.get("decode") or {}, ref.get("decode") or {}
    common = [k for k in d1 if k.isdigit() and int(k) <= t and k in d2]
    return bool(c1) and c1 == c2 and s1 == s2 and all(d1[k] == d2[k] for k in common)


def split_check():
    from concurrent.futures import ThreadPoolExecutor
    ref = b16_runs("LOCAL3_SLOW_D8_A", (260,)).get(260)
    jobs = [("W_SPLIT", dict(arm="W_SPLIT", seed=260, iters=SPLIT_CHECK_ITERS, thr=0.0, total=ITERS))]
    if "W_SPLIT_M" in ARMS:
        jobs.append(("W_SPLIT_M", dict(arm="W_SPLIT_M", seed=260, iters=MUON_CHECK_ITERS, thr=0.0, mlr=MUON_LR, total=ITERS)))
        if M_REF_FROM[0] != "batch 16":
            jobs.append(("W_M_REF", dict(arm="W_M_REF", seed=260, iters=MUON_CHECK_ITERS, mlr=MUON_LR)))
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        res = dict(zip([j[0] for j in jobs], ex.map(lambda j: child("run", j[1]), jobs)))
    r = res["W_SPLIT"]
    eq = ref is not None and eq_upto(r, ref, SPLIT_CHECK_ITERS)
    ok = eq and r["splits"] == 0 and len(r["checks"]) >= 2
    rows = [(f"with the trigger's threshold at 0, W_SPLIT|260 equals batch 16's LOCAL3_SLOW_D8_A|260 ({ref['cpu'] if ref else '?'}) "
             f"bit for bit through {SPLIT_CHECK_ITERS} (curve {r['curve']}; statistics and the decoder at "
             f"{sorted(int(k) for k in (ref or {}).get('decode', {}) if k.isdigit() and int(k) <= SPLIT_CHECK_ITERS)} equal: {eq}; "
             f"checks logged at {[c['step'] for c in r['checks']]}, {r['splits']} splits)", ok)]
    if "W_SPLIT_M" in res:
        rm = res["W_SPLIT_M"]
        pm = res["W_M_REF"] if "W_M_REF" in res else b16_runs("LOCAL3_SLOW_D8_M", (260,)).get(260)
        okm = pm is not None and eq_upto(rm, pm, MUON_CHECK_ITERS) and rm["splits"] == 0 and len(rm["checks"]) >= 1
        rows.append((f"with the threshold at 0, W_SPLIT_M|260 equals its Muon pair ({M_REF_FROM[0]}) bit for bit through "
                     f"{MUON_CHECK_ITERS} (curve {rm['curve']}; {okm})", okm))
    return rows


def check():
    rows = [tuple(x) for x in child("checks", dict(mlr=MUON_LR, muon="W_SPLIT_M" in ARMS))]
    rows += split_check()
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def map_steps(a, runs):
    last = max([a["iters"]] + [r.get("stopped_at") or 0 for r in runs.values()])
    return [t for t in range(MAP_EVERY, a["iters"] + 1, MAP_EVERY) if t <= last]


def maps_table(arm, a, runs, plab, pair):
    steps = map_steps(a, runs)
    print(f"    streams per channel (routing_k's map) at every check and at the end; '.' = the run had stopped (its last map holds):")
    print(f"    {'seed':>4} | {'outcome':<22} {'trans':>5} {'stop':>5} {'acc':>5} | " + " ".join(f"{t:>9}" for t in steps)
          + f" | {'end':>15} | {plab}")
    for s in a["seeds"]:
        r = runs.get(s)
        if r is None:
            print(f"    {s:>4} | not run")
            continue
        cells = []
        for t in steps:
            stop = r.get("stopped_at")
            cells.append(f"{'.':>9}" if stop is not None and stop < t else f"{rp.s33.per_channel(rp.s33.map_at(r, t)):>9}")
        q = pair.get(s)
        print(f"    {s:>4} | {r['outcome']:<22} {str(r['transition']):>5} {str(r.get('stopped_at')):>5} {r['acc']:5.2f} | "
              + " ".join(cells) + f" | {rp.s33.per_channel(r['end'].get('ch_map')):>15} | "
              + (f"{q['outcome']} (trans {q.get('transition')})" if q else "--"))


def splits_block(runs):
    print("    the trigger's checks (step: probe acc; E eligible, F fired, B blocked by gap/cap), and each split's target (KEYMASS "
          "c* -> c0; labelled ok = c* holds >= 2 streams and c0 none; streams on c* before; map before -> after):")
    for s, r in sorted(runs.items()):
        cks = r.get("checks") or []
        marks = " ".join(f"{c['step']}:{c['acc']:.2f}" + ("F" if c["fired"] else f"B{c.get('blocked_by', '')[:1]}" if c["blocked"]
                                                         else "E" if c.get("eligible") else "") for c in cks)
        print(f"      {s}: {r.get('splits')} split(s); checks {marks}")
        for c in cks:
            if c["fired"]:
                mb = c["map_before"]
                print(f"         split at {c['step']}: c* {c['key_cs']} -> c0 {c['key_c0']}, labelled ok {c['key_ok']} (streams on c* "
                      f"{[i for i, ch in enumerate(mb) if ch == c['key_cs']]}, on c0 {[i for i, ch in enumerate(mb) if ch == c['key_c0']]}); "
                      f"{rp.s33.per_channel(mb)} -> {rp.s33.per_channel(c.get('map_after'))}; maps {mb} -> {c.get('map_after')}")


def first_split(r):
    f = [c["step"] for c in r.get("checks") or [] if c.get("fired")]
    return f[0] if f else None


def agreement(arm, runs, pair, plab):
    """Each run's equality with its pair up to where they diverge (through the run's length in a dry run)."""
    rows = {}
    if arm == "W_LONG":
        for s, r in runs.items():
            if s in pair:
                rows[s] = eq_upto(r, pair[s], min(ITERS, r["iters"]))
        what = f"equal to {plab} through {ITERS} (curve, statistics, common decoder steps)"
    elif "SPLIT" in arm:
        for s, r in runs.items():
            if s not in pair:
                continue
            t = first_split(r)
            rows[s] = (t, eq_upto(r, pair[s], t if t is not None else r["iters"])
                       and (t is not None or r["iters"] < pair[s]["iters"] or r["curve"] == pair[s]["curve"]))
        what = f"equal to {plab} through the first split (step), or throughout when none fired"
    else:
        return None
    print(f"    agreement before the arms diverge — {what}: {rows}")
    return rows


def reading(o, n_seeds):
    if not o["complete"]:
        return f"INCOMPLETE ({o['n']}/{n_seeds}), no reading"
    nb = o["bound_routed"]
    v = "improves" if nb >= IMPROVES_N else "does not" if nb <= NOT_N else "neither"
    return (f"{v} (BOUND ROUTED {nb}/{o['n']}; improves: >= {IMPROVES_N}/10; does not: <= {NOT_N}/10; vs {o['plab']}: "
            f"arm only {o['b']}, pair only {o['c']}, p = {o['p']:.3g})")


def report():
    me = sys.modules[__name__]
    import explore_common2 as c2
    c2.print_screen_header2(me)
    store = ec.load_store(NAME)
    got = {arm: c16.runs_of(me, arm, store=store) for arm in ARMS}
    d8a = b16_runs("LOCAL3_SLOW_D8_A")
    mlab, mruns = m_pair(store)
    print(f"  pairs: batch 16's LOCAL3_SLOW_D8_A (code {B16_SHA}; {len(d8a)} runs on {b16_cpus('LOCAL3_SLOW_D8_A')}), BOUND ROUTED "
          f"{sum(rp.br(r) for r in d8a.values())}/{len(d8a)}; Muon pair: {mlab}, BOUND ROUTED "
          f"{sum(rp.br(r) for r in mruns.values())}/{len(mruns)}")
    if CUT:
        print(f"  cut by the runtime rule: {sorted(CUT)} (no runs, no reading)")
    out = {}
    for arm, a in ARMS.items():
        rs = got[arm]
        if a.get("pair") == "D8_A":
            plab, pr = "LOCAL3_SLOW_D8_A", d8a
        elif a.get("pair") == "D8_M":
            plab, pr = mlab, mruns
        else:
            plab, pr = "batch 16's LOCAL3_SLOW_D8_M (2.80GHz)", b16_runs("LOCAL3_SLOW_D8_M")
        if arm == "W_SPLIT_D98" and "W_D98" in ARMS:
            pre_lab, pre = "W_D98", got["W_D98"]
        else:
            pre_lab, pre = plab, pr
        print(f"  ARM {arm} ({a['label']}), seeds {ec.fmt_seeds(a['seeds'])}; {a['sched']}")
        maps_table(arm, a, rs, plab, pr)
        d = rp.compare(rs, pr, a["seeds"], arm, plab.split(" ")[0])
        print(f"    failure classes at the end: {arm} {rp.classes(rs)}; {plab} {rp.classes({s: pr[s] for s in a['seeds'] if s in pr})}")
        print(f"    transitions: {arm} {rp.transitions(rs)}; {plab} {rp.transitions({s: pr[s] for s in a['seeds'] if s in pr})}")
        if "SPLIT" in arm:
            splits_block(rs)
        meds = rp.decode_block(rs, steps=DIAG, chance=1 / 8)
        agr = agreement(arm, rs, pre, pre_lab)
        nb, n = sum(rp.br(r) for r in rs.values()), len(rs)
        out[arm] = dict(d, bound_routed=nb, n=n, complete=n == len(a["seeds"]), classes=rp.classes(rs), meds=meds, plab=plab,
                        agreement=agr, bound_seeds=sorted(s for s, r in rs.items() if rp.br(r)))
        if a.get("ref"):
            print(f"  S48 {arm}: reference (no reading): BOUND ROUTED {nb}/{n}")
        else:
            out[arm]["reading"] = reading(out[arm], len(a["seeds"]))
            print(f"  READING S48 {arm}: {out[arm]['reading']}")
        print()
    return out
