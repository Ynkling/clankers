#!/usr/bin/env python
"""
explore_window_fail.py — EXPLORATORY, not a result. Screen S49 (batch 17), descriptive: S43's follow-up. What LOCAL3_SLOW's
nine unbound two-stream runs did (S43, batch 16: LOCAL3_SLOW 31/40 DISCOVERED on seeds 160-199; the nine others,
test_router_layout.fail_class at the end: STREAM-PARTIAL 163, 183, 185, 188, 197; KEY 169, 173, 186, 192).

THE RUNS (explore_window_fail_child): S43's LOCAL3_SLOW run code on those nine seeds, unchanged, with the gate measured
after updates 0, 25, 50, ..., 1200, then every 100 through 2400, then every 1200 to 24000, on the run's own probe (the
batch fail_class's statistics use). Each rerun must reproduce S43's record bit for bit (curve and statistics; printed per
run). The seeds are S43's unbound LOCAL3_SLOW seeds read from explore_out/window_recipe_results.json (CHECK).

QUESTIONS AND RULES (fixed before any run; k = 2, p = the gate's channel-0 probability; LOCAL3's read gate = write gate):
  (1) What the STREAM-PARTIAL gates do at VAL positions (the writes). At the end, at VAL positions 3j+2: the mean p in each
      (stream, key) cell; per key k: "split" if |p(stream 0, k) - p(stream 1, k)| >= 0.5 (the two streams' writes go to
      different channels), else "shared" on channel 0 (mean p >= 0.5) or 1; the share of soft gates (0.1 < p < 0.9);
      eta^2 of p by stream, key, cell and value token. The same table at KEY positions (the reads) beside it.
  (2) Whether KEY splits form before or after update 600. A KEY split is eta^2(p at KEY positions by key) >= 0.5
      (test_router_layout.ETA_SPLIT, fail_class's KEY threshold). It "formed at" the first measured update from which it
      holds at every later measurement to the end of the run; "by update 600" if that update is <= 600, "after update 600"
      otherwise; "none at the end" if it does not hold at the end (the first crossing, if any, is printed). Asked of all
      nine runs.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16

NAME = "window_fail"
CHILD = "explore_window_fail_child"
MAIN = False
IDEA = "classify LOCAL3_SLOW's nine unbound two-stream runs (S43): the gate at the write positions, and when key splits form"
SOURCE = "batch 16's S43 (LOCAL3_SLOW 31/40; failures STREAM-PARTIAL 5, KEY 4); the user's batch 17"
CHANGE = "none to the runs (S43's run code on the nine seeds); the gate measured densely early and at every evaluation"
PAIRING = "each rerun against its own S43 record (must reproduce it bit for bit)"
LR = ec.SUB_LR
ITERS = ec.MAX_ITERS
NINE = (163, 169, 173, 183, 185, 186, 188, 192, 197)
S43, S43_SHA = "window_recipe", "46867e792241"
EARLY = 600
ETA_SPLIT = 0.5
SPLIT_D = 0.5
SHOW = (0, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1600, 2400, 4800, 12000, 24000)
ARMS = {
    "LOCAL3_SLOW_RERUN": dict(key="LOCAL3_SLOW_RERUN", opt="adam", lr=LR, muon_lr=None, iters=ITERS, seeds=NINE,
                              label="S43's LOCAL3_SLOW on its nine unbound seeds, rerun with dense gate measurements",
                              sched=f"gate W_in/W_g/window {LR:g} throughout; every other trainable parameter {LR / 10:g} for "
                                    f"updates 1-2400, {LR:g} after; W_h frozen (unused)"),
}
CUT = set()


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(seed=seed, iters=ARMS[arm]["iters"]))


def timing(arms):
    return child("timing", dict(steps=300))


def n_meas(arm):
    return ITERS // 1200                                   # S43's decoder at every evaluation


def n_extra(arm):
    return 0


def n_cells(arm):
    return 48 + 12 + ITERS // 1200


def s43_runs():
    st = ec.load_store(S43)["runs"]
    out = {}
    for k, r in st.items():
        a, s, h, cpu = k.split("|", 3)
        if a == "LOCAL3_SLOW" and h == S43_SHA and r.get("ok"):
            out[int(s)] = r
    return out


def check():
    rows = [tuple(x) for x in child("checks", {})]
    rec = s43_runs()
    unbound = sorted(s for s, r in rec.items() if not r.get("discovered"))
    rows.append((f"the nine seeds are S43's unbound LOCAL3_SLOW seeds ({len(rec)} records of code {S43_SHA}; not DISCOVERED: "
                 f"{unbound}; classes {[rec[s].get('fail') for s in unbound]})", tuple(unbound) == NINE and len(rec) == 40))
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def series(r, f):
    return sorted((int(t), d[f]) for t, d in (r.get("dense") or {}).items() if f in d)


def formed(r):
    s = series(r, "eta_key_by_key")
    first = next((t for t, v in s if v >= ETA_SPLIT), None)
    if not s or s[-1][1] < ETA_SPLIT:
        return None, first
    t0 = s[-1][0]
    for t, v in reversed(s):
        if v >= ETA_SPLIT:
            t0 = t
        else:
            break
    return t0, first


def key_label(t0):
    return "none at the end" if t0 is None else f"by update {EARLY}" if t0 <= EARLY else f"after update {EARLY}"


def val_pattern(cell):
    labs = []
    for k in range(len(cell[0])):
        d = cell[0][k] - cell[1][k]
        if abs(d) >= SPLIT_D:
            labs.append("split")
        else:
            labs.append("ch0" if (cell[0][k] + cell[1][k]) / 2 >= 0.5 else "ch1")
    n_split = labs.count("split")
    shared = [lab for lab in labs if lab != "split"]
    kind = ("stream-routed on every key" if n_split == len(labs) else
            ("one channel" if len(set(shared)) == 1 else "key-routed (both channels)") if n_split == 0 else
            f"streams split on {n_split} of {len(labs)} keys, shared on the rest")
    return labs, kind


def at(r, t, f):
    d = (r.get("dense") or {}).get(str(t))
    return None if d is None else d.get(f)


def same_as(r, ref):
    """Curve and statistics equal S43's record through the rerun's length (and the end statistics when as long)."""
    t = r["iters"]
    full = t == ref["iters"]
    return ([c for c in r["curve"] if c[0] <= t] == [c for c in ref["curve"] if c[0] <= t]
            and [x for x in r["stats"] if x["step"] != "end" and x["step"] <= t]
            == [x for x in ref["stats"] if x["step"] != "end" and x["step"] <= t]
            and (not full or (r["curve"] == ref["curve"] and r["end"] == ref["end"])))


def report():
    me = sys.modules[__name__]
    import explore_common2 as c2
    c2.print_screen_header2(me)
    rs = c16.runs_of(me, "LOCAL3_SLOW_RERUN")
    rec = s43_runs()
    print(f"  {len(rs)}/{len(NINE)} reruns; each compared with its S43 record (code {S43_SHA})")
    f2 = c16.fmt2
    out = {}
    print("  (2) KEY SPLITS: eta^2 of p at KEY positions by key (>= 0.5 = a key split), by stream, and the query routing margin, "
          "at updates " + ", ".join(str(t) for t in SHOW) + ":")
    for s in NINE:
        r = rs.get(s)
        if r is None:
            print(f"    {s}: not run")
            continue
        t0, first = formed(r)
        rep = same_as(r, rec[s]) if s in rec else None
        cls_now = r.get("fail")
        out[s] = dict(s43=rec.get(s, {}).get("fail"), now=cls_now, reproduces=rep, formed=t0, first=first, key=key_label(t0))
        print(f"    {s}: S43 class {rec.get(s, {}).get('fail')}, rerun class {cls_now}, reproduces S43's record bit for bit "
              f"(curve and statistics): {rep}; acc {r['acc']:.3f}; KEY split {key_label(t0)}"
              + (f" (formed at {t0}" + (f"; first crossing {first}" if first != t0 else "") + ")" if t0 is not None
                 else (f" (first crossing {first}, not held)" if first is not None else "")))
        e_end = series(r, "eta_key_by_key")[-1][1] if series(r, "eta_key_by_key") else None
        if e_end is not None and abs(e_end - ETA_SPLIT) < 0.02:
            print(f"       note: eta^2 by key at the end {e_end:.4f} is on the threshold {ETA_SPLIT} (a gate that splits the streams "
                  f"on two keys and puts the other two keys on different channels has eta^2 by key 0.5 exactly), so this "
                  f"run's KEY-split label follows sampling noise")
            out[s]["on_threshold"] = True
        for f, lab in (("eta_key_by_key", "by key   "), ("eta_key_by_stream", "by stream"), ("margin", "margin   ")):
            print(f"       {lab} " + " ".join(f2(at(r, t, f)) for t in SHOW))
    print("  (1) THE GATE AT VAL POSITIONS (the writes) AT THE END, beside KEY positions (the reads): mean p = P(channel 0) per "
          "(stream, key) cell; per key: split (|p(s0) - p(s1)| >= 0.5) or shared on ch0 / ch1; soft = share of 0.1 < p < 0.9; "
          "eta^2 of p by stream / key / cell / value token:")
    for s in NINE:
        r = rs.get(s)
        if r is None:
            continue
        e = (r.get("dense") or {}).get(str(r["curve"][-1][0])) or {}
        if not e:
            continue
        for role in ("val", "key"):
            cell = e[f"cell_{role}"]
            labs, kind = val_pattern(cell)
            extra = f" by value {e['eta_val_by_value']:.2f}" if role == "val" else ""
            print(f"    {s} {role.upper():<3} stream 0 [{' '.join(f'{v:.2f}' for v in cell[0])}]  stream 1 "
                  f"[{' '.join(f'{v:.2f}' for v in cell[1])}]  keys {labs}: {kind}; soft {e[f'soft_{role}']:.2f}; eta^2 by stream "
                  f"{e[f'eta_{role}_by_stream']:.2f} key {e[f'eta_{role}_by_key']:.2f} cell {e[f'eta_{role}_by_cell']:.2f}{extra}")
            out[s][f"{role}_pattern"] = kind
            out[s][f"{role}_labels"] = labs
        print(f"         margin {e['margin']:.3f}")
    print("  (1') THE VAL PATTERN OVER TIME (per key: split / ch0 / ch1) at updates 600, 1200, 2400, 4800, 12000, end:")
    for s in NINE:
        r = rs.get(s)
        if r is None:
            continue
        cols = []
        for t in (600, 1200, 2400, 4800, 12000, r["curve"][-1][0]):
            e = (r.get("dense") or {}).get(str(t))
            cols.append("--" if not e else "/".join(val_pattern(e["cell_val"])[0]))
        print(f"    {s} ({out[s]['s43']}): " + "  ".join(cols))
    keyruns = [s for s in out if out[s]["s43"] == "KEY"]
    partial = [s for s in out if out[s]["s43"] == "STREAM-PARTIAL"]
    summ = dict(
        reproduce=f"{sum(1 for o in out.values() if o['reproduces'])}/{len(out)} reruns reproduce S43's records",
        key=(f"KEY runs ({len(keyruns)}): " + ", ".join(f"{s} {out[s]['key']}" + (f" ({out[s]['formed']})" if out[s]['formed']
                                                                                 is not None else "")
                                                      + (" [on the threshold]" if out[s].get("on_threshold") else "") for s in keyruns)),
        key_partial=(f"STREAM-PARTIAL runs ({len(partial)}): " + ", ".join(
            f"{s} {out[s]['key']}" + (f" ({out[s]['formed']})" if out[s]['formed'] is not None else "")
            + (" [on the threshold]" if out[s].get("on_threshold") else "") for s in partial)),
        val=(f"STREAM-PARTIAL at VAL: " + "; ".join(f"{s} {out[s].get('val_pattern')} {out[s].get('val_labels')}" for s in partial)),
        val_key=(f"KEY at VAL: " + "; ".join(f"{s} {out[s].get('val_pattern')}" for s in keyruns)))
    for v in summ.values():
        print(f"  S49 {v}")
    return dict(runs=out, summary=summ)
