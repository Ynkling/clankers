#!/usr/bin/env python
"""
explore_window_recipe.py — EXPLORATORY, not a result. Screen S43 (batch 16): the width-3 window gate (batch 1's LOCAL3)
with S5's slow memory, no hinge, at S=2, P=4, k=2, no convolution, 24000 steps.

BACKGROUND (docstring; the audit of 6 October 2026, Appendix A)
- Batch 1, S1 LOCAL3 (explore_local_gate.py): gate from a width-3 causal conv of the embeddings, no recurrence;
  DISCOVERED 15/20 vs arm A 7/20 on seeds 160-179 (12 vs 4, p = 0.077); failures 4 KEY + 1 routed-unbound (the
  audit's count; test_router_layout.fail_class on the records gives KEY 3, STREAM-PARTIAL 2); dropped after the header
  layout (2/10, key splits). On the same seeds SLOW_HINGE 18/20, SLOW_MEM 12/20. recipe_scope Q2: slow memory removes
  key splits.
- The read is at the query's KEY position (CTX 1 back); the write is at VAL positions (CTX 2 back). A width-3 window
  sees CTX at both. S38-S41 decoded KEY and CTX positions only.
- W_h ~ N(0, 0.1^2): rho ~ 0.6, sigma_max ~ 1.08 at init.
- Every eight-stream run so far: Adam 1e-3 or Muon 0.005, decay 0.95.

ARMS (explore_window_recipe_child; this branch's modules, as batches 1-5; lr = test_channel_binding.SUB_LR = 1e-3):
  LOCAL3       batch 1's S1 run, unchanged, on seeds 180-199; seeds 160-179 are batch 1's records
               (explore_out/local_gate_results.json), which this machine reproduces (CHECK: LOCAL3|160 through 2400).
  LOCAL3_SLOW  LOCAL3 + S5's SLOW schedule (gate group W_in, W_g, the window's weights gate_conv.conv_w at 1e-3
               throughout; every other trainable parameter at 1e-4 for updates 1-2400, 1e-3 after; W_h, unused by the
               window gate, frozen and in no group), no hinge, seeds 160-199.
PAIRING: same seeds = same initial parameters (arm A's, plus the window from its own generator) and batches as S13's
SLOW_HINGE (seeds 160-199, 34/40 DISCOVERED), S5/S7's SLOW_MEM (160-199, 24/40) and LOCAL3.

READINGS (DISCOVERED = bound and final VAL cos < 0.5; fixed before any run)
- LOCAL3_SLOW vs SLOW_HINGE over 160-199: "the window gate replaces the recurrent gate and the hinge" if LOCAL3_SLOW
  >= 36/40 and (SLOW_HINGE only) - (LOCAL3_SLOW only) <= 1; "it does not" if LOCAL3_SLOW <= 30/40; otherwise neither.
- vs SLOW_MEM (24/40): printed with McNemar (exact, two-sided). Also printed: LOCAL3 over 160-199 (batch 1's records
  + the new runs) and LOCAL3_SLOW vs LOCAL3.
DIAGNOSTICS: failure classes (prediction: no POSITION; KEY failures fall from 4/20 to <= 1/20 on 160-179);
transitions; the stream decodability of the gate state (S38's decoder) at CTX, KEY and VAL positions at 1200, 2400,
4800 and the end (new runs only: batch 1's records have no decoder).
CHECKS: the gate at t ignores token t-3 and sees t-2 (batch 1's check); with the window at [1, 0, 0] and W_h = 0 the
gate equals arm A's with W_h = 0; the SLOW groups cover every trainable parameter once (gate group W_in, W_g, window),
W_h in none, the lrs at updates 1, 2400, 2401; LOCAL3|160 with the decoder probe reproduces batch 1's record through
2400 (curve and statistics).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16

NAME = "window_recipe"
CHILD = "explore_window_recipe_child"
MAIN = False
IDEA = "the width-3 window gate (batch 1's LOCAL3) with S5's slow memory and no hinge"
SOURCE = ("the audit of 6 October 2026, section 3.1: LOCAL3 beat arm A 15/20 vs 7/20 in batch 1 and was dropped; it "
          "cannot position-split, and slow memory removes key splits (Q2)")
CHANGE = ("LOCAL3_SLOW: LOCAL3 + S5's two-group schedule (gate W_in, W_g, window 1e-3 throughout; the rest 1e-4 for "
          "1-2400, 1e-3 after); LOCAL3 unchanged on 20 new seeds")
PAIRING = ("seeds 160-199 with S13's SLOW_HINGE, S5/S7's SLOW_MEM and LOCAL3 (batch 1's records on 160-179); same initial "
           "parameters and batches")
LR = ec.SUB_LR
ITERS = ec.MAX_ITERS
OLD = tuple(range(160, 180))
NEW = tuple(range(180, 200))
ALL = OLD + NEW
REPLACE_N, REPLACE_D, NOT_N = 36, 1, 30
DIAG = ("1200", "2400", "4800", "end")
REPRO_SEED, REPRO_ITERS = 160, 2400
CUT = set()
ARMS = {
    "LOCAL3": dict(key="LOCAL3", opt="adam", lr=LR, muon_lr=None, iters=ITERS, seeds=NEW, prio=1,
                   label="batch 1's S1 LOCAL3, unchanged (160-179: batch 1's records)", sched=f"lr {LR:g} throughout"),
    "LOCAL3_SLOW": dict(key="LOCAL3_SLOW", opt="adam", lr=LR, muon_lr=None, iters=ITERS, seeds=ALL, prio=1,
                        label="LOCAL3 + S5's SLOW schedule, no hinge",
                        sched=f"gate W_in/W_g/window {LR:g} throughout; every other trainable parameter {LR / 10:g} for "
                              f"updates 1-2400, {LR:g} after; W_h frozen (unused)"),
}


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ARMS[arm]["iters"]))


def timing(arms):
    return child("timing", dict(which=list(arms), steps=300))


def n_meas(arm):
    return ARMS[arm]["iters"] // 1200                      # the decoder runs at every evaluation (for "end")


# ── Records ──────────────────────────────────────────────────────────────────
def old_runs(store, arm, seeds):
    st = ec.load_store(store)
    return {s: st["runs"][f"{arm}|{s}"] for s in seeds if f"{arm}|{s}" in st["runs"]}


def slow_mem():
    return {**old_runs("slow_mem", "SLOW_MEM", OLD), **old_runs("slow_mem_ext", "SLOW_MEM", NEW)}


# ── CHECKs ───────────────────────────────────────────────────────────────────
def check():
    ok = True
    for nm, v in child("checks", {}):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    rec = old_runs("local_gate", "LOCAL3", (REPRO_SEED,))[REPRO_SEED]
    r = child("repro", dict(seed=REPRO_SEED, iters=REPRO_ITERS))
    want = [c for c in rec["curve"] if c[0] <= REPRO_ITERS]
    st_new = [s for s in r["stats"] if s["step"] != "end" and s["step"] <= REPRO_ITERS]
    st_old = [s for s in rec["stats"] if s["step"] != "end" and s["step"] <= REPRO_ITERS]
    v = r["curve"] == want and st_new == st_old
    print(f"  CHECK {NAME}: LOCAL3|{REPRO_SEED} with the decoder probe reproduces batch 1's record through {REPRO_ITERS} "
          f"(curve {r['curve']} vs {want}; statistics at {[s['step'] for s in st_new]} equal {st_new == st_old}; decoder at "
          f"1200/2400 h_key {[round(r['decode'][t]['h_key'], 3) for t in ('1200', '2400')]}, h_val "
          f"{[round(r['decode'][t]['h_val'], 3) for t in ('1200', '2400')]}): {'ok' if v else 'FAIL'}", flush=True)
    return ok and v


# ── Report ───────────────────────────────────────────────────────────────────
def disc(r):
    return bool(r is not None and ec.discovered(r))


def fails(runs, seeds):
    out = {}
    for s in seeds:
        r = runs.get(s)
        if r is not None and not ec.bound(r):
            c = ec.fail_class(r)
            out[c] = out.get(c, 0) + 1
    return out


def dec(r, t, f):
    d = (r.get("decode") or {}).get(t) if r else None
    return None if d is None else d[f]


def report():
    me = sys.modules[__name__]
    import explore_common2 as c2
    c2.print_screen_header2(me)
    loc_new = c16.runs_of(me, "LOCAL3")
    loc = {**old_runs("local_gate", "LOCAL3", OLD), **loc_new}
    ls = c16.runs_of(me, "LOCAL3_SLOW")
    sh = old_runs("slow_hinge", "SLOW_HINGE", ALL)
    sm = slow_mem()
    print("  per seed (outcome: DISCOVERED, BOUND unrouted, or test_router_layout.fail_class; transition; decodability of "
          "the gate state h by S38's decoder at KEY / VAL positions at 1200, 2400, 4800, end; chance 0.5):")
    print(f"    {'seed':>4} | {'LOCAL3':<16} {'trans':>5} | {'LOCAL3_SLOW':<16} {'trans':>5} "
          f"{'h@KEY 1200/2400/4800/end':>26} {'h@VAL 1200/2400/4800/end':>26} | {'S13 SLOW_HINGE':<16} {'S5/S7 SLOW_MEM':<16}")
    for s in ALL:
        a, b = loc.get(s), ls.get(s)
        src = "" if s in NEW else "*"
        kk = "/".join(c16.fmt2(dec(b, t, "h_key")) for t in DIAG)
        vv = "/".join(c16.fmt2(dec(b, t, "h_val")) for t in DIAG)
        print(f"    {s:>4} | {ec.tag(a) + src:<16} {str(a['transition'] if a else '--'):>5} | {ec.tag(b):<16} "
              f"{str(b['transition'] if b else '--'):>5} {kk:>26} {vv:>26} | {ec.tag(sh.get(s)):<16} {ec.tag(sm.get(s)):<16}")
    print("    (* batch 1's record)")
    if loc_new:
        print("  LOCAL3 (new runs, 180-199) decodability of h at KEY / VAL, per seed, 1200/2400/4800/end:")
        for s, r in sorted(loc_new.items()):
            print(f"    {s}: KEY {'/'.join(c16.fmt2(dec(r, t, 'h_key')) for t in DIAG)}  VAL "
                  f"{'/'.join(c16.fmt2(dec(r, t, 'h_val')) for t in DIAG)}  CTX "
                  f"{'/'.join(c16.fmt2(dec(r, t, 'h_ctx')) for t in DIAG)}")
    D_ls = {s: disc(r) for s, r in ls.items()}
    rows = [("LOCAL3_SLOW", "S13 SLOW_HINGE", D_ls, {s: disc(r) for s, r in sh.items()}, ALL),
            ("LOCAL3_SLOW", "S5/S7 SLOW_MEM", D_ls, {s: disc(r) for s, r in sm.items()}, ALL),
            ("LOCAL3_SLOW", "LOCAL3", D_ls, {s: disc(r) for s, r in loc.items()}, ALL),
            ("LOCAL3_SLOW", "LOCAL3 (160-179)", D_ls, {s: disc(r) for s, r in loc.items()}, OLD)]
    print("  DISCOVERED, paired (same seeds, initial parameters and batches; exact McNemar, two-sided):")
    res = {}
    for a, b, A, B, ss in rows:
        d = c16.mcnemar(A, B, ss)
        res[b] = d
        print(f"    {a:<12} {d['new']:>2}/{d['n']}   {b:<16} {d['old']:>2}/{d['n']}   {a} only {d['b']}, {b} only {d['c']}, "
              f"p = {d['p']:.3g}")
    nl = sum(disc(loc.get(s)) for s in ALL if s in loc)
    print(f"    LOCAL3 over 160-199: {nl}/{len(loc)} (batch 1's 160-179 {sum(disc(loc.get(s)) for s in OLD if s in loc)}/20, "
          f"new 180-199 {sum(disc(r) for r in loc_new.values())}/{len(loc_new)})")
    print("  failure classes (unbound runs; test_router_layout.fail_class):")
    for nm, runs in (("LOCAL3_SLOW", ls), ("LOCAL3", loc), ("S13 SLOW_HINGE", sh), ("S5/S7 SLOW_MEM", sm)):
        print(f"    {nm:<15} 160-179 {fails(runs, OLD)}   180-199 {fails(runs, NEW)}")
    f_old = fails(ls, OLD)
    n_old = sum(1 for s in OLD if s in ls)
    print(f"  prediction (diagnostic, fixed before any run): no POSITION failure and KEY failures <= 1/20 on 160-179 "
          f"(batch 1's LOCAL3: 4 by the audit's count, {fails(loc, OLD).get('KEY', 0)} by fail_class): LOCAL3_SLOW POSITION "
          f"{f_old.get('POSITION', 0) + fails(ls, NEW).get('POSITION', 0)} (160-199), KEY {f_old.get('KEY', 0)}/{n_old} on 160-179"
          f" -> {'met' if n_old == 20 and not f_old.get('POSITION') and not fails(ls, NEW).get('POSITION') and f_old.get('KEY', 0) <= 1 else 'not met' if n_old == 20 else 'incomplete'}")
    for nm, runs in (("LOCAL3_SLOW", ls), ("LOCAL3", loc), ("S13 SLOW_HINGE", sh)):
        tr = [r["transition"] for r in runs.values() if r and r["transition"] is not None]
        print(f"  transitions {nm:<15} median {c16.med(tr)} (n {len(tr)}): {sorted(tr)}")
    print("  median decodability of the gate state over the new runs (chance 0.50):")
    print(f"    {'arm':<12} {'at':>5} {'n':>3} {'h@CTX':>6} {'h@KEY':>6} {'h@VAL':>6}")
    med = {}
    for nm, runs in (("LOCAL3", loc_new), ("LOCAL3_SLOW", ls)):
        for t in DIAG:
            vals = {f: [dec(r, t, f) for r in runs.values() if dec(r, t, f) is not None] for f in ("h_ctx", "h_key", "h_val")}
            n = len(vals["h_key"])
            med[(nm, t)] = {f: c16.med(v) for f, v in vals.items()}
            print(f"    {nm:<12} {t:>5} {n:>3} " + " ".join(f"{c16.fmt2(med[(nm, t)][f]):>6}" for f in ("h_ctx", "h_key", "h_val")))
    d = res["S13 SLOW_HINGE"]
    n_ls, n_want = len(ls), len(ARMS["LOCAL3_SLOW"]["seeds"])
    if n_ls < n_want:
        reading = f"INCOMPLETE, {n_ls}/{n_want} runs, no reading"
    elif d["new"] >= REPLACE_N and d["c"] - d["b"] <= REPLACE_D:
        reading = "the window gate replaces the recurrent gate and the hinge"
    elif d["new"] <= NOT_N:
        reading = "it does not"
    else:
        reading = "neither"
    rd = (f"{reading} (LOCAL3_SLOW {d['new']}/{d['n']} DISCOVERED; SLOW_HINGE {d['old']}/{d['n']}; SLOW_HINGE only {d['c']}, "
          f"LOCAL3_SLOW only {d['b']}; replaces: >= {REPLACE_N}/40 and SLOW_HINGE only - LOCAL3_SLOW only <= {REPLACE_D}; "
          f"does not: <= {NOT_N}/40)")
    print(f"  READING S43: {rd}")
    return dict(reading=rd, short=reading, n=d["new"], vs_sh=d, vs_sm=res["S5/S7 SLOW_MEM"], fails=fails(ls, ALL),
                med=med)
