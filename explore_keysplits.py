#!/usr/bin/env python
"""
explore_keysplits.py — EXPLORATORY, not a result. Screen S52 (batch 18): attack the window gate's two-stream key splits.

BACKGROUND: S43 (batch 16): LOCAL3 33/40, LOCAL3_SLOW 31/40, S13's SLOW_HINGE 34/40 DISCOVERED at S=2, P=4, k=2 (seeds
160-199). S49 (batch 17): the window gate's failures are KEY splits that form before update 600 (four runs: held from
updates 0-300) and STREAM-PARTIAL gates that separate the streams on 2-3 of the 4 keys; two of those sat exactly on the
0.5 KEY threshold (a gate that splits the streams on two keys and puts the other two on different channels has eta^2 by
key 0.5).

ARMS (explore_keysplits_child; this branch's modules, S43's run path: test_short_conv's TASK, S=2, P=4, grouped, no
convolution, 24000 updates; lr = test_channel_binding.SUB_LR 1e-3; LOCAL3 + SLOW; 1 torch thread per run):
  W2_K4        k = 4 (two spare channels). Outcome BOUND ROUTED (test_stream_recipe.outcome, copied from main: bound, and
               each stream on its own channel at the value positions with every stream's held-out accuracy >= 0.9).
  W2_KEYHINGE  k = 2, plus the hinge's KEY term only: 1.0 x relu(eta2_key - 0.2) on the read gate at key positions, weight
               1 on updates 1-2400 and 0 after (the key-position analogue of S34's WINDOW; S19 / S22's term). Outcome
               DISCOVERED (bound and final VAL cos < 0.5, as S43).
PAIRING: seeds 160-199 with S43's LOCAL3_SLOW (31/40 DISCOVERED; explore_out/window_recipe_results.json, code
46867e792241) and S13's SLOW_HINGE (34/40; explore_out/slow_hinge_results.json): same batches; W2_KEYHINGE has
LOCAL3_SLOW's initial parameters; W2_K4's W_g has 4 rows, so the parameters drawn after it differ (the window is the seed's).
W2_K4's BOUND ROUTED is compared with their DISCOVERED (printed with that caveat).

READINGS (fixed before any run; N = the arm's seeds, 40, or 20 if the runtime rule cuts W2_KEYHINGE to 160-179):
  per arm: "removes the key splits" if successes >= ceil(0.9 N) (36/40; 18/20) and KEY failures <= floor(N / 40) (1/40;
  0/20); "does not" if successes <= floor(31 N / 40) (31/40; 15/20); otherwise neither. Successes: W2_K4 BOUND ROUTED,
  W2_KEYHINGE DISCOVERED. KEY failures: unbound runs whose failure class is KEY (W2_K4: test_scale_axes.fail_class_k on the
  pooled eta^2; W2_KEYHINGE: test_router_layout.fail_class, S43's).
PRINTED: every unsuccessful run's class with its eta^2 values (by key, by stream, margin), and the update at which any key
split first held (the pooled eta^2 of the read gate at key positions by key >= 0.5 at a measured update and at every later
one; S49's rule; measured after updates 0, 25, ..., 1200, 1300, ..., 2400, then every 1200); McNemar (exact, two-sided)
against LOCAL3_SLOW and SLOW_HINGE; the key term's firings (W2_KEYHINGE).
CHECKS: the copied functions equal main's (syntax trees); at k = 2 the pooled eta^2 equals routing_stats' channel-0 eta^2
and the hinge's eta2_key equals S19's key_penalty; the term's gradient reaches only the gate and the embedding; the term
equals test_slow_start's eta2 code restricted to key identity (explore_b18_gates.eta2_pool with eta2_hinge's own groupings
equals eta2_hinge, in a main-line child); weight 1 at update 2400, 0 at 2401; with weight 0 the arm equals LOCAL3_SLOW bit
for bit (1200 updates); the dense measurements are inert; W2_K4's model has 4 channels and the seed's window; the paired
records are all present; VALIDITY: with k = 4 the perfect gate (streams 0, 1 -> channels 0, 1) binds and routes on seeds
160-161. Run as this layout's own recorded validity arm, test_short_conv's ceiling_conv (the perfect gate + conv "layer",
mult 8; 5/5 at 1200 in test_short_conv), padded to k = 4: found in the CHECK pass before the dry run, the plain perfect gate
(no convolution) does not bind on this grouped two-stream layout at k = 2 or k = 4 (seed 160 at 0.54, 161 at 0.33 after
9600 updates; k = 2 and k = 4 bit-identical), so the check as first written could not pass at any k; a CHECK prints that
the padding is inert (k = 4 equals k = 2 bit for bit) and the plain gate's outcome.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16

NAME = "two_stream_keysplits"
CHILD = "explore_keysplits_child"
MAIN = False
IDEA = "attack the window gate's two-stream key splits: two spare channels (k=4), or the hinge's key term on updates 1-2400"
SOURCE = "S43 (LOCAL3_SLOW 31/40) and S49 (its KEY splits form before update 600); S19 / S22's key term; the user's batch 18"
CHANGE = "W2_K4: LOCAL3 + SLOW with k=4; W2_KEYHINGE: LOCAL3 + SLOW, k=2, + 1.0 x relu(eta2_key - 0.2) on updates 1-2400"
PAIRING = "seeds 160-199 with S43's LOCAL3_SLOW and S13's SLOW_HINGE (same batches; KEYHINGE also the same initial parameters)"
LR = ec.SUB_LR
ITERS = ec.MAX_ITERS
SEEDS = tuple(range(160, 200))
S43, S43_SHA = "window_recipe", "46867e792241"
PERFECT_SEEDS, PERFECT_ITERS, PLAIN_ITERS = (160, 161), 9600, 2400
SHOW = (0, 100, 200, 300, 400, 600, 800, 1200, 2400, 4800, 12000, 24000)
_sl = (f"gate W_in/W_g/window {LR:g} throughout; every other trainable parameter {LR / 10:g} for updates 1-2400, {LR:g} "
       f"after; W_h frozen (unused)")
ARMS = {
    "W2_K4": dict(key="W2_K4", opt="adam", lr=LR, muon_lr=None, iters=ITERS, seeds=SEEDS, outcome="BOUND ROUTED",
                  label="LOCAL3 + SLOW with k=4 (two spare channels)", sched=_sl + "; k=4"),
    "W2_KEYHINGE": dict(key="W2_KEYHINGE", opt="adam", lr=LR, muon_lr=None, iters=ITERS, seeds=SEEDS, outcome="DISCOVERED",
                        label="LOCAL3 + SLOW, k=2, + the hinge's key term on updates 1-2400",
                        sched=_sl + "; + 1.0 x relu(eta2_key - 0.2) on the read gate at key positions, updates 1-2400 only"),
}
CUT = set()


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ARMS[arm]["iters"]))


def timing(arms):
    return child("timing", dict(which=list(arms), steps=300))


def n_meas(arm):
    return ARMS[arm]["iters"] // 1200


def n_extra(arm):
    return 0


def n_cells(arm):
    dense = set(range(25, 1201, 25)) | set(range(1300, 2401, 100))
    return len(dense | set(range(1200, ARMS[arm]["iters"] + 1, 1200))) + 1


def old_runs(store, arm, seeds, sha=None):
    st = ec.load_store(store)["runs"]
    out = {}
    for s in seeds:
        if sha is None:
            r = st.get(f"{arm}|{s}")
        else:
            r = next((v for k, v in st.items() if k.startswith(f"{arm}|{s}|{sha}|") and v.get("ok")), None)
        if r is not None:
            out[s] = r
    return out


def pairs():
    return (old_runs(S43, "LOCAL3_SLOW", SEEDS, S43_SHA), old_runs("slow_hinge", "SLOW_HINGE", SEEDS))


# ── CHECKs ───────────────────────────────────────────────────────────────────
def check():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as ex:
        f1 = ex.submit(child, "checks", {})
        f2 = ex.submit(child, "perfect_k4", dict(seeds=list(PERFECT_SEEDS), iters=PERFECT_ITERS, arm="ceiling_conv", k=4))
        f4 = ex.submit(lambda: (child("perfect_k4", dict(seeds=[PERFECT_SEEDS[0]], iters=PLAIN_ITERS, arm="ceiling", k=4)),
                                child("perfect_k4", dict(seeds=[PERFECT_SEEDS[0]], iters=PLAIN_ITERS, arm="ceiling", k=2))))
        f3 = ex.submit(c16.run_child, "explore_b18_gates", "check_vs_slow_start", {}, True)
        rows = [tuple(x) for x in f1.result()]
        pk, vs = f2.result(), f3.result()
        p4, p2 = f4.result()
    rows.append((f"validity: with k = 4 the perfect gate (streams 0, 1 -> channels 0, 1; this layout's validity arm, test_short_conv's "
                 f"ceiling_conv: the perfect gate + conv 'layer') binds and routes on seeds {list(PERFECT_SEEDS)}: "
                 + "; ".join(f"{s}: {v['outcome']}, transition {v['transition']}, map {v['ch_map']}, per-stream accuracy "
                             f"{[round(x, 3) for x in v['stream_acc'] or []]}, k {v['k']}" for s, v in pk.items()),
                 all(v["outcome"] == "BOUND ROUTED" and v["k"] == 4 for v in pk.values()) and len(pk) == len(PERFECT_SEEDS)))
    a4, a2 = p4[str(PERFECT_SEEDS[0])], p2[str(PERFECT_SEEDS[0])]
    rows.append((f"the plain perfect gate (no convolution) padded to k = 4 equals it at k = 2 bit for bit through {PLAIN_ITERS} (curves "
                 f"{a4['curve']} vs {a2['curve']}): the padding is inert; it does not bind on this layout at either k (outcome "
                 f"{a4['outcome']}), which is why the validity check above uses the layout's own validity arm",
                 a4["curve"] == a2["curve"] and a4["k"] == 4 and a2["k"] == 2))
    rows.append((f"the key term is test_slow_start's eta2 code restricted to key identity: explore_b18_gates.eta2_pool with "
                 f"eta2_hinge's groupings (index, half) equals test_slow_start.eta2_hinge (main, in a main-line child; EPS "
                 f"{vs['tss_eps']}) on random gates: {[(r['S'], r['k'], r['index'], r['half']) for r in vs['rows']]}; max |diff| "
                 f"{vs['worst']:.1e} <= 1e-6", vs["ok"]))
    ls, sh = pairs()
    rows.append((f"the paired records are present: S43's LOCAL3_SLOW {len(ls)}/40 (DISCOVERED {sum(1 for r in ls.values() if ec.discovered(r))}), "
                 f"S13's SLOW_HINGE {len(sh)}/40 (DISCOVERED {sum(1 for r in sh.values() if ec.discovered(r))})",
                 len(ls) == 40 and len(sh) == 40))
    ok = True
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def series(r, f="etak_key_by_key"):
    return sorted((int(t), d[f]) for t, d in (r.get("dense") or {}).items() if f in d)


def formed(r, thr=0.5):
    s = series(r)
    first = next((t for t, v in s if v >= thr), None)
    if not s or s[-1][1] < thr:
        return None, first
    t0 = s[-1][0]
    for t, v in reversed(s):
        if v >= thr:
            t0 = t
        else:
            break
    return t0, first


def success(arm, r):
    if r is None:
        return False
    return r.get("outcome") == "BOUND ROUTED" if ARMS[arm]["outcome"] == "BOUND ROUTED" else bool(ec.discovered(r))


def label(arm, r):
    if ARMS[arm]["outcome"] == "BOUND ROUTED":
        return r.get("outcome")
    return "DISCOVERED" if ec.discovered(r) else ("bound, not discovered" if r.get("transition") is not None else r.get("fail"))


def thresholds(n):
    return math.ceil(0.9 * n), n // 40, (31 * n) // 40


def report():
    me = sys.modules[__name__]
    import explore_common2 as c2
    import explore_b16_report as rp
    c2.print_screen_header2(me)
    store = ec.load_store(NAME)
    ls, sh = pairs()
    dl = {s: bool(ec.discovered(r)) for s, r in ls.items()}
    dh = {s: bool(ec.discovered(r)) for s, r in sh.items()}
    print(f"  pairs: S43's LOCAL3_SLOW {sum(dl.values())}/{len(dl)} DISCOVERED; S13's SLOW_HINGE {sum(dh.values())}/{len(dh)} DISCOVERED")
    if CUT:
        print(f"  cut by the runtime rule: {sorted(CUT)}")
    out = {}
    for arm, a in ARMS.items():
        rs = c16.runs_of(me, arm, store=store)
        seeds = a["seeds"]
        print(f"  ARM {arm} ({a['label']}), seeds {ec.fmt_seeds(seeds)}; {a['sched']}; outcome {a['outcome']}")
        print(f"    {'seed':>4} | {arm + ' outcome':<28} {'trans':>5} {'acc':>5} {'disc':>5} | eta2 key/stream/margin at the end  "
              f"key split held from (first crossing) | LOCAL3_SLOW  SLOW_HINGE")
        keyf, cls = 0, {}
        for s in seeds:
            r = rs.get(s)
            if r is None:
                print(f"    {s:>4} | not run")
                continue
            e = r["end"]
            kk = e.get("etak_key_by_key", e.get("eta_key_by_key"))
            ks = e.get("etak_key_by_stream", e.get("eta_key_by_stream"))
            t0, first = formed(r)
            lab = label(arm, r)
            if not success(arm, r):
                cls[lab] = cls.get(lab, 0) + 1
                if r.get("fail") == "KEY" and r.get("transition") is None:
                    keyf += 1
            print(f"    {s:>4} | {str(lab):<28} {str(r['transition']):>5} {r['acc']:5.2f} {str(bool(ec.discovered(r))):>5} | "
                  f"{kk:.3f} / {ks:.3f} / {e.get('margin', float('nan')):.3f}{'  [on the 0.5 threshold]' if abs(kk - 0.5) < 0.02 else '':<24} "
                  f"{str(t0):>6} ({str(first):>5}) | {'DISC' if dl.get(s) else '-':<11}  {'DISC' if dh.get(s) else '-'}")
        n_ok = sum(1 for s in seeds if success(arm, rs.get(s)))
        new = {s: success(arm, r) for s, r in rs.items()}
        d1 = c16.mcnemar(new, dl, seeds)
        d2 = c16.mcnemar(new, dh, seeds)
        print(f"    {a['outcome']} {n_ok}/{len(rs)}; KEY failures {keyf}; unsuccessful classes {cls}")
        print(f"    vs S43 LOCAL3_SLOW (DISCOVERED): arm only {d1['b']}, LOCAL3_SLOW only {d1['c']}, McNemar two-sided p = {d1['p']:.3g}"
              + ("  [the arm's BOUND ROUTED against their DISCOVERED]" if a["outcome"] == "BOUND ROUTED" else ""))
        print(f"    vs S13 SLOW_HINGE (DISCOVERED): arm only {d2['b']}, SLOW_HINGE only {d2['c']}, McNemar two-sided p = {d2['p']:.3g}")
        if a["outcome"] == "BOUND ROUTED":
            print(f"    also DISCOVERED (bound and VAL cos < 0.5): {sum(1 for r in rs.values() if ec.discovered(r))}/{len(rs)}; "
                  f"maps at the end: {rp.classes({s: dict(outcome=str(r['end'].get('ch_map'))) for s, r in rs.items()})}")
        else:
            fired = [r["hinge"]["fired"] for r in rs.values() if r.get("hinge")]
            print(f"    the key term fired on {sorted(fired)} of the first 2400 updates per run (median "
                  f"{c16.med(fired)}); eta2_key at updates 1 / 600 / 2400 (median): "
                  + " / ".join(c16.fmt2(c16.med([(r.get('hinge') or {}).get('eta_at', {}).get(t) for r in rs.values()]))
                               for t in ("1", "600", "2400")))
        print(f"    the pooled eta^2 by key at updates {list(SHOW)} (median over the arm's runs): "
              + " ".join(c16.fmt2(c16.med([dict(series(r)).get(t) for r in rs.values()])) for t in SHOW))
        hold = {s: formed(r)[0] for s, r in rs.items() if formed(r)[0] is not None}
        print(f"    runs with a key split held to the end: {len(hold)} ({hold})")
        N = len(seeds)
        hi, kmax, lo = thresholds(N)
        if len(rs) < N:
            rd = f"INCOMPLETE ({len(rs)}/{N}), no reading"
        else:
            v = "removes the key splits" if (n_ok >= hi and keyf <= kmax) else "does not" if n_ok <= lo else "neither"
            rd = (f"{v} ({a['outcome']} {n_ok}/{N}, KEY failures {keyf}; removes: >= {hi}/{N} and KEY <= {kmax}; does not: <= {lo}/{N}; "
                  f"vs LOCAL3_SLOW {d1['b']}/{d1['c']}, p = {d1['p']:.3g}; vs SLOW_HINGE {d2['b']}/{d2['c']}, p = {d2['p']:.3g})")
        print(f"  READING S52 {arm}: {rd}\n")
        out[arm] = dict(n_ok=n_ok, n=len(rs), key_failures=keyf, classes=cls, reading=rd, vs_ls=d1, vs_sh=d2, held=hold)
    return out
