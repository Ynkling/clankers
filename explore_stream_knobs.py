#!/usr/bin/env python
"""
explore_stream_knobs.py — EXPLORATORY, not a result. Screen S46 (batch 16), descriptive: the eight-stream learning rate
and memory decay, never varied so far (every eight-stream run: Adam 1e-3 or Muon 0.005, decay 0.95).

BACKGROUND: the audit of 6 October 2026, section 3.4 — at two streams conv_lr showed 4e-3 changes the outcome
qualitatively; recall failures in recurrent models are often an optimization problem with a narrow rate window
(Okpekpe & Orvieto, arXiv 2508.19029); decay 0.95 has half-life 14 tokens on 99-token sequences (the earliest
same-stream triple weighted 1/118 of the latest), decay 0.98 half-life 34.

ARMS (explore_stream_knobs_child; S=8, P=4, k=16, conv, all 8 streams from step 1, 28800 steps, seeds 260-264, Adam;
X's HINGE_D8 recipe with one knob changed): LR3E4 (every lr x 0.3: 3e-4), LR3E3 (x 3: 3e-3), WH_SLOW (W_h at 1e-4 =
the gate's lr / 10), DECAY98 (decay 0.98), PERFECT98 (the perfect gate at decay 0.98, its own plain recipe: a validity
check). Reference rows: X's recorded HINGE_D8 260-264 (lr 1e-3, decay 0.95; 0/5 bound).
PRINTED (no reading): bound, outcome, final accuracy, decodability of the gate state at KEY and VAL (and CTX) positions
at 1200 and 2400, rho(W_h) and sigma_max(W_h) every 400 updates.
CHECKS: decay 0.98 enters the decay mask and nothing else (parameters equal; only GatedAttention.decay differs;
layer-1 scores scale by (0.98/0.95)^(t-s); setting it back restores the logits bit for bit); the perfect gate's routing is
unchanged; the generic Adam recipe at the default lrs reproduces X's HINGE_D8|260 through 1200; each arm's groups cover
every parameter once with the lrs at updates 1, 2400, 2401.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common16 as c16
import explore_b16_report as rp

NAME = "stream_knobs"
CHILD = "explore_stream_knobs_child"
MAIN = True
IDEA = "the eight-stream knobs never varied: the learning rate (x0.3, x3), W_h's rate (/10), the memory decay (0.98)"
SOURCE = "the audit of 6 October 2026, section 3.4; Okpekpe & Orvieto (arXiv 2508.19029)"
CHANGE = "X's HINGE_D8 with one knob changed per arm; the perfect gate at decay 0.98 (validity)"
PAIRING = "seeds 260-264, printed beside X's recorded HINGE_D8 (same initial parameters and batches); descriptive"
ITERS = 28800
S5 = tuple(range(260, 265))
_h = "; + main's HINGE hinge"
ARMS = {
    "LR3E4": dict(key="LR3E4", opt="adam", lr=3e-4, muon_lr=None, iters=ITERS, seeds=S5, label="HINGE_D8 at lr 3e-4",
                  sched="gate W_in/W_h/W_g 0.0003 throughout; every other parameter 3e-05 for updates 1-2400, 0.0003 after" + _h),
    "LR3E3": dict(key="LR3E3", opt="adam", lr=3e-3, muon_lr=None, iters=ITERS, seeds=S5, label="HINGE_D8 at lr 3e-3",
                  sched="gate W_in/W_h/W_g 0.003 throughout; every other parameter 0.0003 for updates 1-2400, 0.003 after" + _h),
    "WH_SLOW": dict(key="WH_SLOW", opt="adam", lr=1e-3, muon_lr=None, iters=ITERS, seeds=S5, label="HINGE_D8 with W_h at lr/10",
                    sched="W_in/W_g 0.001 throughout; W_h 0.0001 throughout; every other parameter 0.0001 for updates 1-2400, "
                          "0.001 after" + _h),
    "DECAY98": dict(key="DECAY98", opt="adam", lr=1e-3, muon_lr=None, iters=ITERS, seeds=S5, label="HINGE_D8 at decay 0.98",
                    sched="gate W_in/W_h/W_g 0.001 throughout; every other parameter 0.0001 for updates 1-2400, 0.001 after; "
                          "decay 0.98" + _h),
    "PERFECT98": dict(key="PERFECT98", opt="adam", lr=1e-3, muon_lr=None, iters=ITERS, seeds=S5,
                      label="the perfect gate at decay 0.98 (validity)",
                      sched="every parameter 0.001 throughout (the perfect gate's plain recipe); decay 0.98; no hinge"),
}
CUT = set()


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ARMS[arm]["iters"]))


def timing(arms):
    return child("timing", dict(which=list(arms), steps=300))


def n_meas(arm):
    return 0 if arm == "PERFECT98" else 2


def n_extra(arm):
    return 0


def check():
    ok = True
    for nm, v in child("checks", {}):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def report():
    me = sys.modules[__name__]
    import explore_common2 as c2
    import explore_window_scale as s44
    c2.print_screen_header2(me)
    if CUT:
        print(f"  cut by the runtime rule: {sorted(CUT)} (no runs)")
    xr = s44.x_runs("HINGE_D8", S5)
    print("  X HINGE_D8 (recorded; lr 1e-3, decay 0.95): " + "  ".join(
        f"{s}: {r['outcome']}, acc {r['acc']:.3f}" for s, r in sorted(xr.items())))
    out = {}
    for arm, a in ARMS.items():
        rs = c16.runs_of(me, arm)
        print(f"  ARM {arm} ({a['label']}; {a['sched']}), seeds {', '.join(map(str, a['seeds']))}")
        print(f"    {'seed':>4} | {'outcome':<24} {'bound':>5} {'trans':>5} {'acc':>6} | h@CTX/KEY/VAL 1200 | h@CTX/KEY/VAL 2400 | "
              f"X HINGE_D8")
        for s in a["seeds"]:
            r = rs.get(s)
            if r is None:
                print(f"    {s:>4} | not run")
                continue
            dd = " | ".join(" ".join(rp.f2(rp.dec(r, t, f), 4) for f in ("h_ctx", "h_key", "h_val")) + "    "
                            for t in ("1200", "2400"))
            print(f"    {s:>4} | {r['outcome']:<24} {('yes' if r['transition'] is not None else 'no'):>5} {str(r['transition']):>5} "
                  f"{r['acc']:6.3f} | {dd} | {(xr.get(s) or {}).get('outcome', '--')}")
        if arm != "PERFECT98":
            print("    rho(W_h) every 400 updates from 0 (sigma_max(W_h) in brackets every 2400):")
            for s, r in sorted(rs.items()):
                rh = r.get("rho") or {}
                ks = sorted(rh, key=int)
                print(f"      {s}: " + " ".join(f"{rh[k]['rho']:.2f}" + (f"[{rh[k]['sigma']:.2f}]" if int(k) % 2400 == 0 else "")
                                            for k in ks))
            meds = {t: {f: c16.med([rp.dec(r, t, f) for r in rs.values()]) for f in ("h_ctx", "h_key", "h_val")}
                    for t in ("1200", "2400")}
            print("    medians: " + "; ".join(f"{t}: CTX {rp.f2(m['h_ctx'], 4)} KEY {rp.f2(m['h_key'], 4)} VAL {rp.f2(m['h_val'], 4)}"
                                        for t, m in meds.items()))
        else:
            meds = None
        nb = sum(1 for r in rs.values() if r["transition"] is not None)
        nbr = sum(rp.br(r) for r in rs.values())
        print(f"  S46 {arm}: bound {nb}/{len(rs)}, BOUND ROUTED {nbr}/{len(rs)}, median final accuracy "
              f"{rp.f2(c16.med([r['acc'] for r in rs.values()]), 4)} (descriptive)")
        out[arm] = dict(bound=nb, bound_routed=nbr, n=len(rs), complete=len(rs) == len(a["seeds"]),
                        acc=c16.med([r["acc"] for r in rs.values()]), meds=meds, classes=rp.classes(rs))
    return out
