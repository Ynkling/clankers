#!/usr/bin/env python
"""
explore_f_probe.py — EXPLORATORY, not a result. Session F, screen S77: do bound delta single channels tag their keys by
stream? S53's post-hoc diagnostic (one seed, 307) found near-orthogonal layer-3 keys for the two writes of a key in a bound
single delta channel. This puts it on ten seeds per arm with Session G's S55 probe.

ARMS (reruns of recorded runs, bit for bit on this CPU, asserted; explore_f_probe_child):
  S53_DELTA  S53's SINGLE_DELTA_H: one channel, delta β = 1, L2 keys, header layout P = 4, 24000; seeds 300-309 (bound 8/10
             in S53: all but 302, 303).
  S59_GDN    S59's SINGLE_GDN: one channel, delta β = 1, L2 keys, GDN decay, header P = 8, 48000; seeds 350-359 (bound 8/10).
  S69_GDN    S69's S69_GDN: one HEBBIAN channel with the GDN decay (β = 0, write 1, raw keys), header P = 8, 48000; seeds
             350-359, the same batches as S59_GDN (bound 0/10). Of S69's three Hebbian arms this is the one that pairs
             with S59_GDN (same decay).
PROBE (Session G's S55, copied unchanged from claude/explore-G at 3932db7 into explore_f_probe_common): logistic regression
(sklearn, lbfgs, standardised features), fit on 80% of 2000 probe sequences (seed 55000, split by sequence), held-out
accuracy on the other 20%, from the raw embedding, the residual stream entering layers 1, 2 and 3 (layer 1's = LN of the
embedding) and the final layer, at the body's KEY and VAL positions, to the true stream (the block's CTX); chance 0.5.
Fitted on the trained model at the end of each run and on the same seed's model at initialisation.

READINGS (fixed before any run). "Bound" = BOUND in the rerun (= the record's). Let L3 = the probe on the residual entering
layer 3 and L1 = the residual entering layer 1.
  "the delta stack tags its keys by stream" if, among the bound delta runs (S53_DELTA and S59_GDN together), L3 >= 0.9 at
     KEY and at VAL on >= 8 runs while L1 <= 0.6 at KEY and at VAL on those same runs;
  "the tag is where binding is" if the median over all bound runs of L3 (mean of KEY and VAL) exceeds the median over all
     unbound runs (every arm) by >= 0.2;
  both can apply; otherwise descriptive, with the probe by layer and position for every arm (medians, Wilson on the pooled
  held-out counts), at the end and at initialisation.
CHECKS (child.checks + explore_f_delta_checks): G's synthetic probe check (informative feature ~0.998, random ~0.5); G's
residual loop reproduces each model's logits bit for bit on these memories; the probe's labels; each arm's record through
2400 reproduced bit for bit; every full rerun asserts its curve equals its record.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore_common16 as c16
import explore_f_common as fc

NAME = fc.name("probe")
CHILD = "explore_f_probe_child"
MAIN = False
ARMS = {
    "S53_DELTA": dict(opt="adam", lr=1e-3, iters=24000, seeds=tuple(range(300, 310)), label="S53 SINGLE_DELTA_H rerun + probe",
                      sched="Adam 1e-3 throughout"),
    "S59_GDN": dict(opt="adam", lr=1e-3, iters=48000, seeds=tuple(range(350, 360)), label="S59 SINGLE_GDN rerun + probe",
                    sched="Adam 1e-3 throughout"),
    "S69_GDN": dict(opt="adam", lr=1e-3, iters=48000, seeds=tuple(range(350, 360)), label="S69 S69_GDN (Hebbian) rerun + probe",
                    sched="Adam 1e-3 throughout"),
}


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed))


def timing(arm, steps):
    return child("timing", dict(arm=arm, steps=steps))


def check():
    ok = True
    for nm, v in child("checks", {}):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def line(arm, r):
    if not r.get("ok"):
        return f"FAILED {r.get('error')}"
    p = r["probe"]
    return (f"acc {r['acc']:.3f} trans {r['transition']} repro {r['repro']}  probe L1 K/V {p['L1|K']['acc']:.2f}/{p['L1|V']['acc']:.2f}"
            f"  L3 K/V {p['L3|K']['acc']:.2f}/{p['L3|V']['acc']:.2f}")


def report():
    import explore_f_reports3 as fr
    return fr.report_s77()


if __name__ == "__main__":
    import explore_f_probe as me
    fc.drive("s77", [me])
