#!/usr/bin/env python
"""
explore_oracle_d128.py — EXPLORATORY, not a result. Screen S69 (batch 21): is the D = 128 oracle a usable validity check
for the key-load spec draft (explore_out/spec_draft_keyload.md), on its recorded recipe and on SLOW?

QUESTION (the user's batch 21: "the D128 oracle's validity on ten seeds (single Adam and with SLOW) before your spec goes
to V"). The draft makes a width VALID on a machine if its oracle binds on both of the machine's first two seeds within
the budget (VALID_N = 2). At D = 128 the oracle (the perfect gate, a single Adam at 1e-3) bound one of S67's two seeds
(271 at 12,000); 270 did not bind by 43,200 and, in S68, not by 86,400 (0.38).

CONFIGURATION (the draft's, as S67 ran it; explore_scale_child): test_stream_recipe.run_attempt with ARM["ceiling4k16"]
(the perfect gate: stream s -> channel s, channels 4-15 unused; k = 16, conv 'layer' width 4, ctx_pad 16, evaluation every
1200 on held-out sequences, run_one's early stop) on test_channel_binding.task_for(16, 4) (S = 4, P = 16, n_vals 16,
n_q 1, grouped layout, length 195), at mult 2, D 128 (N = 256), 3 layers, for 40,800 updates (the draft's budget).
ARMS (paired by seed: the same initial weights and the same batches):
  ORC_A_D128  the recorded validity recipe: one Adam at 1e-3 on every parameter, no groups, no switch (S67's ORC_P16_D128)
  ORC_S_D128  SLOW: every trainable parameter at 1e-4 for updates 1-2400, 1e-3 after (the learned arms' "rest" group and
              switch; the perfect gate has no parameters, so nothing stays at 1e-3 from the start)
SEEDS: 270-279 on both arms. These are S67's seeds, so ORC_A_D128 on 270 and 271 must reproduce S67's records through 2400
(a CHECK). They are not the draft's seeds (X 660-679, L 1660-1679, kept unused) and not on the main line.
OUTCOME: BOUND (a transition held to the end: the run path's transition; explore_scale_child's oracle outcome), with the
transition. b_A, b_S = how many of the ten bind within 40,800.

READINGS (fixed before any S69 run).
  Per arm (b of 10): "reliable" b >= 9; "at risk" 5 <= b <= 8; "unreliable" b <= 4. Printed with it: the chance that two
  seeds drawn from these ten both bind, b(b-1)/90 (the draft's VALID_N = 2 on a machine), and the Wilson 95% interval of b/10.
  FOR THE SPEC (one name per outcome; rules in order):
    1. b_A >= 9                    "keep the draft's oracle" (the recorded recipe at D128; the draft unchanged)
    2. else b_S >= 9               "the D128 oracle on SLOW" (the draft's option (a); the D32 oracle keeps its recorded
                                   recipe, which bound S67's two seeds at 24,000 and 27,600; the asymmetry is written in)
    3. else max(b_A, b_S) >= 5     "VALID_N = 1 at D128" (the draft's option (b)), with the arm of the larger b (ties: the
                                   recorded recipe)
    4. else                        "no D128 validity check at 40,800" (the draft does not go to V as written)
  An arm with a seed missing or failed has no reading ("INCOMPLETE"), and then there is no spec reading.
DESCRIPTIVE (no reading): ORC_S_D128 against ORC_A_D128 paired (both bind / A only / S only / neither; exact McNemar,
two-sided); the transitions; each run's final held-out accuracy and per-stream accuracy at the end.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common16 as c16
import explore_common19 as c19

NAME = "oracle_d128"
CHILD = "explore_oracle_child"
MAIN = True
IDEA = "is the D = 128 oracle a usable validity check for the key-load draft? ten seeds, the recorded recipe and SLOW"
SOURCE = "the user's batch 21; S67 / S68 (the D128 oracle bound 271 at 12,000; 270 unbound at 86,400); the draft's VALID_N = 2"
CHANGE = "the oracle's recipe (one Adam at 1e-3 vs SLOW); ten seeds at the draft's budget, 40,800"
PAIRING = "seeds 270-279: ORC_S_D128 with ORC_A_D128 on each seed (same initial weights and batches)"
LR = 1e-3
ITERS = 40800
SEEDS = tuple(range(270, 280))
S67_NAME, S67_SHA, S67_ARM = "capacity", "17e56410a1db", "ORC_P16_D128"
S67_SEEDS = (270, 271)
REL_B, RISK_B = 9, 5
CUT = set()
_A = "the perfect gate; a single Adam at 0.001 on every parameter (the recorded validity recipe)"
_S = "the perfect gate; SLOW: every trainable parameter 0.0001 for updates 1-2400, 0.001 after (no gate parameters)"
ARMS_ORDER = ("ORC_A_D128", "ORC_S_D128")


def _spec(slow):
    sp = dict(cfg="k16", task="P16S4", gate="perfect", size=[2, 128, 3])
    if slow:
        sp["slow"] = True
    return sp


def build_arms(iters=ITERS, seeds=SEEDS):
    return {
        "ORC_A_D128": dict(key="ORC_A_D128", spec=_spec(False), seeds=tuple(seeds), iters=iters, lr=LR, opt="adam", role="oracle",
                           sched=_A, label="oracle, P=16, D128 (N=256, D=128), the recorded recipe (single Adam)"),
        "ORC_S_D128": dict(key="ORC_S_D128", spec=_spec(True), seeds=tuple(seeds), iters=iters, lr=LR, opt="adam", role="oracle",
                           sched=_S, label="oracle, P=16, D128 (N=256, D=128), SLOW"),
    }


ARMS = build_arms()


def make_dry(iters=2400):
    global NAME, ITERS, ARMS
    NAME = "dry_" + NAME
    ITERS = iters
    ARMS = build_arms(iters=iters, seeds=(270,))


# ── Readings ─────────────────────────────────────────────────────────────────
def arm_reading(b, complete):
    if not complete:
        return "INCOMPLETE"
    if b >= REL_B:
        return "reliable"
    if b >= RISK_B:
        return "at risk"
    return "unreliable"


def spec_reading(b_a, b_s, complete):
    if not complete:
        return "INCOMPLETE, no reading"
    if b_a >= REL_B:
        return "keep the draft's oracle"
    if b_s >= REL_B:
        return "the D128 oracle on SLOW"
    if max(b_a, b_s) >= RISK_B:
        return f"VALID_N = 1 at D128 (with {'ORC_S_D128' if b_s > b_a else 'ORC_A_D128'})"
    return "no D128 validity check at 40,800"


def both_two(b, n=10):
    return b * (b - 1) / (n * (n - 1)) if n >= 2 else float("nan")


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar2(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


# ── The S67 records (CHECK) ──────────────────────────────────────────────────
def s67_record(seed):
    st = ec.load_store(S67_NAME)
    for k, r in st["runs"].items():
        a, s, h, _ = k.split("|", 3)
        if a == S67_ARM and int(s) == seed and h == S67_SHA and r.get("ok"):
            return r
    return None


def same_upto(r, rec, upto):
    """(ok, why): r equals rec through `upto`: curve, statistics (stats[0] is update 0), early curve."""
    if rec is None or not r or not r.get("ok"):
        return False, "missing"
    c1 = [x for x in r.get("curve") or [] if x[0] <= upto]
    c0 = [x for x in rec.get("curve") or [] if x[0] <= upto]
    if c1 != c0:
        return False, "curve differs"
    n = len(c0) + 1
    if (r.get("stats") or [])[:n] != (rec.get("stats") or [])[:n]:
        return False, "statistics differ"
    if [x for x in r.get("early") or [] if x[0] <= upto] != [x for x in rec.get("early") or [] if x[0] <= upto]:
        return False, "early curve differs"
    return True, ""


# ── Evaluation and report ────────────────────────────────────────────────────
def evaluate_all(store=None):
    me = sys.modules[__name__]
    st = ec.load_store(NAME) if store is None else store
    rows, b, comp = {}, {}, {}
    for arm in ARMS_ORDER:
        a = ARMS[arm]
        runs = c16.runs_of(me, arm, store=st)
        rows[arm] = {s: runs.get(s) for s in a["seeds"]}
        comp[arm] = all(rows[arm][s] is not None and rows[arm][s].get("ok") for s in a["seeds"])
        b[arm] = sum(1 for s in a["seeds"] if c19.bound(rows[arm][s]))
    ba, bs = b["ORC_A_D128"], b["ORC_S_D128"]
    pairs = dict(both=0, a_only=0, s_only=0, neither=0)
    for s in ARMS["ORC_A_D128"]["seeds"]:
        x, y = c19.bound(rows["ORC_A_D128"][s]), c19.bound(rows["ORC_S_D128"].get(s))
        pairs["both" if x and y else "a_only" if x else "s_only" if y else "neither"] += 1
    return dict(rows=rows, b=b, complete=comp,
                arm_reading={k: arm_reading(b[k], comp[k]) for k in ARMS_ORDER},
                spec=spec_reading(ba, bs, comp["ORC_A_D128"] and comp["ORC_S_D128"]),
                pairs=pairs, p_mcnemar=mcnemar2(pairs["a_only"], pairs["s_only"]))


def _end_streams(r):
    e = r.get("end") or {}
    v = e.get("stream_acc") or e.get("per_stream") or e.get("acc_by_stream")
    return "/".join(f"{x:.2f}" for x in v) if v else "-"


def report():
    ev = evaluate_all()
    for arm in ARMS_ORDER:
        a = ARMS[arm]
        print(f"  ARM {arm} ({a['label']}), seeds {ec.fmt_seeds(a['seeds'])}; {a['sched']}; {a['iters']} updates; lr {a['lr']:g}")
        print("    seed | outcome     trans   stop   acc  | per-stream at the end | S67 record (ORC_P16_D128, 43,200)")
        for s in a["seeds"]:
            r = ev["rows"][arm][s]
            if r is None:
                print(f"    {s:>4} | not run")
                continue
            if not r.get("ok"):
                print(f"    {s:>4} | FAILED {r.get('error')}")
                continue
            rec = s67_record(s) if arm == "ORC_A_D128" else None
            ref = (f"{rec.get('outcome')} trans {rec.get('transition')} acc {rec.get('acc'):.2f}" if rec else "-")
            print(f"    {s:>4} | {str(r.get('outcome')):<10} {str(r.get('transition')):>6} {r.get('stopped_at'):>6} {r.get('acc'):5.2f} | "
                  f"{_end_streams(r):<21} | {ref}" + (f"  [resumed at {r.get('resumed')}]" if r.get("resumed") else ""))
        b = ev["b"][arm]
        lo, hi = wilson(b, len(a["seeds"]))
        tr = sorted(r["transition"] for r in ev["rows"][arm].values() if c19.bound(r))
        print(f"  READING S69 {arm}: {ev['arm_reading'][arm]} (bound {b}/{len(a['seeds'])}, Wilson 95% [{lo:.2f}, {hi:.2f}]; two seeds "
              f"both bind: {both_two(b, len(a['seeds'])):.2f}; transitions {tr}; reliable >= {REL_B}, at risk >= {RISK_B}, else unreliable)")
        print()
    p = ev["pairs"]
    print(f"  READING S69 for the spec: {ev['spec']} (b_A {ev['b']['ORC_A_D128']}, b_S {ev['b']['ORC_S_D128']}; rules: b_A >= {REL_B} keep; "
          f"else b_S >= {REL_B} SLOW; else max >= {RISK_B} VALID_N = 1; else no D128 validity check)")
    print(f"  DESCRIPTIVE (no reading): paired, both bind {p['both']}, single Adam only {p['a_only']}, SLOW only {p['s_only']}, "
          f"neither {p['neither']}; exact McNemar two-sided p = {ev['p_mcnemar']:.3g}")
    return ev
