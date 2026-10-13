#!/usr/bin/env python
"""
docs/review_2026-10/rev9/check_S75_v6.py -- referee V: session I's S75 at four sources (S = 4, GTX 1660 SUPER) against
the SIXTH version of the Revision 9 draft (claude/rev9-draft at 42c4557, committed 2026-10-12 23:14:04 UTC,
multichannel_hebbian_report_v9.tex = "tex6") and against session I's own write-up as it stood on claude/explore-I at
that moment (9034b83: report_2.md and s75_S4_descriptions.md as regenerated at 559b942, the pre-run note).

The v6 copy of check_S75.py (which checked c5e4508 = tex5 against 4c3984a). Every record recomputation of check_S75.py
is kept; every draft row is anchored by a verbatim fragment located in tex6 (the line number is found, not assumed);
rows were added for every sentence of the S75 passages that changed between c5e4508 and 42c4557 (tex6:458, 478, 516)
and for report_2.md's section 6 and table row and the regenerated descriptions (559b942), which the draft's provenance
row now cites.

Read-only: every file is read with `git show <ref>:<path>`; nothing is run but this script (no torch, no session code).
Output: check_S75_v6.md (stdout).

    python3 -I docs/review_2026-10/rev9/check_S75_v6.py > docs/review_2026-10/rev9/check_S75_v6.md
                                                       (from the repository root, after `git fetch`)

Conventions (unchanged from check_S75.py)
  * Statuses: MATCH (reproduces at the stated precision), PARTIAL (reproduces only in part, or a stated bound holds
    only after rounding, or the statement is stale), DISAGREE, NOT RECORD-CHECKABLE (the quantity is not in the
    records; its source is named).
  * "chance" for the source probe: four sources, so 0.25; the empirical band is taken from the ORACLE runs, whose stack
    has no reason to carry the source (all their probe values at every layer and evaluation: printed below); "above
    chance" means > 0.27.
  * eta^2 is the record field routing[L]["eta"] (the registered ROUTED quantity, explore_i_dense.routed); the
    word-index-restricted fields eta_pos12 / eta_pos5p are reported in notes only.
  * The source probe "from the residual entering layer l" is the record field probe["L<l>"]["acc"]
    (explore_i_common.residuals: feats["L<l>"] is the residual entering layer l; "final" is the lm_head input).
"""

import datetime as dt
import hashlib
import json
import math
import re
import statistics
import subprocess

BR = "origin/claude/explore-I"
DRAFT = "42c4557"
DRAFT_T = "2026-10-12T23:14:04Z"
REC = "4c3984a"                     # the last commit that changed the S75 results file
OLD_DRAFT = "c5e4508"
GPU = "explore_out/I/gpu/"
RES = GPU + "s75_S4_NVIDIA_GeForce_GTX_1660_SUPER_results.json"
TEX = "multichannel_hebbian_report_v9.tex"


def show(ref, path, binary=False):
    out = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout
    return out if binary else out.decode()


def git_time(c, fmt="%cI"):
    return subprocess.run(["git", "show", "-s", f"--format={fmt}", c], capture_output=True, text=True,
                          check=True).stdout.strip()


def parent(c):
    return subprocess.run(["git", "rev-parse", "--short=7", f"{c}^"], capture_output=True, text=True,
                          check=True).stdout.strip()


ROWS = []


def row(loc, stmt, recomputed, status, note="", src=""):
    ROWS.append((loc, stmt, recomputed, status, (f"[{src}] " if src else "") + note))


def by_arm(st):
    R = {}
    for k, r in st["runs"].items():
        a, s = k.split("|")[:2]
        if r.get("ok"):
            R.setdefault(a, {})[int(s)] = r
    return R


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def band(k, n):
    if k >= math.ceil(0.9 * n):
        return "RELIABLE"
    if k >= math.ceil(0.5 * n):
        return "MAJORITY"
    return "MINORITY" if k >= 1 else "NEVER"


def routed_at(e):
    rt = (e or {}).get("routing") or {}
    return bool(rt) and all(v["eta"] >= 0.9 and v["one_to_one"] for v in rt.values())


def at(r, step):
    return next((e for e in r["curve"] if e["step"] == step), None)


def rng(v, d=3):
    v = list(v)
    return f"{min(v):.{d}f}-{max(v):.{d}f}"


def spread(src_gate):
    """largest difference between two sources' mean gate on the same channel (rows of the source x channel table)."""
    S = len(src_gate)
    return max(max(src_gate[i][j] for i in range(S)) - min(src_gate[i][j] for i in range(S))
               for j in range(len(src_gate[0])))


# ── load ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
ST = json.loads(show(HEAD, RES))
R = by_arm(ST)
LEARNED = ("WIDE_SPLIT", "REG3", "WIDE_SLOW")
REG = ("ORACLE", "SINGLE", "WIDE_SPLIT", "REG3")
tex = show(DRAFT, TEX).splitlines()
S74 = json.loads(show(HEAD, "explore_out/I/s74_pilot_results.json"))
assert show(HEAD, "explore_out/I/s74_pilot_results.json") == show("a2588da", "explore_out/I/s74_pilot_results.json")

for ln, needle in [(36, "0.880$ and $0.892$ against a bar of $0.9$"), (61, "route $0/10$ each"),
                   (456, "about 16 hours of training"), (458, "below $0.003$ at every evaluation"),
                   (458, "$0.05$--$0.38$ at the fires"), (458, "below $0.001$ at every evaluation"),
                   (462, "neither learned gate routed"), (464, "half of the perfect gate's shortfall"),
                   (472, "neither one channel nor either learned gate did"),
                   (478, "the window gate without the split on the present configuration first"),
                   (516, "report 2, \\texttt{03bffdf}) & \\texttt{03bffdf} & pilot; 32")]:
    assert needle in tex[ln - 1], (ln, needle)

O, SG, WS, RG, WL = (R[a] for a in ("ORACLE", "SINGLE", "WIDE_SPLIT", "REG3", "WIDE_SLOW"))
oset = statistics.fmean(r["end"]["set"] for r in O.values())
oset_f = {f: statistics.fmean(r["end"][f] for r in O.values()) for f in ("set", "set_1", "set_2", "set_3-4", "set_5+")}

# ── tex5:36 / 61 / 462 / 472: abstract, index, limits, conclusion ─────────────────────────────────────────────────────
oe = {s: r["end"]["set"] for s, r in O.items()}
row("tex5:36, tex5:61, tex5:458", "perfect gate ended at 0.880 and 0.892 (bar 0.9)",
    f"ORACLE 830 {oe[830]:.4f}, 831 {oe[831]:.4f}", "MATCH", "end.set; both < 0.9 -> not VALID", "record")
kws, krg = sum(routed_at(r["end"]) for r in WS.values()), sum(routed_at(r["end"]) for r in RG.values())
row("tex5:61, tex5:36, tex5:462, tex5:472", "window gate with split and register gate route 0/10 each; neither "
    "learned gate routed (ten seeds each)", f"ROUTED WIDE_SPLIT {kws}/{len(WS)}, REG3 {krg}/{len(RG)}", "MATCH",
    "end.routing[L].eta >= 0.9 and one_to_one at every gate layer (explore_i_dense.routed)", "record")
sp_all = {s: max(e["probe"][L]["acc"] for e in r["curve"] for L in e["probe"]) for s, r in SG.items()}
sp_end = {s: max(r["end"]["probe"][L]["acc"] for L in r["end"]["probe"]) for s, r in SG.items()}
row("tex5:61, tex5:36, tex5:462, tex5:472", "one channel learns to carry the source on none of ten (at four sources)",
    f"SINGLE probe max over layers: end {rng(sp_end.values())}; any evaluation {rng(sp_all.values())} "
    f"(838: {sp_all[838]:.3f})", "MATCH",
    "descriptive; against 0.96-0.98 at two sources (S74, below); the largest value, SINGLE 838's 0.340 (L3, 8400), "
    "is far from carrying", "record probe[L].acc")

# ── tex5:456: the dense-loss paragraph (S74 records) ─────────────────────────────────────────────────────────────────
s74 = {}
for k, r in S74["runs"].items():
    a, s = k.split("|")[:2]
    s74.setdefault(a, {})[int(s)] = r
it74 = {r["iters"] for rr in s74.values() for r in rr.values()}
seeds74 = {a: sorted(rr) for a, rr in s74.items()}
row("tex5:456", "pilot: perfect gate, one channel, single-source control at two sources, 12,000 updates, four seeds",
    f"iters {sorted(it74)}; arms {sorted(seeds74)}; seeds {seeds74['C2_SINGLE']}", "MATCH",
    "C1 has no NOMIX record (stopped after the decision, report_1 l.18)", "S74 record")


def transition(r):
    c = r["curve"]
    j = max(range(1, len(c)), key=lambda i: c[i]["probe"]["L3"]["acc"] - c[i - 1]["probe"]["L3"]["acc"])
    return c[j - 1], c[j]


moved = {s: transition(r) for s, r in s74["C2_SINGLE"].items()
         if max(e["probe"]["L3"]["acc"] for e in r["curve"]) >= 0.9}
row("tex5:456", "three of four runs of the shorter-block configuration (C2: blocks of 3-7 words)",
    f"C2 SINGLE runs whose L3 probe reaches >= 0.9: {len(moved)}/4 ({sorted(moved)})", "MATCH", "", "S74 record")
rise = {s: b["set"] - a["set"] for s, (a, b) in moved.items()}
row("tex5:456", "a step transition raised the set accuracy by 0.09-0.22",
    f"{rng(rise.values(), 4)} (" + ", ".join(f"{s}: +{v:.4f}" for s, v in sorted(rise.items())) + ")", "MATCH",
    "transition = the evaluation with the largest one-evaluation rise of the L3 probe (as recompute_I)", "S74 record")
pre = {s: a["probe"]["L3"]["acc"] for s, (a, b) in moved.items()}
post = {s: b["probe"]["L3"]["acc"] for s, (a, b) in moved.items()}
row("tex5:456", "probe from the residual entering layer 3 from 0.54-0.58 to 0.96-0.98",
    f"before {rng(pre.values(), 4)}; at the transition {rng(post.values(), 4)}", "MATCH", "probe['L3'].acc",
    "S74 record")
O74 = s74["C2_ORACLE"]
closed = {}
for s, (a, b) in moved.items():
    oa = at(O74[s], a["step"])["set"]
    r = s74["C2_SINGLE"][s]
    closed[s] = 1 - (O74[s]["end"]["set"] - r["end"]["set"]) / (oa - a["set"])
row("tex5:456", "closing 44-83% of each seed's gap to the perfect gate",
    ", ".join(f"{s}: {100 * v:.0f}%" for s, v in sorted(closed.items())), "MATCH",
    "1 - (ORACLE end - SINGLE end) / (ORACLE - SINGLE at the evaluation before the transition), same seed; "
    "report_1 l.32 says 60-80% (another definition; recompute_I row 7)", "S74 record")
omax = max(e["probe"][L]["acc"] for a in ("C1_ORACLE", "C2_ORACLE") for r in s74[a].values() for e in r["curve"]
           for L in e["probe"])
row("tex5:456", "the perfect gate's own stack never carried the source", f"S74 ORACLE probe max {omax:.4f}", "MATCH",
    "every layer, every evaluation, C1 and C2", "S74 record")
cfg = ST["meta"]["config"]
row("tex5:456", "four sources need sequences of about 770 tokens; S = 4, L = 768; 24,000 updates",
    f"config {cfg}; iters {ST['meta']['iters']}; L = S*(NB+M) = {cfg['S'] * (cfg['NB'] + cfg['M'])}", "MATCH",
    "meta.config, meta.iters; every record iters = " + str({r["iters"] for rr in R.values() for r in rr.values()}),
    "record")
n32 = sum(len(R[a]) for a in REG)
trh = sum(r["train_secs"] for a in REG for r in R[a].values()) / 3600
wallh = sum(r["secs"] for a in REG for r in R[a].values()) / 3600
row("tex5:456", "S75: 32 runs, about 16 hours of training on that card",
    f"{n32} registered runs; train_secs {trh:.2f} h, wall secs {wallh:.2f} h", "MATCH",
    "ORACLE 2 + SINGLE 10 + WIDE_SPLIT 10 + REG3 10; all R = " + str({r["R"] for rr in R.values() for r in rr.values()}),
    "record")
row("tex5:456", "perfect gate on two seeds; both must end at a set accuracy of at least 0.9; ten seeds each",
    f"ORACLE seeds {sorted(O)}; others " + "; ".join(f"{a} {min(R[a])}-{max(R[a])} (n={len(R[a])})"
                                                      for a in ("SINGLE", "WIDE_SPLIT", "REG3")),
    "MATCH", "VALID rule: s75_s4.py docstring OUTCOMES and report() (valid = both ORACLE end set >= 0.9)",
    "record + code")
row("tex5:456", "window gate at width 16 (the longest block plus two)", f"lmax {cfg['lmax']} + 2 = {cfg['lmax'] + 2}",
    "MATCH", "explore_gpu/run.py make_model: ic.to_window(m, seed, spec.get('width') or task.lmax + 2); "
    "S75_ARMS WIDE_SPLIT has no width", "code + record meta")

# ── tex5:458: Four sources (S75) ─────────────────────────────────────────────────────────────────────────────────────
row("tex5:458", "bound of 0.927", "not in the records", "NOT RECORD-CHECKABLE",
    "pre-run note l.53 (explore_i_task.ideal_set_acc, 144-512 sequences); the draft quotes it correctly",
    "session I note")
res5 = json.loads(show("5bc6c2e", RES))
arms5 = sorted({k.split("|")[0] for k in res5["runs"]})
resb = json.loads(show("b649751", RES))
first_learned = [c for c in ("86652b1", "5bc6c2e", "1e31409", "b649751")
                 if any(k.split("|")[0] in LEARNED for k in json.loads(show(c, RES))["runs"])]
row("tex5:458", "pre-run note committed after the perfect-gate and one-channel runs and before any learned-arm record "
    "had been pushed", f"results file at 5bc6c2e: {len(res5['runs'])} records, arms {arms5}; first commit with a "
    f"learned-arm record: {first_learned[0]} ({git_time(first_learned[0])})", "MATCH",
    f"5bc6c2e {git_time('5bc6c2e')}; parent {parent('5bc6c2e')} = 86652b1", "git")
row("tex5:458", "(three window-gate runs had by then finished on the card, unseen by the session)",
    "no per-run timestamp in the records; see the timeline section", "NOT RECORD-CHECKABLE",
    "a reconstruction from meta.started + secs that assumes (a) one uninterrupted invocation from ORACLE 830 to "
    "REG3 839 and (b) a UTC clock; under them WIDE_SPLIT 832 ends 00:44:50, 98 s before 5bc6c2e (00:46:28), and 833 is "
    "already running; (a) is contradicted by 86652b1's per-GPU summary (written by drive() only when an invocation ends, it "
    "lists no WIDE_SPLIT/REG3 record); without (a) the records allow 0-3 finished runs", "record + git")
row("tex5:458", "follow-up fixed: fresh seeds from 840; 32 blocks per source at L = 1536, bound 0.964; a bar from "
    "perfect-gate and one-channel quantities; a split that does not fire on a one-to-one map",
    "note l.111-119: 840 and up; NB = 32, M = 352, L = 1536, bound 0.964; ORACLE/SINGLE inputs only; guard or spare",
    "MATCH", "not record quantities; the S75b draft (1e31409/114cd08) gives that configuration's bound as 0.963",
    "session I note")
eta_max = {a: max((v["eta"], s, e["step"], L) for s, r in R[a].items() for e in r["curve"]
                  for L, v in e["routing"].items()) for a in LEARNED}
eta12 = {a: max(v["eta_pos12"] or 0 for r in R[a].values() for e in r["curve"] for v in e["routing"].values())
         for a in LEARNED}
nev = {a: sum(len(r["curve"]) for r in R[a].values()) for a in LEARNED}
row("tex5:458", "eta^2 by source below 0.003 at every evaluation of every gate layer in all twenty runs",
    "; ".join(f"{a} max {eta_max[a][0]:.5f} (seed {eta_max[a][1]}, step {eta_max[a][2]}, {eta_max[a][3]})"
              for a in ("WIDE_SPLIT", "REG3")), "MATCH",
    f"curve[i].routing[L].eta over {nev['WIDE_SPLIT']} + {nev['REG3']} evaluations (REG3: 3 layers); the restricted "
    f"field eta_pos12 reaches {eta12['REG3']:.4f} (REG3) / {eta12['WIDE_SPLIT']:.4f} (WIDE_SPLIT), not the "
    "registered quantity", "record")
oto = {a: sum(v["one_to_one"] for r in R[a].values() for e in r["curve"] for v in e["routing"].values())
       for a in LEARNED}
row("tex5:458", "no map was one-to-one at any evaluation",
    "; ".join(f"{a} {oto[a]} one-to-one of {nev[a] * len(next(iter(R[a].values()))['end']['routing'])} "
              f"(evaluation x layer)" for a in ("WIDE_SPLIT", "REG3")), "MATCH",
    "curve[i].routing[L].one_to_one; also 0 in the 64-sequence map_before of all 69 eligible checks", "record")
ws_set = [r["end"]["set"] for r in WS.values()]
rg9 = {s: r["end"]["set"] for s, r in RG.items()}
sg9 = {s: r["end"]["set"] for s, r in SG.items() if s != 838}
row("tex5:458", "window gate ended 0.529-0.536", rng(ws_set, 4) + f" -> {min(ws_set):.3f}-{max(ws_set):.3f}", "MATCH",
    "end.set, ten seeds", "record")
row("tex5:458", "register gate ended 0.535-0.540",
    f"ten seeds {rng(rg9.values(), 4)}; nine seeds (without 839) {rng([v for s, v in rg9.items() if s != 839], 4)}",
    "PARTIAL", f"holds on nine seeds; REG3 839 ended {rg9[839]:.4f} (the sentence does not say 'nine'; 839 is "
    "mentioned later only as 'one register seed')", "record")
row("tex5:458", "0.534-0.540 on nine seeds of one channel", f"SINGLE without 838: {rng(sg9.values(), 4)}", "MATCH",
    f"838 ended {SG[838]['end']['set']:.4f}", "record")
row("tex5:458", "0.880-0.892 for the perfect gate", f"{min(oe.values()):.4f}-{max(oe.values()):.4f}", "MATCH", "",
    "record")
fires = {s: [c for c in r["checks"] if c["fired"]] for s, r in WS.items()}
row("tex5:458", "the split fired its three permitted times in every window-gate run",
    f"splits field {sorted({r['splits'] for r in WS.values()})}; fired checks per run "
    f"{sorted({len(v) for v in fires.values()})}", "MATCH", "record splits, checks[].fired", "record")
first3 = {}
for s, r in WS.items():
    el = [c["step"] for c in r["checks"] if c["eligible"]]
    first3[s] = ([c["step"] for c in fires[s]], el)
nine = [s for s, (f, el) in first3.items() if f == [7200, 12000, 16800]]
row("tex5:458", "at the first three checks at which the trigger allowed a fire (7200, 12,000 and 16,800) in nine of ten",
    f"fires at 7200/12000/16800 on {len(nine)}/10; seed 830 at {first3[830][0]} (first eligible check "
    f"{first3[830][1][0]}; at 7200 its probe rose {WS[830]['checks'][2]['acc'] - WS[830]['checks'][2]['prev']:.4f} "
    ">= 0.02)", "MATCH", "the fires are always the earliest eligible checks the 4800 gap leaves", "record checks")
facc = [c["acc"] for v in fires.values() for c in v]
row("tex5:458", "the trigger's probe read 0.518-0.542 at the fires", rng(facc, 4), "MATCH", "checks[].acc (fired)",
    "record")
row("tex5:458", "threshold 0.88, set from the pilot's perfect gate", f"meta.split_thr {ST['meta']['split_thr']}",
    "MATCH", "docstring: C1's lowest ORACLE final SET 0.910 - 0.03", "record + code")
fm = [m for v in fires.values() for c in v for m in c["masses"]]
em = [m for s, v in fires.items() for c in v for m in at(WS[s], c["step"])["routing"]["L1"]["mass"]]
row("tex5:458", "the mass over the four channels was uneven, 0.05-0.38 at the fires",
    f"checks[].masses {rng(fm, 4)}; evaluation at the fire step {rng(em, 4)}", "MATCH", "", "record")
fsp = [spread(at(WS[s], c["step"])["routing"]["L1"]["src_gate"]) for s, v in fires.items() for c in v]
row("tex5:458", "identical across sources to within 0.02 (at the fires)",
    f"largest per-channel difference between sources at the 30 fire-step evaluations {max(fsp):.4f}; "
    f"{sum(x > 0.02 for x in fsp)}/30 exceed 0.020", "PARTIAL",
    "routing.L1.src_gate at the evaluation of the fire step (the checks hold only pooled masses); holds after "
    "rounding to two decimals only", "record")
fmg = [at(WS[s], c["step"])["routing"]["L1"]["margin"] for s, v in fires.items() for c in v]
amg = max(v["margin"] for a in LEARNED for r in R[a].values() for e in r["curve"] for v in e["routing"].values())
row("tex5:458", "margin at most 0.04", f"at the fires max {max(fmg):.4f}; any evaluation of any learned arm max {amg:.4f}",
    "MATCH", "routing[L].margin", "record")
f121 = sum(len(set(c["map_before"])) == 4 for v in fires.values() for c in v)
rbf = [s for s, v in fires.items() if any(routed_at(at(WS[s], c["step"])) for c in v) and not routed_at(WS[s]["end"])]
row("tex5:458", "the hazard (a split on a routed run overwrites a live channel) did not arise, no run having routed",
    f"fires on a one-to-one map_before: {f121}; runs ROUTED at the evaluation of a fire and not at the end: {len(rbf)}; "
    f"evaluations ROUTED in any learned run: {sum(routed_at(e) for a in LEARNED for r in R[a].values() for e in r['curve'])}",
    "MATCH", "", "record")
rise48 = {s: (r["end"]["set"] - at(r, 4800)["set"], max(e["set"] for e in r["curve"] if e["step"] >= 4800)
              - at(r, 4800)["set"]) for s, r in SG.items()}
nine_r = {s: v for s, v in rise48.items() if s != 838}
row("tex5:458", "one channel: on nine of ten seeds its accuracy rose by at most 0.03 after 4800 updates",
    f"nine seeds: end - SET(4800) {rng([v[0] for v in nine_r.values()], 4)}; max after 4800 - SET(4800) "
    f"{rng([v[1] for v in nine_r.values()], 4)}; 838: {rise48[838][0]:.4f}", "PARTIAL",
    "holds at two decimals; unrounded, end-rise exceeds 0.030 on " +
    ", ".join(f"{s} ({v[0]:.4f})" for s, v in sorted(nine_r.items()) if v[0] > 0.03) +
    "; the max-after-4800 rise on " + ", ".join(f"{s} ({v[1]:.4f})" for s, v in sorted(nine_r.items()) if v[1] > 0.03),
    "record curve[].set")
o_probe = [e["probe"][L]["acc"] for r in O.values() for e in r["curve"] for L in e["probe"]]
s9_probe = max(e["probe"][L]["acc"] for s, r in SG.items() if s != 838 for e in r["curve"] for L in e["probe"])
row("tex5:458", "...and a probe of the source from the residual stayed at chance (nine one-channel seeds)",
    f"nine SINGLE seeds, every layer and evaluation: max {s9_probe:.4f}; ORACLE (the chance band) {rng(o_probe, 4)}",
    "MATCH", "the nine-seed maximum (SINGLE 830, L3, 1200) is 0.003 above the ORACLE band; from 3600 on their "
    "maximum is " + f"{max(e['probe'][L]['acc'] for s, r in SG.items() if s != 838 for e in r['curve'] if e['step'] >= 3600 for L in e['probe']):.4f}",
    "record probe[L].acc")
p838 = max((e["probe"][L]["acc"], e["step"], L) for e in SG[838]["curve"] for L in e["probe"])
p839 = [(e["step"], max(e["probe"][L]["acc"] for L in ("L2", "L3", "final"))) for e in RG[839]["curve"]]
rg_rise = {s: r["end"]["set"] - at(r, 4800)["set"] for s, r in RG.items()}
row("tex5:458", "on the tenth, as on one register seed, a late rise of about 0.06 took the probe to 0.29 and 0.28",
    f"SINGLE 838: SET 4800->end +{rise48[838][0]:.4f}, end {SG[838]['end']['set']:.4f} vs nine-seed median "
    f"{statistics.median(sg9.values()):.4f} (+{SG[838]['end']['set'] - statistics.median(sg9.values()):.4f}); end probe "
    f"max {sp_end[838]:.4f} (final), peak {p838[0]:.4f} ({p838[2]}, {p838[1]}). REG3 839: SET 4800->end "
    f"+{rg_rise[839]:.4f}, end {RG[839]['end']['set']:.4f} vs median of the other nine "
    f"{statistics.median(v for s, v in rg9.items() if s != 839):.4f}; end probe max "
    f"{max(RG[839]['end']['probe'][L]['acc'] for L in RG[839]['end']['probe']):.4f} (L3)", "PARTIAL",
    "0.29 / 0.28 are the end values (max over layers); 'about 0.06' is the excess over the other seeds "
    "(0.057 / 0.049), the rise after 4800 is 0.091 / 0.084; SINGLE 838's probe peaked at 0.340 at 8400 before settling; "
    "REG3 839's probe never returned to the chance band after 1200 (3600: " +
    f"{dict(p839)[3600]:.3f}, 7200: {dict(p839)[7200]:.3f}), so the rise did not 'take' it to 0.28", "record")
p1200 = {a: [r["curve"][0]["probe"]["L2"]["acc"] for r in R[a].values()] for a in R}
assert all(r["curve"][0]["step"] == 1200 for rr in R.values() for r in rr.values())
row("tex5:458", "probe from the residual entering layer 2 read 0.35-0.61 at 1200 updates in every gated run",
    "; ".join(f"{a} {rng(p1200[a], 4)}" for a in LEARNED), "MATCH",
    f"combined {rng(p1200['WIDE_SPLIT'] + p1200['REG3'] + p1200['WIDE_SLOW'], 4)}; probe['L2'] is the residual "
    "entering layer 2 (explore_i_common.residuals)", "record")
row("tex5:458", "...chance in the perfect-gate and one-channel runs (at 1200)",
    f"ORACLE {rng(p1200['ORACLE'], 4)}; SINGLE {rng(p1200['SINGLE'], 4)}", "MATCH", "", "record")
p3600 = {a: {s: max(at(r, 3600)["probe"][L]["acc"] for L in ("L2", "L3", "final")) for s, r in R[a].items()}
         for a in R}
above = [(a, s, round(v, 4)) for a in LEARNED for s, v in p3600[a].items() if v > 0.27]
row("tex5:458", "...and had lost it by 3600", "; ".join(f"{a} max {max(p3600[a].values()):.4f}" for a in LEARNED),
    "PARTIAL", f"above the ORACLE band (> 0.27) at 3600: {above}; 29 of 30 gated runs (19 of 20 registered) at chance",
    "record")
late = {a: max((e["probe"][L]["acc"], s, e["step"], L) for s, r in R[a].items() for e in r["curve"]
               if e["step"] >= 3600 for L in e["probe"]) for a in R}
n_above = {a: sum(max(e["probe"][L]["acc"] for L in e["probe"]) > 0.27 for r in R[a].values() for e in r["curve"]
                  if e["step"] >= 3600) for a in R}
row("caller's paraphrase", "chance in every arm from 3600",
    "; ".join(f"{a} max {late[a][0]:.4f} (seed {late[a][1]}, {late[a][2]}, {late[a][3]}; evaluations > 0.27: "
              f"{n_above[a]})" for a in ("ORACLE", "SINGLE", "WIDE_SPLIT", "REG3", "WIDE_SLOW")), "PARTIAL",
    "holds except SINGLE 838 (above 0.27 at " + str(sum(max(e["probe"][L]["acc"] for L in e["probe"]) > 0.27
                                                      for e in SG[838]["curve"] if e["step"] >= 3600))
    + " of 18 evaluations from 3600, from " + str(next(e["step"] for e in SG[838]["curve"]
                                                      if max(e["probe"][L]["acc"] for L in e["probe"]) > 0.27))
    + " on) and REG3 839 (above 0.27 at " + str(sum(max(e["probe"][L]["acc"] for L in e["probe"]) > 0.27
                                                   for e in RG[839]["curve"] if e["step"] >= 3600))
    + " of 18); the draft itself names those two seeds, but with their end values only", "record")
# shortfall audit (note), partly recomputable: the bin weights are fixed by the fixed evaluation set
import numpy as np  # noqa: E402  (numpy only for the 4-weight least squares)
A = np.array([[e[f"set_{b}"] for b in ("1", "2", "3-4", "5+")] for rr in R.values() for r in rr.values()
              for e in r["curve"]])
y = np.array([e["set"] for rr in R.values() for r in rr.values() for e in r["curve"]])
w = np.linalg.lstsq(A, y, rcond=None)[0]
resid = float(np.abs(A @ w - y).max())
share = {s: w[0] * (0.944 - r["end"]["set_1"]) / (0.927 - r["end"]["set"]) for s, r in O.items()}
row("tex5:458, tex5:464", "half of the perfect gate's shortfall at a block's first word",
    f"bin weights {np.round(w, 4).tolist()} (max residual {resid:.1e}); bin-1 share of (0.927 - SET) with the note's "
    f"bin-1 bound 0.944: 830 {share[830]:.2f}, 831 {share[831]:.2f}", "MATCH",
    "the two bounds are the note's (not records); set_1 830 "
    f"{O[830]['end']['set_1']:.4f}, 831 {O[831]['end']['set_1']:.4f}", "record + session I note")
row("tex5:458", "decay-faded ideal learner 0.680 at four sources against 0.840 at two", "not in the records",
    "NOT RECORD-CHECKABLE", "pre-run note l.55-56; quoted correctly", "session I note")
row("tex5:458", "WIDE_SLOW: nine seeds run so far (the tenth is queued)",
    f"WIDE_SLOW records at {HEAD}: seeds {sorted(WL)}", "MATCH", "839 absent; 'queued' is not record-checkable "
    "(4c3984a's message says 'pending')", "record + git")
wl9 = [r["end"]["set"] for r in WL.values()]
ws9 = [WS[s]["end"]["set"] for s in WL]
row("tex5:458", "0.530-0.536 against 0.529-0.536", f"WIDE_SLOW {rng(wl9, 4)}; WIDE_SPLIT same nine seeds {rng(ws9, 4)}",
    "MATCH", "", "record")
row("tex5:458", "WIDE_SLOW eta^2 below 0.001 at every evaluation, no one-to-one map at any",
    f"max eta {eta_max['WIDE_SLOW'][0]:.5f}; one-to-one {oto['WIDE_SLOW']} of {nev['WIDE_SLOW']}", "MATCH",
    f"eta_pos12 reaches {eta12['WIDE_SLOW']:.4f}", "record")
diffs = {s: WL[s]["end"]["set"] - WS[s]["end"]["set"] for s in WL}
pair_r = [(routed_at(WS[s]["end"]), routed_at(WL[s]["end"])) for s in WL]
row("tex5:458", "matched the split arm seed for seed (nine seeds)",
    f"ROUTED pairs (split, slow): {sorted(set(pair_r))} on {len(pair_r)} seeds; |SET difference| max "
    f"{max(abs(v) for v in diffs.values()):.5f} (seed {max(diffs, key=lambda s: abs(diffs[s]))})", "MATCH", "", "record")
row("caller's paraphrase", "WIDE_SLOW's nine seeds within 0.003 of WIDE_SPLIT (SET)",
    ", ".join(f"{s}: {v:+.5f}" for s, v in sorted(diffs.items())), "PARTIAL",
    f"seed 831: {WS[831]['end']['set']:.5f} vs {WL[831]['end']['set']:.5f} = {diffs[831]:+.5f} > 0.003 unrounded "
    "(0.533 vs 0.536 at three decimals); the draft's own text says only 'seed for seed'", "record")
row("tex5:478", "Next: ... the window gate without the split on the present configuration first",
    f"{len(WL)}/10 WIDE_SLOW runs recorded at {HEAD} and described in tex5:458", "PARTIAL",
    "stale at c5e4508: only seed 839 remains", "record + git")
row("tex5:516", "provenance: records 86652b1, b649751; pre-run note 5bc6c2e; report 2 03bffdf; commit 03bffdf; runs "
    "'pilot; 32'", f"tex5:458 also reads the WIDE_SLOW records of {HEAD} ({len(WL)} runs; {git_time(HEAD)})", "PARTIAL",
    f"4c3984a is not listed, the commit column stops at 03bffdf and the count omits the 9 WIDE_SLOW runs; the other "
    f"commits exist with the stated content (driver 5f03929 = the code SHA's tree, below)", "git")

# ── the S75 rule ──────────────────────────────────────────────────────────────────────────────────────────────────────
valid = len(O) == 2 and all(r["end"]["set"] >= 0.9 for r in O.values())
done = all(len(R[a]) == 10 for a in ("WIDE_SPLIT", "REG3")) and len(O) == 2
best = max(kws, krg)
tag = ("UNTESTED" if not valid else ("carries to four sources" if best >= 8 else "does not carry")) if done else "incomplete"
summ = show(HEAD, GPU + "s75_S4_summary.md")
row("rule (s75_s4.py READING, report())", "VALID := both ORACLE runs end with SET >= 0.9; reading UNTESTED if not "
    "VALID; else 'carries' iff the better of WIDE_SPLIT and REG3 is ROUTED on >= 8/10",
    f"VALID {valid} (0.8915, 0.8801); done {done}; ROUTED {{'WIDE_SPLIT': {kws}, 'REG3': {krg}}}; verdict {tag}",
    "MATCH" if tag == "UNTESTED" and "READING (pre-fixed): UNTESTED" in summ else "DISAGREE",
    "summary prints 'READING (pre-fixed): UNTESTED'; had VALID held, 0/10 < 8/10 would read 'does not carry'",
    "record + code")
lo, hi = wilson(0, 10)
lo2, hi2 = wilson(2, 2)
row("rule", "ROUTED 0/10 [0.00, 0.28] NEVER (both read arms); ORACLE 2/2 [0.34, 1.00] RELIABLE",
    f"Wilson 0/10 [{lo:.2f}, {hi:.2f}] {band(0, 10)}; 2/2 [{lo2:.2f}, {hi2:.2f}] {band(2, 2)}", "MATCH",
    "explore_i_common.wilson / band (z = 1.96)", "record")
us = [s for s, r in SG.items() if r["end"]["set"] <= oset - 0.2]
row("rule (printed, not read)", "S74 usability at S = 4: ORACLE mean 0.886 (>= 0.9: False); SINGLE <= ORACLE - 0.2 "
    "on 10/10", f"ORACLE mean {oset:.4f}; {len(us)}/10; smallest margin {oset - SG[838]['end']['set']:.4f} (838)",
    "MATCH", "note l.23: 'smallest margin 0.29'", "record")
d0, d1 = show("5d9dcf9", "explore_gpu/s75_s4.py"), show(HEAD, "explore_gpu/s75_s4.py")
pick = lambda t: (re.search(r"READING \(fixed\):.*?\n(?=RUNNING)", t, re.S).group(0),
                  re.findall(r"^(ITERS|SPLIT_THR|SEEDS|ORACLE_SEEDS|READ_ARMS|EXTENDED|HI) = .*$", t, re.M),
                  re.search(r"OUTCOMES.*?\n(?=READING)", t, re.S).group(0))
same = pick(d0) == pick(d1)
consts = re.findall(r"^(?:ITERS|SPLIT_THR|SEEDS|ORACLE_SEEDS|READ_ARMS|EXTENDED|HI) = .*$", d1, re.M)
row("rule", "readings, outcomes and constants fixed before the first run and unchanged since",
    f"READING / OUTCOMES text and {len(consts)} constants identical at 5d9dcf9 and {HEAD}: {same}", "MATCH" if same else
    "DISAGREE", f"5d9dcf9 {git_time('5d9dcf9')}; first record: meta.started {ST['meta']['started']} (machine-local "
    "clock; holds for any zone, the gap is ~21.5 h)", "git + record")


def code_sha(c):
    h = hashlib.sha1()
    for n in ST["meta"]["modules"]:
        p = "explore_gpu/__init__.py" if n == "explore_gpu" else n.replace(".", "/") + ".py"
        h.update(n.encode())
        h.update(show(c, p, binary=True))
    return h.hexdigest()[:12]


shas = {c: code_sha(c) for c in ("5d9dcf9", "ab65d9b", "9faa37b", "5f03929", "86652b1", "5bc6c2e", "b649751",
                                  "03bffdf", HEAD)}
rec_sha = {r["sha"] for rr in R.values() for r in rr.values()}
first_eq = next(c for c, v in shas.items() if v == ST["meta"]["sha"])
row("rule / provenance tex5:516", "code SHA 5fb120a92d69 (driver 5f03929), unchanged through the note and every "
    "record commit", f"records' sha {sorted(rec_sha)}; recomputed: " + ", ".join(f"{c} {v}" for c, v in shas.items()),
    "MATCH", f"SHA-1 over meta.modules' sources as explore_gpu/run.code_sha does; first commit with that SHA {first_eq} "
    f"({git_time(first_eq)}); the note's 'as committed in 5d9dcf9' holds for the rules, the code is 5f03929's",
    "git + record")
row("rule", "pre-run note 5bc6c2e (00:46:28 UTC 12 Oct) committed before any WIDE_SPLIT/REG3 run",
    "not establishable from the records (see the timeline)", "NOT RECORD-CHECKABLE",
    "git establishes only 'before any learned-arm record was pushed' (first: b649751 13:53:09 UTC)", "record + git")
row("rule", "descriptions generator (s75_writeup.py) and S75b draft committed blind (1e31409, parent 5bc6c2e)",
    f"1e31409 parent {parent('1e31409')}; 1e31409 {git_time('1e31409')}, b649751 {git_time('b649751')}", "MATCH",
    "parentage as stated; note that 1e31409 was committed 23 min after b649751 reached the branch (merged in 142f47e); "
    "what to describe had been fixed in 5bc6c2e", "git")

# ── pre-run note (5bc6c2e) ───────────────────────────────────────────────────────────────────────────────────────────
NOTE = "explore_out/I/gpu/s75_S4_prerun_note_2026-10-12.md"
assert show("5bc6c2e", NOTE) == show(HEAD, NOTE)
best_o = {s: max((e["set"], e["step"]) for e in r["curve"]) for s, r in O.items()}
row(f"{NOTE}:22", "no ORACLE evaluation ever reached 0.9 (bests 0.892 at 24k, 0.881 at 21.6k)",
    ", ".join(f"{s}: {v[0]:.4f} at {v[1]}" for s, v in best_o.items()), "MATCH", "", "record")
row(f"{NOTE}:23", "SINGLE 0.534-0.540 on 9 seeds, 0.595 on 838; <= ORACLE - 0.2 on 10/10 (smallest margin 0.29)",
    f"{rng(sg9.values(), 4)}; 838 {SG[838]['end']['set']:.4f}; {len(us)}/10; margin {oset - SG[838]['end']['set']:.4f}",
    "MATCH", "", "record")
row(f"{NOTE}:59-61", "ORACLE ends 0.035 and 0.047 under the bound; bin 1 0.719 and 0.712; beats the faded learner by "
    "~0.20", f"0.927 - SET: {0.927 - O[830]['end']['set']:.4f}, {0.927 - O[831]['end']['set']:.4f}; set_1 "
    f"{O[830]['end']['set_1']:.4f}, {O[831]['end']['set_1']:.4f}; mean - 0.680 = {oset - 0.680:.3f}", "MATCH",
    "bounds and the faded learner are the note's constants", "record + note")
row(f"{NOTE}:63-64", "with bin 1 at its bound seed 831 would reach about 0.900",
    f"831: {O[831]['end']['set'] + w[0] * (0.944 - O[831]['end']['set_1']):.4f}", "MATCH",
    "using the recomputed bin-1 weight and the note's bin-1 bound 0.944", "record + note")
row(f"{NOTE}:61-62", "a uniform loss of 0.017-0.036 on bins 2+", "per-bin bounds are not in the records",
    "NOT RECORD-CHECKABLE", "", "note")


def replay(curve, thr, last):
    """the trigger's rule on an evaluation curve: checks at 2400 (reference), 4800, ... through `last`."""
    c = {e["step"]: e["set"] for e in curve}
    prev, out = None, []
    for step in range(2400, last + 1, 2400):
        acc = c[step]
        if prev is not None and acc < thr and acc - prev < 0.02 and len(out) < 3 and (not out or step - out[-1] >= 4800):
            out.append(step)
        prev = acc
    return out


rp = {thr: {s: replay(r["curve"], thr, 21600) for s, r in O.items()} for thr in (0.88, 0.87, 0.86, 0.85)}
rp2 = {thr: sum(len(replay(r["curve"], thr, 9600)) for a in ("C1_ORACLE", "C2_ORACLE") for r in s74[a].values())
       for thr in (0.88, 0.87, 0.86, 0.85)}
row(f"{NOTE}:73-77", "replay: 830 fires at 7200 (0.88); 831 at 4800, 9600, 14400 (0.88), 4800 (0.87), 7200 (0.86); "
    "none at 0.85; S = 2 ORACLEs (8 runs) never", f"S=4: {rp}; S=2 fires per threshold {rp2}", "MATCH"
    if rp[0.88] == {830: [7200], 831: [4800, 9600, 14400]} and rp[0.87] == {830: [], 831: [4800]}
    and rp[0.86] == {830: [], 831: [7200]} and rp[0.85] == {830: [], 831: []} and not any(rp2.values()) else "DISAGREE",
    "replayed on the 1024-sequence evaluation curves as the note did; S=2 at 12000 updates, checks through 9600",
    "record (S75 + S74)")

# ── report_2.md ───────────────────────────────────────────────────────────────────────────────────────────────────────
REP = "explore_out/I/report_2.md"
assert show("03bffdf", REP) == show(HEAD, REP)
gap = lambda r, s, f="set": (r["end"][f] - SG[s]["end"][f]) / (oset_f[f] - SG[s]["end"][f])
mgap = {a: statistics.median(gap(r, s) for s, r in R[a].items()) for a in R}
row(f"{REP}:5-8", "table: median SET (range) ORACLE 0.886 (0.880, 0.892); SINGLE 0.538 (0.534-0.595); WIDE_SPLIT "
    "0.534 (0.529-0.536); REG3 0.539 (0.535-0.588)", "; ".join(
        f"{a} {statistics.median(r['end']['set'] for r in R[a].values()):.3f} ({rng([r['end']['set'] for r in R[a].values()])})"
        for a in REG), "MATCH", "", "record")
row(f"{REP}:5-8", "median GAP ORACLE 1.00, SINGLE 0, WIDE_SPLIT -0.01, REG3 0.00",
    "; ".join(f"{a} {mgap[a]:+.4f}" for a in REG), "MATCH", "GAP as explore_i_dense.gap, mean ORACLE", "record")
row(f"{REP}:7-8", "WIDE_SPLIT 30 splits (3 per run); eta^2 at the end 0.00 per gate layer",
    f"splits {sum(r['splits'] for r in WS.values())}; end eta max WIDE_SPLIT "
    f"{max(r['end']['routing']['L1']['eta'] for r in WS.values()):.5f}, REG3 "
    f"{max(v['eta'] for r in RG.values() for v in r['end']['routing'].values()):.5f}", "MATCH", "", "record")
gaps_s = [oset - SG[s]["end"]["set"] for s in SG]
row(f"{REP}:14, :51, :75", "the interference gap it divides by is 0.35; routing is worth 0.35 SET",
    f"mean ORACLE - SINGLE per seed {rng(gaps_s)}; median {statistics.median(gaps_s):.3f}", "MATCH",
    "0.29 on seed 838", "record")
row(f"{REP}:18", "all four registered arms on the 1660, R = 1, code SHA 5fb120a92d69",
    f"hw {sorted({r['hw'] for rr in R.values() for r in rr.values()})}; R {sorted({r['R'] for rr in R.values() for r in rr.values()})}",
    "MATCH", "", "record")
row(f"{REP}:24", "regenerating the summary reproduces the pushed copy exactly", "not run (session code not executed)",
    "NOT RECORD-CHECKABLE", "the summary's numbers that report_2 quotes are recomputed in these rows; s75_S4_summary.md and "
    "the per-GPU summary at 4c3984a are byte-identical and omit WIDE_SLOW (both written before 4c3984a)", "record")
row(f"{REP}:41-42", "in all 20 learned runs, at each of the 20 evaluations, eta^2 0.00 at every gate layer; no map "
    "one-to-one; no run ROUTED at any point", f"max eta {max(eta_max['WIDE_SPLIT'][0], eta_max['REG3'][0]):.5f}; "
    f"one-to-one {oto['WIDE_SPLIT'] + oto['REG3']}; evaluations per run {sorted({len(r['curve']) for a in REG for r in R[a].values()})}",
    "MATCH", "", "record")
m830 = WS[830]["end"]["routing"]["L1"]["mass"]
endsp = max(spread(v["src_gate"]) for a in ("WIDE_SPLIT", "REG3") for r in R[a].values() for v in r["end"]["routing"].values())
allsp = max(spread(v["src_gate"]) for a in ("WIDE_SPLIT", "REG3") for r in R[a].values() for e in r["curve"]
            for v in e["routing"].values())
row(f"{REP}:43-45", "WIDE_SPLIT 830 ended with gate mass 0.16 / 0.29 / 0.38 / 0.16", " / ".join(f"{x:.4f}" for x in m830),
    "MATCH", "end.routing.L1.mass", "record")
row(f"{REP}:45", "rows of the source x channel table differ by at most 0.02 in any run",
    f"at the end max {endsp:.4f}; at any evaluation max {allsp:.4f}", "PARTIAL",
    "holds at the end after rounding to two decimals; unrounded 0.023, and 0.042 at some evaluation (REG3)", "record")
mmed = {a: statistics.median(statistics.fmean(v["margin"] for v in r["end"]["routing"].values()) for r in R[a].values())
        for a in ("WIDE_SPLIT", "REG3")}
row(f"{REP}:45", "the median margin is 0.03", "; ".join(f"{a} {v:.4f}" for a, v in mmed.items()), "PARTIAL",
    "0.03 for WIDE_SPLIT; REG3's is 0.02 (the summary prints both)", "record")
l3 = {s: max(r["end"]["routing"]["L3"]["mass"]) for s, r in RG.items()}
l3src = {s: [max(g) for g in r["end"]["routing"]["L3"]["src_gate"]] for s, r in RG.items()}
row(f"{REP}:46", "REG3's third-layer gate puts about 0.7 on one channel, identically for every source",
    f"end L3 largest channel mass per seed {rng(l3.values())} (median {statistics.median(l3.values()):.3f})", "PARTIAL",
    "the median is about 0.7, the range is " + rng(l3.values(), 2) + "; per-source maxima within "
    f"{max(max(v) - min(v) for v in l3src.values()):.3f}", "record")
row(f"{REP}:48-49", "WIDE_SPLIT 0.529-0.536; REG3 0.535-0.540 on nine seeds; 839 0.588 (GAP 0.14)",
    f"{rng(ws_set)}; {rng([v for s, v in rg9.items() if s != 839])}; {rg9[839]:.4f} (GAP {gap(RG[839], 839):.4f})",
    "MATCH", "", "record")
row(f"{REP}:50", "seed 838's negative GAP (-0.21 and -0.19) comes from SINGLE 838 (0.595)",
    f"WIDE_SPLIT {gap(WS[838], 838):+.4f}, REG3 {gap(RG[838], 838):+.4f}; SINGLE 838 {SG[838]['end']['set']:.4f}",
    "MATCH", "", "record")
lp = {a: [r["end"]["probe"][L]["acc"] for r in R[a].values() for L in r["end"]["probe"]] for a in ("WIDE_SPLIT", "REG3")}
row(f"{REP}:52-53", "the source probe reads chance (0.25 to 0.28) at every layer in every learned run, as in SINGLE",
    f"at the end {rng(lp['WIDE_SPLIT'] + lp['REG3'])}; at 1200 the L2 probe reads "
    f"{rng(p1200['WIDE_SPLIT'] + p1200['REG3'])}", "PARTIAL",
    "true of the end only; at 1200 every learned run's residual carried the source (per-run max over L2/L3/final "
    + rng([max(r["curve"][0]["probe"][L]["acc"] for L in ("L2", "L3", "final")) for a in ("WIDE_SPLIT", "REG3")
           for r in R[a].values()]) + ")",
    "record")
bs = [r["beta_src"] for r in RG.values()]
bw = [r["beta_word"] for r in RG.values()]
row(f"{REP}:55-57", "beta high at SRC (median 0.80, range 0.44-0.90), low at words (median 0.15)",
    f"src median {statistics.median(bs):.4f}, range {min(bs):.4f}-{max(bs):.4f}; word median {statistics.median(bw):.4f}",
    "MATCH", "record beta_src / beta_word", "record")
row(f"{REP}:59", "30 fires, 3 per run, at 7200, 12,000 and 16,800 (seed 830: 9600, 14,400, 19,200)",
    f"{len(facc)} fires; {len(nine)}/10 at 7200/12000/16800; 830 {first3[830][0]}", "MATCH", "", "record")
row(f"{REP}:60-61", "every fire on a map that was not one-to-one, on_target true, at probe SET 0.518-0.542",
    f"one-to-one map_before {f121}/30; on_target {sum(c['on_target'] for v in fires.values() for c in v)}/30; {rng(facc)}",
    "MATCH", "", "record")
row(f"{REP}:62", "fires on a one-to-one map 0; runs ROUTED before a fire and not at the end 0", f"{f121}; {len(rbf)}",
    "MATCH", "", "record")
nxt = [len(set(nx["map_before"])) == 4 for s, r in WS.items() for i, c in enumerate(r["checks"]) if c["fired"]
       for nx in r["checks"][i + 1:i + 2]]
row(f"{REP}:64-65", "by the next check the map had merged again", f"one-to-one at the check after a fire: {sum(nxt)}/{len(nxt)}",
    "MATCH", "checks[i+1].map_before (the last fire of each run has a following check)", "record")
row(f"{REP}:78-80", "S75b design commit 1e31409 has parent 5bc6c2e, which holds only the ORACLE and SINGLE records",
    f"parent {parent('1e31409')}; records at 5bc6c2e: {arms5}", "MATCH", "", "git")

# ── timeline (report_2 l.27-36, the draft's parenthesis) ─────────────────────────────────────────────────────────────
started = dt.datetime.strptime(ST["meta"]["started"], "%Y-%m-%d %H:%M:%S")
TL = []
t = started
for k, r in ST["runs"].items():                     # insertion order = order of saving (load / update / save)
    a, s = k.split("|")[:2]
    TL.append((a, s, t, t + dt.timedelta(seconds=r["secs"]), r["secs"]))
    t = t + dt.timedelta(seconds=r["secs"])
order_ok = [k.split("|")[0] for k in ST["runs"]] == sorted([k.split("|")[0] for k in ST["runs"]],
                                                           key=lambda x: ("ORACLE", "SINGLE", "WIDE_SPLIT", "REG3",
                                                                          "WIDE_SLOW").index(x))
note_t = dt.datetime.fromisoformat(git_time("5bc6c2e")).replace(tzinfo=None)
end = {(a, s): (t0, t1) for a, s, t0, t1, _ in TL}
w832, w833 = end[("WIDE_SPLIT", "832")], end[("WIDE_SPLIT", "833")]
row(f"{REP}:29-31", "from the durations and the start time (18:04 on 10-11), WIDE_SPLIT 830-832 ran 23:11 (10-11) to "
    "00:44 (10-12); the note was committed at 00:46 UTC", f"under one uninterrupted invocation: WIDE_SPLIT 830 starts "
    f"{end[('WIDE_SPLIT', '830')][0]:%H:%M:%S}, 832 ends {w832[1]:%H:%M:%S}; 5bc6c2e {note_t:%H:%M:%S}", "PARTIAL",
    "the arithmetic reproduces (record secs, meta.started); the premise of one uninterrupted invocation is not in the "
    "records and is contradicted by 86652b1's per-GPU summary (see the timeline section)", "record + git")
row(f"{REP}:31", "the other seven WIDE_SPLIT runs and all ten REG3 runs came after it",
    f"under the same premises WIDE_SPLIT 833 started {w833[0]:%H:%M:%S}, "
    f"{(note_t - w833[0]).total_seconds():.1f} s before the note's commit", "DISAGREE", "by report_2's own reconstruction a fourth run (833) was in progress at 00:46:28; the "
    "statement should read 'three had finished and a fourth had started'", "record + git")
row(f"{REP}:32-33", "the 1660's timestamps can only be UTC (ORACLE and SINGLE committed 23:42 UTC, so they ended before)",
    f"SINGLE 839 ends {end[('SINGLE', '839')][1]:%H:%M:%S} machine time; 86652b1 authored {git_time('86652b1', '%aI')}",
    "PARTIAL", "the commit bounds the clock only from one side: machine time - UTC >= "
    f"{(end[('SINGLE', '839')][1] - dt.datetime(2026, 10, 11, 23, 42, 23)).total_seconds() / 60:.0f} min; a clock "
    "ahead of UTC is not excluded (UTC is plausible: meta.started 18:04:26 is 31 s before 203a5ad's 18:04:57 UTC)",
    "record + git")
row(f"{REP}:34", "no learned-arm record was available until b649751 (13:53 UTC on 10-12)",
    f"first commit with a learned-arm record: {first_learned[0]} {git_time(first_learned[0])}", "MATCH", "", "git")
row(f"{REP}:20-21", "deviation: the note assigned WIDE_SPLIT and REG3 to the RTX 3070; the 1660 ran them instead",
    "not in the records", "NOT RECORD-CHECKABLE", "inconsistent with l.29-31: runs that started at 23:11, before the "
    "note, were not run 'instead' of the note's assignment; one of the two accounts is wrong", "report_2")

# ── descriptions (03bffdf) ────────────────────────────────────────────────────────────────────────────────────────────
DES = "explore_out/I/gpu/s75_S4_descriptions.md"
dtext = show(HEAD, DES).splitlines()
assert show("03bffdf", DES) == show(HEAD, DES)
mk = lambda e: "R" if routed_at(e) else ("o" if all(v["one_to_one"] for v in e["routing"].values()) else ".")
n_ok = n_bad = 0
bad = []
for i, line in enumerate(dtext, 1):
    m = re.match(r"^    (\d{3})   (\S+)\s+(\S)\s+(\S+)$", line)
    if not m:
        continue
    arm = "WIDE_SPLIT" if i < 23 else "REG3"
    r = R[arm][int(m.group(1))]
    exp = ("".join(mk(e) for e in r["curve"]), "R" if routed_at(r["end"]) else "-",
           "/".join(f"{v['eta']:.2f}" for v in r["end"]["routing"].values()))
    if exp == (m.group(2), m.group(3), m.group(4)):
        n_ok += 1
    else:
        n_bad += 1
        bad.append((i, exp, m.groups()))
row(f"{DES}:9-35", "section 1: per-run ROUTED trajectories (20 runs x 20 evaluations), end mark and end eta^2",
    f"{n_ok} lines reproduce, {n_bad} differ {bad}", "MATCH" if n_bad == 0 and n_ok == 20 else "DISAGREE",
    "", "record")
row(f"{DES}:37, :124-135", "WIDE_SLOW: no records yet / pending for 10 of 10 seeds",
    f"{len(WL)} WIDE_SLOW records at {HEAD}", "PARTIAL", "stale: generated at 03bffdf, before 4c3984a; not regenerated "
    "(s75_writeup promises to regenerate when they land)", "record + git")
n_ok = 0
bad = []
for i, line in enumerate(dtext, 1):
    m = re.match(r"^\| (\d{3}) \| (\d+) \| (\S+) \((\S+)\) \| (\[.*?\]) \| (\w+) \| (\w+) \| ([^|]+?) \| (\S) \| (\S) \| (\S) \|$",
                 line)
    if not m:
        continue
    s, step = int(m.group(1)), int(m.group(2))
    r = WS[s]
    c = next(c for c in r["checks"] if c["step"] == step)
    nf = sum(x["fired"] for x in r["checks"] if x["step"] < step)
    lastf = max([x["step"] for x in r["checks"] if x["fired"] and x["step"] < step], default=None)
    status = "fired" if c["fired"] else ("blocked (cap)" if nf >= 3 else
                                         "blocked (gap)" if lastf is not None and step - lastf < 4800 else "?")
    e0, e1 = at(r, step), at(r, step + 1200)
    exp = (f"{c['acc']:.3f}", f"{c['prev']:.3f}", str(c["map_before"]), "yes" if len(set(c["map_before"])) == 4 else "no",
           str(c["on_target"]), status, "R" if routed_at(e0) else "-", "R" if routed_at(e1) else "-",
           "R" if routed_at(r["end"]) else "-")
    got = m.groups()[2:]
    elig = c["eligible"] and (c["fired"] or c["blocked"])
    if exp == tuple(got) and elig and (c.get("blocked_by") in (None, status[9:12])):
        n_ok += 1
    else:
        bad.append((i, exp, got, c.get("blocked_by")))
n_elig = sum(c["eligible"] for r in WS.values() for c in r["checks"])
row(f"{DES}:42-112", "section 2: every eligible check of the split trigger (seed, step, probe SET (prev), map before, "
    "one-to-one, on_target, fired / blocked (gap / cap), routed@step / next / end)",
    f"{n_ok} rows reproduce of {n_elig} eligible checks in the records; differing {bad}",
    "MATCH" if not bad and n_ok == n_elig else "DISAGREE",
    "status recomputed independently (cap: 3 earlier fires; gap: < 4800 since the last) and compared with blocked_by",
    "record")
row(f"{DES}:116-118", "fires 30 over 10 runs; on a one-to-one map 0; routed before a fire and not at the end 0 []; "
    "runs with at least one fire 10/10", f"{len(facc)}; {f121}; {len(rbf)}; {sum(bool(v) for v in fires.values())}/10",
    "MATCH", "", "record")
n_ok = 0
bad = []
for i, line in enumerate(dtext, 1):
    m = re.match(r"^\| (\d{3}) \| (\S) \| (\d) \| (\S+) \| (\S+) \| pending \| - \| - \|$", line)
    if not m:
        continue
    r = WS[int(m.group(1))]
    exp = ("R" if routed_at(r["end"]) else "-", str(r["splits"]), f"{r['end']['routing']['L1']['eta']:.2f}",
           f"{r['end']['set']:.3f}")
    if exp == m.groups()[1:]:
        n_ok += 1
    else:
        bad.append((i, exp, m.groups()))
row(f"{DES}:124-133", "section 4: WIDE_SPLIT's columns (routed@end, fires, eta^2, SET) per seed",
    f"{n_ok}/10 reproduce; differing {bad}", "MATCH" if n_ok == 10 and not bad else "DISAGREE", "", "record")
row(f"{DES}:3", "generated under code SHA 5fb120a92d69; the registered result is UNTESTED", f"{ST['meta']['sha']}; {tag}",
    "MATCH", "", "record")

# ── print ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
from collections import Counter  # noqa: E402

cnt = Counter(r[3] for r in ROWS)
print(f"Counts: {len(ROWS)} rows; " + ", ".join(f"{k} {cnt[k]}" for k in
                                                ("MATCH", "PARTIAL", "DISAGREE", "NOT RECORD-CHECKABLE")))
print()
print("| # | line or file:line | statement | recomputed | status | note |")
print("|---|---|---|---|---|---|")
for i, (loc, stmt, rec, stt, note) in enumerate(ROWS, 1):
    esc = lambda x: str(x).replace("|", "\\|")
    print(f"| {i} | {esc(loc)} | {esc(stmt)} | {esc(rec)} | {stt} | {esc(note)} |")
print()
print("TIMELINE (machine clock of meta.started, one uninterrupted invocation assumed; record secs):")
print(f"  record order is the arms' order: {order_ok}")
for a, s, t0, t1, secs in TL:
    flag = " <- straddles 5bc6c2e" if t0 < note_t < t1 else ""
    print(f"  {a:<10} {s} {t0:%m-%d %H:%M:%S} -> {t1:%m-%d %H:%M:%S} ({secs:7.1f} s){flag}")
sum2 = sum(r["secs"] for a in ("WIDE_SPLIT", "REG3") for r in R[a].values())
bt = dt.datetime.fromisoformat(git_time("b649751")).replace(tzinfo=None)
print(f"  WIDE_SPLIT + REG3 wall time {sum2:.0f} s = {sum2 / 3600:.2f} h; b649751 at {bt}; latest possible start of "
      f"WIDE_SPLIT 830 if contiguous to REG3 839: {bt - dt.timedelta(seconds=sum2):%m-%d %H:%M:%S}")
sumw = sum(r["secs"] for r in WL.values())
ht = dt.datetime.fromisoformat(git_time(HEAD)).replace(tzinfo=None)
print(f"  WIDE_SLOW 830-838 wall time {sumw:.0f} s; 4c3984a at {ht}; latest start {ht - dt.timedelta(seconds=sumw):%H:%M:%S}; "
      f"the uninterrupted reconstruction would end 838 at {TL[-1][3]:%m-%d %H:%M:%S} and 839 by "
      f"{TL[-1][3] + dt.timedelta(seconds=1860):%H:%M:%S}, yet 839 was pending at {ht:%H:%M}: WIDE_SLOW ran in a later "
      "invocation, so gaps between invocations are real and unrecorded")
print(f"  meta.task_check.secs at 86652b1 / b649751 / {HEAD}: " + " / ".join(
    str(json.loads(show(c, RES))["meta"]["task_check"]["secs"]) for c in ("86652b1", "b649751", HEAD)))
s86 = show("86652b1", GPU + "s75_S4_NVIDIA_GeForce_GTX_1660_SUPER_summary.md")
print("  86652b1 per-GPU summary rows: " + "; ".join(l for l in s86.splitlines() if l.startswith("| WIDE_SPLIT")
                                                       or l.startswith("| REG3") or l.startswith("READING")))
print("CHANCE BAND (ORACLE probe, all layers and evaluations): " + rng(o_probe, 4))
