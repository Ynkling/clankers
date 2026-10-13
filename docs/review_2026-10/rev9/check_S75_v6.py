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
def rev_at(branch, when):
    return subprocess.run(["git", "rev-list", "-1", f"--before={when}", branch], capture_output=True, text=True,
                          check=True).stdout.strip()[:7]


HEAD = rev_at(BR, DRAFT_T)                          # session I's head when the sixth draft was committed
TIP = subprocess.run(["git", "rev-parse", "--short=7", BR], capture_output=True, text=True, check=True).stdout.strip()
assert git_time(DRAFT).startswith("2026-10-12T23:14:04")
ST = json.loads(show(HEAD, RES))
assert show(HEAD, RES) == show(REC, RES), "the S75 results file changed after 4c3984a"
ST_TIP = json.loads(show(TIP, RES))
R = by_arm(ST)
LEARNED = ("WIDE_SPLIT", "REG3", "WIDE_SLOW")
REG = ("ORACLE", "SINGLE", "WIDE_SPLIT", "REG3")
tex = show(DRAFT, TEX).splitlines()
tex5 = show(OLD_DRAFT, TEX).splitlines()
assert len(tex) == len(tex5)
CHANGED = [i + 1 for i, (a, b) in enumerate(zip(tex5, tex)) if a != b]
S74 = json.loads(show(HEAD, "explore_out/I/s74_pilot_results.json"))
assert show(HEAD, "explore_out/I/s74_pilot_results.json") == show("a2588da", "explore_out/I/s74_pilot_results.json")


def T(*frags):
    """'tex6:N' for every line holding each verbatim fragment (asserted present)."""
    out = []
    for f in frags:
        hits = [i + 1 for i, x in enumerate(tex) if f in x]
        assert hits, f"tex6 fragment not found: {f!r}"
        out += [f"tex6:{h}" for h in hits]
    return ", ".join(dict.fromkeys(out))


O, SG, WS, RG, WL = (R[a] for a in ("ORACLE", "SINGLE", "WIDE_SPLIT", "REG3", "WIDE_SLOW"))
oset = statistics.fmean(r["end"]["set"] for r in O.values())
oset_f = {f: statistics.fmean(r["end"][f] for r in O.values()) for f in ("set", "set_1", "set_2", "set_3-4", "set_5+")}

# ── abstract, index, limits, conclusion ───────────────────────────────────────────────────────────────────────────────
oe = {s: r["end"]["set"] for s, r in O.items()}
row(T("the perfect gate ended at $0.880$ and $0.892$ against a bar of $0.9$", "the perfect gate at $0.880$ and $0.892$ against $0.9$",
      "the perfect gate ended at $0.880$ and $0.892$ against a bound of $0.927$"),
    "perfect gate ended at 0.880 and 0.892 (bar 0.9)",
    f"ORACLE 830 {oe[830]:.4f}, 831 {oe[831]:.4f}", "MATCH", "end.set; both < 0.9 -> not VALID", "record")
kws, krg = sum(routed_at(r["end"]) for r in WS.values()), sum(routed_at(r["end"]) for r in RG.values())
row(T("route $0/10$ each", "neither one channel nor either learned gate did", "neither learned gate routed"),
    "window gate with split and register gate route 0/10 each; neither "
    "learned gate routed (ten seeds each)", f"ROUTED WIDE_SPLIT {kws}/{len(WS)}, REG3 {krg}/{len(RG)}", "MATCH",
    "end.routing[L].eta >= 0.9 and one_to_one at every gate layer (explore_i_dense.routed)", "record")
sp_all = {s: max(e["probe"][L]["acc"] for e in r["curve"] for L in e["probe"]) for s, r in SG.items()}
sp_end = {s: max(r["end"]["probe"][L]["acc"] for L in r["end"]["probe"]) for s, r in SG.items()}
row(T("one channel learns to carry the source on none of ten", "neither one channel nor either learned gate did",
      "one channel did not carry the source either"),
    "one channel learns to carry the source on none of ten (at four sources)",
    f"SINGLE probe max over layers: end {rng(sp_end.values())}; any evaluation {rng(sp_all.values())} "
    f"(838: {sp_all[838]:.3f})", "MATCH",
    "descriptive; against 0.96-0.98 at two sources (S74, below); the largest value, SINGLE 838's 0.340 (L3, 8400), "
    "is far from carrying", "record probe[L].acc")

# ── tex6:456: the dense-loss paragraph (S74 records) ─────────────────────────────────────────────────────────────────
s74 = {}
for k, r in S74["runs"].items():
    a, s = k.split("|")[:2]
    s74.setdefault(a, {})[int(s)] = r
it74 = {r["iters"] for rr in s74.values() for r in rr.values()}
seeds74 = {a: sorted(rr) for a, rr in s74.items()}
row(T("one channel and a single-source control at two sources, 12{,}000 updates, four seeds"),
    "pilot: perfect gate, one channel, single-source control at two sources, 12,000 updates, four seeds",
    f"iters {sorted(it74)}; arms {sorted(seeds74)}; seeds {seeds74['C2_SINGLE']}", "MATCH",
    "C1 has no NOMIX record (stopped after the decision, report_1 l.18)", "S74 record")


def transition(r):
    c = r["curve"]
    j = max(range(1, len(c)), key=lambda i: c[i]["probe"]["L3"]["acc"] - c[i - 1]["probe"]["L3"]["acc"])
    return c[j - 1], c[j]


moved = {s: transition(r) for s, r in s74["C2_SINGLE"].items()
         if max(e["probe"]["L3"]["acc"] for e in r["curve"]) >= 0.9}
row(T("in three of four runs of the shorter-block configuration"),
    "three of four runs of the shorter-block configuration (C2: blocks of 3-7 words)",
    f"C2 SINGLE runs whose L3 probe reaches >= 0.9: {len(moved)}/4 ({sorted(moved)})", "MATCH", "", "S74 record")
rise = {s: b["set"] - a["set"] for s, (a, b) in moved.items()}
row(T("a step transition raised the set accuracy by $0.09$--$0.22$"), "a step transition raised the set accuracy by 0.09-0.22",
    f"{rng(rise.values(), 4)} (" + ", ".join(f"{s}: +{v:.4f}" for s, v in sorted(rise.items())) + ")", "MATCH",
    "transition = the evaluation with the largest one-evaluation rise of the L3 probe (as recompute_I)", "S74 record")
pre = {s: a["probe"]["L3"]["acc"] for s, (a, b) in moved.items()}
post = {s: b["probe"]["L3"]["acc"] for s, (a, b) in moved.items()}
row(T("from $0.54$--$0.58$ to $0.96$--$0.98$"), "probe from the residual entering layer 3 from 0.54-0.58 to 0.96-0.98",
    f"before {rng(pre.values(), 4)}; at the transition {rng(post.values(), 4)}", "MATCH", "probe['L3'].acc",
    "S74 record")
O74 = s74["C2_ORACLE"]
closed = {}
for s, (a, b) in moved.items():
    oa = at(O74[s], a["step"])["set"]
    r = s74["C2_SINGLE"][s]
    closed[s] = 1 - (O74[s]["end"]["set"] - r["end"]["set"]) / (oa - a["set"])
row(T("closing $44$--$83\\%$ of each seed's gap to the perfect gate"), "closing 44-83% of each seed's gap to the perfect gate",
    ", ".join(f"{s}: {100 * v:.0f}%" for s, v in sorted(closed.items())), "MATCH",
    "1 - (ORACLE end - SINGLE end) / (ORACLE - SINGLE at the evaluation before the transition), same seed; "
    "report_1 l.32 says 60-80% (another definition; recompute_I row 7)", "S74 record")
omax = max(e["probe"][L]["acc"] for a in ("C1_ORACLE", "C2_ORACLE") for r in s74[a].values() for e in r["curve"]
           for L in e["probe"])
row(T("the perfect gate's own stack never carried the source"), "the perfect gate's own stack never carried the source",
    f"S74 ORACLE probe max {omax:.4f}", "MATCH", "every layer, every evaluation, C1 and C2", "S74 record")
cfg = ST["meta"]["config"]
row(T("four sources need sequences of about 770 tokens", "$S = 4$ and $L = 768$, 24{,}000 updates"),
    "four sources need sequences of about 770 tokens; S = 4, L = 768; 24,000 updates",
    f"config {cfg}; iters {ST['meta']['iters']}; L = S*(NB+M) = {cfg['S'] * (cfg['NB'] + cfg['M'])}", "MATCH",
    "meta.config, meta.iters; every record iters = " + str({r["iters"] for rr in R.values() for r in rr.values()}),
    "record")
n32 = sum(len(R[a]) for a in REG)
trh = sum(r["train_secs"] for a in REG for r in R[a].values()) / 3600
wallh = sum(r["secs"] for a in REG for r in R[a].values()) / 3600
row(T("S75: 32 runs, about 16 hours of training on that card"), "S75: 32 runs, about 16 hours of training on that card",
    f"{n32} registered runs; train_secs {trh:.2f} h, wall secs {wallh:.2f} h", "MATCH",
    "ORACLE 2 + SINGLE 10 + WIDE_SPLIT 10 + REG3 10; all R = " + str({r["R"] for rr in R.values() for r in rr.values()}),
    "record")
row(T("the perfect gate on two seeds as the validity condition (both must end at a set accuracy of at least $0.9$)"),
    "perfect gate on two seeds; both must end at a set accuracy of at least 0.9; ten seeds each",
    f"ORACLE seeds {sorted(O)}; others " + "; ".join(f"{a} {min(R[a])}-{max(R[a])} (n={len(R[a])})"
                                                      for a in ("SINGLE", "WIDE_SPLIT", "REG3")),
    "MATCH", "VALID rule: s75_s4.py docstring OUTCOMES and report() (valid = both ORACLE end set >= 0.9)",
    "record + code")
row(T("the window gate at width 16 (the longest block plus two)"), "window gate at width 16 (the longest block plus two)",
    f"lmax {cfg['lmax']} + 2 = {cfg['lmax'] + 2}",
    "MATCH", "explore_gpu/run.py make_model: ic.to_window(m, seed, spec.get('width') or task.lmax + 2); "
    "S75_ARMS WIDE_SPLIT has no width", "code + record meta")

# ── tex6:458: Four sources (S75) ─────────────────────────────────────────────────────────────────────────────────────
row(T("against a bound of $0.927$"), "bound of 0.927", "not in the records", "NOT RECORD-CHECKABLE",
    "pre-run note l.53 (explore_i_task.ideal_set_acc, 144-512 sequences); the draft quotes it correctly",
    "session I note")
res5 = json.loads(show("5bc6c2e", RES))
arms5 = sorted({k.split("|")[0] for k in res5["runs"]})
resb = json.loads(show("b649751", RES))
first_learned = [c for c in ("86652b1", "5bc6c2e", "1e31409", "b649751")
                 if any(k.split("|")[0] in LEARNED for k in json.loads(show(c, RES))["runs"])]
row(T("committed after the perfect-gate and one-channel runs and before any learned-arm record had been pushed"),
    "pre-run note committed after the perfect-gate and one-channel runs and before any learned-arm record "
    "had been pushed", f"results file at 5bc6c2e: {len(res5['runs'])} records, arms {arms5}; first commit with a "
    f"learned-arm record: {first_learned[0]} ({git_time(first_learned[0])})", "MATCH",
    f"5bc6c2e {git_time('5bc6c2e')}; parent {parent('5bc6c2e')} = 86652b1", "git")
row(T("(whether learned-arm runs had started on the card by then is not recorded)"),
    "(whether learned-arm runs had started on the card by then is not recorded) [changed: tex5 said three window-gate "
    "runs had finished]",
    "no record carries a per-run start or end time; meta.started is written once (the first invocation's start); "
    "git gives commit times only", "MATCH",
    "the fifth pass's fix, adopted. The timeline section below still shows what the records allow: under one "
    "uninterrupted invocation and a UTC clock, WIDE_SPLIT 832 ends 98 s before 5bc6c2e and 833 is running; without that "
    "premise (contradicted by 86652b1's per-GPU summary) 0-3 had finished. Session I's own explore_out/README.md at "
    f"{HEAD} still says three had (row below)", "record + git")
row(T("fresh seeds from 840", "has a bound of $0.964$", "a bar derived only from perfect-gate and one-channel quantities"),
    "follow-up fixed: fresh seeds from 840; 32 blocks per source at L = 1536, bound 0.964; a bar from "
    "perfect-gate and one-channel quantities; a split that does not fire on a one-to-one map",
    "note l.111-119: 840 and up; NB = 32, M = 352, L = 1536, bound 0.964; ORACLE/SINGLE inputs only; guard or spare",
    "MATCH", "not record quantities; the S75b draft (1e31409/114cd08) gives that configuration's bound as 0.963",
    "session I note")
eta_max = {a: max((v["eta"], s, e["step"], L) for s, r in R[a].items() for e in r["curve"]
                  for L, v in e["routing"].items()) for a in LEARNED}
eta12 = {a: max(v["eta_pos12"] or 0 for r in R[a].values() for e in r["curve"] for v in e["routing"].values())
         for a in LEARNED}
nev = {a: sum(len(r["curve"]) for r in R[a].values()) for a in LEARNED}
row(T("was below $0.003$ at every evaluation of every gate layer in all twenty runs"),
    "eta^2 by source below 0.003 at every evaluation of every gate layer in all twenty runs",
    "; ".join(f"{a} max {eta_max[a][0]:.5f} (seed {eta_max[a][1]}, step {eta_max[a][2]}, {eta_max[a][3]})"
              for a in ("WIDE_SPLIT", "REG3")), "MATCH",
    f"curve[i].routing[L].eta over {nev['WIDE_SPLIT']} + {nev['REG3']} evaluations (REG3: 3 layers); the restricted "
    f"field eta_pos12 reaches {eta12['REG3']:.4f} (REG3) / {eta12['WIDE_SPLIT']:.4f} (WIDE_SPLIT), not the "
    "registered quantity", "record")
oto = {a: sum(v["one_to_one"] for r in R[a].values() for e in r["curve"] for v in e["routing"].values())
       for a in LEARNED}
row(T("no map was one-to-one at any evaluation"), "no map was one-to-one at any evaluation",
    "; ".join(f"{a} {oto[a]} one-to-one of {nev[a] * len(next(iter(R[a].values()))['end']['routing'])} "
              f"(evaluation x layer)" for a in ("WIDE_SPLIT", "REG3")), "MATCH",
    "curve[i].routing[L].one_to_one; also 0 in the 64-sequence map_before of all 69 eligible checks", "record")
ws_set = [r["end"]["set"] for r in WS.values()]
rg9 = {s: r["end"]["set"] for s, r in RG.items()}
sg9 = {s: r["end"]["set"] for s, r in SG.items() if s != 838}
row(T("both ended where one channel ends ($0.529$--$0.536$,"), "window gate ended 0.529-0.536",
    rng(ws_set, 4) + f" -> {min(ws_set):.3f}-{max(ws_set):.3f}", "MATCH", "end.set, ten seeds", "record")
rg_nine = [v for s, v in rg9.items() if s != 839]
row(T("$0.535$--$0.540$ on nine seeds with one at $0.588$"),
    "register gate ended 0.535-0.540 on nine seeds with one at 0.588 [changed: tex5 gave 0.535-0.540 for all ten]",
    f"nine seeds (without 839) {rng(rg_nine, 4)}; 839 {rg9[839]:.4f}",
    "MATCH" if f"{min(rg_nine):.3f}-{max(rg_nine):.3f}" == "0.535-0.540" and f"{rg9[839]:.3f}" == "0.588" else "DISAGREE",
    "the fifth pass's item, resolved", "record")
row(T("$0.534$--$0.540$ on nine seeds of one channel with one at $0.595$"),
    "0.534-0.540 on nine seeds of one channel with one at 0.595 [', with one at 0.595' added]",
    f"SINGLE without 838: {rng(sg9.values(), 4)}; 838 {SG[838]['end']['set']:.4f}",
    "MATCH" if f"{SG[838]['end']['set']:.3f}" == "0.595" else "DISAGREE", "", "record")
row(T("and $0.880$--$0.892$ for the perfect gate"), "0.880-0.892 for the perfect gate",
    f"{min(oe.values()):.4f}-{max(oe.values()):.4f}", "MATCH", "", "record")
fires = {s: [c for c in r["checks"] if c["fired"]] for s, r in WS.items()}
row(T("The split fired its three permitted times in every window-gate run"),
    "the split fired its three permitted times in every window-gate run",
    f"splits field {sorted({r['splits'] for r in WS.values()})}; fired checks per run "
    f"{sorted({len(v) for v in fires.values()})}", "MATCH", "record splits, checks[].fired", "record")
first3 = {}
for s, r in WS.items():
    el = [c["step"] for c in r["checks"] if c["eligible"]]
    first3[s] = ([c["step"] for c in fires[s]], el)
nine = [s for s, (f, el) in first3.items() if f == [7200, 12000, 16800]]
row(T("at the first three checks at which the trigger allowed a fire (7200, 12{,}000 and 16{,}800 updates) in nine of ten"),
    "at the first three checks at which the trigger allowed a fire (7200, 12,000 and 16,800) in nine of ten",
    f"fires at 7200/12000/16800 on {len(nine)}/10; seed 830 at {first3[830][0]} (first eligible check "
    f"{first3[830][1][0]}; at 7200 its probe rose {WS[830]['checks'][2]['acc'] - WS[830]['checks'][2]['prev']:.4f} "
    ">= 0.02)", "MATCH", "the fires are always the earliest eligible checks the 4800 gap leaves", "record checks")
facc = [c["acc"] for v in fires.values() for c in v]
row(T("the trigger's probe read $0.518$--$0.542$ at the fires"), "the trigger's probe read 0.518-0.542 at the fires",
    rng(facc, 4), "MATCH", "checks[].acc (fired)", "record")
row(T("($0.88$, set from the pilot's perfect gate"), "threshold 0.88, set from the pilot's perfect gate",
    f"meta.split_thr {ST['meta']['split_thr']}", "MATCH", "docstring: C1's lowest ORACLE final SET 0.910 - 0.03",
    "record + code")
fm = [m for v in fires.values() for c in v for m in c["masses"]]
em = [m for s, v in fires.items() for c in v for m in at(WS[s], c["step"])["routing"]["L1"]["mass"]]
row(T("the mass over the four channels was uneven, $0.05$--$0.38$ at the fires"),
    "the mass over the four channels was uneven, 0.05-0.38 at the fires",
    f"checks[].masses {rng(fm, 4)}; evaluation at the fire step {rng(em, 4)}", "MATCH", "", "record")
fsp = [spread(at(WS[s], c["step"])["routing"]["L1"]["src_gate"]) for s, v in fires.items() for c in v]
row(T("identical across sources to within $0.024$"),
    "identical across sources to within 0.024 (at the fires) [changed: tex5 said 0.02]",
    f"largest per-channel difference between sources at the 30 fire-step evaluations {max(fsp):.4f}; "
    f"{sum(x > 0.02 for x in fsp)}/30 exceed 0.020, {sum(x > 0.024 for x in fsp)}/30 exceed 0.024",
    "MATCH" if max(fsp) <= 0.024 else "PARTIAL",
    "routing.L1.src_gate at the evaluation of the fire step (the checks hold only pooled masses); the fifth pass's "
    "item, resolved", "record")
fmg = [at(WS[s], c["step"])["routing"]["L1"]["margin"] for s, v in fires.items() for c in v]
amg = max(v["margin"] for a in LEARNED for r in R[a].values() for e in r["curve"] for v in e["routing"].values())
row(T("margin at most $0.04$"), "margin at most 0.04",
    f"at the fires max {max(fmg):.4f}; any evaluation of any learned arm max {amg:.4f}",
    "MATCH", "routing[L].margin", "record")
f121 = sum(len(set(c["map_before"])) == 4 for v in fires.values() for c in v)
rbf = [s for s, v in fires.items() if any(routed_at(at(WS[s], c["step"])) for c in v) and not routed_at(WS[s]["end"])]
row(T("that hazard did not arise, no run having routed"),
    "the hazard (a split on a routed run overwrites a live channel) did not arise, no run having routed",
    f"fires on a one-to-one map_before: {f121}; runs ROUTED at the evaluation of a fire and not at the end: {len(rbf)}; "
    f"evaluations ROUTED in any learned run: {sum(routed_at(e) for a in LEARNED for r in R[a].values() for e in r['curve'])}",
    "MATCH", "", "record")
rise48 = {s: (r["end"]["set"] - at(r, 4800)["set"], max(e["set"] for e in r["curve"] if e["step"] >= 4800)
              - at(r, 4800)["set"]) for s, r in SG.items()}
nine_r = {s: v for s, v in rise48.items() if s != 838}
mx_end, mx_any = max(v[0] for v in nine_r.values()), max(v[1] for v in nine_r.values())
row(T("on nine of ten seeds its accuracy rose by at most $0.033$ after 4800 updates"),
    "one channel: on nine of ten seeds its accuracy rose by at most 0.033 after 4800 updates [changed: tex5 said 0.03]",
    f"nine seeds: end - SET(4800) {rng([v[0] for v in nine_r.values()], 4)}; max after 4800 - SET(4800) "
    f"{rng([v[1] for v in nine_r.values()], 4)}; 838: {rise48[838][0]:.4f}",
    "MATCH" if max(mx_end, mx_any) <= 0.033 else "PARTIAL",
    "both readings of 'rose' hold unrounded (largest 0.0329, seed 839); the fifth pass's item, resolved",
    "record curve[].set")
o_probe = [e["probe"][L]["acc"] for r in O.values() for e in r["curve"] for L in e["probe"]]
s9_probe = max(e["probe"][L]["acc"] for s, r in SG.items() if s != 838 for e in r["curve"] for L in e["probe"])
row(T("a probe of the source from the residual stayed at chance"),
    "...and a probe of the source from the residual stayed at chance (nine one-channel seeds)",
    f"nine SINGLE seeds, every layer and evaluation: max {s9_probe:.4f}; ORACLE (the chance band) {rng(o_probe, 4)}",
    "MATCH", "the nine-seed maximum (SINGLE 830, L3, 1200) is 0.003 above the ORACLE band; from 3600 on their "
    "maximum is " + f"{max(e['probe'][L]['acc'] for s, r in SG.items() if s != 838 for e in r['curve'] if e['step'] >= 3600 for L in e['probe']):.4f}",
    "record probe[L].acc")
p838 = max((e["probe"][L]["acc"], e["step"], L) for e in SG[838]["curve"] for L in e["probe"])
p839 = [(e["step"], max(e["probe"][L]["acc"] for L in ("L2", "L3", "final"))) for e in RG[839]["curve"]]
pk839 = {lo: max((max(e["probe"][L]["acc"] for L in e["probe"]), e["step"]) for e in RG[839]["curve"] if e["step"] >= lo)
         for lo in (0, 2400, 3600, 4800)}
rg_rise = {s: r["end"]["set"] - at(r, 4800)["set"] for s, r in RG.items()}
end839 = max(RG[839]['end']['probe'][L]['acc'] for L in RG[839]['end']['probe'])
row(T("a late rise ($0.09$ and $0.08$ after 4800)"),
    "on the tenth, as on one register seed, a late rise (0.09 and 0.08 after 4800) [changed: tex5 said 'of about 0.06']",
    f"SET(end) - SET(4800): SINGLE 838 +{rise48[838][0]:.4f}, REG3 839 +{rg_rise[839]:.4f}",
    "MATCH" if (f"{rise48[838][0]:.2f}", f"{rg_rise[839]:.2f}") == ("0.09", "0.08") else "DISAGREE",
    f"for reference, the excess over the other seeds' median end: SINGLE 838 "
    f"+{SG[838]['end']['set'] - statistics.median(sg9.values()):.4f}, REG3 839 "
    f"+{RG[839]['end']['set'] - statistics.median(v for s, v in rg9.items() if s != 839):.4f}; the fifth pass's item, "
    "resolved", "record curve[].set")
row(T("took the probe no higher than $0.34$ ($0.29$ and $0.28$ at the end)"),
    "...took the probe no higher than 0.34 (0.29 and 0.28 at the end) [changed: tex5 said 'took the probe to 0.29 and "
    "0.28']",
    f"SINGLE 838: peak {p838[0]:.4f} ({p838[2]}, {p838[1]}), end max {sp_end[838]:.4f}; REG3 839: peak from 3600 "
    f"{pk839[3600][0]:.4f} ({pk839[3600][1]}), from 4800 {pk839[4800][0]:.4f} ({pk839[4800][1]}), end max {end839:.4f} "
    f"(its early hold: {pk839[0][0]:.4f} at {pk839[0][1]}, {pk839[2400][0]:.4f} at 2400)",
    "MATCH" if p838[0] <= 0.34 else "PARTIAL",
    f"the bound holds only after rounding: SINGLE 838's peak is {p838[0]:.4f} > 0.34 (stated bounds are MATCH only "
    "unrounded, as everywhere in this check); 0.29 / 0.28 are the end values, as now stated. REG3 839's probe never "
    f"returned to the chance band after 1200 (3600: {dict(p839)[3600]:.3f}, 7200: {dict(p839)[7200]:.3f}), so for it "
    "'took the probe' describes a level held since the early transient more than a late rise; 'no higher than 0.34' "
    "holds for it from 2400 on", "record")
p1200 = {a: [r["curve"][0]["probe"]["L2"]["acc"] for r in R[a].values()] for a in R}
assert all(r["curve"][0]["step"] == 1200 for rr in R.values() for r in rr.values())
row(T("read $0.35$--$0.61$ at 1200 updates"),
    "probe from the residual entering layer 2 read 0.35-0.61 at 1200 updates in every gated run",
    "; ".join(f"{a} {rng(p1200[a], 4)}" for a in LEARNED), "MATCH",
    f"combined {rng(p1200['WIDE_SPLIT'] + p1200['REG3'] + p1200['WIDE_SLOW'], 4)}; probe['L2'] is the residual "
    "entering layer 2 (explore_i_common.residuals)", "record")
row(T("chance in the perfect-gate and one-channel runs)"), "...chance in the perfect-gate and one-channel runs (at 1200)",
    f"ORACLE {rng(p1200['ORACLE'], 4)}; SINGLE {rng(p1200['SINGLE'], 4)}", "MATCH", "", "record")
p3600 = {a: {s: max(at(r, 3600)["probe"][L]["acc"] for L in ("L2", "L3", "final")) for s, r in R[a].items()}
         for a in R}
above = [(a, s, round(v, 4)) for a in LEARNED for s, v in p3600[a].items() if v > 0.27]
second = sorted(((v, a, s) for a in LEARNED for s, v in p3600[a].items()), reverse=True)[1]
n_gated = sum(len(R[a]) for a in LEARNED)
row(T("and all but one (a register run, at $0.29$) had lost it by 3600"),
    "...and all but one (a register run, at 0.29) had lost it by 3600 [changed: tex5 said all had]",
    "; ".join(f"{a} max {max(p3600[a].values()):.4f}" for a in LEARNED) + f"; above 0.27 at 3600: {above} of "
    f"{n_gated} gated runs", "MATCH" if [(a, s) for a, s, _ in above] == [("REG3", 839)] and round(above[0][2], 2) == 0.29
    else "DISAGREE",
    f"the next highest at 3600 is {second[1]} {second[2]} at {second[0]:.4f}, {second[0] - max(o_probe):.4f} above the "
    f"ORACLE band's top ({max(o_probe):.4f}) and below 0.27, so the count of one depends on the 0.27 line; the fifth "
    "pass's item, resolved", "record")
late = {a: max((e["probe"][L]["acc"], s, e["step"], L) for s, r in R[a].items() for e in r["curve"]
               if e["step"] >= 3600 for L in e["probe"]) for a in R}
n_above = {a: sum(max(e["probe"][L]["acc"] for L in e["probe"]) > 0.27 for r in R[a].values() for e in r["curve"]
                  if e["step"] >= 3600) for a in R}
row("caller's paraphrase (check_S75)", "chance in every arm from 3600",
    "; ".join(f"{a} max {late[a][0]:.4f} (seed {late[a][1]}, {late[a][2]}, {late[a][3]}; evaluations > 0.27: "
              f"{n_above[a]})" for a in ("ORACLE", "SINGLE", "WIDE_SPLIT", "REG3", "WIDE_SLOW")), "PARTIAL",
    "holds except SINGLE 838 (above 0.27 at " + str(sum(max(e["probe"][L]["acc"] for L in e["probe"]) > 0.27
                                                      for e in SG[838]["curve"] if e["step"] >= 3600))
    + " of 18 evaluations from 3600, from " + str(next(e["step"] for e in SG[838]["curve"]
                                                      if max(e["probe"][L]["acc"] for L in e["probe"]) > 0.27))
    + " on) and REG3 839 (above 0.27 at " + str(sum(max(e["probe"][L]["acc"] for L in e["probe"]) > 0.27
                                                   for e in RG[839]["curve"] if e["step"] >= 3600))
    + " of 18); tex6:458 now names both", "record")
# shortfall audit (note), partly recomputable: the bin weights are fixed by the fixed evaluation set
import numpy as np  # noqa: E402  (numpy only for the 4-weight least squares)
A = np.array([[e[f"set_{b}"] for b in ("1", "2", "3-4", "5+")] for rr in R.values() for r in rr.values()
              for e in r["curve"]])
y = np.array([e["set"] for rr in R.values() for r in rr.values() for e in r["curve"]])
w = np.linalg.lstsq(A, y, rcond=None)[0]
resid = float(np.abs(A @ w - y).max())
share = {s: w[0] * (0.944 - r["end"]["set_1"]) / (0.927 - r["end"]["set"]) for s, r in O.items()}
row(T("puts half of it at a block's first word", "half of the perfect gate's shortfall falls at the block's first word"),
    "half of the perfect gate's shortfall at a block's first word",
    f"bin weights {np.round(w, 4).tolist()} (max residual {resid:.1e}); bin-1 share of (0.927 - SET) with the note's "
    f"bin-1 bound 0.944: 830 {share[830]:.2f}, 831 {share[831]:.2f}", "MATCH",
    "the two bounds are the note's (not records); set_1 830 "
    f"{O[830]['end']['set_1']:.4f}, 831 {O[831]['end']['set_1']:.4f}", "record + session I note")
row(T("scores $0.680$ at four sources against $0.840$ at two"),
    "decay-faded ideal learner 0.680 at four sources against 0.840 at two", "not in the records",
    "NOT RECORD-CHECKABLE", "pre-run note l.55-56; quoted correctly", "session I note")
row(T("on the nine seeds run so far (the tenth is queued)"), "WIDE_SLOW: nine seeds run so far (the tenth is queued)",
    f"WIDE_SLOW records at {HEAD} (I's head at {DRAFT}'s commit time): seeds {sorted(WL)}; at the current tip {TIP}: "
    f"{sorted(int(k.split('|')[1]) for k in ST_TIP['runs'] if k.startswith('WIDE_SLOW'))}", "MATCH",
    f"839 has not landed (results file last changed at {REC}); 'queued' is not record-checkable (I's 4c3984a and 559b942 "
    "say 'pending')", "record + git")
wl9 = [r["end"]["set"] for r in WL.values()]
ws9 = [WS[s]["end"]["set"] for s in WL]
row(T("$0.530$--$0.536$ against $0.529$--$0.536$"), "0.530-0.536 against 0.529-0.536",
    f"WIDE_SLOW {rng(wl9, 4)}; WIDE_SPLIT same nine seeds {rng(ws9, 4)}", "MATCH", "", "record")
row(T("$\\eta^2$ below $0.001$ at every evaluation, no one-to-one map at any"),
    "WIDE_SLOW eta^2 below 0.001 at every evaluation, no one-to-one map at any",
    f"max eta {eta_max['WIDE_SLOW'][0]:.5f}; one-to-one {oto['WIDE_SLOW']} of {nev['WIDE_SLOW']}", "MATCH",
    f"eta_pos12 reaches {eta12['WIDE_SLOW']:.4f}", "record")
diffs = {s: WL[s]["end"]["set"] - WS[s]["end"]["set"] for s in WL}
pair_r = [(routed_at(WS[s]["end"]), routed_at(WL[s]["end"])) for s in WL]
row(T("matched the split arm seed for seed"), "matched the split arm seed for seed (nine seeds)",
    f"ROUTED pairs (split, slow): {sorted(set(pair_r))} on {len(pair_r)} seeds; |SET difference| max "
    f"{max(abs(v) for v in diffs.values()):.5f} (seed {max(diffs, key=lambda s: abs(diffs[s]))})", "MATCH", "", "record")
row("caller's paraphrase (check_S75)", "WIDE_SLOW's nine seeds within 0.003 of WIDE_SPLIT (SET)",
    ", ".join(f"{s}: {v:+.5f}" for s, v in sorted(diffs.items())), "PARTIAL",
    f"seed 831: {WS[831]['end']['set']:.5f} vs {WL[831]['end']['set']:.5f} = {diffs[831]:+.5f} > 0.003 unrounded "
    "(0.533 vs 0.536 at three decimals); the draft's own text says only 'seed for seed'; report_2 l.76 now says it "
    "(row below)", "record")
nxt_old = "the window gate without the split on the present configuration first"
row(T("The dense-loss task at four sources again, in session~I's committed follow-up"),
    "Next: the dense-loss task at four sources again, in session I's committed follow-up: a configuration piloted so "
    "that the perfect gate reaches its bar, fresh seeds, and a split that does not fire on a one-to-one map "
    "[changed: tex5 ended '; the window gate without the split on the present configuration first']",
    f"stale item removed: present in tex5 {any(nxt_old in x for x in tex5)}, in tex6 {any(nxt_old in x for x in tex)}; "
    f"WIDE_SLOW {len(WL)}/10 recorded at {HEAD}, its tenth still pending and described at tex6:458 as queued; "
    "the follow-up as the pre-run note fixes it (l.111-119: fresh seeds from 840, a configuration where 0.9 is "
    "reachable, a split guard) and as S75b's driver (8cb8dce, WIP) implements it (a 2-seed ORACLE pilot with "
    "proceed/stop rules, a guarded trigger)", "MATCH", "the fifth pass's item, resolved", "git + session I note")
prov_runs = len(ST["runs"])
row(T("records \\texttt{86652b1}, \\texttt{b649751}, \\texttt{4c3984a}; pre-run note \\texttt{5bc6c2e}; report 2, "
      "\\texttt{03bffdf}, \\texttt{559b942}) & \\texttt{559b942} & pilot; 41"),
    "provenance: records 86652b1, b649751, 4c3984a; pre-run note 5bc6c2e; report 2 03bffdf, 559b942; commit 559b942; "
    "runs 'pilot; 41' [changed: tex5 stopped at 03bffdf, 'pilot; 32']",
    f"the results file changed at {[c for c in subprocess.run(['git', 'log', '--format=%h', HEAD, '--', RES], capture_output=True, text=True).stdout.split()]}; "
    f"{prov_runs} S75 records at {HEAD}; report_2.md changed at "
    f"{subprocess.run(['git', 'log', '--format=%h', HEAD, '--', 'explore_out/I/report_2.md'], capture_output=True, text=True).stdout.split()}; "
    f"I's commits after 559b942 up to {HEAD}: "
    f"{subprocess.run(['git', 'log', '--format=%h %s', f'559b942..{HEAD}'], capture_output=True, text=True).stdout.strip().splitlines()}",
    "MATCH" if prov_runs == 41 else "DISAGREE",
    "the commits after 559b942 are window_gate's pre-run note and S75b's driver and CHECKs, not S75 material; the "
    "fifth pass's item, resolved", "git + record")

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
row("rule / provenance " + T("driver \\texttt{5f03929}"), "code SHA 5fb120a92d69 (driver 5f03929), unchanged through the note and every "
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
REPT = show(HEAD, REP).splitlines()
REP0 = show("03bffdf", REP).splitlines()
REP_CHANGED = [i + 1 for i in range(min(len(REPT), len(REP0))) if REPT[i] != REP0[i]]
assert REP_CHANGED[0] == 9 and all(n >= 68 for n in REP_CHANGED[1:]), REP_CHANGED   # 559b942: table row, section 6


def RP(frag, n=1):
    """report_2.md at HEAD: the line holding `frag` (asserted unique), with n lines of span."""
    hits = [i + 1 for i, x in enumerate(REPT) if frag in x]
    assert len(hits) == 1, (frag, hits)
    return f"{REP}:{hits[0]}" + (f"-{hits[0] + n - 1}" if n > 1 else "")


gap = lambda r, s, f="set": (r["end"][f] - SG[s]["end"][f]) / (oset_f[f] - SG[s]["end"][f])
mgap = {a: statistics.median(gap(r, s) for s, r in R[a].items()) for a in R}
row(RP("| ORACLE | 830–831 |", 4), "table: median SET (range) ORACLE 0.886 (0.880, 0.892); SINGLE 0.538 (0.534-0.595); WIDE_SPLIT "
    "0.534 (0.529-0.536); REG3 0.539 (0.535-0.588)", "; ".join(
        f"{a} {statistics.median(r['end']['set'] for r in R[a].values()):.3f} ({rng([r['end']['set'] for r in R[a].values()])})"
        for a in REG), "MATCH", "", "record")
row(RP("| ORACLE | 830–831 |", 4), "median GAP ORACLE 1.00, SINGLE 0, WIDE_SPLIT -0.01, REG3 0.00",
    "; ".join(f"{a} {mgap[a]:+.4f}" for a in REG), "MATCH", "GAP as explore_i_dense.gap, mean ORACLE", "record")
row(RP("| WIDE_SPLIT | 830–839 |", 2), "WIDE_SPLIT 30 splits (3 per run); eta^2 at the end 0.00 per gate layer",
    f"splits {sum(r['splits'] for r in WS.values())}; end eta max WIDE_SPLIT "
    f"{max(r['end']['routing']['L1']['eta'] for r in WS.values()):.5f}, REG3 "
    f"{max(v['eta'] for r in RG.values() for v in r['end']['routing'].values()):.5f}", "MATCH", "", "record")
gaps_s = [oset - SG[s]["end"]["set"] for s in SG]
row(", ".join((RP("the interference gap it divides by is 0.35"), RP("Neither label-free gate closed any of the 0.35 gap"), RP("routing is worth 0.35 SET"))), "the interference gap it divides by is 0.35; routing is worth 0.35 SET",
    f"mean ORACLE - SINGLE per seed {rng(gaps_s)}; median {statistics.median(gaps_s):.3f}", "MATCH",
    "0.29 on seed 838", "record")
row(RP("All four registered arms ran on the GTX 1660 SUPER"), "all four registered arms on the 1660, R = 1, code SHA 5fb120a92d69",
    f"hw {sorted({r['hw'] for rr in R.values() for r in rr.values()})}; R {sorted({r['R'] for rr in R.values() for r in rr.values()})}",
    "MATCH", "", "record")
row(RP("regenerating it reproduces the copy the user pushed exactly"), "regenerating the summary reproduces the pushed copy exactly", "not run (session code not executed)",
    "NOT RECORD-CHECKABLE", "the summary's numbers that report_2 quotes are recomputed in these rows; s75_S4_summary.md and "
    "the per-GPU summary at 4c3984a are byte-identical and omit WIDE_SLOW (both written before 4c3984a)", "record")
row(RP("In all 20 learned runs, at each of the 20 evaluations", 2), "in all 20 learned runs, at each of the 20 evaluations, eta^2 0.00 at every gate layer; no map "
    "one-to-one; no run ROUTED at any point", f"max eta {max(eta_max['WIDE_SPLIT'][0], eta_max['REG3'][0]):.5f}; "
    f"one-to-one {oto['WIDE_SPLIT'] + oto['REG3']}; evaluations per run {sorted({len(r['curve']) for a in REG for r in R[a].values()})}",
    "MATCH", "", "record")
m830 = WS[830]["end"]["routing"]["L1"]["mass"]
endsp = max(spread(v["src_gate"]) for a in ("WIDE_SPLIT", "REG3") for r in R[a].values() for v in r["end"]["routing"].values())
allsp = max(spread(v["src_gate"]) for a in ("WIDE_SPLIT", "REG3") for r in R[a].values() for e in r["curve"]
            for v in e["routing"].values())
row(RP("WIDE_SPLIT 830 ended with gate mass", 3), "WIDE_SPLIT 830 ended with gate mass 0.16 / 0.29 / 0.38 / 0.16", " / ".join(f"{x:.4f}" for x in m830),
    "MATCH", "end.routing.L1.mass", "record")
row(RP("source × channel table differ by at most 0.02 in any run"), "rows of the source x channel table differ by at most 0.02 in any run",
    f"at the end max {endsp:.4f}; at any evaluation max {allsp:.4f}", "PARTIAL",
    "holds at the end after rounding to two decimals; unrounded 0.023, and 0.042 at some evaluation (REG3)", "record")
mmed = {a: statistics.median(statistics.fmean(v["margin"] for v in r["end"]["routing"].values()) for r in R[a].values())
        for a in ("WIDE_SPLIT", "REG3")}
row(RP("and the median margin is 0.03"), "the median margin is 0.03", "; ".join(f"{a} {v:.4f}" for a, v in mmed.items()), "PARTIAL",
    "0.03 for WIDE_SPLIT; REG3's is 0.02 (the summary prints both)", "record")
l3 = {s: max(r["end"]["routing"]["L3"]["mass"]) for s, r in RG.items()}
l3src = {s: [max(g) for g in r["end"]["routing"]["L3"]["src_gate"]] for s, r in RG.items()}
row(RP("REG3's third-layer gate puts about 0.7 on one channel"), "REG3's third-layer gate puts about 0.7 on one channel, identically for every source",
    f"end L3 largest channel mass per seed {rng(l3.values())} (median {statistics.median(l3.values()):.3f})", "PARTIAL",
    "the median is about 0.7, the range is " + rng(l3.values(), 2) + "; per-source maxima within "
    f"{max(max(v) - min(v) for v in l3src.values()):.3f}", "record")
row(RP("*WIDE_SPLIT* spans 0.529–0.536", 2), "WIDE_SPLIT 0.529-0.536; REG3 0.535-0.540 on nine seeds; 839 0.588 (GAP 0.14)",
    f"{rng(ws_set)}; {rng([v for s, v in rg9.items() if s != 839])}; {rg9[839]:.4f} (GAP {gap(RG[839], 839):.4f})",
    "MATCH", "", "record")
row(RP("*Seed 838's negative GAP*"), "seed 838's negative GAP (-0.21 and -0.19) comes from SINGLE 838 (0.595)",
    f"WIDE_SPLIT {gap(WS[838], 838):+.4f}, REG3 {gap(RG[838], 838):+.4f}; SINGLE 838 {SG[838]['end']['set']:.4f}",
    "MATCH", "", "record")
lp = {a: [r["end"]["probe"][L]["acc"] for r in R[a].values() for L in r["end"]["probe"]] for a in ("WIDE_SPLIT", "REG3")}
row(RP("The source probe reads chance (0.25 to 0.28", 2), "the source probe reads chance (0.25 to 0.28) at every layer in every learned run, as in SINGLE",
    f"at the end {rng(lp['WIDE_SPLIT'] + lp['REG3'])}; at 1200 the L2 probe reads "
    f"{rng(p1200['WIDE_SPLIT'] + p1200['REG3'])}", "PARTIAL",
    "true of the end only; at 1200 every learned run's residual carried the source (per-run max over L2/L3/final "
    + rng([max(r["curve"][0]["probe"][L]["acc"] for L in ("L2", "L3", "final")) for a in ("WIDE_SPLIT", "REG3")
           for r in R[a].values()]) + ")",
    "record")
bs = [r["beta_src"] for r in RG.values()]
bw = [r["beta_word"] for r in RG.values()]
row(RP("The register's write gate β is high at SRC tokens", 3), "beta high at SRC (median 0.80, range 0.44-0.90), low at words (median 0.15)",
    f"src median {statistics.median(bs):.4f}, range {min(bs):.4f}-{max(bs):.4f}; word median {statistics.median(bw):.4f}",
    "MATCH", "record beta_src / beta_word", "record")
row(RP("*Counts.* 30 fires, 3 per run"), "30 fires, 3 per run, at 7200, 12,000 and 16,800 (seed 830: 9600, 14,400, 19,200)",
    f"{len(facc)} fires; {len(nine)}/10 at 7200/12000/16800; 830 {first3[830][0]}", "MATCH", "", "record")
row(RP("Every fire was on a map that was not one-to-one", 2), "every fire on a map that was not one-to-one, on_target true, at probe SET 0.518-0.542",
    f"one-to-one map_before {f121}/30; on_target {sum(c['on_target'] for v in fires.values() for c in v)}/30; {rng(facc)}",
    "MATCH", "", "record")
row(RP("Fires on a one-to-one map: 0."), "fires on a one-to-one map 0; runs ROUTED before a fire and not at the end 0", f"{f121}; {len(rbf)}",
    "MATCH", "", "record")
nxt = [len(set(nx["map_before"])) == 4 for s, r in WS.items() for i, c in enumerate(r["checks"]) if c["fired"]
       for nx in r["checks"][i + 1:i + 2]]
row(RP("by the next check the map had merged again"), "by the next check the map had merged again", f"one-to-one at the check after a fire: {sum(nxt)}/{len(nxt)}",
    "MATCH", "checks[i+1].map_before (the last fire of each run has a following check)", "record")
row(RP("design commit 1e31409 has parent 5bc6c2e", 2), "S75b design commit 1e31409 has parent 5bc6c2e, which holds only the ORACLE and SINGLE records",
    f"parent {parent('1e31409')}; records at 5bc6c2e: {arms5}", "MATCH", "", "git")

# report_2.md as changed at 559b942 (table row, section 6) and its line 22, which 559b942 left
wl_end = [r["end"]["set"] for r in WL.values()]
lo9, hi9 = wilson(0, len(WL))
row(RP("| WIDE_SLOW (no-split twin, printed) |"), "table row (559b942): WIDE_SLOW 830-838 (839 pending); 0/9 [0.00, 0.30] "
    "NEVER; median SET 0.534 (0.530-0.536); eta^2 at the end 0.00; no split",
    f"seeds {sorted(WL)}; ROUTED {sum(routed_at(r['end']) for r in WL.values())}/{len(WL)}, Wilson [{lo9:.2f}, {hi9:.2f}] "
    f"{band(0, len(WL))}; median {statistics.median(wl_end):.4f} ({rng(wl_end, 4)}); end eta max "
    f"{max(v['eta'] for r in WL.values() for v in r['end']['routing'].values()):.5f}; splits "
    f"{sorted({r.get('splits') for r in WL.values()})}", "MATCH", "", "record")
row(RP("**WIDE_SLOW** is running on the 1660 now"), "WIDE_SLOW is running on the 1660 now",
    f"{len(WL)}/10 WIDE_SLOW records at {HEAD} (since 4c3984a); 839 pending", "PARTIAL",
    "stale since 4c3984a; 559b942 updated the table row and section 6 but left this line (section 6 says 839 is pending)",
    "record + git")
row(RP("**WIDE_SLOW, the no-split twin: seeds 830–838**", 2), "section 6: seeds 830-838 (4c3984a; seed 839 is pending)",
    f"WIDE_SLOW seeds at {HEAD} {sorted(WL)}; results file last changed at {REC}", "MATCH", "", "record + git")
row(RP("*ROUTED.* 0/9 at the end, and never at any of the 20 evaluations"),
    "ROUTED 0/9 at the end, and never at any of the 20 evaluations of any run",
    f"end {sum(routed_at(r['end']) for r in WL.values())}/9; evaluations ROUTED "
    f"{sum(routed_at(e) for r in WL.values() for e in r['curve'])} of {nev['WIDE_SLOW']} "
    f"({sorted({len(r['curve']) for r in WL.values()})} per run)", "MATCH", "", "record")
wl_end_sp = max(spread(v["src_gate"]) for r in WL.values() for v in r["end"]["routing"].values())
wl_any_sp = max(spread(v["src_gate"]) for r in WL.values() for e in r["curve"] for v in e["routing"].values())
row(RP("table differ by at most 0.02, the margin is 0.01–0.03"),
    "eta^2 0.00; the rows of the source x channel table differ by at most 0.02",
    f"eta max any evaluation {eta_max['WIDE_SLOW'][0]:.5f}; row spread at the end max {wl_end_sp:.4f}, at any "
    f"evaluation {wl_any_sp:.4f}", "PARTIAL" if wl_end_sp > 0.02 or wl_any_sp > 0.02 else "MATCH",
    "holds at the end after rounding to two decimals only (0.0219); 0.0297 at some evaluation; the same pattern as "
    "section 1's WIDE_SPLIT statement (row above)", "record")
wl_m_end = [v["margin"] for r in WL.values() for v in r["end"]["routing"].values()]
wl_m_any = [v["margin"] for r in WL.values() for e in r["curve"] for v in e["routing"].values()]
row(RP("table differ by at most 0.02, the margin is 0.01–0.03"), "the margin is 0.01-0.03",
    f"end margins {rng(wl_m_end, 4)}; any evaluation {rng(wl_m_any, 4)}",
    "MATCH" if f"{min(wl_m_end):.2f}-{max(wl_m_end):.2f}" == "0.01-0.03" else "PARTIAL",
    "at the end (as section 1's 'median margin'); over every evaluation the range is wider", "record")
wl_pe = [r["end"]["probe"][L]["acc"] for r in WL.values() for L in r["end"]["probe"]]
wl_p36 = [e["probe"][L]["acc"] for r in WL.values() for e in r["curve"] if e["step"] >= 3600 for L in e["probe"]]
row(RP("*SET and probe.* SET is 0.530–0.536"), "SET is 0.530-0.536, SINGLE's level, and the source probe is at chance "
    "(0.24-0.26) at every layer",
    f"SET {rng(wl_end, 4)}; probe at the end {rng(wl_pe, 4)}; from 3600 {rng(wl_p36, 4)}; at 1200 (L2) "
    f"{rng(p1200['WIDE_SLOW'], 4)}", "PARTIAL",
    "true at the end (and from 3600); at 1200 every WIDE_SLOW run's residual carried the source, as in the other gated "
    "arms (tex6:458 says so) -- the same omission as report_2's section 3 (row above)", "record")
adiff = [abs(v) for v in diffs.values()]
row(RP("Per seed, SET differs by at most 0.003"), "paired by seed with WIDE_SPLIT: neither arm ROUTED on any of the 9 "
    "shared seeds; per seed SET differs by at most 0.003 (median absolute difference 0.0006)",
    f"ROUTED pairs {sorted(set(pair_r))}; |difference| max {max(adiff):.5f} (seed 831), median "
    f"{statistics.median(adiff):.5f}", "PARTIAL" if max(adiff) > 0.003 else "MATCH",
    "holds after rounding only: seed 831 differs by 0.00325 (0.53274 vs 0.53598, 0.533 vs 0.536 at three decimals); "
    "the median reproduces", "record")
row(RP("*As a description of these runs:*"), "as a description: the window gate stayed source-blind with the split and without it; the 30 "
    "fires neither helped nor harmed routing that never formed",
    f"ROUTED at any evaluation: WIDE_SPLIT {sum(routed_at(e) for r in WS.values() for e in r['curve'])}, WIDE_SLOW "
    f"{sum(routed_at(e) for r in WL.values() for e in r['curve'])}", "MATCH", "descriptive, labelled so", "record")
README = "explore_out/README.md"
rd = show(HEAD, README).splitlines()
rl = next(i + 1 for i, x in enumerate(rd) if "Three WIDE_SPLIT runs had already finished" in x)
row(f"{README}:{rl}", "Correction to the pre-run note: three WIDE_SPLIT runs had already finished on the 1660 when it "
    "was committed", "no per-run time in the records (timeline section)", "NOT RECORD-CHECKABLE",
    f"session I's index at {HEAD} (unchanged by 559b942) still states report_2's reconstruction as fact; the draft "
    "(tex6:458) no longer does", "session I")

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
row(RP("From each record's duration and the results file's start time", 3), "from the durations and the start time (18:04 on 10-11), WIDE_SPLIT 830-832 ran 23:11 (10-11) to "
    "00:44 (10-12); the note was committed at 00:46 UTC", f"under one uninterrupted invocation: WIDE_SPLIT 830 starts "
    f"{end[('WIDE_SPLIT', '830')][0]:%H:%M:%S}, 832 ends {w832[1]:%H:%M:%S}; 5bc6c2e {note_t:%H:%M:%S}", "PARTIAL",
    "the arithmetic reproduces (record secs, meta.started); the premise of one uninterrupted invocation is not in the "
    "records and is contradicted by 86652b1's per-GPU summary (see the timeline section)", "record + git")
row(RP("The other seven WIDE_SPLIT runs and all ten REG3 runs came after it"), "the other seven WIDE_SPLIT runs and all ten REG3 runs came after it",
    f"under the same premises WIDE_SPLIT 833 started {w833[0]:%H:%M:%S}, "
    f"{(note_t - w833[0]).total_seconds():.1f} s before the note's commit", "DISAGREE", "by report_2's own reconstruction a fourth run (833) was in progress at 00:46:28; the "
    "statement should read 'three had finished and a fourth had started'", "record + git")
row(RP("Its timestamps can only be UTC", 2), "the 1660's timestamps can only be UTC (ORACLE and SINGLE committed 23:42 UTC, so they ended before)",
    f"SINGLE 839 ends {end[('SINGLE', '839')][1]:%H:%M:%S} machine time; 86652b1 authored {git_time('86652b1', '%aI')}",
    "PARTIAL", "the commit bounds the clock only from one side: machine time - UTC >= "
    f"{(end[('SINGLE', '839')][1] - dt.datetime(2026, 10, 11, 23, 42, 23)).total_seconds() / 60:.0f} min; a clock "
    "ahead of UTC is not excluded (UTC is plausible: meta.started 18:04:26 is 31 s before 203a5ad's 18:04:57 UTC)",
    "record + git")
row(RP("No learned-arm record was available anywhere I could read until b649751"), "no learned-arm record was available until b649751 (13:53 UTC on 10-12)",
    f"first commit with a learned-arm record: {first_learned[0]} {git_time(first_learned[0])}", "MATCH", "", "git")
row(RP("The pre-run note assigned WIDE_SPLIT and REG3 to the RTX 3070", 2), "deviation: the note assigned WIDE_SPLIT and REG3 to the RTX 3070; the 1660 ran them instead",
    "not in the records", "NOT RECORD-CHECKABLE", "inconsistent with l.29-31: runs that started at 23:11, before the "
    "note, were not run 'instead' of the note's assignment; one of the two accounts is wrong", "report_2")

# ── descriptions (regenerated at 559b942; read at HEAD) ─────────────────────────────────────────────────────────────
DES = "explore_out/I/gpu/s75_S4_descriptions.md"
dtext = show(HEAD, DES).splitlines()
des_commits = subprocess.run(["git", "log", "--format=%h", HEAD, "--", DES], capture_output=True, text=True).stdout.split()


def DL(frag, last=None):
    """descriptions line(s): the line of `frag` (asserted unique), or a span to the line of `last`."""
    a = [i + 1 for i, x in enumerate(dtext) if frag in x]
    assert len(a) == 1, (frag, a)
    if last is None:
        return f"{DES}:{a[0]}"
    b = [i + 1 for i, x in enumerate(dtext) if last in x and i + 1 > a[0]]
    return f"{DES}:{a[0]}-{b[0]}"


mk = lambda e: "R" if routed_at(e) else ("o" if all(v["one_to_one"] for v in e["routing"].values()) else ".")
n_ok = n_bad = 0
bad = []
arm = None
per_arm = {}
for i, line in enumerate(dtext, 1):
    h = re.match(r"^\*\*(WIDE_SPLIT|REG3|WIDE_SLOW)\*\* \((\d+)/(\d+) ROUTED at the end\)$", line)
    if h:
        arm = h.group(1)
        hdr_ok = int(h.group(2)) == sum(routed_at(r["end"]) for r in R[arm].values()) and int(h.group(3)) == len(R[arm])
        per_arm[arm] = [0, hdr_ok]
        continue
    if line.startswith("## 2."):
        arm = None
    m = re.match(r"^    (\d{3})   (\S+)\s+(\S)\s+(\S+)$", line)
    if not m or arm is None:
        continue
    r = R[arm][int(m.group(1))]
    exp = ("".join(mk(e) for e in r["curve"]), "R" if routed_at(r["end"]) else "-",
           "/".join(f"{v['eta']:.2f}" for v in r["end"]["routing"].values()))
    if exp == (m.group(2), m.group(3), m.group(4)):
        n_ok += 1
        per_arm[arm][0] += 1
    else:
        n_bad += 1
        bad.append((i, arm, exp, m.groups()))
row(DL("**WIDE_SPLIT** (0/10 ROUTED at the end)") + "-" + str(next(i + 1 for i, x in enumerate(dtext) if x.startswith("## 2.")) - 2),
    "section 1: per-run ROUTED trajectories (WIDE_SPLIT 10, REG3 10, WIDE_SLOW 9 runs x 20 evaluations), end mark, end "
    "eta^2 and each block's 'k/n ROUTED at the end'",
    f"{n_ok} lines reproduce, {n_bad} differ {bad}; per block (lines, header reproduces): {per_arm}",
    "MATCH" if n_bad == 0 and n_ok == 29 and all(v[1] for v in per_arm.values()) else "DISAGREE",
    f"regenerated at 559b942 (file history to {HEAD}: {des_commits}); the WIDE_SLOW block is new", "record")
row(DL("**WIDE_SLOW** (0/9 ROUTED at the end)"), "WIDE_SLOW: 0/9 ROUTED at the end [was 'no records yet' at 03bffdf]",
    f"{len(WL)} WIDE_SLOW records at {HEAD}, ROUTED {sum(routed_at(r['end']) for r in WL.values())}", "MATCH",
    "check_S75's stale-WIDE_SLOW PARTIAL, resolved by 559b942", "record + git")
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
sec2 = [i + 1 for i, x in enumerate(dtext) if re.match(r"^\| \d{3} \| \d+ \| ", x)]
row(f"{DES}:{sec2[0]}-{sec2[-1]}", "section 2: every eligible check of the split trigger (seed, step, probe SET (prev), "
    "map before, one-to-one, on_target, fired / blocked (gap / cap), routed@step / next / end)",
    f"{n_ok} rows reproduce of {n_elig} eligible checks in the records; differing {bad}",
    "MATCH" if not bad and n_ok == n_elig else "DISAGREE",
    "status recomputed independently (cap: 3 earlier fires; gap: < 4800 since the last) and compared with blocked_by",
    "record")
row(DL("- fires: 30 over 10 WIDE_SPLIT runs", "- runs with at least one fire"),
    "fires 30 over 10 runs; on a one-to-one map 0; routed before a fire and not at the end 0 []; "
    "runs with at least one fire 10/10", f"{len(facc)}; {f121}; {len(rbf)}; {sum(bool(v) for v in fires.values())}/10",
    "MATCH", "", "record")
n_ok = n_slow = 0
bad = []
pend = []
s4 = []
for i, line in enumerate(dtext, 1):
    m = re.match(r"^\| (\d{3}) \| (\S) \| (\d) \| (\S+) \| (\S+) \| (\S+) \| (\S+) \| (\S+) \|$", line)
    if not m:
        continue
    s4.append(i)
    sd = int(m.group(1))
    r = WS[sd]
    exp = ("R" if routed_at(r["end"]) else "-", str(r["splits"]), f"{r['end']['routing']['L1']['eta']:.2f}",
           f"{r['end']['set']:.3f}")
    if sd in WL:
        q = WL[sd]
        exp += ("R" if routed_at(q["end"]) else "-", f"{q['end']['routing']['L1']['eta']:.2f}", f"{q['end']['set']:.3f}")
    else:
        exp += ("pending", "-", "-")
        pend.append(sd)
    if exp == m.groups()[1:]:
        n_ok += 1
        n_slow += sd in WL
    else:
        bad.append((i, exp, m.groups()))
row(f"{DES}:{s4[0]}-{s4[-1]}", "section 4: WIDE_SPLIT's columns (routed@end, fires, eta^2, SET) and WIDE_SLOW's "
    "(routed@end, eta^2, SET) per seed; 839 pending",
    f"{n_ok}/10 rows reproduce ({n_slow} with WIDE_SLOW values, pending {pend}); differing {bad}",
    "MATCH" if n_ok == 10 and not bad and pend == [839] else "DISAGREE",
    "WIDE_SLOW's three columns new at 559b942; SET at three decimals, so 831 reads 0.533 vs 0.536 (difference 0.00325)",
    "record")
pairs9 = sum(1 for s in WL)
nb = sum(routed_at(WS[s]["end"]) and routed_at(WL[s]["end"]) for s in WL)
row(DL("Paired seeds with both records:"), "paired seeds with both records 9; ROUTED in both 0, WIDE_SPLIT only 0, "
    "WIDE_SLOW only 0, neither 9; pending for 1 of 10 seeds",
    f"{pairs9} pairs; both {nb}, split only {sum(routed_at(WS[s]['end']) and not routed_at(WL[s]['end']) for s in WL)}, "
    f"slow only {sum(routed_at(WL[s]['end']) and not routed_at(WS[s]['end']) for s in WL)}, neither "
    f"{sum(not routed_at(WS[s]['end']) and not routed_at(WL[s]['end']) for s in WL)}; pending {10 - len(WL)}", "MATCH",
    "", "record")
row(DL("Generated by `python -m explore_gpu.s75_writeup`"), "generated under code SHA 5fb120a92d69; the registered "
    "result is UNTESTED", f"{ST['meta']['sha']}; {tag}", "MATCH", "", "record")

# ── print (the whole of check_S75_v6.md) ─────────────────────────────────────────────────────────────────────────────
from collections import Counter  # noqa: E402

esc = lambda x: str(x).replace("|", "\\|")
cnt = Counter(r[3] for r in ROWS)
KS = ("MATCH", "PARTIAL", "DISAGREE", "NOT RECORD-CHECKABLE")
is_draft = lambda loc: loc.startswith("tex6:") or loc.startswith("caller") or loc.startswith("rule / provenance tex6")
cd_ = Counter(r[3] for r in ROWS if is_draft(r[0]))
ci_ = Counter(r[3] for r in ROWS if not is_draft(r[0]))
S75_LINES = sorted({int(x.split(":")[1]) for r in ROWS for x in r[0].replace(",", " ").split() if x.startswith("tex6:")})
print("# Check (v6): S75 at four sources (session I, GTX 1660 SUPER) against the sixth Revision 9 draft and session I's "
      "write-up")
print()
print("Referee V. Generated in full by `docs/review_2026-10/rev9/check_S75_v6.py` (`python3 -I docs/review_2026-10/rev9/"
      "check_S75_v6.py > docs/review_2026-10/rev9/check_S75_v6.md` from the repository root after `git fetch`). It reads "
      "every file with `git show`, runs no session code and trains nothing. It is the v6 copy of `check_S75.py`: every "
      "record recomputation is kept, every draft row is anchored by a verbatim fragment located in tex6, and rows were "
      "added for every changed sentence in its scope.")
print()
print(f"**Pins.** Draft: `claude/rev9-draft` at `{DRAFT}` ({git_time(DRAFT)}), `{TEX}` (= tex6, {len(tex)} lines; "
      f"identical in length to `{OLD_DRAFT}` = tex5, the changes being in-place line replacements). Lines changed "
      f"tex5 -> tex6: {CHANGED}; of the lines this check reads ({S75_LINES}), changed: "
      f"{[n for n in S75_LINES if n in CHANGED]}. Session I: `{BR}` at `{HEAD}` ({git_time(HEAD)}), its head when "
      f"`{DRAFT}` was committed (current tip `{TIP}`, {git_time(TIP)}). The results file "
      f"`{RES}` is byte-identical at `{REC}` and `{HEAD}` ({len(ST['runs'])} records; WIDE_SLOW seeds {sorted(WL)}); at "
      f"the tip it holds {len(ST_TIP['runs'])} records and WIDE_SLOW "
      f"{sorted(int(k.split('|')[1]) for k in ST_TIP['runs'] if k.startswith('WIDE_SLOW'))}. "
      f"report_2.md and s75_S4_descriptions.md are read at `{HEAD}` (last changed `559b942`); the pre-run note at "
      f"`5bc6c2e` (unchanged); S74's records at `a2588da` (unchanged).")
print()
print(f"**Branch status at {DRAFT}'s time (23:14:04 UTC).** Session I's tenth WIDE_SLOW seed (839) **has not landed**: "
      f"the S75 results file last changed at `{REC}` ({git_time(REC)}), and neither `559b942` (report 2 section 6, "
      "'seed 839 pending'), `30807ec` (window_gate pre-run note for the 1660), `8cb8dce` / `9034b83` (S75b driver and "
      f"CHECKs) nor the later `{TIP}` adds it. So the WIDE_SLOW-vs-WIDE_SPLIT pairing stays at nine seeds: ROUTED "
      f"neither on all nine, |SET difference| max {max(abs(v) for v in diffs.values()):.5f} (831), median "
      f"{statistics.median(abs(v) for v in diffs.values()):.5f}; the end set, eta^2 and one_to_one recomputations for "
      "839 cannot be made.")
print()
print("**Conventions.** As check_S75: MATCH reproduces at the stated precision; a stated bound ('at most', 'within', "
      "'no higher than') is MATCH only if the unrounded value satisfies it, PARTIAL if only after rounding; chance for "
      f"the source probe is the ORACLE band {rng(o_probe, 4)}, 'above chance' > 0.27; eta^2 = routing[L].eta.")
print()
print("## Counts")
print()
print(f"{len(ROWS)} rows: **" + ", ".join(f"{k} {cnt[k]}" for k in KS) + "**. Draft rows (tex6 and the caller's "
      f"paraphrases): {sum(cd_.values())} (" + ", ".join(f"{k} {cd_[k]}" for k in KS) + "); session I's files and the "
      f"S75 rule: {sum(ci_.values())} (" + ", ".join(f"{k} {ci_[k]}" for k in KS) + "). check_S75 (tex5, 4c3984a) had "
      "91 rows: MATCH 67, PARTIAL 16, DISAGREE 1, NOT RECORD-CHECKABLE 7.")
print()
print("### The fifth pass's S75 items at tex6")
print()
FIFTH = [("SINGLE's rise 'at most 0.03' (0.0329)", "at most 0.033 after 4800"),
         ("'had lost it by 3600' (REG3 839 0.291)", "all but one (a register run"),
         ("'within 0.02' (0.0235)", "within 0.024"),
         ("REG3 '0.535-0.540' (nine seeds; 839 0.588)", "register gate ended 0.535-0.540 on nine"),
         ("'late rise of about 0.06 ... 0.29 and 0.28'", "a late rise (0.09 and 0.08"),
         ("'late rise ... took the probe to 0.29 and 0.28' (the peak 0.340)", "took the probe no higher than 0.34"),
         ("the 'three window-gate runs' parenthesis", "(whether learned-arm runs had started"),
         ("next step tex5:478 ('the window gate without the split ... first')", "Next: the dense-loss task"),
         ("I's provenance row (4c3984a; 32 runs)", "provenance: records 86652b1, b649751, 4c3984a")]
print("| fifth-pass item | tex6 | row | now |")
print("|---|---|---|---|")
for item, key in FIFTH:
    k = next(n for n, r in enumerate(ROWS, 1) if key in r[1])
    print(f"| {esc(item)} | {ROWS[k - 1][0]} | {k} | **{ROWS[k - 1][3]}** |")
print()
print("### Every non-MATCH row")
print()
print("| # | where | statement | recomputed | status |")
print("|---|---|---|---|---|")
for i, (loc, stmt, rec, stt, note) in enumerate(ROWS, 1):
    if stt != "MATCH":
        print(f"| {i} | {esc(loc)} | {esc(stmt)} | {esc(rec)} | **{stt}** |")
print()
print("## Rows")
print()
print("| # | line or file:line | statement | recomputed | status | note |")
print("|---|---|---|---|---|---|")
for i, (loc, stmt, rec, stt, note) in enumerate(ROWS, 1):
    print(f"| {i} | {esc(loc)} | {esc(stmt)} | {esc(rec)} | {stt} | {esc(note)} |")
print()
print("## Timeline (unchanged from check_S75; printed for the tex6:458 parenthesis)")
print()
print("```")
print("machine clock of meta.started, ONE UNINTERRUPTED INVOCATION ASSUMED (a premise git contradicts); record secs:")
print(f"  record order is the arms' order: {order_ok}")
for a, s, t0, t1, secs in TL:
    flag = " <- straddles 5bc6c2e" if t0 < note_t < t1 else ""
    print(f"  {a:<10} {s} {t0:%m-%d %H:%M:%S} -> {t1:%m-%d %H:%M:%S} ({secs:7.1f} s){flag}")
sum2 = sum(r["secs"] for a in ("WIDE_SPLIT", "REG3") for r in R[a].values())
bt = dt.datetime.fromisoformat(git_time("b649751")).replace(tzinfo=None)
print(f"  WIDE_SPLIT + REG3 wall time {sum2:.0f} s = {sum2 / 3600:.2f} h; b649751 at {bt}; latest possible start of "
      f"WIDE_SPLIT 830 if contiguous to REG3 839: {bt - dt.timedelta(seconds=sum2):%m-%d %H:%M:%S}")
sumw = sum(r["secs"] for r in WL.values())
ht = dt.datetime.fromisoformat(git_time(REC)).replace(tzinfo=None)
hh = dt.datetime.fromisoformat(git_time(HEAD)).replace(tzinfo=None)
dd = dt.datetime.fromisoformat(git_time(DRAFT)).replace(tzinfo=None)
print(f"  WIDE_SLOW 830-838 wall time {sumw:.0f} s; {REC} at {ht}; latest start {ht - dt.timedelta(seconds=sumw):%H:%M:%S}; "
      f"the uninterrupted reconstruction would end 838 at {TL[-1][3]:%m-%d %H:%M:%S} and 839 by "
      f"{TL[-1][3] + dt.timedelta(seconds=1860):%H:%M:%S}, yet 839 was pending at {ht:%H:%M}, still at {HEAD} "
      f"({hh:%H:%M}) and at {DRAFT} ({dd:%H:%M}): WIDE_SLOW ran in a later invocation, so gaps between invocations are "
      "real and unrecorded")
print(f"  meta.task_check.secs at 86652b1 / b649751 / {REC}: " + " / ".join(
    str(json.loads(show(c, RES))["meta"]["task_check"]["secs"]) for c in ("86652b1", "b649751", REC)))
s86 = show("86652b1", GPU + "s75_S4_NVIDIA_GeForce_GTX_1660_SUPER_summary.md")
print("  86652b1 per-GPU summary rows: " + "; ".join(l for l in s86.splitlines() if l.startswith("| WIDE_SPLIT")
                                                       or l.startswith("| REG3") or l.startswith("READING")))
print("CHANCE BAND (ORACLE probe, all layers and evaluations): " + rng(o_probe, 4))
print("```")
print()
print("Under one uninterrupted invocation and a UTC clock, WIDE_SPLIT 832 ends 98 s before `5bc6c2e` and 833 is "
      "running; without that premise the records allow 0-3 finished WIDE_SPLIT runs at the note's commit. tex6:458's "
      "'whether learned-arm runs had started on the card by then is not recorded' is what the records support.")
print()
print("## What could not be checked")
print()
print("- WIDE_SLOW seed 839: not recorded on any ref at 42c4557's time (nor at the current tip); 'queued' (tex6:458) and "
      "'pending' (session I) are not record-checkable.")
print("- Per-run start and end times of S75 (none in the records); the 1660's clock zone.")
print("- The bounds 0.927 / 0.964 / 0.963, the faded learner 0.680 / 0.840 and the per-bin bounds: session I's note, "
      "not records.")
print("- Regenerating session I's summary (session code not run).")
