"""Referee V: numbers for keyload_options_review.md (the five options in the status section of
explore_out/spec_draft_keyload.md at claude/outside-ideas 419368a).

Stdlib only. Reads git objects with `git show` and nothing else; trains nothing.
Run from the repository root:  python3 docs/review_2026-10/keyload_options_checks.py
"""
import json
import math
import subprocess
from math import comb

REF21 = "419368a"   # batch 21 complete (oracle_d128_results.json, the draft's status section)
REF19 = "e3f5809"   # batch 19 complete (capacity_results.json, S67)


def show(ref, path):
    return subprocess.run(["git", "show", f"{ref}:{path}"], check=True, capture_output=True,
                          text=True).stdout


def binom_sf(k, n, p):
    """P(X >= k), X ~ Bin(n, p)."""
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def wilson(x, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 1.0)
    ph = x / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def beta_inc_inv(a, b, q):
    """Inverse regularised incomplete beta by bisection (for Clopper-Pearson)."""
    def I(x):  # regularised incomplete beta via binomial sum for integer a, b
        n = a + b - 1
        return sum(comb(n, j) * x ** j * (1 - x) ** (n - j) for j in range(a, n + 1))
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if I(mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper(x, n, alpha=0.05):
    lo = 0.0 if x == 0 else beta_inc_inv(x, n - x + 1, alpha / 2)
    hi = 1.0 if x == n else beta_inc_inv(x + 1, n - x, 1 - alpha / 2)
    return lo, hi


def beta_moment(a, b, k):
    """E[p^k] for p ~ Beta(a, b)."""
    r = 1.0
    for i in range(k):
        r *= (a + i) / (a + b + i)
    return r


def fisher_two_sided(a, b, c, d):
    """Fisher exact two-sided p for [[a, b], [c, d]] (sum of tables no more likely than observed)."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    def pr(x):
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)
    p0 = pr(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 * (1 + 1e-9))


def kofn(k, n, p):
    return binom_sf(k, n, p)


def sec(t):
    print()
    print("=" * 100)
    print(t)
    print("=" * 100)


# ── §1 Recount batch 21 from its record ──────────────────────────────────────────────────────────
sec("§1 S69 (batch 21) recount from explore_out/oracle_d128_results.json at " + REF21)
o = json.loads(show(REF21, "explore_out/oracle_d128_results.json"))
runs = o["runs"]
by = {}
for key, r in runs.items():
    arm, seed = key.split("|")[:2]
    by.setdefault(arm, {})[int(seed)] = r
for arm in sorted(by):
    rs = by[arm]
    bound = sorted(s for s, r in rs.items() if r.get("ok") and r.get("transition") is not None)
    n = len(rs)
    w = wilson(len(bound), n)
    cp = clopper(len(bound), n)
    print(f"{arm}: n {n}, bound {len(bound)} {bound}, transitions "
          f"{sorted(rs[s]['transition'] for s in bound)}; Wilson [{w[0]:.3f}, {w[1]:.3f}], "
          f"Clopper-Pearson [{cp[0]:.3f}, {cp[1]:.3f}]; iters {sorted(set(r['iters'] for r in rs.values()))}")
    for s in sorted(rs):
        r = rs[s]
        c = r["curve"]
        mx = max(a for _, a, *rest in c)
        tail = [round(a, 3) for st, a, *rest in c if st >= c[-1][0] - 9600]
        print(f"   {s}: {r['outcome']:<10} trans {r['transition']}  final {r['acc']:.3f}  max {mx:.3f}  "
              f"last 9 evals {tail}  cpu {r['cpu'][-8:]}")
A = by["ORC_A_D128"]
bA = sum(1 for r in A.values() if r.get("transition") is not None)
close = sorted(s for s, r in A.items() if r.get("transition") is None and r["acc"] > 0.8)
stuck = sorted(s for s, r in A.items() if r.get("transition") is None and r["acc"] <= 0.8)
print(f"ORC_A_D128: bound {bA}/10; close (unbound, final > 0.8) {close}; stuck (unbound, final <= 0.8) {stuck}")

# S68: the D128 oracle on 270 to 86,400
s68 = json.loads(show(REF21, "explore_out/s67_continue_results.json"))
for key, r in s68["runs"].items():
    if "ORC" in key:
        c = r["curve"]
        print(f"S68 {key.split('|')[0]}|{key.split('|')[1]}: outcome {r['outcome']}, final step {c[-1][0]}, "
              f"acc {c[-1][1]:.3f}; max over run {max(a for _, a, *x in c):.3f}")

# ── §2 S67: oracle vs learned on the same seeds; the P = 8 and D32 oracles ───────────────────────
sec("§2 S67 (batch 19) from explore_out/capacity_results.json at " + REF19)
cap = json.loads(show(REF19, "explore_out/capacity_results.json"))["runs"]
cb = {}
for key, r in cap.items():
    arm, seed = key.split("|")[:2]
    cb.setdefault(arm, {})[int(seed)] = r
for arm in sorted(cb):
    if arm.startswith("ORC"):
        rs = cb[arm]
        print(f"{arm}: " + ", ".join(f"{s} {'BOUND at ' + str(r['transition']) if r['transition'] is not None else 'not bound (acc %.2f)' % r['acc']}"
                                     for s, r in sorted(rs.items())))
learned = {s: (r["outcome"] == "BOUND ROUTED") for s, r in cb["P16_D128"].items()}
orc = {s: (A[s].get("transition") is not None) for s in A}
both = sorted(s for s in orc if orc[s] and learned[s])
o_only = sorted(s for s in orc if orc[s] and not learned[s])
l_only = sorted(s for s in orc if not orc[s] and learned[s])
neither = sorted(s for s in orc if not orc[s] and not learned[s])
print(f"Seed by seed, S69 ORC_A_D128 (40,800) against S67 P16_D128 learned BOUND ROUTED (43,200):")
print(f"   both {both}; oracle only {o_only}; learned only {l_only}; neither {neither}")
p_f = fisher_two_sided(len(both), len(o_only), len(l_only), len(neither))
print(f"   Fisher exact two-sided p = {p_f:.3f}; learned rate given oracle bound "
      f"{len(both)}/{len(both)+len(o_only)}, given oracle unbound {len(l_only)}/{len(l_only)+len(neither)}")
print(f"   learned P16_D128 270 transition {cb['P16_D128'][270]['transition']} (its oracle: unbound at 40,800 here and at 86,400 in S68)")
print("   SLOW: oracle 0/10 (S69 ORC_S_D128) against learned 4/10 (S67 P16_D128) on the same seeds and recipe schedule")

# ── §3 Validity pass probabilities ───────────────────────────────────────────────────────────────
sec("§3 Validity pass probability: k of n validity seeds bind, rate p per seed, seeds independent")
wl, wh = wilson(4, 10)
cpl, cph = clopper(4, 10)
rates = [("Wilson lower", wl), ("CP lower", cpl), ("point 4/10", 0.4), ("Wilson upper", wh), ("CP upper", cph),
         ("0.7 (close seeds bind)", 0.7), ("0.9", 0.9), ("0.95", 0.95)]
print("Hypergeometric, two seeds drawn from S69's ten (the draft's model):")
print(f"   both bind: 4*3/90 = {4*3/90:.3f}   at least one binds: 1 - 6*5/90 = {1-6*5/90:.3f}")
print("Binomial (fresh seeds, as 660-661 / 1660-1661 are):")
print(f"{'rate':<26}" + "".join(f"{f'{k}of{n} mach':>11}{f'{k}of{n} both':>11}" for k, n in [(2, 2), (1, 2), (1, 3), (1, 4), (2, 4), (1, 6), (2, 6)]))
for name, p in rates:
    row = f"{name + f' {p:.3f}':<26}"
    for k, n in [(2, 2), (1, 2), (1, 3), (1, 4), (2, 4), (1, 6), (2, 6)]:
        m = kofn(k, n, p)
        row += f"{m:>11.3f}{m*m:>11.3f}"
    print(row)
print("Smallest n with P(>= 1 of n binds on both machines) >= 0.90 and >= 0.80:")
for name, p in rates[:5]:
    out = []
    for target in (0.90, 0.80):
        n = 1
        while kofn(1, n, p) ** 2 < target:
            n += 1
        out.append(f"{target}: n = {n}")
    print(f"   {name} {p:.3f}: " + "; ".join(out))

print("\nRecords with 2/2 (D32 oracle at P=16 in S67; D128 and D32 oracles at P=8 in S67): rate unknown")
w22 = wilson(2, 2)
cp22 = clopper(2, 2)
print(f"   2/2: Wilson [{w22[0]:.3f}, 1], Clopper-Pearson [{cp22[0]:.3f}, 1]")
for prior, (a, b) in [("uniform Beta(1,1)", (3, 1)), ("Jeffreys Beta(.5,.5)", (2.5, 0.5))]:
    print(f"   predictive under {prior}: P(next 2 both bind) = {beta_moment(a, b, 2):.3f}; "
          f"P(4 of 4: both machines, VALID_N 2) = {beta_moment(a, b, 4):.3f}")
for p in (w22[0], 0.8, 0.9, 0.95):
    print(f"   rate {p:.3f}: VALID_N 2 per machine {p**2:.3f}, both machines {p**4:.3f}")
print("Joint: pooled claims need D128 AND D32 VALID on both machines (draft l.113-115).")
for pd128_name, pd128 in [("D128 2of2 @0.4", 0.4 ** 2), ("D128 1of2 @0.4", kofn(1, 2, 0.4)),
                          ("D128 1of4 @0.4", kofn(1, 4, 0.4)), ("D128 1of6 @0.4", kofn(1, 6, 0.4)),
                          ("D128 none (option 4)", 1.0)]:
    row = f"   {pd128_name:<22}"
    for pd32 in (0.8, 0.9, 0.95, 1.0):
        row += f"  D32 2of2 @{pd32:.2f}: {(pd128 * pd32**2) ** 2:.3f}"
    print(row)

# ── §4 Screen operating characteristics ──────────────────────────────────────────────────────────
sec("§4 Screen rules: P(rule passes | true per-seed rate)")
rules = [("S69's 'reliable' >= 9/10", 9, 10), (">= 10/10", 10, 10), (">= 18/20", 18, 20), (">= 19/20", 19, 20)]
ps = [0.4, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
print(f"{'rule':<28}" + "".join(f"{p:>8}" for p in ps))
for name, k, n in rules:
    print(f"{name:<28}" + "".join(f"{binom_sf(k, n, p):>8.3f}" for p in ps))
for x, n in [(9, 10), (10, 10), (18, 20), (20, 20)]:
    w = wilson(x, n)
    cp = clopper(x, n)
    print(f"   {x}/{n}: Wilson [{w[0]:.3f}, {w[1]:.3f}], Clopper-Pearson [{cp[0]:.3f}, {cp[1]:.3f}]")
print("Selection over m configurations screened at the same true rate, the best one passing >= 9/10:")
for p in (0.7, 0.8):
    one = binom_sf(9, 10, p)
    print(f"   p = {p}: one config {one:.3f}; " + ", ".join(f"m={m}: {1-(1-one)**m:.3f}" for m in (2, 3, 4)))
print("Option 1 ceiling on S69's own seeds: 270 is unbound at 86,400 (S68), so at any budget <= 86,400 the ten seeds")
print("   give at most 9/10; 'reliable' (>= 9/10) on 270-279 needs 272 and 276 (flat at 0.29-0.30) and all of 275, 277, 279.")

# ── §5 The draft's claim thresholds (re-derived) ─────────────────────────────────────────────────
sec("§5 K1 / K2 thresholds and power as stated in the draft (l.139-145), re-derived")
for n, k in [(40, 13), (40, 12), (20, 8), (20, 7)]:
    print(f"   K2 n={n}, >= {k}: P(>= k | 0.2) = {binom_sf(k, n, 0.2):.4f}; power at 0.4 = {binom_sf(k, n, 0.4):.3f}")
print(f"   K1 with c = 0: b = 5 gives one-sided p = {0.5**5:.4f}; b = 4 gives {0.5**4:.4f}")
for pd128 in (0.2, 0.3, 0.4):
    for pd32 in (0.0, 0.1):
        # independent outcomes per arm; exact one-sided McNemar SHOWN if P(Bin(b+c, .5) >= b) < 0.05
        def shown(b, c):
            m = b + c
            return m > 0 and sum(comb(m, i) for i in range(b, m + 1)) / 2 ** m < 0.05
        pb = pd128 * (1 - pd32)
        pc = pd32 * (1 - pd128)
        n = 40
        tot = 0.0
        for b in range(n + 1):
            for c in range(n - b + 1):
                pr = math.factorial(n) / (math.factorial(b) * math.factorial(c) * math.factorial(n - b - c)) \
                    * pb ** b * pc ** c * (1 - pb - pc) ** (n - b - c)
                if shown(b, c):
                    tot += pr
        print(f"   K1 power, 40 pairs, D128 {pd128} vs D32 {pd32} (independent): {tot:.3f}")

# ── §6 Extra: a near-certain positive control; how many screen seeds make option 3's rate informative ─
sec("§6 Validity at a 10/10-screened check, and the D128 oracle rate's lower bound with more screen seeds")
w1010 = wilson(10, 10)[0]
cp1010 = clopper(10, 10)[0]
for name, p in [("10/10 Wilson lower", w1010), ("10/10 CP lower", cp1010), ("2/2 Wilson lower", wilson(2, 2)[0])]:
    print(f"   {name} {p:.3f}: 2 of 2 per machine {p**2:.3f}, both {p**4:.3f}; "
          f">= 1 of 2 per machine {kofn(1, 2, p):.3f}, both {kofn(1, 2, p)**2:.3f}")
print("If a fresh screen keeps the D128 oracle's observed rate at 0.4 (x of n), its Wilson lower bound, and the")
print("smallest number of validity seeds per machine for P(>= 1 binds on both machines) >= 0.90 at that bound:")
for x, n in [(4, 10), (8, 20), (12, 30), (16, 40)]:
    lo = wilson(x, n)[0]
    m = 1
    while kofn(1, m, lo) ** 2 < 0.90:
        m += 1
    print(f"   {x}/{n}: Wilson lower {lo:.3f}; validity seeds needed {m}")
