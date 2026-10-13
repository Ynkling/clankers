#!/usr/bin/env python3
"""
docs/review_2026-10/spec_checks_s75b.py -- Referee V's checks for explore_out/I/s75b_spec_draft.md (claude/explore-I).

Reads git objects only (git cat-file / show / merge-base / ls-tree / grep); edits, checks out and runs nothing else.
Optional: one read-only GitHub API call (the branch's push activity) through `gh`, skipped if gh fails.

    python3 docs/review_2026-10/spec_checks_s75b.py > docs/review_2026-10/spec_checks_s75b_out.txt

Sections
  A  provenance: parents, ancestry, times, trees, the records each commit holds
  B  numbers in each spec version that appear only in S75's learned-arm summaries (b649751), not in the allowed inputs
  C  S75's learned-arm values (b649751, 4c3984a) beside the spec's constants
  D  K9 / cost reconciliation
  E  ORACLE- and SINGLE-derived numbers re-derived from the records
  F  the pilot rule's operating characteristics (normal model, as the spec's own stop rates assume)
  G  the guard: S75's trigger rows replayed under the S75b precedence
  H  seeds 840-859 on every branch
"""

import json
import math
import re
import statistics as st
import subprocess
import sys

REPO = "/home/user/clankers"
REC = "explore_out/I/gpu/s75_S4_NVIDIA_GeForce_GTX_1660_SUPER_results.json"
SPEC = "explore_out/I/s75b_spec_draft.md"
C = dict(c86="86652b1", pre="5bc6c2e", lrn="b649751", design="1e31409", merge="142f47e", cost="ee97443",
         chk3070="13de308", merge2="15c5e4f", rev="114cd08", wup="03bffdf", slow="4c3984a")


def git(*a, check=True):
    r = subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"git {a}: {r.stderr}")
    return r


def show(ref, path):
    return git("show", f"{ref}:{path}").stdout


def records(ref):
    try:
        return json.loads(show(ref, REC))["runs"]
    except RuntimeError:
        return {}


def hdr(s):
    print()
    print("=" * 100)
    print(s)
    print("=" * 100)


def Phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ── A ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
hdr("A. Provenance")
print("commit   parents                         author date (UTC)          committer date (UTC)       author")
for k, c in C.items():
    o = git("cat-file", "-p", c).stdout
    parents = [l.split()[1][:7] for l in o.splitlines() if l.startswith("parent ")]
    ad = git("log", "-1", "--format=%ad", "--date=iso-strict-local", c, check=False)
    a = git("log", "-1", "--format=%aI|%cI|%an <%ae>", c).stdout.strip().split("|")
    to_utc = git("log", "-1", "--format=%ad|%cd", "--date=format-local:%Y-%m-%d %H:%M:%S", c).stdout.strip()
    print(f"{c}  {' '.join(parents):<30}  {a[0]:<26} {a[1]:<26} {a[2]}  (UTC: {to_utc})")

print("\nIs X an ancestor of Y?  (git merge-base --is-ancestor X Y)")
ys = ["design", "merge", "cost", "merge2", "rev", "wup", "slow"]
print(" " * 10 + "".join(f"{C[y]:>9}" for y in ys))
for x in ["c86", "pre", "lrn", "chk3070", "wup", "slow"]:
    row = []
    for y in ys:
        r = git("merge-base", "--is-ancestor", C[x], C[y], check=False).returncode
        row.append("yes" if r == 0 else "no")
    print(f"{C[x]:<10}" + "".join(f"{v:>9}" for v in row))

print("\nFiles 1e31409 changes against its parent 5bc6c2e (git diff --name-status):")
print(git("diff", "--name-status", C["pre"], C["design"]).stdout.rstrip())
print("\nS75 files in each tree (explore_out/I/gpu/s75_*), with blob ids:")
for k in ["pre", "lrn", "design", "merge", "rev", "slow"]:
    t = git("ls-tree", "-r", C[k], "explore_out/I/gpu/").stdout.splitlines()
    t = [l for l in t if "/s75_" in l]
    print(f"  {C[k]}:")
    for l in t:
        print("     ", l.split()[2][:10], l.split("\t")[1])

print("\nS75 records each commit holds (arm: seeds):")
for k in ["c86", "pre", "lrn", "design", "merge", "cost", "rev", "slow"]:
    R = records(C[k])
    by = {}
    for key in R:
        arm, seed = key.split("|")[:2]
        by.setdefault(arm, []).append(int(seed))
    desc = "; ".join(f"{a} {min(s)}-{max(s)} ({len(s)})" if len(s) > 2 else f"{a} {sorted(s)}"
                     for a, s in sorted(by.items()))
    print(f"  {C[k]}: {len(R)} records: {desc}")

print("\nRemote branches containing b649751:", git("branch", "-r", "--contains", C["lrn"]).stdout.split())
print("Remote branches containing 1e31409:", git("branch", "-r", "--contains", C["design"]).stdout.split())

print("\nPush activity of claude/explore-I (GitHub API, read-only; skipped on error):")
try:
    r = subprocess.run(["gh", "api", "-X", "GET", "repos/Ynkling/clankers/activity",
                        "-f", "ref=refs/heads/claude/explore-I", "-f", "per_page=100"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode:
        print("  unavailable:", (r.stderr or r.stdout).strip()[:300])
    else:
        acts = json.loads(r.stdout)
        for a in sorted(acts, key=lambda a: a["timestamp"]):
            if a["timestamp"] >= "2026-10-11T23" and a["timestamp"] <= "2026-10-12T20":
                print(f"  {a['timestamp']}  {a['activity_type']:<12} {a['before'][:7]} -> {a['after'][:7]}  "
                      f"by {(a.get('actor') or {}).get('login')}")
except Exception as e:  # noqa: BLE001
    print("  unavailable:", e)

# ── B ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
hdr("B. Numbers in the spec that occur in S75's learned-arm summaries but in none of the allowed inputs")
NUM = re.compile(r"(?<![\w.])(\d+\.\d+|\d{3,}(?:,\d{3})*)(?![\w])")


def nums(text):
    return {m.replace(",", "") for m in NUM.findall(text)}


allowed_txt = "".join([
    show(C["pre"], "explore_out/I/gpu/s75_S4_prerun_note_2026-10-12.md"),
    show(C["pre"], "explore_out/I/gpu/s75_S4_NVIDIA_GeForce_GTX_1660_SUPER_summary.md"),
    show(C["rev"], "explore_gpu/checks_cuda_NVIDIA_GeForce_GTX_1660_SUPER_output.txt"),
    show(C["rev"], "explore_gpu/checks_cuda_NVIDIA_GeForce_RTX_3070_Laptop_GPU_output.txt"),
    show(C["rev"], "explore_gpu/s75_s4.py"),
    show(C["rev"], "explore_i_common.py"),
])
learned_txt = (show(C["lrn"], "explore_out/I/gpu/s75_S4_summary.md")
               + show(C["lrn"], "explore_out/I/gpu/s75_S4_NVIDIA_GeForce_GTX_1660_SUPER_summary.md"))
learned_only = nums(learned_txt) - nums(allowed_txt)
print(f"tokens only in the learned-arm summaries: {len(learned_only)}: {sorted(learned_only, key=float)}")
print("(a check of literal numbers only: a number derived from the learned arms and then rounded or transformed is not caught)")
for ref in ["design", "cost", "rev"]:
    sp = show(C[ref], SPEC)
    hit = sorted(nums(sp) & learned_only, key=float)
    print(f"\n  {C[ref]} ({len(sp.splitlines())} lines): {len(hit)} hit(s)")
    for h in hit:
        for i, line in enumerate(sp.splitlines(), 1):
            if h in line.replace(",", ""):
                print(f"    {h:>8}  l.{i}: {line.strip()[:110]}")
        for line in learned_txt.splitlines():
            if h in line.replace(",", ""):
                print(f"    {'':>8}  (learned summary: {line.strip()[:100]})")
                break

# ── C ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
hdr("C. S75's learned-arm values (records at b649751; WIDE_SLOW at 4c3984a)")
R = records(C["slow"])


def arm(a):
    return {int(k.split("|")[1]): v for k, v in R.items() if k.startswith(a + "|") and v.get("ok")}


for a in ["ORACLE", "SINGLE", "WIDE_SPLIT", "REG3", "WIDE_SLOW"]:
    rr = arm(a)
    if not rr:
        continue
    sets = [r["end"]["set"] for r in rr.values()]
    spu = [r["train_secs"] / r["iters"] for r in rr.values()]
    pk = [r.get("peak_mem_gb_batch") for r in rr.values()]
    rt = [r["end"].get("routing") or {} for r in rr.values()]
    eta = [v["eta"] for x in rt for v in x.values()]
    mar = [v["margin"] for x in rt for v in x.values()]
    o2o_end = sum(all(v["one_to_one"] for v in x.values()) for x in rt if x)
    o2o_eval = sum(1 for r in rr.values() for e in r["curve"] for v in (e.get("routing") or {}).values()
                   if v["one_to_one"])
    n_eval = sum(1 for r in rr.values() for e in r["curve"] for v in (e.get("routing") or {}).values())
    print(f"{a:<11} n={len(rr):<2} SET {min(sets):.4f}-{max(sets):.4f} (median {st.median(sets):.4f}); "
          f"train s/update median {st.median(spu):.4f} ({min(spu):.4f}-{max(spu):.4f}); peak GB median {st.median(pk)}")
    if eta:
        print(f"{'':<11} eta_end max {max(eta):.4f}; margin_end {min(mar):.3f}-{max(mar):.3f}; one-to-one at end "
              f"{o2o_end}/{len(rr)}; one-to-one maps over all evaluations x layers {o2o_eval}/{n_eval}")
    if a == "REG3":
        b = [r["beta_src"] for r in rr.values()]
        w = [r["beta_word"] for r in rr.values()]
        print(f"{'':<11} beta_src median {st.median(b):.3f} ({min(b):.3f}-{max(b):.3f}); beta_word median {st.median(w):.3f}")
    if a == "WIDE_SPLIT":
        rows = [(s, c) for s, r in rr.items() for c in r["checks"]]
        fires = [c for _, c in rows if c["fired"]]
        print(f"{'':<11} check steps {sorted({c['step'] for _, c in rows})}")
        print(f"{'':<11} fires {len(fires)} at steps {sorted({c['step'] for c in fires})}; probe acc at fires "
              f"{min(c['acc'] for c in fires):.4f}-{max(c['acc'] for c in fires):.4f}; on_target {sum(c['on_target'] for c in fires)}")
        rises = [c["acc"] - c["prev"] for _, c in rows if c["prev"] is not None]
        print(f"{'':<11} rises between checks {min(rises):+.4f}..{max(rises):+.4f}")

# ── D ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
hdr("D. K9 and the cost section (114cd08 l.180-216)")
K9 = {}
for g in ["GTX_1660_SUPER", "RTX_3070_Laptop_GPU"]:
    K9[g] = json.loads(show(C["rev"], f"explore_gpu/checks_cuda_NVIDIA_GeForce_{g}_output.json"))["checks"]["K9"]["bench"]
print("arm            1660 s/upd  3070 s/upd  3070/1660   K9 peak GB  K9 reserved  3070 used-reserved")
for a in ["ORACLE", "SINGLE", "WIDE_SPLIT", "REG3"]:
    x, y = K9["GTX_1660_SUPER"][f"{a} R=1"], K9["RTX_3070_Laptop_GPU"][f"{a} R=1"]
    print(f"{a:<14} {x['secs_per_update']:.4f}      {y['secs_per_update']:.4f}      {y['secs_per_update'] / x['secs_per_update']:.3f}"
          f"       {x['peak_gb']:.3f}       {x['peak_reserved_gb']:.3f}        {y['device_used_gb'] - y['peak_reserved_gb']:.3f}")
print("R=2 per-run vs R=1 (1660):", {a: round(K9['GTX_1660_SUPER'][f'{a} R=2']['secs_per_run_update']
                                             / K9['GTX_1660_SUPER'][f'{a} R=1']['secs_per_update'], 3)
                                       for a in ["ORACLE", "SINGLE", "WIDE_SPLIT", "REG3"]})
spec1536 = {"ORACLE": 0.22, "SINGLE": 0.20, "WIDE_SPLIT": 0.24}
print("spec's 1660 s/update at L=1536 over K9 at L=768:",
      {a: round(v / K9["GTX_1660_SUPER"][f"{a} R=1"]["secs_per_update"], 2) for a, v in spec1536.items()},
      "; REG3 factor assumed 3.5-4 ->", [round(K9["GTX_1660_SUPER"]["REG3 R=1"]["secs_per_update"] * f, 3) for f in (3.5, 4)])
for T in (767, 1535):
    print(f"gate tensor 32 x 1 x {T} x {T} float32 = {32 * T * T * 4 / 2 ** 30:.3f} GiB")
print("peak in S75 records minus K9 update-only peak (1660), per arm:")
for a in ["ORACLE", "SINGLE", "WIDE_SPLIT", "REG3", "WIDE_SLOW"]:
    rr = arm(a)
    kk = K9["GTX_1660_SUPER"].get(f"{a if a != 'WIDE_SLOW' else 'WIDE_SPLIT'} R=1")
    pk = st.median([r["peak_mem_gb_batch"] for r in rr.values()])
    print(f"  {a:<11} record {pk:.3f} - K9 {kk['peak_gb']:.3f} = {pk - kk['peak_gb']:+.3f}"
          + ("   (learned arm)" if a in ("WIDE_SPLIT", "REG3", "WIDE_SLOW") else ""))
print("memory table 114cd08 - ee97443:", {"ORACLE": round(2.19 - 1.91, 2), "SINGLE": round(1.63 - 1.63, 2),
                                          "WIDE": round(3.33 - 3.05, 2), "REG3 scan": round(4.80 - 4.52, 2),
                                          "REG3 loop": round(3.96 - 3.68, 2)})
print("x1.15:", {k: round(v * 1.15, 2) for k, v in dict(ORACLE=2.19, SINGLE=1.63, WIDE=3.33, REG3s=4.80, REG3l=3.96).items()})
print("hours per 24k run at the spec's s/update:", {k: round(v * 24000 / 3600, 2) for k, v in
                                                    dict(ORACLE=0.22, SINGLE=0.20, WIDE=0.24, REG3s_lo=0.33, REG3s_hi=0.38,
                                                         REG3l_lo=0.55, REG3l_hi=0.8).items()})
h1660 = 2 * 0.22 * 24000 / 3600 + 10 * 0.20 * 24000 / 3600 + 10 * 0.24 * 24000 / 3600
print(f"stage 2 on the 1660 (ORACLE x2, SINGLE x10, WIDE_SLOW x10): {h1660:.1f} h (spec: about 32 h)")
h3070 = [10 * 0.24 * 24000 / 3600 * f for f in (0.77, 0.79)]
r3070 = [10 * 0.0740 * f * 24000 / 3600 for f in (3.5, 4)]
print(f"stage 2 on the 3070: WIDE_SPLIT_G {h3070[0]:.1f}-{h3070[1]:.1f} h + REG3 {r3070[0]:.1f}-{r3070[1]:.1f} h "
      f"= {h3070[0] + r3070[0]:.1f}-{h3070[1] + r3070[1]:.1f} h (spec: 12.5 + 17-20 = 30-32 h)")
allin = [h1660 + 2 * 0.22 * 24000 / 3600 + 10 * 0.24 * 24000 / 3600 + 10 * x * 24000 / 3600 for x in (0.55, 0.8)]
print(f"everything on the 1660 incl. pilot, REG3 loop: {allin[0]:.0f}-{allin[1]:.0f} h (spec: 85-101 h)")
print(f"K9 s75_train_hours R=1: 1660 {json.loads(show(C['rev'], 'explore_gpu/checks_cuda_NVIDIA_GeForce_GTX_1660_SUPER_output.json'))['checks']['K9']['s75_train_hours']['R=1']:.2f} h")

# ── E ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
hdr("E. ORACLE/SINGLE-derived numbers (records at 5bc6c2e)")
P = records(C["pre"])
orc = {int(k.split("|")[1]): v["end"] for k, v in P.items() if k.startswith("ORACLE|")}
sng = [v["end"]["set"] for k, v in P.items() if k.startswith("SINGLE|")]
for s, e in sorted(orc.items()):
    print(f"ORACLE {s}: set {e['set']:.5f}  bins 1/2/3-4/5+ {e['set_1']:.4f} {e['set_2']:.4f} {e['set_3-4']:.4f} {e['set_5+']:.4f}")
spread = orc[830]["set"] - orc[831]["set"]
print(f"ORACLE spread (unrounded) {spread:.5f}; 0.9 + spread = {0.9 + spread:.5f}; sd estimate spread/sqrt(2) = {spread / math.sqrt(2):.5f}")
lo, hi = spread / math.sqrt(2) * math.sqrt(1 / 5.0239), spread / math.sqrt(2) * math.sqrt(1 / 0.000982)
print(f"  95% interval for sigma from one pair (1 df): {lo:.4f} - {hi:.3f}")
om = (orc[830]["set"] + orc[831]["set"]) / 2
print(f"ORACLE mean {om:.5f}; SINGLE median {st.median(sng):.5f}; SINGLE median - 0.496 = {st.median(sng) - 0.496:+.4f}; "
      f"0.517 + that = {0.517 + st.median(sng) - 0.496:.3f}")
for fade, bnd in [(0.679, 0.927), (0.680, 0.927)]:
    share = (om - fade) / (bnd - fade)
    print(f"gap share with faded {fade}, bound {bnd}: {share:.4f} -> at B (faded 0.691): bound 0.964 -> "
          f"{0.691 + share * (0.964 - 0.691):.4f}; bound 0.963 -> {0.691 + share * (0.963 - 0.691):.4f}")
for s in (830, 831):
    print(f"ORACLE {s} - 0.03 = {orc[s]['set'] - 0.03:.4f} (S75's rule applied to S75's ORACLE)")

# ── F ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
hdr("F. Pilot proceed rule: P(stop) under a normal model, two independent pilot runs N(mu, sigma)")


def p_old(mu, sd):
    return Phi((mu - 0.912) / sd) ** 2


def p_new(mu, sd, n=200_000):
    # P(mean >= 0.912 and both >= 0.9): integrate over x1 numerically
    tot, a, b = 0.0, mu - 8 * sd, mu + 8 * sd
    h = (b - a) / 4000
    for i in range(4000):
        x = a + (i + 0.5) * h
        if x < 0.9:
            continue
        f = math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
        x2min = max(0.9, 2 * 0.912 - x)
        tot += f * (1 - Phi((x2min - mu) / sd)) * h
    return tot


def p_valid(mu, sd):
    return Phi((mu - 0.9) / sd) ** 2


print("mu      sigma   P(stop) old rule   P(stop) new rule   P(stage-2 VALID | mu)   SPLIT_THR median if proceed")
for mu in (0.900, 0.905, 0.910, 0.914, 0.916, 0.920, 0.922, 0.928):
    for sd in (0.004, 0.008, 0.012):
        print(f"{mu:.3f}   {sd:.3f}   {1 - p_old(mu, sd):.3f}              {1 - p_new(mu, sd):.3f}              "
              f"{p_valid(mu, sd):.3f}")
print("\nThe spec's figures (l.89-93): old 0.20 (per-bin, mu 0.922) / 0.64 (fixed reach, mu 0.914); new 0.05 / 0.40;"
      " VALID 0.92-0.99; sigma 0.008")
for mu, lab in ((0.922, "per-bin"), (0.914, "fixed reach")):
    print(f"  {lab:<12} mu {mu}: old {1 - p_old(mu, 0.008):.3f}  new {1 - p_new(mu, 0.008):.3f}  VALID {p_valid(mu, 0.008):.3f}")
print("  per-bin as two seed-specific means 0.928 / 0.916, sigma 0.008 per run:")
p1, p2 = Phi((0.928 - 0.912) / 0.008), Phi((0.916 - 0.912) / 0.008)
print(f"    old rule P(stop) = {1 - p1 * p2:.3f}")
print("\nSPLIT_THR given proceed: min(pilot) >= 0.9 is required, so SPLIT_THR >= 0.87; S75 used 0.88.")

# ── G ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
hdr("G. The guard: S75 WIDE_SPLIT trigger rows replayed under S75b's precedence (rules 1-4, 114cd08 l.121-127)")
cnt = dict(not_eligible=0, cap=0, gap=0, cap_and_gap=0, one_to_one_blocked=0, fire=0, one_to_one_any=0,
           eligible_o2o_but_capped=0, not_on_target_fire=0, rows=0)
changed = 0
for s, r in arm("WIDE_SPLIT").items():
    n, last, prev = 0, None, None
    for c in r["checks"]:
        cnt["rows"] += 1
        o2o = len(set(c["map_before"])) == 4
        cnt["one_to_one_any"] += o2o
        eligible = c["eligible"]
        cap = n >= 3
        gap = last is not None and c["step"] - last < 4800
        if not eligible:
            cnt["not_eligible"] += 1
            act = "none"
        elif cap or gap:
            cnt["cap_and_gap"] += cap and gap
            cnt["cap" if cap else "gap"] += 1
            cnt["eligible_o2o_but_capped"] += o2o
            act = "cap" if cap else "gap"
        elif o2o:
            cnt["one_to_one_blocked"] += 1
            act = "o2o"
        else:
            cnt["fire"] += 1
            cnt["not_on_target_fire"] += not c["on_target"]
            n, last = n + 1, c["step"]
            act = "fire"
        rec_act = "fire" if c["fired"] else (c.get("blocked_by") or "none")
        changed += act != rec_act
print(cnt)
print(f"rows whose action differs from S75's record under the guard: {changed}")

# ── H ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
hdr("H. Seeds 840-859 in records on any remote branch (key 'ARM|seed|' or '\"seed\": 84x/85x')")
brs = [b.strip() for b in git("branch", "-r").stdout.splitlines() if "->" not in b]
pat = r"\|8[45][0-9]\||\"seed\": 8[45][0-9][,}]"
for b in brs:
    r = git("grep", "-c", "-E", pat, b, "--", "*.json", "*.jsonl", check=False)
    hits = [l for l in r.stdout.splitlines() if l.strip()]
    print(f"  {b}: {len(hits)} file(s){': ' + '; '.join(hits[:5]) if hits else ''}")
