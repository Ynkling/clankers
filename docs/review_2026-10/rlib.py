"""rlib.py — the referee's helpers: record loading at pinned commits and the outcome definitions,
re-implemented from the report's Section 3 (not imported from the tests), plus the statistics.
Read-only: nothing here writes outside docs/review_2026-10/."""
import json
import math
import statistics
import subprocess
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIND = 0.95          # report Sec. 3, "Bound"
ROUTED_ACC = 0.9     # report Sec. 3, "Bound routed": every stream >= 0.9
DISC_COS = 0.5       # test_window_gate docstring: DISCOVERED = bound and final VAL cos < 0.5
COLLAPSE = 0.15      # report Sec. 3: collapsed when final accuracy < 0.15
MARGIN_PARTIAL, ETA_SPLIT = 0.25, 0.5   # failure classes (report Sec. 3; test_router_layout constants)

_cache = {}


def git(*args):
    return subprocess.check_output(("git", "-C", ROOT) + args).decode()


def load(ref, path):
    """A JSON record file at a git ref (ref None = the working tree of this checkout)."""
    key = (ref, path)
    if key not in _cache:
        if ref is None:
            with open(os.path.join(ROOT, path)) as f:
                _cache[key] = json.load(f)
        else:
            _cache[key] = json.loads(subprocess.check_output(("git", "-C", ROOT, "show", f"{ref}:{path}")))
    return _cache[key]


def runs(store, arm):
    """{seed: record} for an arm; keys are 'ARM|seed' or 'ARM|seed|sha[|cpu]'."""
    out = {}
    for k, r in store["runs"].items():
        p = k.split("|")
        if p[0] == arm:
            s = int(p[1])
            if s in out:
                raise ValueError(f"duplicate record for {arm}|{s}")
            out[s] = r
    return out


# ── outcomes ─────────────────────────────────────────────────────────────────
def transition(curve):
    """First evaluation >= 0.95 that holds to the end of the run (report Sec. 3)."""
    t = None
    for step, acc, *_ in reversed(curve):
        if acc >= BIND:
            t = step
        else:
            break
    return t


def ok(r):
    return bool(r and r.get("ok"))


def bound(r):
    return ok(r) and transition(r["curve"]) is not None


def discovered(r):
    return bound(r) and r["val_cos"] < DISC_COS


TIES = []   # (record id, stream) where the rounded stream_gate cannot decide the argmax


def ch_map(r):
    """Stream -> channel map at value positions: argmax of each stream's mean read gate, recomputed from
    the recorded stream_gate (rounded to 4 decimals). Where the top two channels tie after rounding the
    recorded ch_map (from unrounded values) decides, and the tie is logged in TIES."""
    out = []
    for i, row in enumerate(r["end"]["stream_gate"]):
        order = sorted(range(len(row)), key=lambda c: -row[c])
        if len(row) > 1 and abs(row[order[0]] - row[order[1]]) < 1e-4 + 1e-12:
            TIES.append((r.get("arm"), r.get("seed"), i))
            out.append(r["end"]["ch_map"][i])
        else:
            out.append(order[0])
    return out


def top_mass(r):
    """Each stream's largest channel mass at value positions (how sharp 'one channel per stream' is)."""
    return [max(row) for row in r["end"]["stream_gate"]]


def one_to_one(r):
    m = ch_map(r)
    return len(set(m)) == len(m)


def bound_routed(r):
    return bound(r) and one_to_one(r) and min(r["end"]["stream_acc"]) >= ROUTED_ACC


def shared_max(r):
    m = r["end"]["ch_map"]
    return max(m.count(c) for c in m)


def fail_class(r):
    """k=2 / generic failure class on the end statistics (test_router_layout.fail_class rule)."""
    e = r["end"]
    if e["margin"] >= MARGIN_PARTIAL:
        return "STREAM-PARTIAL"
    if e["eta_key_by_key"] >= ETA_SPLIT:
        return "KEY"
    if max(e["eta_key_by_half"], e["eta_key_by_index"]) >= ETA_SPLIT:
        return "POSITION"
    return "OTHER"


def fail_class_k(r):
    e = r["end"]
    return fail_class({"end": dict(margin=e["margin"], eta_key_by_key=e["etak_key_by_key"],
                                   eta_key_by_half=e["etak_key_by_block"], eta_key_by_index=e["etak_key_by_index"])})


def outcome_k(r):
    """k >= S outcome string (bound routed / bound not routed / MERGED j / non-stream class)."""
    if bound(r):
        return "BOUND ROUTED" if bound_routed(r) else "BOUND NOT routed"
    fc = fail_class_k(r)
    if fc == "STREAM-PARTIAL":
        return f"MERGED ({shared_max(r)} share)" if shared_max(r) >= 2 else "STREAM-PARTIAL one-to-one"
    return f"non-stream {fc}"


# ── statistics ───────────────────────────────────────────────────────────────
def binom_tail(n, b):
    return sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n


def mcnemar1(b, c):
    """Exact one-sided McNemar: P(Bin(b+c, 1/2) >= b)."""
    return 1.0 if b + c == 0 else binom_tail(b + c, b)


def mcnemar2(b, c):
    """Exact two-sided McNemar (doubled smaller tail, capped at 1)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * binom_tail(n, max(b, c)))


def fisher1(a, n1, c, n2):
    """One-sided Fisher exact: P(X >= a) for X ~ Hypergeom(successes a+c, draws n1, total n1+n2)."""
    K, N = a + c, n1 + n2
    tot = math.comb(N, n1)
    return sum(math.comb(K, x) * math.comb(N - K, n1 - x) for x in range(a, min(K, n1) + 1)) / tot


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return 0.0, 1.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def band(c, n):
    return ("RELIABLE" if c >= math.ceil(0.9 * n) else "MAJORITY" if c >= math.ceil(0.5 * n)
            else "MINORITY" if c >= 1 else "NEVER")


def pairs(ra, rb, f, seeds=None):
    """Discordant counts (a only, b only) over seeds both arms ran."""
    ss = sorted(set(ra) & set(rb)) if seeds is None else seeds
    a = sum(1 for s in ss if f(ra[s]) and not f(rb[s]))
    b = sum(1 for s in ss if f(rb[s]) and not f(ra[s]))
    return a, b, len(ss)


def med(xs):
    return statistics.median(xs) if xs else None
