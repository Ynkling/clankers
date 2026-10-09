#!/usr/bin/env python
"""
explore_batch19.py — EXPLORATORY, not a result. Batch 19 of the outside-ideas screens (the user's specification of
9 October 2026): does the window recipe survive width and depth? Same harness, rules and labels as batches 16-18
(explore_common16, imported read-only; explore_common19 adds checkpointed runs and dependent arms). Everything below was
fixed before any run.

  S64 explore_scale_width   S=4, P=4, k=16, conv: N = 256, 512, 1024 at D = 32; D = 64, 128 at N = 256 (seeds 270-279)
  S65 explore_scale_depth   the same at 2, 4, 6 layers against 3 (seeds 280-289); RES6 if 6 layers fails
  S66 explore_scale_eight   S=8, 43200, LOCAL3 + SLOW + KEYMASS split: S64's best width and S65's best depth vs N256 / 3
                            layers (seeds 290-299)
  S67 explore_capacity      S=4, P = 8 and 16 at N=256 and at S64's best width, the oracle first (seeds 270-279)

BACKGROUND, DEFINITIONS, ARMS, READINGS: in each screen's docstring.

ORDER AND REPORTS: S64 and S65 (phase 1) run first, the oracles before the learned arms. When every phase-1 arm is complete
(or UNTESTED, or not needed), the driver prints the report of S64 + S65, applies the phase-2 rule (S66 / S67's widths and
depth: explore_scale_width.best_candidate), stores the decision and the phase-2 projection, and phase 2 (S66, then S67)
starts in the same pool. A full report at the end of every segment and after S66 + S67.

HARNESS: 1 torch thread per run; every segment reproduces X's recorded arm A (seed 160, 2400 steps) bit for bit (a
failure stops the segment); runs are cached under arm|seed|code SHA|CPU (all arms are Adam: records from any CPU that
passed the repro check are accepted); a run with ok = False is retried from scratch (at most 3 attempts). Runs longer than a
segment: every run checkpoints after each evaluation and a stopped run resumes from its checkpoint in the next segment,
bit-identical to an uninterrupted run (explore_b19_model; CHECKed on the S=4 path, the S=8 path with splits before the
checkpoint, and RES6). Checkpoints live outside the repository (--ckpt-dir) and are deleted when the run is saved.

RUNTIME: the user gave no budget and no cut order, only "projection first; never cut the reference arms". The projection
(pool-load timing of every arm and of every phase-2 candidate; worst case: every run to its iters) is printed in the
first segment and stored; nothing is cut.

SEGMENTS: --resume, --budget MIN (default 116), --report-only, --dry (one seed per arm, phase-1 runs capped at 2400
updates and phase-2 runs at 1200, every arm marked ready, separate dry_ stores; not for interpretation), --ckpt-dir DIR.
A watcher outside this file pushes every saved run.

Run:  python explore_batch19.py [--workers 4] [--dry] [--resume] [--budget MIN] [--report-only] [--ckpt-dir DIR]
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import explore_common as ec
import explore_common2 as c2
import explore_common16 as c16
import explore_common19 as c19
import explore_main9c as mt
import explore_scale_width as s64
import explore_scale_depth as s65
import explore_scale_eight as s66
import explore_capacity as s67
from test_binding_recipe import makespan
from test_binding_onset import EVAL_EVERY

PHASE1 = (s64, s65)
PHASE2 = (s66, s67)
ALL = PHASE1 + PHASE2
CHILD = c19.CHILD
DRY_ITERS1, DRY_ITERS2 = 2400, 1200
CHECKS_RECORD = os.path.join(ec.OUT_DIR, "batch19_checks.json")
RUNTIME_STORE = "batch19_runtime"
DEFAULT_BUDGET_MIN = 116
DEFAULT_CKPT = os.path.join(os.environ.get("TMPDIR", "/tmp"), "b19_ckpt")
STATE = dict(dry=False, configured=False, timings={})


def runtime_store():
    return ("dry_" if STATE["dry"] else "") + RUNTIME_STORE


def make_dry():
    c2.DRY.update(on=True, iters=DRY_ITERS1)
    for m in PHASE1:
        m.NAME = "dry_" + m.NAME
        m.ARMS = {k: dict(a, iters=min(a["iters"], DRY_ITERS1), seeds=tuple(a["seeds"][:1])) for k, a in m.ARMS.items()}
    for m in PHASE2:
        m.NAME = "dry_" + m.NAME


def dry_cap(m):
    if STATE["dry"]:
        m.ARMS.update({k: dict(a, iters=min(a["iters"], DRY_ITERS2), seeds=tuple(a["seeds"][:1])) for k, a in m.ARMS.items()})


def screens():
    return PHASE1 + (PHASE2 if STATE["configured"] else ())


# ── Specs, timings, projection ───────────────────────────────────────────────
def spec_key(sp):
    return f"{sp['cfg']}|{sp.get('task', 'P4S8' if sp['cfg'] == 'b' else 'P4S4')}|{sp['gate']}|" + "x".join(map(str, sp["size"])) \
        + ("|T" if sp.get("trigger") else "")


def candidate_specs():
    """Every phase-2 spec any decision could need (S66 at each width / depth, S67 at each width and P)."""
    out = {}
    for w, (mult, d) in s64.SIZES.items():
        out[f"S66 {w}_L3"] = dict(cfg="b", task="P4S8", gate="local", size=[mult, d, 3], trigger=True)
        for P in s67.PS:
            out[f"S67 ORC_P{P}_{w}"] = dict(cfg="k16", task=f"P{P}S4", gate="perfect", size=[mult, d, 3])
            out[f"S67 P{P}_{w}"] = dict(cfg="k16", task=f"P{P}S4", gate="local", size=[mult, d, 3])
    for L in s65.DEPTHS:
        if L != 3:
            out[f"S66 N256_L{L}"] = dict(cfg="b", task="P4S8", gate="local", size=[8, 32, L], trigger=True)
    return out


def measure_timings(workers, steps=50):
    specs = {}
    for m in PHASE1:
        for k, a in m.ARMS.items():
            specs[spec_key(a["spec"])] = a["spec"]
    for k, sp in candidate_specs().items():
        specs[spec_key(sp)] = sp
    keys = sorted(specs)
    chunks = [keys[i::workers] for i in range(workers)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda ch: c16.run_child(CHILD, "timing", dict(which={k: specs[k] for k in ch}, steps=steps), main=True)
                          if ch else {}, chunks))
    t = {}
    for r in res:
        t.update(r)
    return t


def per_run(spec, iters, t):
    st, te = t[spec_key(spec)]
    return iters * st + (iters // EVAL_EVERY) * te


def projection(workers, dry, t_start):
    st = ec.load_store(runtime_store())
    tm = st["meta"].get("timings")
    if tm and not dry:
        print(f"  PROJECTION (stored {st['meta']['projection_time']} on {st['meta']['projection_cpu']}; the stored timings are reused)")
        STATE["timings"] = tm
        return tm
    t0 = time.time()
    tm = measure_timings(workers)
    STATE["timings"] = tm
    print(f"  PROJECTION (worst case: every run to its iters; pool-load timing on {workers} parallel children, 1 thread each, this "
          f"machine; {time.time() - t0:.0f}s to measure):")
    durs1 = []
    full1 = {s64: s64.ARMS, s65: s65.ARMS} if not dry else {s64: FULL1["scale_width"], s65: FULL1["scale_depth"]}
    for m, arms in full1.items():
        for k, a in arms.items():
            d = per_run(a["spec"], a["iters"], tm)
            st_, te = tm[spec_key(a["spec"])]
            n = len(a["seeds"])
            durs1 += [d] * n
            print(f"    {m.NAME.replace('dry_', ''):<12} {k:<9} {n:>2} runs x {d / 60:6.1f} min (step {st_ * 1e3:6.1f} ms, evaluation {te:5.2f} s)"
                  + ("  [only if 6 layers fails]" if k == "RES6" else "") + ("  [only if its oracle binds]" if a.get("oracle") else ""))
    exp1 = []
    for m, arms in full1.items():
        for k, a in arms.items():
            st_, te = tm[spec_key(a["spec"])]
            exp1 += [8400 * st_ + 7 * te] * len(a["seeds"])
    print(f"    PHASE 1 (S64 + S65, RES6 included): {sum(durs1) / 3600:.1f} h of runs, makespan on {workers} workers "
          f"{makespan(durs1, workers) / 3600:.1f} h worst case; {makespan(exp1, workers) / 3600:.1f} h if every run stops at 8400 "
          f"(batch 16's S44(a) stopped 18 of 20 runs there)")
    cs = candidate_specs()
    print("    PHASE 2 per candidate (worst case; S66 10 runs x 43200, S67 per width 2 oracles x 2 P + 10 learned x 2 P, x 43200):")
    ref66 = 10 * per_run(cs["S66 N256_L3"], 43200, tm)
    ref67 = sum(2 * per_run(cs[f"S67 ORC_P{P}_N256"], 43200, tm) + 10 * per_run(cs[f"S67 P{P}_N256"], 43200, tm) for P in s67.PS)
    print(f"      fixed: S66 N256_L3 {ref66 / 3600:.1f} h; S67 at N256 {ref67 / 3600:.1f} h (P=8 oracle {per_run(cs['S67 ORC_P8_N256'], 43200, tm) / 3600:.2f} "
          f"h/run, P=16 oracle {per_run(cs['S67 ORC_P16_N256'], 43200, tm) / 3600:.2f} h/run)")
    wc = {}
    for w in s64.SIZES:
        if w == "N256":
            continue
        a66 = 10 * per_run(cs[f"S66 {w}_L3"], 43200, tm)
        a67 = sum(2 * per_run(cs[f"S67 ORC_P{P}_{w}"], 43200, tm) + 10 * per_run(cs[f"S67 P{P}_{w}"], 43200, tm) for P in s67.PS)
        wc[w] = a66 + a67
        print(f"      best width {w:<6}: S66 {a66 / 3600:6.1f} h + S67 {a67 / 3600:6.1f} h (P=16 learned run {per_run(cs[f'S67 P16_{w}'], 43200, tm) / 3600:.2f} h)")
    dc = {}
    for L in s65.DEPTHS:
        if L == 3:
            continue
        dc[L] = 10 * per_run(cs[f"S66 N256_L{L}"], 43200, tm)
        print(f"      best depth L{L}: S66 {dc[L] / 3600:6.1f} h")
    lo = ref66 + ref67 + min(wc.values()) + min(dc.values())
    hi = ref66 + ref67 + max(wc.values()) + max(dc.values())
    print(f"    PHASE 2 total, worst case: {lo / 3600:.0f}-{hi / 3600:.0f} h of runs ({lo / 3600 / workers:.0f}-{hi / 3600 / workers:.0f} h "
          f"on {workers} workers if perfectly packed); a run binding early stops early (an oracle that binds at 1200 stops at 8400)")
    print("    no budget was given: nothing is cut (and the reference arms never are); the decision is the user's", flush=True)
    if not dry:
        st["meta"].update(timings=tm, projection_time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                          projection_cpu=ec.cpu_model())
        st["meta"].setdefault("provenance", ec.provenance())
        ec.save_store(runtime_store(), st)
    return tm


FULL1 = {}


def dur_fn(j):
    try:
        return per_run(j["spec"], j["iters"], STATE["timings"])
    except KeyError:
        return 0.0


# ── Phase decision ───────────────────────────────────────────────────────────
def phase1_settled():
    for m in PHASE1:
        st = ec.load_store(m.NAME)
        for arm in m.ARMS:
            state, _ = c19.arm_state(m, arm, st, STATE["dry"])
            if state == "never":
                continue
            if state == "wait" or not c19.complete(m, arm, st):
                return False
    return True


def configure(dec):
    s66.configure(dec)
    s67.configure(dec)
    for m in PHASE2:
        dry_cap(m)
    STATE["configured"] = dec is not None
    if dec is not None:
        for m in PHASE2:
            st = ec.load_store(m.NAME)
            st["meta"].setdefault("provenance", ec.provenance())
            st["meta"]["arms"] = {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in a.items()} for k, a in m.ARMS.items()}
            st["meta"]["decision"] = dec
            st["meta"]["dry"] = STATE["dry"]
            ec.save_store(m.NAME, st)


def decide():
    """Called by the pool when idle or after a run: phase 2's arms once phase 1 is settled."""
    if STATE["configured"]:
        return False
    st = ec.load_store(runtime_store())
    dec = st["meta"].get("phase2")
    if dec is None:
        if not phase1_settled():
            return False
        print("#" * 100)
        print(f"REPORT AFTER S64 + S65 — {ec.BANNER}{c2.dry_label()}")
        print("#" * 100, flush=True)
        r64 = s64.report()
        r65 = s65.report()
        w, d = r64.get("best"), r65.get("best")
        how = "the rule"
        if STATE["dry"] and (w is None or d is None):
            w, d, how = w or "N512", d or "L4", "the rule, with a dry-run fallback (N512 / L4) where it gave none"
        dec = dict(best_width=w, best_depth=int(d[1:]) if d else None, rule=how, reading_s64=r64.get("reading"),
                   reading_s65=r65.get("reading"), time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), cpu=ec.cpu_model(),
                   width_summary={k: dict(br=v.get("br"), n=v.get("n"), trans_med=v.get("trans_med"), params=v.get("params"),
                                          tested=v.get("tested")) for k, v in r64.get("summary", {}).items() if not k.startswith("ORC")},
                   depth_summary={k: dict(br=v.get("br"), n=v.get("n"), trans_med=v.get("trans_med"), tested=v.get("tested"))
                                  for k, v in r65.get("summary", {}).items() if not k.startswith("ORC")})
        st["meta"]["phase2"] = dec
        ec.save_store(runtime_store(), st)
        print(f"  PHASE-2 DECISION ({how}; stored): S64's best width {w}; S65's best depth {d}. S66 arms: N256_L3"
              + (f", {w}_L3" if w else "") + (f", {d.replace('L', 'N256_L')}" if d else "")
              + f"; S67 widths: N256" + (f", {w}" if w else ""), flush=True)
        tm = STATE["timings"]
        if tm:
            configure(dec)
            durs = [dur_fn(dict(spec=a["spec"], iters=a["iters"])) for m in PHASE2 for a in m.ARMS.values() for _ in a["seeds"]]
            print(f"  PHASE-2 PROJECTION (worst case): {sum(durs) / 3600:.1f} h of runs, makespan on {ec.WORKERS} workers "
                  f"{makespan(durs, ec.WORKERS) / 3600:.1f} h", flush=True)
            for m in PHASE2:
                c2.print_screen_header2(m)
            return True
    configure(dec)
    return True


# ── CHECKs, constants ────────────────────────────────────────────────────────
def parent_checks():
    rows = []
    rd = s64.reading_of
    rows.append(("S64/S65 reading: d = (0, -1, -1, 0) -> scale-free; (0, -4) -> width-sensitive; (-2, 0) -> neither; incomplete -> "
                 "no reading; an untested arm takes no part",
                 rd({"a": dict(tested=True, complete=True, d=0), "b": dict(tested=True, complete=True, d=-1),
                     "c": dict(tested=True, complete=True, d=-1), "e": dict(tested=False, complete=True, d=None)}, True, "R").startswith("scale-free")
                 and rd({"a": dict(tested=True, complete=True, d=0), "b": dict(tested=True, complete=True, d=-4)}, True, "R").startswith("width-sensitive")
                 and rd({"a": dict(tested=True, complete=True, d=-2), "b": dict(tested=True, complete=True, d=0)}, True, "R").startswith("neither")
                 and rd({"a": dict(tested=True, complete=False, d=0)}, True, "R").startswith("INCOMPLETE")
                 and rd({"a": dict(tested=True, complete=True, d=0)}, False, "R").startswith("INCOMPLETE")))
    sm = {"N256": dict(role="ref", tested=True, complete=True, n=10, br=10, trans_med=3600, params=1),
          "N512": dict(role="arm", tested=True, complete=True, n=10, br=9, trans_med=3600, params=2),
          "N1024": dict(role="arm", tested=True, complete=True, n=10, br=9, trans_med=3600, params=4),
          "D64": dict(role="arm", tested=True, complete=True, n=10, br=9, trans_med=2400, params=3),
          "D128": dict(role="arm", tested=False, complete=True, n=0, br=0, trans_med=None, params=None)}
    b1 = s64.best_candidate(sm, "N256")
    sm["D64"]["trans_med"] = 3600
    b2 = s64.best_candidate(sm, "N256")
    sm["N1024"]["br"] = 8
    sm["D64"]["br"] = 8
    b3 = s64.best_candidate(sm, "N256")
    rows.append((f"phase-2 rule: most BOUND ROUTED, then the earlier median transition, then the larger model; never the reference "
                 f"or an untested arm (synthetic: {b1}, {b2}, {b3} == D64, N1024, N512)", (b1, b2, b3) == ("D64", "N1024", "N512")))
    cr = s67.cap_reading
    rows.append(("S67 reading: P=16 oracle 0/2 at N256 and 1/2 at W -> capacity-bound; 1/2 at N256 -> not; 0 and 0 -> neither width",
                 cr(dict(bound=0, complete=True), dict(bound=1, complete=True), "N1024").startswith("capacity-bound")
                 and cr(dict(bound=1, complete=True), dict(bound=2, complete=True), "N1024").startswith("not capacity-bound")
                 and cr(dict(bound=0, complete=True), dict(bound=0, complete=True), "N1024").startswith("the oracle binds P=16 at neither")
                 and cr(dict(bound=0, complete=False), dict(bound=1, complete=True), "N1024").startswith("INCOMPLETE")))
    rows.append((f"S65's RES6 condition: L6 - L3 <= {s65.FAIL_D} (misses the scale-free margin of {-s64.FREE_D}); S66 carries at d >= "
                 f"{s66.CARRY_D}; S64 scale-free at d >= {s64.FREE_D}, width-sensitive at d <= {s64.SENS_D}",
                 s65.FAIL_D == s64.FREE_D - 1 and s66.CARRY_D == -1 and s64.FREE_D == -1 and s64.SENS_D == -4))
    return rows


def run_checks():
    t0 = time.time()
    jobs = [("checks", {}, None),
            ("resume_check", dict(spec=dict(cfg="k16", task="P4S4", gate="local", size=[8, 32, 3]), seed=270, iters=3600, halt_at=2400),
             "S=4 k16 LOCAL3 N256 L3, seed 270: halted after the checkpoint at 2400, resumed, against 3600 uninterrupted"),
            ("resume_check", dict(spec=dict(cfg="b", task="P4S8", gate="local", size=[8, 32, 3], trigger=True), seed=290, iters=6000,
                                  halt_at=4800, thr=1.01, rise=1.0, total=43200),
             "S=8 D8 LOCAL3 + trigger (threshold above 1: a split fires at 4800), seed 290: halted after the checkpoint at 4800 (after "
             "the split), resumed, against 6000 uninterrupted"),
            ("resume_check", dict(spec=dict(cfg="k16", task="P4S4", gate="resgate", size=[8, 32, 3]), seed=280, iters=2400, halt_at=1200),
             "RES gate at 3 layers, seed 280: halted at 1200, resumed, against 2400 uninterrupted")]
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        res = list(ex.map(lambda j: c16.run_child(CHILD, j[0], j[1], main=True), jobs))
    rows = [tuple(x) for x in res[0]]
    for (fn, p, desc), r in zip(jobs[1:], res[1:]):
        ok = (r["ok"] and not r["diff"] and r["halted"].get("halted") and r["halted"]["step"] == p["halt_at"]
              and r["resumed"] == [p["halt_at"]])
        if p["spec"].get("trigger"):
            ok &= r["splits_full"] == r["splits_res"] == [4800]
        rows.append((f"{desc}: halted at {r['halted'].get('step')}, resumed from {r['resumed']}; fields differing {r['diff']} of "
                     f"{r['n_keys']} compared (curve, statistics, gradient norms, end statistics, transition, early curve, the "
                     f"trigger's rows and splits {r['splits_full']} / {r['splits_res']}, final weights' SHA-1, outcome, initial scales)", ok))
    rows += parent_checks()
    diff = mt.differing()
    rows.append((f"the modules served from main are exactly the test modules that differ between this branch and {mt.MAIN_SHA[:7]} "
                 f"({diff})", diff == sorted(mt.SERVED)))
    ok = True
    for nm, v in rows:
        print(f"  CHECK batch 19: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    if not ok:
        print("  ! a CHECK failed; stopping.")
        sys.exit(1)
    print(f"  all CHECKs passed ({(time.time() - t0) / 60:.1f} min)", flush=True)


def constants():
    k = c16.run_child(CHILD, "constants", {}, main=True)
    assert k["LR"] == 1e-3 and k["LR_WARM"] == 1e-4 and k["WARM"] == 2400 and k["tsr_LR"] == k["tscur_LR"] == 1e-3, k
    assert k["MULT_A4k16"] == 8 and k["D"] == 32 and k["ARCH"]["n_layer"] == 3 and k["WIDTH"] == 3 and k["MAX_SPLITS"] == 3, k
    assert k["GAP"] == 4800 and k["FIRST"] == 4800 and k["EVERY"] == 2400 and k["EVAL_EVERY"] == 1200, k
    print(f"  recipe read in a child from main at {k['main_sha'][:7]}: test_stream_recipe.LR {k['tsr_LR']:g}, "
          f"test_stream_curriculum.LR {k['tscur_LR']:g}; SLOW: gate {k['LR']:g} throughout, others {k['LR_WARM']:g} for updates "
          f"1-{k['WARM']}; window width {k['WIDTH']}; the model: ARCH {k['ARCH']}, D {k['D']}, A4k16's mult {k['MULT_A4k16']} (N = "
          f"{k['MULT_A4k16'] * k['D']}), h_gate {k['H_GATE']}, {k['conv']}; evaluation every {k['EVAL_EVERY']}, bound at "
          f"{k['BIND_PASS']} held, early stop after {k['STOP_AFTER']} in a row (after {k['T_CHECK']} on the k16 path); the "
          f"trigger: every {k['EVERY']} from {k['FIRST']} (reference {k['REF_AT']}), acc < {k['ACC_THR']}, rise < {k['RISE']}, gap "
          f"{k['GAP']}, <= {k['MAX_SPLITS']} splits; tasks {k['tasks']}; 1 torch thread per run", flush=True)
    for m in screens():
        for kk, a in m.ARMS.items():
            print(f"  lr for {m.NAME} {kk}: {a['lr']:g}; iters {a['iters']}; seeds {ec.fmt_seeds(a['seeds'])}; size {a['spec']['size']}")


def line(j, rec):
    acc = rec.get("acc")
    s = f"acc {acc:.3f}  trans {str(rec.get('transition')):>5}  stop {rec.get('stopped_at')}  outcome {rec.get('outcome')}"
    if j["spec"].get("trigger"):
        s += f"; splits {rec.get('splits')} at {[c['step'] for c in rec.get('checks') or [] if c.get('fired')]}"
    return s


# ── Main ─────────────────────────────────────────────────────────────────────
def report_all(tag=""):
    print("#" * 100)
    print(f"REPORTS — {ec.BANNER}{c2.dry_label()}")
    print("#" * 100)
    r = dict(s64=s64.report(), s65=s65.report())
    if STATE["configured"]:
        r["s66"] = s66.report()
        r["s67"] = s67.report()
    print("#" * 100)
    print(f"SUMMARY — {ec.BANNER}{c2.dry_label()}")
    print(f"  {tag}S64 scale_width: {r['s64']['reading']}")
    print(f"  {tag}S65 scale_depth: {r['s65']['reading']}; RES6: {r['s65']['res6'].get('why') or r['s65']['res6'].get('state')}")
    if STATE["configured"]:
        for arm, o in r["s66"].items():
            if o.get("reading"):
                print(f"  {tag}S66 scale_eight {arm}: {o['reading']}")
        print(f"  {tag}S67 capacity: {r['s67'].get('reading')}")
    else:
        print("  S66, S67: not configured yet (phase 1 incomplete)")
    print("#" * 100)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=ec.WORKERS)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--budget", type=float, default=None, help="minutes left in this segment")
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT)
    args = ap.parse_args()
    t_start = time.time()
    STATE["dry"] = args.dry
    if args.resume:
        print("#" * 20 + f" RESUME {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC — a new segment; saved runs are reused, "
              f"stopped runs resume from their checkpoints ({ec.BANNER}) " + "#" * 20)
    cpu = ec.cpu_model()
    rec = None
    if os.path.exists(CHECKS_RECORD) and not args.dry:
        with open(CHECKS_RECORD) as f:
            rec = json.load(f)
    c16.STATE["batch_cpu"] = rec["cpu"] if rec else cpu
    FULL1.update({m.NAME: {k: dict(a) for k, a in m.ARMS.items()} for m in PHASE1})
    if args.dry:
        make_dry()
    dl = c2.dry_label()
    pv = ec.print_banner("batch 19: scale_width (S64), scale_depth (S65), scale_eight (S66), capacity (S67)" + dl)
    sha, rows = c16.code_sha(CHILD, True)
    print(f"  code SHA (one child for all four screens): {sha[:12]} over {len(rows)} modules ({sum(1 for r in rows if r[2] != 'branch')} "
          f"served from main at {mt.MAIN_SHA[:7]}): " + " ".join(r[0] for r in rows), flush=True)
    c16.STATE["shas"] = {m.NAME: sha for m in ALL}
    c16.STATE["other_cpus"] = sorted(set(c16.other_cpus_seen(PHASE1)) | {cpu})
    print(f"  this segment's CPU: {cpu}; the batch's CPU (the full CHECK pass): {c16.STATE['batch_cpu']}; checkpoints in "
          f"{args.ckpt_dir}", flush=True)
    st = ec.load_store(runtime_store())
    if args.report_only:
        print("  REPORT ONLY: from the saved runs (no repro check, CHECKs or runs in this call)")
        STATE["timings"] = st["meta"].get("timings") or {}
        configure(st["meta"].get("phase2"))
        report_all()
        return
    if not ec.repro_check():
        print("  ! this container does not reproduce X: the recorded pairing is invalid; stopping.")
        sys.exit(1)
    projection(args.workers, args.dry, t_start)
    constants()
    if rec and rec.get("sha") == sha and not args.dry:
        print(f"  CHECKs skipped (repro check above still run): the code SHA equals the one at the full CHECK pass ({rec['time']}, "
              f"git {rec['git']}, on {rec['cpu']}; its output is earlier in this log)")
    else:
        if rec:
            print(f"  running every CHECK (code SHA at the full pass {rec.get('sha')} vs now {sha})")
        run_checks()
        if not args.dry:
            with open(CHECKS_RECORD, "w") as f:
                json.dump(dict(git=ec.git_head(), sha=sha, cpu=c16.STATE["batch_cpu"], banner=ec.BANNER,
                               time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())), f, indent=1)
            print(f"  recorded the full CHECK pass (code SHA, the batch's CPU {c16.STATE['batch_cpu']}) in {CHECKS_RECORD}")
    for m in PHASE1:
        c2.print_screen_header2(m)
        s_ = ec.load_store(m.NAME)
        s_["meta"].setdefault("provenance", pv)
        s_["meta"]["dry"] = args.dry
        s_["meta"]["arms"] = {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in a.items()} for k, a in m.ARMS.items()}
        s_["meta"].setdefault("code_shas", [])
        if sha not in s_["meta"]["code_shas"]:
            s_["meta"]["code_shas"].append(sha)
        ec.save_store(m.NAME, s_)
    dec = ec.load_store(runtime_store())["meta"].get("phase2")
    if dec is not None:
        print(f"  phase-2 decision (stored {dec['time']}): best width {dec['best_width']}, best depth {dec['best_depth']}")
        configure(dec)
        for m in PHASE2:
            c2.print_screen_header2(m)
    budget = None if args.budget is None else args.budget * 60 - (time.time() - t_start)
    print(f"  pool: {args.workers} workers (1 torch thread each)" + (f"; segment budget {budget / 60:.0f} min left" if budget else ""),
          flush=True)
    t0 = time.time()
    n = c19.run_pool19(screens, args.workers, budget, line, args.ckpt_dir, cpu, args.dry, dur_fn, advance=decide)
    print(f"  {n} runs saved in this segment; batch wall clock {(time.time() - t0) / 60:.1f} min (this segment)", flush=True)
    report_all("DRY RUN " if args.dry else "")


if __name__ == "__main__":
    main()
