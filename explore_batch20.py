#!/usr/bin/env python
"""
explore_batch20.py — EXPLORATORY, not a result. Batch 20 of the outside-ideas screens (the user's specification of
11 October 2026): S68 explore_continue — are batch 19's unbound P = 16 runs slow or stuck? Same harness, rules, labels,
segments and checkpoints as batch 19 (explore_common16 / explore_common19, imported read-only). Everything below was fixed
before any S68 run (the readings: explore_continue's docstring).

DEVIATION FROM THE SPECIFICATION (agreed with the user before any run): batch 19 deleted every checkpoint when its run was
saved, so the runs are regenerated from update 0 to 86,400 and each must reproduce its S67 record through 43,200 bit for bit
(explore_continue). The user's premise check is kept: C_P8_N256 272 halts after its 43,200 checkpoint and a new process
resumes it; RCHK_P8_N256 272 computes 43,200 -> 44,400 without that checkpoint; both must agree at 44,400 bit for bit.

HARNESS: 1 torch thread per run; every segment reproduces X's recorded arm A (seed 160, 2400 steps) bit for bit (a failure
stops the segment); runs are cached under arm|seed|code SHA|CPU; a run with ok = False is retried from scratch (at most 3
attempts); every run checkpoints after each evaluation and a stopped run resumes from its checkpoint in the next segment
(explore_b19_model). The CHECKs (first segment, recorded; later segments skip them while the code SHA is unchanged):
  - the batch-20 additions do not change a run: P8_N256 seed 272 with every measurement equals batch 19's run path without
    them through 2400 (curve, statistics, early curve, end statistics, final weights' SHA-1);
  - every arm reproduces its S67 record through 2400 (P8_N256 272, P16_N256 270, P16_D128 271, ORC_P16_D128 270);
  - the halt-and-new-process path at small scale: P8_N256 272 halted at 1200, the process replaced, resumed, to 2400, equals
    the uninterrupted run (curve, statistics, measurements, weights' SHA-1), resumed exactly at 1200;
  - the checkpoint guard (no checkpoint after ckpt_until) writes nothing after it and writes at it;
  - the S67 records: every S68 seed has a saved S67 run at 43,200, unbound, with the class the specification names;
  - the readings and the budget rule on synthetic counts; the served modules.
RUNTIME: the projection (batch 19's stored pool-load timings of these exact specs; worst case, every run to its iters) is
printed first; nothing is cut (no budget was given).
SEGMENTS: --resume, --budget MIN (default 116), --report-only, --dry (one seed per arm, every step scaled to 2400, separate
dry_ store; it exercises the code and the reproduction through 2400; not for interpretation), --ckpt-dir DIR.

Run:  python explore_batch20.py [--workers 4] [--dry] [--resume] [--budget MIN] [--report-only] [--ckpt-dir DIR]
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
import explore_continue as s68
import explore_main9c as mt
from test_binding_recipe import makespan
from test_binding_onset import EVAL_EVERY

CHILD = s68.CHILD
CHECKS_RECORD = os.path.join(ec.OUT_DIR, "batch20_checks.json")
B19_RUNTIME = "batch19_runtime"
RUNTIME_STORE = "batch20_runtime"
DEFAULT_CKPT = os.path.join(os.environ.get("TMPDIR", "/tmp"), "b20_ckpt")
DRY_ITERS = 2400
STATE = dict(dry=False, timings={})

c19.CHILD = CHILD                      # the pool's child module (run_pool19 reads it at each start)
c19.ROLE_RANK["check"] = -1            # the reproduction's run first


def screens():
    return [s68]


def spec_key(sp):
    return f"{sp['cfg']}|{sp['task']}|{sp['gate']}|" + "x".join(map(str, sp["size"]))


def per_run(spec, iters, t):
    st, te = t[spec_key(spec)]
    return iters * st + (iters // EVAL_EVERY) * te


def dur_fn(j):
    try:
        return per_run(j["spec"], j["iters"], STATE["timings"])
    except KeyError:
        return 0.0


def projection(workers):
    tm = ec.load_store(B19_RUNTIME)["meta"]["timings"]
    STATE["timings"] = tm
    print(f"  PROJECTION (worst case: every run to its iters; batch 19's pool-load timing of these specs, stored "
          f"{ec.load_store(B19_RUNTIME)['meta']['projection_time']}; S67's observed rates were ~1.5x faster):")
    durs = []
    for k, a in s68.ARMS.items():
        d = per_run(a["spec"], a["iters"], tm)
        st_, te = tm[spec_key(a["spec"])]
        durs += [d] * len(a["seeds"])
        print(f"    {k:<15} {len(a['seeds']):>2} runs x {d / 3600:5.1f} h (step {st_ * 1e3:6.1f} ms, evaluation {te:5.1f} s, "
              f"{a['iters']} updates)")
    print(f"    TOTAL {sum(durs) / 3600:.0f} h of runs; makespan on {workers} workers {makespan(durs, workers) / 3600:.1f} h worst case "
          f"(a run that binds stops early); nothing is cut (no budget was given)", flush=True)
    if not STATE["dry"]:
        st = ec.load_store(RUNTIME_STORE)
        st["meta"].update(projection_time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), projection_cpu=ec.cpu_model(),
                          projection_h=sum(durs) / 3600, makespan_h=makespan(durs, workers) / 3600)
        st["meta"].setdefault("provenance", ec.provenance())
        ec.save_store(RUNTIME_STORE, st)


# ── CHECKs ───────────────────────────────────────────────────────────────────
FIELDS = ("curve", "stats", "early", "end", "w_sha", "transition", "stopped_at", "outcome", "acc_check", "passed")


def parent_checks():
    rows = []
    wr, jr, bo = s68.width_reading, s68.joint_reading, s68.budget_of
    rows.append(("S68 width reading: 6/6, 4/6 -> slow, not stuck; 3/6, 1/6 -> unresolved; 0/6 -> stuck; incomplete -> INCOMPLETE",
                 [wr(k, True) for k in (6, 4, 3, 1, 0)] == ["slow, not stuck"] * 2 + ["unresolved"] * 2 + ["stuck"]
                 and wr(6, False) == "INCOMPLETE"))
    rows.append(("S68 joint reading: (D128, N256) = (4, 4) budget; (6, 1), (4, 0) width; (4, 2), (3, 0), (0, 6) neither; incomplete",
                 [jr(*x, True) for x in ((4, 4), (6, 1), (4, 0), (4, 2), (3, 0), (0, 6))]
                 == ["the budget is the lead", "the width is the lead", "the width is the lead", "neither", "neither", "neither"]
                 and jr(6, 6, False).startswith("INCOMPLETE")))
    rows.append(("budget rule: latest transition + 2400, rounded up to a multiple of 2400 (S67 alone: [19200, 26400, 37200, 38400] -> "
                 "40800; 40000 -> 43200; none -> None)",
                 bo([19200, 26400, 37200, 38400]) == 40800 and bo([40000]) == 43200 and bo([]) is None))
    st67 = s68.s67_store()
    exp = {("P16_N256", s): "STREAM-PARTIAL one-to-one" for s in s68.ONE2ONE["N256"]}
    exp.update({("P16_D128", s): "STREAM-PARTIAL one-to-one" for s in s68.ONE2ONE["D128"]})
    ok, bad = True, []
    for arm, a in s68.build_arms().items():
        for s in a["seeds"]:
            r = s68.s67_record(a["s67"], s, st67)
            good = (r is not None and r.get("stopped_at") == s68.S67_ITERS and r.get("transition") is None
                    and (exp.get((a["s67"], s)) is None or r.get("outcome") == exp[(a["s67"], s)])
                    and (a["s67"] != "P16_N256" or s not in s68.MERGED_N256 or str(r.get("outcome")).startswith("MERGED")))
            if not good:
                ok, bad = False, bad + [f"{a['s67']}|{s}"]
    rows.append((f"S67 records: every S68 seed has a saved S67 run (code {s68.S67_SHA}) stopped at 43,200, unbound, with the class the "
                 f"specification names (one-to-one N256 {s68.ONE2ONE['N256']}, D128 {s68.ONE2ONE['D128']}; MERGED N256 "
                 f"{s68.MERGED_N256}); P16_D128 has no other unbound seed{'; bad: ' + str(bad) if bad else ''}",
                 ok and sorted(int(k.split('|')[1]) for k, r in st67["runs"].items() if k.startswith("P16_D128|") and r.get("transition") is None)
                 == sorted(s68.ONE2ONE["D128"])))
    return rows


def run_checks():
    t0 = time.time()
    arms = s68.build_arms()
    small = dict(marks=[1200, 2400], sha_at=[1200, 2400])
    sp8 = dict(arms["C_P8_N256"]["spec"], **small)
    sp8.pop("halt_once_at")
    jobs = {
        "guard": ("save_guard_check", {}),
        "plain8": ("plain", dict(spec=sp8, seed=272, iters=2400)),
        "meas8": ("run", dict(spec=sp8, seed=272, iters=2400, ckpt=None)),
        "n256": ("run", dict(spec=dict(arms["C_P16_N256"]["spec"], **small), seed=270, iters=2400, ckpt=None)),
        "d128": ("run", dict(spec=dict(arms["C_P16_D128"]["spec"], **small), seed=271, iters=2400, ckpt=None)),
        "orc": ("run", dict(spec=dict(arms["C_ORC_P16_D128"]["spec"], **small), seed=270, iters=2400, ckpt=None)),
    }
    import tempfile
    td = tempfile.mkdtemp(prefix="b20chk_")
    jobs["halt8"] = ("run", dict(spec=dict(sp8, halt_once_at=1200), seed=272, iters=2400, ckpt=os.path.join(td, "ck.pt")))
    keys = list(jobs)
    with ThreadPoolExecutor(max_workers=len(keys)) as ex:
        res = dict(zip(keys, ex.map(lambda k: c16.run_child(CHILD, jobs[k][0], jobs[k][1], main=True), keys)))
    rows = [("the checkpoint guard writes nothing after ckpt_until and writes at it", res["guard"]["ok"])]
    diff = [k for k in FIELDS if res["plain8"].get(k) != res["meas8"].get(k)]
    rows.append((f"the batch-20 measurements do not change a run: P8_N256 272 with marks and weight hashes at 1200 / 2400 equals batch "
                 f"19's run path without them through 2400 (fields differing {diff} of {len(FIELDS)}: {', '.join(FIELDS)})",
                 res["plain8"].get("ok") and res["meas8"].get("ok") and not diff))
    s67 = s68.s67_store()
    for k, arm, seed in (("meas8", "P8_N256", 272), ("n256", "P16_N256", 270), ("d128", "P16_D128", 271), ("orc", "ORC_P16_D128", 270)):
        ok, why = s68.reproduced(res[k], s68.s67_record(arm, seed, s67), upto=2400)
        rows.append((f"{arm} seed {seed} reproduces its S67 record through 2400 (curve, statistics, early curve){' — ' + why if why else ''}"
                     f" [curve {res[k].get('curve')}]", ok))
    h = res["halt8"]
    diff = [k for k in FIELDS if res["meas8"].get(k) != h.get(k)] + (["marks"] if h.get("marks") != res["meas8"].get("marks") else []) \
        + (["shas"] if h.get("shas") != res["meas8"].get("shas") else [])
    rows.append((f"the halt-and-new-process path: P8_N256 272 halted after its checkpoint at 1200, its process replaced (os.execv), "
                 f"resumed from {h.get('resumed')}, marker {h.get('halted_once_at')}, to 2400: fields differing from the uninterrupted "
                 f"run {diff} (of {len(FIELDS)} + marks + weight hashes)", h.get("ok") and not diff and h.get("halted_once_at") == 1200
                 and h.get("resumed") == [1200] and not os.path.exists(os.path.join(td, "ck.pt.halted"))))
    rows += parent_checks()
    d = mt.differing()
    rows.append((f"the modules served from main are exactly the test modules that differ between this branch and {mt.MAIN_SHA[:7]} ({d})",
                 d == sorted(mt.SERVED)))
    ok = True
    for nm, v in rows:
        print(f"  CHECK batch 20: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    if not ok:
        print("  ! a CHECK failed; stopping.")
        sys.exit(1)
    print(f"  all CHECKs passed ({(time.time() - t0) / 60:.1f} min)", flush=True)


def constants():
    k = c16.run_child(CHILD, "constants", {}, main=True)
    assert k["LR"] == 1e-3 and k["LR_WARM"] == 1e-4 and k["WARM"] == 2400 and k["tsr_LR"] == 1e-3, k
    assert k["MULT_A4k16"] == 8 and k["D"] == 32 and k["ARCH"]["n_layer"] == 3 and k["WIDTH"] == 3 and k["EVAL_EVERY"] == 1200, k
    print(f"  recipe read in a child from main at {k['main_sha'][:7]}: test_stream_recipe.LR {k['tsr_LR']:g}; SLOW: gate {k['LR']:g} "
          f"throughout, others {k['LR_WARM']:g} for updates 1-{k['WARM']}; window width {k['WIDTH']}; ARCH {k['ARCH']}, D {k['D']}; "
          f"evaluation every {k['EVAL_EVERY']}, bound at {k['BIND_PASS']} held, early stop after {k['STOP_AFTER']} in a row (after "
          f"{k['T_CHECK']}); no trigger (as S67); 1 torch thread per run", flush=True)
    for kk, a in s68.ARMS.items():
        print(f"  lr for {s68.NAME} {kk}: {a['lr']:g}; iters {a['iters']}; seeds {ec.fmt_seeds(a['seeds'])}; size {a['spec']['size']}; "
              f"task {a['spec']['task']}; gate {a['spec']['gate']}; marks {a['spec']['marks'][0]}..{a['spec']['marks'][-1]}"
              + (f"; halts once at {a['spec']['halt_once_at']}" if a['spec'].get('halt_once_at') else "")
              + (f"; no checkpoint after {a['spec']['ckpt_until']}" if a['spec'].get('ckpt_until') is not None else ""))


def line(j, rec):
    return (f"acc {rec.get('acc'):.3f}  trans {str(rec.get('transition')):>5}  stop {rec.get('stopped_at')}  outcome {rec.get('outcome')}"
            + (f"; halted once at {rec.get('halted_once_at')}" if rec.get("halted_once_at") else ""))


def report_all(tag=""):
    print("#" * 100)
    print(f"REPORTS — {ec.BANNER}{c2.dry_label()}")
    print("#" * 100)
    r = s68.report()
    print("#" * 100)
    print(f"SUMMARY — {ec.BANNER}{c2.dry_label()}")
    rp = r["repro44"]
    print(f"  {tag}S68 reproduction of a resumed run at {s68.REPRO_AT}: "
          + ("not run yet" if rp["ok"] is None else ("BIT-IDENTICAL" if rp["ok"] else "DIFFERS: " + rp["why"])))
    print(f"  {tag}S68 N256: {r['width']['N256']} ({r['k']['N256']}/6); D128: {r['width']['D128']} ({r['k']['D128']}/6); joint: {r['joint']}")
    print(f"  {tag}S68 budget for the main-line test: {r['budget']}" + (f"; 86,400 IS NOT ENOUGH ({r['rising']})" if r["rising"] else ""))
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
    if args.dry:
        c2.DRY.update(on=True, iters=DRY_ITERS)
        s68.make_dry(DRY_ITERS)
    pv = ec.print_banner("batch 20: continue (S68)" + c2.dry_label())
    sha, rows = c16.code_sha(CHILD, True)
    print(f"  code SHA (the child): {sha[:12]} over {len(rows)} modules ({sum(1 for r in rows if r[2] != 'branch')} served from main "
          f"at {mt.MAIN_SHA[:7]}): " + " ".join(r[0] for r in rows), flush=True)
    c16.STATE["shas"] = {s68.NAME: sha}
    c16.STATE["other_cpus"] = sorted(set(c16.other_cpus_seen([s68])) | {cpu})
    print(f"  this segment's CPU: {cpu}; the batch's CPU (the full CHECK pass): {c16.STATE['batch_cpu']}; checkpoints in "
          f"{args.ckpt_dir}", flush=True)
    if args.report_only:
        print("  REPORT ONLY: from the saved runs (no repro check, CHECKs or runs in this call)")
        report_all()
        return
    if not ec.repro_check():
        print("  ! this container does not reproduce X: the recorded pairing is invalid; stopping.")
        sys.exit(1)
    projection(args.workers)
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
    c2.print_screen_header2(s68)
    s_ = ec.load_store(s68.NAME)
    s_["meta"].setdefault("provenance", pv)
    s_["meta"]["dry"] = args.dry
    s_["meta"]["arms"] = {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in a.items()} for k, a in s68.ARMS.items()}
    s_["meta"].setdefault("code_shas", [])
    if sha not in s_["meta"]["code_shas"]:
        s_["meta"]["code_shas"].append(sha)
    ec.save_store(s68.NAME, s_)
    budget = None if args.budget is None else args.budget * 60 - (time.time() - t_start)
    print(f"  pool: {args.workers} workers (1 torch thread each)" + (f"; segment budget {budget / 60:.0f} min left" if budget else ""),
          flush=True)
    t0 = time.time()
    n = c19.run_pool19(screens, args.workers, budget, line, args.ckpt_dir, cpu, args.dry, dur_fn)
    print(f"  {n} runs saved in this segment; batch wall clock {(time.time() - t0) / 60:.1f} min (this segment)", flush=True)
    report_all("DRY RUN " if args.dry else "")


if __name__ == "__main__":
    main()
