#!/usr/bin/env python
"""
explore_batch21.py — EXPLORATORY, not a result. Batch 21 of the outside-ideas screens (the user's request of 13 October
2026: "the D128 oracle's validity on ten seeds (single Adam and with SLOW) before your spec goes to V"): S69
explore_oracle_d128. Same harness, rules, labels, segments and checkpoints as batches 19 and 20 (explore_common16 /
explore_common19, imported read-only). Everything below and the readings (explore_oracle_d128's docstring) were fixed
before any S69 run.

HARNESS: 1 torch thread per run; every segment reproduces X's recorded arm A (seed 160, 2400 steps) bit for bit (a failure
stops the segment); runs are cached under arm|seed|code SHA|CPU; a run with ok = False is retried from scratch (at most 3
attempts); every run checkpoints after each evaluation and a stopped run resumes from its checkpoint in the next segment
(explore_b19_model). The CHECKs (first segment, recorded; later segments skip them while the code SHA is unchanged):
  - ORC_A_D128 on seeds 270 and 271 reproduces S67's ORC_P16_D128 records (explore_out/capacity_results.json, code
    17e56410a1db, a 2.80GHz Xeon) through 2400 bit for bit (curve, statistics, early curve);
  - the recipes through the real run path (forward and evaluation stubbed), 2401 updates at D128 on P16S4: ORC_A_D128 one
    group at 1e-3 throughout; ORC_S_D128 an empty gate group and one group holding every trainable parameter, at 1e-4 at
    updates 1 and 2400 and 1e-3 at 2401; the model is the perfect gate at n_embd 128;
  - ORC_S_D128 is not ORC_A_D128: their curves differ at 1200 (the recipe reaches the run);
  - a checkpoint across the SLOW switch: the oracle on SLOW (P4S4, N256, D32, for speed; the recipe code is size-free)
    halted after its checkpoint at 3600 and resumed to 4800 equals the uninterrupted run in every compared field;
  - the readings on synthetic counts; the modules served from main.
RUNTIME: the projection (batch 19's stored pool-load timing of this spec; worst case, every run to 40,800) is printed
first; nothing is cut (no budget was given).
SEGMENTS: --resume, --budget MIN (default 116), --report-only, --dry (one seed per arm, 2400 updates, separate dry_ store;
not for interpretation), --ckpt-dir DIR.

Run:  python explore_batch21.py [--workers 4] [--dry] [--resume] [--budget MIN] [--report-only] [--ckpt-dir DIR]
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
import explore_oracle_d128 as s69
import explore_main9c as mt
from test_binding_recipe import makespan
from test_binding_onset import EVAL_EVERY

CHILD = s69.CHILD
CHECKS_RECORD = os.path.join(ec.OUT_DIR, "batch21_checks.json")
B19_RUNTIME = "batch19_runtime"
RUNTIME_STORE = "batch21_runtime"
DEFAULT_CKPT = os.path.join(os.environ.get("TMPDIR", "/tmp"), "b21_ckpt")
DRY_ITERS = 2400
STATE = dict(dry=False, timings={})

c19.CHILD = CHILD                      # the pool's child module (run_pool19 reads it at each start)


def screens():
    return [s69]


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
    st19 = ec.load_store(B19_RUNTIME)["meta"]
    tm = st19["timings"]
    STATE["timings"] = tm
    print(f"  PROJECTION (worst case: every run to its iters; batch 19's pool-load timing of this spec, stored "
          f"{st19['projection_time']}; S68's observed rate for the D128 oracle was faster):")
    durs = []
    for k, a in s69.ARMS.items():
        d = per_run(a["spec"], a["iters"], tm)
        st_, te = tm[spec_key(a["spec"])]
        durs += [d] * len(a["seeds"])
        print(f"    {k:<11} {len(a['seeds']):>2} runs x {d / 3600:5.1f} h (step {st_ * 1e3:6.1f} ms, evaluation {te:5.1f} s, "
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
RESUME_FIELDS = 15


def parent_checks():
    rows = []
    ar, sr = s69.arm_reading, s69.spec_reading
    rows.append(("S69 arm reading: 10, 9 -> reliable; 8, 5 -> at risk; 4, 0 -> unreliable; incomplete -> INCOMPLETE",
                 [ar(b, True) for b in (10, 9, 8, 5, 4, 0)] == ["reliable"] * 2 + ["at risk"] * 2 + ["unreliable"] * 2
                 and ar(10, False) == "INCOMPLETE"))
    cases = {(9, 0): "keep the draft's oracle", (10, 10): "keep the draft's oracle", (8, 9): "the D128 oracle on SLOW",
             (4, 10): "the D128 oracle on SLOW", (5, 8): "VALID_N = 1 at D128 (with ORC_S_D128)",
             (8, 8): "VALID_N = 1 at D128 (with ORC_A_D128)", (6, 2): "VALID_N = 1 at D128 (with ORC_A_D128)",
             (4, 4): "no D128 validity check at 40,800", (0, 0): "no D128 validity check at 40,800"}
    got = {k: sr(*k, True) for k in cases}
    rows.append((f"S69 spec reading on {len(cases)} synthetic (b_A, b_S) pairs, each outcome one name; incomplete -> no reading",
                 got == cases and sr(10, 10, False).startswith("INCOMPLETE")))
    rows.append(("two seeds both bind: b(b-1)/90 (10 -> 1, 9 -> 0.8, 1 -> 0); exact McNemar two-sided (0 vs 5 -> 0.0625; 3 vs 3 -> 1)",
                 s69.both_two(10) == 1 and abs(s69.both_two(9) - 0.8) < 1e-12 and s69.both_two(1) == 0
                 and abs(s69.mcnemar2(0, 5) - 0.0625) < 1e-12 and s69.mcnemar2(3, 3) == 1.0))
    recs = {s: s69.s67_record(s) for s in s69.S67_SEEDS}
    rows.append((f"S67 records: ORC_P16_D128 on {s69.S67_SEEDS} saved (code {s69.S67_SHA}): 270 not bound, 271 BOUND at 12000",
                 recs[270] is not None and recs[270].get("transition") is None and recs[271] is not None
                 and recs[271].get("transition") == 12000))
    return rows


def run_checks():
    t0 = time.time()
    arms = s69.build_arms()
    spA, spS = arms["ORC_A_D128"]["spec"], arms["ORC_S_D128"]["spec"]
    small_slow = dict(cfg="k16", task="P4S4", gate="perfect", size=[8, 32, 3], slow=True)
    jobs = {
        "a270": ("run", dict(spec=spA, seed=270, iters=2400, ckpt=None)),
        "a271": ("run", dict(spec=spA, seed=271, iters=2400, ckpt=None)),
        "s270": ("run", dict(spec=spS, seed=270, iters=1200, ckpt=None)),
        "lrs": ("lrs_check", dict(which={"A": spA, "S": spS})),
        "resume": ("resume_check", dict(spec=small_slow, seed=5, iters=4800, halt_at=3600)),
    }
    keys = list(jobs)
    with ThreadPoolExecutor(max_workers=len(keys)) as ex:
        res = dict(zip(keys, ex.map(lambda k: c16.run_child(CHILD, jobs[k][0], jobs[k][1], main=True), keys)))
    rows = []
    for k, s in (("a270", 270), ("a271", 271)):
        ok, why = s69.same_upto(res[k], s69.s67_record(s), 2400)
        rows.append((f"ORC_A_D128 seed {s} reproduces S67's ORC_P16_D128 record through 2400 (curve, statistics, early curve)"
                     f"{' — ' + why if why else ''} [curve {res[k].get('curve')}]", ok))
    W = 2400
    la, ls = res["lrs"]["A"], res["lrs"]["S"]
    rows.append((f"ORC_A_D128 through the real path: one group at {la['at1']} at update 1, {la['at_w']} at {W}, {la['at_w1']} at {W + 1}; "
                 f"groups {la['groups']} (none: onset_run's single Adam); covers every trainable parameter {la['cover']}; "
                 f"gate {la['gate_kind']}, n_embd {la['n_embd']}",
                 la["ok"] and la["n"] == W + 1 and la["at1"] == la["at_w"] == la["at_w1"] == [1e-3] and la["cover"]
                 and la["gate_kind"] == "perfect" and la["n_embd"] == 128))
    rows.append((f"ORC_S_D128 through the real path: groups {ls['groups']} (an empty gate group, every trainable parameter "
                 f"({ls['n_trainable']}) in 'rest'), lrs at update 1 {ls['at1']}, {W} {ls['at_w']}, {W + 1} {ls['at_w1']}; covers every "
                 f"trainable parameter {ls['cover']}; gate {ls['gate_kind']}, n_embd {ls['n_embd']}",
                 ls["ok"] and ls["n"] == W + 1 and [list(g) for g in ls["groups"]] == [["gate", 0], ["rest", ls["n_trainable"]]]
                 and ls["at1"] == [1e-3, 1e-4] and ls["at_w"] == [1e-3, 1e-4] and ls["at_w1"] == [1e-3, 1e-3] and ls["cover"]
                 and ls["gate_kind"] == "perfect" and ls["n_embd"] == 128))
    ca, cs = res["a270"].get("curve") or [], res["s270"].get("curve") or []
    rows.append((f"ORC_S_D128 differs from ORC_A_D128 at 1200 on seed 270 (the recipe reaches the run): {cs[:1]} vs {ca[:1]}",
                 res["s270"].get("ok") and cs[:1] and ca[:1] and cs[0] != ca[0]))
    r = res["resume"]
    rows.append((f"a checkpoint across the SLOW switch: the oracle on SLOW (P4S4, N256, D32, seed 5) halted after its checkpoint at 3600 "
                 f"and resumed (from {r.get('resumed')}) to 4800 equals the uninterrupted run: fields differing {r.get('diff')} of "
                 f"{r.get('n_keys')}", r.get("ok") and r.get("diff") == [] and r.get("resumed") == [3600]))
    rows += parent_checks()
    d = mt.differing()
    rows.append((f"the modules served from main are exactly the test modules that differ between this branch and {mt.MAIN_SHA[:7]} ({d})",
                 d == sorted(mt.SERVED)))
    ok = True
    for nm, v in rows:
        print(f"  CHECK batch 21: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    if not ok:
        print("  ! a CHECK failed; stopping.")
        sys.exit(1)
    print(f"  all CHECKs passed ({(time.time() - t0) / 60:.1f} min)", flush=True)


def constants():
    k = c16.run_child(CHILD, "constants", {}, main=True)
    assert k["LR"] == 1e-3 and k["LR_WARM"] == 1e-4 and k["WARM"] == 2400 and k["tsr_LR"] == 1e-3, k
    assert k["D"] == 32 and k["ARCH"]["n_layer"] == 3 and k["EVAL_EVERY"] == 1200, k
    t = k["tasks"].get("P16S4")
    print(f"  recipe read in a child from main at {k['main_sha'][:7]}: test_stream_recipe.LR {k['tsr_LR']:g} (the recorded validity "
          f"recipe's single Adam); SLOW: {k['LR_WARM']:g} for updates 1-{k['WARM']}, then {k['LR']:g}; ARCH {k['ARCH']}; evaluation every "
          f"{k['EVAL_EVERY']}, bound at {k['BIND_PASS']} held, early stop after {k['STOP_AFTER']} in a row (after {k['T_CHECK']}); "
          f"task P16S4 {t}; 1 torch thread per run", flush=True)
    for kk, a in s69.ARMS.items():
        print(f"  lr for {s69.NAME} {kk}: {a['lr']:g}; iters {a['iters']}; seeds {ec.fmt_seeds(a['seeds'])}; size {a['spec']['size']}; "
              f"task {a['spec']['task']}; gate {a['spec']['gate']}; {'SLOW' if a['spec'].get('slow') else 'single Adam'}")


def line(j, rec):
    return f"acc {rec.get('acc'):.3f}  trans {str(rec.get('transition')):>5}  stop {rec.get('stopped_at')}  outcome {rec.get('outcome')}"


def report_all(tag=""):
    print("#" * 100)
    print(f"REPORTS — {ec.BANNER}{c2.dry_label()}")
    print("#" * 100)
    c2.print_screen_header2(s69)
    ev = s69.report()
    print("#" * 100)
    print(f"SUMMARY — {ec.BANNER}{c2.dry_label()}")
    print(f"  {tag}S69 ORC_A_D128 (single Adam): {ev['arm_reading']['ORC_A_D128']} ({ev['b']['ORC_A_D128']}/10); ORC_S_D128 (SLOW): "
          f"{ev['arm_reading']['ORC_S_D128']} ({ev['b']['ORC_S_D128']}/10)")
    print(f"  {tag}S69 for the spec: {ev['spec']}")
    print("#" * 100)
    return ev


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
        s69.make_dry(DRY_ITERS)
    pv = ec.print_banner("batch 21: the D128 oracle's validity (S69)" + c2.dry_label())
    sha, rows = c16.code_sha(CHILD, True)
    print(f"  code SHA (the child): {sha[:12]} over {len(rows)} modules ({sum(1 for r in rows if r[2] != 'branch')} served from main "
          f"at {mt.MAIN_SHA[:7]}): " + " ".join(r[0] for r in rows), flush=True)
    c16.STATE["shas"] = {s69.NAME: sha}
    c16.STATE["other_cpus"] = sorted(set(c16.other_cpus_seen([s69])) | {cpu})
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
    c2.print_screen_header2(s69)
    s_ = ec.load_store(s69.NAME)
    s_["meta"].setdefault("provenance", pv)
    s_["meta"]["dry"] = args.dry
    s_["meta"]["arms"] = {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in a.items()} for k, a in s69.ARMS.items()}
    s_["meta"].setdefault("code_shas", [])
    if sha not in s_["meta"]["code_shas"]:
        s_["meta"]["code_shas"].append(sha)
    ec.save_store(s69.NAME, s_)
    budget = None if args.budget is None else args.budget * 60 - (time.time() - t_start)
    print(f"  pool: {args.workers} workers (1 torch thread each)" + (f"; segment budget {budget / 60:.0f} min left" if budget else ""),
          flush=True)
    t0 = time.time()
    n = c19.run_pool19(screens, args.workers, budget, line, args.ckpt_dir, cpu, args.dry, dur_fn)
    print(f"  {n} runs saved in this segment; batch wall clock {(time.time() - t0) / 60:.1f} min (this segment)", flush=True)
    report_all("DRY RUN " if args.dry else "")


if __name__ == "__main__":
    main()
